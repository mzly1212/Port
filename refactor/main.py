import logging
from logging.handlers import RotatingFileHandler
import os
import socket
import selectors
import json
import struct
import time
from config import Config
from smooth_engine import PerceptionFilterEngine
from input_adapter import NebulaInputAdapter
from output_adapter import ProtocolAdapter

# --- 配置生产级日志 ---
log_dir = os.path.join(Config.BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "perception_server.log")

handler = RotatingFileHandler(log_file, maxBytes=50*1024*1024, backupCount=5, encoding='utf-8')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[handler, logging.StreamHandler()])
logger = logging.getLogger(__name__)


def main():
    logger.info("初始化组件中...")
    engine = PerceptionFilterEngine(Config, use_ai=True)
    input_adapter = NebulaInputAdapter(Config)
    output_adapter = ProtocolAdapter(Config)

    # 如果开启了回放数据保存，打开文件流
    replay_file = None
    if getattr(Config, 'SAVE_REPLAY_DATA', False):
        replay_file = open(Config.REPLAY_FILE_PATH, "a", encoding="utf-8")
        logger.info(f"💾 数据落盘已开启，回放数据将保存至: {Config.REPLAY_FILE_PATH}")

    # 建立 web Socket 准备发送给 前端 系统
    web_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    web_sock.connect((Config.TCP_WEB_IP, Config.TCP_WEB_PORT))

    # ==========================================
    # 📡 多数据源接入: selectors 单线程多路复用
    #   - 数据源 1: 星云互联 UDP (原有, 端口 10001)
    #   - 数据源 2: 第三方 UDP  (新增, 端口 10011, 报文格式相同)
    # 单线程事件循环串行喂引擎, 规避 LaneQueueTracker 的线程安全
    # 问题 (active_vehicles 字典等共享状态无锁)。
    # ==========================================
    sel = selectors.DefaultSelector()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((Config.UDP_NEBULALINK_IP, Config.UDP_NEBULALINK_PORT))
    sel.register(sock, selectors.EVENT_READ, 'nebula')
    logger.info(f"📡 UDP Server 启动成功，监听星云互联数据于 {Config.UDP_NEBULALINK_IP}:{Config.UDP_NEBULALINK_PORT}")

    # 第三方原始数据包落盘 (按收包原样保存二进制报文)
    thirdparty_raw_file = None
    if getattr(Config, 'ENABLE_THIRDPARTY_INPUT', False):
        third_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        third_sock.bind((Config.UDP_THIRDPARTY_IP, Config.UDP_THIRDPARTY_PORT))
        sel.register(third_sock, selectors.EVENT_READ, 'thirdparty')
        logger.info(f"📡 UDP Server 启动成功，监听第三方数据于 {Config.UDP_THIRDPARTY_IP}:{Config.UDP_THIRDPARTY_PORT}")

        if getattr(Config, 'SAVE_THIRDPARTY_RAW', False):
            raw_path = getattr(Config, 'THIRDPARTY_RAW_FILE',
                                os.path.join(Config.BASE_DIR, "logs", "thirdparty_raw.bin"))
            os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            thirdparty_raw_file = open(raw_path, "ab")
            logger.info(f"💾 第三方原始数据包落盘已开启: {raw_path} "
                        f"(记录格式: 8B接收时刻ms + 4B包长 + 原始报文)")

    def save_thirdparty_raw(data):
        """保存第三方原始报文: [接收时刻 epoch ms][包长][原始字节]"""
        if thirdparty_raw_file is not None:
            thirdparty_raw_file.write(struct.pack('<QI', int(time.time() * 1000), len(data)))
            thirdparty_raw_file.write(data)
            thirdparty_raw_file.flush()

    def handle_packet(data, source):
        """单包处理: 解析 -> 引擎 -> 前端下发 -> 可选落盘"""
        # [0. 第三方原始包落盘] (保存动作先于解析: 解析失败的坏包也留痕)
        if source == 'thirdparty':
            save_thirdparty_raw(data)

        # [1. 输入解析] (两路数据报文格式相同, 共用适配器)
        raw_frame = input_adapter.parse_udp_packet(data)
        if not raw_frame or not raw_frame.vehicles:
            return

        # [2. 算法处理]
        processed_frame = engine.process_frame(raw_frame)

        # [3. 输出转换]
        tcp_bytes_to_send = output_adapter.to_tcp_bytes(processed_frame)
        web_sock.sendall(tcp_bytes_to_send)

        # [4. 异常打印]
        if processed_frame.alerts:
            logger.warning(f"异常捕获: {processed_frame.alerts}")

        # ==========================================
        # 💾 [5. 可选功能] 数据落盘用于本地可视化回放
        # ==========================================
        if replay_file:
            record = {
                "timestamp": processed_frame.timestamp_ms,
                # ===== 原始数据：补全业务属性 =====
                "raw": [
                    {
                        "id": v.object_id,
                        "rel_x": v.rel_x,
                        "rel_y": v.rel_y,
                        "heading": getattr(v, 'radar_heading', 0.0),
                        # --- 新增透传字段 ---
                        "itc_obj_type": getattr(v, 'itc_obj_type', 1),
                        "itc_sub_type": getattr(v, 'itc_sub_type', 99),
                        "plate_num": getattr(v, 'plate_num', ""),
                        "lane_no": getattr(v, 'lane_no', "")
                    }
                    for v in raw_frame.vehicles
                ],
                # ===== 修复后数据：补全业务属性 =====
                "fixed": [
                    {
                        "id": v.fixed_id,
                        "rel_x": v.x,
                        "rel_y": v.y,
                        "is_predicted": v.is_predicted,
                        "psi": getattr(v, 'psi', 0.0),
                        # --- 新增透传字段 ---
                        "itc_obj_type": getattr(v, 'itc_obj_type', 1),
                        "itc_sub_type": getattr(v, 'itc_sub_type', 99),
                        "plate_num": getattr(v, 'plate_num', ""),
                        "lane_no": getattr(v, 'lane_no', "")
                    }
                    for v in processed_frame.vehicles
                ],
                "alerts": processed_frame.alerts
            }
            replay_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            replay_file.flush()

    while True:
        try:
            # 多路复用等待: 任一数据源有包到达即唤醒
            for key, _ in sel.select(timeout=1.0):
                data, addr = key.fileobj.recvfrom(65535)
                handle_packet(data, key.data)
        except Exception as e:
            logger.error(f"运行时异常: {e}", exc_info=True)

if __name__ == "__main__":
    main()