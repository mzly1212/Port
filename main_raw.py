import logging
from logging.handlers import RotatingFileHandler
import os
import socket
import selectors
import struct
import time
import json

from config import Config
from input_adapter import NebulaInputAdapter
from output_adapter_raw import ProtocolAdapter

# 导入数据结构 (根据你的实际存放位置，可能在 data.py 或 engine.py 中)
from data import ProcessedFrame, ProcessedVehicle

# --- 配置纯透传模式的独立日志 ---
log_dir = os.path.join(Config.BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "perception_raw_server.log")

handler = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024, backupCount=5, encoding='utf-8')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] [RAW] %(message)s',
                    handlers=[handler, logging.StreamHandler()])
logger = logging.getLogger(__name__)


class SourceMerger:
    """
    透传版多源目标合并器。

    问题背景: 透传版没有跟踪状态, 两路 UDP 数据交错到达时, 每个下发帧
    只含单源的目标 —— 前端按帧渲染, 所有目标以两源交替频率一闪一闪。
    (算法版无此问题: 跟踪器内目标跨帧存活, 每帧输出都含全部目标)

    解决: 缓存各数据源最近一帧的透传目标, 每收到任意源的新帧, 合并
    所有「新鲜」源的目标后下发; 超过 stale_sec 未更新的源, 其目标
    视为已消失, 从输出中剔除。
    """
    def __init__(self, stale_sec=1.0):
        self.stale_sec = stale_sec
        self.cache = {}   # source -> (monotonic_time, vehicles)

    def merge(self, source, vehicles, now=None):
        """更新指定源缓存, 返回所有新鲜源目标的合并列表"""
        if now is None:
            now = time.monotonic()
        if vehicles:
            self.cache[source] = (now, vehicles)
        else:
            # 该源当前无目标: 剔除其历史目标 (其余源不受影响)
            self.cache.pop(source, None)

        merged = []
        for src in list(self.cache):
            t, vehs = self.cache[src]
            if now - t <= self.stale_sec:
                merged.extend(vehs)
            else:
                del self.cache[src]   # 源数据过期: 目标已消失
        return merged


def main():
    logger.info("🚀 初始化组件中 (纯透传对比模式 - 算法已关闭)...")

    # 仅初始化输入和输出适配器，不初始化 Engine
    input_adapter = NebulaInputAdapter(Config)
    output_adapter = ProtocolAdapter(Config)

    # 如果需要记录纯原始数据用于本地回放对比
    replay_file = None
    if getattr(Config, 'SAVE_REPLAY_DATA', False):
        raw_replay_path = os.path.join(Config.BASE_DIR, "logs", "raw_only_replay.jsonl")
        replay_file = open(raw_replay_path, "a", encoding="utf-8")
        logger.info(f"💾 原始数据落盘已开启，将保存至: {raw_replay_path}")

    # 建立 TCP Socket 准备发送给前端/ITC系统
    web_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    web_sock.connect((Config.TCP_WEB_IP, Config.TCP_WEB_PORT))

    # ==========================================
    # 📡 多数据源接入: selectors 单线程多路复用
    #   - 数据源 1: 星云互联 UDP (原有, 端口 10001)
    #   - 数据源 2: 第三方 UDP  (新增, 端口 10011, 报文格式相同)
    # ⚠ 透传版必须合并两源目标后下发 (SourceMerger), 否则两路帧
    #   交错到达时每帧只含单源目标, 前端按帧渲染 -> 全部目标闪烁。
    # ==========================================
    sel = selectors.DefaultSelector()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((Config.UDP_NEBULALINK_IP, Config.UDP_NEBULALINK_PORT))
    sel.register(sock, selectors.EVENT_READ, 'nebula')
    logger.info(f"📡 UDP Server 启动成功，监听星云互联数据于 {Config.UDP_NEBULALINK_IP}:{Config.UDP_NEBULALINK_PORT}")

    if getattr(Config, 'ENABLE_THIRDPARTY_INPUT', False):
        third_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        third_sock.bind((Config.UDP_THIRDPARTY_IP, Config.UDP_THIRDPARTY_PORT))
        sel.register(third_sock, selectors.EVENT_READ, 'thirdparty')
        logger.info(f"📡 UDP Server 启动成功，监听第三方数据于 {Config.UDP_THIRDPARTY_IP}:{Config.UDP_THIRDPARTY_PORT}")

    # 第三方原始数据包落盘 (按收包原样保存二进制报文)
    thirdparty_raw_file = None
    if getattr(Config, 'ENABLE_THIRDPARTY_INPUT', False) \
            and getattr(Config, 'SAVE_THIRDPARTY_RAW', False):
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

    merger = SourceMerger(stale_sec=1.0)

    def handle_packet(data, source):
        """单包处理: 解析 -> 透传包装 -> 多源合并 -> 前端下发"""
        # [0. 第三方原始包落盘] (保存动作先于解析: 解析失败的坏包也留痕)
        if source == 'thirdparty':
            save_thirdparty_raw(data)

        # [1. 输入解析] (两路数据报文格式相同, 共用适配器)
        raw_frame = input_adapter.parse_udp_packet(data)
        if not raw_frame:
            return

        # [2. 直接透传包装 (跳过任何平滑、预测和修复)]
        raw_pass_vehicles = []
        for rv in raw_frame.vehicles:
            # 把 RawVehicle 的属性原封不动地塞进 ProcessedVehicle
            raw_pass_vehicles.append(
                ProcessedVehicle(
                    original_id=rv.object_id,
                    fixed_id=rv.object_id,  # ID 直接透传，不修复跳变
                    x=rv.rel_x,  # 坐标直接透传，不平滑
                    y=rv.rel_y,
                    v=0.0,  # 透传模式不计算速度
                    psi=0.0,  # 透传模式直接用雷达原生航向角
                    is_predicted=False,  # 永远为 False
                    itc_obj_type=rv.itc_obj_type,
                    itc_sub_type=rv.itc_sub_type,
                    plate_num=rv.plate_num,
                    lane_no=rv.lane_no,
                    radar_heading=rv.radar_heading,
                    type_reliability=rv.type_reliability
                )
            )

        # [3. 多源合并: 所有新鲜源的目标合成一帧, 消除交错闪烁]
        merged = merger.merge(source, raw_pass_vehicles)
        if not merged:
            return

        mock_frame = ProcessedFrame(
            timestamp_ms=raw_frame.timestamp_ms,
            vehicles=merged,
            alerts=[]  # 无算法介入，自然无异常告警
        )

        # [4. 输出转换 + 发送给前端]
        tcp_bytes_to_send = output_adapter.to_tcp_bytes(mock_frame)
        web_sock.sendall(tcp_bytes_to_send)

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
