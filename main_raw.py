import logging
from logging.handlers import RotatingFileHandler
import os
import socket
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

    # 建立 UDP 监听星云互联数据
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((Config.UDP_NEBULALINK_IP, Config.UDP_NEBULALINK_PORT))
    logger.info(f"📡 UDP Server 启动成功，监听星云互联数据于 {Config.UDP_NEBULALINK_IP}:{Config.UDP_NEBULALINK_PORT}")

    # 建立 TCP Socket 准备发送给前端/ITC系统
    web_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    web_sock.connect((Config.TCP_WEB_IP, Config.TCP_WEB_PORT))

    while True:
        try:
            data, addr = sock.recvfrom(65535)

            # [1. 输入解析]
            raw_frame = input_adapter.parse_udp_packet(data)
            if not raw_frame or not raw_frame.vehicles:
                continue

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

            mock_frame = ProcessedFrame(
                timestamp_ms=raw_frame.timestamp_ms,
                vehicles=raw_pass_vehicles,
                alerts=[]  # 无算法介入，自然无异常告警
            )

            # [3. 输出转换]
            tcp_bytes_to_send = output_adapter.to_tcp_bytes(mock_frame)

            # [发送给前端]
            web_sock.sendall(tcp_bytes_to_send)

        except Exception as e:
            logger.error(f"运行时异常: {e}", exc_info=True)


if __name__ == "__main__":
    main()
