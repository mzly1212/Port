# -*- coding: utf-8 -*-
"""
main.py — tracker_v2 入口 (复用根目录 I/O 适配器, 只换算法引擎)

与根目录 main.py 的区别:
  - 引擎换为 tracker_v2 的 MTT 架构实现
  - 日志/回放文件独立命名, 不与 v1 生产文件冲突
"""
import bootstrap  # noqa: F401  (根目录 -> sys.path: data/config/适配器)

import json
import logging
import os
import socket
from logging.handlers import RotatingFileHandler

from config import Config
from engine import PerceptionFilterEngine
from input_adapter import NebulaInputAdapter
from output_adapter import ProtocolAdapter

# --- 生产级日志 (独立文件, 避免与 v1 并行运行时互写) ---
log_dir = os.path.join(Config.BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "perception_server_v2.log")

handler = RotatingFileHandler(log_file, maxBytes=50 * 1024 * 1024,
                              backupCount=5, encoding='utf-8')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[handler, logging.StreamHandler()])
logger = logging.getLogger(__name__)


def main():
    logger.info("tracker_v2 初始化组件中...")
    engine = PerceptionFilterEngine(Config)
    input_adapter = NebulaInputAdapter(Config)
    output_adapter = ProtocolAdapter(Config)

    replay_file = None
    if getattr(Config, 'SAVE_REPLAY_DATA', False):
        replay_path = os.path.join(Config.BASE_DIR, "tracker_v2",
                                   "replay_v2.jsonl")
        os.makedirs(os.path.dirname(replay_path), exist_ok=True)
        replay_file = open(replay_path, "a", encoding="utf-8")
        logger.info(f"数据落盘已开启: {replay_path}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((Config.UDP_NEBULALINK_IP, Config.UDP_NEBULALINK_PORT))
    logger.info(f"UDP Server 启动成功, 监听星云互联数据于 "
                f"{Config.UDP_NEBULALINK_IP}:{Config.UDP_NEBULALINK_PORT}")

    web_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    web_sock.connect((Config.TCP_WEB_IP, Config.TCP_WEB_PORT))
    logger.info(f"已连接前端 TCP {Config.TCP_WEB_IP}:{Config.TCP_WEB_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(65535)

            # [1. 输入解析]
            raw_frame = input_adapter.parse_udp_packet(data)
            if not raw_frame or not raw_frame.vehicles:
                continue

            # [2. 算法处理 (v2 MTT 引擎)]
            processed_frame = engine.process_frame(raw_frame)

            # [3. 输出转换]
            tcp_bytes_to_send = output_adapter.to_tcp_bytes(processed_frame)
            web_sock.sendall(tcp_bytes_to_send)

            # [4. 异常打印]
            if processed_frame.alerts:
                logger.warning(f"跟踪告警: {processed_frame.alerts}")

            # [5. 可选: 数据落盘用于回放]
            if replay_file:
                record = {
                    "timestamp": processed_frame.timestamp_ms,
                    "raw": [
                        {
                            "id": v.object_id,
                            "rel_x": v.rel_x,
                            "rel_y": v.rel_y,
                            "heading": getattr(v, 'radar_heading', 0.0),
                            "itc_obj_type": getattr(v, 'itc_obj_type', 1),
                            "itc_sub_type": getattr(v, 'itc_sub_type', 99),
                            "plate_num": getattr(v, 'plate_num', ""),
                            "lane_no": getattr(v, 'lane_no', "")
                        }
                        for v in raw_frame.vehicles
                    ],
                    "fixed": [
                        {
                            "id": v.fixed_id,
                            "rel_x": v.x,
                            "rel_y": v.y,
                            "is_predicted": v.is_predicted,
                            "psi": getattr(v, 'psi', 0.0),
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
        except Exception as e:
            logger.error(f"运行时异常: {e}", exc_info=True)


if __name__ == "__main__":
    main()
