# replay_sender.py
import argparse
import json
import time
import socket
import math
from collections import deque
from typing import List, Optional

from config import Config
from data import ProcessedVehicle, ProcessedFrame
from output_adapter import ProtocolAdapter

"""
克隆 修改 提交 测试 2
"""

class ReplaySender:
    def __init__(self, mode: str = "fixed", speed: float = 1.0, loop: bool = False,
                 start_idx: Optional[int] = None, end_idx: Optional[int] = None):
        """
        :param mode: "raw" 或 "fixed"
        :param speed: 播放速度倍率
        :param loop: 是否循环播放
        :param start_idx: 起始帧索引（包含），默认为 0
        :param end_idx: 结束帧索引（包含），默认为最后一帧
        """
        self.mode = mode
        self.speed = speed
        self.loop = loop
        self.start_idx = start_idx if start_idx is not None else 0
        self.end_idx = end_idx
        self.adapter = ProtocolAdapter(Config)
        self.records = self._load_records()
        self.block_set = {17164}
        if not self.records:
            raise RuntimeError("没有加载到任何回放数据，请检查 replay_data.jsonl 是否存在。")

        # 校验并修正索引范围
        total = len(self.records)
        if self.start_idx < 0:
            self.start_idx = max(0, total + self.start_idx)
        if self.end_idx is None or self.end_idx >= total:
            self.end_idx = total - 1
        elif self.end_idx < 0:
            self.end_idx = max(0, total + self.end_idx)

        if self.start_idx > self.end_idx:
            raise ValueError(f"起始索引 {self.start_idx} 大于结束索引 {self.end_idx}，无效范围。")
        if self.start_idx >= total:
            raise ValueError(f"起始索引 {self.start_idx} 超出总帧数 {total}。")

        # 用于速度差分计算的上一帧数据缓存
        self.prev_raw = {}   # id -> (x, y, timestamp)
        self.prev_fixed = {} # id -> (x, y, timestamp)

    def _load_records(self) -> List[dict]:
        """加载 JSONL 文件"""
        path = Config.REPLAY_FILE_PATH
        records = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            print(f"✅ 成功加载 {len(records)} 帧回放数据")
        except Exception as e:
            print(f"❌ 加载回放文件失败: {e}")
        return records

    def _build_frame_from_raw(self, raw_list: List[dict], timestamp: int) -> ProcessedFrame:
        vehicles = []
        current_pos = {v['id']: (v['rel_x'], v['rel_y']) for v in raw_list}

        for v in raw_list:
            obj_id = v['id']
            x, y = v['rel_x'], v['rel_y']
            heading_deg = v.get('heading', 0.0)
            psi = math.radians(heading_deg)

            # 速度差分
            speed = 0.0
            prev = self.prev_raw.get(obj_id)
            if prev:
                prev_x, prev_y, prev_t = prev
                dt = (timestamp - prev_t) / 1000.0
                if dt > 0:
                    dist = math.hypot(x - prev_x, y - prev_y)
                    speed = dist / dt
                    if speed > 30.0:
                        speed = 0.0

            pv = ProcessedVehicle(
                original_id=obj_id,
                fixed_id=obj_id,
                x=x,
                y=y,
                v=speed,
                psi=psi,
                is_predicted=False,
                itc_obj_type=v.get('itc_obj_type', 1),
                itc_sub_type=v.get('itc_sub_type', 99),
                plate_num=v.get('plate_num', ""),
                lane_no=v.get('lane_no', ""),
                radar_heading=heading_deg,
                type_reliability=1.0
            )
            vehicles.append(pv)

        self.prev_raw = {v['id']: (v['rel_x'], v['rel_y'], timestamp) for v in raw_list}
        return ProcessedFrame(timestamp_ms=timestamp, vehicles=vehicles, alerts=[])

    def _build_frame_from_fixed(self, fixed_list: List[dict], timestamp: int) -> ProcessedFrame:
        filtered_fixed = [v for v in fixed_list if v['id'] not in self.block_set]

        vehicles = []
        for v in filtered_fixed:
            obj_id = v['id']
            x, y = v['rel_x'], v['rel_y']
            psi = v.get('psi', 0.0)
            is_pred = v.get('is_predicted', False)

            speed = 0.0
            prev = self.prev_fixed.get(obj_id)
            if prev:
                prev_x, prev_y, prev_t = prev
                dt = (timestamp - prev_t) / 1000.0
                if dt > 0:
                    dist = math.hypot(x - prev_x, y - prev_y)
                    speed = dist / dt
                    if speed > 30.0:
                        speed = 0.0

            pv = ProcessedVehicle(
                original_id=obj_id,
                fixed_id=obj_id,
                x=x,
                y=y,
                v=speed,
                psi=psi,
                is_predicted=is_pred,
                itc_obj_type=v.get('itc_obj_type', 1),
                itc_sub_type=v.get('itc_sub_type', 99),
                plate_num=v.get('plate_num', ""),
                lane_no=v.get('lane_no', ""),
                radar_heading=math.degrees(psi),
                type_reliability=1.0
            )
            vehicles.append(pv)

        self.prev_fixed = {v['id']: (v['rel_x'], v['rel_y'], timestamp) for v in filtered_fixed}
        return ProcessedFrame(timestamp_ms=timestamp, vehicles=vehicles, alerts=[])

    def send_all(self):
        """主循环：发送指定范围内的所有记录"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((Config.TCP_WEB_IP, Config.TCP_WEB_PORT))
            print(f"🔗 已连接前端服务器 {Config.TCP_WEB_IP}:{Config.TCP_WEB_PORT}")
        except Exception as e:
            print(f"❌ TCP 连接失败: {e}")
            return

        total = len(self.records)
        start = self.start_idx
        end = self.end_idx
        range_len = end - start + 1
        print(f"📌 将发送第 {start} 帧至第 {end} 帧（共 {range_len} 帧）")

        idx = start
        while True:
            record = self.records[idx]
            timestamp = record['timestamp']

            if self.mode == 'raw':
                frame = self._build_frame_from_raw(record.get('raw', []), timestamp)
            else:  # fixed
                frame = self._build_frame_from_fixed(record.get('fixed', []), timestamp)

            try:
                data_bytes = self.adapter.to_tcp_bytes(frame)
                sock.sendall(data_bytes)
                print(f"📤 发送帧 {idx+1}/{total} [模式: {self.mode}]  车辆数: {len(frame.vehicles)}")
            except Exception as e:
                print(f"❌ 发送失败: {e}")
                break

            # 计算下一次发送的等待时间
            if idx < end:
                next_ts = self.records[idx+1]['timestamp']
                delta_ms = next_ts - timestamp
                if delta_ms > 0:
                    sleep_sec = delta_ms / 1000.0 / self.speed
                    if sleep_sec > 5.0:
                        sleep_sec = 0.1
                    time.sleep(sleep_sec)

            idx += 1
            if idx > end:
                if self.loop:
                    print("🔄 循环播放，重新开始...")
                    # 重置速度缓存，因为新循环起始帧没有前序帧
                    self.prev_raw.clear()
                    self.prev_fixed.clear()
                    idx = start
                else:
                    print("✅ 播放完毕。")
                    break

        sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线数据回放发送工具")
    parser.add_argument('--mode', choices=['raw', 'fixed'], default='fixed',
                        help='发送模式：raw=原始雷达数据，fixed=算法修复数据')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='播放速度倍数，例如 2.0 为两倍速')
    parser.add_argument('--loop', action='store_true',
                        help='是否循环播放')
    parser.add_argument('--start', type=int, default=None,
                        help='起始帧索引（从0开始），默认为0')
    parser.add_argument('--end', type=int, default=None,
                        help='结束帧索引（包含），默认为最后一帧')
    args = parser.parse_args()

    sender = ReplaySender(mode=args.mode, speed=args.speed, loop=args.loop,
                          start_idx=args.start, end_idx=args.end)
    sender.send_all()