# engine.py
from config import Config
from map_manager import MapManager
from lane_tracker import LaneQueueTracker
from data import ProcessedFrame


class PerceptionFilterEngine:
    def __init__(self, config: Config, use_ai: bool = True):
        self.config = config
        # 初始化地图和新的追踪器
        self.map_mgr = MapManager(config.LANE_PATH)
        self.tracker = LaneQueueTracker(self.map_mgr)

    def process_frame(self, frame) -> ProcessedFrame:
        """核心对外接口"""
        current_time = frame.timestamp_ms

        # 将原始帧抛给基于车道的追踪器，直接拿回组装好的平滑车辆数据
        processed_vehicles = self.tracker.process_frame(frame.vehicles, current_time)

        # 告警信息等可根据业务补充
        alert_msgs = []

        return ProcessedFrame(
            timestamp_ms=current_time,
            vehicles=processed_vehicles,
            alerts=alert_msgs
        )