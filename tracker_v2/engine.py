# -*- coding: utf-8 -*-
"""
engine.py — tracker_v2 感知滤波引擎

对外接口与根目录 smooth_engine.PerceptionFilterEngine 完全一致:
    PerceptionFilterEngine(Config, use_ai=...).process_frame(raw_frame)
        -> ProcessedFrame
可直接替换根目录 main.py 中的引擎 (见本目录 main.py)。
"""
import bootstrap  # noqa: F401  (根目录 -> sys.path, 复用 data 契约)
from data import ProcessedFrame

from map_model import LaneMap
from track_manager import TrackManager


class PerceptionFilterEngine:
    def __init__(self, config=None, use_ai=False):
        # config/use_ai 仅为接口兼容保留: v2 所有可调参数在 params.py
        self.lane_map = LaneMap()
        self.tracker = TrackManager(self.lane_map)

    def process_frame(self, raw_frame):
        vehicles, alerts = self.tracker.process_frame(
            raw_frame.vehicles, raw_frame.timestamp_ms)
        return ProcessedFrame(
            timestamp_ms=raw_frame.timestamp_ms,
            vehicles=vehicles,
            alerts=alerts,
        )
