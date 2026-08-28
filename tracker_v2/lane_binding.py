# -*- coding: utf-8 -*-
"""
lane_binding.py — 车道绑定层 (纯输出注释)

职责:
  - 用 KF 位置/速度给航迹标注 lane_id / s / l
  - 车道变更确认 (粘滞护城河 + 连续帧确认, 防相邻车道画龙)
  - 方向裁决: KF 速度矢量在车道切向上的投影 -> dir 状态机
    (死区 + 连续确认帧数, v1 已验证的滞回语义)

与 v1 的本质区别: 绑定结果只写进 track 的注释字段, 绝不回馈
运动滤波。丢一帧注释, 下一帧重算即可 —— 不存在 "离道导致
运动状态被重置" 的失效模式。
"""
import math

from shapely.geometry import Point

import params as P
from geo_utils import math_heading_of, ang_diff_deg


class LaneBinding:
    def __init__(self, lane_map):
        self.map = lane_map

    # ------------------------------------------------------------------
    def annotate(self, tr, t_ms):
        x, y = tr.kf.pos
        speed = tr.kf.speed

        # 参考航向: 速度足够时用运动矢量 (物理真相), 否则雷达滤波航向
        if speed >= P.MOTION_HEADING_MIN_SPEED:
            ref = math_heading_of(*tr.kf.vel)
        else:
            ref = tr.heading.filtered

        lane_id, s, l = self.map.match(
            x, y, ref_heading=ref, v=speed,
            last_lane_id=tr.lane_id if tr.lane_id is not None else tr.last_lane_id,
        )

        if lane_id is not None:
            if tr.lane_id is not None and lane_id != tr.lane_id:
                # 候选新车道: 需连续 LANE_HYSTERESIS_FRAMES 帧确认
                tr.lane_hysteresis += 1
                if tr.lane_hysteresis >= P.LANE_HYSTERESIS_FRAMES:
                    tr.lane_id = lane_id
                    tr.s, tr.l = s, l
                    tr.lane_hysteresis = 0
                else:
                    # 维持老车道, 在老车道上重新投影
                    self._project(tr, tr.lane_id, x, y)
            else:
                if tr.lane_id is None:
                    tr.last_lane_id = lane_id
                tr.lane_id = lane_id
                tr.lane_hysteresis = 0
                tr.s, tr.l = s, l
        else:
            # 离道: 记住最后车道 (重入轨时恢复粘滞)
            if tr.lane_id is not None:
                tr.last_lane_id = tr.lane_id
                tr.lane_id = None

        self._update_direction(tr, t_ms)

    # ------------------------------------------------------------------
    def _project(self, tr, lane_id, x, y):
        lane = self.map.lanes.get(lane_id)
        if lane is None:
            return
        tr.s = lane['line'].project(Point(x, y))
        tr.l = self.map.signed_offset(lane_id, x, y, tr.s)

    # ------------------------------------------------------------------
    # 方向裁决: 切向投影 + 滞回状态机
    # ------------------------------------------------------------------
    def _update_direction(self, tr, t_ms):
        if tr.lane_id is None:
            return
        tx, ty = self.map.tangent(tr.lane_id, tr.s)
        vx, vy = tr.kf.vel
        vt = vx * tx + vy * ty

        if abs(vt) > P.DIR_DEADZONE:
            want = 1 if vt > 0 else -1
            if tr.dir is None:
                # 初次建立: 直接采纳 (无需确认, 确认机制只约束翻转)
                tr.dir = want
                tr.dir_confirm = 0
            elif want != tr.dir:
                tr.dir_confirm += 1
                if tr.dir_confirm >= P.DIR_CONFIRM_N:
                    tr.dir = want
                    tr.dir_confirm = 0
            else:
                tr.dir_confirm = 0
            return

        # 速度不足以裁决: 方向未知期用可靠雷达航向兜底初始化
        if tr.dir is None and tr.heading.reliable(t_ms):
            lane_h = self.map.lane_heading(tr.lane_id)
            if lane_h is not None:
                tr.dir = -1 if ang_diff_deg(tr.heading.filtered, lane_h) \
                    > P.DIR_UNKNOWN_FLIP else 1
