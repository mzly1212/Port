# -*- coding: utf-8 -*-
"""
track.py — 单航迹 (纯状态 + 单量测应用)

生命周期: 创建 -> (命中累积) -> 活跃 -> (断检) -> 盲区推演 -> 超时删除

「航迹」与 v1「VehicleState」的本质区别:
  - 运动状态只有卡尔曼 (全局米制), 没有任何车道系 (s/方向/变道) 状态
    回写运动滤波 —— v1 的 "缝合清历史导致方向丢失" 失效模式在结构上不存在。
  - 车道绑定 (lane_id/s/l/dir) 是输出注释, 由 lane_binding 每帧重算,
    任何一帧丢失都不影响下一帧。
  - 航向有两个独立观点: KF 速度矢量 (物理真相) 与 HeadingEstimator
    (雷达观点), 最终仲裁在 presenter。

本文件只做单航迹状态维护: 量测应用 / 车型投票 / 设施锚定。
航迹间的事 (关联/合并/生命周期) 全部在 track_manager。
"""
import math
from collections import deque

import params as P
from motion_filter import KalmanCV2D, HeadingEstimator
from geo_utils import math_heading_of, norm_deg, ang_diff_deg


def _is_facility(sub_type):
    return sub_type in P.FIXED_FACILITY_SUB_TYPES


class Track:
    def __init__(self, fixed_id, rv, t_ms):
        self.fixed_id = fixed_id        # 修复后的唯一 ID
        self.initial_id = rv.object_id  # 创建时的原始 ID
        self.last_object_id = rv.object_id  # 最近一次喂入量测的原始 ID

        self.attrs = {
            "itc_obj_type": rv.itc_obj_type,
            "itc_sub_type": rv.itc_sub_type,
            "plate_num": rv.plate_num,
            "lane_no": rv.lane_no,
            "type_reliability": rv.type_reliability,
        }

        # ---- 运动滤波 (全局米制) ----
        self.kf = KalmanCV2D(t_ms, rv.rel_x, rv.rel_y)
        self.heading = HeadingEstimator(rv.radar_heading, t_ms)

        # ---- 生命周期 ----
        self.hits = 1
        self.misses = 0
        self.last_update_t = t_ms
        self.first_seen_t = t_ms
        self.meas_history = deque(maxlen=30)  # 3 秒量测位置历史
        self.meas_history.append((t_ms, rv.rel_x, rv.rel_y))

        # ---- 车型投票 (v1 已验证机制) ----
        self.type_votes = {}      # sub_type -> 加权票数
        self.type_frames = {}     # sub_type -> 观测帧数
        self.locked_type = None
        self.vote_type(rv.itc_sub_type, rv.type_reliability, t_ms)

        # ---- 固定设施锚定 (v1 已验证机制) ----
        self.is_facility = False
        self.anchor_x = None
        self.anchor_y = None
        self.facility_obs = deque(maxlen=P.FACILITY_ANCHOR_SAMPLES)
        self.drift_since = None
        if _is_facility(rv.itc_sub_type):
            self.facility_obs.append((rv.rel_x, rv.rel_y))

        # ---- 车道绑定注释 (lane_binding 每帧重算) ----
        self.lane_id = None
        self.last_lane_id = None
        self.s = 0.0
        self.l = 0.0
        self.dir = None           # 沿车道切向符号: +1/-1/None(未建立)
        self.dir_confirm = 0
        self.lane_hysteresis = 0

        # ---- 输出状态 (presenter 维护) ----
        self.out_heading = None
        self.last_output_t = None

    # ------------------------------------------------------------------
    # 时间推进 (盲区时速度指数衰减)
    # ------------------------------------------------------------------
    def predict(self, t_ms):
        self.kf.predict(t_ms, coasting=self.misses > 0)

    # ------------------------------------------------------------------
    # 量测应用
    # ------------------------------------------------------------------
    def update(self, rv, t_ms):
        self.meas_history.append((t_ms, rv.rel_x, rv.rel_y))
        self.last_object_id = rv.object_id
        self.last_update_t = t_ms
        self.hits += 1
        self.misses = 0

        self.kf.predict(t_ms)  # 已被 manager 推进时 dt<=0, 无副作用

        if self.is_facility:
            # 设施: 漂移超阈值拒绝观测 (误绑定防护), 持续超阈值才重锚
            d = math.hypot(rv.rel_x - self.anchor_x, rv.rel_y - self.anchor_y)
            if d < P.FACILITY_MAX_DRIFT:
                self.kf.update(rv.rel_x, rv.rel_y)
            self._facility_drift(rv, t_ms, d)
        else:
            self.kf.update(rv.rel_x, rv.rel_y)

        self.heading.update(rv.radar_heading, t_ms)

        # 设施锚定确认
        if _is_facility(rv.itc_sub_type):
            self.facility_obs.append((rv.rel_x, rv.rel_y))
            if not self.is_facility \
                    and len(self.facility_obs) >= P.FACILITY_ANCHOR_SAMPLES \
                    and self._stationary():
                self._calc_anchor()

        # 车型投票
        self.vote_type(rv.itc_sub_type, rv.type_reliability, t_ms)

        # 属性透传
        self.attrs.update({
            "itc_obj_type": rv.itc_obj_type,
            "itc_sub_type": rv.itc_sub_type,
            "plate_num": rv.plate_num,
            "lane_no": rv.lane_no,
            "type_reliability": rv.type_reliability,
        })

    # ------------------------------------------------------------------
    # 车型投票 (置信度加权 + 时间衰减 + 双门槛, v1 已验证)
    # ------------------------------------------------------------------
    def vote_type(self, sub_type, reliability, t_ms):
        if sub_type is None or sub_type == 99:
            return

        for k in self.type_votes:
            self.type_votes[k] *= P.TYPE_DECAY

        w = reliability if reliability else 0.9
        w = max(0.1, min(1.0, w))
        self.type_votes[sub_type] = self.type_votes.get(sub_type, 0.0) + w
        self.type_frames[sub_type] = self.type_frames.get(sub_type, 0) + 1

        is_fac = sub_type in P.FIXED_FACILITY_SUB_TYPES
        # 设施类型需要静止证据: 移动目标报设施是感知误分类, 拒绝锁定
        if is_fac and not self._stationary():
            return

        total = sum(self.type_votes.values())
        if total <= 0:
            return
        ratio = self.type_votes[sub_type] / total
        frames = self.type_frames[sub_type]

        if self.locked_type is None:
            need_ratio = P.TYPE_FAC_LOCK_RATIO if is_fac else P.TYPE_LOCK_RATIO
            need_frames = P.TYPE_FAC_LOCK_FRAMES if is_fac else P.TYPE_LOCK_FRAMES
            if ratio >= need_ratio and frames >= need_frames:
                self.locked_type = sub_type
        elif sub_type != self.locked_type:
            # 已锁定但被另一类型大比例持续压倒 -> 纠正
            if ratio >= P.TYPE_FIX_RATIO and frames >= P.TYPE_LOCK_FRAMES:
                self.locked_type = sub_type

    # ------------------------------------------------------------------
    # 固定设施锚定
    # ------------------------------------------------------------------
    def _calc_anchor(self):
        xs = [p[0] for p in self.facility_obs]
        ys = [p[1] for p in self.facility_obs]
        self.anchor_x = sum(xs) / len(xs)
        self.anchor_y = sum(ys) / len(ys)
        self.is_facility = True
        # 设施静止: 清零速度, 防止滤波残速导致锚点漂移
        self.kf.x[2] = 0.0
        self.kf.x[3] = 0.0

    def _facility_drift(self, rv, t_ms, d):
        """抖动静区钉死 / 大幅持续偏移才重锚 (兼容吊装搬运)"""
        if d <= P.FACILITY_JITTER_DEADZONE:
            self.drift_since = None
        elif d >= P.FACILITY_MAX_DRIFT:
            if self.drift_since is None:
                self.drift_since = t_ms
            elif t_ms - self.drift_since >= P.FACILITY_REANCHOR_MS:
                self.anchor_x = rv.rel_x
                self.anchor_y = rv.rel_y
                self.kf.x[0], self.kf.x[1] = rv.rel_x, rv.rel_y
                self.kf.x[2] = self.kf.x[3] = 0.0
                self.drift_since = None
        else:
            self.drift_since = None

    # ------------------------------------------------------------------
    # 静止证据 (设施锁定/锚定的前提)
    # ------------------------------------------------------------------
    def _stationary(self):
        if len(self.meas_history) < 4:
            return True  # 证据不足按静止处理 (允许冷启动锚定)
        xs = [p[1] for p in self.meas_history]
        ys = [p[2] for p in self.meas_history]
        return (max(xs) - min(xs) < 1.0) and (max(ys) - min(ys) < 1.0)
