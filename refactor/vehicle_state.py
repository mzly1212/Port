# -*- coding: utf-8 -*-
"""
vehicle_state.py — 单目标跟踪状态 (纯状态 + 测量滤波, 不做关联决策)

职责边界:
  - 本模块只维护"一个目标"的状态与滤波, 不感知其他目标的存在
  - 所有关联/去重/车道仲裁决策都在 lane_tracker.py (流水线层)
  - self.s 的写入点收拢为三个: apply_radar_s (测量), advance_s (推演/追击),
    hard_reset_s (变道/上轨硬切) — 消除散落写入导致的交叉污染
"""

import math
from collections import deque

import numpy as np

from tracker_params import (
    FIXED_FACILITY_SUB_TYPES, FACILITY_ANCHOR_SAMPLES,
    FACILITY_JITTER_DEADZONE, FACILITY_MAX_DRIFT, FACILITY_REANCHOR_MS,
    DIR_DEADZONE, DIR_CONFIRM_N,
    TYPE_DECAY, TYPE_LOCK_RATIO, TYPE_LOCK_FRAMES,
    TYPE_FAC_LOCK_RATIO, TYPE_FAC_LOCK_FRAMES, TYPE_FIX_RATIO,
    CHASE_TRIGGER_GAP_MS, CHASE_TRIGGER_DIST,
    HEADING_NOISE_JUMP, OFFLANE_MIN_MOTION, OFFLANE_HEADING_ERRATIC,
    HEADING_FLIP_MARK, HEADING_FLIP_TRUST_MS,
    ang_diff_deg,
)


class VehicleState:
    """单个跟踪目标的全量状态"""

    def __init__(self, obj_id, lane_id, s, l_offset, v, attrs, current_time,
                 rel_x, rel_y, raw_heading):
        # ---- 身份 ----
        self.fixed_id = obj_id
        self.first_seen_time = current_time   # 存活资历 (新车保护期)
        self.last_radar_time = current_time   # 最后真实雷达时刻

        # ---- 车道状态 ----
        self.lane_id = lane_id
        self.s = s                    # 纵向位置 (在轨时的权威坐标)
        self.l = l_offset
        self.filtered_l = l_offset    # 滤波后的横向偏移
        self.last_lane_id = lane_id   # 离道前最后绑定车道
        self.last_s = s
        self.is_off_lane = lane_id is None
        self.went_off_lane_time = None  # 转入离道的时刻 (边缘抖动判定)

        # ---- 变道意图确认状态机 ----
        self.pending_lane_id = None
        self.lane_change_counter = 0

        # ---- 运动状态 ----
        self.v = v
        self.signed_v = 0.0           # 带符号 S 轴速度 (方向状态机输入)
        self.s_history = deque(maxlen=50)      # 5 秒 ST 历史 (测速回归)
        self.s_history.append((current_time, s))
        self.xy_history = deque(maxlen=30)     # 3 秒原始 2D 轨迹 (运动矢量/静止证据)
        self.heading_history = deque(maxlen=10)  # 1 秒原始航向 (杂乱度检测)
        self.heading_flip_times = deque(maxlen=10)  # >120° 航向跳变时刻 (前后混淆标记)

        # ---- 原始测量滤波 ----
        self.raw_x = rel_x
        self.raw_y = rel_y
        self.raw_heading = raw_heading

        # ---- 追击机制 (推演位置与真实位置的温和收敛) ----
        self.target_s = s             # 真实雷达车所在位置 (物理层)
        self.is_chasing = False

        # ---- 行驶方向滞回状态机 ----
        self.drive_direction = 0      # 0=未知, 1=正向, -1=反向
        self.dir_candidate = 0
        self.dir_confirm = 0

        # ---- 车型投票锁定 ----
        self.type_votes = {}
        self.type_obs_count = 0
        self.locked_type = None
        self.locked_obj_type = None

        # ---- 固定设施位置锚定 ----
        self.anchor_x = None
        self.anchor_y = None
        self.anchor_obs = deque(maxlen=FACILITY_ANCHOR_SAMPLES)
        self.drift_since = None

        # ---- 变道/上轨平滑动画 ----
        self.is_changing_lane = False
        self.lc_start_time = 0
        self.lc_duration = 2000
        self.lc_offset_x = 0.0
        self.lc_offset_y = 0.0
        self.lc_offset_heading = 0.0
        self.is_reverse_driving = False    # 海一路借道逆行标志

        # ---- 海一路入轨轨迹拟合缓存 ----
        self.haiyi_fitted_math_angle = None
        self.haiyi_fitted_heading = None

        # ---- 输出缓存 (前端渲染的视觉状态) ----
        self.out_x = None
        self.out_y = None
        self.out_heading = raw_heading
        self.attrs = attrs

    # =====================================================
    # 便捷判定
    # =====================================================
    @property
    def is_fixed_facility(self):
        return self.attrs.get("itc_sub_type", 99) in FIXED_FACILITY_SUB_TYPES

    def age_ms(self, current_time):
        """距最后真实雷达信号的时间"""
        return current_time - self.last_radar_time

    # =====================================================
    # 测量滤波: 坐标 / 航向 / 横向偏移
    # =====================================================
    def update_raw_xy(self, new_x, new_y, current_time=None):
        """
        二维物理坐标预平滑 (送入车道匹配前)。
        同时统一记录原始轨迹 — 运动矢量推导航向 / 静止判定 /
        设施锁定静止证据的唯一来源。
        """
        self.xy_history.append((new_x, new_y))

        if self.is_fixed_facility:
            return self._update_fixed_facility_xy(new_x, new_y, current_time)

        alpha = 0.3  # 30% 信任新雷达点, 70% 沿用上一帧物理惯性
        self.raw_x = self.raw_x * (1 - alpha) + new_x * alpha
        self.raw_y = self.raw_y * (1 - alpha) + new_y * alpha
        return self.raw_x, self.raw_y

    def _update_fixed_facility_xy(self, new_x, new_y, current_time=None):
        """
        固定设施位置锚定: 确认期(中位数锁锚) -> 锚定期(三段式限移)。
        """
        # ---- 阶段 1: 锚点确认期 ----
        if self.anchor_x is None:
            self.anchor_obs.append((new_x, new_y))
            alpha = 0.3
            self.raw_x = self.raw_x * (1 - alpha) + new_x * alpha
            self.raw_y = self.raw_y * (1 - alpha) + new_y * alpha
            if len(self.anchor_obs) >= self.anchor_obs.maxlen:
                xs = [p[0] for p in self.anchor_obs]
                ys = [p[1] for p in self.anchor_obs]
                self.anchor_x = float(np.median(xs))
                self.anchor_y = float(np.median(ys))
                self.raw_x, self.raw_y = self.anchor_x, self.anchor_y
            return self.raw_x, self.raw_y

        # ---- 阶段 2: 锚定期 ----
        drift = math.hypot(new_x - self.anchor_x, new_y - self.anchor_y)

        if drift <= FACILITY_JITTER_DEADZONE:
            # 传感器小波动: 完全忽略, 钉死在锚点
            self.drift_since = None
            self.raw_x, self.raw_y = self.anchor_x, self.anchor_y
        elif drift <= FACILITY_MAX_DRIFT:
            # 中等偏移: 重滤波缓慢收敛 (视觉上近似静止)
            self.drift_since = None
            alpha = 0.05
            self.raw_x = self.raw_x * (1 - alpha) + new_x * alpha
            self.raw_y = self.raw_y * (1 - alpha) + new_y * alpha
        else:
            # 大幅偏移: 大概率是过车误绑定, 拒绝该观测
            if current_time is not None:
                if self.drift_since is None:
                    self.drift_since = current_time
                elif current_time - self.drift_since >= FACILITY_REANCHOR_MS:
                    # 持续大幅偏移超 15 秒: 真实搬运 (如舱盖板被吊走), 重新锚定
                    self.anchor_x, self.anchor_y = new_x, new_y
                    self.raw_x, self.raw_y = new_x, new_y
                    self.drift_since = None
                    print(f'[固定设施重新锚定] ID:{int(self.fixed_id) % 10000} '
                          f'-> ({new_x:.2f}, {new_y:.2f})')

        return self.raw_x, self.raw_y

    def update_raw_heading(self, new_heading, time_diff_ms=None, noise_gate=False,
                           current_time=None):
        """
        航向角环形低通滤波 (解决 0°/360° 交界插值错误)。
        noise_gate=True: 连续追踪下单帧突变超过 HEADING_NOISE_JUMP 度
        直接拒绝 (物理上不可能的横摆率)。
        同时标记 >120° 跳变时刻 (前后混淆签名, 供缝合/去重豁免查询):
        真实转弯的滤波滞后约 36°/帧, 单帧 >120° 只可能是前后混淆。
        """
        self.heading_history.append(new_heading)

        diff = (new_heading - self.raw_heading + 180) % 360 - 180

        # 前后混淆标记: 无论本次观测是否被噪点门限拒绝,
        # 雷达报出 >120° 的单帧跳变本身就是"该车航向不可信"的证据
        if abs(diff) > HEADING_FLIP_MARK and current_time is not None:
            self.heading_flip_times.append(current_time)

        if noise_gate and time_diff_ms is not None and time_diff_ms < 500 \
                and abs(diff) > HEADING_NOISE_JUMP:
            return self.raw_heading

        alpha = 0.05 if self.is_fixed_facility else 0.2
        self.raw_heading = (self.raw_heading + alpha * diff) % 360
        return self.raw_heading

    def heading_recently_flipped(self, current_time, window_ms=None):
        """近 window_ms 内雷达航向是否出现过 >120° 单帧跳变 (前后混淆)。
        真对向车航向稳定永不触发; 触发说明该车雷达航向不可信,
        航向否决依据失效, 缝合/去重应回到位置连续性仲裁。"""
        if window_ms is None:
            window_ms = HEADING_FLIP_TRUST_MS
        return any(current_time - ft <= window_ms
                   for ft in self.heading_flip_times)

    def update_l(self, raw_l):
        """
        横向偏移动态低通: 偏差大(变道中)快跟随, 偏差小(巡航)强抗噪。
        """
        err = abs(raw_l - self.filtered_l)
        alpha = 0.5 if err > 0.6 else 0.2
        self.filtered_l = self.filtered_l * (1 - alpha) + raw_l * alpha
        return self.filtered_l

    # =====================================================
    # 测速与方向状态机
    # =====================================================
    def update_and_estimate_speed(self, current_time, current_s):
        """
        最小二乘回归求 S-T 斜率 (真实车速)。
        仅用 5 秒 ~ 1 秒前的数据, 免疫目标刚出现/拼接时的高频跳动。
        """
        self.s_history.append((current_time, current_s))

        valid_history = [h for h in self.s_history if (current_time - h[0]) > 1000]
        if len(valid_history) < 3:
            return self.v

        t_list = [h[0] for h in valid_history]
        s_list = [h[1] for h in valid_history]
        t0 = t_list[0]
        x = [(t - t0) / 1000.0 for t in t_list]
        y = s_list

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        denominator = (n * sum_xx - sum_x * sum_x)
        if denominator == 0:
            return self.v

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        self.signed_v = slope
        self._update_drive_direction()

        return max(0.0, min(30.0, slope))

    def _update_drive_direction(self):
        """
        行驶方向滞回状态机: |signed_v| 超死区且同方向连续 DIR_CONFIRM_N 次
        才翻转。死区内维持既有方向。
        """
        if abs(self.signed_v) < DIR_DEADZONE:
            return self.drive_direction

        want = 1 if self.signed_v > 0 else -1
        if want == self.drive_direction:
            self.dir_candidate = 0
            self.dir_confirm = 0
            return self.drive_direction

        if want == self.dir_candidate:
            self.dir_confirm += 1
        else:
            self.dir_candidate = want
            self.dir_confirm = 1

        if self.dir_confirm >= DIR_CONFIRM_N:
            self.drive_direction = want
            self.dir_candidate = 0
            self.dir_confirm = 0
        return self.drive_direction

    def reset_drive_direction(self):
        """车道系更换(尤其 S 轴反转)后, 方向需重新建立"""
        self.drive_direction = 0
        self.dir_candidate = 0
        self.dir_confirm = 0

    # =====================================================
    # 测量应用: 纵向位置 (s 写入点 1/3)
    # =====================================================
    def apply_radar_s(self, current_time, raw_s):
        """
        处理真实雷达纵向观测: 测速 + 追击触发/单向棘轮。
        """
        if self.is_fixed_facility:
            # 设施: 纵向直接采用锚定坐标投影, 不参与棘轮/追击/测速
            self.v = 0.0
            self.signed_v = 0.0
            self.target_s = raw_s
            self.s = raw_s
            return

        self.v = self.update_and_estimate_speed(current_time, raw_s)
        self.target_s = raw_s

        was_predicted = (current_time - self.last_radar_time > CHASE_TRIGGER_GAP_MS)

        if not self.is_chasing:
            if was_predicted and abs(self.s - self.target_s) > CHASE_TRIGGER_DIST:
                # 刚恢复且偏差大: 触发追击, s 停在推演位置等待收敛
                self.is_chasing = True
            else:
                # 常态单向棘轮 (由方向状态机裁决, 免疫回归斜率抖动)
                if self.drive_direction > 0:
                    if self.target_s >= self.s:
                        self.s = self.target_s
                elif self.drive_direction < 0:
                    if self.target_s <= self.s:
                        self.s = self.target_s
                else:
                    # 方向未确认: 完全相信雷达
                    self.s = self.target_s

    def hard_reset_s(self, s):
        """
        变道确认/离道上轨的硬切重置 (s 写入点 3/3):
        打断追击、清空测速历史, 防止跨车道坐标系错乱。
        """
        self.s_history.clear()
        self.signed_v = 0.0
        self.s = s
        self.target_s = s
        self.is_chasing = False

    # =====================================================
    # 物理证据 (轨迹校验)
    # =====================================================
    def get_motion_heading(self, min_disp=None):
        """
        从最近 3 秒位移矢量推导运动方向。
        位移不足 min_disp 返回 None (静止, 航向无物理意义)。
        """
        if min_disp is None:
            min_disp = OFFLANE_MIN_MOTION
        if len(self.xy_history) < 8:
            return None
        x0, y0 = self.xy_history[0]
        x1, y1 = self.xy_history[-1]
        dx, dy = x1 - x0, y1 - y0
        if math.hypot(dx, dy) < min_disp:
            return None
        return math.degrees(math.atan2(dy, dx)) % 360

    def is_heading_erratic(self, thresh=None):
        """最近 1 秒雷达航向是否杂乱无章 (噪点特征)"""
        if thresh is None:
            thresh = OFFLANE_HEADING_ERRATIC
        hs = list(self.heading_history)
        if len(hs) < 4:
            return False
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                if ang_diff_deg(hs[i], hs[j]) > thresh:
                    return True
        return False

    def _is_physically_static(self):
        """最近 3 秒原始轨迹是否真实静止 (设施锁定的运动学证据)"""
        if len(self.xy_history) < 15:
            return False
        xs = [p[0] for p in self.xy_history]
        ys = [p[1] for p in self.xy_history]
        return (max(xs) - min(xs)) < 1.0 and (max(ys) - min(ys)) < 1.0

    # =====================================================
    # 车型投票锁定
    # =====================================================
    def vote_type(self, sub_type, obj_type, reliability, current_time):
        """
        置信度加权投票锁定: 双门槛(占比+帧数)达标才锁定;
        压倒性反证可纠正; 设施票需物理静止证据。
        """
        if sub_type is None or sub_type == 99:
            return

        rel = reliability if 0 < reliability <= 1.0 else 0.6

        # 全体票数时间衰减: 持续稳定观测主导, 早期误检退场
        for k in list(self.type_votes.keys()):
            self.type_votes[k] *= TYPE_DECAY
            if self.type_votes[k] < 0.01:
                del self.type_votes[k]

        is_fac = sub_type in FIXED_FACILITY_SUB_TYPES
        if is_fac and self.locked_type is None and not self._is_physically_static():
            # 行驶中目标投来的设施票: 大概率设备误报, 丢弃
            return

        self.type_votes[sub_type] = self.type_votes.get(sub_type, 0.0) + rel
        self.type_obs_count += 1

        if not self.type_votes:
            return
        best_type = max(self.type_votes, key=self.type_votes.get)
        best_votes = self.type_votes[best_type]
        total = sum(self.type_votes.values())
        if total <= 0:
            return
        ratio = best_votes / total

        if self.locked_type is None:
            # ---- 阶段 1: 未锁定 ----
            if best_type in FIXED_FACILITY_SUB_TYPES:
                if ratio >= TYPE_FAC_LOCK_RATIO and self.type_obs_count >= TYPE_FAC_LOCK_FRAMES \
                        and self._is_physically_static():
                    self.locked_type = best_type
                    self.locked_obj_type = obj_type
            elif ratio >= TYPE_LOCK_RATIO and self.type_obs_count >= TYPE_LOCK_FRAMES:
                self.locked_type = best_type
                self.locked_obj_type = obj_type
        else:
            # ---- 阶段 2: 已锁定, 压倒性反证才纠正 ----
            if best_type != self.locked_type and ratio >= TYPE_FIX_RATIO \
                    and best_votes > self.type_votes.get(self.locked_type, 0.0):
                if self.locked_type in FIXED_FACILITY_SUB_TYPES and self._is_physically_static():
                    return
                old_lock = self.locked_type
                self.locked_type = best_type
                self.locked_obj_type = obj_type
                print(f'[车型纠正] ID:{int(self.fixed_id) % 10000} | {old_lock} -> {best_type}')
