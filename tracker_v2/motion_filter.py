# -*- coding: utf-8 -*-
"""
motion_filter.py — 运动滤波层 (全局米制坐标系)

包含两个互不感知的滤波器:
  1. KalmanCV2D     匀速模型卡尔曼: 状态 [x, y, vx, vy]。
                    - 马氏距离门限 + NIS 鲁棒更新: 位置跳变平滑收敛不瞬移
                    - 盲区推演: 速度指数衰减 (v1 已验证的物理一致模型)
  2. HeadingEstimator  雷达航向环形低通 + 前后混淆签名检测:
                    - 环形最短路径滤波, 免疫 0°/360° 交界 180° 甩头
                    - 单帧 >120° 跳变打「翻转标记」(前后混淆专属签名,
                      真实转弯滤波滞后仅 ~36°/帧不可能触发)

设计要点: 航向的最终裁决不在这一层 —— 输出层用 KF 速度矢量
(物理真相) 与本层滤波航向 (雷达观点) 仲裁。倒行/来回翻转问题
在结构上被消灭: 速度矢量不会被任何「重置」清掉。
"""
import math
from collections import deque

import numpy as np

import params as P
from geo_utils import norm_deg, wrap180, ang_diff_deg


class KalmanCV2D:
    """2D 匀速模型卡尔曼滤波器"""

    # 量测矩阵: 只观测位置
    _H = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0]])

    def __init__(self, t_ms, x, y, vx=0.0, vy=0.0):
        self.x = np.array([x, y, vx, vy], dtype=float)
        self.P = np.diag([1.0, 1.0, 9.0, 9.0])   # 初速不确定度 3 m/s
        self.R = np.diag([P.KF_SIGMA_R ** 2] * 2)
        self.last_t = t_ms

    # ------------------------------------------------------------------
    @property
    def pos(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self):
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self):
        return float(math.hypot(self.x[2], self.x[3]))

    # ------------------------------------------------------------------
    def _transition(self, dt):
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        # 分段常加速过程噪声 (白噪声加速度离散化)
        q = P.KF_SIGMA_A ** 2
        G = np.array([[dt * dt / 2.0, 0.0],
                      [0.0, dt * dt / 2.0],
                      [dt, 0.0],
                      [0.0, dt]])
        Q = q * (G @ G.T)
        return F, Q

    def predict(self, t_ms, coasting=False):
        """时间推进; coasting=True 时速度指数衰减 (盲区物理一致推演)"""
        dt = (t_ms - self.last_t) / 1000.0
        if dt <= 0:
            return
        if coasting:
            decay = math.exp(-dt / P.COAST_DECAY_TAU)
            self.x[2] *= decay
            self.x[3] *= decay
            sp = self.speed
            if sp < P.COAST_MIN_SPEED:
                self.x[2] = 0.0
                self.x[3] = 0.0
        F, Q = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.last_t = t_ms

    # ------------------------------------------------------------------
    def innovation_cov(self):
        return self._H @ self.P @ self._H.T + self.R

    def mahalanobis(self, zx, zy):
        """量测到预测位置的马氏距离 (关联门限用)"""
        y = np.array([zx, zy]) - self._H @ self.x
        S = self.innovation_cov()
        return float(math.sqrt(max(y @ np.linalg.solve(S, y), 0.0)))

    def update(self, zx, zy):
        """
        鲁棒量测更新: NIS 超门限时按超出倍数膨胀量测噪声。
        效果: 感知交接的位置前偏 / 缝合重连的跳变以渐进方式
        收敛 (v1 的「追击机制」在这里是滤波器的自然行为)。
        """
        z = np.array([zx, zy])
        y = z - self._H @ self.x
        S = self.innovation_cov()
        nis = float(y @ np.linalg.solve(S, y))

        if nis > P.KF_NIS_GATE:
            S = self._H @ self.P @ self._H.T + self.R * (nis / P.KF_NIS_GATE)

        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self._H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return nis


class HeadingEstimator:
    """雷达航向环形低通滤波 + 前后混淆签名检测 (移植 v1 已验证逻辑)"""

    def __init__(self, initial_heading, t_ms):
        self.filtered = norm_deg(initial_heading)
        self.last_t = t_ms
        self.history = deque(maxlen=10)       # 1 秒原始航向 (杂乱度检测)
        self.flip_times = deque(maxlen=10)    # >120° 单帧跳变时刻 (混淆签名)

    def update(self, raw_heading, t_ms):
        """环形低通 + 噪点门限 + 翻转标记; 返回滤波后航向"""
        self.history.append(raw_heading)
        dt_ms = t_ms - self.last_t

        diff = wrap180(raw_heading - self.filtered)

        # 前后混淆标记: 无论观测是否被噪点门限拒绝, 雷达报出 >120° 的
        # 单帧跳变本身就是「该车航向不可信」的证据 (真实转弯滤波滞后
        # 约 36°/帧, 物理上达不到)
        if abs(diff) > P.ASSOC_FLIP_MARK:
            self.flip_times.append(t_ms)

        # 噪点门限: 连续追踪下单帧突变 >45° 物理上不可能, 拒绝观测
        if dt_ms is not None and dt_ms < 500 \
                and abs(diff) > P.HEADING_NOISE_JUMP:
            self.last_t = t_ms
            return self.filtered

        alpha = P.HEADING_ALPHA
        self.filtered = norm_deg(self.filtered + alpha * diff)
        self.last_t = t_ms
        return self.filtered

    # ------------------------------------------------------------------
    def recently_flipped(self, t_ms):
        """近 HEADING_FLIP_WINDOW_MS 内出现过 >120° 单帧跳变 (前后混淆)"""
        return any(t_ms - ft <= P.ASSOC_FLIP_WINDOW_MS
                   for ft in self.flip_times)

    def erratic(self, thresh=None):
        """最近 1 秒雷达航向是否杂乱无章 (噪点特征)"""
        if thresh is None:
            thresh = P.OFFLANE_HEADING_ERRATIC
        hs = list(self.history)
        if len(hs) < 4:
            return False
        for i in range(len(hs)):
            for j in range(i + 1, len(hs)):
                if ang_diff_deg(hs[i], hs[j]) > thresh:
                    return True
        return False

    def reliable(self, t_ms):
        """航向可作为否决依据: 既无翻转标记也不杂乱"""
        return not self.recently_flipped(t_ms) and not self.erratic()
