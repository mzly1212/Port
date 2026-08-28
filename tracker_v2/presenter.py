# -*- coding: utf-8 -*-
"""
presenter.py — 输出合成层 (航向仲裁 / 限转速 / 前端视图组装)

航向仲裁规则 (v1 三重防线在统一框架下的重建):

  在轨且方向已建立:
      基准 = 车道切向 * dir (物理事实, 免疫雷达前后混淆)
      倒车豁免: 雷达航向可靠且与基准近乎相反 (>90°) -> 用雷达航向
      (车在倒着开, 车头朝向不变, 这是 v1 倒车用例的正确行为)

  离道 / 方向未建立:
      静止 (v < 0.5 m/s)      -> 冻结航向 (原地感知翻转不进画面)
      运动且雷达航向杂乱      -> 运动矢量纠正 (噪点特征)
      运动且雷达与运动相反    -> 雷达航向 (倒行)
      其余                    -> 运动矢量

  最后统一限转速 HEADING_MAX_RATE (度/秒), 无论上游多离谱,
  前端每帧最多转 4.5° —— 这是观感的最终保证。
"""
import bootstrap  # noqa: F401  (根目录 -> sys.path)
from data import ProcessedVehicle

import params as P
from geo_utils import math_heading_of, norm_deg, wrap180, ang_diff_deg, clamp


class Presenter:
    def __init__(self, lane_map):
        self.map = lane_map

    # ------------------------------------------------------------------
    def synthesize(self, tr, t_ms):
        # 确认门槛: 两连击才上画面 (单帧噪点永不可见)
        if tr.hits < P.CONFIRM_HITS:
            return None
        if t_ms - tr.first_seen_t < P.NEW_TRACK_GRACE_MS:
            return None

        if tr.is_facility:
            x, y = tr.anchor_x, tr.anchor_y
            v = 0.0
        else:
            x, y = tr.kf.pos
            v = tr.kf.speed

        psi = self._resolve_heading(tr, t_ms)
        is_pred = tr.misses > 0 \
            and (t_ms - tr.last_update_t) > P.PREDICTED_FLAG_MS

        sub = tr.locked_type if tr.locked_type is not None \
            else tr.attrs.get("itc_sub_type", 99)

        return ProcessedVehicle(
            original_id=tr.last_object_id,
            fixed_id=tr.fixed_id,
            x=x, y=y, v=v,
            psi=psi,
            is_predicted=is_pred,
            itc_obj_type=tr.attrs.get("itc_obj_type", 1),
            itc_sub_type=sub,
            plate_num=tr.attrs.get("plate_num", ""),
            lane_no=tr.attrs.get("lane_no", ""),
            radar_heading=psi,  # 兼容 v1 前端字段: 修复后的展示航向
            type_reliability=tr.attrs.get("type_reliability", 0.9),
        )

    # ------------------------------------------------------------------
    # 航向仲裁
    # ------------------------------------------------------------------
    def _resolve_heading(self, tr, t_ms):
        vx, vy = tr.kf.vel
        speed = tr.kf.speed
        radar_h = tr.heading.filtered
        reliable = tr.heading.reliable(t_ms)
        motion_h = math_heading_of(vx, vy) \
            if speed >= P.MOTION_HEADING_MIN_SPEED else None

        if tr.is_facility:
            target = radar_h
        elif tr.lane_id is not None and tr.dir is not None:
            # 在轨: 车道切向为基准 (免疫雷达前后混淆)
            tx, ty = self.map.tangent(tr.lane_id, tr.s)
            base = math_heading_of(tx, ty)
            if tr.dir < 0:
                base = norm_deg(base + 180.0)
            if reliable and ang_diff_deg(radar_h, base) > P.HEADING_MOTION_CONFLICT:
                target = radar_h  # 倒车: 车头朝向以雷达为准
            else:
                target = base
        else:
            # 离道 / 方向未建立
            if speed < P.OFFLANE_STATIC_SPEED:
                # 静止冻结: 原地航向翻转不进画面
                target = tr.out_heading if tr.out_heading is not None else radar_h
            elif motion_h is not None:
                if tr.heading.erratic():
                    target = motion_h          # 雷达杂乱 -> 运动矢量纠正
                elif reliable and ang_diff_deg(radar_h, motion_h) \
                        > P.HEADING_MOTION_CONFLICT:
                    target = radar_h           # 倒行
                else:
                    target = motion_h
            else:
                target = radar_h

        # ---- 输出限转速 (最终防线) ----
        if tr.out_heading is None or tr.last_output_t is None:
            tr.out_heading = norm_deg(target)
        else:
            dt = max((t_ms - tr.last_output_t) / 1000.0, 0.001)
            max_d = P.HEADING_MAX_RATE * dt
            d = wrap180(target - tr.out_heading)
            tr.out_heading = norm_deg(tr.out_heading + clamp(d, -max_d, max_d))
        tr.last_output_t = t_ms
        return tr.out_heading
