# -*- coding: utf-8 -*-
"""
association.py — 关联层 (量测 <-> 航迹)

v2 的结构性核心: v1 中三套互相纠缠的机制 ——
  - ID 跳变「缝合」(新 ID 认领老车)
  - 分裂「去重」(两个活跃目标合并)
  - 断联「重连」(丢失后找回)
在这里坍缩为同一个几何关联问题: 新量测与既有航迹的匹配。
航迹在盲区中由卡尔曼外推维持, 老车的「幽灵」就是它的预测位置,
因此重连不需要任何特殊缝合代码。

门限 = 马氏距离 (协方差自适应: 盲区越久容忍越大)
     ∪ v1 验证过的重连窗口 (dist < 30 + v·gap, 封顶 70m)
否决 = 航向近乎正对 (>150°, 对向车签名), 带前后混淆豁免:
       老车航向出现过 >120° 单帧跳变时, 雷达航向对它不可信,
       否决依据失效, 回到纯几何仲裁。真对向车航向稳定永不豁免。
"""
import math

import params as P
from geo_utils import ang_diff_deg


def _is_facility(sub_type):
    return sub_type in P.FIXED_FACILITY_SUB_TYPES


class Associator:
    def __init__(self):
        pass

    def associate(self, tracks, measurements, t_ms):
        """
        tracks: 待匹配航迹列表 (已 predict 到 t_ms)
        measurements: [(idx, raw_vehicle), ...]
        返回 (matched_pairs, unmatched_meas_idx)
            matched_pairs: [(track, meas_idx, raw_vehicle), ...]
        """
        pairs = []
        for tr in tracks:
            gap_ms = t_ms - tr.last_update_t
            for mi, rv in measurements:
                cost = self._pair_cost(tr, rv, gap_ms, t_ms)
                if cost is not None:
                    pairs.append((cost, tr, mi, rv))

        # 代价从小到大贪心分配 (全局最近邻的轻量近似, 港口规模 O(n²) 足够)
        pairs.sort(key=lambda p: p[0])
        used_tracks = set()
        used_meas = set()
        matched = []
        for cost, tr, mi, rv in pairs:
            if id(tr) in used_tracks or mi in used_meas:
                continue
            used_tracks.add(id(tr))
            used_meas.add(mi)
            matched.append((tr, mi, rv))

        unmatched = [mi for mi, _ in measurements if mi not in used_meas]
        return matched, unmatched

    # ------------------------------------------------------------------
    def _pair_cost(self, tr, rv, gap_ms, t_ms):
        """计算 (航迹, 量测) 的关联代价; 不可关联返回 None"""
        zx, zy = rv.rel_x, rv.rel_y

        # ---- 固定设施身份隔离 (v1 三重保护, 按观测类型而非锚定状态判定) ----
        # 航迹身份 = 已锚定 或 最近观测为设施类型; 量测身份 = 本帧观测类型。
        # 1. 车辆量测不认领设施航迹   2. 设施量测不认领车辆航迹
        # 3. 未知类型(99) 且 <=5m 的量测允许缝合设施 (兼容 ID 跳变重连)
        tr_fac = tr.is_facility or _is_facility(tr.attrs.get("itc_sub_type", 99))
        meas_fac = _is_facility(rv.itc_sub_type)
        if tr_fac or meas_fac:
            if not (tr_fac and meas_fac):
                if tr_fac and rv.itc_sub_type == 99:
                    ax = tr.anchor_x if tr.anchor_x is not None else tr.kf.x[0]
                    ay = tr.anchor_y if tr.anchor_y is not None else tr.kf.x[1]
                    if math.hypot(zx - ax, zy - ay) > 5.0:
                        return None
                else:
                    return None

        # ---- 几何门限 ----
        px, py = tr.kf.pos
        dist = math.hypot(zx - px, zy - py)
        if dist > P.ASSOC_HARD_DIST:
            return None

        d_maha = tr.kf.mahalanobis(zx, zy)
        chi2_ok = d_maha * d_maha <= P.ASSOC_CHI2_GATE
        if not chi2_ok:
            # 位置证据不足, 先过航向否决 (对向车签名)。
            # ⚠ 卡方门内不否决: 位置在统计上重合的目标物理上不可能是
            # 两辆车, 航向冲突只能解释为雷达前后混淆 —— 此时吸收量测
            # 才能让 HeadingEstimator 记录翻转签名 (否则陷入死锁:
            # 否决阻止更新, 签名永远无法记录, v1 T11 教训)。
            if self._heading_veto(tr, rv, t_ms):
                return None

            if gap_ms > P.ASSOC_STALE_GAP_MS:
                # 断联航迹: v1 验证过的重连窗口兜底 (协方差已膨胀,
                # 卡方门限其实已自适应放大, 此窗口是极端交接偏差的保底)
                rejoin = min(P.ASSOC_REJOIN_MAX,
                             P.ASSOC_REJOIN_BASE + tr.kf.speed * gap_ms / 1000.0)
                if dist >= rejoin:
                    return None
            elif dist > P.ASSOC_SPLIT_CLAIM:
                # 新鲜航迹: 只认极近距离的同车双报 (分裂),
                # 防止重连窗口吞并远处正常出现的新车
                return None

        # ---- 代价: 马氏距离为主, 同 ID 软偏好 ----
        cost = d_maha
        if rv.object_id == tr.fixed_id:
            cost = max(0.0, cost - P.ASSOC_ID_BONUS)
        return cost

    def _heading_veto(self, tr, rv, t_ms):
        """
        航向近乎正对 (>150°) 的两个目标是同车道对向行驶, 绝非同一辆车。
        豁免条件 (缺一即不否决, 回到几何仲裁):
          - 航迹速度未建立 (<1 m/s): 运动证据不足
          - 航迹航向不可信 (近期有 >120° 翻转标记或杂乱): 前后混淆
        """
        if tr.is_facility:
            return False
        if tr.kf.speed < P.ASSOC_MIN_SPEED_VETO:
            return False
        if not tr.heading.reliable(t_ms):
            return False
        return ang_diff_deg(tr.heading.filtered, rv.radar_heading) \
            > P.ASSOC_HEADING_VETO
