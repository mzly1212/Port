# -*- coding: utf-8 -*-
"""
track_manager.py — 航迹管理器 (每帧流水线编排)

流水线 (对应 v1 六阶段, 但各阶段解耦):
    1. predict    全航迹时间推进 (盲区航迹速度指数衰减)
    2. associate  量测 <-> 航迹匹配 (ID 跳变缝合/分裂/断联重连
                  坍缩为同一几何关联问题)
    3. create     未匹配量测建新航迹 (含设施误绑定门卫)
    4. coast      未命中航迹 misses++ (下一帧起按盲区衰减推演)
    5. merge      分裂航迹合并 (全对比较, 对向车否决, 静止收紧)
    6. annotate   车道绑定 + 方向裁决 (纯输出注释)
    7. cleanup    超时删除 (tentative/在轨/离道/设施四档)
    8. output     presenter 合成 ProcessedVehicle 列表
"""
import math

import bootstrap  # noqa: F401  (根目录 -> sys.path)
from config import Config

import params as P
from association import Associator
from track import Track
from lane_binding import LaneBinding
from presenter import Presenter
from geo_utils import math_heading_of, ang_diff_deg


def _is_facility(sub_type):
    return sub_type in P.FIXED_FACILITY_SUB_TYPES


class TrackManager:
    def __init__(self, lane_map):
        self.lane_map = lane_map
        self.associator = Associator()
        self.lane_binding = LaneBinding(lane_map)
        self.presenter = Presenter(lane_map)
        self.tracks = {}          # fixed_id -> Track
        self._id_seq = 900000000  # ID 冲突时分配的合成 ID 段

    # ==================================================================
    # 主入口
    # ==================================================================
    def process_frame(self, raw_vehicles, t_ms):
        # ---- 1. 时间推进 ----
        for tr in self.tracks.values():
            tr.predict(t_ms)

        # ---- 2. 关联 ----
        measurements = list(enumerate(raw_vehicles))
        matched, unmatched_idx = self.associator.associate(
            list(self.tracks.values()), measurements, t_ms)

        matched_fids = set()
        for tr, _mi, rv in matched:
            tr.update(rv, t_ms)
            matched_fids.add(tr.fixed_id)

        # ---- 3. 新航迹 ----
        for mi in unmatched_idx:
            self._create_track(raw_vehicles[mi], t_ms)

        # ---- 4. 盲区计数 ----
        for tr in self.tracks.values():
            if tr.fixed_id not in matched_fids:
                tr.misses += 1

        # ---- 5. 分裂合并 ----
        alerts = self._merge_duplicates()

        # ---- 6. 车道注释 ----
        for tr in self.tracks.values():
            self.lane_binding.annotate(tr, t_ms)

        # ---- 7. 生命周期 ----
        self._cleanup(t_ms)

        # ---- 8. 输出 ----
        out = []
        for tr in self.tracks.values():
            pv = self.presenter.synthesize(tr, t_ms)
            if pv is not None:
                out.append(pv)
        return out, alerts

    # ==================================================================
    # 新航迹创建 (含 v1 输入门卫的两条防线)
    # ==================================================================
    def _create_track(self, rv, t_ms):
        # 门卫 1: 边缘盲区冷启动航向初始化 (防新车原地旋转)
        if self.lane_map.in_zone(rv.rel_x, rv.rel_y, 'AREA_INTER_R'):
            rv.radar_heading = getattr(Config, 'ENTRY_HEADING_INTER_R', 162.0)

        # 门卫 2: 设施点误绑定 —— 距新鲜活跃车辆过近的设施量测直接丢弃
        # (关联层已拒绝设施量测认领车辆航迹, 能走到这里的设施点
        #  若贴着活跃车辆, 就是感知把车误报成了设施)
        if _is_facility(rv.itc_sub_type):
            for tr in self.tracks.values():
                if tr.is_facility:
                    continue
                if t_ms - tr.last_update_t > P.FACILITY_MISBIND_FRESH_MS:
                    continue
                d = math.hypot(tr.kf.x[0] - rv.rel_x, tr.kf.x[1] - rv.rel_y)
                if d <= P.FACILITY_MISBIND_DIST:
                    return

        fid = rv.object_id if rv.object_id not in self.tracks else self._alloc_id()
        self.tracks[fid] = Track(fid, rv, t_ms)

    def _alloc_id(self):
        while self._id_seq in self.tracks:
            self._id_seq += 1
        fid = self._id_seq
        self._id_seq += 1
        return fid

    # ==================================================================
    # 分裂航迹合并 (全对比较, v1 教训: 相邻比较会漏)
    # ==================================================================
    def _merge_duplicates(self):
        alerts = []
        tracks = list(self.tracks.values())
        removed = set()
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                a, b = tracks[i], tracks[j]
                if a.fixed_id in removed or b.fixed_id in removed:
                    continue
                senior, junior = self._seniority(a, b)
                if self._mergeable(senior, junior):
                    del self.tracks[junior.fixed_id]
                    removed.add(junior.fixed_id)
                    alerts.append(
                        f"MERGE:{junior.fixed_id}->{senior.fixed_id}")
        return alerts

    @staticmethod
    def _seniority(a, b):
        """保留信息量大的航迹: 命中多者优先, 其次更早创建"""
        if a.hits != b.hits:
            return (a, b) if a.hits > b.hits else (b, a)
        if a.first_seen_t != b.first_seen_t:
            return (a, b) if a.first_seen_t <= b.first_seen_t else (b, a)
        return (a, b) if a.fixed_id < b.fixed_id else (b, a)

    @staticmethod
    def _mergeable(a, b):
        # 设施隔离: 设施只与设施合并 (锚点贴近)
        if a.is_facility or b.is_facility:
            return (a.is_facility and b.is_facility
                    and a.anchor_x is not None and b.anchor_x is not None
                    and math.hypot(a.anchor_x - b.anchor_x,
                                   a.anchor_y - b.anchor_y) <= P.MERGE_DIST)

        ax, ay = a.kf.pos
        bx, by = b.kf.pos
        dist = math.hypot(ax - bx, ay - by)
        if dist > P.MERGE_DIST:
            return False

        sa, sb = a.kf.speed, b.kf.speed
        # 双方均低速: 收紧距离 (静止车队间距 >5m, 同车双报 <3m)
        if sa < 1.0 and sb < 1.0 and dist > P.MERGE_STATIC_DIST:
            return False
        if abs(sa - sb) > P.MERGE_V_DIFF:
            return False

        # 对向车否决: 双方速度已建立且运动方向正对
        if sa >= P.ASSOC_MIN_SPEED_VETO and sb >= P.ASSOC_MIN_SPEED_VETO:
            ha = math_heading_of(*a.kf.vel)
            hb = math_heading_of(*b.kf.vel)
            if ang_diff_deg(ha, hb) > P.MERGE_HEADING_VETO:
                return False
        return True

    # ==================================================================
    # 生命周期
    # ==================================================================
    def _cleanup(self, t_ms):
        for fid in list(self.tracks):
            tr = self.tracks[fid]
            gap = t_ms - tr.last_update_t
            if tr.hits < P.CONFIRM_HITS:
                limit = P.COAST_TENTATIVE_MS
            elif tr.is_facility:
                limit = P.COAST_FACILITY_MS
            elif tr.lane_id is not None:
                limit = P.COAST_ONLANE_MS
            else:
                limit = P.COAST_OFFLANE_MS
            if gap > limit:
                del self.tracks[fid]
