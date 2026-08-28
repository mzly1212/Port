# -*- coding: utf-8 -*-
"""
map_model.py — 车道地图模型 (只读)

职责:
  - 加载根目录 map/ GeoJSON, 投影为局部 UTM 米制坐标
  - LINE_SE 起止点校准车道绘制方向 (画反自动 reverse, 移植 v1)
  - 位置 -> 车道匹配 (评分制: 距离 + 航向软惩罚 - 老车道粘滞)
  - 车道切向矢量 (方向裁决的物理基准)

与 v1 的区别: 本模块不维护任何车辆状态, 是纯几何服务。
地图数据只读引用根目录 map/, 避免多份拷贝不一致。
"""
import json
import math

from shapely.geometry import LineString, Point, Polygon

import bootstrap  # noqa: F401  (确保根目录在 sys.path)
from config import Config
from geo_utils import ang_diff_deg
import params as P


class LaneMap:
    def __init__(self, geojson_path=None):
        self.lanes = {}   # uid -> {line, heading_math, length, props}
        self.zones = {}   # zone_name -> shapely Polygon (局部米制)
        self._load_zones()
        self._load_map(geojson_path or Config.LANE_PATH)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def _proj(self, lon, lat):
        x, y = Config.PROJ(lon, lat)
        return x - Config.ORIGIN_X, y - Config.ORIGIN_Y

    def _load_zones(self):
        for name, gps_coords in getattr(Config, 'SPECIAL_AREAS', {}).items():
            pts = [self._proj(lon, lat) for lon, lat in gps_coords]
            if len(pts) >= 3:
                self.zones[name] = Polygon(pts)

    def _load_map(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        line_se = getattr(Config, 'LINE_SE2', {})
        for feature in data.get('features', []):
            coords = feature['geometry']['coordinates']
            props = feature['properties']
            uid = str(props['uid'])

            utm_coords = [self._proj(lon, lat) for lon, lat in coords]

            # LINE_SE 权威起止点校准绘制方向 (移植 v1: 画反自动翻转)
            if uid in line_se and len(utm_coords) >= 2:
                sx, sy = self._proj(*line_se[uid][0])
                head, tail = utm_coords[0], utm_coords[-1]
                if math.hypot(tail[0] - sx, tail[1] - sy) < \
                        math.hypot(head[0] - sx, head[1] - sy):
                    utm_coords.reverse()

            line = LineString(utm_coords)
            self.lanes[uid] = {
                'line': line,
                'length': line.length,
                # 车道航向以线形几何为准: approx_heading 标注与实测切向
                # 偏差可达 120°+ (lane A 实测), 不可作为运行时航向基准。
                # 取中点切向, LINE_SE 已校准线方向。
                'heading_math': self._line_heading(line),
                'props': props,
            }

    @staticmethod
    def _line_heading(line):
        eps = min(1.0, line.length / 2.0)
        s0 = line.length / 2.0
        p0 = line.interpolate(s0)
        p1 = line.interpolate(min(s0 + eps, line.length))
        if p1.distance(p0) < 1e-9:
            p1 = line.interpolate(max(s0 - eps, 0.0))
        return math.degrees(math.atan2(p1.y - p0.y, p1.x - p0.x)) % 360.0

    # ------------------------------------------------------------------
    # 区域判定
    # ------------------------------------------------------------------
    def in_zone(self, x, y, zone_name):
        poly = self.zones.get(zone_name)
        return poly is not None and poly.contains(Point(x, y))

    # ------------------------------------------------------------------
    # 车道匹配 (评分制, 移植 v1 语义, 数学角航向输入)
    # ------------------------------------------------------------------
    def match(self, x, y, ref_heading=None, v=0.0, last_lane_id=None):
        """
        返回 (lane_id, s, signed_l); 无匹配返回 (None, None, None)。
        评分 = 距离 + 航向软惩罚 - 老车道粘滞护城河, 越小越好。
        """
        dynamic_max = max(P.LANE_BIND_BASE_DIST,
                          min(P.LANE_BIND_DYNAMIC,
                              P.LANE_BIND_BASE_DIST + abs(v) * 0.1))
        point = Point(x, y)
        in_t_zone = self.in_zone(x, y, 'AREA_T')

        best = (None, 0.0, 0.0)
        min_score = float('inf')

        for uid, lane in self.lanes.items():
            dist = lane['line'].distance(point)
            if dist > dynamic_max:
                continue

            angle_penalty = 0.0
            if ref_heading is not None:
                angle_diff = ang_diff_deg(lane['heading_math'], ref_heading)
                # 兼容倒车/对向: 180° 反向视为合法
                eff_diff = min(angle_diff, 180.0 - angle_diff)

                threshold = P.LANE_BIND_HEADING_THRESH
                if not in_t_zone and (
                        (last_lane_id is not None and uid == last_lane_id)
                        or dist < 2.0):
                    # 老车道/极近距离放宽 (T 字路口除外, 急转需斩断粘滞)
                    threshold = P.LANE_BIND_HEADING_RELAX
                if eff_diff > threshold:
                    continue
                angle_penalty = eff_diff * P.LANE_ANGLE_WEIGHT

            history_bonus = 1.5 if (last_lane_id is not None
                                    and uid == last_lane_id) else 0.0
            # v1 粘滞量级, 见 params 说明
            history_bonus = P.LANE_HYSTERESIS_M if history_bonus else 0.0

            score = dist + angle_penalty - history_bonus
            if score < min_score:
                min_score = score
                s = lane['line'].project(point)
                best = (uid, s, self.signed_offset(uid, x, y, s))

        return best

    # ------------------------------------------------------------------
    # 几何服务
    # ------------------------------------------------------------------
    def signed_offset(self, lane_id, x, y, s=None):
        """带符号横向偏移: 沿车道方向左侧为正"""
        lane = self.lanes.get(lane_id)
        if lane is None:
            return 0.0
        line = lane['line']
        point = Point(x, y)
        dist = line.distance(point)
        if line.length == 0 or dist == 0:
            return dist
        if s is None:
            s = line.project(point)
        p_line = line.interpolate(s)
        tx, ty = self.tangent(lane_id, s)
        cross = tx * (y - p_line.y) - ty * (x - p_line.x)
        return dist if cross >= 0 else -dist

    def tangent(self, lane_id, s):
        """车道切向单位矢量 (沿 s 增大方向)"""
        lane = self.lanes.get(lane_id)
        if lane is None:
            return 1.0, 0.0
        line = lane['line']
        eps = min(0.1, line.length / 2.0)
        p = line.interpolate(min(max(s, 0.0), line.length))
        if s + eps <= line.length:
            q = line.interpolate(s + eps)
            dx, dy = q.x - p.x, q.y - p.y
        else:
            q = line.interpolate(max(s - eps, 0.0))
            dx, dy = p.x - q.x, p.y - q.y
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            return 1.0, 0.0
        return dx / norm, dy / norm

    def offset_lateral(self, lane_id, s, l):
        """已知车道/纵向/横向偏移反推物理坐标 (signed_offset 的逆运算)"""
        lane = self.lanes.get(lane_id)
        if lane is None:
            return None, None
        line = lane['line']
        if line.length == 0:
            return None, None
        s = max(0.0, min(line.length, s))
        p = line.interpolate(s)
        tx, ty = self.tangent(lane_id, s)
        # 左侧单位法线 (切向逆时针旋转 90°)
        return p.x - ty * l, p.y + tx * l

    def get_xy_from_s(self, lane_id, s):
        lane = self.lanes.get(lane_id)
        if lane is None:
            return None, None
        pt = lane['line'].interpolate(max(0.0, min(lane['length'], s)))
        return pt.x, pt.y

    def lane_heading(self, lane_id):
        lane = self.lanes.get(lane_id)
        return None if lane is None else lane['heading_math']
