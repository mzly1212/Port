import json
import math
from shapely.geometry import LineString, Point, Polygon
from config import Config

from pyproj import Proj


class ZoneManager:
    """
    通用几何区域管理器：
    负责在系统初始化时将经纬度 GPS 多边形一次性投影为局部 UTM 多边形，
    并为上层算法提供高性能、标准化的区域判定 API。
    """

    def __init__(self, special_areas_config):
        self.zones = {}
        for zone_name, gps_coords in special_areas_config.items():
            utm_coords = []
            for lon, lat in gps_coords:
                # 经纬度 -> 局部 UTM 相对物理坐标
                x_utm, y_utm = Config.PROJ(lon, lat)
                utm_coords.append((x_utm - Config.ORIGIN_X, y_utm - Config.ORIGIN_Y))

            # 至少需要3个点才能构成封闭多边形
            if len(utm_coords) >= 3:
                self.zones[zone_name] = Polygon(utm_coords)
                # print(f"📍 特殊几何区域 [{zone_name}] 投影加载完成。")

    def is_in_zone(self, x, y, zone_name):
        """标准 API：判断局部物理坐标 (x, y) 是否在指定名称的区域内"""
        if zone_name not in self.zones:
            return False
        return self.zones[zone_name].contains(Point(x, y))

    def get_zones_for_point(self, x, y):
        """进阶 API：查询指定坐标点所属的所有区域列表（适用于重叠多边形的多重业务触发）"""
        pt = Point(x, y)
        return [name for name, poly in self.zones.items() if poly.contains(pt)]

class MapManager:
    def __init__(self, geojson_path):
        self.lanes = {}  # lane_uid -> dict(LineString, properties)
        self.line_se = Config.LINE_SE2  # 各车道起止点
        self.zone_mgr = ZoneManager(getattr(Config, 'SPECIAL_AREAS', {}))
        self.load_map(geojson_path)


    def load_map(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for feature in data.get('features', []):
            coords = feature['geometry']['coordinates']
            props = feature['properties']
            uid = str(props['uid'])

            # 1. 坐标投影：将经纬度转换为相对 UTM 坐标
            utm_coords = []
            for lon, lat in coords:
                x_utm, y_utm = Config.PROJ(lon, lat)
                utm_coords.append((x_utm - Config.ORIGIN_X, y_utm - Config.ORIGIN_Y))

            # ==========================================
            # 🚀 新增逻辑：利用 LineSE 校准车道线的拓扑方向
            # ==========================================
            if uid in self.line_se and len(utm_coords) >= 2:
                # 获取 LineSE 中规定的权威起点 (lon, lat)
                start_gps = self.line_se[uid][0]

                # 将规定的起点转换为 UTM 相对坐标
                sx_utm, sy_utm = Config.PROJ(start_gps[0], start_gps[1])
                sx_rel = sx_utm - Config.ORIGIN_X
                sy_rel = sy_utm - Config.ORIGIN_Y

                # 取出 GeoJSON 中原始的线头和线尾
                line_head = utm_coords[0]
                line_tail = utm_coords[-1]

                # 计算线头和线尾，谁离“真理起点”更近
                dist_head = math.hypot(line_head[0] - sx_rel, line_head[1] - sy_rel)
                dist_tail = math.hypot(line_tail[0] - sx_rel, line_tail[1] - sy_rel)

                # 如果线尾反而比线头离起点更近，说明原地图的线画反了！
                if dist_tail < dist_head:
                    utm_coords.reverse()  # 原地翻转数组序列，硬性纠正方向！
                    # print(f"⚠️ 车道 {uid} 绘制方向反转，已自动纠正。")


            self.lanes[uid] = {
                'line': LineString(utm_coords),
                'props': props,
                'heading': props.get('approx_heading', 0.0)
            }


    # =======计算带有左右方向的横向偏移======
    def get_signed_offset(self, lane_id, x, y, s=None):
        """计算带方向的横向偏移（沿车道方向，左侧为正，右侧为负）"""
        if lane_id not in self.lanes:
            return 0.0

        line = self.lanes[lane_id]['line']
        point = Point(x, y)
        dist = line.distance(point)

        # 兜底极短车道线或原地坐标
        if line.length == 0 or dist == 0:
            return dist

        if s is None:
            s = line.project(point)

        # 取投影点
        p_line = line.interpolate(s)

        # 向前或向后取 0.1 米，构造车道切向矢量
        epsilon = min(0.1, line.length / 2.0)
        if s + epsilon <= line.length:
            p_ahead = line.interpolate(s + epsilon)
            dx_line = p_ahead.x - p_line.x
            dy_line = p_ahead.y - p_line.y
        else:
            p_behind = line.interpolate(s - epsilon)
            dx_line = p_line.x - p_behind.x
            dy_line = p_line.y - p_behind.y

        # 构造从投影点到实际车辆坐标的相对矢量
        dx_point = x - p_line.x
        dy_point = y - p_line.y

        # 2D 叉乘判断左右
        cross_product = dx_line * dy_point - dy_line * dx_point

        return dist if cross_product >= 0 else -dist

    def offset_lateral(self, lane_id, s, l):
        """
        已知车道、纵向距离 s 与带符号横向偏移 l(左正右负)，反推真实物理坐标。
        与 get_signed_offset 的符号定义严格一致，互为逆运算：
        车道切向逆时针旋转 90° 即"左侧"法线方向。
        """
        if lane_id not in self.lanes:
            return None, None
        line = self.lanes[lane_id]['line']
        if line.length == 0:
            return None, None

        s = max(0.0, min(line.length, s))
        p_line = line.interpolate(s)

        # 构造车道切向矢量 (取样方式与 get_signed_offset 保持一致)
        epsilon = min(1.0, line.length / 2.0)
        if s + epsilon <= line.length:
            p_ahead = line.interpolate(s + epsilon)
            dx_line = p_ahead.x - p_line.x
            dy_line = p_ahead.y - p_line.y
        else:
            p_behind = line.interpolate(s - epsilon)
            dx_line = p_line.x - p_behind.x
            dy_line = p_line.y - p_behind.y

        norm = math.hypot(dx_line, dy_line)
        if norm < 1e-9:
            return p_line.x, p_line.y

        # 左侧单位法线 (与 get_signed_offset 的"左正右负"定义互逆)
        nx = -dy_line / norm
        ny = dx_line / norm
        return p_line.x + nx * l, p_line.y + ny * l

    def match_to_lane(self, x, y, veh_heading=None, v=0.0, last_lane_id=None, base_max_dist=3.0, forced_lane=None):
        """
        强化版核心映射：带有动态阈值、航向约束和历史粘滞的综合评分系统
        veh_heading: 北0°  东90°
        """
        # 1. 动态距离阈值: 根据速度自适应扩大搜索半径 (假设 108km/h = 30m/s 时额外放宽 3米)
        dynamic_max_dist = max(base_max_dist, min(6.0, base_max_dist + abs(v) * 0.1))

        point = Point(x, y)
        best_lane = None
        min_score = float('inf')
        best_s = 0.0
        best_l_real = 0.0  # 记录真实的横向偏移

        for uid, lane_data in self.lanes.items():

            # ==========================================
            # 🚀 新增：执行向量夹角绝对仲裁
            # 如果是海一路范围内的入轨匹配，且由于向量夹角已经确定了唯一目标车道，
            # 则对海一路的“另一条对向落选车道”实施一票否决，直接跳过！
            # ==========================================
            if forced_lane is not None and uid in {'137438953490_1', '137438953490_2', '137438953506_1', '137438953506_2'}:
                if uid != forced_lane:
                    continue  # 绝对不给落选车道任何竞争机会！


            line = lane_data['line']
            dist = line.distance(point)

            if dist > dynamic_max_dist:
                continue

            # 2. 航向约束与惩罚
            angle_penalty = 0.0
            if veh_heading is not None:
                map_heading = lane_data['heading']
                base_geo_heading = (90 - map_heading) % 360

                # 计算两者的最小夹角 (0~180度)
                angle_diff = abs((base_geo_heading - veh_heading + 180) % 360 - 180)

                # 兼容会车错乱或者正常倒车的情况(角度差接近 180 度)
                effective_angle_diff = min(angle_diff, 180 - angle_diff)


                # [Fix: 航向角瞬时噪点豁免机制]
                # 默认最大允许 45 度偏差。
                angle_threshold = 45.0

                # 🚀 规范化调用：向 ZoneManager 询问当前点是否在 T 字交叉路口区域
                in_t_area = self.zone_mgr.is_in_zone(x, y, 'AREA_T')

                if not in_t_area:
                    # 只有在非 T 字区域的普通路段，才赋予老车道或极近距离的 90 度航向角豁免
                    # T 字路口急转弯时，绝对禁止 90 度豁免，必须用严格的 45 度斩断老车道粘滞！
                    if (last_lane_id is not None and uid == last_lane_id) or dist < 2:
                        angle_threshold = 90.0

                if effective_angle_diff > angle_threshold:  # 如果车辆实际航向与车道走向夹角大于动态阈值，拒绝匹配！
                    print('angle_threshold continue')
                    continue

                # 在 0~45 度之间，角度偏差越大，加一定的惩罚权重 (每度相当于偏离 0.02 米)
                angle_penalty = effective_angle_diff * 0.2 # 0.02

            # 3. 历史车道“粘滞”优势 (防止在两条相邻车道线中间画龙跳动)
            history_bonus = 0.0
            if last_lane_id is not None and uid == last_lane_id:
                # 赋予老车道 1.5 米的“护城河”。新车道必须比老车道近至少 1.5m 才能发生换道。
                history_bonus = 1.5 # 1.5

                # 综合评分 (越小越好) = 物理距离 + 角度惩罚 - 历史优势
            score = dist + angle_penalty - history_bonus

            if score < min_score:
                min_score = score
                best_lane = uid
                best_s = line.project(point)
                best_l_real = dist  # 返回真实的物理距离，而不是加了权重的 score

        # ==========================================
        # 🚀 确定最优车道后，一次性为其赋予左正右负的符号
        # ==========================================
        if best_lane is not None:
            best_l_real = self.get_signed_offset(best_lane, x, y, best_s)

        return best_lane, best_s, best_l_real

    def get_xy_from_s(self, lane_id, s):
        """已知车道和纵向距离，反推平滑后的物理坐标"""
        if lane_id not in self.lanes:
            return None, None
        line = self.lanes[lane_id]['line']
        pt = line.interpolate(s)
        return pt.x, pt.y

if __name__ == '__main__':
    map_manager = MapManager(Config.LANE_PATH)
    x_utm, y_utm = Config.PROJ(113.8660123,
      22.4922919)
    x = x_utm - Config.ORIGIN_X
    y = y_utm - Config.ORIGIN_Y
    print(map_manager.zone_mgr.is_in_zone(x, y, 'AREA_T'))
    print(map_manager.match_to_lane(x, y))