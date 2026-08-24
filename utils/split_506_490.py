import json
import math
from typing import List, Tuple, Dict, Any

# ---------------------------- 工具函数 ----------------------------
def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """计算两点间的球面距离（米）"""
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R*c

def compute_length(coords: List[List[float]]) -> float:
    """计算线串总长度（米）"""
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(len(coords)-1):
        total += haversine(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
    return total

def find_point_index(coords: List[List[float]], target: Tuple[float, float], tol: float = 1e-6) -> int:
    """在坐标列表中查找目标点（经纬度），返回索引，未找到则抛出异常"""
    for i, (lon, lat) in enumerate(coords):
        if abs(lon - target[0]) < tol and abs(lat - target[1]) < tol:
            return i
    raise ValueError(f"Point {target} not found in coordinates")

def split_attribute_array(attr_str: str, start_idx: int, end_idx: int) -> str:
    """裁剪逗号分隔的字符串数组，保留索引区间 [start_idx, end_idx]（含两端）"""
    if not attr_str:
        return attr_str
    parts = attr_str.split(',')
    # 去除可能的空格
    parts = [p.strip() for p in parts]
    # 确保索引不越界
    if start_idx < 0:
        start_idx = 0
    if end_idx >= len(parts):
        end_idx = len(parts) - 1
    if start_idx > end_idx:
        return ""
    return ','.join(parts[start_idx:end_idx+1])

def update_feature_properties(props: Dict[str, Any], new_coords: List[List[float]],
                              new_uid: str, start_idx: int, end_idx: int) -> Dict[str, Any]:
    """根据新的坐标和原始属性，生成新属性字典"""
    new_props = props.copy()
    # 更新uid
    new_props['uid'] = new_uid
    # 重新计算总长度
    new_props['total_length'] = compute_length(new_coords)
    # 裁剪数组属性（这些属性每个点一个值）
    array_attrs = ['left_edge_dist', 'right_edge_dist', 'road_edge_dist', 'cross_slope', 'slope']
    for attr in array_attrs:
        if attr in new_props:
            new_props[attr] = split_attribute_array(new_props[attr], start_idx, end_idx)
    return new_props

# ---------------------------- 主处理逻辑 ----------------------------
def process_lane(feature: Dict[str, Any], point1: Tuple[float, float], point2: Tuple[float, float]) -> List[Dict[str, Any]]:
    """拆分一条车道，返回两个新Feature"""
    coords = feature['geometry']['coordinates']
    # 查找两个点的索引
    i1 = find_point_index(coords, point1)
    i2 = find_point_index(coords, point2)
    if i1 > i2:
        i1, i2 = i2, i1  # 确保 i1 < i2
    if i1 == i2:
        raise ValueError("Point1 and point2 are the same, cannot split")

    # 新车道1：从起点到 i1（含）
    coords1 = coords[:i1+1]
    # 新车道2：从 i2 到终点（含）
    coords2 = coords[i2:]

    # 生成新属性
    props = feature['properties']
    uid_base = str(props['uid'])
    new_uid1 = f"{uid_base}_1"
    new_uid2 = f"{uid_base}_2"

    props1 = update_feature_properties(props, coords1, new_uid1, 0, i1)
    props2 = update_feature_properties(props, coords2, new_uid2, i2, len(coords)-1)

    # 构建新Feature
    new_feature1 = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords1
        },
        "properties": props1
    }
    new_feature2 = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords2
        },
        "properties": props2
    }
    return [new_feature1, new_feature2]

def main(input_file: str, output_file: str):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 需要拆分的车道及分割点
    split_tasks = [
        {
            "uid": 137438953506,
            "point1": (113.864865, 22.495516),
            "point2": (113.864764, 22.495806)
        },
        {
            "uid": 137438953490,
            "point1": (113.864903, 22.495526),
            "point2": (113.864801, 22.495823)
        }
    ]

    new_features = []
    removed_uids = set()

    for feature in data['features']:
        uid = feature['properties'].get('uid')
        # 检查是否是需要拆分的车道
        task = None
        for t in split_tasks:
            if t['uid'] == uid:
                task = t
                break
        if task is None:
            # 不是目标车道，直接保留
            new_features.append(feature)
        else:
            # 执行拆分
            print(f"Splitting lane uid={uid}")
            try:
                split_features = process_lane(feature, task['point1'], task['point2'])
                new_features.extend(split_features)
                removed_uids.add(uid)
            except Exception as e:
                print(f"Error splitting lane {uid}: {e}")
                # 如果拆分失败，保留原车道（或选择抛出异常）
                new_features.append(feature)

    # 写入新GeoJSON
    data['features'] = new_features
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"处理完成，结果保存至 {output_file}")
    if removed_uids:
        print(f"已移除并拆分的车道UID: {removed_uids}")

if __name__ == "__main__":
    # 假设输入文件名为 '1-3.geojson'，输出为 '1-3_split.geojson'
    main('../map/1-3.geojson', '../map/1-3_split.geojson')