#!/usr/bin/env python3
"""
车道线高精地图拆分工具（删除中间 N 个点）

用法：直接修改顶部配置，然后运行 python split_lane.py
"""

import json
import math
import sys
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Optional

# ========== 用户配置区域 ==========
INPUT_FILE = "../map/1-5.geojson"          # 输入文件路径
OUTPUT_FILE = "../map/1-5.geojson"        # 输出文件路径
TARGET_UIDS = [                       # 需要拆分的车道 uid 列表
    137438958178,
    137438958162,
137438958194,
137438958210,
137438958226,
137438961730,
137438961714,
137438961698,
137438961762,
137438961778,
137438961794,
137438961810
]
DELETE_POINTS = 10                     # 中间删除的点数
# ==================================


def calculate_length(coords: List[List[float]]) -> float:
    """计算线串总长度（欧氏距离）"""
    if len(coords) < 2:
        return 0.0
    length = 0.0
    for i in range(len(coords) - 1):
        dx = coords[i+1][0] - coords[i][0]
        dy = coords[i+1][1] - coords[i][1]
        dz = coords[i+1][2] - coords[i][2] if len(coords[i]) > 2 else 0.0
        length += math.sqrt(dx*dx + dy*dy + dz*dz)
    return length


def delete_middle_points(coords: List[List[float]], delete_count: int) -> Optional[Tuple[List[List[float]], List[List[float]]]]:
    """
    从坐标列表中间删除指定数量的连续点（按索引），
    返回前段和后段坐标。若点数不足，返回 None。
    """
    n = len(coords)
    if n <= delete_count:
        print(f"警告: 点数 {n} <= 删除点数 {delete_count}，无法拆分", file=sys.stderr)
        return None

    # 计算删除段的起始索引，使得删除段尽可能居中
    start = (n - delete_count) // 2
    end = start + delete_count  # 删除区间为 [start, end)

    # 前段：0 -> start-1
    first_half = coords[:start]
    # 后段：end -> n-1
    second_half = coords[end:]

    if not first_half or not second_half:
        print(f"警告: 删除后某段为空，无法拆分", file=sys.stderr)
        return None

    return first_half, second_half


def split_feature(feature: Dict[str, Any], uid: str, delete_count: int) -> List[Dict[str, Any]]:
    """拆分单个 Feature，返回两个新 Feature，若失败返回空列表"""
    geom = feature.get('geometry')
    if not geom or geom.get('type') != 'LineString':
        print(f"警告: uid {uid} 不是 LineString，跳过", file=sys.stderr)
        return []

    coords = geom.get('coordinates', [])
    if len(coords) < 2:
        print(f"警告: uid {uid} 点数少于 2，无法拆分", file=sys.stderr)
        return []

    split_result = delete_middle_points(coords, delete_count)
    if split_result is None:
        print(f"警告: uid {uid} 拆分失败（点数不足）", file=sys.stderr)
        return []

    first_coords, second_coords = split_result

    props = deepcopy(feature.get('properties', {}))

    new_feature1 = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": first_coords},
        "properties": deepcopy(props)
    }
    new_feature2 = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": second_coords},
        "properties": deepcopy(props)
    }

    # 更新 uid
    new_feature1['properties']['uid'] = f"{uid}_1"
    new_feature2['properties']['uid'] = f"{uid}_2"

    # 更新总长度
    if 'total_length' in new_feature1['properties']:
        new_feature1['properties']['total_length'] = calculate_length(first_coords)
        new_feature2['properties']['total_length'] = calculate_length(second_coords)

    return [new_feature1, new_feature2]


def main():
    # 读取 GeoJSON
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    if data.get('type') != 'FeatureCollection':
        print("错误: 仅支持 FeatureCollection", file=sys.stderr)
        sys.exit(1)

    features = data.get('features', [])
    uid_set = {str(uid) for uid in TARGET_UIDS}

    new_features = []
    removed_count = 0

    for feature in features:
        props = feature.get('properties', {})
        uid = str(props.get('uid'))
        if uid in uid_set:
            splits = split_feature(feature, uid, DELETE_POINTS)
            if splits:
                new_features.extend(splits)
                removed_count += 1
            else:
                # 拆分失败则保留原样
                new_features.append(feature)
        else:
            new_features.append(feature)

    data['features'] = new_features

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"成功拆分 {removed_count} 条车道（删除中间 {DELETE_POINTS} 个点），结果已保存至 {OUTPUT_FILE}")
    except Exception as e:
        print(f"写入文件失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()