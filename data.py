from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RawVehicle:
    """输入：单辆车原始感知数据"""
    object_id: int
    lon: float          # 原始经度
    lat: float          # 原始纬度
    rel_x: float        # 相对/局部坐标X (由经纬度投影)
    rel_y: float        # 相对/局部坐标Y
    # ==== 修改为严格映射后的透传字段 ====
    itc_obj_type: int  # ITC协议的标准父类
    itc_sub_type: int  # ITC协议的标准子类
    plate_num: str
    lane_no: str
    radar_heading: float  # 雷达原始航向角
    type_reliability: float

@dataclass
class RawFrame:
    """输入：单帧感知数据"""
    timestamp_ms: int
    vehicles: List[RawVehicle]

@dataclass
class ProcessedVehicle:
    """输出：修复后的单辆车数据"""
    original_id: int    # 原始ID (对于被修复的跳变，这是它本来的面目)
    fixed_id: int       # 修复后的统一ID
    x: float
    y: float
    v: float
    psi: float
    is_predicted: bool  # 是否处于预测状态
    # ==== 对应的透传字段 ====
    itc_obj_type: int
    itc_sub_type: int
    plate_num: str
    lane_no: str
    radar_heading: float
    type_reliability: float

@dataclass
class ProcessedFrame:
    """输出：处理后的单帧数据"""
    timestamp_ms: int
    vehicles: List[ProcessedVehicle]
    alerts: List[str]   # 本帧产生的告警(跳变/分裂信息)