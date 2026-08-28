# -*- coding: utf-8 -*-
"""
geo_utils.py — 环形角度运算 (全模块唯一角度工具)

坐标系约定 (与根目录 input_adapter 契约一致):
  - 雷达航向 / 地图 approx_heading 之外的运行时角度一律用
    「数学角约定: 东=0°, 北=90°」, 即 atan2(dy, dx) 的角度制。
  - 地图 GeoJSON 的 approx_heading 为「北=0°, 东=90°」罗盘约定,
    经 math_heading() 转换后进入运行时。
"""
import math


def norm_deg(a):
    """归一化至 [0, 360)"""
    return a % 360.0


def wrap180(a):
    """归一化至 (-180, 180] (最短几何路径差)"""
    return (a + 180.0) % 360.0 - 180.0


def ang_diff_deg(a, b):
    """两角最短几何夹角 [0°, 180°]"""
    return abs(wrap180(a - b))


def math_heading_of(vx, vy):
    """速度矢量 -> 数学角航向 (度)"""
    return math.degrees(math.atan2(vy, vx)) % 360.0


def compass_to_math(compass_deg):
    """地图罗盘角 (北0°东90°) -> 数学角 (东0°北90°)"""
    return (90.0 - compass_deg) % 360.0


def clamp(v, lo, hi):
    return max(lo, min(hi, v))
