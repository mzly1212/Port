import struct
import math
from datetime import datetime, timedelta
import pandas as pd
from data import ProcessedFrame
from config import Config


class ProtocolAdapter:
    def __init__(self, config: Config):
        self.config = config
        self.PROTOCOL_VERSION = 120  # V1.20 版本对应数字

    def _convert_core_state(self, v):
        """将引擎内部的物理状态转换为协议要求的单位和坐标"""
        # 1. 坐标转换：相对 UTM -> 全局 UTM -> 经纬度
        local_x = v.x + self.config.ORIGIN_X
        local_y = v.y + self.config.ORIGIN_Y
        lon, lat = self.config.PROJ(local_x, local_y, inverse=True)

        # 2. 速度转换：m/s -> km/h
        speed_kmh = v.v * 3.6

        # 3. 航向角转换：弧度 (-pi~pi) -> 角度 (0~360)
        degree = math.degrees(v.psi)
        if degree < 0:
            degree += 360.0

        # 4. 可信度评估：如果是 AI 预测接管的盲区点，适当降低可信度
        area_reliability = 0.8 if v.is_predicted else 1.0

        return lon, lat, speed_kmh, degree, area_reliability

    def to_webapi_json(self, frame: ProcessedFrame) -> list:
        """
        转换为 WebApi (1.2.1 上传感知数据) 的 JSON 列表格式
        """
        # 转换时间为 2023-05-12T08:54:02.8262 格式
        dt = pd.to_datetime(frame.timestamp_ms, unit='ms') + pd.Timedelta(hours=8)
        time_str = dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:24]

        payload = []
        for v in frame.vehicles:
            lon, lat, speed_kmh, degree, area_rel = self._convert_core_state(v)

            obj_dict = {
                "SysTime": time_str,
                "BeiDouTime": time_str,
                "ObjectId": str(v.fixed_id),
                "ObjectType": str(v.itc_obj_type),  # 使用翻译后的父类
                "SubObjectType": str(v.itc_sub_type),  # 使用翻译后的子类
                "TruckNo": v.plate_num,  # 透传车牌
                "LonX": round(lon, 7),
                "LatY": round(lat, 7),
                "Degree": round(degree, 2),  # 严格透传雷达的正北航向角
                "BodyDegree": 0.0,
                "Speed": round(speed_kmh, 2),
                "DataDmlType": "0",
                "PointsCount": 0,
                "Points": "",
                "LaneNo": v.lane_no,  # 透传车道号
                "AreaReliability": area_rel,
                "ObjectTypeReliability": round(v.type_reliability, 2)
            }
            payload.append(obj_dict)

        return payload

    def to_tcp_bytes(self, frame: ProcessedFrame) -> bytes:
        """
        转换为 TCP协议 (2.2.2. 感知数据包 0x11) 的二进制流
        使用 struct.pack 进行严格的字节对齐打包
        """
        # --- 报文体组装 ---
        body_bytes = bytearray()

        # 数据推送形式 (1 byte): 0 全量
        body_bytes.extend(struct.pack('B', 0))

        # 物体个数 (4 byte Unsigned Int)
        body_bytes.extend(struct.pack('<I', len(frame.vehicles)))

        for v in frame.vehicles:
            lon, lat, speed_kmh, degree, area_rel = self._convert_core_state(v)

            # 物体 ID (32 byte Unsigned char)
            obj_id_bytes = str(v.fixed_id).encode('utf-8').ljust(32, b'\x00')
            body_bytes.extend(obj_id_bytes[:32])

            # 父类(1 byte), 子类(1 byte)
            body_bytes.extend(struct.pack('BB', v.itc_obj_type, v.itc_sub_type))

            # 车牌号 (20 byte)，注意中文字符可能占3个字节，安全截断
            truck_no_bytes = v.plate_num.encode('gbk', errors='ignore')[:20].ljust(20, b'\x00')
            body_bytes.extend(truck_no_bytes)

            # 经度(8), 纬度(8), 航向角(8), 拖挂航向(8), 速度(8) (全是 Double)
            body_bytes.extend(struct.pack('<ddddd', lon, lat, v.radar_heading, 0.0, speed_kmh))
            # body_bytes.extend(struct.pack('<ddddd', lon, lat, degree, 0.0, speed_kmh))

            # 信息类型(1 byte)
            body_bytes.extend(struct.pack('B', 0))  # 0:默认

            # 多边形顶点个数(4 byte) -> 填 0
            body_bytes.extend(struct.pack('<I', 0))
            # 注意：因为顶点个数是0，所以跳过坐标串的拼接

            # 最后感知时间戳 (8 byte Long)
            body_bytes.extend(struct.pack('<q', frame.timestamp_ms))

            # 车道ID (10 byte)
            lane_bytes = v.lane_no.encode('ascii', errors='ignore')[:10].ljust(10, b'\x00')
            body_bytes.extend(lane_bytes)

            # 1.20版本新增：定位精度(8 Double), 类型可信度(8 Double)
            body_bytes.extend(struct.pack('<dd', area_rel, v.type_reliability))

        # --- 报文头组装 ---
        # 字节长度 = 报文体的总长度 + 时间戳(16)
        total_len = len(body_bytes) + 16

        header = struct.pack('<IBIqq',
                             self.PROTOCOL_VERSION,  # 版本号 (4 byte)
                             0x11,  # 指令 ID (1 byte)
                             total_len,  # 字节长度 (4 byte)
                             frame.timestamp_ms,  # 系统时间戳 (8 byte)
                             frame.timestamp_ms  # 北斗时间戳 (8 byte)
                             )

        return header + body_bytes
