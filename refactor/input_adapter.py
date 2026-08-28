import struct
# 导入你刚刚编译生成的 protobuf 模块 (请确保文件名与实际生成的一致)
import pb2

from config import Config
from data import RawFrame, RawVehicle


class NebulaInputAdapter:
    def __init__(self, config: Config):
        self.config = config

        # 协议规定的报文头常量
        self.HEADER_MAGIC = b'\xda\xdb\xdc\xdd'
        self.DATA_TYPE_PERCEPTION = 0x01
        self.PERCEP_TYPE_MEC = 0x07

    def _convert_gps_to_relative(self, lon, lat):
        """将星云互联传来的经纬度转换为算法引擎需要的局部相对坐标"""
        x_utm, y_utm = self.config.PROJ(lon, lat)
        rel_x = x_utm - self.config.ORIGIN_X
        rel_y = y_utm - self.config.ORIGIN_Y
        return rel_x, rel_y

    def parse_udp_packet(self, udp_data: bytes) -> RawFrame:
        """
        解析星云互联的 UDP 数据包：8字节头 + Protobuf 数据体
        """
        # 1. 校验报文长度是否足够包含头部
        if len(udp_data) < 8:
            return None

        # 2. 解析 8 字节报文头
        # 格式: 4字节Magic + 1字节数据类型 + 1字节感知类型 + 2字节长度 (short)
        # 注意：这里使用大端序 '>4sBBH'，如果不通可以尝试小端序 '<4sBBH'
        try:
            magic, data_type, percep_type, payload_length = struct.unpack('>4sBBH', udp_data[:8])
        except struct.error:
            return None

        if magic != self.HEADER_MAGIC:
            # 非法报文头，直接丢弃
            return None

        if data_type != self.DATA_TYPE_PERCEPTION or percep_type != self.PERCEP_TYPE_MEC:
            # 不是MEC感知数据，忽略
            return None

        # 3. 提取 Protobuf Payload
        payload = udp_data[8:8 + payload_length]

        # 4. 反序列化 Protobuf 数据
        msg = pb2.PerceptronSet()  #
        try:
            msg.ParseFromString(payload)
        except Exception as e:
            print(f"⚠️ Protobuf 解析失败: {e}")
            return None

        # 5. 提取业务数据
        frame_time = msg.time_stamp  # 获取时间戳
        raw_vehicles = []

        # 遍历感知目标列表
        for target in msg.perceptron:
            # ==========================================
            # 0 是未 知，1 是机动车，2 是 机动车，3 是行人，4 是 rsu 自身
            # ==========================================
            obj_type_itc = getattr(target, 'object_class_type', 0)

            lon = target.point_gps.object_longitude
            lat = target.point_gps.object_latitude
            if lon < 1.0 or lat < 1.0:
                continue

            rel_x, rel_y = self._convert_gps_to_relative(lon, lat)
            plate_str = target.plate_num.decode('utf-8', errors='ignore') if getattr(target, 'plate_num', b'') else ""
            lane_str = target.lane_id if getattr(target, 'lane_id', '') else ""
            type_rel = getattr(target, 'ptc_Exttype_cfd', 0.0)
            type_rel = type_rel if type_rel > 0 else 1.0

            # ==========================================
            # 2. 车型子类提取与映射 (星云 -> ITC)
            # ==========================================
            ttype = getattr(target, 'ptc_Exttype', 0)

            sub_type_itc = 99  # 默认(未知)

            if ttype == 1:
                sub_type_itc = 11
            elif ttype == 2:
                sub_type_itc = 10
            elif ttype == 3:
                sub_type_itc = 6
            elif ttype == 4:
                sub_type_itc = 7
            elif ttype == 5:
                sub_type_itc = 8
            elif ttype == 6:
                sub_type_itc = 15
            elif ttype == 7:
                sub_type_itc = 9
            elif ttype == 8:
                sub_type_itc = 12
            elif ttype == 9:
                sub_type_itc = 5
            elif ttype == 10:
                sub_type_itc = 2
            elif ttype == 11:
                sub_type_itc = 2
            elif ttype == 12:
                sub_type_itc = 13
            elif ttype == 13:
                sub_type_itc = 14
            elif ttype == 14:
                sub_type_itc = 1
            elif ttype == 15:
                sub_type_itc = 99
            elif ttype == 16:
                sub_type_itc = 13
            else:
                print(f'Unknown ptc_Exttype: {ttype}')

                # 提取雷达原生方向角
            radar_heading = getattr(target, 'object_heading', 0.0) # 东0°, 北90°

            raw_vehicles.append(RawVehicle(
                object_id=target.object_id,
                lon=lon,
                lat=lat,
                rel_x=rel_x,
                rel_y=rel_y,
                itc_obj_type=obj_type_itc,
                itc_sub_type=sub_type_itc,  # 传入经历了三重考验后的精准子类
                plate_num=plate_str,
                lane_no=lane_str,
                radar_heading=radar_heading,
                type_reliability=type_rel
            ))

        return RawFrame(timestamp_ms=frame_time, vehicles=raw_vehicles)
