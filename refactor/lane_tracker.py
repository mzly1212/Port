# -*- coding: utf-8 -*-
"""
lane_tracker.py — 感知平滑跟踪器 (六阶段流水线)

对外接口与原版完全一致:
    LaneQueueTracker(map_manager).process_frame(raw_vehicles, current_time)
        -> List[ProcessedVehicle]

流水线结构 (每帧按序执行, 各阶段职责单一):
    Stage 1  输入门卫     _gate_keep          入场航向初始化 / 设施身份防劫持
    Stage 2  ID 关联      _resolve_identity   海一路仲裁 / 预滤波 / 车道匹配 /
                                                 防劫持护盾 / 2D 缝合
    Stage 3  测量应用     _apply_measurement  同车道 / 变道确认 / 离道上轨
    Stage 4  物理推演     _advance_physics    离道去重 / 490拦截 / 在轨去重 /
                                                 断联推演 / 追击收敛
    Stage 5  生命周期     _cleanup_stale      双轨超时清理
    Stage 6  输出合成     _generate_output    坐标融合 / 航向合成 / 输出平滑

与旧版的行为差异 (有意修复的功能冲突):
    1. 2D 缝合基线统一: 旧版被缝合的点用"原始坐标"参与后续计算, 与老车下帧
       的"滤波坐标"存在 1~2m 基线差, 造成缝合瞬间 S 跳变。新版缝合后立即用
       老车的滤波器重新处理观测并重新匹配车道, 测量基线全程一致。
    2. 对向车误合并防护 (全局): 缝合与在轨去重均增加 >150° 航向正对否决。
       旧版仅海一路有航向否决, 其他车道对向车会被近距离缝合/去重吞并;
       新车测速暂态期 v 偏低, 仅靠速度差会漏判。150° 阈值不阻断感知交界处
       同车双报的合法缝合 (历史教训: 90°~120° 曾误伤, 已回退)。
    3. 缝合重连跳变平滑: 同车道重连位置跳变 >2m 时启动输出过渡动画
       (复用变道动画机制), 消除缝合瞬间的画面瞬移; 保留测速历史与方向
       状态不清空, 纵向收敛交给追击/棘轮机制, 与旧版一致。
       (教训: 曾在缝合时 hard_reset 清历史, ID 跳变高发区方向状态机
        永远无法确认, 车头退回雷达航向兜底而来回翻转/倒行, 已回退)
    4. 车道边缘短暂离道 (<1s 重回原车道) 不硬重置方向状态与测速历史:
       按同车道测量继续处理, 防止边缘抖动导致车头来回翻转。
    5. 海一路车道集合/阈值统一为 tracker_params 常量, 消除 5 处重复定义。
    6. lane_queues 从实例字段降级为局部变量 (它每帧重建, 从未被跨帧读取)。
"""

import math

import numpy as np
from shapely.geometry import Point

from data import ProcessedVehicle
from config import Config
from vehicle_state import VehicleState
from tracker_params import (
    FIXED_FACILITY_SUB_TYPES, FACILITY_TIMEOUT_MS, FACILITY_REFRESH_MS,
    PRED_SPEED_DECAY_TAU, PRED_MIN_SPEED,
    PRED_MAX_GAP_BASE, PRED_MAX_GAP_RATIO, CHASE_MAX_SPEED,
    IDM_BRAKE_DIST, IDM_BRAKE_DECEL,
    SUTURE_REJOIN_DIST_BASE, SUTURE_REJOIN_DIST_MAX, SUTURE_GAP_MS,
    SUTURE_SPLIT_DIST, SUTURE_FACILITY_UNKNOWN_DIST,
    SUTURE_HEADING_VETO_GLOBAL, SUTURE_REJOIN_ANIM_MAX, CHASE_TRIGGER_DIST,
    DEDUP_LANE_S_WINDOW, DEDUP_LANE_V_DIFF, DEDUP_OFFLANE_DIST,
    HEADING_MAX_RATE, HEADING_MOTION_CONFLICT,
    LANE_CHANGE_CONFIRM_FRAMES, LANE_CHANGE_CONFIRM_DIST, REENTRY_BLIP_MS,
    NEW_VEHICLE_GRACE_MS, OFFLANE_TIMEOUT_MS, ONLANE_TIMEOUT_MS,
    PREDICTED_FLAG_MS, RECENT_GAP_MS,
    OUTPUT_SMOOTH_ALPHA, OUTPUT_HARD_CUT_DIST,
    HAIYI_LANES, HAIYI_490_LANES, HAIYI_490_1, HAIYI_506_1, HAIYI_490_2,
    HAIYI_FIT_MIN_DIST, HAIYI_FIT_MAX_DIST,
    HIJACK_LANE_JUMP_L, HIJACK_HEADING_JUMP, HIJACK_DT_SEC,
    SUTURE_HEADING_VETO, SUTURE_HEADING_VETO_MS,
    HAIYI_DEPART_L, HEADING_FALLBACK_FLIP,
    MATCH_BASE_MAX_DIST,
    ang_diff_deg, geo_heading_of,
)

# ===== 特殊业务车道: 堆场内车道 =====
# 业务上这些车道不存在逆行, 车头航向严格锁定为车道正向:
# 跳过方向状态机 -1 分支 / 逆行先验 / 雷达反向兼容等一切翻转逻辑。
# (车道清单在 config.py SPECIAL_LINES['DUICHANG'], 此处只读引用)
DUICHANG_LANES = frozenset(
    getattr(Config, 'SPECIAL_LINES', {}).get('DUICHANG', []))


class LaneQueueTracker:
    def __init__(self, map_manager):
        self.last_update_time = None
        self.map_mgr = map_manager
        self.active_vehicles = {}  # fixed_id -> VehicleState

    # =====================================================
    # 主入口: 六阶段编排
    # =====================================================
    def process_frame(self, raw_vehicles, current_time):
        current_radar_ids = set()

        for rv in raw_vehicles:
            # ---- Stage 1: 输入门卫 ----
            if not self._gate_keep(rv, current_time):
                continue

            # ---- Stage 2: ID 关联 (含 2D 缝合与缝合后的基线重整) ----
            assoc = self._resolve_identity(rv, current_time)
            if assoc is None:
                continue  # 劫持/无效点, 直接丢弃
            fixed_id, old_veh, lane_id, s, l, match_x, match_y, attrs = assoc

            current_radar_ids.add(fixed_id)

            # ---- Stage 3: 测量应用 ----
            if lane_id is None:
                self._update_off_lane_vehicle(fixed_id, rv, attrs, current_time)
                continue

            if old_veh is None:
                self.active_vehicles[fixed_id] = VehicleState(
                    fixed_id, lane_id, s, l, 0.0, attrs, current_time,
                    rv.rel_x, rv.rel_y, rv.radar_heading)
            else:
                self._apply_measurement(old_veh, rv, lane_id, s, l,
                                        match_x, match_y, attrs, current_time)
                self._update_vehicle_type(old_veh, rv, attrs, current_time)
                old_veh.attrs = attrs
                old_veh.last_radar_time = current_time
                old_veh.is_off_lane = False

        # ---- Stage 4: 物理推演 (去重 -> 拦截 -> 推演/追击) ----
        self._advance_physics(current_time, current_radar_ids)

        # ---- Stage 5: 生命周期 ----
        self._cleanup_stale_vehicles(current_time)

        # ---- Stage 6: 输出合成 ----
        return self._generate_processed_vehicles(current_time)

    # =====================================================
    # Stage 1: 输入门卫
    # =====================================================
    def _gate_keep(self, rv, current_time):
        """
        帧级输入过滤, 返回 False 表示丢弃该雷达点:
        a) 新车在边缘盲区 → 强制初始化航向 (防冷启动原地旋转)
        b) 设施身份防劫持: 已确认车辆收到设施(13/14)点 → 误绑定, 丢弃
           (反向"车辆点绑到设施 ID"由位置锚定机制冻结, 无需丢弃)
        """
        fixed_id = rv.object_id
        old_veh = self.active_vehicles.get(fixed_id)

        if old_veh is None:
            if self.map_mgr.zone_mgr.is_in_zone(rv.rel_x, rv.rel_y, 'AREA_INTER_R'):
                rv.radar_heading = getattr(Config, 'ENTRY_HEADING_INTER_R', 162.0)
            return True

        # ---- 固定设施低频刷新节流 ----
        # 设施数据来自第三方低频专用通道, 状态每分钟刷新一次已足够。
        # 非刷新窗口内的设施量测仅作心跳续命 (防超时清理), 位姿保持不变
        # —— 位置不动 + 航向冻结, 前端图标完全静止。
        if old_veh.is_fixed_facility and rv.itc_sub_type in FIXED_FACILITY_SUB_TYPES:
            if current_time - old_veh.last_facility_refresh_t < FACILITY_REFRESH_MS:
                old_veh.last_radar_time = current_time   # 心跳续命
                return False                              # 跳过本帧量测
            old_veh.last_facility_refresh_t = current_time

        _old_sub = old_veh.attrs.get("itc_sub_type", 99)
        if rv.itc_sub_type in FIXED_FACILITY_SUB_TYPES \
                and _old_sub not in FIXED_FACILITY_SUB_TYPES and _old_sub != 99:
            return False

        return True

    # =====================================================
    # Stage 2: ID 关联
    # =====================================================
    def _resolve_identity(self, rv, current_time):
        """
        完成单点的身份解析与车道匹配, 返回关联结果元组:
            (fixed_id, old_veh, lane_id, s, l, match_x, match_y, attrs)
        返回 None 表示该点被防劫持护盾拒绝。
        """
        fixed_id = rv.object_id
        old_veh = self.active_vehicles.get(fixed_id)

        # 海一路西侧入轨轨迹拟合与向量夹角车道仲裁
        forced_haiyi_lane, hijacked_heading = self._haiyi_arbitrate(old_veh, rv)

        # 坐标/航向双层预滤波 (劫持航向优先喂给滤波器)
        input_heading = hijacked_heading if hijacked_heading is not None else rv.radar_heading
        if old_veh is not None:
            match_x, match_y = old_veh.update_raw_xy(rv.rel_x, rv.rel_y, current_time)
            match_heading = old_veh.update_raw_heading(input_heading,
                                                       current_time=current_time)
        else:
            match_x, match_y = rv.rel_x, rv.rel_y
            match_heading = input_heading

        veh_v = old_veh.v if old_veh else 0.0
        last_lane = old_veh.lane_id if old_veh else None

        lane_id, s, l = self.map_mgr.match_to_lane(
            match_x, match_y,
            veh_heading=match_heading,
            v=veh_v,
            last_lane_id=last_lane,
            base_max_dist=MATCH_BASE_MAX_DIST,
            forced_lane=forced_haiyi_lane,
        )

        # 防劫持护盾 (海一路会车错乱, 丢弃而非生成临时 ID)
        if old_veh is not None and self._is_hijacked(old_veh, rv, lane_id, current_time):
            return None

        # 全域 2D 物理缝合: 新 ID 认领已有目标 (T 路口除外)
        if fixed_id not in self.active_vehicles \
                and not self.map_mgr.zone_mgr.is_in_zone(match_x, match_y, 'AREA_T'):
            matched_id = self._find_best_match_2d(match_x, match_y, rv.radar_heading,
                                                  current_time,
                                                  incoming_sub_type=rv.itc_sub_type)
            if matched_id is not None:
                fixed_id = matched_id
                old_veh = self.active_vehicles[matched_id]
                # =================================================
                # 缝合基线统一 (本版核心修复):
                # 旧版缝合点直接沿用"原始坐标"的匹配结果, 与老车下帧的
                # "滤波坐标"基线差 1~2m, 造成缝合瞬间 S 跳变。
                # 现在用老车的滤波器重新处理观测并重新匹配, 基线全程一致。
                # =================================================
                match_x, match_y = old_veh.update_raw_xy(rv.rel_x, rv.rel_y, current_time)
                match_heading = old_veh.update_raw_heading(rv.radar_heading,
                                                            current_time=current_time)
                lane_id, s, l = self.map_mgr.match_to_lane(
                    match_x, match_y,
                    veh_heading=match_heading,
                    v=old_veh.v,
                    last_lane_id=old_veh.lane_id,
                    base_max_dist=MATCH_BASE_MAX_DIST,
                )
                # ---- 缝合重连跳变平滑 ----
                # 同车道重连但位置跳变超过阈值: 输出从上帧视觉位置过渡到
                # 新轨迹 (复用变道动画), 消除缝合瞬间的画面瞬移。
                # ⚠ 绝不在此清空测速历史 (hard_reset_s): ID 跳变高发区若
                # 每次缝合都清历史, 方向状态机永远无法确认(需 >1s 的历史
                # 点 + 5 帧确认), 车头会退化到雷达航向兜底而来回翻转/倒行。
                # 纵向跳变交给 apply_radar_s 的追击/棘轮机制自然收敛,
                # 与旧版行为一致。
                if lane_id is not None and lane_id == old_veh.lane_id \
                        and abs(s - old_veh.s) > CHASE_TRIGGER_DIST:
                    self._start_rejoin_animation(old_veh, lane_id, s, current_time)

        attrs = {
            "itc_obj_type": rv.itc_obj_type,
            "plate_num": rv.plate_num,
            "lane_no": rv.lane_no,
            "type_reliability": rv.type_reliability,
            "itc_sub_type": rv.itc_sub_type,
        }

        return fixed_id, old_veh, lane_id, s, l, match_x, match_y, attrs

    def _is_hijacked(self, old_veh, rv, lane_id, current_time):
        """海一路专属会车防错乱护盾 (仅 490/506 车道生效)"""
        dt_sec = max((current_time - old_veh.last_radar_time) / 1000.0, 0.1)

        if old_veh.lane_id not in HAIYI_LANES:
            return False

        # 检查 A: 在老车道中心却瞬跳对向车道 (无合法借道意图)
        if lane_id in HAIYI_LANES and lane_id != old_veh.lane_id:
            if old_veh.lane_change_counter == 0 \
                    and abs(old_veh.filtered_l) < HIJACK_LANE_JUMP_L:
                print('hijack fixed: A')
                return True

        # 检查 B: 雷达航向突变 > 90 度且 dt < 1s (非逆行)
        heading_jump = ang_diff_deg(old_veh.raw_heading, rv.radar_heading)
        if heading_jump > HIJACK_HEADING_JUMP and dt_sec < HIJACK_DT_SEC \
                and not old_veh.is_reverse_driving:
            print('hijack fixed: B')
            return True

        return False

    def _haiyi_arbitrate(self, old_veh, rv):
        """
        海一路西侧入轨轨迹拟合与向量夹角车道仲裁。
        返回 (forced_lane, hijacked_heading):
            forced_lane       向量裁决出的目标车道 (或 None)
            hijacked_heading  拟合出的干净航向, 用于替代雷达瞬时航向 (或 None)
        """
        forced_haiyi_lane = None
        hijacked_heading = None

        if old_veh is None:
            return forced_haiyi_lane, hijacked_heading

        if HAIYI_506_1 not in self.map_mgr.lanes or HAIYI_490_2 not in self.map_mgr.lanes:
            return forced_haiyi_lane, hijacked_heading

        line_506 = self.map_mgr.lanes[HAIYI_506_1]['line']
        pt = Point(rv.rel_x, rv.rel_y)
        dist_506 = line_506.distance(pt)

        # 1. 蓝色感知区 (2.5m ~ 8.0m) 内持续线性拟合
        if HAIYI_FIT_MIN_DIST <= dist_506 <= HAIYI_FIT_MAX_DIST and old_veh.is_off_lane:
            if len(old_veh.xy_history) >= 2:
                xs = [p[0] for p in old_veh.xy_history]
                ys = [p[1] for p in old_veh.xy_history]

                if max(xs) - min(xs) > 0.01 or max(ys) - min(ys) > 0.01:
                    m, b = np.polyfit(xs, ys, 1)
                    dx = xs[-1] - xs[0]
                    dy = ys[-1] - ys[0]

                    vec_base = np.array([1, m])
                    vec_points = np.array([dx, dy])
                    direction_sign = 1 if np.dot(vec_base, vec_points) >= 0 else -1
                    vec_fit = direction_sign * vec_base

                    theta_fit_deg = math.degrees(math.atan2(vec_fit[1], vec_fit[0]))
                    old_veh.haiyi_fitted_math_angle = theta_fit_deg
                    old_veh.haiyi_fitted_heading = (90 - theta_fit_deg) % 360

        elif dist_506 > HAIYI_FIT_MAX_DIST:
            old_veh.haiyi_fitted_math_angle = None
            old_veh.haiyi_fitted_heading = None

        # 2. 逼近入轨界限 (< 2.5m), 向量夹角绝对仲裁
        if dist_506 < HAIYI_FIT_MIN_DIST and old_veh.haiyi_fitted_math_angle is not None:
            fit_angle = old_veh.haiyi_fitted_math_angle
            angle_506 = self.map_mgr.lanes[HAIYI_506_1]['heading']
            angle_490 = self.map_mgr.lanes[HAIYI_490_2]['heading']

            diff_506 = ang_diff_deg(fit_angle, angle_506)
            diff_490 = ang_diff_deg(fit_angle, angle_490)

            # 谁的夹角小, 车就是顺着哪条道开进来的
            forced_haiyi_lane = HAIYI_506_1 if diff_506 <= diff_490 else HAIYI_490_2
            hijacked_heading = old_veh.haiyi_fitted_heading

            print(f"===== [海一路向量仲裁] ID:{int(old_veh.fixed_id) % 10000} | "
                  f"轨迹角:{fit_angle:.1f}° | 对506夹角:{diff_506:.1f}° | "
                  f"对490夹角:{diff_490:.1f}° | 最终裁决: {forced_haiyi_lane} =====")

            if not old_veh.is_off_lane:
                old_veh.haiyi_fitted_math_angle = None

        return forced_haiyi_lane, hijacked_heading

    def _find_best_match_2d(self, rel_x, rel_y, rv_heading, current_time,
                            incoming_sub_type=99):
        """
        全域 2D 物理缝合: 新 ID 认领已有目标。
        场景 1: 断联重连 (time_diff > 100ms 且距离与车速相关)
        场景 2: 极近距离分裂噪点 (< 18m 强行认领)
        """
        best_id = None
        min_dist = float('inf')

        for v_id, veh in self.active_vehicles.items():
            dist = math.hypot(veh.raw_x - rel_x, veh.raw_y - rel_y)
            time_diff = current_time - veh.last_radar_time
            dt_sec = max(time_diff / 1000.0, 0.1)

            # ---- 固定设施身份隔离 ----
            # 1. 车辆点绝不认领设施 ID   2. 设施点绝不认领车辆 ID
            # 3. 设施点绝不认领设施 ID: 舱盖板等设施密集排布, 相邻间距常
            #    < 18m (缝合分裂阈值), 设施-设施缝合会把整排舱盖板吞成
            #    一个目标 (透传显示多个, 算法版只剩极少)。设施静止,
            #    没有 ID 跳变缝合的必要 —— 每个设施独立持有 ID, 只锚定位姿。
            # 4. 仅"未知类型(99)且 <= 5m"的点允许缝合设施 (兼容 ID 跳变重连)
            if veh.is_fixed_facility:
                if incoming_sub_type in FIXED_FACILITY_SUB_TYPES:
                    continue
                if not (incoming_sub_type == 99
                        and dist <= SUTURE_FACILITY_UNKNOWN_DIST):
                    continue
            elif incoming_sub_type in FIXED_FACILITY_SUB_TYPES:
                continue

            # ---- 航向方向否决 (严防对向会车误缝合) ----
            # 海一路: >90° 即否决 (会车错乱高发区, 严格)
            # 其他车道: 仅近乎正对 (>150°) 才否决 —— 感知交界处同车双报
            #   的航向差通常 < 150°, 而对向车必 > 150°, 以此区分。
            #   (历史教训: 全车道 90°~120° 否决曾阻断交界处合法缝合, 已回退)
            # ---- 前后混淆豁免 ----
            # 老车近数秒内观测航向出现过 >120° 单帧跳变 (前后混淆签名)时,
            # 雷达航向对该车不可信, 航向否决依据失效 —— 放行缝合,
            # 位置连续性是唯一可靠证据。真对向车航向稳定, 不触发豁免。
            heading_unreliable = veh.heading_recently_flipped(current_time)
            if time_diff < SUTURE_HEADING_VETO_MS and not veh.is_reverse_driving \
                    and not heading_unreliable:
                angle_diff = ang_diff_deg(veh.raw_heading, rv_heading)
                if veh.lane_id in HAIYI_LANES:
                    if angle_diff > SUTURE_HEADING_VETO:
                        continue
                elif angle_diff > SUTURE_HEADING_VETO_GLOBAL:
                    continue

            # ---- 核心缝合 ----
            is_match = False

            # 场景 1: 断联重连
            max_allow_dist = min(SUTURE_REJOIN_DIST_MAX,
                                 SUTURE_REJOIN_DIST_BASE + veh.v * dt_sec)
            if time_diff > SUTURE_GAP_MS and dist < max_allow_dist:
                is_match = True
            # 场景 2: 极近距离分裂噪点
            elif dist < SUTURE_SPLIT_DIST:
                is_match = True

            if is_match and dist < min_dist:
                min_dist = dist
                best_id = v_id

        return best_id

    # =====================================================
    # Stage 3: 测量应用
    # =====================================================
    def _apply_measurement(self, veh, rv, lane_id, s, l,
                           match_x, match_y, attrs, current_time):
        """
        按车的当前车道状态分派测量:
            情况 A: 同车道正常行驶
            情况 B: 触发变道意图 (确认期 / 确认 / 拒绝)
            情况 C: 离道游离 -> 上轨
        """
        if veh.lane_id == lane_id:
            # 【情况 A】
            veh.pending_lane_id = None
            veh.lane_change_counter = 0
            veh.update_l(l)
            veh.apply_radar_s(current_time, s)
            return

        if veh.lane_id is not None:
            # 【情况 B】变道意图
            self._handle_lane_change_intent(veh, lane_id, s, l,
                                            match_x, match_y, current_time)
            return

        # 【情况 C】离道 -> 上轨
        self._handle_reentry(veh, lane_id, s, l, current_time)

    def _handle_lane_change_intent(self, veh, lane_id, s, l,
                                   match_x, match_y, current_time):
        """变道意图确认状态机: 连续 N 帧同一新车道 + 真实偏离达标才确认"""
        if veh.pending_lane_id == lane_id:
            veh.lane_change_counter += 1
        else:
            veh.pending_lane_id = lane_id
            veh.lane_change_counter = 1

        # 统一用滤波坐标计算相对老车道的偏离 (与情况 A 的 S/L 基线一致)
        dist_to_old = self.map_mgr.get_signed_offset(veh.lane_id, match_x, match_y)
        veh.update_l(dist_to_old)

        if veh.lane_change_counter >= LANE_CHANGE_CONFIRM_FRAMES \
                and abs(dist_to_old) > LANE_CHANGE_CONFIRM_DIST:
            # ===== 正式确认变道 =====
            self._start_lane_change_animation(veh, lane_id, s, current_time)

            # 车道系反转检测: 对向互变时 S 轴翻转, 方向状态需重建
            old_map_h = self.map_mgr.lanes[veh.lane_id]['heading']
            new_map_h = self.map_mgr.lanes[lane_id]['heading']
            if ang_diff_deg(old_map_h, new_map_h) > 90:
                veh.reset_drive_direction()

            veh.hard_reset_s(s)
            veh.lane_id = lane_id
            veh.v = veh.update_and_estimate_speed(current_time, s)
            veh.filtered_l = l
            veh.pending_lane_id = None
            veh.lane_change_counter = 0
        else:
            # 拒绝瞬间跨道: 沿老车道投影继续追踪
            old_line = self.map_mgr.lanes[veh.lane_id]['line']
            s_old = old_line.project(Point(match_x, match_y))
            veh.apply_radar_s(current_time, s_old)

    def _start_lane_change_animation(self, veh, lane_id, s, current_time):
        """
        变道平滑过渡动画 (海一路双向车道特化):
        Cosine ease-in-out 从上帧视觉位置丝滑过渡到新车道位置。
        非海一路车道不打动画 (直接硬切, 由输出级平滑兜底)。
        """
        if veh.lane_id not in HAIYI_LANES or lane_id not in HAIYI_LANES:
            veh.is_changing_lane = False
            veh.is_reverse_driving = False
            return

        new_x, new_y = self.map_mgr.get_xy_from_s(lane_id, s)
        if new_x is None:
            veh.is_changing_lane = False
            veh.is_reverse_driving = False
            return

        veh.is_changing_lane = True
        veh.lc_start_time = current_time

        start_x = veh.out_x if veh.out_x is not None else veh.raw_x
        start_y = veh.out_y if veh.out_y is not None else veh.raw_y
        veh.lc_offset_x = start_x - new_x
        veh.lc_offset_y = start_y - new_y

        target_geo = geo_heading_of(self.map_mgr.lanes[lane_id]['heading'])

        # 逆行判定: 雷达原生航向与新车道正常航向反向 => 逆行超车
        if ang_diff_deg(target_geo, veh.raw_heading) > 90:
            veh.is_reverse_driving = True
            target_geo = (target_geo + 180) % 360
        else:
            veh.is_reverse_driving = False

        veh.lc_offset_heading = (veh.out_heading - target_geo + 180) % 360 - 180

    def _start_rejoin_animation(self, veh, lane_id, s, current_time):
        """
        缝合重连过渡动画: 输出坐标从上帧视觉位置丝滑滑向新轨迹。
        复用变道动画机制 (Cosine ease-in-out), 期间豁免输出级硬切。
        """
        new_x, new_y = self.map_mgr.get_xy_from_s(lane_id, s)
        if new_x is None:
            return
        start_x = veh.out_x if veh.out_x is not None else veh.raw_x
        start_y = veh.out_y if veh.out_y is not None else veh.raw_y
        if math.hypot(start_x - new_x, start_y - new_y) > SUTURE_REJOIN_ANIM_MAX:
            return
        veh.is_changing_lane = True
        veh.lc_start_time = current_time
        veh.lc_offset_x = start_x - new_x
        veh.lc_offset_y = start_y - new_y
        veh.lc_offset_heading = 0.0  # 航向不变, 仅坐标过渡

    def _handle_reentry(self, veh, lane_id, s, l, current_time):
        """离道游离 -> 上轨: 记录时空偏差启动动画 + 硬切重置坐标系"""
        # 车道边缘短暂抖动 (离道 < REENTRY_BLIP_MS 且重回原车道):
        # 保留方向状态机与测速历史, 按同车道测量处理。
        # ⚠ 若每次边缘抖动都硬重置, 方向永远无法确认, 车头会退回
        # 雷达航向兜底 (前后混淆时) 而长期倒行/来回翻转。
        if veh.last_lane_id == lane_id and veh.went_off_lane_time is not None \
                and current_time - veh.went_off_lane_time < REENTRY_BLIP_MS:
            veh.lane_id = lane_id
            veh.update_l(l)
            veh.apply_radar_s(current_time, s)
            veh.pending_lane_id = None
            veh.lane_change_counter = 0
            return

        new_x, new_y = self.map_mgr.get_xy_from_s(lane_id, s)
        if new_x is not None:
            veh.is_changing_lane = True
            veh.lc_start_time = current_time

            # 视觉起点: 优先用离道期的最后平滑输出, 实现无缝衔接
            start_x = veh.out_x if veh.out_x is not None else veh.raw_x
            start_y = veh.out_y if veh.out_y is not None else veh.raw_y
            veh.lc_offset_x = start_x - new_x
            veh.lc_offset_y = start_y - new_y

            target_geo = geo_heading_of(self.map_mgr.lanes[lane_id]['heading'])

            if lane_id in HAIYI_LANES:
                # 海一路: 逆行判定
                if ang_diff_deg(target_geo, veh.raw_heading) > 90:
                    veh.is_reverse_driving = True
                    target_geo = (target_geo + 180) % 360
                else:
                    veh.is_reverse_driving = False
            else:
                veh.is_reverse_driving = False
                # 其他车道: 掉头兼容 (>150° 翻转, 海一路 490 除外)
                if ang_diff_deg(target_geo, veh.raw_heading) > HEADING_FALLBACK_FLIP \
                        and lane_id not in HAIYI_490_LANES:
                    target_geo = (target_geo + 180) % 360

            veh.lc_offset_heading = (veh.out_heading - target_geo + 180) % 360 - 180

        veh.reset_drive_direction()  # S 坐标系更换, 方向重新建立
        veh.hard_reset_s(s)
        veh.v = veh.update_and_estimate_speed(current_time, s)
        veh.filtered_l = l  # 车道系已更换, 直接采用新车道横向偏移
        veh.pending_lane_id = None
        veh.lane_change_counter = 0
        veh.lane_id = lane_id

    def _update_vehicle_type(self, veh, rv, attrs, current_time):
        """车型投票锁定的统一入口: 投票后以锁定结果覆写透传属性"""
        veh.vote_type(rv.itc_sub_type, rv.itc_obj_type, rv.type_reliability, current_time)
        if veh.locked_type is not None and attrs.get("itc_sub_type") != veh.locked_type:
            attrs["itc_sub_type"] = veh.locked_type
            if veh.locked_obj_type is not None:
                attrs["itc_obj_type"] = veh.locked_obj_type

    def _update_off_lane_vehicle(self, fixed_id, rv, attrs, current_time):
        """离道车测量更新 (含新建)"""
        if fixed_id in self.active_vehicles:
            veh = self.active_vehicles[fixed_id]
            if not veh.is_off_lane:
                # 在轨 -> 离道: 保存最后车道信息, 保留输出缓存实现无缝衔接
                veh.last_lane_id = veh.lane_id
                veh.last_s = veh.s
                veh.is_off_lane = True
                veh.lane_id = None
                veh.is_changing_lane = False  # 打断变道动画
                veh.went_off_lane_time = current_time  # 边缘抖动判定基准

            veh.update_raw_xy(rv.rel_x, rv.rel_y, current_time)
            # 噪点门限: 离道车无车道航向约束, 拒绝物理不可能的单帧突变
            veh.update_raw_heading(rv.radar_heading,
                                   time_diff_ms=current_time - veh.last_radar_time,
                                   noise_gate=True,
                                   current_time=current_time)

            self._update_vehicle_type(veh, rv, attrs, current_time)
            veh.last_radar_time = current_time
            veh.attrs = attrs
        else:
            new_veh = VehicleState(fixed_id, None, 0, 0, 0, attrs, current_time,
                                   rv.rel_x, rv.rel_y, rv.radar_heading)
            new_veh.is_off_lane = True
            new_veh.last_lane_id = None
            self.active_vehicles[fixed_id] = new_veh

    # =====================================================
    # Stage 4: 物理推演
    # =====================================================
    def _advance_physics(self, current_time, current_radar_ids):
        """去重 -> 490 离道拦截 -> 在轨去重 -> 断联推演/追击"""
        if self.last_update_time is None:
            self.last_update_time = current_time
        dt = (current_time - self.last_update_time) / 1000.0
        if dt <= 0:
            dt = 0.1

        self._dedup_off_lane()
        lane_groups = self._collect_on_lane(current_time, current_radar_ids)
        deduped_groups = self._dedup_on_lane(lane_groups, current_time)
        self._advance_lane_physics(current_time, current_radar_ids, deduped_groups, dt)

        self.last_update_time = current_time

    def _dedup_off_lane(self):
        """
        离道车 2D 物理去重: 距离 < 18m 视为分裂,
        雷达数据更新鲜者为真身。固定设施豁免。
        """
        off_lane_vehicles = [veh for veh in self.active_vehicles.values()
                             if veh.is_off_lane]
        to_delete = set()

        for i in range(len(off_lane_vehicles)):
            for j in range(i + 1, len(off_lane_vehicles)):
                v1, v2 = off_lane_vehicles[i], off_lane_vehicles[j]
                if v1.fixed_id in to_delete or v2.fixed_id in to_delete:
                    continue
                if v1.is_fixed_facility or v2.is_fixed_facility:
                    continue

                dist_2d = math.hypot(v1.raw_x - v2.raw_x, v1.raw_y - v2.raw_y)
                if dist_2d < DEDUP_OFFLANE_DIST:
                    if v1.last_radar_time >= v2.last_radar_time:
                        ghost, survivor = v2, v1
                        reason = "后车(离道)雷达数据陈旧"
                    else:
                        ghost, survivor = v1, v2
                        reason = "前车(离道)雷达数据陈旧"

                    to_delete.add(ghost.fixed_id)
                    print(f"[离道去重] 删ID:{int(ghost.fixed_id) % 10000} | "
                          f"留ID:{int(survivor.fixed_id) % 10000} | 原因: {reason} | "
                          f"2D相距:{dist_2d:.2f}m")

        for ghost_id in to_delete:
            self.active_vehicles.pop(ghost_id, None)

    def _collect_on_lane(self, current_time, current_radar_ids):
        """
        收集在轨车 -> {lane_id: [(veh, s), ...]}
        含海一路 490 车道横穿离场前置拦截。
        """
        lane_groups = {}

        for veh in self.active_vehicles.values():
            if veh.is_off_lane or veh.lane_id is None:
                continue

            # 海一路 490 车道主动离道拦截:
            # 信号丢失且滤波横向偏移显著偏左 (> 2m) => 横穿离场, 剥夺在轨身份
            if veh.fixed_id not in current_radar_ids and veh.lane_id in HAIYI_490_LANES:
                if veh.filtered_l > HAIYI_DEPART_L:
                    veh.is_off_lane = True
                    veh.last_lane_id = veh.lane_id
                    veh.is_changing_lane = False
                    print(f"[离道拦截] ID:{int(veh.fixed_id) % 10000} 从 490 车道向左"
                          f"(l={veh.filtered_l:.2f})横穿离场，中止推演！")
                    continue

            lane_groups.setdefault(veh.lane_id, []).append((veh, veh.s))

        return lane_groups

    def _dedup_on_lane(self, lane_groups, current_time):
        """
        在轨同车道去重 (全对比较):
        S 差 < 15m 且速度差 < 3 m/s 视为分裂,
        雷达数据更新鲜者为真身。固定设施豁免。
        """
        deduped = {}
        for lane_id, items in lane_groups.items():
            items.sort(key=lambda x: x[1], reverse=True)

            survivors = []
            ghosts_to_delete = set()
            for veh, s_val in items:
                if veh.fixed_id in ghosts_to_delete:
                    continue
                if veh.is_fixed_facility:
                    survivors.append((veh, s_val))
                    continue

                matched = False
                for si, (sv_veh, sv_s) in enumerate(survivors):
                    if sv_veh.is_fixed_facility:
                        continue
                    # 已确认对向行驶的两车绝非分裂 (速度大小可能相近,
                    # 但方向状态机给出了相反的物理证据)
                    if sv_veh.drive_direction * veh.drive_direction < 0:
                        continue
                    # 航向近乎正对 (>150°) 的两车是同车道对向行驶, 绝非分裂
                    # (分裂回波航向差通常远小于 150°; 新车测速暂态期 v 偏低,
                    #  仅靠速度差会漏判对向车, 航向是更硬的物理证据)
                    # 前后混淆豁免: 任一方近数秒内观测航向出现过 >120° 单帧
                    # 跳变时其雷达航向不可信, 该否决失效 (放行合并,
                    # 交给位置/速度仲裁)
                    if ang_diff_deg(sv_veh.raw_heading, veh.raw_heading) \
                            > SUTURE_HEADING_VETO_GLOBAL \
                            and not (sv_veh.heading_recently_flipped(current_time)
                                     or veh.heading_recently_flipped(current_time)):
                        continue
                    s_diff = abs(sv_s - s_val)
                    if s_diff < DEDUP_LANE_S_WINDOW \
                            and abs(sv_veh.v - veh.v) < DEDUP_LANE_V_DIFF:
                        if sv_veh.last_radar_time >= veh.last_radar_time:
                            ghosts_to_delete.add(veh.fixed_id)
                        else:
                            ghosts_to_delete.add(sv_veh.fixed_id)
                            survivors[si] = (veh, s_val)
                        matched = True
                        break

                if not matched:
                    survivors.append((veh, s_val))

            for ghost_id in ghosts_to_delete:
                self.active_vehicles.pop(ghost_id, None)

            deduped[lane_id] = survivors

        return deduped

    def _advance_lane_physics(self, current_time, current_radar_ids, lane_groups, dt):
        """断联推演 (物理一致模型) / 在线追击收敛"""
        for items in lane_groups.values():
            for idx, (veh, s_val) in enumerate(items):
                if veh.fixed_id not in current_radar_ids:
                    self._extrapolate_vehicle(veh, s_val, idx, items, dt, current_time)
                elif veh.is_chasing:
                    self._advance_chase(veh, dt, current_time)

    def _extrapolate_vehicle(self, veh, s_val, idx, items, dt, current_time):
        """
        断联推演 (物理一致模型):
        速度随盲区时间指数衰减, 方向由行驶方向状态机裁决。
        跟车时距前车过近按 IDM 减速。设施位置恒定不推演。
        """
        if veh.is_fixed_facility:
            return

        elapsed = (current_time - veh.last_radar_time) / 1000.0
        pred_v = veh.v * math.exp(-elapsed / PRED_SPEED_DECAY_TAU)

        if pred_v < PRED_MIN_SPEED:
            # 速度已衰减到接近静止: 不再推进 (静止车丢检后停在原地)
            new_s = s_val
        else:
            dir_sign = -1.0 if veh.drive_direction < 0 else 1.0

            if idx == 0:
                new_s = s_val + dir_sign * pred_v * dt
            else:
                leader_veh, leader_s = items[idx - 1]
                gap = (leader_s - s_val) * dir_sign
                if 0 < gap < IDM_BRAKE_DIST:
                    pred_v = max(0.0, pred_v - IDM_BRAKE_DECEL * dt)
                new_s = s_val + dir_sign * pred_v * dt

        if not veh.is_off_lane:
            veh.s = new_s          # (s 写入点 2/3: 推演)
            veh.target_s = new_s
            px, py = self.map_mgr.get_xy_from_s(veh.lane_id, new_s)
            if px is not None:
                veh.raw_x, veh.raw_y = px, py
        else:
            veh.last_s = new_s

    def _advance_chase(self, veh, dt, current_time):
        """
        追击收敛: 推演位置与真实位置的温和合流。
        偏差超物理可达极限直接瞬移 (大概率关联错误)。
        """
        gap = veh.target_s - veh.s
        elapsed = (current_time - veh.last_radar_time) / 1000.0

        # 物理可达性校验
        max_gap = veh.v * elapsed * PRED_MAX_GAP_RATIO + PRED_MAX_GAP_BASE
        if abs(gap) > max(max_gap, 30.0):
            veh.s = veh.target_s
            veh.is_chasing = False
            return

        # 温和追击: 速度差与偏差成正比, 上限 CHASE_MAX_SPEED
        catch_speed = min(CHASE_MAX_SPEED, abs(gap) / max(elapsed, 0.5))
        if gap > 0:
            veh.s += (veh.v + catch_speed) * dt
            if veh.s >= veh.target_s:
                veh.s = veh.target_s
                veh.is_chasing = False
        else:
            if veh.v < 0.5:
                # 真实车停了: 硬拉回
                veh.s = veh.target_s
                veh.is_chasing = False
            elif veh.s <= veh.target_s:
                veh.s = veh.target_s
                veh.is_chasing = False

    # =====================================================
    # Stage 5: 生命周期
    # =====================================================
    def _cleanup_stale_vehicles(self, current_time):
        """
        双轨超时清理: 离道车 2.5 秒快杀 / 在轨推演车 8 秒保护 / 设施 10 秒。
        """
        expired_ids = []
        for v_id, veh in self.active_vehicles.items():
            time_silent = current_time - veh.last_radar_time

            if veh.is_fixed_facility:
                if time_silent > FACILITY_TIMEOUT_MS:
                    expired_ids.append(v_id)
                continue

            if veh.is_off_lane or veh.lane_id is None:
                if time_silent > OFFLANE_TIMEOUT_MS:
                    expired_ids.append(v_id)
            else:
                if time_silent > ONLANE_TIMEOUT_MS:
                    expired_ids.append(v_id)

        if expired_ids:
            print("expired_ids: " + ' '.join(map(str, expired_ids)))
            for v_id in expired_ids:
                del self.active_vehicles[v_id]

    # =====================================================
    # Stage 6: 输出合成
    # =====================================================
    def _generate_processed_vehicles(self, current_time):
        res = []
        for veh in self.active_vehicles.values():
            if (current_time - veh.first_seen_time) < NEW_VEHICLE_GRACE_MS:
                continue

            is_predicted_flag = (veh.age_ms(current_time) > PREDICTED_FLAG_MS) \
                or veh.is_chasing

            if veh.is_off_lane:
                x, y, heading_rad, output_heading = self._compose_offlane_output(
                    veh, current_time)
                x, y = self._alpha_filter_xy(veh, x, y)
            else:
                x, y, heading_rad, output_heading = self._compose_onlane_output(
                    veh, current_time)

            veh.out_heading = output_heading

            res.append(ProcessedVehicle(
                original_id=veh.fixed_id,
                fixed_id=veh.fixed_id,
                x=x, y=y, v=veh.v,
                psi=heading_rad,
                is_predicted=is_predicted_flag,
                radar_heading=output_heading,
                **veh.attrs
            ))
        return res

    def _compose_onlane_output(self, veh, current_time):
        """在轨车输出: S+L 融合坐标 + 分层航向合成 + 变道动画 + 输出平滑"""
        x, y = self.map_mgr.get_xy_from_s(veh.lane_id, veh.s)

        # 输出融合带符号横向偏移 L: 输出贴合真实位置而非钉死中心线
        lx, ly = self.map_mgr.offset_lateral(veh.lane_id, veh.s, veh.filtered_l)
        if lx is not None:
            x, y = lx, ly

        geo_heading = self._compose_onlane_heading(veh, current_time)

        # 注入平滑变道曲线与航向角插值 (Cosine ease-in-out)
        if veh.is_changing_lane:
            elapsed = current_time - veh.lc_start_time
            if elapsed < veh.lc_duration:
                ratio = 0.5 * (1 + math.cos(math.pi * elapsed / veh.lc_duration))
                x += veh.lc_offset_x * ratio
                y += veh.lc_offset_y * ratio
                geo_heading += veh.lc_offset_heading * ratio
            else:
                veh.is_changing_lane = False  # 动画自然结束

        geo_heading = geo_heading % 360
        heading_rad = math.radians(geo_heading)

        # 输出级平滑 (动画期间豁免 3m 硬切保护)
        x, y = self._alpha_filter_xy(veh, x, y, is_lc=veh.is_changing_lane)

        return x, y, heading_rad, geo_heading

    def _compose_onlane_heading(self, veh, current_time):
        """
        在轨车航向合成 (优先级从高到低):
          0. 堆场车道锁定: 业务上不存在逆行, 严格锁定车道正向
          1. 方向滞回状态机裁决 (有 5 帧确认, 即时生效)
          2. 方向未知期: 海一路逆行先验 is_reverse_driving
          3. 方向未知期: 雷达反向兼容 (>150° 翻转, 海一路 490 除外)

        注意: 在轨车【故意】不做输出级限转速 — 方向状态机的翻转
        有滞回保护且代表物理事实 (真实掉头/逆行), 应尽快反映给前端;
        离道车无可靠方向信息才需要限转速平滑。
        """
        map_heading = self.map_mgr.lanes[veh.lane_id]['heading']
        base_geo = geo_heading_of(map_heading)

        # 堆场车道: 车头严格锁定车道正向。测速噪声误判 drive_direction=-1、
        # 雷达前后混淆报反向航向等一切"逆行"迹象都是感知错误, 不进画面。
        if veh.lane_id in DUICHANG_LANES:
            return base_geo

        if veh.drive_direction > 0:
            return base_geo
        if veh.drive_direction < 0:
            return (base_geo + 180) % 360

        # 方向未知期兜底
        geo = base_geo
        if veh.is_reverse_driving:
            return (geo + 180) % 360

        is_recent_gap = veh.age_ms(current_time) > RECENT_GAP_MS
        if ang_diff_deg(geo, veh.raw_heading) > HEADING_FALLBACK_FLIP \
                and not is_recent_gap and veh.lane_id not in HAIYI_490_LANES:
            print(f'angle reverse : {int(veh.fixed_id) % 10000}')
            return (geo + 180) % 360

        return geo

    def _compose_offlane_output(self, veh, current_time):
        """
        离道车输出: 物理坐标 + 航向三重防线 + 输出级限转速。
        """
        x, y = veh.raw_x, veh.raw_y

        # ---- 航向三重防线 ----
        # 1) 运动矢量纠正: 航向与真实运动方向严重不符且杂乱 => 噪点, 用运动方向
        # 2) 静止冻结: 无明显位移 => 保持上帧输出, 图标不随噪点旋转
        # 3) 限转速: 每帧最多转 HEADING_MAX_RATE/10 度
        target_h = veh.raw_heading
        motion_h = veh.get_motion_heading()
        if motion_h is not None:
            if ang_diff_deg(veh.raw_heading, motion_h) > HEADING_MOTION_CONFLICT \
                    and veh.is_heading_erratic():
                # 倒车等"航向稳定但与运动反向"不受影响: 稳定则不杂乱
                target_h = motion_h
        else:
            target_h = veh.out_heading if veh.out_heading is not None else veh.raw_heading

        if veh.out_heading is None:
            veh.out_heading = target_h
        d_h = (target_h - veh.out_heading + 180) % 360 - 180
        max_step = HEADING_MAX_RATE / 10.0
        d_h = max(-max_step, min(max_step, d_h))
        veh.out_heading = (veh.out_heading + d_h) % 360

        heading_rad = math.radians(veh.out_heading)
        return x, y, heading_rad, veh.out_heading

    @staticmethod
    def _alpha_filter_xy(veh, x, y, alpha=OUTPUT_SMOOTH_ALPHA, is_lc=False):
        """
        输出级坐标平滑 (Alpha 稳态卡尔曼近似):
        - 常规: 一阶低通
        - 突变 > 3m: 硬切重置 (防止画面横向滑移假象)
        - 变道动画期间豁免硬切
        """
        if veh.out_x is None or veh.out_y is None:
            veh.out_x = x
            veh.out_y = y
        else:
            dist_jump = math.hypot(x - veh.out_x, y - veh.out_y)

            if dist_jump > OUTPUT_HARD_CUT_DIST and not is_lc:
                veh.out_x = x
                veh.out_y = y
            else:
                veh.out_x = veh.out_x * (1 - alpha) + x * alpha
                veh.out_y = veh.out_y * (1 - alpha) + y * alpha

        return veh.out_x, veh.out_y
