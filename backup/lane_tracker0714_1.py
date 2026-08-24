import math
import time
from collections import deque
from data import ProcessedVehicle
from config import Config
from shapely.geometry import Point


class VehicleState:
    def __init__(self, obj_id, lane_id, s, l_offset, v, attrs, current_time, rel_x, rel_y, raw_heading):
        self.fixed_id = obj_id
        self.lane_id = lane_id
        self.s = s  # 纵向距离
        self.l = l_offset  # 横向偏移
        self.filtered_l = l_offset  # 滤波后的横向偏移
        self.v = v  # 当前速度
        self.attrs = attrs  # 业务透传属性
        self.last_radar_time = current_time
        self.first_seen_time = current_time  # 记录存活资历，用于消灭后来分裂出的李鬼
        self.is_off_lane = False  # 离线标识

        # 变道意图确认状态机
        self.pending_lane_id = None
        self.lane_change_counter = 0

        # fallback 2D 坐标 (用于离线游离状态)
        self.raw_x = rel_x
        self.raw_y = rel_y
        self.raw_heading = raw_heading

        # 5秒历史轨迹队列 (10Hz下50帧)
        self.s_history = deque(maxlen=50)
        self.s_history.append((current_time, s))

        # 最终输出平滑坐标缓存
        self.out_x = None
        self.out_y = None

        self.last_lane_id = lane_id  # 记录最后绑定车道
        self.last_s = s  # 记录最后S坐标
        self.pred_x = None  # 预测坐标缓存
        self.pred_y = None
        # ===== 新增：幽灵推演车追击机制 =====
        self.target_s = s  # 真实雷达车所在的位置 (物理层)
        self.is_chasing = False  # 是否处于推演车追击状态

    def update_s_and_chase(self, current_time, raw_s):
        """处理真实坐标更新，并触发/维持推演车追击状态"""
        # 1. 速度估算永远依赖【真实雷达点】
        # 防噪点保护：强制真实目标 S 单调递增，防止雷达噪点飘到车后
        actual_s = max(self.target_s if self.target_s is not None else self.s, raw_s)
        self.v = self.update_and_estimate_speed(current_time, actual_s)
        self.target_s = actual_s

        # 2. 判断是否刚从断联中恢复
        was_predicted = (current_time - self.last_radar_time > 300)

        if not self.is_chasing:
            # 如果刚恢复，且物理偏差大于 2 米，触发幽灵车追击！
            if was_predicted and abs(self.s - self.target_s) > 2.0:
                self.is_chasing = True
                # 注意：此时 self.s 保持不变（停留在推演车当前位置），仅 target_s 变为真实位置
            else:
                self.s = self.target_s  # 正常同轨行驶，合体

    def update_l(self, raw_l):
        """对横向偏移 L 进行一阶低通滤波"""
        alpha = 0.2  # 滤波系数：越小越平滑，抗噪越强 (0.2 相当于极度信任历史)
        self.filtered_l = self.filtered_l * (1 - alpha) + raw_l * alpha
        return self.filtered_l


    def update_and_estimate_speed(self, current_time, current_s):
        """
        利用最小二乘法，对过去 5 秒的历史轨迹进行线性回归，求出最稳定的 S-T 图像斜率(即真实车速)
        这能彻底免疫雷达短时间内的位置跳动噪点。
        """
        self.s_history.append((current_time, current_s))

        # 数据点太少，无法回归，返回 0.0 或上一帧速度
        if len(self.s_history) < 3: return self.v

        t_list = [h[0] for h in self.s_history]
        s_list = [h[1] for h in self.s_history]

        # 将时间戳转换为相对秒数 (防止数字过大导致精度丢失)
        t0 = t_list[0]
        x = [(t - t0) / 1000.0 for t in t_list]
        y = s_list

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        denominator = (n * sum_xx - sum_x * sum_x)
        if denominator == 0: return self.v
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        # 物理极限制裁：防止算出离谱速度 (例如限制在 -20m/s 到 +30m/s 之间)
        return max(0.0, min(30.0, slope))


class LaneQueueTracker:
    def __init__(self, map_manager):
        self.last_update_time = None
        self.map_mgr = map_manager
        # 核心数据结构：按 lane_id 分组维护车辆列表
        self.lane_queues = {}
        self.active_vehicles = {}  # fixed_id -> VehicleState

    def process_frame(self, raw_vehicles, current_time):
        current_radar_ids = set()

        for rv in raw_vehicles:
            # ==========================================
            # 提取历史状态，辅助更鲁棒的车道匹配
            # ==========================================
            fixed_id = rv.object_id
            old_veh = self.active_vehicles.get(fixed_id)

            veh_v = old_veh.v if old_veh else 0.0
            last_lane = old_veh.lane_id if old_veh else None

            # 1. 初始雷达点映射
            lane_id, s, l = self.map_mgr.match_to_lane(
                rv.rel_x, rv.rel_y,
                veh_heading=rv.radar_heading,
                v=veh_v,
                last_lane_id=last_lane,
                base_max_dist=3.0  # 基础阈值设为 3.0 米
            )

            attrs = {
                "itc_obj_type": rv.itc_obj_type,
                "plate_num": rv.plate_num,
                "lane_no": rv.lane_no,
                "type_reliability": rv.type_reliability,
                "itc_sub_type": rv.itc_sub_type
            }

            fixed_id = rv.object_id

            # ==========================================
            # 全域 2D 物理缝合 (解决换道残留)
            # ==========================================
            if fixed_id not in self.active_vehicles:
                matched_id = self._find_best_match_2d(rv.rel_x, rv.rel_y, current_time)
                if matched_id:
                    fixed_id = matched_id

            # 【单帧去重防抖】
            if fixed_id in current_radar_ids: continue
            current_radar_ids.add(fixed_id)

            # ==========================================
            # 最终状态更新 (融合变道迟滞确认机制)
            # ==========================================
            if lane_id is None:
                self._update_off_lane_vehicle(fixed_id, rv, attrs, current_time)
                continue

            if fixed_id not in self.active_vehicles:
                self.active_vehicles[fixed_id] = VehicleState(fixed_id, lane_id, s, l, 0.0, attrs, current_time,
                                                              rv.rel_x, rv.rel_y, rv.radar_heading)
            else:
                veh = self.active_vehicles[fixed_id]

                if veh.lane_id == lane_id:
                    # 【情况 A：正常同车道行驶】
                    veh.pending_lane_id = None
                    veh.lane_change_counter = 0
                    veh.update_l(l)  # 刷新滤波 L
                    # --- 替换旧的纵向更新逻辑 ---
                    # actual_s = max(veh.s, s)
                    # veh.v = veh.update_and_estimate_speed(current_time, actual_s)
                    # veh.s = actual_s
                    veh.update_s_and_chase(current_time, s)  # ✅ 应用追击逻辑
                    veh.lane_id = lane_id

                elif veh.lane_id is not None:
                    # 【情况 B：触发变道意图 (老车道 -> 新车道)】
                    if veh.pending_lane_id == lane_id:
                        veh.lane_change_counter += 1
                    else:
                        veh.pending_lane_id = lane_id
                        veh.lane_change_counter = 1

                    # 计算它相对【老车道】的真实横向距离，并丢入低通滤波
                    old_line = self.map_mgr.lanes[veh.lane_id]['line']
                    pt = Point(rv.rel_x, rv.rel_y)
                    dist_to_old = old_line.distance(pt)
                    veh.update_l(dist_to_old)

                    # 门限确认：连续 5 帧匹配到同一新车道，且滤波后的偏离距离 > 1.2 米 (集卡较宽，阈值适中)
                    if veh.lane_change_counter >= 5 and abs(veh.filtered_l) > 2.5:  # 1.2
                        # ✅ 正式确认变道！
                        veh.s_history.clear()
                        veh.lane_id = lane_id
                        # 变道瞬间允许硬切重置，打断追击防止跨车道错乱
                        veh.target_s = s
                        veh.s = s
                        veh.is_chasing = False

                        veh.v = veh.update_and_estimate_speed(current_time, s)
                        veh.filtered_l = l
                        veh.pending_lane_id = None
                        veh.lane_change_counter = 0
                    else:
                        # ❌ 仍在确认期或意图不足，拒绝瞬间跨道！
                        lane_id = veh.lane_id
                        s = old_line.project(pt)
                        # --- 替换旧的纵向更新逻辑 ---
                        veh.update_s_and_chase(current_time, s)  # ✅ 拒识变道时，依然应用沿车道追击

                else:
                    # 【情况 C：之前是离线游离状态，现在成功上轨】
                    veh.s_history.clear()
                    veh.lane_id = lane_id
                    veh.s = s
                    veh.target_s = s  # 新增：同步 target_s
                    veh.is_chasing = False  # 新增：从游离态强制拉回，关闭追击
                    veh.update_l(l)
                    veh.v = veh.update_and_estimate_speed(current_time, s)
                    veh.pending_lane_id = None
                    veh.lane_change_counter = 0

                    # [Fix 体验优化] 将车辆当前的 2D 物理坐标无缝接管为输出级平滑滤波的起点
                    # 这样后续的一阶低通滤波器会用 3~5 帧的时间，把车辆从自由 2D 轨迹极其丝滑地“拉回”到新车道中心线上，视觉体验拉满
                    veh.out_x = veh.raw_x
                    veh.out_y = veh.raw_y

                # 统一更新基础属性
                veh.attrs = attrs
                veh.last_radar_time = current_time
                veh.is_off_lane = False
                veh.raw_x = rv.rel_x
                veh.raw_y = rv.rel_y
                veh.raw_heading = rv.radar_heading

        self._update_physics_queues(current_time, current_radar_ids)
        self._cleanup_stale_vehicles(current_time)
        return self._generate_processed_vehicles(current_time)

    def _find_best_match_2d(self, rel_x, rel_y, current_time):
        """跨越车道的全域 2D 物理缝合，并寻找距离最近的最优解"""
        best_id = None
        min_dist = float('inf')

        for v_id, veh in self.active_vehicles.items():
            # 使用最真实的物理坐标进行欧氏距离计算
            dist = math.hypot(veh.raw_x - rel_x, veh.raw_y - rel_y)
            time_diff = current_time - veh.last_radar_time

            is_match = False
            # 场景 1: 无论是否换道/离线，只要断联且距离 < 20米，大概率是它
            if time_diff > 100 and dist < 50.0: # 40
                is_match = True
            # 场景 2: 极近距离雷达瞬间分裂噪点
            elif dist < 20.0:   # 15
                is_match = True

            if is_match and dist < min_dist:
                min_dist = dist
                best_id = v_id

        return best_id

    def _update_physics_queues(self, current_time, current_radar_ids):
        # 初始化时间步长
        if self.last_update_time is None:
            self.last_update_time = current_time
        dt = (current_time - self.last_update_time) / 1000.0
        if dt <= 0:
            dt = 0.1  # 安全保护

        self.lane_queues.clear()

        # 收集所有有车道信息的车辆（包括离线但有 last_lane_id 的）
        for veh in self.active_vehicles.values():
            if not veh.is_off_lane and veh.lane_id is not None:
                lane = veh.lane_id
                s = veh.s
                if lane not in self.lane_queues:
                    self.lane_queues[lane] = []
                self.lane_queues[lane].append((veh, s))  # 存储 (vehicle, s) 元组

        for lane_id, items in self.lane_queues.items():
            # 按 s 降序排序
            items.sort(key=lambda x: x[1], reverse=True)

            # ----- 去重逻辑（基于 s 和速度差）-----
            survivors = []
            for veh, s_val in items:
                if not survivors:
                    survivors.append((veh, s_val))
                else:
                    leader_veh, leader_s = survivors[-1]
                    if (leader_s - s_val) < 15.0 and abs(leader_veh.v - veh.v) < 2.0:
                        # 合并：删除后出现的（或资历浅的）
                        if leader_veh.first_seen_time <= veh.first_seen_time:
                            ghost = veh
                        else:
                            ghost = leader_veh
                            survivors[-1] = (veh, s_val)
                        if ghost.fixed_id in self.active_vehicles:
                            del self.active_vehicles[ghost.fixed_id]
                    else:
                        survivors.append((veh, s_val))

            # 更新队列为去重后的 (vehicle, s) 列表
            self.lane_queues[lane_id] = survivors

            # ----- 推演每个车辆的 S（断联推演 or 在线追击）-----
            for idx, (veh, s_val) in enumerate(survivors):
                if veh.fixed_id not in current_radar_ids:
                    # ==========================================
                    # 1. 断联/离线状态：执行 IDM 盲区推演
                    # ==========================================
                    if veh.v < 1.0:
                        veh.v = 2.0  # 可调整，建议保守值 1.0~2.0

                    if idx == 0:
                        # 头车：匀速
                        new_s = s_val + veh.v * dt
                    else:
                        # 跟车：简单 IDM
                        leader_veh, leader_s = survivors[idx - 1]
                        dist = leader_s - s_val
                        if dist < 15.0:
                            veh.v = max(0, veh.v - 2.0 * dt)
                        else:
                            veh.v = leader_veh.v
                        new_s = s_val + veh.v * dt

                    # 更新车辆状态
                    if not veh.is_off_lane:
                        veh.s = new_s
                        veh.target_s = new_s  # ✅ 断联推演时，两者保持同步
                        # S 坐标往前推演了，必须同步更新 2D 物理坐标
                        px, py = self.map_mgr.get_xy_from_s(veh.lane_id, new_s)
                        if px is not None:
                            veh.raw_x, veh.raw_y = px, py
                    else:
                        # ⚠️ 恢复原本的脱轨逻辑：离线且脱离车道时，仅记录推演 S
                        veh.last_s = new_s

                    print(f'idm: {veh.fixed_id}-{idx}')

                else:
                    # ==========================================
                    # 2. 在线/观测状态：正常更新 或 执行推演车追击
                    # ==========================================
                    if getattr(veh, 'is_chasing', False):
                        # ===== 核心：推演车追击真实车逻辑 =====
                        gap = veh.target_s - veh.s

                        # 如果物理偏差离谱(>30米)，说明 IDM 推演出大错，放弃追击直接瞬移
                        if abs(gap) > 30.0:
                            veh.s = veh.target_s
                            veh.is_chasing = False
                        else:
                            catch_speed = 6.0  # 追击速度差：比真实车快/慢 6m/s

                            if gap > 0:
                                # 推演车在真实车后方，加速追赶
                                veh.s += (veh.v + catch_speed) * dt
                                if veh.s >= veh.target_s:
                                    veh.s = veh.target_s
                                    veh.is_chasing = False
                            else:
                                # 推演车冲过头了(盲区预测过快)，减速甚至后退等待真实车
                                chase_v = veh.v - catch_speed
                                if chase_v > 0:
                                    veh.s += chase_v * dt
                                else:
                                    veh.s -= 2.0 * dt  # 速度不够扣时，强制以 2m/s 后退靠拢，防止死锁

                                if veh.s <= veh.target_s:
                                    veh.s = veh.target_s
                                    veh.is_chasing = False

                        # ✅ 追击过程中，将推演车的 s 转换为 raw_x/y，保证输出平滑器正常工作
                        px, py = self.map_mgr.get_xy_from_s(veh.lane_id, veh.s)
                        if px is not None:
                            veh.raw_x, veh.raw_y = px, py
                    print(f'idm: {veh.fixed_id}-{idx}')
        self.last_update_time = current_time

    def _generate_processed_vehicles(self, current_time):
        res = []
        for veh in self.active_vehicles.values():
            if (current_time - veh.first_seen_time) < 300:
                continue

            # ✅ 新增：如果是推演追击状态，强行保持 is_predicted = True 通知前端
            is_predicted_flag = (current_time - veh.last_radar_time > 500) or getattr(veh, 'is_chasing', False)

            if veh.is_off_lane:
                # 已经脱离车道，放弃 S 坐标约束，直接使用自带滤波的 2D 物理坐标
                x, y = veh.raw_x, veh.raw_y
                # 航向使用原始航向（或车道方向，可由业务决定）
                heading_rad = math.radians(veh.raw_heading)
                output_radar_heading = veh.raw_heading
            else:
                x, y = self.map_mgr.get_xy_from_s(veh.lane_id, veh.s)
                map_heading = self.map_mgr.lanes[veh.lane_id]['heading']
                geo_heading = (90 - map_heading) % 360

                # 雷达反向感知/倒车兼容处理
                # 计算雷达感知航向与车道标准航向的夹角绝对值 (0~180度)
                angle_diff = abs((geo_heading - veh.raw_heading + 180) % 360 - 180)
                is_predicted = (current_time - veh.last_radar_time > 100)
                # 如果偏差超过 150°，强制翻转 180° 取反方向
                if angle_diff > 150 and not is_predicted and (veh.lane_id != '137438953490'):   # 只有在非预测(实时追踪)状态下，才允许根据雷达原始航向翻转车身。
                    geo_heading = (geo_heading + 180) % 360
                    print(f'angle reverse : {veh.fixed_id}')

                heading_rad = math.radians(geo_heading)
                output_radar_heading = geo_heading

                if not veh.is_off_lane:
                    # 在轨车辆应用输出级平滑滤波
                    x, y = self._alpha_filter_xy(veh, x, y)

            res.append(ProcessedVehicle(
                original_id=veh.fixed_id,
                fixed_id=veh.fixed_id,
                x=x, y=y, v=veh.v,
                psi=heading_rad,
                is_predicted=is_predicted_flag,
                radar_heading=output_radar_heading,
                **veh.attrs
            ))
        return res

    @staticmethod
    def _alpha_filter_xy(veh: VehicleState, x, y, alpha=0.2):
        """
        引入滤波平滑输出
        对(x,y) 进行滤波，使前端轨迹更平滑，同时避免位置突变。
        alpha = 0.3 意味着 30% 信任新算出的目标位置，70% 保持原有轨迹的惯性
        """
        if veh.out_x is None or veh.out_y is None:
            # 第一次输出，直接赋值
            veh.out_x = x
            veh.out_y = y
        else:
            # 计算期望目标坐标与当前平滑坐标的几何距离
            dist_jump = math.hypot(x - veh.out_x, y - veh.out_y)

            if dist_jump > 3.0:
                # 【防拉扯机制】：如果突变超过 3 米（如 3 帧确认后瞬间切入新车道）
                # 直接重置平滑坐标，强行硬切。防止前端画面出现车辆在马路上横向“滑移”漂过去的假象！
                veh.out_x = x
                veh.out_y = y
            else:
                # 【常规平滑机制】：一阶低通滤波 (Alpha 稳态卡尔曼近似)
                veh.out_x = veh.out_x * (1 - alpha) + x * alpha
                veh.out_y = veh.out_y * (1 - alpha) + y * alpha

        return veh.out_x, veh.out_y

    def _update_off_lane_vehicle(self, fixed_id, rv, attrs, current_time):
        if fixed_id in self.active_vehicles:
            veh = self.active_vehicles[fixed_id]
            if not veh.is_off_lane:
                # 转入离线，保存最后车道信息
                veh.last_lane_id = veh.lane_id
                veh.last_s = veh.s
                veh.is_off_lane = True
                veh.lane_id = None
                # 可重置输出平滑缓存，避免突变
                veh.out_x = veh.out_y = None
            # 更新原始坐标（用于重入匹配）
            alpha = 0.1
            veh.raw_x = veh.raw_x * (1 - alpha) + rv.rel_x * alpha
            veh.raw_y = veh.raw_y * (1 - alpha) + rv.rel_y * alpha
            veh.raw_heading = rv.radar_heading
            veh.last_radar_time = current_time
            veh.attrs = attrs
        else:
            # 新建离线车辆（没有车道信息）
            new_veh = VehicleState(
                fixed_id, None, 0, 0, 0, attrs, current_time,
                rv.rel_x, rv.rel_y, rv.radar_heading
            )
            new_veh.is_off_lane = True
            new_veh.last_lane_id = None
            self.active_vehicles[fixed_id] = new_veh

    def _cleanup_stale_vehicles(self, current_time, timeout=10000):
        """清理长时间未收到雷达信号的车辆 (默认 10 秒)"""
        expired_ids = [
            v_id for v_id, veh in self.active_vehicles.items()
            if (current_time - veh.last_radar_time) > timeout
        ]
        for v_id in expired_ids:
            del self.active_vehicles[v_id]