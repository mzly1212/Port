import math
import time
from collections import deque
from data import ProcessedVehicle
from config import Config
import numpy as np
from shapely.geometry import Point

# ==========================================
# 🚀 固定设施类子类型：舱盖板(14)、候工亭/锁销框(13)
# 这类目标物理位置恒定不动，执行"限移三重保护"策略：
#   1. 去重豁免 —— 不参与 2D 去重仲裁，防止被误杀
#   2. 身份隔离 —— 不与车辆互相缝合 ID，防止误绑定后跟随车辆移动
#   3. 位置锚定 —— 坐标钉死在锚点附近，消除传感器波动导致的抖动
# ==========================================
FIXED_FACILITY_SUB_TYPES = {13, 14}

# 固定设施锚定参数
FACILITY_ANCHOR_SAMPLES = 10     # 锚点确认采样帧数 (10Hz 下约 1 秒)
FACILITY_JITTER_DEADZONE = 0.5   # 抖动静区：偏移 <= 0.5m 视为传感器波动，坐标钉死在锚点
FACILITY_MAX_DRIFT = 3.0         # 误绑定阈值：偏移 > 3.0m 判定为车辆误绑定，拒绝移动
FACILITY_REANCHOR_MS = 15000     # 持续大幅偏移超过 15 秒才重新锚定 (兼容舱盖板被吊装搬运)
FACILITY_TIMEOUT_MS = 10000      # 固定设施断检存活期 (防止过车遮挡导致平台图标闪烁)

# ==========================================
# 🚗 行驶方向滞回状态机参数 (根治车头 180° 来回翻转)
#   旧策略直接用最小二乘回归斜率 signed_v 的正负翻转航向角,
#   低速/雷达噪声下斜率在阈值附近抖动, 导致车头前后甩动。
#   新策略: 死区 + 连续确认的滞回状态机, 只有持续显著的反向
#   物理证据才允许翻转一次方向。
# ==========================================
DIR_DEADZONE = 0.8    # 死区: |signed_v| < 0.8 m/s 视为噪声, 不参与方向判定
DIR_CONFIRM_N = 5     # 翻转方向需连续 5 个有效采样确认 (10Hz 下约 0.5 秒)

# ==========================================
# 🚗 车型投票锁定参数 (取代旧的"首次非99即永久锁定")
#   旧策略第一帧误分类会被永久锁死。新策略按置信度加权投票:
#   占比与帧数双门槛达标才锁定; 锁定后若出现压倒性反证仍可纠正;
#   固定设施(13/14)额外要求"物理静止"证据, 防止行驶中的车辆被
#   误锁成设施后触发位置锚定而"钉死"在原地。
# ==========================================
TYPE_DECAY = 0.98           # 票数时间衰减系数 (半衰期约 3.4 秒 @10Hz)
TYPE_LOCK_RATIO = 0.6       # 普通类型锁定占比门槛
TYPE_LOCK_FRAMES = 8        # 普通类型锁定最小观测帧数
TYPE_FAC_LOCK_RATIO = 0.85  # 固定设施锁定占比门槛 (更严格)
TYPE_FAC_LOCK_FRAMES = 10   # 固定设施锁定最小观测帧数
TYPE_FIX_RATIO = 0.8        # 已锁定类型的纠正占比门槛 (反证占压倒性优势才纠正)


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
        # =====新增：海一路入场轨迹拟合相关状态======
        self.xy_history = deque(maxlen=30)  # 记录最近 3 秒的真实 2D 坐标
        self.haiyi_fitted_heading = None  # 缓存的平滑拟合航向 (地理角)

        # ===== 新增：平滑变道过渡动画机制 =====
        self.is_changing_lane = False
        self.lc_start_time = 0
        self.lc_duration = 2000  # 变道动画持续时间 2000ms (2秒，可根据前端视觉效果微调)
        self.lc_offset_x = 0.0
        self.lc_offset_y = 0.0
        self.lc_offset_heading = 0.0

        # ===== 新增：海一路逆行变道控制 =====
        self.out_heading = raw_heading  # 缓存用于前端渲染的视觉航向角
        self.is_reverse_driving = False  # 是否处于逆行借道状态
        self.signed_v = 0.0  # ✅ 新增：记录带有方向的真实 S 轴速度

        # ===== 新增：行驶方向滞回状态机 (根治车头 180° 翻转抖动) =====
        self.drive_direction = 0    # 0=未知, 1=沿车道正向, -1=沿车道反向
        self.dir_candidate = 0      # 候选方向 (等待连续确认)
        self.dir_confirm = 0        # 候选方向连续确认计数

        # ===== 新增：车型投票锁定状态 =====
        self.type_votes = {}        # sub_type -> 加权票数 (按置信度加权, 随时间衰减)
        self.type_obs_count = 0     # 有效观测帧数 (不含 99 未知)
        self.locked_type = None     # 投票锁定后的子类型
        self.locked_obj_type = None # 锁定时的父类型 (保证父子类型一致)

        # ===== 新增：固定设施位置锚定 =====
        self.anchor_x = None               # 锚点坐标 (确认期结束后锁定)
        self.anchor_y = None
        self.anchor_obs = deque(maxlen=FACILITY_ANCHOR_SAMPLES)  # 确认期观测采样 (中位数免疫误绑定尖峰)
        self.drift_since = None            # 持续大幅偏移的起始时间 (用于真实搬运后重新锚定)

    def update_s_and_chase(self, current_time, raw_s):
        """处理真实坐标更新，并触发/维持推演车追击状态"""
        # 🚀 固定设施：纵向位置直接采用锚定坐标的车道投影结果，
        # 不参与棘轮/追击/测速 (设施速度恒为 0，杜绝一切纵向移动来源)
        if self.attrs.get("itc_sub_type", 99) in FIXED_FACILITY_SUB_TYPES:
            self.v = 0.0
            self.signed_v = 0.0
            self.target_s = raw_s
            self.s = raw_s
            return

        # 1. 速度估算永远依赖【真实雷达点】，将抗噪任务全权交给后续的最小二乘法 (允许轻微噪点回退)
        self.v = self.update_and_estimate_speed(current_time, raw_s)
        self.target_s = raw_s  # 物理层永远保持为真实的雷达纵向坐标

        # 2. 判断是否刚从断联中恢复
        was_predicted = (current_time - self.last_radar_time > 100) # 300

        if not self.is_chasing:
            # 如果刚恢复，且物理偏差大于 2 米，触发幽灵车追击！
            if was_predicted and abs(self.s - self.target_s) > 2.0: # 2.0
                self.is_chasing = True
                # 注意：此时 self.s 保持不变（停留在推演车当前位置），仅 target_s 变为真实位置
            else:
                # ==========================================
                # 常态防后退/防瞬移 (单向棘轮) 机制
                # 彻底解决雷达噪点回跳导致的画面拉扯与视觉倒退问题
                # ==========================================
                if self.drive_direction > 0:
                    # 明确正向行驶中：不允许 S 减小。若雷达噪点回跳，停在原地等待真实信号跟上
                    # (改用方向状态机裁决而非瞬时 signed_v, 消除回归斜率抖动导致的棘轮方向反复反转)
                    if self.target_s >= self.s:
                        self.s = self.target_s
                elif self.drive_direction < 0:
                    # 明确倒车/逆行中：不允许 S 增加。若雷达噪点前跳，停在原地等待
                    if self.target_s <= self.s:
                        self.s = self.target_s
                else:
                    # 方向未确认：完全相信雷达并交给底层的 _alpha_filter_xy 2D平滑
                    # (防止单向机制导致静止车辆的坐标随噪点不断向前/向后累积蠕动)
                    self.s = self.target_s

    def update_l(self, raw_l):
        """
        对横向偏移 L 进行一阶低通滤波 (动态系数)：
        - 偏差大 (> 0.6m, 变道/入轨修正中): alpha=0.5 快速跟随真实横移
        - 偏差小 (正常巡航): alpha=0.2 极度平滑抗噪
        旧的固定 alpha=0.2 会让变道时的横向偏移收敛滞后数秒。
        """
        err = abs(raw_l - self.filtered_l)
        alpha = 0.5 if err > 0.6 else 0.2
        self.filtered_l = self.filtered_l * (1 - alpha) + raw_l * alpha
        return self.filtered_l

    # ==========================================
    # �� 新增：原始坐标前置滤波方法
    # ==========================================
    def update_raw_xy(self, new_x, new_y, current_time=None):
        """
        在送入地图匹配前，对二维物理坐标进行预平滑。
        不仅能消除雷达高频抖动，更能让 match_to_lane 的评分与横向距离计算更加稳健。
        """
        # 🚀 固定设施：走"位置锚定"策略 —— 限制移动、消除抖动
        if self.attrs.get("itc_sub_type", 99) in FIXED_FACILITY_SUB_TYPES:
            return self._update_fixed_facility_xy(new_x, new_y, current_time)

        alpha = 0.3  # 30% 信任新雷达点，70% 沿用上一帧物理惯性
        self.raw_x = self.raw_x * (1 - alpha) + new_x * alpha
        self.raw_y = self.raw_y * (1 - alpha) + new_y * alpha
        return self.raw_x, self.raw_y

    def _update_fixed_facility_xy(self, new_x, new_y, current_time=None):
        """
        🚀 固定设施位置锚定策略 (限制移动 + 消除抖动)：
        - 确认期 (前 FACILITY_ANCHOR_SAMPLES 帧)：采样观测取中位数锁定锚点，
          中位数天然免疫确认期内偶发的误绑定尖峰
        - 锚定期：
            偏移 <= 0.5m      -> 传感器波动，坐标钉死在锚点 (彻底消除抖动)
            0.5m < 偏移 <= 3m -> 重滤波缓慢收敛 (纠正锚点小偏差，视觉上近似静止)
            偏移 > 3m         -> 判定为过车误绑定，拒绝该观测、位置保持不动；
                                 持续超过 15 秒才重新锚定 (兼容舱盖板被吊装搬运的真实场景)
        """
        # ---- 阶段 1：锚点确认期 ----
        if self.anchor_x is None:
            self.anchor_obs.append((new_x, new_y))
            # 确认期内先用常规轻滤波，保证有连续输出
            alpha = 0.3
            self.raw_x = self.raw_x * (1 - alpha) + new_x * alpha
            self.raw_y = self.raw_y * (1 - alpha) + new_y * alpha
            if len(self.anchor_obs) >= self.anchor_obs.maxlen:
                xs = [p[0] for p in self.anchor_obs]
                ys = [p[1] for p in self.anchor_obs]
                self.anchor_x = float(np.median(xs))
                self.anchor_y = float(np.median(ys))
                self.raw_x, self.raw_y = self.anchor_x, self.anchor_y
            return self.raw_x, self.raw_y

        # ---- 阶段 2：锚定期 ----
        drift = math.hypot(new_x - self.anchor_x, new_y - self.anchor_y)

        if drift <= FACILITY_JITTER_DEADZONE:
            # 传感器小波动：完全忽略，钉死在锚点
            self.drift_since = None
            self.raw_x, self.raw_y = self.anchor_x, self.anchor_y
        elif drift <= FACILITY_MAX_DRIFT:
            # 中等偏移：重滤波缓慢收敛 (alpha 极小，视觉上近似静止)
            self.drift_since = None
            alpha = 0.05
            self.raw_x = self.raw_x * (1 - alpha) + new_x * alpha
            self.raw_y = self.raw_y * (1 - alpha) + new_y * alpha
        else:
            # 大幅偏移：大概率是过车误绑定，拒绝该观测，位置保持不动
            if current_time is not None:
                if self.drift_since is None:
                    self.drift_since = current_time
                elif current_time - self.drift_since >= FACILITY_REANCHOR_MS:
                    # 持续大幅偏移超过 15 秒：判定为真实搬运 (如舱盖板被吊走)，重新锚定
                    self.anchor_x, self.anchor_y = new_x, new_y
                    self.raw_x, self.raw_y = new_x, new_y
                    self.drift_since = None
                    print(f'[固定设施重新锚定] ID:{int(self.fixed_id) % 10000} -> ({new_x:.2f}, {new_y:.2f})')

        return self.raw_x, self.raw_y

    # ==========================================
    # �� 新增：原始雷达航向角环形预滤波方法
    # ==========================================
    def update_raw_heading(self, new_heading):
        """
        对原始雷达航向角进行一阶低通滤波（严格解决 360° 环形临界点插值错误）
        避免在 0° 和 360° 交界处滤波时发生 180° 甩头异变。
        """
        alpha = 0.2  # 滤波系数：30% 信任新航向，70% 保持惯性

        # 🚀 固定设施：航向恒定，用更重的滤波彻底抑制图标旋转抖动
        if self.attrs.get("itc_sub_type", 99) in FIXED_FACILITY_SUB_TYPES:
            alpha = 0.05

        # 1. 计算两角之间的最短几何路径差 (-180° ~ 180°)
        # 例如：old = 350, new = 10 -> diff = (10 - 350 + 180)%360 - 180 = +20
        diff = (new_heading - self.raw_heading + 180) % 360 - 180

        # 2. 沿最短路径累加滤波增量，并归一化至 0° ~ 360°
        self.raw_heading = (self.raw_heading + alpha * diff) % 360
        return self.raw_heading

    def update_and_estimate_speed(self, current_time, current_s):
        """
        利用最小二乘法，对历史轨迹进行线性回归，求出最稳定的 S-T 图像斜率(即真实车速)
        根据最新需求：排除最近 1 秒的数据，仅使用过去 5 秒 ~ 1 秒的轨迹进行测速
        这能彻底免疫雷达短时间内（如目标刚出现或刚拼接时）的高频位置跳动噪点。
        """
        self.s_history.append((current_time, current_s))

        # 过滤出 1 秒 (1000ms) 以前的数据
        valid_history = [h for h in self.s_history if (current_time - h[0]) > 1000]

        # 过滤后的有效数据点太少 (不足3点)，无法回归，保持上一帧的惯性速度
        if len(valid_history) < 3:
            return self.v

        t_list = [h[0] for h in valid_history]
        s_list = [h[1] for h in valid_history]

        # 将时间戳转换为相对秒数 (防止数字过大导致最小二乘法计算精度丢失)
        t0 = t_list[0]
        x = [(t - t0) / 1000.0 for t in t_list]
        y = s_list

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi * xi for xi in x)

        denominator = (n * sum_xx - sum_x * sum_x)
        if denominator == 0:
            return self.v

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # ✅ 记录 S 的真实变化率 (正代表沿车道，负代表逆车道)
        self.signed_v = slope

        # ✅ 方向滞回状态机推进: 消除回归斜率噪声导致的车头 180° 来回翻转
        self._update_drive_direction()

        # 物理极限制裁：防止算出离谱速度 (例如限制在 0m/s 到 +30m/s 之间)
        return max(0.0, min(30.0, slope))

    def _update_drive_direction(self):
        """
        🚗 行驶方向滞回状态机 (0=未知, 1=正向, -1=反向)。
        旧逻辑直接以 signed_v 正负(阈值0.5)翻转航向角, 低速或雷达噪声下
        回归斜率会在阈值附近抖动, 造成车头 180° 来回甩动、单向棘轮反复反转。
        新逻辑: 只有当 |signed_v| 超过死区(DIR_DEADZONE), 且同一方向连续
        DIR_CONFIRM_N 次采样确认后才翻转方向; 死区内维持既有方向。
        """
        if abs(self.signed_v) < DIR_DEADZONE:
            # 死区: 斜率不显著, 维持既有方向 (候选保留, 允许短暂回落后继续确认)
            return self.drive_direction

        want = 1 if self.signed_v > 0 else -1
        if want == self.drive_direction:
            # 与当前方向一致, 清空翻转候选
            self.dir_candidate = 0
            self.dir_confirm = 0
            return self.drive_direction

        if want == self.dir_candidate:
            self.dir_confirm += 1
        else:
            self.dir_candidate = want
            self.dir_confirm = 1

        if self.dir_confirm >= DIR_CONFIRM_N:
            self.drive_direction = want
            self.dir_candidate = 0
            self.dir_confirm = 0
        return self.drive_direction

    def reset_drive_direction(self):
        """车道切换(尤其 S 坐标系反转, 如对向借道)后调用, 方向需基于新车道重新建立"""
        self.drive_direction = 0
        self.dir_candidate = 0
        self.dir_confirm = 0

    def _is_physically_static(self):
        """基于最近 3 秒原始 2D 轨迹判断目标是否真实静止 (固定设施锁定的运动学证据)"""
        if len(self.xy_history) < 15:  # 至少 1.5 秒历史
            return False
        xs = [p[0] for p in self.xy_history]
        ys = [p[1] for p in self.xy_history]
        return (max(xs) - min(xs)) < 1.0 and (max(ys) - min(ys)) < 1.0

    def vote_type(self, sub_type, obj_type, reliability, current_time):
        """
        🚗 车型投票锁定机制 (取代旧的"首次非 99 即永久锁定")：
        - 按感知置信度加权投票, 票数随时间指数衰减 (早期误检影响自动消退)
        - 占比与帧数双门槛达标才锁定, 杜绝首帧误分类被永久锁死
        - 锁定后若另一类型持续获得压倒性证据, 允许纠正
        - 固定设施(13/14)额外要求"物理静止"证据: 行驶中的目标即使被设备
          误报为设施也不予采纳, 防止车辆被位置锚定机制"钉死"在原地
        """
        if sub_type is None or sub_type == 99:
            return  # 未知类型不投票

        # 置信度归一 (设备未提供有效值时给中性权重)
        rel = reliability if 0 < reliability <= 1.0 else 0.6

        # 全体票数时间衰减: 让持续稳定的观测主导, 早期偶发误检自动退场
        for k in list(self.type_votes.keys()):
            self.type_votes[k] *= TYPE_DECAY
            if self.type_votes[k] < 0.01:
                del self.type_votes[k]

        is_fac = sub_type in FIXED_FACILITY_SUB_TYPES
        if is_fac and self.locked_type is None and not self._is_physically_static():
            # 行驶中的目标投来的设施票: 大概率是设备误报, 直接丢弃
            return

        self.type_votes[sub_type] = self.type_votes.get(sub_type, 0.0) + rel
        self.type_obs_count += 1

        if not self.type_votes:
            return
        best_type = max(self.type_votes, key=self.type_votes.get)
        best_votes = self.type_votes[best_type]
        total = sum(self.type_votes.values())
        if total <= 0:
            return
        ratio = best_votes / total

        if self.locked_type is None:
            # ---- 阶段 1: 未锁定, 寻找达标类型 ----
            if best_type in FIXED_FACILITY_SUB_TYPES:
                if ratio >= TYPE_FAC_LOCK_RATIO and self.type_obs_count >= TYPE_FAC_LOCK_FRAMES \
                        and self._is_physically_static():
                    self.locked_type = best_type
                    self.locked_obj_type = obj_type
            elif ratio >= TYPE_LOCK_RATIO and self.type_obs_count >= TYPE_LOCK_FRAMES:
                self.locked_type = best_type
                self.locked_obj_type = obj_type
        else:
            # ---- 阶段 2: 已锁定, 允许压倒性反证纠正 ----
            if best_type != self.locked_type and ratio >= TYPE_FIX_RATIO \
                    and best_votes > self.type_votes.get(self.locked_type, 0.0):
                # 已锁定为设施且目标确实静止: 保持设施身份, 不纠正
                if self.locked_type in FIXED_FACILITY_SUB_TYPES and self._is_physically_static():
                    return
                old_lock = self.locked_type
                self.locked_type = best_type
                self.locked_obj_type = obj_type
                print(f'[车型纠正] ID:{int(self.fixed_id) % 10000} | {old_lock} -> {best_type}')


class LaneQueueTracker:
    def __init__(self, map_manager):
        self.last_update_time = None
        self.map_mgr = map_manager
        # 核心数据结构：按 lane_id 分组维护车辆列表
        self.lane_queues = {}
        self.active_vehicles = {}  # fixed_id -> VehicleState



    def process_frame(self, raw_vehicles, current_time):
        current_radar_ids = set()

        # �� 预处理：先找出本帧已经有明确 ID 匹配的在线健康车辆
        # 这些车辆获得了真正的雷达点，绝不能再让其他噪点或断联车去缝合/吸附它们！
        direct_match_ids = {rv.object_id for rv in raw_vehicles if rv.object_id in self.active_vehicles}

        for rv in raw_vehicles:
            # ==========================================
            # 提取历史状态，辅助更鲁棒的车道匹配
            # ==========================================
            fixed_id = rv.object_id

            # ==========================================
            # �� [新增] 边缘盲区入场航向角强制初始化 (防止冷启动原地旋转)
            # ==========================================
            if fixed_id not in self.active_vehicles:  # 仅针对首帧出现的新车
                if self.map_mgr.zone_mgr.is_in_zone(rv.rel_x, rv.rel_y, 'AREA_INTER_R'):
                    # 用配置的合理航向角强行覆盖雷达不准确的初始瞬时值
                    rv.radar_heading = getattr(Config, 'ENTRY_HEADING_INTER_R', 162.0)

            old_veh = self.active_vehicles.get(fixed_id)

            # ==========================================
            # 🚀 固定设施身份防劫持：
            # 设备偶发把固定设施的检测点错绑到某辆车的 ID 上。
            # 若历史状态是已确认车辆(非未知)、新点却是设施(13/14)，必为误绑定：
            # 直接丢弃该点，保护车辆轨迹不被设施位置污染。
            # (反向"车辆点错绑到设施ID"由位置锚定机制冻结，无需丢弃)
            # ==========================================
            if old_veh is not None:
                _old_sub = old_veh.attrs.get("itc_sub_type", 99)
                if rv.itc_sub_type in FIXED_FACILITY_SUB_TYPES \
                        and _old_sub not in FIXED_FACILITY_SUB_TYPES and _old_sub != 99:
                    continue

            veh_v = old_veh.v if old_veh else 0.0
            last_lane = old_veh.lane_id if old_veh else None

            # 默认使用雷达瞬时航向
            veh_heading = rv.radar_heading

            # ==========================================
            # 海一路西侧入轨轨迹拟合与向量夹角车道仲裁
            # ==========================================
            forced_haiyi_lane, hijacked_heading = self.haiyi_match_to_lane(old_veh, rv, fixed_id)

            # ==========================================
            # 坐标 (x,y) 与 航向角 (heading) 双层前置预滤波
            # ==========================================
            # 如果触发了航向角劫持 (hijacked_heading 不为 None)，则将干净的拟合角度喂给角度滤波器！
            # 否则，按正常流程将雷达瞬时航向 rv.radar_heading 喂给滤波器
            input_heading = hijacked_heading if hijacked_heading is not None else rv.radar_heading

            if old_veh:
                # 针对追踪到的老车，同时对 2D 坐标和 360° 航向角执行滤波
                match_x, match_y = old_veh.update_raw_xy(rv.rel_x, rv.rel_y, current_time)
                match_heading = old_veh.update_raw_heading(input_heading)
            else:
                # 新车首帧无历史惯性，直接沿用原值
                match_x, match_y = rv.rel_x, rv.rel_y
                match_heading = input_heading

            # ==========================================
            # 初始雷达点映射, 车道匹配
            # ==========================================
            lane_id, s, l = self.map_mgr.match_to_lane(
                match_x, match_y,
                veh_heading=match_heading,
                v=veh_v,
                last_lane_id=last_lane,
                base_max_dist=2.5,  # 基础阈值设为 3.0 米
                forced_lane=forced_haiyi_lane  # �� 新增入参：向量裁决车道
            )

            # ==========================================
            # 透传属性
            # ==========================================
            attrs = {
                "itc_obj_type": rv.itc_obj_type,
                "plate_num": rv.plate_num,
                "lane_no": rv.lane_no,
                "type_reliability": rv.type_reliability,
                "itc_sub_type": rv.itc_sub_type
            }

            # ==========================================
            # 防劫持护盾 (将会车错乱限制在海一路)
            # ==========================================
            if fixed_id in self.active_vehicles:
                old_veh = self.active_vehicles[fixed_id]
                dt_sec = max((current_time - old_veh.last_radar_time) / 1000.0, 0.1)
                is_hijacked = False

                # # 基础物理位移护盾 (全地图通用)：收紧至 15.0*dt + 4.0                                           有必要吗???????
                # dist_2d = math.hypot(match_x - old_veh.raw_x, match_y - old_veh.raw_y)
                # if dist_2d > (20.0 * dt_sec + 4.0):
                #     is_hijacked = True
                #     print(f'hijack fixed: T  ->  {int(fixed_id)%10000}')


                # ��️ 海一路专属会车防错乱护盾 (仅在 490 和 506 车道生效)
                haiyi_lanes = {'137438953490_1', '137438953490_2', '137438953506_1', '137438953506_2'}
                if old_veh.lane_id in haiyi_lanes:
                    # 检查 A: 会车瞬间跳向了方向相反的对向车道，且没有触发合法借道意图
                    if lane_id in haiyi_lanes and lane_id != old_veh.lane_id:
                        # �� 修复核心：如果上一次的横向偏移 abs(old_veh.filtered_l) 很小（小于 1.0m，说明车在老车道中心），
                        # 却瞬间跳到了对向车道，那是噪点劫持。
                        # 但如果 abs(old_veh.filtered_l) 已经 > 1.0m，说明车已经真实压线偏移，必须放行允许变道！
                        if old_veh.lane_change_counter == 0 and abs(old_veh.filtered_l) < 1.0:
                            is_hijacked = True
                            print('hijack fixed: A')

                    # 检查 B: 雷达航向角突变 (>90度说明在对向车道会车时误认成了来车)
                    heading_jump = abs((old_veh.raw_heading - rv.radar_heading + 180) % 360 - 180)
                    if heading_jump > 90 and dt_sec < 1.0 and not getattr(old_veh, 'is_reverse_driving', False):
                        is_hijacked = True
                        print('hijack fixed: B')

                if is_hijacked:
                    # �� 核心修复：判定为错乱或劫持时，直接丢弃该噪点(continue)
                    # 绝不生成随机临时ID，彻底从源头掐灭“幽灵发生器”和图标拖尾！
                    continue

            # ==========================================
            # 全域 2D 物理缝合 (解决换道残留)
            # ==========================================     T路口不匹配
            if fixed_id not in self.active_vehicles and not self.map_mgr.zone_mgr.is_in_zone(match_x, match_y, 'AREA_T'):
                matched_id = self._find_best_match_2d(match_x, match_y, rv.radar_heading, current_time,
                                                      incoming_sub_type=rv.itc_sub_type)
                if matched_id:
                    fixed_id = matched_id

            # # 【单帧去重防抖】                                                                          (注释后缓解轨迹卡顿)
            # if fixed_id in current_radar_ids: continue

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
                    veh.update_s_and_chase(current_time, s)  # ✅ 应用追击逻辑

                elif veh.lane_id is not None:
                    # 【情况 B：触发变道意图 (老车道 -> 新车道)】
                    if veh.pending_lane_id == lane_id:
                        veh.lane_change_counter += 1
                    else:
                        veh.pending_lane_id = lane_id
                        veh.lane_change_counter = 1

                    # 计算它相对【老车道】的真实横向距离，并丢入低通滤波
                    # ✅ 统一使用滤波后坐标 (与情况A的 S/L 基线完全一致):
                    #    旧代码用原始雷达坐标投影, 与情况A的滤波坐标之间存在
                    #    1~2 米的基线差, 车辆一进入变道确认期 S 就会瞬间前跳,
                    #    触发输出级 3m 硬切保护造成图标跳变
                    old_line = self.map_mgr.lanes[veh.lane_id]['line']
                    pt = Point(match_x, match_y)
                    # dist_to_old = old_line.distance(pt)
                    dist_to_old = self.map_mgr.get_signed_offset(veh.lane_id, match_x, match_y)
                    veh.update_l(dist_to_old)

                    # 门限确认：连续 5 帧匹配到同一新车道，且相对老车道的真实偏离距离 > 1.0 米
                    # (改用真实距离而非滤波值 filtered_l: 低通滤波的滞后会让确认时机推迟数秒)
                    if veh.lane_change_counter >= 5 and abs(dist_to_old) > 1.0:
                    # if veh.lane_change_counter >= 5:    # 门限确认：连续 4 帧匹配到同一新车道
                        # ==========================================
                        # �� 新增：变道平滑过渡机制 (目前仅针对海一路双向车道特化)
                        # ==========================================
                        haiyi_lanes = {'137438953490_1', '137438953490_2', '137438953506_1', '137438953506_2'}
                        if veh.lane_id in haiyi_lanes and lane_id in haiyi_lanes:
                            new_x, new_y = self.map_mgr.get_xy_from_s(lane_id, s)
                            if new_x is not None:
                                veh.is_changing_lane = True
                                veh.lc_start_time = current_time

                                start_x = veh.out_x if veh.out_x is not None else veh.raw_x
                                start_y = veh.out_y if veh.out_y is not None else veh.raw_y
                                veh.lc_offset_x = start_x - new_x
                                veh.lc_offset_y = start_y - new_y

                                new_map_heading = self.map_mgr.lanes[lane_id]['heading']
                                target_geo = (90 - new_map_heading) % 360

                                # 判定逆行：对比雷达原生航向与新车道正常航向 (如果反向，说明是逆行超车)
                                angle_diff = abs((target_geo - veh.raw_heading + 180) % 360 - 180)
                                if angle_diff > 90:
                                    veh.is_reverse_driving = True
                                    target_geo = (target_geo + 180) % 360
                                else:
                                    veh.is_reverse_driving = False

                                start_heading = getattr(veh, 'out_heading', veh.raw_heading)
                                veh.lc_offset_heading = (start_heading - target_geo + 180) % 360 - 180
                        else:
                            veh.is_changing_lane = False
                            veh.is_reverse_driving = False
                        # ==========================================
                        # 🚗 车道系反转检测: 互为对向的车道互变(如海一路借道)时
                        # S 轴方向翻转, 历史方向状态失效, 需基于新车道重新建立
                        # ==========================================
                        old_map_h = self.map_mgr.lanes[veh.lane_id]['heading']
                        new_map_h = self.map_mgr.lanes[lane_id]['heading']
                        if abs((old_map_h - new_map_h + 180) % 360 - 180) > 90:
                            veh.reset_drive_direction()
                        # ==========================================
                        # ✅ 正式确认变道！
                        veh.s_history.clear()
                        veh.signed_v = 0.0  # <--- 新增：清空历史的同时重置方向速度
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
                    # 【情况 C：之前是离线游离状态，现在成功上轨 (车辆入道)】
                    # ======== 新增：记录入道瞬间的时空偏差 ========
                    new_x, new_y = self.map_mgr.get_xy_from_s(lane_id, s)
                    if new_x is not None:
                        veh.is_changing_lane = True
                        veh.lc_start_time = current_time

                        # 视觉起点：优先用之前的平滑输出坐标，若无则用当前真实的 2D 坐标
                        start_x = veh.out_x if veh.out_x is not None else veh.raw_x
                        start_y = veh.out_y if veh.out_y is not None else veh.raw_y
                        veh.lc_offset_x = start_x - new_x
                        veh.lc_offset_y = start_y - new_y

                        # 计算航向角偏差（预测目标车道地理角，并兼容 180 度倒车翻转）
                        new_map = self.map_mgr.lanes[lane_id]['heading']
                        target_geo = (90 - new_map) % 360

                        # === [修改] 海一路入道逆行判定 ===
                        haiyi_lanes = {'137438953490_1', '137438953490_2', '137438953506_1', '137438953506_2'}
                        if lane_id in haiyi_lanes:
                            angle_diff = abs((target_geo - veh.raw_heading + 180) % 360 - 180)
                            if angle_diff > 90:
                                veh.is_reverse_driving = True
                                target_geo = (target_geo + 180) % 360
                            else:
                                veh.is_reverse_driving = False
                        else:
                            veh.is_reverse_driving = False
                            # 保留原有其他车道的掉头兼容逻辑
                            angle_diff = abs((target_geo - veh.raw_heading + 180) % 360 - 180)
                            if angle_diff > 150 and (lane_id != '137438953490_1') and (lane_id != '137438953490_2'):
                                target_geo = (target_geo + 180) % 360

                        start_heading = getattr(veh, 'out_heading', veh.raw_heading)
                        veh.lc_offset_heading = (start_heading - target_geo + 180) % 360 - 180
                    # ==========================================

                    veh.s_history.clear()
                    veh.signed_v = 0.0  # <--- 新增：同样重置
                    veh.reset_drive_direction()  # 🚗 S 坐标系更换, 行驶方向需重新建立
                    veh.lane_id = lane_id
                    veh.s = s
                    veh.target_s = s
                    veh.is_chasing = False
                    veh.filtered_l = l  # 🚗 车道系已更换, 直接采用新车道横向偏移 (旧滤波值无连续性意义)
                    veh.v = veh.update_and_estimate_speed(current_time, s)
                    veh.pending_lane_id = None
                    veh.lane_change_counter = 0
                    # ✅ 保留 out_x/out_y 平滑缓存: 上轨起点承接离道时的最后输出,
                    #    配合变道动画实现离道->上轨的坐标无缝衔接

                # 统一更新基础属性 (🚗 车型投票锁定, 取代旧的"首次非99即永久锁定")
                self._update_vehicle_type(veh, rv, attrs, current_time)

                veh.attrs = attrs
                veh.last_radar_time = current_time
                veh.is_off_lane = False
                # update_raw_xy 已经更新过 raw_x/y
                # veh.raw_x = rv.rel_x
                # veh.raw_y = rv.rel_y
                # veh.raw_heading = rv.radar_heading

        self._update_physics_queues(current_time, current_radar_ids)
        self._cleanup_stale_vehicles(current_time)
        return self._generate_processed_vehicles(current_time)

    def _update_vehicle_type(self, veh, rv, attrs, current_time):
        """
        🚗 车型投票锁定的统一入口: 先投票, 再以锁定结果覆写透传属性。
        未锁定期间保持设备原始输出, 锁定后以投票裁决值为准。
        """
        veh.vote_type(rv.itc_sub_type, rv.itc_obj_type, rv.type_reliability, current_time)
        if veh.locked_type is not None and attrs.get("itc_sub_type") != veh.locked_type:
            attrs["itc_sub_type"] = veh.locked_type
            if veh.locked_obj_type is not None:
                attrs["itc_obj_type"] = veh.locked_obj_type

    def haiyi_match_to_lane(self, old_veh, rv, fixed_id):
        """
        海一路西侧入轨轨迹拟合与向量夹角车道仲裁
        """
        forced_haiyi_lane = None  # 记录向量裁决出的目标车道
        hijacked_heading = None  # �� 新增：记录需要向外传递的劫持航向角

        if old_veh:
            old_veh.xy_history.append((rv.rel_x, rv.rel_y))

            if '137438953506_1' in self.map_mgr.lanes and '137438953490_2' in self.map_mgr.lanes:
                line_506 = self.map_mgr.lanes['137438953506_1']['line']
                pt = Point(rv.rel_x, rv.rel_y)
                dist_506 = line_506.distance(pt)

                # 1. 车辆在蓝色感知区 (2.5m ~ 8.0m) 内，持续进行线性拟合
                if 2.5 <= dist_506 <= 8.0 and old_veh.is_off_lane:
                    if len(old_veh.xy_history) >= 2:
                        xs = [p[0] for p in old_veh.xy_history]
                        ys = [p[1] for p in old_veh.xy_history]

                        if max(xs) - min(xs) > 0.01 or max(ys) - min(ys) > 0.01:
                            m, b = np.polyfit(xs, ys, 1)
                            dx = xs[-1] - xs[0]
                            dy = ys[-1] - ys[0]

                            vec_base = np.array([1, m])
                            vec_points = np.array([dx, dy])

                            # 根据首尾点确定方向，避免 180° 歧义
                            direction_sign = 1 if np.dot(vec_base, vec_points) >= 0 else -1
                            vec_fit = direction_sign * vec_base

                            # 得到轨迹拟合向量的数学极角 (-180° ~ 180°)
                            theta_fit_deg = math.degrees(math.atan2(vec_fit[1], vec_fit[0]))

                            # 缓存拟合的数学角度和地理角度
                            old_veh.haiyi_fitted_math_angle = theta_fit_deg
                            old_veh.haiyi_fitted_heading = (90 - theta_fit_deg) % 360

                elif dist_506 > 8.0:
                    old_veh.haiyi_fitted_math_angle = None
                    old_veh.haiyi_fitted_heading = None

                # 2. 逼近入轨界限 (< 2.5m)，利用向量夹角进行车道绝对仲裁！
                if dist_506 < 2.5 and getattr(old_veh, 'haiyi_fitted_math_angle', None) is not None:
                    fit_angle = old_veh.haiyi_fitted_math_angle

                    # 获取地图中两条对向车道的标准数学角度 (约 -72° 和 108°)
                    angle_506 = self.map_mgr.lanes['137438953506_1']['heading']
                    angle_490 = self.map_mgr.lanes['137438953490_2']['heading']

                    # --- 计算向量夹角 (0° ~ 180° 最小几何夹角) ---
                    # 与 L1 (506车道) 的夹角
                    diff_506 = abs((fit_angle - angle_506 + 180) % 360 - 180)
                    # 与 L2 (490车道) 的夹角
                    diff_490 = abs((fit_angle - angle_490 + 180) % 360 - 180)

                    # �� 核心裁决：谁的夹角小，说明车就是顺着哪条道开进来的！
                    if diff_506 <= diff_490:
                        forced_haiyi_lane = '137438953506_1'
                    else:
                        forced_haiyi_lane = '137438953490_2'

                    # 将雷达瞬时航向也劫持为拟合航向，提高稳定性
                    # �� 核心修复：将拟合出来的航向角通过变量向外传出，切断无效局部变量！
                    hijacked_heading = old_veh.haiyi_fitted_heading

                    print(
                        f"===== [海一路向量仲裁] ID:{int(fixed_id) % 10000} | 轨迹角:{fit_angle:.1f}° | 对506夹角:{diff_506:.1f}° | 对490夹角:{diff_490:.1f}° | 最终裁决: {forced_haiyi_lane} =====")

                    if not old_veh.is_off_lane:
                        old_veh.haiyi_fitted_math_angle = None
        # �� 必须同时返回 仲裁车道 和 劫持航向角
        return  forced_haiyi_lane, hijacked_heading

    def _find_best_match_2d(self, rel_x, rel_y, rv_heading, current_time, incoming_sub_type=99):
        """
        �� 全域 2D 物理缝合（融合版）：
        1. 严格保留原版“极近距离瞬间分裂噪点”的消除机制
        2. 将会车方向互斥严格限制在海一路(490/506)，彻底杜绝会车错乱与图标残留
        """
        haiyi_lanes = {'137438953490_1', '137438953490_2', '137438953506_1', '137438953506_2'}
        best_id = None
        min_dist = float('inf')

        for v_id, veh in self.active_vehicles.items():
            dist = math.hypot(veh.raw_x - rel_x, veh.raw_y - rel_y)
            time_diff = current_time - veh.last_radar_time
            dt_sec = max(time_diff / 1000.0, 0.1)

            # ==========================================
            # 🚀 固定设施身份隔离防线：
            # 1. 车辆点绝不认领固定设施的 ID (防止误绑定后设施跟随车辆移动)
            # 2. 固定设施点绝不认领车辆的 ID (防止设施瞬移进车流、污染车辆轨迹)
            # 3. 仅"未知类型(99)且距离 <= 5m"的点允许缝合设施
            #    (兼容设备偶发把设施上报为未知类型的 ID 跳变重连)
            # ==========================================
            if self._is_fixed_facility(veh):
                if incoming_sub_type not in FIXED_FACILITY_SUB_TYPES:
                    if not (incoming_sub_type == 99 and dist <= 5.0):
                        continue
            elif incoming_sub_type in FIXED_FACILITY_SUB_TYPES:
                continue

            # ==========================================
            # ��️ 隔离防线：海一路专属 - 运动学方向一票否决
            # ==========================================
            # 即使雷达点就在车身身边（dist < 20.0），只要当前车在海一路上，
            # 且雷达点的航向与历史航向相反（夹角 > 90°），说明这绝对是对向会车车辆的点！
            # 直接一票否决，绝不允许把对向车的点当成自己的分裂噪点认领！
            if veh.lane_id in haiyi_lanes:
                angle_diff = abs((veh.raw_heading - rv_heading + 180) % 360 - 180)
                # �� 修复：如果车辆处于长断联恢复期 (time_diff > 500ms)，第一帧航向极可能是噪点，暂不触发 90 度方向否决
                if angle_diff > 90 and not getattr(veh, 'is_reverse_driving', False):
                    if time_diff < 500:  # 仅对连续追踪的车辆实施严苛方向打击
                        continue

            # ==========================================
            # 核心缝合：完整复用原版的两类经典场景
            # ==========================================
            is_match = False

            # 场景 1: 无论是否换道/离线，只要断联且在合理距离内，大概率是它
            max_allow_dist = min(70.0, 30.0 + veh.v * dt_sec)
            if time_diff > 100 and dist < max_allow_dist:
                is_match = True

            # 场景 2: 极近距离雷达瞬间分裂噪点
            # (即使 time_diff <= 100 甚至本帧已匹配过，只要 < 20米，均强行认领！
            # 认领后，外部的 `if fixed_id in current_radar_ids: continue` 会将分裂噪点作为重复项完美抹除)
            elif dist < 18.0:  # 15
                is_match = True

            if is_match and dist < min_dist:
                min_dist = dist
                best_id = v_id

        return best_id

    # 🚀 固定设施类子类集合：这些目标位置固定不动，不参与 2D 去重，防止被误杀
    # itc_sub_type: 舱盖板=14, 候工亭/锁销框=13
    DEDUP_EXEMPT_SUB_TYPES = FIXED_FACILITY_SUB_TYPES

    @classmethod
    def _is_fixed_facility(cls, veh):
        """
        🚀 固定设施判定：舱盖板(14)、候工亭/锁销框(13)。
        这类目标物理位置恒定，全程享受"限移三重保护"：
        去重豁免 / 身份隔离 / 位置锚定。
        """
        return veh.attrs.get("itc_sub_type", 99) in FIXED_FACILITY_SUB_TYPES

    @classmethod
    def _is_dedup_exempt(cls, veh):
        """
        🚀 2D 去重豁免判定：固定设施(舱盖板(14)、候工亭/锁销框(13))
        因位置恒定、不会分裂移动，一律不参与 2D 去重仲裁，
        防止被其他目标去重误杀（对应现象：后台有检测结果，但平台上没有）。
        """
        return cls._is_fixed_facility(veh)

    def _update_physics_queues(self, current_time, current_radar_ids):
        # 初始化时间步长
        if self.last_update_time is None:
            self.last_update_time = current_time
        dt = (current_time - self.last_update_time) / 1000.0
        if dt <= 0:
            dt = 0.1  # 安全保护

        # ==========================================
        # �� [新增] 离道 (Off-lane) 车辆专属 2D 物理去重
        # ==========================================
        off_lane_vehicles = [veh for veh in self.active_vehicles.values() if veh.is_off_lane]
        off_lane_to_delete = set()

        for i in range(len(off_lane_vehicles)):
            for j in range(i + 1, len(off_lane_vehicles)):
                v1 = off_lane_vehicles[i]
                v2 = off_lane_vehicles[j]

                # 如果其中一个已经被标记为待删幽灵，则跳过
                if v1.fixed_id in off_lane_to_delete or v2.fixed_id in off_lane_to_delete:
                    continue

                # 🚀 固定设施去重豁免：只要任意一方是舱盖板(14),候工亭/锁销框(13)，
                # 本对目标不参与 2D 去重，防止固定目标被误杀
                if self._is_dedup_exempt(v1) or self._is_dedup_exempt(v2):
                    continue

                # 计算两个游离目标的 2D 物理距离
                dist_2d = math.hypot(v1.raw_x - v2.raw_x, v1.raw_y - v2.raw_y)

                # 判定阈值：2D距离 < 15.0米，且速度差 < 3.0m/s
                if dist_2d < 18.0:
                    # 仲裁标准与在轨车保持绝对一致：谁拥有更新鲜的雷达数据，谁才是真身！
                    if v1.last_radar_time >= v2.last_radar_time:
                        ghost = v2
                        survivor = v1
                        reason = "后车(离道)雷达数据陈旧"
                    else:
                        ghost = v1
                        survivor = v2
                        reason = "前车(离道)雷达数据陈旧"

                    off_lane_to_delete.add(ghost.fixed_id)
                    print(
                        f"��️ [离道去重] 删ID:{int(ghost.fixed_id) % 10000} | 留ID:{int(survivor.fixed_id) % 10000} | 原因: {reason} | 2D相距:{dist_2d:.2f}m")

        # 统一执行删除动作
        for ghost_id in off_lane_to_delete:
            if ghost_id in self.active_vehicles:
                del self.active_vehicles[ghost_id]
        # ==========================================

        self.lane_queues.clear()

        # 收集所有有车道信息的车辆（包括离线但有 last_lane_id 的）
        for veh in self.active_vehicles.values():
            if not veh.is_off_lane and veh.lane_id is not None:
                # ==========================================
                # �� [新增] 海一路 490 车道主动离道前置拦截
                # ==========================================
                # 如果车辆在内侧车道 (490) 上且当前帧已丢失雷达信号
                if veh.fixed_id not in current_radar_ids and (veh.lane_id == '137438953490_1' or veh.lane_id == '137438953490_2'):
                    # 检查其最后消失前的滤波横向偏移 l 是否显著偏向左侧 (正数)
                    # (偏移大于 2m 说明车身大半已压线或出线)
                    if veh.filtered_l > 2.0:
                        # 判定为横穿离场，直接剥夺在轨身份！
                        veh.is_off_lane = True
                        veh.last_lane_id = veh.lane_id
                        # veh.lane_id = None
                        # ✅ 保留 out_x/out_y: 离道后输出从最后平滑位置自然收敛到
                        #    最后物理位置, 避免图标瞬移
                        veh.is_changing_lane = False  # 打断任何可能的变道动画
                        print(
                            f"[离道拦截] ID:{int(veh.fixed_id) % 10000} 从 490 车道向左(l={veh.filtered_l:.2f})横穿离场，中止推演！")
                        continue  # 核心：直接跳过，不加入后续的 IDM 推演队列

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

                    # 🚀 固定设施去重豁免：只要任意一方是舱盖板(14)/候工亭/锁销框(13)，
                    # 本对目标不参与同车道去重合并，防止固定目标被误杀
                    if self._is_dedup_exempt(leader_veh) or self._is_dedup_exempt(veh):
                        survivors.append((veh, s_val))
                    elif (leader_s - s_val) < 15.0 and abs(leader_veh.v - veh.v) < 3.0:
                        # 仲裁：谁拥有更近的真实雷达点，谁才是真身！
                        if leader_veh.last_radar_time >= veh.last_radar_time:
                            ghost = veh  # 丢弃落后的/旧的
                            survivor = leader_veh
                            reason = "后车雷达数据陈旧"
                        else:
                            ghost = leader_veh  # 丢弃老的推演车
                            survivor = veh
                            survivors[-1] = (veh, s_val)
                            reason = "前车(推演车)雷达数据陈旧"

                        if ghost.fixed_id in self.active_vehicles:
                            # �� [新增日志 1]：打印被同车道真实车去重合并的原因
                            print(
                                f"��️ [去重合并] 车道:{lane_id} | 删ID:{int(ghost.fixed_id) % 10000} | 留ID:{int(survivor.fixed_id) % 10000} | 原因: {reason} | S相差:{leader_s - s_val:.2f}m")
                            del self.active_vehicles[ghost.fixed_id]
                    else:
                        survivors.append((veh, s_val))

            # 更新队列为去重后的 (vehicle, s) 列表
            self.lane_queues[lane_id] = survivors

            add_v = 1.5
            # ----- 推演每个车辆的 S（断联推演 or 在线追击）-----
            for idx, (veh, s_val) in enumerate(survivors):
                if veh.fixed_id not in current_radar_ids:
                    # ==========================================
                    # 🚀 固定设施：位置恒定，绝不参与 IDM 盲区推演
                    # (防止断检期间被"推演着走"，速度被虚增至 2m/s)
                    # ==========================================
                    if self._is_fixed_facility(veh):
                        continue

                    # ==========================================
                    # 1. 断联/离线状态：执行 IDM 盲区推演
                    # ==========================================
                    if veh.v < 1.0:
                        veh.v = 2.0  # 可调整，建议保守值 1.0~2.0

                    if idx == 0:
                        # 头车：匀速
                        new_s = s_val + (veh.v+add_v) * dt
                    else:
                        # 跟车：简单 IDM
                        leader_veh, leader_s = survivors[idx - 1]
                        dist = leader_s - s_val
                        if dist < 15.0:
                            veh.v = max(0, veh.v - 2.0 * dt)
                        else:
                            veh.v = leader_veh.v
                        new_s = s_val + (veh.v+add_v) * dt

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

                    print(f'idm: {int(veh.fixed_id)%10000}-{idx}')

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
                            catch_speed = 4.0  # 追击速度差：比真实车快/慢 6m/s

                            if gap > 0:
                                # 推演车在真实车后方，加速追赶
                                veh.s += (veh.v + catch_speed) * dt
                                if veh.s >= veh.target_s:
                                    veh.s = veh.target_s
                                    veh.is_chasing = False
                            else:
                                # �� 修复点：推演车冲过头了，原地等待。
                                # 但如果真实车停了，或者偏差过大(>15m)，切勿死等，直接硬拉回真实点解除追击！
                                if gap < -15.0 or veh.v < 0.5:
                                    veh.s = veh.target_s
                                    veh.is_chasing = False
                                elif veh.s <= veh.target_s:
                                    veh.s = veh.target_s
                                    veh.is_chasing = False
                    print(f'idm: {int(veh.fixed_id)%10000}-{idx}')
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
                # ✅ 离道车辆同样应用输出级平滑: 消除在轨(车道坐标)->离道(物理坐标)
                #    切换瞬间的图标跳变, 保证全程输出坐标连续
                x, y = self._alpha_filter_xy(veh, x, y)
            else:
                x, y = self.map_mgr.get_xy_from_s(veh.lane_id, veh.s)
                # ==========================================
                # 🚗 输出融合带符号横向偏移 L:
                # 旧逻辑只反算 S 坐标, 车辆永远贴在车道中心线上, 变道时输出
                # 严重滞后于真实位置。融合 L 后输出贴合真实雷达位置, 变道
                # 确认前车辆即可自然横向滑出, 确认后新老车道的坐标天然连续。
                # ==========================================
                lx, ly = self.map_mgr.offset_lateral(veh.lane_id, veh.s, veh.filtered_l)
                if lx is not None:
                    x, y = lx, ly
                map_heading = self.map_mgr.lanes[veh.lane_id]['heading']
                geo_heading = (90 - map_heading) % 360
                is_predicted = (current_time - veh.last_radar_time > 100)

                # ==========================================
                # 🚗 航向角最终防线: 由方向滞回状态机裁决 (根治车头 180° 甩头)
                # 旧策略直接用回归斜率 signed_v 正负(阈值0.5)翻转航向, 低速与
                # 雷达噪声下斜率在阈值附近抖动, 造成车头前后反复甩动。
                # 新策略: 只有方向状态机给出明确裁决时才覆写; 方向未知期回落到
                # 海一路逆行先验 / 雷达反向兼容逻辑兜底。
                # ==========================================
                base_map_geo = (90 - map_heading) % 360
                if veh.drive_direction > 0:
                    # S 持续增加: 沿车道正向行驶
                    geo_heading = base_map_geo
                elif veh.drive_direction < 0:
                    # S 持续减小: 逆车道方向行驶 (倒车/借道逆行)
                    geo_heading = (base_map_geo + 180) % 360
                else:
                    # 方向未确认期: 入场/变道先验与雷达反向兼容兜底
                    if getattr(veh, 'is_reverse_driving', False):
                        # 借道超车状态下，将基础航向强制翻转180度
                        geo_heading = (geo_heading + 180) % 360
                    else:
                        # 保留原有的雷达反向感知/倒车兼容处理
                        angle_diff = abs((geo_heading - veh.raw_heading + 180) % 360 - 180)
                        if angle_diff > 150 and not is_predicted and (veh.lane_id != '137438953490_1' and veh.lane_id != '137438953490_2'):
                            geo_heading = (geo_heading + 180) % 360
                            print(f'angle reverse : {int(veh.fixed_id)%10000}')

                # ==========================================
                # 注入平滑变道曲线与航向角插值
                # ==========================================
                if getattr(veh, 'is_changing_lane', False):
                    elapsed = current_time - veh.lc_start_time
                    if elapsed < veh.lc_duration:
                        # 核心数学：Cosine ease-in-out 函数，实现 (1.0 -> 0.0) 的丝滑过渡衰减
                        ratio = 0.5 * (1 + math.cos(math.pi * elapsed / veh.lc_duration))

                        x += veh.lc_offset_x * ratio
                        y += veh.lc_offset_y * ratio
                        geo_heading += veh.lc_offset_heading * ratio
                    else:
                        veh.is_changing_lane = False  # 动画自然结束，完全贴合新车道

                # 更新修正后的航向角
                geo_heading = geo_heading % 360
                heading_rad = math.radians(geo_heading)
                output_radar_heading = geo_heading
                # ==========================================

                if not veh.is_off_lane:
                    # �� 在轨车辆应用输出级平滑滤波。传递 is_lc 参数防止动画期间触发 3m 的硬切保护！
                    is_animating = getattr(veh, 'is_changing_lane', False)
                    x, y = self._alpha_filter_xy(veh, x, y, is_lc=is_animating)

            # === [新增]：无论离线还是在轨，缓存最后用于前端渲染的视角航向 ===
            veh.out_heading = output_radar_heading

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
    def _alpha_filter_xy(veh: VehicleState, x, y, alpha=0.2, is_lc=False):
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

            # ✅ 新增：如果正在播放变道动画(is_lc=True)，绝对禁止触发硬切拉扯机制！
            if dist_jump > 3.0 and not is_lc:
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
                # ✅ 保留输出平滑缓存 out_x/out_y: 在轨最后输出(s+l融合坐标)与
                #    离道物理坐标本就接近, 保留缓存配合输出级平滑可实现无缝衔接;
                #    置空反而会造成切换瞬间的图标瞬移
                veh.is_changing_lane = False  # ✅ 新增：离线时强制打断变道动画

            # �� 同步调用坐标与航向的预滤波方法
            veh.update_raw_xy(rv.rel_x, rv.rel_y, current_time)
            veh.update_raw_heading(rv.radar_heading)


            # 离道车辆同步应用车型投票锁定机制
            self._update_vehicle_type(veh, rv, attrs, current_time)

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

    def _cleanup_stale_vehicles(self, current_time, default_timeout=2500):
        """
        �� 双轨超时清理机制：
        1. 游离在车道外、无约束的废弃噪点点位 -> 2秒快速销毁，绝不拖尾
        2. 绑定在合规车道上、正由 IDM 引擎进行盲区推演的真实车辆 -> 给予 8秒 保护期！
        """
        expired_ids = []
        for v_id, veh in self.active_vehicles.items():
            time_silent = current_time - veh.last_radar_time

            # 🚀 固定设施：位置恒定，延长存活期至 FACILITY_TIMEOUT_MS (10秒)
            # (防止被过车短暂遮挡导致平台图标闪烁：后台有、平台上没有)
            if self._is_fixed_facility(veh):
                if time_silent > FACILITY_TIMEOUT_MS:
                    expired_ids.append(v_id)
                continue

            if veh.is_off_lane or veh.lane_id is None:
                # 游离状态：严格执行 2 秒快杀
                if time_silent > default_timeout:
                    expired_ids.append(v_id)
            else:
                # 盲区推演状态：放宽至 8000 毫秒 (8秒)，保障合法车不被超时误杀
                if time_silent > 8000:
                    expired_ids.append(v_id)

        if expired_ids:
            print("expired_ids: " + ' '.join(map(str, expired_ids)))
            for v_id in expired_ids:
                del self.active_vehicles[v_id]