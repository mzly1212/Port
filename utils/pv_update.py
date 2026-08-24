import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 核心规划函数（带加速度/加加速度限幅） ====================
def calc_quintic_coeffs(x0, v0, xf, vf, T):
    """五次多项式系数计算，边界加速度为0"""
    X = xf - x0
    if abs(T) < 1e-9:
        T = 1.0
    c0 = x0
    c1 = v0
    c2 = 0.0  # 初始加速度为0
    c3 = (10 * X - (4 * vf + 6 * v0) * T) / (T ** 3)
    c4 = (-15 * X + (7 * vf + 8 * v0) * T) / (T ** 4)
    c5 = (6 * X - 3 * (vf + v0) * T) / (T ** 5)
    return np.array([c0, c1, c2, c3, c4, c5])


def evaluate_trajectory(coeffs, t):
    """计算 t 时刻的位置、速度、加速度、加加速度"""
    c0, c1, c2, c3, c4, c5 = coeffs
    x = c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5
    v = c1 + 2 * c2 * t + 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4
    a = 2 * c2 + 6 * c3 * t + 12 * c4 * t ** 2 + 20 * c5 * t ** 3
    j = 6 * c3 + 24 * c4 * t + 60 * c5 * t ** 2
    return x, v, a, j


def check_constraints(coeffs, T, a_max, j_max, num_samples=5000):
    """检验是否满足约束"""
    t = np.linspace(0, T, num_samples)
    _, _, a, j = evaluate_trajectory(coeffs, t)
    return (np.max(np.abs(a)) <= a_max) and (np.max(np.abs(j)) <= j_max)


def find_optimal_time(x0, v0, xf, vf, a_max, j_max):
    """二分搜索满足约束的最短规划时间 T"""
    D = xf - x0
    v_avg = (v0 + vf) / 2.0
    if abs(v_avg) < 1e-6:
        T_low = 1.0
    else:
        T_low = abs(D / v_avg)  # 理论下界
    if T_low <= 0:
        T_low = 1.0
    T_high = max(T_low * 2, 2.0)

    # 指数增长找可行上界
    for _ in range(60):
        coeffs = calc_quintic_coeffs(x0, v0, xf, vf, T_high)
        if check_constraints(coeffs, T_high, a_max, j_max):
            break
        T_high *= 2.0

    # 二分精搜
    for _ in range(80):
        T_mid = (T_low + T_high) / 2.0
        coeffs = calc_quintic_coeffs(x0, v0, xf, vf, T_mid)
        if check_constraints(coeffs, T_mid, a_max, j_max):
            T_high = T_mid
        else:
            T_low = T_mid
    return T_high


# ==================== 场景参数设置 ====================
x0, v0 = 0.0, 10.0  # 当前位置(m)，当前速度(m/s)
xf, vf = 120.0, 18.0  # 目标位置(m)，目标速度(m/s) —— 加速通过，无需减速
a_max = 3.5  # 最大加速度限制 (m/s²)
j_max = 6.0  # 最大加加速度限制 (m/s³)

print("正在规划轨迹并计算最优时间...")
T_opt = find_optimal_time(x0, v0, xf, vf, a_max, j_max)
coeffs = calc_quintic_coeffs(x0, v0, xf, vf, T_opt)

# 生成高密度时间序列用于绘图
t_samples = np.linspace(0, T_opt, 3000)
x_vals, v_vals, a_vals, j_vals = evaluate_trajectory(coeffs, t_samples)

print(f"规划完成！总时长 T = {T_opt:.3f} 秒")
print(f"峰值加速度 = {np.max(np.abs(a_vals)):.3f} m/s² (限制 {a_max})")
print(f"峰值加加速度 = {np.max(np.abs(j_vals)):.3f} m/s³ (限制 {j_max})")

# ==================== 创建动态可视化布局 ====================
fig = plt.figure(figsize=(12, 10))
fig.suptitle("一维车辆平滑变速动态演示 (加速度/加加速度限幅)", fontsize=16, y=0.98)

# 使用 GridSpec 灵活布局
gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1, 1, 1], hspace=0.3)
ax1 = fig.add_subplot(gs[0])  # 道路/位置
ax2 = fig.add_subplot(gs[1])  # 速度曲线
ax3 = fig.add_subplot(gs[2])  # 加速度曲线
ax4 = fig.add_subplot(gs[3])  # 加加速度曲线

# ---------- 图1：一维道路 ----------
ax1.set_xlim(x0 - 20, xf + 20)
ax1.set_ylim(-1.5, 1.5)
ax1.set_ylabel("位置 (m)")
ax1.set_yticks([])  # 隐藏y轴刻度
ax1.axhline(y=0, color='black', linewidth=2, alpha=0.5)  # 道路线
# 起点和终点标记
ax1.axvline(x=x0, color='blue', linestyle='--', alpha=0.6, label=f'起点 {x0}m')
ax1.axvline(x=xf, color='red', linestyle='--', alpha=0.6, label=f'目标 {xf}m')
# 车辆（动态更新的点）
vehicle, = ax1.plot([], [], 'bo', markersize=18, zorder=5)
# 位置标签（动态）
pos_text = ax1.text(x0, 0.8, '', fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax1.legend(loc='upper left')

# ---------- 图2：速度曲线 ----------
ax2.plot(t_samples, v_vals, 'g-', linewidth=2, alpha=0.6, label='速度曲线')
ax2.axhline(y=v0, color='gray', linestyle=':', alpha=0.5, label=f'初速 {v0}')
ax2.axhline(y=vf, color='gray', linestyle=':', alpha=0.5, label=f'末速 {vf}')
ax2.set_ylabel("速度 (m/s)")
ax2.set_xlim(0, T_opt * 1.05)
ax2.set_ylim(min(v0, vf) - 2, max(v0, vf) + 2)
ax2.grid(True, linestyle='--', alpha=0.3)
# 速度游标
point_v, = ax2.plot([], [], 'ro', markersize=10, zorder=5)
line_v, = ax2.plot([], [], 'r--', linewidth=1, alpha=0.5)  # 垂直虚线
ax2.legend(loc='upper right')

# ---------- 图3：加速度曲线 ----------
ax3.plot(t_samples, a_vals, 'orange', linewidth=2, alpha=0.6, label='加速度')
ax3.axhline(y=a_max, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'限幅 ±{a_max}')
ax3.axhline(y=-a_max, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax3.fill_between(t_samples, -a_max, a_max, alpha=0.08, color='red')
ax3.set_ylabel("加速度 (m/s²)")
ax3.set_xlim(0, T_opt * 1.05)
ax3.grid(True, linestyle='--', alpha=0.3)
point_a, = ax3.plot([], [], 'ro', markersize=10, zorder=5)
line_a, = ax3.plot([], [], 'r--', linewidth=1, alpha=0.5)
ax3.legend(loc='upper right')

# ---------- 图4：加加速度曲线 ----------
ax4.plot(t_samples, j_vals, 'purple', linewidth=2, alpha=0.6, label='加加速度')
ax4.axhline(y=j_max, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'限幅 ±{j_max}')
ax4.axhline(y=-j_max, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax4.fill_between(t_samples, -j_max, j_max, alpha=0.08, color='red')
ax4.set_xlabel("时间 (s)")
ax4.set_ylabel("加加速度 (m/s³)")
ax4.set_xlim(0, T_opt * 1.05)
ax4.grid(True, linestyle='--', alpha=0.3)
point_j, = ax4.plot([], [], 'ro', markersize=10, zorder=5)
line_j, = ax4.plot([], [], 'r--', linewidth=1, alpha=0.5)
ax4.legend(loc='upper right')

# 顶部实时数据显示
time_text = fig.text(0.5, 0.93, '', fontsize=14, ha='center',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))


# ==================== 动画更新函数 ====================
def update(frame):
    """每一帧的更新逻辑"""
    t = t_samples[frame]
    x, v, a, j = evaluate_trajectory(coeffs, t)

    # 1. 更新车辆位置（道路图）
    vehicle.set_data([x], [0])
    pos_text.set_position((x, 0.8))
    pos_text.set_text(f'{x:.1f} m')

    # 2. 更新速度曲线游标
    point_v.set_data([t], [v])
    line_v.set_data([t, t], [ax2.get_ylim()[0], ax2.get_ylim()[1]])

    # 3. 更新加速度曲线游标
    point_a.set_data([t], [a])
    line_a.set_data([t, t], [ax3.get_ylim()[0], ax3.get_ylim()[1]])

    # 4. 更新加加速度曲线游标
    point_j.set_data([t], [j])
    line_j.set_data([t, t], [ax4.get_ylim()[0], ax4.get_ylim()[1]])

    # 5. 更新顶部数据面板
    time_text.set_text(
        f'⏱️ 时间: {t:.2f}s   |   📍 位置: {x:.1f} m   |   🚀 速度: {v:.2f} m/s   |   '
        f'⚡ 加速度: {a:.2f} m/s²   |   🔄 加加速度: {j:.2f} m/s³'
    )

    # 返回所有更新对象
    return vehicle, pos_text, point_v, line_v, point_a, line_a, point_j, line_j, time_text


# ==================== 运行动画 ====================
ani = FuncAnimation(
    fig, update, frames=range(0, len(t_samples), 3),  # 每3个点取一帧，加快播放速度
    interval=20,  # 每帧间隔20ms (约50fps)
    blit=False,  # 设为False避免复杂对象重绘报错
    repeat=True
)

plt.tight_layout()
plt.show()

# 如果想把动画保存为GIF，可以取消下面注释（需要安装pillow）
# ani.save('smooth_vehicle.gif', writer='pillow', fps=30)
print("动画播放中，请关闭图形窗口结束程序...")