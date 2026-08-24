import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 存储所有的点击点
points_x = []
points_y = []

# ================== 1. 向量与几何参数初始化 ==================
# L1 角度 (-72°, 即车道137438953506)，指向下方
angle1_deg = -72
theta1 = np.radians(angle1_deg)
# L1 方向单位向量
u1 = np.array([np.cos(theta1), np.sin(theta1)])

# L2 角度 (108°, 即车道137438953490)，指向上方，与L1平行且反向
angle2_deg = 108
theta2 = np.radians(angle2_deg)
# L2 方向单位向量
u2 = np.array([np.cos(theta2), np.sin(theta2)])

# 定义垂直向量，用于平移
n_p = np.array([np.sin(theta1), -np.cos(theta1)])  # “左侧”法向量

# 参考点
P0 = np.array([5.0, 5.0])  # L1 穿过的基准点
P2 = P0 + 5.0 * (-n_p)  # L2 位于 L1 右侧 5m (沿 -n_p 方向)

# ================== 2. 初始化画布 ==================
fig, ax = plt.subplots(figsize=(10, 8))
# 必须设置为 equal，否则视觉上的 5m 距离和角度会发生畸变
ax.set_aspect('equal')
ax.set_xlim(-5, 15)
ax.set_ylim(-5, 15)
ax.grid(True, linestyle='--', alpha=0.5)

# ================== 3. 绘制参考线、感知区域及箭头 ==================
# T 用于定义参考线伸展长度
T = 30
t_vals = np.array([-T, T])

# 画 L1
L1_x = P0[0] + t_vals * u1[0]
L1_y = P0[1] + t_vals * u1[1]
ax.plot(L1_x, L1_y, 'g--', linewidth=2, alpha=0.8, label=f'L1 参考线 ({angle1_deg}°)')

# 画 L2
L2_x = P2[0] + t_vals * u2[0]
L2_y = P2[1] + t_vals * u2[1]
ax.plot(L2_x, L2_y, color='orange', linestyle='--', linewidth=2, alpha=0.8, label=f'L2 参考线 ({angle2_deg}°)')

# 画箭头 (起点为参考点，顺着方向向量延伸)
arrow_len = 3.0
ax.arrow(P0[0], P0[1], u1[0] * arrow_len, u1[1] * arrow_len,
         head_width=0.6, head_length=0.8, fc='green', ec='green', zorder=5)
ax.arrow(P2[0], P2[1], u2[0] * arrow_len, u2[1] * arrow_len,
         head_width=0.6, head_length=0.8, fc='orange', ec='orange', zorder=5)

# ========== 修改点 1：绘制感知区域 (L1 沿 n_p方向外推 2.5m 到 8.0m) ==========
d_min = 2.5
d_max = 8.0
poly_A = P0 + T * u1 + d_min * n_p  # 起点向左推 2.5m
poly_B = P0 - T * u1 + d_min * n_p  # 终点向左推 2.5m
poly_C = P0 - T * u1 + d_max * n_p  # 终点向左推 8.0m
poly_D = P0 + T * u1 + d_max * n_p  # 起点向左推 8.0m
zone = Polygon(np.array([poly_A, poly_B, poly_C, poly_D]),
               closed=True, color='blue', alpha=0.1, label=f'感知区域 (L1左侧 {d_min}m 到 {d_max}m)')
ax.add_patch(zone)

# ================== 4. 动态图层初始化 ==================
# 区分有效点(红色)和无效点(灰色)
scatter_valid, = ax.plot([], [], 'ro', markersize=8, label='车辆有效轨迹点')
scatter_invalid, = ax.plot([], [], 'o', color='gray', markersize=6, alpha=0.5, label='无效点(区域外)')
line_plot, = ax.plot([], [], 'b-', linewidth=2, label='拟合直线')

# 调整图例位置放置在画布外，避免遮挡
ax.legend()


def onclick(event):
    if event.inaxes != ax:
        return

    # 添加或清空点
    if event.button == 1:
        points_x.append(event.xdata)
        points_y.append(event.ydata)
    elif event.button == 3:
        points_x.clear()
        points_y.clear()

    # == 核心逻辑：判断点是否在感知区域内 ==
    valid_x, valid_y = [], []
    invalid_x, invalid_y = [], []

    for x, y in zip(points_x, points_y):
        w = np.array([x, y]) - P0
        d_p = np.dot(w, n_p)

        # ========== 修改点 2：距离过滤条件修改为 2.5 到 8.0 ==========
        if d_min <= d_p <= d_max:
            valid_x.append(x)
            valid_y.append(y)
        else:
            invalid_x.append(x)
            invalid_y.append(y)

    scatter_valid.set_data(valid_x, valid_y)
    scatter_invalid.set_data(invalid_x, invalid_y)

    # 仅使用有效点 (感知区域内的点) 进行拟合
    if len(valid_x) >= 2:
        m, b = np.polyfit(valid_x, valid_y, 1)
        curr_xlim = np.array(ax.get_xlim())
        y_vals = m * curr_xlim + b
        line_plot.set_data(curr_xlim, y_vals)

        # 确定拟合线方向 (基于有效点)
        dx = valid_x[-1] - valid_x[0]
        dy = valid_y[-1] - valid_y[0]

        vec_base = np.array([1, m])
        vec_points = np.array([dx, dy])

        if dx == 0 and dy == 0:
            direction_sign = 1
        else:
            direction_sign = 1 if np.dot(vec_base, vec_points) >= 0 else -1

        vec_fit = direction_sign * vec_base
        theta_fit_deg = np.degrees(np.arctan2(vec_fit[1], vec_fit[0]))

        # === 计算与 L1 和 L2 的矢量夹角 ===
        # L1 夹角
        angle_diff_L1 = theta_fit_deg - angle1_deg
        angle_diff_L1 = (angle_diff_L1 + 180) % 360 - 180

        # L2 夹角
        angle_diff_L2 = theta_fit_deg - angle2_deg
        angle_diff_L2 = 180 - (angle_diff_L2 + 180) % 360

        # 更新标题展示两条线的夹角
        title_str = (f'拟合方程: y = {m:.2f}x + {b:.2f} | 有效点数: {len(valid_x)}\n'
                     f'拟合方向: {theta_fit_deg:.1f}° | 对L1夹角: {angle_diff_L1:.1f}° | 对L2夹角: {angle_diff_L2:.1f}°')
        ax.set_title(title_str, fontsize=13)
    else:
        line_plot.set_data([], [])
        ax.set_title('交互式直线拟合\n(请在蓝色【感知区域】内至少添加 2 个有效点)', fontsize=14)

    fig.canvas.draw()


fig.canvas.mpl_connect('button_press_event', onclick)
ax.set_title('交互式直线拟合\n(左键添加点，右键清空所有点)', fontsize=14)
plt.show()