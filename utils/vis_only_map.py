import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 1. 加载地理数据
# with open('../map/area.geojson', 'r') as f:
#     areas = json.load(f)

with open('../map/merged_lanes.geojson', 'r') as f:
    lanes = json.load(f)

# 2. 初始化画布
fig, ax = plt.subplots(figsize=(12, 10))

# 3. 绘制静态高精地图（车道）
for feature in lanes['features']:
    if feature['geometry']['type'] == 'LineString':
        xs = [pt[0] for pt in feature['geometry']['coordinates']]
        ys = [pt[1] for pt in feature['geometry']['coordinates']]
        ax.plot(xs, ys, color='blue', linewidth=1, alpha=0.3)

# # （可选）绘制区域，如有需要可取消注释
# for feature in areas['features']:
#     if feature['geometry']['type'] == 'Polygon':
#         for ring in feature['geometry']['coordinates']:
#             xs = [pt[0] for pt in ring]
#             ys = [pt[1] for pt in ring]
#             ax.fill(xs, ys, color='lightgray', edgecolor='black', alpha=0.5)


# 四个点：左下、右下、右上、左上（可任意调整）
# points = [
#     (113.865855, 22.492540),
#     (113.865979, 22.492562),
#     (113.863711, 22.499154),
#     (113.863580, 22.499110)
# ]
# points = [
#     (113.865442, 22.491884),
#     (113.867130, 22.492423),
#     (113.864547, 22.499801),
#     (113.862874, 22.499315)
# ]
# points = [
#     (113.855752, 22.485587),
#     (113.859742, 22.485656),
#     (113.859169, 22.490425),
#     (113.855935, 22.490059)
# ]


# polygon = Polygon(
#     points,
#     closed=True,
#     edgecolor='red',
#     facecolor='none',
#     linewidth=2,
#     linestyle='--',
#     label='Target Area'
# )
# ax.add_patch(polygon)
ax.legend()

# 5. 设置图形属性
ax.set_aspect('equal', 'datalim')
plt.title('Port Map with Custom Quadrilateral (Click to print coordinates)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# ---------- 新增：鼠标点击事件 ----------
def on_click(event):
    # 检查鼠标点击是否在坐标轴内（避免点击工具栏等区域）
    if event.inaxes == ax:
        lon, lat = event.xdata, event.ydata
        print(f"Clicked at: Longitude = {lon:.6f}, Latitude = {lat:.6f}")
    else:
        print("Click outside axes")

# 绑定点击事件到画布
fig.canvas.mpl_connect('button_press_event', on_click)

# 6. 显示静态地图（阻塞式，等待交互）
plt.show()