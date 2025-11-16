import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, MultiPoint
import json


def get_perpendicular_line_from_p2(p1, p2, length=5.0):
    """
    通过 P2 点构建一条垂直于线段 p1p2 的线段，并从 P2 点延长该垂线
    :param p1: 原始线段的第一个端点 (x1, y1)
    :param p2: 原始线段的第二个端点 (x2, y2)
    :param length: 垂线的长度
    :return: 返回通过 P2 点的垂线的 LineString 对象
    """
    # 创建原始线段
    line = LineString([p1, p2])

    # 计算原始线段的方向向量
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # 垂直方向向量，旋转 90 度（逆时针）
    perpendicular_direction = (-dy, dx)  # 逆时针旋转 90 度

    # 归一化垂直方向
    norm = np.sqrt(perpendicular_direction[0]**2 + perpendicular_direction[1]**2)
    perpendicular_direction = (perpendicular_direction[0] / norm, perpendicular_direction[1] / norm)

    # 计算垂线的两端
    p1_extended = (p2[0] + perpendicular_direction[0] * length, p2[1] + perpendicular_direction[1] * length)
    p2_extended = (p2[0] - perpendicular_direction[0] * length, p2[1] - perpendicular_direction[1] * length)

    # 返回垂线
    return [p1_extended, p2_extended]

def get_intersection(p1, p2, q1, q2):
    # 使用 Shapely 创建线段
    line1 = LineString([p1, p2])
    line2 = LineString([q1, q2])

    # 计算交点
    intersection = line1.intersection(line2)

    # 如果交点是一个点（而非线或其他几何体），返回交点的坐标
    if intersection.is_empty:
        return None, None
    elif intersection.geom_type == 'Point':
        return intersection.x, intersection.y
    else:
        # 如果交点是线段（相交部分有长度），选择线段的一个点
        return intersection.coords[0]  # 返回第一个交点

# 基础几何函数：归一化向量
def normalize(v):
    v = np.array(v)
    return v / np.linalg.norm(v)

# 延长线段的函数：延长线段两端
def extend_line(p1, p2, length=5.0):
    """
    对给定的线段(p1, p2)进行延长，延长长度为 length（默认为2米）
    返回延长后的两个点
    """
    p1, p2 = np.array(p1), np.array(p2)

    # 计算方向向量
    direction = normalize(p2 - p1)

    # 延长前端
    p1_extended = p1 - direction * length
    p2_extended = p2 + direction * length

    return p1_extended, p2_extended

frame_idx = "180"

# 加载墙体数据
wall_path = f"extracted_lidar_data_code/wall_batch/laserscan_000{frame_idx}_walls.json"
with open(wall_path, "r") as f:
    wall_info = json.load(f)
raw_walls = wall_info['walls']

walls = []
for raw_wall in raw_walls:
    if raw_wall["p1"][0] < -2 and raw_wall["p2"][0] < -2:
        continue
    walls.append(raw_wall)

# 加载中心线数据
centerlines_path = f"extracted_lidar_data_code/road_network/laserscan_000{frame_idx}_centerlines.json"
with open(centerlines_path, "r") as f:
    centerlines_info = json.load(f)

centerlines = centerlines_info["centerlines"]
self_centerline = centerlines[0]

# 获取相关墙体
self_wall_ids = self_centerline["wall_pair"]

extended_walls = []
self_wall_extended = []
intersection_coords = []

# 扩展墙体并记录交点
for wall in walls:
    p1 = wall['p1']
    p2 = wall['p2']
    extend_p1, extend_p2 = extend_line(p1, p2)
    if wall["id"] in self_wall_ids:
        # intersection_coords.append(p2)
        self_wall_extended.append(
            [extend_p1, extend_p2]
        )
    else:
        extended_walls.append(
            [extend_p1, extend_p2]
        )

perpendicular_centerline = get_perpendicular_line_from_p2(self_centerline['p1'], self_centerline['p2'])
for line in self_wall_extended:
    intersection_x, intersection_y = get_intersection(line[0], line[1], perpendicular_centerline[0], perpendicular_centerline[1])
    intersection_coords.append([intersection_x, intersection_y])


# 计算延长线与其他墙体的交点
for line in self_wall_extended:
    for wall in extended_walls:
        intersection_x, intersection_y = get_intersection(line[0], line[1], wall[0], wall[1])
        if intersection_x:
            intersection_coords.append(
                [intersection_x, intersection_y]
            )

# 打印交点坐标
print("交点坐标:", intersection_coords)

# 创建多边形（四边形）
polygon = None
if len(intersection_coords) >= 3:
    # 使用 Shapely 的 MultiPoint 来计算凸包
    points = [tuple(coord) for coord in intersection_coords]  # 转换为元组形式
    multipoint = MultiPoint(points)  # 创建 MultiPoint 对象
    polygon = multipoint.convex_hull  # 计算凸包
else:
    polygon = None  # 如果交点不足以形成多边形，设置为 None

# 打印凸包多边形的坐标
if polygon:
    print("生成的凸包多边形坐标:", list(polygon.exterior.coords))

# 获取多边形的质心
if polygon:
    centroid = polygon.centroid
    centroid_coords = (centroid.x, centroid.y)
else:
    centroid_coords = None

# 创建并排的子图
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 左侧图：原始的 centerlines 和 walls
ax1 = axes[0]
# 绘制墙体 (Walls)
for wall in walls:
    p1 = wall["p1"]
    p2 = wall["p2"]
    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', label=f'Wall {wall["id"]}' if wall["id"] == 0 else "")

# 绘制中心线 (Centerline)
center_p1 = self_centerline["p1"]
center_p2 = self_centerline["p2"]
ax1.plot([center_p1[0], center_p2[0]], [center_p1[1], center_p2[1]], 'k-', label="Centerline")

ax1.set_title("Original Walls and Centerlines")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.legend()
ax1.grid(True)

# 右侧图：原始的 centerlines 和 walls + polygon 区域
ax2 = axes[1]
# 绘制墙体 (Walls)
for wall in walls:
    p1 = wall["p1"]
    p2 = wall["p2"]
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', label=f'Wall {wall["id"]}' if wall["id"] == 0 else "")

# 绘制延长后的墙体 (Extended Walls)
for extended_wall in extended_walls:
    p1, p2 = extended_wall
    ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', label="Extended Wall")

# 绘制交点 (Intersection Points)
for intersection in intersection_coords:
    ax2.plot(intersection[0], intersection[1], 'go', label="Intersection" if intersection_coords.index(intersection) == 0 else "")

# 绘制中心线 (Centerline)
ax2.plot([center_p1[0], center_p2[0]], [center_p1[1], center_p2[1]], 'k-', label="Centerline")

# 绘制多边形区域 (Polygon)
if polygon:
    x, y = polygon.exterior.xy
    ax2.fill(x, y, alpha=0.3, color='orange', label="Polygon Area")

# 绘制多边形的质心
if centroid_coords:
    ax2.plot(centroid_coords[0], centroid_coords[1], 'ro', label="Centroid")

ax2.set_title("With Polygon Area")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.legend()
ax2.grid(True)

# 保存图像
plt.tight_layout()
plt.savefig(f"{frame_idx}_comparison_with_centroid.png")
