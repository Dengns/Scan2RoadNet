import json
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, Point


output_image_path = "intersection.png"

###############################################################
# 输入数据
###############################################################

# walls = {
#     0: {"p1":[-3.6222,-1.0316], "p2":[2.4841,-1.2340]},
#     1: {"p1":[-3.3698,-3.7267], "p2":[-3.4073,-4.9560]},
#     2: {"p1":[4.3624,1.8079],   "p2":[4.2042,-2.0796]},
#     3: {"p1":[-2.5641,1.3324],  "p2":[2.5678,1.1046]}
# }

# # 已知道路中心线
# centerline = {
#     "p1": [-2.6038226478394515, 0.13418892963311857],
#     "p2": [ 2.522850290936412, -0.06455884136045931]
# }


# load walls
walls_path = "extracted_lidar_data_code/wall/laserscan_000136_walls.json"
with open(walls_path, "r") as f:
    walls_info = json.load(f)

raw_walls = walls_info["walls"]
walls = {}
for wall in raw_walls:
    walls[wall["id"]] = {
        "p1": wall["p1"],
        "p2": wall["p2"]
    }

# load centerline
centerline_path = "extracted_lidar_data_code/road_network/laserscan_000136_centerlines.json"
with open(centerline_path, "r") as f:
    centerlines_info = json.load(f)

centerlines = centerlines_info["centerlines"]


# Todo: need to fine the current centerline
centerline = centerlines[0]


###############################################################
# 1. 道路方向向量
###############################################################
p1 = np.array(centerline["p1"])
p2 = np.array(centerline["p2"])
road_dir = p2 - p1
road_dir = road_dir / np.linalg.norm(road_dir)   # 单位化


###############################################################
# 2. 分类水平墙（道路边界）与竖直墙（前向边界）
###############################################################
def wall_angle(w):
    p1 = np.array(w["p1"])
    p2 = np.array(w["p2"])
    d = p2 - p1
    return np.degrees(np.arctan2(d[1], d[0]))

horizontal_walls = []
vertical_walls = []

for wid, w in walls.items():
    ang = wall_angle(w)
    if abs(ang) < 20 or abs(abs(ang)-180) < 20:
        horizontal_walls.append(w)
    elif abs(abs(ang)-90) < 20:
        vertical_walls.append(w)

# 根据 y 判断上下边界
horizontal_sorted = sorted(horizontal_walls, key=lambda w: (w["p1"][1] + w["p2"][1]) / 2)

W_bottom = horizontal_sorted[0]
W_top = horizontal_sorted[-1]


###############################################################
# 3. 找前方竖直墙：投影最大（即沿道路方向最靠前）
###############################################################
def projection_to_road(w):
    p1 = np.array(w["p1"])
    p2 = np.array(w["p2"])
    return max(np.dot(p1, road_dir), np.dot(p2, road_dir))

W_front = max(vertical_walls, key=lambda w: projection_to_road(w))


###############################################################
# 4. 取水平墙的“最前端点”（沿道路方向）
###############################################################
def furthest_point_along_road(w):
    pts = [np.array(w["p1"]), np.array(w["p2"])]
    return max(pts, key=lambda p: np.dot(p, road_dir))

P_top = furthest_point_along_road(W_top)
P_bottom = furthest_point_along_road(W_bottom)


###############################################################
# 5. 求与前方墙 W_front 的交点（严格几何）
###############################################################
front_line = LineString([W_front["p1"], W_front["p2"]])

def intersect_horizontal_to_front(pt):
    """从 pt 沿道路方向画一条射线到前方墙"""
    ray_end = pt + road_dir * 100  # 向前延伸足够长
    ray = LineString([tuple(pt), tuple(ray_end)])
    inter = ray.intersection(front_line)

    if inter.is_empty:
        return None

    if isinstance(inter, Point):
        return np.array([inter.x, inter.y])
    else:
        return None

P_top_R = intersect_horizontal_to_front(P_top)
P_bottom_R = intersect_horizontal_to_front(P_bottom)


###############################################################
# 6. 构成四边形
###############################################################
poly = Polygon([tuple(P_top), tuple(P_top_R), tuple(P_bottom_R), tuple(P_bottom)])
print("四边形顶点:")
print(list(poly.exterior.coords))


###############################################################
# 7. 可视化
###############################################################
plt.figure(figsize=(10,8))

# 画墙
for wid, w in walls.items():
    plt.plot([w["p1"][0], w["p2"][0]], [w["p1"][1], w["p2"][1]], 'k-')

# 画四边形
x, y = poly.exterior.xy
plt.fill(x, y, alpha=0.4, color="red")

plt.axis("equal")
plt.grid(True)
plt.savefig(output_image_path)
print(f"intersection save to {output_image_path}")
