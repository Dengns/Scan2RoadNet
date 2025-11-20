#!/usr/bin/env python3
"""
整合脚本：从原始 LiDAR scan 数据到完整的道路网络
包含：原始点云 + 墙体检测 + 道路中心线 + 路口识别
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from shapely.geometry import LineString, MultiPoint, Point
import json
import os
import sys
from functools import partial

# 添加当前目录到Python路径，以便导入其他模块的函数
sys.path.insert(0, os.path.dirname(__file__))


# =========================
# 配置参数
# =========================
# 输入文件路径
INPUT_LASERSCAN_PATH = "extracted_lidar_data_code/extracted_lidar_data/laserscan_json2/laserscan_000093.json"
# 输出目录
OUTPUT_DIR = "extracted_lidar_data_code/all_in_one_results"

# 墙体检测参数
MAX_RANGE = 50.0
X_LIMIT = 10.0
Y_LIMIT = 5.0
JUMP_DIST_THRESH = 0.2
MIN_SEGMENT_POINTS = 8
RDP_EPSILON = 0.03
MIN_SPLIT_POINTS = 3
MIN_RDP_SEGMENT_LENGTH = 0.3  # RDP分割后子段最小长度（米）
WALL_RMSE_THRESH = 0.07
MIN_SEGMENT_LENGTH = 0.5
PARAM_DBSCAN_EPS = 0.3
PARAM_DBSCAN_MIN_SAMPLES = 1
ALPHA_SCALE = 0.30  # θ的缩放系数
BETA_SCALE = 0.3    # rho的动态阈值系数
GAP_THRESH = 1.8
MIN_WALL_LENGTH = 0.8
MIN_SEGMENTS_PER_WALL = 0.8

# 道路中心线参数
MAX_ROAD_WIDTH = 3.0
MIN_ROAD_WIDTH = 0.8
PARALLEL_ANGLE_THRESH = 10
MIN_OVERLAP_LENGTH = 0.5  # 最小重叠长度（米）

# 交叉口识别参数
EXTEND_LENGTH = 4.0


# =========================
# 1. 墙体检测核心函数
# =========================

def laserscan_to_xy(laserscan, max_range=None):
    """将 LaserScan 数据转换为笛卡尔坐标"""
    angle_min = laserscan['angle_min']
    angle_increment = laserscan['angle_increment']
    ranges = np.asarray(laserscan['ranges'], dtype=np.float64)

    # 有效性与量程过滤
    valid = np.isfinite(ranges) & (ranges > 0.0)
    if max_range is not None:
        valid &= (ranges <= max_range)

    if not np.any(valid):
        return np.empty((0, 2), dtype=np.float64)

    ranges = ranges[valid]
    thetas = angle_min + np.arange(len(ranges)) * angle_increment

    x = ranges * np.cos(thetas)
    y = ranges * np.sin(thetas)
    return np.column_stack([x, y])


def segment_by_jump(points_xy, jump_thresh, min_points):
    """基于扫描顺序的跳变分段"""
    if len(points_xy) == 0:
        return []

    segments = []
    start_idx = 0

    for i in range(1, len(points_xy)):
        d = np.linalg.norm(points_xy[i] - points_xy[i-1])
        if not np.isfinite(d) or d > jump_thresh:
            seg = points_xy[start_idx:i]
            if len(seg) >= min_points:
                segments.append(seg)
            start_idx = i

    last = points_xy[start_idx:]
    if len(last) >= min_points:
        segments.append(last)

    return segments


def fit_line_pca(points):
    """PCA 直线拟合"""
    if len(points) < 2:
        return None

    mean = np.mean(points, axis=0)
    centered = points - mean

    # PCA
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 主方向（最大特征值对应的特征向量）
    direction = eigvecs[:, 1]

    # 计算 RMSE
    normal = np.array([-direction[1], direction[0]])
    distances = np.abs((centered @ normal))
    rmse = np.sqrt(np.mean(distances**2))

    # 计算端点
    t = centered @ direction
    t_min, t_max = np.min(t), np.max(t)
    p1 = mean + t_min * direction
    p2 = mean + t_max * direction
    length = float(t_max - t_min)

    return {
        'p1': p1,
        'p2': p2,
        'dir': direction,
        'length': length,
        'rmse': rmse,
        'mean': mean
    }


def rdp_recursive(points, eps, start_idx=0, end_idx=None):
    """RDP 算法递归实现"""
    if end_idx is None:
        end_idx = len(points) - 1

    if end_idx - start_idx < 1:
        return [start_idx, end_idx]

    # 计算点到线段的最大距离
    p1 = points[start_idx]
    p2 = points[end_idx]
    vec = p2 - p1
    norm = np.linalg.norm(vec)

    if norm < 1e-12:
        return [start_idx, end_idx]

    # 计算所有中间点到线段的距离
    distances = np.abs(np.cross(points[start_idx+1:end_idx] - p1, vec)) / norm

    if len(distances) == 0:
        return [start_idx, end_idx]

    max_dist = np.max(distances)
    max_idx = start_idx + 1 + np.argmax(distances)

    if max_dist > eps:
        # 递归处理左右两段
        left_indices = rdp_recursive(points, eps, start_idx, max_idx)
        right_indices = rdp_recursive(points, eps, max_idx, end_idx)
        return left_indices[:-1] + right_indices
    else:
        return [start_idx, end_idx]


def split_by_rdp(points, eps, min_points, min_length=0.0):
    """使用 RDP 算法分割点云"""
    if len(points) < min_points:
        return []

    key_indices = rdp_recursive(points, eps)

    segments = []
    for i in range(len(key_indices) - 1):
        seg = points[key_indices[i]:key_indices[i+1]+1]

        # 过滤条件1：点数太少
        if len(seg) < min_points:
            continue

        # 过滤条件2：线段太短（如果指定了min_length）
        if min_length > 0:
            seg_length = np.linalg.norm(seg[-1] - seg[0])
            if seg_length < min_length:
                continue

        segments.append(seg)

    return segments


def angle_diff_undirected(a, b):
    """计算无向直线的角度差，范围 [-π/2, π/2]"""
    d = a - b
    d = (d + 0.5 * np.pi) % np.pi - 0.5 * np.pi
    return d


def seg_metric(u, v, theta_scale=0.3, k_rho=0.3):
    """自定义线段距离度量（带角度wrap和距离自适应）"""
    theta_u, rho_u, r_u = u[0], u[1], u[2]
    theta_v, rho_v, r_v = v[0], v[1], v[2]

    # ----- 角度部分：带 wrap -----
    dtheta = angle_diff_undirected(theta_u, theta_v)
    dtheta_norm = dtheta / theta_scale

    # ----- rho 部分：动态阈值 T(r) = k_rho * r -----
    drho = rho_u - rho_v
    r_bar = 0.5 * (r_u + r_v) + 1e-3
    T_r = k_rho * r_bar
    drho_norm = drho / T_r

    # 综合距离（欧氏距离）
    return np.sqrt(dtheta_norm**2 + drho_norm**2)


def segment_to_hessian(p1, p2):
    """将线段转换为Hessian法线形式 (θ, ρ)（修复版）"""
    # 中点
    m = 0.5 * (p1 + p2)

    # 方向向量
    d = p2 - p1

    # 法向量（逆时针旋转90度）：方向(dx, dy) → 法向(-dy, dx)
    nx = -d[1]
    ny = d[0]
    norm = np.sqrt(nx**2 + ny**2) + 1e-12
    nx /= norm
    ny /= norm

    # 原点到直线的有符号距离
    rho = nx * m[0] + ny * m[1]

    # 法向量的角度
    theta = np.arctan2(ny, nx)

    # 归一化到 [0, π)（因为直线无方向性）
    # 如果 theta < 0，加 π，同时翻转 rho 符号（因为法向量反向了）
    if theta < 0:
        theta += np.pi
        rho = -rho

    return float(theta), float(rho)


def cluster_segments_in_param_space(fitted_segments, eps, min_samples, alpha, beta):
    """在 (θ, ρ, r) 参数空间对线段进行DBSCAN聚类（带角度wrap和距离自适应）"""
    if len(fitted_segments) == 0:
        return np.array([]), np.array([])

    # 预分配数组
    n_segs = len(fitted_segments)
    features = np.zeros((n_segs, 3), dtype=np.float64)  # 现在是3维：[theta, rho, r]

    # 向量化处理所有线段
    for i, seg in enumerate(fitted_segments):
        fit = seg['fit']
        p1 = fit['p1']
        p2 = fit['p2']

        # 转换到 (θ, ρ) 空间
        theta, rho = segment_to_hessian(p1, p2)

        # 计算线段中点到原点的距离
        midpoint = 0.5 * (p1 + p2)
        r = np.linalg.norm(midpoint)

        # 特征：[theta, rho, r]（注意：theta不除以alpha，alpha传递给metric）
        features[i, 0] = theta
        features[i, 1] = rho
        features[i, 2] = r

    # 使用自定义metric进行DBSCAN聚类
    # 将alpha和k_rho参数固化到metric函数中
    metric_fn = partial(seg_metric, theta_scale=alpha, k_rho=beta)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric_fn)
    labels = db.fit_predict(features)

    return features, labels


def merge_segments_in_cluster(cluster_segments, gap_thresh):
    """合并簇内的线段"""
    if len(cluster_segments) == 0:
        return []

    # 计算平均方向
    all_directions = np.array([seg['fit']['dir'] for seg in cluster_segments])
    avg_dir = np.mean(all_directions, axis=0)
    avg_dir = avg_dir / np.linalg.norm(avg_dir)

    # 收集所有点
    all_points = []
    for seg in cluster_segments:
        all_points.append(seg['points'])
    all_points = np.vstack(all_points)

    # 投影到平均方向
    t_vals = all_points @ avg_dir

    # 构建区间
    intervals = []
    for seg in cluster_segments:
        pts = seg['points']
        t = pts @ avg_dir
        intervals.append((np.min(t), np.max(t)))

    # 合并区间
    intervals = sorted(intervals)
    merged = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start - current_end <= gap_thresh:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    # 为每个区间重新拟合
    walls = []
    wall_normal = np.array([-avg_dir[1], avg_dir[0]])

    for t_min, t_max in merged:
        mask = (t_vals >= t_min - 0.01) & (t_vals <= t_max + 0.01)
        interval_points = all_points[mask]

        if len(interval_points) >= 2:
            fit = fit_line_pca(interval_points)
            if fit is not None:
                walls.append({'fit': fit, 'points': interval_points})

    return walls


def detect_walls(xy):
    """墙体检测主流程"""
    # 1. Jump 分段
    segments_points = segment_by_jump(xy, JUMP_DIST_THRESH, MIN_SEGMENT_POINTS)

    # 2. RDP 分割
    split_segments = []
    for seg in segments_points:
        split_segments.extend(split_by_rdp(seg, RDP_EPSILON, MIN_SPLIT_POINTS, MIN_RDP_SEGMENT_LENGTH))

    # 3. PCA 拟合
    fitted_segments = []
    for seg_pts in split_segments:
        fit = fit_line_pca(seg_pts)
        if fit is None:
            continue
        if fit['rmse'] > WALL_RMSE_THRESH:
            continue
        if fit['length'] < MIN_SEGMENT_LENGTH:
            continue
        fitted_segments.append({'fit': fit, 'points': seg_pts})

    # 保存初步拟合的线段供可视化使用
    initial_fitted_segments = fitted_segments.copy()

    # 4. 参数空间聚类
    if len(fitted_segments) == 0:
        return [], {
            'segments_points': segments_points,
            'split_segments': split_segments,
            'initial_fitted_segments': [],
            'features': np.array([]),
            'labels': np.array([])
        }

    features, labels = cluster_segments_in_param_space(
        fitted_segments, PARAM_DBSCAN_EPS, PARAM_DBSCAN_MIN_SAMPLES, ALPHA_SCALE, BETA_SCALE
    )

    # 5. 区间合并
    merged_walls = []
    wall_id = 0
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue

        cluster_mask = (labels == cluster_id)
        cluster_segments = [fitted_segments[i] for i in np.where(cluster_mask)[0]]

        if len(cluster_segments) < MIN_SEGMENTS_PER_WALL:
            continue

        walls = merge_segments_in_cluster(cluster_segments, GAP_THRESH)

        for wall in walls:
            if wall['fit']['length'] >= MIN_WALL_LENGTH:
                wall['id'] = wall_id
                wall_id += 1
                merged_walls.append(wall)

    # 返回墙体和中间结果
    intermediate_data = {
        'segments_points': segments_points,
        'split_segments': split_segments,
        'initial_fitted_segments': initial_fitted_segments,
        'features': features,
        'labels': labels
    }

    return merged_walls, intermediate_data


# =========================
# 2. 道路中心线核心函数
# =========================

def are_parallel(wall1, wall2, angle_thresh_deg):
    """判断两面墙是否平行"""
    dot = np.abs(np.dot(wall1['fit']['dir'], wall2['fit']['dir']))
    angle_diff = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return angle_diff < angle_thresh_deg or angle_diff > (180 - angle_thresh_deg)


def compute_wall_distance(wall1, wall2):
    """计算两面墙之间的距离"""
    normal1 = np.array([-wall1['fit']['dir'][1], wall1['fit']['dir'][0]])
    d1 = np.dot(wall2['fit']['p1'] - wall1['fit']['p1'], normal1)
    d2 = np.dot(wall2['fit']['p2'] - wall1['fit']['p1'], normal1)
    return np.abs(0.5 * (d1 + d2))


def compute_projection_overlap(wall1, wall2):
    """计算两面墙的投影重叠长度（米）"""
    dir_vec = wall1['fit']['dir']

    t1_start = np.dot(wall1['fit']['p1'], dir_vec)
    t1_end = np.dot(wall1['fit']['p2'], dir_vec)
    if t1_start > t1_end:
        t1_start, t1_end = t1_end, t1_start

    t2_start = np.dot(wall2['fit']['p1'], dir_vec)
    t2_end = np.dot(wall2['fit']['p2'], dir_vec)
    if t2_start > t2_end:
        t2_start, t2_end = t2_end, t2_start

    overlap_start = max(t1_start, t2_start)
    overlap_end = min(t1_end, t2_end)
    overlap_length = max(0, overlap_end - overlap_start)

    # 返回绝对重叠长度（米）
    return overlap_length


def find_wall_pairs(walls):
    """找到可以配对的墙体"""
    pairs = []
    used_walls = set()

    for i, wall1 in enumerate(walls):
        if i in used_walls:
            continue

        best_match = None
        best_score = -1

        for j, wall2 in enumerate(walls):
            if i >= j or j in used_walls:
                continue

            if not are_parallel(wall1, wall2, PARALLEL_ANGLE_THRESH):
                continue

            dist = compute_wall_distance(wall1, wall2)
            if dist < MIN_ROAD_WIDTH or dist > MAX_ROAD_WIDTH:
                continue

            # 🔑 改进：检查绝对重叠长度（米），而不是重叠比例
            overlap_length = compute_projection_overlap(wall1, wall2)
            if overlap_length < MIN_OVERLAP_LENGTH:
                continue

            # 计算评分（使用重叠长度参与评分）
            ideal_width = 1.5
            width_score = 1.0 - abs(dist - ideal_width) / MAX_ROAD_WIDTH
            # 重叠长度越长越好，归一化到[0,1]，假设5米为满分
            overlap_score = min(overlap_length / 5.0, 1.0)
            score = overlap_score * 0.7 + width_score * 0.3

            if score > best_score:
                best_score = score
                best_match = (j, wall2, dist)

        if best_match is not None:
            j, wall2, dist = best_match
            pairs.append((wall1, wall2, dist))
            used_walls.add(i)
            used_walls.add(j)

    return pairs


def generate_centerline(wall1, wall2):
    """根据墙体对生成道路中心线"""
    dir_vec = wall1['fit']['dir']

    # 投影端点
    t1 = np.dot(wall1['fit']['p1'], dir_vec)
    t2 = np.dot(wall1['fit']['p2'], dir_vec)
    t3 = np.dot(wall2['fit']['p1'], dir_vec)
    t4 = np.dot(wall2['fit']['p2'], dir_vec)

    # 重叠区间
    t_start = max(min(t1, t2), min(t3, t4))
    t_end = min(max(t1, t2), max(t3, t4))

    # 计算中心线端点
    def get_point_at_t(wall, t):
        t1 = np.dot(wall['fit']['p1'], dir_vec)
        t2 = np.dot(wall['fit']['p2'], dir_vec)
        if abs(t2 - t1) < 1e-6:
            return wall['fit']['p1']
        ratio = (t - t1) / (t2 - t1)
        return wall['fit']['p1'] + ratio * (wall['fit']['p2'] - wall['fit']['p1'])

    start1 = get_point_at_t(wall1, t_start)
    start2 = get_point_at_t(wall2, t_start)
    centerline_start = 0.5 * (start1 + start2)

    end1 = get_point_at_t(wall1, t_end)
    end2 = get_point_at_t(wall2, t_end)
    centerline_end = 0.5 * (end1 + end2)

    width = compute_wall_distance(wall1, wall2)

    # 🔑 确保方向：p1（起点）在后，p2（终点）在前
    # 使用朝向判断：中心线方向应该指向前方（X轴正方向）
    centerline_dir = centerline_end - centerline_start

    # 如果方向向量的X分量为负（朝后），则交换起点和终点
    if centerline_dir[0] < 0:
        centerline_start, centerline_end = centerline_end, centerline_start

    return {
        'p1': centerline_start,
        'p2': centerline_end,
        'width': width,
        'wall_pair': (wall1, wall2)
    }


def select_main_road(centerlines, robot_pos=np.array([0.0, 0.0])):
    """选择机器人所在的主干道（距离机器人最近的中心线）"""
    if len(centerlines) == 0:
        return None

    min_dist = float('inf')
    main_centerline = None

    for cl in centerlines:
        # 计算机器人到中心线的最近距离（点到线段的距离）
        p1 = cl['p1']
        p2 = cl['p2']

        # 线段方向向量
        line_vec = p2 - p1
        line_length_sq = np.dot(line_vec, line_vec)

        if line_length_sq < 1e-12:
            # 退化为点的情况
            dist = np.linalg.norm(robot_pos - p1)
        else:
            # 计算投影参数 t
            robot_vec = robot_pos - p1
            t = np.dot(robot_vec, line_vec) / line_length_sq
            t = np.clip(t, 0.0, 1.0)  # 限制在线段范围内

            # 最近点
            closest_point = p1 + t * line_vec
            dist = np.linalg.norm(robot_pos - closest_point)

        if dist < min_dist:
            min_dist = dist
            main_centerline = cl

    return main_centerline


# =========================
# 3. 交叉口识别核心函数
# =========================

def normalize(v):
    """归一化向量"""
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


def extend_line(p1, p2, length):
    """延长线段"""
    p1, p2 = np.array(p1), np.array(p2)
    direction = normalize(p2 - p1)
    p1_extended = p1 - direction * length
    p2_extended = p2 + direction * length
    return p1_extended, p2_extended


def get_perpendicular_line(p1, p2, length):
    """构建垂线"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    perp_dir = (-dy, dx)
    norm = np.sqrt(perp_dir[0]**2 + perp_dir[1]**2)
    if norm < 1e-6:
        return None

    perp_dir = (perp_dir[0] / norm, perp_dir[1] / norm)

    p1_ext = (p2[0] + perp_dir[0] * length, p2[1] + perp_dir[1] * length)
    p2_ext = (p2[0] - perp_dir[0] * length, p2[1] - perp_dir[1] * length)

    return [p1_ext, p2_ext]


def get_intersection(p1, p2, q1, q2):
    """计算两条线段的交点"""
    line1 = LineString([p1, p2])
    line2 = LineString([q1, q2])

    intersection = line1.intersection(line2)

    if intersection.is_empty:
        return None
    elif intersection.geom_type == 'Point':
        return np.array([intersection.x, intersection.y])
    else:
        return None


def detect_intersection(walls, centerline):
    """检测交叉口区域"""
    if centerline is None:
        return None, None, []

    # 获取道路边界墙体
    wall1, wall2 = centerline['wall_pair']
    self_wall_ids = [wall1['id'], wall2['id']]
    self_walls = [wall1, wall2]
    other_walls = [w for w in walls if w['id'] not in self_wall_ids]

    # 延长墙体
    self_wall_extended = []
    for wall in self_walls:
        p1_ext, p2_ext = extend_line(wall['fit']['p1'], wall['fit']['p2'], EXTEND_LENGTH)
        self_wall_extended.append([p1_ext, p2_ext])

    other_wall_extended = []
    for wall in other_walls:
        p1_ext, p2_ext = extend_line(wall['fit']['p1'], wall['fit']['p2'], EXTEND_LENGTH)
        other_wall_extended.append([p1_ext, p2_ext])

    # 构建垂线（在中心线终点p2，即前方）
    perp_line = get_perpendicular_line(centerline['p1'], centerline['p2'], EXTEND_LENGTH)

    intersection_coords = []

    # 🔑 改进的交点收集策略
    # 1. 道路边界墙与垂线的交点（垂线在前方，这些点构成路口的后边界）
    if perp_line is not None:
        for line in self_wall_extended:
            inter = get_intersection(line[0], line[1], perp_line[0], perp_line[1])
            if inter is not None:
                intersection_coords.append(inter)

    # 2. 道路边界墙与其他墙的交点
    # 只保留在道路前方的交点
    road_direction = centerline['p2'] - centerline['p1']  # 道路方向向量

    for self_line in self_wall_extended:
        for other_line in other_wall_extended:
            inter = get_intersection(self_line[0], self_line[1], other_line[0], other_line[1])
            if inter is not None:
                # 🔑 筛选条件：交点在道路前方
                # 计算交点相对于中心线终点的方向
                to_intersection = inter - centerline['p2']
                # 如果与道路方向同向（点积>0），说明在前方
                if np.dot(to_intersection, road_direction) > 0:
                    intersection_coords.append(inter)

    # 计算凸包
    polygon = None
    centroid = None

    if len(intersection_coords) >= 3:
        points = [tuple(coord) for coord in intersection_coords]
        multipoint = MultiPoint(points)
        polygon = multipoint.convex_hull

        if polygon and not polygon.is_empty:
            centroid = np.array([polygon.centroid.x, polygon.centroid.y])

    # 🔑 返回额外的调试信息：延长线和垂线
    debug_info = {
        'self_wall_extended': self_wall_extended,
        'other_wall_extended': other_wall_extended,
        'perp_line': perp_line,
        'intersection_coords': intersection_coords
    }

    return polygon, centroid, debug_info


def detect_successor_lanes(walls, centerline, polygon, centroid, intersection_coords):
    """检测路口的后继车道（4个方向：前、后、左、右）

    Args:
        walls: 所有墙体列表
        centerline: 主干道中心线
        polygon: 路口多边形
        centroid: 路口质心
        intersection_coords: 路口交点坐标列表

    Returns:
        successor_lanes: 过滤后的后继车道列表
        rectangle_info: 矩形信息（用于可视化）
    """
    if centerline is None or polygon is None or centroid is None or len(intersection_coords) < 3:
        return [], None

    # 1. 计算中心线方向（自车方向）
    road_dir = normalize(centerline['p2'] - centerline['p1'])
    road_angle = np.arctan2(road_dir[1], road_dir[0])

    # 2. 构建旋转矩阵（将中心线方向旋转到X轴）
    cos_a = np.cos(-road_angle)
    sin_a = np.sin(-road_angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # 3. 旋转交点坐标到aligned坐标系
    centered_coords = np.array(intersection_coords) - centroid
    rotated_coords = centered_coords @ R.T

    # 4. 计算AABB（轴对齐边界框 = 最小矩形）
    x_min, y_min = rotated_coords.min(axis=0)
    x_max, y_max = rotated_coords.max(axis=0)

    # 5. 矩形的3条边的中点（在rotated坐标系）
    # 只保留3个后继方向：前（直行）、左、右
    # backward不需要，因为main_centerline已经代表当前车道
    edge_centers_rotated = {
        'forward': np.array([x_max, (y_min + y_max) / 2]),
        'left': np.array([(x_min + x_max) / 2, y_max]),
        'right': np.array([(x_min + x_max) / 2, y_min])
    }

    # 6. 旋转回世界坐标系
    R_inv = R.T
    edge_centers_world = {}
    for direction, pt_rot in edge_centers_rotated.items():
        pt_world = pt_rot @ R_inv.T + centroid
        edge_centers_world[direction] = pt_world

    # 7. 生成3个后继方向的车道（从路口质心向外延伸3米）
    SUCCESSOR_LENGTH = 3.0
    successor_lanes = []

    for direction, edge_center in edge_centers_world.items():
        lane_dir = normalize(edge_center - centroid)
        p1 = centroid
        p2 = centroid + lane_dir * SUCCESSOR_LENGTH

        successor_lanes.append({
            'direction': direction,
            'p1': p1,
            'p2': p2,
            'lane_dir': lane_dir,
            'edge_center': edge_center
        })

    # 8. 墙体遮挡检测
    filtered_lanes = []
    for lane in successor_lanes:
        if not is_blocked_by_wall(lane, walls, centerline):
            filtered_lanes.append(lane)

    # 返回矩形信息和过滤后的车道
    # 将矩形的4个角点转回世界坐标（用于可视化）
    rect_corners_rotated = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])
    rect_corners_world = rect_corners_rotated @ R_inv.T + centroid

    rectangle_info = {
        'centroid': centroid,
        'road_angle': road_angle,
        'corners': rect_corners_world,
        'R': R
    }

    return filtered_lanes, rectangle_info


def is_blocked_by_wall(lane, walls, main_centerline):
    """检测车道是否被墙体遮挡

    策略：直接检查后继车道线段是否与墙体距离很近（说明被墙堵住）
    不区分方向，所有4个方向都用同样的逻辑检查
    """
    BLOCKING_DISTANCE = 0.5  # 如果车道与墙体距离小于0.5米，认为被阻挡

    # 获取主干道墙体ID（排除它们，因为它们是自车所在道路的边界）
    road_wall_ids = set()
    if main_centerline is not None:
        wall1, wall2 = main_centerline['wall_pair']
        road_wall_ids = {wall1['id'], wall2['id']}

    # 创建车道线段
    lane_line = LineString([lane['p1'], lane['p2']])

    # 检查所有其他墙体
    for wall in walls:
        # 跳过主干道的墙体（它们是自车所在道路的边界）
        if wall['id'] in road_wall_ids:
            continue

        # 创建墙体线段
        wall_line = LineString([wall['fit']['p1'], wall['fit']['p2']])

        # 计算车道线段到墙体的最短距离
        dist = lane_line.distance(wall_line)

        # 如果距离很近，说明这个方向被墙堵住了
        if dist < BLOCKING_DISTANCE:
            return True

    return False


# =========================
# 4. 完整处理流程
# =========================

def process_full_pipeline(laserscan_path):
    """完整处理流程：点云 → 墙体 → 中心线 → 路口"""
    print("=" * 70)
    print(" 道路网络完整处理流程")
    print("=" * 70)

    # 1. 加载点云数据
    print("\n【步骤1】加载点云数据")
    with open(laserscan_path, "r") as f:
        data = json.load(f)
    laserscan = data["laserscan"] if "laserscan" in data else data

    xy = laserscan_to_xy(laserscan, max_range=MAX_RANGE)

    # 空间过滤
    mask = (xy[:,0] >= -X_LIMIT) & (xy[:,0] <= X_LIMIT) & \
           (xy[:,1] >= -Y_LIMIT) & (xy[:,1] <= Y_LIMIT)
    xy = xy[mask]
    xy = xy[np.isfinite(xy).all(axis=1)]

    print(f"  点云数量: {len(xy)}")

    # 2. 墙体检测
    print("\n【步骤2】墙体检测")
    walls, intermediate_data = detect_walls(xy)
    print(f"  检测到墙体数: {len(walls)}")

    # 3. 道路中心线生成
    print("\n【步骤3】道路中心线生成")
    wall_pairs = find_wall_pairs(walls)
    print(f"  找到墙体对: {len(wall_pairs)}")

    centerlines = []
    for wall1, wall2, dist in wall_pairs:
        cl = generate_centerline(wall1, wall2)
        centerlines.append(cl)
        print(f"  中心线 #{len(centerlines)}: 宽度={cl['width']:.2f}m")

    # 🔑 选择主干道（距离机器人最近的中心线）
    main_centerline = select_main_road(centerlines, robot_pos=np.array([0.0, 0.0]))
    if main_centerline is not None:
        main_idx = centerlines.index(main_centerline)
        print(f"  ✅ 选择主干道: 中心线 #{main_idx + 1} (距离机器人最近)")
    else:
        print(f"  ⚠️  未找到主干道")

    # 4. 交叉口识别（只为主干道检测）
    print("\n【步骤4】交叉口识别")
    polygons = []
    centroids = []
    all_debug_info = []

    if main_centerline is not None:
        poly, cent, debug_info = detect_intersection(walls, main_centerline)
        if poly is not None:
            polygons.append(poly)
            centroids.append(cent)
            all_debug_info.append(debug_info)
            print(f"  ✅ 主干道检测到交叉口")
        else:
            print(f"  ⚠️  主干道未检测到交叉口")
    else:
        print(f"  ⚠️  无主干道，跳过交叉口识别")

    # 5. 后继车道检测（基于路口矩形）
    print("\n【步骤5】后继车道检测")
    successor_lanes = []
    rectangle_info = None

    if main_centerline is not None and len(polygons) > 0 and len(all_debug_info) > 0:
        poly = polygons[0]
        cent = centroids[0]
        debug_info = all_debug_info[0]
        intersection_coords = debug_info['intersection_coords']

        successor_lanes, rectangle_info = detect_successor_lanes(
            walls, main_centerline, poly, cent, intersection_coords
        )

        if len(successor_lanes) > 0:
            print(f"  ✅ 检测到 {len(successor_lanes)} 条后继车道:")
            for lane in successor_lanes:
                print(f"     - {lane['direction']}")
        else:
            print(f"  ⚠️  未检测到后继车道")
    else:
        print(f"  ⚠️  无路口信息，跳过后继车道检测")

    return xy, walls, centerlines, main_centerline, polygons, centroids, all_debug_info, intermediate_data, successor_lanes, rectangle_info


def visualize_all(xy, walls, centerlines, main_centerline, polygons, centroids, all_debug_info, intermediate_data, successor_lanes, rectangle_info, output_path):
    """可视化所有结果 - 5x2 完整流程图（新增图9：后继车道）"""
    fig, axes = plt.subplots(5, 2, figsize=(20, 35))

    # 提取中间数据
    segments_points = intermediate_data['segments_points']
    split_segments = intermediate_data['split_segments']
    initial_fitted_segments = intermediate_data['initial_fitted_segments']
    features = intermediate_data['features']
    labels = intermediate_data['labels']

    # ========== 第1行左：原始点云 ==========
    ax1 = axes[0, 0]
    if len(xy) > 0:
        ax1.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=3, alpha=0.6)
    ax1.set_title(f"1. Raw Points (N={len(xy)})", fontsize=12, fontweight='bold')
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, linestyle=':', alpha=0.4)

    # ========== 第1行右：Jump-distance 分段 ==========
    ax2 = axes[0, 1]
    colors_jump = plt.cm.tab20(np.linspace(0, 1, max(1, len(segments_points))))
    for k, seg in enumerate(segments_points):
        col = colors_jump[k % len(colors_jump)]
        ax2.plot(seg[:, 0], seg[:, 1], '.', color=col, markersize=3, alpha=0.9)
    ax2.set_title(f"2. Jump-distance Segments (N={len(segments_points)})", fontsize=12, fontweight='bold')
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_xlim(-10, 10)
    ax2.set_ylim(-10, 10)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle=':', alpha=0.4)

    # ========== 第2行左：RDP 分割 ==========
    ax3 = axes[1, 0]
    colors_split = plt.cm.tab20b(np.linspace(0, 1, max(1, len(split_segments))))
    for k, seg in enumerate(split_segments):
        col = colors_split[k % len(colors_split)]
        ax3.plot(seg[:, 0], seg[:, 1], '.', color=col, markersize=3, alpha=0.9)
    ax3.set_title(f"3. RDP Segments (N={len(split_segments)})", fontsize=12, fontweight='bold')
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_xlim(-10, 10)
    ax3.set_ylim(-10, 10)
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True, linestyle=':', alpha=0.4)

    # ========== 第2行右：参数空间聚类 ==========
    ax4 = axes[1, 1]
    if len(features) > 0:
        noise_mask = (labels == -1)
        cluster_mask = (labels != -1)

        if np.any(noise_mask):
            ax4.scatter(features[noise_mask, 0], features[noise_mask, 1],
                       c='gray', s=80, alpha=0.6, marker='x', linewidths=2,
                       label=f'Noise ({np.sum(noise_mask)})')

        if np.any(cluster_mask):
            unique_labels = sorted(set(labels[cluster_mask]))
            for label in unique_labels:
                mask = (labels == label)
                n_segs = np.sum(mask)
                color = plt.cm.tab10(label % 10)
                ax4.scatter(features[mask, 0], features[mask, 1],
                           c=[color], s=100, alpha=0.85,
                           edgecolors='black', linewidths=1.5,
                           label=f'Cluster {label} ({n_segs})')

        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = np.sum(labels == -1)
        ax4.set_title(f"4. Param-space DBSCAN (C={num_clusters}, N={num_noise})", fontsize=12, fontweight='bold')
        ax4.set_xlabel(f"θ / {ALPHA_SCALE:.2f}")
        ax4.set_ylabel("ρ (m)")
        ax4.legend(fontsize=8, loc='best', framealpha=0.9)
    else:
        ax4.text(0.5, 0.5, 'No segments', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14, color='red')
        ax4.set_title("4. Param-space DBSCAN", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.5)

    # ========== 第3行左：拟合线段（聚类着色）==========
    ax5 = axes[2, 0]
    if len(xy) > 0:
        ax5.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=2, alpha=0.3)

    if len(initial_fitted_segments) > 0 and len(labels) > 0:
        for i, seg_dict in enumerate(initial_fitted_segments):
            fit = seg_dict['fit']
            p1, p2 = fit['p1'], fit['p2']

            if i < len(labels):
                if labels[i] == -1:
                    color = 'gray'
                    lw = 1.5
                    alpha = 0.5
                else:
                    cluster_id = labels[i]
                    color = plt.cm.tab10(cluster_id % 10)
                    lw = 2.5
                    alpha = 0.85

                ax5.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color=color, linewidth=lw, alpha=alpha)

    ax5.set_title(f"5. Fitted Segments by Cluster (N={len(initial_fitted_segments)})", fontsize=12, fontweight='bold')
    ax5.set_xlabel("X (m)")
    ax5.set_ylabel("Y (m)")
    ax5.set_xlim(-10, 10)
    ax5.set_ylim(-10, 10)
    ax5.set_aspect('equal', adjustable='box')
    ax5.grid(True, linestyle=':', alpha=0.4)

    # ========== 第3行右：最终墙体 ==========
    ax6 = axes[2, 1]
    if len(xy) > 0:
        ax6.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=2, alpha=0.3)

    for wall in walls:
        p1, p2 = wall['fit']['p1'], wall['fit']['p2']
        ax6.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color='tab:red', linewidth=3.5, alpha=0.95)

    ax6.set_title(f"6. Final Walls (N={len(walls)})", fontsize=12, fontweight='bold')
    ax6.set_xlabel("X (m)")
    ax6.set_ylabel("Y (m)")
    ax6.set_xlim(-10, 10)
    ax6.set_ylim(-10, 10)
    ax6.set_aspect('equal', adjustable='box')
    ax6.grid(True, linestyle=':', alpha=0.4)

    # ========== 第4行左：墙体 + 中心线 ==========
    ax7 = axes[3, 0]
    if len(xy) > 0:
        ax7.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=1, alpha=0.3)

    # 🔑 获取主干道边界墙的ID（用于区分左右墙）
    road_wall_ids = set()
    if main_centerline is not None:
        wall1, wall2 = main_centerline['wall_pair']
        road_wall_ids = {wall1['id'], wall2['id']}

    # 绘制墙体（区分左右墙）
    for i, wall in enumerate(walls):
        fit = wall['fit']

        # 🎨 根据墙体类型选择颜色
        if wall['id'] in road_wall_ids:
            # 主干道边界墙：用蓝色和绿色区分
            if main_centerline is not None:
                wall1, wall2 = main_centerline['wall_pair']
                if wall['id'] == wall1['id']:
                    color, label = 'b', 'Left Wall (Main)'
                else:
                    color, label = 'g', 'Right Wall (Main)'
            else:
                color, label = 'b', 'Road Wall'
        else:
            # 其他墙：灰色
            color, label = 'gray', 'Other Walls' if i == len(walls)-1 else ''

        ax7.plot([fit['p1'][0], fit['p2'][0]],
                [fit['p1'][1], fit['p2'][1]],
                color=color, linewidth=3, alpha=0.8, label=label if label else '')

    # 绘制中心线（区分主干道和其他道路）
    for i, cl in enumerate(centerlines):
        if main_centerline is not None and cl is main_centerline:
            # 主干道：红色粗虚线
            ax7.plot([cl['p1'][0], cl['p2'][0]],
                    [cl['p1'][1], cl['p2'][1]],
                    'r--', linewidth=4, alpha=0.9, label='Main Road Centerline')
        else:
            # 其他道路：灰色细虚线
            ax7.plot([cl['p1'][0], cl['p2'][0]],
                    [cl['p1'][1], cl['p2'][1]],
                    color='gray', linestyle=':', linewidth=2, alpha=0.5,
                    label='Other Centerlines' if i == 0 and main_centerline is not None else '')

    ax7.plot(0, 0, 'k^', markersize=10, markeredgewidth=2, label='Robot')
    ax7.set_title(f"7. Walls + Centerlines (CL={len(centerlines)})", fontsize=12, fontweight='bold')
    ax7.set_xlabel("X (m)")
    ax7.set_ylabel("Y (m)")
    ax7.set_xlim(-10, 10)
    ax7.set_ylim(-10, 10)
    ax7.set_aspect('equal', adjustable='box')
    ax7.grid(True, linestyle=':', alpha=0.4)
    ax7.legend(fontsize=9, loc='best', framealpha=0.9)

    # ========== 第4行右：完整道路网络（墙体+中心线+交叉口+延长线+交点）==========
    ax8 = axes[3, 1]
    if len(xy) > 0:
        ax8.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=1, alpha=0.3)

    # 🔑 绘制延长线和垂线（如果有调试信息）
    if len(all_debug_info) > 0:
        debug_info = all_debug_info[0]  # 取第一个路口的调试信息

        # 绘制道路边界墙的延长线（蓝色和绿色虚线）
        for i, line in enumerate(debug_info['self_wall_extended']):
            color = 'b' if i == 0 else 'g'
            ax8.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                    color=color, linestyle=':', linewidth=1.5, alpha=0.5,
                    label='Extended Road Walls' if i == 0 else '')

        # 绘制其他墙的延长线（灰色虚线）
        for i, line in enumerate(debug_info['other_wall_extended']):
            ax8.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                    color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                    label='Extended Other Walls' if i == 0 else '')

        # 绘制垂线（紫色虚线）
        if debug_info['perp_line'] is not None:
            perp = debug_info['perp_line']
            ax8.plot([perp[0][0], perp[1][0]], [perp[0][1], perp[1][1]],
                    color='purple', linestyle='--', linewidth=2, alpha=0.7,
                    label='Perpendicular Line')

        # 绘制交点（黄色圆点）
        for coord in debug_info['intersection_coords']:
            ax8.plot(coord[0], coord[1], 'yo', markersize=8, markeredgecolor='black',
                    markeredgewidth=1, label='Intersection Points' if coord is debug_info['intersection_coords'][0] else '')

    # 绘制墙体（区分左右墙）
    for i, wall in enumerate(walls):
        fit = wall['fit']

        # 🎨 根据墙体类型选择颜色（与图7一致）
        if wall['id'] in road_wall_ids:
            if main_centerline is not None:
                wall1, wall2 = main_centerline['wall_pair']
                if wall['id'] == wall1['id']:
                    color, label = 'b', 'Left Wall (Main)'
                else:
                    color, label = 'g', 'Right Wall (Main)'
            else:
                color, label = 'b', 'Road Wall'
        else:
            color, label = 'gray', 'Other Walls' if i == len(walls)-1 else ''

        ax8.plot([fit['p1'][0], fit['p2'][0]],
                [fit['p1'][1], fit['p2'][1]],
                color=color, linewidth=3, alpha=0.8, label=label if label else '')

    # 绘制中心线（区分主干道和其他道路）
    for i, cl in enumerate(centerlines):
        if main_centerline is not None and cl is main_centerline:
            # 主干道：红色粗虚线
            ax8.plot([cl['p1'][0], cl['p2'][0]],
                    [cl['p1'][1], cl['p2'][1]],
                    'r--', linewidth=4, alpha=0.9, label='Main Road Centerline')
        else:
            # 其他道路：灰色细虚线
            ax8.plot([cl['p1'][0], cl['p2'][0]],
                    [cl['p1'][1], cl['p2'][1]],
                    color='gray', linestyle=':', linewidth=2, alpha=0.5,
                    label='Other Centerlines' if i == 0 and main_centerline is not None else '')

    # 绘制交叉口
    for i, poly in enumerate(polygons):
        if poly and not poly.is_empty:
            x, y = poly.exterior.xy
            ax8.fill(x, y, alpha=0.3, color='orange', label='Intersection' if i == 0 else '')
            ax8.plot(x, y, 'r-', linewidth=2, alpha=0.7)

    # 绘制质心
    for i, cent in enumerate(centroids):
        if cent is not None:
            ax8.plot(cent[0], cent[1], 'ro', markersize=10, label='Centroid' if i == 0 else '')

    ax8.plot(0, 0, 'k^', markersize=10, markeredgewidth=2, label='Robot')
    ax8.set_title(f"8. Complete Network + Debug (INT={len(polygons)})", fontsize=12, fontweight='bold')
    ax8.set_xlabel("X (m)")
    ax8.set_ylabel("Y (m)")
    ax8.set_xlim(-10, 10)
    ax8.set_ylim(-10, 10)
    ax8.set_aspect('equal', adjustable='box')
    ax8.grid(True, linestyle=':', alpha=0.4)
    ax8.legend(fontsize=8, loc='best', framealpha=0.9, ncol=2)

    # ========== 第5行左：路口矩形 + 后继车道 ==========
    ax9 = axes[4, 0]
    if len(xy) > 0:
        ax9.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=1, alpha=0.2)

    # 绘制墙体
    for i, wall in enumerate(walls):
        fit = wall['fit']
        if wall['id'] in road_wall_ids:
            if main_centerline is not None:
                wall1, wall2 = main_centerline['wall_pair']
                if wall['id'] == wall1['id']:
                    color = 'b'
                else:
                    color = 'g'
            else:
                color = 'b'
        else:
            color = 'gray'
        ax9.plot([fit['p1'][0], fit['p2'][0]],
                [fit['p1'][1], fit['p2'][1]],
                color=color, linewidth=2.5, alpha=0.7)

    # 绘制主干道中心线
    if main_centerline is not None:
        ax9.plot([main_centerline['p1'][0], main_centerline['p2'][0]],
                [main_centerline['p1'][1], main_centerline['p2'][1]],
                'r--', linewidth=3, alpha=0.9, label='Main Centerline')

    # 绘制路口矩形
    if rectangle_info is not None:
        corners = rectangle_info['corners']
        # 闭合矩形（添加第一个点到末尾）
        rect_x = np.append(corners[:, 0], corners[0, 0])
        rect_y = np.append(corners[:, 1], corners[0, 1])
        ax9.plot(rect_x, rect_y, 'purple', linestyle='--', linewidth=2.5, alpha=0.8, label='Junction Rectangle')
        ax9.fill(rect_x, rect_y, alpha=0.15, color='purple')

        # 绘制质心
        cent = rectangle_info['centroid']
        ax9.plot(cent[0], cent[1], 'mo', markersize=10, markeredgecolor='black',
                markeredgewidth=1.5, label='Centroid')

    # 绘制后继车道（3个方向，不同颜色）
    if len(successor_lanes) > 0:
        direction_colors = {
            'forward': 'lime',
            'left': 'yellow',
            'right': 'orange'
        }

        for lane in successor_lanes:
            direction = lane['direction']
            color = direction_colors.get(direction, 'white')
            ax9.plot([lane['p1'][0], lane['p2'][0]],
                    [lane['p1'][1], lane['p2'][1]],
                    color=color, linewidth=3.5, alpha=0.95,
                    label=f"{direction.capitalize()}")
            # 绘制箭头
            ax9.arrow(lane['p1'][0], lane['p1'][1],
                     (lane['p2'][0] - lane['p1'][0]) * 0.8,
                     (lane['p2'][1] - lane['p1'][1]) * 0.8,
                     head_width=0.3, head_length=0.2, fc=color, ec=color, alpha=0.8)

    ax9.plot(0, 0, 'k^', markersize=10, markeredgewidth=2, label='Robot')
    ax9.set_title(f"9. Junction Rectangle + Successor Lanes (N={len(successor_lanes)})",
                 fontsize=12, fontweight='bold')
    ax9.set_xlabel("X (m)")
    ax9.set_ylabel("Y (m)")
    ax9.set_xlim(-10, 10)
    ax9.set_ylim(-10, 10)
    ax9.set_aspect('equal', adjustable='box')
    ax9.grid(True, linestyle=':', alpha=0.4)
    ax9.legend(fontsize=9, loc='best', framealpha=0.9)

    # ========== 第5行右：空白（预留）==========
    ax10 = axes[4, 1]
    ax10.axis('off')
    ax10.text(0.5, 0.5, 'Reserved for future use', ha='center', va='center',
             transform=ax10.transAxes, fontsize=14, color='gray', style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 可视化结果已保存: {output_path}")


def main():
    """主函数"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 完整处理流程
    input_path = os.path.join(os.path.dirname(__file__), INPUT_LASERSCAN_PATH)
    xy, walls, centerlines, main_centerline, polygons, centroids, all_debug_info, intermediate_data, successor_lanes, rectangle_info = process_full_pipeline(input_path)

    # 可视化结果
    output_filename = os.path.splitext(os.path.basename(input_path))[0] + "_complete.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    visualize_all(xy, walls, centerlines, main_centerline, polygons, centroids, all_debug_info, intermediate_data, successor_lanes, rectangle_info, output_path)

    print("\n" + "=" * 70)
    print(" 处理完成!")
    print("=" * 70)
    print(f"点云数量: {len(xy)}")
    print(f"墙体数量: {len(walls)}")
    print(f"中心线数量: {len(centerlines)}")
    print(f"交叉口数量: {len(polygons)}")
    print(f"后继车道数量: {len(successor_lanes)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
