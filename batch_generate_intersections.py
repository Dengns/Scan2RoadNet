#!/usr/bin/env python3
"""
批量生成交叉口识别结果
基于 generate_intersection.py 的算法，批量处理所有墙体和中心线数据
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, MultiPoint
import json
import os
from pathlib import Path
from tqdm import tqdm


# =========================
# 配置参数
# =========================
WALL_DIR = "/home/qzl/test_road/test_online_road/extracted_lidar_data_code/wall_batch"
CENTERLINE_DIR = "/home/qzl/test_road/test_online_road/extracted_lidar_data_code/road_network_batch"
OUTPUT_DIR = "/home/qzl/test_road/test_online_road/extracted_lidar_data_code/intersection_results1V2"

# 延长线长度
EXTEND_LENGTH = 5.0


# =========================
# 核心函数（复用 generate_intersection.py）
# =========================

def get_perpendicular_line_from_p2(p1, p2, length=5.0):
    """
    通过 P2 点构建一条垂直于线段 p1p2 的线段
    """
    # 计算原始线段的方向向量
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # 垂直方向向量，旋转 90 度（逆时针）
    perpendicular_direction = (-dy, dx)

    # 归一化垂直方向
    norm = np.sqrt(perpendicular_direction[0]**2 + perpendicular_direction[1]**2)
    if norm < 1e-6:
        return None
    perpendicular_direction = (perpendicular_direction[0] / norm, perpendicular_direction[1] / norm)

    # 计算垂线的两端
    p1_extended = (p2[0] + perpendicular_direction[0] * length, p2[1] + perpendicular_direction[1] * length)
    p2_extended = (p2[0] - perpendicular_direction[0] * length, p2[1] - perpendicular_direction[1] * length)

    return [p1_extended, p2_extended]


def get_intersection(p1, p2, q1, q2):
    """计算两条线段的交点"""
    # 使用 Shapely 创建线段
    line1 = LineString([p1, p2])
    line2 = LineString([q1, q2])

    # 计算交点
    intersection = line1.intersection(line2)

    # 如果交点是一个点，返回交点的坐标
    if intersection.is_empty:
        return None, None
    elif intersection.geom_type == 'Point':
        return intersection.x, intersection.y
    else:
        # 如果交点是线段，返回第一个交点
        return intersection.coords[0]


def normalize(v):
    """归一化向量"""
    v = np.array(v)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


def extend_line(p1, p2, length=5.0):
    """
    延长线段两端
    """
    p1, p2 = np.array(p1), np.array(p2)

    # 计算方向向量
    direction = normalize(p2 - p1)

    # 延长前端和后端
    p1_extended = p1 - direction * length
    p2_extended = p2 + direction * length

    return p1_extended, p2_extended


def process_single_frame(wall_path, centerline_path):
    """
    处理单个帧，生成交叉口识别结果

    返回:
        walls: 墙体列表
        centerline: 中心线
        polygon: 交叉口多边形
        centroid_coords: 质心坐标
        intersection_coords: 交点列表
    """
    # 加载墙体数据
    with open(wall_path, "r") as f:
        wall_info = json.load(f)
    raw_walls = wall_info['walls']

    walls = []
    for raw_wall in raw_walls:
        # 过滤掉完全在左侧的墙体
        if raw_wall["p1"][0] < -2 and raw_wall["p2"][0] < -2:
            continue
        walls.append(raw_wall)

    # 加载中心线数据
    with open(centerline_path, "r") as f:
        centerlines_info = json.load(f)

    centerlines = centerlines_info["centerlines"]

    # 如果没有中心线，返回空结果
    if len(centerlines) == 0:
        return walls, None, None, None, []

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
        extend_p1, extend_p2 = extend_line(p1, p2, EXTEND_LENGTH)

        if wall["id"] in self_wall_ids:
            self_wall_extended.append([extend_p1, extend_p2])
        else:
            extended_walls.append([extend_p1, extend_p2])

    # 构建垂线
    perpendicular_centerline = get_perpendicular_line_from_p2(
        self_centerline['p1'],
        self_centerline['p2']
    )

    if perpendicular_centerline is None:
        return walls, self_centerline, None, None, []

    # 计算道路边界墙与垂线的交点
    for line in self_wall_extended:
        intersection_x, intersection_y = get_intersection(
            line[0], line[1],
            perpendicular_centerline[0], perpendicular_centerline[1]
        )
        if intersection_x is not None:
            intersection_coords.append([intersection_x, intersection_y])

    # 计算延长线与其他墙体的交点
    for line in self_wall_extended:
        for wall in extended_walls:
            intersection_x, intersection_y = get_intersection(
                line[0], line[1], wall[0], wall[1]
            )
            if intersection_x is not None:
                intersection_coords.append([intersection_x, intersection_y])

    # 创建多边形（凸包）
    polygon = None
    centroid_coords = None

    if len(intersection_coords) >= 3:
        # 使用 Shapely 的 MultiPoint 来计算凸包
        points = [tuple(coord) for coord in intersection_coords]
        multipoint = MultiPoint(points)
        polygon = multipoint.convex_hull

        # 获取多边形的质心
        if polygon and not polygon.is_empty:
            centroid = polygon.centroid
            centroid_coords = (centroid.x, centroid.y)

    return walls, self_centerline, polygon, centroid_coords, intersection_coords


def visualize_and_save(walls, centerline, polygon, centroid_coords,
                       intersection_coords, output_path, frame_idx, has_centerline, has_intersection):
    """
    可视化并保存交叉口识别结果
    """
    # 创建并排的子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左侧图：原始的 centerlines 和 walls
    ax1 = axes[0]

    # 绘制墙体
    for wall in walls:
        p1 = wall["p1"]
        p2 = wall["p2"]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)

    # 绘制中心线
    if centerline:
        center_p1 = centerline["p1"]
        center_p2 = centerline["p2"]
        ax1.plot([center_p1[0], center_p2[0]], [center_p1[1], center_p2[1]],
                'k-', linewidth=2.5, label="Centerline")

    # 添加状态标签
    status = "Normal"
    if not has_centerline:
        status = "No Centerline"
    elif not has_intersection:
        status = "No Intersection"

    ax1.set_title(f"Frame {frame_idx:04d} - Original [{status}]", fontsize=12)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    if centerline:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # 右侧图：原始的 centerlines 和 walls + polygon 区域
    ax2 = axes[1]

    # 绘制墙体
    for wall in walls:
        p1 = wall["p1"]
        p2 = wall["p2"]
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)

    # 绘制交点
    for intersection in intersection_coords:
        ax2.plot(intersection[0], intersection[1], 'go', markersize=6)

    # 绘制中心线
    if centerline:
        center_p1 = centerline["p1"]
        center_p2 = centerline["p2"]
        ax2.plot([center_p1[0], center_p2[0]], [center_p1[1], center_p2[1]],
                'k-', linewidth=2.5, label="Centerline")

    # 绘制多边形区域
    if polygon and not polygon.is_empty:
        x, y = polygon.exterior.xy
        ax2.fill(x, y, alpha=0.3, color='orange', label="Intersection Area")
        ax2.plot(x, y, 'r-', linewidth=2)

    # 绘制多边形的质心
    if centroid_coords:
        ax2.plot(centroid_coords[0], centroid_coords[1], 'ro',
                markersize=10, label="Centroid")

    ax2.set_title(f"Frame {frame_idx:04d} - With Intersection [{status}]", fontsize=12)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    if centerline or polygon:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """主函数：批量处理所有文件"""
    print("=" * 70)
    print(" 批量生成交叉口识别结果")
    print("=" * 70)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有墙体文件
    wall_files = sorted(Path(WALL_DIR).glob("laserscan_*_walls.json"))
    print(f"\n找到 {len(wall_files)} 个墙体文件")

    if len(wall_files) == 0:
        print("错误：未找到任何墙体文件！")
        return

    # 统计信息
    success_count = 0
    fail_count = 0
    no_centerline_count = 0
    no_intersection_count = 0

    # 批量处理
    for wall_path in tqdm(wall_files, desc="处理文件"):
        try:
            # 构建对应的中心线文件路径
            frame_name = wall_path.stem  # 例如：laserscan_000136_walls
            frame_idx = int(frame_name.split('_')[1])  # 提取帧号
            centerline_filename = f"{frame_name}_centerlines.json"
            centerline_path = Path(CENTERLINE_DIR) / centerline_filename

            has_centerline = True
            has_intersection = True
            walls = []
            centerline = None
            polygon = None
            centroid_coords = None
            intersection_coords = []

            # 检查中心线文件是否存在
            if not centerline_path.exists():
                # 无中心线，只加载墙体
                has_centerline = False
                no_centerline_count += 1
                with open(wall_path, "r") as f:
                    wall_info = json.load(f)
                raw_walls = wall_info['walls']
                for raw_wall in raw_walls:
                    if raw_wall["p1"][0] < -2 and raw_wall["p2"][0] < -2:
                        continue
                    walls.append(raw_wall)
            else:
                # 处理单个帧
                walls, centerline, polygon, centroid_coords, intersection_coords = \
                    process_single_frame(str(wall_path), str(centerline_path))

                # 如果没有中心线
                if centerline is None:
                    has_centerline = False
                    no_centerline_count += 1

                # 如果没有交叉口
                if polygon is None or polygon.is_empty:
                    has_intersection = False
                    no_intersection_count += 1

            # 保存可视化结果（包括无中心线和无路口的情况）
            output_path = os.path.join(OUTPUT_DIR, f"intersection_{frame_idx:06d}.png")
            visualize_and_save(walls, centerline, polygon, centroid_coords,
                             intersection_coords, output_path, frame_idx,
                             has_centerline, has_intersection)

            success_count += 1

        except Exception as e:
            fail_count += 1
            print(f"\n警告：处理文件 {wall_path.name} 时出错: {e}")
            continue

    # 打印统计信息
    print("\n" + "=" * 70)
    print(" 处理完成统计")
    print("=" * 70)
    print(f"总文件数: {len(wall_files)}")
    print(f"成功处理: {success_count}")
    print(f"无中心线: {no_centerline_count}")
    print(f"无交叉口: {no_intersection_count}")
    print(f"处理失败: {fail_count}")
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
