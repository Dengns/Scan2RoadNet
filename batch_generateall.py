#!/usr/bin/env python3
"""
批量处理所有LiDAR扫描数据
从原始点云到完整的道路网络（墙体+中心线+交叉口）
只保存最终结果图（图8：Complete Network + Debug）

使用优化后的墙体检测算法：
- 角度wrap处理（解决0度突变问题）
- 距离自适应DBSCAN（远处容忍度更大）
- 修复的Hessian法线计算
- RDP长度过滤
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
import sys

# 导入generateall.py中的函数
sys.path.insert(0, os.path.dirname(__file__))
from generateall import (
    laserscan_to_xy, detect_walls, find_wall_pairs, generate_centerline,
    detect_intersection, MAX_RANGE, X_LIMIT, Y_LIMIT, RDP_EPSILON, MIN_SPLIT_POINTS,
    MIN_RDP_SEGMENT_LENGTH, PARAM_DBSCAN_EPS, ALPHA_SCALE, BETA_SCALE, GAP_THRESH
)


# =========================
# 配置参数
# =========================
LASERSCAN_DIR = "extracted_lidar_data_code/extracted_lidar_data/laserscan_json"
OUTPUT_DIR = "extracted_lidar_data_code/batch_all_in_one_results"


def visualize_final_only(xy, walls, centerlines, polygons, centroids, all_debug_info, output_path, frame_idx):
    """只可视化最终结果图（图8）"""
    fig, ax = plt.subplots(figsize=(14, 14))

    # 绘制点云背景
    if len(xy) > 0:
        ax.scatter(xy[:, 0], xy[:, 1], c='lightgray', s=1, alpha=0.3)

    # 获取道路边界墙的ID
    road_wall_ids = set()
    if len(centerlines) > 0:
        wall1, wall2 = centerlines[0]['wall_pair']
        road_wall_ids = {wall1['id'], wall2['id']}

    # 🔑 绘制延长线和垂线（如果有调试信息）
    if len(all_debug_info) > 0:
        debug_info = all_debug_info[0]

        # 绘制道路边界墙的延长线（蓝色和绿色虚线）
        for i, line in enumerate(debug_info['self_wall_extended']):
            color = 'b' if i == 0 else 'g'
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                   color=color, linestyle=':', linewidth=1.5, alpha=0.5,
                   label='Extended Road Walls' if i == 0 else '')

        # 绘制其他墙的延长线（灰色虚线）
        for i, line in enumerate(debug_info['other_wall_extended']):
            ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]],
                   color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                   label='Extended Other Walls' if i == 0 else '')

        # 绘制垂线（紫色虚线）
        if debug_info['perp_line'] is not None:
            perp = debug_info['perp_line']
            ax.plot([perp[0][0], perp[1][0]], [perp[0][1], perp[1][1]],
                   color='purple', linestyle='--', linewidth=2, alpha=0.7,
                   label='Perpendicular Line')

        # 绘制交点（黄色圆点）
        for i, coord in enumerate(debug_info['intersection_coords']):
            ax.plot(coord[0], coord[1], 'yo', markersize=8, markeredgecolor='black',
                   markeredgewidth=1, label='Intersection Points' if i == 0 else '')

    # 绘制墙体（区分左右墙）
    label_shown = {'Left Wall': False, 'Right Wall': False, 'Other Walls': False, 'Road Wall': False}

    for wall in walls:
        fit = wall['fit']

        if wall['id'] in road_wall_ids:
            if len(centerlines) > 0:
                wall1, wall2 = centerlines[0]['wall_pair']
                if wall['id'] == wall1['id']:
                    color, label = 'b', 'Left Wall'
                else:
                    color, label = 'g', 'Right Wall'
            else:
                color, label = 'b', 'Road Wall'
        else:
            color, label = 'gray', 'Other Walls'

        # 只在第一次出现时显示label
        show_label = label if not label_shown[label] else ''
        if not label_shown[label]:
            label_shown[label] = True

        ax.plot([fit['p1'][0], fit['p2'][0]],
               [fit['p1'][1], fit['p2'][1]],
               color=color, linewidth=3, alpha=0.8, label=show_label)

    # 绘制中心线
    for i, cl in enumerate(centerlines):
        ax.plot([cl['p1'][0], cl['p2'][0]],
               [cl['p1'][1], cl['p2'][1]],
               'r--', linewidth=3, alpha=0.9, label='Centerline' if i == 0 else '')

    # 绘制交叉口
    for i, poly in enumerate(polygons):
        if poly and not poly.is_empty:
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.3, color='orange', label='Intersection' if i == 0 else '')
            ax.plot(x, y, 'r-', linewidth=2, alpha=0.7)

    # 绘制质心
    for i, cent in enumerate(centroids):
        if cent is not None:
            ax.plot(cent[0], cent[1], 'ro', markersize=10, label='Centroid' if i == 0 else '')

    # 绘制机器人位置
    ax.plot(0, 0, 'k^', markersize=12, markeredgewidth=2, label='Robot')

    ax.set_title(f"Frame {frame_idx:04d} - Complete Network (W={len(walls)}, CL={len(centerlines)}, INT={len(polygons)})",
                fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.4)

    # 处理图例，去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc='best', framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def process_single_scan(laserscan_path, output_path, frame_idx):
    """处理单个扫描文件（总是生成可视化，即使检测失败）"""
    import json

    try:
        # 1. 加载点云数据
        with open(laserscan_path, "r") as f:
            data = json.load(f)
        laserscan = data["laserscan"] if "laserscan" in data else data

        xy = laserscan_to_xy(laserscan, max_range=MAX_RANGE)

        # 空间过滤
        mask = (xy[:,0] >= -X_LIMIT) & (xy[:,0] <= X_LIMIT) & \
               (xy[:,1] >= -Y_LIMIT) & (xy[:,1] <= Y_LIMIT)
        xy = xy[mask]
        xy = xy[np.isfinite(xy).all(axis=1)]

        # 初始化空结果（如果后续步骤失败，就用空结果）
        walls = []
        centerlines = []
        polygons = []
        centroids = []
        all_debug_info = []
        status_msg = ""

        if len(xy) == 0:
            status_msg = "No valid points"
        else:
            # 2. 墙体检测
            walls, _ = detect_walls(xy)

            if len(walls) == 0:
                status_msg = "No walls detected"
            else:
                # 3. 道路中心线生成
                wall_pairs = find_wall_pairs(walls)

                if len(wall_pairs) == 0:
                    status_msg = "No road pairs found"
                else:
                    for wall1, wall2, dist in wall_pairs:
                        cl = generate_centerline(wall1, wall2)
                        centerlines.append(cl)

                    # 4. 交叉口识别
                    for cl in centerlines:
                        poly, cent, debug_info = detect_intersection(walls, cl)
                        if poly is not None:
                            polygons.append(poly)
                            centroids.append(cent)
                            all_debug_info.append(debug_info)

                    status_msg = f"W={len(walls)}, CL={len(centerlines)}, INT={len(polygons)}"

        # 5. 可视化（总是生成，即使结果为空）
        visualize_final_only(xy, walls, centerlines, polygons, centroids, all_debug_info, output_path, frame_idx)

        # 如果有错误信息，返回False；否则返回True
        success = (status_msg != "" and "No " not in status_msg)
        return success, status_msg if status_msg else f"W={len(walls)}, CL={len(centerlines)}, INT={len(polygons)}"

    except Exception as e:
        return False, str(e)


def main():
    """主函数：批量处理"""
    print("=" * 70)
    print(" 批量处理所有LiDAR扫描数据（优化版）")
    print("=" * 70)

    # 打印关键参数
    print("\n📋 墙体检测参数:")
    print(f"  - RDP精度: {RDP_EPSILON}m")
    print(f"  - 最少点数: {MIN_SPLIT_POINTS}")
    print(f"  - 最小长度: {MIN_RDP_SEGMENT_LENGTH}m")
    print(f"  - DBSCAN eps: {PARAM_DBSCAN_EPS}")
    print(f"  - θ缩放: {ALPHA_SCALE}")
    print(f"  - ρ动态系数: {BETA_SCALE}")
    print(f"  - 合并间隔: {GAP_THRESH}m")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 获取所有laserscan文件
    laserscan_dir = Path(LASERSCAN_DIR)
    scan_files = sorted(laserscan_dir.glob("laserscan_*.json"))

    print(f"\n找到 {len(scan_files)} 个扫描文件")

    if len(scan_files) == 0:
        print("错误：未找到任何扫描文件！")
        return

    # 详细统计信息
    success_count = 0
    no_points_count = 0
    no_walls_count = 0
    no_road_pairs_count = 0
    exception_count = 0
    error_details = []  # 存储错误详情

    # 批量处理
    for scan_file in tqdm(scan_files, desc="处理中"):
        try:
            # 提取帧号
            frame_name = scan_file.stem  # 例如：laserscan_000136
            frame_idx = int(frame_name.split('_')[1])

            # 输出路径
            output_filename = f"frame_{frame_idx:06d}_network.png"
            output_path = Path(OUTPUT_DIR) / output_filename

            # 处理
            success, info = process_single_scan(str(scan_file), str(output_path), frame_idx)

            if success:
                success_count += 1
            else:
                # 详细分类错误类型
                if "No valid points" in info:
                    no_points_count += 1
                elif "No walls detected" in info:
                    no_walls_count += 1
                elif "No road pairs found" in info:
                    no_road_pairs_count += 1
                else:
                    # 其他异常错误
                    exception_count += 1
                    error_details.append((scan_file.name, info))

        except Exception as e:
            exception_count += 1
            error_details.append((scan_file.name, str(e)))
            continue

    # 打印统计信息
    total_count = len(scan_files)
    fail_count = no_points_count + no_walls_count + no_road_pairs_count + exception_count

    print("\n" + "=" * 70)
    print(" 处理完成统计")
    print("=" * 70)
    print(f"总文件数:        {total_count}")
    print(f"成功处理:        {success_count} ({100*success_count/total_count:.1f}%)")
    print(f"\n失败分类:")
    print(f"  ├─ 无有效点云:  {no_points_count} ({100*no_points_count/total_count:.1f}%)")
    print(f"  ├─ 无墙体:      {no_walls_count} ({100*no_walls_count/total_count:.1f}%)")
    print(f"  ├─ 无道路对:    {no_road_pairs_count} ({100*no_road_pairs_count/total_count:.1f}%)")
    print(f"  └─ 异常错误:    {exception_count} ({100*exception_count/total_count:.1f}%)")
    print(f"\n总失败数:        {fail_count} ({100*fail_count/total_count:.1f}%)")

    # 打印前5个异常错误详情
    if len(error_details) > 0:
        print(f"\n异常错误详情（前{min(5, len(error_details))}个）：")
        for i, (filename, error) in enumerate(error_details[:5]):
            print(f"  {i+1}. {filename}: {error}")

    print(f"\n输出目录: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
