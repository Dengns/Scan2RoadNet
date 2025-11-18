#!/usr/bin/env python3
"""
批量处理墙体数据，生成道路中心线gif动画
- 读取所有墙体JSON文件
- 生成道路中心线（不包含路口检测）
- 生成中心线可视化gif动画
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
import time
from tqdm import tqdm
import imageio


# =========================
# 全局配置参数
# =========================
# 墙体数据输入目录
WALL_JSON_DIR = "/home/qzl/Main/MobiMind/test/test_online_road/extracted_lidar_data_code/wall_batch2/"

# 输出目录
OUT_DIR = "/home/qzl/Main/MobiMind/test/test_online_road/extracted_lidar_data_code/road_network_batch2"
# GIF_OUTPUT = "/home/qzl/Main/MobiMind/test/test_online_road/extracted_lidar_data_code/centerlines_animation.gif"

# ===== 道路配对参数 =====
MAX_ROAD_WIDTH = 3.0        # 最大道路宽度（米）
MIN_ROAD_WIDTH = 0.8        # 最小道路宽度（米）
PARALLEL_ANGLE_THRESH = 10  # 平行判定角度阈值（度）
MIN_OVERLAP_RATIO = 0.3     # 最小重叠比例（墙体投影重叠）

# GIF参数
GIF_FPS = 10  # 帧率
X_LIMIT = 10.0   # ±10m
Y_LIMIT = 5.0    # ±5m


# =========================================================================
# ===== 数据结构定义 =====
# =========================================================================

@dataclass
class Wall:
    """墙体数据结构"""
    id: int
    p1: np.ndarray  # 起点
    p2: np.ndarray  # 终点
    length: float
    direction: np.ndarray  # 方向向量
    rmse: float

    @property
    def center(self) -> np.ndarray:
        """墙体中心点"""
        return 0.5 * (self.p1 + self.p2)

    @property
    def angle(self) -> float:
        """墙体角度（弧度）"""
        return np.arctan2(self.direction[1], self.direction[0])


@dataclass
class Centerline:
    """道路中心线数据结构"""
    id: int
    p1: np.ndarray  # 起点
    p2: np.ndarray  # 终点
    width: float    # 道路宽度
    wall_pair: Tuple[int, int]  # 配对的墙体ID

    @property
    def center(self) -> np.ndarray:
        """中心线中点"""
        return 0.5 * (self.p1 + self.p2)

    @property
    def length(self) -> float:
        """中心线长度"""
        return np.linalg.norm(self.p2 - self.p1)

    @property
    def direction(self) -> np.ndarray:
        """方向向量"""
        vec = self.p2 - self.p1
        return vec / (np.linalg.norm(vec) + 1e-12)


# =========================================================================
# ===== 复用原始代码中的核心函数 =====
# =========================================================================

def load_walls(json_path: str) -> List[Wall]:
    """从JSON文件加载墙体数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    walls = []
    for w in data['walls']:
        walls.append(Wall(
            id=w['id'],
            p1=np.array(w['p1']),
            p2=np.array(w['p2']),
            length=w['length'],
            direction=np.array(w['direction']),
            rmse=w['rmse']
        ))

    return walls


def are_parallel(wall1: Wall, wall2: Wall, angle_thresh_deg: float) -> bool:
    """判断两面墙是否平行"""
    dot = np.abs(np.dot(wall1.direction, wall2.direction))
    angle_diff = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    return angle_diff < angle_thresh_deg or angle_diff > (180 - angle_thresh_deg)


def compute_wall_distance(wall1: Wall, wall2: Wall) -> float:
    """计算两面平行墙体之间的平均距离"""
    normal1 = np.array([-wall1.direction[1], wall1.direction[0]])
    d1 = np.dot(wall2.p1 - wall1.p1, normal1)
    d2 = np.dot(wall2.p2 - wall1.p1, normal1)
    return np.abs(0.5 * (d1 + d2))


def compute_projection_overlap(wall1: Wall, wall2: Wall) -> float:
    """计算两面墙体在它们方向上的投影重叠比例"""
    dir_vec = wall1.direction

    # 投影墙1的端点
    t1_start = np.dot(wall1.p1, dir_vec)
    t1_end = np.dot(wall1.p2, dir_vec)
    if t1_start > t1_end:
        t1_start, t1_end = t1_end, t1_start

    # 投影墙2的端点
    t2_start = np.dot(wall2.p1, dir_vec)
    t2_end = np.dot(wall2.p2, dir_vec)
    if t2_start > t2_end:
        t2_start, t2_end = t2_end, t2_start

    # 计算重叠区间
    overlap_start = max(t1_start, t2_start)
    overlap_end = min(t1_end, t2_end)
    overlap_length = max(0, overlap_end - overlap_start)

    # 计算重叠比例（相对于较短的墙）
    min_length = min(t1_end - t1_start, t2_end - t2_start)
    if min_length < 1e-6:
        return 0.0

    return overlap_length / min_length


def find_wall_pairs(walls: List[Wall],
                    max_width: float,
                    min_width: float,
                    angle_thresh: float,
                    min_overlap: float) -> List[Tuple[Wall, Wall]]:
    """找到可以配对形成道路的墙体对"""
    pairs = []
    used_walls = set()

    for i, wall1 in enumerate(walls):
        if wall1.id in used_walls:
            continue

        best_match = None
        best_score = -1

        for j, wall2 in enumerate(walls):
            if i >= j or wall2.id in used_walls:
                continue

            # 1. 检查是否平行
            if not are_parallel(wall1, wall2, angle_thresh):
                continue

            # 2. 检查距离是否在道路宽度范围内
            dist = compute_wall_distance(wall1, wall2)
            if dist < min_width or dist > max_width:
                continue

            # 3. 检查投影重叠度
            overlap = compute_projection_overlap(wall1, wall2)
            if overlap < min_overlap:
                continue

            # 4. 计算配对得分（重叠度越高越好，距离适中越好）
            ideal_width = 1.5
            width_score = 1.0 - abs(dist - ideal_width) / max_width
            score = overlap * 0.7 + width_score * 0.3

            if score > best_score:
                best_score = score
                best_match = (wall2, dist)

        # 如果找到了最佳配对
        if best_match is not None:
            wall2, dist = best_match
            pairs.append((wall1, wall2))
            used_walls.add(wall1.id)
            used_walls.add(wall2.id)

    return pairs


def generate_centerline_from_pair(wall1: Wall, wall2: Wall, pair_id: int) -> Centerline:
    """根据墙体对生成道路中心线"""
    # 使用墙1的方向作为道路方向
    dir_vec = wall1.direction

    # 投影所有端点到方向轴
    t1 = np.dot(wall1.p1, dir_vec)
    t2 = np.dot(wall1.p2, dir_vec)
    t3 = np.dot(wall2.p1, dir_vec)
    t4 = np.dot(wall2.p2, dir_vec)

    # 找到重叠区间
    t_start = max(min(t1, t2), min(t3, t4))
    t_end = min(max(t1, t2), max(t3, t4))

    # 中心线的起点和终点在重叠区间内
    def get_point_at_t(wall: Wall, t: float) -> np.ndarray:
        """在投影位置t处获取墙上的点"""
        t1 = np.dot(wall.p1, dir_vec)
        t2 = np.dot(wall.p2, dir_vec)
        if abs(t2 - t1) < 1e-6:
            return wall.p1
        ratio = (t - t1) / (t2 - t1)
        return wall.p1 + ratio * (wall.p2 - wall.p1)

    # 起点: 两墙在t_start处的中点
    start1 = get_point_at_t(wall1, t_start)
    start2 = get_point_at_t(wall2, t_start)
    centerline_start = 0.5 * (start1 + start2)

    # 终点: 两墙在t_end处的中点
    end1 = get_point_at_t(wall1, t_end)
    end2 = get_point_at_t(wall2, t_end)
    centerline_end = 0.5 * (end1 + end2)

    # 计算道路宽度
    width = compute_wall_distance(wall1, wall2)

    return Centerline(
        id=pair_id,
        p1=centerline_start,
        p2=centerline_end,
        width=width,
        wall_pair=(wall1.id, wall2.id)
    )


# =========================================================================
# ===== 处理单个文件的核心函数 =====
# =========================================================================

def process_single_file(wall_json_path: str) -> Tuple[List[Wall], List[Centerline]]:
    """
    处理单个墙体文件，生成道路中心线

    返回:
        walls: 墙体列表
        centerlines: 中心线列表
    """
    # 1. 加载墙体数据
    walls = load_walls(wall_json_path)

    # 2. 墙体配对
    wall_pairs = find_wall_pairs(
        walls,
        max_width=MAX_ROAD_WIDTH,
        min_width=MIN_ROAD_WIDTH,
        angle_thresh=PARALLEL_ANGLE_THRESH,
        min_overlap=MIN_OVERLAP_RATIO
    )

    # 3. 生成道路中心线
    centerlines = []
    for i, (wall1, wall2) in enumerate(wall_pairs):
        cl = generate_centerline_from_pair(wall1, wall2, i)
        centerlines.append(cl)

    return walls, centerlines


def plot_centerline_frame(walls: List[Wall], centerlines: List[Centerline],
                          frame_idx: int) -> np.ndarray:
    """
    生成单帧中心线可视化图像

    返回:
        image: RGB图像数组 (H, W, 3)
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)

    # 绘制墙体（黑色实线）
    for wall in walls:
        ax.plot([wall.p1[0], wall.p2[0]],
               [wall.p1[1], wall.p2[1]],
               'k-', linewidth=2.5, alpha=0.7)

    # 绘制道路中心线（红色虚线，加粗）
    for cl in centerlines:
        ax.plot([cl.p1[0], cl.p2[0]],
               [cl.p1[1], cl.p2[1]],
               'r--', linewidth=3.5, alpha=0.95)

        # 标注中心线宽度
        mid = cl.center
        ax.text(mid[0], mid[1], f'{cl.width:.2f}m',
               fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # 绘制原点（自车坐标系）
    ax.plot(0, 0, 'ko', markersize=10, markeredgewidth=2,
            markerfacecolor='black', label='Robot Origin')

    ax.set_title(f"Frame {frame_idx:04d} - Centerlines (N={len(centerlines)})",
                  fontsize=14, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)

    # 添加图例
    if len(walls) > 0:
        ax.plot([], [], 'k-', linewidth=2.5, label='Walls')
    if len(centerlines) > 0:
        ax.plot([], [], 'r--', linewidth=3.5, label='Centerlines')
    ax.legend(loc='upper right', fontsize=10)

    # 转换为图像数组
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image = image[:, :, :3]  # 只取RGB通道

    plt.close(fig)
    return image


def save_centerline_json(walls: List[Wall], centerlines: List[Centerline],
                         wall_json_path: str, output_dir: str):
    """保存中心线数据为JSON"""
    road_network_data = {
        'centerlines': [],
        'metadata': {
            'input_file': str(wall_json_path),
            'num_centerlines': len(centerlines),
            'num_walls': len(walls),
            'params': {
                'max_road_width': MAX_ROAD_WIDTH,
                'min_road_width': MIN_ROAD_WIDTH,
                'parallel_angle_thresh': PARALLEL_ANGLE_THRESH,
                'min_overlap_ratio': MIN_OVERLAP_RATIO
            }
        }
    }

    # 保存中心线
    for cl in centerlines:
        road_network_data['centerlines'].append({
            'id': cl.id,
            'p1': cl.p1.tolist(),
            'p2': cl.p2.tolist(),
            'width': cl.width,
            'length': cl.length,
            'wall_pair': cl.wall_pair
        })

    # 保存JSON
    basename = os.path.splitext(os.path.basename(wall_json_path))[0]
    out_json_path = os.path.join(output_dir, f"{basename}_centerlines.json")
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(road_network_data, f, indent=2, ensure_ascii=False)


# =========================================================================
# ===== 主函数：批量处理 =====
# =========================================================================

def main():
    """批量处理所有墙体数据，生成中心线gif"""
    print("="*70)
    print(" 批量道路中心线生成与GIF制作")
    print("="*70)

    # 创建输出目录
    os.makedirs(OUT_DIR, exist_ok=True)

    # 获取所有墙体JSON文件（从第130帧开始）
    all_files = sorted(Path(WALL_JSON_DIR).glob("*_walls.json"))
    # 只保留从130帧开始的文件
    input_files = [f for f in all_files if int(f.stem.split('_')[1]) >= 130]
    print(f"\n找到 {len(all_files)} 个墙体数据文件")
    print(f"从第130帧开始处理，实际处理 {len(input_files)} 个文件")

    if len(input_files) == 0:
        print("错误：未找到任何输入文件！")
        return

    # 批量处理
    frames = []
    total_start = time.time()
    total_centerlines = 0

    for idx, wall_json_path in enumerate(tqdm(input_files, desc="处理文件")):
        try:
            # 处理单个文件
            walls, centerlines = process_single_file(str(wall_json_path))
            total_centerlines += len(centerlines)

            # 生成可视化图像
            image = plot_centerline_frame(walls, centerlines, idx)
            frames.append(image)

            # 保存中心线JSON
            save_centerline_json(walls, centerlines, str(wall_json_path), OUT_DIR)

        except Exception as e:
            print(f"\n警告：处理文件 {wall_json_path.name} 时出错: {e}")
            continue

    total_time = time.time() - total_start

    # 生成GIF
    if len(frames) > 0:
        print(f"\n生成GIF动画...")
        imageio.mimsave(GIF_OUTPUT, frames, fps=GIF_FPS, loop=0)
        print(f"✅ GIF已保存: {GIF_OUTPUT}")
    else:
        print("错误：没有生成任何帧！")

    # 统计信息
    print("\n" + "="*70)
    print(" 处理完成统计")
    print("="*70)
    print(f"总文件数: {len(input_files)}")
    print(f"成功处理: {len(frames)}")
    print(f"总中心线数: {total_centerlines}")
    print(f"平均每帧中心线: {total_centerlines/max(len(frames), 1):.1f}")
    print(f"总耗时: {total_time:.2f}s")
    print(f"平均速度: {len(frames)/total_time:.2f} FPS")
    print(f"\n输出位置:")
    print(f"  - GIF动画: {GIF_OUTPUT}")
    print(f"  - 中心线JSON: {OUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()
