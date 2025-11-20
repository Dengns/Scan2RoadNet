#!/usr/bin/env python3
"""
批量生成MP4视频和GIF动图
从PNG序列帧生成视频文件
"""

import os
import imageio
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_sorted_png_files(folder_path):
    """获取文件夹中按序号排序的PNG文件列表"""
    folder = Path(folder_path)
    png_files = sorted([f for f in folder.glob("frame_*_network.png")])
    return png_files


def create_mp4(png_files, output_path, fps=10):
    """生成MP4视频"""
    print(f"  生成MP4视频: {output_path}")
    print(f"  帧数: {len(png_files)}, 帧率: {fps}fps, 时长: {len(png_files)/fps:.1f}秒")

    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')

    for png_file in tqdm(png_files, desc="  写入MP4帧"):
        img = imageio.imread(png_file)
        writer.append_data(img)

    writer.close()
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ MP4生成完成，大小: {file_size_mb:.1f}MB\n")


def create_gif(png_files, output_path, fps=5, max_width=800):
    """生成GIF动图（压缩版）"""
    print(f"  生成GIF动图: {output_path}")
    print(f"  帧数: {len(png_files)}, 帧率: {fps}fps, 时长: {len(png_files)/fps:.1f}秒")
    print(f"  压缩设置: 宽度缩放到{max_width}px")

    frames = []

    for png_file in tqdm(png_files, desc="  读取并压缩帧"):
        img = Image.open(png_file)

        # 计算缩放比例（保持宽高比）
        width, height = img.size
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 转换为RGB（GIF需要）
        if img.mode != 'RGB':
            img = img.convert('RGB')

        frames.append(np.array(img))

    print("  正在写入GIF文件...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✅ GIF生成完成，大小: {file_size_mb:.1f}MB\n")


def process_folder(folder_path):
    """处理单个文件夹，生成MP4和GIF"""
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    print("=" * 70)
    print(f"处理文件夹: {folder_name}")
    print("=" * 70)

    # 获取PNG文件
    png_files = get_sorted_png_files(folder_path)

    if len(png_files) == 0:
        print(f"❌ 未找到PNG文件！")
        return

    print(f"找到 {len(png_files)} 个PNG文件\n")

    # 输出路径
    mp4_output = folder_path / f"{folder_name}.mp4"
    gif_output = folder_path / f"{folder_name}.gif"

    # 生成MP4
    create_mp4(png_files, str(mp4_output), fps=10)

    # 生成GIF
    create_gif(png_files, str(gif_output), fps=5, max_width=800)

    print("=" * 70)
    print(f"✅ {folder_name} 处理完成！")
    print(f"   MP4: {mp4_output}")
    print(f"   GIF: {gif_output}")
    print("=" * 70)
    print()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print(" 批量生成MP4视频和GIF动图")
    print("=" * 70)
    print()

    # 两个文件夹路径
    folders = [
        "extracted_lidar_data_code/batch_all_in_one_results",
        "extracted_lidar_data_code/batch_all_in_one_results2"
    ]

    # 处理每个文件夹
    for folder in folders:
        folder_path = os.path.join(os.path.dirname(__file__), folder)
        if os.path.exists(folder_path):
            process_folder(folder_path)
        else:
            print(f"❌ 文件夹不存在: {folder_path}\n")

    print("\n" + "=" * 70)
    print(" 全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
