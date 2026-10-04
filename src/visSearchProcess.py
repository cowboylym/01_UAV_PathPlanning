#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
搜索过程可视化 — FMM / A* / RRT* 三张论文插图
================================================
2D 俯视图: 底色=DSM 地形高程 (terrain 配色), 叠加各算法搜索过程数据.

输入文件 (../data/):
  - SF_Downtown.tif              DSM 高程 (底色)
  - search_FMM_Tfield.csv        FMM T 场 2D 切片 (x,y,T)
  - search_ASTAR_nodes.csv       A* 搜索节点 (x,y,z,parent_idx)
  - search_RRTSTAR_tree.csv      RRT* 搜索树 (x,y,z,parent_idx,cost)
  - search_path_FMM.csv          FMM 最终路径 (x,y,z)
  - search_path_ASTAR.csv        A* 最终路径 (x,y,z)
  - search_path_RRTSTAR.csv      RRT* 最终路径 (x,y,z)

输出 (../analysis/):
  - 搜索过程_FMM.png             波前等值线 + 地形底色 + 路径
  - 搜索过程_ASTAR.png           访问节点散点 + 边 + 地形底色 + 路径
  - 搜索过程_RRTSTAR.png         搜索树 + 地形底色 + 路径

依赖: pip install numpy matplotlib
      (可选) pip install rasterio   # GeoTIFF 读取
"""

import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.collections import LineCollection

# -------------------------- 全局样式 --------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

OUTPUT_DIR = '../analysis'
DATA_DIR = '../data/compare4/final_v1/raw'
DSM_PATH = '../data/SF_Downtown.tif'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Okabe-Ito 色盲友好调色板 (Nature 期刊风格)
OKABE_ITO = {
    'blue':   '#0072B2',
    'orange': '#E69F00',
    'green':  '#009E73',
    'red':    '#D55E00',
    'purple': '#CC79A7',
    'sky':    '#56B4E9',
    'yellow': '#F0E442',
    'black':  '#000000',
}


# -------------------------- GeoTIFF 读取 (简化版) --------------------------
def read_geotiff(filepath):
    """读取 GeoTIFF, 返回 (elevation, extent) where extent=[x_min,x_max,y_min,y_max]."""
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            elev = src.read(1).astype(np.float32)
            gt = src.transform
            x_min = gt.c
            x_max = gt.c + elev.shape[1] * gt.a
            y_max = gt.f
            y_min = gt.f + elev.shape[0] * gt.e
        print(f"[rasterio] DSM shape={elev.shape} extent=[{x_min:.1f},{x_max:.1f},{y_min:.1f},{y_max:.1f}]")
        return elev, [x_min, x_max, y_min, y_max]
    except ImportError:
        pass

    try:
        from osgeo import gdal
        ds = gdal.Open(filepath)
        band = ds.GetRasterBand(1)
        elev = band.ReadAsArray().astype(np.float32)
        gt = ds.GetGeoTransform()
        x_min = gt[0]
        x_max = gt[0] + elev.shape[1] * gt[1]
        y_max = gt[3]
        y_min = gt[3] + elev.shape[0] * gt[5]
        print(f"[GDAL] DSM shape={elev.shape} extent=[{x_min:.1f},{x_max:.1f},{y_min:.1f},{y_max:.1f}]")
        return elev, [x_min, x_max, y_min, y_max]
    except ImportError:
        pass

    from PIL import Image
    img = Image.open(filepath)
    elev = np.array(img).astype(np.float32)
    tags = img.tag_v2 if hasattr(img, 'tag_v2') else {}
    scale = tags.get(33550, (1.0, 1.0, 1.0))
    tiepoint = tags.get(33922, (0, 0, 0, 0, 0, 0))
    x_min = tiepoint[3]
    y_max = tiepoint[4]
    x_max = x_min + elev.shape[1] * scale[0]
    y_min = y_max - elev.shape[0] * scale[1]
    print(f"[PIL] DSM shape={elev.shape} extent=[{x_min:.1f},{x_max:.1f},{y_min:.1f},{y_max:.1f}]")
    return elev, [x_min, x_max, y_min, y_max]


# -------------------------- CSV 加载 --------------------------
def load_csv(filepath):
    """加载 CSV, 返回 (header, list_of_rows)."""
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows


def load_path_csv(filepath):
    """加载路径 CSV (x,y,z), 返回 numpy 数组 (N, 3)."""
    _, rows = load_csv(filepath)
    return np.array([[float(v) for v in row] for row in rows], dtype=np.float64)


# -------------------------- 通用绘图元素 --------------------------
# FMM T 场色标: 绿(近 goal, T=0) → 黄 → 红(远 goal, T 大)
# 与等值线配合: 终点附近绿色 (波前起点), 远处红色 (波前最后到达)
FMM_CMAP = LinearSegmentedColormap.from_list(
    'fmm_green_red',
    [(0.00, '#009E73'),  # 绿: T=0 (goal, 波源)
     (0.25, '#56B4E9'),  # 蓝绿: T 较小 (近 goal)
     (0.50, '#F0E442'),  # 黄: T 中等
     (0.75, '#E69F00'),  # 橙: T 较大
     (1.00, '#D55E00')], # 红: T 大 (远 goal, 波前最后到达)
)


def draw_terrain_background(ax, elev, extent, data_bounds, shrink=1):
    """绘制 DSM 地形底色 (灰度, 低 alpha 作为背景参考).
    学术论文标准做法: 灰底 + 彩色搜索数据, 对比最强."""
    x_min, x_max, y_min, y_max = extent
    dx, dy = data_bounds
    # margin: 数据范围外扩 20%
    mx = (dx[1] - dx[0]) * 0.2
    my = (dy[1] - dy[0]) * 0.2
    cx_min, cx_max = dx[0] - mx, dx[1] + mx
    cy_min, cy_max = dy[0] - my, dy[1] + my

    # 裁剪到 DSM 范围
    cx_min = max(cx_min, x_min); cx_max = min(cx_max, x_max)
    cy_min = max(cy_min, y_min); cy_max = min(cy_max, y_max)

    # 计算像素范围
    px_min = int((cx_min - x_min) / (x_max - x_min) * elev.shape[1])
    px_max = int((cx_max - x_min) / (x_max - x_min) * elev.shape[1])
    py_min = int((y_max - cy_max) / (y_max - y_min) * elev.shape[0])
    py_max = int((y_max - cy_min) / (y_max - y_min) * elev.shape[0])
    px_min = max(0, px_min); px_max = min(elev.shape[1], px_max)
    py_min = max(0, py_min); py_max = min(elev.shape[0], py_max)

    sub = elev[py_min:py_max:shrink, px_min:px_max:shrink]
    sub_extent = [cx_min, cx_max, cy_min, cy_max]

    im = ax.imshow(sub, cmap='gray', origin='upper', extent=sub_extent,
                   aspect='equal', interpolation='nearest', alpha=0.35)
    return im, (cx_min, cx_max, cy_min, cy_max)


def draw_start_goal(ax, path):
    """绘制起点 (绿色五角星) 和终点 (红色五角星), 白色描边确保醒目."""
    # 白色底层 (外描边效果)
    ax.scatter(path[0, 0], path[0, 1], marker='*', s=500, c='white',
               edgecolors='white', linewidths=2.5, zorder=19)
    ax.scatter(path[-1, 0], path[-1, 1], marker='*', s=500, c='white',
               edgecolors='white', linewidths=2.5, zorder=19)
    # 彩色顶层
    ax.scatter(path[0, 0], path[0, 1], marker='*', s=350, c=OKABE_ITO['green'],
               edgecolors='black', linewidths=1.0, zorder=20, label='Start')
    ax.scatter(path[-1, 0], path[-1, 1], marker='*', s=350, c=OKABE_ITO['red'],
               edgecolors='black', linewidths=1.0, zorder=20, label='Goal')


def draw_final_path(ax, path):
    """绘制最终路径 (黑色粗线 + 白色底衬, 任何背景下都清晰)."""
    ax.plot(path[:, 0], path[:, 1], color='white', linewidth=4.5, zorder=14)
    ax.plot(path[:, 0], path[:, 1], color=OKABE_ITO['black'], linewidth=2.5,
            zorder=15, label='Final path')


def finalize_axes(ax, bounds, title):
    """设置坐标轴、标题、图例 (白底黑框确保可见)."""
    cx_min, cx_max, cy_min, cy_max = bounds
    ax.set_xlim(cx_min, cx_max)
    ax.set_ylim(cy_min, cy_max)
    ax.set_xlabel('Easting X (m)', fontsize=11)
    ax.set_ylabel('Northing Y (m)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    legend = ax.legend(loc='upper left', fontsize=10, framealpha=1.0,
                       edgecolor='black', facecolor='white', fancybox=False,
                       borderaxespad=0.6)
    legend.get_frame().set_linewidth(1.2)
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)


# -------------------------- FMM: 波前等值线 --------------------------
def render_fmm(dsm_path, tfield_csv, path_csv, out_path, dpi=300):
    """渲染 FMM 搜索过程: T 场等值线 (波前传播) + 地形底色 + 最终路径."""
    print(f"\n=== 渲染 FMM 搜索过程 ===")
    elev, extent = read_geotiff(dsm_path)
    path = load_path_csv(path_csv)
    print(f"  路径点数: {len(path)}")

    # 加载 T 场 CSV
    header, rows = load_csv(tfield_csv)
    xs = np.array([float(r[0]) for r in rows])
    ys = np.array([float(r[1]) for r in rows])
    ts = np.array([float(r[2]) if r[2] != 'inf' else np.nan for r in rows])
    print(f"  T 场点数: {len(xs)}")

    # 重建规则网格
    x_unique = np.unique(xs)
    y_unique = np.unique(ys)
    nx, ny = len(x_unique), len(y_unique)
    T_grid = ts.reshape(ny, nx)
    X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
    print(f"  T 场网格: {nx} × {ny}, T 范围 [{np.nanmin(ts):.2f}, {np.nanmax(ts):.2f}]")

    # 数据范围
    data_x = (min(xs.min(), path[:, 0].min()), max(xs.max(), path[:, 0].max()))
    data_y = (min(ys.min(), path[:, 1].min()), max(ys.max(), path[:, 1].max()))

    fig, ax = plt.subplots(figsize=(12, 10))

    # 底色: DSM 地形 (低 alpha, T 场为主)
    im_terrain, bounds = draw_terrain_background(ax, elev, extent, (data_x, data_y))

    # 计算起点处的 T 值作为色标上限 (波前传播到起点的时间)
    # 找最近网格点
    ix_start = np.argmin(np.abs(x_unique - path[0, 0]))
    iy_start = np.argmin(np.abs(y_unique - path[0, 1]))
    t_at_start = T_grid[iy_start, ix_start]
    if not np.isfinite(t_at_start) or t_at_start <= 0:
        t_at_start = np.nanpercentile(ts[np.isfinite(ts)], 95)
    t_display_max = t_at_start * 1.05  # 略留余量
    print(f"  起点 T={t_at_start:.1f}, 色标上限={t_display_max:.1f}")

    # T 场填充等值线 (contourf: 绿-黄-红渐变, 近 goal 绿, 远 goal 红)
    levels = np.linspace(0, t_display_max, 20)
    # 将超出 t_display_max 的值截到 t_display_max, 避免大面积纯色
    T_display = np.where(np.isfinite(T_grid), np.minimum(T_grid, t_display_max), np.nan)
    contourf = ax.contourf(X_grid, Y_grid, T_display, levels=levels, cmap=FMM_CMAP,
                           alpha=0.85, zorder=3, extend='neither')
    # 叠加白色等值线 (每 4 条取 1 条, 标注 T 值)
    contour_lines = ax.contour(X_grid, Y_grid, T_display, levels=levels[::4],
                               colors='white', linewidths=0.6, alpha=0.5, zorder=4)
    ax.clabel(contour_lines, inline=True, fontsize=7, fmt='%.0f')

    # 最终路径
    draw_final_path(ax, path)
    # 起终点
    draw_start_goal(ax, path)

    finalize_axes(ax, bounds, 'FMM Search Process: Wavefront Propagation')

    # 色标: T 场
    cbar = fig.colorbar(contourf, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Arrival cost T (m)', fontsize=10)

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  [输出] {out_path}")


# -------------------------- A*: 搜索节点 + 边 --------------------------
def render_graph_search(dsm_path, nodes_csv, path_csv, out_path,
                        algorithm, title, dpi=300):
    """渲染格网搜索过程: 已发现节点、父子边、最终路径与地形背景。"""
    print(f"\n=== 渲染 {algorithm} 搜索过程 ===")
    elev, extent = read_geotiff(dsm_path)
    path = load_path_csv(path_csv)
    print(f"  路径点数: {len(path)}")

    # 加载搜索节点
    header, rows = load_csv(nodes_csv)
    nodes = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows], dtype=np.float64)
    parents = np.array([int(r[3]) for r in rows])
    print(f"  搜索节点数: {len(nodes)}")

    # 数据范围
    all_x = np.concatenate([nodes[:, 0], path[:, 0]])
    all_y = np.concatenate([nodes[:, 1], path[:, 1]])
    data_x = (all_x.min(), all_x.max())
    data_y = (all_y.min(), all_y.max())

    fig, ax = plt.subplots(figsize=(12, 10))

    # 底色: DSM 地形
    im_terrain, bounds = draw_terrain_background(ax, elev, extent, (data_x, data_y))

    # parent-child 边 (Okabe-Ito 蓝色, 灰底上高对比)
    edges = []
    for i in range(len(nodes)):
        p = parents[i]
        if p >= 0:
            edges.append([(nodes[i, 0], nodes[i, 1]), (nodes[p, 0], nodes[p, 1])])
    if edges:
        lc = LineCollection(edges, colors=OKABE_ITO['blue'], linewidths=0.5,
                            alpha=0.4, zorder=5)
        ax.add_collection(lc)
    print(f"  搜索边数: {len(edges)}")

    # 已发现节点散点 (按 Z 高度着色: 低=绿, 高=红, 绿红渐变)
    sc = ax.scatter(nodes[:, 0], nodes[:, 1], c=nodes[:, 2], cmap='RdYlGn_r',
                    s=4, alpha=0.7, zorder=10, label='Search nodes',
                    vmin=np.percentile(nodes[:, 2], 2),
                    vmax=np.percentile(nodes[:, 2], 98))

    # 最终路径
    draw_final_path(ax, path)
    # 起终点
    draw_start_goal(ax, path)

    finalize_axes(ax, bounds, title)

    # 色标: 节点高度
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Node altitude Z (m)', fontsize=10)

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  [输出] {out_path}")


# -------------------------- RRT*: 搜索树 --------------------------
def render_rrtstar(dsm_path, tree_csv, path_csv, out_path, dpi=300):
    """渲染 RRT* 搜索过程: 搜索树 + 地形底色 + 最终路径."""
    print(f"\n=== 渲染 RRT* 搜索过程 ===")
    elev, extent = read_geotiff(dsm_path)
    path = load_path_csv(path_csv)
    print(f"  路径点数: {len(path)}")

    # 加载搜索树
    header, rows = load_csv(tree_csv)
    nodes = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows], dtype=np.float64)
    parents = np.array([int(r[3]) for r in rows])
    costs = np.array([float(r[4]) for r in rows])
    print(f"  树节点数: {len(nodes)}")

    # 数据范围
    all_x = np.concatenate([nodes[:, 0], path[:, 0]])
    all_y = np.concatenate([nodes[:, 1], path[:, 1]])
    data_x = (all_x.min(), all_x.max())
    data_y = (all_y.min(), all_y.max())

    fig, ax = plt.subplots(figsize=(12, 10))

    # 底色: DSM 地形
    im_terrain, bounds = draw_terrain_background(ax, elev, extent, (data_x, data_y))

    # 搜索树边 (统一 Okabe-Ito 蓝色, 灰底上高对比)
    edges = []
    for i in range(len(nodes)):
        p = parents[i]
        if p >= 0:
            edges.append([(nodes[i, 0], nodes[i, 1]), (nodes[p, 0], nodes[p, 1])])
    if edges:
        lc = LineCollection(edges, colors=OKABE_ITO['blue'], linewidths=0.6,
                            alpha=0.45, zorder=5)
        ax.add_collection(lc)
    print(f"  树边数: {len(edges)}")

    # 高亮回溯路径 (白色底衬 + 红色线, 清晰可见)
    goal_node_idx = int(np.argmin(np.sum((nodes[:, :2] - path[-1, :2])**2, axis=1)))
    backtrace_x, backtrace_y = [], []
    idx = goal_node_idx
    while idx >= 0:
        backtrace_x.append(nodes[idx, 0])
        backtrace_y.append(nodes[idx, 1])
        idx = parents[idx]
    ax.plot(backtrace_x, backtrace_y, color='white', linewidth=3.5, zorder=11)
    ax.plot(backtrace_x, backtrace_y, color=OKABE_ITO['red'], linewidth=1.8,
            zorder=12, label='Backtracked tree path')
    print(f"  回溯路径节点数: {len(backtrace_x)}")

    # 树节点散点 (蓝色小点)
    ax.scatter(nodes[:, 0], nodes[:, 1], c=OKABE_ITO['blue'], s=4,
               alpha=0.5, zorder=8, label='Tree nodes')

    # 最终路径 (平滑后)
    draw_final_path(ax, path)
    # 起终点
    draw_start_goal(ax, path)

    finalize_axes(ax, bounds, 'RRT* Search Process: Random Tree Expansion')

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  [输出] {out_path}")


# -------------------------- 主函数 --------------------------
def main():
    parser = argparse.ArgumentParser(
        description='搜索过程可视化: FDGA* / FMM / A* / RRT* 四张论文插图',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=DATA_DIR, help=f'搜索数据目录 (默认: {DATA_DIR})')
    parser.add_argument('--dsm', default=DSM_PATH, help=f'DSM 文件 (默认: {DSM_PATH})')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help=f'输出目录 (默认: {OUTPUT_DIR})')
    parser.add_argument('--dpi', type=int, default=300, help='输出分辨率 (默认: 300)')
    parser.add_argument('--shrink', type=int, default=1,
                        help='DSM 底色降采样因子 (默认: 1, 大数据可设 2-4)')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    dsm_path = args.dsm
    if not os.path.exists(dsm_path):
        raise FileNotFoundError(dsm_path)

    # FDGA*
    nodes_fdga = os.path.join(data_dir, 'search_FGDASTAR_nodes.csv')
    path_fdga = os.path.join(data_dir, 'search_path_FGDASTAR.csv')
    if os.path.exists(nodes_fdga) and os.path.exists(path_fdga):
        render_graph_search(
            dsm_path, nodes_fdga, path_fdga,
            os.path.join(out_dir, 'search_process_FDGA.png'),
            'FDGA*', 'FDGA* Search Process: Direction-Aware Node Expansion',
            args.dpi,
        )
    else:
        print(f"[跳过] FDGA* 数据不存在: {nodes_fdga} 或 {path_fdga}")

    # FMM
    tfield_csv = os.path.join(data_dir, 'search_FMM_Tfield.csv')
    path_fmm = os.path.join(data_dir, 'search_path_FMM.csv')
    if os.path.exists(tfield_csv) and os.path.exists(path_fmm):
        render_fmm(dsm_path, tfield_csv, path_fmm,
                   os.path.join(out_dir, 'search_process_FMM.png'), args.dpi)
    else:
        print(f"[跳过] FMM 数据不存在: {tfield_csv} 或 {path_fmm}")

    # A*
    nodes_csv = os.path.join(data_dir, 'search_ASTAR_nodes.csv')
    path_astar = os.path.join(data_dir, 'search_path_ASTAR.csv')
    if os.path.exists(nodes_csv) and os.path.exists(path_astar):
        render_graph_search(
            dsm_path, nodes_csv, path_astar,
            os.path.join(out_dir, 'search_process_ASTAR.png'),
            'A*', 'A* Search Process: Discovered Nodes and Expansion Edges',
            args.dpi,
        )
    else:
        print(f"[跳过] A* 数据不存在: {nodes_csv} 或 {path_astar}")

    # RRT*
    tree_csv = os.path.join(data_dir, 'search_RRTSTAR_tree.csv')
    path_rrt = os.path.join(data_dir, 'search_path_RRTSTAR.csv')
    if os.path.exists(tree_csv) and os.path.exists(path_rrt):
        render_rrtstar(dsm_path, tree_csv, path_rrt,
                       os.path.join(out_dir, 'search_process_RRTSTAR.png'), args.dpi)
    else:
        print(f"[跳过] RRT* 数据不存在: {tree_csv} 或 {path_rrt}")

    print(f"\n渲染完成! 输出目录: {out_dir}")


if __name__ == '__main__':
    main()
