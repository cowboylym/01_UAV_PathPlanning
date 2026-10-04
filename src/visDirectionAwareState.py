#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""方向感知联合状态二维对比图。

分别输出：
(a) 原始 A*：同一栅格仅保留一个位置状态；
(b) FDGA*：同一栅格按入射方向维护多个独立状态 (c, q)。
"""

import argparse
import heapq
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch, Rectangle

OUTPUT_DIR = '../analysis'
COLORS = {
    'blue': '#0072B2',
    'orange': '#D55E00',
    'green': '#009E73',
    'red': '#C0392B',
    'gray': '#7A7A7A',
    'cell': '#EFEFEF',
    'cell_edge': '#C9C9C9',
    'yellow': '#F7E58B',
    'dark': '#222222',
}
BLUE_FILL = '#E8F1FA'
ORANGE_FILL = '#FBEDE4'

# (dz, dy, dx) 字典序枚举，q = 0..25
DIRECTIONS = [
    (dx, dy, dz)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]
Q_INDEX = {d: q for q, d in enumerate(DIRECTIONS)}


def configure_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 13,
        'axes.titlesize': 15,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


# ---------------------------------------------------------------- 通用图元

def draw_grid(axis, nx, ny, highlight=None):
    """绘制以整数坐标为中心的栅格平面。"""
    for ix in range(nx):
        for iy in range(ny):
            face = COLORS['yellow'] if highlight == (ix, iy) else COLORS['cell']
            axis.add_patch(Rectangle((ix - 0.5, iy - 0.5), 1, 1,
                                     facecolor=face, edgecolor=COLORS['cell_edge'],
                                     linewidth=0.7, zorder=1))


def path_line(axis, points, color, linestyle='-', linewidth=2.6, alpha=1.0,
              arrow=True, mutation_scale=15, zorder=6):
    points = np.asarray(points, dtype=float)
    axis.plot(points[:, 0], points[:, 1], color=color, linestyle=linestyle,
              linewidth=linewidth, alpha=alpha, zorder=zorder,
              solid_capstyle='round')
    if arrow and len(points) >= 2:
        p0, p1 = points[-2], points[-1]
        mid = p0 + 0.60 * (p1 - p0)
        axis.add_patch(FancyArrowPatch(
            mid, p1, arrowstyle='-|>', mutation_scale=mutation_scale,
            color=color, linewidth=linewidth, alpha=alpha, zorder=zorder))


def node(axis, xy, color, size=150, edgecolor='white', zorder=10):
    axis.scatter(xy[0], xy[1], s=size, color=color, edgecolor=edgecolor,
                 linewidth=1.2, zorder=zorder)


def turn_arc(axis, center, radius, a0, a1, color, label, label_radius=None):
    angles = np.linspace(np.deg2rad(a0), np.deg2rad(a1), 40)
    cx, cy = center
    axis.plot(cx + radius * np.cos(angles), cy + radius * np.sin(angles),
              color=color, linewidth=2.0, zorder=11)
    mid = np.deg2rad((a0 + a1) / 2)
    rr = label_radius or radius + 0.26
    axis.text(cx + rr * np.cos(mid), cy + rr * np.sin(mid), label,
              color=color, fontsize=9.5, fontweight='bold', ha='center',
              va='center', zorder=12)


def text_box(axis, xy, text, fontsize=9.5, color=COLORS['dark'],
             facecolor='white', edgecolor='#9A9A9A', pad=0.32, zorder=14):
    axis.text(xy[0], xy[1], text, ha='center', va='center', fontsize=fontsize,
              color=color, zorder=zorder,
              bbox=dict(boxstyle=f'round,pad={pad}', facecolor=facecolor,
                        edgecolor=edgecolor, linewidth=0.9))


def setup_plan_axis(axis):
    axis.set_xlim(-0.8, 5.8)
    axis.set_ylim(-1.05, 6.45)
    axis.set_aspect('equal')
    axis.set_axis_off()


# ----------------------------------------------------- (a) 传统 A* 标签合并

def draw_conventional_panel(axis):
    setup_plan_axis(axis)
    draw_grid(axis, 6, 7, highlight=(3, 2))

    # 蓝色入射分支：代价略大，到达 c 前被合并丢弃
    path_line(axis, [(0, 2), (1, 2), (2, 2)], COLORS['blue'], arrow=False)
    path_line(axis, [(2, 2), (2.5, 2)], COLORS['blue'], linestyle='--',
              linewidth=2.0, alpha=0.55, arrow=False)
    axis.plot([2.62, 2.92], [2, 2], color=COLORS['gray'], linestyle=':',
              linewidth=1.4, zorder=5)
    axis.scatter(2.58, 2, marker='x', s=110, color=COLORS['red'],
                 linewidth=2.4, zorder=13)
    axis.text(1.25, 2.55, r'$g_1>g_2$，入射方向 $q_1$', color=COLORS['blue'],
              fontsize=8.8, ha='center')
    axis.text(1.25, 1.42, r'若直行：$\theta_1=0^\circ$（转向更优）',
              color=COLORS['blue'], fontsize=8.3, ha='center', style='italic')

    # 橙色入射分支：唯一保留的标签
    path_line(axis, [(3, 0), (3, 1), (3, 2)], COLORS['orange'])
    axis.text(3.72, 1.05, r'仅保留 $g_2$（入射方向 $q_2$）',
              color=COLORS['orange'], fontsize=8.8, ha='left', va='center')

    # 唯一节点与后续航段
    node(axis, (3, 2), 'white', size=260, edgecolor=COLORS['dark'])
    axis.text(3, 2, r'$c$', ha='center', va='center', fontsize=11,
              fontweight='bold', zorder=12)
    path_line(axis, [(3.18, 2), (4, 2), (5, 2)], COLORS['green'])
    axis.text(4.55, 1.50, r'$\theta_1=0^\circ$ 的直行可能已丢失',
              color=COLORS['gray'], fontsize=8.2, ha='center')
    turn_arc(axis, (3, 2), 0.52, 0, 90, COLORS['red'], r'$\theta_2=90^\circ$',
             label_radius=0.86)

    text_box(axis, (2.5, 6.05),
             r'单一标签：$g(c)=\min\{g_1,g_2\}=g_2$', fontsize=10.5)
    text_box(axis, (2.5, -0.62),
             '位置相同即视为同一状态，更优后续转向的入射分支被提前舍弃',
             fontsize=8.8, facecolor='#FBF3F2', edgecolor=COLORS['red'])
    axis.set_title('(a) 传统 A*：同一栅格仅保留单一标签', pad=6)


# --------------------------------------------- (b) 方向感知联合状态“炸开”

def draw_state_card(axis, x0, y0, w, h, edge, fill, title, g_text):
    card = FancyBboxPatch((x0, y0), w, h,
                          boxstyle='round,pad=0.02,rounding_size=0.12',
                          facecolor=fill, edgecolor=edge, linewidth=1.8,
                          zorder=15)
    axis.add_patch(card)
    cx = x0 + w / 2
    axis.text(cx, y0 + h - 0.30, title, ha='center', va='center',
              fontsize=11, fontweight='bold', color=edge, zorder=16)
    axis.text(cx, y0 + 0.66, g_text, ha='center', va='center', fontsize=9,
              color=COLORS['dark'], zorder=16)
    axis.text(cx, y0 + 0.28, '独立累计代价 / 父状态指针', ha='center',
              va='center', fontsize=7.8, color=COLORS['gray'], zorder=16)


def draw_fdga_panel(axis):
    setup_plan_axis(axis)
    draw_grid(axis, 6, 7, highlight=(3, 2))

    # 两条入射路径到达同一栅格的两个不同状态
    path_line(axis, [(0, 2), (1, 2), (2, 2), (2.84, 2)], COLORS['blue'])
    path_line(axis, [(3, 0), (3, 1), (3.16, 2)], COLORS['orange'])

    # 起点：无方向标签
    axis.scatter(0, 2, marker='*', s=240, color='#E6B800',
                 edgecolor=COLORS['dark'], linewidth=0.8, zorder=12)
    text_box(axis, (0.95, 2.78),
             r'起点 $s_s=(c_s,q_\emptyset)$，$g=0$', fontsize=7.8,
             edgecolor='#BFA030', facecolor='#FFFBE6', pad=0.22)

    # 同位置两个状态节点（并排小点表示同格不同标签）
    node(axis, (2.84, 2), COLORS['blue'], size=120)
    node(axis, (3.16, 2), COLORS['orange'], size=120)

    # 各自扩展：蓝色直行 theta=0；橙色转 90°（虚线弯折到上方，代价高）
    path_line(axis, [(3.10, 2), (4, 2), (4.62, 2)], COLORS['green'],
              mutation_scale=13)
    axis.text(3.95, 1.55, r'$\theta_1=0^\circ$，继续直行',
              color=COLORS['green'], fontsize=8.3, ha='center')
    corner_xy = (3.16, 2.62)
    path_line(axis, [(3.16, 2.12), corner_xy], COLORS['orange'],
              linestyle='--', linewidth=1.8, arrow=False, alpha=0.9)
    path_line(axis, [corner_xy, (4.66, 2.62)], COLORS['orange'],
              linestyle='--', linewidth=1.8, alpha=0.9, mutation_scale=12)
    turn_arc(axis, (3.16, 2.12), 0.30, 0, 90, COLORS['red'], '',
             label_radius=0.5)
    axis.text(2.58, 2.86, r'$\theta_2=90^\circ$', color=COLORS['red'],
              fontsize=8.6, fontweight='bold', ha='center')
    axis.text(4.05, 3.16, '该候选仍可扩展，但曲率代价高',
              color=COLORS['orange'], fontsize=7.8, ha='center')

    # 状态卡片（概念上“炸”到栅格上方）
    draw_state_card(axis, 0.45, 4.25, 2.55, 1.40, COLORS['blue'], BLUE_FILL,
                    r'$s_1=(c,q_1)$', r'$g(c,q_1)=g_1$')
    draw_state_card(axis, 3.05, 4.25, 2.55, 1.40, COLORS['orange'], ORANGE_FILL,
                    r'$s_2=(c,q_2)$', r'$g(c,q_2)=g_2$')
    for card_cx, xyb, color in ((1.725, (2.84, 2.10), COLORS['blue']),
                                (4.325, (3.16, 2.10), COLORS['orange'])):
        axis.plot([card_cx, xyb[0]], [4.25, xyb[1]], color=color,
                  linestyle=':', linewidth=1.4, zorder=9)

    text_box(axis, (2.5, 6.05),
             r'联合状态 $s=(c,q)$，$g(c,q_1)\neq g(c,q_2),\ q_1\neq q_2$',
             fontsize=10)
    text_box(axis, (2.5, -0.62),
             '同格不同入射方向视为不同搜索状态，扩展时可显式计算转角与曲率代价',
             fontsize=8.8, facecolor='#F0F8F4', edgecolor=COLORS['green'])
    axis.set_title('(b) FDGA*：栅格—入射方向联合状态', pad=6)


# ------------------------------------------ (c) 26 个离散方向的九宫格编码

def direction_color(dx, dy, dz):
    norm_sq = dx * dx + dy * dy + dz * dz
    return {1: COLORS['green'], 2: COLORS['blue'], 3: COLORS['orange']}[norm_sq]


def draw_layer(axis, ox, dz):
    """绘制一张 3x3 九宫格，箭头由邻格指向中心 c（入射方向）。"""
    # 九宫格底框
    axis.add_patch(Rectangle((ox - 0.5, -0.5), 3, 3, facecolor='white',
                             edgecolor='#C4C4C4', linewidth=1.0, zorder=1))
    for ix in range(3):
        for iy in range(3):
            dx, dy = ix - 1, iy - 1
            x, y = ox + ix, iy
            if dx == 0 and dy == 0:
                if dz == 0:
                    face, edge = COLORS['yellow'], COLORS['dark']
                else:
                    face, edge = 'white', COLORS['green']
                axis.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                         facecolor=face, edgecolor=edge,
                                         linewidth=1.1, zorder=2))
                continue
            axis.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                     facecolor=COLORS['cell'],
                                     edgecolor=COLORS['cell_edge'],
                                     linewidth=0.7, zorder=2))

    # 入射箭头与 q 编号
    for (dx, dy, dzz), q in Q_INDEX.items():
        if dzz != dz:
            continue
        color = direction_color(dx, dy, dz)
        if dx == 0 and dy == 0:
            # 竖直面邻域：俯视图中用出/入平面符号表示（mathtext 保证字形可用）
            glyph = r'$\odot$' if dz == 1 else r'$\otimes$'  # ⊙ / ⊗
            axis.text(ox + 1, 1, glyph, ha='center', va='center',
                      fontsize=15, color=COLORS['green'], zorder=5)
            axis.text(ox + 0.64, 0.62, str(q), fontsize=6.6, color=color,
                      ha='center', va='center', zorder=6,
                      bbox=dict(boxstyle='round,pad=0.10', facecolor='white',
                                edgecolor='none', alpha=0.9))
            continue
        start = (ox + 1 + dx * 0.66, 1 + dy * 0.66)
        end = (ox + 1 + dx * 0.30, 1 + dy * 0.30)
        axis.add_patch(FancyArrowPatch(
            start, end, arrowstyle='-|>', mutation_scale=8,
            color=color, linewidth=1.5, zorder=4))
        axis.text(ox + 1 + dx * 0.82, 1 + dy * 0.82, str(q),
                  ha='center', va='center', fontsize=6.6, color=color,
                  zorder=5,
                  bbox=dict(boxstyle='round,pad=0.08', facecolor='white',
                            edgecolor='none', alpha=0.8))

    if dz == 0:
        axis.text(ox + 1, 1, r'$c$', ha='center', va='center',
                  fontsize=10, fontweight='bold', zorder=6)
    axis.text(ox + 1, 2.78, f'$z={dz:+d}$ 层' if dz else r'$z=0$ 层',
              ha='center', va='bottom', fontsize=10, fontweight='bold')


def draw_iso_cube(axis):
    """层叠立方体小插图：表示三张九宫格对应 3x3x3 邻域的三个 z 层。"""
    axis.set_axis_off()
    axis.set_aspect('equal')
    # 不透明白底，避免与后方九宫格线条重叠
    axis.add_patch(Rectangle((-1.35, -0.95), 2.9, 2.65, facecolor='white',
                             edgecolor='#CCCCCC', linewidth=0.8, zorder=0))
    ex = np.array([1.0, 0.30])
    ey = np.array([-1.0, 0.30])
    ez = np.array([0.0, 1.05])

    def corner(i, j, k):
        return i * ex + j * ey + k * ez

    for i in (0, 1):
        for j in (0, 1):
            axis.plot(*zip(corner(i, j, 0), corner(i, j, 1)),
                      color=COLORS['gray'], linewidth=1.1)
    for i in (0, 1):
        axis.plot(*zip(corner(i, 0, 0), corner(i, 1, 0)),
                  color=COLORS['gray'], linewidth=1.1)
        axis.plot(*zip(corner(i, 0, 1), corner(i, 1, 1)),
                  color=COLORS['gray'], linewidth=1.1)
    for j in (0, 1):
        axis.plot(*zip(corner(0, j, 0), corner(1, j, 0)),
                  color=COLORS['gray'], linewidth=1.1)
        axis.plot(*zip(corner(0, j, 1), corner(1, j, 1)),
                  color=COLORS['gray'], linewidth=1.1)
    # 可见面上的 z 分层线（k=1/3, 2/3）
    for kk in (1 / 3, 2 / 3):
        axis.plot(*zip(corner(0, 1, kk), corner(1, 1, kk)),
                  color=COLORS['blue'], linewidth=0.9, linestyle='--')
        axis.plot(*zip(corner(1, 0, kk), corner(1, 1, kk)),
                  color=COLORS['blue'], linewidth=0.9, linestyle='--')
    for k, label in ((5 / 6, '+1'), (1 / 2, '0'), (1 / 6, '-1')):
        p = corner(1, 0, k) + np.array([0.12, 0.0])
        axis.text(p[0], p[1], label, fontsize=7.5, ha='left', va='center',
                  color=COLORS['dark'])
    axis.text(0, -0.42, '三张九宫格\n即三个 z 层', fontsize=7.4, ha='center',
              va='top', color=COLORS['gray'], linespacing=1.25, clip_on=True)
    axis.set_xlim(-1.35, 1.55)
    axis.set_ylim(-0.95, 1.70)


def draw_neighborhood_panel(axis):
    axis.set_xlim(-1.5, 9.9)
    axis.set_ylim(-1.72, 3.35)
    axis.set_aspect('equal')
    axis.set_axis_off()

    for ox, dz in ((0.0, -1), (3.4, 0), (6.8, 1)):
        draw_layer(axis, ox, dz)

    inset = axis.inset_axes([0.005, 0.50, 0.17, 0.48])
    draw_iso_cube(inset)
    inset.patch.set_facecolor('white')
    inset.patch.set_alpha(0.94)
    inset.patch.set_edgecolor('#CCCCCC')
    inset.patch.set_linewidth(0.8)

    handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['green'],
               markeredgecolor=COLORS['green'], markersize=7,
               label='6 个面邻域（单位步长）'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['blue'],
               markeredgecolor=COLORS['blue'], markersize=7,
               label=r'12 个边邻域（$\sqrt{2}$ 步长）'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['orange'],
               markeredgecolor=COLORS['orange'], markersize=7,
               label=r'8 个角邻域（$\sqrt{3}$ 步长）'),
    ]
    axis.legend(handles=handles, loc='lower center',
                bbox_to_anchor=(0.08, 0.07, 0.88, 0.14), ncol=3,
                frameon=False, fontsize=8.4, handletextpad=0.35,
                columnspacing=1.1, mode='expand')
    axis.text(4.3, -1.52,
              r'箭头由父栅格指向当前栅格 $c$（入射方向）；共 26 个离散方向，'
              r'起点使用无方向标签 $q_\emptyset$（$\odot$ 指向观察者，$\otimes$ 背离观察者）',
              ha='center', va='center', fontsize=8.6, color=COLORS['dark'])
    axis.set_title('(c) 三维 26 邻域离散入射方向编码（按 z 层展开为九宫格）',
                   pad=6)


# ----------------------------------------------- 黑白原始 A* 栅格搜索状态图

def compute_astar_snapshot(width, height, start, goal, obstacles):
    """运行四邻域 A*，返回搜索标签、Open/Closed 集合和最终路径。"""
    def heuristic(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    open_heap = [(heuristic(start), 0, start)]
    open_set = {start}
    closed = set()
    parent = {}
    g_score = {start: 0}
    order = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current not in open_set:
            continue
        open_set.remove(current)
        closed.add(current)
        if current == goal:
            break

        x, y = current
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            nx, ny = neighbor
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if neighbor in obstacles or neighbor in closed:
                continue
            tentative_g = g_score[current] + 10
            if tentative_g >= g_score.get(neighbor, float('inf')):
                continue
            parent[neighbor] = current
            g_score[neighbor] = tentative_g
            order += 1
            heapq.heappush(open_heap,
                           (tentative_g + 10 * heuristic(neighbor), order, neighbor))
            open_set.add(neighbor)

    path = []
    if goal in g_score:
        cell = goal
        while True:
            path.append(cell)
            if cell == start:
                break
            cell = parent[cell]
        path.reverse()
    return g_score, open_set, closed, path


def draw_black_white_astar(axis):
    """参考经典 A* 示意图绘制纯黑白搜索状态。"""
    width, height = 10, 8
    start, goal = (2, 3), (8, 3)
    obstacles = {
        (4, 2), (4, 3), (4, 4), (4, 5),
        (5, 5), (6, 5), (6, 4), (6, 3),
    }
    g_score, open_set, closed, path = compute_astar_snapshot(
        width, height, start, goal, obstacles)

    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect('equal')
    axis.set_xticks(range(width + 1))
    axis.set_yticks(range(height + 1))
    axis.grid(color='black', linewidth=0.85)
    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if cell in obstacles:
                face = 'black'
            elif cell == start:
                face = 'white'
            elif cell == goal:
                face = '#B0B0B0'
            elif cell in closed:
                face = '#D0D0D0'
            elif cell in open_set:
                face = '#EEEEEE'
            else:
                face = 'white'
            axis.add_patch(Rectangle((x, y), 1, 1, facecolor=face,
                                     edgecolor='none', zorder=0))

            if cell in g_score and cell not in obstacles:
                g = g_score[cell]
                h = 10 * (abs(x - goal[0]) + abs(y - goal[1]))
                f = g + h
                axis.text(x + 0.50, y + 0.76, str(f), ha='center', va='center',
                          fontsize=8.3, color='black', zorder=4)
                axis.text(x + 0.15, y + 0.20, str(g), ha='left', va='center',
                          fontsize=6.8, color='black', zorder=4)
                axis.text(x + 0.85, y + 0.20, str(h), ha='right', va='center',
                          fontsize=6.8, color='black', zorder=4)

    if path:
        centers = np.array([(x + 0.5, y + 0.5) for x, y in path])
        axis.plot(centers[:, 0], centers[:, 1], color='black', linewidth=2.7,
                  solid_capstyle='round', solid_joinstyle='round', zorder=6)
        axis.scatter(centers[:, 0], centers[:, 1], s=22, facecolor='white',
                     edgecolor='black', linewidth=0.9, zorder=7)

    sx, sy = start
    gx, gy = goal
    axis.text(sx + 0.5, sy + 0.5, 'S', ha='center', va='center', fontsize=12,
              fontweight='bold', color='black', zorder=9,
              bbox=dict(boxstyle='circle,pad=0.18', facecolor='white',
                        edgecolor='black', linewidth=1.2))
    axis.text(gx + 0.5, gy + 0.5, 'G', ha='center', va='center', fontsize=12,
              fontweight='bold', color='black', zorder=9,
              bbox=dict(boxstyle='circle,pad=0.18', facecolor='white',
                        edgecolor='black', linewidth=1.2))

    legend_items = [
        Rectangle((0, 0), 1, 1, facecolor='black', edgecolor='black', label='障碍物'),
        Rectangle((0, 0), 1, 1, facecolor='#D0D0D0', edgecolor='black', label='Closed 集合'),
        Rectangle((0, 0), 1, 1, facecolor='#EEEEEE', edgecolor='black', label='Open 集合'),
        Line2D([0], [0], color='black', linewidth=2.7, marker='o',
               markerfacecolor='white', markeredgecolor='black', label='最终路径'),
    ]
    axis.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, -0.035),
                ncol=4, frameon=False, fontsize=8.5, columnspacing=1.2,
                handlelength=1.8)
    axis.set_title(
        '原始 A* 的栅格搜索状态\n'
        r'每个栅格仅维护一个状态标签：$f(c)=g(c)+h(c)$',
        fontsize=12.5, pad=12, linespacing=1.35)
    axis.text(0.08, height - 0.12,
              r'单元格标注：上方 $f$，左下 $g$，右下 $h$',
              ha='left', va='top', fontsize=7.8, color='#333333',
              bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.92),
              zorder=10)


# -------------------------------------- 彩色 8 邻域与方向感知 A* 搜索图

SCI_COLORS = {
    'obstacle': '#3F4650',
    'closed': '#173F5F',
    'open': '#3F6478',
    'path': '#D55E00',
    'start': '#009E73',
    'goal': '#CC79A7',
    'direction': '#0072B2',
    'direction_alt': '#7B61A8',
    'grid': '#7F8C8D',
}
NEIGHBORS_8 = [
    (-1, -1), (0, -1), (1, -1), (-1, 0),
    (1, 0), (-1, 1), (0, 1), (1, 1),
]


def octile_heuristic(cell, goal):
    dx, dy = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
    return 10 * (dx + dy) + (14 - 20) * min(dx, dy)


def is_valid_step(cell, direction, width, height, obstacles):
    """检查 8 邻域移动，并禁止对角穿越障碍角点。"""
    dx, dy = direction
    neighbor = (cell[0] + dx, cell[1] + dy)
    if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
        return False
    if neighbor in obstacles:
        return False
    if dx and dy:
        if (cell[0] + dx, cell[1]) in obstacles or (cell[0], cell[1] + dy) in obstacles:
            return False
    return True


def compute_astar_8(width, height, start, goal, obstacles):
    """运行 8 邻域 A*，使用 octile 启发且禁止对角切角。"""
    heap = [(octile_heuristic(start, goal), 0, start)]
    open_set, closed = {start}, set()
    parent, g_score = {}, {start: 0}
    order = 0
    while heap:
        _, _, current = heapq.heappop(heap)
        if current not in open_set:
            continue
        open_set.remove(current)
        closed.add(current)
        if current == goal:
            break
        for dx, dy in NEIGHBORS_8:
            neighbor = (current[0] + dx, current[1] + dy)
            if not is_valid_step(current, (dx, dy), width, height, obstacles):
                continue
            if neighbor in closed:
                continue
            step = 14 if dx and dy else 10
            tentative = g_score[current] + step
            if tentative >= g_score.get(neighbor, float('inf')):
                continue
            parent[neighbor], g_score[neighbor] = current, tentative
            order += 1
            heapq.heappush(heap, (tentative + octile_heuristic(neighbor, goal),
                                  order, neighbor))
            open_set.add(neighbor)
    path = []
    if goal in g_score:
        cell = goal
        while True:
            path.append(cell)
            if cell == start:
                break
            cell = parent[cell]
        path.reverse()
    return g_score, open_set, closed, path


def compute_direction_aware_astar(width, height, start, goal, obstacles,
                                  arrival_time, clearance, allowed_cells=None):
    """以 FMM 到达代价为启发运行方向感知联合状态 A*。"""
    start_state = (start, None)
    heap = [(arrival_time[start[1], start[0]], 0, start_state)]
    open_states, closed_states = {start_state}, set()
    parent, g_score = {}, {start_state: 0.0}
    order = 0
    goal_state = None
    while heap:
        _, _, state = heapq.heappop(heap)
        if state not in open_states:
            continue
        open_states.remove(state)
        closed_states.add(state)
        cell, incoming = state
        if cell == goal:
            goal_state = state
            break
        for direction in NEIGHBORS_8:
            dx, dy = direction
            neighbor = (cell[0] + dx, cell[1] + dy)
            if not is_valid_step(cell, direction, width, height, obstacles):
                continue
            if allowed_cells is not None and neighbor not in allowed_cells:
                continue
            length_cost = np.hypot(dx, dy)
            curvature_cost = 0.0
            if incoming is not None:
                previous_length = np.hypot(*incoming)
                angle = np.arccos(np.clip(
                    (incoming[0] * dx + incoming[1] * dy) /
                    (previous_length * length_cost), -1.0, 1.0))
                if angle > np.deg2rad(60) + 1e-9:
                    continue
                support_length = 0.5 * (previous_length + length_cost)
                curvature_cost = angle ** 2 / support_length
            risk_cost = length_cost * 0.5 * (
                1.0 / max(clearance[cell[1], cell[0]], 0.1) +
                1.0 / max(clearance[neighbor[1], neighbor[0]], 0.1))
            next_state = (neighbor, direction)
            tentative = (g_score[state] + length_cost +
                         0.2 * risk_cost + 0.1 * curvature_cost)
            if tentative >= g_score.get(next_state, float('inf')):
                continue
            parent[next_state], g_score[next_state] = state, tentative
            order += 1
            priority = tentative + arrival_time[neighbor[1], neighbor[0]]
            heapq.heappush(heap, (priority, order, next_state))
            open_states.add(next_state)

    path_states = []
    if goal_state is not None:
        state = goal_state
        while True:
            path_states.append(state)
            if state == start_state:
                break
            state = parent[state]
        path_states.reverse()
    return g_score, open_states, closed_states, path_states


def astar_map_definition():
    """两幅彩色图共用的开放式凹形障碍与水平死胡同地图。"""
    width, height = 22, 12

    # 用细墙构造水平死胡同，避免底部出现大面积连成一片的障碍。
    obstacles = {(x, 5) for x in range(5, 16)}
    obstacles.update({(x, 3) for x in range(5, 16)})
    obstacles.add((15, 4))

    # 起点两侧的短墙使公共路径以向上的入射方向到达 p3。
    obstacles.update({(3, y) for y in range(0, 4)})
    obstacles.update({(5, y) for y in range(0, 3)})
    return width, height, (4, 1), (20, 4), obstacles


def compute_fmm_arrival_field(width, height, goal, obstacles):
    """用 ESDF 净空调制的速度场计算二维 FMM 到达时间示意场。"""
    obstacle_points = np.asarray(list(obstacles), dtype=float)
    clearance = np.zeros((height, width), dtype=float)
    for y in range(height):
        for x in range(width):
            if (x, y) in obstacles:
                continue
            clearance[y, x] = np.min(np.linalg.norm(
                obstacle_points - np.array((x, y), dtype=float), axis=1))

    speed = np.zeros_like(clearance)
    free_mask = clearance > 0
    speed[free_mask] = 1.0 / (1.0 + 0.2 / clearance[free_mask])

    arrival = np.full((height, width), np.inf, dtype=float)
    arrival[goal[1], goal[0]] = 0.0
    heap = [(0.0, goal)]
    while heap:
        current_time, cell = heapq.heappop(heap)
        if current_time > arrival[cell[1], cell[0]]:
            continue
        for dx, dy in NEIGHBORS_8:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if not is_valid_step(cell, (dx, dy), width, height, obstacles):
                continue
            mean_speed = 0.5 * (speed[cell[1], cell[0]] +
                                speed[neighbor[1], neighbor[0]])
            travel_time = np.hypot(dx, dy) / max(mean_speed, 1e-6)
            candidate = current_time + travel_time
            if candidate < arrival[neighbor[1], neighbor[0]]:
                arrival[neighbor[1], neighbor[0]] = candidate
                heapq.heappush(heap, (candidate, neighbor))
    obstacle_y, obstacle_x = zip(*[(y, x) for x, y in obstacles])
    arrival[np.asarray(obstacle_y), np.asarray(obstacle_x)] = np.nan
    return arrival, clearance


def build_fmm_guidance_corridor(width, height, start, goal, obstacles):
    """定义 FMM 初始引导路径及其局部搜索走廊。"""
    guide_cells = [
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
        (5, 7), (6, 8),
        *[(x, 8) for x in range(7, 17)],
        (17, 7), (18, 6), (19, 5), (20, 4),
    ]
    guide_cells = [cell for cell in guide_cells
                   if 0 <= cell[0] < width and 0 <= cell[1] < height
                   and cell not in obstacles]
    corridor = set()
    radius = 1
    for x, y in guide_cells:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                cell = (x + dx, y + dy)
                if (0 <= cell[0] < width and 0 <= cell[1] < height
                        and cell not in obstacles):
                    corridor.add(cell)
    corridor.update({start, goal})
    return guide_cells, corridor


def draw_fmm_background(axis, arrival, width, height):
    """叠加 FMM 到达时间色场，不绘制等值线。"""
    finite = arrival[np.isfinite(arrival)]
    upper = float(np.percentile(finite, 95))
    field = np.ma.masked_invalid(np.clip(arrival, 0.0, upper))
    image = axis.imshow(field, origin='lower', extent=(0, width, 0, height),
                        cmap='YlGnBu', alpha=0.62, interpolation='nearest',
                        zorder=-3)
    return image


def setup_color_grid(axis, width, height):
    axis.set_xlim(0, width)
    axis.set_ylim(0, height)
    axis.set_aspect('equal')
    axis.set_xticks(range(width + 1))
    axis.set_yticks(range(height + 1))
    axis.grid(color=SCI_COLORS['grid'], linewidth=0.7, alpha=0.8)
    axis.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in axis.spines.values():
        spine.set_color(SCI_COLORS['grid'])
        spine.set_linewidth(1.0)


def draw_color_cells(axis, width, height, obstacles, open_cells, closed_cells,
                     reveal_fmm=False):
    for x in range(width):
        for y in range(height):
            cell = (x, y)
            if cell in obstacles:
                axis.add_patch(Rectangle(
                    (x, y), 1, 1, facecolor=SCI_COLORS['obstacle'],
                    edgecolor='none', zorder=2))
            elif cell in closed_cells:
                axis.add_patch(Rectangle(
                    (x + 0.045, y + 0.045), 0.91, 0.91,
                    facecolor='none', edgecolor=SCI_COLORS['closed'],
                    linewidth=2.0, linestyle='-', zorder=3))
            elif cell in open_cells:
                axis.add_patch(Rectangle(
                    (x + 0.045, y + 0.045), 0.91, 0.91,
                    facecolor='none', edgecolor=SCI_COLORS['open'],
                    linewidth=2.0, linestyle=(0, (3, 2)), zorder=3))


def draw_start_goal(axis, start, goal):
    for cell, label, color in ((start, 'S', SCI_COLORS['start']),
                               (goal, 'G', SCI_COLORS['goal'])):
        x, y = cell
        axis.scatter(x + 0.5, y + 0.5, s=245, facecolor=color,
                     edgecolor='white', linewidth=1.7, zorder=9)
        axis.text(x + 0.5, y + 0.5, label, ha='center', va='center',
                  fontsize=13, fontweight='bold', color='white', zorder=10)


def label_path_points(axis, path, offsets=None, labels=None):
    """依次标注最终路径节点 p_0, p_1, ...，必要时允许替换分叉标签。"""
    offsets = offsets or {}
    labels = labels or {}
    for index, (x, y) in enumerate(path):
        dx, dy = offsets.get(index, (0.10, 0.15))
        label = labels.get(index, rf'$p_{{{index}}}$')
        axis.text(x + 0.5 + dx, y + 0.5 + dy, label,
                  ha='left' if dx >= 0 else 'right',
                  va='bottom' if dy >= 0 else 'top',
                  fontsize=10.2, fontweight='bold', color='#25313C',
                  bbox=dict(facecolor='white', edgecolor='none',
                            alpha=0.78, pad=0.5),
                  zorder=11)


def draw_turn_angle_vectors(axis, path, vertex_index):
    """在 p_i 处用共起点矢量表示 v_{i-1} 与 v_i，并标注 theta_i。"""
    previous = np.array(path[vertex_index - 1], dtype=float)
    vertex = np.array(path[vertex_index], dtype=float)
    following = np.array(path[vertex_index + 1], dtype=float)
    incoming = vertex - previous
    outgoing = following - vertex
    origin = vertex + 0.5
    scale = 0.72
    color_previous = SCI_COLORS['direction_alt']
    color_current = SCI_COLORS['direction']
    for vector, color, label, shift in (
            (incoming, color_previous, r'$\mathbf{v}_{i-1}$', (-0.03, 0.08)),
            (outgoing, color_current, r'$\mathbf{v}_{i}$', (0.03, 0.08))):
        end = origin + scale * vector / np.linalg.norm(vector)
        axis.add_patch(FancyArrowPatch(origin, end, arrowstyle='-|>',
                                       mutation_scale=11, linewidth=1.8,
                                       color=color, zorder=13))
        axis.text(end[0] + shift[0], end[1] + shift[1], label,
                  fontsize=8.2, color=color, ha='center', va='bottom',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.4),
                  zorder=14)
    angle_start = np.degrees(np.arctan2(incoming[1], incoming[0]))
    angle_end = np.degrees(np.arctan2(outgoing[1], outgoing[0]))
    while angle_end < angle_start:
        angle_end += 360
    if angle_end - angle_start > 180:
        angle_start, angle_end = angle_end, angle_start + 360
    radius = 0.44
    axis.add_patch(Arc(origin, 2 * radius, 2 * radius,
                       theta1=angle_start, theta2=angle_end,
                       linewidth=1.4, color='#25313C', zorder=13))
    mid_angle = np.deg2rad(0.5 * (angle_start + angle_end))
    axis.text(origin[0] + 0.56 * np.cos(mid_angle),
              origin[1] + 0.56 * np.sin(mid_angle),
              r'$\theta_i$', fontsize=8.3, color='#25313C',
              ha='center', va='center', zorder=14,
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.3))


def draw_color_legend(axis, direction_aware=False):
    closed_handle = Rectangle(
        (0, 0), 1, 1, facecolor='none', edgecolor=SCI_COLORS['closed'],
        linewidth=2.0, label='Closed set (outline)')
    open_handle = Rectangle(
        (0, 0), 1, 1, facecolor='none', edgecolor=SCI_COLORS['open'],
        linewidth=2.0, linestyle=(0, (3, 2)), label='Open set (outline)')
    items = [
        Rectangle((0, 0), 1, 1, facecolor=SCI_COLORS['obstacle'],
                  edgecolor='none', label='Obstacle'),
        closed_handle,
        open_handle,
        Line2D([0], [0], color=SCI_COLORS['path'], linewidth=2.7,
               marker='o', markerfacecolor='white', label='Final path'),
    ]
    if direction_aware:
        items.extend([
            Line2D([0, 1], [0, 0], color='#1E6F9F', linewidth=3.0,
                   linestyle=(0, (7, 3)), marker='>', markersize=7,
                   markevery=[1], label='FMM initial guidance path'),
            Rectangle((0, 0), 1, 1, facecolor='#74ADD1', alpha=0.22,
                      edgecolor='#2A6F97', linewidth=1.4,
                      linestyle=(0, (5, 3)), label='FMM guidance corridor'),
        ])
    axis.legend(handles=items, loc='upper center', bbox_to_anchor=(0.5, -0.055),
                ncol=3 if direction_aware else len(items), frameon=False,
                fontsize=12.0, columnspacing=1.15, handlelength=2.2,
                labelspacing=0.8)


def draw_color_astar_8(axis):
    width, height, start, goal, obstacles = astar_map_definition()
    g_score, open_cells, closed_cells, path = compute_astar_8(
        width, height, start, goal, obstacles)
    setup_color_grid(axis, width, height)
    axis.set_xlim(-7.0, width)
    axis.add_patch(Rectangle(
        (-7.0, 0), 7.0, height, facecolor='white', edgecolor='#B8C2C9',
        linewidth=1.0, zorder=5, clip_on=False))
    draw_color_cells(axis, width, height, obstacles, open_cells, closed_cells)

    key_cells = {start, goal, (4, 6), (5, 6), (5, 7), (8, 4), (12, 4), (14, 4)}
    key_cells.update(path[::4])
    for cell in sorted(key_cells):
        if cell not in g_score or cell in obstacles:
            continue
        g = g_score[cell]
        h = octile_heuristic(cell, goal)
        x, y = cell
        axis.text(x + 0.50, y + 0.72, f'{g + h}', ha='center', va='center',
                  fontsize=8.8, color='#25313C', zorder=4)
        axis.text(x + 0.12, y + 0.16, f'{g}', ha='left', va='center',
                  fontsize=7.1, color='#40505E', zorder=4)
        axis.text(x + 0.88, y + 0.16, f'{h}', ha='right', va='center',
                  fontsize=7.1, color='#40505E', zorder=4)

    centers = np.array([(x + 0.5, y + 0.5) for x, y in path])
    axis.plot(centers[:, 0], centers[:, 1], color=SCI_COLORS['path'],
              linewidth=3.0, zorder=6, solid_joinstyle='round')
    axis.scatter(centers[:, 0], centers[:, 1], s=22, facecolor='white',
                 edgecolor=SCI_COLORS['path'], linewidth=1.0, zorder=7)

    # 两种算法在公共节点 p3 后首次产生路径分歧。
    p3, p4a, p4d = (4, 6), (5, 6), (5, 7)
    for cell, label, color, offset in (
            (p3, r'$p_3$', '#25313C', (-0.34, -0.05)),
            (p4a, r'$p_4^{\mathrm{A}}$', '#C44E21', (0.18, -0.42)),
            (p4d, r'$p_4^{\mathrm{D}}$', '#456A83', (0.28, 0.34))):
        x, y = cell
        axis.scatter(x + 0.5, y + 0.5, s=78, facecolor='white',
                     edgecolor=color, linewidth=1.8, zorder=10)
        axis.text(x + 0.5 + offset[0], y + 0.5 + offset[1], label,
                  fontsize=12.5, fontweight='bold', color=color, zorder=12,
                  ha='center', va='center',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.86, pad=0.3))

    fork = np.array(p3, dtype=float) + 0.5
    axis.add_patch(FancyArrowPatch(
        fork, np.array(p4a, dtype=float) + 0.5, arrowstyle='-|>',
        mutation_scale=15, linewidth=2.4, color='#C44E21', zorder=11))
    axis.add_patch(FancyArrowPatch(
        fork, np.array(p4d, dtype=float) + 0.5, arrowstyle='-|>',
        mutation_scale=13, linewidth=1.7, linestyle='--', color='#607D8B', zorder=11))
    axis.annotate('Misleading dead-end\nexpansion', xy=(6.2, 4.55),
                  xytext=(-3.5, 4.35), fontsize=12.5, color='#A6421C',
                  ha='center', va='center',
                  arrowprops=dict(arrowstyle='->', color='#C44E21',
                                  linewidth=1.7),
                  bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                            edgecolor='#C44E21', alpha=0.96), zorder=13)
    axis.text(15.45, 4.5, 'sealed end', rotation=90, fontsize=11.0,
              color='#F3F3F3', ha='center', va='center', zorder=5)

    axis.text(-3.5, 2.0,
              r'Cell values' + '\n' +
              r'top: $f=g+h_{oct}$' + '\n' +
              r'lower-left: $g$' + '\n' +
              r'lower-right: $h_{oct}$',
              fontsize=11.5, color='#40505E', ha='center', va='center',
              linespacing=1.35,
              bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                        edgecolor='#78909C', alpha=0.98), zorder=13)

    # 放大两种算法在 p3 后产生分歧的关键局部。
    inset = axis.inset_axes([0.018, 0.44, 0.255, 0.47])
    inset.set_facecolor('white')
    inset.set_xlim(3.65, 6.35)
    inset.set_ylim(5.65, 8.35)
    inset.set_aspect('equal')
    inset.set_xticks(np.arange(4, 7))
    inset.set_yticks(np.arange(6, 9))
    inset.grid(color=SCI_COLORS['grid'], linewidth=0.8, alpha=0.9)
    inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    zoom_color = '#8E44AD'
    for spine in inset.spines.values():
        spine.set_color(zoom_color)
        spine.set_linewidth(2.4)
    for cell in ((4, 6), (5, 6), (5, 7)):
        x, y = cell
        inset.add_patch(Rectangle(
            (x + 0.045, y + 0.045), 0.91, 0.91,
            facecolor='none', edgecolor=SCI_COLORS['closed'],
            linewidth=2.0, zorder=3))
        g = g_score[cell]
        h = octile_heuristic(cell, goal)
        inset.text(x + 0.5, y + 0.72, f'{g + h}', ha='center', va='center',
                   fontsize=10.0, color='#25313C', zorder=4)
        inset.text(x + 0.12, y + 0.16, f'{g}', ha='left', va='center',
                   fontsize=8.0, color='#40505E', zorder=4)
        inset.text(x + 0.88, y + 0.16, f'{h}', ha='right', va='center',
                   fontsize=8.0, color='#40505E', zorder=4)
    inset.add_patch(FancyArrowPatch(
        fork, np.array(p4a, dtype=float) + 0.5, arrowstyle='-|>',
        mutation_scale=17, linewidth=2.8, color='#C44E21', zorder=7))
    inset.add_patch(FancyArrowPatch(
        fork, np.array(p4d, dtype=float) + 0.5, arrowstyle='-|>',
        mutation_scale=15, linewidth=2.0, linestyle='--',
        color='#607D8B', zorder=7))
    for cell, label, color, offset in (
            (p3, r'$p_3$', '#25313C', (-0.30, -0.12)),
            (p4a, r'$p_4^{\mathrm{A}}$', '#C44E21', (0.12, -0.30)),
            (p4d, r'$p_4^{\mathrm{D}}$', '#456A83', (0.18, 0.28))):
        x, y = cell
        inset.scatter(x + 0.5, y + 0.5, s=95, facecolor='white',
                      edgecolor=color, linewidth=2.0, zorder=8)
        inset.text(x + 0.5 + offset[0], y + 0.5 + offset[1], label,
                   fontsize=13.5, fontweight='bold', color=color,
                   ha='center', va='center', zorder=9,
                   bbox=dict(facecolor='white', edgecolor='none',
                             alpha=0.82, pad=0.2))
    inset.text(5.5, 8.08, r'Discarded $p_4^{\mathrm{D}}$',
               fontsize=10.5, fontweight='bold', color='#456A83',
               ha='center', va='center', zorder=10,
               bbox=dict(facecolor='white', edgecolor='#456A83',
                         boxstyle='round,pad=0.18', alpha=0.96))
    inset.set_title('Local decision at $p_3$', fontsize=12.0,
                    color='#25313C', pad=4)

    # 用醒目边框圈出主图中被局部放大的关键区域。
    axis.add_patch(Rectangle(
        (3.72, 5.72), 2.56, 2.56, facecolor='none',
        edgecolor=zoom_color, linewidth=2.6, linestyle=(0, (5, 2)),
        zorder=14))
    # 两条引导线从主图高亮区域展开至左侧局部放大图。
    axis.add_patch(FancyArrowPatch(
        (3.72, 8.28), (-0.55, 9.8), arrowstyle='-',
        linewidth=2.0, color=zoom_color, zorder=12,
        connectionstyle='arc3,rad=0', clip_on=False))
    axis.add_patch(FancyArrowPatch(
        (3.72, 5.72), (-0.55, 5.95), arrowstyle='-',
        linewidth=2.0, color=zoom_color, zorder=12,
        connectionstyle='arc3,rad=0', clip_on=False))

    draw_start_goal(axis, start, goal)
    draw_color_legend(axis)


def draw_direction_state_arrows(axis, state_groups, path_cells, fork_cell):
    """仅在 p3 八邻域的少量格子中展示共起点方向状态矢量。"""
    candidates = sorted(
        ((cell, dirs) for cell, dirs in state_groups.items()
         if len(dirs) >= 2
         and cell not in path_cells
         and max(abs(cell[0] - fork_cell[0]),
                 abs(cell[1] - fork_cell[1])) == 1),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]))[:3]
    for cell, directions in candidates:
        x, y = cell
        origin = (x + 0.5, y + 0.5)
        for idx, (dx, dy) in enumerate(directions[:3]):
            color = SCI_COLORS['direction'] if idx == 0 else SCI_COLORS['direction_alt']
            length = 0.34 / np.hypot(dx, dy)
            end = (origin[0] + length * dx, origin[1] + length * dy)
            axis.add_patch(FancyArrowPatch(origin, end, arrowstyle='-|>',
                                           mutation_scale=8, linewidth=1.25,
                                           color=color, zorder=5))
    return [cell for cell, _ in candidates]


def draw_color_direction_aware_astar(axis):
    width, height, start, goal, obstacles = astar_map_definition()
    g_score, open_states, closed_states, path_states = compute_direction_aware_astar(
        width, height, start, goal, obstacles)
    open_cells = {state[0] for state in open_states}
    closed_cells = {state[0] for state in closed_states}
    arrival_time, clearance = compute_fmm_arrival_field(width, height, goal, obstacles)
    setup_color_grid(axis, width, height)
    axis.set_xlim(0, 14.2)
    fmm_image = draw_fmm_background(axis, arrival_time, width, height)
    colorbar_axis = axis.inset_axes([0.735, 0.925, 0.235, 0.032])
    colorbar = axis.figure.colorbar(
        fmm_image, cax=colorbar_axis, orientation='horizontal')
    colorbar.set_label(r'FMM arrival cost $T_{\mathrm{FMM}}(c)$',
                       fontsize=13.0, labelpad=4)
    colorbar.ax.tick_params(labelsize=11.5, length=3)
    colorbar.outline.set_linewidth(0.8)
    draw_color_cells(axis, width, height, obstacles, open_cells, closed_cells,
                     reveal_fmm=True)

    # 每个栅格只显示当前最优方向状态 q* 的标签，完整方向标签仍保存在状态空间中。
    states_by_cell = {}
    for state, g_value in g_score.items():
        states_by_cell.setdefault(state[0], []).append((state, g_value))
    best_state_by_cell = {
        cell: min(states, key=lambda item: item[1])
        for cell, states in states_by_cell.items()
    }
    for cell, ((_, direction), _) in best_state_by_cell.items():
        if cell in obstacles:
            continue
        x, y = cell
        g_value = min(value for (state, value) in states_by_cell[cell])
        h_value = arrival_time[y, x]
        f_value = g_value + h_value
        axis.text(x + 0.50, y + 0.76, f'{f_value:.1f}',
                  ha='center', va='center', fontsize=11.0,
                  color='#25313C', zorder=6)
        axis.text(x + 0.12, y + 0.16, f'{g_value:.1f}',
                  ha='left', va='center', fontsize=9.0,
                  color='#40505E', zorder=6)
        axis.text(x + 0.88, y + 0.16, f'{h_value:.1f}',
                  ha='right', va='center', fontsize=9.0,
                  color='#40505E', zorder=6)
        if direction is None:
            state_label = rf'$(({x},{y}),q_{{\varnothing}})$'
        else:
            q_index = NEIGHBORS_8.index(direction)
            state_label = rf'$(({x},{y}),q_{{{q_index}}})$'
        axis.text(x + 0.50, y + 0.43, state_label,
                  ha='center', va='center', fontsize=9.2,
                  color=SCI_COLORS['direction'], zorder=6)

    centers = np.array([(cell[0] + 0.5, cell[1] + 0.5)
                        for cell, _ in path_states])
    axis.plot(centers[:, 0], centers[:, 1], color=SCI_COLORS['path'],
              linewidth=2.8, zorder=7, solid_joinstyle='round')
    axis.scatter(centers[:, 0], centers[:, 1], s=25, facecolor='white',
                 edgecolor=SCI_COLORS['path'], linewidth=1.0, zorder=8)

    for (cell, direction) in path_states[1:]:
        x, y = cell
        dx, dy = direction
        axis.add_patch(FancyArrowPatch(
            (x + 0.5 - 0.18 * dx, y + 0.5 - 0.18 * dy),
            (x + 0.5 + 0.17 * dx, y + 0.5 + 0.17 * dy),
            arrowstyle='-|>', mutation_scale=8, color=SCI_COLORS['path'],
            linewidth=1.0, zorder=9))

    direction_path = [state[0] for state in path_states]
    label_path_points(axis, direction_path, {
        0: (-0.16, 0.18), 1: (0.08, -0.25), 2: (-0.12, 0.16),
        3: (-0.18, -0.23), 4: (-0.16, 0.15), 5: (0.08, 0.14),
        6: (0.08, 0.14), 7: (0.09, -0.25), 8: (0.10, 0.15),
    }, labels={4: r'$p_4^{\mathrm{D}}$'})

    # 公共分叉点 p3：三条矢量从同一点出发，直接对比两个候选决策。
    fork = np.array(direction_path[3], dtype=float) + 0.5
    incoming = np.array((1.0, 1.0)) / np.sqrt(2.0)
    accepted = incoming
    rejected = np.array((1.0, -1.0)) / np.sqrt(2.0)
    for vector, length, color, label, label_offset in (
            (incoming, 0.53, SCI_COLORS['direction_alt'],
             r'$\mathbf{v}_{i-1}$', (-0.13, 0.03)),
            (accepted, 0.82, SCI_COLORS['start'],
             r'$\mathbf{v}_i^{\mathrm{D}}$', (0.08, 0.02)),
            (rejected, 0.82, '#C0392B',
             r'$\mathbf{v}_i^{\mathrm{A}}$', (0.10, -0.10))):
        end = fork + length * vector
        axis.add_patch(FancyArrowPatch(fork, end, arrowstyle='-|>',
                                       mutation_scale=12, linewidth=1.8,
                                       linestyle='--' if color == '#C0392B' else '-',
                                       color=color, zorder=13))
        axis.text(end[0] + label_offset[0], end[1] + label_offset[1], label,
                  fontsize=11.0, color=color, ha='center', va='center',
                  bbox=dict(facecolor='white', edgecolor='none', alpha=0.86, pad=0.3),
                  zorder=14)
    # 标示入射方向与被拒绝候选方向之间的 90° 转角。
    axis.add_patch(Arc(fork, 0.76, 0.76, theta1=-45, theta2=45,
                       linewidth=1.3, color='#C0392B', zorder=13))
    axis.text(fork[0] + 0.50, fork[1], r'$\theta_i=90^\circ$',
              fontsize=10.5, color='#C0392B', ha='center', va='center',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.86, pad=0.2),
              zorder=14)
    axis.text(fork[0] + 0.42, fork[1] + 0.47, r'$\theta_i=0^\circ$',
              fontsize=10.5, color=SCI_COLORS['start'], ha='center', va='center',
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.86, pad=0.2),
              zorder=14)
    rejected_cell = (5, 4)
    rejected_point = np.array(rejected_cell, dtype=float) + 0.5
    # 为便于比较，继续计算该不可行候选的曲率代价；它仍不会写入 Open 集合。
    fork_state = path_states[3]
    incoming = (np.array(path_states[3][0], dtype=float) -
                np.array(path_states[2][0], dtype=float))
    outgoing = (np.array(rejected_cell, dtype=float) -
                np.array(path_states[3][0], dtype=float))
    theta_trial = np.arccos(np.clip(
        np.dot(incoming, outgoing) /
        (np.linalg.norm(incoming) * np.linalg.norm(outgoing)), -1.0, 1.0))
    support_length = 0.5 * (np.linalg.norm(incoming) + np.linalg.norm(outgoing))
    curvature_trial = theta_trial ** 2 / support_length
    # 按图中的 g 与 FMM 启发口径计算该候选的假设 f，但它不会写入 Open 集合。
    trial_length = 14.0
    trial_risk = trial_length * 0.5 * (
        1.0 / max(clearance[fork_state[0][1], fork_state[0][0]], 0.1) +
        1.0 / max(clearance[rejected_cell[1], rejected_cell[0]], 0.1))
    g_trial = (g_score[fork_state] + trial_length +
               0.2 * trial_risk + 0.1 * curvature_trial)
    h_trial = arrival_time[rejected_cell[1], rejected_cell[0]]
    f_trial = g_trial + h_trial
    # 与其他格网保持相同布局：顶部 f，左下 g，右下 h。
    axis.text(rejected_cell[0] + 0.50, rejected_cell[1] + 0.78,
              f'{f_trial:.1f}', fontsize=10.0, color='#C0392B',
              ha='center', va='center', zorder=15)
    axis.text(rejected_cell[0] + 0.12, rejected_cell[1] + 0.16,
              f'{g_trial:.1f}', fontsize=8.2, color='#C0392B',
              ha='left', va='center', zorder=15)
    axis.text(rejected_cell[0] + 0.88, rejected_cell[1] + 0.16,
              f'{h_trial:.1f}', fontsize=8.2, color='#C0392B',
              ha='right', va='center', zorder=15)
    rejected_q = NEIGHBORS_8.index(tuple(
        np.asarray(outgoing, dtype=int)))
    axis.text(rejected_point[0], rejected_cell[1] + 0.55,
              r'$p_4^{\mathrm{A}}$', fontsize=10.5, fontweight='bold',
              color='#C0392B', ha='center', va='center', zorder=15,
              bbox=dict(facecolor='white', edgecolor='none', alpha=0.86, pad=0.4))
    axis.scatter(rejected_point[0], rejected_cell[1] + 0.34, marker='x', s=62,
                 linewidth=1.9, color='#C0392B', zorder=15)
    # 红绿说明框采用完全一致的字段顺序：候选点、状态、转角和曲率代价。
    rejected_text = (
        r'$p_4^{\mathrm{A}}$: Rejected' + '\n' +
        rf'$s_4^{{\mathrm{{A}}}}=(({rejected_cell[0]},{rejected_cell[1]}),q_{{{rejected_q}}})$' + '\n' +
        r'$\theta_i=90^\circ>\theta_{\max}=60^\circ$' + '\n' +
        rf'$K_i={curvature_trial:.1f}$')
    axis.annotate(
        rejected_text,
        xy=(rejected_point[0], rejected_cell[1] + 0.34),
        xytext=(11.95, 5.55),
        fontsize=10.0, color='#C0392B', ha='center', va='center',
        linespacing=1.22,
        arrowprops=dict(arrowstyle='->', color='#C0392B', linewidth=1.1,
                        linestyle='--'),
        bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                  edgecolor='#C0392B', alpha=0.96), zorder=14)
    accepted_q = NEIGHBORS_8.index((1, 1))
    accepted_cell = path_states[4][0]
    accepted_text = (
        r'$p_4^{\mathrm{D}}$: Accepted' + '\n' +
        rf'$s_4^{{\mathrm{{D}}}}=(({accepted_cell[0]},{accepted_cell[1]}),q_{{{accepted_q}}})$' + '\n' +
        r'$\theta_i=0^\circ\leq\theta_{\max}=60^\circ$' + '\n' +
        r'$K_i=0$')
    axis.annotate(
        accepted_text, xy=fork + 0.72 * accepted, xytext=(11.95, 7.00),
        fontsize=10.0, color=SCI_COLORS['start'], ha='center', va='center',
        linespacing=1.22,
        arrowprops=dict(arrowstyle='->', color=SCI_COLORS['start'],
                        linewidth=1.1, linestyle='--'),
        bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                  edgecolor=SCI_COLORS['start'], alpha=0.96), zorder=14)

    # 以 p4D 所在栅格为例展示被接受的联合状态。
    focus = accepted_cell
    fx, fy = focus
    axis.add_patch(Rectangle((fx + 0.05, fy + 0.05), 0.90, 0.90,
                             facecolor='none', edgecolor=SCI_COLORS['direction_alt'],
                             linewidth=2.0, zorder=10))
    table_lines = [
        r'Joint state: $s=(c,q)=((5,6),q_7)$',
        r'$c=(5,6)$: current grid-cell position',
        r'$q_7$: incoming-direction index in the 8-neighbor set',
        r'$q_7\leftrightarrow(\Delta x,\Delta y)=(+1,+1)$',
        r'entered from predecessor $(4,5)$ to current cell $(5,6)$',
    ]
    axis.annotate(
        '\n'.join(table_lines),
        xy=(fx + 0.5, fy + 0.92), xytext=(11.95, 3.95),
        fontsize=9.2, color='#25313C',
        ha='center', va='center', linespacing=1.30,
        arrowprops=dict(arrowstyle='->', color=SCI_COLORS['direction_alt'],
                        linewidth=1.2),
        bbox=dict(boxstyle='round,pad=0.36', facecolor='white',
                  edgecolor=SCI_COLORS['direction_alt'], alpha=0.97),
        zorder=12)

    transition_text = (
        'Values shown in each searched cell\n' +
        r'top: $f(s)=g(s)+h(s)$  --- estimated total cost' + '\n' +
        r'lower left: $g(s)$  --- accumulated transition cost' + '\n' +
        r'lower right: $h(s)=T_{\mathrm{FMM}}(c)$  --- FMM cost-to-go' + '\n' +
        r'$g(s_j)=g(s_i)+1.0L_{ij}+0.2R_{ij}+0.1K_i$' + '\n' +
        r'$L$: length; $R$: clearance risk; $K$: curvature')
    axis.text(11.95, 2.05, transition_text, fontsize=10.2,
              color='#25313C', ha='center', va='center', linespacing=1.28,
              bbox=dict(boxstyle='round,pad=0.38', facecolor='white',
                        edgecolor=SCI_COLORS['direction'], linewidth=1.2,
                        alpha=0.97), zorder=14)

    heuristic_text = (
        'Obstacle-aware heuristic\n' +
        r'$h(s_j)=1.0\,T_{\mathrm{FMM}}(c_j)$' + '\n' +
        'ESDF speed field encodes global obstacles\n'
        'and clearance; gray-blue dashed curves: FMM isochrones')
    axis.text(11.95, 0.55, heuristic_text, fontsize=10.2,
              color='#25313C', ha='center', va='center', linespacing=1.30,
              bbox=dict(boxstyle='round,pad=0.38', facecolor='white',
                        edgecolor='#456A83', linewidth=1.2, alpha=0.97),
              zorder=14)
    axis.text(0.18, 7.78,
              r'FMM arrival-time field $T_{\mathrm{FMM}}$: dark $\rightarrow$ high, light $\rightarrow$ low',
              fontsize=9.5, color='#31556D', ha='left', va='top', zorder=15,
              bbox=dict(facecolor='white', edgecolor='#456A83', alpha=0.90,
                        boxstyle='round,pad=0.22'))
    draw_start_goal(axis, start, goal)
    draw_color_legend(axis, direction_aware=True)
    axis.set_title(
        '(b) FDGA*: direction-aware joint-state search with FMM heuristic\n'
        r'$s=(c,q)$; $\theta_i\leq\theta_{\max}=60^\circ$; '
        r'$\Delta g_{ij}=1.0L_{ij}+0.2R_{ij}+0.1K_i$; '
        r'$h=T_{\mathrm{FMM}}$',
        fontsize=16.5, pad=16, linespacing=1.35, color='#25313C')


def draw_color_direction_aware_astar_v2(axis):
    width, height, start, goal, obstacles = astar_map_definition()
    arrival_time, clearance = compute_fmm_arrival_field(
        width, height, goal, obstacles)
    guide_cells, corridor_cells = build_fmm_guidance_corridor(
        width, height, start, goal, obstacles)
    g_score, open_states, closed_states, path_states = compute_direction_aware_astar(
        width, height, start, goal, obstacles, arrival_time, clearance,
        allowed_cells=corridor_cells)
    open_cells = {state[0] for state in open_states if state[0] in corridor_cells}
    closed_cells = {state[0] for state in closed_states if state[0] in corridor_cells}

    setup_color_grid(axis, width, height)
    axis.set_xlim(-8.0, width)
    axis.add_patch(Rectangle(
        (-8.0, 0), 8.0, height, facecolor='white', edgecolor='#B8C2C9',
        linewidth=1.0, zorder=5, clip_on=False))
    fmm_image = draw_fmm_background(axis, arrival_time, width, height)

    # FMM 初始路径周围的一格宽走廊是后续联合状态搜索的唯一扩展区域。
    for x, y in corridor_cells:
        axis.add_patch(Rectangle(
            (x + 0.03, y + 0.03), 0.94, 0.94,
            facecolor='#74ADD1', edgecolor='none', alpha=0.20, zorder=0))
    corridor_boundary = np.ones((height, width), dtype=float)
    for x, y in corridor_cells:
        corridor_boundary[y, x] = 0.0
    axis.contour(
        np.arange(width) + 0.5, np.arange(height) + 0.5,
        corridor_boundary, levels=[0.5], colors='#2A6F97',
        linewidths=1.5, linestyles='--', zorder=4)

    draw_color_cells(axis, width, height, obstacles, open_cells, closed_cells,
                     reveal_fmm=True)
    colorbar_axis = axis.inset_axes([0.31, 0.945, 0.43, 0.035])
    colorbar = axis.figure.colorbar(
        fmm_image, cax=colorbar_axis, orientation='horizontal')
    colorbar.set_label(r'FMM global arrival cost $T_{\mathrm{FMM}}(c)$',
                       fontsize=12.5, labelpad=3)
    colorbar.ax.tick_params(labelsize=10.5, length=3)

    path_cells = [state[0] for state in path_states]
    centers = np.array([(x + 0.5, y + 0.5) for x, y in path_cells])
    axis.plot(centers[:, 0], centers[:, 1], color=SCI_COLORS['path'],
              linewidth=3.1, zorder=8, solid_joinstyle='round')
    axis.scatter(centers[:, 0], centers[:, 1], s=23, facecolor='white',
                 edgecolor=SCI_COLORS['path'], linewidth=1.0, zorder=9)
    for (cell, direction) in path_states[1:]:
        x, y = cell
        dx, dy = direction
        axis.add_patch(FancyArrowPatch(
            (x + 0.5 - 0.16 * dx, y + 0.5 - 0.16 * dy),
            (x + 0.5 + 0.16 * dx, y + 0.5 + 0.16 * dy),
            arrowstyle='-|>', mutation_scale=8, color=SCI_COLORS['path'],
            linewidth=1.0, zorder=10))

    states_by_cell = {}
    for state, value in g_score.items():
        states_by_cell.setdefault(state[0], []).append((state, value))
    key_cells = {start, goal, (4, 6), (5, 6), (5, 7)}
    key_cells.update(path_cells[::4])
    for cell in sorted(key_cells):
        if cell not in states_by_cell or cell in obstacles:
            continue
        state, g_value = min(states_by_cell[cell], key=lambda item: item[1])
        x, y = cell
        h_value = arrival_time[y, x]
        axis.text(x + 0.50, y + 0.75, f'{g_value + h_value:.1f}',
                  ha='center', va='center', fontsize=8.3,
                  color='#25313C', zorder=11)
        axis.text(x + 0.10, y + 0.15, f'{g_value:.1f}', ha='left', va='center',
                  fontsize=6.8, color='#40505E', zorder=11)
        axis.text(x + 0.90, y + 0.15, f'{h_value:.1f}', ha='right', va='center',
                  fontsize=6.8, color='#40505E', zorder=11)

    # 角度约束示范放在 A* 与 FDGA* 最终路径首次分歧的格网。
    p3, p4a, p4d = (4, 6), (5, 6), (5, 7)
    fork = np.array(p3, dtype=float) + 0.5
    incoming = np.array((0.0, 1.0))
    accepted = np.array((1.0, 1.0))
    rejected = np.array((1.0, 0.0))
    for vector, length, color, linestyle in (
            (incoming, 0.55, SCI_COLORS['direction_alt'], '-'),
            (accepted, 0.92, SCI_COLORS['start'], '-'),
            (rejected, 0.92, '#C0392B', '--')):
        unit_vector = vector / np.linalg.norm(vector)
        axis.add_patch(FancyArrowPatch(
            fork, fork + length * unit_vector, arrowstyle='-|>', mutation_scale=14,
            linewidth=2.1, linestyle=linestyle, color=color, zorder=14))
    # 主图仅保留候选方向和节点标记；详细文字统一放入局部放大图。
    axis.add_patch(Arc(fork, 0.82, 0.82, theta1=0, theta2=90,
                       linewidth=1.6, color='#C0392B', zorder=14))
    for cell, color in (
            (p3, '#25313C'),
            (p4a, '#C0392B'),
            (p4d, SCI_COLORS['start'])):
        x, y = cell
        axis.scatter(x + 0.5, y + 0.5, s=82, facecolor='white',
                     edgecolor=color, linewidth=1.9, zorder=13)

    guide = np.array([(x + 0.5, y + 0.5) for x, y in guide_cells])
    axis.plot(guide[:, 0], guide[:, 1], color='#1E6F9F', linewidth=3.6,
              linestyle=(0, (7, 3)), alpha=0.98, zorder=12)
    axis.add_patch(FancyArrowPatch(guide[-2], guide[-1], arrowstyle='-|>',
                                   mutation_scale=18, linewidth=2.8,
                                   color='#1E6F9F', zorder=13))
    axis.text(10.8, 9.05, 'FMM initial guidance path', fontsize=13.2,
              color='#245B78', fontweight='bold', ha='center', va='center',
              bbox=dict(boxstyle='round,pad=0.24', facecolor='white',
                        edgecolor='#2A6F97', alpha=0.93), zorder=15)
    axis.annotate('Search corridor\n(outside cells are not expanded)',
                  xy=(13.5, 7.15), xytext=(12.0, 6.25), fontsize=12.0,
                  color='#245B78', ha='center', va='center',
                  arrowprops=dict(arrowstyle='->', color='#2A6F97', linewidth=1.5),
                  bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                            edgecolor='#2A6F97', alpha=0.94), zorder=15)

    axis.add_patch(Rectangle((5.05, 7.05), 0.90, 0.90, facecolor='none',
                             edgecolor=SCI_COLORS['direction_alt'],
                             linewidth=2.3, zorder=12))
    # 计算局部放大区域中三个状态的代价，用于格内标注和左侧详细说明。
    _, p3_g = min(states_by_cell[p3], key=lambda item: item[1])
    _, p4d_g = min(states_by_cell[p4d], key=lambda item: item[1])
    p3_h = arrival_time[p3[1], p3[0]]
    p4a_h = arrival_time[p4a[1], p4a[0]]
    p4d_h = arrival_time[p4d[1], p4d[0]]
    length_a, length_d = 1.0, np.sqrt(2.0)
    risk_a = length_a * 0.5 * (
        1.0 / clearance[p3[1], p3[0]] +
        1.0 / clearance[p4a[1], p4a[0]])
    risk_d = length_d * 0.5 * (
        1.0 / clearance[p3[1], p3[0]] +
        1.0 / clearance[p4d[1], p4d[0]])
    curvature_a = (np.pi / 2.0) ** 2
    curvature_d = (np.pi / 4.0) ** 2 / (0.5 * (1.0 + length_d))
    delta_a = length_a + 0.2 * risk_a + 0.1 * curvature_a
    delta_d = length_d + 0.2 * risk_d + 0.1 * curvature_d
    p4a_g = p3_g + delta_a
    local_costs = {
        p3: (p3_g + p3_h, p3_g, p3_h),
        p4a: (p4a_g + p4a_h, p4a_g, p4a_h),
        p4d: (p4d_g + p4d_h, p4d_g, p4d_h),
    }

    # 参照原始 A* 图，在左侧放大 p3 处的角度约束决策。
    inset = axis.inset_axes([0.012, 0.47, 0.285, 0.46])
    inset.set_facecolor('white')
    inset.set_xlim(3.65, 6.35)
    inset.set_ylim(5.65, 8.35)
    inset.set_aspect('equal')
    inset.set_xticks(np.arange(4, 7))
    inset.set_yticks(np.arange(6, 9))
    inset.grid(color=SCI_COLORS['grid'], linewidth=0.8, alpha=0.9)
    inset.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    zoom_color = '#8E44AD'
    for spine in inset.spines.values():
        spine.set_color(zoom_color)
        spine.set_linewidth(2.4)
    # 放大图只保留决策所需信息，避免代价值与状态文字相互遮挡。
    for cell, edge in (
            (p3, SCI_COLORS['closed']),
            (p4a, '#C0392B'),
            (p4d, SCI_COLORS['start'])):
        x, y = cell
        inset.add_patch(Rectangle(
            (x + 0.045, y + 0.045), 0.91, 0.91,
            facecolor='none', edgecolor=edge, linewidth=2.0, zorder=3))

    for vector, length, color, linestyle in (
            (incoming, 0.58, SCI_COLORS['direction_alt'], '-'),
            (accepted, 0.96, SCI_COLORS['start'], '-'),
            (rejected, 0.96, '#C0392B', '--')):
        unit_vector = vector / np.linalg.norm(vector)
        inset.add_patch(FancyArrowPatch(
            fork, fork + length * unit_vector, arrowstyle='-|>',
            mutation_scale=17, linewidth=2.5, linestyle=linestyle,
            color=color, zorder=8))
    inset.add_patch(Arc(fork, 0.90, 0.90, theta1=0, theta2=90,
                        linewidth=2.0, color='#C0392B', zorder=8))

    # 在三个关键格网内恢复 f/g/h 代价，版式与主图保持一致。
    for cell, (f_value, g_value, h_value) in local_costs.items():
        x, y = cell
        inset.text(x + 0.50, y + 0.84, f'{f_value:.1f}',
                   fontsize=7.8, color='#25313C', ha='center', va='center', zorder=6)
        inset.text(x + 0.10, y + 0.10, f'{g_value:.1f}',
                   fontsize=6.8, color='#40505E', ha='left', va='center', zorder=6)
        inset.text(x + 0.90, y + 0.10, f'{h_value:.1f}',
                   fontsize=6.8, color='#40505E', ha='right', va='center', zorder=6)

    # 三个节点标签分别放在各自格网的中部空白区域。
    inset.scatter(4.5, 6.5, s=100, facecolor='white',
                  edgecolor='#25313C', linewidth=2.1, zorder=9)
    inset.scatter(5.5, 6.5, s=100, facecolor='white',
                  edgecolor='#C0392B', linewidth=2.1, zorder=9)
    inset.scatter(5.5, 7.5, s=100, facecolor='white',
                  edgecolor=SCI_COLORS['start'], linewidth=2.1, zorder=9)
    inset.text(4.18, 6.28, r'$p_3$', fontsize=11.8, fontweight='bold',
               color='#25313C', ha='center', va='center', zorder=11)
    inset.text(5.52, 6.28, r'$p_4^{\mathrm{A}}\ (q_4)$', fontsize=10.6,
               fontweight='bold', color='#C0392B', ha='center', va='center',
               zorder=11)
    inset.text(5.52, 7.72, r'$p_4^{\mathrm{D}}\ (q_7)$', fontsize=10.6,
               fontweight='bold', color=SCI_COLORS['start'],
               ha='center', va='center', zorder=11)

    # 将 90° 标注移入红色候选格，避开绿色接受方向线。
    inset.text(5.04, 6.78, r'$\theta_i^{\mathrm{A}}=90^\circ$',
               fontsize=10.2, color='#C0392B', fontweight='bold',
               ha='left', va='center', zorder=10,
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.94, pad=0.10))
    inset.text(4.72, 7.38, r'$\theta_i^{\mathrm{D}}=45^\circ$',
               fontsize=10.2, color=SCI_COLORS['start'], fontweight='bold',
               ha='left', va='center', zorder=10,
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.94, pad=0.10))
    inset.text(5.50, 5.88, 'Rejected', fontsize=10.5, fontweight='bold',
               color='#C0392B', ha='center', va='center', zorder=11)
    inset.text(5.50, 8.12, 'Accepted', fontsize=10.5, fontweight='bold',
               color=SCI_COLORS['start'], ha='center', va='center', zorder=11)
    inset.set_title(r'Local decision at $p_3$: $\theta_{\max}=60^\circ$',
                    fontsize=11.5, color='#25313C', pad=4)

    # 用醒目边框圈出主图中被局部放大的关键区域。
    axis.add_patch(Rectangle(
        (3.72, 5.72), 2.56, 2.56, facecolor='none',
        edgecolor=zoom_color, linewidth=2.6, linestyle=(0, (5, 2)),
        zorder=16))
    # 用高对比紫色连接主图高亮区域与局部放大图。
    axis.add_patch(FancyArrowPatch(
        (3.72, 8.28), (-0.48, 10.28), arrowstyle='-',
        linewidth=2.0, color=zoom_color, zorder=12,
        connectionstyle='arc3,rad=0', clip_on=False))
    axis.add_patch(FancyArrowPatch(
        (3.72, 5.72), (-0.48, 6.02), arrowstyle='-',
        linewidth=2.0, color=zoom_color, zorder=12,
        connectionstyle='arc3,rad=0', clip_on=False))
    cost_text = (
        'Local transition costs from $p_3$\n' +
        rf'$p_4^{{\mathrm{{A}}}}$: $L={length_a:.1f}$, $R={risk_a:.2f}$, '
        rf'$K={curvature_a:.2f}$, $\Delta g={delta_a:.2f}$' + '\n' +
        rf'$p_4^{{\mathrm{{D}}}}$: $L=\sqrt{{2}}$, $R={risk_d:.2f}$, '
        rf'$K={curvature_d:.2f}$, $\Delta g={delta_d:.2f}$' + '\n' +
        r'$f=g+T_{\mathrm{FMM}}$; top: $f$, lower-left: $g$, lower-right: $T_{\mathrm{FMM}}$' + '\n' +
        r'$p_4^{\mathrm{A}}$ is discarded because $90^\circ>\theta_{\max}=60^\circ$.')
    axis.text(-4.0, 3.65, cost_text, fontsize=10.3, color='#25313C',
              ha='center', va='center', linespacing=1.32,
              bbox=dict(boxstyle='round,pad=0.40', facecolor='white',
                        edgecolor=SCI_COLORS['direction'], linewidth=1.4), zorder=15)
    axis.text(-4.0, 1.15,
              'FMM is computed first and extracts an\ninitial guidance path and local corridor.\nJoint-state search is restricted to this corridor;\noutside cells are not expanded.',
              fontsize=11.2, color='#31556D', ha='center', va='center',
              linespacing=1.32,
              bbox=dict(boxstyle='round,pad=0.40', facecolor='white',
                        edgecolor='#456A83', linewidth=1.4), zorder=15)
    axis.text(-4.0, 5.18,
              r'Cell labels: top $f$, lower-left $g$, lower-right $T_{\mathrm{FMM}}$',
              fontsize=10.5, color='#31556D', ha='center', va='center', zorder=15,
              bbox=dict(facecolor='white', edgecolor='#456A83', alpha=0.96,
                        boxstyle='round,pad=0.24'))
    draw_start_goal(axis, start, goal)
    draw_color_legend(axis, direction_aware=True)


def draw_color_direction_aware_astar_compact(axis):
    """绘制不含左侧说明区和局部放大图的 FDGA* 主图。"""
    draw_color_direction_aware_astar_v2(axis)

    # 删除局部放大坐标轴，保留顶部 FMM 色条。
    for child_axis in list(axis.child_axes):
        if child_axis.get_position().height > 0.10:
            child_axis.remove()

    # 删除左侧说明文字、说明区背景及局部放大连接线。
    for text in list(axis.texts):
        if text.get_position()[0] < 0:
            text.remove()
    for patch in list(axis.patches):
        if isinstance(patch, Rectangle) and (
                patch.get_x() < 0 or patch.get_width() > 2.0):
            patch.remove()
        elif isinstance(patch, FancyArrowPatch) and not patch.get_clip_on():
            patch.remove()

    axis.set_xlim(0, 22)


# ---------------------------------------------------------------- 主渲染流程

def save_single_panel(draw_panel, output_base, dpi, figsize=(7.2, 7.7)):
    figure, axis = plt.subplots(figsize=figsize)
    draw_panel(axis)
    figure.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.03)
    output_path = f'{output_base}.png'
    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f'[Output] {output_path}')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description='分别绘制原始 A* 与方向感知 A* 的二维搜索状态图。')
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    configure_style()
    os.makedirs(args.output_dir, exist_ok=True)
    save_single_panel(
        draw_conventional_panel,
        os.path.join(args.output_dir, 'FigE1a_conventional_astar_state'),
        args.dpi,
    )
    save_single_panel(
        draw_fdga_panel,
        os.path.join(args.output_dir, 'FigE1b_direction_aware_state'),
        args.dpi,
    )
    save_single_panel(
        draw_black_white_astar,
        os.path.join(args.output_dir, 'FigE1a_conventional_astar_state_bw'),
        args.dpi,
        figsize=(10.0, 8.3),
    )
    save_single_panel(
        draw_color_astar_8,
        os.path.join(args.output_dir, 'FigE1a_astar_8direction_color'),
        args.dpi,
        figsize=(15.5, 9.2),
    )
    save_single_panel(
        draw_color_direction_aware_astar_v2,
        os.path.join(args.output_dir, 'FigE1b_direction_aware_astar_2d_color_state_explanation'),
        args.dpi,
        figsize=(19.0, 9.2),
    )
    save_single_panel(
        draw_color_direction_aware_astar_compact,
        os.path.join(args.output_dir, 'FigE1b_direction_aware_astar_compact'),
        args.dpi,
        figsize=(14.2, 8.6),
    )


if __name__ == '__main__':
    main()
