#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三算法路径叠加 DSM 三维可视化 — 真实深度渲染 (PyVista/VTK)
============================================================
为何弃用 matplotlib:
  matplotlib 的 3D 是"画家算法"(painter's algorithm), 没有真实深度缓冲(z-buffer),
  曲面与路径的遮挡关系靠排序近似, 复杂地形下路径会错误地穿透山体或悬浮.
  本脚本改用 PyVista (VTK 内核), 由 OpenGL 硬件执行真实 z-buffer 深度测试,
  曲面与路径之间、路径自身的前后遮挡完全符合几何实况, 如实体现深度关系.

功能:
  1. 读取 GeoTIFF DSM, 构建三角网格曲面, 应用灰白色系 + 山体阴影纹理
  2. 读取 FMM / A* / RRT* 三条路径, 以 3D 管状(流线)渲染, 深色高对比
  3. 真实光照 + 深度缓冲, 路径被山体遮挡处正确隐藏
  4. 输出适用于论文插图的 DSM 三维主视角

依赖:
  pip install pyvista numpy matplotlib
  # 无显示器环境(服务器/Docker)还需:
  pip install pyvista[osmesa]   # 或安装 mesa / xvfb

用法:
  python visDSM.py                        # 默认: 不带风场 (exp_nowind)
  python visDSM.py --exp wind             # 渲染带风场实验路径 (exp_wind)
  python visDSM.py --exp nowind           # 渲染不带风场实验路径 (exp_nowind)
  python visDSM.py --dsm ./data/SF_Downtown.tif --path-dir ./data/exp_wind
  python visDSM.py --z-exag 2.0 --dpi 300
"""

import os
import argparse
import math
import numpy as np
import pyvista as pv
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap

# -------------------------- 全局样式配置 --------------------------
# 输出目录 (与 Analysis.py 一致)
OUTPUT_DIR = '../analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_DSM = '../data/SF_Downtown.tif'
DEFAULT_PATH_DIR = '../data'

# 风场实验目录映射 (--exp 参数): main.cpp 按 use_wind_cost 分目录输出,
#   nowind → data/exp_nowind/   (原实验, 无风场代价)
#   wind   → data/exp_wind/     (节能规划, 风场代价)
EXP_DIRS = {
    'nowind': '../data/exp_nowind',
    'wind':   '../data/exp_wind',
}

# 三算法路径配色: 明亮高饱和色 (在灰白 DSM 表面上最醒目)
PLANNERS = [
    ('FMM',     'search_path_FMM.csv',     '#1FF70A', 'FMM'),   # 霓虹绿
    ('ASTAR',   'search_path_ASTAR.csv',   '#FF1744', 'A*'),    # 霓虹红
    ('RRTSTAR', 'search_path_RRTSTAR.csv', '#FFEA00', 'RRT*'),  # 霓虹黄
]

# DSM 灰白色系 colormap: 深灰 → 浅白 (低海拔深, 高海拔浅)
DSM_CMAP = LinearSegmentedColormap.from_list(
    'gray_white', ['#2a2a2a', '#5a5a5a', '#8a8a8a', '#b8b8b8', '#e8e8e8', '#ffffff'], N=256
)


# -------------------------- GeoTIFF 读取 --------------------------
def read_geotiff(filepath):
    """读取 GeoTIFF, 返回 (elevation, geo_transform, nodata, crs_name).
    geo_transform = (origin_x, pixel_w, 0, origin_y, 0, pixel_h).
    优先 rasterio, 其次 GDAL, 最后 PIL + 手动解析."""
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            elev = src.read(1).astype(np.float32)
            gt = src.transform
            nodata = src.nodata
            crs = src.crs.to_string() if src.crs else 'unknown'
        geo = (gt.c, gt.a, gt.b, gt.f, gt.d, gt.e)
        print(f"[rasterio] 读取成功 | shape={elev.shape} crs={crs}")
        return elev, geo, nodata, crs
    except ImportError:
        pass

    try:
        from osgeo import gdal
        ds = gdal.Open(filepath)
        band = ds.GetRasterBand(1)
        elev = band.ReadAsArray().astype(np.float32)
        gt = ds.GetGeoTransform()
        nodata = band.GetNoDataValue()
        proj = ds.GetProjection()
        crs = proj.split('"')[-2] if 'PROJCS' in proj or 'GEOGCS' in proj else 'unknown'
        print(f"[GDAL] 读取成功 | shape={elev.shape} crs={crs[:40]}")
        return elev, gt, nodata, crs
    except ImportError:
        pass

    from PIL import Image
    img = Image.open(filepath)
    elev = np.array(img).astype(np.float32)
    tags = img.tag_v2 if hasattr(img, 'tag_v2') else {}
    scale = tags.get(33550, (1.0, 1.0, 1.0))
    pixel_w = abs(scale[0]); pixel_h = abs(scale[1])
    tiepoint = tags.get(33922, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    origin_x = tiepoint[3]; origin_y = tiepoint[4]
    nodata_raw = tags.get(42113, None)
    try:
        nodata = float(nodata_raw) if nodata_raw is not None else None
    except (ValueError, TypeError):
        nodata = None
    crs = 'unknown'
    if 34737 in tags:
        crs = str(tags[34737]).split('|')[0].strip() or 'unknown'
    geo = (origin_x, pixel_w, 0.0, origin_y, 0.0, -pixel_h)
    print(f"[PIL] 读取成功 | shape={elev.shape} pixel=({pixel_w}m, {pixel_h}m)")
    return elev, geo, nodata, crs


# -------------------------- 路径读取 --------------------------
def load_path_csv(csv_path):
    """读取路径 CSV (x,y,z), 返回 Nx3 数组."""
    return np.loadtxt(csv_path, delimiter=',', skiprows=1)


# -------------------------- DSM 高程查询 (用于路径贴地) --------------------------
def make_elevation_sampler(elev, geo):
    """构造高程查询函数: (x, y) → DSM 高程值 (双线性插值)."""
    origin_x, pixel_w, _, origin_y, _, pixel_h = geo
    pixel_h = abs(pixel_h)
    rows, cols = elev.shape

    def sample(x, y):
        col = (x - origin_x) / pixel_w
        row = (origin_y - y) / pixel_h
        if col < 0 or col >= cols - 1 or row < 0 or row >= rows - 1:
            return np.nan
        c0, r0 = int(col), int(row)
        dc, dr = col - c0, row - r0
        z = (elev[r0, c0]     * (1 - dc) * (1 - dr) +
             elev[r0, c0+1]   * dc       * (1 - dr) +
             elev[r0+1, c0]   * (1 - dc) * dr +
             elev[r0+1, c0+1] * dc       * dr)
        return float(z)

    return sample


# -------------------------- 相机视角 (matplotlib 约定 → PyVista) --------------------------
def camera_from_elev_azim(elev, azim, focal, distance):
    """把 matplotlib 的 (elev, azim) 视角约定转换为 PyVista 相机位置.
    elev: 仰角(度); azim: 方位角(度, 0=正南视角, 逆时针增大)."""
    elev_r = math.radians(elev)
    azim_r = math.radians(azim)
    offset = np.array([
        math.cos(elev_r) * math.sin(azim_r),
        math.cos(elev_r) * math.cos(azim_r),
        math.sin(elev_r),
    ])
    pos = np.asarray(focal, dtype=float) + offset * distance
    return [pos.tolist(), list(focal), [0, 0, 1]]


# -------------------------- 视图配置 --------------------------
# 仅输出论文使用的主视角
VIEWS = [
    ('主视角', 5, 0),
]


# -------------------------- 主可视化函数 --------------------------
def visualize_paths_on_dsm(dsm_path, path_dir, dpi=200,
                           z_exaggeration=1.5, path_lift=0.0,
                           output_name='DSM_3D',
                           output_dir=None):
    """渲染不叠加航线的 DSM 三维高程图。"""
    print("=" * 60)
    print("DSM 三维高程可视化 (PyVista/VTK 真实深度渲染)")
    print("=" * 60)

    out_dir = output_dir if output_dir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. 读取 DSM ----
    elev, geo, nodata, crs = read_geotiff(dsm_path)
    origin_x, pixel_w, _, origin_y, _, pixel_h = geo
    pixel_h = abs(pixel_h)
    rows, cols = elev.shape

    if nodata is not None:
        nodata_mask = (np.abs(elev - nodata) < 1e-3) | ~np.isfinite(elev)
    else:
        nodata_mask = ~np.isfinite(elev)
    elev_clean = elev.copy()
    if nodata_mask.any():
        valid_min = np.nanmin(elev_clean[~nodata_mask])
        elev_clean[nodata_mask] = valid_min

    z_min, z_max = np.nanmin(elev_clean), np.nanmax(elev_clean)
    print(f"[DSM] 尺寸: {rows}x{cols}, 高程: {z_min:.1f}~{z_max:.1f}m")

    # ---- 2. 降采样 DSM (控制网格规模, 保证 VTK 渲染流畅) ----
    target_pts = 400000
    step = max(1, int(np.sqrt(rows * cols / target_pts)))
    elev_sub = elev_clean[::step, ::step]
    print(f"[DSM] 降采样 step={step} → {elev_sub.shape[0]}x{elev_sub.shape[1]}")

    x_coords = origin_x + np.arange(0, cols, step) * pixel_w
    y_coords = origin_y - np.arange(0, rows, step) * pixel_h   # 北向 Y 递减
    X, Y = np.meshgrid(x_coords, y_coords)
    Z = elev_sub * z_exaggeration

    # ---- 3. 构建 DSM 曲面网格 (PyVista) ----
    # StructuredGrid → PolyData (extract_surface 显式指定 algorithm 消除未来默认值变更警告)
    surf = pv.StructuredGrid(X, Y, Z).extract_surface(algorithm='dataset_surface')

    # ---- 4. 灰白 + 山体阴影纹理 (烘培到曲面) ----
    norm = Normalize(vmin=z_min, vmax=z_max)
    ls = LightSource(azdeg=315, altdeg=55)
    rgba = ls.shade(elev_sub, cmap=DSM_CMAP, blend_mode='overlay',
                    vert_exag=z_exaggeration, dx=pixel_w * step, dy=pixel_h * step,
                    fraction=1.5)
    rgb = np.clip(rgba[:, :, :3] * 255, 0, 255).astype(np.uint8)
    tex = pv.Texture(rgb)

    # 纹理坐标: 用 texture_map_to_plane 自动生成 (按 XY 平面投影)
    # 新版 PyVista 不允许直接赋值 t_coords, 此函数会正确写入 point_data
    surf.texture_map_to_plane(inplace=True)
    print(f"[DSM] 曲面网格点: {surf.n_points}, 纹理 {elev_sub.shape[1]}x{elev_sub.shape[0]}")

    surf['Elevation (m)'] = surf.points[:, 2] / z_exaggeration

    # ---- 5. DSM 三维可视化 ----
    x_range = (x_coords.min(), x_coords.max())
    y_range = (y_coords.min(), y_coords.max())
    z_range = (z_min * z_exaggeration, z_max * z_exaggeration)
    for view_name, elev_angle, azim_angle in VIEWS:
        print(f"\n[渲染] {view_name} (elev={elev_angle}, azim={azim_angle})")
        _render_full_view(
            surf, tex, view_name, elev_angle, azim_angle,
            x_range, y_range, z_range, z_exaggeration,
            dpi, out_dir, output_name)

    print("\n完成!")


def _render_full_view(surf, tex, view_name, elev_angle, azim_angle,
                      x_range, y_range, z_range, z_exag,
                      dpi, out_dir, output_name):
    """渲染带坐标轴和高程色标的 DSM 三维曲面。"""
    xspan = x_range[1] - x_range[0]
    yspan = y_range[1] - y_range[0]
    zspan = z_range[1] - z_range[0]
    diag = np.sqrt(xspan * xspan + yspan * yspan + zspan * zspan)

    p = pv.Plotter(off_screen=True, window_size=(1600, 1200))

    # DSM 曲面及高程色标；不叠加航线和起终点标记
    p.add_mesh(
        surf,
        scalars='Elevation (m)',
        # 滨海城市 DSM 配色：浅蓝灰低地过渡到沙金色及暖色高层建筑
        cmap=['#D7E8EF', '#A8CAD3', '#80ADB5', '#D8CEA8',
              '#E8B85C', '#D9824B', '#A94F3D'],
        clim=[z_range[0] / z_exag, z_range[1] / z_exag],
        lighting=True,
        ambient=0.65,
        diffuse=0.55,
        specular=0.08,
        smooth_shading=True,
        show_edges=False,
        scalar_bar_args={
            'title': 'Elevation (m)',
            'vertical': True,
            'title_font_size': 20,
            'label_font_size': 17,
            'font_family': 'arial',
            'n_labels': 6,
            'fmt': '%.0f',
            'width': 0.040,
            'height': 0.74,
            'position_x': 0.83,
            'position_y': 0.10,
            'bold': True,
        },
        name='dsm',
    )

    # ---- 相机设置 ----
    p.reset_camera()
    p.camera.clipping_range = [0.1, diag * 5.0]

    # ===== 主视角 =====
    p.camera.enable_parallel_projection = False
    p.camera.elevation = elev_angle
    p.camera.azimuth = azim_angle
    p.camera.distance = diag * 0.5

    # 坐标框仅置于 DSM 外围，不绘制穿过 DSM 的内部网格线
    bounds_actor = p.show_bounds(
        xtitle='东向 X (km)', ytitle='北向 Y (km)', ztitle='高度 Z (km)',
        axes_ranges=(x_range[0] / 1000, x_range[1] / 1000,
                     y_range[0] / 1000, y_range[1] / 1000,
                     z_range[0] / z_exag / 1000,
                     z_range[1] / z_exag / 1000),
        font_size=14,
        fmt='%.1f',
        n_xlabels=3,
        n_ylabels=3,
        n_zlabels=5,
        grid=False,
        location='outer',
        use_2d=False,
    )
    # 增大刻度与各自轴线的距离，使底部公共端点的两个数值沿两轴分开
    bounds_actor.SetLabelOffset(24.0)

    # 左下角三维方位坐标轴：X 指东，Y 指北，Z 指上
    p.add_axes(
        xlabel='E',
        ylabel='N',
        zlabel='UP',
        line_width=3,
        labels_off=False,
        cone_radius=0.30,
        shaft_length=0.72,
        tip_length=0.24,
        viewport=(0.155, 0.065, 0.275, 0.215),
    )

    # 放大主体并让方位坐标轴贴近渲染区域，减少无效留白
    p.camera.zoom(0.92)
    p.set_background('white')

    # 使用纯 ASCII 文件名，避免 Docker 挂载目录拒绝中文路径。
    tag = 'main_view' if view_name == '主视角' else view_name.replace(' ', '_')
    png_path = os.path.join(out_dir, f'{output_name}_{tag}.png')
    p.screenshot(png_path, transparent_background=False,
                 window_size=(1800, 1300))
    p.close()
    print(f"  [输出] {png_path}")


def _auto_radius(pts, x_range, y_range):
    """根据路径几何规模自动估计管径 (物理单位), 保证视觉醒目且不过度遮盖."""
    span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
    return max(span * 0.0012, 1.0)


# -------------------------- 命令行入口 --------------------------
def main():
    parser = argparse.ArgumentParser(
        description='DSM 三维高程可视化 (PyVista 真实深度渲染)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visDSM.py
  python visDSM.py --dsm ../data/SF_Downtown.tif
  python visDSM.py --z-exag 2.0 --dpi 300

输出:
  DSM_3D_main_view.png
""")
    parser.add_argument('--dsm', default=DEFAULT_DSM,
                        help=f'DSM GeoTIFF 路径 (默认: {DEFAULT_DSM})')
    parser.add_argument('--dpi', type=int, default=200,
                        help='输出分辨率 (默认: 200)')
    parser.add_argument('--z-exag', type=float, default=1.5,
                        help='Z 轴垂直夸张系数 (默认: 1.5)')
    parser.add_argument('--output-dir', default=None,
                        help=f'输出目录 (默认: {OUTPUT_DIR})')
    args = parser.parse_args()

    visualize_paths_on_dsm(
        dsm_path=args.dsm,
        path_dir=None,
        dpi=args.dpi,
        z_exaggeration=args.z_exag,
        output_dir=args.output_dir,
        output_name='DSM_3D',
    )


if __name__ == '__main__':
    main()