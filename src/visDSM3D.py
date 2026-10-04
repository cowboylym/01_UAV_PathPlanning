#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DSM 三维高程可视化 — 论文插图生成
================================================
功能:
  1. 读取 GeoTIFF DSM (支持 PIL / rasterio / GDAL 三种后端, 自动检测)
  2. 解析 GeoTIFF 仿射变换, 生成真实地理坐标 XYZ 网格 (不降采样)
  3. 三维着色渲染 (高程映射颜色 + 山体阴影 hillshade 增强立体感)
  4. 添加 XYZ 坐标轴 + 高程色标, 输出高分辨率 PNG

配色:
  - 默认 viridis (感知均匀, Nature/Science 等顶刊常用, 色盲友好)
  - 可选 plasma / magma / terrain / cividis
  - 自定义 hypsometric (蓝-绿-棕-白) 高程分层着色

依赖:
  pip install numpy matplotlib pillow
  (可选) pip install rasterio   # 更精确的 GeoTIFF 读取

用法:
  python visDSM3D.py
  python visDSM3D.py --file ./data/SF_Downtown.tif --cmap viridis
  python visDSM3D.py --cmap terrain --no-shade --dpi 600
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource, Normalize, LinearSegmentedColormap
import matplotlib.ticker as mticker

# -------------------------- 全局样式配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# 输出目录 (与 Analysis.py 一致)
OUTPUT_DIR = './analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 默认输入文件 (相对 src 目录, 数据在 ./data)
DEFAULT_DSM = './data/SF_Downtown.tif'


# -------------------------- GeoTIFF 读取 --------------------------
def read_geotiff(filepath):
    """
    读取 GeoTIFF, 返回 (elevation, geo_transform, nodata, crs_name)
    优先使用 rasterio, 其次 GDAL, 最后 PIL + 手动解析 GeoTIFF tags.
    geo_transform = (origin_x, pixel_w, 0, origin_y, 0, pixel_h)
    """
    # --- 后端 1: rasterio (最精确) ---
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            elev = src.read(1).astype(np.float32)
            gt = src.transform          # Affine
            nodata = src.nodata
            crs = src.crs.to_string() if src.crs else 'unknown'
        # 转为 (origin_x, pixel_w, 0, origin_y, 0, pixel_h)
        geo = (gt.c, gt.a, gt.b, gt.f, gt.d, gt.e)
        print(f"[rasterio] 读取成功 | shape={elev.shape} crs={crs}")
        return elev, geo, nodata, crs
    except ImportError:
        pass

    # --- 后端 2: GDAL ---
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

    # --- 后端 3: PIL + 手动解析 GeoTIFF tags ---
    from PIL import Image
    img = Image.open(filepath)
    elev = np.array(img).astype(np.float32)

    # 解析 GeoTIFF tags
    tags = img.tag_v2 if hasattr(img, 'tag_v2') else {}

    # ModelPixelScaleTag (33550): (ScaleX, ScaleY, ScaleZ)
    scale = tags.get(33550, (1.0, 1.0, 1.0))
    pixel_w = abs(scale[0])
    pixel_h = abs(scale[1])

    # ModelTiepointTag (33922): (I, J, K, X, Y, Z) — 像素(I,J)对应地理(X,Y)
    tiepoint = tags.get(33922, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    origin_x = tiepoint[3]
    origin_y = tiepoint[4]

    # NoData (Tag 42113 = GDAL_NODATA, PIL 读出为 ASCII 字符串, 需转 float)
    nodata_raw = tags.get(42113, None)
    try:
        nodata = float(nodata_raw) if nodata_raw is not None else None
    except (ValueError, TypeError):
        nodata = None

    # 坐标系名 (Tag 34737 = GeoAsciiParamsTag)
    crs = 'unknown'
    if 34737 in tags:
        crs = str(tags[34737]).split('|')[0].strip() or 'unknown'

    # GDAL GeoTransform: (origin_x, pixel_w, 0, origin_y, 0, -pixel_h)
    # 注: GDAL 约定 pixel_h 为负 (Y 轴向下), 这里转成 GDAL 标准格式
    geo = (origin_x, pixel_w, 0.0, origin_y, 0.0, -pixel_h)
    print(f"[PIL] 读取成功 | shape={elev.shape} pixel=({pixel_w}m, {pixel_h}m) crs={crs[:40]}")
    return elev, geo, nodata, crs


# -------------------------- 自定义高程配色 --------------------------
def make_hypsometric_cmap():
    """
    高程分层着色 (hypsometric tint): 深蓝-浅蓝-绿-黄绿-棕-白
    经典地形学配色, 适合展示从低地到山峰的过渡.
    """
    colors = [
        '#313695',  # 深蓝 (低地/水面)
        '#4575b4',  # 蓝
        '#74add1',  # 浅蓝
        '#abd9e9',  # 淡蓝
        '#fee090',  # 淡黄 (低地)
        '#fdae61',  # 橙黄 (丘陵)
        '#f46d43',  # 橙红 (山地)
        '#a50026',  # 深红 (高山)
        '#ffffff',  # 白 (雪线)
    ]
    return LinearSegmentedColormap.from_list('hypsometric', colors, N=256)


# -------------------------- 主可视化函数 --------------------------
def visualize_dsm(filepath, cmap_name='viridis', use_shade=True,
                  dpi=200, elev_angle=35, azim_angle=315, z_exaggeration=1.5,
                  output_prefix='DSM_3D', output_dir=None):
    """
    DSM 三维高程可视化

    参数:
        filepath: GeoTIFF 路径
        cmap_name: 配色名称 (viridis/plasma/magma/terrain/cividis/hypsometric)
        use_shade: 是否启用山体阴影 (hillshade) 增强立体感
        dpi: 输出分辨率
        elev_angle: 仰角 (度)
        azim_angle: 方位角 (度)
        z_exaggeration: Z 轴垂直夸张系数 (地形通常需放大 1.5~3 倍以凸显起伏)
        output_prefix: 输出文件前缀
    """
    print("=" * 60)
    print("DSM 三维高程可视化 — 论文插图生成")
    print("=" * 60)

    # 输出目录 (默认 ./analysis, 可通过参数覆盖)
    out_dir = output_dir if output_dir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. 读取数据 ----
    elev, geo, nodata, crs = read_geotiff(filepath)
    origin_x, pixel_w, _, origin_y, _, pixel_h = geo
    pixel_h = abs(pixel_h)  # 取绝对值, 统一处理

    rows, cols = elev.shape
    print(f"\n[数据] 尺寸: {rows} x {cols} 像素")
    print(f"[数据] 分辨率: {pixel_w}m x {pixel_h}m")
    print(f"[数据] 坐标系: {crs}")

    # ---- 2. 处理 NoData ----
    if nodata is not None:
        nodata_mask = (np.abs(elev - nodata) < 1e-3) | ~np.isfinite(elev)
    else:
        nodata_mask = ~np.isfinite(elev)

    elev_clean = elev.copy()
    if nodata_mask.any():
        # 用有效值最小值填充 NoData, 避免渲染异常; 后续遮罩处理
        valid_min = np.nanmin(elev_clean[~nodata_mask])
        elev_clean[nodata_mask] = valid_min
        print(f"[数据] NoData 像素: {nodata_mask.sum()} 个 ({100*nodata_mask.mean():.2f}%), 已填充")

    z_min, z_max = np.nanmin(elev_clean), np.nanmax(elev_clean)
    print(f"[数据] 高程范围: {z_min:.2f} ~ {z_max:.2f} m")

    # ---- 3. 生成地理坐标网格 (不降采样, 全分辨率) ----
    # X 轴: 列方向 (东向, 递增)
    # Y 轴: 行方向 (北向, 递减 → 顶部 Y 大, 底部 Y 小)
    x_coords = origin_x + np.arange(cols) * pixel_w
    y_coords_top = origin_y - np.arange(rows) * pixel_h  # 顶部 Y 大

    X, Y = np.meshgrid(x_coords, y_coords_top)
    Z = elev_clean * z_exaggeration  # 垂直夸张

    print(f"[网格] X 范围: {x_coords.min():.1f} ~ {x_coords.max():.1f} m")
    print(f"[网格] Y 范围: {y_coords_top.min():.1f} ~ {y_coords_top.max():.1f} m")
    print(f"[网格] 总网格点: {rows*cols:,} (全分辨率, 未降采样)")

    # ---- 4. 配色 ----
    if cmap_name == 'hypsometric':
        cmap = make_hypsometric_cmap()
        cmap_label = '高程 (m)'
    else:
        cmap = plt.get_cmap(cmap_name)
        cmap_label = '高程 (m)'

    norm = Normalize(vmin=z_min, vmax=z_max)

    # ---- 5. 山体阴影 (hillshade) ----
    if use_shade:
        # LightSource 计算阴影, 增强地形立体感 (Nature 系列地形图标准做法)
        ls = LightSource(azdeg=315, altdeg=45)
        # rgb 阵列: 颜色 × 阴影系数
        rgb = ls.shade(elev_clean, cmap=cmap, blend_mode='soft',
                       vert_exag=z_exaggeration, dx=pixel_w, dy=pixel_h)
        facecolors = rgb
        print("[渲染] 启用山体阴影 (hillshade, soft blend)")
    else:
        facecolors = cmap(norm(elev_clean))
        print("[渲染] 纯高程着色 (无阴影)")

    # ---- 6. 绘图 ----
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 全分辨率 surface (rstride=1, cstride=1 不降采样)
    surf = ax.plot_surface(X, Y, Z, facecolors=facecolors,
                           rstride=1, cstride=1,
                           linewidth=0, antialiased=True,
                           shade=False)

    # ---- 7. 坐标轴设置 ----
    ax.set_xlabel('东向 X (m)', labelpad=12, fontsize=11)
    ax.set_ylabel('北向 Y (m)', labelpad=12, fontsize=11)
    ax.set_zlabel(f'高程 Z (m) ×{z_exaggeration}', labelpad=8, fontsize=11)

    # 轴刻度格式化 (千位分隔, 避免数字过长)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.zaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/z_exaggeration:.0f}'))

    # 刻度密度 (避免过密)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(8))

    # 视角
    ax.view_init(elev=elev_angle, azim=azim_angle)

    # 背景与网格
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3, linestyle='--')

    # ---- 8. 色标 ----
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array(elev_clean)
    cbar = fig.colorbar(m, ax=ax, shrink=0.5, aspect=20, pad=0.1,
                        label=cmap_label)
    cbar.ax.tick_params(labelsize=9)

    # ---- 9. 保存 ----
    plt.tight_layout()
    png_path = os.path.join(out_dir, f'{output_prefix}_高程可视化.png')
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
    print(f"\n[输出] PNG: {png_path} (dpi={dpi})")

    plt.close()
    print("\n可视化完成!")


# -------------------------- 命令行入口 --------------------------
def main():
    parser = argparse.ArgumentParser(
        description='DSM 三维高程可视化 (论文插图)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visDSM3D.py
  python visDSM3D.py --file ./data/SF_Downtown.tif --cmap viridis
  python visDSM3D.py --cmap hypsometric --no-shade --dpi 150
  python visDSM3D.py --z-exag 2.0 --elev 25 --azim 45
""")
    parser.add_argument('--file', default=DEFAULT_DSM,
                        help=f'DSM GeoTIFF 路径 (默认: {DEFAULT_DSM})')
    parser.add_argument('--cmap', default='viridis',
                        choices=['viridis', 'plasma', 'magma', 'terrain',
                                 'cividis', 'hypsometric'],
                        help='高程配色 (默认: viridis, 顶刊感知均匀配色)')
    parser.add_argument('--no-shade', action='store_true',
                        help='禁用山体阴影 (默认启用)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='输出分辨率 (默认: 200, surface 栅格化 dpi)')
    parser.add_argument('--elev', type=float, default=35,
                        help='视角仰角 (度, 默认: 35)')
    parser.add_argument('--azim', type=float, default=315,
                        help='视角方位角 (度, 默认: 315)')
    parser.add_argument('--z-exag', type=float, default=1.5,
                        help='Z 轴垂直夸张系数 (默认: 1.5)')
    parser.add_argument('--prefix', default='DSM_3D',
                        help='输出文件前缀 (默认: DSM_3D)')
    parser.add_argument('--output-dir', default=None,
                        help=f'输出目录 (默认: {OUTPUT_DIR})')
    args = parser.parse_args()

    visualize_dsm(
        filepath=args.file,
        cmap_name=args.cmap,
        use_shade=not args.no_shade,
        dpi=args.dpi,
        elev_angle=args.elev,
        azim_angle=args.azim,
        z_exaggeration=args.z_exag,
        output_prefix=args.prefix,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
