# -*- coding: utf-8 -*-
# ==============================================================
# DOM (Digital Orthophoto Map) GeoTIFF 可视化
#   读取正射影像, 叠加经纬度坐标轴/比例尺/指北针, 输出论文用图.
#   支持 rasterio / GDAL / PIL 三后端读取.
#
# 用法:
#   python visDOM.py                          # 默认 ./data/DOM.tif
#   python visDOM.py --file ./data/DOM.tif
#   python visDOM.py --file ./data/DOM.tif --dpi 400 --crs wgs84
#
# 依赖: numpy matplotlib (rasterio 或 GDAL 至少一个)
# ==============================================================
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.scale import LinearScale
from matplotlib.transforms import Affine2D

# -------------------------- 全局配置 --------------------------
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.8

OUTPUT_DIR = './analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAVE_DPI = 300


# -------------------------- GeoTIFF 读取 --------------------------
def read_dom_geotiff(filepath):
    """
    读取 DOM GeoTIFF, 返回 (image_rgb, geo_transform, crs_name, width, height)
    image_rgb: (H, W, 3) uint8
    geo_transform: (origin_x, pixel_w, 0, origin_y, 0, pixel_h)  [pixel_h 为负]
    crs_name: 坐标系名称字符串
    """
    # --- 后端 1: rasterio ---
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            bands = src.count
            if bands >= 3:
                r = src.read(1)
                g = src.read(2)
                b = src.read(3)
                img = np.dstack([r, g, b])
            else:
                gray = src.read(1)
                img = np.dstack([gray, gray, gray])

            # 归一化到 0-255 uint8
            if img.dtype != np.uint8:
                vmin, vmax = np.nanpercentile(img, [2, 98])
                img = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

            gt = src.transform
            geo = (gt.c, gt.a, gt.b, gt.f, gt.d, gt.e)
            crs = src.crs.to_string() if src.crs else 'unknown'
            w, h = src.width, src.height

        print(f"[rasterio] 读取成功 | shape={img.shape} crs={crs} size={w}x{h}")
        return img, geo, crs, w, h
    except ImportError:
        pass

    # --- 后端 2: GDAL ---
    try:
        from osgeo import gdal
        ds = gdal.Open(filepath)
        bands = ds.RasterCount
        if bands >= 3:
            r = ds.GetRasterBand(1).ReadAsArray()
            g = ds.GetRasterBand(2).ReadAsArray()
            b = ds.GetRasterBand(3).ReadAsArray()
            img = np.dstack([r, g, b])
        else:
            gray = ds.GetRasterBand(1).ReadAsArray()
            img = np.dstack([gray, gray, gray])

        if img.dtype != np.uint8:
            vmin, vmax = np.nanpercentile(img, [2, 98])
            img = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

        geo = ds.GetGeoTransform()
        proj = ds.GetProjection()
        crs = proj.split('"')[-2] if 'PROJCS' in proj or 'GEOGCS' in proj else 'unknown'
        w, h = ds.RasterXSize, ds.RasterYSize

        print(f"[GDAL] 读取成功 | shape={img.shape} crs={crs[:40]} size={w}x{h}")
        return img, geo, crs, w, h
    except ImportError:
        pass

    # --- 后端 3: PIL ---
    from PIL import Image
    pil_img = Image.open(filepath)
    img = np.array(pil_img.convert('RGB'))
    w, h = pil_img.size

    tags = pil_img.tag_v2 if hasattr(pil_img, 'tag_v2') else {}
    scale = tags.get(33550, (1.0, 1.0, 1.0))
    tiepoint = tags.get(33922, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    origin_x = tiepoint[3]
    origin_y = tiepoint[4]
    pixel_w = abs(scale[0])
    pixel_h = abs(scale[1])
    geo = (origin_x, pixel_w, 0.0, origin_y, 0.0, -pixel_h)
    crs = str(tags.get(34737, 'unknown')).split('|')[0].strip() or 'unknown'

    print(f"[PIL] 读取成功 | shape={img.shape} pixel=({pixel_w}m, {pixel_h}m) crs={crs[:40]}")
    return img, geo, crs, w, h


# -------------------------- 坐标转换 --------------------------
def try_transform_to_wgs84(geo, crs_str, width, height):
    """
    尝试将投影坐标转换为 WGS84 经纬度.
    返回 (lon_extent, lat_extent, success)
      lon_extent = (min_lon, max_lon)
      lat_extent = (min_lat, max_lat)
    若 CRS 不可解析或已是 WGS84, 返回 None.
    """
    origin_x, pixel_w, _, origin_y, _, pixel_h = geo
    x_min = origin_x
    x_max = origin_x + width * pixel_w
    y_max = origin_y  # pixel_h < 0, origin 在左上
    y_min = origin_y + height * pixel_h

    # 已经是 WGS84 (EPSG:4326)
    if '4326' in crs_str or 'WGS 84' in crs_str or 'GEOGCS' in crs_str.upper():
        return (x_min, x_max), (y_min, y_max), True

    # 尝试 pyproj
    try:
        from pyproj import Transformer, CRS
        crs_obj = CRS.from_string(crs_str) if crs_str != 'unknown' else None
        if crs_obj is not None and crs_obj.is_geographic:
            return (x_min, x_max), (y_min, y_max), True
        if crs_obj is not None:
            transformer = Transformer.from_crs(crs_obj, 'EPSG:4326', always_xy=True)
            lon_min, lat_min = transformer.transform(x_min, y_min)
            lon_max, lat_max = transformer.transform(x_max, y_max)
            return (min(lon_min, lon_max), max(lon_min, lon_max)), \
                   (min(lat_min, lat_max), max(lat_min, lat_max)), True
    except Exception:
        pass

    return None


# -------------------------- 比例尺计算 --------------------------
def compute_scale_bar(geo, ax_extent_lon, ax_extent_lat):
    """
    计算合适的比例尺长度.
    返回 (length_meters, label_text)
    """
    origin_x, pixel_w = geo[0], geo[1]
    # 影像宽度对应的地面距离 (米)
    # pixel_w 单位通常为米 (UTM) 或度 (WGS84)
    if abs(pixel_w) < 0.01:
        # 经纬度: 1度 ≈ 111320m
        total_width_m = abs(ax_extent_lon[1] - ax_extent_lon[0]) * 111320
    else:
        total_width_m = abs(pixel_w) * 1000  # 假设影像宽度约 1000 像素

    # 选择 1/4 影像宽度, 取整到好看的数字
    target = total_width_m * 0.25
    candidates = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    best = candidates[0]
    for c in candidates:
        if c <= target:
            best = c

    if best >= 1000:
        label = f'{best / 1000:.0f} km'
    else:
        label = f'{best} m'
    return best, label


# -------------------------- 主可视化函数 --------------------------
def visualize_dom(filepath, dpi=300, output_name='DOM_study_area.png', display_crs='auto'):
    """
    可视化 DOM 正射影像, 叠加地理坐标.

    参数:
        filepath: GeoTIFF 路径
        dpi: 输出分辨率
        output_name: 输出文件名
        display_crs: 'auto' (自动检测), 'wgs84' (强制经纬度), 'projected' (投影坐标)
    """
    print(f"\n{'=' * 60}")
    print(f"DOM 可视化: {filepath}")
    print(f"{'=' * 60}")

    img, geo, crs, w, h = read_dom_geotiff(filepath)
    origin_x, pixel_w, _, origin_y, _, pixel_h = geo

    # 计算四角坐标
    x_min = origin_x
    x_max = origin_x + w * pixel_w
    y_min = origin_y + h * pixel_h  # pixel_h < 0
    y_max = origin_y

    print(f"\n--- 基本信息---")
    print(f"  影像尺寸: {w} x {h} 像素")
    print(f"  像素分辨率: {abs(pixel_w):.4f} x {abs(pixel_h):.4f}")
    print(f"  坐标系: {crs}")
    print(f"  地理范围 X: [{x_min:.4f}, {x_max:.4f}]")
    print(f"  地理范围 Y: [{y_min:.4f}, {y_max:.4f}]")
    print(f"  影像宽度: {abs(x_max - x_min):.2f}")
    print(f"  影像高度: {abs(y_max - y_min):.2f}")

    # 尝试转换为 WGS84
    wgs84 = try_transform_to_wgs84(geo, crs, w, h)
    use_wgs84 = False
    if display_crs == 'wgs84' and wgs84 is not None:
        use_wgs84 = True
    elif display_crs == 'auto':
        if wgs84 is not None:
            use_wgs84 = True
        # 投影坐标范围很大 (UTM 米), 用经纬度更直观
        elif abs(x_max - x_min) > 100000:
            use_wgs84 = False  # 无法转换, 只能用投影坐标

    if use_wgs84 and wgs84 is not None:
        lon_extent, lat_extent = wgs84
        xlabel = 'Longitude'
        ylabel = 'Latitude'
        fmt_coord = '{:.6f}°'
        print(f"  WGS84 经度范围: [{lon_extent[0]:.6f}, {lon_extent[1]:.6f}]")
        print(f"  WGS84 纬度范围: [{lat_extent[0]:.6f}, {lat_extent[1]:.6f}]")
        extent = [lon_extent[0], lon_extent[1], lat_extent[0], lat_extent[1]]
    else:
        xlabel = f'X ({crs[:20]})'
        ylabel = f'Y ({crs[:20]})'
        fmt_coord = '{:.1f}'
        extent = [x_min, x_max, y_min, y_max]

    # -------------------------- 绘图 --------------------------
    fig, ax = plt.subplots(figsize=(10, 9))

    # 显示正射影像
    ax.imshow(img, extent=extent, origin='upper')

    # 坐标轴格式化
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')

    # 坐标刻度格式
    if use_wgs84:
        # 经纬度: 度分秒格式更论文风
        def fmt_lon(x, pos):
            if x < 0:
                return f'{abs(x):.4f}°W'
            elif x > 0:
                return f'{x:.4f}°E'
            return '0°'

        def fmt_lat(y, pos):
            if y < 0:
                return f'{abs(y):.4f}°S'
            elif y > 0:
                return f'{y:.4f}°N'
            return '0°'

        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))
    else:
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))

    ax.tick_params(axis='both', labelsize=9, direction='out', length=4, width=0.8)

    # 恢复顶部和右侧轴线 (地图图框需要完整边框)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    # -------------------------- 比例尺 --------------------------
    scale_len_m, scale_label = compute_scale_bar(geo, (extent[0], extent[1]), (extent[2], extent[3]))

    # 比例尺位置: 左下角
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    bar_x_start = extent[0] + x_range * 0.05
    bar_y = extent[2] + y_range * 0.05

    # 比例尺像素长度
    if use_wgs84:
        # 经度转米: 1度 ≈ 111320m (近似)
        bar_pixel_len = scale_len_m / 111320.0
    else:
        bar_pixel_len = scale_len_m / abs(pixel_w)

    # 绘制比例尺 (黑白交替条)
    bar_height = y_range * 0.01
    segments = [bar_pixel_len / 4] * 4
    colors_bar = ['black', 'white', 'black', 'white']
    x_cursor = bar_x_start
    for seg_len, seg_color in zip(segments, colors_bar):
        rect = mpatches.Rectangle(
            (x_cursor, bar_y), seg_len, bar_height,
            facecolor=seg_color, edgecolor='black', linewidth=0.8, zorder=10
        )
        ax.add_patch(rect)
        x_cursor += seg_len

    # 比例尺文字
    ax.text(
        bar_x_start + bar_pixel_len / 2, bar_y - bar_height * 2.5,
        f'0    {scale_label}',
        ha='center', va='top', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray'),
        zorder=11
    )

    # -------------------------- 指北针 --------------------------
    arrow_x = extent[0] + x_range * 0.92
    arrow_y_start = extent[2] + y_range * 0.88
    arrow_y_end = extent[2] + y_range * 0.96

    ax.annotate(
        'N',
        xy=(arrow_x, arrow_y_end),
        xytext=(arrow_x, arrow_y_start),
        arrowprops=dict(arrowstyle='->', color='black', lw=2),
        ha='center', va='bottom', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.2),
        zorder=12
    )

    # -------------------------- 信息标注框 --------------------------
    info_text = (
        f'Image size: {w} × {h} px\n'
        f'Resolution: {abs(pixel_w):.2f} × {abs(pixel_h):.2f}\n'
        f'Coordinate system: {crs[:30]}'
    )
    if use_wgs84:
        lon_c = (lon_extent[0] + lon_extent[1]) / 2
        lat_c = (lat_extent[0] + lat_extent[1]) / 2
        info_text += f'\nCenter: {lat_c:.4f}°N, {lon_c:.4f}°E'

    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes, fontsize=8, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'),
        zorder=11
    )

    # 网格线
    ax.grid(True, alpha=0.2, linestyle='--', color='gray', linewidth=0.5)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n已保存: {output_path} (DPI={dpi})")


# -------------------------- 命令行入口 --------------------------
def main():
    parser = argparse.ArgumentParser(description='DOM 正射影像可视化 (带经纬度坐标)')
    parser.add_argument('--file', default='./data/DOM.tif',
                        help='DOM GeoTIFF 路径 (默认: ./data/DOM.tif)')
    parser.add_argument('--dpi', type=int, default=SAVE_DPI,
                        help=f'输出分辨率 (默认: {SAVE_DPI})')
    parser.add_argument('--crs', default='auto', choices=['auto', 'wgs84', 'projected'],
                        help='坐标显示方式: auto(自动) / wgs84(经纬度) / projected(投影坐标)')
    parser.add_argument('--output', default='DOM_study_area.png',
                        help='输出文件名 (默认: DOM_study_area.png)')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"错误: 文件不存在: {args.file}")
        sys.exit(1)

    visualize_dom(args.file, dpi=args.dpi, output_name=args.output, display_crs=args.crs)


if __name__ == '__main__':
    main()
