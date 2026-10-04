# -*- coding: utf-8 -*-
"""
visWind.py — 解析 wind_20260102.nc (ERA5 风场) 并在 SF_Downtown.tif 地理范围内
渲染各气压层风场图（每层一张）。

数据说明:
  - wind_20260102.nc: ERA5 再分析风场 (GRIB→netCDF4/HDF5), 变量 u/v (m/s),
    维度 [valid_time × pressure_level × latitude × longitude],
    3×3 网格 (lat 37.5~38, lon -122.5~-122, 0.25° 步长), 4 个气压层
    (1000/975/950/925 hPa), 6 个时次 (逐小时)。
  - SF_Downtown.tif: 旧金山市区 DSM (EPSG:7131), 作为底图与裁剪范围。

读取方案:
  - 环境无 netCDF4/h5py, 使用 rasterio (GDAL netCDF 驱动) 打开子数据集
    netcdf:"<nc>":u / :v, 按 band 标签 NETCDF_DIM_valid_time /
    NETCDF_DIM_pressure_level 解析出 [time, level, lat, lon] 四维数组。

渲染方案 (每个气压层一张):
  - DSM 山体阴影灰度底图 (LightSource)
  - 风速 sqrt(u²+v²) 半透明填色 (全层统一色标, 便于层间对比)
  - 风羽图 barbs (气象站图 WMO 惯例: 杆指向风的来向;
    旗=50节, 长羽=10节, 短羽=5节, 1 m/s ≈ 1.944 节;
    白色粗羽描边 + 黑色细羽, 保证彩色底图上可读)
  - 标题标注气压层 + 标准大气近似高度

用法:
  python visWind.py                        # 默认 6 个时次平均
  python visWind.py --time 0               # 仅用第 1 个时次 (0-based)
  python visWind.py --time avg --dpi 300

依赖: numpy matplotlib scipy rasterio (UAV_PathPlanning venv 均已具备)

输出: ../analysis/风场_1000hPa.png 等 4 张
"""

import os
import sys
import argparse
import warnings
from datetime import datetime, timezone, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import rasterio
from rasterio.warp import transform_bounds
from scipy.interpolate import LinearNDInterpolator

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NC_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'wind_20260102.nc'))
DSM_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'SF_Downtown.tif'))
OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'analysis'))


# ----------------------------------------------------------------------
# 1. 解析 netCDF 风场
# ----------------------------------------------------------------------
def read_wind_nc(nc_path):
    """
    通过 rasterio(GDAL netCDF 驱动) 读取 u/v 子数据集。

    返回:
      times   : 排序后的 Unix 时间戳列表
      levels  : 排序后的气压层列表 (hPa, 降序: 1000→925)
      u, v    : ndarray [time, level, lat, lon]
      lat, lon: 一维坐标数组 (像元中心, 度)
    """
    data = {}  # (t, lev) -> (u2d, v2d)
    lats, lons = None, None

    for var in ('u', 'v'):
        sub = f'netcdf:"{nc_path}":{var}'
        with rasterio.open(sub) as src:
            if lats is None:
                # 像元中心坐标: GDAL transform (origin_x, 0.25, 0, origin_y, 0, -0.25)
                tf = src.transform
                ny, nx = src.shape
                lons = np.array([tf.c + (j + 0.5) * tf.a for j in range(nx)])
                lats = np.array([tf.f + (i + 0.5) * tf.e for i in range(ny)])
            for b in range(1, src.count + 1):
                tg = src.tags(b)
                t = int(tg['NETCDF_DIM_valid_time'])
                lev = int(tg['NETCDF_DIM_pressure_level'])
                grid = src.read(b).astype(np.float64)
                grid[grid > 1e30] = np.nan          # GRIB missingValue
                if (t, lev) not in data:
                    data[(t, lev)] = {}
                data[(t, lev)][var] = grid

    times = sorted({k[0] for k in data})
    levels = sorted({k[1] for k in data}, reverse=True)  # 1000 → 925 (自下而上)
    nt, nl = len(times), len(levels)

    u = np.full((nt, nl, 3, 3), np.nan)
    v = np.full((nt, nl, 3, 3), np.nan)
    for (t, lev), d in data.items():
        it, il = times.index(t), levels.index(lev)
        u[it, il] = d['u']
        v[it, il] = d['v']

    print(f"[nc] 时次 {nt} 个 | 气压层 {nl} 个: {levels} | 网格 lat {lats[0]:.2f}~{lats[-1]:.2f}, "
          f"lon {lons[0]:.2f}~{lons[-1]:.2f}")
    return times, levels, u, v, lats, lons


def pressure_to_altitude(p_hpa):
    """标准大气: 气压(hPa) → 近似海拔(m)"""
    return 44330.0 * (1.0 - (p_hpa / 1013.25) ** 0.19)


# ----------------------------------------------------------------------
# 2. SF_Downtown.tif 范围与 DSM
# ----------------------------------------------------------------------
def read_dsm(dsm_path):
    """读取 DSM 及其 WGS84 经纬度范围。"""
    with rasterio.open(dsm_path) as src:
        dem = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan
        b4326 = transform_bounds(src.crs, 'EPSG:4326', *src.bounds, densify_pts=21)
    left, bottom, right, top = b4326
    print(f"[dsm] shape={dem.shape} WGS84 范围: lon {left:.5f}~{right:.5f}, "
          f"lat {bottom:.5f}~{top:.5f}")
    return dem, left, right, bottom, top


# ----------------------------------------------------------------------
# 3. 风场插值 (3×3 → SF 细网格)
# ----------------------------------------------------------------------
def interp_wind(u2d, v2d, lats, lons, lat_grid, lon_grid):
    """3×3 源网格双线性插值到目标经纬度网格。"""
    src_lat, src_lon = np.meshgrid(lats, lons, indexing='ij')
    pts = np.column_stack([src_lat.ravel(), src_lon.ravel()])
    tgt_lat, tgt_lon = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    tgt = np.column_stack([tgt_lat.ravel(), tgt_lon.ravel()])

    fu = LinearNDInterpolator(pts, u2d.ravel())(tgt).reshape(tgt_lat.shape)
    fv = LinearNDInterpolator(pts, v2d.ravel())(tgt).reshape(tgt_lat.shape)
    return fu, fv


# ----------------------------------------------------------------------
# 4. 渲染单个气压层
# ----------------------------------------------------------------------
def render_level(lev, u2d, v2d, lats, lons, dem, extent, vmax, out_path,
                 time_label, dpi):
    left, right, bottom, top = extent
    lat_mid = 0.5 * (bottom + top)

    # 目标网格 (~0.001° ≈ 100m)
    lon_grid = np.arange(left, right + 1e-9, 0.001)
    lat_grid = np.arange(bottom, top + 1e-9, 0.001)
    fu, fv = interp_wind(u2d, v2d, lats, lons, lat_grid, lon_grid)
    speed = np.hypot(fu, fv)

    fig, ax = plt.subplots(figsize=(10, 9), dpi=dpi)

    # 底图: DSM 山体阴影
    ls = LightSource(azdeg=315, altdeg=55)
    shade = ls.hillshade(np.nan_to_num(dem, nan=0.0), vert_exag=0.02,
                         dx=2, dy=2)
    ax.imshow(shade, extent=[left, right, bottom, top], origin='upper',
              cmap='gray', vmin=0, vmax=1, zorder=1, interpolation='bilinear')

    # 风速填色 (半透明)
    im = ax.pcolormesh(lon_grid, lat_grid, speed, cmap='turbo', alpha=0.62,
                       vmin=0, vmax=vmax, shading='auto', zorder=2)

    # 风羽图 (气象站图 WMO 惯例: 杆指向风的来向, 单位: 节)
    # u/v 取负 → 杆指向来向; 旗=50节 长羽=10节 短羽=5节
    knot = 1.943844
    step = 4                                    # 抽样间隔 (风羽占位大, 需更稀)
    blon = lon_grid[::step]
    blat = lat_grid[::step]
    bu = -fu[::step, ::step] * knot
    bv = -fv[::step, ::step] * knot
    ax.barbs(blon, blat, bu, bv, length=6.2, linewidth=3.0,
             color='white', alpha=0.9, zorder=3)          # 白色描边
    ax.barbs(blon, blat, bu, bv, length=6.2, linewidth=1.15,
             color='black', zorder=3.1)                   # 黑色风羽

    # 比例与标注
    alt = pressure_to_altitude(lev)
    ax.set_title(f'旧金山市区风场 — {lev} hPa (约 {alt:.0f} m)\n'
                 f'ERA5 2026-01-02 · {time_label}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('经度 (°E)', fontsize=10)
    ax.set_ylabel('纬度 (°N)', fontsize=10)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(lat_mid)))   # 经纬度等比校正
    ax.grid(color='white', linestyle=':', linewidth=0.4, alpha=0.5, zorder=4)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label('风速 (m/s)', fontsize=10)

    # 简易指北针
    ax.annotate('N', xy=(0.955, 0.90), xycoords='axes fraction',
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='circle,pad=0.28', fc='#333', ec='white',
                          alpha=0.85, lw=1.2))
    ax.annotate('', xy=(0.955, 0.955), xytext=(0.955, 0.865),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='white', lw=1.6))

    ax.text(0.02, 0.02,
            '风羽: 旗=50节 长羽=10节 短羽=5节 (杆指来向)\n'
            '填色: 风速 m/s | 底图: SF_Downtown DSM 山体阴影',
            transform=ax.transAxes, fontsize=8, color='white', alpha=0.92,
            va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='#333', ec='none', alpha=0.6))

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"[out] {out_path}  (风速 {np.nanmin(speed):.2f}~{np.nanmax(speed):.2f} m/s)")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='ERA5 风场分气压层可视化 (SF Downtown 范围)')
    parser.add_argument('--nc', default=NC_PATH, help='netCDF 风场文件路径')
    parser.add_argument('--dsm', default=DSM_PATH, help='DSM GeoTIFF 路径 (范围+底图)')
    parser.add_argument('--out', default=OUT_DIR, help='输出目录')
    parser.add_argument('--time', default='avg',
                        help="时次选择: 'avg' 全部时次平均, 或时次索引 (0-based)")
    parser.add_argument('--dpi', type=int, default=200, help='输出 DPI')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 读取数据
    times, levels, u, v, lats, lons = read_wind_nc(args.nc)
    dem, left, right, bottom, top = read_dsm(args.dsm)

    # 范围检查: SF 范围必须落在风场网格内
    if not (lats.min() <= bottom and top <= lats.max()
            and lons.min() <= left and right <= lons.max()):
        print('[warn] SF 范围未完全落入风场网格, 边缘将外推受限!')

    # 时次选择
    tz_cn = timezone(timedelta(hours=8))
    t_strs = [datetime.fromtimestamp(t, tz=tz_cn).strftime('%H:%M') for t in times]
    if args.time.lower() == 'avg':
        u_sel = u.mean(axis=0)     # [level, lat, lon]
        v_sel = v.mean(axis=0)
        time_label = f'{len(times)} 个时次平均 ({t_strs[0]}–{t_strs[-1]} 北京时间)'
    else:
        it = int(args.time)
        if not (0 <= it < len(times)):
            sys.exit(f'[err] 时次索引 {it} 超出范围 0~{len(times) - 1}')
        u_sel, v_sel = u[it], v[it]
        time_label = f'北京时间 {t_strs[it]}'
    print(f"[time] {time_label}")

    # 统一色标上限 (各层对比)
    vmax = float(np.nanmax(np.hypot(u_sel, v_sel)))
    vmax = np.ceil(vmax) if vmax > 1 else 1.0

    # 每个气压层渲染一张
    for il, lev in enumerate(levels):
        out_path = os.path.join(args.out, f'风场_{lev}hPa.png')
        render_level(lev, u_sel[il], v_sel[il], lats, lons, dem,
                     (left, right, bottom, top), vmax, out_path,
                     time_label, args.dpi)

    print('[done] 全部完成')


if __name__ == '__main__':
    main()
