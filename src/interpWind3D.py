# -*- coding: utf-8 -*-
"""
interpWind3D.py — 基于 ERA5 风场 + SF_Downtown 地表, 生成实验区三维插值风场 (NetCDF)

输入:
  - 风场: data/wind_20260102.nc (ERA5, GRIB→netCDF4/HDF5, 变量 u/v,
    维度 [valid_time × pressure_level × latitude × longitude], 0.25° 网格,
    6 个时次, 气压层 1000/975/950/925 hPa —— 仅使用 1000 与 975 hPa, 其余忽略)
  - 地表: data/SF_Downtown.tif (EPSG:7131, 2 m, DSM 含建筑, 作为 DEM 使用)

实验区: SF_Downtown.tif 全范围 (与 SF_Downtown_sdf.mhd 的 ESDF 体积一致:
        2074 m × 2056 m), 外加富余缓冲 (默认 20 m, --buffer 可调)。

插值方案:
  1) 水平: ERA5 3×3 网格 → 实验区 2 m 网格, 双线性 (RegularGridInterpolator)
  2) 垂直: z = DEM(x,y) + h_agl, h_agl ∈ [0, 340] m 间隔 2 m (171 层)
     标准大气层海拔: 1000 hPa ≈ 111 m ASL, 975 hPa ≈ 323 m ASL
     - 111 ≤ z ≤ 323 (两层间, 对数律插值):
         w(z) = w1000 + (w975 - w1000) · [ln z - ln 111] / [ln 323 - ln 111]
     - z < 111 (近地面外推):
         · DEM < 111 m:  w(z) = w1000 · ln((z-DEM)/z0) / ln((111-DEM)/z0), z0=0.1 m
           (h_agl=0 时 ln(0)→-inf, 截断 h 至 z0, 即地面风速 0)
         · DEM ≥ 111 m (1000 hPa 在地下): w(z) = w975 · (z/323)^0.2
           (数值保护: 若 111-DEM ≤ z0, 同样走幂律分支)
     - z > 323 (高于 975 hPa): w(z) = w975 · (z/323)^0.2
     v 分量与 u 完全独立、公式相同。
     遮挡置零: 网格点位于建筑内部或地表以下 (z ≤ DEM(x,y), 与 ESDF 障碍体
               判定一致) 时 u = v = 0; 整层遮挡的层 (当前构造下仅 h_agl = 0
               地表层, z = DEM 处处成立) 跳过插值且不写入文件,
               实际输出 h_agl 自 2 m 起。

输出 (每个时次一个文件, 文件名 UTC 时间):
  data/interpolated_wind_YYYYMMDD_HHMM_2m.nc  (NetCDF-3 CDF-2, float32, 无压缩)
  坐标变量: lon (1D, WGS84 度, 升序), lat (1D, WGS84 度, 南→北升序),
            height_agl (1D, m, 2~340 间隔 2, 整层遮挡的 0 m 层已剔除)
  数据变量: u, v  维度 (height_agl, lat, lon), 单位 m/s

用法 (默认 UAV_PathPlanning 虚拟环境):
  D:\\Study\\docker\\UAV_PathPlanning\\Scripts\\python.exe src\\interpWind3D.py
  可选: --time all|0..5   --buffer 20   --out ../data   --overwrite

注意:
  - 默认网格约 1057×1048×170 × 2 变量 × float32 ≈ 1.50 GB/时次, 6 个时次 ≈ 9.0 GB,
    处理时峰值内存 ≈ 2.5 GB; 可用 --time 0 只生成单个时次。
  - 依赖: numpy scipy rasterio (无需 netCDF4, 写出用 scipy.io.netcdf_file)
"""

import os
import sys
import math
import argparse
import warnings
import time as _t
from datetime import datetime, timezone, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as crs_transform
from scipy.interpolate import RegularGridInterpolator
from scipy.io import netcdf_file

# ---------------- 规格常量 ----------------
LEV_LOW_HPA, LEV_HIGH_HPA = 1000, 975      # 仅使用的两个气压层 (hPa)
Z_LOW_AS, Z_HIGH_AS = 111.0, 323.0         # 两层对应标准大气海拔 (m ASL)
Z0 = 0.1                                   # 地面粗糙度 (m)
ALPHA = 0.2                                # 幂律指数
H_AGL_MAX, DZ = 340.0, 2.0                 # AGL 上限与垂直间隔 (m)
BUFFER_M = 20.0                            # 实验区外扩富余 (m)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NC_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'wind_20260102.nc'))
DEM_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'SF_Downtown.tif'))
OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'data'))

TZ_CN = timezone(timedelta(hours=8))


def human_size(nbytes):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024 or unit == 'TB':
            return f'{nbytes:.2f} {unit}' if unit != 'B' else f'{int(nbytes)} B'
        nbytes /= 1024


# ----------------------------------------------------------------------
# 1. 解析 ERA5 风场 netCDF (rasterio/GDAL netCDF 驱动)
# ----------------------------------------------------------------------
def read_wind_nc(nc_path):
    """解析风场元数据与数据。

    返回 dict:
      times  升序 Unix 时间戳列表
      levels 降序气压层列表 (hPa)
      u, v   ndarray [nt, nl, nlat, nlon]  (nlat 北→南)
      lats   降序 (北→南), lons 升序 (西→东), 单位: 度
    """
    data = {}
    lats = lons = None

    for var in ('u', 'v'):
        sub = f'netcdf:"{nc_path}":{var}'
        with rasterio.open(sub) as src:
            if lats is None:
                tf = src.transform
                ny, nx = src.shape
                lons = tf.c + (np.arange(nx) + 0.5) * tf.a      # 像元中心, 升序
                lats = tf.f + (np.arange(ny) + 0.5) * tf.e      # 像元中心, 降序
            for b in range(1, src.count + 1):
                tg = src.tags(b)
                t = tg.get('NETCDF_DIM_valid_time')
                lev = tg.get('NETCDF_DIM_pressure_level')
                if t is None or lev is None:
                    continue
                g = src.read(b).astype(np.float64)
                g[g > 1e30] = np.nan                             # GRIB missingValue
                data.setdefault((int(t), int(lev)), {})[var] = g

    times = sorted({k[0] for k in data})
    levels = sorted({k[1] for k in data}, reverse=True)
    nt, nl = len(times), len(levels)
    ny, nx = len(lats), len(lons)

    u = np.full((nt, nl, ny, nx), np.nan)
    v = np.full((nt, nl, ny, nx), np.nan)
    for (t, lev), d in data.items():
        it, il = times.index(t), levels.index(lev)
        if 'u' in d:
            u[it, il] = d['u']
        if 'v' in d:
            v[it, il] = d['v']

    t_utc = [datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d %H:%MZ')
             for t in times]
    print(f'[wind] 变量 u/v | 时次 {nt} 个: {t_utc[0]} ~ {t_utc[-1]}')
    print(f'[wind] 气压层 {levels} hPa | 源网格 {ny}×{nx} '
          f'(lat {lats[-1]:.3f}~{lats[0]:.3f}, lon {lons[0]:.3f}~{lons[-1]:.3f}, 0.25°)')
    return dict(times=times, levels=levels, u=u, v=v, lats=lats, lons=lons)


# ----------------------------------------------------------------------
# 2. 解析 DEM (SF_Downtown.tif, EPSG:7131, 2 m)
# ----------------------------------------------------------------------
def read_dsm(dem_path):
    """返回 dem[ny,nx](北→南), 列中心 x_c(升序), 行中心 y_c(降序), transform, crs。"""
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        tf, crs, nodata = src.transform, src.crs, src.nodata
        b7131 = src.bounds

    x_c = tf.c + (np.arange(dem.shape[1]) + 0.5) * tf.a          # 升序 (西→东)
    y_c = tf.f + (np.arange(dem.shape[0]) + 0.5) * tf.e          # 降序 (北→南)

    n_bad = 0
    if nodata is not None:
        bad = np.isclose(dem, nodata)
        n_bad = int(bad.sum())
        dem[bad] = np.nan
    if np.isnan(dem).any():
        fill = float(np.nanmean(dem))
        dem = np.where(np.isnan(dem), fill, dem)
        print(f'[dem ] nodata {n_bad} 像元已用均值 {fill:.1f} m 填充')

    print(f'[dem ] shape={dem.shape} 分辨率 {abs(tf.a):.1f}×{abs(tf.e):.1f} m | '
          f'EPSG:{crs.to_epsg()} 范围 X {b7131.left:.0f}~{b7131.right:.0f}, '
          f'Y {b7131.bottom:.0f}~{b7131.top:.0f} | 高程 {dem.min():.1f}~{dem.max():.1f} m')
    return dem, x_c, y_c, tf, crs


# ----------------------------------------------------------------------
# 3. 垂直插值: 单个 AGL 高度层, 四分支公式 (u/v 通用)
# ----------------------------------------------------------------------
def wind_layer(w1000, w975, dem, h, z0=Z0, z_low=Z_LOW_AS, z_high=Z_HIGH_AS,
               alpha=ALPHA):
    """计算某 AGL 高度 h (m) 的风分量层。

    w1000/w975: 该层 1000/975 hPa 水平插值后的 2D 场
    dem:        地表海拔 2D (m ASL)
    返回:       2D 风分量 (m/s); 建筑内部/地表以下 (z ≤ dem) 置 0
    """
    z = dem + h                                                  # 海拔 ASL
    out = np.empty(z.shape, dtype=np.float64)
    ln_low, ln_high = math.log(z_low), math.log(z_high)

    # 分支 1: z_low <= z <= z_high —— 两层间对数律插值
    m_mid = (z >= z_low) & (z <= z_high)
    if m_mid.any():
        lz = np.log(z[m_mid])
        out[m_mid] = (w1000[m_mid]
                      + (w975[m_mid] - w1000[m_mid]) * (lz - ln_low) / (ln_high - ln_low))

    # 分支 2: z < z_low —— 近地面外推
    m_low = z < z_low
    m_low_log = m_low & ((z_low - dem) > z0)      # 对数律分母有效 (DEM 远低于 111 m)
    m_low_pow = m_low & ~m_low_log                # DEM ≥ ~111 m: 1000 hPa 在地下 → 幂律
    if m_low_log.any():
        # z - DEM = h; h=0 → 截断至 z0 (地面风速 0, 避免 ln(0)=-inf)
        num = math.log(max(h, z0) / z0)
        den = np.log((z_low - dem[m_low_log]) / z0)
        out[m_low_log] = w1000[m_low_log] * (num / den)
    # 分支 3 (与 2b 合并): 幂律 w = w975 · (z/z_high)^alpha
    m_pow = m_low_pow | (z > z_high)
    if m_pow.any():
        zc = np.maximum(z[m_pow], z0)                              # 数值保护
        out[m_pow] = w975[m_pow] * (zc / z_high) ** alpha

    # 遮挡置零: 建筑内部 / 地表以下 (z ≤ DEM, 与 ESDF 障碍体判定一致) → 风速 0
    # (整层遮挡层已在主流程剔除, 此掩码为防御性保留)
    solid = z <= dem
    if solid.any():
        out[solid] = 0.0
    return out


# ----------------------------------------------------------------------
# 4. 写单个时次的 NetCDF
# ----------------------------------------------------------------------
def write_netcdf(out_path, it, wind, u1000, u975, v1000, v975,
                 dem_asc, lon_asc, lat_asc, h_agl, x0, y0, dx, dy):
    t = wind['times'][it]
    t_utc = datetime.fromtimestamp(t, tz=timezone.utc)
    t_cn = datetime.fromtimestamp(t, tz=TZ_CN)

    nlev, nlat, nlon = len(h_agl), len(lat_asc), len(lon_asc)
    f = netcdf_file(out_path, 'w', version=2)      # CDF-2 (64-bit offset)

    f.createDimension('height_agl', nlev)
    f.createDimension('lat', nlat)
    f.createDimension('lon', nlon)

    vlon = f.createVariable('lon', 'f', ('lon',))
    vlon[:] = lon_asc.astype('f4')
    vlon.units = 'degrees_east'
    vlon.long_name = 'longitude (WGS84)'
    vlon.standard_name = 'longitude'

    vlat = f.createVariable('lat', 'f', ('lat',))
    vlat[:] = lat_asc.astype('f4')
    vlat.units = 'degrees_north'
    vlat.long_name = 'latitude (WGS84, ascending south to north)'
    vlat.standard_name = 'latitude'

    vh = f.createVariable('height_agl', 'f', ('height_agl',))
    vh[:] = h_agl.astype('f4')
    vh.units = 'm'
    vh.long_name = 'height above ground level'
    vh.positive = 'up'

    vu = f.createVariable('u', 'f', ('height_agl', 'lat', 'lon'))
    vu.units = 'm s-1'
    vu.long_name = 'zonal (eastward) wind'
    vu.standard_name = 'eastward_wind'

    vv = f.createVariable('v', 'f', ('height_agl', 'lat', 'lon'))
    vv.units = 'm s-1'
    vv.long_name = 'meridional (northward) wind'
    vv.standard_name = 'northward_wind'

    # 全局属性 (ASCII, NetCDF-3 属性编码限制)
    f.title = 'SF Downtown 3D interpolated wind field (2m grid, 0-340m AGL)'
    f.source = 'ERA5 wind (wind_20260102.nc) + SF_Downtown.tif DSM'
    f.history = (f'created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} by '
                 f'interpWind3D.py')
    f.time_utc = t_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    f.time_epoch = int(t)
    f.time_beijing = t_cn.strftime('%Y-%m-%d %H:%M:%S')
    f.pressure_levels_used_hpa = f'{LEV_LOW_HPA}, {LEV_HIGH_HPA}'
    f.level_altitudes_m_asl = f'1000hPa={Z_LOW_AS}, 975hPa={Z_HIGH_AS}'
    f.roughness_length_m = Z0
    f.power_law_alpha = ALPHA
    f.horizontal_interpolation = 'bilinear from ERA5 0.25deg regular grid'
    f.vertical_interpolation = ('log-law between 111-323m ASL; surface log profile '
                                'below 111m (z0=0.1m); power law (z/323)^0.2 above '
                                '323m and where 1000hPa is below ground')
    f.occlusion_rule = ('u=v=0 where z <= DEM(x,y), i.e. inside buildings or '
                        'below terrain (consistent with ESDF obstacle convention); '
                        'fully occluded layers are omitted from this file '
                        '(h_agl starts at 2 m since the 0 m layer lies at z=DEM)')
    f.dem_note = 'SF_Downtown.tif (DSM incl. buildings) used as ground elevation'
    # EPSG:7131 网格锚点: 风场网格本身构建在 7131 规则 2m 网格上 (DEM 像元+缓冲),
    # lon/lat 仅为 WGS84 元数据. C++ WindField (wind_field.hpp) 用以下属性直接索引:
    #   列 i ↔ x = x0 + i*dx (自西向东), 行 j ↔ y = y0 + j*dy (自南向北, 与 lat 升序一致)
    f.grid_crs = 'EPSG:7131'
    f.x_origin_7131 = float(x0)
    f.y_origin_7131 = float(y0)
    f.grid_dx_m = float(dx)
    f.grid_dy_m = float(dy)

    # 逐层计算并写入 (分片赋值, 控制峰值内存)
    u_min, u_max = np.inf, -np.inf
    v_min, v_max = np.inf, -np.inf
    for k, h in enumerate(h_agl):
        lu = wind_layer(u1000, u975, dem_asc, h)
        lv = wind_layer(v1000, v975, dem_asc, h)
        vu[k] = lu
        vv[k] = lv
        u_min = min(u_min, float(lu.min()))
        u_max = max(u_max, float(lu.max()))
        v_min = min(v_min, float(lv.min()))
        v_max = max(v_max, float(lv.max()))
        if (k + 1) % 25 == 0 or k == nlev - 1:
            print(f'    height_agl {h:5.0f} m  ({k + 1}/{nlev})', flush=True)
    f.close()

    print(f'    u ∈ [{u_min:.2f}, {u_max:.2f}]  v ∈ [{v_min:.2f}, {v_max:.2f}] m/s')
    return t_utc


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='实验区三维插值风场生成 (ERA5 + SF_Downtown DSM, 2m 网格, 0-340m AGL)')
    parser.add_argument('--nc', default=NC_PATH, help='ERA5 风场 netCDF 路径')
    parser.add_argument('--dem', default=DEM_PATH, help='DEM/DSM GeoTIFF 路径')
    parser.add_argument('--out', default=OUT_DIR, help='输出目录')
    parser.add_argument('--time', default='all',
                        help="时次: 'all' 全部, 或索引 0..N-1 (默认 all)")
    parser.add_argument('--buffer', type=float, default=BUFFER_M,
                        help=f'实验区外扩富余 (m, 默认 {BUFFER_M:.0f})')
    parser.add_argument('--hmax', type=float, default=H_AGL_MAX, help='AGL 上限 (m)')
    parser.add_argument('--dz', type=float, default=DZ, help='垂直间隔 (m)')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在文件')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- 解析输入元数据 ----
    wind = read_wind_nc(args.nc)
    dem, x_c, y_c, tf, crs = read_dsm(args.dem)

    # 气压层检查: 仅用 1000 与 975 hPa
    levels = wind['levels']
    missing = [lv for lv in (LEV_LOW_HPA, LEV_HIGH_HPA) if lv not in levels]
    if missing:
        sys.exit(f'[err] 风场缺少气压层 {missing} hPa, 现有: {levels}')
    ignored = [lv for lv in levels if lv not in (LEV_LOW_HPA, LEV_HIGH_HPA)]
    if ignored:
        print(f'[wind] 忽略气压层: {ignored} hPa (仅使用 {LEV_LOW_HPA}/{LEV_HIGH_HPA} hPa)')
    il_low, il_high = levels.index(LEV_LOW_HPA), levels.index(LEV_HIGH_HPA)

    # ---- 目标网格: DEM 2m 网格 + 富余缓冲 ----
    px = abs(tf.a)
    nb = int(round(args.buffer / px))
    if nb > 0:
        dx, dy = tf.a, tf.e
        x_e = np.concatenate([x_c[0] - dx * np.arange(nb, 0, -1), x_c,
                              x_c[-1] + dx * np.arange(1, nb + 1)])
        y_e = np.concatenate([y_c[0] - dy * np.arange(nb, 0, -1), y_c,
                              y_c[-1] + dy * np.arange(1, nb + 1)])
        dem_e = np.pad(dem, nb, mode='edge')
    else:
        x_e, y_e, dem_e = x_c, y_c, dem
    print(f'[grid] 实验区网格 {dem.shape[1]}×{dem.shape[0]} + 缓冲 {nb}px '
          f'({args.buffer:.0f} m) → {len(x_e)}×{len(y_e)}')

    # 网格中心转 WGS84 (7131 度带投影, 1D 近似分离, 2km 内误差 << 2 m)
    mid_x, mid_y = x_e[len(x_e) // 2], y_e[len(y_e) // 2]
    lon_asc, _ = crs_transform(crs, CRS.from_epsg(4326),
                               list(map(float, x_e)), [float(mid_y)] * len(x_e))
    _, lat_1 = crs_transform(crs, CRS.from_epsg(4326),
                             [float(mid_x)] * len(y_e), list(map(float, y_e)))
    lon_asc = np.asarray(lon_asc)                 # 升序 (西→东)
    lat_asc = np.asarray(lat_1)[::-1]             # 升序 (南→北)
    dem_asc = dem_e[::-1]                         # 行序与 lat_asc 对齐
    print(f'[grid] WGS84 范围: lon {lon_asc[0]:.5f}~{lon_asc[-1]:.5f}, '
          f'lat {lat_asc[0]:.5f}~{lat_asc[-1]:.5f}')

    # 范围检查: 目标区必须落在风场网格内
    if not (wind['lats'].min() <= lat_asc[0] and lat_asc[-1] <= wind['lats'].max()
            and wind['lons'].min() <= lon_asc[0] and lon_asc[-1] <= wind['lons'].max()):
        sys.exit('[err] 实验区超出风场网格范围!')

    # 垂直 AGL 层 + 整层地下/遮挡层剔除
    # 判定 (数据驱动): 层内所有网格点均满足遮挡条件 z = DEM + h ≤ DEM → 整层遮挡。
    # 当前构造下仅 h_agl = 0 命中 (z = DEM 处处成立, 即地表层);
    # 若垂直基准未来改为 DTM, 建筑体内的层会被此判定自动跳过。
    h_all = np.arange(0.0, args.hmax + 1e-9, args.dz)
    occl = np.array([(dem_asc + h <= dem_asc).all() for h in h_all])
    h_agl = h_all[~occl]
    if occl.any():
        skipped = ', '.join(f'{h:.0f}' for h in h_all[occl])
        print(f'[vert] 整层地下/遮挡层 [{skipped}] m: 跳过插值且不写入文件 '
              f'(z = DEM 处处成立)')
    if len(h_agl) == 0:
        sys.exit('[err] 所有垂直层均整层遮挡, 无可输出层')
    print(f'[vert] 保留 {len(h_agl)} 层: h_agl {h_agl[0]:.0f}~{h_agl[-1]:.0f} m, '
          f'间隔 {args.dz:.0f} m')
    nlev = len(h_agl)
    per_file = nlev * len(lat_asc) * len(lon_asc) * 4 * 2

    # 时次选择
    times = wind['times']
    if args.time.lower() == 'all':
        idx_list = list(range(len(times)))
    else:
        it = int(args.time)
        if not (0 <= it < len(times)):
            sys.exit(f'[err] --time 索引 {it} 超出 0..{len(times) - 1}')
        idx_list = [it]

    print(f'[size] 预计每文件 {human_size(per_file)} × {len(idx_list)} 个时次 '
          f'≈ {human_size(per_file * len(idx_list))} (float32, 无压缩)')
    print(f'[size] 处理期间峰值内存约 2.5 GB')

    # ---- 水平插值器 (ERA5 规则网格, 双线性; 源 lat 转升序) ----
    order = np.argsort(wind['lats'])
    lat_src = wind['lats'][order]
    lon_src = wind['lons']
    lat_g, lon_g = np.meshgrid(lat_asc, lon_asc, indexing='ij')
    tgt = np.column_stack([lat_g.ravel(), lon_g.ravel()])

    def hinterp(field):
        rgi = RegularGridInterpolator((lat_src, lon_src), field[order, :],
                                      method='linear', bounds_error=False,
                                      fill_value=None)
        return rgi(tgt).reshape(lat_g.shape)

    # ---- 逐时次生成 ----
    t0 = _t.perf_counter()
    for n, it in enumerate(idx_list, 1):
        t = times[it]
        t_utc = datetime.fromtimestamp(t, tz=timezone.utc)
        out_path = os.path.join(
            args.out, f'interpolated_wind_{t_utc:%Y%m%d_%H%M}_2m.nc')
        if os.path.exists(out_path) and not args.overwrite:
            print(f'[{n}/{len(idx_list)}] {os.path.basename(out_path)} 已存在, 跳过 '
                  f'(--overwrite 可覆盖)')
            continue
        print(f'[{n}/{len(idx_list)}] {os.path.basename(out_path)} '
              f'(UTC {t_utc:%Y-%m-%d %H:%M} / 北京 '
              f'{datetime.fromtimestamp(t, tz=TZ_CN):%m-%d %H:%M}) 计算中...', flush=True)

        u1000 = hinterp(wind['u'][it, il_low])
        u975 = hinterp(wind['u'][it, il_high])
        v1000 = hinterp(wind['v'][it, il_low])
        v975 = hinterp(wind['v'][it, il_high])

        write_netcdf(out_path, it, wind, u1000, u975, v1000, v975,
                     dem_asc, lon_asc, lat_asc, h_agl,
                     float(x_e[0]), float(y_e[-1]),
                     float(abs(tf.a)), float(abs(tf.e)))
        print(f'    写入完成: {out_path} ({human_size(os.path.getsize(out_path))}, '
              f'耗时 {_t.perf_counter() - t0:.0f}s)', flush=True)

    print('[done] 全部完成')


if __name__ == '__main__':
    main()
