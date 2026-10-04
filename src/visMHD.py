#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MHD 体数据 3D 体渲染 — 论文插图 (视角与 visDSM3D.py 一致)
================================================
渲染方式:
  - SDF:  VTK 光线投射体渲染 (能看到体素内部渐变)
  - Mask: VTK MarchingCubes 等值面
  - Cost: VTK 等值面

视角匹配 visDSM3D.py:
  1. 正交投影 (ParallelProjectionOn) — matplotlib 也是正交
  2. 相机角度与 matplotlib view_init(elev, azim) 约定一致
     (azim=0 → 从+X看, azim=90 → 从+Y看)

SDF 颜色/透明度传递函数:
  - 障碍内 (d<0):  红色,   不透明度高 (1.0)
  - 表面   (d=0):  红色,   不透明 0.95
  - 近障碍 (d=3):  橙色,   不透明 0.7
  - 过渡   (d=7):  黄色,   不透明 0.4
  - 安全   (d=10): 绿色,   不透明 0.2
  - 远处   (d>10): 绿色,   不透明 0.1 (不完全透明)

依赖: pip install numpy vtk matplotlib pillow

用法:
  python visMHD.py ../data/SF_Downtown_sdf.mhd
  python visMHD.py ../data/SF_Downtown_mask.mhd
  python visMHD.py ../data/SF_Downtown_sdf.mhd --shrink 10
"""

import os
import argparse
import math
import numpy as np
import vtk
from vtk.util import numpy_support
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.cm as cm
from PIL import Image

# -------------------------- 全局样式 --------------------------
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'SimHei',
                                   'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

OUTPUT_DIR = '../analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MET_TYPE_MAP = {
    'MET_FLOAT': np.float32, 'MET_DOUBLE': np.float64,
    'MET_UCHAR': np.uint8,   'MET_CHAR': np.int8,
    'MET_USHORT': np.uint16, 'MET_SHORT': np.int16,
    'MET_UINT': np.uint32,   'MET_INT': np.int32,
}

# ESDF 色标: 危险→安全渐变 (深红→火砖→橙→金黄→柠檬绿→深绿)
# 非线性映射: d≤0 同色, 0~10 剧变, d>10 微变
ESDF_CMAP = LinearSegmentedColormap.from_list(
    'esdf_danger',
    [(0.00, '#4A0E0E'),  # d≤0:  深红 (障碍, 同一颜色)
     (0.08, '#B22222'),  # d=0:  火砖红 (表面)
     (0.28, '#FF4500'),  # d=2.5: 橙红
     (0.48, '#FFA500'),  # d=5:  橙
     (0.68, '#FFD700'),  # d=7.5: 金黄
     (0.85, '#32CD32'),  # d=10: 柠檬绿 (安全边界)
     (0.93, '#228B22'),  # d>10: 森林绿 (微变)
     (1.00, '#006400')], # d>>10: 深绿 (安全, 微变)
)


class ESDFNorm(Normalize):
    """ESDF 非线性归一化.
    
    映射规则:
      d ≤ 0    → 0.0        (同一颜色: 深红, 障碍物)
      0 < d ≤ 10 → 0.08~0.85 (剧变: 占 77% 色域, 0~10m 细节鲜明)
      d > 10   → 0.85~1.0   (微变: 占 15% 色域, 安全区颜色趋同)
    """
    def __init__(self, d_min, d_max, clip=False):
        self._d_max = max(float(d_max), 20.0)
        super().__init__(vmin=float(d_min), vmax=float(d_max), clip=clip)

    def __call__(self, value, clip=None):
        d = np.ma.asanyarray(value, dtype=float)
        result = np.ma.zeros_like(d)
        # 障碍内部单独占用 0~0.08 色域，避免最小负值与 0 刻度重叠
        m0 = d <= 0
        if float(self.vmin) < 0:
            d_min = float(self.vmin)
            result[m0] = 0.08 * np.clip((d[m0] - d_min) / (-d_min), 0.0, 1.0)
        else:
            result[m0] = 0.0
        # 0 < d ≤ 10: 剧变区 (0.08 → 0.85)
        m1 = (d > 0) & (d <= 10)
        result[m1] = 0.08 + 0.77 * (d[m1] / 10.0)
        # d > 10: 微变区 (0.85 → 1.0)
        m2 = d > 10
        result[m2] = 0.85 + 0.15 * np.minimum(1.0, (d[m2] - 10.0) / (self._d_max - 10.0))
        if np.ma.is_masked(d):
            result.mask = d.mask
        return result

    def inverse(self, value):
        v = np.ma.asanyarray(value, dtype=float)
        result = np.ma.zeros_like(v)
        m1 = (v > 0.08) & (v <= 0.85)
        result[m1] = (v[m1] - 0.08) / 0.77 * 10.0
        m2 = v > 0.85
        result[m2] = 10.0 + (v[m2] - 0.85) / 0.15 * (self._d_max - 10.0)
        return result


# -------------------------- .mhd 解析 --------------------------
def parse_mhd(mhd_path):
    with open(mhd_path, 'r') as f:
        header = {}
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                header[key.strip()] = val.strip()
    return {
        'DimSize': tuple(int(x) for x in header['DimSize'].split()),
        'Spacing': tuple(float(x) for x in header['ElementSpacing'].split()),
        'Offset': tuple(float(x) for x in header['Offset'].split()),
        'dtype': MET_TYPE_MAP[header['ElementType']],
        'raw_path': os.path.join(os.path.dirname(os.path.abspath(mhd_path)),
                                 header['ElementDataFile']),
        'compressed': header.get('CompressedData', 'False').lower() == 'true',
    }


# -------------------------- 降采样读取 --------------------------
def load_downsampled(info, shrink):
    nx, ny, nz = info['DimSize']
    sx, sy, sz = info['Spacing']
    if not os.path.exists(info['raw_path']):
        raise FileNotFoundError(f".raw 不存在: {info['raw_path']}")
    arr = np.memmap(info['raw_path'], dtype=info['dtype'], mode='r', shape=(nz, ny, nx))
    sub = arr[::shrink, ::shrink, ::shrink].copy()
    nz2, ny2, nx2 = sub.shape
    sp = (sx * shrink, sy * shrink, sz * shrink)
    print(f"  降采样 shrink={shrink}: {nx}×{ny}×{nz} → {nx2}×{ny2}×{nz2} "
          f"({sp[0]:.1f}×{sp[1]:.1f}×{sp[2]:.1f}m, {sub.nbytes/1e6:.1f}MB)")
    return sub, (nx2, ny2, nz2), sp, info['Offset']


# -------------------------- numpy → vtkImageData --------------------------
def numpy_to_vtk_image(data, dims, spacing, origin):
    nx, ny, nz = dims
    flat = np.ascontiguousarray(data).ravel(order='C')
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True,
                                         array_type=numpy_support.get_vtk_array_type(data.dtype))
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.GetPointData().SetScalars(vtk_arr)
    return img


# -------------------------- 相机（与 DSM 三维主视角一致）--------------------------
def set_camera(renderer, bounds, elev=5, azim=0):
    """严格复现 visDSM.py 中 PyVista 主视角的相机操作顺序。"""
    cam = renderer.GetActiveCamera()

    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    xspan = bounds[1] - bounds[0]
    yspan = bounds[3] - bounds[2]
    zspan = bounds[5] - bounds[4]
    diag = math.sqrt(xspan**2 + yspan**2 + zspan**2)

    # PyVista Plotter 的默认相机为等轴测方向，reset_camera() 仅拟合场景。
    cam.SetFocalPoint(cx, cy, cz)
    cam.SetPosition(cx + diag, cy + diag, cz + diag)
    cam.SetViewUp(0, 0, 1)
    renderer.ResetCamera()
    cam.SetClippingRange(0.1, diag * 5.0)

    cam.ParallelProjectionOff()
    cam.Elevation(elev)
    cam.Azimuth(azim)
    cam.SetDistance(diag * 0.5)
    cam.Zoom(0.92)


# -------------------------- 坐标轴 --------------------------
def add_cube_axes(renderer, bounds):
    """采用与 DSM 主视角一致的 km 坐标轴样式。"""
    axes = vtk.vtkCubeAxesActor()
    axes.SetBounds(bounds)
    axes.SetCamera(renderer.GetActiveCamera())
    axes.SetXTitle('X (km)')
    axes.SetYTitle('Y (km)')
    axes.SetZTitle('Z (km)')
    axes.SetXAxisRange(bounds[0] / 1000.0, bounds[1] / 1000.0)
    axes.SetYAxisRange(bounds[2] / 1000.0, bounds[3] / 1000.0)
    axes.SetZAxisRange(bounds[4] / 1000.0, bounds[5] / 1000.0)
    axes.SetXLabelFormat('%.1f')
    axes.SetYLabelFormat('%.1f')
    axes.SetZLabelFormat('%.1f')
    axes.SetFlyModeToOuterEdges()
    axes.SetLabelOffset(14.0)
    axes.DrawXGridlinesOff()
    axes.DrawYGridlinesOff()
    axes.DrawZGridlinesOff()
    axes.XAxisMinorTickVisibilityOff()
    axes.YAxisMinorTickVisibilityOff()
    axes.ZAxisMinorTickVisibilityOff()
    for i in range(3):
        title_prop = axes.GetTitleTextProperty(i)
        label_prop = axes.GetLabelTextProperty(i)
        title_prop.SetColor(0, 0, 0)
        title_prop.SetFontSize(26)
        title_prop.SetBold(False)
        label_prop.SetColor(0.05, 0.05, 0.05)
        label_prop.SetFontSize(22)
        label_prop.SetBold(False)
    renderer.AddActor(axes)


# -------------------------- SDF 体渲染 --------------------------
def render_sdf_volume(img_data, bounds, output_path, dpi, elev, azim, dims, spacing,
                      data_min, data_max):
    """
    SDF 光线投射体渲染.
    颜色: 障碍红 → 橙 → 黄 → 安全绿 (0~10m 线性过渡, >10m 保持绿色)
    透明度: 近障碍不透明, 远处半透明 (最低 0.1, 不完全透明)

    注意: 传递函数必须覆盖 [data_min, data_max] 全范围, 否则 VTK 会裁剪
          超出范围的体素 (不渲染), 导致 d>50 的安全区消失.
    """
    # 颜色传递函数: 危险→安全渐变 (与三视图 ESDF_CMAP 一致)
    #   d ≤ 0:  深红 (同一颜色, 障碍物)
    #   0~10:   火砖红→橙→金黄→柠檬绿 (剧变, 占主要色域)
    #   d > 10: 柠檬绿→森林绿→深绿 (微变)
    ctf = vtk.vtkColorTransferFunction()
    # d ≤ 0 同色: 在 data_min 和 0 处都放深红
    ctf.AddRGBPoint(data_min, 0.290, 0.055, 0.055)  # #4A0E0E 深红
    ctf.AddRGBPoint(-1e-3,   0.290, 0.055, 0.055)  # #4A0E0E 深红 (d≤0 保持同色)
    ctf.AddRGBPoint(0.0,     0.698, 0.133, 0.133)  # #B22222 火砖红 (表面)
    ctf.AddRGBPoint(2.5,     1.000, 0.271, 0.000)  # #FF4500 橙红
    ctf.AddRGBPoint(5.0,     1.000, 0.647, 0.000)  # #FFA500 橙
    ctf.AddRGBPoint(7.5,     1.000, 0.843, 0.000)  # #FFD700 金黄
    ctf.AddRGBPoint(10.0,    0.196, 0.804, 0.196)  # #32CD32 柠檬绿 (安全边界)
    ctf.AddRGBPoint(30.0,    0.133, 0.545, 0.133)  # #228B22 森林绿 (微变)
    ctf.AddRGBPoint(data_max, 0.000, 0.392, 0.000) # #006400 深绿

    # 不透明度 (不用 GradientOpacity, 避免安全区被乘到几乎透明)
    # 单体素 α, 光线穿过 ~2000 层累积:
    #   障碍 α=0.06 → 50层累积 0.96 (不透明)
    #   表面 α=0.04 → 100层累积 0.98 (不透明)
    #   d=3  α=0.015 → 累积可观
    #   d=10 α=0.003 → 2000层累积 1-(0.997)^2000 ≈ 0.998? 太高!
    #   实际上安全区从 d=10 到 d_max 只有 370m, 光线穿过 ~370 层:
    #     α=0.001 → 1-(0.999)^370 ≈ 0.31 (半透明可看穿)
    otf = vtk.vtkPiecewiseFunction()
    otf.AddPoint(data_min, 0.06)   # 障碍内
    otf.AddPoint(0.0,     0.04)    # 表面
    otf.AddPoint(3.0,     0.015)   # 近障碍
    otf.AddPoint(7.0,     0.006)   # 过渡
    otf.AddPoint(10.0,    0.002)   # 安全边界
    otf.AddPoint(50.0,    0.001)   # 安全区半透明
    otf.AddPoint(data_max, 0.0008) # 远处最低 (不完全透明)

    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(img_data)
    mapper.SetBlendModeToComposite()
    # 关闭自动采样距离: 大数据集上 VTK 会把步长设很大导致全透明
    mapper.SetAutoAdjustSampleDistances(0)
    mapper.SetSampleDistance(spacing[0])  # 显式设置为 1 个体素步长

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(ctf)
    prop.SetScalarOpacity(otf)
    # 不用 GradientOpacity: 安全区梯度≈1 时 gotf≈0.27 会让整体过暗
    prop.SetInterpolationTypeToLinear()
    prop.ShadeOn()
    prop.SetAmbient(0.72)
    prop.SetDiffuse(0.38)
    prop.SetSpecular(0.06)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)
    add_cube_axes(renderer, bounds)
    renderer.SetBackground(1, 1, 1)
    renderer.ResetCamera()
    set_camera(renderer, bounds, elev, azim)

    save_image(renderer, output_path, dpi, elev, azim,
               title='欧氏有向距离场 SDF (体渲染)',
               cb_range=(data_min, data_max), cb_label='Distance (m)', cb_cmap=ESDF_CMAP,
               dims=dims, spacing=spacing,
               cb_norm=ESDFNorm(0.0, data_max),
               cb_ticks=[0, 2.5, 5, 7.5, 10, data_max])


# -------------------------- Mask 等值面 --------------------------
def render_mask_surface(img_data, bounds, output_path, dpi, elev, azim, dims, spacing):
    """二值掩码: MarchingCubes 等值面, 按高程着色."""
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(img_data)
    mc.SetValue(0, 0.5)
    mc.ComputeNormalsOn()
    mc.Update()
    surface = mc.GetOutput()
    if surface.GetNumberOfPoints() == 0:
        print("  [警告] 未提取到表面"); return

    z_min, z_max = surface.GetBounds()[4], surface.GetBounds()[5]
    elev_f = vtk.vtkElevationFilter()
    elev_f.SetInputData(surface)
    elev_f.SetLowPoint(0, 0, z_min)
    elev_f.SetHighPoint(0, 0, z_max)
    elev_f.SetScalarRange(z_min, z_max)
    elev_f.Update()

    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.6, 0.0)
    lut.SetTableRange(z_min, z_max)
    lut.Build()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(elev_f.GetOutputPort())
    mapper.SetScalarRange(z_min, z_max)
    mapper.SetLookupTable(lut)
    mapper.UseLookupTableScalarRangeOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetSpecular(0.3)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    add_cube_axes(renderer, bounds)
    renderer.SetBackground(1, 1, 1)
    renderer.ResetCamera()
    set_camera(renderer, bounds, elev, azim)

    save_image(renderer, output_path, dpi, elev, azim,
               title='体素占据掩码 (障碍物表面)',
               cb_range=(z_min, z_max), cb_label='高程 Z (m)',
               cb_cmap='terrain_r', dims=dims, spacing=spacing)
    print(f"  表面: {surface.GetNumberOfPoints()} 顶点")


# -------------------------- 三视图切片 (matplotlib imshow) --------------------------
def render_slices(img_data, output_path, dpi, dims, spacing, d_min, d_max,
                  is_binary=False, is_cost=False):
    """
    三视图切片: XY俯视 / XZ侧视 / YZ前视.
    从 vtkImageData 提取中间切片, matplotlib imshow 渲染 2D 断面.
    2D 切片能清晰展示体渲染难以呈现的内部精细结构.
    """
    nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
    sx, sy, sz = spacing[0], spacing[1], spacing[2]

    # 获取 numpy 视图 (共享 VTK 内存, 不复制)
    vtk_arr = img_data.GetPointData().GetScalars()
    arr = numpy_support.vtk_to_numpy(vtk_arr)
    arr = arr.reshape(nz, ny, nx)  # VTK Fortran order → C order (z, y, x)

    # 三组切片: Z 取 50/100/150m 三个高度, 展示中高空变化; X/Y 每组不同
    #   行0 (低空/南西):  Z=50m,  Y=30%, X=30%
    #   行1 (中空/中心):  Z=100m, Y=50%, X=50%
    #   行2 (高空/北东):  Z=150m, Y=70%, X=70%
    z_vals = [50.0, 100.0, 150.0]
    slice_positions = []
    for i, zv in enumerate(z_vals):
        z_idx = min(nz - 1, max(1, int(zv / sz)))
        y_idx = max(1, int(ny * (0.3 + i * 0.2)))
        x_idx = max(1, int(nx * (0.3 + i * 0.2)))
        slice_positions.append((z_idx, y_idx, x_idx))
    # 物理坐标范围
    x_ext = [0, nx * sx]
    y_ext = [0, ny * sy]
    z_ext = [0, nz * sz]

    # 配色与显示范围 (SDF 用 ESDF_CMAP + 非线性 norm, 与体渲染一致)
    if is_binary:
        cmap = 'gray'
        norm = Normalize(vmin=0, vmax=1)
        cb_label = 'Occupancy (0=free, 1=occupied)'
    elif is_cost:
        cmap = 'viridis'
        vmax = max(10.0, d_max * 0.1)
        norm = Normalize(vmin=1.0, vmax=vmax)
        cb_label = 'Risk cost'
    else:
        # SDF 图例从 0 m 开始；负值体素仍按最低端颜色显示
        cmap = ESDF_CMAP
        norm = ESDFNorm(0.0, d_max)
        cb_label = 'Distance (m)'

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for row, (z_idx, y_idx, x_idx) in enumerate(slice_positions):
        z_val = z_idx * sz
        y_val = y_idx * sy
        x_val = x_idx * sx
        ax_xy = axes[row, 0]
        ax_xz = axes[row, 1]
        ax_yz = axes[row, 2]

        # XY 俯视 (固定 Z=z_idx)
        xy_slice = arr[z_idx, :, :]
        im = ax_xy.imshow(xy_slice, cmap=cmap, norm=norm,
                          origin='lower', extent=[*x_ext, *y_ext], aspect='equal',
                          interpolation='nearest')
        ax_xy.set_title(f'Top View (XY), Z = {z_val:.0f} m', fontsize=11)
        ax_xy.set_xlabel('X (m)')
        ax_xy.set_ylabel('Y (m)')
        # 参考线: 标出 XZ/YZ 切片所在位置
        ax_xy.axhline(y=y_val, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_xy.axvline(x=x_val, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_xy.grid(alpha=0.2, linestyle='--', linewidth=0.5)

        # XZ 侧视 (固定 Y=y_idx, 每行不同)
        xz_slice = arr[:, y_idx, :]
        ax_xz.imshow(xz_slice, cmap=cmap, norm=norm,
                     origin='lower', extent=[*x_ext, *z_ext], aspect='auto',
                     interpolation='nearest')
        ax_xz.set_title(f'South View (XZ), Y = {y_val:.0f} m', fontsize=11)
        ax_xz.set_xlabel('X (m)')
        ax_xz.set_ylabel('Z (m)')
        # 参考线: Z 切片高度 + X 切片位置
        ax_xz.axhline(y=z_val, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_xz.axvline(x=x_val, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_xz.grid(alpha=0.2, linestyle='--', linewidth=0.5)

        # YZ 前视 (固定 X=x_idx, 每行不同)
        yz_slice = arr[:, :, x_idx]
        ax_yz.imshow(yz_slice, cmap=cmap, norm=norm,
                     origin='lower', extent=[*y_ext, *z_ext], aspect='auto',
                     interpolation='nearest')
        ax_yz.set_title(f'West View (YZ), X = {x_val:.0f} m', fontsize=11)
        ax_yz.set_xlabel('Y (m)')
        ax_yz.set_ylabel('Z (m)')
        # 参考线: Z 切片高度 + Y 切片位置
        ax_yz.axhline(y=z_val, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_yz.axvline(x=y_val, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_yz.grid(alpha=0.2, linestyle='--', linewidth=0.5)

    # 共享色标 (SDF 用非线性 norm 的刻度, 其他用线性)
    fig.subplots_adjust(right=0.92, wspace=0.25, hspace=0.35)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cb_label, fontsize=10)
    # SDF: 手动设置刻度, 突出 0 和 10 这两个关键值
    if not is_binary and not is_cost:
        cbar.set_ticks([0, 2.5, 5, 7.5, 10, d_max])
        cbar.set_ticklabels(['0.0', '2.5', '5.0', '7.5', '10.0', f'{d_max:.1f}'])

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


# -------------------------- Cost 等值面 --------------------------
def render_cost_surface(img_data, bounds, output_path, dpi, elev, azim, dims, spacing,
                        data_max):
    """代价场: 多层等值面."""
    renderer = vtk.vtkRenderer()
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.66, 0.0)
    lut.SetTableRange(1.0, max(data_max * 0.1, 10.0))
    lut.Build()

    for iso_val, alpha in [(2.0, 0.3), (5.0, 0.6)]:
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(img_data)
        mc.SetValue(0, iso_val)
        mc.Update()
        surf = mc.GetOutput()
        if surf.GetNumberOfPoints() == 0:
            continue
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(surf)
        mapper.SetScalarRange(1.0, max(data_max * 0.1, 10.0))
        mapper.SetLookupTable(lut)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(alpha)
        renderer.AddActor(actor)
        print(f"  等值面 {iso_val}: {surf.GetNumberOfPoints()} 顶点")

    add_cube_axes(renderer, bounds)
    renderer.SetBackground(1, 1, 1)
    renderer.ResetCamera()
    set_camera(renderer, bounds, elev, azim)

    save_image(renderer, output_path, dpi, elev, azim,
               title='风险代价场 (等值面)',
               cb_range=(1.0, max(data_max * 0.1, 10.0)), cb_label='代价值',
               cb_cmap='plasma', dims=dims, spacing=spacing)


# -------------------------- 保存 (VTK offscreen + matplotlib 色标) --------------------------
def save_image(renderer, output_path, dpi, elev, azim, title, cb_range,
               cb_label, cb_cmap, dims, spacing, cb_norm=None, cb_ticks=None):
    """VTK offscreen 渲染 → matplotlib 叠加标题/色标.

    cb_norm:  自定义非线性归一化 (如 ESDFNorm), None 时用线性 Normalize
    cb_ticks: 色标刻度位置 (数据值列表), None 时自动
    """
    win = vtk.vtkRenderWindow()
    win.AddRenderer(renderer)
    win.SetSize(int(dpi * 11), int(dpi * 7.7))
    win.SetOffScreenRendering(True)
    win.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetReadFrontBuffer(False)
    w2i.Update()

    tmp = output_path + '.tmp.png'
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(tmp)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    img = Image.open(tmp)
    img_arr = np.array(img)

    fig = plt.figure(figsize=(12, 7.6))
    ax = fig.add_axes([0.025, 0.025, 0.795, 0.95])
    ax.imshow(img_arr)
    ax.axis('off')

    # 独立色标轴便于精确控制字号和留白
    norm = cb_norm if cb_norm is not None else Normalize(vmin=cb_range[0], vmax=cb_range[1])
    sm = cm.ScalarMappable(cmap=cb_cmap, norm=norm)
    sm.set_array([])
    # PyVista 字号以屏幕像素计；Matplotlib 字号以 point 计，按输出 DPI 换算。
    title_size = 40.0 * 72.0 / dpi
    label_size = 34.0 * 72.0 / dpi
    cax = fig.add_axes([0.72, 0.19, 0.020, 0.62])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_title(
        cb_label,
        fontsize=title_size,
        fontfamily='DejaVu Sans',
        fontweight='bold',
        pad=4
    )
    if cb_ticks is not None:
        cbar.set_ticks(cb_ticks)
        cbar.set_ticklabels([f'{value:.1f}' for value in cb_ticks])
    cbar.ax.tick_params(labelsize=label_size, width=1.0, length=5, pad=5)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontfamily('DejaVu Sans')
        tick.set_fontweight('bold')
    cbar.outline.set_linewidth(1.2)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.08)
    plt.close()
    os.remove(tmp)
    print(f"[输出] {output_path} (dpi={dpi})")


# -------------------------- 主函数 --------------------------
def main():
    parser = argparse.ArgumentParser(
        description='MHD 体数据 3D 体渲染 (正交投影, 视角与 visDSM3D.py 一致)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visMHD.py ../data/SF_Downtown_sdf.mhd
  python visMHD.py ../data/SF_Downtown_mask.mhd
""")
    parser.add_argument('mhd', help='.mhd 文件路径')
    parser.add_argument('--shrink', type=int, default=1,
                        help='降采样因子 (默认: 1=不降采样, 用 VTK 流式读取)')
    parser.add_argument('--dpi', type=int, default=300, help='输出分辨率 (默认: 300)')
    parser.add_argument('--elev', type=float, default=5,
                        help='仰角 (默认: 5, 与 DSM_3D_主视角.png 一致)')
    parser.add_argument('--azim', type=float, default=0,
                        help='方位角 (默认: 0, 与 DSM_3D_主视角.png 一致)')
    parser.add_argument('--output-dir', default=None, help=f'输出目录 (默认: {OUTPUT_DIR})')
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print(f"解析: {args.mhd}")
    info = parse_mhd(args.mhd)
    print(f"  维度: {info['DimSize']}, 分辨率: {info['Spacing']}, 类型: {info['dtype']}")

    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(args.mhd)
    reader.Update()

    # GPU 体纹理的单轴尺寸通常限制为 2048；留出余量后自动降采样，
    # 避免超限时 vtkSmartVolumeMapper 输出空白或整块单色图像。
    source_dims = reader.GetOutput().GetDimensions()
    texture_limit = 1024
    auto_shrink = max(1, math.ceil(max(source_dims) / texture_limit))
    shrink = max(args.shrink, auto_shrink)
    if shrink > 1:
        print(f"读取 + VTK 降采样 (shrink={shrink}, GPU 纹理安全尺寸)...")
        shrink_filter = vtk.vtkImageShrink3D()
        shrink_filter.SetInputConnection(reader.GetOutputPort())
        shrink_filter.SetShrinkFactors(shrink, shrink, shrink)
        shrink_filter.AveragingOn()
        shrink_filter.Update()
        img_data = shrink_filter.GetOutput()
    else:
        print("读取 (VTK MetaImageReader, 无需降采样)...")
        img_data = reader.GetOutput()

    dims = img_data.GetDimensions()
    spacing = img_data.GetSpacing()
    bounds = img_data.GetBounds()
    d_min, d_max = img_data.GetScalarRange()
    print(f"  实际维度: {dims[0]}×{dims[1]}×{dims[2]}, "
          f"分辨率: {spacing[0]:.1f}×{spacing[1]:.1f}×{spacing[2]:.1f}m")
    print(f"  数据范围: [{d_min:.2f}, {d_max:.2f}]")

    base = os.path.splitext(os.path.basename(args.mhd))[0]
    is_binary = (info['dtype'] == np.uint8 and d_max <= 1)
    is_cost = 'cost' in base.lower()
    out_path = os.path.join(out_dir, f'{base}_volume_rendering.png')

    if is_binary:
        print("渲染 Mask (等值面)...")
        render_mask_surface(img_data, bounds, out_path, args.dpi,
                            args.elev, args.azim, dims, spacing)
    elif is_cost:
        print("渲染 Cost (等值面)...")
        render_cost_surface(img_data, bounds, out_path, args.dpi,
                            args.elev, args.azim, dims, spacing, d_max)
    else:
        print("渲染 SDF (体渲染)...")
        render_sdf_volume(img_data, bounds, out_path, args.dpi,
                          args.elev, args.azim, dims, spacing, d_min, d_max)

    # 三视图切片 (补充体渲染难以展示的内部细节)
    slices_path = os.path.join(out_dir, f'{base}_orthographic_views.png')
    print("渲染三视图切片...")
    render_slices(img_data, slices_path, args.dpi, dims, spacing, d_min, d_max,
                  is_binary=is_binary, is_cost=is_cost)
    print(f"  → {slices_path}")

    print("渲染完成!")


if __name__ == '__main__':
    main()
