#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ESDF 梯度场可视化 — 切片 + 箭头叠加 (论文插图)
================================================
仅依赖梯度矢量场 (grad.mhd), 无需 ESDF 标量场.

渲染方式:
  - 底色: 梯度模长 |∇d| = √(gx²+gy²+gz²), diverging 配色以 1.0 为中心
          |∇d|≈1.0 (理想距离场) → 白色
          |∇d|<1   (距离场退化, 如骨架处) → 蓝色
          |∇d|>1   (距离场异常, 如锐角处) → 红色
  - 箭头: ∇ESDF 梯度方向投影到切片平面 (quiver), 统一长度, 黑色
  - 布局: 3 行 × 3 列 (行=Z高度 50/100/150m, 列=XY/XZ/YZ 三个正交视图)

物理含义:
  - 梯度方向 = ESDF 上升最快方向 = 远离障碍物/地形表面的方向
  - 箭头从障碍物指向自由空间, 直观展示"逃生方向"
  - 梯度模长 |∇d| ≈ 1.0 (理想距离场性质), 偏离 1.0 的区域说明 ESDF 退化

输入文件:
  - 梯度矢量场:  SF_Downtown_grad.mhd  (float32, 3 通道 gx,gy,gz)

依赖: pip install numpy matplotlib

用法:
  python visGradient.py
  python visGradient.py --grad ../data/SF_Downtown_grad.mhd
  python visGradient.py --arrow-step 10 --dpi 300
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# -------------------------- 全局样式 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
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

# 梯度模长 diverging 色标: 以 1.0 为中心
#   |∇d| < 1 (退化): 蓝 → 青
#   |∇d| ≈ 1 (理想): 浅黄/白
#   |∇d| > 1 (异常): 橙 → 红
GMAG_CMAP = LinearSegmentedColormap.from_list(
    'gmag_div',
    [(0.00, '#2166AC'),  # 强蓝 (|∇d|≈0, 严重退化)
     (0.25, '#67A9CF'),  # 浅蓝
     (0.45, '#D1E5F0'),  # 淡青
     (0.50, '#FFF7BC'),  # 浅黄 (|∇d|≈1, 理想)
     (0.55, '#FDDBC7'),  # 淡橙
     (0.75, '#EF8A62'),  # 橙红
     (1.00, '#B2182B')], # 深红 (|∇d|>>1, 异常)
)


# -------------------------- .mhd 解析 --------------------------
def parse_mhd(mhd_path):
    """解析 .mhd 头文件, 返回元数据字典."""
    with open(mhd_path, 'r') as f:
        header = {}
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                header[key.strip()] = val.strip()

    n_channels = int(header.get('ElementNumberOfChannels', '1'))
    return {
        'DimSize': tuple(int(x) for x in header['DimSize'].split()),
        'Spacing': tuple(float(x) for x in header['ElementSpacing'].split()),
        'Offset': tuple(float(x) for x in header['Offset'].split()),
        'dtype': MET_TYPE_MAP[header['ElementType']],
        'n_channels': n_channels,
        'raw_path': os.path.join(os.path.dirname(os.path.abspath(mhd_path)),
                                 header['ElementDataFile']),
        'compressed': header.get('CompressedData', 'False').lower() == 'true',
    }


def load_vector_mhd(info):
    """加载矢量 MHD (3通道) 为 numpy 数组 (nz, ny, nx, 3). 用 memmap 懒加载."""
    nx, ny, nz = info['DimSize']
    nc = info['n_channels']
    if nc != 3:
        raise ValueError(f"梯度场应为 3 通道, 实际 {nc} 通道")
    if not os.path.exists(info['raw_path']):
        raise FileNotFoundError(f".raw 不存在: {info['raw_path']}")
    # ITK VectorImage 内存布局: interleaved [v0_c0, v0_c1, v0_c2, v1_c0, ...]
    arr = np.memmap(info['raw_path'], dtype=info['dtype'], mode='r',
                    shape=(nz, ny, nx, nc))
    return np.array(arr)


# -------------------------- 渲染: 切片 + 箭头叠加 --------------------------
def render_gradient_quiver(gmag_data, grad_data, dims, spacing, output_path,
                           dpi=300, arrow_step=10):
    """
    三视图切片 + 梯度箭头叠加. 底色=梯度模长, 箭头=梯度方向.

    参数:
      gmag_data  - 梯度模长标量场 (nz, ny, nx), float
      grad_data  - 梯度矢量场 (nz, ny, nx, 3), float32
      dims       - (nx, ny, nz)
      spacing    - (sx, sy, sz) 物理分辨率 (m)
      output_path- 输出 PNG 路径
      dpi        - 输出分辨率
      arrow_step - 箭头采样间隔 (体素), 默认 10
    """
    nx, ny, nz = dims
    sx, sy, sz = spacing

    # 梯度模长范围 (用于 diverging 配色, 以 1.0 为中心)
    gmag_min = float(np.min(gmag_data))
    gmag_max = float(np.max(gmag_data))
    # 确保 vmax 对称地偏离 1.0, 使色标以 1.0 为视觉中心
    vmax = max(gmag_max, 2.0 - gmag_min)  # 保证 1.0 居中
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)

    # 物理坐标范围
    x_ext = [0, nx * sx]
    y_ext = [0, ny * sy]
    z_ext = [0, nz * sz]

    # 三组切片: Z=50/100/150m, X/Y 位置随行递增 (与 visMHD.py 一致)
    z_vals = [50.0, 100.0, 150.0]
    slice_positions = []
    for i, zv in enumerate(z_vals):
        z_idx = min(nz - 1, max(1, int(zv / sz)))
        y_idx = max(1, int(ny * (0.3 + i * 0.2)))
        x_idx = max(1, int(nx * (0.3 + i * 0.2)))
        slice_positions.append((z_idx, y_idx, x_idx))
    row_labels = ['低空/南西', '中空/中心', '高空/北东']

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for row, (z_idx, y_idx, x_idx) in enumerate(slice_positions):
        z_val = z_idx * sz
        y_val = y_idx * sy
        x_val = x_idx * sx
        ax_xy = axes[row, 0]
        ax_xz = axes[row, 1]
        ax_yz = axes[row, 2]

        # ===== XY 俯视 (固定 Z=z_idx) =====
        xy_gmag = gmag_data[z_idx, :, :]                    # (ny, nx)
        xy_gx   = grad_data[z_idx, :, :, 0]                 # (ny, nx)
        xy_gy   = grad_data[z_idx, :, :, 1]                 # (ny, nx)
        im = ax_xy.imshow(xy_gmag, cmap=GMAG_CMAP, norm=norm,
                          origin='lower', extent=[*x_ext, *y_ext],
                          aspect='equal', interpolation='nearest')
        _add_quiver(ax_xy, xy_gx, xy_gy, x_ext, y_ext,
                    arrow_step, sx, sy, nx, ny)
        ax_xy.set_title(f'XY 俯视 ({row_labels[row]}, Z={z_val:.0f}m)', fontsize=11)
        ax_xy.set_xlabel('东向 X (m)')
        ax_xy.set_ylabel('北向 Y (m)')
        ax_xy.axhline(y=y_val, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_xy.axvline(x=x_val, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_xy.grid(alpha=0.2, linestyle='--', linewidth=0.5)

        # ===== XZ 侧视 (固定 Y=y_idx) =====
        xz_gmag = gmag_data[:, y_idx, :]                    # (nz, nx)
        xz_gx   = grad_data[:, y_idx, :, 0]                 # (nz, nx)
        xz_gz   = grad_data[:, y_idx, :, 2]                 # (nz, nx)
        ax_xz.imshow(xz_gmag, cmap=GMAG_CMAP, norm=norm,
                     origin='lower', extent=[*x_ext, *z_ext],
                     aspect='auto', interpolation='nearest')
        _add_quiver(ax_xz, xz_gx, xz_gz, x_ext, z_ext,
                    arrow_step, sx, sz, nx, nz)
        ax_xz.set_title(f'XZ 侧视 (Y={y_val:.0f}m)', fontsize=11)
        ax_xz.set_xlabel('东向 X (m)')
        ax_xz.set_ylabel('高程 Z (m)')
        ax_xz.axhline(y=z_val, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_xz.axvline(x=x_val, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_xz.grid(alpha=0.2, linestyle='--', linewidth=0.5)

        # ===== YZ 前视 (固定 X=x_idx) =====
        yz_gmag = gmag_data[:, :, x_idx]                    # (nz, ny)
        yz_gy   = grad_data[:, :, x_idx, 1]                 # (nz, ny)
        yz_gz   = grad_data[:, :, x_idx, 2]                 # (nz, ny)
        ax_yz.imshow(yz_gmag, cmap=GMAG_CMAP, norm=norm,
                     origin='lower', extent=[*y_ext, *z_ext],
                     aspect='auto', interpolation='nearest')
        _add_quiver(ax_yz, yz_gy, yz_gz, y_ext, z_ext,
                    arrow_step, sy, sz, ny, nz)
        ax_yz.set_title(f'YZ 前视 (X={x_val:.0f}m)', fontsize=11)
        ax_yz.set_xlabel('北向 Y (m)')
        ax_yz.set_ylabel('高程 Z (m)')
        ax_yz.axhline(y=z_val, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_yz.axvline(x=y_val, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax_yz.grid(alpha=0.2, linestyle='--', linewidth=0.5)

    # 共享色标
    fig.subplots_adjust(right=0.92, wspace=0.25, hspace=0.35)
    cax = fig.add_axes([0.93, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('梯度模长 |∇ESDF| (理想值=1.0)', fontsize=10)
    # 刻度突出 1.0 这个理想值
    cbar.set_ticks([0.0, 0.5, 1.0, 1.5, round(vmax, 1)])

    fig.suptitle('ESDF 梯度场 矢量切片 '
                 '(底色=|∇ESDF| 模长, 箭头=∇ESDF 远离障碍物方向)\n'
                 f'体素 {nx}×{ny}×{nz}  分辨率 {sx:.1f}×{sy:.1f}×{sz:.1f}m  '
                 f'箭头间隔 {arrow_step} 体素',
                 fontsize=13, fontweight='bold', y=0.99)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"[输出] {output_path} (dpi={dpi})")


def _add_quiver(ax, gx, gz_or_gy, ext_x, ext_y, step, sp_x, sp_y, n_x, n_y):
    """
    在切片上叠加梯度箭头.

    参数:
      ax        - matplotlib 子图
      gx        - 梯度 X 分量 (n_y, n_x) 或 (n_z, n_x)
      gz_or_gy  - 梯度第二分量 (Z 或 Y)
      ext_x     - X 物理范围 [xmin, xmax]
      ext_y     - Y 物理范围 [ymin, ymax]
      step      - 箭头采样间隔 (体素)
      sp_x, sp_y- 物理分辨率 (m)
      n_x, n_y  - 切片维度
    """
    # 降采样网格 (每隔 step 体素取一个箭头)
    ys = np.arange(0, n_y, step)
    xs = np.arange(0, n_x, step)
    Xs, Ys = np.meshgrid(xs, ys)  # (len(ys), len(xs))

    # 采样梯度分量
    u = gx[ys][:, xs]             # X 方向分量
    v = gz_or_gy[ys][:, xs]       # Y/Z 方向分量

    # 转换为物理坐标
    Xp = Xs * sp_x
    Yp = Ys * sp_y

    # 统一箭头长度 (方向用梯度方向, 长度统一)
    ax.quiver(Xp, Yp, u, v,
              angles='uv',          # 方向来自 (u,v), 长度统一
              scale=25,             # 越大箭头越短 (统一长度)
              scale_units='width',  # 以子图宽度为参考
              color='black',
              linewidth=0.6,
              alpha=0.85,
              pivot='middle')


# -------------------------- 主函数 --------------------------
def main():
    parser = argparse.ArgumentParser(
        description='ESDF 梯度场切片+箭头可视化 (底色=|∇ESDF|, 箭头=∇ESDF)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python visGradient.py
  python visGradient.py --grad ../data/SF_Downtown_grad.mhd
  python visGradient.py --arrow-step 10 --dpi 300
""")
    parser.add_argument('--grad', default='../data/SF_Downtown_grad.mhd',
                        help='梯度矢量场 .mhd 路径 (默认: ../data/SF_Downtown_grad.mhd)')
    parser.add_argument('--arrow-step', type=int, default=10,
                        help='箭头采样间隔 (体素, 默认: 10)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='输出分辨率 (默认: 300)')
    parser.add_argument('--output-dir', default=None,
                        help=f'输出目录 (默认: {OUTPUT_DIR})')
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载梯度矢量场
    print(f"加载梯度场: {args.grad}")
    grad_info = parse_mhd(args.grad)
    print(f"  维度: {grad_info['DimSize']}, 分辨率: {grad_info['Spacing']}, "
          f"通道: {grad_info['n_channels']}")
    if grad_info['n_channels'] != 3:
        print(f"[错误] 梯度场应为 3 通道, 实际 {grad_info['n_channels']} 通道")
        return 1
    grad_data = load_vector_mhd(grad_info)

    # 2. 计算梯度模长 (作为底色, 同时验证 ESDF 质量)
    gmag_data = np.sqrt(grad_data[..., 0]**2 + grad_data[..., 1]**2 + grad_data[..., 2]**2)
    print(f"  梯度模长: max={float(np.max(gmag_data)):.4f}, "
          f"mean={float(np.mean(gmag_data)):.4f}, "
          f"min={float(np.min(gmag_data)):.4f} (理想值≈1.0)")

    # 3. 渲染
    dims = grad_info['DimSize']        # (nx, ny, nz)
    spacing = grad_info['Spacing']     # (sx, sy, sz)
    base = os.path.splitext(os.path.basename(args.grad))[0].replace('_grad', '')
    out_path = os.path.join(out_dir, f'{base}_梯度场矢量切片.png')
    print(f"渲染梯度场切片 (箭头间隔={args.arrow_step} 体素)...")
    render_gradient_quiver(gmag_data, grad_data, dims, spacing, out_path,
                           dpi=args.dpi, arrow_step=args.arrow_step)
    print("渲染完成!")


if __name__ == '__main__':
    main()
