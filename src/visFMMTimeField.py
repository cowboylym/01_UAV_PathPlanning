#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render orthographic views of a complete 3-D FMM arrival-cost field."""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np


OUTPUT_DIR = '../analysis'
DEFAULT_TFIELD = '../data/compare4/final_v1/raw/search_FMM_Tfield_3D.mhd'
DEFAULT_PATH = '../data/compare4/final_v1/raw/search_path_FMM.csv'

FMM_CMAP = LinearSegmentedColormap.from_list(
    'fmm_green_red',
    [
        (0.00, '#009E73'),
        (0.25, '#56B4E9'),
        (0.50, '#F0E442'),
        (0.75, '#E69F00'),
        (1.00, '#D55E00'),
    ],
)
FMM_CMAP.set_bad('white')

MET_TYPE_MAP = {
    'MET_FLOAT': np.float32,
    'MET_DOUBLE': np.float64,
    'MET_UCHAR': np.uint8,
    'MET_CHAR': np.int8,
    'MET_USHORT': np.uint16,
    'MET_SHORT': np.int16,
    'MET_UINT': np.uint32,
    'MET_INT': np.int32,
}


def parse_mhd(mhd_path):
    header = {}
    with open(mhd_path, 'r') as stream:
        for line in stream:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                header[key.strip()] = value.strip()

    return {
        'dims': tuple(int(value) for value in header['DimSize'].split()),
        'spacing': tuple(float(value) for value in header['ElementSpacing'].split()),
        'origin': tuple(float(value) for value in header['Offset'].split()),
        'dtype': MET_TYPE_MAP[header['ElementType']],
        'raw_path': os.path.join(
            os.path.dirname(os.path.abspath(mhd_path)),
            header['ElementDataFile'],
        ),
    }


def load_volume(info, shrink):
    nx, ny, nz = info['dims']
    if not os.path.exists(info['raw_path']):
        raise FileNotFoundError(f"RAW file not found: {info['raw_path']}")

    source = np.memmap(
        info['raw_path'],
        dtype=info['dtype'],
        mode='r',
        shape=(nz, ny, nx),
    )
    data = source[::shrink, ::shrink, ::shrink].copy()
    spacing = tuple(value * shrink for value in info['spacing'])
    print(
        f'[FMM] dimensions: {nx}x{ny}x{nz} -> '
        f'{data.shape[2]}x{data.shape[1]}x{data.shape[0]}, shrink={shrink}'
    )
    return data, spacing, info['origin']


def load_path_endpoints(path_csv):
    if not path_csv or not os.path.exists(path_csv):
        return None, None
    path = np.loadtxt(path_csv, delimiter=',', skiprows=1, ndmin=2)
    return path[0, :3], path[-1, :3]


def point_to_index(point, origin, spacing, shape):
    index_xyz = np.rint(
        (np.asarray(point) - np.asarray(origin)) / np.asarray(spacing)
    ).astype(int)
    nx, ny, nz = shape[2], shape[1], shape[0]
    return np.clip(index_xyz, (0, 0, 0), (nx - 1, ny - 1, nz - 1))


def display_limit(data, start, origin, spacing):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError('The FMM volume contains no finite values.')

    fallback = float(np.percentile(finite, 95))
    if start is None:
        return fallback * 1.05

    ix, iy, iz = point_to_index(start, origin, spacing, data.shape)
    value = float(data[iz, iy, ix])
    if not np.isfinite(value) or value <= 0:
        value = fallback
    print(f'[FMM] T(start)={value:.2f}')
    return value * 1.05


def source_indices(data, goal, origin, spacing):
    if goal is not None:
        return point_to_index(goal, origin, spacing, data.shape)

    finite_data = np.where(np.isfinite(data), data, np.inf)
    iz, iy, ix = np.unravel_index(np.argmin(finite_data), data.shape)
    return np.array([ix, iy, iz])


def render_orthographic_views(data, spacing, origin, start, goal, output_path, dpi):
    t_max = display_limit(data, start, origin, spacing)
    display = np.ma.masked_invalid(data)
    display = np.ma.minimum(display, t_max)

    if start is None or goal is None:
        raise ValueError('A path CSV is required to mark the start and goal.')
    start_local = np.asarray(start) - np.asarray(origin)
    goal_local = np.asarray(goal) - np.asarray(origin)

    nx, ny, nz = data.shape[2], data.shape[1], data.shape[0]
    sx, sy, sz = spacing
    reference_slices = [
        (48.0, 615.0, 621.0),
        (99.0, 1026.0, 1035.0),
        (150.0, 1437.0, 1449.0),
    ]
    slice_positions = []
    for z_value, y_value, x_value in reference_slices:
        z_index = min(nz - 1, max(1, int(round(z_value / sz))))
        y_index = min(ny - 1, max(1, int(round(y_value / sy))))
        x_index = min(nx - 1, max(1, int(round(x_value / sx))))
        slice_positions.append(
            (z_index, y_index, x_index, z_value, y_value, x_value)
        )

    x_extent = [0, nx * sx]
    y_extent = [0, ny * sy]
    z_extent = [0, nz * sz]
    norm = Normalize(vmin=0.0, vmax=t_max)
    figure, axes = plt.subplots(3, 3, figsize=(18, 16))

    for row, (
        z_index, y_index, x_index, z_value, y_value, x_value
    ) in enumerate(slice_positions):
        axis_xy = axes[row, 0]
        axis_xz = axes[row, 1]
        axis_yz = axes[row, 2]

        image = axis_xy.imshow(
            display[z_index, :, :], cmap=FMM_CMAP, norm=norm,
            origin='lower', extent=[*x_extent, *y_extent], aspect='equal',
            interpolation='nearest',
        )
        axis_xy.set_title(f'Top View (XY), Z = {z_value:.0f} m', fontsize=11)
        axis_xy.set_xlabel('X (m)')
        axis_xy.set_ylabel('Y (m)')
        axis_xy.axhline(y=y_value, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        axis_xy.axvline(x=x_value, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        axis_xy.grid(alpha=0.2, linestyle='--', linewidth=0.5)

        axis_xz.imshow(
            display[:, y_index, :], cmap=FMM_CMAP, norm=norm,
            origin='lower', extent=[*x_extent, *z_extent], aspect='auto',
            interpolation='nearest',
        )
        axis_xz.set_title(f'South View (XZ), Y = {y_value:.0f} m', fontsize=11)
        axis_xz.set_xlabel('X (m)')
        axis_xz.set_ylabel('Z (m)')
        axis_xz.axhline(y=z_value, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        axis_xz.axvline(x=x_value, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        axis_xz.grid(alpha=0.2, linestyle='--', linewidth=0.5)

        axis_yz.imshow(
            display[:, :, x_index], cmap=FMM_CMAP, norm=norm,
            origin='lower', extent=[*y_extent, *z_extent], aspect='auto',
            interpolation='nearest',
        )
        axis_yz.set_title(f'West View (YZ), X = {x_value:.0f} m', fontsize=11)
        axis_yz.set_xlabel('Y (m)')
        axis_yz.set_ylabel('Z (m)')
        axis_yz.axhline(y=z_value, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        axis_yz.axvline(x=y_value, color='white', linestyle=':', linewidth=1.5, alpha=0.8)
        axis_yz.grid(alpha=0.2, linestyle='--', linewidth=0.5)

        marker_style = {
            'marker': '*',
            's': 145,
            'edgecolors': 'black',
            'linewidths': 0.8,
            'zorder': 20,
        }
        axis_xy.scatter(start_local[0], start_local[1], c='#009E73', **marker_style)
        axis_xy.scatter(goal_local[0], goal_local[1], c='#D55E00', **marker_style)
        axis_xz.scatter(start_local[0], start_local[2], c='#009E73', **marker_style)
        axis_xz.scatter(goal_local[0], goal_local[2], c='#D55E00', **marker_style)
        axis_yz.scatter(start_local[1], start_local[2], c='#009E73', **marker_style)
        axis_yz.scatter(goal_local[1], goal_local[2], c='#D55E00', **marker_style)

    figure.subplots_adjust(right=0.92, bottom=0.08, wspace=0.25, hspace=0.35)
    color_axis = figure.add_axes([0.93, 0.12, 0.012, 0.76])
    colorbar = figure.colorbar(image, cax=color_axis)
    colorbar.set_label('Arrival cost (m)', fontsize=10)
    legend_handles = [
        Line2D(
            [0], [0], marker='*', linestyle='none', markersize=11,
            markerfacecolor='#009E73', markeredgecolor='black', label='Start',
        ),
        Line2D(
            [0], [0], marker='*', linestyle='none', markersize=11,
            markerfacecolor='#D55E00', markeredgecolor='black', label='Goal',
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc='lower center',
        bbox_to_anchor=(0.52, 0.012),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor='black',
        fontsize=11,
    )

    figure.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(figure)
    print(f'[output] {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Render orthographic views of a complete 3-D FMM arrival-cost field.'
    )
    parser.add_argument('mhd', nargs='?', default=DEFAULT_TFIELD)
    parser.add_argument('--path', default=DEFAULT_PATH)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument(
        '--output-name',
        default='FMM_arrival_cost_orthographic_views.png',
    )
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--shrink', type=int, default=1)
    args = parser.parse_args()

    info = parse_mhd(args.mhd)
    data, spacing, origin = load_volume(info, max(1, args.shrink))
    start, goal = load_path_endpoints(args.path)
    os.makedirs(args.output_dir, exist_ok=True)
    render_orthographic_views(
        data,
        spacing,
        origin,
        start,
        goal,
        os.path.join(args.output_dir, args.output_name),
        args.dpi,
    )


if __name__ == '__main__':
    main()
