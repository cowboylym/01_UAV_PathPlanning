#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render the FMM-guided FDGA* search corridor over the urban DSM."""

import argparse
import os

import numpy as np
import pyvista as pv
import vtk
from matplotlib.colors import LinearSegmentedColormap


OUTPUT_DIR = '../analysis'
DEFAULT_DATA_DIR = '../data/compare4/final_v1/raw'
DEFAULT_DSM = '../data/SF_Downtown.tif'
DSM_CMAP = LinearSegmentedColormap.from_list(
    'urban_grayscale',
    ['#E8E8E8', '#D2D2D2', '#B8B8B8', '#969696', '#707070'],
)
FMM_CMAP = LinearSegmentedColormap.from_list(
    'fmm_green_red',
    ['#009E73', '#56B4E9', '#F0E442', '#E69F00', '#D55E00'],
)
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
    with open(mhd_path, 'r', encoding='utf-8') as stream:
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


def load_field_info(mhd_path):
    info = parse_mhd(mhd_path)
    nx, ny, nz = info['dims']
    print(f'[Corridor field] dimensions={nx}x{ny}x{nz}')
    return info


def load_volume(mhd_path):
    info = parse_mhd(mhd_path)
    nx, ny, nz = info['dims']
    volume = np.memmap(
        info['raw_path'], dtype=info['dtype'], mode='r', shape=(nz, ny, nx)
    )
    return info, volume


def sample_volume_nearest(points, info, volume, z_exaggeration):
    physical = np.asarray(points, dtype=float).copy()
    physical[:, 2] /= z_exaggeration
    origin = np.asarray(info['origin'])
    spacing = np.asarray(info['spacing'])
    indices = np.rint((physical - origin) / spacing).astype(int)
    nx, ny, nz = info['dims']
    indices[:, 0] = np.clip(indices[:, 0], 0, nx - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, ny - 1)
    indices[:, 2] = np.clip(indices[:, 2], 0, nz - 1)
    values = np.asarray(volume[
        indices[:, 2], indices[:, 1], indices[:, 0]
    ], dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def build_continuous_corridor(points, radius, z_exaggeration):
    centerline = np.asarray(points, dtype=float).copy()
    centerline[:, 2] *= z_exaggeration

    polyline = pv.lines_from_points(centerline, close=False)
    margin = radius * 1.15
    bounds = (
        centerline[:, 0].min() - margin,
        centerline[:, 0].max() + margin,
        centerline[:, 1].min() - margin,
        centerline[:, 1].max() + margin,
        centerline[:, 2].min() - margin,
        centerline[:, 2].max() + margin,
    )
    modeller = vtk.vtkImplicitModeller()
    modeller.SetInputData(polyline)
    modeller.SetModelBounds(bounds)
    modeller.SetSampleDimensions(220, 300, 90)
    modeller.SetMaximumDistance(0.20)
    modeller.SetAdjustDistance(0.0)
    modeller.CappingOff()
    modeller.Update()

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputConnection(modeller.GetOutputPort())
    contour.SetValue(0, radius)
    contour.ComputeNormalsOn()
    contour.Update()
    surface = pv.wrap(contour.GetOutput()).clean()
    surface = surface.smooth_taubin(
        n_iter=40,
        pass_band=0.08,
        normalize_coordinates=True,
    ).compute_normals(
        point_normals=True,
        cell_normals=True,
        auto_orient_normals=True,
    )
    print(
        f'[Corridor] continuous swept volume, radius={radius:.2f} m, '
        f'surface points={surface.n_points}'
    )
    return surface


def read_dsm(dsm_path):
    import rasterio

    with rasterio.open(dsm_path) as source:
        elevation = source.read(1).astype(np.float32)
        transform = source.transform
        nodata = source.nodata
    invalid = ~np.isfinite(elevation)
    if nodata is not None:
        invalid |= np.isclose(elevation, nodata)
    if invalid.any():
        elevation[invalid] = np.nanmin(elevation[~invalid])
    return elevation, transform


def build_dsm_surface(dsm_path, z_exaggeration, target_points=400000):
    elevation, transform = read_dsm(dsm_path)
    rows, cols = elevation.shape
    pixel_w = abs(transform.a)
    pixel_h = abs(transform.e)
    step = max(1, int(np.sqrt(rows * cols / target_points)))
    sampled = elevation[::step, ::step]

    x = transform.c + np.arange(0, cols, step)[:sampled.shape[1]] * pixel_w
    y = transform.f - np.arange(0, rows, step)[:sampled.shape[0]] * pixel_h
    xx, yy = np.meshgrid(x, y)
    zz = sampled * z_exaggeration
    surface = pv.StructuredGrid(xx, yy, zz).extract_surface(
        algorithm='dataset_surface'
    )
    surface['Elevation (m)'] = surface.points[:, 2] / z_exaggeration
    bounds = (
        float(x.min()), float(x.max()),
        float(y.min()), float(y.max()),
        float(sampled.min() * z_exaggeration),
        float(sampled.max() * z_exaggeration),
    )
    print(f'[DSM] full area={sampled.shape[1]}x{sampled.shape[0]}, step={step}')
    return surface, bounds


def load_csv_points(csv_path, columns):
    table = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=float)
    table = np.atleast_1d(table)
    return np.column_stack([table[column] for column in columns])


def make_tube(points, z_exaggeration, radius):
    displayed = np.asarray(points, dtype=float).copy()
    displayed[:, 2] *= z_exaggeration
    displayed[:, 2] += 1.5 * z_exaggeration
    line = pv.lines_from_points(displayed, close=False)
    return line.tube(radius=radius, n_sides=16)


def add_bounds(plotter, subplot, bounds, z_exaggeration, top_view=False):
    plotter.subplot(*subplot)
    actor = plotter.show_bounds(
        xtitle='Easting X (km)',
        ytitle='Northing Y (km)',
        ztitle='' if top_view else 'Altitude Z (km)',
        axes_ranges=(
            bounds[0] / 1000.0, bounds[1] / 1000.0,
            bounds[2] / 1000.0, bounds[3] / 1000.0,
            bounds[4] / z_exaggeration / 1000.0,
            bounds[5] / z_exaggeration / 1000.0,
        ),
        fmt='%.2f',
        n_xlabels=3,
        n_ylabels=3,
        n_zlabels=3,
        font_size=12,
        grid=False,
        location='outer',
        use_2d=top_view,
    )
    actor.SetLabelOffset(24.0)


def add_scene(plotter, subplot, dsm, corridor, guidance_tube,
              start, goal, bounds, z_exaggeration, top_view, show_legend,
              arrival_range):
    plotter.subplot(*subplot)
    plotter.add_mesh(
        dsm,
        scalars='Elevation (m)',
        cmap=DSM_CMAP,
        lighting=not top_view,
        ambient=0.48,
        diffuse=0.72,
        specular=0.08,
        specular_power=14.0,
        smooth_shading=not top_view,
        show_scalar_bar=False,
        name=f'dsm-{subplot}',
    )
    cell_normals = np.asarray(corridor.cell_data['Normals'])
    upper_corridor = corridor.extract_cells(cell_normals[:, 2] > 0.0)
    lower_corridor = corridor.extract_cells(cell_normals[:, 2] <= 0.0)
    material = {
        'scalars': 'Arrival cost (m)',
        'cmap': FMM_CMAP,
        'clim': arrival_range,
        'lighting': True,
        'pbr': False,
        'ambient': 0.42,
        'diffuse': 0.72,
        'specular': 0.48,
        'specular_power': 48.0,
        'smooth_shading': True,
        'show_edges': False,
    }
    plotter.add_mesh(
        lower_corridor,
        opacity=1.0,
        name=f'corridor-lower-{subplot}',
        scalar_bar_args={
            'title': 'Arrival cost (m)',
            'vertical': True,
            'position_x': 0.86,
            'position_y': 0.16,
            'width': 0.025,
            'height': 0.62,
            'title_font_size': 16,
            'label_font_size': 14,
            'fmt': '%.0f',
        },
        **material,
    )
    plotter.add_mesh(
        upper_corridor,
        opacity=0.38,
        name=f'corridor-upper-{subplot}',
        show_scalar_bar=False,
        **material,
    )
    plotter.add_mesh(
        guidance_tube,
        color='#174A7E',
        smooth_shading=True,
        name=f'guidance-{subplot}',
        label='FMM guidance path',
    )

    marker_points = np.vstack([start, goal]).astype(float)
    marker_points[:, 2] *= z_exaggeration
    marker_points[:, 2] += 3.0 * z_exaggeration
    plotter.add_points(
        marker_points[:1], color='#009E73', point_size=18,
        render_points_as_spheres=True, label='Start', name=f'start-{subplot}'
    )
    plotter.add_points(
        marker_points[1:], color='#E69F00', point_size=18,
        render_points_as_spheres=True, label='Goal', name=f'goal-{subplot}'
    )
    add_bounds(plotter, subplot, bounds, z_exaggeration, top_view)
    if show_legend:
        legend = plotter.add_legend(
            bcolor='white', border=True, size=(0.22, 0.13),
            loc='upper left', face=None
        )
        legend.SetPosition(0.18, 0.73)
        legend_text = legend.GetEntryTextProperty()
        legend_text.SetFontSize(14)
        legend_text.SetBold(False)
        legend_text.SetFontFamilyToArial()
    plotter.set_background('white')


def render(args):
    corridor_path = os.path.join(args.data_dir, 'search_FGDASTAR_corridor.mhd')
    arrival_path = os.path.join(
        args.data_dir, 'search_FGDASTAR_arrival_cost_3D.mhd'
    )
    guidance_path = os.path.join(args.data_dir, 'search_FGDASTAR_guidance_path.csv')
    for path in (corridor_path, arrival_path, guidance_path, args.dsm):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    info = load_field_info(corridor_path)
    nx, ny, nz = info['dims']
    sx, sy, sz = info['spacing']
    ox, oy, oz = info['origin']
    field_bounds = (
        ox, ox + (nx - 1) * sx,
        oy, oy + (ny - 1) * sy,
        oz * args.z_exag, (oz + (nz - 1) * sz) * args.z_exag,
    )
    dsm, scene_bounds = build_dsm_surface(args.dsm, args.z_exag)

    guidance = load_csv_points(guidance_path, ('x', 'y', 'z'))
    corridor_radius = args.corridor_radius_cells * sx
    corridor = build_continuous_corridor(
        guidance, corridor_radius, args.z_exag
    )
    arrival_info, arrival_volume = load_volume(arrival_path)
    arrival_values = sample_volume_nearest(
        corridor.points, arrival_info, arrival_volume, args.z_exag
    )
    finite_arrival = arrival_values[np.isfinite(arrival_values)]
    if finite_arrival.size == 0:
        raise ValueError('No finite arrival-cost samples found on corridor surface.')
    fill_value = float(np.nanmax(finite_arrival))
    corridor['Arrival cost (m)'] = np.nan_to_num(
        arrival_values, nan=fill_value, posinf=fill_value, neginf=0.0
    )
    arrival_range = (
        float(np.percentile(finite_arrival, 2.0)),
        float(np.percentile(finite_arrival, 98.0)),
    )
    line_radius = max(field_bounds[1] - field_bounds[0],
                      field_bounds[3] - field_bounds[2]) * 0.0014
    guidance_tube = make_tube(guidance, args.z_exag, line_radius)
    start, goal = guidance[0], guidance[-1]

    plotter = pv.Plotter(
        shape=(1, 1),
        off_screen=True,
        window_size=(1600, 1200),
        border=False,
    )
    add_scene(
        plotter, (0, 0), dsm, corridor, guidance_tube,
        start, goal, scene_bounds, args.z_exag, False, True,
        arrival_range,
    )
    plotter.subplot(0, 0)
    diag = np.linalg.norm([
        scene_bounds[1] - scene_bounds[0],
        scene_bounds[3] - scene_bounds[2],
        scene_bounds[5] - scene_bounds[4],
    ])
    plotter.renderer.remove_all_lights()
    plotter.enable_lightkit()
    plotter.reset_camera()
    plotter.camera.clipping_range = (0.1, diag * 5.0)
    plotter.camera.enable_parallel_projection = False
    plotter.camera.elevation = 5
    plotter.camera.azimuth = 0
    plotter.camera.distance = diag * 0.5
    plotter.camera.zoom(0.92)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    plotter.screenshot(
        output_path,
        transparent_background=False,
        window_size=(1800, 1300),
    )
    plotter.close()
    print(f'[Output] {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize the FMM-guided FDGA* search corridor.'
    )
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--dsm', default=DEFAULT_DSM)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument(
        '--output-name', default='FDGA_guidance_corridor_comparison.png'
    )
    parser.add_argument('--z-exag', type=float, default=1.5)
    parser.add_argument(
        '--corridor-radius-cells', type=float, default=5.0,
        help='连续走廊半径，单位为规划栅格数（默认：5）',
    )
    parser.add_argument('--dpi', type=int, default=300,
                        help='Reserved for publication export compatibility.')
    args = parser.parse_args()
    render(args)


if __name__ == '__main__':
    main()
