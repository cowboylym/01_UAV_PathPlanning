#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render four planning paths over the urban DSM in main and top views."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.lines import Line2D

from visFGDACorridor import DSM_CMAP, add_bounds, build_dsm_surface, load_csv_points
from visSearchProcess import read_geotiff


OUTPUT_DIR = '../analysis'
DEFAULT_DATA_DIR = '../data/compare4/final_v1/raw'
DEFAULT_DSM = '../data/SF_Downtown.tif'

PLANNERS = (
    ('FDGA*', 'search_path_FGDASTAR.csv', '#0072B2'),
    ('FMM', 'search_path_FMM.csv', '#E69F00'),
    ('A*', 'search_path_ASTAR.csv', '#D55E00'),
    ('RRT*', 'search_path_RRTSTAR.csv', '#CC79A7'),
)


def make_path_tube(points, z_exaggeration, radius):
    displayed = np.asarray(points, dtype=float).copy()
    displayed[:, 2] *= z_exaggeration
    displayed[:, 2] += 2.0 * z_exaggeration
    return pv.lines_from_points(displayed, close=False).tube(
        radius=radius,
        n_sides=18,
    )


def legend_handles(paths):
    handles = [
        Line2D([0], [0], color=color, linewidth=2.4, label=label)
        for label, _, color in paths
    ]
    handles.extend([
        Line2D([0], [0], marker='*', markersize=13, linestyle='None',
               markerfacecolor='#009E73', markeredgecolor='black', label='Start'),
        Line2D([0], [0], marker='*', markersize=13, linestyle='None',
               markerfacecolor='#E69F00', markeredgecolor='black', label='Goal'),
    ])
    return handles


def add_scene(plotter, dsm, paths, bounds, z_exaggeration, top_view):
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
        show_edges=False,
        show_scalar_bar=False,
        name='dsm',
    )

    scene_span = max(bounds[1] - bounds[0], bounds[3] - bounds[2])
    path_radius = scene_span * 0.00125
    for label, points, color in paths:
        tube = make_path_tube(points, z_exaggeration, path_radius)
        plotter.add_mesh(
            tube,
            color=color,
            lighting=True,
            ambient=0.45,
            diffuse=0.75,
            specular=0.25,
            specular_power=25.0,
            smooth_shading=True,
            label=label,
            name=f'path-{label}',
        )

    start = paths[0][1][0].copy()
    goal = paths[0][1][-1].copy()
    markers = np.vstack([start, goal])
    markers[:, 2] *= z_exaggeration
    markers[:, 2] += 4.0 * z_exaggeration
    plotter.add_points(
        markers[:1],
        color='#009E73',
        point_size=18,
        render_points_as_spheres=True,
        label='Start',
        name='start',
    )
    plotter.add_points(
        markers[1:],
        color='#E69F00',
        point_size=18,
        render_points_as_spheres=True,
        label='Goal',
        name='goal',
    )

    add_bounds(plotter, (0, 0), bounds, z_exaggeration, top_view=top_view)
    plotter.set_background('white')


def set_main_camera(plotter, bounds):
    diag = np.linalg.norm([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
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


def render_top_view(paths, args):
    elevation, extent = read_geotiff(args.dsm)
    valid = np.isfinite(elevation)
    if np.any(valid):
        low, high = np.percentile(elevation[valid], [2.0, 98.0])
    else:
        low, high = 0.0, 1.0

    figure, ax = plt.subplots(figsize=(9.0, 8.4), constrained_layout=True)
    ax.imshow(
        elevation,
        cmap='gray',
        origin='upper',
        extent=extent,
        vmin=low,
        vmax=high,
        interpolation='nearest',
        alpha=0.48,
        zorder=1,
    )

    for label, points, color in paths:
        ax.plot(points[:, 0], points[:, 1], color='white', linewidth=4.2, zorder=4)
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2.4,
                label=label, zorder=5)

    start = paths[0][1][0]
    goal = paths[0][1][-1]
    ax.scatter(start[0], start[1], marker='*', s=330, color='#009E73',
               edgecolor='black', linewidth=1.1, zorder=7)
    ax.scatter(goal[0], goal[1], marker='*', s=330, color='#E69F00',
               edgecolor='black', linewidth=1.1, zorder=7)

    legend = ax.legend(handles=legend_handles(paths), loc='upper left', fontsize=10,
                       framealpha=1.0, edgecolor='black', facecolor='white',
                       fancybox=False, borderaxespad=0.6)
    legend.get_frame().set_linewidth(1.2)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Easting X (m)', fontsize=11)
    ax.set_ylabel('Northing Y (m)', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.2, linestyle='--', linewidth=0.5)

    output_path = os.path.join(
        args.output_dir, 'four_algorithms_paths_DSM_top_view.png'
    )
    figure.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(figure)
    print(f'[Output] {output_path}')


def render_main_view(dsm, paths, bounds, args):
    plotter = pv.Plotter(
        shape=(1, 1),
        off_screen=True,
        window_size=(1600, 1200),
        border=False,
    )
    add_scene(plotter, dsm, paths, bounds, args.z_exag, top_view=False)
    set_main_camera(plotter, bounds)
    output_path = os.path.join(
        args.output_dir, 'four_algorithms_paths_DSM_main_view.png'
    )
    image = plotter.screenshot(
        return_img=True,
        transparent_background=False,
        window_size=(1800, 1300),
    )
    plotter.close()

    figure, ax = plt.subplots(figsize=(9.0, 6.5), dpi=200)
    ax.imshow(image)
    ax.set_axis_off()
    legend = ax.legend(handles=legend_handles(paths), loc='upper left',
                       bbox_to_anchor=(0.16, 0.82), fontsize=10,
                       framealpha=1.0, edgecolor='black', facecolor='white',
                       fancybox=False, borderaxespad=0.6)
    legend.get_frame().set_linewidth(1.2)
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure.savefig(output_path, dpi=200, facecolor='white')
    plt.close(figure)
    print(f'[Output] {output_path}')


def render(args):
    path_specs = []
    for label, filename, color in PLANNERS:
        path = os.path.join(args.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        path_specs.append((label, load_csv_points(path, ('x', 'y', 'z')), color))
    if not os.path.exists(args.dsm):
        raise FileNotFoundError(args.dsm)

    dsm, bounds = build_dsm_surface(args.dsm, args.z_exag)
    os.makedirs(args.output_dir, exist_ok=True)
    render_main_view(dsm, path_specs, bounds, args)
    render_top_view(path_specs, args)


def main():
    parser = argparse.ArgumentParser(
        description='Render FDGA*, FMM, A*, and RRT* paths over the urban DSM.'
    )
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--dsm', default=DEFAULT_DSM)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    parser.add_argument('--z-exag', type=float, default=1.5)
    args = parser.parse_args()
    render(args)


if __name__ == '__main__':
    main()
