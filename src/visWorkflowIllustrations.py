#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate ten compact, high-legibility illustrations for the workflow diagram."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pyvista as pv
import vtk


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "compare4" / "final_v1" / "raw"
DEFAULT_OUTPUT = ROOT / "analysis" / "workflow_illustrations"
DSM_PATH = DATA / "SF_Downtown.tif"
SDF_PATH = DATA / "SF_Downtown_sdf.mhd"

COLORS = {
    "FDGA*": "#0072B2",
    "FMM": "#E69F00",
    "A*": "#D55E00",
    "RRT*": "#CC79A7",
    "Start": "#009E73",
    "Goal": "#D55E00",
}
PLANNERS = (
    ("FDGA*", "search_path_FGDASTAR.csv"),
    ("FMM", "search_path_FMM.csv"),
    ("A*", "search_path_ASTAR.csv"),
    ("RRT*", "search_path_RRTSTAR.csv"),
)
FMM_CMAP = LinearSegmentedColormap.from_list(
    "arrival", ["#009E73", "#56B4E9", "#F0E442", "#E69F00", "#D55E00"]
)
SDF_CMAP = LinearSegmentedColormap.from_list(
    "esdf_danger",
    ["#4A0E0E", "#B22222", "#FF4500", "#FFA500", "#FFD700", "#32CD32", "#228B22", "#006400"],
)
DSM_CMAP = LinearSegmentedColormap.from_list(
    "urban", ["#E8E8E8", "#D2D2D2", "#B8B8B8", "#969696", "#707070"]
)
MET_TYPES = {
    "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64,
    "MET_UCHAR": np.uint8, "MET_CHAR": np.int8,
    "MET_USHORT": np.uint16, "MET_SHORT": np.int16,
    "MET_UINT": np.uint32, "MET_INT": np.int32,
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "font.weight": "bold", "axes.labelweight": "bold",
        "axes.titleweight": "bold", "axes.titlesize": 18,
        "axes.labelsize": 16, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "axes.unicode_minus": False,
    })


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def parse_mhd(path: Path) -> dict:
    header = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                header[key.strip()] = value.strip()
    return {
        "dims": tuple(int(v) for v in header["DimSize"].split()),
        "spacing": tuple(float(v) for v in header["ElementSpacing"].split()),
        "origin": tuple(float(v) for v in header.get("Offset", "0 0 0").split()),
        "dtype": MET_TYPES[header["ElementType"]],
        "raw": path.parent / header["ElementDataFile"],
    }


def open_volume(path: Path) -> tuple[dict, np.memmap]:
    info = parse_mhd(path)
    nx, ny, nz = info["dims"]
    volume = np.memmap(info["raw"], dtype=info["dtype"], mode="r", shape=(nz, ny, nx))
    return info, volume


def read_dsm() -> tuple[np.ndarray, list[float], object]:
    import rasterio
    with rasterio.open(DSM_PATH) as source:
        elevation = source.read(1).astype(np.float32)
        transform = source.transform
        nodata = source.nodata
    invalid = ~np.isfinite(elevation)
    if nodata is not None:
        invalid |= np.isclose(elevation, nodata)
    elevation[invalid] = np.nanmin(elevation[~invalid])
    extent = [transform.c, transform.c + elevation.shape[1] * transform.a,
              transform.f + elevation.shape[0] * transform.e, transform.f]
    return elevation, extent, transform


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"[output] {path}")


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(width=1.6, length=5)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def add_clear_legend(ax: plt.Axes, handles=None, ncol=1, loc="upper left", **kwargs):
    legend = ax.legend(handles=handles, ncol=ncol, loc=loc, fontsize=14,
                       frameon=True, fancybox=False, facecolor="white",
                       edgecolor="black", framealpha=1.0, **kwargs)
    legend.get_frame().set_linewidth(1.6)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    return legend


def path_legend_handles() -> list[Line2D]:
    handles = [Line2D([0], [0], color=COLORS[name], lw=5, label=name)
               for name, _ in PLANNERS]
    handles += [
        Line2D([0], [0], marker="*", markersize=16, linestyle="none",
               markerfacecolor=COLORS["Start"], markeredgecolor="black", label="Start"),
        Line2D([0], [0], marker="*", markersize=16, linestyle="none",
               markerfacecolor=COLORS["Goal"], markeredgecolor="black", label="Goal"),
    ]
    return handles


def build_dsm_surface(z_exag: float = 1.5):
    elevation, extent, transform = read_dsm()
    rows, cols = elevation.shape
    step = max(1, int(np.sqrt(rows * cols / 400000)))
    sampled = elevation[::step, ::step]
    x = transform.c + np.arange(sampled.shape[1]) * transform.a * step
    y = transform.f + np.arange(sampled.shape[0]) * transform.e * step
    xx, yy = np.meshgrid(x, y)
    surface = pv.StructuredGrid(xx, yy, sampled * z_exag).extract_surface(
        algorithm="dataset_surface"
    )
    surface["Elevation (m)"] = surface.points[:, 2] / z_exag
    bounds = (float(x.min()), float(x.max()), float(y.min()), float(y.max()),
              float(sampled.min() * z_exag), float(sampled.max() * z_exag))
    return surface, bounds, elevation, extent


def set_camera(plotter: pv.Plotter, bounds) -> None:
    diag = np.linalg.norm([
        bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]
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


def add_pv_bounds(plotter: pv.Plotter, bounds) -> None:
    actor = plotter.show_bounds(xtitle="X (km)", ytitle="Y (km)", ztitle="Z (km)",
        axes_ranges=(bounds[0] / 1000, bounds[1] / 1000, bounds[2] / 1000,
                     bounds[3] / 1000, bounds[4] / 1500, bounds[5] / 1500),
        fmt="%.1f", n_xlabels=3, n_ylabels=3, n_zlabels=3, font_size=18,
        grid=False, location="outer", use_2d=False)
    actor.SetLabelOffset(26)


def new_plotter() -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=(2400, 1733), border=False)
    plotter.set_background("white")
    return plotter


def render_dsm(surface, bounds, output: Path, dpi: int) -> None:
    colors = [
        "#D7E8EF", "#A8CAD3", "#80ADB5", "#D8CEA8",
        "#E8B85C", "#D9824B", "#A94F3D",
    ]
    elevation = np.asarray(surface["Elevation (m)"])
    value_range = (float(np.nanmin(elevation)), float(np.nanmax(elevation)))

    plotter = new_plotter()
    plotter.add_mesh(
        surface,
        scalars="Elevation (m)",
        cmap=colors,
        clim=value_range,
        lighting=True,
        smooth_shading=True,
        ambient=0.65,
        diffuse=0.55,
        specular=0.08,
        show_edges=False,
        show_scalar_bar=False,
    )
    add_pv_bounds(plotter, bounds)
    set_camera(plotter, bounds)
    image = plotter.screenshot(return_img=True, window_size=(2400, 1600))
    plotter.close()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    cmap = LinearSegmentedColormap.from_list("dsm_elevation", colors)
    cax = fig.add_axes([0.84, 0.22, 0.022, 0.56])
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(
            norm=Normalize(vmin=value_range[0], vmax=value_range[1]),
            cmap=cmap,
        ),
        cax=cax,
    )
    colorbar.set_label("Elevation (m)", fontsize=16, fontweight="bold", labelpad=10)
    colorbar.set_ticks(np.linspace(value_range[0], value_range[1], 5))
    colorbar.ax.tick_params(labelsize=13, width=1.4, length=5)
    for tick in colorbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    colorbar.outline.set_linewidth(1.4)

    fig.savefig(output, dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)
    print(f"[output] {output}")


def screenshot_with_legend(plotter: pv.Plotter, output: Path, dpi: int,
                           handles, anchor=(0.12, 0.86), ncol=1,
                           colorbar=None) -> None:
    image = plotter.screenshot(return_img=True, window_size=(2400, 1733))
    plotter.close()
    fig, ax = plt.subplots(figsize=(12, 8.665))
    ax.imshow(image)
    ax.axis("off")
    add_clear_legend(ax, handles=handles, ncol=ncol, loc="upper left",
                     bbox_to_anchor=anchor, borderaxespad=0.0)
    fig.subplots_adjust(0, 0, 1, 1)
    if colorbar is not None:
        cmap, norm, label = colorbar
        cax = fig.add_axes([0.90, 0.23, 0.022, 0.50])
        bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        bar.set_label(label, fontsize=15, fontweight="bold")
        bar.ax.tick_params(labelsize=12, width=1.4)
        for tick in bar.ax.get_yticklabels():
            tick.set_fontweight("bold")
    save_figure(fig, output, dpi)


def render_corridor(surface, bounds, output: Path, dpi: int) -> None:
    guidance = load_csv(RAW / "search_FGDASTAR_guidance_path.csv")[:, 4:7]
    arrival_info, arrival = open_volume(RAW / "search_FGDASTAR_arrival_cost_3D.mhd")
    z_exag = 1.5
    centerline = guidance.copy()
    centerline[:, 2] *= z_exag
    radius = 5.0 * arrival_info["spacing"][0]
    polyline = pv.lines_from_points(centerline, close=False)
    margin = radius * 1.15
    modeller = vtk.vtkImplicitModeller()
    modeller.SetInputData(polyline)
    modeller.SetModelBounds(
        centerline[:, 0].min() - margin, centerline[:, 0].max() + margin,
        centerline[:, 1].min() - margin, centerline[:, 1].max() + margin,
        centerline[:, 2].min() - margin, centerline[:, 2].max() + margin,
    )
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
    corridor = pv.wrap(contour.GetOutput()).clean().smooth_taubin(
        n_iter=40, pass_band=0.08, normalize_coordinates=True
    ).compute_normals(point_normals=True, cell_normals=True, auto_orient_normals=True)

    physical = corridor.points.copy()
    physical[:, 2] /= z_exag
    idx = np.rint((physical - np.asarray(arrival_info["origin"])) /
                  np.asarray(arrival_info["spacing"])).astype(int)
    nx, ny, nz = arrival_info["dims"]
    idx = np.clip(idx, [0, 0, 0], [nx - 1, ny - 1, nz - 1])
    values = np.asarray(arrival[idx[:, 2], idx[:, 1], idx[:, 0]], dtype=float)
    finite = values[np.isfinite(values)]
    fill = float(np.max(finite))
    corridor["Arrival cost (m)"] = np.nan_to_num(values, nan=fill, posinf=fill, neginf=0.0)
    normals = np.asarray(corridor.cell_data["Normals"])
    upper = corridor.extract_cells(normals[:, 2] > 0.0)
    lower = corridor.extract_cells(normals[:, 2] <= 0.0)
    shown = guidance.copy()
    shown[:, 2] = shown[:, 2] * z_exag + 1.5 * z_exag
    line_radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.0014
    guide_tube = pv.lines_from_points(shown, close=False).tube(radius=line_radius, n_sides=16)

    plotter = new_plotter()
    plotter.add_mesh(surface, scalars="Elevation (m)", cmap=DSM_CMAP, show_scalar_bar=False,
                     lighting=True, smooth_shading=True, ambient=0.48, diffuse=0.72,
                     specular=0.08, specular_power=14.0)
    arrival_limits = np.percentile(finite, [2, 98])
    material = dict(scalars="Arrival cost (m)", cmap=FMM_CMAP, clim=arrival_limits,
                    lighting=True, ambient=0.42, diffuse=0.72, specular=0.48,
                    specular_power=48.0, smooth_shading=True, show_edges=False,
                    show_scalar_bar=False)
    plotter.add_mesh(lower, opacity=1.0, **material)
    plotter.add_mesh(upper, opacity=0.38, **material)
    plotter.add_mesh(guide_tube, color="#174A7E", smooth_shading=True)
    markers = guidance[[0, -1]].copy()
    markers[:, 2] = markers[:, 2] * z_exag + 3.0 * z_exag
    plotter.add_points(markers[:1], color=COLORS["Start"], point_size=28, render_points_as_spheres=True)
    plotter.add_points(markers[1:], color="#E69F00", point_size=28, render_points_as_spheres=True)
    add_pv_bounds(plotter, bounds)
    set_camera(plotter, bounds)
    handles = [Line2D([0], [0], color="#174A7E", lw=6, label="FMM guidance path"),
               Line2D([0], [0], color="#E69F00", lw=9, alpha=.7, label="Search corridor"),
               *path_legend_handles()[-2:]]
    screenshot_with_legend(plotter, output, dpi, handles, anchor=(0.12, 0.86),
        colorbar=(FMM_CMAP, Normalize(*arrival_limits), "Arrival cost (m)"))


def load_paths() -> list[tuple[str, np.ndarray]]:
    return [(name, load_csv(RAW / filename)[:, :3]) for name, filename in PLANNERS]


def render_paths_main(surface, bounds, paths, output: Path, dpi: int) -> None:
    plotter = new_plotter()
    plotter.add_mesh(surface, scalars="Elevation (m)", cmap=DSM_CMAP, show_scalar_bar=False,
                     lighting=True, smooth_shading=True, ambient=0.48, diffuse=0.72,
                     specular=0.08, specular_power=14.0, show_edges=False)
    radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2]) * 0.00125
    for name, points in paths:
        shown = points.copy(); shown[:, 2] = shown[:, 2] * 1.5 + 2.0 * 1.5
        plotter.add_mesh(pv.lines_from_points(shown, close=False).tube(radius=radius, n_sides=18),
                         color=COLORS[name], lighting=True, ambient=0.45, diffuse=0.75,
                         specular=0.25, specular_power=25.0, smooth_shading=True)
    markers = paths[0][1][[0, -1]].copy(); markers[:, 2] = markers[:, 2] * 1.5 + 4.0 * 1.5
    plotter.add_points(markers[:1], color=COLORS["Start"], point_size=30, render_points_as_spheres=True)
    plotter.add_points(markers[1:], color=COLORS["Goal"], point_size=30, render_points_as_spheres=True)
    add_pv_bounds(plotter, bounds)
    set_camera(plotter, bounds)
    screenshot_with_legend(plotter, output, dpi, path_legend_handles(), anchor=(0.12, 0.86), ncol=2)


def render_paths_top(elevation, extent, paths, output: Path, dpi: int) -> None:
    valid = elevation[np.isfinite(elevation)]
    low, high = np.percentile(valid, [2, 98])
    fig, ax = plt.subplots(figsize=(10, 9), constrained_layout=True)
    ax.imshow(elevation, cmap="gray", origin="upper", extent=extent, vmin=low, vmax=high,
              interpolation="nearest", alpha=0.48)
    for name, points in paths:
        ax.plot(points[:, 0], points[:, 1], color="white", lw=4.2, zorder=4)
        ax.plot(points[:, 0], points[:, 1], color=COLORS[name], lw=2.4, zorder=5)
    start, goal = paths[0][1][0], paths[0][1][-1]
    ax.scatter(start[0], start[1], marker="*", s=400, color=COLORS["Start"], edgecolor="black", lw=1.3, zorder=8)
    ax.scatter(goal[0], goal[1], marker="*", s=400, color="#E69F00", edgecolor="black", lw=1.3, zorder=8)
    add_clear_legend(ax, path_legend_handles(), ncol=2)
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel="Easting X (m)", ylabel="Northing Y (m)")
    ax.set_aspect("equal"); ax.grid(alpha=.2, ls="--"); style_axis(ax)
    save_figure(fig, output, dpi)


class ESDFNorm(Normalize):
    def __init__(self, d_max: float):
        self._d_max = max(float(d_max), 20.0)
        super().__init__(vmin=0.0, vmax=float(d_max))

    def __call__(self, value, clip=None):
        d = np.ma.asanyarray(value, dtype=float)
        result = np.ma.zeros_like(d)
        middle = (d > 0) & (d <= 10)
        result[middle] = 0.08 + 0.77 * d[middle] / 10.0
        far = d > 10
        result[far] = 0.85 + 0.15 * np.minimum(1.0, (d[far] - 10.0) / (self._d_max - 10.0))
        if np.ma.is_masked(d):
            result.mask = d.mask
        return result

    def inverse(self, value):
        v = np.ma.asanyarray(value, dtype=float)
        result = np.ma.zeros_like(v)
        middle = (v > 0.08) & (v <= 0.85)
        result[middle] = (v[middle] - 0.08) / 0.77 * 10.0
        far = v > 0.85
        result[far] = 10.0 + (v[far] - 0.85) / 0.15 * (self._d_max - 10.0)
        return result


def volume_slices(mhd_path: Path, output: Path, dpi: int, kind: str) -> None:
    info, volume = open_volume(mhd_path)
    nx, ny, nz = info["dims"]; sx, sy, sz = info["spacing"]
    ox, oy, oz = info["origin"]
    if kind == "fmm":
        path = load_csv(RAW / "search_path_FMM.csv")[:, :3]
        start, goal = path[0] - np.asarray(info["origin"]), path[-1] - np.asarray(info["origin"])
        ix = int(np.clip(round(start[0] / sx), 0, nx - 1))
        iy = int(np.clip(round(start[1] / sy), 0, ny - 1))
        iz = int(np.clip(round(start[2] / sz), 0, nz - 1))
        finite = np.asarray(volume[:, ::4, ::4]); finite = finite[np.isfinite(finite)]
        fallback = float(np.percentile(finite, 95))
        start_value = float(volume[iz, iy, ix])
        vmax = (start_value if np.isfinite(start_value) and start_value > 0 else fallback) * 1.05
        display = np.ma.minimum(np.ma.masked_invalid(volume), vmax)
        positions = [(48.0, 615.0, 621.0), (99.0, 1026.0, 1035.0),
                     (150.0, 1437.0, 1449.0)]
        cmap, norm, label = FMM_CMAP, Normalize(0.0, vmax), "Arrival cost (m)"
    else:
        display = volume
        vmax = float(np.nanmax(np.asarray(volume[:, ::8, ::8])))
        positions = [(50.0, ny * sy * .3, nx * sx * .3),
                     (100.0, ny * sy * .5, nx * sx * .5),
                     (150.0, ny * sy * .7, nx * sx * .7)]
        cmap, norm, label = SDF_CMAP, ESDFNorm(vmax), "Distance (m)"
        start = goal = None

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    x_extent, y_extent, z_extent = [0, nx * sx], [0, ny * sy], [0, nz * sz]
    image = None
    for row, (z_value, y_value, x_value) in enumerate(positions):
        zi = int(np.clip(round(z_value / sz), 1, nz - 1))
        yi = int(np.clip(round(y_value / sy), 1, ny - 1))
        xi = int(np.clip(round(x_value / sx), 1, nx - 1))
        z_value, y_value, x_value = zi * sz, yi * sy, xi * sx
        panels = [
            (display[zi, :, :], x_extent + y_extent, "Top View (XY)", f"Z = {z_value:.0f} m", "X (m)", "Y (m)", "equal", x_value, y_value),
            (display[:, yi, :], x_extent + z_extent, "South View (XZ)", f"Y = {y_value:.0f} m", "X (m)", "Z (m)", "auto", x_value, z_value),
            (display[:, :, xi], y_extent + z_extent, "West View (YZ)", f"X = {x_value:.0f} m", "Y (m)", "Z (m)", "auto", y_value, z_value),
        ]
        for col, (data, extent, view, position, xlabel, ylabel, aspect, vline, hline) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(data, cmap=cmap, norm=norm, origin="lower", extent=extent,
                              aspect=aspect, interpolation="nearest")
            ax.set_title(f"{view}, {position}", fontsize=15)
            ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
            ax.axhline(hline, color="white", ls=":", lw=1.5, alpha=.8)
            ax.axvline(vline, color="white", ls=":", lw=1.5, alpha=.8)
            ax.grid(alpha=.2, ls="--", lw=.5); style_axis(ax)
            if kind == "fmm":
                coords = ((start[0], start[1]), (goal[0], goal[1])) if col == 0 else (
                    ((start[0], start[2]), (goal[0], goal[2])) if col == 1 else
                    ((start[1], start[2]), (goal[1], goal[2])))
                ax.scatter(*coords[0], marker="*", s=220, c=COLORS["Start"], edgecolors="black", lw=1.0, zorder=20)
                ax.scatter(*coords[1], marker="*", s=220, c=COLORS["Goal"], edgecolors="black", lw=1.0, zorder=20)
    fig.subplots_adjust(right=.92, bottom=.08, wspace=.25, hspace=.35)
    cax = fig.add_axes([.93, .12, .012, .76])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label(label, fontweight="bold", fontsize=16)
    if kind == "sdf":
        cbar.set_ticks([0, 2.5, 5, 7.5, 10, vmax])
        cbar.set_ticklabels(["0.0", "2.5", "5.0", "7.5", "10.0", f"{vmax:.1f}"])
    cbar.ax.tick_params(labelsize=13, width=1.5)
    for tick in cbar.ax.get_yticklabels(): tick.set_fontweight("bold")
    if kind == "fmm":
        fig.legend(handles=path_legend_handles()[-2:], loc="lower center",
                   bbox_to_anchor=(.52, .012), ncol=2, fontsize=14,
                   frameon=True, fancybox=False, edgecolor="black")
    save_figure(fig, output, dpi)


def terrain_crop(ax, elevation, extent, x_values, y_values):
    mx = (x_values.max() - x_values.min()) * .20
    my = (y_values.max() - y_values.min()) * .20
    bounds = (max(extent[0], x_values.min() - mx), min(extent[1], x_values.max() + mx),
              max(extent[2], y_values.min() - my), min(extent[3], y_values.max() + my))
    ax.imshow(elevation, cmap="gray", origin="upper", extent=extent, alpha=.35,
              aspect="equal", interpolation="nearest")
    ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:]); return bounds


def markers_and_path(ax, path):
    ax.plot(path[:, 0], path[:, 1], color="white", lw=5.2, zorder=14)
    ax.plot(path[:, 0], path[:, 1], color="black", lw=3.0, label="Final path", zorder=15)
    ax.scatter(path[0, 0], path[0, 1], marker="*", s=430, c=COLORS["Start"], edgecolor="black", lw=1.3, label="Start", zorder=20)
    ax.scatter(path[-1, 0], path[-1, 1], marker="*", s=430, c=COLORS["Goal"], edgecolor="black", lw=1.3, label="Goal", zorder=20)


def finalize_search(ax, title):
    ax.set_title(title); ax.set_xlabel("Easting X (m)"); ax.set_ylabel("Northing Y (m)")
    ax.set_aspect("equal"); ax.grid(alpha=.2, ls="--"); style_axis(ax); add_clear_legend(ax)


def render_graph_search(elevation, extent, nodes_file: str, path_file: str,
                        output: Path, dpi: int, title: str, rrt=False) -> None:
    nodes = load_csv(RAW / nodes_file); path = load_csv(RAW / path_file)[:, :3]
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    all_x = np.r_[nodes[:, 0], path[:, 0]]; all_y = np.r_[nodes[:, 1], path[:, 1]]
    terrain_crop(ax, elevation, extent, all_x, all_y)
    parents = nodes[:, 3].astype(int); valid = np.where(parents >= 0)[0]
    segments = np.stack((nodes[valid, :2], nodes[parents[valid], :2]), axis=1)
    edge_label = "Tree edges" if rrt else "Search edges"
    ax.add_collection(LineCollection(segments, colors="#0072B2", lw=.6 if rrt else .5,
                                     alpha=.45 if rrt else .4, label=edge_label, zorder=5))
    if rrt:
        goal_index = int(np.argmin(np.sum((nodes[:, :2] - path[-1, :2]) ** 2, axis=1)))
        backtrace = []
        index = goal_index
        while index >= 0:
            backtrace.append(nodes[index, :2])
            index = parents[index]
        backtrace = np.asarray(backtrace)
        ax.plot(backtrace[:, 0], backtrace[:, 1], color="white", lw=3.5, zorder=11)
        ax.plot(backtrace[:, 0], backtrace[:, 1], color=COLORS["A*"], lw=1.8,
                label="Backtracked tree path", zorder=12)
        ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c="#0072B2", alpha=.5,
                   label="Tree nodes", zorder=8)
    else:
        points = ax.scatter(nodes[:, 0], nodes[:, 1], s=4, c=nodes[:, 2], cmap="RdYlGn_r",
                            alpha=.7, vmin=np.percentile(nodes[:, 2], 2),
                            vmax=np.percentile(nodes[:, 2], 98), label="Search nodes", zorder=10)
        cbar = fig.colorbar(points, ax=ax, shrink=.7, pad=.02)
        cbar.set_label("Node altitude Z (m)", fontweight="bold", fontsize=16)
    markers_and_path(ax, path); finalize_search(ax, title)
    save_figure(fig, output, dpi)


def render_fmm_search(elevation, extent, output: Path, dpi: int) -> None:
    path = load_csv(RAW / "search_path_FMM.csv")[:, :3]
    table = load_csv(RAW / "search_FMM_Tfield.csv")
    xs, ys, values = table[:, 0], table[:, 1], table[:, 2]
    ux, uy = np.unique(xs), np.unique(ys); field = values.reshape(len(uy), len(ux))
    ix = np.argmin(abs(ux - path[0, 0])); iy = np.argmin(abs(uy - path[0, 1]))
    vmax = field[iy, ix] if np.isfinite(field[iy, ix]) else np.nanpercentile(field, 95)
    shown = np.where(np.isfinite(field), np.minimum(field, vmax * 1.05), np.nan)
    levels = np.linspace(0, vmax * 1.05, 20)
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    terrain_crop(ax, elevation, extent, np.r_[xs, path[:, 0]], np.r_[ys, path[:, 1]])
    filled = ax.contourf(ux, uy, shown, levels=levels, cmap=FMM_CMAP, alpha=.86, zorder=3)
    ax.contour(ux, uy, shown, levels=levels[::4], colors="white", lw=1.0, alpha=.6, zorder=4)
    markers_and_path(ax, path); finalize_search(ax, "FMM Wavefront")
    cbar = fig.colorbar(filled, ax=ax, shrink=.76, pad=.02)
    cbar.set_label("Arrival cost (m)", fontweight="bold", fontsize=16)
    for tick in cbar.ax.get_yticklabels(): tick.set_fontweight("bold")
    save_figure(fig, output, dpi)


def validate_inputs() -> None:
    required = [DSM_PATH, SDF_PATH, RAW / "search_FMM_Tfield_3D.mhd",
                RAW / "search_FGDASTAR_arrival_cost_3D.mhd",
                RAW / "search_FGDASTAR_guidance_path.csv"]
    required += [RAW / filename for _, filename in PLANNERS]
    required += [RAW / "search_FGDASTAR_nodes.csv", RAW / "search_ASTAR_nodes.csv",
                 RAW / "search_RRTSTAR_tree.csv", RAW / "search_FMM_Tfield.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style(); validate_inputs()

    surface, bounds, elevation, extent = build_dsm_surface()
    paths = load_paths()
    render_dsm(surface, bounds, output / "DSM_3D_main_view.png", args.dpi)
    render_corridor(surface, bounds, output / "FDGA_guidance_corridor_comparison.png", args.dpi)
    volume_slices(RAW / "search_FMM_Tfield_3D.mhd", output / "FMM_arrival_cost_orthographic_views.png", args.dpi, "fmm")
    render_paths_main(surface, bounds, paths, output / "four_algorithms_paths_DSM_main_view.png", args.dpi)
    render_paths_top(elevation, extent, paths, output / "four_algorithms_paths_DSM_top_view.png", args.dpi)
    render_graph_search(elevation, extent, "search_ASTAR_nodes.csv", "search_path_ASTAR.csv",
                        output / "search_process_ASTAR.png", args.dpi,
                        "A* Search Process: Discovered Nodes and Expansion Edges")
    render_graph_search(elevation, extent, "search_FGDASTAR_nodes.csv", "search_path_FGDASTAR.csv",
                        output / "search_process_FDGA.png", args.dpi,
                        "FDGA* Search Process: Direction-Aware Node Expansion")
    render_fmm_search(elevation, extent, output / "search_process_FMM.png", args.dpi)
    render_graph_search(elevation, extent, "search_RRTSTAR_tree.csv", "search_path_RRTSTAR.csv",
                        output / "search_process_RRTSTAR.png", args.dpi,
                        "RRT* Search Process: Random Tree Expansion", rrt=True)
    volume_slices(SDF_PATH, output / "SF_Downtown_sdf_orthographic_views.png", args.dpi, "sdf")
    print(f"Generated 10 workflow illustrations in {output}")


if __name__ == "__main__":
    main()
