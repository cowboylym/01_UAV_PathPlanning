from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / 'analysis'
OUTPUT_PATH = ANALYSIS_DIR / 'FDGA_technical_workflow.png'

COLORS = {
    'sidebar': '#AFC8DF',
    'header': '#F3D3B7',
    'panel': '#FFFDF8',
    'subpanel': '#F7F9FB',
    'blue': '#3579A8',
    'blue_light': '#DDECF5',
    'green': '#3A9473',
    'green_light': '#DDF0E7',
    'orange': '#D9822B',
    'orange_light': '#FCE7D2',
    'purple': '#7868A6',
    'purple_light': '#E9E4F3',
    'line': '#69747D',
    'text': '#1F2933',
}


def add_text(ax, x, y, text, size=10, weight='normal', color=None,
             ha='center', va='center', rotation=0):
    ax.text(
        x, y, text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS['text'],
        ha=ha,
        va=va,
        rotation=rotation,
        transform=ax.transAxes,
        zorder=20,
    )


def add_box(ax, x, y, w, h, text='', facecolor='white', edgecolor=None,
            radius=0.008, linewidth=1.0, size=9, weight='normal'):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad=0.004,rounding_size={radius}',
        facecolor=facecolor,
        edgecolor=edgecolor or COLORS['line'],
        linewidth=linewidth,
        transform=ax.transAxes,
        zorder=5,
    )
    ax.add_patch(box)
    if text:
        add_text(ax, x + w / 2, y + h / 2, text, size=size, weight=weight)
    return box


def add_arrow(ax, start, end, color=None, width=1.7, mutation_scale=15):
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle='-|>',
        mutation_scale=mutation_scale,
        linewidth=width,
        color=color or COLORS['blue'],
        transform=ax.transAxes,
        zorder=15,
    )
    ax.add_patch(arrow)


def add_image(ax, path, x, y, w, h, crop=None):
    image = Image.open(path).convert('RGB')
    if crop is not None:
        left, top, right, bottom = crop
        iw, ih = image.size
        image = image.crop((left * iw, top * ih, right * iw, bottom * ih))
    inset = ax.inset_axes([x, y, w, h], transform=ax.transAxes, zorder=8)
    inset.imshow(image)
    inset.set_axis_off()
    return inset


def add_section(ax, y, h, title, color):
    ax.add_patch(patches.Rectangle(
        (0.006, y), 0.988, h,
        facecolor='white', edgecolor='#AAB2B8', linewidth=1.0,
        transform=ax.transAxes, zorder=0,
    ))
    ax.add_patch(patches.Rectangle(
        (0.006, y), 0.055, h,
        facecolor=COLORS['sidebar'], edgecolor='#7EA2BF', linewidth=0.8,
        transform=ax.transAxes, zorder=2,
    ))
    add_text(ax, 0.0335, y + h / 2, title, size=12, weight='bold', rotation=90)


def add_panel(ax, x, y, w, h, title, header_color=None):
    add_box(ax, x, y, w, h, facecolor=COLORS['panel'], edgecolor='#AAB2B8', radius=0.004)
    header_h = 0.12 * h
    ax.add_patch(patches.Rectangle(
        (x, y + h - header_h), w, header_h,
        facecolor=header_color or COLORS['header'], edgecolor='none',
        transform=ax.transAxes, zorder=6,
    ))
    add_text(ax, x + w / 2, y + h - header_h / 2, title, size=10.5, weight='bold')
    return header_h


def draw_workflow():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'mathtext.fontset': 'dejavusans',
    })

    fig = plt.figure(figsize=(16, 11), dpi=180, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    add_section(ax, 0.555, 0.435, 'Data and Environment Modeling', COLORS['green'])
    add_section(ax, 0.245, 0.295, 'FDGA* Path Planning', COLORS['orange'])
    add_section(ax, 0.015, 0.215, 'Experimental Evaluation', COLORS['purple'])

    # Data and environment modeling
    add_panel(ax, 0.075, 0.805, 0.895, 0.165, 'Urban Digital Surface Model (DSM)')
    add_image(
        ax,
        ANALYSIS_DIR / 'DSM_3D_main_view.png',
        0.245, 0.812, 0.555, 0.132,
        crop=(0.08, 0.13, 0.89, 0.88),
    )

    add_arrow(ax, (0.50, 0.802), (0.50, 0.772), color=COLORS['green'], mutation_scale=20)

    add_panel(ax, 0.075, 0.575, 0.275, 0.19, 'DSM-Based 3-D Occupancy Modeling')
    add_image(
        ax,
        ANALYSIS_DIR / 'DSM_3D_main_view.png',
        0.090, 0.590, 0.150, 0.125,
        crop=(0.10, 0.18, 0.86, 0.86),
    )
    add_arrow(ax, (0.242, 0.645), (0.282, 0.645), color=COLORS['green'])
    add_box(ax, 0.286, 0.606, 0.048, 0.075, '1 m\nvoxel\ngrid', COLORS['green_light'], COLORS['green'], size=8.3, weight='bold')
    add_text(ax, 0.212, 0.584, 'DSM elevation → occupied voxels', size=8.2, color='#52606D')

    add_panel(ax, 0.375, 0.575, 0.275, 0.19, 'Three-Dimensional ESDF')
    add_image(
        ax,
        ANALYSIS_DIR / 'SF_Downtown_sdf_volume_rendering.png',
        0.390, 0.590, 0.170, 0.125,
        crop=(0.09, 0.15, 0.89, 0.89),
    )
    add_box(ax, 0.566, 0.615, 0.067, 0.065,
            '$D>0$  free\n$D=0$  surface\n$D<0$  occupied',
            COLORS['green_light'], COLORS['green'], size=8.0)
    add_text(ax, 0.510, 0.584, 'Euclidean distance and clearance risk', size=8.2, color='#52606D')

    add_panel(ax, 0.675, 0.575, 0.295, 0.19, 'Risk-Aware Speed / Cost Field')
    add_box(ax, 0.700, 0.655, 0.082, 0.046, 'Path length', COLORS['blue_light'], COLORS['blue'], size=8.4)
    add_box(ax, 0.700, 0.598, 0.082, 0.046, 'ESDF clearance', COLORS['green_light'], COLORS['green'], size=8.4)
    add_arrow(ax, (0.790, 0.678), (0.835, 0.650), color=COLORS['orange'])
    add_arrow(ax, (0.790, 0.621), (0.835, 0.650), color=COLORS['orange'])
    add_box(ax, 0.842, 0.620, 0.105, 0.060, 'Risk-aware\ntravel cost', COLORS['orange_light'], COLORS['orange'], size=9.0, weight='bold')
    add_text(ax, 0.822, 0.584, 'Low speed near obstacles; high speed in open space', size=8.2, color='#52606D')

    add_arrow(ax, (0.350, 0.667), (0.375, 0.667), color=COLORS['green'])
    add_arrow(ax, (0.650, 0.667), (0.675, 0.667), color=COLORS['green'])
    add_arrow(ax, (0.50, 0.555), (0.50, 0.542), color=COLORS['green'], mutation_scale=20)

    # FDGA* planning
    panel_y, panel_h = 0.270, 0.245
    widths = [0.205, 0.205, 0.215, 0.205]
    xs = [0.075, 0.300, 0.525, 0.760]
    titles = [
        'FMM Arrival-Cost Field',
        'Guidance Path and Corridor',
        'Direction-Aware A* Search',
        'Raw FDGA* Path',
    ]
    for x, w, title in zip(xs, widths, titles):
        add_panel(ax, x, panel_y, w, panel_h, title)

    add_image(
        ax,
        ANALYSIS_DIR / 'FMM_arrival_cost_orthographic_views.png',
        0.087, 0.315, 0.181, 0.157,
        crop=(0.02, 0.04, 0.98, 0.92),
    )
    add_text(ax, 0.1775, 0.292, 'Solve the Eikonal equation from the goal', size=8.1, color='#52606D')

    add_image(
        ax,
        ANALYSIS_DIR / 'FDGA_guidance_corridor_comparison.png',
        0.312, 0.315, 0.181, 0.157,
        crop=(0.10, 0.10, 0.90, 0.90),
    )
    add_text(ax, 0.4025, 0.292, '$T$-descending path + spherical dilation', size=8.1, color='#52606D')

    add_image(
        ax,
        ANALYSIS_DIR / 'FDGA_direction_aware_state_comparison.png',
        0.538, 0.329, 0.189, 0.140,
        crop=(0.28, 0.02, 0.995, 0.98),
    )
    add_box(ax, 0.548, 0.290, 0.074, 0.027, '$s=(c,q)$', COLORS['blue_light'], COLORS['blue'], size=8.5, weight='bold')
    add_box(ax, 0.630, 0.290, 0.084, 0.027, r'$\theta\leq\theta_{max}$', COLORS['orange_light'], COLORS['orange'], size=8.5, weight='bold')

    add_box(ax, 0.780, 0.389, 0.165, 0.052,
            'State backtracking', COLORS['purple_light'], COLORS['purple'], size=9.3, weight='bold')
    add_arrow(ax, (0.862, 0.383), (0.862, 0.350), color=COLORS['purple'])
    add_box(ax, 0.780, 0.305, 0.165, 0.045,
            'Direct search output', COLORS['green_light'], COLORS['green'], size=9.2, weight='bold')
    add_text(ax, 0.862, 0.288, 'No smoothing or path post-processing', size=8.0, color='#52606D')

    for left, right in zip([0.280, 0.505, 0.740], [0.300, 0.525, 0.760]):
        add_arrow(ax, (left, 0.390), (right, 0.390), color=COLORS['orange'], mutation_scale=18)

    add_arrow(ax, (0.50, 0.245), (0.50, 0.232), color=COLORS['orange'], mutation_scale=20)

    # Experimental evaluation
    add_panel(ax, 0.075, 0.035, 0.315, 0.165, 'Experimental Task Design and Algorithm Comparison')
    add_box(ax, 0.095, 0.137, 0.080, 0.035, '7 waypoints', COLORS['blue_light'], COLORS['blue'], size=8.5, weight='bold')
    add_arrow(ax, (0.181, 0.154), (0.215, 0.154), color=COLORS['purple'])
    add_box(ax, 0.220, 0.137, 0.105, 0.035, '42 directed OD pairs', COLORS['orange_light'], COLORS['orange'], size=8.3, weight='bold')
    planners = [
        ('FDGA*', '#0072B2'),
        ('FMM', '#E69F00'),
        ('A*', '#D55E00'),
        ('RRT*', '#CC79A7'),
    ]
    for i, (name, color) in enumerate(planners):
        x = 0.103 + i * 0.067
        ax.plot([x, x + 0.022], [0.092, 0.092], color=color, linewidth=3.0,
                transform=ax.transAxes, zorder=12)
        add_text(ax, x + 0.027, 0.092, name, size=7.9, weight='bold', ha='left')
    add_text(ax, 0.232, 0.058, 'Common environment, start–goal conditions, and safety threshold', size=7.7, color='#52606D')

    add_panel(ax, 0.430, 0.035, 0.540, 0.165, 'Path-Quality and Planning-Efficiency Evaluation')
    metric_labels = [
        'Path length', 'Planning time', 'Mean curvature',
        'Integrated squared curvature', 'High-curvature ratio',
        'Maximum turning angle', 'Minimum ESDF distance',
    ]
    for i, label in enumerate(metric_labels):
        col = i % 4
        row = i // 4
        x = 0.450 + col * 0.122
        y = 0.137 - row * 0.042
        add_box(ax, x, y - 0.014, 0.110, 0.028, label,
                COLORS['subpanel'], '#B8C0C6', size=7.5)
    add_box(ax, 0.548, 0.054, 0.300, 0.027,
            'Safety · Path efficiency · Geometric continuity · Computational efficiency',
            COLORS['green_light'], COLORS['green'], size=8.0, weight='bold')

    add_arrow(ax, (0.390, 0.116), (0.430, 0.116), color=COLORS['purple'])

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', pad_inches=0.04, facecolor='white')
    plt.close(fig)
    print(f'Saved: {OUTPUT_PATH}')


if __name__ == '__main__':
    draw_workflow()
