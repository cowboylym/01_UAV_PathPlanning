# ==============================================================
# 路径规划算法分析入口
# compare4 与 B0-B3 新流程使用 Friedman/Kendall W，事后采用配对
# Wilcoxon signed-rank，并在每个指标内独立执行 Holm 校正。
# 历史 nowind/wind 分析保留原有统计口径。
#   [统计口径] 描述统计补报 中位数; 新增"稳健统计"小节 (中位数 + 几何均值),
#          直接回应"均值由离群值驱动"质疑 (曲率统计口径已由总曲率改为平均曲率)
#   [图表] 按"效率优势→质量不输→平滑更优→综合第一→机理"叙事链重组:
#          01 耗时箱线图(对数轴) | 02 耗时-距离伸缩曲线 | 03 长度+ESDF双联图(标注ns)
#          04 平均曲率+转角双联图 | 05 综合评分+贡献分解 | 06 优胜次数
#          07 耗时-平均曲率Pareto散点(log-log) | 08 权重敏感性 | A1 相关性热力图(附录)
#   [新增] 综合评分权重 ±20% 扰动敏感性分析 (回应"权重人为设定"质疑)
#   [新增] 配对显著性星标直接标注到箱线图上 (ns 与 *** 同等醒目)
#   [新增] --exp 参数切换风场实验: nowind(默认)=data/exp_nowind, wind=data/exp_wind
# 依赖库：pandas numpy scipy matplotlib seaborn openpyxl
# 用法:
#   python Analysis.py                # 分析无风场实验 (data/exp_nowind/)
#   python Analysis.py --exp wind     # 分析风场节能实验 (data/exp_wind/, 含 W 系列三算法风场对比)
#   python Analysis.py --exp compare  # 带风 vs 不带风 跨实验对比 (读两目录, 输出 analysis_compare/)
#   python Analysis.py --exp ablation --batch-dir <批次目录> # B0-B3 机制消融
#   python Analysis.py --exp compare4 --batch-dir <批次目录> # 四算法对比 (FGDA* + FMM + A* + RRT*)
#   python Analysis.py --exp nowind --csv ../data/path_comparison.csv
# ==============================================================

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
import argparse
import tempfile

# -------------------------- 全局配置 --------------------------
print(f"当前工作目录: {os.getcwd()}")

# --exp 实验目录映射 (与 main.cpp 分目录输出、visDSM.py 的 EXP_DIRS 对应)
EXP_DIRS = {
    'nowind': '../data/exp_nowind',
    'wind':   '../data/exp_wind',
}

# CA-ESDF-FMM 消融与曲率权重敏感性实验目录。
# w_kappa 下允许直接放 path_comparison.csv，或按权重建立任意名称子目录；
# 权重从目录名（如 0.3、w_kappa_0.3、wk-0.3）中提取。
ABLATION_DIRS = {
    'B0': '../data/ablation/latest/B0',
    'B1': '../data/ablation/latest/B1',
    'B2': '../data/ablation/latest/B2',
    'B3': '../data/ablation/latest/B3',
}
W_KAPPA_DIR = '../data/w_kappa'
ABLATION_VARIANTS = {'B0': 'B0', 'B1': 'B1', 'B2': 'B2', 'B3': 'B3'}
ABLATION_METRICS = [
    '路径长度(m)', '平均曲率(度/m)', '曲率平方积分(1/m)', '高曲率段占比(%)',
    '后处理修正量(m)', '最大转角(度)', '最大连续急转弯段数',
    '最小ESDF距离(m)', '安全违反率(%)', '穿墙次数', '规划耗时(ms)'
]
LOWER_IS_BETTER = set(ABLATION_METRICS) - {'最小ESDF距离(m)'}

# 中文字体 (Windows: SimHei; macOS: Arial Unicode MS; Linux 服务器: Noto Sans CJK)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Noto Sans CJK SC', 'Noto Sans CJK JP']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
# Nature 期刊风格
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4
plt.rcParams['font.size'] = 10

SAVE_DPI = 300   # 论文用图分辨率


_original_savefig = plt.savefig


def _safe_savefig(path, *args, **kwargs):
    """先写英文临时文件再替换，兼容 Docker Desktop 的 Windows 挂载目录。"""
    output_dir = os.path.dirname(path)
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix='.plot_', suffix='.png', dir=output_dir)
    os.close(fd)
    try:
        _original_savefig(temp_path, *args, **kwargs)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


plt.savefig = _safe_savefig

# 常规三算法综合评分权重（对所有算法使用相同规则）
WEIGHTS = {
    '路径长度(m)': 0.30,      # 越小越好
    '规划耗时(ms)': 0.25,     # 越小越好
    '平均曲率(度/m)': 0.20,   # 越小越好
    '最小ESDF距离(m)': 0.15,  # 越大越好
    '高度方差': 0.10          # 越小越好
}

# 四算法对比采用效率、曲率、安全与路径代价维度：
# 效率、全局曲率能量、局部极端转角、安全净空与路径代价。
# 评分规则不含算法名称奖励，所有算法统一 Min-Max 归一化。
COMPARE4_WEIGHTS = {
    '路径长度(m)': 0.15,          # 越小越好
    '规划耗时(ms)': 0.20,         # 越小越好
    '曲率平方积分(1/m)': 0.25,    # 越小越好
    '最大转角(度)': 0.20,         # 越小越好
    '最小ESDF距离(m)': 0.20       # 越大越好
}

COMPARE4_ORDER = ['FGDA*', 'FMM', 'A*', 'RRT*']

NATURE_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']

# 贡献分解图专用配色: 与 WEIGHTS 顺序一致, 色相/明度拉开避免相近
# (路径长度-深蓝 vs 最小ESDF-亮青 vs 高度方差-天蓝; 曲率-橙红; 耗时-绿)
CONTRIB_COLORS = {
    '路径长度(m)': '#0072B2',     # 深蓝
    '规划耗时(ms)': '#009E73',    # 绿
    '平均曲率(度/m)': '#D55E00',  # 橙红
    '最小ESDF距离(m)': '#F0E442', # 黄 (与深蓝/绿对比强)
    '高度方差': '#CC79A7'         # 粉紫
}


def _sig_marker(p_val):
    """显著性星标: ns 与 *** 同等醒目."""
    if p_val < 0.001: return '***'
    elif p_val < 0.01: return '**'
    elif p_val < 0.05: return '*'
    return 'ns'


# -------------------------- RRT* 浮动范围 (rrt_fluctuation.csv) 解析 --------------------------
# rrt_fluctuation.csv 表头形如: 实验编号,路径长度(m)[min，max],...,高度方差(m²)[min，max]
# 用于把 RRT* 单条"中位数代表"记录扩充为区间 [min, max], 在统计检验/评分中体现随机采样波动.
_UNIT_TO_CORE = {'高度方差(m²)': '高度方差'}  # rrt 表头单位与主分析 core 列名的差异


def _metric_from_rrt_col(col):
    """从 rrt_fluctuation.csv 列名解析出 core_metrics 指标名; 非区间列返回 None."""
    col = str(col)
    for suf in ('[min，max]', '[min,max]'):
        if col.endswith(suf):
            base = col[:-len(suf)]
            return _UNIT_TO_CORE.get(base, base)
    return None


def _parse_range(val):
    """解析 '[min，max]' 或 '[min,max]' 字符串 -> (min, max); 失败返回 (None, None)."""
    s = str(val).strip()
    if not (s.startswith('[') and s.endswith(']')):
        return None, None
    inner = s[1:-1]
    for sep in ('，', ','):
        if sep in inner:
            parts = [x.strip() for x in inner.split(sep)]
            if len(parts) == 2:
                try:
                    return float(parts[0]), float(parts[1])
                except ValueError:
                    return None, None
    return None, None


def load_rrt_ranges(rrt_csv_path):
    """读取 rrt_fluctuation.csv, 返回 {实验编号: {指标: (min, max)}}."""
    rrt_df = pd.read_csv(rrt_csv_path)
    ranges = {}
    for _, row in rrt_df.iterrows():
        eid = int(row['实验编号'])
        d = {}
        for col in rrt_df.columns:
            if col == '实验编号':
                continue
            key = _metric_from_rrt_col(col)
            if key is None:
                continue
            mn, mx = _parse_range(row[col])
            d[key] = (mn, mx)
        ranges[eid] = d
    return ranges


# -------------------------- 跨实验对比 (带风 vs 不带风) --------------------------
def run_cross_analysis(nowind_dir, wind_dir, OUTPUT_DIR):
    """对比同一批 OD 对在 无风场/带风场 两种规划下的路径差异.

    输入: 两个实验目录各自的 path_comparison.csv + path_wind_metrics.csv
    核心指标:
      加长率(%)  = (L_有风 - L_无风)/L_无风 × 100    (绕路代价)
      Δ顶风(m/s) = 顶风_有风 - 顶风_无风              (借风收益, 负=更顺风)
      节能率(%)  = (E_无风 - E_有风)/E_无风 × 100    (净收益)
        能耗代理 (立方律): E = L·(1 + 顶风/v_g)³, v_g = 10 m/s
        阻力功率 P ∝ v_air³ → 单位距离能耗 ∝ (1+顶风/v_g)³
    """
    print("=" * 60)
    print("带风场 vs 不带风场 跨实验对比")
    print("=" * 60)

    files = {'无风场路径指标': os.path.join(nowind_dir, 'path_comparison.csv'),
             '风场路径指标':   os.path.join(wind_dir, 'path_comparison.csv'),
             '无风场风场统计': os.path.join(nowind_dir, 'path_wind_metrics.csv'),
             '风场风场统计':   os.path.join(wind_dir, 'path_wind_metrics.csv')}
    missing = [f'{k}: {v}' for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print("[错误] 缺少输入文件, 无法进行跨实验对比:")
        for s in missing:
            print(f"       - {s}")
        print("       需先完成两组实验: ./uav_planner 与 ./uav_planner --wind")
        return

    nd = pd.read_csv(files['无风场路径指标'])
    wd = pd.read_csv(files['风场路径指标'])
    nwm = pd.read_csv(files['无风场风场统计'])
    wwm = pd.read_csv(files['风场风场统计'])
    for frame in (nd, wd, nwm, wwm):
        if '规划状态' in frame.columns:
            frame.drop(frame.index[frame['规划状态'].astype(str).str.strip() != '成功'],
                       inplace=True)
    keep = ['实验编号', '规划器', '平均顶风分量(m/s)', '平均风速(m/s)']
    nwm = nwm[keep]
    wwm = wwm[keep]

    # 按 实验编号+规划器 配对两组成功实验
    key = ['实验编号', '规划器']
    m = (nd.merge(wd, on=key, suffixes=('_无风', '_有风'))
           .merge(nwm.rename(columns={'平均顶风分量(m/s)': '顶风_无风(m/s)',
                                      '平均风速(m/s)': '风速_无风(m/s)'}), on=key)
           .merge(wwm.rename(columns={'平均顶风分量(m/s)': '顶风_有风(m/s)',
                                      '平均风速(m/s)': '风速_有风(m/s)'}), on=key))

    planners = list(dict.fromkeys(m['规划器']))
    PLANNER_COLORS = {p: NATURE_PALETTE[i % len(NATURE_PALETTE)]
                      for i, p in enumerate(planners)}
    unpaired = (len(nd) - len(m)) + (len(wd) - len(m))
    print(f"配对成功 {len(m)} 条 (无风场 {len(nd)} 条 / 风场 {len(wd)} 条; "
          f"未配对 {unpaired} 条已丢弃 — 多为超时或失败组)")

    # ---- 派生指标 ----
    V_G = 10.0
    L0, L1 = m['路径长度(m)_无风'], m['路径长度(m)_有风']
    h0, h1 = m['顶风_无风(m/s)'], m['顶风_有风(m/s)']
    m['加长率(%)'] = (L1 - L0) / L0 * 100
    m['Δ顶风(m/s)'] = h1 - h0
    E0 = L0 * (1.0 + h0 / V_G) ** 3
    E1 = L1 * (1.0 + h1 / V_G) ** 3
    m['节能率(%)'] = (E0 - E1) / E0 * 100

    # ---- 安全性核查 (一票否决项): 风场版不得劣化安全约束 ----
    print("\n安全性核查 (风场版实验, 一票否决项):")
    safety = m.groupby('规划器').agg(
        配对组数=('路径长度(m)_有风', 'size'),
        最小ESDF最小值m=('最小ESDF距离(m)_有风', 'min'),
        穿墙合计=('穿墙次数_有风', 'sum'))
    print(safety)

    # ---- 分算法汇总 + 节能率单侧 Wilcoxon (H1: 节能率 > 0) ----
    print("\n分算法节能效果汇总:")
    rows = []
    for p in planners:
        sub = m[m['规划器'] == p]
        try:
            _, p_val = stats.wilcoxon(sub['节能率(%)'], alternative='greater')
        except (ValueError, TypeError):
            p_val = float('nan')
        rows.append({
            '规划器': p,
            '配对组数': len(sub),
            '节能率均值(%)': round(float(sub['节能率(%)'].mean()), 2),
            '节能率中位数(%)': round(float(sub['节能率(%)'].median()), 2),
            '加长率均值(%)': round(float(sub['加长率(%)'].mean()), 2),
            'Δ顶风均值(m/s)': round(float(sub['Δ顶风(m/s)'].mean()), 3),
            '节能显著p值(单侧)': round(float(p_val), 4),
        })
    cross_summary = pd.DataFrame(rows)
    print(cross_summary.to_string(index=False))

    # ---- 图 C1: 节能率 + 加长率 双联箱线图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, metric, title in zip(
            axes, ['节能率(%)', '加长率(%)'],
            ['节能率 (立方律能耗代理, 正=风场版更省)', '路径加长率 (绕路代价)']):
        sns.boxplot(x='规划器', y=metric, data=m, ax=ax,
                    palette=[PLANNER_COLORS[p] for p in planners],
                    width=0.5, order=planners)
        sns.stripplot(x='规划器', y=metric, data=m, ax=ax, color='black',
                      alpha=0.4, size=3, jitter=True, order=planners)
        ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
        means = [f'{p}\n均值 {m[m["规划器"] == p][metric].mean():+.1f}%'
                 for p in planners]
        ax.set_xticks(range(len(planners)))
        ax.set_xticklabels(means, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'C1_节能与加长双联箱线图.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    print("\n已保存：C1_节能与加长双联箱线图.png")
    plt.close()

    # ---- 图 C2: 节能率-加长率 权衡散点 (象限判读) ----
    fig, ax = plt.subplots(figsize=(9.5, 7))
    for p in planners:
        sub = m[m['规划器'] == p]
        ax.scatter(sub['加长率(%)'], sub['节能率(%)'], s=70, alpha=0.7, label=p,
                   color=PLANNER_COLORS[p], edgecolors='black', linewidth=0.5)
        ax.scatter(sub['加长率(%)'].mean(), sub['节能率(%)'].mean(), marker='*',
                   s=350, color=PLANNER_COLORS[p], edgecolors='black',
                   linewidth=1.5, zorder=5)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.text(0.03, 0.97, '节能↑ 且更短\n(双赢区)', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='#555555')
    ax.text(0.97, 0.97, '节能↑ 绕路↑\n(有代价的收益)', transform=ax.transAxes,
            ha='right', va='top', fontsize=9, color='#555555')
    ax.text(0.97, 0.03, '更耗能且绕路\n(失败区)', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#555555')
    ax.set_title('节能率-加长率权衡 (★为各算法均值, 左上方向为最优)', fontsize=12)
    ax.set_xlabel('路径加长率 (%) ← 越小越优', fontsize=11)
    ax.set_ylabel('节能率 (%) ← 越大越优', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(title='规划器', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'C2_节能加长权衡散点.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：C2_节能加长权衡散点.png")
    plt.close()

    # ---- 图 C3: 顶风分量前后对比 (借风直接证据) ----
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(planners))
    w_bar = 0.36
    hw0 = [m[m['规划器'] == p]['顶风_无风(m/s)'].mean() for p in planners]
    hw1 = [m[m['规划器'] == p]['顶风_有风(m/s)'].mean() for p in planners]
    b1 = ax.bar(x - w_bar / 2, hw0, w_bar, label='不带风场规划',
                color=[PLANNER_COLORS[p] for p in planners], alpha=0.40,
                edgecolor='black', linewidth=0.8)
    b2 = ax.bar(x + w_bar / 2, hw1, w_bar, label='带风场规划',
                color=[PLANNER_COLORS[p] for p in planners],
                edgecolor='black', linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.03 if v >= 0 else -0.10), f'{v:+.2f}',
                    ha='center', fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax.set_xticks(x)
    ax.set_xticklabels(planners, fontsize=11)
    ax.set_ylabel('平均顶风分量 (m/s)', fontsize=11)
    ax.set_title('风场感知规划前后平均顶风对比 (越低越顺风)\n'
                 '浅色=不带风场规划, 深色=带风场规划 (节能的直接证据)', fontsize=12)
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'C3_顶风前后对比.png'),
                dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：C3_顶风前后对比.png")
    plt.close()

    # ---- Excel 导出 ----
    detail_cols = ['实验编号', '规划器', '路径长度(m)_无风', '路径长度(m)_有风',
                   '加长率(%)', '顶风_无风(m/s)', '顶风_有风(m/s)', 'Δ顶风(m/s)',
                   '节能率(%)']
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, '对比分析结果汇总.xlsx')) as writer:
        m[detail_cols].round(3).to_excel(writer, sheet_name='逐组明细', index=False)
        cross_summary.to_excel(writer, sheet_name='分算法汇总', index=False)
        safety.to_excel(writer, sheet_name='安全性核查')

    print("\n跨实验对比完成! 结果已保存到 " + OUTPUT_DIR)
    print("- C1_节能与加长双联箱线图.png  节能收益与绕路代价")
    print("- C2_节能加长权衡散点.png      逐组 trade-off (象限判读)")
    print("- C3_顶风前后对比.png           借风直接证据")
    print("- 对比分析结果汇总.xlsx         逐组明细 + 分算法汇总 + 安全性核查")


# -------------------------- G0-G3 消融 + w_kappa 敏感性配对分析 --------------------------
def _holm_adjust(p_values):
    """Holm step-down 校正；NaN 保持为 NaN。"""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def _rank_biserial(x, y):
    """配对 Wilcoxon 的秩二列效应量，正值表示 x > y。"""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if not len(diff):
        return 0.0
    ranks = stats.rankdata(np.abs(diff), method='average')
    total = ranks.sum()
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / total)


def _weight_from_path(path):
    """从 w_kappa 文件/目录名解析权重。"""
    normalized = path.replace('κ', 'kappa')
    tagged = re.findall(r'(?:w[_-]?kappa|wk)[_=-]?([0-9]+(?:\.[0-9]+)?)',
                        normalized, flags=re.IGNORECASE)
    if tagged:
        return float(tagged[-1])
    for part in reversed(os.path.normpath(path).split(os.sep)):
        if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', part):
            return float(part)
    return None


def _read_ablation_csv(csv_path, group, expected_planner=None, weight=None):
    """读取单组结果并过滤规划失败/非数值记录。"""
    raw = pd.read_csv(csv_path)
    if expected_planner and '规划器' in raw.columns:
        chosen = raw[raw['规划器'] == expected_planner]
        # 兼容旧版 G0 CSV（同时包含三算法）以及用户自定义的统一规划器名。
        if chosen.empty and raw['规划器'].nunique() == 1:
            chosen = raw
        raw = chosen
    input_count = len(raw)
    if '规划状态' in raw.columns:
        raw = raw[raw['规划状态'].astype(str).str.strip() == '成功'].copy()
    required = ['实验编号', '路径点数', '路径长度(m)']
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'{csv_path} 缺少必要列: {missing}')
    available = [m for m in ABLATION_METRICS if m in raw.columns]
    numeric = list(dict.fromkeys(required[1:] + available))
    for col in numeric:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    valid = (raw['路径点数'] >= 2) & (raw['路径长度(m)'] > 0)
    valid &= np.isfinite(raw[numeric]).all(axis=1)
    filtered = raw.loc[valid].copy()
    filtered['实验组'] = group
    filtered['w_kappa'] = weight
    return filtered, input_count - len(filtered), available


def _collect_ablation_data(ablation_dirs, w_kappa_dir):
    """汇集 G0-G3 及 w_kappa 各目录，并保留每组唯一的实验编号。"""
    frames, filter_rows, metric_sets = [], [], []
    for group, directory in ablation_dirs.items():
        csv_path = os.path.join(directory, 'path_comparison.csv')
        if not os.path.isfile(csv_path):
            print(f'[警告] {group} 缺少 {csv_path}，联合分析将跳过该组')
            continue
        part, dropped, metrics = _read_ablation_csv(
            csv_path, group, ABLATION_VARIANTS.get(group))
        frames.append(part)
        metric_sets.append(set(metrics))
        filter_rows.append({'实验组': group, '输入记录': len(part) + dropped,
                            '失败过滤': dropped, '有效记录': len(part)})

    if os.path.isdir(w_kappa_dir):
        for root, _, files in os.walk(w_kappa_dir):
            for filename in sorted(files):
                if not (filename.lower().startswith('path_comparison') and
                        filename.lower().endswith('.csv')):
                    continue
                csv_path = os.path.join(root, filename)
                weight = _weight_from_path(csv_path)
                probe = pd.read_csv(csv_path, nrows=1)
                if weight is None and 'w_kappa' in probe.columns and len(probe):
                    weight = float(probe['w_kappa'].iloc[0])
                if weight is None:
                    print(f'[警告] 无法从路径解析 w_kappa，跳过: {csv_path}')
                    continue
                group = f'w_kappa={weight:g}'
                part, dropped, metrics = _read_ablation_csv(csv_path, group, weight=weight)
                frames.append(part)
                metric_sets.append(set(metrics))
                filter_rows.append({'实验组': group, '输入记录': len(part) + dropped,
                                    '失败过滤': dropped, '有效记录': len(part)})

    if len(frames) < 2:
        raise ValueError('至少需要两个有效的 G0-G3/w_kappa 实验目录才能进行配对分析')
    data = pd.concat(frames, ignore_index=True)
    if data['实验组'].nunique() != len(frames):
        duplicate_groups = [g for g, n in data.groupby('实验组').size().items()
                            if n > data.loc[data['实验组'] == g, '实验编号'].nunique()]
        raise ValueError(f'w_kappa 权重标签重复，请确保每个权重仅有一个 CSV: {duplicate_groups}')
    duplicated = data.duplicated(['实验组', '实验编号'], keep=False)
    if duplicated.any():
        dup = data.loc[duplicated, ['实验组', '实验编号']].drop_duplicates()
        raise ValueError(f'同一实验组存在重复实验编号，无法唯一配对:\n{dup.to_string(index=False)}')
    metrics = [m for m in ABLATION_METRICS
               if metric_sets and all(m in columns for columns in metric_sets)]
    if not metrics:
        raise ValueError('各实验组没有共同的可分析指标')
    return data, metrics, pd.DataFrame(filter_rows)


def _paired_subset(data, groups):
    """返回指定实验组按实验编号取交集后的完整配对数据。"""
    available = [g for g in groups if g in set(data['实验组'])]
    if len(available) != len(groups):
        return data.iloc[0:0].copy(), []
    id_sets = [set(data.loc[data['实验组'] == g, '实验编号']) for g in groups]
    complete_ids = sorted(set.intersection(*id_sets))
    return data[(data['实验组'].isin(groups)) &
                (data['实验编号'].isin(complete_ids))].copy(), complete_ids


def _normality_table(data, metrics, analysis):
    """逐实验、逐指标完整报告 Shapiro-Wilk 正态性检验。"""
    rows = []
    for group in list(dict.fromkeys(data['实验组'])):
        sub = data[data['实验组'] == group]
        for metric in metrics:
            values = sub[metric].dropna().values
            statistic, p_val = np.nan, np.nan
            if 3 <= len(values) <= 5000:
                statistic, p_val = stats.shapiro(values)
            rows.append({'分析': analysis, '实验组': group, '指标': metric,
                         '样本量': len(values), 'Shapiro_W': float(statistic),
                         'p值': float(p_val),
                         '正态性满足': ('是' if p_val >= 0.05 else '否')
                         if np.isfinite(p_val) else '样本量不足'})
    return pd.DataFrame(rows)


def _friedman_table(data, metrics, groups, analysis):
    """仅在同一实验设计内执行 Friedman 检验。"""
    rows = []
    for metric in metrics:
        wide = data.pivot(index='实验编号', columns='实验组', values=metric)[groups]
        try:
            chi2, p_val = stats.friedmanchisquare(*(wide[g].values for g in groups))
        except ValueError:
            chi2, p_val = 0.0, 1.0
        kendall_w = float(chi2 / (len(wide) * (len(groups) - 1)))
        rows.append({'分析': analysis, '指标': metric, '配对组数': len(wide),
                     '组数': len(groups), 'Friedman卡方': float(chi2),
                     'p值': float(p_val), 'Kendall_W': kendall_w})
    return pd.DataFrame(rows)


def _wilcoxon_table(data, metrics, groups, analysis):
    """指定实验设计内两两 Wilcoxon，并按指标分别进行 Holm 校正。"""
    rows = []
    for metric in metrics:
        wide = data.pivot(index='实验编号', columns='实验组', values=metric)[groups]
        metric_rows = []
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                x, y = wide[g1].values, wide[g2].values
                try:
                    w_val, p_val = stats.wilcoxon(x, y, alternative='two-sided')
                except ValueError:
                    w_val, p_val = 0.0, 1.0
                metric_rows.append({'分析': analysis, '指标': metric,
                                    '对比': f'{g1} vs {g2}', '配对组数': len(x),
                                    'W值': float(w_val), '原始p值': float(p_val),
                                    '秩二列效应量': _rank_biserial(x, y)})
        for row, adjusted in zip(metric_rows,
                                 _holm_adjust([r['原始p值'] for r in metric_rows])):
            row['Holm校正p'] = adjusted
            row['Holm显著'] = 'sig' if adjusted < 0.05 else 'ns'
        rows.extend(metric_rows)
    return pd.DataFrame(rows)


def run_paired_planner_analysis(data, output_dir, groups, analysis,
                                group_column='规划器', split_column=None,
                                single_metric_plots=False,
                                excluded_plot_metrics=None):
    """对完整配对数据执行 Friedman/Wilcoxon-Holm，并输出统计表与指标图。"""
    excluded_plot_metrics = set(excluded_plot_metrics or [])
    os.makedirs(output_dir, exist_ok=True)
    metric_candidates = [
        '路径长度(m)', '路径伸缩率', '曲率平方积分(1/m)',
        '平均曲率(度/m)', '高曲率段占比(%)', '最大转角(度)',
        '最小ESDF距离(m)', '规划耗时(ms)'
    ]
    work = data.copy()
    if '路径伸缩率' not in work.columns:
        coord_cols = ['起点X', '起点Y', '起点Z', '终点X', '终点Y', '终点Z']
        if all(column in work.columns for column in coord_cols) and '路径长度(m)' in work.columns:
            direct = np.sqrt((work['终点X'] - work['起点X']) ** 2 +
                             (work['终点Y'] - work['起点Y']) ** 2 +
                             (work['终点Z'] - work['起点Z']) ** 2)
            work['路径伸缩率'] = work['路径长度(m)'] / direct.replace(0, np.nan)
        else:
            print('[警告] 缺少坐标或路径长度列，无法计算路径伸缩率')
    metrics = [metric for metric in metric_candidates if metric in work.columns]
    missing = [metric for metric in metric_candidates if metric not in work.columns]
    if missing:
        print(f'[警告] {analysis} 缺少指标列，将跳过: {missing}')
    if not metrics:
        raise ValueError(f'{analysis} 没有可分析指标')
    for metric in metrics:
        work[metric] = pd.to_numeric(work[metric], errors='coerce')

    summaries = []
    friedman_frames = []
    wilcoxon_frames = []
    split_values = [None] if split_column is None else list(dict.fromkeys(work[split_column]))
    for split in split_values:
        subset = work if split is None else work[work[split_column] == split]
        tag = analysis if split is None else f'{analysis}-{split}'
        stats_data = subset.rename(columns={group_column: '实验组'}) if group_column != '实验组' else subset.copy()
        duplicated = stats_data.duplicated(['实验编号', '实验组'], keep=False)
        if duplicated.any():
            raise ValueError(f'{tag} 存在重复实验编号-实验组记录')
        wide_ids = [set(stats_data.loc[stats_data['实验组'] == group, '实验编号'])
                    for group in groups]
        complete_ids = sorted(set.intersection(*wide_ids)) if wide_ids else []
        stats_data = stats_data[(stats_data['实验编号'].isin(complete_ids)) &
                                (stats_data['实验组'].isin(groups))].copy()
        if len(complete_ids) < 2:
            print(f'[警告] {tag} 完整配对仅 {len(complete_ids)} 组，跳过')
            continue
        usable = [metric for metric in metrics
                  if not stats_data.pivot(index='实验编号', columns='实验组', values=metric)[groups]
                  .isna().any().any()]
        skipped = sorted(set(metrics) - set(usable))
        if skipped:
            print(f'[警告] {tag} 指标含缺失值，将跳过: {skipped}')
        if not usable:
            continue
        summaries.append(stats_data.groupby('实验组')[usable].agg(['count', 'mean', 'median', 'std']))
        if len(groups) >= 3:
            friedman_frames.append(_friedman_table(stats_data, usable, groups, tag))
        wilcoxon_frames.append(_wilcoxon_table(stats_data, usable, groups, tag))

        plot_metrics = [metric for metric in usable
                        if metric not in excluded_plot_metrics]
        safe_tag = re.sub(r'[^0-9A-Za-z_\u4e00-\u9fff-]+', '_', str(tag))
        if single_metric_plots:
            ylabel_map = {
                '路径长度(m)': 'Path Length (m)',
                '曲率平方积分(1/m)': 'Integrated Squared Curvature (1/m)',
                '平均曲率(度/m)': 'Mean Curvature (deg/m)',
                '高曲率段占比(%)': 'High-Curvature Segment Ratio (%)',
                '最大转角(度)': 'Maximum Turning Angle (deg)',
                '最小ESDF距离(m)': 'Minimum ESDF Distance (m)',
                '规划耗时(ms)': 'Planning Time (ms)',
            }
            legacy_path = os.path.join(output_dir, f'{safe_tag}_指标分布.png')
            if os.path.exists(legacy_path):
                os.remove(legacy_path)
            for metric in plot_metrics:
                fig, axis = plt.subplots(figsize=(7, 5.5))
                sns.boxplot(data=stats_data, x='实验组', y=metric,
                            hue='实验组', order=groups, hue_order=groups,
                            palette=NATURE_PALETTE[:len(groups)], width=0.5,
                            linewidth=1.3, legend=False, ax=axis)
                sns.stripplot(data=stats_data, x='实验组', y=metric, order=groups,
                              color='black', alpha=0.35, size=3.5, ax=axis)
                axis.set_title('')
                axis.set_xlabel('')
                axis.set_ylabel(ylabel_map.get(metric, metric), fontsize=16, labelpad=10)
                axis.tick_params(axis='x', labelsize=15, pad=7)
                axis.tick_params(axis='y', labelsize=14)
                axis.margins(x=0.025)
                safe_metric = re.sub(r'[^0-9A-Za-z_\u4e00-\u9fff-]+', '_', metric).strip('_')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{safe_tag}_{safe_metric}.png'),
                            dpi=SAVE_DPI, bbox_inches='tight')
                plt.close(fig)
        elif plot_metrics:
            columns = 2
            rows = int(np.ceil(len(plot_metrics) / columns))
            fig, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False)
            for axis, metric in zip(axes.flat, plot_metrics):
                sns.boxplot(data=stats_data, x='实验组', y=metric,
                            hue='实验组', order=groups, hue_order=groups,
                            palette=NATURE_PALETTE[:len(groups)],
                            legend=False, ax=axis)
                sns.stripplot(data=stats_data, x='实验组', y=metric, order=groups,
                              color='black', alpha=0.35, size=3, ax=axis)
                axis.set_title(metric)
                axis.set_xlabel('')
            for axis in axes.flat[len(plot_metrics):]:
                axis.set_visible(False)
            fig.suptitle(tag)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{safe_tag}_指标分布.png'),
                        dpi=SAVE_DPI, bbox_inches='tight')
            plt.close(fig)

    friedman = pd.concat(friedman_frames, ignore_index=True) if friedman_frames else pd.DataFrame()
    wilcoxon = pd.concat(wilcoxon_frames, ignore_index=True) if wilcoxon_frames else pd.DataFrame()
    friedman.to_csv(os.path.join(output_dir, 'Friedman_KendallW.csv'), index=False, encoding='utf-8-sig')
    wilcoxon.to_csv(os.path.join(output_dir, 'Wilcoxon_Holm_效应量.csv'), index=False, encoding='utf-8-sig')
    if summaries:
        pd.concat(summaries).to_csv(os.path.join(output_dir, '描述统计.csv'), encoding='utf-8-sig')
    work.to_csv(os.path.join(output_dir, '分析输入_含派生指标.csv'), index=False, encoding='utf-8-sig')
    return friedman, wilcoxon


def _load_g0_reference_planners(csv_path):
    """从 G0 原始 CSV 读取 A*/RRT* 基准耗时，不使用消融目录中的替代数据。"""
    raw = pd.read_csv(csv_path)
    required = ['实验编号', '规划器', '路径点数', '路径长度(m)', '规划耗时(ms)']
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'{csv_path} 缺少 G0 基准必要列: {missing}')
    refs = raw[raw['规划器'].isin(['A*', 'RRT*'])].copy()
    if '规划状态' in refs.columns:
        refs = refs[refs['规划状态'].astype(str).str.strip() == '成功'].copy()
    for col in ['路径点数', '路径长度(m)', '规划耗时(ms)']:
        refs[col] = pd.to_numeric(refs[col], errors='coerce')
    valid = ((refs['路径点数'] >= 2) & (refs['路径长度(m)'] > 0) &
             np.isfinite(refs['规划耗时(ms)']))
    refs = refs.loc[valid, ['实验编号', '规划器', '规划耗时(ms)']]
    duplicated = refs.duplicated(['规划器', '实验编号'], keep=False)
    if duplicated.any():
        raise ValueError('G0 原始 CSV 的 A*/RRT* 存在重复实验编号，无法唯一配对')
    return refs


def _paired_wilcoxon(data, left, right, metric, alternative):
    """按实验编号配对并执行有方向的 Wilcoxon，返回样本量、均值和 p 值。"""
    paired, ids = _paired_subset(data, [left, right])
    if len(ids) < 2 or metric not in paired.columns:
        return None
    wide = paired.pivot(index='实验编号', columns='实验组', values=metric)
    wide = wide[[left, right]].dropna()
    if len(wide) < 2:
        return None
    try:
        _, p_val = stats.wilcoxon(wide[left], wide[right], alternative=alternative)
    except ValueError:
        p_val = 1.0
    return len(wide), float(wide[left].mean()), float(wide[right].mean()), float(p_val)


def _seven_checks(data, g0_references):
    """按七项验收要求执行预设的有方向配对检验并给出 PASS/FAIL。"""
    rows = []

    def add(name, passed, detail):
        rows.append({'序号': len(rows) + 1, '判据': name,
                     '结果': 'PASS' if passed else 'FAIL', '依据': detail})

    smooth_metrics = ['平均曲率(度/m)', '曲率平方积分(1/m)']

    def smoothness_check(left, right):
        results = []
        for metric in smooth_metrics:
            result = _paired_wilcoxon(data, left, right, metric, 'greater')
            if result is not None:
                n, left_mean, right_mean, raw_p = result
                results.append([metric, n, left_mean, right_mean, raw_p, np.nan])
        if results:
            adjusted = _holm_adjust([r[4] for r in results])
            for result, adjusted_p in zip(results, adjusted):
                result[5] = float(adjusted_p)
        return results

    g01_smooth = smoothness_check('G0', 'G1')
    passed = any(b < a and adjusted_p < 0.05
                 for _, _, a, b, _, adjusted_p in g01_smooth)
    detail = '；'.join(
        f'{m}: n={n}, 均值{a:.4g}/{b:.4g}, 原始单侧p={raw_p:.3g}, Holm校正p={adjusted_p:.3g}'
        for m, n, a, b, raw_p, adjusted_p in g01_smooth
    ) or '缺少 G0-G1 平滑性配对指标'
    add('G1平均曲率或Eκ显著降低', passed, detail)

    turn = _paired_wilcoxon(data, 'G0', 'G1', '最大转角(度)', 'less')
    if turn is None:
        add('G1最大转角不恶化', False, '缺少 G0-G1 最大转角配对数据')
    else:
        n, g0_mean, g1_mean, p_worse = turn
        add('G1最大转角不恶化', g1_mean <= g0_mean,
            f'n={n}，G0/G1均值={g0_mean:.3f}/{g1_mean:.3f}°，'
            f'G1显著恶化原始单侧p={p_worse:.3g}，Holm校正p={p_worse:.3g}')

    length = _paired_wilcoxon(data, 'G0', 'G1', '路径长度(m)', 'two-sided')
    if length is None:
        add('G1长度均值增幅≤3%', False, '缺少 G0-G1 路径长度配对数据')
    else:
        n, g0_mean, g1_mean, p_val = length
        change = (g1_mean - g0_mean) / abs(g0_mean) * 100 if g0_mean else np.nan
        add('G1长度均值增幅≤3%', np.isfinite(change) and change <= 3.0,
            f'n={n}，G0/G1均值={g0_mean:.3f}/{g1_mean:.3f}m，增幅={change:+.2f}%，'
            f'原始双侧p={p_val:.3g}，Holm校正p={p_val:.3g}')

    esdf = _paired_wilcoxon(data, 'G0', 'G1', '最小ESDF距离(m)', 'greater')
    if esdf is None:
        add('G1最小ESDF不显著降低', False, '缺少 G0-G1 最小ESDF配对数据')
    else:
        n, g0_mean, g1_mean, p_drop = esdf
        add('G1最小ESDF不显著降低', p_drop >= 0.05,
            f'n={n}，G0/G1均值={g0_mean:.3f}/{g1_mean:.3f}m，'
            f'降低原始单侧p={p_drop:.3g}，Holm校正p={p_drop:.3g}')

    g1 = data[data['实验组'] == 'G1']
    wall_total = float(g1['穿墙次数'].sum()) if '穿墙次数' in g1 else np.nan
    add('G1共42组成功且零碰撞', len(g1) == 42 and wall_total == 0,
        f'成功组数={len(g1)}/42，穿墙次数合计={wall_total:g}' if np.isfinite(wall_total)
        else f'成功组数={len(g1)}/42，缺少穿墙次数')

    time_results = []
    g1_time = data.loc[data['实验组'] == 'G1', ['实验编号', '规划耗时(ms)']]
    for planner in ['A*', 'RRT*']:
        ref = g0_references[g0_references['规划器'] == planner]
        wide = g1_time.merge(ref, on='实验编号', suffixes=('_G1', '_ref'))
        if len(wide) < 2:
            time_results.append([planner, None, np.nan, np.nan, np.nan, np.nan])
            continue
        try:
            _, raw_p = stats.wilcoxon(wide['规划耗时(ms)_G1'],
                                      wide['规划耗时(ms)_ref'], alternative='less')
        except ValueError:
            raw_p = 1.0
        mean_g1 = float(wide['规划耗时(ms)_G1'].mean())
        mean_ref = float(wide['规划耗时(ms)_ref'].mean())
        time_results.append([planner, len(wide), mean_g1, mean_ref, float(raw_p), np.nan])
    testable = [result for result in time_results if result[1] is not None]
    if testable:
        adjusted = _holm_adjust([r[4] for r in testable])
        for result, adjusted_p in zip(testable, adjusted):
            result[5] = float(adjusted_p)
    time_pass = len(testable) == 2 and all(
        mean_g1 < mean_ref and adjusted_p < 0.05
        for _, _, mean_g1, mean_ref, _, adjusted_p in testable)
    time_details = [
        f'{planner}: 配对不足' if n is None else
        f'{planner}: n={n}, 均值G1/基准={mean_g1:.3f}/{mean_ref:.3f}ms, '
        f'原始单侧p={raw_p:.3g}, Holm校正p={adjusted_p:.3g}'
        for planner, n, mean_g1, mean_ref, raw_p, adjusted_p in time_results
    ]
    add('G1耗时显著低于G0原CSV的A*/RRT*', time_pass, '；'.join(time_details))

    g23_smooth = smoothness_check('G3', 'G2')
    passed = any(b < a and adjusted_p < 0.05
                 for _, _, a, b, _, adjusted_p in g23_smooth)
    detail = '；'.join(
        f'{m}: n={n}, 均值G3/G2={a:.4g}/{b:.4g}, '
        f'原始单侧p={raw_p:.3g}, Holm校正p={adjusted_p:.3g}'
        for m, n, a, b, raw_p, adjusted_p in g23_smooth
    ) or '缺少 G2-G3 平滑性配对指标'
    add('G2平均曲率或Eκ显著优于G3', passed, detail)
    return pd.DataFrame(rows)


def run_ablation_analysis(output_dir, ablation_dirs=None, w_kappa_dir=W_KAPPA_DIR):
    """分别对 G0-G3 消融与 w_kappa 扫描执行完整配对分析。"""
    ablation_dirs = ABLATION_DIRS if ablation_dirs is None else ablation_dirs
    data, metrics, filter_df = _collect_ablation_data(ablation_dirs, w_kappa_dir)
    ablation_groups = ['G0', 'G1', 'G2', 'G3']
    weight_groups = sorted(
        [g for g in data['实验组'].unique() if str(g).startswith('w_kappa=')],
        key=lambda g: float(str(g).split('=')[1]))
    paired_frames, friedman_frames, pair_frames, normality_frames = [], [], [], []

    analyses = [('G0-G3消融', ablation_groups), ('w_kappa敏感性', weight_groups)]
    for analysis, groups in analyses:
        if len(groups) < 3:
            print(f'[警告] {analysis}仅有 {len(groups)} 个实验组，跳过 Friedman')
            continue
        paired, complete_ids = _paired_subset(data, groups)
        if len(complete_ids) < 2:
            print(f'[警告] {analysis}完整配对仅 {len(complete_ids)} 组，跳过统计检验')
            continue
        paired['分析'] = analysis
        paired_frames.append(paired)
        friedman_frames.append(_friedman_table(paired, metrics, groups, analysis))
        pair_frames.append(_wilcoxon_table(paired, metrics, groups, analysis))
        normality_frames.append(_normality_table(paired, metrics, analysis))
        print(f'{analysis}: {groups}，完整配对 {len(complete_ids)} 组')
        for group in groups:
            mask = filter_df['实验组'] == group
            filter_df.loc[mask, '未进入本分析配对'] = (
                filter_df.loc[mask, '有效记录'] - len(complete_ids))

    if not paired_frames:
        raise ValueError('G0-G3 消融与 w_kappa 敏感性均无足够完整配对数据')
    paired = pd.concat(paired_frames, ignore_index=True)
    friedman_df = pd.concat(friedman_frames, ignore_index=True)
    pair_df = pd.concat(pair_frames, ignore_index=True)
    normality_df = pd.concat(normality_frames, ignore_index=True)
    g0_csv = os.path.join(ablation_dirs['G0'], 'path_comparison.csv')
    g0_references = _load_g0_reference_planners(g0_csv)
    checks_df = _seven_checks(data, g0_references)

    print(filter_df.to_string(index=False))
    print('\n分实验 Friedman + Kendall W:')
    print(friedman_df.to_string(index=False))
    print('\n完整正态性检验表:')
    print(normality_df.to_string(index=False))
    print('\n七项验收判定:')
    print(checks_df.to_string(index=False))

    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(os.path.join(output_dir, '消融与权重分开配对分析.xlsx')) as writer:
        filter_df.to_excel(writer, sheet_name='失败与配对过滤', index=False)
        paired.to_excel(writer, sheet_name='分实验配对明细', index=False)
        friedman_df.to_excel(writer, sheet_name='分开Friedman_KendallW', index=False)
        normality_df.to_excel(writer, sheet_name='正态性完整表', index=False)
        pair_df.to_excel(writer, sheet_name='Wilcoxon_Holm_效应量', index=False)
        checks_df.to_excel(writer, sheet_name='七项PASS_FAIL', index=False)

    # ---------------------- 消融统计图 (与主分析同风格) ----------------------
    ablation_paired = paired[paired['分析'] == 'G0-G3消融'] \
        if {'G0', 'G1', 'G2', 'G3'}.issubset(set(data['实验组'])) else None

    def _sig_stars(adjusted_p):
        if not np.isfinite(adjusted_p):
            return 'ns'
        return '***' if adjusted_p < 0.001 else '**' if adjusted_p < 0.01 \
            else '*' if adjusted_p < 0.05 else 'ns'

    def _annotate_sig(ax, metric, y_max):
        """在箱线图上方标注相邻组 Holm 校正显著性 (G0-G1, G1-G2, G2-G3)。"""
        present_groups = [g for g in ['G0', 'G1', 'G2', 'G3']
                          if g in ablation_paired['实验组'].unique()]
        for left, right in zip(present_groups[:-1], present_groups[1:]):
            hit = pair_df[(pair_df['分析'] == 'G0-G3消融') &
                          (pair_df['指标'] == metric) &
                          (pair_df['对比'] == f'{left} vs {right}')]
            if not len(hit):
                continue
            stars = _sig_stars(float(hit.iloc[0]['Holm校正p']))
            i, j = present_groups.index(left), present_groups.index(right)
            ax.plot([i, j], [y_max, y_max], color='gray', lw=1)
            ax.text((i + j) / 2, y_max, stars, ha='center', va='bottom', fontsize=10)

    if ablation_paired is not None and not ablation_paired.empty:
        group_labels = {'G0': 'G0\n普通+平滑', 'G1': 'G1\n曲率+平滑',
                        'G2': 'G2\n曲率+投影', 'G3': 'G3\n普通+投影'}
        group_colors = {g: NATURE_PALETTE[i] for i, g in enumerate(['G0', 'G1', 'G2', 'G3'])}

        # A1. 平滑性四联箱线图 (H1: 曲率感知是否改善)
        smooth_layout = [('平均曲率(度/m)', '平均曲率 (°/m)'),
                         ('曲率平方积分(1/m)', '曲率平方积分 $E_\\kappa$ (1/m)'),
                         ('高曲率段占比(%)', '高曲率段占比 (%)'),
                         ('最大连续急转弯段数', '最大连续急转弯段数')]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
        for ax, (metric, ylabel) in zip(axes, smooth_layout):
            if metric not in ablation_paired.columns:
                ax.set_visible(False)
                continue
            sns.boxplot(x='实验组', y=metric, data=ablation_paired, ax=ax,
                        order=['G0', 'G1', 'G2', 'G3'], palette=group_colors)
            sns.stripplot(x='实验组', y=metric, data=ablation_paired, ax=ax,
                          order=['G0', 'G1', 'G2', 'G3'], color='black',
                          size=2.5, alpha=0.35)
            y_max = ablation_paired[metric].max()
            span = y_max - ablation_paired[metric].min()
            ax.set_ylim(top=y_max + 0.18 * span if span > 0 else y_max * 1.2)
            _annotate_sig(ax, metric, y_max + 0.06 * span if span > 0 else y_max * 1.1)
            ax.set_xlabel('')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xticklabels([group_labels[g.get_text()] for g in ax.get_xticklabels()])
        fig.suptitle('G0-G3 消融: 平滑性指标对比 (H1, 星标为 Holm 校正显著性)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'A1_平滑性四联箱线图.png'),
                    dpi=SAVE_DPI, bbox_inches='tight')
        plt.close()
        print('已保存：A1_平滑性四联箱线图.png')

        # A2. 长度/安全/后处理依赖三联箱线图 (H2 + H3)
        tradeoff_layout = [('路径长度(m)', '路径长度 (m)'),
                           ('最小ESDF距离(m)', '最小 ESDF 距离 (m)'),
                           ('后处理修正量(m)', '后处理修正量 $D_{post}$ (m)')]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, (metric, ylabel) in zip(axes, tradeoff_layout):
            if metric not in ablation_paired.columns:
                ax.set_visible(False)
                continue
            sns.boxplot(x='实验组', y=metric, data=ablation_paired, ax=ax,
                        order=['G0', 'G1', 'G2', 'G3'], palette=group_colors)
            sns.stripplot(x='实验组', y=metric, data=ablation_paired, ax=ax,
                          order=['G0', 'G1', 'G2', 'G3'], color='black',
                          size=2.5, alpha=0.35)
            y_max = ablation_paired[metric].max()
            span = y_max - ablation_paired[metric].min()
            ax.set_ylim(top=y_max + 0.18 * span if span > 0 else y_max * 1.2)
            _annotate_sig(ax, metric, y_max + 0.06 * span if span > 0 else y_max * 1.1)
            ax.set_xlabel('')
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xticklabels([group_labels[g.get_text()] for g in ax.get_xticklabels()])
        fig.suptitle('G0-G3 消融: 绕行代价 (H2) 与后处理依赖 (H3)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'A2_长度安全与后处理三联箱线图.png'),
                    dpi=SAVE_DPI, bbox_inches='tight')
        plt.close()
        print('已保存：A2_长度安全与后处理三联箱线图.png')

    # A3. w_kappa 敏感性曲线 (实验二)
    weight_data = data[data['实验组'].astype(str).str.startswith('w_kappa=')]
    if not weight_data.empty:
        weight_data = weight_data.copy()
        weight_data['w_kappa值'] = weight_data['实验组'].astype(str).str.split('=').str[1].astype(float)
        sensitivity_metrics = [('平均曲率(度/m)', '平均曲率 (°/m)'),
                               ('曲率平方积分(1/m)', '曲率平方积分 (1/m)'),
                               ('路径长度(m)', '路径长度 (m)'),
                               ('最小ESDF距离(m)', '最小 ESDF 距离 (m)')]
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, (metric, ylabel) in zip(axes, sensitivity_metrics):
            if metric not in weight_data.columns:
                ax.set_visible(False)
                continue
            summary = weight_data.groupby('w_kappa值')[metric].agg(['mean', 'std']).reset_index()
            ax.errorbar(summary['w_kappa值'], summary['mean'],
                        yerr=summary['std'], marker='o', capsize=4,
                        color=NATURE_PALETTE[0], lw=1.5)
            ax.set_xlabel('$w_\\kappa$', fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(alpha=0.3, linestyle='--')
        fig.suptitle('曲率权重 $w_\\kappa$ 敏感性: 均值 ± 标准差 (42 组)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'A3_w_kappa敏感性曲线.png'),
                    dpi=SAVE_DPI, bbox_inches='tight')
        plt.close()
        print('已保存：A3_w_kappa敏感性曲线.png')

    print(f'消融与权重分开配对分析完成: {output_dir}')


def run_b0_b3_batch(batch_dir):
    groups = ['B0', 'B1', 'B2', 'B3']
    frames = []
    for group in groups:
        csv_path = os.path.join(batch_dir, group, 'path_comparison.csv')
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f'{group} 结果不存在: {csv_path}')
        part = pd.read_csv(csv_path)
        required = {'实验编号', '规划器', '规划状态', '后处理模式'}
        missing = required - set(part.columns)
        if missing:
            raise ValueError(f'{csv_path} 缺少必要列: {sorted(missing)}')
        if set(part['规划器'].astype(str).str.strip()) != {group}:
            raise ValueError(f'{csv_path} 的规划器标签必须严格为 {group}')
        if set(part['后处理模式'].astype(str).str.strip()) != {'raw'}:
            raise ValueError(f'{csv_path} 消融必须统一使用 raw 模式')
        duplicate = part.duplicated(['实验编号', '规划器'], keep=False)
        if duplicate.any():
            raise ValueError(f'{csv_path} 存在重复实验编号')
        frames.append(part[part['规划状态'].astype(str).str.strip() == '成功'].copy())
    combined = pd.concat(frames, ignore_index=True)
    paired, ids = _paired_subset(combined.rename(columns={'规划器': '实验组'}), groups)
    if len(ids) < 2:
        raise ValueError(f'B0-B3 完整成功配对仅 {len(ids)} 组，至少需要 2 组')
    output_dir = os.path.join(batch_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    paired.to_csv(os.path.join(output_dir, 'B0-B3完整成功配对.csv'),
                  index=False, encoding='utf-8-sig')
    run_paired_planner_analysis(paired, output_dir, groups, 'B0-B3机制消融', '实验组')
    print(f'B0-B3 消融分析完成，完整配对 {len(ids)} 组: {output_dir}')


# -------------------------- 命令行入口 --------------------------
def main():
    parser = argparse.ArgumentParser(
        description='路径规划算法对比分析 (读取 path_comparison.csv, 输出统计图表)')
    parser.add_argument('--exp', choices=['nowind', 'wind', 'compare', 'ablation', 'compare4'], default='nowind',
                        help='实验类型: nowind、wind、compare、ablation(B0-B3)、compare4')
    parser.add_argument('--csv', default=None,
                        help='直接指定 path_comparison.csv 路径')
    parser.add_argument('--batch-dir', default=None,
                        help='compare4 或 B0-B3 的批次根目录')
    args = parser.parse_args()

    if args.exp == 'ablation':
        if not args.batch_dir:
            parser.error('--exp ablation 必须指定 --batch-dir')
        run_b0_b3_batch(os.path.abspath(args.batch_dir))
        return

    if args.exp == 'compare4':
        from analyze_compare4 import latest_result_dir, main as compare4_main
        if args.batch_dir:
            sys.argv = ['analyze_compare4.py', os.path.abspath(args.batch_dir)]
        else:
            root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'compare4'))
            sys.argv = ['analyze_compare4.py', latest_result_dir(root)]
        compare4_main()
        return

    # compare 模式: 读两组实验目录, 独立输出到 analysis_compare/
    if args.exp == 'compare':
        output_dir = './analysis_compare'
        os.makedirs(output_dir, exist_ok=True)
        print(f"[exp] 跨实验对比 | 输入: {EXP_DIRS['nowind']} vs {EXP_DIRS['wind']} "
              f"| 输出目录: {output_dir}")
        run_cross_analysis(EXP_DIRS['nowind'], EXP_DIRS['wind'], output_dir)
        return

    # 输入: --csv 优先, 否则按 --exp 取对应实验目录
    csv_path = args.csv if args.csv else os.path.join(EXP_DIRS[args.exp],
                                                      'path_comparison.csv')
    # 输出: 分目录归档, 避免两类实验图表互相覆盖
    out_tag = '_wind' if args.exp == 'wind' else ''
    output_dir = f'./analysis{out_tag}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"[exp] 实验类型: {args.exp} | 输入: {csv_path} | 输出目录: {output_dir}")
    run_analysis(csv_path, output_dir, args.exp)


# -------------------------- 分析主流程 --------------------------
def run_analysis(csv_path, OUTPUT_DIR, exp_name='nowind'):
    global df, desc_stats, assumption_df, normality_detail_df, kw_df, pair_results_df, \
           param_pair_df, win_counts, score_summary, contrib_summary, \
           sens_df, score_df, planners, PLANNER_COLORS

    # -------------------------- 1. 数据读取与预处理 --------------------------
    df = pd.read_csv(csv_path)
    if '规划状态' in df.columns:
        df = df[df['规划状态'].astype(str).str.strip() == '成功'].copy()

    print("=" * 50)
    exp_label = {'wind': '风场节能', 'compare4': '四算法对比'}.get(exp_name, '无风场')
    print(f"数据集基本信息 ({exp_label} 实验):")
    print(f"总记录数：{len(df)} 条")
    print(f"实验组数：{df['实验编号'].nunique()} 组")
    print(f"规划器类型：{df['规划器'].unique().tolist()}")
    print(f"穿墙次数合计：{df['穿墙次数'].sum()} (应为 0)")
    print("=" * 50)

    # 派生字段: 起终点直线距离 (用于耗时伸缩曲线)
    df['直线距离(m)'] = np.sqrt((df['终点X'] - df['起点X']) ** 2 +
                                (df['终点Y'] - df['起点Y']) ** 2 +
                                (df['终点Z'] - df['起点Z']) ** 2)

    observed_planners = list(df['规划器'].unique())
    if exp_name == 'compare4':
        planners = [p for p in COMPARE4_ORDER if p in observed_planners]
        planners += [p for p in observed_planners if p not in planners]
    else:
        planners = observed_planners
    score_weights = COMPARE4_WEIGHTS if exp_name == 'compare4' else WEIGHTS
    PLANNER_COLORS = {p: NATURE_PALETTE[i % len(NATURE_PALETTE)] for i, p in enumerate(planners)}
    sns.set_palette([PLANNER_COLORS[p] for p in planners])

    core_metrics = ['路径长度(m)', '平均曲率(度/m)', '最大转角(度)',
                    '最小ESDF距离(m)', '规划耗时(ms)', '高度方差']
    # 新版 main.cpp 输出的平滑性/后处理指标；兼容尚未重跑生成的旧 CSV。
    new_metrics = ['曲率平方积分(1/m)', '高曲率段占比(%)',
                   '后处理修正量(m)', '最大连续急转弯段数']
    core_metrics.extend(m for m in new_metrics if m in df.columns)

    # 显著性标记: 基于 Bonferroni 校正后 α; 另提供原始 p 的星标
    N_PAIRS = len(planners) * (len(planners) - 1) // 2
    BONF_ALPHA = 0.05 / N_PAIRS

    def pair_sig(p1, p2, metric):
        """便捷查询: 某对比某指标的 Bonferroni 结论 ('sig'/'ns') 与 p 值."""
        row = pair_results_df[(pair_results_df['对比'] == f'{p1} vs {p2}') &
                              (pair_results_df['指标'] == metric)]
        if row.empty:
            row = pair_results_df[(pair_results_df['对比'] == f'{p2} vs {p1}') &
                                  (pair_results_df['指标'] == metric)]
        return row.iloc[0]['p值'], row.iloc[0]['Bonferroni显著']

    # -------------------------- 2. 描述性统计分析 --------------------------
    desc_stats = df.groupby('规划器')[core_metrics].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
    print("\n核心指标描述性统计 (含 mean/median)：")
    print(desc_stats)

    # -------------------------- 2.5 稳健统计口径 (中位数 + 几何均值) --------------------------
    # 回应审稿意见: 均值易被极端离群值 (如 RRT* 未收敛路径) 主导,
    # 正文结论需同时报告 中位数 与 几何均值 (稳健中心趋势), 三者排序一致才稳健.
    print("\n" + "=" * 50)
    print("稳健统计口径 (中位数 / 几何均值) 对比:")
    print("  —— 用于佐证'均值结论是否被离群值驱动'; 若三种口径排序一致, 结论稳健.")
    robust_metrics = ['路径长度(m)', '平均曲率(度/m)', '规划耗时(ms)', '最小ESDF距离(m)']
    robust_rows = []
    for p in planners:
        sub = df[df['规划器'] == p]
        row = {'规划器': p}
        for m in robust_metrics:
            vals = sub[m].values
            row[f'{m} (中位数)'] = round(float(np.median(vals)), 3)
            pos = vals[vals > 0]
            g = float(stats.gmean(pos)) if len(pos) else np.nan
            row[f'{m} (几何均值)'] = round(g, 3) if np.isfinite(g) else np.nan
        robust_rows.append(row)
    robust_df = pd.DataFrame(robust_rows)
    print(robust_df.to_string(index=False))

    # 中位数口径下各指标最优者 (直接回应'中位数口径结论是否反转')
    print("\n中位数口径下的最优算法 (路径长度/平均曲率/耗时越小越优; 最小ESDF 越大越优):")
    for m in robust_metrics:
        col = f'{m} (中位数)'
        sub = robust_df[['规划器', col]]
        best = sub.loc[sub[col].idxmax() if m == '最小ESDF距离(m)' else sub[col].idxmin(), '规划器']
        print(f"  {m}: {best}")

    # -------------------------- 3. 统计显著性检验 --------------------------
    # 3.0 前置假设检验 (保留, 用于论证"为何主检验采用非参数方法")
    print("\n" + "=" * 50)
    print("前置假设检验 (α=0.05)：")
    assumption_results = []
    normality_results = []
    for metric in core_metrics:
        groups = [df[df['规划器'] == p][metric].dropna().values for p in planners]
        norm_pass = True
        for p, g in zip(planners, groups):
            statistic, p_norm = np.nan, np.nan
            if 3 <= len(g) <= 5000:
                statistic, p_norm = stats.shapiro(g)
                norm_pass &= p_norm >= 0.05
            else:
                norm_pass = False
            normality_results.append({
                '规划器': p, '指标': metric, '样本量': len(g),
                'Shapiro_W': float(statistic), 'p值': float(p_norm),
                '正态性满足': ('是' if p_norm >= 0.05 else '否')
                if np.isfinite(p_norm) else '样本量不足'
            })
        _, p_lev = stats.levene(*groups, center='median')
        assumption_results.append({
            '指标': metric,
            '正态性满足': '是' if norm_pass else '否',
            '方差齐性满足': '是' if p_lev >= 0.05 else '否',
            'Levene_p值': round(p_lev, 4)
        })
    normality_detail_df = pd.DataFrame(normality_results)
    assumption_df = pd.DataFrame(assumption_results).set_index('指标')
    print(assumption_df)
    print('\nShapiro-Wilk 正态性完整表:')
    print(normality_detail_df.to_string(index=False))
    print("注: 多数指标不满足正态/方差齐性 → 本文以非参数检验为主, 参数检验仅作对照。")

    # 3.1 [主检验] Kruskal-Wallis 全局检验
    print("\n" + "=" * 50)
    print("Kruskal-Wallis 全局检验 (主方法)：")
    kw_results = {}
    for metric in core_metrics:
        groups = [df[df['规划器'] == p][metric].values for p in planners]
        h_val, p_val = stats.kruskal(*groups)
        kw_results[metric] = {'H值': round(h_val, 4), 'P值': p_val}
        print(f"{metric}: H={h_val:.4f}, p={p_val:.3e} {_sig_marker(p_val)}")
    kw_df = pd.DataFrame(kw_results).T

    # 3.2 [主检验] Wilcoxon 符号秩配对检验 (Bonferroni 校正)
    print("\n" + "=" * 50)
    print(f"Wilcoxon 配对检验 (主方法, Bonferroni α=0.05→{BONF_ALPHA:.4f})：")
    pair_results = []
    # 预先按实验编号自连接一次, 避免内层循环重复 merge
    _mg_all = df.merge(df, on='实验编号', suffixes=('_1', '_2'))
    for i in range(len(planners)):
        for j in range(i + 1, len(planners)):
            p1, p2 = planners[i], planners[j]
            pair_data = _mg_all[(_mg_all['规划器_1'] == p1) & (_mg_all['规划器_2'] == p2)]
            print(f"\n--- {p1} vs {p2} ---")
            for metric in core_metrics:
                v1, v2 = pair_data[f'{metric}_1'], pair_data[f'{metric}_2']
                try:
                    w_val, p_val = stats.wilcoxon(v1, v2)
                except ValueError:      # 全部差值为 0
                    w_val, p_val = 0.0, 1.0
                diff_pct = ((v2.median() - v1.median()) / v1.median()) * 100 if v1.median() != 0 else 0.0
                sig_bonf = 'sig' if p_val < BONF_ALPHA else 'ns'
                print(f"{metric}: 中位数变化 {diff_pct:+.2f}% (p={p_val:.3e}, Bonferroni {sig_bonf})")
                pair_results.append({
                    '对比': f'{p1} vs {p2}', '指标': metric,
                    '中位数变化(%)': round(diff_pct, 2), 'W值': round(float(w_val), 1),
                    'p值': p_val, 'Bonferroni显著': sig_bonf
                })
    pair_results_df = pd.DataFrame(pair_results)

    # 3.3 [对照] 原参数检验 (ANOVA + 配对t), 仅存档
    print("\n" + "=" * 50)
    print("参数检验对照 (仅供参考, 存档用)：")
    anova_results = {}
    for metric in core_metrics:
        groups = [df[df['规划器'] == p][metric].values for p in planners]
        f_val, p_val = stats.f_oneway(*groups)
        anova_results[metric] = {'F值': round(f_val, 4), 'P值': p_val}
    param_pair = []
    for i in range(len(planners)):
        for j in range(i + 1, len(planners)):
            p1, p2 = planners[i], planners[j]
            pair_data = _mg_all[(_mg_all['规划器_1'] == p1) & (_mg_all['规划器_2'] == p2)]
            for metric in core_metrics:
                t_val, p_val = stats.ttest_rel(pair_data[f'{metric}_1'], pair_data[f'{metric}_2'])
                param_pair.append({'对比': f'{p1} vs {p2}', '指标': metric,
                                   't值': round(t_val, 4), 'p值': p_val,
                                   'Bonferroni显著': 'sig' if p_val < BONF_ALPHA else 'ns'})
    param_pair_df = pd.DataFrame(param_pair)

    # -------------------------- 4. 各维度优胜次数统计 --------------------------
    win_metrics = [m for m in core_metrics if m != '路径点数']
    print("\n" + "=" * 50)
    print("各指标优胜次数统计：")
    win_counts = pd.DataFrame(index=planners, columns=win_metrics, data=0)
    for exp_id, group in df.groupby('实验编号'):
        for metric in win_metrics:
            best_val = group[metric].max() if metric == '最小ESDF距离(m)' else group[metric].min()
            for w in group.loc[group[metric] == best_val, '规划器']:
                win_counts.loc[w, metric] += 1
    print(win_counts)

    # -------------------------- 5. 综合性能评分 + 贡献分解 + 权重敏感性 --------------------------
    def compute_scores(data, weights):
        """按给定权重计算每条记录的综合得分与各指标贡献."""
        out = data.copy()
        out['综合得分'] = 0.0
        contrib = pd.DataFrame(index=data.index)
        for metric, w in weights.items():
            s = data[metric]
            rng = s.max() - s.min()
            rng = rng if rng > 0 else 1.0
            norm = (s - s.min()) / rng if metric == '最小ESDF距离(m)' else (s.max() - s) / rng
            out['综合得分'] += norm * w
            contrib[metric] = norm * w
        return out, contrib

    score_df, contrib_df = compute_scores(df, score_weights)
    contrib_df['规划器'] = df['规划器'].values
    contrib_summary = contrib_df.groupby('规划器').mean().round(4)
    score_summary = score_df.groupby('规划器')['综合得分'].agg(['mean', 'std', 'min', 'max']).round(4)
    print("\n" + "=" * 50)
    print("综合性能评分结果：")
    print(score_summary.sort_values('mean', ascending=False))
    print("\n综合得分贡献分解 (均值)：")
    print(contrib_summary)

    # 5.1 权重敏感性: 各权重 ±20% 均匀扰动, 统计排名第一频率
    print("\n" + "=" * 50)
    print("权重敏感性分析 (各权重 ×U(0.8,1.2), 500 次扰动)：")
    rng = np.random.default_rng(42)
    N_TRIALS = 500
    base_w = np.array(list(score_weights.values()))
    metric_keys = list(score_weights.keys())
    rank1_count = {p: 0 for p in planners}
    for _ in range(N_TRIALS):
        w_trial = base_w * rng.uniform(0.8, 1.2, len(base_w))
        sd_t, _ = compute_scores(df, dict(zip(metric_keys, w_trial)))
        best = sd_t.groupby('规划器')['综合得分'].mean().idxmax()
        rank1_count[best] += 1
    sens_df = pd.DataFrame({'规划器': list(rank1_count.keys()),
                            '排名第一次数': list(rank1_count.values())})
    sens_df['排名第一频率(%)'] = (sens_df['排名第一次数'] / N_TRIALS * 100).round(1)
    print(sens_df.to_string(index=False))

    # -------------------------- 5.5 RRT* 浮动稳健性分析 --------------------------
    # 背景: RRT* 为随机算法, path_comparison.csv 仅存"多次运行取长度中位数"的代表值,
    #       若统计检验/评分只基于单一代表值, 会低估 RRT* 的随机波动, 结论不公正。
    # 做法: 读 rrt_fluctuation.csv 的 [min,max], 用蒙特卡洛在区间内采样 RRT* 各指标,
    #       重新计算 KW 全局 + 配对 Wilcoxon 的 p 值分布与"显著率", 判定结论是否稳健。
    rrt_csv = os.path.join(os.path.dirname(csv_path), 'rrt_fluctuation.csv')
    float_kw_df = None
    float_pair_df = None
    score_range_df = None
    has_rrt_range = os.path.exists(rrt_csv)

    if has_rrt_range:
        print("\n" + "=" * 50)
        print("RRT* 浮动稳健性分析 (随机采样波动是否颠覆结论):")
        rrt_ranges = load_rrt_ranges(rrt_csv)
        RRT_NAME = 'RRT*'
        other_planners = [p for p in planners if p != RRT_NAME]
        M = 200                       # 蒙特卡洛采样次数
        mc_rng = np.random.default_rng(2024)

        rrt_sub = df[df['规划器'] == RRT_NAME].sort_values('实验编号').reset_index(drop=True)
        rrt_ids = rrt_sub['实验编号'].astype(int).values

        def sample_rrt(metric):
            """每组按 [min,max] 均匀采样一个 RRT* 值; 无区间信息时回退中位数代表."""
            arr = np.empty(len(rrt_ids))
            for k, eid in enumerate(rrt_ids):
                med = float(rrt_sub.loc[k, metric])
                r0 = rrt_ranges.get(eid, {}).get(metric)
                if (r0 is not None and r0[0] is not None and r0[1] is not None
                        and np.isfinite(r0[0]) and np.isfinite(r0[1])):
                    lo, hi = min(r0), max(r0)
                    arr[k] = med if hi <= lo else mc_rng.uniform(lo, hi)
                else:
                    arr[k] = med
            return arr

        # (a) KW 全局检验 p 值分布
        print(f"\n[KW 全局] RRT* 区间采样下 p 值分布 (M={M}):")
        float_kw_rows = []
        for metric in core_metrics:
            others = [df[df['规划器'] == p][metric].values for p in other_planners]
            ps = []
            for _ in range(M):
                groups = others + [sample_rrt(metric)]
                _, pv = stats.kruskal(*groups)
                ps.append(pv)
            ps = np.array(ps)
            float_kw_rows.append({
                '指标': metric,
                '原始p值(代表值)': round(float(kw_results[metric]['P值']), 4),
                'p值中位数': round(float(np.median(ps)), 4),
                'p值5%-95%分位': f"{np.percentile(ps, 5):.2e} ~ {np.percentile(ps, 95):.2e}",
                '显著率(%)': round(float(100 * np.mean(ps < 0.05)), 1),
            })
        float_kw_df = pd.DataFrame(float_kw_rows)
        print(float_kw_df.to_string(index=False))

        # (b) 配对 Wilcoxon (确定性算法 vs RRT*) p 值分布
        print(f"\n[配对 Wilcoxon] 确定性算法 vs RRT* (区间采样, M={M}, Bonferroni α={BONF_ALPHA:.4f}):")
        float_pair_rows = []
        for other in other_planners:
            base = df[df['规划器'] == other].set_index('实验编号')
            for metric in core_metrics:
                if metric not in base.columns:
                    continue
                v_other = base[metric].reindex([int(x) for x in rrt_ids]).values
                valid = np.isfinite(v_other)
                if valid.sum() < 3:
                    continue
                ps = []
                for _ in range(M):
                    v_rrt = sample_rrt(metric)
                    try:
                        _, pv = stats.wilcoxon(v_other[valid], v_rrt[valid])
                    except ValueError:
                        pv = 1.0
                    ps.append(pv)
                ps = np.array(ps)
                float_pair_rows.append({
                    '对比': f'{other} vs {RRT_NAME}', '指标': metric,
                    'p值中位数': round(float(np.median(ps)), 4),
                    'p值95%分位': round(float(np.percentile(ps, 95)), 4),
                    '显著率(%)': round(float(100 * np.mean(ps < BONF_ALPHA)), 1),
                })
        float_pair_df = pd.DataFrame(float_pair_rows)
        print(float_pair_df.to_string(index=False))

        # (c) 综合评分浮动区间 (RRT* best/worst 边界)
        print("\n[综合评分] RRT* 浮动区间 (每组取 最优/最劣 边界):")
        higher_better = {'最小ESDF距离(m)': True}
        g_min = {m: float(df[m].min()) for m in score_weights}
        g_max = {m: float(df[m].max()) for m in score_weights}

        def norm_score(metric, val):
            rng = g_max[metric] - g_min[metric]
            rng = rng if rng > 0 else 1.0
            if higher_better.get(metric, False):
                return (val - g_min[metric]) / rng
            return (g_max[metric] - val) / rng

        score_range_rows = []
        for p in planners:
            base_sc = float(score_summary.loc[p, 'mean'])
            if p != RRT_NAME:
                score_range_rows.append({'规划器': p, '综合得分_代表值': round(base_sc, 4),
                                         '综合得分_下限': round(base_sc, 4),
                                         '综合得分_上限': round(base_sc, 4)})
            else:
                best_tot = 0.0
                worst_tot = 0.0
                for metric, w in score_weights.items():
                    b_vals, w_vals = [], []
                    for k, eid in enumerate(rrt_ids):
                        med = float(rrt_sub.loc[k, metric])
                        r0 = rrt_ranges.get(eid, {}).get(metric)
                        lo, hi = med, med
                        if (r0 is not None and r0[0] is not None and r0[1] is not None
                                and np.isfinite(r0[0]) and np.isfinite(r0[1])):
                            lo, hi = min(r0), max(r0)
                        if higher_better.get(metric, False):
                            b_vals.append(hi); w_vals.append(lo)
                        else:
                            b_vals.append(lo); w_vals.append(hi)
                    best_tot += norm_score(metric, float(np.mean(b_vals))) * w
                    worst_tot += norm_score(metric, float(np.mean(w_vals))) * w
                score_range_rows.append({'规划器': p, '综合得分_代表值': round(base_sc, 4),
                                         '综合得分_下限': round(worst_tot, 4),
                                         '综合得分_上限': round(best_tot, 4)})
        score_range_df = pd.DataFrame(score_range_rows)
        print(score_range_df.to_string(index=False))
    else:
        print(f"\n[提示] 未找到 RRT* 浮动范围文件: {rrt_csv}")
        print("       跳过浮动稳健性分析 (需先以 --rrt-runs>1 运行 main.cpp 生成 rrt_fluctuation.csv)")

    # -------------------------- 6. 数据可视化 (叙事链: 效率→质量→平滑→综合→机理) --------------------------
    metric_units = {
        '路径长度(m)': '(m)', '规划耗时(ms)': '(ms)', '平均曲率(度/m)': '(°/m)',
        '曲率平方积分(1/m)': '(1/m)', '高曲率段占比(%)': '(%)',
        '后处理修正量(m)': '(m)', '最大连续急转弯段数': '',
        '最小ESDF距离(m)': '(m)', '最大转角(度)': '(°)', '高度方差': '(m²)'
    }

    def _box_with_sig(ax, metric, log_y=False, kw_text_pos=0.98):
        """箱线+散点, 标注 KW 全局检验结果 (ns 与 *** 同等醒目)."""
        sns.boxplot(x='规划器', y=metric, hue='规划器', data=df, ax=ax,
                    palette=PLANNER_COLORS, width=0.5, order=planners,
                    hue_order=planners, legend=False)
        sns.stripplot(x='规划器', y=metric, data=df, ax=ax,
                      color='black', alpha=0.4, size=3, jitter=True, order=planners)
        if log_y:
            ax.set_yscale('log')
        p_val = kw_results[metric]['P值']
        sig = _sig_marker(p_val)
        label = f'KW p={p_val:.2e} ({sig})' if sig != 'ns' else f'KW p={p_val:.2f} (ns, 组间差异不显著)'
        ax.text(0.5, kw_text_pos, label, transform=ax.transAxes, ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round',
                          facecolor='lightgray' if sig == 'ns' else 'wheat', alpha=0.8))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xlabel('')

    # ========== 图 01: 规划耗时箱线图 (对数轴) —— 核心卖点: FMM 快一个数量级 ==========
    fig, ax = plt.subplots(figsize=(8, 6))
    _box_with_sig(ax, '规划耗时(ms)', log_y=True)
    ax.set_title('规划耗时分布对比 (对数坐标)', fontsize=14)
    ax.set_ylabel('规划耗时 (ms, log)', fontsize=11)
    # 均值写入 x 轴刻度标签 (避免与箱体/散点重叠)
    mean_labels = [f'{p}\n均值 {df[df["规划器"] == p]["规划耗时(ms)"].mean():.0f} ms' for p in planners]
    ax.set_xticks(range(len(planners)))
    ax.set_xticklabels(mean_labels, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_耗时箱线图_对数.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("\n已保存：01_耗时箱线图_对数.png")
    plt.close()

    # ========== 图 02: 耗时-任务距离伸缩曲线 —— scalability 证据 ==========
    dist_bins = [0, 800, 1500, 2200, 4000]
    dist_labels = ['<800', '800-1500', '1500-2200', '>2200']
    df['距离桶'] = pd.cut(df['直线距离(m)'], bins=dist_bins, labels=dist_labels)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for p in planners:
        sub = df[df['规划器'] == p]
        ax.scatter(sub['直线距离(m)'], sub['规划耗时(ms)'], s=35, alpha=0.35,
                   color=PLANNER_COLORS[p], edgecolors='none')
        grp = sub.groupby('距离桶', observed=True)['规划耗时(ms)'].agg(['mean', 'sem'])
        x_pos = [(dist_bins[i] + dist_bins[i + 1]) / 2 for i in range(len(grp))]
        ax.errorbar(x_pos, grp['mean'], yerr=grp['sem'], marker='o', markersize=8,
                    linewidth=2.2, capsize=5, color=PLANNER_COLORS[p], label=p,
                    markeredgecolor='black', markeredgewidth=0.8)
    ax.set_yscale('log')
    ax.set_title('规划耗时随任务距离的伸缩性 (均值±标准误, 对数坐标)', fontsize=14)
    ax.set_xlabel('起终点直线距离 (m)', fontsize=11)
    ax.set_ylabel('规划耗时 (ms, log)', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--', which='both')
    ax.legend(title='规划器', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_耗时距离伸缩曲线.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：02_耗时距离伸缩曲线.png")
    plt.close()

    # ========== 图 03: 长度 + min ESDF 双联箱线图 —— "快而不牺牲质量" (ns 醒目) ==========
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _box_with_sig(axes[0], '路径长度(m)')
    axes[0].set_title('路径长度分布 (几何质量)', fontsize=13)
    axes[0].set_ylabel('路径长度 (m)', fontsize=11)
    _box_with_sig(axes[1], '最小ESDF距离(m)')
    axes[1].set_title('最小 ESDF 距离分布 (安全裕度)', fontsize=13)
    axes[1].set_ylabel('最小 ESDF 距离 (m)', fontsize=11)
    quality_title = ('四算法路径质量与安全性对比' if exp_name == 'compare4'
                     else '路径质量与安全性: 组间差异不显著 → 效率优势非质量置换')
    fig.suptitle(quality_title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_长度与安全双联箱线图.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：03_长度与安全双联箱线图.png")
    plt.close()

    # ========== 图 04: 平均曲率 + 转角双联箱线图 —— FMM 平滑性优势 ==========
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _box_with_sig(axes[0], '平均曲率(度/m)', log_y=True)
    axes[0].set_title('平均曲率分布 (对数坐标)', fontsize=13)
    axes[0].set_ylabel('平均曲率 (°/m, log)', fontsize=11)
    # 四算法模式以 FGDA* 为主要对照；常规模式沿用 FMM。
    focal_planner = 'FGDA*' if exp_name == 'compare4' else 'FMM'
    txt = []
    for p2 in planners:
        if p2 == focal_planner:
            continue
        pv, sg = pair_sig(focal_planner, p2, '平均曲率(度/m)')
        txt.append(f'{focal_planner} vs {p2}: p={pv:.1e} ({sg})')
    axes[0].text(0.02, 0.02, '\n'.join(txt), transform=axes[0].transAxes, fontsize=9,
                 va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    _box_with_sig(axes[1], '最大转角(度)')
    axes[1].set_title('最大转角分布', fontsize=13)
    axes[1].set_ylabel('最大转角 (°)', fontsize=11)
    fig.suptitle('路径平滑性对比 (Wilcoxon 配对检验, Bonferroni 校正)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_平滑性双联箱线图.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：04_平滑性双联箱线图.png")
    plt.close()

    # ========== 图 05: 综合评分 + 贡献分解 双联图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1.4]})
    mean_scores = score_summary['mean'].sort_values(ascending=False)
    std_scores = score_summary.loc[mean_scores.index, 'std']
    colors_sorted = [PLANNER_COLORS[p] for p in mean_scores.index]
    bars = axes[0].bar(mean_scores.index, mean_scores.values, yerr=std_scores.values,
                       color=colors_sorted, alpha=0.85, capsize=8, edgecolor='black', linewidth=0.8,
                       width=0.35)
    axes[0].set_title('综合性能评分 (权重见正文)', fontsize=13)
    axes[0].set_ylabel('综合得分 (满分 1.0)', fontsize=11)
    axes[0].set_ylim(0, 1.15)
    for bar, m_v, s_v in zip(bars, mean_scores.values, std_scores.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, m_v + s_v + 0.02,
                     f'{m_v:.3f}±{s_v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    # 右: 贡献分解堆叠条 (顺序与左图一致: 按综合得分降序)
    contrib_sorted = contrib_summary.loc[mean_scores.index]
    bottom = np.zeros(len(contrib_sorted))
    for metric in score_weights:
        vals = contrib_sorted[metric].values
        axes[1].bar(contrib_sorted.index, vals, bottom=bottom,
                    label=f'{metric} (w={score_weights[metric]})',
                    color=CONTRIB_COLORS.get(metric), edgecolor='white', linewidth=0.6, alpha=0.9, width=0.35)
        bottom += vals
    axes[1].set_title('综合得分贡献分解 (均值)', fontsize=13)
    axes[1].set_ylabel('贡献值', fontsize=11)
    axes[1].set_ylim(0, bottom.max() * 1.15)
    axes[1].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_综合评分与贡献分解.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：05_综合评分与贡献分解.png")
    plt.close()

    # ========== 图 06: 优胜次数堆叠柱状图 ==========
    fig, ax = plt.subplots(figsize=(11, 6))
    win_counts.T.plot(kind='bar', stacked=True, color=[PLANNER_COLORS[p] for p in planners],
                      ax=ax, edgecolor='white', linewidth=0.8)
    ax.set_title('各指标维度优胜次数统计 (共 42 组实验, 并列计次)', fontsize=14)
    ax.set_ylabel('优胜次数', fontsize=11)
    ax.set_xlabel('评估指标', fontsize=11)
    plt.xticks(rotation=20, ha='right')
    ax.legend(title='规划器', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for container in ax.containers:
        ax.bar_label(container, label_type='center', fontsize=9, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_优胜次数统计.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：06_优胜次数统计.png")
    plt.close()

    # ========== 图 07: 耗时-平均曲率 Pareto 散点 (log-log) —— FMM 占据双目标最优角 ==========
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for p in planners:
        sub = df[df['规划器'] == p]
        ax.scatter(sub['规划耗时(ms)'], sub['平均曲率(度/m)'], s=80, alpha=0.7, label=p,
                   color=PLANNER_COLORS[p], edgecolors='black', linewidth=0.5)
        ax.scatter(sub['规划耗时(ms)'].mean(), sub['平均曲率(度/m)'].mean(),
                   marker='*', s=350, color=PLANNER_COLORS[p], edgecolors='black',
                   linewidth=1.5, zorder=5)
    # 非支配解: 耗时越小 + 平均曲率越小
    pts = df[['规划耗时(ms)', '平均曲率(度/m)']].values
    pareto_pts = []
    for i, (ti, ci) in enumerate(pts):
        dominated = any(tj <= ti and cj <= ci and (tj < ti or cj < ci)
                        for j, (tj, cj) in enumerate(pts) if i != j)
        if not dominated:
            pareto_pts.append((ti, ci))
    if len(pareto_pts) >= 2:
        pareto_pts.sort(key=lambda t: t[0])
        px, py = zip(*pareto_pts)
        ax.plot(px, py, 'k--', alpha=0.6, linewidth=1.5, label='Pareto 前沿')
        ax.scatter(px, py, facecolors='none', edgecolors='black', s=120, linewidth=1.5, zorder=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('效率-平滑性双目标分布 (log-log)\n左下方向为最优, ★为均值', fontsize=13)
    ax.set_xlabel('规划耗时 (ms, log) ← 越小越优', fontsize=11)
    ax.set_ylabel('平均曲率 (°/m, log) ← 越小越优', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--', which='both')
    ax.legend(title='规划器', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_耗时曲率Pareto散点.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：07_耗时曲率Pareto散点.png")
    plt.close()

    # ========== 图 08: 权重敏感性分析 ==========
    fig, ax = plt.subplots(figsize=(8, 5))
    sens_sorted = sens_df.sort_values('排名第一频率(%)', ascending=False)
    bars = ax.bar(sens_sorted['规划器'], sens_sorted['排名第一频率(%)'],
                  color=[PLANNER_COLORS[p] for p in sens_sorted['规划器']],
                  alpha=0.85, edgecolor='black', linewidth=0.8)
    for bar, v in zip(bars, sens_sorted['排名第一频率(%)']):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.1f}%',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_title(f'综合评分权重敏感性 (各权重 ±20% 均匀扰动, {N_TRIALS} 次)', fontsize=13)
    ax.set_ylabel('综合得分排名第一频率 (%)', fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '08_权重敏感性分析.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：08_权重敏感性分析.png")
    plt.close()

    # ========== 图 A1: 指标相关性热力图 (附录) ==========
    fig, ax = plt.subplots(figsize=(9, 7.5))
    corr_matrix = df[core_metrics].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True,
                linewidths=0.8, mask=mask, vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': '相关系数', 'shrink': 0.7, 'fraction': 0.046},
                annot_kws={'size': 10, 'weight': 'bold'})
    ax.set_title('路径规划指标相关性热力图 (附录)', fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'A1_指标相关性热力图_附录.png'), dpi=SAVE_DPI, bbox_inches='tight')
    print("已保存：A1_指标相关性热力图_附录.png")
    plt.close()

    # -------------------------- 6.5 三算法风场分析对比 (仅 --exp wind) --------------------------
    # 数据源: path_wind_metrics.csv (main.cpp 输出, 与 path_comparison.csv 同目录)
    #   列: 实验编号, 规划器, 平均顶风分量(m/s), 平均风速(m/s), 风场代价
    # 分析口径与主流程一致: KW 全局 + Wilcoxon 配对 (Bonferroni), ns 与 *** 同等醒目
    wind_pair_df = None
    wind_sheets = []
    if exp_name == 'wind':
        wind_csv = os.path.join(os.path.dirname(csv_path), 'path_wind_metrics.csv')
        if not os.path.exists(wind_csv):
            print(f"\n[警告] 未找到风场指标文件: {wind_csv}")
            print("       跳过三算法风场对比分析 (需 main.cpp --wind 实验产物)")
        else:
            print("\n" + "=" * 50)
            print("三算法风场指标对比 (节能规划实验):")
            wdf = pd.read_csv(wind_csv)
            if '规划状态' in wdf.columns:
                wdf = wdf[wdf['规划状态'].astype(str).str.strip() == '成功'].copy()
            wdf = wdf.dropna(subset=['平均顶风分量(m/s)'])
            # 合并路径长度 (能耗代理需要)
            wdf = wdf.merge(df[['实验编号', '规划器', '路径长度(m)']],
                            on=['实验编号', '规划器'], how='left')
            # 能耗代理 (立方律): E = Σ 段长·(1+顶风/v_g)³, v_g=10 m/s
            #   阻力功率 P ∝ v_air³ → 单位距离能耗 ∝ (1+顶风/v_g)³;
            #   与跨实验对比 (C 系列) 同一公式, 保证口径一致
            V_G = 10.0
            wdf['能耗代理'] = wdf['路径长度(m)'] * (1.0 + wdf['平均顶风分量(m/s)'] / V_G) ** 3

            wind_metrics = ['平均顶风分量(m/s)', '平均风速(m/s)', '能耗代理']

            # 描述性统计
            wind_desc = wdf.groupby('规划器')[wind_metrics].agg(['mean', 'std']).round(3)
            print(wind_desc)

            # KW 全局检验
            kw_wind = {}
            for metric in wind_metrics:
                groups = [wdf[wdf['规划器'] == p][metric].values for p in planners]
                h_val, p_val = stats.kruskal(*groups)
                kw_wind[metric] = {'H值': round(h_val, 4), 'P值': p_val}
            kw_wind_df = pd.DataFrame(kw_wind).T
            for metric in wind_metrics:
                print(f"{metric}: H={kw_wind[metric]['H值']:.4f}, "
                      f"p={kw_wind[metric]['P值']:.3e} {_sig_marker(kw_wind[metric]['P值'])}")

            # Wilcoxon 符号秩配对检验 (Bonferroni 校正)
            print(f"\nWilcoxon 配对检验 (风场指标, Bonferroni α={BONF_ALPHA:.4f}):")
            _wmg = wdf.merge(wdf, on='实验编号', suffixes=('_1', '_2'))
            wind_pair_results = []
            for i in range(len(planners)):
                for j in range(i + 1, len(planners)):
                    p1, p2 = planners[i], planners[j]
                    wp = _wmg[(_wmg['规划器_1'] == p1) & (_wmg['规划器_2'] == p2)]
                    for metric in wind_metrics:
                        v1, v2 = wp[f'{metric}_1'], wp[f'{metric}_2']
                        try:
                            _, p_val = stats.wilcoxon(v1, v2)
                        except ValueError:      # 全部差值为 0
                            p_val = 1.0
                        diff_pct = ((v2.median() - v1.median()) / abs(v1.median()) * 100
                                    if v1.median() != 0 else 0.0)
                        wind_pair_results.append({
                            '对比': f'{p1} vs {p2}', '指标': metric,
                            '中位数变化(%)': round(diff_pct, 2), 'p值': p_val,
                            'Bonferroni显著': 'sig' if p_val < BONF_ALPHA else 'ns'
                        })
            wind_pair_df = pd.DataFrame(wind_pair_results)
            print(wind_pair_df.to_string(index=False))

            # 风场维度优胜统计: 顶风最低 / 能耗代理最低 / 净顺风组数
            wind_win = pd.DataFrame(0, index=planners,
                                    columns=['顶风最低', '能耗代理最低', '净顺风组数'])
            for _, grp in wdf.groupby('实验编号'):
                wind_win.loc[grp.loc[grp['平均顶风分量(m/s)'].idxmin(), '规划器'], '顶风最低'] += 1
                wind_win.loc[grp.loc[grp['能耗代理'].idxmin(), '规划器'], '能耗代理最低'] += 1
            wind_win['净顺风组数'] = (wdf[wdf['平均顶风分量(m/s)'] < 0]
                                     .groupby('规划器').size()
                                     .reindex(planners).fillna(0).astype(int))
            print("\n各算法风场维度优胜统计:")
            print(wind_win)

            # ---- 图 W1: 顶风分量 + 风速 双联箱线图 (KW 标注, 虚线=零风) ----
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            for ax, metric, title in zip(
                    axes, ['平均顶风分量(m/s)', '平均风速(m/s)'],
                    ['平均顶风分量 (负=净顺风, 越低越优)', '平均风速 (大风区暴露度)']):
                sns.boxplot(x='规划器', y=metric, data=wdf, ax=ax,
                            palette=[PLANNER_COLORS[p] for p in planners],
                            width=0.5, order=planners)
                sns.stripplot(x='规划器', y=metric, data=wdf, ax=ax,
                              color='black', alpha=0.4, size=3, jitter=True, order=planners)
                p_val = kw_wind[metric]['P值']
                sig = _sig_marker(p_val)
                label = (f'KW p={p_val:.2e} ({sig})' if sig != 'ns'
                         else f'KW p={p_val:.2f} (ns)')
                ax.text(0.5, 0.98, label, transform=ax.transAxes, ha='center',
                        va='top', fontsize=9,
                        bbox=dict(boxstyle='round',
                                  facecolor='lightgray' if sig == 'ns' else 'wheat',
                                  alpha=0.8))
                ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
                ax.set_title(title, fontsize=12)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_xlabel('')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'W1_顶风与风速双联箱线图.png'),
                        dpi=SAVE_DPI, bbox_inches='tight')
            print("\n已保存：W1_顶风与风速双联箱线图.png")
            plt.close()

            # ---- 图 W2: 能耗代理箱线图 (节能效果的综合裁决指标) ----
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x='规划器', y='能耗代理', data=wdf, ax=ax,
                        palette=[PLANNER_COLORS[p] for p in planners],
                        width=0.5, order=planners)
            sns.stripplot(x='规划器', y='能耗代理', data=wdf, ax=ax,
                          color='black', alpha=0.4, size=3, jitter=True, order=planners)
            p_val = kw_wind['能耗代理']['P值']
            sig = _sig_marker(p_val)
            label = (f'KW p={p_val:.2e} ({sig})' if sig != 'ns'
                     else f'KW p={p_val:.2f} (ns)')
            ax.text(0.5, 0.98, label, transform=ax.transAxes, ha='center', va='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round',
                              facecolor='lightgray' if sig == 'ns' else 'wheat', alpha=0.8))
            mean_labels = [f'{p}\n均值 {wdf[wdf["规划器"] == p]["能耗代理"].mean():.0f}'
                           for p in planners]
            ax.set_xticks(range(len(planners)))
            ax.set_xticklabels(mean_labels, fontsize=10)
            ax.set_title('风场能耗代理 E = Σ 长度·(1+顶风/v_g)³, v_g=10 m/s\n'
                         '越低越优 (同时计入绕路代价与风阻收益)', fontsize=12)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_xlabel('')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'W2_风场能耗代理对比.png'),
                        dpi=SAVE_DPI, bbox_inches='tight')
            print("已保存：W2_风场能耗代理对比.png")
            plt.close()

            # ---- 图 W3: 长度-顶风 散点 (借风收益 vs 绕路代价 trade-off) ----
            fig, ax = plt.subplots(figsize=(9, 6.5))
            for p in planners:
                sub = wdf[wdf['规划器'] == p]
                ax.scatter(sub['路径长度(m)'], sub['平均顶风分量(m/s)'], s=70,
                           alpha=0.7, label=p, color=PLANNER_COLORS[p],
                           edgecolors='black', linewidth=0.5)
                ax.scatter(sub['路径长度(m)'].mean(), sub['平均顶风分量(m/s)'].mean(),
                           marker='*', s=350, color=PLANNER_COLORS[p],
                           edgecolors='black', linewidth=1.5, zorder=5)
            ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
            ax.set_title('路径长度-平均顶风权衡 (左下方向为最优, ★为均值)\n'
                         '纵轴负值=净顺风(借风), 横轴小=少绕路', fontsize=12)
            ax.set_xlabel('路径长度 (m) ← 越小越优', fontsize=11)
            ax.set_ylabel('平均顶风分量 (m/s) ← 越低越优', fontsize=11)
            ax.grid(alpha=0.3, linestyle='--')
            ax.legend(title='规划器', frameon=False)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'W3_长度顶风权衡散点.png'),
                        dpi=SAVE_DPI, bbox_inches='tight')
            print("已保存：W3_长度顶风权衡散点.png")
            plt.close()

            # 汇总到 Excel (与主结果同工作簿追加 sheet)
            wind_sheets = [('风场描述统计', wind_desc),
                           ('风场KW检验', kw_wind_df),
                           ('风场Wilcoxon配对', wind_pair_df),
                           ('风场优胜统计', wind_win)]

    # -------------------------- 6.8 分析结论与综合评估 --------------------------
    conclusions = []

    def _best_by_median(metric):
        med = desc_stats[metric]['median']
        return med.idxmin() if metric != '最小ESDF距离(m)' else med.idxmax()

    # (1) 描述统计: 各指标中位数口径最优算法
    for metric in core_metrics:
        conclusions.append(f"[描述统计] {metric} 中位数最优: {_best_by_median(metric)}")

    # (2) 稳健统计口径
    robust_best = {}
    for m in ['路径长度(m)', '平均曲率(度/m)', '规划耗时(ms)', '最小ESDF距离(m)']:
        col = f'{m} (中位数)'
        idx = robust_df[col].idxmin() if m != '最小ESDF距离(m)' else robust_df[col].idxmax()
        robust_best[m] = robust_df.loc[idx, '规划器']
    conclusions.append("[稳健统计] 中位数口径最优: " + "，".join(f"{m}={b}" for m, b in robust_best.items()))

    # (3) 假设检验
    non_normal = assumption_df.index[assumption_df['正态性满足'] == '否'].tolist()
    conclusions.append(f"[假设检验] 不满足正态性指标: {non_normal if non_normal else '无'} → 主检验采用非参数")

    # (4) KW 全局显著指标
    sig_kw = [m for m in kw_df.index if kw_df.loc[m, 'P值'] < 0.05]
    conclusions.append(f"[KW全局] 组间显著(p<0.05)指标: {sig_kw if sig_kw else '无'}")

    # (5) 配对显著(FMM 相关)
    fmm_sig = pair_results_df[(pair_results_df['对比'].str.contains('FMM'))
                              & (pair_results_df['Bonferroni显著'] == 'sig')]
    conclusions.append(f"[配对显著] FMM 相关 Bonferroni 显著对比 {len(fmm_sig)} 项")

    # (6) 优胜次数
    win_tot = win_counts.sum(axis=1).sort_values(ascending=False)
    conclusions.append("[优胜次数] 总优胜: " + "，".join(f"{i}:{int(v)}" for i, v in win_tot.items()))

    # (7) 综合评分 (中位数代表)
    rank = score_summary['mean'].sort_values(ascending=False)
    conclusions.append("[综合评分] 排名: " + " > ".join(f"{i}({v:.3f})" for i, v in rank.items()))

    # (8) 权重敏感性
    sens_rank1 = sens_df.sort_values('排名第一频率(%)', ascending=False)
    conclusions.append("[权重敏感性] 排名第一频率: " + "，".join(
        f"{r['规划器']}={r['排名第一频率(%)']}%" for _, r in sens_rank1.iterrows()))

    # (9) 浮动稳健性
    if has_rrt_range:
        if float_kw_df is not None and len(float_kw_df):
            unstable_kw = float_kw_df[float_kw_df['显著率(%)'] < 95]['指标'].tolist()
            conclusions.append("[浮动稳健性-KW] " +
                               (f"RRT* 浮动下结论可能不稳定的指标: {unstable_kw}"
                                if unstable_kw else "全部指标 KW 显著结论在 RRT* 浮动下稳定(显著率≥95%)"))
        if float_pair_df is not None and len(float_pair_df):
            unstable_pair = float_pair_df[float_pair_df['显著率(%)'] < 95]
            conclusions.append("[浮动稳健性-配对] " +
                               ("RRT* 浮动下可能不稳定的对比: " + "，".join(
                                   f"{r['对比']}:{r['指标']}" for _, r in unstable_pair.iterrows()))
                               if len(unstable_pair) else "确定性算法 vs RRT* 配对结论在浮动下稳定")
    else:
        conclusions.append("[浮动稳健性] 未提供 rrt_fluctuation.csv, 无法检验 RRT* 波动影响")

    # (10) 综合评估结果
    best_planner = rank.index[0]
    conclusions.append(f"[综合评估-推荐] 综合得分(中位数代表)第一: {best_planner}")
    best_freq = float(sens_df.loc[sens_df['规划器'] == best_planner, '排名第一频率(%)'].iloc[0])
    conclusions.append(f"[综合评估-权重稳健] {best_planner} 在权重±20%扰动下排名第一频率: {best_freq}%")
    if has_rrt_range and score_range_df is not None and len(score_range_df):
        sr = score_range_df.set_index('规划器')
        if best_planner != 'RRT*' and 'RRT*' in sr.index:
            best_sc = float(sr.loc[best_planner, '综合得分_代表值'])
            rrt_up = float(sr.loc['RRT*', '综合得分_上限'])
            if rrt_up < best_sc:
                conclusions.append(f"[综合评估-浮动稳健] RRT* 区间最好情形得分({rrt_up:.4f})"
                                   f"仍低于 {best_planner}({best_sc:.4f}), 推荐结论对 RRT* 波动稳健")
            else:
                conclusions.append(f"[综合评估-浮动稳健] 注意: RRT* 区间最好情形得分({rrt_up:.4f})"
                                   f"可逼近 {best_planner}({best_sc:.4f}), 论述需说明确定性算法无波动")
        elif best_planner == 'RRT*':
            conclusions.append("[综合评估-浮动稳健] RRT* 综合第一但为随机算法, 其得分为区间(见'综合评分浮动'表), 需报告波动范围")

    conclusion_df = pd.DataFrame({'结论': conclusions})
    print("\n" + "=" * 50)
    print("分析结论与综合评估结果:")
    for i, c in enumerate(conclusions, 1):
        print(f"  {i}. {c}")
    print("=" * 50)

    # -------------------------- 7. 结果导出到 Excel --------------------------
    with pd.ExcelWriter(os.path.join(OUTPUT_DIR, '分析结果汇总.xlsx')) as writer:
        desc_stats.to_excel(writer, sheet_name='描述性统计')
        robust_df.to_excel(writer, sheet_name='稳健统计_中位数几何均值', index=False)
        assumption_df.to_excel(writer, sheet_name='假设检验')
        normality_detail_df.to_excel(writer, sheet_name='正态性完整表', index=False)
        kw_df.to_excel(writer, sheet_name='KruskalWallis检验_主')
        pair_results_df.to_excel(writer, sheet_name='Wilcoxon配对_主', index=False)
        pd.DataFrame(anova_results).T.to_excel(writer, sheet_name='ANOVA_对照')
        param_pair_df.to_excel(writer, sheet_name='配对t检验_对照', index=False)
        win_counts.to_excel(writer, sheet_name='优胜次数统计')
        pd.DataFrame({'指标': list(score_weights.keys()),
                      '权重': list(score_weights.values())}).to_excel(
                          writer, sheet_name='综合评分权重', index=False)
        score_summary.to_excel(writer, sheet_name='综合评分')
        contrib_summary.to_excel(writer, sheet_name='贡献分解')
        sens_df.to_excel(writer, sheet_name='权重敏感性', index=False)
        if float_kw_df is not None:
            float_kw_df.to_excel(writer, sheet_name='RRT浮动_KW稳健性', index=False)
        if float_pair_df is not None:
            float_pair_df.to_excel(writer, sheet_name='RRT浮动_配对稳健性', index=False)
        if score_range_df is not None:
            score_range_df.to_excel(writer, sheet_name='RRT浮动_综合评分区间', index=False)
        conclusion_df.to_excel(writer, sheet_name='分析结论', index=False)
        # 明细得分移除不统计的指标 (路径点数、安全/爬升违反率、累计升降、平均飞行高度)
        detail_drop = ['路径点数', '安全违反率(%)', '爬升违反率(%)', '总爬升高度(m)',
                       '总下降高度(m)', '平均飞行高度(m)']
        score_df.drop(columns=[c for c in detail_drop if c in score_df.columns],
                      inplace=True)
        score_df.to_excel(writer, sheet_name='明细得分', index=False)
        for sheet_name, sheet_df in wind_sheets:
            sheet_df.to_excel(writer, sheet_name=sheet_name)

    print("\n" + "=" * 50)
    print(f"分析完成！所有结果已统一保存到 {OUTPUT_DIR} 目录。")
    print("叙事链图表：")
    print("- 01_耗时箱线图_对数.png        [论点1] FMM 快一个数量级")
    print("- 02_耗时距离伸缩曲线.png       [论点1] 随距离绝对增量最小")
    print("- 03_长度与安全双联箱线图.png   [论点2] 质量不输 (ns 醒目标注)")
    print("- 04_平滑性双联箱线图.png       [论点3] 平均曲率/转角优势")
    print("- 05_综合评分与贡献分解.png     [论点4] 综合第一+耗时贡献来源")
    print("- 06_优胜次数统计.png           [论点4] 耗时 42/42 全胜")
    print("- 07_耗时曲率Pareto散点.png     [论点5] 双目标最优角")
    print("- 08_权重敏感性分析.png         [稳健性] 排名对权重扰动的稳定性")
    print("- A1_指标相关性热力图_附录.png  [附录]")
    print("- 分析结果汇总.xlsx             (KW/Wilcoxon 为主, ANOVA/t 对照存档)")
    if exp_name == 'wind' and wind_sheets:
        print("风场对比图表 (--exp wind 专属):")
        print("- W1_顶风与风速双联箱线图.png  [风场] 三算法借风能力 (负=净顺风)")
        print("- W2_风场能耗代理对比.png      [风场] 节能综合裁决 (含绕路代价)")
        print("- W3_长度顶风权衡散点.png      [风场] 借风收益 vs 绕路代价 trade-off")
    print("=" * 50)


if __name__ == '__main__':
    main()