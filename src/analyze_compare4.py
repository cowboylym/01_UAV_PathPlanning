import argparse
import os
import sys

import pandas as pd

from Analysis import COMPARE4_ORDER, run_paired_planner_analysis


LEGACY_NAMES = {'CA-FMM', 'FMM-CA', 'FMM-CA-NS'}
KEY = ['实验编号', '规划器']
COORDS = ['起点X', '起点Y', '起点Z', '终点X', '终点Y', '终点Z']
REQUIRED_COLUMNS = set(KEY + COORDS + ['规划状态', '后处理模式'])


def latest_result_dir(compare4_root):
    candidates = []
    if os.path.isdir(compare4_root):
        for name in os.listdir(compare4_root):
            path = os.path.join(compare4_root, name)
            if (os.path.isdir(path) and
                    os.path.isfile(os.path.join(path, 'raw', 'path_comparison.csv')) and
                    os.path.isfile(os.path.join(path, 'smoothed', 'path_comparison.csv'))):
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f'没有找到同时包含 raw/smoothed 的完整批次: {compare4_root}')
    return max(candidates, key=os.path.getmtime)


def read_mode(csv_path, expected_mode):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'找不到实验结果: {csv_path}')
    data = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f'{csv_path} 缺少必要列: {sorted(missing)}')
    data['规划器'] = data['规划器'].astype(str).str.strip()
    old = sorted(set(data['规划器']) & LEGACY_NAMES)
    if old:
        raise ValueError(f'{csv_path} 含旧算法名 {old}；旧 CA-FMM 数据不是 FGDA*，禁止混用')
    unknown = sorted(set(data['规划器']) - set(COMPARE4_ORDER))
    if unknown:
        raise ValueError(f'{csv_path} 含非 compare4 算法: {unknown}')
    duplicated = data.duplicated(KEY, keep=False)
    if duplicated.any():
        rows = data.loc[duplicated, KEY].drop_duplicates()
        raise ValueError('存在重复的“实验编号-规划器”记录:\n' + rows.to_string(index=False))
    modes = set(data['后处理模式'].astype(str).str.strip())
    if modes != {expected_mode}:
        raise ValueError(f'{csv_path} 后处理模式应仅为 {expected_mode}，实际为 {sorted(modes)}')
    for col in COORDS:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    for experiment, rows in data.groupby('实验编号'):
        for col in COORDS:
            values = rows[col].dropna()
            if values.empty or values.max() - values.min() > 1e-6:
                raise ValueError(f'{csv_path} 实验 {experiment} 的 {col} 在算法间不一致')
    data['规划状态'] = data['规划状态'].astype(str).str.strip()
    return data


def complete_success(data, mode):
    success = data[data['规划状态'] == '成功'].copy()
    sets = success.groupby('实验编号')['规划器'].agg(set)
    required = set(COMPARE4_ORDER)
    ids = sorted(sets[sets.apply(lambda value: value == required)].index)
    paired = success[(success['实验编号'].isin(ids)) &
                     (success['规划器'].isin(COMPARE4_ORDER))].copy()
    order = {name: index for index, name in enumerate(COMPARE4_ORDER)}
    paired['_order'] = paired['规划器'].map(order)
    paired = paired.sort_values(['实验编号', '_order']).drop(columns='_order')
    if len(ids) < 2:
        raise ValueError(f'{mode} 四算法完整成功配对仅 {len(ids)} 组，至少需要 2 组')
    return paired, ids


def success_rates(raw, smoothed):
    rows = []
    for mode, data in [('raw', raw), ('smoothed', smoothed)]:
        total_ids = data['实验编号'].nunique()
        for planner in COMPARE4_ORDER:
            part = data[data['规划器'] == planner]
            successful = part.loc[part['规划状态'] == '成功', '实验编号'].nunique()
            rows.append({'后处理模式': mode, '规划器': planner,
                         '全部场景数': total_ids, '成功场景数': successful,
                         '成功率(%)': 100.0 * successful / total_ids if total_ids else float('nan')})
    return pd.DataFrame(rows)


def main():
    default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'compare4'))
    parser = argparse.ArgumentParser(description='FGDA*、FMM、A*、RRT* 的 raw/smoothed 配对分析')
    parser.add_argument('result_dir', nargs='?', help='批次根目录；省略时选取 data/compare4 最新完整批次')
    args = parser.parse_args()
    result_dir = os.path.abspath(args.result_dir) if args.result_dir else latest_result_dir(default_root)
    output_dir = os.path.join(result_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    raw = read_mode(os.path.join(result_dir, 'raw', 'path_comparison.csv'), 'raw')
    smoothed = read_mode(os.path.join(result_dir, 'smoothed', 'path_comparison.csv'), 'smoothed')
    raw_paired, raw_ids = complete_success(raw, 'raw')
    smoothed_paired, smoothed_ids = complete_success(smoothed, 'smoothed')
    common_ids = sorted(set(raw_ids) & set(smoothed_ids))
    if len(common_ids) < 2:
        raise ValueError(f'raw/smoothed 共同完整成功实验仅 {len(common_ids)} 组，至少需要 2 组')
    common = pd.concat([
        raw_paired[raw_paired['实验编号'].isin(common_ids)],
        smoothed_paired[smoothed_paired['实验编号'].isin(common_ids)]
    ], ignore_index=True)

    raw_paired.to_csv(os.path.join(output_dir, 'raw_四算法完整成功配对.csv'), index=False, encoding='utf-8-sig')
    smoothed_paired.to_csv(os.path.join(output_dir, 'smoothed_四算法完整成功配对.csv'), index=False, encoding='utf-8-sig')
    common.to_csv(os.path.join(output_dir, 'raw_smoothed共同实验.csv'), index=False, encoding='utf-8-sig')
    rates = success_rates(raw, smoothed)
    rates.to_csv(os.path.join(output_dir, '全部场景成功率.csv'), index=False, encoding='utf-8-sig')

    run_paired_planner_analysis(
        raw_paired,
        os.path.join(output_dir, 'raw'),
        COMPARE4_ORDER,
        'raw 四算法',
        '规划器',
        single_metric_plots=True,
        excluded_plot_metrics={'路径伸缩率', '高度方差'}
    )
    run_paired_planner_analysis(smoothed_paired, os.path.join(output_dir, 'smoothed'),
                                COMPARE4_ORDER, 'smoothed 四算法', '规划器')
    effect = common.rename(columns={'后处理模式': '实验组'})
    run_paired_planner_analysis(effect, os.path.join(output_dir, 'postprocess_effect'),
                                ['raw', 'smoothed'], '各算法后处理变化', '实验组',
                                split_column='规划器')
    print(f'[完成] 批次: {result_dir}')
    print(f'[配对] raw={len(raw_ids)}，smoothed={len(smoothed_ids)}，共同={len(common_ids)}')


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f'[错误] {error}', file=sys.stderr)
        raise SystemExit(1)
