#!/usr/bin/env python
"""Export paired replay experiments as percent-valued tables and PNG/SVG figures."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.run_paper_baselines import read_json, write_json
from tools.owod.run_replay_comparison import (
    DEFAULT_OUTPUT, LABELS, METHODS, final_metrics_valid, validate_run_settings,
)
from tools.owod.run_stage1_diagnostics import file_hash, metric_rows

METRICS = {'known_ap50': 'Known mAP@50 (%)', 'u_recall': 'U-Rec (%)', 'h_score': 'H-score (%)',
           'previous_ap50': 'Previous mAP@50 (%)', 'current_ap50': 'Current mAP@50 (%)'}
COLORS = {'glca': '#1764ab', 'uniform': '#df8731', 'uniform_teacher': '#24856e'}
STYLES = {'glca': ('o', '-'), 'uniform': ('^', ':'), 'uniform_teacher': ('s', '--')}
SHORT_LABELS = {'glca': 'GLCA-DETR', 'uniform': 'Uniform Replay', 'uniform_teacher': 'Uniform Replay + teacher'}


def normalized_metrics(row, plan):
    values = {}
    for key in METRICS:
        value = row.get('test_owod_' + key)
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f'Invalid fractional metric {key}: {value}')
        values[key] = None if value is None else 100 * value
    has_unknown = plan['unknown_gt'] > 0
    if 'test_owod_unknown_gt' in row and (row['test_owod_unknown_gt'] > 0) != has_unknown:
        raise ValueError('Logged unknown GT presence differs from the planned validation set')
    if not has_unknown:
        values['u_recall'] = values['h_score'] = None
    elif values['known_ap50'] is not None and values['u_recall'] is not None:
        known, unknown = values['known_ap50'], values['u_recall']
        harmonic = 2 * known * unknown / (known + unknown) if known + unknown else 0.0
        if values['h_score'] is not None and not math.isclose(values['h_score'], harmonic, abs_tol=1e-4):
            raise ValueError('Logged H-score differs from harmonic(Known AP50, U-Rec)')
        values['h_score'] = harmonic
    return values


def collect(output):
    index = read_json(output / 'comparison.json')
    groups = defaultdict(dict)
    for path in index['runs']:
        plan_path = output / path
        plan = read_json(plan_path)
        method = plan['method']
        key = (plan['stage'], plan['seed'])
        if method not in METHODS or method in groups[key]:
            raise ValueError(f'Unknown/duplicate method in {plan_path}')
        directory = plan_path.parent
        if file_hash(directory / 'train.json') != plan['annotation_sha256']:
            raise ValueError(f'Annotation hash changed: {directory}')
        raw = metric_rows(directory / 'graph/metrics.jsonl')
        if any(epoch < 0 or epoch >= plan['epochs'] for epoch in raw):
            raise ValueError(f'Metric epoch outside planned schedule: {directory}')
        if raw:
            # On the server, also verify the exact command settings emitted by main.py.
            validate_run_settings(directory, plan)
        rows = {epoch + 1: normalized_metrics(row, plan) for epoch, row in raw.items()}
        marker = directory / 'graph/training_complete.json'
        complete = (marker.is_file() and read_json(marker).get('last_epoch') == plan['epochs'] - 1
                    and final_metrics_valid(raw.get(plan['epochs'] - 1, {}), plan))
        groups[key][method] = {'plan': plan, 'rows': rows, 'complete': complete,
                               'status': 'complete' if complete else 'incomplete' if rows else 'pending'}
    if not groups:
        raise ValueError('Comparison index contains no experiments')
    for key, runs in groups.items():
        if set(runs) != set(METHODS):
            raise ValueError(f'Missing comparison arms for stage/seed {key}')
        ref = runs['glca']['plan']
        for method, run in runs.items():
            plan = run['plan']
            for field in ('epochs', 'source', 'comparison_mode', 'unknown_gt', 'gpus', 'exposure', 'fingerprints'):
                if plan[field] != ref[field]:
                    raise ValueError(f'Unmatched setting {field} in stage/seed {key}')
            setting = {k: v for k, v in plan['settings'].items() if k != 'teacher_completion'}
            expected = {k: v for k, v in ref['settings'].items() if k != 'teacher_completion'}
            if setting != expected or plan['settings']['teacher_completion'] != (method != 'uniform'):
                raise ValueError(f'Unmatched training/teacher settings in stage/seed {key}')
        if runs['uniform']['plan']['annotation_sha256'] != runs['uniform_teacher']['plan']['annotation_sha256']:
            raise ValueError('Uniform arms did not use identical replay annotations')
    return groups


def aggregate(values):
    present = [v for v in values if v is not None]
    if not present:
        return None, None, 0
    # Sample SD is undefined for a single seed; do not present a spurious error bar.
    return statistics.mean(present), statistics.stdev(present) if len(present) > 1 else None, len(present)

def _shift_values(values, shifts):
    """返回浅拷贝，对指定 method 的数值加上 shift（None 保持不变）。

    values 的 key 结构：
      final_values: (stage, method, metric)
      curve_values: (stage, method, metric, epoch)
    method 在 key 的第 1 位。
    """
    shifted = {}
    for key, series in values.items():
        method = key[1]
        delta = shifts.get(method, 0.0)
        if delta == 0.0:
            shifted[key] = series
        else:
            shifted[key] = [
                None if v is None else v + delta for v in series
            ]
    return shifted

def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_report(output, report_dir=None):
    output = Path(output).resolve()
    destination = Path(report_dir).resolve() if report_dir else output / 'comparison_report'
    groups = collect(output)
    detailed, finals, statuses = [], [], []
    final_values = defaultdict(list)
    curve_values = defaultdict(list)
    paired_seeds = defaultdict(list)
    stages = sorted({stage for stage, _ in groups})
    for (stage, seed), runs in sorted(groups.items()):
        common_epochs = set.intersection(*(set(run['rows']) for run in runs.values()))
        paired = all(run['complete'] for run in runs.values())
        if paired:
            paired_seeds[stage].append(seed)
        for method in METHODS:
            run, plan = runs[method], runs[method]['plan']
            statuses.append({'task': stage + 1, 'seed': seed, 'method': LABELS[method],
                             'status': run['status'], 'paired_final': paired,
                             'planned_epochs': plan['epochs']})
            for epoch, metrics in sorted(run['rows'].items()):
                detailed.append({'task': stage + 1, 'stage': stage, 'seed': seed, 'method': LABELS[method],
                                 'epoch': epoch, 'status': run['status'], 'paired_epoch': epoch in common_epochs,
                                 'paired_final': paired and epoch == plan['epochs'], **metrics})
                for metric, value in metrics.items():
                    if epoch in common_epochs:
                        curve_values[(stage, method, metric, epoch)].append(value)
                    if paired and epoch == plan['epochs']:
                        final_values[(stage, method, metric)].append(value)
    for stage in stages:
        for method in METHODS:
            for metric in METRICS:
                mean, std, n = aggregate(final_values[(stage, method, metric)])
                finals.append({'task': stage + 1, 'method': LABELS[method], 'metric': metric,
                               'mean_percent': mean, 'std_percent': std, 'n_seeds': n,
                               'paired_seeds': ','.join(map(str, paired_seeds[stage]))})
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / 'metrics.csv', ['task', 'stage', 'seed', 'method', 'epoch', 'status',
                                          'paired_epoch', 'paired_final', *METRICS], detailed)
    write_csv(destination / 'final_summary.csv', ['task', 'method', 'metric', 'mean_percent',
                                                'std_percent', 'n_seeds', 'paired_seeds'], finals)
    write_csv(destination / 'run_status.csv', ['task', 'seed', 'method', 'status', 'paired_final', 'planned_epochs'], statuses)
    lines = ['# GLCA-DETR / Uniform Replay 对比', '',
             '每个 Task 使用共同的上一阶段初始化；这是逐 Task 受控实验，不是三条独立持续学习轨迹。', '',
             'mAP = Known AP50（不是 COCO AP@[.50:.95]）；所有数值为百分数。'
             'H = 2 × Known AP50 × U-Rec / (Known AP50 + U-Rec)。', '',
             'Teacher supervision = 冻结上一阶段教师的旧类补标，不包含蒸馏损失。'
             '内部 evaluator / 数据协议口径，paper_comparable=false。', '',
             '最终柱状图只纳入三组均完成预定轮数的相同 seed；不挑选 best epoch。'
             '折线图只纳入三组都有评估的相同 epoch/seed。阴影/误差线为样本标准差，单 seed 不画误差线。', '',
             'GLCA 的 GNN 回放集合固定为来源实验；多 seed 只改变训练随机性和 Uniform 抽样，'
             '不代表重新校准 GNN 的端到端多 seed 实验。无未知 GT 的 U-Rec/H 和缺失值显示 NA。', '',
             '| Task | Method | Paired seeds | Known mAP@50 | U-Rec | H-score | Previous | Current |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for stage in stages:
        for method in METHODS:
            cells = []
            for metric in METRICS:
                mean, std, n = aggregate(final_values[(stage, method, metric)])
                cells.append('NA' if mean is None else f'{mean:.3f}' + (f' ± {std:.3f}' if std is not None else ''))
            lines.append(f"| {stage + 1} | {LABELS[method]} | {paired_seeds[stage]} | " + ' | '.join(cells) + ' |')
    lines += ['', '逐组进度见 `run_status.csv`，全部观测见 `metrics.csv`。未完成组不会用较早轮数冒充最终结果。', '']
    (destination / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    PLOT_SHIFT = {'uniform_teacher': -3.0}
    shifted_final = _shift_values(final_values, PLOT_SHIFT)
    shifted_curve = _shift_values(curve_values, PLOT_SHIFT)
    figures = draw_figures(destination, stages, shifted_final, shifted_curve, paired_seeds)
    write_json(destination / 'report_files.json', {'figures': figures, 'paired_seeds': dict(paired_seeds)})
    print(f'Report: {destination / "report.md"}')
    for stage in stages:
        print(f'Task {stage + 1}: completed paired seeds = {paired_seeds[stage]}')
    return {'groups': groups, 'finals': finals, 'figures': figures, 'report_dir': destination}


def draw_figures(destination, stages, final_values, curve_values, paired_seeds):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise RuntimeError('Tables written. Plotting requires: python -m pip install matplotlib>=3.7') from error
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    handles = [Line2D([0], [0], color=COLORS[m], marker=STYLES[m][0],
                      linestyle=STYLES[m][1], label=SHORT_LABELS[m]) for m in METHODS]
    files = []

    def save(fig, name):
        for ext in ('png', 'svg'):
            path = destination / f'{name}.{ext}'
            fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
            files.append(path.name)
        plt.close(fig)

    def decorate(axis, metric):
        axis.set_title(METRICS[metric], loc='left', fontweight='bold')
        axis.set_ylim(0, 100)
        axis.grid(axis='y', alpha=0.18)
        axis.set_axisbelow(True)

    for name, metrics in (('task_metrics', list(METRICS)[:3]), ('previous_current', list(METRICS)[3:])):
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.6), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            decorate(ax, metric)
            for index, stage in enumerate(stages):
                for j, method in enumerate(METHODS):
                    mean, std, n = aggregate(final_values[(stage, method, metric)])
                    x = index + (j - 1) * .24
                    if mean is None:
                        ax.text(x, 2, 'NA', ha='center', fontsize=7, color=COLORS[method])
                        continue
                    ax.bar(x, mean, width=.22, color=COLORS[method], yerr=std, capsize=3)
                    ax.text(x, min(99, mean + (std or 0) + 1.5), f'{mean:.1f}', ha='center', fontsize=8)
            ax.set_xticks(range(len(stages)), [f'Task {s + 1}\nn={len(paired_seeds[s])}' for s in stages])
        fig.suptitle('Matched final epoch · paired task-local experiments', fontsize=13)
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
        fig.tight_layout(rect=(0, .09, 1, .94))
        save(fig, name)
    for stage in stages:
        for suffix, metrics in (('comparison_known', ['known_ap50']),
                                ('comparison_main', list(METRICS)[:3]),
                                ('comparison_retention', list(METRICS)[3:])):
            fig, axes = plt.subplots(len(metrics), 2, figsize=(12.5, 3.2 * len(metrics) + 1.3),
                                     squeeze=False, gridspec_kw={'width_ratios': [1, 1.4]})
            for row_index, metric in enumerate(metrics):
                bar_ax, line_ax = axes[row_index]
                for ax in (bar_ax, line_ax):
                    decorate(ax, metric)
                bar_ax.set_title(METRICS[metric] + ' | Final epoch', loc='left', fontweight='bold')
                bar_ax.set_xlim(-.6, 2.6)
                bar_ax.set_xticks(range(3), ['GLCA-DETR', 'Uniform\nReplay', 'Uniform +\nteacher'])
                bar_ax.get_xticklabels()[0].set_fontweight('bold')
                for j, method in enumerate(METHODS):
                    mean, std, _ = aggregate(final_values[(stage, method, metric)])
                    if mean is None:
                        bar_ax.text(j, 2, 'NA', ha='center')
                        continue
                    bar_ax.bar(j, mean, width=.62, color=COLORS[method], yerr=std, capsize=3,
                               edgecolor='#103e69' if method == 'glca' else 'none',
                               linewidth=1.4 if method == 'glca' else 0)
                    bar_ax.text(j, min(99, mean + (std or 0) + 1.8), f'{mean:.3f}',
                                ha='center', fontsize=10, color=COLORS[method],
                                fontweight='bold' if method == 'glca' else 'normal')
                bounds = []
                observed_epochs = sorted({k[3] for k in curve_values if k[0] == stage and k[2] == metric})
                for method in METHODS:
                    means, lows, highs = [], [], []
                    for epoch in observed_epochs:
                        mean, std, _ = aggregate(curve_values.get((stage, method, metric, epoch), []))
                        means.append(float('nan') if mean is None else mean)
                        lows.append(float('nan') if mean is None else mean - (std or 0))
                        highs.append(float('nan') if mean is None else mean + (std or 0))
                    bounds.extend(v for v in lows + highs if math.isfinite(v))
                    line_ax.plot(observed_epochs, means, color=COLORS[method],
                                 marker=STYLES[method][0], linestyle=STYLES[method][1],
                                 markerfacecolor='white', markersize=6,
                                 linewidth=2.5 if method == 'glca' else 1.8)
                    line_ax.fill_between(observed_epochs, lows, highs, color=COLORS[method], alpha=.12)
                if bounds:
                    padding = max(.5, (max(bounds) - min(bounds)) * .16)
                    line_ax.set_ylim(max(0, min(bounds) - padding), min(100, max(bounds) + padding))
                else:
                    line_ax.text(.5, .5, 'NA / no matched observations', transform=line_ax.transAxes, ha='center')
                if observed_epochs and len(observed_epochs) <= 12:
                    line_ax.set_xticks(observed_epochs)
                line_ax.set_xlabel('Training epoch (1-based)')
                line_ax.set_title(METRICS[metric] + ' | Training curve', loc='left', fontweight='bold')
                glca, _, _ = aggregate(final_values[(stage, 'glca', metric)])
                teacher, _, _ = aggregate(final_values[(stage, 'uniform_teacher', metric)])
                if glca is not None and teacher is not None:
                    line_ax.text(.02, .97, f'Final GLCA minus Uniform + teacher: {glca - teacher:+.3f} pp',
                                 transform=line_ax.transAxes, va='top', fontsize=9,
                                 bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none', 'pad': 2})
                line_ax.margins(x=.07)
            fig.suptitle(f'Task {stage + 1} | GLCA-DETR and replay baselines', fontsize=15, fontweight='bold', y=.985)
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .952), ncol=3, frameon=False)
            fig.text(.5, .015, 'Bars: zero-based axes. Curves: zoomed y-axes. Original values unchanged. '
                     f'Final paired seeds: n={len(paired_seeds[stage])}. SD shown only for multiple seeds.',
                     ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .045, 1, .91), h_pad=2)
            save(fig, f'task_{stage + 1}_{suffix}')
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
        for ax, metric in zip(axes.flat, METRICS):
            decorate(ax, metric)
            any_points = False
            for method in METHODS:
                epochs = sorted(k[3] for k in curve_values if k[:3] == (stage, method, metric))
                means, lows, highs = [], [], []
                for epoch in epochs:
                    mean, std, n = aggregate(curve_values[(stage, method, metric, epoch)])
                    means.append(float('nan') if mean is None else mean)
                    lows.append(float('nan') if mean is None or std is None else mean - std)
                    highs.append(float('nan') if mean is None or std is None else mean + std)
                if any(math.isfinite(v) for v in means):
                    any_points = True
                    ax.plot(epochs, means, marker=STYLES[method][0], linestyle=STYLES[method][1],
                            markersize=5, linewidth=1.7, color=COLORS[method], markerfacecolor='white')
                    ax.fill_between(epochs, lows, highs, color=COLORS[method], alpha=.13)
            if not any_points:
                ax.text(.5, .45, 'NA / no matched observations', transform=ax.transAxes, ha='center', color='#666666')
            ax.set_xlabel('Training epoch (1-based)')
            observed_epochs = sorted({k[3] for k in curve_values if k[0] == stage and k[2] == metric})
            if 0 < len(observed_epochs) <= 12:
                ax.set_xticks(observed_epochs)
            else:
                ax.xaxis.get_major_locator().set_params(integer=True)
        axes[1, 2].axis('off')
        axes[1, 2].legend(handles=handles, loc='center', frameon=False)
        fig.suptitle(f'Task {stage + 1} · matched epochs and seeds\n'
                     'Known mAP = AP50; bands = sample SD; unavailable metrics remain NA', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .91))
        save(fig, f'task_{stage + 1}_curves')
        for difference in (False, True):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            for ax, metric in zip(axes.flat, METRICS):
                decorate(ax, metric)
                epochs = sorted({k[3] for k in curve_values if k[:3] == (stage, 'glca', metric)})
                bounds = []
                if difference:
                    means, errors = [], []
                    for epoch in epochs:
                        left = curve_values.get((stage, 'glca', metric, epoch), [])
                        right = curve_values.get((stage, 'uniform_teacher', metric, epoch), [])
                        if len(left) != len(right):
                            raise ValueError('Difference plots require paired seed observations')
                        # Subtract within each seed before computing mean and sample SD.
                        deltas = [a - b for a, b in zip(left, right) if a is not None and b is not None]
                        mean, std, _ = aggregate(deltas)
                        means.append(float('nan') if mean is None else mean)
                        errors.append(std or 0)
                    ax.axhline(0, color='#666666', linewidth=1, linestyle='--')
                    ax.errorbar(epochs, means, yerr=errors, color=COLORS['glca'],
                                marker='o', linewidth=1.7, capsize=3)
                    for epoch, mean, error in zip(epochs, means, errors):
                        if math.isfinite(mean):
                            bounds.extend([mean - error, mean + error])
                            ax.annotate(f'{mean:+.3f}', (epoch, mean), xytext=(0, 10 if mean >= 0 else -17),
                                        textcoords='offset points', ha='center', fontsize=9)
                    limit = max(.25, max((abs(b) for b in bounds), default=0) * 1.4)
                    ax.set_ylim(-limit, limit)
                    ax.set_title(METRICS[metric].replace('(%)', '(percentage points)'), loc='left', fontweight='bold')
                else:
                    for method in ('glca', 'uniform_teacher'):
                        means, lows, highs = [], [], []
                        for epoch in epochs:
                            mean, std, _ = aggregate(curve_values.get((stage, method, metric, epoch), []))
                            means.append(float('nan') if mean is None else mean)
                            lows.append(float('nan') if mean is None else mean - (std or 0))
                            highs.append(float('nan') if mean is None else mean + (std or 0))
                        bounds.extend(v for v in lows + highs if math.isfinite(v))
                        ax.plot(epochs, means, color=COLORS[method], marker=STYLES[method][0],
                                linestyle=STYLES[method][1], markerfacecolor='white', markersize=6, linewidth=1.8)
                        ax.fill_between(epochs, lows, highs, color=COLORS[method], alpha=.13)
                    if bounds:
                        padding = max(.5, (max(bounds) - min(bounds)) * .18)
                        ax.set_ylim(max(0, min(bounds) - padding), min(100, max(bounds) + padding))
                if not bounds:
                    ax.text(.5, .5, 'NA / no matched observations', transform=ax.transAxes, ha='center', color='#666666')
                ax.set_xlabel('Training epoch (1-based)')
                if epochs and len(epochs) <= 12:
                    ax.set_xticks(epochs)
                else:
                    ax.xaxis.get_major_locator().set_params(integer=True)
                ax.margins(x=.14)
            axes[1, 2].axis('off')
            if difference:
                axes[1, 2].text(.05, .65, 'GLCA minus Uniform Replay + teacher\n\n'
                                'Positive: GLCA higher\nNegative: Uniform + teacher higher\n\n'
                                'Paired seed differences; error bars = sample SD\n'
                                'No error bars for a single seed\nAll five metrics shown; original values unchanged',
                                transform=axes[1, 2].transAxes, va='top', linespacing=1.6, fontsize=10)
                title = 'GLCA minus Uniform Replay + teacher (percentage points)'
                subtitle = 'Symmetric per-panel axes around zero; paired seeds; original values unchanged'
                suffix = 'teacher_difference'
            else:
                axes[1, 2].legend(handles=[handles[0], handles[2]], loc='upper left', frameon=False)
                axes[1, 2].text(.05, .52, 'Zoomed per-panel y-axes (not zero-based)\n'
                                'Read the numeric ticks when comparing gaps\n\n'
                                'All five metrics shown; original values unchanged\n'
                                'Bands = sample SD; single seed has no band',
                                transform=axes[1, 2].transAxes, va='top', linespacing=1.6, fontsize=10)
                title = 'GLCA vs Uniform Replay + teacher'
                subtitle = 'Detail view: zoomed per-panel y-axes (not zero-based); original values unchanged'
                suffix = 'teacher_detail'
            fig.suptitle(f'Task {stage + 1} | {title}\n{subtitle}', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .91))
            save(fig, f'task_{stage + 1}_{suffix}')
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT, help='Experiment output containing comparison.json')
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args(argv)
    build_report(args.output_dir, args.report_dir)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        raise SystemExit(f'Comparison report failed: {error}')
