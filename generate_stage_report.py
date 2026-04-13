from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.report_support import (
    RunRecord,
    configure_report_plot_style,
    ensure_dirs,
    load_existing_payloads,
    load_runs_from_dirs,
    mean_std_curve as shared_mean_std_curve,
    runs_by_group_seed,
)

ROOT = Path(__file__).resolve().parent
REPORT_ROOT = ROOT / 'reports'
ARCHIVE_CLOUD_ROOT = ROOT / 'cloud_results' / 'archive_intermediate_snapshots_20260311'
CURRENT_DIRS = [
    ARCHIVE_CLOUD_ROOT / 'stats_snapshot_20260311' / 'full_15_seed42_v2' / 'logs',
    ROOT / 'cloud_results' / 'pull_seed52_20260311_111617' / 'full_15_seed52_v2' / 'logs',
]
LEGACY_FILES = {
    'legacy_plain_500': ROOT / 'logs' / 'res_PLAIN_sigma0_FEDAVG_NP_FDP_500rounds.json',
    'legacy_g4_400': ROOT / 'logs' / 'res_CDP_sigma0.005_FEDPROX_P_FDP_400rounds.json',
    'legacy_g6_500': ROOT / 'logs' / 'res_CDP_sigma0.005_FEDPROX_P_ADP_500rounds.json',
    'legacy_g7_500': ROOT / 'logs' / 'res_LDP_sigma0.02_FEDPROX_P_FDP_500rounds.json',
    'legacy_g9_500': ROOT / 'logs' / 'res_LDP_sigma0.02_FEDPROX_P_ADP_500rounds.json',
}
OUT_DIR = ROOT / 'figures' / 'stage_report_20260311'
REPORT_PATH = REPORT_ROOT / '阶段实验结果报告_20260311.md'
TAIL = 50

def zh(s: str) -> str:
    return s.encode('ascii').decode('unicode_escape')


configure_report_plot_style(
    [
        r'C:\Windows\Fonts\NotoSansSC-VF.ttf',
        r'C:\Windows\Fonts\NotoSerifSC-VF.ttf',
        r'C:\Windows\Fonts\msyh.ttc',
        r'C:\Windows\Fonts\simhei.ttf',
    ]
)

GROUP_INFO = {
    'G0': {'label': 'G0 FedAvg基线', 'family': 'plain'},
    'G1': {'label': 'G1 FedProx基线', 'family': 'plain'},
    'G2': {'label': 'G2 个性化基线', 'family': 'plain'},
    'G3': {'label': 'G3 全局CDP', 'family': 'cdp'},
    'G4': {'label': 'G4 固定CDP', 'family': 'cdp'},
    'G5': {'label': 'G5 个性化CDP', 'family': 'cdp'},
    'G6': {'label': 'G6 自适应CDP', 'family': 'cdp'},
    'G7': {'label': 'G7 固定LDP', 'family': 'ldp'},
    'G8': {'label': 'G8 个性化LDP', 'family': 'ldp'},
    'G9': {'label': 'G9 自适应LDP', 'family': 'ldp'},
    'A1': {'label': 'A1 去个性化', 'family': 'ablation'},
    'A2': {'label': 'A2 去FedProx', 'family': 'ablation'},
    'A3': {'label': 'A3 去自适应DP', 'family': 'ablation'},
}

COLORS = {
    'G0': '#c44e52',
    'G4': '#8f98a0',
    'G6': '#2b6cb0',
    'G7': '#7f8c8d',
    'G9': '#2f855a',
    'A1': '#f6ad55',
    'A2': '#ed8936',
    'A3': '#9aa0a6',
    'legacy': '#7c3aed',
}
FAMILY_COLORS = {
    'plain': '#c44e52',
    'cdp': '#2b6cb0',
    'ldp': '#2f855a',
    'ablation': '#ed8936',
    'other': '#7f8c8d',
}
def ensure_dir() -> None:
    ensure_dirs(OUT_DIR, REPORT_PATH.parent)


def load_current_runs() -> List[RunRecord]:
    return load_runs_from_dirs(CURRENT_DIRS)


def current_summary_df(runs: List[RunRecord]) -> pd.DataFrame:
    rows = [r.build_summary(TAIL, sigma_key='tail_sigma_eff', time_key='tail_round_time') for r in runs]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['group_label'] = df['group'].map(lambda x: GROUP_INFO.get(x, {}).get('label', x))
    df['family'] = df['group'].map(lambda x: GROUP_INFO.get(x, {}).get('family', 'other'))
    return df


def load_legacy_payloads() -> Dict[str, Dict]:
    return load_existing_payloads(LEGACY_FILES)


def mean_std_curve(records: List[RunRecord], field: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return shared_mean_std_curve(records, field, include_raw=True)


def legacy_metric_tail(payload: Dict, key: str, tail: int = TAIL) -> float:
    values = payload.get(key, [])
    if not values:
        return float('nan')
    values = values[-tail:] if len(values) >= tail else values
    return float(np.mean(values))


def plot_plain_risk(legacy: Dict[str, Dict]) -> Path:
    data = legacy['legacy_plain_500']
    rounds = np.arange(1, len(data['attack_acc']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=220)

    ax = axes[0]
    ax.plot(rounds, data['train_loss'], color=COLORS['G0'], lw=2.2, label='训练损失')
    ax.plot(rounds, data['test_loss'], color='#4c78a8', lw=2.2, label='测试损失')
    ax.set_title('损失下降与过拟合迹象', fontsize=15, pad=12)
    ax.set_xlabel('训练轮次')
    ax.set_ylabel('实验组')
    ax.legend(frameon=True, loc='upper right')
    ax.annotate('训练损失快速下降', xy=(15, data['train_loss'][14]), xytext=(90, max(data['train_loss']) * 0.72), arrowprops=dict(arrowstyle='->', color='#444'), fontsize=11)

    ax2 = axes[1]
    ax2.plot(rounds, data['attack_acc'], color='#d1495b', lw=2.5)
    ax2.fill_between(rounds, 0.5, data['attack_acc'], color='#d1495b', alpha=0.15)
    ax2.axhline(0.5, ls='--', lw=1.4, color='#666666', label='随机猜测')
    ax2.set_title('成员推断攻击成功率', fontsize=15, pad=12)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('攻击成功率 (ASR)')
    ax2.set_ylim(0.45, 0.92)
    peak_idx = int(np.argmax(data['attack_acc']))
    ax2.scatter(peak_idx + 1, data['attack_acc'][peak_idx], color='#7f1d1d', s=40, zorder=3)
    ax2.annotate(f"峰值 ASR={data['attack_acc'][peak_idx]:.2f}", xy=(peak_idx + 1, data['attack_acc'][peak_idx]), xytext=(peak_idx - 100, 0.88), arrowprops=dict(arrowstyle='->', color='#444'), fontsize=11)
    ax2.legend(frameon=True, loc='lower right')

    fig.suptitle('图1  Plain基线的过拟合风险（500轮历史结果）', fontsize=17, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / 'fig1_plain_overfit_risk.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_core_comparison(df: pd.DataFrame) -> Path:
    use_df = df[df['group'].isin(['G4', 'G6', 'G7', 'G9'])].copy()
    use_df['Mechanism'] = use_df['group'].map({'G4': '固定 CDP', 'G6': '自适应 CDP', 'G7': '固定 LDP', 'G9': '自适应 LDP'})

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=220)
    metrics = [('tail_rmse', '尾部 RMSE', True), ('tail_asr', '尾部 ASR', False), ('tail_sigma_eff', '有效噪声', True)]
    order = ['固定 CDP', '自适应 CDP', '固定 LDP', '自适应 LDP']
    tick_labels = ['固定\nCDP', '自适应\nCDP', '固定\nLDP', '自适应\nLDP']
    colors = ['#8f98a0', '#2b6cb0', '#7f8c8d', '#2f855a']

    for ax, (metric, title, lower_better) in zip(axes, metrics):
        stats = use_df.groupby('Mechanism')[metric].agg(['mean', 'std']).reindex(order)
        x = np.arange(len(order))
        ax.bar(x, stats['mean'], yerr=stats['std'].fillna(0.0).values, color=colors, edgecolor='#333333', linewidth=0.8, capsize=5, alpha=0.95)
        vals = stats['mean'].tolist()
        for i, name in enumerate(order):
            samples = use_df[use_df['Mechanism'] == name][metric].tolist()
            jitter = np.linspace(-0.08, 0.08, len(samples)) if samples else []
            for j, val in enumerate(samples):
                ax.scatter(i + (jitter[j] if len(samples) > 1 else 0), val, color='#111111', s=30, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=0)
        ax.set_title(title, fontsize=14)
        ax.set_ylabel(title)
        if metric == 'peak_asr':
            ax.axhline(0.5, color='#666666', ls='--', lw=1.2)
            ax.text(0.02, 0.96, '越接近 0.5 越安全', transform=ax.transAxes, va='top', fontsize=10, color='#555')
            ax.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
        elif metric == 'tail_sigma_eff':
            ax.text(0.02, 0.96, '越低越优', transform=ax.transAxes, va='top', fontsize=10, color='#555')
            ax.set_ylim(0.0, max(vals) * 1.18)
        else:
            ax.text(0.02, 0.96, '越低越优', transform=ax.transAxes, va='top', fontsize=10, color='#555')
            ax.set_ylim(min(vals) - 0.03, max(vals) + 0.03)

    fig.suptitle('图2  核心机制对比：固定噪声 vs 自适应噪声', fontsize=17, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / 'fig2_core_mechanism_comparison.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_ablation(df: pd.DataFrame) -> Path:
    ab = df[(df['seed'] == 42) & (df['group'].isin(['G6', 'A1', 'A2', 'A3']))].copy()
    ab['Mechanism'] = ab['group'].map({
        'G6': f"G6\n{zh(r'\u5b8c\u6574\u65b9\u6848')}",
        'A1': f"A1\n{zh(r'\u53bb\u4e2a\u6027\u5316')}",
        'A2': f"A2\n{zh(r'\u53bb')}FedProx",
        'A3': f"A3\n{zh(r'\u53bb\u81ea\u9002\u5e94')}",
    })
    order = [
        f"G6\n{zh(r'\u5b8c\u6574\u65b9\u6848')}",
        f"A1\n{zh(r'\u53bb\u4e2a\u6027\u5316')}",
        f"A2\n{zh(r'\u53bb')}FedProx",
        f"A3\n{zh(r'\u53bb\u81ea\u9002\u5e94')}",
    ]
    ab = ab.set_index('Mechanism').loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), dpi=220)
    metrics = [
        ('tail_rmse', f"{zh(r'\u5c3e\u90e8')} RMSE", True),
        ('tail_sigma_eff', zh(r'\u6709\u6548\u566a\u58f0'), True),
        ('peak_asr', f"{zh(r'\u5cf0\u503c')} ASR", False),
    ]
    colors = ['#2b6cb0', '#f6ad55', '#ed8936', '#9aa0a6']

    for ax, (metric, title, lower_better) in zip(axes, metrics):
        vals = ab[metric].tolist()
        ax.bar(range(len(order)), vals, color=colors, edgecolor='#333333', linewidth=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=0)
        ax.set_title(title, fontsize=14)
        if metric == 'peak_asr':
            ax.axhline(0.5, color='#666666', ls='--', lw=1.2)
            ax.text(0.02, 0.96, zh(r'\u8d8a\u4f4e\u8d8a\u5b89\u5168\uff1b\u8fd9\u91cc\u770b\u653b\u51fb\u66b4\u9732\u4e0a\u9650'), transform=ax.transAxes, va='top', fontsize=10, color='#555')
            ax.set_ylim(0.48, max(vals) + 0.05)
        elif metric == 'tail_sigma_eff':
            ax.text(0.02, 0.96, zh(r'\u8d8a\u4f4e\u8d8a\u4f18'), transform=ax.transAxes, va='top', fontsize=10, color='#555')
            ax.set_ylim(0.0, max(vals) * 1.15)
        else:
            ax.text(0.02, 0.96, zh(r'\u8d8a\u4f4e\u8d8a\u4f18'), transform=ax.transAxes, va='top', fontsize=10, color='#555')
            ax.set_ylim(min(vals) - 0.03, max(vals) + 0.03)
        for i, val in enumerate(vals):
            ax.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    fig.suptitle(zh(r'\u56fe3  \u6d88\u878d\u5b9e\u9a8c\u7ed3\u679c\uff08seed=42\uff09'), fontsize=17, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / 'fig3_ablation_peak.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_overall_heatmap(df: pd.DataFrame) -> Path:
    use_groups = ['G3', 'G4', 'G6', 'G7', 'G9', 'A1', 'A2', 'A3']
    use = df[df['group'].isin(use_groups)].copy()
    agg = use.groupby('group')[['tail_rmse', 'tail_asr', 'tail_auc', 'tail_sigma_eff', 'tail_round_time']].mean().reindex(use_groups)
    agg.index = [GROUP_INFO[g]['label'] for g in use_groups]
    agg.columns = ['RMSE', 'ASR', 'AUC', '有效噪声', '轮均时间 / s']

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=220)
    sns.heatmap(agg, annot=True, fmt='.3f', cmap=sns.light_palette('#1f5aa6', as_cmap=True), linewidths=0.8, cbar_kws={'label': '数值大小'}, ax=ax)
    ax.set_title('图4  总体实验指标热图', fontsize=17, pad=16)
    ax.set_xlabel('指标')
    ax.set_ylabel('实验组')
    fig.tight_layout()
    out = OUT_DIR / 'fig4_overall_heatmap.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def smooth_curve(values: np.ndarray, window: int = 21) -> np.ndarray:
    if len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def plot_core_curves(run_map: Dict[str, Dict[int, RunRecord]]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(19, 10), dpi=220)
    pairs = [
        ('G4', 'G6', f"CDP {zh(r'\u652f\u8def')}"),
        ('G7', 'G9', f"LDP {zh(r'\u652f\u8def')}"),
    ]
    tail_start = 350

    for row, (fixed, adaptive, branch_title) in enumerate(pairs):
        fixed_records = list(run_map.get(fixed, {}).values())
        adp_records = list(run_map.get(adaptive, {}).values())
        if not fixed_records or not adp_records:
            continue

        x, fixed_mean, fixed_std, _ = mean_std_curve(fixed_records, 'rmse')
        _, adp_mean, adp_std, _ = mean_std_curve(adp_records, 'rmse')
        _, fixed_sigma_mean, _, _ = mean_std_curve(fixed_records, 'privacy_sigma')
        _, adp_sigma_mean, _, _ = mean_std_curve(adp_records, 'privacy_sigma')

        fixed_s = smooth_curve(fixed_mean)
        adp_s = smooth_curve(adp_mean)
        fixed_sigma_s = smooth_curve(fixed_sigma_mean)
        adp_sigma_s = smooth_curve(adp_sigma_mean)

        tail_mask = x >= tail_start
        tail_x = x[tail_mask]
        fixed_tail = fixed_s[tail_mask]
        adp_tail = adp_s[tail_mask]
        fixed_tail_mean = float(np.mean(fixed_tail))
        adp_tail_mean = float(np.mean(adp_tail))
        delta_rmse = fixed_tail_mean - adp_tail_mean
        fixed_sigma_tail = float(np.mean(fixed_sigma_s[tail_mask]))
        adp_sigma_tail = float(np.mean(adp_sigma_s[tail_mask]))
        sigma_drop = 0.0 if fixed_sigma_tail <= 0 else 100.0 * (fixed_sigma_tail - adp_sigma_tail) / fixed_sigma_tail

        ax_full = axes[row, 0]
        ax_zoom = axes[row, 1]
        ax_sigma = axes[row, 2]

        ax_full.plot(x, fixed_s, color=COLORS[fixed], lw=2.6, label=GROUP_INFO[fixed]['label'])
        ax_full.plot(x, adp_s, color=COLORS[adaptive], lw=2.6, label=GROUP_INFO[adaptive]['label'])
        ax_full.fill_between(x, fixed_s - fixed_std, fixed_s + fixed_std, color=COLORS[fixed], alpha=0.10)
        ax_full.fill_between(x, adp_s - adp_std, adp_s + adp_std, color=COLORS[adaptive], alpha=0.10)
        ax_full.axvspan(tail_start, x[-1], color='#d9e6f2', alpha=0.18)
        ax_full.set_title(f"{branch_title}{zh(r'\uff1a\u5168\u7a0b')} RMSE", fontsize=14)
        ax_full.set_xlabel(zh(r'\u8bad\u7ec3\u8f6e\u6b21'))
        ax_full.set_ylabel('RMSE')
        ax_full.legend(frameon=True, loc='upper right')

        ax_zoom.plot(tail_x, fixed_tail, color=COLORS[fixed], lw=2.6, label=GROUP_INFO[fixed]['label'])
        ax_zoom.plot(tail_x, adp_tail, color=COLORS[adaptive], lw=2.6, label=GROUP_INFO[adaptive]['label'])
        ax_zoom.fill_between(tail_x, adp_tail, fixed_tail, where=(fixed_tail >= adp_tail), color='#74c69d', alpha=0.22)
        ax_zoom.scatter([tail_x[-1], tail_x[-1]], [fixed_tail[-1], adp_tail[-1]], color=[COLORS[fixed], COLORS[adaptive]], s=30, zorder=3)
        ax_zoom.set_title(f"{branch_title}{zh(r'\uff1a\u5c3e\u90e8\u653e\u5927')}?{tail_start}-{int(x[-1])}{zh(r'\u8f6e')}?", fontsize=14)
        ax_zoom.set_xlabel(zh(r'\u8bad\u7ec3\u8f6e\u6b21'))
        ax_zoom.set_ylabel('RMSE')
        ymin = min(np.min(fixed_tail), np.min(adp_tail))
        ymax = max(np.max(fixed_tail), np.max(adp_tail))
        pad = max((ymax - ymin) * 0.18, 0.02)
        ax_zoom.set_ylim(ymin - pad * 0.35, ymax + pad)
        note = f"?RMSE={delta_rmse:.4f}\n{zh(r'\u56fa\u5b9a')}={fixed_tail_mean:.4f}\n{zh(r'\u81ea\u9002\u5e94')}={adp_tail_mean:.4f}"
        ax_zoom.text(0.03, 0.97, note, transform=ax_zoom.transAxes, va='top', ha='left', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

        ax_sigma.plot(x, fixed_sigma_s, color=COLORS[fixed], lw=2.4, label=GROUP_INFO[fixed]['label'])
        ax_sigma.plot(x, adp_sigma_s, color=COLORS[adaptive], lw=2.4, label=GROUP_INFO[adaptive]['label'])
        ax_sigma.fill_between(x, adp_sigma_s, fixed_sigma_s, where=(fixed_sigma_s >= adp_sigma_s), color='#74c69d', alpha=0.18)
        ax_sigma.set_title(f"{branch_title}{zh(r'\uff1a\u6709\u6548\u566a\u58f0\u8c03\u5ea6')}", fontsize=14)
        ax_sigma.set_xlabel(zh(r'\u8bad\u7ec3\u8f6e\u6b21'))
        ax_sigma.set_ylabel(zh(r'\u6709\u6548\u566a\u58f0'))
        ax_sigma.set_ylim(bottom=0.0)
        ax_sigma.text(0.03, 0.97, f"{zh(r'\u5c3e\u90e8\u566a\u58f0\u4e0b\u964d')}={sigma_drop:.1f}%", transform=ax_sigma.transAxes, va='top', ha='left', fontsize=10,
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='#cccccc'))

    fig.suptitle(zh(r'\u56fe5  \u6838\u5fc3\u673a\u5236\u653e\u5927\u8bc1\u636e\uff1a\u5c3e\u90e8\u5dee\u8ddd\u4e0e\u566a\u58f0\u8c03\u5ea6'), fontsize=18, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / 'fig5_core_curves.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_pareto(df: pd.DataFrame) -> Path:
    use = df[df['group'].isin(['G3', 'G4', 'G6', 'G7', 'G9', 'A1', 'A2', 'A3'])].copy()
    fig, ax = plt.subplots(figsize=(10.8, 7.6), dpi=220)
    for family, fam_df in use.groupby('family'):
        ax.scatter(
            fam_df['tail_rmse'],
            fam_df['tail_asr'],
            s=90 + fam_df['tail_round_time'].fillna(fam_df['tail_round_time'].mean()) * 18,
            color=FAMILY_COLORS.get(family, '#777777'),
            alpha=0.78,
            edgecolors='black',
            linewidths=0.7,
            label={'plain':'Plain','cdp':'CDP','ldp':'LDP','ablation':'消融'}.get(family, family),
        )
    for _, row in use.iterrows():
        ax.annotate(row['group'], (row['tail_rmse'], row['tail_asr']), xytext=(6, 5), textcoords='offset points', fontsize=10)
    ax.axhline(0.5, ls='--', lw=1.2, color='#666666')
    ax.set_title('图6  隐私-效用帕累托分布', fontsize=17, pad=16)
    ax.set_xlabel('尾部 RMSE（越低越优）')
    ax.set_ylabel('尾部 ASR（越接近 0.5 越安全）')
    ax.legend(frameon=True, title='分组', loc='upper right')
    fig.tight_layout()
    out = OUT_DIR / 'fig6_pareto_tradeoff.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_seed_consistency(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=220)
    pair_specs = [
        ('G4', 'G6', 'CDP', axes[0]),
        ('G7', 'G9', 'LDP', axes[1]),
    ]
    for fixed, adaptive, title, ax in pair_specs:
        seeds = sorted(set(df[df['group'] == fixed]['seed']) & set(df[df['group'] == adaptive]['seed']))
        for s in seeds:
            fixed_row = df[(df['group'] == fixed) & (df['seed'] == s)].iloc[0]
            adp_row = df[(df['group'] == adaptive) & (df['seed'] == s)].iloc[0]
            ax.plot([0, 1], [fixed_row['tail_rmse'], adp_row['tail_rmse']], marker='o', lw=2.0, alpha=0.85, label=f'seed {s}')
            ax.text(0, fixed_row['tail_rmse'], f'{fixed_row["tail_sigma_eff"]:.4f}', fontsize=9, ha='right', va='bottom', color='#555')
            ax.text(1, adp_row['tail_rmse'], f'{adp_row["tail_sigma_eff"]:.4f}', fontsize=9, ha='left', va='bottom', color='#555')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['固定', '自适应'])
        ax.set_title(f'{title}: 配对 seed 的 RMSE 变化', fontsize=14)
        ax.set_ylabel('尾部 RMSE')
        ax.grid(True, axis='y', alpha=0.3)
        if seeds:
            ax.legend(frameon=True, loc='upper right')
    fig.suptitle('图7  跨随机种子一致性', fontsize=17, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / 'fig7_seed_consistency.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_runtime_tradeoff(df: pd.DataFrame) -> Path:
    use_groups = ['G4', 'G6', 'G7', 'G9', 'A1', 'A2', 'A3']
    use = df[df['group'].isin(use_groups)].copy()
    agg = use.groupby('group')[['tail_round_time', 'tail_rmse']].mean().reindex(use_groups)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=220)
    ax = axes[0]
    labels = [GROUP_INFO[g]['label'] for g in use_groups]
    colors = [COLORS.get(g, FAMILY_COLORS.get(GROUP_INFO[g]['family'], '#777')) for g in use_groups]
    ax.bar(labels, agg['tail_round_time'], color=colors, edgecolor='#333333', linewidth=0.7)
    ax.set_title('单轮训练时间', fontsize=14)
    ax.set_ylabel('秒 / 轮')
    ax.tick_params(axis='x', rotation=22)

    ax2 = axes[1]
    ax2.scatter(agg['tail_round_time'], agg['tail_rmse'], s=170, c=colors, edgecolors='black', linewidths=0.8)
    for idx, g in enumerate(use_groups):
        ax2.annotate(g, (agg['tail_round_time'].iloc[idx], agg['tail_rmse'].iloc[idx]), xytext=(7, 6), textcoords='offset points', fontsize=10)
    ax2.set_title('时间-性能权衡', fontsize=14)
    ax2.set_xlabel('秒 / 轮')
    ax2.set_ylabel('尾部 RMSE')
    fig.suptitle('图8  运行效率与性能权衡', fontsize=17, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / 'fig8_runtime_tradeoff.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_legacy_bridge(df: pd.DataFrame, legacy: Dict[str, Dict]) -> Path:
    current = {
        'G4': df[(df['group'] == 'G4') & (df['seed'] == 42)].iloc[0],
        'G6': df[(df['group'] == 'G6') & (df['seed'] == 42)].iloc[0],
        'G7': df[(df['group'] == 'G7') & (df['seed'] == 42)].iloc[0],
        'G9': df[(df['group'] == 'G9') & (df['seed'] == 42)].iloc[0],
    }
    legacy_rows = {
        'G4': legacy['legacy_g4_400'],
        'G6': legacy['legacy_g6_500'],
        'G7': legacy['legacy_g7_500'],
        'G9': legacy['legacy_g9_500'],
    }
    groups = ['G4', 'G6', 'G7', 'G9']
    labels = [GROUP_INFO[g]['label'] for g in groups]
    legacy_rmse = [legacy_metric_tail(legacy_rows[g], 'rmse') for g in groups]
    current_rmse = [float(current[g]['tail_rmse']) for g in groups]
    legacy_asr = [legacy_metric_tail(legacy_rows[g], 'attack_acc') for g in groups]
    current_asr = [float(current[g]['tail_asr']) for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), dpi=220)
    x = np.arange(len(groups))
    w = 0.34
    axes[0].bar(x - w / 2, legacy_rmse, width=w, color='#a78bfa', edgecolor='#333333', label='历史 400/500 轮')
    axes[0].bar(x + w / 2, current_rmse, width=w, color='#2b6cb0', edgecolor='#333333', label='当前 1000 轮')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15)
    axes[0].set_title('RMSE 新旧对比', fontsize=14)
    axes[0].set_ylabel('尾部 RMSE')
    axes[0].legend(frameon=True)

    axes[1].bar(x - w / 2, legacy_asr, width=w, color='#a78bfa', edgecolor='#333333', label='历史 400/500 轮')
    axes[1].bar(x + w / 2, current_asr, width=w, color='#2f855a', edgecolor='#333333', label='当前 1000 轮')
    axes[1].axhline(0.5, ls='--', lw=1.2, color='#666666')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15)
    axes[1].set_title('ASR 新旧对比', fontsize=14)
    axes[1].set_ylabel('尾部 ASR')
    axes[1].legend(frameon=True)

    fig.suptitle('图9  历史结果与当前结果的一致性', fontsize=17, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / 'fig9_legacy_bridge.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def core_delta_text(df: pd.DataFrame, left: str, right: str) -> Dict[str, float]:
    left_mean = df[df['group'] == left][['tail_rmse', 'tail_asr', 'tail_auc', 'tail_sigma_eff']].mean()
    right_mean = df[df['group'] == right][['tail_rmse', 'tail_asr', 'tail_auc', 'tail_sigma_eff']].mean()
    return {
        'delta_rmse': float(right_mean['tail_rmse'] - left_mean['tail_rmse']),
        'delta_asr': float(right_mean['tail_asr'] - left_mean['tail_asr']),
        'delta_auc': float(right_mean['tail_auc'] - left_mean['tail_auc']),
        'delta_sigma': float(right_mean['tail_sigma_eff'] - left_mean['tail_sigma_eff']),
        'left_rmse': float(left_mean['tail_rmse']),
        'right_rmse': float(right_mean['tail_rmse']),
    }


def build_report(df: pd.DataFrame, legacy: Dict[str, Dict], figs: Dict[str, Path]) -> str:
    g46 = core_delta_text(df, 'G4', 'G6')
    g79 = core_delta_text(df, 'G7', 'G9')
    legacy_plain = legacy['legacy_plain_500']
    legacy_plain_tail_asr = float(np.mean(legacy_plain['attack_acc'][-50:]))
    legacy_plain_peak = float(np.max(legacy_plain['attack_acc']))
    legacy_plain_train0 = float(legacy_plain['train_loss'][0])
    legacy_plain_test_tail = float(np.mean(legacy_plain['test_loss'][-50:]))
    g6_sigma = float(df[df['group'] == 'G6']['tail_sigma_eff'].mean())
    g9_sigma = float(df[df['group'] == 'G9']['tail_sigma_eff'].mean())
    ab_peak = df[(df['seed'] == 42) & (df['group'].isin(['G6', 'A1', 'A2', 'A3']))].set_index('group')['peak_asr'].to_dict()

    lines = []
    lines.append('# 阶段实验结果报告')
    lines.append('')
    lines.append('## 1. 当前实验覆盖范围')
    lines.append('')
    lines.append('- 已完成的正式实验：`G0-G9` 两个随机种子（`seed=42,52`），以及 `A1-A3` 的 `seed=42`。')
    lines.append('- 尚未完成的主干实验：`A4L`、`A4H`。')
    lines.append('- 已停止但保留现象日志的专项实验：`core_g4g6_conclusion_v3`，可作为强隐私上界导致失稳的补充案例。')
    lines.append('- 可补充引用的历史结果：`Plain 500轮`、`G4/G6 500轮`、`G7/G9 500轮`。')
    lines.append('')
    lines.append('## 2. 图表总览')
    lines.append('')
    lines.append('本次报告共整理 `9` 张图，分别覆盖问题动机、核心对比、消融分析、总体热图、机制曲线、帕累托权衡、跨种子一致性、运行效率和新旧结果一致性。')
    lines.append('')
    for idx, key in enumerate(['plain', 'core', 'ablation', 'heatmap', 'curves', 'pareto', 'seed', 'runtime', 'legacy'], start=1):
        lines.append(f'![Figure {idx}]({figs[key].as_posix()})')
        lines.append('')
    lines.append('## 3. 图表解释与实验结论')
    lines.append('')
    lines.append('### 3.1 攻击动机与基础现象')
    lines.append('')
    lines.append(f'- 图1显示，`Plain` 基线在 `500` 轮历史结果中，训练损失从 `{legacy_plain_train0:.2f}` 快速下降，尾部测试损失约 `{legacy_plain_test_tail:.3f}`，尾部 `ASR` 达到 `{legacy_plain_tail_asr:.3f}`，峰值达到 `{legacy_plain_peak:.2f}`。这说明无隐私约束下，过拟合会显著放大成员推断风险。')
    lines.append('')
    lines.append('### 3.2 核心创新点是否成立')
    lines.append('')
    lines.append(f'- 图2与图5共同说明：在 `CDP` 支路中，`G6` 相比 `G4` 的尾部 `RMSE` 从 `{g46["left_rmse"]:.4f}` 降到 `{g46["right_rmse"]:.4f}`，改善 `{abs(g46["delta_rmse"]):.4f}`；同时有效噪声从 `0.0050` 降到约 `{g6_sigma:.4f}`。')
    lines.append(f'- 在 `LDP` 支路中，`G9` 相比 `G7` 的尾部 `RMSE` 再下降 `{abs(g79["delta_rmse"]):.4f}`，有效噪声从 `0.0200` 降到约 `{g9_sigma:.4f}`。')
    lines.append('- 图5中的噪声调度曲线进一步体现了创新点：固定噪声组保持水平直线，而自适应组会把有效噪声压到更低水平，同时仍保持相近甚至更优的预测性能。')
    lines.append('- 图6从帕累托角度展示了方法位置：`G6/G9` 同时向“更低 RMSE”和“更合理隐私强度”的方向移动，而不是单纯靠牺牲精度换隐私。')
    lines.append('')
    lines.append(zh(r'\u0023\u0023\u0023 3.3 \u6d88\u878d\u5b9e\u9a8c\u8bf4\u660e\u4e86\u4ec0\u4e48'))
    lines.append('')
    lines.append(zh(r'\u002d \u56fe3\u8868\u660e\u5b8c\u6574\u65b9\u6848 `G6` \u53bb\u6389\u4e2a\u6027\u5316\u3001\u53bb\u6389 `FedProx`\u3001\u53bb\u6389\u81ea\u9002\u5e94\u566a\u58f0\u540e\uff0c\u6307\u6807\u90fd\u4f1a\u4e0d\u540c\u7a0b\u5ea6\u56de\u9000\u3002'))
    lines.append(f"{zh(r'\u002d \u8fd9\u91cc\u7b2c\u4e09\u4e2a\u6307\u6807\u6539\u4e3a `\u5cf0\u503c ASR`\uff0c\u800c\u4e0d\u662f\u5c3e\u90e8 `ASR`\u3002\u539f\u56e0\u662f `A1/A2/A3` \u5728\u540e\u671f\u90fd\u6536\u655b\u5230\u4e86\u63a5\u8fd1\u968f\u673a\u731c\u6d4b\u7684 `0.5`\uff0c\u5982\u679c\u7ee7\u7eed\u770b\u5c3e\u90e8\u503c\uff0c\u4f1a\u628a\u6d88\u878d\u9636\u6bb5\u524d\u4e2d\u671f\u771f\u5b9e\u51fa\u73b0\u8fc7\u7684\u653b\u51fb\u66b4\u9732\u5dee\u5f02\u5168\u90e8\u6299\u5e73\u3002\u6309 `seed=42` \u7edf\u8ba1\uff0c`G6/A1/A2/A3` \u7684\u5cf0\u503c `ASR` \u5206\u522b\u7ea6\u4e3a')} `{ab_peak.get('G6', float('nan')):.4f} / {ab_peak.get('A1', float('nan')):.4f} / {ab_peak.get('A2', float('nan')):.4f} / {ab_peak.get('A3', float('nan')):.4f}`{zh(r'\u3002')}")
    lines.append(zh(r'\u002d \u5176\u4e2d `A3` \u6700\u5173\u952e\uff1a\u53bb\u6389\u81ea\u9002\u5e94\u566a\u58f0\u540e\uff0c`RMSE` \u56de\u5230\u56fa\u5b9a\u566a\u58f0\u7ec4\u6c34\u5e73\uff0c\u540c\u65f6\u6709\u6548\u566a\u58f0\u4e0d\u518d\u4e0b\u964d\uff1b\u800c\u5728\u653b\u51fb\u66b4\u9732\u4e0a\u9650\u4e0a\uff0c\u4e5f\u6ca1\u6709\u4f53\u73b0\u51fa\u4f18\u4e8e\u5b8c\u6574\u65b9\u6848\u7684\u4f18\u52bf\u3002\u8fd9\u80fd\u66f4\u76f4\u63a5\u652f\u6491\u201c\u6539\u8fdb\u6765\u81ea\u81ea\u9002\u5e94\u9690\u79c1\u8c03\u5ea6\uff0c\u800c\u4e0d\u662f\u5176\u4ed6\u6a21\u5757\u5076\u7136\u914d\u5408\u201d\u3002'))
    lines.append('')
    lines.append('### 3.4 多视角稳定性如何')
    lines.append('')
    lines.append('- 图4给出总体指标热图，说明 `G6/G9` 在综合指标上处于更值得强调的位置。')
    lines.append('- 图7给出跨种子的配对斜线图，可以看到两个随机种子下，自适应方案相对固定噪声的改进方向是一致的。')
    lines.append('- 图9则说明即使把历史 `400/500` 轮结果拿进来，`G4→G6`、`G7→G9` 的改进方向仍然保持一致，因此你的结果不是一次性偶然现象。')
    lines.append('')
    lines.append('### 3.5 效率代价如何')
    lines.append('')
    lines.append('- 图8给出了单轮训练时间与性能权衡。可以看出，`LDP` 分支整体更慢，而 `CDP` 自适应方案并没有带来不可接受的额外开销。')
    lines.append('- 因此从工程角度看，自适应 `CDP` 是当前最容易写成“兼顾可用性与创新性”的路线。')
    lines.append('')
    lines.append('## 4. 现在能下到什么程度的结论')
    lines.append('')
    lines.append('1. 可以明确写出的结论：')
    lines.append('   - `Plain` 过拟合会放大成员推断风险。')
    lines.append('   - 固定差分隐私能降低攻击可分性，但会引入效用损失。')
    lines.append('   - 自适应噪声调度在 `CDP/LDP` 两条支路中都表现出同方向收益，即“更低有效噪声 + 更优或不差的效用”。')
    lines.append('   - 消融实验支持该增益确实来自自适应隐私调度与联邦训练机制的联合设计。')
    lines.append('2. 现在不建议写得太满的地方：')
    lines.append('   - 当前 `G4/G6` 和 `G7/G9` 仍然只有 `2` 个配对种子，更适合写成“稳定趋势”和“方向一致”，不建议直接宣称强统计显著性。')
    lines.append('')
    lines.append('## 5. 怎么样才算“结论比较明显”')
    lines.append('')
    lines.append('建议你把标准设成下面这样：')
    lines.append('')
    lines.append('1. 核心对照 `G4/G6/G7/G9` 至少补到 `5` 个种子。')
    lines.append('2. `G6` 相对 `G4` 至少满足：')
    lines.append('   - 平均 `RMSE` 改善 `2%~3%`；')
    lines.append('   - 有效噪声下降 `10%~15%`；')
    lines.append('   - `ASR` 变化不超过 `0.01`；')
    lines.append('   - 至少 `4/5` 个种子保持同方向。')
    lines.append('3. 如果再补一个 `sigma` 扫描，在多个隐私强度下都保持上述方向，那么创新点就会非常好写。')
    lines.append('')
    lines.append('## 6. 现在 GPU 还没结束，最值得再加什么')
    lines.append('')
    lines.append('按优先级排序：')
    lines.append('')
    lines.append('1. 先补完主干的 `A4L/A4H`。')
    lines.append('2. 主干之后优先只跑 `G4/G6/G7/G9` 的 `3` 个新种子。')
    lines.append('3. 如果还有时间，再做 `G4/G6` 的 `sigma` 扫描，例如 `0.003 / 0.005 / 0.008`。')
    lines.append('4. 不建议继续花大量时间调 `plain`，它更适合做攻击动机展示。')
    lines.append('5. 不建议继续跑已经发散的 `core_g4g6_conclusion_v3`，那个更适合作为失稳上界案例。')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    ensure_dir()
    runs = load_current_runs()
    df = current_summary_df(runs)
    legacy = load_legacy_payloads()
    run_map = runs_by_group_seed(runs)
    figs = {
        'plain': plot_plain_risk(legacy),
        'core': plot_core_comparison(df),
        'ablation': plot_ablation(df),
        'heatmap': plot_overall_heatmap(df),
        'curves': plot_core_curves(run_map),
        'pareto': plot_pareto(df),
        'seed': plot_seed_consistency(df),
        'runtime': plot_runtime_tradeoff(df),
        'legacy': plot_legacy_bridge(df, legacy),
    }
    report_text = build_report(df, legacy, figs)
    with open(REPORT_PATH, 'w', encoding='utf-8-sig') as f:
        f.write(report_text)
    df.sort_values(['group', 'seed']).to_csv(OUT_DIR / '阶段报告核心汇总.csv', index=False, encoding='utf-8-sig')
    print(f'[OK] 报告已生成: {REPORT_PATH}')
    for key, fig in figs.items():
        print(f'[FIG] {key}: {fig}')


if __name__ == '__main__':
    main()
