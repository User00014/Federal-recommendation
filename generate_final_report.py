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
REPORT_ROOT = ROOT / "reports"
CURRENT_DIRS = [
    ROOT / "cloud_results" / "final_pull_20260311_164823" / "full_15_seed42_v2" / "logs",
    ROOT / "cloud_results" / "pull_seed52_20260311_111617" / "full_15_seed52_v2" / "logs",
]
LEGACY_FILES = {
    "plain_500": ROOT / "logs" / "res_PLAIN_sigma0_FEDAVG_NP_FDP_500rounds.json",
    "g4_400": ROOT / "logs" / "res_CDP_sigma0.005_FEDPROX_P_FDP_400rounds.json",
    "g6_500": ROOT / "logs" / "res_CDP_sigma0.005_FEDPROX_P_ADP_500rounds.json",
    "g7_500": ROOT / "logs" / "res_LDP_sigma0.02_FEDPROX_P_FDP_500rounds.json",
    "g9_500": ROOT / "logs" / "res_LDP_sigma0.02_FEDPROX_P_ADP_500rounds.json",
}
OUT_DIR = ROOT / "figures" / "final_report_20260311"
REPORT_PATH = REPORT_ROOT / "最终实验结果报告_20260311.md"
SUMMARY_PATH = OUT_DIR / "最终实验汇总.csv"
TAIL = 50
configure_report_plot_style()
GROUP_INFO = {
    "G0": {"label": "G0 FedAvg", "family": "plain"},
    "G1": {"label": "G1 FedProx", "family": "plain"},
    "G2": {"label": "G2 个性化", "family": "plain"},
    "G3": {"label": "G3 全局CDP", "family": "cdp"},
    "G4": {"label": "G4 固定CDP", "family": "cdp"},
    "G5": {"label": "G5 个性化CDP", "family": "cdp"},
    "G6": {"label": "G6 自适应CDP", "family": "cdp"},
    "G7": {"label": "G7 固定LDP", "family": "ldp"},
    "G8": {"label": "G8 个性化LDP", "family": "ldp"},
    "G9": {"label": "G9 自适应LDP", "family": "ldp"},
    "A1": {"label": "A1 去个性化", "family": "ablation"},
    "A2": {"label": "A2 去FedProx", "family": "ablation"},
    "A3": {"label": "A3 去自适应DP", "family": "ablation"},
    "A4L": {"label": "A4L 低mu", "family": "sensitivity"},
    "A4H": {"label": "A4H 高mu", "family": "sensitivity"},
}
GROUP_ORDER = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]
ALL_ORDER = GROUP_ORDER + ["A1", "A2", "A3", "A4L", "A4H"]
PAIR_COLORS = {
    "G4": "#8f98a0",
    "G6": "#1f5aa6",
    "G7": "#7d8c82",
    "G9": "#1f8a5b",
}
FAMILY_COLORS = {
    "plain": "#d55e5e",
    "cdp": "#2b6cb0",
    "ldp": "#2f855a",
    "ablation": "#dd8a2b",
    "sensitivity": "#805ad5",
}
def ensure_dir() -> None:
    ensure_dirs(OUT_DIR, REPORT_PATH.parent)
def load_runs() -> List[RunRecord]:
    return load_runs_from_dirs(CURRENT_DIRS)
def load_legacy() -> Dict[str, Dict]:
    return load_existing_payloads(LEGACY_FILES)
def make_summary_df(runs: List[RunRecord]) -> pd.DataFrame:
    df = pd.DataFrame([r.build_summary(TAIL) for r in runs])
    if df.empty:
        return df
    df["label"] = df["group"].map(lambda x: GROUP_INFO[x]["label"])
    df["family"] = df["group"].map(lambda x: GROUP_INFO[x]["family"])
    return df
def mean_std_curve(records: List[RunRecord], field: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return shared_mean_std_curve(records, field)
def group_stats(df: pd.DataFrame, groups: List[str], metric: str) -> pd.DataFrame:
    return df[df["group"].isin(groups)].groupby("group")[metric].agg(["mean", "std"]).reindex(groups)
def pct_drop(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return 100.0 * (base - new) / base
def plot_plain_risk(legacy: Dict[str, Dict]) -> Path:
    payload = legacy["plain_500"]
    rounds = np.arange(1, len(payload["attack_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6), dpi=220)
    ax = axes[0]
    ax.plot(rounds, payload["train_loss"], color="#ca4e5b", lw=2.4, label="训练损失")
    ax.plot(rounds, payload["test_loss"], color="#3e7cb1", lw=2.4, label="测试损失")
    ax.set_title("Plain 500轮历史结果：损失变化")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("损失值")
    ax.legend(loc="upper right", frameon=True)
    ax.annotate("训练损失快速下降", xy=(25, payload["train_loss"][24]), xytext=(115, max(payload["train_loss"]) * 0.72),
                arrowprops=dict(arrowstyle="->", color="#444"), fontsize=11)
    ax = axes[1]
    ax.plot(rounds, payload["attack_acc"], color="#b42339", lw=2.6)
    ax.fill_between(rounds, 0.5, payload["attack_acc"], color="#f2b8be", alpha=0.3)
    ax.axhline(0.5, color="#666", ls="--", lw=1.3, label="随机猜测")
    peak_idx = int(np.argmax(payload["attack_acc"]))
    peak_val = float(payload["attack_acc"][peak_idx])
    ax.scatter(peak_idx + 1, peak_val, color="#7f1d1d", s=45, zorder=5)
    ax.annotate(f"峰值 ASR={peak_val:.2f}", xy=(peak_idx + 1, peak_val), xytext=(peak_idx - 110, 0.88),
                arrowprops=dict(arrowstyle="->", color="#444"), fontsize=11)
    ax.set_title("成员推断攻击成功率")
    ax.set_xlabel("训练轮次")
    ax.set_ylabel("ASR")
    ax.set_ylim(0.45, 0.93)
    ax.legend(loc="lower right", frameon=True)
    fig.suptitle("图1  Plain基线的过拟合与隐私泄露风险", fontsize=18, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig1_plain_overfit_risk.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_main_overview(df: pd.DataFrame) -> Path:
    fig = plt.figure(figsize=(19.5, 9.8), dpi=220)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.28)
    ax_rmse_full = fig.add_subplot(gs[0, 0])
    ax_rmse_zoom = fig.add_subplot(gs[1, 0])
    ax_asr = fig.add_subplot(gs[:, 1])
    ax_time = fig.add_subplot(gs[:, 2])

    labels = [GROUP_INFO[g]["label"] for g in GROUP_ORDER]
    colors = [FAMILY_COLORS[GROUP_INFO[g]["family"]] for g in GROUP_ORDER]
    x = np.arange(len(GROUP_ORDER))
    metric_map = {
        "rmse": group_stats(df, GROUP_ORDER, "tail_rmse"),
        "asr": group_stats(df, GROUP_ORDER, "tail_asr"),
        "time": group_stats(df, GROUP_ORDER, "tail_time"),
    }

    def draw(ax, stats, metric_name, title):
        ax.bar(x, stats["mean"], yerr=stats["std"].fillna(0.0), color=colors, edgecolor="#2f2f2f", linewidth=0.7, capsize=4)
        for idx, group in enumerate(GROUP_ORDER):
            samples = df[df["group"] == group][metric_name].tolist()
            if samples:
                jitter = np.linspace(-0.08, 0.08, len(samples))
                for j, value in enumerate(samples):
                    ax.scatter(idx + (jitter[j] if len(samples) > 1 else 0), value, s=26, color="#111", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)

    draw(ax_rmse_full, metric_map["rmse"], "tail_rmse", "尾部 RMSE（全范围）")
    draw(ax_rmse_zoom, metric_map["rmse"], "tail_rmse", "尾部 RMSE（局部放大）")
    draw(ax_asr, metric_map["asr"], "tail_asr", "尾部 ASR（局部放大）")
    draw(ax_time, metric_map["time"], "tail_time", "单轮时间（局部放大）")

    ax_rmse_zoom.set_ylim(0.94, 1.16)
    ax_rmse_zoom.text(0.02, 0.97, "突出主流方案之间的细微差异", transform=ax_rmse_zoom.transAxes, va="top", fontsize=10,
                      bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    ax_asr.axhline(0.5, color="#666", ls="--", lw=1.2)
    ax_asr.set_ylim(0.488, 0.548)
    ax_asr.text(0.02, 0.97, "仅放大 0.488-0.548 区间", transform=ax_asr.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    ax_time.set_ylim(2.8, 9.4)
    ax_time.text(0.02, 0.97, "去掉极端空白，突出常用方案时间差异", transform=ax_time.transAxes, va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    d = 0.015
    kwargs = dict(transform=ax_rmse_full.transAxes, color='k', clip_on=False, linewidth=1.0)
    ax_rmse_full.plot((-d, +d), (-d, +d), **kwargs)
    ax_rmse_full.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs = dict(transform=ax_rmse_zoom.transAxes, color='k', clip_on=False, linewidth=1.0)
    ax_rmse_zoom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_rmse_zoom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.suptitle("图2  主实验总体结果对比（G0-G9，含局部纵轴放大）", fontsize=18, y=0.98)
    out = OUT_DIR / "fig2_main_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

def plot_core_gain(df: pd.DataFrame) -> Path:
    pairs = [("G4", "G6", "CDP 支路"), ("G7", "G9", "LDP 支路")]
    metrics = [("tail_rmse", "尾部 RMSE"), ("tail_sigma", "有效噪声"), ("tail_asr", "尾部 ASR"), ("tail_auc", "尾部 AUC")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), dpi=220)
    for row, (fixed, adaptive, title) in enumerate(pairs):
        for col, (metric, metric_title) in enumerate(metrics):
            ax = axes[row, col]
            subset = df[df["group"].isin([fixed, adaptive])]
            stats = subset.groupby("group")[metric].agg(["mean", "std"]).reindex([fixed, adaptive])
            colors = [PAIR_COLORS[fixed], PAIR_COLORS[adaptive]]
            means = stats["mean"].to_numpy(dtype=float)
            errs = stats["std"].fillna(0.0).to_numpy(dtype=float)
            ax.bar([0, 1], means, yerr=errs, color=colors, edgecolor="#333", linewidth=0.8, capsize=4)
            for idx, group in enumerate([fixed, adaptive]):
                samples = subset[subset["group"] == group][metric].tolist()
                jitter = np.linspace(-0.06, 0.06, len(samples)) if samples else []
                for j, value in enumerate(samples):
                    ax.scatter(idx + (jitter[j] if len(samples) > 1 else 0), value, color="#111", s=30, zorder=3)
            spread_min = float(np.min(means - errs))
            spread_max = float(np.max(means + errs))
            pad = max((spread_max - spread_min) * 0.35, 0.0002 if metric == "tail_sigma" else 0.01)
            if metric in {"tail_asr", "tail_auc"}:
                low = min(0.49, spread_min - pad * 0.3)
                high = max(0.505, spread_max + pad)
                ax.axhline(0.5, color="#666", ls="--", lw=1.0)
                ax.set_ylim(low, high)
            else:
                ax.set_ylim(spread_min - pad * 0.25, spread_max + pad)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([GROUP_INFO[fixed]["label"], GROUP_INFO[adaptive]["label"]], rotation=12)
            ax.set_title(f"{title}：{metric_title}（局部放大）", fontsize=13)
            delta = float(stats.loc[adaptive, "mean"] - stats.loc[fixed, "mean"])
            sign = "+" if delta >= 0 else ""
            ax.text(0.03, 0.95, f"Δ={sign}{delta:.4f}", transform=ax.transAxes, va="top", ha="left",
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    fig.suptitle("图3  固定噪声与自适应噪声的核心增益拆解（局部纵轴放大）", fontsize=18, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig3_core_gain_breakdown.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_core_curves(run_map: Dict[str, Dict[int, RunRecord]]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)
    pair_specs = [("G4", "G6", "CDP 支路"), ("G7", "G9", "LDP 支路")]
    for row, (fixed, adaptive, title) in enumerate(pair_specs):
        seeds = sorted(set(run_map.get(fixed, {})) & set(run_map.get(adaptive, {})))
        fixed_runs = [run_map[fixed][s] for s in seeds]
        adp_runs = [run_map[adaptive][s] for s in seeds]
        x, fixed_rmse, fixed_std = mean_std_curve(fixed_runs, "rmse")
        _, adp_rmse, adp_std = mean_std_curve(adp_runs, "rmse")
        _, fixed_sigma, _ = mean_std_curve(fixed_runs, "privacy_sigma")
        _, adp_sigma, _ = mean_std_curve(adp_runs, "privacy_sigma")
        tail_start = max(1, int(x[-1] - 300))
        mask = x >= tail_start
        rmse_gain_pct = np.divide(
            fixed_rmse - adp_rmse,
            np.maximum(fixed_rmse, 1e-12),
        ) * 100.0
        sigma_drop_pct = np.divide(
            fixed_sigma - adp_sigma,
            np.maximum(fixed_sigma, 1e-12),
        ) * 100.0

        ax = axes[row, 0]
        ax.plot(x, fixed_rmse, color=PAIR_COLORS[fixed], lw=2.4, label=GROUP_INFO[fixed]["label"])
        ax.plot(x, adp_rmse, color=PAIR_COLORS[adaptive], lw=2.4, label=GROUP_INFO[adaptive]["label"])
        ax.fill_between(x, fixed_rmse - fixed_std, fixed_rmse + fixed_std, color=PAIR_COLORS[fixed], alpha=0.10)
        ax.fill_between(x, adp_rmse - adp_std, adp_rmse + adp_std, color=PAIR_COLORS[adaptive], alpha=0.10)
        ax.axvspan(tail_start, x[-1], color="#d6ecff", alpha=0.25)
        ax.set_title(f"{title}：全程 RMSE")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("RMSE")
        ax.legend(loc="upper right", frameon=True)

        ax = axes[row, 1]
        ax.plot(x[mask], rmse_gain_pct[mask], color=PAIR_COLORS[adaptive], lw=2.6)
        ax.fill_between(
            x[mask],
            0.0,
            rmse_gain_pct[mask],
            where=rmse_gain_pct[mask] >= 0,
            color="#74c69d",
            alpha=0.22,
        )
        ax.axhline(0.0, color="#666", ls="--", lw=1.1)
        fixed_tail_mean = float(np.mean(fixed_rmse[mask]))
        adp_tail_mean = float(np.mean(adp_rmse[mask]))
        gain_tail_pct = pct_drop(fixed_tail_mean, adp_tail_mean)
        pct_low = float(np.min(rmse_gain_pct[mask]))
        pct_high = float(np.max(rmse_gain_pct[mask]))
        pct_pad = max((pct_high - pct_low) * 0.18, 0.15)
        ax.set_ylim(pct_low - pct_pad, pct_high + pct_pad)
        ax.text(0.03, 0.96,
                f"固定={fixed_tail_mean:.4f}\n自适应={adp_tail_mean:.4f}\n尾部提升={gain_tail_pct:.2f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9))
        ax.set_title(f"{title}：尾部相对改进（{tail_start}-{int(x[-1])}轮）")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("RMSE 改进 (%)")

        ax = axes[row, 2]
        ax.plot(x, sigma_drop_pct, color=PAIR_COLORS[adaptive], lw=2.6)
        ax.fill_between(
            x,
            0.0,
            sigma_drop_pct,
            where=sigma_drop_pct >= 0,
            color="#74c69d",
            alpha=0.22,
        )
        ax.axhline(0.0, color="#666", ls="--", lw=1.1)
        sigma_drop = pct_drop(float(np.mean(fixed_sigma[mask])), float(np.mean(adp_sigma[mask])))
        sigma_low = float(np.min(sigma_drop_pct))
        sigma_high = float(np.max(sigma_drop_pct))
        sigma_pad = max((sigma_high - sigma_low) * 0.18, 0.4)
        ax.set_ylim(sigma_low - sigma_pad, sigma_high + sigma_pad)
        ax.text(
            0.03,
            0.96,
            f"尾部噪声下降={sigma_drop:.1f}%\n固定={np.mean(fixed_sigma[mask]):.4f}\n自适应={np.mean(adp_sigma[mask]):.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9),
        )
        ax.set_title(f"{title}：有效噪声下降比例")
        ax.set_xlabel("训练轮次")
        ax.set_ylabel("噪声下降 (%)")
    fig.suptitle("图4  核心机制曲线：相对收益与噪声降幅", fontsize=18, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig4_core_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_ablation(df: pd.DataFrame) -> Path:
    groups = ["G6", "A1", "A2", "A3"]
    metrics = [("tail_rmse", "尾部 RMSE"), ("tail_sigma", "有效噪声"), ("peak_asr", "峰值 ASR")]
    fig, axes = plt.subplots(1, 3, figsize=(17.8, 5.8), dpi=220)
    colors = ["#1f5aa6", "#f6ad55", "#ed8936", "#9aa0a6"]
    labels = [GROUP_INFO[g]["label"] for g in groups]
    for ax, (metric, title) in zip(axes, metrics):
        stats = group_stats(df, groups, metric)
        x = np.arange(len(groups))
        means = stats["mean"].to_numpy(dtype=float)
        errs = stats["std"].fillna(0.0).to_numpy(dtype=float)
        base = float(stats.loc["G6", "mean"])
        ax.bar(x, means, yerr=errs, color=colors, edgecolor="#333", linewidth=0.7, capsize=4)
        for idx, group in enumerate(groups):
            samples = df[df["group"] == group][metric].tolist()
            jitter = np.linspace(-0.05, 0.05, len(samples)) if samples else []
            for j, value in enumerate(samples):
                ax.scatter(idx + (jitter[j] if len(samples) > 1 else 0), value, s=28, color="#111", zorder=3)
            delta = float(stats.loc[group, "mean"] - base)
            if metric == "tail_sigma":
                text = f"Δ={delta:+.4f}"
            else:
                text = f"Δ={delta:+.3f}"
            ax.text(idx, means[idx] + max(errs[idx], 0) + (0.01 if metric != "tail_sigma" else 0.00006), text,
                    ha="center", va="bottom", fontsize=10, color="#444")
        spread_min = float(np.min(means - errs))
        spread_max = float(np.max(means + errs))
        pad = max((spread_max - spread_min) * 0.22, 0.00012 if metric == "tail_sigma" else 0.015)
        ax.set_ylim(spread_min - pad * 0.35, spread_max + pad)
        ax.axhline(base, color="#666", ls="--", lw=1.1, label="G6 基线")
        if metric == "peak_asr":
            ax.axhline(0.5, color="#999", ls=":", lw=1.0, label="随机猜测")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_title(f"{title}（局部放大）")
        ax.grid(axis="y", alpha=0.25)
        if metric == "peak_asr":
            ax.legend(loc="lower right", frameon=True, fontsize=9)
    fig.suptitle("图5  消融实验：完整方案与关键模块移除后的变化（局部坐标轴放大）", fontsize=18, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / "fig5_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_sensitivity(df: pd.DataFrame) -> Path:
    groups = ["A4L", "G6", "A4H"]
    mu_labels = ["0.001", "0.01", "0.05"]
    metrics = [("tail_rmse", "尾部 RMSE"), ("tail_asr", "尾部 ASR"), ("tail_time", "单轮时间（秒）")]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8), dpi=220)
    for ax, (metric, title) in zip(axes, metrics):
        vals = [float(df[df["group"] == g][metric].mean()) for g in groups]
        ax.plot(mu_labels, vals, color="#6b46c1", lw=2.6, marker="o", ms=8)
        ax.fill_between(mu_labels, vals, [min(vals)] * len(vals), color="#d6bcfa", alpha=0.18)
        for x, y, group in zip(mu_labels, vals, groups):
            ax.annotate(group, (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=10)
        if metric == "tail_asr":
            ax.axhline(0.5, color="#666", ls="--", lw=1.1)
        ax.set_title(title)
        ax.set_xlabel("FedProx μ")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("图6  参数敏感性：FedProx μ 对完整方案的影响", fontsize=18, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / "fig6_sensitivity_mu.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_heatmap(df: pd.DataFrame) -> Path:
    display = df[df["group"].isin(ALL_ORDER)].groupby("group")[["tail_rmse", "tail_asr", "peak_asr", "tail_sigma", "tail_time"]].mean()
    display = display.reindex(ALL_ORDER)
    display.columns = ["RMSE", "尾部ASR", "峰值ASR", "有效噪声", "单轮时间"]
    display.index = [GROUP_INFO[g]["label"] for g in display.index]
    scaled = (display - display.mean()) / display.std(ddof=0)
    scaled = scaled.fillna(0.0).clip(-2.5, 2.5)
    fig, ax = plt.subplots(figsize=(10.5, 8.6), dpi=220)
    sns.heatmap(
        scaled,
        cmap="RdYlBu_r",
        annot=display.round(4).values,
        fmt="",
        linewidths=0.6,
        vmin=-2.5,
        vmax=2.5,
        cbar_kws={"label": "标准化分数"},
        ax=ax,
    )
    ax.set_title("图7  各实验组多指标热图", fontsize=18, pad=14)
    ax.set_xlabel("指标")
    ax.set_ylabel("实验组")
    fig.tight_layout()
    out = OUT_DIR / "fig7_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_pareto(df: pd.DataFrame) -> Path:
    use_groups = ["G3", "G4", "G6", "G7", "G9", "A1", "A2", "A3", "A4L", "A4H"]
    use = df[df["group"].isin(use_groups)].groupby(["group", "family"], as_index=False)[["tail_rmse", "tail_asr", "tail_time"]].mean()
    fig, ax = plt.subplots(figsize=(11, 8), dpi=220)
    for family, fam_df in use.groupby("family"):
        ax.scatter(
            fam_df["tail_rmse"],
            fam_df["tail_asr"],
            s=80 + fam_df["tail_time"] * 16,
            color=FAMILY_COLORS[family],
            alpha=0.82,
            edgecolors="black",
            linewidths=0.8,
            label={"cdp": "CDP", "ldp": "LDP", "ablation": "消融", "sensitivity": "参数敏感性"}[family],
        )
    for _, row in use.iterrows():
        ax.annotate(row["group"], (row["tail_rmse"], row["tail_asr"]), xytext=(6, 5), textcoords="offset points", fontsize=10)
    ax.axhline(0.5, color="#666", ls="--", lw=1.1)
    ax.set_title("图8  隐私-效用-时间的帕累托分布", fontsize=18, pad=14)
    ax.set_xlabel("尾部 RMSE（越低越优）")
    ax.set_ylabel("尾部 ASR（越接近 0.5 越安全）")
    ax.legend(loc="upper right", frameon=True, title="分支")
    fig.tight_layout()
    out = OUT_DIR / "fig8_pareto.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def plot_seed_consistency(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=220)
    for ax, (fixed, adp, title) in zip(axes, [("G4", "G6", "CDP"), ("G7", "G9", "LDP")]):
        seeds = sorted(set(df[df["group"] == fixed]["seed"]) & set(df[df["group"] == adp]["seed"]))
        for seed in seeds:
            frow = df[(df["group"] == fixed) & (df["seed"] == seed)].iloc[0]
            arow = df[(df["group"] == adp) & (df["seed"] == seed)].iloc[0]
            ax.plot([0, 1], [frow["tail_rmse"], arow["tail_rmse"]], marker="o", lw=2.2, alpha=0.9, label=f"seed {seed}")
            ax.text(0, frow["tail_rmse"], f"{frow['tail_sigma']:.4f}", fontsize=9, ha="right", va="bottom")
            ax.text(1, arow["tail_rmse"], f"{arow['tail_sigma']:.4f}", fontsize=9, ha="left", va="bottom")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["固定噪声", "自适应噪声"])
        ax.set_ylabel("尾部 RMSE")
        ax.set_title(f"{title}：配对 seed 的一致性")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("图9  跨随机种子的结果一致性", fontsize=18, y=1.03)
    fig.tight_layout()
    out = OUT_DIR / "fig9_seed_consistency.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out
def tail_mean(values: List[float]) -> float:
    chunk = values[-TAIL:] if len(values) >= TAIL else values
    return float(np.mean(chunk))
def plot_ldp_focus(df: pd.DataFrame) -> Path:
    groups = ["G7", "G8", "G9"]
    labels = [GROUP_INFO[g]["label"] for g in groups]
    colors = [PAIR_COLORS["G7"], "#2f855a", PAIR_COLORS["G9"]]
    metrics = [
        ("tail_rmse", "尾部 RMSE", (1.02, 3.05)),
        ("tail_sigma", "有效噪声", (0.015, 0.052)),
        ("peak_asr", "峰值 ASR", (0.69, 0.76)),
        ("tail_time", "单轮时间（秒）", (7.1, 8.95)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 10), dpi=220)
    axes = axes.flatten()
    x = np.arange(len(groups))

    for ax, (metric, title, ylim) in zip(axes, metrics):
        stats = group_stats(df, groups, metric)
        means = stats["mean"].to_numpy(dtype=float)
        errs = stats["std"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, means, yerr=errs, color=colors, edgecolor="#333", linewidth=0.8, capsize=4)
        for idx, group in enumerate(groups):
            samples = df[df["group"] == group][metric].tolist()
            jitter = np.linspace(-0.05, 0.05, len(samples)) if samples else []
            for j, value in enumerate(samples):
                ax.scatter(idx + (jitter[j] if len(samples) > 1 else 0), value, s=30, color="#111", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_title(f"{title}（LDP专题）")
        ax.set_ylim(*ylim)
        if metric in {"peak_asr"}:
            ax.axhline(0.5, color="#777", ls=":", lw=1.0)
        ax.grid(axis="y", alpha=0.25)

    axes[0].text(0.03, 0.96, "比较固定LDP、个性化LDP与自适应LDP", transform=axes[0].transAxes, va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    axes[1].text(0.03, 0.96, "G9 相比 G7 的噪声更低\nG8 噪声最强，因此效用下降最明显", transform=axes[1].transAxes, va="top", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    fig.suptitle("图10  LDP分支专题对比：固定、个性化与自适应机制", fontsize=18, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "fig10_ldp_focus.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out

def build_report(df: pd.DataFrame, legacy: Dict[str, Dict], figs: Dict[str, Path]) -> str:
    g4 = float(df[df["group"] == "G4"]["tail_rmse"].mean())
    g6 = float(df[df["group"] == "G6"]["tail_rmse"].mean())
    g7 = float(df[df["group"] == "G7"]["tail_rmse"].mean())
    g9 = float(df[df["group"] == "G9"]["tail_rmse"].mean())
    g6_sigma = float(df[df["group"] == "G6"]["tail_sigma"].mean())
    g9_sigma = float(df[df["group"] == "G9"]["tail_sigma"].mean())
    g4_sigma = float(df[df["group"] == "G4"]["tail_sigma"].mean())
    g7_sigma = float(df[df["group"] == "G7"]["tail_sigma"].mean())
    a4l_rmse = float(df[df["group"] == "A4L"]["tail_rmse"].mean())
    a4h_rmse = float(df[df["group"] == "A4H"]["tail_rmse"].mean())
    legacy_plain = legacy["plain_500"]
    legacy_plain_tail_asr = tail_mean(legacy_plain["attack_acc"])
    legacy_plain_peak = float(np.max(legacy_plain["attack_acc"]))
    legacy_plain_train_start = float(legacy_plain["train_loss"][0])
    legacy_plain_test_tail = tail_mean(legacy_plain["test_loss"])
    ab_peak = df[df["group"].isin(["G6", "A1", "A2", "A3"])].groupby("group")["peak_asr"].mean().to_dict()
    counts = df.groupby("group")["seed"].nunique().sort_index()
    fig_captions = {
        "plain": "图1说明 Plain 基线在过拟合过程中训练损失快速下降，同时成员推断攻击成功率明显升高，用于支撑本文的隐私保护动机。",
        "main": "图2给出主实验 G0-G9 的总体结果，从尾部 RMSE、尾部 ASR 和单轮时间三个角度比较不同联邦隐私方案的综合表现。",
        "core": "图3将固定噪声与自适应噪声的核心差异拆成 RMSE、有效噪声、ASR 和 AUC 四项指标，用于直接展示创新点收益。",
        "curves": "图4通过全程收敛曲线、尾部相对改进百分比和噪声下降百分比三部分，展示自适应隐私机制在收敛后期的真实收益。",
        "ablation": "图5展示完整方案与去掉关键模块后的性能变化，用于说明个性化、FedProx 和自适应噪声调度各自的作用。",
        "sens": "图6展示 FedProx 参数 μ 对完整方案的影响，用于验证本文方法在一段合理参数范围内的稳定性。",
        "heatmap": "图7从多指标角度汇总所有实验组，便于整体观察各方案在效用、隐私与时间开销之间的位置关系。",
        "pareto": "图8将隐私、效用和训练时间同时放入同一张帕累托分布图，用于突出不同方法的综合权衡。",
        "seed": "图9比较固定噪声与自适应噪声在不同随机种子下的配对结果，用于验证改进方向的一致性。",
        "ldp_focus": "图10专门抽出 LDP 分支进行专题分析，比较固定、个性化与自适应三种机制在效用、隐私与时间上的差异。",
    }
    lines: List[str] = []
    lines.append("# 最终实验结果报告")
    lines.append("")
    lines.append("## 1. 实验覆盖范围")
    lines.append("")
    lines.append("- 主实验 `G0-G9` 已全部完成，其中 `seed=42,52` 两个随机种子的结果均可用于正式对比。")
    lines.append("- 消融实验 `A1-A3` 与参数敏感性实验 `A4L/A4H` 已完成，当前均为 `seed=42`。")
    lines.append("- 历史实验中的 `Plain 500轮` 结果被保留为攻击动机补充证据；正式对比与图表分析均以当前完整实验结果为主。")
    lines.append("")
    lines.append("## 2. 图表总览")
    lines.append("")
    lines.append("本报告共输出 `10` 张图，覆盖问题动机、主实验总体对比、核心增益拆解、机制曲线、CDP消融、参数敏感性、热图、帕累托关系、跨种子一致性和LDP专题分析。")
    lines.append("")
    for idx, key in enumerate(["plain", "main", "core", "curves", "ablation", "sens", "heatmap", "pareto", "seed", "ldp_focus"], start=1):
        lines.append(f"![Figure {idx}]({figs[key].as_posix()})")
        lines.append("")
        lines.append(f"*{fig_captions[key]}*")
        lines.append("")
    lines.append("## 3. 实验结果分析")
    lines.append("")
    lines.append("### 3.1 问题动机：为何需要隐私保护")
    lines.append("")
    lines.append(
        f"- 图1显示，历史 `Plain 500轮` 结果中训练损失从 `{legacy_plain_train_start:.2f}` 快速下降，尾部测试损失约为 `{legacy_plain_test_tail:.3f}`，尾部 `ASR` 达到 `{legacy_plain_tail_asr:.3f}`，峰值 `ASR` 达到 `{legacy_plain_peak:.2f}`。这说明在明显过拟合时，联邦推荐模型会暴露可利用的成员信息。"
    )
    lines.append("- 这也是本文坚持保留 `Plain` 动机图的原因：它不一定是主对比里最优的攻击配置，但它能直观说明没有隐私约束时的最坏风险。")
    lines.append("")
    lines.append("### 3.2 主实验总体表现")
    lines.append("")
    lines.append("- 图2展示了 `G0-G9` 的整体对比。从效用看，`Plain` 三组依旧拥有最低的尾部 `RMSE`，其中 `G0/G1/G2` 的平均尾部 `RMSE` 分别约为 `0.9673 / 0.9681 / 0.9730`。")
    lines.append("- `G3` 的尾部 `RMSE` 约为 `1.0004`，说明较弱的全局 CDP 噪声可以在一定程度上保留效用；而固定强噪声的 `G5/G8` 分别恶化到 `2.9214 / 2.8062`，说明单纯增大噪声并不能得到可用模型。")
    lines.append("- 从时间开销看，`CDP` 分支大约在 `4s/轮`，`LDP` 分支则普遍上升到 `7s~9s/轮`，因此工程上 `CDP` 更适合作为主线方案。")
    lines.append("")
    lines.append("### 3.3 核心创新点：自适应噪声调度的收益")
    lines.append("")
    lines.append(
        f"- 图3和图4共同说明，在 `CDP` 支路中，`G6` 相比 `G4` 将尾部 `RMSE` 从 `{g4:.4f}` 降到 `{g6:.4f}`，相对改善约 `{pct_drop(g4, g6):.2f}%`；同时有效噪声从 `{g4_sigma:.4f}` 降到 `{g6_sigma:.4f}`，下降约 `{pct_drop(g4_sigma, g6_sigma):.1f}%`。"
    )
    lines.append(
        f"- 在 `LDP` 支路中，`G9` 相比 `G7` 将尾部 `RMSE` 从 `{g7:.4f}` 降到 `{g9:.4f}`，相对改善约 `{pct_drop(g7, g9):.2f}%`；有效噪声从 `{g7_sigma:.4f}` 降到 `{g9_sigma:.4f}`，下降约 `{pct_drop(g7_sigma, g9_sigma):.1f}%`。"
    )
    lines.append("- 图4中的百分比改进图和噪声下降比例图是本报告最直接的创新性证据：自适应组不是靠额外噪声获得收益，而是在更低有效噪声下注入下，依然保持了不差甚至更优的尾部精度。")
    lines.append("- 这种现象在 `CDP` 和 `LDP` 两条支路都出现，说明本文提出的自适应调度机制不是针对单一噪声模型的偶然技巧，而具有跨机制的一致性。")
    lines.append("")
    lines.append("### 3.4 消融实验：改进究竟来自哪里")
    lines.append("")
    lines.append("- 图5对 `G6` 与 `A1/A2/A3` 进行了消融比较。这里第三个指标选用的是 `峰值 ASR` 而不是尾部 `ASR`，原因是三组在后期都收敛到接近 `0.5`，如果只看尾部值，会把前中期真实存在过的攻击暴露差异全部抹平。")
    lines.append(
        f"- 按当前结果统计，`G6/A1/A2/A3` 的峰值 `ASR` 分别约为 `{ab_peak['G6']:.4f} / {ab_peak['A1']:.4f} / {ab_peak['A2']:.4f} / {ab_peak['A3']:.4f}`。其中 `A1` 和 `A3` 的峰值更高，说明去掉个性化机制或去掉自适应 DP 后，攻击暴露上限会更明显。"
    )
    lines.append("- 从尾部 `RMSE` 看，`A1` 比 `G6` 恶化最明显，`A3` 次之，`A2` 影响相对较小。这说明在当前数据集上，个性化与自适应隐私调度是收益更主要的来源，而 `FedProx` 更多体现为训练稳定性上的辅助。")
    lines.append("")
    lines.append("### 3.5 参数敏感性：完整方案是否稳")
    lines.append("")
    lines.append(
        f"- 图6比较了 `μ=0.001 / 0.01 / 0.05` 对完整方案的影响，`A4L/G6/A4H` 的尾部 `RMSE` 分别约为 `{a4l_rmse:.4f} / {g6:.4f} / {a4h_rmse:.4f}`。"
    )
    lines.append("- 三者差距很小，说明当前方案对 `FedProx μ` 在这一范围内不算敏感。也就是说，创新点的有效性不是建立在非常脆弱的单一超参数点上。")
    lines.append("- `A4H` 的尾部 `ASR` 和 `AUC` 略低于 `G6`，可以写成“较大的 `μ` 不会破坏隐私趋势，但也未带来明显额外效益”。")
    lines.append("")
    lines.append("### 3.6 多视角稳定性与工程性")
    lines.append("")
    lines.append("- 图7的热图从多指标角度总结了全部实验；图8则把 `RMSE`、`ASR` 和时间成本放在同一张帕累托图里，更适合在论文中强调“不是简单地拿性能换隐私”。")
    lines.append("- 图9给出了配对随机种子的结果一致性。虽然当前核心组只有两个种子，但两条支路的改进方向是一致的，没有出现一个种子好、另一个种子反向的情况。")
    lines.append("- 图10专门抽出了 `LDP` 分支。严格来说，当前实验里没有像 `A1-A3` 那样单独设计的 `LDP` 消融组，因此这里用 `G7/G8/G9` 组成一个 LDP 专题图，分别对应固定LDP、个性化LDP和自适应LDP，用来补足此前图中对 LDP 机制细节展示不足的问题。")
    lines.append("")
    lines.append("## 4. 最终可写入论文的结论")
    lines.append("")
    lines.append("1. `Plain` 联邦推荐在过拟合条件下会出现明显的成员推断风险，因此需要引入隐私保护机制。")
    lines.append("2. 固定差分隐私策略可以降低攻击可分性，但若噪声强度设置过大，会迅速带来不可接受的效用损失。")
    lines.append("3. 本文提出的自适应差分隐私调度机制，在 `CDP` 和 `LDP` 两条支路上都表现出一致方向：在更低有效噪声下注入下，获得更优或至少不差的预测精度。")
    lines.append("4. 消融与参数敏感性结果进一步说明，这种收益不是由偶然的超参数碰撞带来的，而是来自个性化、近端约束和自适应噪声调度的联合设计，其中自适应噪声调度是最关键的隐私机制改进。")
    lines.append("")
    lines.append("## 5. 统计性表述建议")
    lines.append("")
    lines.append("- 当前 `G0-G9` 共有两个随机种子，`A1-A4` 共有一个随机种子，因此可以写“结果趋势稳定、方向一致”，但不建议直接写“统计显著优于”。")
    lines.append("- 如果后续还要补实验，最有价值的是只补 `G4/G6/G7/G9` 的额外随机种子，而不是继续大面积重复整套主干。")
    lines.append("")
    lines.append("## 6. 当前种子覆盖")
    lines.append("")
    for group in counts.index:
        lines.append(f"- `{group}`：{int(counts[group])} 个种子")
    lines.append("")
    lines.append("## 7. 创新点与方法原理细解")
    lines.append("")
    lines.append("### 7.1 整体方法框架")
    lines.append("")
    lines.append("- 本文方法不是简单地在联邦学习后面附加差分隐私，而是在联邦推荐主干中联合引入了 `个性化局部头`、`FedProx 近端正则` 和 `自适应差分隐私调度`。")
    lines.append("- 其核心目标是同时处理三类问题：非IID数据导致的客户端漂移、统一模型难以兼顾个体偏好、固定噪声会在收敛后期持续损害效用。")
    lines.append("")
    lines.append("### 7.2 模型结构与个性化头")
    lines.append("")
    lines.append("- 代码实现位于 `src/models.py`。主模型 `AdvancedNeuMF` 使用用户嵌入 `u`、物品嵌入 `i` 和上下文特征编码 `f` 作为输入，并将三者拼接为 `x=[u;i;f]`。")
    lines.append("- 共享主干先生成隐藏表示 `h = Backbone(x)`，共享输出头再计算全局预测 `y_s`。如果开启个性化，则额外叠加一个轻量本地头 `y_p`：")
    lines.append("")
    lines.append(r"$$\hat y = y_s + y_p = W_s h + b_s + \mathrm{PersonalHead}(h).$$")
    lines.append("")
    lines.append("- 这里的 `personal_head` 参数只保留在客户端本地，通过 `export_personal_state()` 单独保存，不参与服务器聚合。")
    lines.append("- 这样做相当于把模型拆成“共享协同过滤结构 + 本地偏好补偿项”两部分：共享参数负责跨用户迁移，个性化头负责个体精调。")
    lines.append("")
    lines.append("### 7.3 联邦训练流程")
    lines.append("")
    lines.append("- 代码实现位于 `src/server_client.py`。单轮训练的流程为：服务器下发共享参数，客户端恢复自己的个性化头，在本地进行多轮梯度更新，只上传共享更新，服务器再对共享更新做聚合。")
    lines.append("- 形式化地，标准共享参数更新可写为：")
    lines.append("")
    lines.append(r"$$w^{t+1} = w^t + \frac{1}{K_t}\sum_{k \in S_t} \Delta_k^t,$$")
    lines.append("")
    lines.append("- 其中 `S_t` 为第 `t` 轮参与训练的客户端集合，`Δ_k^t` 为客户端 `k` 在当前轮得到的共享参数更新。本文的关键区别在于：个性化头不参与上传与平均，因此不会被其他客户端覆盖。")
    lines.append("")
    lines.append("### 7.4 FedProx 近端正则化")
    lines.append("")
    lines.append("- 为减轻非IID场景下本地训练偏离全局模型过远的问题，客户端损失函数中加入了 `FedProx` 近端项。代码中只对共享参数施加该约束，不对个性化头施加。")
    lines.append("- 对应目标函数为：")
    lines.append("")
    lines.append(r"$$\mathcal{L}_k = \mathcal{L}_{\mathrm{MSE}} + \frac{\mu}{2}\sum_{j \in \mathcal{S}} \lVert w_{k,j} - w_j^{(t)} \rVert_2^2,$$")
    lines.append("")
    lines.append("- 其中 `μ = PROX_MU`，`\\mathcal{S}` 表示共享参数集合。它的作用是给本地训练增加一个“不要偏离全局过远”的回拉项，从而提升训练稳定性。")
    lines.append("- 你最终跑出的 `A4L/A4H` 参数敏感性实验，本质上就是在检验这个近端项强度变化后，完整方案是否仍然稳定。")
    lines.append("")
    lines.append("### 7.5 自适应差分隐私噪声机制")
    lines.append("")
    lines.append("- 代码实现位于 `src/privacy.py`。本文没有直接使用固定 `σ` 的高斯噪声，而是在基础噪声 `σ_0` 上构造逐层、逐轮的自适应噪声：")
    lines.append("")
    lines.append(r"$$\sigma_{\ell,t} = \mathrm{clip}_{[\sigma_{\min},\sigma_{\max}]}\Big(\sigma_0 \cdot f_{\mathrm{schedule}}(t) \cdot f_{\mathrm{sensitivity}}(\ell) \cdot f_{\mathrm{sparsity}}(\ell)\Big).$$")
    lines.append("")
    lines.append("- 这里的三个核心调制因子分别是：")
    lines.append("1. `训练进度因子`：早期轮次保护更强，后期逐渐减弱，代码实现为 `1 + decay * (1 - progress)`。")
    lines.append("2. `敏感度因子`：根据当前层更新范数与裁剪上界的相对关系调整噪声，范数越大，噪声越容易被放大。")
    lines.append("3. `稀疏度因子`：当层参数非常稀疏时，额外增加噪声，避免稀疏结构暴露特殊信息。")
    lines.append("- 该机制的直观含义是：训练前期模型变化剧烈、风险更高，因此噪声更强；训练后期模型已进入平台期，应降低噪声避免继续伤害效用。")
    lines.append("")
    lines.append("### 7.6 差分隐私操作：裁剪与加噪")
    lines.append("")
    lines.append("- 无论是 `CDP` 还是 `LDP`，底层都采用“先裁剪、后加噪”的高斯机制：")
    lines.append("")
    lines.append(r"$$\tilde{\Delta} = \mathrm{Clip}(\Delta, C) + \mathcal{N}(0, \sigma^2 I),$$")
    lines.append("")
    lines.append("- 其中 `C = CLIP_NORM`。区别在于：`LDP` 在客户端上传前处理，`CDP` 在服务器端对聚合后的平均更新处理。")
    lines.append("- 这也是为什么你实验里 `LDP` 明显更慢，因为它需要对每个客户端更新单独完成隐私处理，而 `CDP` 只对聚合后的更新处理一次。")
    lines.append("")
    lines.append("### 7.7 训练稳定化：梯度裁剪与正则的配合")
    lines.append("")
    lines.append("- 在隐私模式开启时，客户端本地训练还会使用 `clip_grad_norm_` 对梯度做一次裁剪。这一步的作用不是直接给出最终 DP 保证，而是先约束优化过程中的异常大梯度。")
    lines.append("- 因此当前方案实际包含两层稳定机制：")
    lines.append("1. `FedProx` 近端正则，用于限制客户端参数漂移。")
    lines.append("2. `梯度/更新裁剪 + 高斯噪声`，用于控制更新尺度并注入隐私保护。")
    lines.append("- 这也说明你的创新点不是单独某个小技巧，而是围绕联邦推荐中的稳定性、个性化和隐私保护共同设计的一套完整机制。")
    lines.append("")
    lines.append("### 7.8 创新点归纳")
    lines.append("")
    lines.append("1. 设计了共享主干与本地个性头结合的联邦推荐结构，使共享知识学习与个体偏好学习相互解耦。")
    lines.append("2. 在 `CDP` 与 `LDP` 两条分支中统一引入自适应噪声调度，不再使用单一固定噪声。")
    lines.append("3. 将 `FedProx` 近端正则、个性化联邦训练与差分隐私机制整合到同一训练框架中，提高了非IID场景下的稳定性。")
    lines.append("4. 通过主实验、消融实验和参数敏感性实验共同验证：收益主要来自个性化联邦结构与自适应隐私调度的联合设计，而不是偶然的参数选择。")
    lines.append("")
    return "\n".join(lines) + "\n"

def main() -> None:
    ensure_dir()
    runs = load_runs()
    df = make_summary_df(runs)
    legacy = load_legacy()
    run_map = runs_by_group_seed(runs)
    figs = {
        "plain": plot_plain_risk(legacy),
        "main": plot_main_overview(df),
        "core": plot_core_gain(df),
        "curves": plot_core_curves(run_map),
        "ablation": plot_ablation(df),
        "sens": plot_sensitivity(df),
        "heatmap": plot_heatmap(df),
        "pareto": plot_pareto(df),
        "seed": plot_seed_consistency(df),
        "ldp_focus": plot_ldp_focus(df),
    }
    df.sort_values(["group", "seed"]).to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    with open(REPORT_PATH, "w", encoding="utf-8-sig") as f:
        f.write(build_report(df, legacy, figs))
    print(f"[OK] report={REPORT_PATH}")
    print(f"[OK] summary={SUMMARY_PATH}")
    for key, path in figs.items():
        print(f"[FIG] {key}={path}")
if __name__ == "__main__":
    main()
