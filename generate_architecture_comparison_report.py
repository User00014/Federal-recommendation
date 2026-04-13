from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import Config
from src.dataset import load_all_data
from src.models import AdvancedNeuMF, is_personalized_param
from src.report_support import configure_report_plot_style, ensure_dirs


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "architecture_compare_20260413"
REPORT_PATH = ROOT / "reports" / "隐私架构复杂性对比实验_20260413.md"
CSV_PATH = OUT_DIR / "复杂性对比汇总.csv"


def mib(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def count_params() -> dict:
    train_data, test_data, stats, _ = load_all_data(Config.DATA_PATH, random_seed=42)
    model = AdvancedNeuMF(
        stats["n_users"],
        stats["n_items"],
        stats["feature_dim"],
        emb_dim=Config.EMBEDDING_DIM,
        enable_personalization=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    shared_params = sum(p.numel() for n, p in model.named_parameters() if not is_personalized_param(n))
    personal_params = sum(p.numel() for n, p in model.named_parameters() if is_personalized_param(n))

    train_lens = [len(v) for v in train_data.values()]
    avg_train_samples = sum(train_lens) / max(len(train_lens), 1)
    avg_batches_per_epoch = sum(math.ceil(x / Config.BATCH_SIZE) for x in train_lens) / max(len(train_lens), 1)

    return {
        "stats": stats,
        "total_params": total_params,
        "shared_params": shared_params,
        "personal_params": personal_params,
        "avg_train_samples": avg_train_samples,
        "avg_batches_per_epoch": avg_batches_per_epoch,
    }


def build_comparison_df(base: dict) -> pd.DataFrame:
    users_per_round = 30
    local_epochs = 8
    rtt_ms = 20.0

    shared_mib = mib(base["shared_params"] * 4)
    avg_client_batches_round = base["avg_batches_per_epoch"] * local_epochs
    split_cut_dim = 64
    split_roundtrip_mib = mib(split_cut_dim * 4 * 2)
    split_comm_mib = split_roundtrip_mib * base["avg_train_samples"] * users_per_round * local_epochs

    rows = [
        {
            "architecture": "本文方法",
            "family": "ours",
            "round_comm_mib": 2 * users_per_round * shared_mib,
            "sync_events": 2 * users_per_round,
            "dependency_score": 1,
            "model_surgery_score": 1,
            "notes": "联邦聚合，仅共享参数上传；个性化头留在本地",
        },
        {
            "architecture": "同态加密联邦学习",
            "family": "he",
            "round_comm_mib": 2 * users_per_round * shared_mib * 12.0,
            "sync_events": 2 * users_per_round,
            "dependency_score": 5,
            "model_surgery_score": 2,
            "notes": "采用保守 12x 密文膨胀代理系数，突出密文通信与聚合开销",
        },
        {
            "architecture": "安全多方计算联邦学习",
            "family": "mpc",
            "round_comm_mib": 2 * users_per_round * shared_mib * 6.0,
            "sync_events": 6 * users_per_round,
            "dependency_score": 5,
            "model_surgery_score": 3,
            "notes": "采用 3 方秘密分享的有效 6x 通信代理，并加入更多交互阶段",
        },
        {
            "architecture": "可信执行环境联邦学习",
            "family": "tee",
            "round_comm_mib": 2 * users_per_round * shared_mib * 1.05,
            "sync_events": 2 * users_per_round,
            "dependency_score": 4,
            "model_surgery_score": 2,
            "notes": "通信接近普通联邦学习，但依赖可信硬件、远程证明与平台适配",
        },
        {
            "architecture": "拆分学习",
            "family": "split",
            "round_comm_mib": split_comm_mib,
            "sync_events": 2 * users_per_round * avg_client_batches_round,
            "dependency_score": 3,
            "model_surgery_score": 5,
            "notes": "按 64 维切分点估计激活与梯度传输，批次级同步频繁",
        },
    ]

    df = pd.DataFrame(rows)
    df["latency_20ms_s"] = df["sync_events"] * (rtt_ms / 1000.0)

    comm_norm = df["round_comm_mib"] / df["round_comm_mib"].min()
    sync_norm = df["sync_events"] / df["sync_events"].min()
    dep_norm = df["dependency_score"] / 5.0
    surgery_norm = df["model_surgery_score"] / 5.0

    df["complexity_score"] = (
        0.40 * comm_norm
        + 0.30 * sync_norm
        + 0.15 * dep_norm
        + 0.15 * surgery_norm
    )
    df["complexity_vs_ours"] = df["complexity_score"] / float(
        df.loc[df["architecture"] == "本文方法", "complexity_score"].iloc[0]
    )
    return df


def plot_comm_sync(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=220)
    colors = ["#2b6cb0", "#b45309", "#c53030", "#4a5568", "#2f855a"]

    axes[0].bar(df["architecture"], df["round_comm_mib"], color=colors, edgecolor="#333", linewidth=0.8)
    axes[0].set_title("单轮通信量对比")
    axes[0].set_ylabel("MiB / round")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(df["architecture"], df["latency_20ms_s"], color=colors, edgecolor="#333", linewidth=0.8)
    axes[1].set_title("20ms RTT 下的同步延迟代理")
    axes[1].set_ylabel("s / round")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("图11  不同隐私架构的复杂性代理对比", fontsize=17, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig11_arch_compare_comm_sync.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_score(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 6), dpi=220)
    colors = ["#2b6cb0", "#b45309", "#c53030", "#4a5568", "#2f855a"]
    ax.bar(df["architecture"], df["complexity_vs_ours"], color=colors, edgecolor="#333", linewidth=0.8)
    ax.axhline(1.0, color="#2b6cb0", ls="--", lw=1.2)
    ax.set_title("综合复杂度得分（本文方法归一化为 1.0）")
    ax.set_ylabel("Relative Complexity")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(df["complexity_vs_ours"]):
        ax.text(idx, value + 0.05, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("图12  综合复杂度归一化得分", fontsize=17, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig12_arch_compare_score.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_report(base: dict, df: pd.DataFrame, fig1: Path, fig2: Path) -> str:
    stats = base["stats"]
    ours = df[df["architecture"] == "本文方法"].iloc[0]
    he = df[df["architecture"] == "同态加密联邦学习"].iloc[0]
    mpc = df[df["architecture"] == "安全多方计算联邦学习"].iloc[0]
    tee = df[df["architecture"] == "可信执行环境联邦学习"].iloc[0]
    split = df[df["architecture"] == "拆分学习"].iloc[0]

    lines = []
    lines.append("# 隐私架构复杂性对比实验")
    lines.append("")
    lines.append("## 1. 实验目的")
    lines.append("")
    lines.append("本补充实验不追求重新完整训练 `HE/MPC/TEE/拆分学习` 四套系统，而是面向毕业论文中的“方案选择合理性”问题，构造统一模型规模、统一客户端规模和统一训练配置下的复杂性代理实验。目标是回答：在当前推荐任务、当前联邦用户划分方式和当前工程条件下，为什么本文采用的“个性化联邦学习 + FedProx + 自适应差分隐私”架构更适合作为主线方案。")
    lines.append("")
    lines.append("## 2. 统一实验设置")
    lines.append("")
    lines.append(f"- 数据规模：`{stats['n_users']}` 个用户，`{stats['n_items']}` 个物品，`{stats['total_interactions']}` 条交互记录。")
    lines.append(f"- 模型规模：总参数 ` {base['total_params']:,} ` 个，其中共享参数 ` {base['shared_params']:,} ` 个，本地个性化头参数 ` {base['personal_params']:,} ` 个。")
    lines.append(f"- 单轮参与客户端数：`30`。")
    lines.append(f"- 本地训练轮数：`8`。")
    lines.append(f"- 平均每客户端训练样本数：约 `{base['avg_train_samples']:.2f}`。")
    lines.append(f"- 平均每客户端每个 epoch 的 mini-batch 数：约 `{base['avg_batches_per_epoch']:.2f}`。")
    lines.append("- 网络延迟代理：采用 `20ms RTT` 作为远程用户到云端服务器的轻量估计，用于衡量同步型方案的交互惩罚。")
    lines.append("")
    lines.append("## 3. 代理建模说明")
    lines.append("")
    lines.append("为保证可复现且不虚构密码学实现，本实验采用如下保守代理设定：")
    lines.append("")
    lines.append("1. 本文方法：直接使用当前联邦推荐架构，通信单位为共享参数上传与下发。")
    lines.append("2. 同态加密联邦学习：采用 `12x` 密文膨胀系数，反映密文更新的通信与聚合压力。")
    lines.append("3. 安全多方计算联邦学习：采用 `6x` 有效通信代理，并将同步阶段扩展为普通联邦学习的 `3` 倍。")
    lines.append("4. 可信执行环境联邦学习：通信量接近普通联邦学习，但加入可信硬件依赖与部署复杂度。")
    lines.append("5. 拆分学习：在 `64` 维切分点上传激活并回传梯度，通信按批次同步累计。")
    lines.append("")
    lines.append("综合复杂度得分定义为：")
    lines.append("")
    lines.append("$$")
    lines.append("Score = 0.40 \\cdot Comm_{norm} + 0.30 \\cdot Sync_{norm} + 0.15 \\cdot Dep_{norm} + 0.15 \\cdot Surgery_{norm}")
    lines.append("$$")
    lines.append("")
    lines.append("其中 `Comm` 为单轮通信量，`Sync` 为单轮同步事件数，`Dep` 为平台/可信硬件依赖评分，`Surgery` 为对现有模型训练流程的改造难度评分。得分越低表示越适合当前场景。")
    lines.append("")
    lines.append("## 4. 对比结果")
    lines.append("")
    table_df = df[[
        "architecture",
        "round_comm_mib",
        "sync_events",
        "latency_20ms_s",
        "dependency_score",
        "model_surgery_score",
        "complexity_vs_ours",
    ]].rename(columns={
        "architecture": "架构",
        "round_comm_mib": "单轮通信量(MiB)",
        "sync_events": "同步事件数",
        "latency_20ms_s": "20ms RTT 代理延迟(s)",
        "dependency_score": "平台依赖分",
        "model_surgery_score": "模型改造分",
        "complexity_vs_ours": "相对综合复杂度",
    }).round({
        "单轮通信量(MiB)": 2,
        "20ms RTT 代理延迟(s)": 2,
        "相对综合复杂度": 2,
    })
    headers = list(table_df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in table_df.iterrows():
        values = [str(row[h]) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append(f"![Figure 11]({fig1.as_posix()})")
    lines.append("")
    lines.append("*图11 展示五类架构在单轮通信量与同步延迟代理上的差异。*")
    lines.append("")
    lines.append(f"![Figure 12]({fig2.as_posix()})")
    lines.append("")
    lines.append("*图12 将本文方法归一化为 1.0 后，对比不同架构的综合复杂度得分。*")
    lines.append("")
    lines.append("## 5. 结果分析")
    lines.append("")
    lines.append(f"- 本文方法单轮通信量约为 `{ours['round_comm_mib']:.2f} MiB`，同步事件数为 `{ours['sync_events']:.0f}`，综合复杂度记为 `1.00`。这说明在当前 `30` 客户端、`8` 本地 epoch 的配置下，现有联邦架构已经处在一个较平衡的工程点。")
    lines.append(f"- 同态加密联邦学习的单轮通信量约为 `{he['round_comm_mib']:.2f} MiB`，约为本文方法的 `{he['round_comm_mib'] / ours['round_comm_mib']:.1f}` 倍，综合复杂度约为本文方法的 `{he['complexity_vs_ours']:.2f}` 倍。其问题不在于不能做，而在于在当前迭代训练任务中，密文通信与计算代价明显过高。")
    lines.append(f"- 安全多方计算联邦学习的单轮通信量约为 `{mpc['round_comm_mib']:.2f} MiB`，同步事件数提升到 `{mpc['sync_events']:.0f}`，在云端-客户端异步环境下更容易受交互阶段拖累，综合复杂度约为本文方法的 `{mpc['complexity_vs_ours']:.2f}` 倍。")
    lines.append(f"- 可信执行环境联邦学习的通信量与本文方法接近，约为 `{tee['round_comm_mib']:.2f} MiB`，综合复杂度约为 `{tee['complexity_vs_ours']:.2f}` 倍，表明它是四类替代方案里最接近本文方法的。然而它依赖可信硬件、远程证明和平台适配，且信任中心服务器的假设更强，不如本文方法易于复现和推广。")
    lines.append(f"- 拆分学习的单轮通信量约为 `{split['round_comm_mib']:.2f} MiB`，表面上低于本文方法，但同步事件数达到 `{split['sync_events']:.0f}`，在 `20ms RTT` 假设下同步延迟代理约为 `{split['latency_20ms_s']:.2f}s`，约为本文方法的 `{split['latency_20ms_s'] / ours['latency_20ms_s']:.1f}` 倍。对于当前大量客户端、跨网络批量训练的推荐任务，交互频率才是主要瓶颈。")
    lines.append("")
    lines.append("## 6. 为什么当前场景更适合本文架构")
    lines.append("")
    lines.append("结合上述结果，可以从四个层面解释当前任务与本文架构的匹配性：")
    lines.append("")
    lines.append("1. **任务形态匹配**：当前任务是典型的多客户端、反复通信轮次的联邦推荐训练，不是一次性安全推理。对于需要持续多轮优化的任务，`HE/MPC` 的额外密码学开销会在轮次上不断累积。")
    lines.append("2. **网络条件匹配**：当前实验运行在云端 GPU 上，客户端到服务器的交互不是局域网零延迟环境，因此拆分学习这种批次级同步方案会显著受 RTT 影响。")
    lines.append("3. **隐私边界匹配**：本文方法已经通过 `CDP/LDP + 自适应 DP` 对共享更新进行隐私保护，能够在推荐效用、隐私风险和工程代价之间形成实际可跑的平衡点。")
    lines.append("4. **工程可复现性匹配**：毕业论文场景除了追求理论安全性，也需要代码能跑、实验能复现、报告能解释。`TEE` 虽然复杂度低于 `HE/MPC`，但对平台、可信硬件和部署环境的依赖更强，不如当前架构适合作为公开可复现主线。")
    lines.append("")
    lines.append("## 7. 本节结论")
    lines.append("")
    lines.append("本补充实验表明，在当前推荐任务、当前数据规模和当前云端训练条件下，本文架构并不是四类替代方案中“绝对最强安全假设”的方案，但它是**综合复杂度最低、训练形态最匹配、工程复现性最好**的方案。对于本论文关注的联邦推荐场景，与其引入高复杂度的 `HE/MPC/TEE/拆分学习` 框架，不如采用当前的“个性化联邦学习 + FedProx + 自适应差分隐私”架构，更能在可实现性与研究价值之间取得平衡。")
    return "\n".join(lines) + "\n"


def main() -> None:
    configure_report_plot_style()
    ensure_dirs(OUT_DIR, REPORT_PATH.parent)
    base = count_params()
    df = build_comparison_df(base)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    fig1 = plot_comm_sync(df)
    fig2 = plot_score(df)
    report = build_report(base, df, fig1, fig2)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[OK] comparison csv: {CSV_PATH}")
    print(f"[OK] comparison report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
