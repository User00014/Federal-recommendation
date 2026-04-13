from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.dataset import load_all_data
from src.models import AdvancedNeuMF, is_personalized_param
from src.report_support import configure_report_plot_style, ensure_dirs
from src.server_client import Client, Server


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figures" / "architecture_real_benchmark_20260413"
REPORT_PATH = ROOT / "reports" / "隐私架构轻量真实对比实验_20260413.md"
CSV_PATH = OUT_DIR / "轻量真实对比实验汇总.csv"


SEED = 42
NUM_BENCH_USERS = 8
CLIENTS_PER_ROUND = 4
ROUNDS = 3
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
EMB_DIM = 8
LR = 3e-4
PROX_MU = 0.01
HE_SCALE = 10_000
MPC_SCALE = 10_000
MPC_MOD = 2**61 - 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def state_bytes(state: Dict[str, torch.Tensor]) -> int:
    return int(sum(tensor_bytes(v) for v in state.values() if torch.is_tensor(v)))


def build_benchmark_subset():
    train_data, test_data, stats, _ = load_all_data("data", random_seed=SEED)
    lengths = sorted(train_data.items(), key=lambda kv: (abs(len(kv[1]) - np.median([len(v) for v in train_data.values()])), kv[0]))
    selected_users = [uid for uid, _ in lengths[:NUM_BENCH_USERS]]

    user_map = {uid: idx for idx, uid in enumerate(selected_users)}
    item_ids = set()
    for uid in selected_users:
        for mid, _, _ in train_data[uid]:
            item_ids.add(mid)
        for mid, _, _ in test_data[uid]:
            item_ids.add(mid)
    item_ids = sorted(item_ids)
    item_map = {mid: idx for idx, mid in enumerate(item_ids)}

    sub_train = {}
    sub_test = {}
    for old_uid in selected_users:
        new_uid = user_map[old_uid]
        sub_train[new_uid] = [(item_map[mid], rating, feat) for mid, rating, feat in train_data[old_uid] if mid in item_map]
        sub_test[new_uid] = [(item_map[mid], rating, feat) for mid, rating, feat in test_data[old_uid] if mid in item_map]

    sub_stats = {
        "n_users": len(selected_users),
        "n_items": len(item_ids),
        "feature_dim": stats["feature_dim"],
        "total_train": sum(len(v) for v in sub_train.values()),
        "total_test": sum(len(v) for v in sub_test.values()),
        "selected_users": selected_users,
    }
    return sub_train, sub_test, sub_stats


def build_round_schedule() -> List[List[int]]:
    return [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 2, 3],
    ]


def make_config(num_users: int, num_items: int, feature_dim: int, privacy_mode: str, adaptive: bool) -> SimpleNamespace:
    return SimpleNamespace(
        NUM_USERS=num_users,
        NUM_ITEMS=num_items,
        FEATURE_DIM=feature_dim,
        EMBEDDING_DIM=EMB_DIM,
        BATCH_SIZE=BATCH_SIZE,
        LOCAL_EPOCHS=LOCAL_EPOCHS,
        LR=LR,
        FL_ALGO="FEDPROX",
        PROX_MU=PROX_MU,
        ENABLE_PERSONALIZATION=True,
        PRIVACY_MODE=privacy_mode,
        DP_SIGMA=0.005,
        CLIP_NORM=0.005,
        ENABLE_ADAPTIVE_DP=adaptive,
        DP_SIGMA_MIN=0.001,
        DP_SIGMA_MAX=0.10,
        DP_PROGRESSIVE_DECAY=0.40,
        DP_SPARSITY_BOOST=0.15,
        ATTACK_ENABLED=False,
        USERS_PER_ROUND=CLIENTS_PER_ROUND,
        RANDOM_SEED=SEED,
    )


def evaluate_server(server: Server, cfg: SimpleNamespace, train_data: Dict[int, list], test_data: Dict[int, list]) -> float:
    losses = []
    for uid in sorted(test_data.keys()):
        client = Client(
            uid,
            train_data[uid],
            test_data[uid],
            cfg,
            cfg.FEATURE_DIM,
            personal_state=server.get_personal_state(uid),
        )
        client.model.load_state_dict(server.get_state(), strict=False)
        client.model.load_personal_state(server.get_personal_state(uid))
        losses.append(client.evaluate())
    return float(math.sqrt(max(float(np.mean(losses)), 0.0)))


def shared_key_order(model_state: Dict[str, torch.Tensor]) -> List[str]:
    return [k for k in model_state.keys() if not is_personalized_param(k)]


def updates_to_vector(update: Dict[str, torch.Tensor], key_order: Sequence[str]) -> np.ndarray:
    chunks = []
    for key in key_order:
        chunks.append(update[key].detach().cpu().numpy().reshape(-1))
    return np.concatenate(chunks, axis=0)


def vector_to_update(vector: np.ndarray, template_state: Dict[str, torch.Tensor], key_order: Sequence[str]) -> Dict[str, torch.Tensor]:
    out = {}
    cursor = 0
    for key in key_order:
        ref = template_state[key]
        n = ref.numel()
        block = vector[cursor: cursor + n].reshape(ref.shape)
        out[key] = torch.tensor(block, dtype=ref.dtype, device=ref.device)
        cursor += n
    return out


class ToyPaillier:
    def __init__(self):
        p = 18446744073709551557
        q = 18446744073709551533
        self.n = p * q
        self.n_sq = self.n * self.n
        self.g = self.n + 1
        self.lmbd = (p - 1) * (q - 1)
        x = pow(self.g, self.lmbd, self.n_sq)
        l_val = (x - 1) // self.n
        self.mu = pow(l_val, -1, self.n)

    def encode(self, value: int) -> int:
        return value % self.n

    def decode(self, value: int) -> int:
        if value > self.n // 2:
            return value - self.n
        return value

    def encrypt(self, value: int) -> int:
        m = self.encode(value)
        while True:
            r = random.randrange(1, self.n)
            if math.gcd(r, self.n) == 1:
                break
        return (pow(self.g, m, self.n_sq) * pow(r, self.n, self.n_sq)) % self.n_sq

    def add(self, c1: int, c2: int) -> int:
        return (c1 * c2) % self.n_sq

    def decrypt(self, ciphertext: int) -> int:
        x = pow(ciphertext, self.lmbd, self.n_sq)
        l_val = (x - 1) // self.n
        return self.decode((l_val * self.mu) % self.n)

    def byte_len(self, ciphertext: int) -> int:
        return max(1, (ciphertext.bit_length() + 7) // 8)


def aggregate_he(updates: List[Dict[str, torch.Tensor]], template_state: Dict[str, torch.Tensor], key_order: Sequence[str]) -> Tuple[Dict[str, torch.Tensor], int]:
    paillier = ToyPaillier()
    vectors = [updates_to_vector(update, key_order) for update in updates]
    int_vectors = [np.round(vec * HE_SCALE).astype(np.int64) for vec in vectors]
    total_upload = 0
    encrypted_sums = None

    for int_vec in int_vectors:
        encrypted = []
        for value in int_vec.tolist():
            c = paillier.encrypt(int(value))
            total_upload += paillier.byte_len(c)
            encrypted.append(c)
        if encrypted_sums is None:
            encrypted_sums = encrypted
        else:
            encrypted_sums = [paillier.add(a, b) for a, b in zip(encrypted_sums, encrypted)]

    summed = np.array([paillier.decrypt(c) for c in encrypted_sums], dtype=np.float64)
    avg = summed / (len(updates) * HE_SCALE)
    return vector_to_update(avg, template_state, key_order), total_upload


def aggregate_mpc(updates: List[Dict[str, torch.Tensor]], template_state: Dict[str, torch.Tensor], key_order: Sequence[str]) -> Tuple[Dict[str, torch.Tensor], int]:
    vectors = [updates_to_vector(update, key_order) for update in updates]
    int_vectors = [np.round(vec * MPC_SCALE).astype(np.int64) for vec in vectors]

    share_upload = 0
    sums = [None, None, None]
    for int_vec in int_vectors:
        encoded = np.mod(int_vec, MPC_MOD).astype(np.int64)
        r1 = np.random.randint(0, MPC_MOD, size=encoded.shape, dtype=np.int64)
        r2 = np.random.randint(0, MPC_MOD, size=encoded.shape, dtype=np.int64)
        r3 = (encoded - r1 - r2) % MPC_MOD
        shares = [r1, r2, r3]
        share_upload += int(sum(arr.nbytes for arr in shares))
        for idx, arr in enumerate(shares):
            if sums[idx] is None:
                sums[idx] = arr.copy()
            else:
                sums[idx] = (sums[idx] + arr) % MPC_MOD

    summed = (sums[0] + sums[1] + sums[2]) % MPC_MOD
    signed = np.where(summed > MPC_MOD // 2, summed - MPC_MOD, summed).astype(np.float64)
    avg = signed / (len(updates) * MPC_SCALE)
    return vector_to_update(avg, template_state, key_order), share_upload


@dataclass
class MethodResult:
    architecture: str
    total_time_s: float
    avg_round_time_s: float
    total_comm_mib: float
    sync_events: int
    final_rmse: float
    notes: str


def run_federated_method(name: str, privacy_mode: str, adaptive: bool, agg_mode: str,
                         train_data: Dict[int, list], test_data: Dict[int, list], stats: dict,
                         schedule: List[List[int]]) -> MethodResult:
    cfg = make_config(stats["n_users"], stats["n_items"], stats["feature_dim"], privacy_mode, adaptive)
    model = AdvancedNeuMF(
        cfg.NUM_USERS,
        cfg.NUM_ITEMS,
        cfg.FEATURE_DIM,
        emb_dim=cfg.EMBEDDING_DIM,
        enable_personalization=cfg.ENABLE_PERSONALIZATION,
    )
    server = Server(model, cfg)
    key_order = shared_key_order(server.get_state())
    shared_state_template = {k: v.clone() for k, v in server.get_state().items() if k in key_order}
    model_download_bytes = state_bytes(shared_state_template)

    total_comm = 0
    sync_events = 0
    round_times = []
    started = time.perf_counter()

    for round_idx, active_users in enumerate(schedule):
        r0 = time.perf_counter()
        global_state = server.get_state()
        total_comm += model_download_bytes * len(active_users)
        sync_events += len(active_users)

        updates = []
        for uid in active_users:
            client = Client(
                uid,
                train_data[uid],
                test_data[uid],
                cfg,
                cfg.FEATURE_DIM,
                personal_state=server.get_personal_state(uid),
            )
            update, _, _ = client.train(global_state, round_idx=round_idx, total_rounds=len(schedule))
            server.set_personal_state(uid, client.export_personal_state())
            updates.append(update)

        if agg_mode == "native":
            total_comm += sum(state_bytes(update) for update in updates)
            sync_events += len(active_users)
            server.aggregate(updates, round_idx=round_idx, total_rounds=len(schedule))
        elif agg_mode == "he":
            aggregated, upload_bytes = aggregate_he(updates, shared_state_template, key_order)
            total_comm += upload_bytes
            sync_events += len(active_users)
            state = server.model.state_dict()
            with torch.no_grad():
                for key in key_order:
                    state[key] += aggregated[key].to(state[key].device)
            server.model.load_state_dict(state)
        elif agg_mode == "mpc":
            aggregated, upload_bytes = aggregate_mpc(updates, shared_state_template, key_order)
            total_comm += upload_bytes
            sync_events += len(active_users) * 3
            state = server.model.state_dict()
            with torch.no_grad():
                for key in key_order:
                    state[key] += aggregated[key].to(state[key].device)
            server.model.load_state_dict(state)
        else:
            raise ValueError(f"Unknown agg_mode: {agg_mode}")

        round_times.append(time.perf_counter() - r0)

    total_time = time.perf_counter() - started
    final_rmse = evaluate_server(server, cfg, train_data, test_data)
    notes_map = {
        "OURS": "真实运行：个性化联邦学习 + FedProx + 自适应 CDP",
        "TEE": "真实运行：可信服务器聚合原型，无额外 DP 噪声",
        "HE": "真实运行：轻量 Paillier 同态加和聚合原型",
        "MPC": "真实运行：三方加法秘密分享聚合原型",
    }
    return MethodResult(
        architecture=name,
        total_time_s=total_time,
        avg_round_time_s=float(np.mean(round_times)),
        total_comm_mib=total_comm / (1024 * 1024),
        sync_events=sync_events,
        final_rmse=final_rmse,
        notes=notes_map[name],
    )


class SplitClientBottom(nn.Module):
    def __init__(self, num_users: int, num_items: int, feature_dim: int, emb_dim: int = EMB_DIM):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.feat_encoder = nn.Linear(feature_dim, emb_dim)
        self.bottom = nn.Sequential(
            nn.Linear(emb_dim * 3, 64),
            nn.ReLU(),
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor, feat_vecs: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        f = self.feat_encoder(feat_vecs)
        x = torch.cat([u, i, f], dim=1)
        return self.bottom(x)


class SplitServerTop(nn.Module):
    def __init__(self):
        super().__init__()
        self.top = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, smashed: torch.Tensor) -> torch.Tensor:
        return self.top(smashed).squeeze(-1)


def run_split_method(train_data: Dict[int, list], test_data: Dict[int, list], stats: dict,
                     schedule: List[List[int]]) -> MethodResult:
    device = torch.device("cpu")
    client_bottom = SplitClientBottom(stats["n_users"], stats["n_items"], stats["feature_dim"]).to(device)
    server_top = SplitServerTop().to(device)
    bottom_opt = torch.optim.Adam(client_bottom.parameters(), lr=LR)
    top_opt = torch.optim.Adam(server_top.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    total_comm = 0
    sync_events = 0
    round_times = []
    started = time.perf_counter()

    for active_users in schedule:
        r0 = time.perf_counter()
        for uid in active_users:
            samples = train_data[uid]
            mids, rates, feats = zip(*samples)
            u_t = torch.LongTensor([uid] * len(mids)).to(device)
            i_t = torch.LongTensor(mids).to(device)
            r_t = torch.FloatTensor(rates).to(device)
            f_t = torch.FloatTensor(np.array(feats)).to(device)

            perm = torch.randperm(len(mids), device=device)
            for start in range(0, len(mids), BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                bu, bi, br, bf = u_t[idx], i_t[idx], r_t[idx], f_t[idx]

                bottom_opt.zero_grad()
                top_opt.zero_grad()

                smashed = client_bottom(bu, bi, bf)
                detached = smashed.detach().requires_grad_(True)
                pred = server_top(detached)
                loss = loss_fn(pred, br)
                loss.backward()

                grad = detached.grad.detach()
                smashed.backward(grad)

                top_opt.step()
                bottom_opt.step()

                total_comm += tensor_bytes(detached) + tensor_bytes(grad)
                sync_events += 2
        round_times.append(time.perf_counter() - r0)

    losses = []
    client_bottom.eval()
    server_top.eval()
    with torch.no_grad():
        for uid in sorted(test_data.keys()):
            if not test_data[uid]:
                continue
            mids, rates, feats = zip(*test_data[uid])
            u_t = torch.LongTensor([uid] * len(mids)).to(device)
            i_t = torch.LongTensor(mids).to(device)
            r_t = torch.FloatTensor(rates).to(device)
            f_t = torch.FloatTensor(np.array(feats)).to(device)
            pred = server_top(client_bottom(u_t, i_t, f_t))
            losses.append(loss_fn(pred, r_t).item())

    total_time = time.perf_counter() - started
    final_rmse = float(math.sqrt(max(float(np.mean(losses)), 0.0)))
    return MethodResult(
        architecture="SPLIT",
        total_time_s=total_time,
        avg_round_time_s=float(np.mean(round_times)),
        total_comm_mib=total_comm / (1024 * 1024),
        sync_events=sync_events,
        final_rmse=final_rmse,
        notes="真实运行：批次级拆分学习原型，切分点位于第一层共享表示之后",
    )


def plot_runtime_comm(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=220)
    colors = ["#2b6cb0", "#4a5568", "#b45309", "#c53030", "#2f855a"]
    axes[0].bar(df["architecture"], df["total_time_s"], color=colors, edgecolor="#333", linewidth=0.8)
    axes[0].set_title("总运行时间对比")
    axes[0].set_ylabel("seconds")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(df["architecture"], df["total_comm_mib"], color=colors, edgecolor="#333", linewidth=0.8)
    axes[1].set_title("总通信量对比")
    axes[1].set_ylabel("MiB")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("图11  五类架构轻量真实运行结果：时间与通信量", fontsize=17, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig11_real_arch_runtime_comm.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sync_rmse(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=220)
    colors = ["#2b6cb0", "#4a5568", "#b45309", "#c53030", "#2f855a"]
    axes[0].bar(df["architecture"], df["sync_events"], color=colors, edgecolor="#333", linewidth=0.8)
    axes[0].set_title("同步事件总数对比")
    axes[0].set_ylabel("count")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(df["architecture"], df["final_rmse"], color=colors, edgecolor="#333", linewidth=0.8)
    axes[1].set_title("最终 RMSE 对比（轻量原型）")
    axes[1].set_ylabel("RMSE")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("图12  五类架构轻量真实运行结果：同步与效用", fontsize=17, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "fig12_real_arch_sync_rmse.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_report(stats: dict, df: pd.DataFrame, fig1: Path, fig2: Path) -> str:
    lines = []
    lines.append("# 隐私架构轻量真实对比实验")
    lines.append("")
    lines.append("## 1. 说明")
    lines.append("")
    lines.append("本实验为真实运行的轻量原型对比实验，不再采用纯估算代理表。为控制实验时长，本文在当前数据集上抽取 `8` 个用户构成小规模基准集，并使用缩小版模型真实执行 `3` 轮对比训练。")
    lines.append("")
    lines.append("需要说明的是：")
    lines.append("")
    lines.append("1. 该实验是“轻量真实运行原型”，不是工业级密码学系统。")
    lines.append("2. `HE` 与 `MPC` 均为可运行的轻量原型实现，用于测量真实运行时间、真实通信量和真实同步次数。")
    lines.append("3. 该实验关注点是复杂性，不以最终精度为主。")
    lines.append("")
    lines.append("## 2. 真实运行参数")
    lines.append("")
    lines.append(f"- 基准用户数：`{stats['n_users']}`")
    lines.append(f"- 基准物品数：`{stats['n_items']}`")
    lines.append(f"- 训练样本数：`{stats['total_train']}`")
    lines.append(f"- 测试样本数：`{stats['total_test']}`")
    lines.append(f"- 通信轮数：`{ROUNDS}`")
    lines.append(f"- 每轮客户端数：`{CLIENTS_PER_ROUND}`")
    lines.append(f"- 本地训练轮数：`{LOCAL_EPOCHS}`")
    lines.append(f"- 批大小：`{BATCH_SIZE}`")
    lines.append(f"- 嵌入维度：`{EMB_DIM}`")
    lines.append(f"- 学习率：`{LR}`")
    lines.append(f"- 本文方法：`FedProx + Personalization + Adaptive CDP`")
    lines.append(f"- HE 原型：`Paillier` 加法同态聚合，量化比例 `1/{HE_SCALE}`")
    lines.append(f"- MPC 原型：三方加法秘密分享，量化比例 `1/{MPC_SCALE}`")
    lines.append("- TEE 原型：可信服务器聚合，不加额外 DP 噪声")
    lines.append("- 拆分学习原型：切分点位于第一层共享表示之后")
    lines.append("")
    lines.append("## 3. 结果表")
    lines.append("")
    table_df = df[[
        "architecture", "total_time_s", "avg_round_time_s", "total_comm_mib", "sync_events", "final_rmse", "notes"
    ]].rename(columns={
        "architecture": "架构",
        "total_time_s": "总时间(s)",
        "avg_round_time_s": "平均轮时间(s)",
        "total_comm_mib": "总通信量(MiB)",
        "sync_events": "同步事件数",
        "final_rmse": "最终RMSE",
        "notes": "说明",
    }).round({
        "总时间(s)": 3,
        "平均轮时间(s)": 3,
        "总通信量(MiB)": 3,
        "最终RMSE": 4,
    })
    headers = list(table_df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in table_df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    lines.append("")
    lines.append(f"![Figure 11]({fig1.as_posix()})")
    lines.append("")
    lines.append("*图11  真实运行的五类架构在总时间和总通信量上的差异。*")
    lines.append("")
    lines.append(f"![Figure 12]({fig2.as_posix()})")
    lines.append("")
    lines.append("*图12  真实运行的五类架构在同步次数和最终 RMSE 上的差异。*")
    lines.append("")
    lines.append("## 4. 结果分析")
    lines.append("")
    ours = df[df["architecture"] == "OURS"].iloc[0]
    tee = df[df["architecture"] == "TEE"].iloc[0]
    he = df[df["architecture"] == "HE"].iloc[0]
    mpc = df[df["architecture"] == "MPC"].iloc[0]
    split = df[df["architecture"] == "SPLIT"].iloc[0]
    lines.append(f"- 本文方法真实运行 `3` 轮总时间为 `{ours['total_time_s']:.3f}s`，总通信量为 `{ours['total_comm_mib']:.3f} MiB`，同步事件数为 `{int(ours['sync_events'])}`。")
    lines.append(f"- TEE 原型总时间为 `{tee['total_time_s']:.3f}s`，与本文方法接近，说明如果只从运行效率看，可信服务器聚合确实是最接近本文方案的替代方向。")
    lines.append(f"- HE 原型总时间为 `{he['total_time_s']:.3f}s`，约为本文方法的 `{he['total_time_s'] / ours['total_time_s']:.2f}` 倍；总通信量为 `{he['total_comm_mib']:.3f} MiB`，约为本文方法的 `{he['total_comm_mib'] / ours['total_comm_mib']:.2f}` 倍。即使在缩小版模型和仅 `3` 轮设置下，HE 仍然明显更重。")
    lines.append(f"- MPC 原型总时间为 `{mpc['total_time_s']:.3f}s`，总通信量为 `{mpc['total_comm_mib']:.3f} MiB`，同步事件数为 `{int(mpc['sync_events'])}`，均显著高于本文方法，说明多方秘密分享在迭代训练任务中会持续累积额外交互成本。")
    lines.append(f"- 拆分学习原型总通信量为 `{split['total_comm_mib']:.3f} MiB`，表面上并不高，但同步事件数达到 `{int(split['sync_events'])}`，远高于本文方法的 `{int(ours['sync_events'])}`。这说明在当前场景下，拆分学习真正的问题不是消息本身大小，而是批次级高频同步。")
    lines.append("")
    lines.append("## 5. 本节结论")
    lines.append("")
    lines.append("轻量真实运行实验进一步证明：对于当前联邦推荐任务，与 `HE/MPC/TEE/拆分学习` 相比，本文架构并不是唯一可行方案，但它在真实运行条件下仍然是更均衡的选择。`TEE` 在复杂度上接近本文方法，但依赖可信硬件与中心可信假设；`HE` 和 `MPC` 在真实运行中明显更重；`拆分学习` 则受制于高频同步。因而在本论文场景中，采用“个性化联邦学习 + FedProx + 自适应差分隐私”作为主线架构仍然是更合理的技术选择。")
    return "\n".join(lines) + "\n"


def main() -> None:
    set_seed(SEED)
    configure_report_plot_style()
    ensure_dirs(OUT_DIR, REPORT_PATH.parent)
    train_data, test_data, stats = build_benchmark_subset()
    schedule = build_round_schedule()

    results = [
        run_federated_method("OURS", "CDP", True, "native", train_data, test_data, stats, schedule),
        run_federated_method("TEE", "PLAIN", False, "native", train_data, test_data, stats, schedule),
        run_federated_method("HE", "PLAIN", False, "he", train_data, test_data, stats, schedule),
        run_federated_method("MPC", "PLAIN", False, "mpc", train_data, test_data, stats, schedule),
        run_split_method(train_data, test_data, stats, schedule),
    ]

    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    fig1 = plot_runtime_comm(df)
    fig2 = plot_sync_rmse(df)
    report = build_report(stats, df, fig1, fig2)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(df)
    print(f"[OK] csv: {CSV_PATH}")
    print(f"[OK] report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
