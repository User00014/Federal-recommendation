import argparse
import ast
import copy
import csv
import datetime as dt
import json
import math
import os
import random
import traceback

import numpy as np
import torch

from src.attack import MembershipAttackTrainer, extract_gradient_features
from src.config import Config
from src.dataset import load_all_data
from src.experiment_io import build_result_path
from src.models import AdvancedNeuMF
from src.server_client import Client, Server


GROUP_CONFIGS = {
    "G0": {"PRIVACY_MODE": "PLAIN", "DP_SIGMA": 0.0, "FL_ALGO": "FEDAVG", "ENABLE_PERSONALIZATION": False, "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "G1": {"PRIVACY_MODE": "PLAIN", "DP_SIGMA": 0.0, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": False, "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "G2": {"PRIVACY_MODE": "PLAIN", "DP_SIGMA": 0.0, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "G3": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.002, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "G4": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.005, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "G5": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.01,  "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "G6": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.005, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.01},
    "G7": {"PRIVACY_MODE": "LDP",   "DP_SIGMA": 0.02,  "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "G8": {"PRIVACY_MODE": "LDP",   "DP_SIGMA": 0.05,  "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "G9": {"PRIVACY_MODE": "LDP",   "DP_SIGMA": 0.02,  "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.01},
    "A1": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.005, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": False, "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.01},
    "A2": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.005, "FL_ALGO": "FEDAVG",  "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.01},
    "A3": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.005, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "A4L": {"PRIVACY_MODE": "CDP",  "DP_SIGMA": 0.005, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.001},
    "A4H": {"PRIVACY_MODE": "CDP",  "DP_SIGMA": 0.005, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.05},
    "P1": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.001, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.01},
    "P2": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.003, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.01},
    "P3": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.007, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.01},
    "P4": {"PRIVACY_MODE": "CDP",   "DP_SIGMA": 0.015, "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": True,  "PROX_MU": 0.01},
    "P5": {"PRIVACY_MODE": "LDP",   "DP_SIGMA": 0.03,  "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
    "P6": {"PRIVACY_MODE": "LDP",   "DP_SIGMA": 0.08,  "FL_ALGO": "FEDPROX", "ENABLE_PERSONALIZATION": True,  "ENABLE_ADAPTIVE_DP": False, "PROX_MU": 0.01},
}

DEFAULT_GROUPS = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "A1", "A2", "A3", "A4L", "A4H"]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_snapshot():
    keys = [
        "ROUNDS", "LOCAL_EPOCHS", "LR", "EMBEDDING_DIM", "BATCH_SIZE",
        "FL_ALGO", "PROX_MU", "USERS_PER_ROUND",
        "ENABLE_PERSONALIZATION",
        "PRIVACY_MODE", "DP_SIGMA", "CLIP_NORM",
        "ENABLE_ADAPTIVE_DP", "DP_SIGMA_MIN", "DP_SIGMA_MAX",
        "DP_PROGRESSIVE_DECAY", "DP_SPARSITY_BOOST",
        "ATTACK_ENABLED", "TAIL_WINDOW", "RANDOM_SEED",
    ]
    return {k: getattr(Config, k) for k in keys}


def parse_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_override_value(raw):
    text = str(raw).strip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def parse_set_overrides(text):
    out = {}
    if not text:
        return out
    for item in parse_list(text):
        if "=" not in item:
            raise ValueError(f"--set ??????: {item}??? KEY=VALUE")
        k, v = item.split("=", 1)
        key = k.strip().upper()
        if not key:
            raise ValueError(f"--set ??????: {item}?KEY ????")
        out[key] = parse_override_value(v)
    return out


def apply_overrides(overrides):
    for k, v in overrides.items():
        setattr(Config, k, v)


def summarize_tail(history, tail_window):
    n = len(history["attack_acc"])
    tail = min(max(1, int(tail_window)), n) if n > 0 else 1
    if n == 0:
        return {"tail": 0, "tail_asr": 0.5, "tail_auc": 0.5, "tail_rmse": 0.0, "round_time_mean": 0.0}

    return {
        "tail": tail,
        "tail_asr": float(np.mean(history["attack_acc"][-tail:])),
        "tail_auc": float(np.mean(history["attack_auc"][-tail:])),
        "tail_rmse": float(np.mean(history["rmse"][-tail:])),
        "round_time_mean": float(np.mean(history["round_time"][-tail:])),
    }


class CloudBatchRunner:
    def __init__(self, args):
        self.args = args
        self.dataset_cache = {}

        self.base_config = {
            k: copy.deepcopy(getattr(Config, k))
            for k in dir(Config)
            if k.isupper()
        }

        if args.resume_dir:
            self.batch_dir = os.path.abspath(args.resume_dir)
            self.state_path = os.path.join(self.batch_dir, "batch_state.json")
            if not os.path.exists(self.state_path):
                raise FileNotFoundError(f"未找到状态文件: {self.state_path}")
            self.state = self._load_state()
        else:
            timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
            name = args.batch_name or f"cloud_batch_{timestamp}"
            self.batch_dir = os.path.abspath(os.path.join(args.output_root, name))
            self.state_path = os.path.join(self.batch_dir, "batch_state.json")
            os.makedirs(self.batch_dir, exist_ok=True)
            self.state = self._build_new_state()
            self._save_state()

        self.logs_dir = os.path.join(self.batch_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    def _build_new_state(self):
        groups = parse_list(self.args.groups) if self.args.groups else list(DEFAULT_GROUPS)
        seeds = [int(x) for x in parse_list(self.args.seeds)]
        extra_overrides = parse_set_overrides(self.args.set)

        runs = []
        idx = 1
        for seed in seeds:
            for group in groups:
                if group not in GROUP_CONFIGS:
                    raise ValueError(f"未知实验组: {group}")

                cfg = dict(GROUP_CONFIGS[group])
                cfg["ROUNDS"] = int(self.args.rounds)
                cfg["RANDOM_SEED"] = int(seed)
                cfg["ATTACK_ENABLED"] = bool(not self.args.disable_attack)

                if self.args.users_per_round is not None:
                    cfg["USERS_PER_ROUND"] = int(self.args.users_per_round)
                if self.args.local_epochs is not None:
                    cfg["LOCAL_EPOCHS"] = int(self.args.local_epochs)
                cfg.update(extra_overrides)

                runs.append({
                    "id": f"R{idx:03d}",
                    "group": group,
                    "seed": int(seed),
                    "config": cfg,
                    "status": "pending",
                    "started_at": None,
                    "ended_at": None,
                    "duration_sec": None,
                    "log_path": None,
                    "tail_metrics": None,
                    "error": None,
                })
                idx += 1

        return {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "batch_dir": self.batch_dir,
            "args": vars(self.args),
            "runs": runs,
        }

    def _load_state(self):
        with open(self.state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_state(self):
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def _restore_base_config(self):
        for k, v in self.base_config.items():
            setattr(Config, k, copy.deepcopy(v))

    def _load_dataset_for_seed(self, seed):
        if seed in self.dataset_cache:
            return self.dataset_cache[seed]

        set_random_seed(seed)
        train_data, test_data, stats, _ = load_all_data(Config.DATA_PATH, random_seed=seed)
        self.dataset_cache[seed] = (train_data, test_data, stats)
        return self.dataset_cache[seed]

    def _init_server_attacker(self, train_data):
        global_model = AdvancedNeuMF(
            Config.NUM_USERS,
            Config.NUM_ITEMS,
            Config.FEATURE_DIM,
            emb_dim=Config.EMBEDDING_DIM,
            enable_personalization=Config.ENABLE_PERSONALIZATION,
        )
        server = Server(global_model, Config)

        attacker = None
        if Config.ATTACK_ENABLED:
            first_uid = next(iter(train_data.keys()))
            dummy_client = Client(
                first_uid,
                train_data[first_uid],
                [],
                Config,
                Config.FEATURE_DIM,
                personal_state=server.get_personal_state(first_uid),
            )
            dummy_update, _, _ = dummy_client.train(server.get_state(), round_idx=0, total_rounds=1)
            if dummy_update:
                feat_dim = len(extract_gradient_features(dummy_update))
                attacker = MembershipAttackTrainer(num_features=feat_dim, lr=float(self.args.attack_lr), buffer_size=int(self.args.attack_buffer_size))

        return server, attacker

    def _run_single(self, run_def):
        run_id = run_def["id"]
        group = run_def["group"]
        seed = int(run_def["seed"])

        self._restore_base_config()
        apply_overrides(run_def["config"])
        set_random_seed(seed)

        train_data, test_data, stats = self._load_dataset_for_seed(seed)
        Config.NUM_USERS = stats["n_users"]
        Config.NUM_ITEMS = stats["n_items"]
        Config.FEATURE_DIM = stats["feature_dim"]

        server, attacker = self._init_server_attacker(train_data)

        history = {
            "train_loss": [],
            "test_loss": [],
            "rmse": [],
            "attack_acc": [],
            "attack_auc": [],
            "attack_precision": [],
            "attack_recall": [],
            "attack_f1": [],
            "attack_loss": [],
            "privacy_sigma": [],
            "round_time": [],
        }

        users = list(train_data.keys())
        total_rounds = int(Config.ROUNDS)

        if len(users) < 2:
            raise RuntimeError("用户数不足，无法进行成员/非成员分组攻击评估。")

        for r in range(total_rounds):
            round_start = dt.datetime.now()

            member_n = min(Config.USERS_PER_ROUND, len(users))
            selected_members = np.random.choice(users, member_n, replace=False)

            candidates = list(set(users) - set(selected_members))
            non_member_n = min(member_n, len(candidates))
            selected_non_members = [] if non_member_n == 0 else np.random.choice(candidates, non_member_n, replace=False)

            member_updates = []
            non_member_grads = []
            train_losses = []
            test_losses = []
            ldp_sigma_trace = []

            global_state = server.get_state()

            for uid in selected_members:
                uid = int(uid)
                client = Client(
                    uid,
                    train_data[uid],
                    test_data[uid],
                    Config,
                    Config.FEATURE_DIM,
                    personal_state=server.get_personal_state(uid),
                )
                up, t_loss, dp_meta = client.train(global_state, round_idx=r, total_rounds=total_rounds)
                server.set_personal_state(uid, client.export_personal_state())

                if up:
                    member_updates.append(up)
                    train_losses.append(t_loss)
                    if Config.PRIVACY_MODE == "LDP":
                        ldp_sigma_trace.append(float(dp_meta.get("avg_sigma", 0.0)))

                test_losses.append(client.evaluate())

            for uid in selected_non_members:
                uid = int(uid)
                shadow_client = Client(
                    uid,
                    train_data[uid],
                    [],
                    Config,
                    Config.FEATURE_DIM,
                    personal_state=server.get_personal_state(uid),
                )
                nm_up, _, _ = shadow_client.train(global_state, round_idx=r, total_rounds=total_rounds)
                if nm_up:
                    non_member_grads.append(nm_up)

            if member_updates:
                server.aggregate(member_updates, round_idx=r, total_rounds=total_rounds)

            attack_metrics = {"acc": 0.5, "auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            att_loss = 0.0
            if attacker and member_updates and non_member_grads:
                attacker.add_data(member_updates, non_member_grads)
                att_loss = attacker.train_epoch(epochs=int(self.args.attack_epochs))
                attack_metrics = attacker.test_metrics(member_updates, non_member_grads)

            avg_train = float(np.mean(train_losses)) if train_losses else 0.0
            avg_test = float(np.mean(test_losses)) if test_losses else 0.0
            rmse = float(math.sqrt(max(avg_test, 0.0)))

            if Config.PRIVACY_MODE == "CDP":
                privacy_sigma = float(server.last_dp_meta.get("avg_sigma", 0.0))
            elif Config.PRIVACY_MODE == "LDP":
                privacy_sigma = float(np.mean(ldp_sigma_trace)) if ldp_sigma_trace else 0.0
            else:
                privacy_sigma = 0.0

            history["train_loss"].append(avg_train)
            history["test_loss"].append(avg_test)
            history["rmse"].append(rmse)
            history["attack_acc"].append(float(attack_metrics["acc"]))
            history["attack_auc"].append(float(attack_metrics["auc"]))
            history["attack_precision"].append(float(attack_metrics["precision"]))
            history["attack_recall"].append(float(attack_metrics["recall"]))
            history["attack_f1"].append(float(attack_metrics["f1"]))
            history["attack_loss"].append(float(att_loss))
            history["privacy_sigma"].append(privacy_sigma)
            history["round_time"].append(float((dt.datetime.now() - round_start).total_seconds()))

            if (r + 1) % int(self.args.log_every) == 0 or r == 0:
                print(
                    f"[{run_id}][{group}][Seed {seed}] Round {r + 1:4d}/{total_rounds} | "
                    f"TrainLoss {avg_train:.4f} | TestLoss {avg_test:.4f} | RMSE {rmse:.4f} | "
                    f"ASR {attack_metrics['acc']:.4f} | AUC {attack_metrics['auc']:.4f} | AttackLoss {att_loss:.4f} | Sigma* {privacy_sigma:.4f}",
                    flush=True,
                )

        tail_metrics = summarize_tail(history, Config.TAIL_WINDOW)

        run_name = f"{run_id}_{group}_seed{seed}"
        log_path = build_result_path(self.logs_dir, Config, run_name=run_name)
        save_data = {k: [float(x) if x is not None else 0.0 for x in v] for k, v in history.items()}
        save_data["meta"] = {
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "config": config_snapshot(),
            "dataset": stats,
            "run": {
                "run_id": run_id,
                "group": group,
                "seed": seed,
                "batch_dir": self.batch_dir,
            },
        }

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        return log_path, tail_metrics

    def _write_summary_csv(self):
        csv_path = os.path.join(self.batch_dir, "batch_summary.csv")
        fields = [
            "id", "group", "seed", "status", "duration_sec", "log_path",
            "tail", "tail_asr", "tail_auc", "tail_rmse", "round_time_mean", "error",
        ]

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for run in self.state["runs"]:
                tm = run.get("tail_metrics") or {}
                writer.writerow({
                    "id": run.get("id"),
                    "group": run.get("group"),
                    "seed": run.get("seed"),
                    "status": run.get("status"),
                    "duration_sec": run.get("duration_sec"),
                    "log_path": run.get("log_path"),
                    "tail": tm.get("tail"),
                    "tail_asr": tm.get("tail_asr"),
                    "tail_auc": tm.get("tail_auc"),
                    "tail_rmse": tm.get("tail_rmse"),
                    "round_time_mean": tm.get("round_time_mean"),
                    "error": run.get("error"),
                })

    def run(self):
        total = len(self.state["runs"])
        print(f"[INFO] Batch Dir: {self.batch_dir}")
        print(f"[INFO] Total Runs: {total}")

        for idx, run in enumerate(self.state["runs"], start=1):
            status = run.get("status", "pending")
            if status == "done":
                print(f"[SKIP] {run['id']} {run['group']} seed={run['seed']} 已完成")
                continue
            if status == "failed" and not self.args.retry_failed:
                print(f"[SKIP] {run['id']} {run['group']} seed={run['seed']} 之前失败，未开启 --retry-failed")
                continue

            start_time = dt.datetime.now()
            run["status"] = "running"
            run["started_at"] = start_time.isoformat(timespec="seconds")
            run["error"] = None
            self._save_state()

            print(
                f"[RUN] ({idx}/{total}) {run['id']} | Group={run['group']} | Seed={run['seed']} | Rounds={run['config']['ROUNDS']}",
                flush=True,
            )

            try:
                log_path, tail_metrics = self._run_single(run)
                end_time = dt.datetime.now()
                run["status"] = "done"
                run["ended_at"] = end_time.isoformat(timespec="seconds")
                run["duration_sec"] = float((end_time - start_time).total_seconds())
                run["log_path"] = os.path.abspath(log_path)
                run["tail_metrics"] = tail_metrics
                print(
                    f"[DONE] {run['id']} | tail_ASR={tail_metrics['tail_asr']:.4f} | tail_RMSE={tail_metrics['tail_rmse']:.4f} | {log_path}",
                    flush=True,
                )
            except KeyboardInterrupt:
                run["status"] = "failed"
                run["ended_at"] = dt.datetime.now().isoformat(timespec="seconds")
                run["error"] = "KeyboardInterrupt"
                self._save_state()
                self._write_summary_csv()
                raise
            except Exception:
                run["status"] = "failed"
                run["ended_at"] = dt.datetime.now().isoformat(timespec="seconds")
                run["error"] = traceback.format_exc(limit=20)
                print(f"[FAIL] {run['id']} 失败，错误已记录到 state 文件。", flush=True)
                if self.args.stop_on_error:
                    self._save_state()
                    self._write_summary_csv()
                    raise
            finally:
                self._save_state()
                self._write_summary_csv()

        done_n = sum(1 for x in self.state["runs"] if x.get("status") == "done")
        fail_n = sum(1 for x in self.state["runs"] if x.get("status") == "failed")
        print(f"[INFO] Batch finished. done={done_n}, failed={fail_n}, total={total}")
        print(f"[INFO] State: {self.state_path}")
        print(f"[INFO] Summary: {os.path.join(self.batch_dir, 'batch_summary.csv')}")


def build_args():
    parser = argparse.ArgumentParser(
        description="云端批量实验运行器（自动串行跑完多组，支持断点续跑）"
    )
    parser.add_argument("--rounds", type=int, default=1000, help="每组通信轮数，默认 1000")
    parser.add_argument("--groups", type=str, default=",".join(DEFAULT_GROUPS), help="实验组列表，逗号分隔")
    parser.add_argument("--seeds", type=str, default="42", help="随机种子列表，逗号分隔，例如 42,52,62")
    parser.add_argument("--output-root", type=str, default="logs", help="批量结果根目录")
    parser.add_argument("--batch-name", type=str, default=None, help="批次目录名，不填则自动生成")
    parser.add_argument("--resume-dir", type=str, default=None, help="从已有批次目录继续跑")
    parser.add_argument("--retry-failed", action="store_true", help="继续跑时是否重试失败组")
    parser.add_argument("--stop-on-error", action="store_true", help="遇到错误立即停止")
    parser.add_argument("--log-every", type=int, default=20, help="每多少轮打印一次日志")
    parser.add_argument("--users-per-round", type=int, default=None, help="覆盖 USERS_PER_ROUND")
    parser.add_argument("--local-epochs", type=int, default=None, help="覆盖 LOCAL_EPOCHS")
    parser.add_argument("--disable-attack", action="store_true", help="关闭成员推断攻击，减少耗时")
    parser.add_argument("--attack-epochs", type=int, default=15, help="每轮成员推断攻击训练epoch数")
    parser.add_argument("--attack-lr", type=float, default=0.003, help="成员推断攻击学习率")
    parser.add_argument("--attack-buffer-size", type=int, default=2000, help="成员推断攻击样本缓存大小")
    parser.add_argument("--set", type=str, default="", help="????????? KEY=VALUE,KEY2=VALUE2")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    runner = CloudBatchRunner(args)
    runner.run()


