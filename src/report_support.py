import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager as fm


DEFAULT_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\NotoSansSC-VF.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
]


def configure_report_plot_style(font_candidates: Optional[Iterable[str]] = None) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.unicode_minus"] = False

    candidates = list(font_candidates or DEFAULT_FONT_CANDIDATES)
    seen = set()
    for font_path in candidates:
        if font_path in seen:
            continue
        seen.add(font_path)
        if Path(font_path).exists():
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            plt.rcParams["font.family"] = [font_name, "DejaVu Sans"]
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_payloads(file_map: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for key, path in file_map.items():
        if path.exists():
            payloads[key] = load_json(path)
    return payloads


def safe_list(payload: Dict[str, Any], key: str, fallback_len: int = 0, default_value: float = 0.0) -> List[float]:
    values = payload.get(key, [])
    if not isinstance(values, list):
        values = []
    if not values and fallback_len > 0:
        values = [default_value] * fallback_len
    return [float(x) for x in values]


@dataclass
class RunRecord:
    group: str
    seed: int
    file_name: str
    rmse: List[float]
    attack_acc: List[float]
    attack_auc: List[float]
    privacy_sigma: List[float]
    round_time: List[float]
    train_loss: List[float]
    test_loss: List[float]
    config: Dict[str, Any]

    def tail_mean(self, values: List[float], tail: int) -> float:
        if not values:
            return float("nan")
        chunk = values[-tail:] if len(values) >= tail else values
        return float(np.mean(chunk))

    def build_summary(
        self,
        tail: int,
        sigma_key: str = "tail_sigma",
        time_key: str = "tail_time",
    ) -> Dict[str, float]:
        return {
            "group": self.group,
            "seed": self.seed,
            "file_name": self.file_name,
            "tail_rmse": self.tail_mean(self.rmse, tail),
            "tail_asr": self.tail_mean(self.attack_acc, tail),
            "peak_asr": float(np.max(self.attack_acc)) if self.attack_acc else float("nan"),
            "tail_auc": self.tail_mean(self.attack_auc, tail),
            sigma_key: self.tail_mean(self.privacy_sigma, tail),
            time_key: self.tail_mean(self.round_time, tail),
            "tail_train_loss": self.tail_mean(self.train_loss, tail),
            "tail_test_loss": self.tail_mean(self.test_loss, tail),
        }


def load_runs_from_dirs(current_dirs: Iterable[Path]) -> List[RunRecord]:
    runs: List[RunRecord] = []
    for current_dir in current_dirs:
        if not current_dir.exists():
            continue
        for path in sorted(current_dir.glob("*.json")):
            data = load_json(path)
            meta = data.get("meta", {})
            run = meta.get("run", {})
            cfg = meta.get("config", {})
            group = run.get("group")
            seed = int(run.get("seed", cfg.get("RANDOM_SEED", 42)))
            rmse = safe_list(data, "rmse")
            base_len = len(rmse) or len(data.get("attack_acc", [])) or len(data.get("test_loss", []))
            if not group or base_len == 0:
                continue
            runs.append(
                RunRecord(
                    group=group,
                    seed=seed,
                    file_name=path.name,
                    rmse=rmse,
                    attack_acc=safe_list(data, "attack_acc", base_len, 0.5),
                    attack_auc=safe_list(data, "attack_auc", base_len, 0.5),
                    privacy_sigma=safe_list(data, "privacy_sigma", base_len, float(cfg.get("DP_SIGMA", 0.0))),
                    round_time=safe_list(data, "round_time", base_len, float("nan")),
                    train_loss=safe_list(data, "train_loss", base_len, float("nan")),
                    test_loss=safe_list(data, "test_loss", base_len, float("nan")),
                    config=cfg,
                )
            )
    return runs


def runs_by_group_seed(runs: Iterable[RunRecord]) -> Dict[str, Dict[int, RunRecord]]:
    grouped: Dict[str, Dict[int, RunRecord]] = {}
    for run in runs:
        grouped.setdefault(run.group, {})[run.seed] = run
    return grouped


def mean_std_curve(records: List[RunRecord], field: str, include_raw: bool = False):
    min_len = min(len(getattr(record, field)) for record in records)
    x = np.arange(1, min_len + 1)
    arr = np.vstack([np.array(getattr(record, field)[:min_len], dtype=float) for record in records])
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    if include_raw:
        return x, mean, std, arr
    return x, mean, std
