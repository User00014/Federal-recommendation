import datetime as dt
import os
import re


def _sanitize_token(text):
    val = str(text).strip()
    if not val:
        return "NA"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", val)


def _float_to_token(value):
    try:
        return f"{float(value):g}"
    except Exception:
        return _sanitize_token(value)


def build_result_filename(config, run_name=None, timestamp=None):
    pflag = "P" if bool(getattr(config, "ENABLE_PERSONALIZATION", False)) else "NP"
    dflag = "ADP" if bool(getattr(config, "ENABLE_ADAPTIVE_DP", False)) else "FDP"

    parts = [
        "res",
        _sanitize_token(getattr(config, "PRIVACY_MODE", "UNK")),
        f"sigma{_float_to_token(getattr(config, 'DP_SIGMA', 0.0))}",
        _sanitize_token(getattr(config, "FL_ALGO", "FEDAVG")),
        pflag,
        dflag,
        f"{int(getattr(config, 'ROUNDS', 0))}rounds",
        f"seed{int(getattr(config, 'RANDOM_SEED', 0))}",
        f"mu{_float_to_token(getattr(config, 'PROX_MU', 0.0))}",
    ]

    if run_name:
        parts.append(_sanitize_token(run_name))

    ts = timestamp or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    parts.append(ts)

    return "_".join(parts) + ".json"


def build_result_path(save_dir, config, run_name=None, timestamp=None):
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, build_result_filename(config, run_name=run_name, timestamp=timestamp))
