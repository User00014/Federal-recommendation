import os
import glob
import json
import math
from collections import defaultdict

base = '/root/Federated_Privacy_Project/logs'
batches = ['full_15_seed42_v2', 'full_15_seed52_v2']
rows = []
issues = []

for b in batches:
    bdir = os.path.join(base, b, 'logs')
    for p in sorted(glob.glob(os.path.join(bdir, '*.json'))):
        d = json.load(open(p, 'r', encoding='utf-8'))
        meta = d.get('meta', {}) if isinstance(d, dict) else {}
        run = meta.get('run', {}) if isinstance(meta, dict) else {}
        cfg = meta.get('config', {}) if isinstance(meta, dict) else {}

        group = run.get('group', 'UNK')
        seed = run.get('seed', 'UNK')
        mode = cfg.get('PRIVACY_MODE', 'UNK')
        sigma = float(cfg.get('DP_SIGMA', 0.0) or 0.0)
        adp = bool(cfg.get('ENABLE_ADAPTIVE_DP', False))

        rmse = [float(x) for x in d.get('rmse', []) if x is not None]
        asr = [float(x) for x in d.get('attack_acc', []) if x is not None]
        auc = [float(x) for x in d.get('attack_auc', []) if x is not None]
        ps = [float(x) for x in d.get('privacy_sigma', []) if x is not None]
        rt = [float(x) for x in d.get('round_time', []) if x is not None]

        n = len(rmse)
        if n == 0:
            issues.append((os.path.basename(p), 'empty_rmse'))
            continue
        tail = min(100, n)

        def mean(a):
            return sum(a) / len(a) if a else float('nan')

        row = {
            'batch': b,
            'group': group,
            'seed': seed,
            'mode': mode,
            'sigma': sigma,
            'adp': adp,
            'n': n,
            'rmse_tail': mean(rmse[-tail:]),
            'asr_tail': mean(asr[-tail:]) if asr else float('nan'),
            'auc_tail': mean(auc[-tail:]) if auc else float('nan'),
            'sigma_tail': mean(ps[-tail:]) if ps else 0.0,
            'rt_tail': mean(rt[-tail:]) if rt else float('nan'),
            'rmse_max': max(rmse),
            'asr_min': min(asr) if asr else float('nan'),
            'asr_max': max(asr) if asr else float('nan'),
        }
        rows.append(row)

        seq = rmse + asr + auc + ps + rt
        if any(math.isnan(x) or math.isinf(x) for x in seq):
            issues.append((os.path.basename(p), 'nan_or_inf'))
        if row['rmse_max'] > 6.0:
            issues.append((os.path.basename(p), f'rmse_too_high:{row["rmse_max"]:.3f}'))
        if mode == 'PLAIN' and row['sigma_tail'] > 1e-6:
            issues.append((os.path.basename(p), f'plain_sigma_not_zero:{row["sigma_tail"]:.6f}'))
        if mode in ('CDP', 'LDP') and row['sigma_tail'] <= 0:
            issues.append((os.path.basename(p), 'dp_sigma_tail_nonpositive'))

print('DONE_LOG_COUNT', len(rows))
for r in sorted(rows, key=lambda x: (str(x['seed']), str(x['group']))):
    print('ROW', r['seed'], r['group'], r['mode'],
          f"sig={r['sigma']:g}",
          'ADP' if r['adp'] else 'FDP',
          f"RMSE@tail={r['rmse_tail']:.4f}",
          f"ASR@tail={r['asr_tail']:.4f}",
          f"AUC@tail={r['auc_tail']:.4f}",
          f"SigmaEff@tail={r['sigma_tail']:.4f}",
          f"T/round={r['rt_tail']:.2f}s")

agg = defaultdict(list)
for r in rows:
    agg[r['mode']].append(r)

print('AGG_BY_MODE')
for mode, arr in agg.items():
    print('MODE', mode, 'n', len(arr),
          'RMSE@tail', round(sum(x['rmse_tail'] for x in arr) / len(arr), 4),
          'ASR@tail', round(sum(x['asr_tail'] for x in arr) / len(arr), 4),
          'AUC@tail', round(sum(x['auc_tail'] for x in arr) / len(arr), 4),
          'T/round', round(sum(x['rt_tail'] for x in arr) / len(arr), 2))

print('ISSUE_COUNT', len(issues))
for k, v in issues[:50]:
    print('ISSUE', k, v)
