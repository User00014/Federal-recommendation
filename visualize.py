import os
import re
import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D


sns.set_theme(style='whitegrid', context='paper')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['figure.facecolor'] = '#f8f9fb'
plt.rcParams['axes.facecolor'] = '#fbfcff'
plt.rcParams['savefig.facecolor'] = '#f8f9fb'

MODE_COLORS = {
    'PLAIN': '#c44e52',
    'CDP': '#4c72b0',
    'LDP': '#55a868',
    'UNK': '#8c8c8c',
}
ALGO_MARKERS = {
    'FEDAVG': 'o',
    'FEDPROX': 's',
}
ADP_STYLES = {
    'ADP': '-',
    'FDP': '--',
    'UNK': '-',
}


def ema(arr, alpha=0.2):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.array([])
    out = [arr[0]]
    for x in arr[1:]:
        out.append(alpha * x + (1 - alpha) * out[-1])
    return np.array(out)


def finish_plot(fig):
    backend = str(plt.get_backend()).lower()
    if 'agg' in backend:
        plt.close(fig)
    else:
        plt.show()


class PaperVisualizer:
    def __init__(self, logs_dir='logs', output_dir='figures/archive_exploratory_20260309', extra_log_dirs=None):
        self.logs_dir = logs_dir
        self.output_dir = output_dir

        # read logs from logs_dir only
        self.log_dirs = [logs_dir]
        self.extra_log_dirs = []
        if extra_log_dirs:
            print('[INFO] Only logs_dir is used; extra_log_dirs ignored.')

        self.logs = {}
        self.summary_df = pd.DataFrame()
        self.active_files = []
        self.run_color_map = {}

    def _build_run_colors(self):
        names = sorted(self.summary_df['file'].tolist()) if not self.summary_df.empty else []
        n = len(names)
        if n == 0:
            self.run_color_map = {}
            return

        palette = sns.husl_palette(n_colors=n, s=0.88, l=0.50)
        self.run_color_map = {name: palette[i] for i, name in enumerate(names)}

    def _color_of(self, file_name):
        return self.run_color_map.get(file_name, '#666666')

    def _parse_filename(self, fname):
        base = os.path.basename(fname)
        pat = re.compile(
            r"res_(?P<mode>[A-Z]+)_sigma(?P<sigma>[-+0-9.eE]+)"
            r"(?:_(?P<algo>FED[A-Z]+))?"
            r"(?:_(?P<personal>P|NP))?"
            r"(?:_(?P<adp>ADP|FDP))?"
            r"_(?P<rounds>\d+)rounds(?:_.*)?\.json$"
        )
        m = pat.search(base)
        if not m:
            return {
                'raw_file': base,
                'mode': 'UNK',
                'sigma': np.nan,
                'algo': 'FEDAVG',
                'personal': 'UNK',
                'adp': 'UNK',
                'rounds': np.nan,
            }

        d = m.groupdict()
        return {
            'raw_file': base,
            'mode': d.get('mode') or 'UNK',
            'sigma': float(d.get('sigma') or 0.0),
            'algo': d.get('algo') or 'FEDAVG',
            'personal': d.get('personal') or 'UNK',
            'adp': d.get('adp') or 'UNK',
            'rounds': int(d.get('rounds') or 0),
        }

    def _safe_list(self, obj, key, fallback_len=None, default_value=0.0):
        arr = obj.get(key, [])
        if not isinstance(arr, list):
            arr = []
        if len(arr) == 0 and fallback_len is not None:
            arr = [default_value] * fallback_len
        return np.array(arr, dtype=float)

    def _dedupe_name(self, base_name, src_dir):
        if base_name not in self.logs:
            return base_name
        suffix = os.path.basename(src_dir) or src_dir
        return f"{base_name} [{suffix}]"

    def _prepare_canvas(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def load_logs(self):
        files = sorted(glob.glob(os.path.join(self.logs_dir, '*.json')))

        if not files:
            print('[ERR] 未在 logs 目录下找到日志文件。')
            return False

        self.logs = {}
        rows = []

        for path in files:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f'[WARN] 跳过文件 {os.path.basename(path)}: {e}')
                continue

            meta = self._parse_filename(path)
            display_name = self._dedupe_name(meta['raw_file'], os.path.dirname(path))

            train_loss = self._safe_list(data, 'train_loss')
            test_loss = self._safe_list(data, 'test_loss')
            base_len = len(test_loss) if len(test_loss) > 0 else len(train_loss)
            if base_len == 0:
                continue

            loss = test_loss if len(test_loss) > 0 else train_loss
            rmse = self._safe_list(data, 'rmse')
            if len(rmse) == 0:
                rmse = np.sqrt(np.clip(loss, a_min=0.0, a_max=None))

            attack_acc = self._safe_list(data, 'attack_acc', fallback_len=base_len, default_value=0.5)
            attack_auc = self._safe_list(data, 'attack_auc', fallback_len=base_len, default_value=0.5)
            round_time = self._safe_list(data, 'round_time', fallback_len=base_len, default_value=np.nan)
            privacy_sigma = self._safe_list(data, 'privacy_sigma', fallback_len=base_len, default_value=0.0)

            tail_default = 50
            tail = tail_default
            if isinstance(data.get('meta'), dict):
                cfg = data['meta'].get('config', {})
                if isinstance(cfg, dict):
                    tail = int(cfg.get('TAIL_WINDOW', tail_default))
            tail = max(10, min(tail, base_len))

            rec = {
                'meta': meta,
                'path': path,
                'raw': data,
                'loss': loss,
                'rmse': rmse,
                'attack_acc': attack_acc,
                'attack_auc': attack_auc,
                'round_time': round_time,
                'privacy_sigma': privacy_sigma,
                'tail': tail,
            }
            self.logs[display_name] = rec

            rows.append({
                'file': display_name,
                'raw_file': meta['raw_file'],
                'source_dir': os.path.basename(os.path.dirname(path)) or os.path.dirname(path),
                'mode': meta['mode'],
                'algo': meta['algo'],
                'personal': meta['personal'],
                'adp': meta['adp'],
                'sigma': meta['sigma'],
                'rounds': base_len,
                'tail': tail,
                'tail_asr_mean': float(np.mean(attack_acc[-tail:])),
                'tail_asr_std': float(np.std(attack_acc[-tail:])),
                'tail_auc_mean': float(np.mean(attack_auc[-tail:])),
                'tail_rmse_mean': float(np.mean(rmse[-tail:])),
                'tail_rmse_std': float(np.std(rmse[-tail:])),
                'round_time_mean': float(np.nanmean(round_time)) if np.any(~np.isnan(round_time)) else np.nan,
                'sigma_eff_mean': float(np.mean(privacy_sigma[-tail:])),
            })

        self.summary_df = pd.DataFrame(rows)
        if self.summary_df.empty:
            print('[ERR] 日志加载失败。')
            return False

        self._build_run_colors()
        self.select_all_logs()
        print(f"[OK] 已从 logs 目录加载 {len(self.summary_df)} 个日志。")
        return True

    def _label(self, row, compact=False):
        if compact:
            if pd.isna(row['sigma']):
                return f"{row['mode']}/{row['algo']}"
            return f"{row['mode']}/σ={row['sigma']:g}/{row['algo']}"
        if pd.isna(row['sigma']):
            return f"{row['mode']}|{row['algo']}|{row['personal']}|{row['adp']}"
        return (
            f"{row['mode']}|σ={row['sigma']:g}|{row['algo']}|"
            f"{row['personal']}|{row['adp']}"
        )

    def _active_summary(self):
        if self.summary_df.empty:
            return pd.DataFrame()
        if not self.active_files:
            return pd.DataFrame()
        return self.summary_df[self.summary_df['file'].isin(self.active_files)].copy()

    def _iter_active(self):
        df = self._active_summary()
        if df.empty:
            return []
        df = df.sort_values(['mode', 'sigma', 'algo', 'personal', 'adp', 'file'])
        out = []
        for _, row in df.iterrows():
            name = row['file']
            if name in self.logs:
                out.append((name, self.logs[name], row))
        return out

    def _ensure_selection(self):
        if not self.active_files:
            print('[WARN] 当前没有选中的日志，请先选择日志。')
            return False
        return True

    def _legend_panel(self, ax_leg, handles, labels, title='图例'):
        ax_leg.axis('off')
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        labels_u = list(uniq.keys())
        handles_u = [uniq[l] for l in labels_u]

        if not handles_u:
            ax_leg.text(0.5, 0.5, '无图例', ha='center', va='center', fontsize=10)
            return

        n = len(handles_u)
        ncol = min(4, max(1, int(np.ceil(n / 2))))
        ax_leg.legend(handles_u, labels_u, loc='center', ncol=ncol, frameon=False, fontsize=8.5, title=title, title_fontsize=9.5)

    def _run_legend_handles(self, df_sorted):
        handles = []
        labels = []
        for i, row in df_sorted.reset_index(drop=True).iterrows():
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=self._color_of(row['file']), markersize=7))
            short = self._label(row, compact=True)
            if len(short) > 26:
                short = short[:25] + '…'
            labels.append(f"#{i + 1} {short}")
        return handles, labels


    def select_all_logs(self):
        self.active_files = self.summary_df['file'].tolist() if not self.summary_df.empty else []
        print(f"[OK] 已选择全部日志: {len(self.active_files)} 个。")

    def _parse_indices(self, text, max_idx):
        text = text.strip().lower()
        if text in ['a', 'all', '*']:
            return list(range(1, max_idx + 1))

        values = set()
        for token in text.split(','):
            token = token.strip()
            if not token:
                continue
            if '-' in token:
                parts = token.split('-', 1)
                if len(parts) != 2:
                    continue
                l, r = parts[0].strip(), parts[1].strip()
                if not l.isdigit() or not r.isdigit():
                    continue
                l, r = int(l), int(r)
                if l > r:
                    l, r = r, l
                for i in range(l, r + 1):
                    if 1 <= i <= max_idx:
                        values.add(i)
            else:
                if token.isdigit():
                    i = int(token)
                    if 1 <= i <= max_idx:
                        values.add(i)
        return sorted(values)

    def select_logs_interactive(self):
        if self.summary_df.empty:
            print('[WARN] 暂无日志可选。')
            return

        df = self.summary_df.sort_values(['mode', 'sigma', 'algo', 'personal', 'adp', 'file']).reset_index(drop=True)

        print('\n' + '=' * 92)
        print('  日志选择器（绘图前置步骤）')
        print('=' * 92)
        for i, row in df.iterrows():
            idx = i + 1
            color_hex = mcolors.to_hex(self._color_of(row['file']))
            desc = (
                f"{row['mode']:<5} | σ={row['sigma']:<6g} | {row['algo']:<7} | "
                f"{row['personal']:<2} | {row['adp']:<3} | R={int(row['rounds']):<4} | "
                f"ASR@tail={row['tail_asr_mean']:.4f} | RMSE@tail={row['tail_rmse_mean']:.4f} | color={color_hex}"
            )
            print(f" [{idx:>2}] {row['file']}\n      {desc}")

        print('-' * 92)
        print('输入格式示例: 1,3,5-7')
        print('输入 a 表示选择全部日志；输入 n 表示清空选择')
        raw = input('请选择参与绘图的日志: ').strip().lower()

        if raw in ['', 'a', 'all', '*']:
            self.active_files = df['file'].tolist()
        elif raw in ['n', 'none', '0']:
            self.active_files = []
        else:
            idxs = self._parse_indices(raw, len(df))
            self.active_files = df.iloc[[i - 1 for i in idxs]]['file'].tolist() if idxs else []

        if self.active_files:
            print(f"[OK] 已选择 {len(self.active_files)} 个日志参与绘图。")
        else:
            print('[WARN] 当前没有选中任何日志。')


    def plot_core_panel(self, save=True):
        if not self._ensure_selection():
            return

        series = self._iter_active()
        if not series:
            print('[WARN] 选中的日志为空。')
            return

        fig = plt.figure(figsize=(19.0, 7.0), dpi=150)
        gs = fig.add_gridspec(2, 3, height_ratios=[14, 2])
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_leg = fig.add_subplot(gs[1, :])

        for name, rec, row in series:
            x = np.arange(1, len(rec['rmse']) + 1)
            color = self._color_of(name)
            ls = ADP_STYLES.get(row['adp'], '-')

            axes[0].plot(x, rec['rmse'], color=color, alpha=0.14, linewidth=1.0)
            axes[0].plot(x, ema(rec['rmse'], alpha=0.18), color=color, linestyle=ls, linewidth=2.1)

            axes[1].plot(x, rec['attack_acc'], color=color, alpha=0.14, linewidth=1.0)
            axes[1].plot(x, ema(rec['attack_acc'], alpha=0.20), color=color, linestyle=ls, linewidth=2.1)

            axes[2].plot(x, rec['attack_auc'], color=color, alpha=0.14, linewidth=1.0)
            axes[2].plot(x, ema(rec['attack_auc'], alpha=0.20), color=color, linestyle=ls, linewidth=2.1)

        axes[0].set_title('收敛曲线：RMSE（越低越好）')
        axes[1].set_title('隐私风险：ASR（越接近0.5越安全）')
        axes[2].set_title('攻击区分度：AUC（越接近0.5越安全）')

        for ax in axes:
            ax.set_xlabel('通信轮次')
            ax.grid(True, linestyle='--', alpha=0.22)
        axes[0].set_ylabel('RMSE')
        axes[1].set_ylabel('ASR')
        axes[2].set_ylabel('AUC')
        axes[1].axhline(0.5, linestyle='--', linewidth=1.1, color='#34495e', alpha=0.9)
        axes[2].axhline(0.5, linestyle='--', linewidth=1.1, color='#34495e', alpha=0.9)

        df_sorted = self._active_summary().sort_values(['mode', 'sigma', 'algo', 'personal', 'adp', 'file'])
        run_handles, run_labels = self._run_legend_handles(df_sorted)
        style_handles = [
            Line2D([0], [0], color='#444444', linestyle='-', linewidth=2),
            Line2D([0], [0], color='#444444', linestyle='--', linewidth=2),
        ]
        style_labels = ['ADP（实线）', 'FDP（虚线）']
        self._legend_panel(ax_leg, run_handles + style_handles, run_labels + style_labels, title='实验组与线型')

        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig1_core_curves.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def plot_pareto_bubble(self, save=True):
        if not self._ensure_selection():
            return

        df = self._active_summary()
        if df.empty:
            return

        work = df.copy()
        work['privacy_gap'] = (work['tail_asr_mean'] - 0.5).abs()
        if work['round_time_mean'].isna().all():
            size = np.full(len(work), 260.0)
        else:
            rt = work['round_time_mean'].fillna(work['round_time_mean'].median())
            rt_min, rt_max = rt.min(), rt.max()
            if rt_max > rt_min:
                size = 170 + 330 * (rt - rt_min) / (rt_max - rt_min)
            else:
                size = np.full(len(work), 260.0)

        work['bubble_size'] = size
        sorted_df = work.sort_values(['mode', 'sigma', 'algo', 'personal', 'adp']).reset_index(drop=True)

        fig = plt.figure(figsize=(10.5, 8.0), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[14, 3])
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[1, 0])

        ymin = max(0.0, float(work['tail_rmse_mean'].min()) * 0.9)
        ymax = float(work['tail_rmse_mean'].max()) * 1.1
        xmin = 0.0
        xmax = max(0.02, float(work['privacy_gap'].max()) * 1.15)

        ax.add_patch(plt.Rectangle((xmin, ymin), max(0.005, xmax * 0.35), max(0.001, (ymax - ymin) * 0.35), color='#2ecc71', alpha=0.10, zorder=0))

        for i, row in sorted_df.iterrows():
            color = self._color_of(row['file'])
            marker = ALGO_MARKERS.get(row['algo'], 'o')
            ax.scatter(
                row['privacy_gap'],
                row['tail_rmse_mean'],
                s=float(row['bubble_size']),
                color=color,
                marker=marker,
                alpha=0.88,
                edgecolors='white',
                linewidths=1.1,
            )
            ax.annotate(f"#{i + 1}", (row['privacy_gap'], row['tail_rmse_mean']), textcoords='offset points', xytext=(7, -5), fontsize=8)

        ax.set_title('隐私-效用 Pareto 气泡图（尾段统计）')
        ax.set_xlabel('|ASR - 0.5|（越低越安全）')
        ax.set_ylabel('Tail RMSE（越低越好）')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle='--', alpha=0.22)

        run_handles, run_labels = self._run_legend_handles(sorted_df)
        algo_handles = [
            Line2D([0], [0], marker=mk, color='#555555', label=algo, linestyle='None', markersize=8)
            for algo, mk in ALGO_MARKERS.items() if algo in sorted_df['algo'].values
        ]
        algo_labels = [h.get_label() for h in algo_handles]
        self._legend_panel(ax_leg, run_handles + algo_handles, run_labels + algo_labels, title='编号颜色与聚合算法')

        print('\n[Pareto 编号映射]')
        for i, row in sorted_df.iterrows():
            print(f"  #{i + 1}: {self._label(row, compact=False)} | color={mcolors.to_hex(self._color_of(row['file']))}")

        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig2_pareto_bubble.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def plot_tail_distribution(self, save=True):
        if not self._ensure_selection():
            return

        items = self._iter_active()
        if not items:
            return

        rows = []
        for idx, (name, rec, row) in enumerate(items, start=1):
            tail = rec['tail']
            short = self._label(row, compact=True)
            if len(short) > 26:
                short = short[:25] + '…'
            label = f"#{idx} {short}"
            for v in rec['attack_acc'][-tail:]:
                rows.append({'exp': label, 'file': name, 'asr': float(v)})

        dist_df = pd.DataFrame(rows)
        if dist_df.empty:
            print('[WARN] 缺少 tail ASR 数据，跳过该图。')
            return

        order = dist_df.groupby('exp')['asr'].mean().sort_values().index.tolist()

        fig = plt.figure(figsize=(12.8, 8.0), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[14, 3])
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[1, 0])

        sns.boxplot(data=dist_df, x='exp', y='asr', order=order, color='#d8e8f8', width=0.62, fliersize=0, ax=ax)

        rng = np.random.default_rng(42)
        exp_to_file = dist_df.groupby('exp')['file'].first().to_dict()
        for i, exp_name in enumerate(order):
            grp = dist_df[dist_df['exp'] == exp_name]
            if grp.empty:
                continue
            c = self._color_of(exp_to_file[exp_name])
            xj = i + rng.normal(0, 0.06, size=len(grp))
            ax.scatter(xj, grp['asr'].values, s=12, alpha=0.45, c=[c], linewidths=0)

        ax.axhline(0.5, linestyle='--', color='#2f3e4d', linewidth=1.1)
        ax.set_title('尾段 ASR 分布（箱线+散点）')
        ax.set_xlabel('实验组')
        ax.set_ylabel('ASR')
        ax.tick_params(axis='x', rotation=14)
        ax.grid(True, linestyle='--', alpha=0.18)

        legend_df = self._active_summary().sort_values(['mode', 'sigma', 'algo', 'personal', 'adp', 'file'])
        run_handles, run_labels = self._run_legend_handles(legend_df)
        self._legend_panel(ax_leg, run_handles, run_labels, title='实验组颜色映射')

        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig3_tail_asr_boxstrip.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def plot_noise_schedule(self, save=True):
        if not self._ensure_selection():
            return

        items = self._iter_active()
        if not items:
            return

        has_sigma = any(np.any(rec['privacy_sigma'] > 0) for _, rec, _ in items)
        if not has_sigma:
            print('[WARN] 当前选择中无 privacy_sigma>0 的日志，跳过噪声图。')
            return

        fig = plt.figure(figsize=(11.0, 7.8), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[14, 3])
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[1, 0])

        plotted_rows = []
        for name, rec, row in items:
            if row['mode'] == 'PLAIN' or not np.any(rec['privacy_sigma'] > 0):
                continue

            x = np.arange(1, len(rec['privacy_sigma']) + 1)
            color = self._color_of(name)
            ls = ADP_STYLES.get(row['adp'], '-')
            ax.plot(x, rec['privacy_sigma'], color=color, alpha=0.12, linewidth=1.0)
            ax.plot(x, ema(rec['privacy_sigma'], alpha=0.20), color=color, linestyle=ls, linewidth=2.2)
            plotted_rows.append(row)

        if not plotted_rows:
            print('[WARN] 当前选择中没有可用噪声曲线。')
            plt.close(fig)
            return

        ax.set_title('自适应噪声调度曲线（有效 σ*）')
        ax.set_xlabel('通信轮次')
        ax.set_ylabel('有效噪声强度 σ*')
        ax.grid(True, linestyle='--', alpha=0.22)

        legend_df = pd.DataFrame(plotted_rows).drop_duplicates(subset=['file'])
        run_handles, run_labels = self._run_legend_handles(legend_df)
        style_handles = [
            Line2D([0], [0], color='#444444', linestyle='-', linewidth=2),
            Line2D([0], [0], color='#444444', linestyle='--', linewidth=2),
        ]
        style_labels = ['ADP（实线）', 'FDP（虚线）']
        self._legend_panel(ax_leg, run_handles + style_handles, run_labels + style_labels, title='实验组与线型')

        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig4_noise_schedule.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def plot_round_time_curve(self, save=True):
        if not self._ensure_selection():
            return

        items = self._iter_active()
        if not items:
            return

        has_time = False
        for _, rec, _ in items:
            rt = np.asarray(rec['round_time'], dtype=float)
            if np.any(~np.isnan(rt)):
                has_time = True
                break
        if not has_time:
            print('[WARN] 当前选择中缺少 round_time 数据，跳过训练时间图。')
            return

        fig = plt.figure(figsize=(11.0, 7.8), dpi=150)
        gs = fig.add_gridspec(2, 1, height_ratios=[14, 3])
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[1, 0])

        plotted_rows = []
        for name, rec, row in items:
            rt = np.asarray(rec['round_time'], dtype=float)
            if not np.any(~np.isnan(rt)):
                continue

            x = np.arange(1, len(rt) + 1)
            rt_fill = np.where(np.isnan(rt), np.nanmedian(rt) if np.any(~np.isnan(rt)) else 0.0, rt)
            color = self._color_of(name)
            ls = ADP_STYLES.get(row['adp'], '-')
            ax.plot(x, rt_fill, color=color, alpha=0.12, linewidth=1.0)
            ax.plot(x, ema(rt_fill, alpha=0.20), color=color, linestyle=ls, linewidth=2.2)
            plotted_rows.append(row)

        ax.set_title('每轮训练时间曲线（Round Time）')
        ax.set_xlabel('通信轮次')
        ax.set_ylabel('每轮耗时（秒）')
        ax.grid(True, linestyle='--', alpha=0.22)

        legend_df = pd.DataFrame(plotted_rows).drop_duplicates(subset=['file'])
        run_handles, run_labels = self._run_legend_handles(legend_df)
        style_handles = [
            Line2D([0], [0], color='#444444', linestyle='-', linewidth=2),
            Line2D([0], [0], color='#444444', linestyle='--', linewidth=2),
        ]
        style_labels = ['ADP（实线）', 'FDP（虚线）']
        self._legend_panel(ax_leg, run_handles + style_handles, run_labels + style_labels, title='实验组与线型')

        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig5_round_time_curve.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def plot_score_heatmap(self, save=True):
        if not self._ensure_selection():
            return

        df = self._active_summary()
        if df.empty or len(df) < 2:
            print('[WARN] 至少需要 2 个日志才能绘制评分热力图。')
            return

        work = df.copy()
        work['privacy_gap'] = (work['tail_asr_mean'] - 0.5).abs()

        def to_score_smaller_better(series):
            s = pd.to_numeric(series, errors='coerce').astype(float)
            if s.isna().all():
                return np.full(len(s), 0.5)
            s = s.fillna(s.median())
            s_min, s_max = s.min(), s.max()
            if s_max <= s_min:
                return np.ones(len(s))
            return 1.0 - (s - s_min) / (s_max - s_min)

        utility_score = to_score_smaller_better(work['tail_rmse_mean'])
        privacy_score = to_score_smaller_better(work['privacy_gap'])
        stability_score = to_score_smaller_better(work['tail_asr_std'])

        if work['round_time_mean'].isna().all():
            efficiency_score = np.full(len(work), 0.5)
        else:
            rt = work['round_time_mean'].fillna(work['round_time_mean'].max())
            efficiency_score = to_score_smaller_better(rt)

        score_df = pd.DataFrame({
            '效用分数': np.asarray(utility_score, dtype=float),
            '隐私分数': np.asarray(privacy_score, dtype=float),
            '稳定性分数': np.asarray(stability_score, dtype=float),
            '效率分数': np.asarray(efficiency_score, dtype=float),
        }, index=work.apply(lambda r: self._label(r, compact=True), axis=1).values)

        score_df['综合分数'] = (
            0.36 * score_df['效用分数']
            + 0.36 * score_df['隐私分数']
            + 0.18 * score_df['稳定性分数']
            + 0.10 * score_df['效率分数']
        )
        score_df = score_df.sort_values('综合分数', ascending=False)

        fig, ax = plt.subplots(figsize=(12.2, 6.6), dpi=150)
        sns.heatmap(score_df, annot=True, fmt='.2f', cmap='YlGnBu', cbar=True, linewidths=0.5, ax=ax)
        ax.set_title('创新综合评分热力图')
        ax.set_xlabel('指标')
        ax.set_ylabel('实验组')

        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig6_score_heatmap.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def plot_metric_correlation(self, save=True):
        if not self._ensure_selection():
            return

        df = self._active_summary()
        if df.empty or len(df) < 2:
            print('[WARN] 至少需要 2 个日志才能绘制相关性热力图。')
            return

        corr_cols = [
            'tail_rmse_mean',
            'tail_asr_mean',
            'tail_asr_std',
            'tail_auc_mean',
            'sigma_eff_mean',
            'round_time_mean',
        ]
        num = df[corr_cols].copy()
        for c in corr_cols:
            num[c] = pd.to_numeric(num[c], errors='coerce')
            if num[c].isna().all():
                num[c] = 0.0
            else:
                num[c] = num[c].fillna(num[c].median())

        corr = num.corr(method='pearson')
        rename_map = {
            'tail_rmse_mean': 'Tail RMSE',
            'tail_asr_mean': 'Tail ASR',
            'tail_asr_std': 'ASR Std',
            'tail_auc_mean': 'Tail AUC',
            'sigma_eff_mean': 'Sigma*',
            'round_time_mean': 'Round Time',
        }
        corr = corr.rename(index=rename_map, columns=rename_map)

        fig, ax = plt.subplots(figsize=(8.8, 7.1), dpi=150)
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            linewidths=0.6,
            cbar_kws={'shrink': 0.85},
            ax=ax,
        )
        ax.set_title('指标相关性热力图（当前选择）')

        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig7_metric_correlation.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def plot_grouped_statistics(self, save=True):
        if not self._ensure_selection():
            return

        df = self._active_summary()
        if df.empty or len(df) < 2:
            print('[WARN] 至少需要 2 个日志才能绘制分组统计图。')
            return

        work = df.copy()
        work['privacy_gap'] = (work['tail_asr_mean'] - 0.5).abs()
        work['group'] = work.apply(lambda r: f"{r['mode']}-{r['algo']}-{r['adp']}", axis=1)

        grouped = work.groupby('group', as_index=False).agg(
            rmse_mean=('tail_rmse_mean', 'mean'),
            rmse_std=('tail_rmse_mean', 'std'),
            pgap_mean=('privacy_gap', 'mean'),
            pgap_std=('privacy_gap', 'std'),
            auc_mean=('tail_auc_mean', 'mean'),
            auc_std=('tail_auc_mean', 'std'),
            rt_mean=('round_time_mean', 'mean'),
            rt_std=('round_time_mean', 'std'),
            n=('group', 'count'),
        )

        for col in ['rmse_std', 'pgap_std', 'auc_std', 'rt_std']:
            grouped[col] = pd.to_numeric(grouped[col], errors='coerce').fillna(0.0)

        grouped = grouped.sort_values('pgap_mean', ascending=True)
        x = np.arange(len(grouped))
        bar_colors = sns.color_palette('Set2', n_colors=len(grouped))

        fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.2), dpi=150)

        def draw(ax, y, yerr, title, ylabel):
            ax.bar(x, y, yerr=yerr, capsize=3, color=bar_colors, alpha=0.88, edgecolor='white', linewidth=0.8)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(grouped['group'], rotation=18)
            ax.grid(True, axis='y', linestyle='--', alpha=0.18)

        draw(axes[0, 0], grouped['rmse_mean'], grouped['rmse_std'], '分组均值 Tail RMSE', 'RMSE')
        draw(axes[0, 1], grouped['pgap_mean'], grouped['pgap_std'], '分组均值 |ASR-0.5|', '|ASR-0.5|')
        draw(axes[1, 0], grouped['auc_mean'], grouped['auc_std'], '分组均值 Tail AUC', 'AUC')

        rt_mean = grouped['rt_mean'].fillna(grouped['rt_mean'].median() if not grouped['rt_mean'].isna().all() else 0.0)
        rt_std = grouped['rt_std'].fillna(0.0)
        draw(axes[1, 1], rt_mean, rt_std, '分组均值 Round Time', '秒')

        fig.suptitle('分组统计对比图（mode-algo-adp）', y=1.02, fontsize=12)
        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig8_grouped_statistics.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def plot_tail_ci_forest(self, save=True):
        if not self._ensure_selection():
            return

        items = self._iter_active()
        if len(items) < 2:
            print('[WARN] 至少需要 2 个日志才能绘制置信区间森林图。')
            return

        rows = []
        for i, (name, rec, row) in enumerate(items, start=1):
            tail = rec['tail']
            asr_vals = np.asarray(rec['attack_acc'][-tail:], dtype=float)
            rmse_vals = np.asarray(rec['rmse'][-tail:], dtype=float)
            n_asr = max(1, len(asr_vals))
            n_rmse = max(1, len(rmse_vals))

            asr_mean = float(np.mean(asr_vals))
            asr_ci = float(1.96 * np.std(asr_vals, ddof=1) / np.sqrt(n_asr)) if n_asr > 1 else 0.0
            rmse_mean = float(np.mean(rmse_vals))
            rmse_ci = float(1.96 * np.std(rmse_vals, ddof=1) / np.sqrt(n_rmse)) if n_rmse > 1 else 0.0

            rows.append({
                'label': f"#{i} {self._label(row, compact=True)}",
                'file': name,
                'asr_mean': asr_mean,
                'asr_ci': asr_ci,
                'rmse_mean': rmse_mean,
                'rmse_ci': rmse_ci,
            })

        stat_df = pd.DataFrame(rows)
        stat_df['asr_gap'] = (stat_df['asr_mean'] - 0.5).abs()
        stat_df = stat_df.sort_values('asr_gap', ascending=True).reset_index(drop=True)

        y = np.arange(len(stat_df))
        colors = [self._color_of(f) for f in stat_df['file']]

        fig, axes = plt.subplots(1, 2, figsize=(14.0, 8.0), dpi=150, sharey=True)

        axes[0].errorbar(
            stat_df['asr_mean'], y,
            xerr=stat_df['asr_ci'],
            fmt='o',
            markersize=5.5,
            ecolor='#444444',
            elinewidth=1,
            capsize=3,
            color='#444444',
        )
        for yy, xx, c in zip(y, stat_df['asr_mean'], colors):
            axes[0].scatter(xx, yy, s=40, color=c, zorder=3)
        axes[0].axvline(0.5, linestyle='--', color='#2f3e4d', linewidth=1.1)
        axes[0].set_title('Tail ASR 均值 ±95% CI')
        axes[0].set_xlabel('ASR')
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(stat_df['label'])
        axes[0].grid(True, axis='x', linestyle='--', alpha=0.2)

        axes[1].errorbar(
            stat_df['rmse_mean'], y,
            xerr=stat_df['rmse_ci'],
            fmt='o',
            markersize=5.5,
            ecolor='#444444',
            elinewidth=1,
            capsize=3,
            color='#444444',
        )
        for yy, xx, c in zip(y, stat_df['rmse_mean'], colors):
            axes[1].scatter(xx, yy, s=40, color=c, zorder=3)
        axes[1].set_title('Tail RMSE 均值 ±95% CI')
        axes[1].set_xlabel('RMSE')
        axes[1].grid(True, axis='x', linestyle='--', alpha=0.2)

        fig.suptitle('尾段置信区间森林图', y=0.98, fontsize=12)
        plt.tight_layout()
        if save:
            self._prepare_canvas()
            out = os.path.join(self.output_dir, 'fig9_tail_ci_forest.png')
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f'[SAVE] 已导出 {out}')

        finish_plot(fig)

    def export_selected_summary_csv(self):
        df = self._active_summary()
        if df.empty:
            print('[WARN] 当前选择为空，未导出汇总。')
            return

        self._prepare_canvas()
        out = os.path.join(self.output_dir, 'selected_experiment_summary.csv')
        df.sort_values(['mode', 'sigma', 'algo', 'personal', 'adp', 'file']).to_csv(out, index=False, encoding='utf-8-sig')
        print(f'[SAVE] 已导出 {out}')


    def generate_core_figures(self):
        self.plot_core_panel(save=True)
        self.plot_pareto_bubble(save=True)
        self.plot_tail_distribution(save=True)
        self.plot_noise_schedule(save=True)
        self.plot_round_time_curve(save=True)
        self.export_selected_summary_csv()

    def generate_stat_figures(self):
        self.plot_score_heatmap(save=True)
        self.plot_metric_correlation(save=True)
        self.plot_grouped_statistics(save=True)
        self.plot_tail_ci_forest(save=True)

    def generate_all_figures(self):
        self.generate_core_figures()
        self.generate_stat_figures()

    def menu(self):
        while True:
            selected_n = len(self.active_files)
            total_n = len(self.summary_df) if not self.summary_df.empty else 0
            print('\n' + '=' * 74)
            print(' FedRec 论文绘图系统（中文增强版）')
            print('=' * 74)
            print(f' 当前选择: {selected_n}/{total_n} 个日志 | 数据源: logs')
            print('-' * 74)
            print(' [s] 选择参与绘图日志（支持全选）')
            print(' [a] 直接选择全部日志')
            print(' [1] 一键生成核心五图（含每轮训练时间）')
            print(' [2] 三联核心曲线（RMSE/ASR/AUC）')
            print(' [3] 隐私-效用 Pareto 气泡图')
            print(' [4] 尾段 ASR 分布图')
            print(' [5] 噪声调度曲线图')
            print(' [6] 每轮训练时间曲线图（Round Time）')
            print(' [7] 创新综合评分热力图')
            print(' [8] 指标相关性热力图')
            print(' [9] 分组统计对比图')
            print(' [0] 尾段95%CI森林图')
            print(' [g] 生成全部图（核心+统计）')
            print(' [e] 导出当前选择汇总 CSV')
            print(' [r] 重新加载 logs')
            print(' [q] 退出')

            c = input('请选择: ').strip().lower()
            if c == 's':
                self.select_logs_interactive()
            elif c == 'a':
                self.select_all_logs()
            elif c == '1':
                self.generate_core_figures()
            elif c == '2':
                self.plot_core_panel(save=False)
            elif c == '3':
                self.plot_pareto_bubble(save=False)
            elif c == '4':
                self.plot_tail_distribution(save=False)
            elif c == '5':
                self.plot_noise_schedule(save=False)
            elif c == '6':
                self.plot_round_time_curve(save=False)
            elif c == '7':
                self.plot_score_heatmap(save=False)
            elif c == '8':
                self.plot_metric_correlation(save=False)
            elif c == '9':
                self.plot_grouped_statistics(save=False)
            elif c == '0':
                self.plot_tail_ci_forest(save=False)
            elif c == 'g':
                self.generate_all_figures()
            elif c == 'e':
                self.export_selected_summary_csv()
            elif c == 'r':
                if self.load_logs():
                    self.select_logs_interactive()
            elif c == 'q':
                break
            else:
                print('无效选项。')


if __name__ == '__main__':
    v = PaperVisualizer(
        logs_dir='logs',
        output_dir='figures/archive_exploratory_20260309',
    )
    if v.load_logs():
        v.select_logs_interactive()
        v.menu()

