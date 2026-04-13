# main.py
import os
import sys
import json
import time
import math
import random
import datetime as dt

import torch
import numpy as np

from src.config import Config
from src.dataset import load_all_data
from src.models import AdvancedNeuMF
from src.server_client import Server, Client
from src.attack import MembershipAttackTrainer, extract_gradient_features
from src.experiment_io import build_result_path


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_snapshot():
    keys = [
        'ROUNDS', 'LOCAL_EPOCHS', 'LR', 'EMBEDDING_DIM', 'BATCH_SIZE',
        'FL_ALGO', 'PROX_MU', 'USERS_PER_ROUND',
        'ENABLE_PERSONALIZATION',
        'PRIVACY_MODE', 'DP_SIGMA', 'CLIP_NORM',
        'ENABLE_ADAPTIVE_DP', 'DP_SIGMA_MIN', 'DP_SIGMA_MAX',
        'DP_PROGRESSIVE_DECAY', 'DP_SPARSITY_BOOST',
        'ATTACK_ENABLED', 'TAIL_WINDOW', 'RANDOM_SEED',
    ]
    return {k: getattr(Config, k) for k in keys}


class InteractiveSystem:
    def __init__(self):
        set_random_seed(Config.RANDOM_SEED)

        self.data_loaded = False
        self.sys_init = False
        self.train_data = None
        self.test_data = None
        self.stats = None
        self.server = None
        self.attacker = None

        self.history = {
            'train_loss': [],
            'test_loss': [],
            'rmse': [],
            'attack_acc': [],
            'attack_auc': [],
            'attack_precision': [],
            'attack_recall': [],
            'attack_f1': [],
            'attack_loss': [],
            'privacy_sigma': [],
            'round_time': [],
        }

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_menu(self):
        print('\n' + '=' * 72)
        print('   FedRec 联邦学习隐私攻防实验平台 (Upgraded: FedProx + Personalization + AdaptiveDP)')
        print('=' * 72)
        print(
            f" [Config] Mode: {Config.PRIVACY_MODE} | Sigma: {Config.DP_SIGMA} | "
            f"Algo: {Config.FL_ALGO} | PHead: {Config.ENABLE_PERSONALIZATION} | ADP: {Config.ENABLE_ADAPTIVE_DP} | Seed: {Config.RANDOM_SEED}"
        )
        print('-' * 72)
        print(' [1]  加载数据 (Load Data)')
        print(' [2]   配置参数 (Settings)')
        print(' [3]  初始化系统 (Init Models)')
        print(' [4]  开始训练 (Start Training)')
        print(' [5]  保存日志 (Save Logs)')
        print(' [q]  退出 (Quit)')
        print('-' * 72)

    def step_1(self):
        print('\n>>> [1] Loading dataset...')
        try:
            self.train_data, self.test_data, self.stats, _ = load_all_data(
                Config.DATA_PATH, random_seed=Config.RANDOM_SEED
            )
            Config.NUM_USERS = self.stats['n_users']
            Config.NUM_ITEMS = self.stats['n_items']
            Config.FEATURE_DIM = self.stats['feature_dim']
            self.data_loaded = True
            print(
                '[OK] Data loaded. '
                f"Users: {Config.NUM_USERS}, Items: {Config.NUM_ITEMS}, FeatureDim: {Config.FEATURE_DIM}"
            )
        except Exception as e:
            print(f'[ERR] Error: {e}')

    def step_2(self):
        print('\n>>> [2] Config Settings')

        mode = input(f"Privacy Mode (Current: {Config.PRIVACY_MODE}) [PLAIN/LDP/CDP]: ").upper().strip()
        if mode in ['PLAIN', 'LDP', 'CDP']:
            Config.PRIVACY_MODE = mode

        sigma = input(f"DP Sigma (Current: {Config.DP_SIGMA}): ").strip()
        if sigma:
            Config.DP_SIGMA = float(sigma)

        algo = input(f"FL Algo (Current: {Config.FL_ALGO}) [FEDAVG/FEDPROX]: ").upper().strip()
        if algo in ['FEDAVG', 'FEDPROX']:
            Config.FL_ALGO = algo

        mu = input(f"FedProx MU (Current: {Config.PROX_MU}): ").strip()
        if mu:
            Config.PROX_MU = float(mu)

        phead = input(f"Enable Personalization (Current: {Config.ENABLE_PERSONALIZATION}) [y/n]: ").lower().strip()
        if phead in ['y', 'n']:
            Config.ENABLE_PERSONALIZATION = (phead == 'y')

        adp = input(f"Enable Adaptive DP (Current: {Config.ENABLE_ADAPTIVE_DP}) [y/n]: ").lower().strip()
        if adp in ['y', 'n']:
            Config.ENABLE_ADAPTIVE_DP = (adp == 'y')

        rounds = input(f"Total Rounds (Current: {Config.ROUNDS}): ").strip()
        if rounds:
            Config.ROUNDS = int(rounds)

        users_per_round = input(f"Users Per Round (Current: {Config.USERS_PER_ROUND}): ").strip()
        if users_per_round:
            Config.USERS_PER_ROUND = int(users_per_round)

        seed = input(f"Random Seed (Current: {Config.RANDOM_SEED}): ").strip()
        if seed:
            Config.RANDOM_SEED = int(seed)
            set_random_seed(Config.RANDOM_SEED)

        print('[OK] Config updated.')

    def step_3(self):
        if not self.data_loaded:
            print('[WARN] Load data first!')
            return

        print('\n>>> [3] Initializing System...')
        global_model = AdvancedNeuMF(
            Config.NUM_USERS,
            Config.NUM_ITEMS,
            Config.FEATURE_DIM,
            emb_dim=Config.EMBEDDING_DIM,
            enable_personalization=Config.ENABLE_PERSONALIZATION,
        )
        self.server = Server(global_model, Config)

        if Config.ATTACK_ENABLED:
            first_uid = next(iter(self.train_data.keys()))
            dummy_client = Client(
                first_uid,
                self.train_data[first_uid],
                [],
                Config,
                Config.FEATURE_DIM,
                personal_state=self.server.get_personal_state(first_uid),
            )
            dummy_update, _, _ = dummy_client.train(self.server.get_state(), round_idx=0, total_rounds=1)

            if dummy_update:
                feat_dim = len(extract_gradient_features(dummy_update))
                self.attacker = MembershipAttackTrainer(num_features=feat_dim, lr=0.003, buffer_size=2000)
                print(f'   --> Attacker Feature Dimension: {feat_dim}')
                print('   --> Attacker Model Ready.')

        self.sys_init = True
        self.history = {k: [] for k in self.history}
        print('[OK] System Initialized.')

    def step_4(self):
        if not self.sys_init:
            print('[WARN] Init system first!')
            return

        total_rounds = int(Config.ROUNDS)
        print(f'\n Starting Training for {total_rounds} rounds...')

        users = list(self.train_data.keys())
        if len(users) < 2:
            print('[ERR] 用户数不足，无法进行成员/非成员分组攻击评估。')
            return

        for r in range(total_rounds):
            round_start = time.time()

            member_n = min(Config.USERS_PER_ROUND, len(users))
            selected_members = np.random.choice(users, member_n, replace=False)

            candidates = list(set(users) - set(selected_members))
            non_member_n = min(member_n, len(candidates))
            if non_member_n == 0:
                selected_non_members = []
            else:
                selected_non_members = np.random.choice(candidates, non_member_n, replace=False)

            member_updates = []
            non_member_grads = []
            train_losses = []
            test_losses = []
            ldp_sigma_trace = []

            global_state = self.server.get_state()

            # Member updates (used by global aggregation)
            for uid in selected_members:
                uid = int(uid)
                client = Client(
                    uid,
                    self.train_data[uid],
                    self.test_data[uid],
                    Config,
                    Config.FEATURE_DIM,
                    personal_state=self.server.get_personal_state(uid),
                )
                up, t_loss, dp_meta = client.train(global_state, round_idx=r, total_rounds=total_rounds)
                self.server.set_personal_state(uid, client.export_personal_state())

                if up:
                    member_updates.append(up)
                    train_losses.append(t_loss)
                    if Config.PRIVACY_MODE == 'LDP':
                        ldp_sigma_trace.append(float(dp_meta.get('avg_sigma', 0.0)))

                v_loss = client.evaluate()
                test_losses.append(v_loss)

            # Non-member shadow gradients (for MIA only)
            for uid in selected_non_members:
                uid = int(uid)
                shadow_client = Client(
                    uid,
                    self.train_data[uid],
                    [],
                    Config,
                    Config.FEATURE_DIM,
                    personal_state=self.server.get_personal_state(uid),
                )
                nm_up, _, _ = shadow_client.train(global_state, round_idx=r, total_rounds=total_rounds)
                if nm_up:
                    non_member_grads.append(nm_up)

            if member_updates:
                self.server.aggregate(member_updates, round_idx=r, total_rounds=total_rounds)

            # Attack simulation
            attack_metrics = {
                'acc': 0.5,
                'auc': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }
            att_loss = 0.0

            if self.attacker and member_updates and non_member_grads:
                self.attacker.add_data(member_updates, non_member_grads)
                att_loss = self.attacker.train_epoch(epochs=15)
                attack_metrics = self.attacker.test_metrics(member_updates, non_member_grads)

            avg_train = float(np.mean(train_losses)) if train_losses else 0.0
            avg_test = float(np.mean(test_losses)) if test_losses else 0.0
            rmse = float(math.sqrt(max(avg_test, 0.0)))

            if Config.PRIVACY_MODE == 'CDP':
                privacy_sigma = float(self.server.last_dp_meta.get('avg_sigma', 0.0))
            elif Config.PRIVACY_MODE == 'LDP':
                privacy_sigma = float(np.mean(ldp_sigma_trace)) if ldp_sigma_trace else 0.0
            else:
                privacy_sigma = 0.0

            self.history['train_loss'].append(avg_train)
            self.history['test_loss'].append(avg_test)
            self.history['rmse'].append(rmse)
            self.history['attack_acc'].append(float(attack_metrics['acc']))
            self.history['attack_auc'].append(float(attack_metrics['auc']))
            self.history['attack_precision'].append(float(attack_metrics['precision']))
            self.history['attack_recall'].append(float(attack_metrics['recall']))
            self.history['attack_f1'].append(float(attack_metrics['f1']))
            self.history['attack_loss'].append(float(att_loss))
            self.history['privacy_sigma'].append(privacy_sigma)
            self.history['round_time'].append(float(time.time() - round_start))

            if (r + 1) % 10 == 0 or r == 0:
                print(
                    f"[Round] [Round {r + 1:3d}/{total_rounds}] "
                    f"Loss(Tr/Te): {avg_train:.4f}/{avg_test:.4f} | RMSE: {rmse:.4f} | "
                    f"ASR: {attack_metrics['acc']:.4f} | AUC: {attack_metrics['auc']:.4f} | "
                    f"Sigma*: {privacy_sigma:.4f}"
                )

        print('\n[OK] Training Complete.')
        best_acc = max(self.history['attack_acc']) if self.history['attack_acc'] else 0.5
        tail = min(Config.TAIL_WINDOW, len(self.history['attack_acc']))
        tail_acc = float(np.mean(self.history['attack_acc'][-tail:])) if tail > 0 else 0.5
        print(f'   Max Privacy Risk (ASR Max): {best_acc:.4f}')
        print(f'   Tail Avg Privacy Risk (ASR Mean@{tail}): {tail_acc:.4f}')

    def step_5(self):
        if not self.history['train_loss']:
            print('[WARN] 没有数据可保存。')
            return

        filename = build_result_path(Config.SAVE_DIR, Config)

        save_data = {k: [float(x) if x is not None else 0.0 for x in v] for k, v in self.history.items()}
        save_data['meta'] = {
            'created_at': dt.datetime.now().isoformat(timespec='seconds'),
            'config': config_snapshot(),
            'dataset': self.stats or {},
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f'[OK] 结果已保存至 {filename}')

    def loop(self):
        while True:
            self.print_menu()
            cmd = input('Command: ').lower().strip()
            if cmd == '1':
                self.step_1()
            elif cmd == '2':
                self.step_2()
            elif cmd == '3':
                self.step_3()
            elif cmd == '4':
                self.step_4()
            elif cmd == '5':
                self.step_5()
            elif cmd == 'q':
                sys.exit()
            else:
                print('Invalid Command')


if __name__ == '__main__':
    InteractiveSystem().loop()
