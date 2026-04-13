# src/server_client.py
import copy
import torch
import torch.optim as optim
import numpy as np

from src.privacy import PrivacyEngine
from src.models import AdvancedNeuMF, is_personalized_param


class Client:
    def __init__(self, uid, train_data, test_data, config, feature_dim, personal_state=None):
        self.uid = uid
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim

        self.model = AdvancedNeuMF(
            self.config.NUM_USERS,
            self.config.NUM_ITEMS,
            self.feature_dim,
            emb_dim=self.config.EMBEDDING_DIM,
            enable_personalization=self.config.ENABLE_PERSONALIZATION,
        ).to(self.device)

        self.loss_fn = torch.nn.MSELoss()
        self.personal_state = personal_state or {}
        self.last_dp_meta = {'avg_sigma': 0.0, 'clip_coef': 1.0, 'total_norm': 0.0}

    def _adaptive_cfg(self):
        return {
            'enabled': bool(self.config.ENABLE_ADAPTIVE_DP),
            'sigma_min': float(self.config.DP_SIGMA_MIN),
            'sigma_max': float(self.config.DP_SIGMA_MAX),
            'progressive_decay': float(self.config.DP_PROGRESSIVE_DECAY),
            'sparsity_boost': float(self.config.DP_SPARSITY_BOOST),
        }

    def train(self, global_state, round_idx=0, total_rounds=1):
        """Local training: returns (shared_update_dict, avg_train_loss, dp_meta)."""
        # Load shared global params in non-strict mode so personalization toggles
        # or personalized-local params won't break state dict alignment.
        self.model.load_state_dict(global_state, strict=False)
        if self.config.ENABLE_PERSONALIZATION:
            self.model.load_personal_state(self.personal_state)

        self.model.train()
        opt = optim.Adam(self.model.parameters(), lr=self.config.LR)

        if not self.train_data:
            return None, 0.0, self.last_dp_meta

        mids, rates, feats = zip(*self.train_data)
        u_t = torch.LongTensor([self.uid] * len(mids)).to(self.device)
        i_t = torch.LongTensor(mids).to(self.device)
        r_t = torch.FloatTensor(rates).to(self.device)
        f_t = torch.FloatTensor(np.array(feats)).to(self.device)

        # Cache global reference on device for FedProx
        global_ref = {k: v.to(self.device) for k, v in global_state.items()}

        batch_size = max(1, int(self.config.BATCH_SIZE))
        n = len(mids)
        epoch_losses = []

        for _ in range(self.config.LOCAL_EPOCHS):
            perm = torch.randperm(n, device=self.device)
            batch_losses = []

            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                bu = u_t[idx]
                bi = i_t[idx]
                br = r_t[idx]
                bf = f_t[idx]

                opt.zero_grad()
                pred = self.model(bu, bi, bf)
                loss = self.loss_fn(pred, br)

                # FedProx regularization for better non-IID stability
                if self.config.FL_ALGO == 'FEDPROX' and self.config.PROX_MU > 0:
                    prox_term = 0.0
                    for name, param in self.model.named_parameters():
                        if is_personalized_param(name):
                            continue
                        prox_term = prox_term + torch.sum((param - global_ref[name]) ** 2)
                    loss = loss + 0.5 * float(self.config.PROX_MU) * prox_term

                loss.backward()

                if self.config.PRIVACY_MODE != 'PLAIN':
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.CLIP_NORM)

                opt.step()
                batch_losses.append(loss.item())

            epoch_losses.append(float(np.mean(batch_losses)) if batch_losses else 0.0)

        # Build shared update only (personalized params stay local)
        update = {}
        new_params = self.model.state_dict()
        for k in new_params:
            if is_personalized_param(k):
                continue
            update[k] = new_params[k] - global_ref[k]

        # LDP on shared update
        if self.config.PRIVACY_MODE == 'LDP':
            update, self.last_dp_meta = PrivacyEngine.clip_and_noise(
                update,
                sigma=self.config.DP_SIGMA,
                max_norm=self.config.CLIP_NORM,
                round_idx=round_idx,
                total_rounds=total_rounds,
                adaptive_cfg=self._adaptive_cfg(),
                return_meta=True,
            )
        else:
            self.last_dp_meta = {'avg_sigma': 0.0, 'clip_coef': 1.0, 'total_norm': 0.0}

        # Save local personalized head after local training
        if self.config.ENABLE_PERSONALIZATION:
            self.personal_state = self.model.export_personal_state()

        avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        return update, avg_epoch_loss, self.last_dp_meta

    def export_personal_state(self):
        return self.personal_state

    def evaluate(self):
        if not self.test_data:
            return 0.0

        self.model.eval()
        with torch.no_grad():
            mids, rates, feats = zip(*self.test_data)
            u_t = torch.LongTensor([self.uid] * len(mids)).to(self.device)
            i_t = torch.LongTensor(mids).to(self.device)
            r_t = torch.FloatTensor(rates).to(self.device)
            f_t = torch.FloatTensor(np.array(feats)).to(self.device)

            pred = self.model(u_t, i_t, f_t)
            loss = self.loss_fn(pred, r_t)
        return float(loss.item())


class Server:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.personal_states = {}
        self.last_dp_meta = {'avg_sigma': 0.0, 'clip_coef': 1.0, 'total_norm': 0.0}

    def _adaptive_cfg(self):
        return {
            'enabled': bool(self.config.ENABLE_ADAPTIVE_DP),
            'sigma_min': float(self.config.DP_SIGMA_MIN),
            'sigma_max': float(self.config.DP_SIGMA_MAX),
            'progressive_decay': float(self.config.DP_PROGRESSIVE_DECAY),
            'sparsity_boost': float(self.config.DP_SPARSITY_BOOST),
        }

    def get_personal_state(self, uid):
        return copy.deepcopy(self.personal_states.get(uid, {}))

    def set_personal_state(self, uid, state):
        if state is None:
            return
        self.personal_states[uid] = copy.deepcopy(state)

    def aggregate(self, updates, round_idx=0, total_rounds=1):
        """Aggregate shared updates only."""
        if not updates:
            return

        avg_up = {}
        for k in updates[0].keys():
            avg_up[k] = torch.stack([u[k] for u in updates]).mean(dim=0)

        if self.config.PRIVACY_MODE == 'CDP':
            avg_up, self.last_dp_meta = PrivacyEngine.clip_and_noise(
                avg_up,
                sigma=self.config.DP_SIGMA,
                max_norm=self.config.CLIP_NORM,
                round_idx=round_idx,
                total_rounds=total_rounds,
                adaptive_cfg=self._adaptive_cfg(),
                return_meta=True,
            )
        else:
            self.last_dp_meta = {'avg_sigma': 0.0, 'clip_coef': 1.0, 'total_norm': 0.0}

        state = self.model.state_dict()
        with torch.no_grad():
            for k in avg_up:
                if k in state:
                    state[k] += avg_up[k].to(self.device)
        self.model.load_state_dict(state)

    def get_state(self):
        return copy.deepcopy(self.model.state_dict())


