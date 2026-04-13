# src/attack.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def extract_gradient_features(param_dict):
    """Sparse-aware gradient feature extractor for MIA."""
    features = []

    for name, grad in sorted(param_dict.items()):
        grad = grad.float().cpu().detach()
        flat_grad = grad.view(-1)

        non_zero_mask = flat_grad.abs() > 1e-8
        non_zero_count = non_zero_mask.sum().item()
        total_params = flat_grad.numel()
        sparsity_ratio = non_zero_count / max(total_params, 1)
        features.append(sparsity_ratio)

        if non_zero_count > 0:
            active_grad = flat_grad[non_zero_mask]

            if active_grad.numel() > 1:
                features.append(active_grad.norm(2).item())
                features.append(active_grad.mean().item())
                features.append(active_grad.std().item())
                features.append(active_grad.max().item())
                features.append(active_grad.min().item())
            elif active_grad.numel() == 1:
                features.append(active_grad.norm(2).item())
                features.append(active_grad.mean().item())
                features.append(0.0)
                features.append(active_grad.max().item())
                features.append(active_grad.min().item())
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            if non_zero_count > 3:
                q = torch.quantile(active_grad, torch.tensor([0.25, 0.5, 0.75]))
                features.extend(q.tolist())
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * 8)

    feature_tensor = torch.FloatTensor(features)
    feature_tensor = torch.sign(feature_tensor) * torch.log1p(feature_tensor.abs())
    return torch.nan_to_num(feature_tensor, 0.0)


def binary_auc(y_true, y_score):
    """AUC without external dependencies. O(P*N), but robust for small batches."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg)
        wins += 0.5 * np.sum(p == neg)
    return float(wins / (len(pos) * len(neg)))


class MembershipAttacker(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class MembershipAttackTrainer:
    def __init__(self, num_features, lr=0.001, buffer_size=2000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MembershipAttacker(num_features).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()

        self.buffer_size = buffer_size
        self.member_buffer = []
        self.non_member_buffer = []

    def add_data(self, member_grads, non_member_grads):
        for g in member_grads:
            self.member_buffer.append(extract_gradient_features(g))
        for g in non_member_grads:
            self.non_member_buffer.append(extract_gradient_features(g))

        if len(self.member_buffer) > self.buffer_size:
            self.member_buffer = self.member_buffer[-self.buffer_size:]
        if len(self.non_member_buffer) > self.buffer_size:
            self.non_member_buffer = self.non_member_buffer[-self.buffer_size:]

    def train_epoch(self, epochs=5):
        if len(self.member_buffer) < 32 or len(self.non_member_buffer) < 32:
            return 0.0

        m_data = torch.stack(self.member_buffer).to(self.device)
        n_data = torch.stack(self.non_member_buffer).to(self.device)

        min_len = min(len(m_data), len(n_data))
        m_indices = torch.randperm(len(m_data), device=self.device)[:min_len]
        n_indices = torch.randperm(len(n_data), device=self.device)[:min_len]

        x = torch.cat([m_data[m_indices], n_data[n_indices]], dim=0)
        y = torch.cat([torch.ones(min_len, 1), torch.zeros(min_len, 1)], dim=0).to(self.device)

        perm = torch.randperm(len(x), device=self.device)
        x, y = x[perm], y[perm]

        self.model.train()
        total_loss = 0.0
        total_steps = 0

        batch_size = 64
        for _ in range(epochs):
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i + batch_size]
                batch_y = y[i:i + batch_size]

                # BatchNorm requires at least 2 samples during training
                if batch_x.shape[0] < 2:
                    continue

                self.optimizer.zero_grad()
                out = self.model(batch_x)
                loss = self.criterion(out, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_steps += 1

        if total_steps == 0:
            return 0.0
        return total_loss / total_steps

    def test_metrics(self, member_grads, non_member_grads):
        if not member_grads or not non_member_grads:
            return {
                'acc': 0.5,
                'auc': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
            }

        self.model.eval()
        with torch.no_grad():
            m_feats = [extract_gradient_features(g) for g in member_grads]
            n_feats = [extract_gradient_features(g) for g in non_member_grads]

            x = torch.stack(m_feats + n_feats).to(self.device)
            y_true = np.array([1] * len(m_feats) + [0] * len(n_feats))

            logits = self.model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            preds = (probs > 0.5).astype(int)

        acc = float((preds == y_true).mean())
        auc = binary_auc(y_true, probs)

        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            'acc': float(acc),
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    def test(self, member_grads, non_member_grads):
        # Backward-compatible API
        return self.test_metrics(member_grads, non_member_grads)['acc']
