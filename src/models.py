# src/models.py
import torch
import torch.nn as nn


PERSONAL_PARAM_PREFIX = 'personal_head'


def is_personalized_param(name: str) -> bool:
    return name.startswith(PERSONAL_PARAM_PREFIX)


class AdvancedNeuMF(nn.Module):
    def __init__(self, num_users, num_items, feature_dim, emb_dim=32, hidden_dim=64, enable_personalization=True):
        super().__init__()
        self.enable_personalization = enable_personalization

        # Shared representation
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.feat_encoder = nn.Linear(feature_dim, emb_dim)

        self.shared_backbone = nn.Sequential(
            nn.Linear(emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.shared_head = nn.Linear(hidden_dim // 2, 1)

        # Personalized lightweight head (kept local, not aggregated)
        if self.enable_personalization:
            self.personal_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        else:
            self.personal_head = None

    def forward(self, user_idx, item_idx, feat_vecs):
        u = self.user_emb(user_idx)
        i = self.item_emb(item_idx)
        f = self.feat_encoder(feat_vecs)

        x = torch.cat([u, i, f], dim=1)
        h = self.shared_backbone(x)
        y = self.shared_head(h)

        if self.personal_head is not None:
            y = y + self.personal_head(h)

        return y.squeeze(-1)

    def export_personal_state(self):
        """Export personalized parameters only (stored on server per client)."""
        out = {}
        for name, param in self.state_dict().items():
            if is_personalized_param(name):
                out[name] = param.detach().cpu().clone()
        return out

    def load_personal_state(self, personal_state):
        if not personal_state:
            return
        state = self.state_dict()
        for name, value in personal_state.items():
            if name in state:
                state[name] = value.to(state[name].device)
        self.load_state_dict(state)


class AttackModel(nn.Module):
    """Legacy attack model kept for compatibility."""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)
