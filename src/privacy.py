# src/privacy.py
import math
import torch


class PrivacyEngine:
    @staticmethod
    def _clip_param_dict(param_dict, max_norm):
        eps = 1e-12
        max_norm = max(float(max_norm), eps)
        total_norm_sq = 0.0
        for p in param_dict.values():
            total_norm_sq += p.detach().data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm_sq)
        clip_coef = max(1.0, total_norm / max_norm)

        clipped = {k: v / clip_coef for k, v in param_dict.items()}
        return clipped, total_norm, clip_coef

    @staticmethod
    def _adaptive_sigma(
        base_sigma,
        tensor,
        max_norm,
        round_idx,
        total_rounds,
        adaptive_cfg,
    ):
        if not adaptive_cfg or not adaptive_cfg.get('enabled', False):
            return max(float(base_sigma), 0.0)

        sigma = max(float(base_sigma), 0.0)
        if sigma == 0.0:
            return 0.0

        progress = float(round_idx) / max(float(total_rounds - 1), 1.0)
        decay = float(adaptive_cfg.get('progressive_decay', 0.4))
        # Early rounds: stronger protection; late rounds: improve utility
        schedule_factor = 1.0 + decay * (1.0 - progress)

        t_norm = tensor.detach().data.norm(2).item()
        sensitivity = math.sqrt(max(t_norm / max(float(max_norm), 1e-12), 1e-12))
        sensitivity_factor = min(1.5, max(0.8, sensitivity))

        sparsity_factor = 1.0
        sparsity_boost = float(adaptive_cfg.get('sparsity_boost', 0.0))
        if sparsity_boost > 0:
            nonzero_ratio = (tensor.detach().abs() > 1e-8).float().mean().item()
            if nonzero_ratio < 0.05:
                sparsity_factor += sparsity_boost

        sigma = sigma * schedule_factor * sensitivity_factor * sparsity_factor

        sigma_min = float(adaptive_cfg.get('sigma_min', 0.0))
        sigma_max = float(adaptive_cfg.get('sigma_max', sigma))
        sigma = max(sigma_min, min(sigma, sigma_max))
        return sigma

    @staticmethod
    def clip_and_noise(
        param_dict,
        sigma,
        max_norm,
        round_idx=0,
        total_rounds=1,
        adaptive_cfg=None,
        return_meta=False,
    ):
        """
        Differential privacy primitive: clipping + Gaussian noise.
        Supports adaptive per-layer sigma for better utility-security trade-off.
        """
        clipped_dict, total_norm, clip_coef = PrivacyEngine._clip_param_dict(param_dict, max_norm)

        noisy_dict = {}
        sigma_trace = []

        for name, param in clipped_dict.items():
            layer_sigma = PrivacyEngine._adaptive_sigma(
                base_sigma=sigma,
                tensor=param,
                max_norm=max_norm,
                round_idx=round_idx,
                total_rounds=total_rounds,
                adaptive_cfg=adaptive_cfg,
            )
            sigma_trace.append(layer_sigma)

            if layer_sigma > 0:
                noise = torch.normal(
                    mean=0.0,
                    std=layer_sigma,
                    size=param.shape,
                    device=param.device,
                    dtype=param.dtype,
                )
                noisy_dict[name] = param + noise
            else:
                noisy_dict[name] = param

        meta = {
            'total_norm': float(total_norm),
            'clip_coef': float(clip_coef),
            'avg_sigma': float(sum(sigma_trace) / max(len(sigma_trace), 1)),
            'max_sigma': float(max(sigma_trace) if sigma_trace else 0.0),
            'min_sigma': float(min(sigma_trace) if sigma_trace else 0.0),
        }

        if return_meta:
            return noisy_dict, meta
        return noisy_dict
