"""Kahlus patch transformer for the scaling sweep (issue #11).

Tokens are 1 s patches, one per channel and second. Attention is causal in time: a token sees every
channel at its own second and earlier, never later. Two heads train together:

* masked: rebuild patches that were hidden from the input;
* future: from each token, predict the next ``horizon`` patches of its channel.

Because no token sees the future, the future head at the last context second is a strictly-future
forecast, which is what the benchmark scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

#: Width, depth and heads per sweep point. The names are the approximate parameter counts.
SIZES = {
    "tiny": {"d_model": 32, "n_layers": 2, "n_heads": 2},  # CPU proofs and tests only
    "10m": {"d_model": 384, "n_layers": 6, "n_heads": 6},
    "50m": {"d_model": 640, "n_layers": 10, "n_heads": 10},
    "200m": {"d_model": 1024, "n_layers": 16, "n_heads": 16},
    "1b": {"d_model": 1792, "n_layers": 26, "n_heads": 28},
}


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    n_channels: int = 2
    patch_len: int = 125
    context: int = 64
    horizon: int = 4
    qk_norm: bool = True  # LayerNorm on queries and keys; bounds attention logits (arXiv 2309.14322)
    objective: str = "signal"  # "signal": targets are patches. "latent": targets are hidden states of a slow copy of the encoder
    mlp_head: bool = False  # heads are Linear(d, d), GELU, Linear(d, out) instead of one Linear (E1); old checkpoints have False


class Block(nn.Module):
    def __init__(self, d: int, heads: int, qk_norm: bool):
        super().__init__()
        self.heads = heads
        self.q_norm = nn.LayerNorm(d // heads) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(d // heads) if qk_norm else nn.Identity()
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(self.q_norm(q), self.k_norm(k), v, attn_mask=mask)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


class KahlusPatchModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Linear(cfg.patch_len, d)
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.channel = nn.Parameter(torch.randn(cfg.n_channels, d) * 0.02)
        # the latent teacher also reads the horizon seconds after the context, so its time table is longer
        seconds = cfg.context + (cfg.horizon if cfg.objective == "latent" else 0)
        self.time = nn.Parameter(torch.randn(seconds, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, cfg.n_heads, cfg.qk_norm) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(d)
        out = d if cfg.objective == "latent" else cfg.patch_len
        head = (lambda n: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n))) if cfg.mlp_head else (lambda n: nn.Linear(d, n))
        self.recon = head(out)
        self.future = head(cfg.horizon * out)
        self.last_target_std = torch.zeros(())  # latent only: mean over features of the std over tokens of the targets
        second = torch.arange(seconds).repeat_interleave(cfg.n_channels)
        self.register_buffer("causal", second[:, None] >= second[None, :], persistent=False)

    def hidden(self, x: torch.Tensor, masked: torch.Tensor | None = None) -> torch.Tensor:
        """Normed encoder states ``(batch, seconds, channels, d_model)``. The state at second s reads seconds 0..s only."""
        b, t, c, _ = x.shape
        h = self.embed(x)
        if masked is not None:
            h = torch.where(masked[..., None], self.mask_token.to(h.dtype), h)
        h = (h + self.channel + self.time[:t, None]).reshape(b, t * c, -1)
        mask = self.causal[: t * c, : t * c]
        for block in self.blocks:
            h = block(h, mask)
        return self.norm(h).view(b, t, c, -1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The state of the last second of every window: ``(batch, channels, d_model)``, for probes and fine-tuning."""
        return self.hidden(x)[:, -1]

    def predict(self, x: torch.Tensor, masked: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` is ``(batch, seconds, channels, patch_len)``. Returns the rebuilt patches, same shape, and
        the forecasts ``(batch, seconds, channels, horizon, patch_len)``."""
        b, t, c, p = x.shape
        h = self.hidden(x, masked)
        return self.recon(h), self.future(h).view(b, t, c, self.cfg.horizon, -1)

    def forward(self, window: torch.Tensor, masked: torch.Tensor, teacher_hidden: torch.Tensor | None = None) -> torch.Tensor:
        """Training losses for ``window`` of ``context + horizon`` seconds: ``[total, masked, future]``.
        ``teacher_hidden`` (latent objective) is the unmasked teacher's states for the whole window; its layer-normed
        values replace the signal as targets. Windows whose ECG tokens are all masked (the ECG-drop of E1) carry no
        loss on the ECG tokens."""
        t, hz = self.cfg.context, self.cfg.horizon
        x = window[:, :t]
        recon, future = self.predict(x, masked)
        seq = window if teacher_hidden is None else F.layer_norm(teacher_hidden.float(), teacher_hidden.shape[-1:])
        if teacher_hidden is not None:
            self.last_target_std = seq.flatten(0, 2).std(0).mean().detach()
        w = torch.ones_like(masked, dtype=torch.float32)
        if x.shape[2] > 1:
            w[:, :, 1] = 1 - masked[:, :, 1].all(1, keepdim=True).float()  # 1 = ECG token counts
        m = (masked.float() * w)[..., None]
        l_masked = (((recon.float() - seq[:, :t]) ** 2) * m).sum() / (m.sum() * recon.shape[-1]).clamp(min=1)
        l_future = (((future.float() - future_targets(seq, t, hz)) ** 2) * w[..., None, None]).sum() / (w.sum() * hz * future.shape[-1])
        return torch.stack([l_masked + l_future, l_masked, l_future])

    @torch.no_grad()
    def forecast(self, context: torch.Tensor) -> torch.Tensor:
        """Strictly-future forecast ``(batch, horizon, channels, patch_len)`` from unmasked context."""
        return self.predict(context)[1][:, -1].transpose(1, 2)


def future_targets(seq: torch.Tensor, t: int, hz: int) -> torch.Tensor:
    """Target of the token at second s: seconds s+1 .. s+horizon of the same channel, ``(batch, t, channels, horizon, features)``.
    Never a second at or before s."""
    return seq.unfold(1, hz, 1)[:, 1 : t + 1].permute(0, 1, 2, 4, 3)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
