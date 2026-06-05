"""
srcae.py
--------
Sparse Regularisation Convolutional AutoEncoder (SRCAE), adapted from:
  "DESC-IDS: Toward Explainable Anomaly Detection for CAN Bus Intrusion
   Detection System Using Deep Evolving Stream Clustering"

Architecture
------------
Input  : (B, N, F)  — sliding window of CAN messages
Encoder: treat window as a 2-D image (B, 1, N, F);
         2 × dilated causal Conv2D + BatchNorm + LeakyReLU →
         AdaptiveAvgPool2d(1) → Linear → latent z ∈ R^d
Decoder: Linear → GELU → Linear → reshape  (MLP, matches other baselines)

Loss
----
  L = J_AE  +  λ · J_wd  +  β · J_sparse
  J_AE     = mean MSE(x, x̂)               (reconstruction)
  J_wd     = Σ ||W_i||²_F over conv layers  (weight-decay / Frobenius penalty)
  J_sparse = mean ||z||_1  over batch        (sparse bottleneck)

Anomaly score: per-sample MSE (same as all other AE baselines).

API: forward(x) → (loss, scores)   — matches LSTMAE / GRUAE / CNNAE.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalDilatedConv2d(nn.Module):
    """
    2-D convolution that is causal in the time axis (dim 2) and
    symmetric in the feature axis (dim 3).

    Causality: left-pad (k_t - 1)*d_t steps, zero right-pad.
    Symmetry : symmetric pad (k_f - 1)*d_f // 2 on both sides.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, dilation: Tuple[int, int] = (1, 1)):
        super().__init__()
        k = kernel_size
        dt, df = dilation
        self.pad_t_left = (k - 1) * dt   # causal: only past
        self.pad_f      = (k - 1) * df // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, (k, k),
            dilation=(dt, df), bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F)
        x = F.pad(x, (self.pad_f, self.pad_f, self.pad_t_left, 0))
        return self.conv(x)


class SRCAE(nn.Module):

    def __init__(
        self,
        n_features: int = 10,
        seq_len: int = 100,
        d_model: int = 64,
        lambda_wd: float = 1e-3,
        beta_sparse: float = 0.1,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.lambda_wd  = lambda_wd
        self.beta_sparse = beta_sparse

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = _CausalDilatedConv2d(1,  16, kernel_size=3, dilation=(1, 1))
        self.bn1  = nn.BatchNorm2d(16)
        self.enc2 = _CausalDilatedConv2d(16, 32, kernel_size=3, dilation=(2, 2))
        self.bn2  = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)           # (B, 32, 1, 1)
        self.latent_proj = nn.Linear(32, d_model)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec_fc1 = nn.Linear(d_model, d_model * 2)
        self.dec_fc2 = nn.Linear(d_model * 2, seq_len * n_features)

        self._conv_weights = [self.enc1.conv.weight, self.enc2.conv.weight]

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, F)  →  z: (B, d_model)"""
        h = x.unsqueeze(1)                           # (B, 1, N, F)
        h = F.leaky_relu(self.bn1(self.enc1(h)), 0.2)
        h = F.leaky_relu(self.bn2(self.enc2(h)), 0.2)
        h = self.pool(h).flatten(1)                  # (B, 32)
        return self.latent_proj(h)                   # (B, d_model)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_model)  →  x_hat: (B, N, F)"""
        h = F.gelu(self.dec_fc1(z))
        return self.dec_fc2(h).reshape(-1, self.seq_len, self.n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        loss   : scalar training loss  L = J_AE + λ·J_wd + β·J_sparse
        scores : per-sample MSE  (B,)  — used as anomaly scores
        """
        z     = self._encode(x)
        x_hat = self._decode(z)

        per_sample_mse = ((x - x_hat) ** 2).mean(dim=(1, 2))   # (B,)
        J_AE     = per_sample_mse.mean()
        J_wd     = sum(w.pow(2).sum() for w in self._conv_weights)
        J_sparse = z.abs().mean()

        loss = J_AE + self.lambda_wd * J_wd + self.beta_sparse * J_sparse
        return loss, per_sample_mse.detach()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, sc = self.forward(x)
        return sc
