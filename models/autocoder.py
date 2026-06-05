"""
AutoCoder: Conv1D × 2 + Multi-head self-attention autoencoder.
Architecture from: Xu et al., "An efficient vehicular network anomaly detection
framework based on encoder and dynamic threshold adjustment", P2P Netw. Appl. 2025.

Encoder : Conv1(k=3,s=2,64) → ReLU → Conv2(k=3,s=2,32) → ReLU → MHA(32,4h)
Decoder : reshape → DeConv1(k=3,s=2,64) → ReLU → DeConv2(k=3,s=2,C) → ReLU
Training loss  : MAE (mean absolute reconstruction error)
Anomaly score  : per-window MAE; threshold = 95th pct of train scores (perthr)
"""

import numpy as np
import torch
import torch.nn as nn


class ConvAttentionEncoder(nn.Module):
    """Conv1D × 2 + Multi-head self-attention encoder."""

    def __init__(self, n_features: int = 10, num_heads: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=2, padding=1)
        self.act   = nn.ReLU()
        self.attn  = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)          # (B, C, T)
        x = self.act(self.conv1(x))    # (B, 64, T/2)
        x = self.act(self.conv2(x))    # (B, 32, T/4)
        x = x.transpose(1, 2)          # (B, T/4, 32)
        h, _ = self.attn(x, x, x)
        return h                        # (B, T/4, 32)


class ConvAttentionDecoder(nn.Module):
    def __init__(self, n_features: int = 10):
        super().__init__()
        self.deconv1 = nn.ConvTranspose1d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, n_features, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.act     = nn.ReLU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = h.transpose(1, 2)           # (B, 32, T/4)
        x = self.act(self.deconv1(x))   # (B, 64, T/2)
        x = self.act(self.deconv2(x))   # (B, n_features, T)
        return x.transpose(1, 2)        # (B, T, n_features)


class ConvAttentionAE(nn.Module):
    def __init__(self, n_features: int = 10, num_heads: int = 4):
        super().__init__()
        self.encoder = ConvAttentionEncoder(n_features, num_heads)
        self.decoder = ConvAttentionDecoder(n_features)

    def forward(self, x: torch.Tensor):
        x_hat  = self.decoder(self.encoder(x))
        scores = torch.mean(torch.abs(x - x_hat), dim=(1, 2))
        return scores.mean(), scores.detach()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, sc = self.forward(x)
        return sc

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(self.decoder(self.encoder(x)) - x), dim=(1, 2))


class AutoCoderModel:
    """
    Fit/score wrapper around ConvAttentionAE.
    Anomaly score = mean L1 reconstruction error per window.
    """

    def __init__(
        self,
        n_features: int = 10,
        seq_len: int    = 100,
        num_heads: int  = 4,
        lr: float       = 1e-3,
        epochs: int     = 50,
        batch_size: int = 64,
        device: str     = "cpu",
    ):
        self.model      = ConvAttentionAE(n_features, num_heads)
        self.seq_len    = seq_len
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.device     = device

    def fit(self, X: np.ndarray):
        from torch.utils.data import TensorDataset, DataLoader
        self.model.to(self.device).train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dl  = DataLoader(TensorDataset(torch.from_numpy(X).float()),
                         batch_size=self.batch_size, shuffle=True, drop_last=True)
        for epoch in range(self.epochs):
            total = 0.0
            for (batch,) in dl:
                batch = batch.to(self.device)
                opt.zero_grad()
                loss = self.model.reconstruction_error(batch).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                total += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  [AutoCoder] epoch {epoch+1}/{self.epochs}  loss={total/len(dl):.4f}")

    def score(self, X: np.ndarray) -> np.ndarray:
        from torch.utils.data import TensorDataset, DataLoader
        self.model.eval()
        dl = DataLoader(TensorDataset(torch.from_numpy(X).float()),
                        batch_size=self.batch_size, shuffle=False)
        scores = []
        with torch.no_grad():
            for (batch,) in dl:
                scores.append(self.model.reconstruction_error(batch.to(self.device)).cpu().numpy())
        return np.concatenate(scores)
