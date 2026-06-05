"""
Conv1D AE — plain 1D convolutional autoencoder, no period detection.
Used as the base variant in the stacked ablation (before TMPO is added).

Encoder : Linear proj → 3 × Conv1D(stride=2) → mean pool → latent z
Decoder : reuses AEDecoder from can_tempo (Linear → reshape)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


from .can_tempo import AEDecoder, stat_vector


class Conv1DEncoder(nn.Module):
    def __init__(self, n_features: int = 10, d_model: int = 256, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Linear(n_features, d_model)
        self.convs = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F)
        h = self.proj(x)                        # (B, N, d_model)
        h = self.convs(h.transpose(1, 2)).transpose(1, 2)  # (B, N, d_model)
        h = self.norm(h)
        if self.pooling == "max":
            return h.max(dim=1).values          # (B, d_model)
        return h.mean(dim=1)                    # (B, d_model)


class ConvAE(nn.Module):
    """1-D CNN encoder + MLP decoder autoencoder baseline."""

    def __init__(self, n_features: int = 10, seq_len: int = 100,
                 d_model: int = 64, n_layers: int = 3, kernel_size: int = 3):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features

        self.input_proj = nn.Linear(n_features, d_model)
        conv_layers = []
        for _ in range(n_layers):
            conv_layers += [
                nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ]
        self.convs   = nn.Sequential(*conv_layers)
        self.dec_fc1 = nn.Linear(d_model, d_model * 2)
        self.dec_fc2 = nn.Linear(d_model * 2, seq_len * n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, nf = x.shape
        h = self.input_proj(x).permute(0, 2, 1)
        z = self.convs(h).mean(dim=-1)
        h     = F.gelu(self.dec_fc1(z))
        x_hat = self.dec_fc2(h).reshape(B, N, nf)
        scores = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return scores.mean(), scores.detach()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, sc = self.forward(x)
        return sc


class LSTMAE(nn.Module):
    """Seq2seq LSTM autoencoder baseline."""

    def __init__(self, n_features: int = 10, seq_len: int = 100,
                 hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.encoder     = nn.LSTM(n_features, hidden_size, n_layers,
                                   batch_first=True, bidirectional=True)
        self.latent_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.decoder     = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.out_proj    = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, nf = x.shape
        _, (h, _) = self.encoder(x)
        z = F.gelu(self.latent_proj(torch.cat([h[-2], h[-1]], dim=-1)))
        dec, _ = self.decoder(z.unsqueeze(1).expand(-1, N, -1))
        x_hat  = self.out_proj(dec)
        scores = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return scores.mean(), scores.detach()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, sc = self.forward(x)
        return sc


class GRUAE(nn.Module):
    """Seq2seq GRU autoencoder baseline."""

    def __init__(self, n_features: int = 10, seq_len: int = 100,
                 hidden_size: int = 64, n_layers: int = 2):
        super().__init__()
        self.encoder     = nn.GRU(n_features, hidden_size, n_layers,
                                  batch_first=True, bidirectional=True)
        self.latent_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.decoder     = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.out_proj    = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, nf = x.shape
        _, h = self.encoder(x)
        z = F.gelu(self.latent_proj(torch.cat([h[-2], h[-1]], dim=-1)))
        dec, _ = self.decoder(z.unsqueeze(1).expand(-1, N, -1))
        x_hat  = self.out_proj(dec)
        scores = ((x - x_hat) ** 2).mean(dim=(1, 2))
        return scores.mean(), scores.detach()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, sc = self.forward(x)
        return sc


class ConvAEStat(nn.Module):
    """
    Conv1D AE with optional stat + FFT auxiliary losses.
    Same interface as CANTEMPOStat — drop-in for the base/stat/stat_fft variants.
    """

    def __init__(
        self,
        n_features:  int   = 10,
        seq_len:     int   = 100,
        d_model:     int   = 256,
        lambda_stat: float = 0.0,
        lambda_fft:  float = 0.0,
        pooling:     str   = "mean",
        **kwargs,                               # absorb unused CANTEMPOStat args
    ):
        super().__init__()
        self.lambda_stat = lambda_stat
        self.lambda_fft  = lambda_fft
        self.encoder = Conv1DEncoder(n_features, d_model, pooling)
        self.decoder = AEDecoder(d_model, seq_len, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        x_hat = self.decoder(z)

        mse_per = ((x - x_hat) ** 2).mean(dim=(1, 2))
        scores  = mse_per.clone()
        loss    = mse_per.mean()

        if self.lambda_stat > 0.0:
            stat_per = ((stat_vector(x) - stat_vector(x_hat)) ** 2).mean(dim=1)
            loss     = loss   + self.lambda_stat * stat_per.mean()
            scores   = scores + self.lambda_stat * stat_per

        if self.lambda_fft > 0.0:
            fft_x    = torch.fft.rfft(x,     dim=1).abs()
            fft_xhat = torch.fft.rfft(x_hat, dim=1).abs()
            fft_per  = ((fft_x - fft_xhat) ** 2).mean(dim=(1, 2))
            loss     = loss   + self.lambda_fft * fft_per.mean()
            scores   = scores + self.lambda_fft * fft_per

        return loss, scores.detach()
