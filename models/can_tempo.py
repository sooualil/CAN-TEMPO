import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── Inception 2D Block ────────────────────────────────────────────────────────

class InceptionBlock2D(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        mid = d_model // 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(d_model, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid), nn.GELU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(d_model, mid, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid), nn.GELU(),
        )
        self.proj = nn.Conv2d(mid * 2, d_model, kernel_size=1)
        self.norm = nn.BatchNorm2d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([self.conv3(x), self.conv5(x)], dim=1)))


# ── TMPO Block ────────────────────────────────────────────────────────────────

class TMPOBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int, top_k: int = 2,
                 per_feature_freqs: bool = False, n_fft_windows: int = 1):
        super().__init__()
        self.seq_len           = seq_len
        self.top_k             = top_k
        self.per_feature_freqs = per_feature_freqs
        self.n_fft_windows     = n_fft_windows
        self.conv2d            = InceptionBlock2D(d_model)
        self.norm              = nn.LayerNorm(d_model)

    def _stft_amp(self, signal: torch.Tensor) -> torch.Tensor:
        B, N, C = signal.shape
        n = self.n_fft_windows
        if n <= 1:
            return torch.fft.rfft(signal, dim=1).abs().mean(dim=0)
        win_len = N // n
        hop     = max(1, (N - win_len) // max(1, n - 1))
        starts  = [i * hop for i in range(n) if i * hop + win_len <= N]
        n_bins  = N // 2 + 1
        out     = torch.zeros(n_bins, C, device=signal.device, dtype=signal.dtype)
        for s in starts:
            a   = torch.fft.rfft(signal[:, s:s + win_len, :], dim=1).abs().mean(dim=0)
            out += F.interpolate(a.T.unsqueeze(0), size=n_bins,
                                 mode='linear', align_corners=False).squeeze(0).T
        return out / len(starts)

    def _get_periods_and_weights(self, x, x_orig):
        B, N, D = x.shape
        if self.per_feature_freqs and x_orig is not None:
            amp_orig      = self._stft_amp(x_orig)
            amp_orig[0,:] = 0
            k_per         = min(self.top_k, amp_orig.shape[0] - 1)
            top_freqs     = torch.unique(torch.cat([
                torch.topk(amp_orig[1:, f], k_per).indices + 1
                for f in range(x_orig.shape[-1])
            ]))
            mean_amp      = self._stft_amp(x).mean(dim=-1)
            mean_amp[0]   = 0
            weights       = torch.softmax(mean_amp[top_freqs], dim=0)
        else:
            mean_amp      = self._stft_amp(x).mean(dim=-1)
            mean_amp[0]   = 0
            k             = min(self.top_k, mean_amp.shape[0] - 1)
            top_freqs     = torch.topk(mean_amp[1:], k).indices + 1
            weights       = torch.softmax(mean_amp[top_freqs], dim=0)
        periods = (N / top_freqs.float()).round().long().clamp(min=2)
        return periods, weights, len(top_freqs)

    def forward(self, x: torch.Tensor,
                x_orig: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        periods, weights, k = self._get_periods_and_weights(x, x_orig)
        out = torch.zeros_like(x)
        for i in range(k):
            p      = periods[i].item()
            T      = math.ceil(N / p)
            x_pad  = F.pad(x, (0, 0, 0, T * p - N))
            x_2d   = x_pad.reshape(B, T, p, D).permute(0, 3, 1, 2)
            x_flat = self.conv2d(x_2d).permute(0, 2, 3, 1).reshape(B, T * p, D)[:, :N, :]
            out    = out + weights[i] * x_flat
        return self.norm(x + out)


# ── Encoder ───────────────────────────────────────────────────────────────────

class CANTEMPOEncoder(nn.Module):
    def __init__(self, n_features: int = 10, seq_len: int = 100,
                 d_model: int = 256, n_layers: int = 4, top_k: int = 2,
                 per_feature_freqs: bool = False, pooling: str = "max",
                 n_fft_windows: int = 1):
        super().__init__()
        self.per_feature_freqs = per_feature_freqs
        self.pooling           = pooling
        self.input_proj        = nn.Linear(n_features, d_model)
        self.blocks            = nn.ModuleList([
            TMPOBlock(seq_len, d_model, top_k, per_feature_freqs, n_fft_windows)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)
        if pooling == "attention":
            self.attn_q = nn.Linear(d_model, d_model)
            self.attn_k = nn.Linear(d_model, d_model)
            self.scale  = d_model ** -0.5

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        if self.pooling == "max":
            return h.max(dim=1).values
        if self.pooling == "attention":
            q = self.attn_q(h.mean(dim=1, keepdim=True))
            k = self.attn_k(h)
            w = torch.softmax((q * k).sum(dim=-1) * self.scale, dim=-1)
            return (w.unsqueeze(-1) * h).sum(dim=1)
        return h.mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_orig = x if self.per_feature_freqs else None
        h = self.input_proj(x)
        for blk in self.blocks:
            h = blk(h, x_orig)
        return self._pool(self.out_norm(h))


# ── Decoder ───────────────────────────────────────────────────────────────────

class AEDecoder(nn.Module):
    def __init__(self, d_model: int, seq_len: int, n_features: int):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.fc1        = nn.Linear(d_model, d_model * 2)
        self.fc2        = nn.Linear(d_model * 2, seq_len * n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(z))).reshape(z.size(0), self.seq_len, self.n_features)


# ── Statistical feature vector ────────────────────────────────────────────────

def stat_vector(x: torch.Tensor) -> torch.Tensor:
    """Returns (B, 3*F): variance, lag-1 autocorr, mean-abs-diff per feature."""
    var      = x.var(dim=1)
    mu       = x.mean(dim=1, keepdim=True)
    xc       = x - mu
    autocorr = (xc[:, :-1, :] * xc[:, 1:, :]).mean(dim=1) / (xc.var(dim=1) + 1e-8)
    mad      = x.diff(dim=1).abs().mean(dim=1)
    return torch.cat([var, autocorr, mad], dim=-1)


# ── CAN-TEMPO ─────────────────────────────────────────────────────────────────

class CANTEMPO(nn.Module):
    """
    CAN-TEMPO autoencoder with optional statistical and spectral auxiliary losses.

    Args:
        n_features   : input features per CAN message (default 10)
        seq_len      : window size in messages (default 100)
        d_model      : embedding dimension (default 256)
        n_layers     : number of TMPO blocks (default 4)
        top_k        : dominant periods per block (default 2)
        lambda_stat  : statistical consistency loss weight (default 4.0; set 0 to disable)
        lambda_fft   : spectral reconstruction loss weight (default 0.1; set 0 to disable)
        pooling      : latent pooling — 'max', 'mean', or 'attention' (default 'max')
    """

    def __init__(
        self,
        n_features:        int   = 10,
        seq_len:           int   = 100,
        d_model:           int   = 256,
        n_layers:          int   = 4,
        top_k:             int   = 2,
        lambda_stat:       float = 4.0,
        lambda_fft:        float = 0.1,
        per_feature_freqs: bool  = False,
        pooling:           str   = "max",
        n_fft_windows:     int   = 1,
    ):
        super().__init__()
        self.lambda_stat = lambda_stat
        self.lambda_fft  = lambda_fft
        self.encoder     = CANTEMPOEncoder(n_features, seq_len, d_model, n_layers,
                                           top_k, per_feature_freqs, pooling, n_fft_windows)
        self.decoder     = AEDecoder(d_model, seq_len, n_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        x_hat = self.decoder(z)

        mse_per = ((x - x_hat) ** 2).mean(dim=(1, 2))
        loss    = mse_per.mean()
        scores  = mse_per.clone()

        if self.lambda_stat > 0.0:
            stat_per = ((stat_vector(x) - stat_vector(x_hat)) ** 2).mean(dim=1)
            loss     = loss   + self.lambda_stat * stat_per.mean()
            scores   = scores + self.lambda_stat * stat_per

        if self.lambda_fft > 0.0:
            fft_per = ((torch.fft.rfft(x, dim=1).abs() -
                        torch.fft.rfft(x_hat, dim=1).abs()) ** 2).mean(dim=(1, 2))
            loss    = loss   + self.lambda_fft * fft_per.mean()
            scores  = scores + self.lambda_fft * fft_per

        return loss, scores.detach()

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, sc = self.forward(x)
        return sc
