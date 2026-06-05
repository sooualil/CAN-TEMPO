"""
rl_ids.py
---------
Baseline 2: RL-IDS — Robust and Lightweight Intrusion Detection System.
(Yuan et al., JISA 2026)

Pipeline:
  Input: (B, 3, 9, 9) CAN images  (27-msg window → 9×9×3 RGB)

  Stage 1 — WGAN-GP discriminator pre-training on normal CAN images.
  Stage 2 — GAA model:
      Encoder1 → z  (latent)
      Decoder  → x̂  (reconstruction)
      Encoder2 → ẑ  (re-encode)
      Discriminator (frozen) evaluates D(x) and D(x̂)
      Loss = ||x - x̂||² + ||D(x) - D(x̂)||² + ||z - ẑ||²

  Stage 3 — Teacher-Student distillation:
      Teacher (GAA-T): complex multi-branch encoder with SAM + SEM
      Student (GAA-S): single-branch lightweight encoder
      Distillation loss = L_S + ||L_S - L_T||²

  Anomaly score = same formula as training loss (Eq. 3 in paper)

  Threshold η: maximise F1 on validation set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Discriminator (WGAN-GP style)
# ────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """Simple DCGAN-style discriminator for 9×9 images."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, padding=1),          nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),               nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 1)

    def gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """WGAN-GP gradient penalty."""
        alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.forward(interp)
        grad = torch.autograd.grad(
            d_interp, interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        gp = ((grad.norm(2, dim=(1, 2, 3)) - 1) ** 2).mean()
        return gp


# ────────────────────────────────────────────────────────────────────────────
# Spatial Attention Module (SAM)
# ────────────────────────────────────────────────────────────────────────────

class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        m   = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * m


# ────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation Module (SEM)
# ────────────────────────────────────────────────────────────────────────────

class SEM(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ────────────────────────────────────────────────────────────────────────────
# DSC: Depthwise Separable Convolution block
# ────────────────────────────────────────────────────────────────────────────

def dsc_block(in_ch: int, out_ch: int, kernel: int, stride: int = 1) -> nn.Sequential:
    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel, stride=stride, padding=pad, groups=in_ch),
        nn.Conv2d(in_ch, out_ch, 1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    )


# ────────────────────────────────────────────────────────────────────────────
# GAA-T: Teacher Encoder (complex multi-branch)
# ────────────────────────────────────────────────────────────────────────────

class EncoderGAAT(nn.Module):
    """
    3-parallel-branch encoder (3×3, 5×5, 7×7 DSC) + SAM after each DSC
    → concatenate → SEM → pointwise conv → FC → 100-dim latent.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 100):
        super().__init__()
        mid = 64

        self.branch3 = nn.Sequential(
            dsc_block(in_channels, mid, 3), SAM(),
            dsc_block(mid, mid, 3), SAM(),
        )
        self.branch5 = nn.Sequential(
            dsc_block(in_channels, mid, 5), SAM(),
            dsc_block(mid, mid, 5), SAM(),
        )
        self.branch7 = nn.Sequential(
            dsc_block(in_channels, mid, 7), SAM(),
            dsc_block(mid, mid, 7), SAM(),
        )

        concat_ch = mid * 3  # 192
        self.sem   = SEM(concat_ch)
        self.pw    = nn.Conv2d(concat_ch, 256, 1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        b7 = self.branch7(x)
        h  = torch.cat([b3, b5, b7], dim=1)    # (B, 192, 9, 9)
        h  = self.sem(h)
        h  = self.pw(h)                         # (B, 256, 9, 9)
        h  = self.pool(h).flatten(1)            # (B, 256)
        return self.fc(h)                        # (B, latent_dim)


# ────────────────────────────────────────────────────────────────────────────
# GAA-S: Student Encoder (lightweight single-branch)
# ────────────────────────────────────────────────────────────────────────────

class EncoderGAAS(nn.Module):
    """Single 3×3 branch encoder — ~65% fewer parameters than GAA-T."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 100):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),         nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)   # (B, 128)
        return self.fc(h)              # (B, latent_dim)


# ────────────────────────────────────────────────────────────────────────────
# Shared Decoder (teacher and student use the same decoder design)
# ────────────────────────────────────────────────────────────────────────────

class DecoderGAAT(nn.Module):
    """3× transposed conv layers (128→64→32→16) → pointwise → Tanh → (B,3,9,9)."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.fc  = nn.Linear(latent_dim, 16 * 2 * 2)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 3, 1),
            nn.AdaptiveAvgPool2d((9, 9)),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 16, 2, 2)
        return self.net(h)   # (B, 3, 9, 9)


class DecoderGAAS(nn.Module):
    """Simpler single transposed-conv decoder."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.fc  = nn.Linear(latent_dim, 16 * 3 * 3)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(16, 3, 3, stride=3, padding=0),
            nn.AdaptiveAvgPool2d((9, 9)),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 16, 3, 3)
        return self.net(h)   # (B, 3, 9, 9)


# ────────────────────────────────────────────────────────────────────────────
# GAA model (teacher or student)
# ────────────────────────────────────────────────────────────────────────────

class GAA(nn.Module):
    """
    Generative Adversarial Autoencoder.

    mode="teacher" → uses GAA-T encoder + decoder
    mode="student" → uses GAA-S encoder + decoder
    """

    def __init__(
        self,
        mode: str = "teacher",
        latent_dim: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert mode in ("teacher", "student")
        self.mode = mode

        if mode == "teacher":
            self.encoder1 = EncoderGAAT(latent_dim=latent_dim)
            self.encoder2 = EncoderGAAT(latent_dim=latent_dim)
            self.decoder  = DecoderGAAT(latent_dim=latent_dim)
        else:
            self.encoder1 = EncoderGAAS(latent_dim=latent_dim)
            self.encoder2 = EncoderGAAS(latent_dim=latent_dim)
            self.decoder  = DecoderGAAS(latent_dim=latent_dim)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple:
        z    = self.drop(self.encoder1(x))    # (B, latent)
        x_hat = self.decoder(z)               # (B, 3, 9, 9)
        z_hat = self.drop(self.encoder2(x_hat))
        return z, x_hat, z_hat

    def gaa_loss(
        self,
        x: torch.Tensor,
        discriminator: Discriminator,
    ) -> torch.Tensor:
        """
        L = ||x - x̂||² + ||D(x) - D(x̂)||² + ||z - ẑ||²
        Discriminator weights must be frozen before calling.
        """
        z, x_hat, z_hat = self.forward(x)
        with torch.no_grad():
            D_x    = discriminator(x)
        D_xhat     = discriminator(x_hat)

        L_x = F.mse_loss(x_hat, x)
        L_d = F.mse_loss(D_xhat, D_x)
        L_z = F.mse_loss(z_hat, z)
        return L_x + L_d + L_z

    @torch.no_grad()
    def anomaly_score(
        self,
        x: torch.Tensor,
        discriminator: Discriminator,
    ) -> torch.Tensor:
        """Per-image anomaly score (same formula as training loss, per sample)."""
        self.eval()
        discriminator.eval()
        z, x_hat, z_hat = self.forward(x)
        D_x    = discriminator(x)
        D_xhat = discriminator(x_hat)

        L_x = ((x_hat - x) ** 2).mean(dim=(1, 2, 3))
        L_d = ((D_xhat - D_x) ** 2).squeeze(1)
        L_z = ((z_hat - z) ** 2).mean(dim=1)
        return L_x + L_d + L_z   # (B,)


# ────────────────────────────────────────────────────────────────────────────
# RL-IDS wrapper (combines discriminator + teacher + student)
# ────────────────────────────────────────────────────────────────────────────

class RLIDS(nn.Module):
    """
    Full RL-IDS: discriminator + GAA-T teacher + GAA-S student.

    Training stages are separated into helper functions in train.py.
    This class holds all components and provides a unified anomaly_score().
    """

    def __init__(self, latent_dim: int = 100, dropout: float = 0.2):
        super().__init__()
        self.discriminator = Discriminator()
        self.teacher       = GAA("teacher", latent_dim, dropout)
        self.student       = GAA("student", latent_dim, dropout)

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Use the student model for inference."""
        return self.student.anomaly_score(x, self.discriminator)

    def student_distillation_loss(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        L = L_S + ||L_S - L_T||²
        where L_S and L_T are per-batch scalar losses.
        """
        L_T = self.teacher.gaa_loss(x, self.discriminator)
        L_S = self.student.gaa_loss(x, self.discriminator)
        return L_S + F.mse_loss(L_S.unsqueeze(0), L_T.detach().unsqueeze(0))
