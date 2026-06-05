"""
dataset_car_hacking.py
----------------------
Dataset loader for the OTIDS / Car-Hacking Dataset
(Lee et al., 2017 — used as primary benchmark in CGTS and RL-IDS papers).

Directory layout expected:
  <root>/
    benign_dataset.csv      ← normal-only
    DoS_dataset.csv         ← contains both R (normal) and T (attack) rows
    Fuzzy_dataset.csv
    gear_dataset.csv
    RPM_dataset.csv

CSV columns (no header):
  timestamp, arbitration_id (hex), dlc, byte_0..byte_7, label (R=normal / T=attack)

All attack files include interleaved normal traffic (label R) — we use only
label-R rows for training and evaluate on the full file.
"""

import os
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple

from .preprocess import make_windows, make_windows_forecasting, make_can_images

ATTACK_FILES = ["DoS_dataset.csv", "Fuzzy_dataset.csv", "gear_dataset.csv", "RPM_dataset.csv"]
NORMAL_FILE  = "benign_dataset.csv"

# Map attack file stem → short attack name
ATTACK_NAMES = {
    "DoS_dataset":   "DoS",
    "Fuzzy_dataset": "Fuzzy",
    "gear_dataset":  "Gear",
    "RPM_dataset":   "RPM",
}


# ── Low-level file loading ────────────────────────────────────────────────────

def load_car_hacking_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a Car-Hacking CSV file.

    Returns:
        features : (N, 9) float32  — [arb_id_norm, byte_0..byte_7_norm]
        labels   : (N,)   int8     — 0=normal, 1=attack
    """
    timestamps, arb_ids, payloads, labels = [], [], [], []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                ts    = float(parts[0])
                arb   = int(parts[1], 16)
                # dlc = parts[2]  (ignored — we always take 8 bytes)
                # bytes: parts[3..10], label: parts[-1]
                label_char = parts[-1].strip().upper()
                if label_char not in ("R", "T"):
                    continue
                lbl = 0 if label_char == "R" else 1
                raw_bytes = parts[3:-1]  # everything between dlc and label
                byte_vals = [int(b, 16) for b in raw_bytes[:8]]
                # zero-pad to 8 bytes
                byte_vals += [0] * (8 - len(byte_vals))
            except (ValueError, IndexError):
                continue

            timestamps.append(ts)
            arb_ids.append(arb)
            payloads.append(byte_vals)
            labels.append(lbl)

    if not timestamps:
        return np.zeros((0, 10), dtype=np.float32), np.zeros(0, dtype=np.int8)

    timestamps = np.array(timestamps, dtype=np.float64)
    arb_ids    = np.array(arb_ids,    dtype=np.int32)
    payloads   = np.array(payloads,   dtype=np.uint8)
    labels     = np.array(labels,     dtype=np.int8)

    # Compute inter-arrival times (delta_t)
    dt = np.diff(timestamps, prepend=timestamps[0])
    dt = np.clip(dt, 0, None)

    # Build feature matrix: arb_id_norm, byte_0..byte_7_norm, dt_norm
    arb_norm  = arb_ids / 0x7FF
    pay_norm  = payloads / 255.0
    dt_norm   = np.clip(dt, 0, 0.01) / 0.01

    features = np.column_stack([arb_norm, pay_norm, dt_norm]).astype(np.float32)
    return features, labels


def load_normal_only(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a file and return only normal (R) rows."""
    features, labels = load_car_hacking_csv(path)
    mask = labels == 0
    return features[mask], labels[mask]


# ── Dataset classes ───────────────────────────────────────────────────────────

class CarHackingWindowDataset(Dataset):
    """Windowed dataset for Car-Hacking."""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 window_size: int = 100, stride: int = 50):
        self.X, self.y = make_windows(features, labels, window_size, stride)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class CarHackingForecastDataset(Dataset):
    """Forecasting dataset for Car-Hacking."""

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 window_size: int = 100, stride: int = 50, horizon: int = 10):
        self.X, self.tgt, self.y = make_windows_forecasting(
            features, labels, window_size, stride, horizon)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import torch
        return (torch.from_numpy(self.X[idx]),
                torch.from_numpy(self.tgt[idx]),
                torch.tensor(self.y[idx], dtype=torch.long))


class CarHackingImageDataset(Dataset):
    """27-msg CAN image dataset for RL-IDS."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.imgs, self.y = make_can_images(features, labels)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        import torch
        return torch.from_numpy(self.imgs[idx]), torch.tensor(self.y[idx], dtype=torch.long)


# ── Build helpers ─────────────────────────────────────────────────────────────

def build_car_hacking_train(root: str,
                             mode: str = "window",
                             window_size: int = 100,
                             stride: int = 50,
                             horizon: int = 10,
                             augment_from_attacks: bool = False):
    """
    Training set: normal-only rows.

    augment_from_attacks=True (default): pool normal rows from benign_dataset.csv
    AND the R-labelled rows from all 4 attack files. This ensures the model sees
    the same inter-arrival timing distribution as the test files, preventing
    session-level covariate shift in timing-based features (dom_ratio, dt_*).
    """
    feat_list, lbl_list = [], []
    # Primary: benign file
    f, l = load_normal_only(os.path.join(root, NORMAL_FILE))
    feat_list.append(f); lbl_list.append(l)

    if augment_from_attacks:
        # Add normal rows from each attack file (no attack labels — clean training)
        for fname in ATTACK_FILES:
            f, l = load_normal_only(os.path.join(root, fname))
            feat_list.append(f); lbl_list.append(l)

    features = np.concatenate(feat_list, axis=0)
    labels   = np.concatenate(lbl_list,  axis=0)
    if mode == "window":
        return CarHackingWindowDataset(features, labels, window_size, stride)
    elif mode == "forecast":
        return CarHackingForecastDataset(features, labels, window_size, stride, horizon)
    elif mode == "image":
        return CarHackingImageDataset(features, labels)
    raise ValueError(f"Unknown mode: {mode}")


def build_car_hacking_test(root: str,
                            attack: str = "DoS",
                            mode: str = "window",
                            window_size: int = 100,
                            stride: int = 50,
                            horizon: int = 10):
    """
    Test set: full attack file (contains interleaved R + T rows).
    attack: one of 'DoS', 'Fuzzy', 'Gear', 'RPM'
    """
    # Find the matching file
    file_map = {v: k + ".csv" for k, v in ATTACK_NAMES.items()}
    fname = file_map.get(attack)
    if fname is None:
        raise ValueError(f"Unknown attack: {attack}. Choose from {list(file_map)}")
    features, labels = load_car_hacking_csv(os.path.join(root, fname))

    if mode == "window":
        return CarHackingWindowDataset(features, labels, window_size, stride)
    elif mode == "forecast":
        return CarHackingForecastDataset(features, labels, window_size, stride, horizon)
    elif mode == "image":
        return CarHackingImageDataset(features, labels)
    raise ValueError(f"Unknown mode: {mode}")


def build_all_car_hacking(root: str,
                           mode: str = "window",
                           window_size: int = 100,
                           stride: int = 50):
    """Return train ds + dict of test datasets keyed by attack name."""
    train_ds = build_car_hacking_train(root, mode, window_size, stride)
    test_ds  = {atk: build_car_hacking_test(root, atk, mode, window_size, stride)
                for atk in ATTACK_NAMES.values()}
    return train_ds, test_ds
