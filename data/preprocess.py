"""
preprocess.py
-------------
Shared feature extraction, windowing, and normalization utilities
for both CAN-Train-and-Test and Survival datasets.

Feature vector per CAN message (9 features, matching RL-IDS):
  [arb_id_int, byte_0, byte_1, ..., byte_7]
All values normalized to [0, 1].
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


# ── Constants ────────────────────────────────────────────────────────────────

N_FEATURES    = 10         # 1 (ID) + 8 (payload bytes) + 1 (inter-arrival time)
MAX_ARB_ID    = 0x7FF     # Standard 11-bit CAN ID
MAX_BYTE      = 255.0
MAX_DELTA_T   = 0.01      # 10ms — clips most normal inter-arrival times to [0,1]


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features_can_train_test(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from CAN-Train-and-Test CSV rows.

    CSV columns: timestamp, arbitration_id (hex str), data_field (hex str), attack (int)

    Returns:
        features : np.ndarray, shape (N, 9), float32, normalized [0, 1]
        labels   : np.ndarray, shape (N,),   int8
    """
    # Arbitration ID: hex string → int → normalize
    arb_ids = df["arbitration_id"].apply(lambda x: int(str(x).strip(), 16)).values.astype(np.float32)
    arb_ids = arb_ids / MAX_ARB_ID

    # Data field: hex string → 8 bytes (zero-padded)
    payload = _hex_field_to_bytes(df["data_field"].values)  # (N, 8)

    # Inter-arrival time: diff of timestamps, clipped and normalized
    ts = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).values.astype(np.float64)
    delta_t = np.diff(ts, prepend=ts[0])                    # (N,) first row = 0
    delta_t = np.clip(delta_t, 0, MAX_DELTA_T) / MAX_DELTA_T  # normalize to [0, 1]
    delta_t = delta_t.astype(np.float32)

    features = np.concatenate([arb_ids[:, None], payload, delta_t[:, None]], axis=1).astype(np.float32)
    labels   = df["attack"].values.astype(np.int8)
    return features, labels


def extract_features_survival(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from Survival TXT rows.

    TXT columns: timestamp, arbitration_id (hex), data_length, byte_0 .. byte_7
    Byte columns may be absent (variable-length payload → zero-pad to 8).

    Returns:
        features : np.ndarray, shape (N, 9), float32, normalized [0, 1]
        labels   : np.ndarray, shape (N,),   int8  (0=normal, 1=attack inferred from file)
    """
    def _safe_arb(x):
        try:
            return int(str(x).strip(), 16)
        except ValueError:
            return 0

    arb_ids = df["arbitration_id"].apply(_safe_arb).values.astype(np.float32)
    arb_ids = arb_ids / MAX_ARB_ID

    byte_cols = [c for c in df.columns if c.startswith("byte_")]
    payload = np.zeros((len(df), 8), dtype=np.float32)
    for i, col in enumerate(byte_cols[:8]):
        payload[:, i] = pd.to_numeric(df[col], errors="coerce").fillna(0).values / MAX_BYTE

    # Inter-arrival time (first column is timestamp)
    ts_col = df.columns[0]
    ts = pd.to_numeric(df[ts_col], errors="coerce").fillna(0).values.astype(np.float64)
    delta_t = np.diff(ts, prepend=ts[0])
    delta_t = np.clip(delta_t, 0, MAX_DELTA_T) / MAX_DELTA_T
    delta_t = delta_t.astype(np.float32)

    features = np.concatenate([arb_ids[:, None], payload, delta_t[:, None]], axis=1).astype(np.float32)
    labels   = df["attack"].values.astype(np.int8) if "attack" in df.columns else np.zeros(len(df), dtype=np.int8)
    return features, labels


# ── Windowing ────────────────────────────────────────────────────────────────

def make_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 100,
    stride: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over a message sequence to produce fixed-size samples.

    Args:
        features    : (N, 9) array of per-message features
        labels      : (N,)   array of per-message labels
        window_size : number of messages per window
        stride      : step size between consecutive windows

    Returns:
        windows      : (M, window_size, 9)  float32
        win_labels   : (M,)                 int8  — 1 if any message in window is attack
    """
    N = len(features)
    starts = range(0, N - window_size + 1, stride)
    windows    = np.stack([features[s:s + window_size] for s in starts])
    win_labels = np.array([labels[s:s + window_size].max() for s in starts], dtype=np.int8)
    return windows, win_labels


def make_windows_forecasting(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 100,
    horizon: int = 10,
    stride: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Like make_windows but also returns the next `horizon` messages as targets.

    Returns:
        windows      : (M, window_size, 9)
        targets      : (M, horizon, 9)
        win_labels   : (M,)
    """
    N = len(features)
    starts = range(0, N - window_size - horizon + 1, stride)
    windows    = np.stack([features[s:s + window_size]                     for s in starts])
    targets    = np.stack([features[s + window_size:s + window_size + horizon] for s in starts])
    win_labels = np.array([labels[s:s + window_size + horizon].max()       for s in starts], dtype=np.int8)
    return windows, targets, win_labels


# ── RL-IDS CAN Image encoding ────────────────────────────────────────────────

def make_can_images(
    features: np.ndarray,
    labels: np.ndarray,
    window_size: int = 27,
    stride: int = 27,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert sequences of 27 messages to 9×9×3 RGB CAN images (RL-IDS style).

    Each message → 9 features in [0, 255].
    27 messages → 27×9 matrix → reshape to 9×9×3 (height=9, width=9, RGB).
    RGB encoding splits the 27-row dimension into 3 channels of 9 rows each.

    Returns:
        images     : (M, 3, 9, 9) float32  in [0, 1]
        win_labels : (M,)         int8
    """
    # RL-IDS CAN image uses only the original 9 features (arb_id + 8 bytes), not delta_t
    feat_9 = features[:, :9]
    feat_255 = (feat_9 * np.array([MAX_ARB_ID] + [MAX_BYTE] * 8, dtype=np.float32)).astype(np.float32)

    N = len(feat_255)
    starts = range(0, N - window_size + 1, stride)
    images     = []
    win_labels = []
    for s in starts:
        block = feat_255[s:s + window_size]   # (27, 9)
        img   = block.reshape(3, 9, 9) / 255.0  # (3, 9, 9) in [0,1]
        images.append(img)
        win_labels.append(labels[s:s + window_size].max())

    images     = np.stack(images).astype(np.float32)      # (M, 3, 9, 9)
    win_labels = np.array(win_labels, dtype=np.int8)
    return images, win_labels


# ── Internal helpers ──────────────────────────────────────────────────────────

def _hex_field_to_bytes(hex_fields: np.ndarray) -> np.ndarray:
    """
    Convert an array of hex strings (e.g. '3000000430000004') to
    an (N, 8) float32 array of byte values in [0, 1].
    Strings shorter than 16 hex chars are zero-padded on the right.
    """
    N = len(hex_fields)
    out = np.zeros((N, 8), dtype=np.float32)
    for i, hx in enumerate(hex_fields):
        hx = str(hx).strip().replace(" ", "")
        # Pad/truncate to exactly 16 hex chars (8 bytes)
        hx = hx.ljust(16, "0")[:16]
        for j in range(8):
            try:
                out[i, j] = int(hx[j * 2: j * 2 + 2], 16) / MAX_BYTE
            except ValueError:
                out[i, j] = 0.0
    return out
