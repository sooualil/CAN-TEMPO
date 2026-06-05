"""
dataset_survival.py
-------------------
Dataset loader for the Survival Analysis Dataset (SAD).

Directory layout expected:
  <root>/
    Sonata/
      FreeDrivingData_20180323_SONATA.txt   ← normal
      Flooding_dataset_SONATA.txt
      Fuzzy_dataset_SONATA.txt
      Malfunction_dataset_SONATA.txt
    Soul/
      FreeDrivingData_20180112_KIA.txt
      Flooding_dataset_KIA.txt
      Fuzzy_dataset_KIA.txt
      Malfunction153_dataset_KIA.txt
    Spark/
      FreeDrivingData_20171231_Spark.txt
      Flooding_dataset_Spark.txt
      Fuzzy_dataset_Spark.txt
      Malfunction18E_dataset_Spark.txt

TXT columns: timestamp, arbitration_id, data_length, byte_0 .. byte_N
(no header — we infer it)
"""

import os
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List, Tuple

from .preprocess import extract_features_survival, make_windows, make_windows_forecasting, make_can_images

VEHICLES = ["Sonata", "Soul", "Spark"]

ATTACK_KEYWORDS = ["Flooding", "Fuzzy", "Malfunction"]
NORMAL_KEYWORDS = ["FreeDriving", "freedriving", "free_driving"]


# ── Low-level file loading ───────────────────────────────────────────────────

def load_survival_txt(path: str, is_attack: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a Survival TXT file.

    If the file contains a label column (last field R/T), labels are read from it.
    Otherwise (FreeDriving files have no label column), is_attack is used for all rows.
    """
    rows = []
    file_has_labels = False
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            if len(parts) == 4 and " " in parts[3]:
                byte_tokens = parts[3].strip().split()
                parts = parts[:3] + byte_tokens
            if parts[-1].strip().upper() in ("R", "T"):
                file_has_labels = True
            rows.append(parts)

    if not rows:
        return np.zeros((0, 9), dtype=np.float32), np.zeros(0, dtype=np.int8)

    max_bytes = 8
    col_names = ["timestamp", "arbitration_id", "data_length"] + [f"byte_{i}" for i in range(max_bytes)]
    records = []
    row_labels = []
    for r in rows:
        if file_has_labels:
            lbl = 0 if r[-1].strip().upper() == "R" else 1
            r = r[:-1]
        else:
            lbl = int(is_attack)
        padded = r + ["0"] * (len(col_names) - len(r))
        records.append(padded[:len(col_names)])
        row_labels.append(lbl)

    df = pd.DataFrame(records, columns=col_names)
    df["attack"] = row_labels

    # arbitration_id is a hex string — keep as-is for extract_features_survival
    # data_length is decimal
    df["data_length"] = pd.to_numeric(df["data_length"].str.strip(), errors="coerce").fillna(0)
    # byte columns are hex strings — parse as hex integers; tolerate non-hex tokens (e.g. 'R' label)
    def _safe_hex(x):
        s = str(x).strip()
        if not s or s in ("0", "nan"):
            return 0
        try:
            return int(s, 16)
        except ValueError:
            return 0

    for col in [c for c in col_names if c.startswith("byte_")]:
        df[col] = df[col].apply(_safe_hex).astype(float)

    features, labels = extract_features_survival(df)
    return features, labels


def _is_normal(filename: str) -> bool:
    name = os.path.basename(filename).lower()
    return any(kw.lower() in name for kw in NORMAL_KEYWORDS)


def _is_attack(filename: str) -> bool:
    name = os.path.basename(filename).lower()
    return any(kw.lower() in name for kw in ATTACK_KEYWORDS)


def _attack_type(filename: str) -> str:
    name = os.path.basename(filename).lower()
    for kw in ATTACK_KEYWORDS:
        if kw.lower() in name:
            return kw.lower()
    return "unknown"


# ── Dataset class ─────────────────────────────────────────────────────────────

class SurvivalWindowDataset(Dataset):
    """Sliding-window dataset over Survival message sequences."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int = 100,
        stride: int = 50,
    ):
        self.windows, self.labels = make_windows(features, labels, window_size, stride)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        import torch
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(int(self.labels[idx])),
        )


class SurvivalImageDataset(Dataset):
    """CAN image dataset (RL-IDS style) over Survival data."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        window_size: int = 27,
        stride: int = 27,
    ):
        self.images, self.labels = make_can_images(features, labels, window_size, stride)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import torch
        return (
            torch.from_numpy(self.images[idx]),
            torch.tensor(int(self.labels[idx])),
        )


# ── High-level builder functions ─────────────────────────────────────────────

def build_survival_train(
    root: str,
    vehicle: str = "Sonata",
    mode: str = "window",
    window_size: int = 100,
    stride: int = 50,
) -> Dataset:
    """
    Build a training dataset from normal (FreeDriving) traffic of one vehicle.
    """
    vehicle_dir = os.path.join(root, vehicle)
    txt_files = glob.glob(os.path.join(vehicle_dir, "*.txt"))
    normal_files = [f for f in txt_files if _is_normal(f)]

    all_features, all_labels = [], []
    for f in normal_files:
        feat, lab = load_survival_txt(f, is_attack=False)
        all_features.append(feat)
        all_labels.append(lab)

    features = np.concatenate(all_features) if all_features else np.zeros((0, 9), dtype=np.float32)
    labels   = np.concatenate(all_labels)   if all_labels   else np.zeros(0, dtype=np.int8)

    return _make_survival_dataset(features, labels, mode, window_size, stride)


def build_survival_test(
    root: str,
    vehicle: str = "Sonata",
    attack_type: str = "flooding",
    mode: str = "window",
    window_size: int = 100,
    stride: int = 50,
) -> Dataset:
    """
    Build a test dataset mixing normal + one attack type for one vehicle.
    """
    vehicle_dir = os.path.join(root, vehicle)
    txt_files   = glob.glob(os.path.join(vehicle_dir, "*.txt"))

    all_features, all_labels = [], []

    for f in txt_files:
        if attack_type.lower() in os.path.basename(f).lower():
            feat, lab = load_survival_txt(f)
            all_features.append(feat)
            all_labels.append(lab)

    features = np.concatenate(all_features) if all_features else np.zeros((0, 9), dtype=np.float32)
    labels   = np.concatenate(all_labels)   if all_labels   else np.zeros(0, dtype=np.int8)

    return _make_survival_dataset(features, labels, mode, window_size, stride)


def build_all_survival(
    root: str,
    mode: str = "window",
    window_size: int = 100,
    stride: int = 50,
) -> dict:
    """
    Build train + all attack-type test datasets for all vehicles.

    Returns:
        {
          "Sonata": {
            "train": Dataset,
            "flooding": Dataset,
            "fuzzy": Dataset,
            "malfunction": Dataset,
          },
          ...
        }
    """
    attack_types = ["flooding", "fuzzy", "malfunction"]
    result = {}
    for vehicle in VEHICLES:
        entry = {}
        entry["train"] = build_survival_train(root, vehicle, mode, window_size, stride)
        for atk in attack_types:
            entry[atk] = build_survival_test(root, vehicle, atk, mode, window_size, stride)
        result[vehicle] = entry
    return result


# ── Internal helper ───────────────────────────────────────────────────────────

def _make_survival_dataset(features, labels, mode, window_size, stride, horizon=10):
    if mode in ("window", "forecast"):
        return SurvivalWindowDataset(features, labels, window_size, stride)
    elif mode == "image":
        return SurvivalImageDataset(features, labels, window_size=27, stride=27)
    else:
        raise ValueError(f"Unknown mode: {mode}")
