"""
Train CAN-TEMPO on CHD or SAD and evaluate with a 95th-percentile threshold.

Usage:
  python train_cantempo.py --dataset chd --data_root ./data/raw --device cuda:0
  python train_cantempo.py --dataset sad --data_root ./data/raw --seed 123 --device cuda:1
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.can_tempo import CANTEMPO
from utils import set_seed, compute_metrics, avg_metrics

CHD_ATTACKS  = ["DoS", "Fuzzy", "Gear", "RPM"]
SAD_VEHICLES = ["Sonata", "Soul", "Spark"]
SAD_ATTACKS  = ["Flooding", "Fuzzy", "Malfunction"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     required=True, choices=["chd", "sad"])
    p.add_argument("--data_root",   default="./data/raw",  help="Root directory of datasets")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="cuda:0")
    # Training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    # Windowing
    p.add_argument("--seq_len",     type=int,   default=100, help="Sliding window length")
    p.add_argument("--stride",      type=int,   default=50,  help="Sliding window stride")
    # Model
    p.add_argument("--d_model",     type=int,   default=256, help="Embedding dimension")
    p.add_argument("--n_layers",    type=int,   default=4,   help="Number of TMPO blocks")
    p.add_argument("--top_k",       type=int,   default=2,   help="Dominant periods K")
    p.add_argument("--lambda_stat", type=float, default=4.0, help="Statistical consistency loss weight")
    p.add_argument("--lambda_fft",  type=float, default=0.1, help="Spectral reconstruction loss weight")
    # Evaluation
    p.add_argument("--percentile",  type=int,   default=95,  help="Anomaly threshold percentile (computed on train set)")
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--ckpt_dir",    default="./checkpoints")
    return p.parse_args()


def train(model, train_ds, args, device):
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    for ep in range(args.epochs):
        total = 0.0
        for x, _ in loader:
            x = x.to(device)
            loss, _ = model(x)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0 or args.epochs < 10:
            print(f"  epoch {ep+1}/{args.epochs}  loss={total/len(loader):.4f}", flush=True)


def score(model, ds, device):
    """Return per-window anomaly scores and labels for a dataset."""
    model.eval()
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            _, sc = model(x.to(device))
            scores.append(sc.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def run_chd(args, device):
    from data.dataset_car_hacking import build_car_hacking_train, build_car_hacking_test
    chd_root = os.path.join(args.data_root, "Car-Hacking")

    set_seed(args.seed)
    model = CANTEMPO(seq_len=args.seq_len, d_model=args.d_model, n_layers=args.n_layers,
                     top_k=args.top_k, lambda_stat=args.lambda_stat,
                     lambda_fft=args.lambda_fft).to(device)

    print(f"[CHD s={args.seed}] training ...", flush=True)
    train_ds = build_car_hacking_train(chd_root, mode="window",
                                       window_size=args.seq_len, stride=args.stride)
    train(model, train_ds, args, device)

    ckpt = os.path.join(args.ckpt_dir, f"cantempo_chd_seed{args.seed}.pt")
    torch.save(model.state_dict(), ckpt)

    # Threshold from training scores at given percentile (no test data leakage)
    tr_sc, _ = score(model, train_ds, device)
    thr = float(np.percentile(tr_sc, args.percentile))
    print(f"  threshold (p{args.percentile}) = {thr:.5f}", flush=True)

    results = {}
    metrics_list = []
    for atk in CHD_ATTACKS:
        test_ds = build_car_hacking_test(chd_root, attack=atk, mode="window",
                                         window_size=args.seq_len, stride=args.seq_len)
        sc, lb = score(model, test_ds, device)
        m = compute_metrics(sc, lb, thr)
        results[atk] = m
        metrics_list.append(m)
        print(f"  {atk:6s}  AUC={m['auc_roc']:.4f}  F1={m['f1']:.4f}  FPR={m['fpr']:.4f}", flush=True)

    results["avg"] = avg_metrics(metrics_list)
    return results


def run_sad(args, device):
    from data.dataset_survival import build_survival_train, build_survival_test
    sad_root = os.path.join(args.data_root, "Survival")

    # Train and evaluate independently per vehicle
    results = {}
    for veh in SAD_VEHICLES:
        set_seed(args.seed)
        model = CANTEMPO(seq_len=args.seq_len, d_model=args.d_model, n_layers=args.n_layers,
                         top_k=args.top_k, lambda_stat=args.lambda_stat,
                         lambda_fft=args.lambda_fft).to(device)

        print(f"[SAD s={args.seed}] {veh} training ...", flush=True)
        train_ds = build_survival_train(sad_root, vehicle=veh, mode="window",
                                        window_size=args.seq_len, stride=args.stride)
        train(model, train_ds, args, device)

        ckpt = os.path.join(args.ckpt_dir, f"cantempo_sad_{veh}_seed{args.seed}.pt")
        torch.save(model.state_dict(), ckpt)

        tr_sc, _ = score(model, train_ds, device)
        thr = float(np.percentile(tr_sc, args.percentile))
        print(f"  threshold (p{args.percentile}) = {thr:.5f}", flush=True)

        veh_results = {}
        metrics_list = []
        for atk in SAD_ATTACKS:
            try:
                test_ds = build_survival_test(sad_root, vehicle=veh, attack_type=atk,
                                              mode="window", window_size=args.seq_len,
                                              stride=args.seq_len)
                sc, lb = score(model, test_ds, device)
                m = compute_metrics(sc, lb, thr)
            except Exception as e:
                print(f"  {veh}/{atk} skipped: {e}", flush=True)
                m = {k: float("nan") for k in
                     ["auc_roc","auc_pr","f1","precision","recall","fpr","tp","tn","fp","fn"]}
            veh_results[atk] = m
            metrics_list.append(m)
            print(f"  {veh}/{atk:12s}  AUC={m['auc_roc']:.4f}  F1={m['f1']:.4f}  "
                  f"FPR={m['fpr']:.4f}", flush=True)

        veh_results["avg"] = avg_metrics(metrics_list)
        results[veh] = veh_results

    return results


def main():
    args   = get_args()
    device = torch.device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir,    exist_ok=True)

    print(f"CAN-TEMPO | dataset={args.dataset} seed={args.seed} device={args.device}", flush=True)
    print(f"  d={args.d_model} L={args.n_layers} K={args.top_k} "
          f"λ_stat={args.lambda_stat} λ_fft={args.lambda_fft}", flush=True)

    if args.dataset == "chd":
        results = run_chd(args, device)
    else:
        results = run_sad(args, device)

    out = os.path.join(args.results_dir, f"cantempo_{args.dataset}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()
