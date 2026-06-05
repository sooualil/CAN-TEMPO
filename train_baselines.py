"""
Train a baseline model on CHD or SAD and evaluate with a 95th-percentile threshold.

Usage:
  python train_baselines.py --model cnn_ae  --dataset chd --data_root ./data/raw
  python train_baselines.py --model rlids   --dataset sad --data_root ./data/raw --seed 7
  python train_baselines.py --model descids --dataset chd --data_root ./data/raw --device cuda:1

Supported models: cnn_ae, lstm_ae, gru_ae, autocoder, cgts, descids, rlids
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import set_seed, compute_metrics, avg_metrics

CHD_ATTACKS  = ["DoS", "Fuzzy", "Gear", "RPM"]
SAD_VEHICLES = ["Sonata", "Soul", "Spark"]
SAD_ATTACKS  = ["Flooding", "Fuzzy", "Malfunction"]

# ── Per-model defaults (paper values) ─────────────────────────────────────────

MODEL_DEFAULTS = {
    "cnn_ae":    dict(epochs=50, batch=64, lr=1e-4, seq_len=100, stride=50),
    "lstm_ae":   dict(epochs=50, batch=64, lr=1e-4, seq_len=100, stride=50),
    "gru_ae":    dict(epochs=50, batch=64, lr=1e-4, seq_len=100, stride=50),
    "autocoder": dict(epochs=50, batch=64, lr=1e-3, seq_len=100, stride=1, test_stride=1),
    "cgts":      dict(epochs=50, batch=64, lr=1e-4, seq_len=100, stride=100),
    "descids":   dict(epochs=50, batch=64, lr=1e-4, seq_len=128, stride=50),
    "rlids":     dict(gan_epochs=300, ae_epochs=500, batch=32, lr=1e-3,
                      seq_len=27, stride=27, latent_dim=100, dropout=0.2),
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   choices=["cnn_ae", "lstm_ae", "gru_ae", "autocoder", "cgts", "descids", "rlids"],
                   help="Baseline model to train")
    p.add_argument("--dataset",     required=True, choices=["chd", "sad"],
                   help="Benchmark dataset: chd=Car-Hacking, sad=Survival Attack")
    p.add_argument("--data_root",   default="./data/raw",  help="Root directory of datasets")
    p.add_argument("--seed",        type=int,   default=42, help="Random seed")
    p.add_argument("--device",      default="cuda:0",       help="PyTorch device string")
    p.add_argument("--percentile",  type=int,   default=95, help="Anomaly threshold percentile (computed on train set)")
    p.add_argument("--epochs",      type=int,   default=None,
                   help="Override epoch count from MODEL_DEFAULTS (all phases for rlids)")
    p.add_argument("--results_dir", default="./results",    help="Directory for JSON result files")
    p.add_argument("--ckpt_dir",    default="./checkpoints",help="Directory for model checkpoints")
    return p.parse_args()


# ── Model factory ──────────────────────────────────────────────────────────────

def build_model(name, cfg, device):
    if name == "cnn_ae":
        from models.conv_ae import ConvAE
        return ConvAE().to(device)
    if name == "lstm_ae":
        from models.conv_ae import LSTMAE
        return LSTMAE().to(device)
    if name == "gru_ae":
        from models.conv_ae import GRUAE
        return GRUAE().to(device)
    if name == "autocoder":
        from models.autocoder import ConvAttentionAE
        return ConvAttentionAE().to(device)
    if name == "cgts":
        from models.cgts import CGTS
        return CGTS().to(device)
    if name == "descids":
        from models.srcae import SRCAE
        return SRCAE(n_features=10, seq_len=cfg["seq_len"]).to(device)
    if name == "rlids":
        from models.rl_ids import RLIDS
        return RLIDS(latent_dim=cfg["latent_dim"], dropout=cfg["dropout"]).to(device)
    raise ValueError(f"Unknown model: {name}")


# ── Training routines ──────────────────────────────────────────────────────────

def train_standard(model, train_ds, cfg, device):
    """Standard autoencoder training: minimise reconstruction loss on normal traffic."""
    loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model.train()
    for ep in range(cfg["epochs"]):
        for x, _ in loader:
            x = x.to(device)
            loss, _ = model(x)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{cfg['epochs']}", flush=True)


def train_rlids(model, train_ds, cfg, device, patience=20):
    """Three-phase RL-IDS training: (1) WGAN discriminator, (2) teacher AE, (3) student AE.
    Teacher and student phases use early stopping with the given patience."""
    loader = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                        drop_last=True, num_workers=4, pin_memory=True)

    # Phase 1: train WGAN discriminator to distinguish real vs reconstructed traffic
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=cfg["lr"], betas=(0.0, 0.9))
    for _ in range(cfg["gan_epochs"]):
        for imgs, _ in loader:
            imgs = imgs.to(device)
            with torch.no_grad(): _, fake, _ = model.teacher(imgs)
            gp     = model.discriminator.gradient_penalty(imgs, fake.detach())
            d_loss = (model.discriminator(fake.detach()).mean()
                      - model.discriminator(imgs).mean() + 10.0 * gp)
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    for p in model.discriminator.parameters(): p.requires_grad_(False)

    # Phase 2 & 3: teacher AE (adversarially guided), then student AE (distilled from teacher)
    for phase, params, loss_fn in [
        ("teacher", model.teacher.parameters(),
         lambda imgs: model.teacher.gaa_loss(imgs, model.discriminator)),
        ("student", model.student.parameters(),
         lambda imgs: model.student_distillation_loss(imgs)),
    ]:
        opt = torch.optim.Adam(params, lr=cfg["lr"], betas=(0.9, 0.999))
        best, no_imp = float("inf"), 0
        for _ in range(cfg["ae_epochs"]):
            ep_loss = 0.0
            for imgs, _ in loader:
                imgs = imgs.to(device)
                loss = loss_fn(imgs)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(params), 1.0)
                opt.step()
                ep_loss += loss.item()
            if ep_loss < best - 1e-6:
                best, no_imp = ep_loss, 0
            else:
                no_imp += 1
                if no_imp >= patience: break
        if phase == "teacher":
            for p in model.teacher.parameters(): p.requires_grad_(False)


def train_cgts(model, train_ds, cfg, device):
    from torch_geometric.loader import DataLoader as GeoLoader
    loader = GeoLoader(train_ds, batch_size=cfg["batch"], shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model.train()
    for ep in range(cfg["epochs"]):
        for batch in loader:
            batch = batch.to(device)
            loss = model(batch)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{cfg['epochs']}", flush=True)


# ── Scoring ────────────────────────────────────────────────────────────────────

def score_ds(model, ds, device):
    """Return per-window anomaly scores and labels. Uses anomaly_score() if available."""
    model.eval()
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            if hasattr(model, "anomaly_score"):
                sc = model.anomaly_score(x.to(device))
            else:
                _, sc = model(x.to(device))
            scores.append(sc.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


# ── Dataset builders ───────────────────────────────────────────────────────────

def get_chd_datasets(args, cfg):
    from data.dataset_car_hacking import build_car_hacking_train, build_car_hacking_test
    root  = os.path.join(args.data_root, "Car-Hacking")
    mode  = "image" if args.model == "rlids" else "window"
    train = build_car_hacking_train(root, mode=mode, window_size=cfg["seq_len"],
                                    stride=cfg["stride"])
    test_stride = cfg.get("test_stride", cfg["seq_len"])
    tests = {atk: build_car_hacking_test(root, attack=atk, mode=mode,
                                          window_size=cfg["seq_len"],
                                          stride=test_stride)
             for atk in CHD_ATTACKS}
    return train, tests


def get_sad_datasets(args, cfg, vehicle):
    from data.dataset_survival import build_survival_train, build_survival_test
    root  = os.path.join(args.data_root, "Survival")
    mode  = "image" if args.model == "rlids" else "window"
    train = build_survival_train(root, vehicle=vehicle, mode=mode,
                                 window_size=cfg["seq_len"], stride=cfg["stride"])
    test_stride = cfg.get("test_stride", cfg["seq_len"])
    tests = {}
    for atk in SAD_ATTACKS:
        try:
            tests[atk] = build_survival_test(root, vehicle=vehicle, attack_type=atk,
                                              mode=mode, window_size=cfg["seq_len"],
                                              stride=test_stride)
        except Exception as e:
            print(f"  {vehicle}/{atk} skipped: {e}", flush=True)
    return train, tests


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(args, cfg, device, label, train_ds, test_ds_dict):
    set_seed(args.seed)
    model = build_model(args.model, cfg, device)

    print(f"[{args.model} s={args.seed}] {label} training ...", flush=True)
    if args.model == "rlids":
        train_rlids(model, train_ds, cfg, device)
    elif args.model == "cgts":
        train_cgts(model, train_ds, cfg, device)
    else:
        train_standard(model, train_ds, cfg, device)

    ckpt = os.path.join(args.ckpt_dir, f"{args.model}_{args.dataset}_{label}_seed{args.seed}.pt")
    torch.save(model.state_dict(), ckpt)

    # Threshold from training scores at given percentile (no test data leakage)
    tr_sc, _ = score_ds(model, train_ds, device)
    thr = float(np.percentile(tr_sc, args.percentile))
    print(f"  threshold (p{args.percentile}) = {thr:.5f}", flush=True)

    results = {}
    metrics_list = []
    for key, test_ds in test_ds_dict.items():
        sc, lb = score_ds(model, test_ds, device)
        m = compute_metrics(sc, lb, thr)
        results[key] = m
        metrics_list.append(m)
        print(f"  {key:15s}  AUC={m['auc_roc']:.4f}  F1={m['f1']:.4f}  "
              f"FPR={m['fpr']:.4f}", flush=True)
    results["avg"] = avg_metrics(metrics_list)
    return results


def main():
    args   = get_args()
    device = torch.device(args.device)
    cfg    = MODEL_DEFAULTS[args.model].copy()
    if args.epochs is not None:
        for key in ("epochs", "gan_epochs", "ae_epochs"):
            if key in cfg:
                cfg[key] = args.epochs
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir,    exist_ok=True)

    print(f"{args.model.upper()} | dataset={args.dataset} seed={args.seed} device={args.device}",
          flush=True)

    if args.dataset == "chd":
        train_ds, test_ds = get_chd_datasets(args, cfg)
        results = run(args, cfg, device, "chd", train_ds, test_ds)
    else:
        # Train and evaluate independently per vehicle
        results = {}
        for veh in SAD_VEHICLES:
            train_ds, test_ds = get_sad_datasets(args, cfg, veh)
            results[veh] = run(args, cfg, device, veh, train_ds, test_ds)

    out = os.path.join(args.results_dir, f"{args.model}_{args.dataset}_seed{args.seed}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()
