"""
Evaluate a saved checkpoint on CHD or SAD.

Usage:
  python evaluate.py --model cantempo --checkpoint ./checkpoints/cantempo_chd_seed42.pt \
                     --dataset chd --data_root ./data/raw
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import compute_metrics, avg_metrics

CHD_ATTACKS  = ["DoS", "Fuzzy", "Gear", "RPM"]
SAD_VEHICLES = ["Sonata", "Soul", "Spark"]
SAD_ATTACKS  = ["Flooding", "Fuzzy", "Malfunction"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       required=True,
                   choices=["cantempo", "cnn_ae", "lstm_ae", "gru_ae",
                            "autocoder", "cgts", "descids", "rlids"])
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--dataset",     required=True, choices=["chd", "sad"])
    p.add_argument("--data_root",   default="./data/raw")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--percentile",  type=int, default=95)
    p.add_argument("--results_dir", default="./results")
    return p.parse_args()


def load_model(args, device):
    ckpt = torch.load(args.checkpoint, map_location=device)
    if args.model == "cantempo":
        from models.can_tempo import CANTEMPO
        model = CANTEMPO()
    elif args.model == "cnn_ae":
        from models.conv_ae import ConvAE
        model = ConvAE()
    elif args.model == "lstm_ae":
        from models.conv_ae import LSTMAE
        model = LSTMAE()
    elif args.model == "gru_ae":
        from models.conv_ae import GRUAE
        model = GRUAE()
    elif args.model == "autocoder":
        from models.autocoder import ConvAttentionAE
        model = ConvAttentionAE()
    elif args.model == "cgts":
        from models.cgts import CGTS
        model = CGTS()
    elif args.model == "descids":
        from models.srcae import SRCAE
        model = SRCAE()
    elif args.model == "rlids":
        from models.rl_ids import RLIDS
        model = RLIDS()
    model.load_state_dict(ckpt)
    return model.to(device)


def score_ds(model, ds, device):
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


def main():
    args   = get_args()
    device = torch.device(args.device)
    os.makedirs(args.results_dir, exist_ok=True)

    model = load_model(args, device)
    mode  = "image" if args.model == "rlids" else "window"

    if args.dataset == "chd":
        from data.dataset_car_hacking import build_car_hacking_train, build_car_hacking_test
        root     = os.path.join(args.data_root, "Car-Hacking")
        train_ds = build_car_hacking_train(root, mode=mode)
        tr_sc, _ = score_ds(model, train_ds, device)
        thr      = float(np.percentile(tr_sc, args.percentile))
        print(f"threshold (p{args.percentile}) = {thr:.5f}", flush=True)

        results, metrics_list = {}, []
        for atk in CHD_ATTACKS:
            test_ds = build_car_hacking_test(root, attack=atk, mode=mode)
            sc, lb  = score_ds(model, test_ds, device)
            m = compute_metrics(sc, lb, thr)
            results[atk] = m
            metrics_list.append(m)
            print(f"  {atk:6s}  AUC={m['auc_roc']:.4f}  F1={m['f1']:.4f}  "
                  f"FPR={m['fpr']:.4f}", flush=True)
        results["avg"] = avg_metrics(metrics_list)

    else:
        from data.dataset_survival import build_survival_train, build_survival_test
        root    = os.path.join(args.data_root, "Survival")
        results = {}
        for veh in SAD_VEHICLES:
            train_ds = build_survival_train(root, vehicle=veh, mode=mode)
            tr_sc, _ = score_ds(model, train_ds, device)
            thr      = float(np.percentile(tr_sc, args.percentile))
            print(f"{veh} threshold (p{args.percentile}) = {thr:.5f}", flush=True)

            veh_results, metrics_list = {}, []
            for atk in SAD_ATTACKS:
                try:
                    test_ds = build_survival_test(root, vehicle=veh, attack_type=atk, mode=mode)
                    sc, lb  = score_ds(model, test_ds, device)
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

    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out  = os.path.join(args.results_dir, f"eval_{name}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()
