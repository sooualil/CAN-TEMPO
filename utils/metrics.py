import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_metrics(scores, labels, threshold):
    lb   = np.asarray(labels, dtype=int)
    sc   = np.asarray(scores, dtype=float)
    pred = (sc >= threshold).astype(int)
    has_both = lb.sum() > 0 and (lb == 0).sum() > 0
    auc_roc = float(roc_auc_score(lb, sc))           if has_both else float("nan")
    auc_pr  = float(average_precision_score(lb, sc)) if has_both else float("nan")
    tp = int(((pred == 1) & (lb == 1)).sum())
    tn = int(((pred == 0) & (lb == 0)).sum())
    fp = int(((pred == 1) & (lb == 0)).sum())
    fn = int(((pred == 0) & (lb == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "auc_roc": auc_roc, "auc_pr": auc_pr,
        "f1": f1, "precision": prec, "recall": rec, "fpr": fpr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def avg_metrics(metric_list):
    out = {}
    for k in metric_list[0]:
        vals = [m[k] for m in metric_list if not np.isnan(m[k])]
        out[f"{k}_mean"] = float(np.nanmean(vals)) if vals else float("nan")
        out[f"{k}_std"]  = float(np.nanstd(vals))  if vals else float("nan")
    return out
