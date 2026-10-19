"""All detection metrics. Convention everywhere: label 1 = spoof (positive), score = P(spoof)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

from ..common.metrics_core import compute_eer


def metrics_at_threshold(scores, labels, thr):
    pred = (np.asarray(scores) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=float((tn + tp) / len(labels)),
                far=float(fp / max(fp + tn, 1)),         # bona fide accepted as spoof (false alarm)
                frr=float(fn / max(fn + tp, 1)))         # spoof missed


def summarize(scores, labels, fixed_thr=0.5):
    """Full metric dict for one set of scores."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    eer, thr = compute_eer(scores, labels)
    at_eer = metrics_at_threshold(scores, labels, thr)
    at_fix = metrics_at_threshold(scores, labels, fixed_thr)
    return dict(n=len(labels), n_bona=int((labels == 0).sum()), n_spoof=int((labels == 1).sum()),
                eer=eer, eer_thr=thr, auc=float(roc_auc_score(labels, scores)),
                acc_at_eer=at_eer["accuracy"], acc_at_0p5=at_fix["accuracy"],
                tn=at_eer["tn"], fp=at_eer["fp"], fn=at_eer["fn"], tp=at_eer["tp"])


def bootstrap_eer_ci(scores, labels, n_boot=1000, alpha=0.05, seed=0):
    rng = np.random.RandomState(seed); scores = np.asarray(scores); labels = np.asarray(labels)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(compute_eer(scores[idx], labels[idx])[0])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def per_group_eer(df: pd.DataFrame, group_col="method_id", ref_thr=None):
    """EER of each spoof group vs ALL bona fide rows in df.  Returns DataFrame sorted by EER desc.
    If ref_thr is given (e.g. the EER threshold of the validation set) also reports detection rate there."""
    bona = df[df.label == 0]
    rows = []
    for g, d in df[df.label == 1].groupby(group_col):
        sub = pd.concat([bona, d])
        r = dict(group=g, n_spoof=len(d), eer=compute_eer(sub.score, sub.label)[0],
                 auc=float(roc_auc_score(sub.label, sub.score)))
        if ref_thr is not None:
            r["detect_rate_at_ref_thr"] = float((d.score >= ref_thr).mean())
        rows.append(r)
    return pd.DataFrame(rows).sort_values("eer", ascending=False).reset_index(drop=True)


def paired_bootstrap_eer_diff(df_a: pd.DataFrame, df_b: pd.DataFrame, n_boot=1000, seed=0):
    """EER(A) - EER(B) on the SAME utterances (merge on utt_id). Returns (mean diff, 95% CI lo, hi, share of
    resamples where A is better i.e. lower EER). Use it to decide if a difference between two systems is real."""
    m = df_a[["utt_id", "label", "score"]].merge(df_b[["utt_id", "score"]], on="utt_id", suffixes=("_a", "_b"))
    rng = np.random.RandomState(seed); y = m.label.to_numpy(); a = m.score_a.to_numpy(); b = m.score_b.to_numpy()
    d = []
    for _ in range(n_boot):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(compute_eer(a[i], y[i])[0] - compute_eer(b[i], y[i])[0])
    d = np.asarray(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d < 0).mean())
