"""The one metric every member needs from Day 1 (training uses it for early stopping)."""
import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(scores, labels):
    """Equal Error Rate.

    labels: 1 = spoof (positive class), 0 = bona fide.
    scores: higher = more likely spoof.
    Returns (eer in [0,1], threshold at the EER point).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("EER needs both classes (bona fide and spoof) in the data")
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])
