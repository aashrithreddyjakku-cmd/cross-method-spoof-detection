"""Standard figures. All functions take a score DataFrame (SCORE_COLUMNS) and an output path."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_curve

from .metrics import per_group_eer


def plot_roc_det(named_dfs: dict, out_png):
    """named_dfs: {'label': score_df}. Left: ROC. Right: DET (normal-deviate axes)."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, df in named_dfs.items():
        fpr, tpr, _ = roc_curve(df.label, df.score)
        ax[0].plot(fpr, tpr, label=name)
        fnr = 1 - tpr; eps = 1e-3
        ax[1].plot(norm.ppf(np.clip(fpr, eps, 1 - eps)), norm.ppf(np.clip(fnr, eps, 1 - eps)), label=name)
    ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
    ticks = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99])
    ax[1].set_xticks(norm.ppf(ticks)); ax[1].set_xticklabels([f"{t*100:g}" for t in ticks])
    ax[1].set_yticks(norm.ppf(ticks)); ax[1].set_yticklabels([f"{t*100:g}" for t in ticks])
    ax[1].set(xlabel="FPR (%)", ylabel="FNR (%)", title="DET")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def plot_per_method_bar(df, out_png, title="EER per spoofing method", seen_methods=()):
    t = per_group_eer(df, "method_id").sort_values("group")
    colors = ["tab:blue" if g in seen_methods else "tab:red" for g in t.group]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(t) + 2), 3.8))
    ax.bar(t.group, t.eer * 100, color=colors)
    ax.set(ylabel="EER (%)", title=title); ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def plot_confusion(tn, fp, fn, tp, out_png, title="Confusion matrix"):
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["bona", "spoof"], yticklabels=["bona", "spoof"],
           xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, out_png, title="", fmt="{:.1f}"):
    """matrix: rows = experiments / held-out method, columns = test method; values EER in %."""
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * matrix.shape[1] + 2), max(3, 0.4 * matrix.shape[0] + 1.5)))
    im = ax.imshow(matrix.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(matrix.shape[1])); ax.set_xticklabels(matrix.columns, rotation=60)
    ax.set_yticks(range(matrix.shape[0])); ax.set_yticklabels(matrix.index)
    for (i, j), v in np.ndenumerate(matrix.values):
        if not np.isnan(v):
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6, color="white")
    fig.colorbar(im, ax=ax, label="EER (%)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
