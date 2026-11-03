"""Report figures for detection scores and D4 aggregation tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

from .metrics import per_group_eer


def _save(fig, out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _score_arrays(df):
    missing = {"label", "score"} - set(df.columns)
    if missing:
        raise ValueError(f"score data are missing columns: {sorted(missing)}")
    try:
        numeric_labels = pd.to_numeric(df["label"], errors="raise").to_numpy(dtype=float)
        scores = pd.to_numeric(df["score"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("score data must contain numeric labels and scores") from exc
    if numeric_labels.ndim != 1 or scores.ndim != 1 or len(numeric_labels) != len(scores) or not len(numeric_labels):
        raise ValueError("score data must contain non-empty, one-dimensional labels and scores")
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(numeric_labels).all()
        or not np.isin(numeric_labels, (0, 1)).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("scores must be finite and labels must use 0=bona fide, 1=spoof")
    labels = numeric_labels.astype(int)
    if len(np.unique(labels)) != 2:
        raise ValueError("a curve requires both bona-fide and spoof examples")
    return labels, scores


def plot_roc_det(named_dfs: dict, out_png):
    """Draw ROC and DET curves for ``{label: score_df}`` inputs."""
    if not named_dfs:
        raise ValueError("at least one labeled score set is required")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, df in named_dfs.items():
        labels, scores = _score_arrays(df)
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        auc = roc_auc_score(labels, scores)
        legend = f"{name} (AUC={auc:.3f})"
        ax[0].plot(fpr * 100, tpr * 100, label=legend, lw=1.8)

        fnr = 1 - tpr
        eps = 1e-3
        ax[1].plot(
            norm.ppf(np.clip(fpr, eps, 1 - eps)),
            norm.ppf(np.clip(fnr, eps, 1 - eps)),
            label=legend,
            lw=1.8,
        )

    ax[0].plot([0, 100], [0, 100], "k:", lw=1, label="Chance")
    ax[0].set(
        xlabel="False positive rate (%)",
        ylabel="True positive rate (%)",
        title="Receiver operating characteristic",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    ax[0].set_xticks(np.arange(0, 101, 20))
    ax[0].set_yticks(np.arange(0, 101, 20))

    ticks = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99])
    det_positions = norm.ppf(ticks)
    tick_labels = [f"{value * 100:g}%" for value in ticks]
    ax[1].plot(det_positions, det_positions, "k:", lw=1, label="Equal error")
    ax[1].set_xticks(det_positions, tick_labels)
    ax[1].set_yticks(det_positions, tick_labels)
    ax[1].set(
        xlabel="False positive rate (%)",
        ylabel="False negative rate (%)",
        title="Detection error tradeoff",
        xlim=(det_positions[0], det_positions[-1]),
        ylim=(det_positions[0], det_positions[-1]),
    )
    for axis in ax:
        axis.grid(True, alpha=0.22)
        axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save(fig, out_png)


def plot_per_method_bar(df, out_png, title="EER per spoofing method", seen_methods=()):
    """Compare each spoofing method against all bona-fide examples in the file."""
    _score_arrays(df)
    if "method_id" not in df.columns:
        raise ValueError("score data are missing method_id")
    t = per_group_eer(df, "method_id").sort_values("group")
    if t.empty:
        raise ValueError("no spoofing methods are available for the per-method EER plot")

    seen_methods = set(seen_methods)
    colors = (
        ["tab:blue" if group in seen_methods else "tab:red" for group in t.group]
        if seen_methods else ["tab:blue"] * len(t)
    )
    fig, ax = plt.subplots(figsize=(max(6.5, 0.48 * len(t) + 2), 4.2))
    ax.bar(t.group, t.eer * 100, color=colors)
    ax.set_xlabel("Spoofing method")
    ax.set(ylabel="Equal error rate (%)", title=title)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=60)
    if seen_methods:
        ax.legend(handles=[
            Patch(color="tab:blue", label="Seen method"),
            Patch(color="tab:red", label="Unseen method"),
        ])
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    _save(fig, out_png)


def plot_confusion(tn, fp, fn, tp, out_png, title="Confusion matrix"):
    """Plot a count matrix with bona fide as class 0 and spoof as class 1."""
    counts = np.asarray([tn, fp, fn, tp], dtype=float)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("confusion-matrix counts must be non-negative integers")
    cm = counts.astype(int).reshape(2, 2)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(int(cm.max()), 1))
    for (i, j), value in np.ndenumerate(cm):
        ax.text(
            j, i, f"{value:,}", ha="center", va="center",
            color="white" if value > cm.max() / 2 else "black",
        )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Bona fide", "Spoof"],
        yticklabels=["Bona fide", "Spoof"],
        xlabel="Predicted class",
        ylabel="True class",
        title=title,
    )
    fig.colorbar(image, ax=ax, label="Utterance count")
    fig.tight_layout()
    _save(fig, out_png)


def plot_heatmap(matrix: pd.DataFrame, out_png, title="", fmt="{:.1f}"):
    """Plot a numeric EER (%) matrix with rows as models and columns as folds."""
    if matrix.empty:
        raise ValueError("cannot plot an empty heatmap")
    values = matrix.to_numpy(dtype=float)
    if values.ndim != 2 or not np.isfinite(values).any():
        raise ValueError("heatmap must contain at least one finite numeric value")
    fig, ax = plt.subplots(
        figsize=(max(6.5, 0.55 * matrix.shape[1] + 2), max(3.4, 0.45 * matrix.shape[0] + 1.5))
    )
    im = ax.imshow(values, cmap="viridis_r", aspect="auto", vmin=0)
    ax.set_xticks(range(matrix.shape[1]), labels=matrix.columns, rotation=55, ha="right")
    ax.set_yticks(range(matrix.shape[0]), labels=matrix.index)
    for (i, j), value in np.ndenumerate(values):
        if np.isfinite(value):
            text_color = "white" if value > np.nanmax(values) * 0.55 else "black"
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=7, color=text_color)
    ax.set_xlabel("Held-out method or family")
    ax.set_ylabel("Model")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Equal error rate (%)")
    fig.tight_layout()
    _save(fig, out_png)


def plot_generalization_gap(frame: pd.DataFrame, out_png, title="Generalization gap"):
    """Plot unseen-minus-seen EER in percentage points, with across-seed SD."""
    required = {"model", "held_out", "gap_unseen_minus_seen"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"generalization-gap data are missing columns: {missing}")
    data = frame.dropna(subset=["gap_unseen_minus_seen"]).copy()
    if data.empty:
        raise ValueError("no matched seen/unseen results are available for the gap plot")
    if data.duplicated(["model", "held_out"]).any():
        raise ValueError("generalization-gap data must have one row per model and held-out group")

    models = sorted(data.model.astype(str).unique())
    fig, axes = plt.subplots(
        len(models), 1,
        figsize=(max(7, 0.62 * data.held_out.nunique() + 3), 3.5 * len(models)),
        squeeze=False,
        sharex=True,
    )
    for axis, model in zip(axes[:, 0], models):
        group = data[data.model.astype(str) == model].sort_values("held_out")
        x = np.arange(len(group))
        mean = group["gap_unseen_minus_seen"].to_numpy(dtype=float) * 100
        std_col = "gap_unseen_minus_seen_std"
        yerr = group[std_col].fillna(0).to_numpy(dtype=float) * 100 if std_col in group else None
        axis.bar(x, mean, yerr=yerr, capsize=3, color="tab:purple", alpha=0.9)
        axis.axhline(0, color="black", lw=0.9, ls=":")
        axis.set_ylabel("EER difference (percentage points)")
        axis.set_xlabel("Held-out method or family")
        axis.set_title(model)
        axis.grid(axis="y", alpha=0.22)
        axis.set_xticks(x, labels=group.held_out.astype(str), rotation=45, ha="right")
    if len(models) == 1:
        axes[0, 0].set_title(f"{title} — {models[0]}")
    else:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97) if len(models) > 1 else None)
    _save(fig, out_png)


def plot_ablation_comparison(frame: pd.DataFrame, out_png, title="Ablation comparison"):
    """Plot D4 mean unseen EER and fold standard deviation per ablation setting."""
    required = {"model", "factor_level", "mean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"ablation data are missing columns: {missing}")
    data = frame.dropna(subset=["mean"]).copy()
    if data.empty:
        raise ValueError("no ablation results are available to plot")
    if "kind" not in data:
        data["kind"] = "all protocols"

    groups = list(data.groupby(["model", "kind"], dropna=False, sort=True))
    fig, axes = plt.subplots(
        len(groups), 1,
        figsize=(9, max(3.4, 0.48 * sum(len(group) for _, group in groups) + 1.8 * len(groups))),
        squeeze=False,
    )
    for axis, ((model, kind), group) in zip(axes[:, 0], groups):
        group = group.sort_values("mean")
        labels = group["factor_level"].astype(str)
        if "n_folds" in group:
            labels = labels + " (" + group["n_folds"].fillna(0).astype(int).astype(str) + " folds)"
        mean = group["mean"].to_numpy(dtype=float) * 100
        std = group["std"].fillna(0).to_numpy(dtype=float) * 100 if "std" in group else None
        axis.barh(labels, mean, xerr=std, capsize=3, color="tab:purple", alpha=0.9)
        axis.set_xlabel("Equal error rate on held-out data (%)")
        axis.set_title(f"{model} — {kind}")
        axis.grid(axis="x", alpha=0.22)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_png)
