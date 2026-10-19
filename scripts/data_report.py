"""Dataset sanity report -> results/tables/data_*.csv and results/figures/data_*.png
  python scripts/data_report.py [--feat mel64_w400]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spoofdet.common import paths
from spoofdet.common.io import read_manifest
from spoofdet.data.features import FeatureStore

ap = argparse.ArgumentParser()
ap.add_argument("--feat", default="mel64_w400")
a = ap.parse_args()
paths.TABLES.mkdir(parents=True, exist_ok=True); paths.FIGURES.mkdir(parents=True, exist_ok=True)
df = read_manifest(paths.MANIFESTS / "master.csv")

# 1. counts
cnt = df.groupby(["dataset", "partition", "label"]).size().unstack(fill_value=0).rename(columns={0: "bonafide", 1: "spoof"})
cnt.to_csv(paths.TABLES / "data_counts.csv"); print(cnt)
df[df.label == 1].groupby(["dataset", "partition", "method_id", "method_family"]).size().rename("n").to_csv(paths.TABLES / "data_per_method.csv")
df.groupby("dataset")["speaker_id"].nunique().rename("n_speakers").to_csv(paths.TABLES / "data_speakers.csv")

# 2. duration histograms (bona fide vs spoof) - duration is a known shortcut!
fig, axes = plt.subplots(1, df.dataset.nunique(), figsize=(4.5 * df.dataset.nunique(), 3.2), squeeze=False)
for ax, (ds, d) in zip(axes[0], df.groupby("dataset")):
    for lab, name in ((0, "bona fide"), (1, "spoof")):
        ax.hist(d[d.label == lab].duration_s.dropna().clip(upper=15), bins=40, alpha=.6, label=name, density=True)
    ax.set(title=f"{ds}: duration (s)", xlabel="seconds"); ax.legend()
fig.tight_layout(); fig.savefig(paths.FIGURES / "data_durations.png", dpi=200); plt.close(fig)
dur = df.groupby(["dataset", "label"]).duration_s.agg(["mean", "median", "std"]); dur.to_csv(paths.TABLES / "data_duration_stats.csv"); print(dur)

# 3. mean spectrum per class and example spectrograms
store = FeatureStore(paths.FEATURES / a.feat)
fig, ax = plt.subplots(figsize=(6, 3.5))
for ds in df.dataset.unique():
    for lab, ls in ((0, "-"), (1, "--")):
        ids = df[(df.dataset == ds) & (df.label == lab) & df.utt_id.isin(store.row.keys())].utt_id
        ids = ids.sample(min(300, len(ids)), random_state=0)
        if len(ids):
            ax.plot(np.mean([store.get(u).mean(1) for u in ids], axis=0), ls, label=f"{ds} {'spoof' if lab else 'bona'}")
ax.set(xlabel="mel band", ylabel="mean dB", title="Mean log-mel profile"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(paths.FIGURES / "data_mean_spectrum.png", dpi=200); plt.close(fig)

la = df[(df.dataset == "la19") & df.utt_id.isin(store.row.keys())]
groups = ["bonafide"] + sorted(m for m in la.method_id.unique() if m != "bonafide")
cols = 5; rows = int(np.ceil(len(groups) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows), squeeze=False)
for ax in axes.ravel(): ax.axis("off")
for ax, g in zip(axes.ravel(), groups):
    u = la[la.method_id == g].utt_id.iloc[0]
    ax.imshow(store.get(u), origin="lower", aspect="auto", cmap="magma"); ax.set_title(g, fontsize=8)
fig.tight_layout(); fig.savefig(paths.FIGURES / "data_examples.png", dpi=150); plt.close(fig)
print("report written to", paths.TABLES, paths.FIGURES)
