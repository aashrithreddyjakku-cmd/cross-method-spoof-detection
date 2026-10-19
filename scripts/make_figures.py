"""Standard figure set from score files and tables (run after evaluate --all and aggregate).
   python scripts/make_figures.py
Writes results/figures/: roc_det_p0_test.png, roc_det_p0_itw.png, gap_<kind>.png, lomo_heatmap.png,
                         ablation.png, confusion_<exp>_<test>.png, per_method_<exp>_<test>.png
Skips any figure whose input does not exist yet."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spoofdet.common import paths
from spoofdet.common.io import read_scores
from spoofdet.eval.aggregate import load_meta
from spoofdet.eval.plots import plot_roc_det, plot_heatmap, plot_confusion, plot_per_method_bar

paths.FIGURES.mkdir(parents=True, exist_ok=True)
res = pd.read_csv(paths.TABLES / "all_results.csv")
meta = load_meta(res.exp_id.unique())
df = res.merge(meta, on="exp_id")

# 1. ROC + DET of the P0 models (seed 1) on the official test set and on In-the-Wild
p0 = meta[(meta.kind == "P0_official") & (meta.seed == meta[meta.kind == "P0_official"].seed.min())] if (meta.kind == "P0_official").any() else meta.iloc[0:0]
for test in ("test", "itw"):
    named = {}
    for _, r in p0.iterrows():
        f = paths.SCORES / r.exp_id / f"{test}.csv"
        if f.exists():
            named[r.model] = read_scores(f)
    if named:
        plot_roc_det(named, paths.FIGURES / f"roc_det_p0_{test}.png")

# 2. seen vs unseen EER per held-out method / family, one panel per model
for kind in ("P1_lomo", "P1_lofo"):
    d = df[(df.kind == kind) & df.test_set.isin(["test_seen", "test_unseen"])]
    if d.empty:
        continue
    g = d.groupby(["model", "held_out", "test_set"]).eer.mean().unstack() * 100
    models = sorted(g.index.get_level_values(0).unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 3.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        t = g.loc[m].sort_index(); x = np.arange(len(t))
        ax.bar(x - .2, t.get("test_seen", 0), .4, label="seen methods"); ax.bar(x + .2, t["test_unseen"], .4, label="held-out method", color="tab:red")
        ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=60, fontsize=8); ax.set(title=m, ylabel="EER (%)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(paths.FIGURES / f"gap_{kind.lower()}.png", dpi=200); plt.close(fig)

# 3. LOMO heat-map: rows = model, columns = held-out method (default features/norm/aug), value = unseen EER %
d = df[(df.kind == "P1_lomo") & (df.test_set == "test_unseen") & (df.feat_id == "mel64_w400") & (df.norm == "utt") & df.augment]
if len(d):
    mat = d.groupby(["model", "held_out"]).eer.mean().unstack() * 100
    plot_heatmap(mat, paths.FIGURES / "lomo_heatmap.png", "EER (%) on the held-out method")

# 4. ablation bars (mean unseen EER over folds)
ab = df[df.exp_id.str.startswith("abl_") & (df.test_set == "test_unseen")]
if len(ab):
    ab = ab.assign(level=ab.apply(lambda r: f"{r.feat_id} | {r.norm} | aug={r.augment}", axis=1))
    t = ab.groupby("level").eer.agg(["mean", "std"]).sort_values("mean") * 100
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(t) + 1.2))
    ax.barh(t.index, t["mean"], xerr=t["std"].fillna(0), color="tab:purple"); ax.invert_yaxis()
    ax.set(xlabel="EER on held-out folds (%)", title="Ablations of the mel-spectrogram front end")
    fig.tight_layout(); fig.savefig(paths.FIGURES / "ablation.png", dpi=200); plt.close(fig)

# 5. confusion matrices and per-method bars for every P0 / P1 experiment's main test files (threshold from val)
for _, r in df[df.test_set.isin(["test", "test_unseen", "itw"])].iterrows():
    k = ("tn_v", "fp_v", "fn_v", "tp_v") if "tn_v" in r and not pd.isna(r.get("tn_v")) else ("tn", "fp", "fn", "tp")
    plot_confusion(*[int(r[c]) for c in k], paths.FIGURES / f"confusion_{r.exp_id}_{r.test_set}.png", f"{r.exp_id}\n{r.test_set}")
    plot_per_method_bar(read_scores(paths.SCORES / r.exp_id / f"{r.test_set}.csv"),
                        paths.FIGURES / f"per_method_{r.exp_id}_{r.test_set}.png", f"{r.exp_id} / {r.test_set}")
print("figures written to", paths.FIGURES)
