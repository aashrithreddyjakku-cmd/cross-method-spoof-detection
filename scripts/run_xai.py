"""Which spectral regions drive the decision, per spoofing method?
   python scripts/run_xai.py --exp lomo_A10_lcnn_mel64_w400_s1 --split test_unseen --n 100
Outputs results/tables/xai_<exp>_band_occlusion.csv and results/figures/xai_<exp>_*.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from spoofdet.common import paths
from spoofdet.common.io import read_manifest
from spoofdet.data.dataset import SpoofDataset
from spoofdet.data.features import FeatureStore
from spoofdet.eval.gradcam import GradCAM, band_occlusion, band_edges_hz
from spoofdet.models import load_checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--exp", required=True)
ap.add_argument("--split-csv", required=True, help="e.g. splits/P1_lomo/A10/test_unseen.csv")
ap.add_argument("--n", type=int, default=100, help="max utterances per method")
ap.add_argument("--bands", type=int, default=8)
a = ap.parse_args()

model, ck = load_checkpoint(paths.CKPT / a.exp / "best.pt")
store = FeatureStore(paths.FEATURES / ck["feat_cfg"]["feat_id"])
df = read_manifest(a.split_csv)
df = pd.concat([d.sample(min(a.n, len(d)), random_state=0) for _, d in df.groupby("method_id")])
tmp = paths.TABLES / "_xai_tmp.csv"; paths.TABLES.mkdir(parents=True, exist_ok=True); df.to_csv(tmp, index=False)
ds = SpoofDataset(tmp, store, ck["norm"], ck["norm_stats"])
cam = GradCAM(model)
M = ck["feat_cfg"]["n_mels"]
prof, occ_rows = {}, []
for i in range(0, len(ds), 32):
    xs = torch.stack([ds[j][0] for j in range(i, min(i + 32, len(ds)))])
    c, p = cam(xs)
    occ = band_occlusion(model, xs, a.bands)
    for k in range(len(xs)):
        r = ds.df.iloc[i + k]
        prof.setdefault(r.method_id, []).append(c[k].mean(axis=1))                  # (M,) importance over mel bands
        occ_rows.append(dict(method_id=r.method_id, label=r.label, **{f"band{b}": occ[k, b] for b in range(a.bands)}))
occ_df = pd.DataFrame(occ_rows)
occ_df.groupby("method_id").mean(numeric_only=True).to_csv(paths.TABLES / f"xai_{a.exp}_band_occlusion.csv")

hz = band_edges_hz(ck["feat_cfg"], a.bands)
paths.FIGURES.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for m, v in sorted(prof.items()):
    ax.plot(np.mean(v, axis=0), label=m, lw=2 if m == "bonafide" else 1, ls="--" if m == "bonafide" else "-")
ax.set(xlabel="mel band (0 = low frequency)", ylabel="mean Grad-CAM", title=f"Where the model looks - {a.exp}")
ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(paths.FIGURES / f"xai_{a.exp}_gradcam_profile.png", dpi=200); plt.close(fig)

mat = occ_df[occ_df.label == 1].groupby("method_id").mean(numeric_only=True).drop(columns="label")
fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(mat) + 1)))
im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-np.abs(mat.values).max(), vmax=np.abs(mat.values).max())
ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index)
ax.set_xticks(range(a.bands)); ax.set_xticklabels([f"{lo/1000:.1f}-{hi/1000:.1f}" for lo, hi in hz], rotation=45, fontsize=7)
ax.set_xlabel("frequency band (kHz)"); fig.colorbar(im, label="drop in P(spoof) when band removed")
fig.tight_layout(); fig.savefig(paths.FIGURES / f"xai_{a.exp}_band_occlusion.png", dpi=200); plt.close(fig)
tmp.unlink()
print("saved xai tables/figures for", a.exp)
