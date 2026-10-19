"""Do cheap "non-speech" cues already separate bona fide from spoof?  (duration, leading/trailing silence)
   python scripts/shortcut_baseline.py --n 2000
AUC far from 0.5 for a cue means a model could cheat with it.  Output: results/tables/shortcut_auc.csv + figure.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from spoofdet.common import paths
from spoofdet.common.io import read_manifest
from spoofdet.data.audio import load_wave


def cues(path, top_db=30):
    y = load_wave(paths.RAW / path)
    _, (i0, i1) = librosa.effects.trim(y, top_db=top_db)
    return dict(duration=len(y) / 16000, lead_sil=i0 / 16000, trail_sil=(len(y) - i1) / 16000)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--datasets", nargs="+", default=["la19", "itw"])
    a = ap.parse_args()
    df = read_manifest(paths.MANIFESTS / "master.csv")
    rows = []
    for ds in a.datasets:
        d = df[df.dataset == ds]
        if ds == "la19":
            d = d[d.partition == "train"]                      # shortcut learned from the training partition
        d = pd.concat([g.sample(min(a.n, len(g)), random_state=0) for _, g in d.groupby("label")])
        c = pd.DataFrame([cues(p) for p in tqdm(d["path"], desc=ds)], index=d.index)
        d = pd.concat([d, c], axis=1)
        for cue in ("duration", "lead_sil", "trail_sil"):
            rows.append(dict(dataset=ds, cue=cue, auc=roc_auc_score(d.label, d[cue]),
                             mean_bona=d[d.label == 0][cue].mean(), mean_spoof=d[d.label == 1][cue].mean()))
        fig, ax = plt.subplots(1, 3, figsize=(11, 3))
        for k, cue in enumerate(("duration", "lead_sil", "trail_sil")):
            for lab, nm in ((0, "bona fide"), (1, "spoof")):
                ax[k].hist(d[d.label == lab][cue].clip(upper=d[cue].quantile(.99)), bins=40, alpha=.6, density=True, label=nm)
            ax[k].set(title=f"{ds}: {cue} (s)"); ax[k].legend()
        fig.tight_layout(); paths.FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(paths.FIGURES / f"shortcut_{ds}.png", dpi=200); plt.close(fig)
    out = pd.DataFrame(rows); paths.TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.TABLES / "shortcut_auc.csv", index=False); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
