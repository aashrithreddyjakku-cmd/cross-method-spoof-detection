"""Explainability analysis plan (playbook C.6) for one held-out fold model.
  python scripts/xai_analysis.py --exp lofo_waveform_modification_lcnn_mel64_w400_s1 [--n 40]

Compares seen methods, the held-out method/family and bona fide clips per frequency band, and runs three sanity checks:
  (a) randomising the last layers must change the Grad-CAM map (Adebayo et al., 2018);
  (b) Grad-CAM and occlusion should highlight similar bands;
  (c) the bona fide profile should not simply be the average spectrum.
Writes results/tables/xai_compare_<exp>.csv, results/tables/xai_sanity.csv and results/figures/xai_compare_<exp>.png.
"""
import argparse
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from spoofdet.common import paths
from spoofdet.common.io import load_yaml, read_manifest
from spoofdet.data.dataset import normalize
from spoofdet.data.features import FeatureStore
from spoofdet.eval.gradcam import GradCAM, band_edges_hz, band_occlusion
from spoofdet.models import load_checkpoint


def corr(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def randomise_last_layers(model, seed=0):
    torch.manual_seed(seed)
    m = copy.deepcopy(model)
    parts = [m.gradcam_layer] + [getattr(m, n) for n in ("head", "fc") if hasattr(m, n)]
    for part in parts:
        for layer in part.modules():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    return m.eval()


def run_group(model, ck, store, ids, bands, batch=32):
    """Returns raw mels, CAMs, scores and per-band occlusion drops."""
    mels, cams, scores, drops = [], [], [], []
    cam_fn = GradCAM(model)
    for i in range(0, len(ids), batch):
        raw = [store.get(u) for u in ids[i:i + batch]]
        x = torch.from_numpy(np.stack([normalize(r, ck["norm"], ck["norm_stats"]) for r in raw]).astype(np.float32))[:, None]
        c, s = cam_fn(x, target_class=1)
        mels += raw; cams.append(c); scores.append(s); drops.append(band_occlusion(model, x, bands))
    cam_fn.close()
    return np.stack(mels), np.concatenate(cams), np.concatenate(scores), np.concatenate(drops)


def by_band(cams, bands):
    edges = np.linspace(0, cams.shape[1], bands + 1, dtype=int)
    return np.array([cams[:, edges[i]:edges[i + 1], :].mean() for i in range(bands)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--n", type=int, default=40, help="max clips per method (and bona fide)")
    ap.add_argument("--bands", type=int, default=8)
    a = ap.parse_args()
    cfg = load_yaml(paths.CKPT / a.exp / "config.yaml")
    if not cfg["protocol"].startswith(("P1_lomo", "P1_lofo")):
        raise SystemExit("xai_analysis needs a held-out fold model (P1_lomo or P1_lofo)")
    fold = paths.SPLITS / cfg["protocol"]
    model, ck = load_checkpoint(paths.CKPT / a.exp / "best.pt")
    store = FeatureStore(paths.FEATURES / ck["feat_cfg"]["feat_id"])
    seen, unseen = read_manifest(fold / "test_seen.csv"), read_manifest(fold / "test_unseen.csv")

    def sample(d):
        return pd.concat(g.sample(min(a.n, len(g)), random_state=0) for _, g in d.groupby("method_id")).utt_id.tolist()

    groups = {"seen spoof": sample(seen[seen.label == 1]), "held-out spoof": sample(unseen[unseen.label == 1]),
              "bona fide": sample(unseen[unseen.label == 0])}
    edges = band_edges_hz(ck["feat_cfg"], a.bands)
    res, rows = {}, []
    for name, ids in groups.items():
        res[name] = run_group(model, ck, store, ids, a.bands)
        cam, drop = by_band(res[name][1], a.bands), res[name][3].mean(0)
        for j, (lo, hi) in enumerate(edges):
            rows.append(dict(group=name, n=len(ids), mean_score=float(res[name][2].mean()), band=j, low_hz=round(lo), high_hz=round(hi),
                             gradcam=float(cam[j]), occlusion_drop=float(drop[j])))
    tbl = pd.DataFrame(rows); paths.TABLES.mkdir(parents=True, exist_ok=True); paths.FIGURES.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(paths.TABLES / f"xai_compare_{a.exp}.csv", index=False)

    # sanity checks
    spoof_ids = groups["held-out spoof"]
    _, trained_cam, _, _ = run_group(model, ck, store, spoof_ids, a.bands)
    _, random_cam, _, _ = run_group(randomise_last_layers(model), ck, store, spoof_ids, a.bands)
    rand_r = float(np.nanmean([corr(p, q) for p, q in zip(trained_cam, random_cam)]))
    g = {k: tbl[tbl.group == k] for k in groups}
    held, seenp, bona = g["held-out spoof"], g["seen spoof"], g["bona fide"]
    mel_bona = res["bona fide"][0]
    edges_i = np.linspace(0, mel_bona.shape[1], a.bands + 1, dtype=int)
    spectrum = [mel_bona[:, edges_i[i]:edges_i[i + 1], :].mean() for i in range(a.bands)]
    sanity = dict(exp=a.exp, randomisation_cam_corr=rand_r, gradcam_vs_occlusion_heldout=corr(held.gradcam, held.occlusion_drop),
                  gradcam_vs_occlusion_seen=corr(seenp.gradcam, seenp.occlusion_drop), bona_profile_vs_spectrum=corr(bona.gradcam, spectrum),
                  seen_vs_heldout_gradcam=corr(seenp.gradcam, held.gradcam), mean_score_seen=float(seenp.mean_score.iloc[0]),
                  mean_score_heldout=float(held.mean_score.iloc[0]), mean_score_bona=float(bona.mean_score.iloc[0]))
    out = paths.TABLES / "xai_sanity.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    old = old[old.exp != a.exp] if len(old) else old
    pd.concat([old, pd.DataFrame([sanity])], ignore_index=True).to_csv(out, index=False)

    labels = [f"{lo / 1000:.1f}-{hi / 1000:.1f}" for lo, hi in edges]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, (name, marker) in enumerate((("seen spoof", "o"), ("held-out spoof", "s"), ("bona fide", "^"))):
        d = g[name]
        ax[0].plot(labels, d.gradcam, marker=marker, label=name); ax[1].plot(labels, d.occlusion_drop, marker=marker, label=name)
    ax[0].set(title="Mean Grad-CAM per band", ylabel="normalised intensity", xlabel="band (kHz)")
    ax[1].set(title="Band occlusion", ylabel="decrease in P(spoof)", xlabel="band (kHz)"); ax[1].axhline(0, color="k", lw=.6)
    for x in ax:
        x.tick_params(axis="x", labelsize=7, rotation=30); x.legend(fontsize=7)
    fig.suptitle(a.exp, fontsize=9); fig.tight_layout(); fig.savefig(paths.FIGURES / f"xai_compare_{a.exp}.png", dpi=200); plt.close(fig)
    print(pd.Series(sanity).to_string())


if __name__ == "__main__":
    main()
