"""Explain a fold-specific checkpoint on its unseen spoofing methods.

Example::

    python scripts/run_xai.py --exp lomo_A10_lcnn_mel64_w400_s1 \
        --split-csv splits/P1_lomo/A10/test_unseen.csv

Writes numerical summaries to ``results/tables/`` and report-ready figures to
``results/figures/``. The runner intentionally accepts only P1 test_unseen folds.
"""

import argparse
from pathlib import Path

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
from spoofdet.eval.gradcam import GradCAM, band_occlusion, band_edges_hz
from spoofdet.models import load_checkpoint


def validate_unseen_split(split_csv, frame):
    """Require a generated LOMO/LOFO ``test_unseen.csv`` with matching labels."""
    path = Path(split_csv)
    parts = path.parts
    if path.name != "test_unseen.csv" or len(parts) < 3 or parts[-3] not in ("P1_lomo", "P1_lofo"):
        raise ValueError("XAI requires splits/P1_lomo/<method>/test_unseen.csv or "
                         "splits/P1_lofo/<family>/test_unseen.csv")
    kind, held_out = parts[-3], parts[-2]
    if frame.empty:
        raise ValueError("unseen test split is empty")
    labels = set(pd.to_numeric(frame.label, errors="coerce").dropna().astype(int).unique())
    if labels != {0, 1}:
        raise ValueError("unseen test split must contain bona fide and spoof samples (labels 0 and 1)")
    if not frame.label.isin([0, 1]).all():
        raise ValueError("unseen test split contains labels outside the frozen 0/1 contract")
    bona = frame[frame.label == 0]
    spoof = frame[frame.label == 1]
    if not bona.method_id.astype(str).eq("bonafide").all():
        raise ValueError("label 0 rows must have method_id=bonafide")
    if kind == "P1_lomo":
        valid = spoof.method_id.astype(str).eq(held_out).all()
        column = "method_id"
    else:
        valid = spoof.method_family.astype(str).eq(held_out).all()
        column = "method_family"
    if not valid:
        raise ValueError(f"{kind} unseen spoof rows must all belong to held-out {column}={held_out}")
    return kind, held_out


def _checkpoint_config(exp_id):
    config_path = paths.CONFIGS / "experiments" / f"{exp_id}.yaml"
    if not config_path.exists():
        config_path = paths.CKPT / exp_id / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"experiment config is missing for {exp_id}; cannot verify its held-out fold")
    return load_yaml(config_path)


def _select_samples(frame, n, seed=0):
    if n < 1:
        raise ValueError("--n must be at least 1")
    # Keep an equal upper limit for each unseen spoof method, plus a bona fide
    # comparison group. Sorting makes output order reproducible.
    selected = [group.sample(min(n, len(group)), random_state=seed)
                for _, group in frame[frame.label == 1].groupby("method_id", sort=True)]
    bona = frame[frame.label == 0]
    if len(bona):
        selected.append(bona.sample(min(n, len(bona)), random_state=seed))
    return pd.concat(selected, ignore_index=True)


def _save_gradcam_overlay(mel, cam, output, exp_id, utt_id, method_id, score):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.imshow(mel, origin="lower", aspect="auto", cmap="gray_r")
    overlay = ax.imshow(cam, origin="lower", aspect="auto", cmap="magma", alpha=0.55,
                        vmin=0.15, vmax=1.0)
    ax.set(xlabel="Time frame", ylabel="Mel band",
           title=f"{exp_id} | {method_id} | {utt_id} | P(spoof)={score:.3f}")
    fig.colorbar(overlay, ax=ax, label="Grad-CAM intensity (normalized)")
    fig.tight_layout()
    fig.savefig(output, dpi=250)
    plt.close(fig)


def run_xai(exp_id, split_csv, n=100, bands=8, device="cpu"):
    """Run explainability and return paths to the generated table and figures."""
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise FileNotFoundError(f"unseen split CSV not found: {split_csv}")
    frame = read_manifest(split_csv)
    kind, held_out = validate_unseen_split(split_csv, frame)
    expected_protocol = f"{kind}/{held_out}"
    config = _checkpoint_config(exp_id)
    if config.get("protocol") != expected_protocol:
        raise ValueError(f"checkpoint {exp_id} was configured for {config.get('protocol')!r}, "
                         f"but input split is {expected_protocol!r}")

    checkpoint_path = paths.CKPT / exp_id / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best checkpoint not found for {exp_id}: {checkpoint_path}")
    selected = _select_samples(frame, n)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({dev}), but CUDA is unavailable")
    try:
        model, ck = load_checkpoint(checkpoint_path, str(dev))
    except (KeyError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"could not load checkpoint {checkpoint_path}: {exc}") from exc
    store = FeatureStore(paths.FEATURES / ck["feat_cfg"]["feat_id"])
    missing = [uid for uid in selected.utt_id if uid not in store]
    if missing:
        raise KeyError(f"{len(missing)} selected unseen-test utterances are missing from the feature store; "
                       f"examples: {missing[:3]}")
    if bands < 1 or bands > int(ck["feat_cfg"]["n_mels"]):
        raise ValueError(f"--bands must be between 1 and n_mels ({ck['feat_cfg']['n_mels']})")
    n_bands = bands

    paths.TABLES.mkdir(parents=True, exist_ok=True)
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    explainer = GradCAM(model)
    per_utterance, profile_rows, overlay_done = [], [], set()
    try:
        batch_size = 32
        for start in range(0, len(selected), batch_size):
            rows = selected.iloc[start:start + batch_size]
            normalized = [normalize(store.get(row.utt_id), ck["norm"], ck["norm_stats"])
                          for row in rows.itertuples()]
            xs = torch.from_numpy(np.stack(normalized).astype(np.float32))[:, None].to(dev)
            cams, scores = explainer(xs, target_class=1)
            occlusion = band_occlusion(model, xs, n_bands)
            for offset, row in enumerate(rows.itertuples()):
                record = {"utt_id": row.utt_id, "method_id": row.method_id,
                          "method_family": row.method_family, "label": int(row.label),
                          "base_score": float(scores[offset]),
                          "gradcam_mean": float(cams[offset].mean())}
                record.update({f"band{j}_drop": float(occlusion[offset, j]) for j in range(n_bands)})
                per_utterance.append(record)
                if int(row.label) == 1:
                    profile_rows.append((str(row.method_id), cams[offset].mean(axis=1)))
                    if row.method_id not in overlay_done:
                        out = paths.FIGURES / f"xai_{exp_id}_{row.method_id}_gradcam.png"
                        _save_gradcam_overlay(normalized[offset], cams[offset], out, exp_id,
                                              row.utt_id, row.method_id, scores[offset])
                        overlay_done.add(row.method_id)
    finally:
        explainer.close()

    raw = pd.DataFrame(per_utterance)
    per_utt_path = paths.TABLES / f"xai_{exp_id}_per_utterance.csv"
    raw.to_csv(per_utt_path, index=False)
    value_columns = ["base_score", "gradcam_mean", *[f"band{j}_drop" for j in range(n_bands)]]
    summary = raw.groupby(["method_id", "method_family", "label"], dropna=False)[value_columns].agg(["mean", "std", "count"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    summary_path = paths.TABLES / f"xai_{exp_id}_band_occlusion.csv"
    summary.to_csv(summary_path, index=False)

    # The profile is calculated from held-out spoof samples only.
    profiles = {}
    for method, values in profile_rows:
        profiles.setdefault(method, []).append(values)
    if profiles:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method, values in sorted(profiles.items()):
            ax.plot(np.mean(values, axis=0), label=method)
        ax.set(xlabel="Mel band (low to high frequency)", ylabel="Mean normalized Grad-CAM intensity",
               title=f"Unseen spoof methods | {exp_id}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(paths.FIGURES / f"xai_{exp_id}_gradcam_profile.png", dpi=250)
        plt.close(fig)

    spoof_summary = summary[summary.label == 1]
    band_means = np.column_stack([spoof_summary[f"band{j}_drop_mean"].to_numpy() for j in range(n_bands)])
    if len(spoof_summary):
        hz = band_edges_hz(ck["feat_cfg"], n_bands)
        limit = float(np.nanmax(np.abs(band_means)))
        limit = limit if np.isfinite(limit) and limit > 0 else 1e-8
        fig, ax = plt.subplots(figsize=(7.2, max(2.5, 0.42 * len(spoof_summary) + 1)))
        im = ax.imshow(band_means, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set_yticks(range(len(spoof_summary)))
        ax.set_yticklabels(spoof_summary.method_id.astype(str))
        ax.set_xticks(range(n_bands))
        ax.set_xticklabels([f"{lo / 1000:.1f}-{hi / 1000:.1f}" for lo, hi in hz], rotation=35, ha="right")
        ax.set(xlabel="Frequency band (kHz)", ylabel="Held-out spoofing method",
               title=f"Mel-band occlusion on unseen spoof samples | {exp_id}")
        fig.colorbar(im, ax=ax, label="Decrease in P(spoof) after masking band")
        fig.tight_layout()
        fig.savefig(paths.FIGURES / f"xai_{exp_id}_band_occlusion.png", dpi=250)
        plt.close(fig)

    outputs = [per_utt_path, summary_path,
               paths.FIGURES / f"xai_{exp_id}_gradcam_profile.png",
               paths.FIGURES / f"xai_{exp_id}_band_occlusion.png"]
    outputs.extend(paths.FIGURES / f"xai_{exp_id}_{m}_gradcam.png" for m in sorted(overlay_done))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", required=True, help="fold-specific experiment/checkpoint id")
    parser.add_argument("--split-csv", required=True, help="P1_lomo/<method>/test_unseen.csv or P1_lofo/<family>/test_unseen.csv")
    parser.add_argument("--n", type=int, default=100, help="maximum held-out spoof utterances per method (and bona fide controls)")
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda:0")
    args = parser.parse_args()
    for output in run_xai(args.exp, args.split_csv, args.n, args.bands, args.device):
        print("saved", output)


if __name__ == "__main__":
    main()
