"""Pick demo clips from real test scores and check that live inference reproduces them.
  python scripts/prepare_demo.py [--exp p0_lcnn_mel64_w400_s1] [--out <folder>]
Copies audio to <SPOOFDET_DATA>/demo_clips (never into git: dataset licence) and writes clips.csv + threshold.txt there.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from spoofdet.common import paths
from spoofdet.common.metrics_core import compute_eer
from spoofdet.infer import Detector


def nearest(df, target, n=1):
    return df.iloc[(df.score - target).abs().argsort()[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="p0_lcnn_mel64_w400_s1")
    ap.add_argument("--out", default=str(paths.DATA / "demo_clips"))
    a = ap.parse_args()
    sc = paths.SCORES / a.exp
    thr = compute_eer(*(lambda v: (v.score, v.label))(pd.read_csv(sc / "val.csv")))[1]
    man = pd.read_csv(paths.MANIFESTS / "master.csv", dtype={"speaker_id": str}).set_index("utt_id")
    te = pd.read_csv(sc / "test.csv"); bona = te[te.label == 0]; spoof = te[te.label == 1]
    picks = [
        ("1_real_correct", nearest(bona[bona.score < thr], 0.02)),
        ("2_tts_griffinlim_A11_correct", nearest(spoof[(spoof.attack_id == "A11") & (spoof.score >= thr)], 0.98)),
        ("3_tts_neural_A10_correct", nearest(spoof[(spoof.attack_id == "A10") & (spoof.score >= thr)], 0.98)),
        ("4_vc_A17_MISSED", nearest(spoof[(spoof.attack_id == "A17") & (spoof.score < thr)], 0.02)),
    ]
    itw = sc / "itw.csv"
    if itw.exists():
        d = pd.read_csv(itw)
        picks.append(("5_inthewild_real_FALSE_ALARM", nearest(d[(d.label == 0) & (d.score >= thr)], 0.98)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    det = Detector(paths.CKPT / a.exp / "best.pt")
    rows = []
    for name, r in picks:
        r = r.iloc[0]; src = paths.RAW / man.loc[r.utt_id, "path"]
        dst = out / f"{name}{src.suffix}"; shutil.copy2(src, dst)
        live = det.score(dst)
        rows.append(dict(clip=dst.name, utt_id=r.utt_id, label=int(r.label), attack=r.attack_id, stored_score=round(float(r.score), 4),
                         live_score=round(live, 4), verdict="SPOOF" if live >= thr else "BONA FIDE"))
    df = pd.DataFrame(rows); df.to_csv(out / "clips.csv", index=False)
    (out / "threshold.txt").write_text(f"{thr:.4f}\n")
    print(df.to_string(index=False)); print(f"val EER threshold = {thr:.4f}\n->", out)
    bad = df[(df.stored_score - df.live_score).abs() > 0.01]
    print("LIVE INFERENCE MATCHES STORED SCORES" if bad.empty else f"MISMATCH on {len(bad)} clips")


if __name__ == "__main__":
    main()
