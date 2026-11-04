"""Morning check before the demo: is the model loadable, do the clips give their expected verdicts, are the figures there?
  python scripts/demo_check.py      (or double-click check_demo.bat)
"""
import sys

import pandas as pd

from spoofdet.common import paths
from spoofdet.infer import Detector

EXP = "p0_lcnn_mel64_w400_s1"
BACKUP = paths.DATA / "backup_2026-10-20" / "checkpoints" / EXP / "best.pt"
FIGURES = ["roc_det_p0_test.png", "roc_det_p0_itw.png", "gap_p1_lofo.png", "lofo_heatmap.png"]
bad = []


def line(ok, text):
    print(("OK      " if ok else "PROBLEM ") + text)
    if not ok:
        bad.append(text)


new = (paths.SCORES / EXP / "DONE").exists()
line(new, f"new trained demo model finished ({EXP})" if new else "new demo model not finished: the launcher will use the backup model")
ckpt = paths.CKPT / EXP / "best.pt" if new else BACKUP
det = Detector(ckpt)
print(f"        model file: {ckpt}  (best validation EER {100 * det.ck['val_eer']:.2f}%)")

clips = paths.DATA / "demo_clips"
t = pd.read_csv(clips / "clips.csv")
thr = float((clips / "threshold.txt").read_text())
print(f"        threshold {thr:.3f}")
for r in t.itertuples():
    s = det.score(clips / r.clip)
    verdict = "SPOOF" if s >= thr else "BONA FIDE"
    line(verdict == r.verdict, f"{r.clip}: {verdict} at {s:.3f} (expected {r.verdict})")
for f in FIGURES:
    line((paths.FIGURES / f).exists(), f"figure {f}")
status = paths.ROOT / "logs" / "programme" / "status.csv"
if status.exists():
    st = pd.read_csv(status)
    print(f"        training log: {st.status.value_counts().to_dict()}")
print("\nREADY FOR THE DEMO" if not bad else f"\n{len(bad)} PROBLEM(S): see the lines above and docs/DEMO_DAY.md")
sys.exit(1 if bad else 0)
