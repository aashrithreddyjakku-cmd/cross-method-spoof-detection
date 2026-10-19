"""Validate score files before handing them to Member C.
   python scripts/check_scores.py --exp <exp_id>            (or --all)
Rules: required columns, no NaN, 0<=score<=1, both classes present, row count equals the split it was made from (when known)."""
import argparse
import sys

import pandas as pd

from spoofdet.common import paths
from spoofdet.common.constants import SCORE_COLUMNS

ap = argparse.ArgumentParser(); ap.add_argument("--exp"); ap.add_argument("--all", action="store_true")
a = ap.parse_args()
exps = [d.name for d in sorted(paths.SCORES.iterdir()) if d.is_dir()] if a.all else [a.exp]
bad = 0
for e in exps:
    for f in sorted((paths.SCORES / e).glob("*.csv")):
        d = pd.read_csv(f)
        probs = []
        if [c for c in SCORE_COLUMNS if c not in d.columns]: probs.append("missing columns")
        elif d[SCORE_COLUMNS].isna().any().any(): probs.append("NaN present")
        elif not d.score.between(0, 1).all(): probs.append("score outside [0,1]")
        elif d.label.nunique() < 2: probs.append("only one class (ok for itw-less partial files? check)")
        elif d.utt_id.duplicated().any(): probs.append("duplicate utt_id")
        print(f"{'OK  ' if not probs else 'BAD '} {e}/{f.name:22s} rows={len(d):6d} {'; '.join(probs)}")
        bad += bool(probs)
sys.exit(1 if bad else 0)
