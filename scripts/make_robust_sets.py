"""Stretch S1: build degraded copies of a test split.
  python scripts/make_robust_sets.py --split splits/P0_official/test.csv --n 3000 --conds noise20 noise10 tel8k mp3_64k
Creates data/raw/robust/<cond>/*.wav, data/manifests/robust_<cond>.csv and splits/robust/<cond>.csv.
Then:  python scripts/extract_features.py --feat mel64_w400 --out-name mel64_w400__robust --manifest data/manifests/robust_*.csv
and score with:  python -m spoofdet.train.predict --exp <exp> --feat mel64_w400__robust --split splits/robust/<cond>.csv --name robust_<cond>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from spoofdet.common import paths
from spoofdet.common.io import read_manifest
from spoofdet.data.audio import load_wave
from spoofdet.data.degrade import CONDITIONS

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--conds", nargs="+", default=["noise20", "noise10", "tel8k"])
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()

df = read_manifest(a.split)
if len(df) > a.n:
    df = df.sample(a.n, random_state=a.seed)
rng = np.random.RandomState(a.seed)
for cond in a.conds:
    out_dir = paths.RAW / "robust" / cond; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=cond):
        y = load_wave(paths.RAW / r["path"])
        y2 = CONDITIONS[cond](y, 16000, rng)
        fn = f"{r['utt_id']}__{cond}.wav"
        sf.write(out_dir / fn, y2, 16000)
        r = r.copy(); r["utt_id"] = f"{r['utt_id']}__{cond}"; r["path"] = f"robust/{cond}/{fn}"
        rows.append(r)
    m = pd.DataFrame(rows)
    paths.MANIFESTS.mkdir(parents=True, exist_ok=True)
    m.to_csv(paths.MANIFESTS / f"robust_{cond}.csv", index=False)
    (paths.SPLITS / "robust").mkdir(parents=True, exist_ok=True)
    m.to_csv(paths.SPLITS / "robust" / f"{cond}.csv", index=False)
    print(cond, len(m), "files")
