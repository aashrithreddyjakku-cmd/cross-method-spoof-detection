"""Task A.7: integrity check of a feature store. Run after every extraction.
  python scripts/check_features.py --feat mel64_w400 [--manifest data/manifests/master.csv]
Exit 1 if shape/coverage/finiteness is wrong or failed.csv exists.
"""
import argparse
import sys

import numpy as np

from spoofdet.common import paths
from spoofdet.common.io import read_manifest
from spoofdet.data.features import FeatureStore, n_frames

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True, help="folder name under data/features")
ap.add_argument("--manifest", nargs="*", default=None,
                help="CSV(s) the store must cover (default: master.csv for the full store, nothing otherwise)")
ap.add_argument("--every", type=int, default=200, help="check every k-th row for values")
a = ap.parse_args()

root = paths.FEATURES / a.feat
fs = FeatureStore(root); arr = fs.arr
cfg = fs.cfg
bad = []
print(f"{root}\n shape {arr.shape} dtype {arr.dtype}, index rows {len(fs)}")
if arr.dtype != np.float16: bad.append("dtype is not float16")
if arr.shape != (len(fs), cfg["n_mels"], n_frames(cfg)): bad.append(f"shape != (N, {cfg['n_mels']}, {n_frames(cfg)})")
if (root / "failed.csv").exists(): bad.append(f"failed.csv exists ({sum(1 for _ in open(root / 'failed.csv')) - 1} files)")

sub = np.asarray(arr[::max(1, a.every)], dtype=np.float32)
n_const = int((sub.std(axis=(1, 2)) == 0).sum())
print(f" sampled {len(sub)} rows: finite={np.isfinite(sub).all()} min={sub.min():.1f} max={sub.max():.1f} dB, "
      f"constant clips={n_const}")
if not np.isfinite(sub).all(): bad.append("non-finite values")
if n_const > max(1, 0.001 * len(sub)): bad.append(f"{n_const} constant (blank) clips in the sample")

manifests = a.manifest if a.manifest is not None else ([str(paths.MANIFESTS / "master.csv")] if "__" not in a.feat and a.feat == cfg["feat_id"] else [])
for m in manifests:
    ids = set(read_manifest(m).utt_id)
    miss = [u for u in ids if u not in fs]
    print(f" coverage of {m}: {len(ids) - len(miss)}/{len(ids)}")
    if miss: bad.append(f"{len(miss)} utt_ids of {m} missing, e.g. {miss[:3]}")

print("\nFEATURE STORE OK" if not bad else "\n" + "\n".join("FAIL: " + b for b in bad))
sys.exit(1 if bad else 0)
