"""Run many experiment configs one after another; finished ones (marker results/scores/<exp>/DONE) are skipped,
so after a Kaggle/Colab timeout you simply start the same command again.
   python scripts/run_queue.py "configs/experiments/lomo_*.yaml" [--dry-run] [--extra train.epochs=2 ...]
"""
import argparse
import glob
import subprocess
import sys
import time

from spoofdet.common import paths
from spoofdet.common.io import load_yaml

ap = argparse.ArgumentParser()
ap.add_argument("pattern", help="glob of experiment yaml files (quote it)")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--extra", nargs="*", default=[], help="overrides passed to every run, e.g. train.epochs=2")
a = ap.parse_args()
files = sorted(glob.glob(a.pattern))
todo = [f for f in files if not (paths.SCORES / load_yaml(f)["exp_id"] / "DONE").exists()]
print(f"{len(files)} configs match, {len(files) - len(todo)} already done, {len(todo)} to run")
for i, f in enumerate(todo, 1):
    exp = load_yaml(f)["exp_id"]
    print(f"[{i}/{len(todo)}] {exp}", flush=True)
    if a.dry_run:
        continue
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "spoofdet.train.run_experiment", f] + a.extra)
    print(f"    {'ok' if r.returncode == 0 else 'FAILED'} in {(time.time()-t0)/60:.1f} min", flush=True)
