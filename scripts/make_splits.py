"""Create every split file used in the project from data/manifests/master.csv.

Outputs (all under splits/):
  P0_official/{train,val,test}.csv           la19 train / dev / eval  (standard protocol)
  P0_official/test_itw.csv                   all In-the-Wild          (cross-dataset)
  P0_official/test_la21.csv, test_df21.csv   (if 2021 data in manifest; subsampled)
  P1_all/{train,val,test}.csv                speaker-disjoint, ALL methods seen (reference)
  P1_lomo/<method>/{train,val,test_seen,test_unseen}.csv   leave-one-METHOD-out (17 folds)
  P1_lofo/<family>/{train,val,test_seen,test_unseen}.csv   leave-one-FAMILY-out
  split_summary.csv                          counts per file / label / method
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spoofdet.common import paths
from spoofdet.common.io import read_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(paths.MANIFESTS / "master.csv"))
ap.add_argument("--out", default=str(paths.SPLITS))
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--frac", type=float, nargs=3, default=[0.60, 0.15, 0.25], help="speaker fractions train/val/test")
ap.add_argument("--cap", type=int, nargs=3, default=[1500, 300, 600],
                help="max spoof utts PER METHOD in train/val/test groups (P1 only)")
ap.add_argument("--cap-bona", type=int, nargs=3, default=[0, 0, 0], help="0 = keep all bona fide")
ap.add_argument("--max-2021", type=int, default=10000, help="subsample size for la21/df21 test files")
a = ap.parse_args()

out = Path(a.out)
df = read_manifest(a.manifest)
rng = np.random.RandomState(a.seed)


def save(d, rel):
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(p, index=False)
    return p


# ------------------------------------------------------------------ P0
la = df[df.dataset == "la19"]
for part, name in (("train", "train"), ("dev", "val"), ("eval", "test")):
    save(la[la.partition == part], f"P0_official/{name}.csv")
for ds, fn in (("itw", "test_itw.csv"), ("la21", "test_la21.csv"), ("df21", "test_df21.csv")):
    d = df[df.dataset == ds]
    if len(d):
        if ds != "itw" and len(d) > a.max_2021:
            d = d.sample(a.max_2021, random_state=a.seed)
        save(d, f"P0_official/{fn}")

# ------------------------------------------------------------------ P1 speaker groups
# Speakers of train+dev partitions form pool "known"; eval speakers form pool "eval".
first_part = la.groupby("speaker_id")["partition"].agg(lambda s: "eval" if (s == "eval").all() else "known")
group_of = {}
for pool in ("known", "eval"):
    spk = sorted(first_part[first_part == pool].index)
    rng.shuffle(spk)
    n = len(spk)
    n_tr = max(1, int(round(a.frac[0] * n)))
    n_va = max(1, int(round(a.frac[1] * n)))
    for i, s in enumerate(spk):
        group_of[s] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
la = la.copy()
la["group"] = la["speaker_id"].map(group_of)
print(la.groupby(["group"])["speaker_id"].nunique().rename("n_speakers"))

# Cap spoof utterances per (method, group); optional cap on bona fide.
parts = []
for g, cap, capb in zip(("train", "val", "test"), a.cap, a.cap_bona):
    gd = la[la.group == g]
    bona = gd[gd.label == 0]
    if capb and len(bona) > capb:
        bona = bona.sample(capb, random_state=a.seed)
    parts.append(bona)
    for m, md in gd[gd.label == 1].groupby("method_id"):
        parts.append(md.sample(min(cap, len(md)), random_state=a.seed))
pool = pd.concat(parts, ignore_index=True)
tr, va, te = (pool[pool.group == g].drop(columns="group") for g in ("train", "val", "test"))
save(tr, "P1_all/train.csv"); save(va, "P1_all/val.csv"); save(te, "P1_all/test.csv")


def make_fold(rel, held_mask):
    for name, d in (("train", tr), ("val", va)):
        save(d[~held_mask(d)], f"{rel}/{name}.csv")
    save(te[~held_mask(te)], f"{rel}/test_seen.csv")
    unseen = te[(te.label == 0) | held_mask(te)]           # bona fide + held-out spoofs
    save(unseen, f"{rel}/test_unseen.csv")


methods = sorted(m for m in pool.method_id.unique() if m != "bonafide")
for m in methods:
    make_fold(f"P1_lomo/{m}", lambda d, m=m: d.method_id == m)
fams = sorted(f for f in pool.method_family.unique() if f not in ("bonafide", "unknown"))
for f in fams:
    make_fold(f"P1_lofo/{f}", lambda d, f=f: d.method_family == f)

# ------------------------------------------------------------------ summary
rows = []
for p in sorted(out.rglob("*.csv")):
    if p.name == "split_summary.csv":
        continue
    d = pd.read_csv(p)
    rows.append(dict(file=p.relative_to(out).as_posix(), n=len(d), bonafide=int((d.label == 0).sum()),
                     spoof=int((d.label == 1).sum()), n_speakers=d.speaker_id.nunique(),
                     methods=";".join(sorted(d[d.label == 1].method_id.unique()))))
pd.DataFrame(rows).to_csv(out / "split_summary.csv", index=False)
print(pd.DataFrame(rows).drop(columns="methods").to_string(index=False))
