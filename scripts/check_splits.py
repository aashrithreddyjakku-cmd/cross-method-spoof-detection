"""Validate every split file. Exit code 1 if any rule is broken.   python scripts/check_splits.py"""
import sys
import pandas as pd
from spoofdet.common import paths
from spoofdet.common.io import read_manifest

S = paths.SPLITS
bad = []


def ck(cond, msg):
    if not cond:
        bad.append(msg); print("FAIL:", msg)


def ids(p): return set(read_manifest(p).utt_id)
def spk(p, lab=None):
    d = read_manifest(p); return set(d.speaker_id)


# ---- P1: speaker-disjoint train/val/test; no utterance overlap
for folder in sorted([p for p in (S / "P1_lomo").glob("*")] + [p for p in (S / "P1_lofo").glob("*")] + [S / "P1_all"]):
    if not folder.is_dir():
        continue
    tr, va = read_manifest(folder / "train.csv"), read_manifest(folder / "val.csv")
    te_name = "test.csv" if (folder / "test.csv").exists() else "test_unseen.csv"
    te = read_manifest(folder / te_name)
    ck(not (set(tr.speaker_id) & set(va.speaker_id)), f"{folder.name}: train/val share speakers")
    ck(not (set(tr.speaker_id) & set(te.speaker_id)), f"{folder.name}: train/test share speakers")
    ck(not (set(va.speaker_id) & set(te.speaker_id)), f"{folder.name}: val/test share speakers")
    ck(not (set(tr.utt_id) & set(te.utt_id)) and not (set(tr.utt_id) & set(va.utt_id)), f"{folder.name}: utterance overlap")
    ck(tr.label.nunique() == 2 and va.label.nunique() == 2 and te.label.nunique() == 2, f"{folder.name}: a split lacks one class")
    if folder.parent.name in ("P1_lomo", "P1_lofo"):
        col = "method_id" if folder.parent.name == "P1_lomo" else "method_family"
        held = folder.name
        ck(held not in set(tr[col]) and held not in set(va[col]), f"{folder.parent.name}/{held}: held-out leaks into train/val")
        un = read_manifest(folder / "test_unseen.csv")
        ck(set(un[un.label == 1][col]) == {held}, f"{folder.parent.name}/{held}: test_unseen spoofs are not exactly the held-out group")
        se = read_manifest(folder / "test_seen.csv")
        ck(held not in set(se[col]), f"{folder.parent.name}/{held}: held-out appears in test_seen")

# ---- P0 official
for n, part in (("train", "train"), ("val", "dev"), ("test", "eval")):
    d = read_manifest(S / "P0_official" / f"{n}.csv")
    ck(set(d.partition) == {part} and set(d.dataset) == {"la19"}, f"P0_official/{n}: wrong partition")
ck(not (spk(S / "P0_official/train.csv") & spk(S / "P0_official/test.csv")), "P0: train/eval speakers overlap")
print("\nALL SPLIT CHECKS PASSED" if not bad else f"\n{len(bad)} PROBLEM(S) FOUND")
sys.exit(1 if bad else 0)
