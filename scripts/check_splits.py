"""Validate every split file. Exit code 1 if any rule is broken.   python scripts/check_splits.py

Rules (Part I §5.4): (i) no speaker shared between train/val/test of a P1 fold; (ii) the held-out
method/family is absent from train and val; (iii) test_seen and test_unseen use the same test speakers and
the same bona fide utterances; (iv) every fold is cut from the same capped pool (P1_all).
"""
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


all_tr, all_va, all_te = (read_manifest(S / "P1_all" / f"{n}.csv") for n in ("train", "val", "test"))

# ---- P1: speaker-disjoint train/val/test; no utterance overlap
for folder in sorted([p for p in (S / "P1_lomo").glob("*")] + [p for p in (S / "P1_lofo").glob("*")] + [S / "P1_all"]):
    if not folder.is_dir():
        continue
    tr, va = read_manifest(folder / "train.csv"), read_manifest(folder / "val.csv")
    te_names = ["test.csv"] if (folder / "test.csv").exists() else ["test_seen.csv", "test_unseen.csv"]
    for te_name in te_names:
        te = read_manifest(folder / te_name)
        tag = f"{folder.parent.name}/{folder.name}/{te_name}"
        ck(not (set(tr.speaker_id) & set(te.speaker_id)), f"{tag}: train/test share speakers")
        ck(not (set(va.speaker_id) & set(te.speaker_id)), f"{tag}: val/test share speakers")
        ck(not (set(tr.utt_id) & set(te.utt_id)) and not (set(va.utt_id) & set(te.utt_id)), f"{tag}: utterance overlap")
        ck(te.label.nunique() == 2, f"{tag}: test lacks one class")
    ck(not (set(tr.speaker_id) & set(va.speaker_id)), f"{folder.name}: train/val share speakers")
    ck(not (set(tr.utt_id) & set(va.utt_id)), f"{folder.name}: train/val utterance overlap")
    ck(tr.label.nunique() == 2 and va.label.nunique() == 2, f"{folder.name}: train or val lacks one class")
    ck(set(tr.dataset) == {"la19"} and set(va.dataset) == {"la19"}, f"{folder.name}: P1 must use la19 only")
    if folder.parent.name in ("P1_lomo", "P1_lofo"):
        col = "method_id" if folder.parent.name == "P1_lomo" else "method_family"
        held = folder.name
        tag = f"{folder.parent.name}/{held}"
        ck(held not in set(tr[col]) and held not in set(va[col]), f"{tag}: held-out leaks into train/val")
        un = read_manifest(folder / "test_unseen.csv")
        ck(set(un[un.label == 1][col]) == {held}, f"{tag}: test_unseen spoofs are not exactly the held-out group")
        se = read_manifest(folder / "test_seen.csv")
        ck(held not in set(se[col]), f"{tag}: held-out appears in test_seen")
        # (iii) same bona fide utterances (hence same test speakers for the bona fide side) in seen/unseen
        ck(set(se[se.label == 0].utt_id) == set(un[un.label == 0].utt_id), f"{tag}: test_seen/test_unseen bona fide differ")
        # (iv) fold = P1_all minus the held-out rows (same capped pool for every fold)
        for name, d, ref in (("train", tr, all_tr), ("val", va, all_va)):
            ck(set(d.utt_id) == set(ref[ref[col] != held].utt_id), f"{tag}: {name} is not P1_all/{name} minus held-out")
        ck(set(se.utt_id) | set(un.utt_id) == set(all_te.utt_id), f"{tag}: seen+unseen do not cover P1_all/test")

# ---- P0 official
for n, part in (("train", "train"), ("val", "dev"), ("test", "eval")):
    d = read_manifest(S / "P0_official" / f"{n}.csv")
    ck(set(d.partition) == {part} and set(d.dataset) == {"la19"}, f"P0_official/{n}: wrong partition")
    ck(d.label.nunique() == 2, f"P0_official/{n}: lacks one class")
ck(not (spk(S / "P0_official/train.csv") & spk(S / "P0_official/test.csv")), "P0: train/eval speakers overlap")
ck(not (spk(S / "P0_official/train.csv") & spk(S / "P0_official/val.csv")), "P0: train/dev speakers overlap")
for fn in ("test_itw.csv", "test_la21.csv", "test_df21.csv"):
    p = S / "P0_official" / fn
    if p.exists():
        d = read_manifest(p)
        ck(d.dataset.nunique() == 1 and d.label.nunique() == 2, f"P0_official/{fn}: mixed datasets or one class")
print("\nALL SPLIT CHECKS PASSED" if not bad else f"\n{len(bad)} PROBLEM(S) FOUND")
sys.exit(1 if bad else 0)
