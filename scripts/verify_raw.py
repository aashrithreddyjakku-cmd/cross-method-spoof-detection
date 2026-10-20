"""Task A.3: verify the raw downloads (file counts, protocol files, In-the-Wild meta.csv).
  python scripts/verify_raw.py            # real data: compares with the official counts, exit 1 on mismatch
  python scripts/verify_raw.py --dummy    # dummy data: only prints what was found
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from spoofdet.common import paths
from spoofdet.data.protocols import LA19_PARTS

EXPECTED_LA19 = {"train": (25380, 2580, 22800), "dev": (24844, 2548, 22296), "eval": (71237, 7355, 63882)}

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(paths.RAW))
ap.add_argument("--la-dir", default="asvspoof2019/LA")
ap.add_argument("--itw-dir", default="in_the_wild")
ap.add_argument("--dummy", "--subset", dest="dummy", action="store_true",
                help="dummy data or a download_subset.py subset: do not compare with the official counts")
a = ap.parse_args()
raw = Path(a.raw)
bad = []

print("== ASVspoof 2019 LA ==")
base = raw / a.la_dir
for part, (audio_dir, proto) in LA19_PARTS.items():
    flac = base / audio_dir / "flac"
    pfile = base / "ASVspoof2019_LA_cm_protocols" / proto
    if not flac.is_dir() or not pfile.exists():
        bad.append(part); print(f"{part:5s}: MISSING {flac if not flac.is_dir() else pfile}")
        continue
    n_files = sum(1 for _ in flac.glob("*.flac"))
    lines = [l.split() for l in open(pfile) if l.strip()]
    n_bona = sum(l[4] == "bonafide" for l in lines); n_spoof = sum(l[4] == "spoof" for l in lines)
    on_disk = {p.stem for p in flac.glob("*.flac")}
    missing = sum(l[1] not in on_disk for l in lines)
    attacks = sorted({l[3] for l in lines if l[4] == "spoof"})
    msg = f"{part:5s}: {n_files} flac, protocol {len(lines)} lines ({n_bona} bona fide / {n_spoof} spoof), " \
          f"{missing} listed but missing, attacks {attacks[0]}..{attacks[-1]} ({len(attacks)})"
    ok = missing == 0
    if not a.dummy:
        ok = ok and (n_files, n_bona, n_spoof) == EXPECTED_LA19[part]
        msg += "" if ok else f"   EXPECTED {EXPECTED_LA19[part]}"
    print(("OK   " if ok else "BAD  ") + msg)
    if not ok:
        bad.append(part)

print("\n== In-the-Wild ==")
metas = list((raw / a.itw_dir).rglob("meta.csv")) if (raw / a.itw_dir).exists() else []
if not metas:
    print("not found (optional until CP2)")
else:
    meta = pd.read_csv(metas[0])
    print(f"meta.csv: {metas[0]}  columns={list(meta.columns)}")
    lcol = next(c for c in meta.columns if c.lower().strip() in ("label", "class", "key"))
    fcol = next(c for c in meta.columns if c.lower().strip() in ("file", "path", "filename"))
    print(meta[lcol].value_counts().to_string())
    missing = sum(not (metas[0].parent / f).exists() for f in meta[fcol].astype(str))
    n_wav = sum(1 for _ in metas[0].parent.glob("*.wav"))
    spk = next((c for c in meta.columns if "speaker" in c.lower()), None)
    print(f"{n_wav} wav files next to meta.csv, {len(meta)} rows, {missing} listed but missing, "
          f"{meta[spk].nunique() if spk else '?'} speakers")
    if missing or meta[lcol].nunique() != 2:
        bad.append("itw")
    elif not a.dummy and not (30000 <= len(meta) <= 33000):
        print("WARNING: expected about 31.8 k clips")

print("\nRAW DATA OK" if not bad else f"\nPROBLEMS: {bad}")
sys.exit(1 if bad else 0)
