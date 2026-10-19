"""Build data/manifests/master.csv from whatever raw datasets are present.

Usage:
  python scripts/build_manifest.py                      # la19 (+ itw if present)
  python scripts/build_manifest.py --la21-meta asvspoof2021/keys/LA/CM/trial_metadata.txt
"""
import argparse
from pathlib import Path

import pandas as pd

from spoofdet.common import paths
from spoofdet.data import protocols as P

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default=str(paths.RAW))
ap.add_argument("--out", default=str(paths.MANIFESTS / "master.csv"))
ap.add_argument("--la-dir", default="asvspoof2019/LA")
ap.add_argument("--itw-dir", default="in_the_wild")
ap.add_argument("--la21-meta", default=None, help="path to LA trial_metadata.txt (relative to --raw or absolute)")
ap.add_argument("--df21-meta", default=None, help="path to DF trial_metadata.txt")
ap.add_argument("--la21-audio", default="asvspoof2021/ASVspoof2021_LA_eval/flac")
ap.add_argument("--df21-audio", default="asvspoof2021/ASVspoof2021_DF_eval/flac")
ap.add_argument("--no-audio-info", action="store_true", help="skip reading audio headers (faster)")
a = ap.parse_args()

raw = Path(a.raw)
parts = []
if (raw / a.la_dir).exists():
    parts.append(P.parse_la19(raw, a.la_dir)); print("la19 rows:", len(parts[-1]))
else:
    raise SystemExit(f"ASVspoof2019 LA not found at {raw/a.la_dir}")
if (raw / a.itw_dir).exists():
    parts.append(P.parse_itw(raw, a.itw_dir)); print("itw rows:", len(parts[-1]))
else:
    print("[info] In-the-Wild not found, skipping")
if a.la21_meta:
    parts.append(P.parse_2021(raw / a.la21_meta, "la21", a.la21_audio))
if a.df21_meta:
    parts.append(P.parse_2021(raw / a.df21_meta, "df21", a.df21_audio))

df = pd.concat(parts, ignore_index=True)
if not a.no_audio_info:
    df = P.add_audio_info(df, raw)
df = P.finalize(df)
assert df["utt_id"].is_unique, "utt_id must be unique"
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(df.groupby(["dataset", "partition", "label"]).size())
print("wrote", a.out)
