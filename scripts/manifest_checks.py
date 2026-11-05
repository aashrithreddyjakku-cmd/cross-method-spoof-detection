"""Task A.4: the manifest sanity checks, saved for the data card.
  python scripts/manifest_checks.py [--dummy]
Writes results/tables/data_manifest_checks.txt. Exit 1 if a hard rule is broken.
"""
import argparse
import io
import sys
from contextlib import redirect_stdout

from spoofdet.common import paths
from spoofdet.common.io import read_manifest

EXPECTED = {("la19", "train"): (2580, 22800), ("la19", "dev"): (2548, 22296), ("la19", "eval"): (7355, 63882)}

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(paths.MANIFESTS / "master.csv"))
ap.add_argument("--dummy", "--subset", dest="dummy", action="store_true",
                help="dummy data or a download_subset.py subset: skip the comparison with the official counts")
a = ap.parse_args()

df = read_manifest(a.manifest)
bad = []
buf = io.StringIO()
with redirect_stdout(buf):
    print("rows:", len(df), " unique utt_id:", df.utt_id.is_unique)
    print("\n-- counts per dataset/partition/label\n", df.groupby(["dataset", "partition", "label"]).size().to_string())
    la = df[df.dataset == "la19"]
    print("\n-- la19 attacks per partition\n", la.groupby(["partition", "attack_id"]).size().unstack(0, fill_value=0).to_string())
    print("\n-- orig_sr\n", df.groupby("dataset").orig_sr.value_counts().to_string())
    print("\n-- duration_s\n", df.groupby("dataset").duration_s.describe().to_string())
    print("NaN durations:", int(df.duration_s.isna().sum()))
    print("\n-- method_family\n", df.groupby("dataset").method_family.value_counts().to_string())
    print("\n-- speakers per dataset\n", df.groupby("dataset").speaker_id.nunique().to_string())
    print("\n-- speakers per la19 partition\n", la.groupby("partition").speaker_id.nunique().to_string())
print(buf.getvalue())

if not df.utt_id.is_unique:
    bad.append("utt_id not unique")
if df.duration_s.isna().any():
    bad.append(f"{int(df.duration_s.isna().sum())} unreadable files (NaN duration)")
unk = la[(la.label == 1) & (la.method_family == "unknown")]
if len(unk):
    bad.append(f"la19 spoofs with unknown family: {sorted(unk.attack_id.unique())}")
if set(la[la.partition.isin(["train", "dev"]) & (la.label == 1)].attack_id) - {f"A0{i}" for i in range(1, 7)}:
    bad.append("train/dev contain attacks other than A01-A06")
if (la[la.label == 1].attack_id == "-").any():
    bad.append("spoof rows with attack '-'")
if not a.dummy:
    for (ds, part), (nb, ns) in EXPECTED.items():
        d = df[(df.dataset == ds) & (df.partition == part)]
        got = (int((d.label == 0).sum()), int((d.label == 1).sum()))
        if got != (nb, ns):
            bad.append(f"{ds}/{part}: got {got}, expected {(nb, ns)}")
    if la.speaker_id.nunique() != 107:
        bad.append(f"la19 has {la.speaker_id.nunique()} speakers, expected 107")
    if (la.orig_sr != 16000).any():
        bad.append("la19 contains files that are not 16 kHz")

paths.TABLES.mkdir(parents=True, exist_ok=True)
out = paths.TABLES / "data_manifest_checks.txt"
out.write_text(buf.getvalue() + "\nPROBLEMS:\n" + ("\n".join(bad) if bad else "none") + "\n", encoding="utf-8")
print("\n".join("FAIL: " + b for b in bad) if bad else "MANIFEST CHECKS PASSED", "->", out)
sys.exit(1 if bad else 0)
