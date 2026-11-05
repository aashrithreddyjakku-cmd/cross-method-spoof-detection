"""Task A.12: assemble the folder to upload as the private Kaggle dataset spoofdet-data-v1.
  python scripts/package_kaggle.py --user <kaggle-username> [--out ../spoofdet-data-v1] [--stores mel64_w400 ...]
Files are hard-linked when possible (no extra disk), otherwise copied. Raw audio is NEVER included (licence).
Upload:  kaggle datasets create -p <out> --dir-mode tar      (first time)
         kaggle datasets version -p <out> --dir-mode tar -m "v2: ablation stores"   (later)
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from spoofdet.common import paths


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username (dataset id = <user>/spoofdet-data-v1)")
    ap.add_argument("--out", default=str(paths.DATA.parent / "spoofdet-data-v1"))
    ap.add_argument("--stores", nargs="*", default=None, help="feature folders to include (default: all)")
    a = ap.parse_args()
    out = Path(a.out)
    stores = a.stores or sorted(p.name for p in paths.FEATURES.iterdir() if (p / "feats.npy").exists())
    for m in paths.MANIFESTS.glob("*.csv"):
        link_or_copy(m, out / "manifests" / m.name)
    for s in stores:
        for fn in ("feats.npy", "index.csv", "feat_config.yaml"):
            link_or_copy(paths.FEATURES / s / fn, out / "features" / s / fn)
        print("store", s)
    meta = {"title": "spoofdet-data-v1", "id": f"{a.user}/spoofdet-data-v1",
            "licenses": [{"name": "other"}],
            "description": "Log-mel feature stores + manifests for the cross-method spoof detection project. "
                           "Derived from ASVspoof 2019 LA / In-the-Wild; research use only, do not redistribute."}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e9
    print(f"{out}  ({size:.2f} GB). In a notebook set SPOOFDET_DATA=/kaggle/input/spoofdet-data-v1")


if __name__ == "__main__":
    main()
