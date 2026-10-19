"""Compute log-mel features for every utterance listed in one or more manifest/split CSVs.

  python scripts/extract_features.py --feat mel64_w400 --manifest data/manifests/master.csv
  python scripts/extract_features.py --feat mel40_w400 --manifest splits/_union_P1.csv       (ablation subsets)
  python scripts/extract_features.py --feat mel64_w400 --out-name mel64_w400__la21 --manifest splits/P0_official/test_la21.csv
"""
import argparse

import pandas as pd

from spoofdet.common import paths
from spoofdet.common.io import read_manifest
from spoofdet.data.features import FeatureStore, load_feat_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True, help="feat_id, e.g. mel64_w400 (configs/features/<feat>.yaml)")
    ap.add_argument("--manifest", nargs="+", default=[str(paths.MANIFESTS / "master.csv")],
                    help="one or more CSVs; the union of their utt_ids is extracted")
    ap.add_argument("--out-name", default=None, help="folder name under data/features (default = feat id)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = load_feat_cfg(a.feat)
    assert cfg["feat_id"] == a.feat, "feat_id inside the yaml must equal the file name"
    df = pd.concat([read_manifest(m) for m in a.manifest], ignore_index=True)
    out_name = a.out_name or a.feat
    print(f"{df.utt_id.nunique()} unique utterances -> data/features/{out_name}")
    FeatureStore.build(df, cfg, paths.FEATURES / out_name, workers=a.workers, overwrite=a.overwrite)


if __name__ == "__main__":      # this guard is REQUIRED on Windows/macOS (multiprocessing 'spawn')
    main()
