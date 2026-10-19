"""Score a split CSV with a trained experiment.

  python -m spoofdet.train.predict --exp <exp_id> --split splits/P0_official/test.csv --name la19_eval
Writes results/scores/<exp_id>/<name>.csv with columns SCORE_COLUMNS (score = P(spoof)).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..common import paths
from ..common.constants import SCORE_COLUMNS
from ..data.dataset import SpoofDataset
from ..data.features import FeatureStore
from ..models import load_checkpoint


@torch.no_grad()
def predict_split(exp_id, split_csv, feat_id=None, batch_size=128, device=None, ckpt_name="best.pt", workers=2):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(paths.CKPT / exp_id / ckpt_name, device)
    store = FeatureStore(paths.FEATURES / (feat_id or ck["feat_cfg"]["feat_id"]))
    ds = SpoofDataset(split_csv, store, ck["norm"], ck["norm_stats"], None)
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=workers)
    s = []
    for x, _ in dl:
        s.append(torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy())
    out = ds.df.copy(); out["score"] = np.concatenate(s)
    return out[SCORE_COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", nargs="+", required=True, help="one or more split CSVs")
    ap.add_argument("--name", nargs="+", required=True, help="output names, same count as --split")
    ap.add_argument("--feat", default=None, help="override feature store (e.g. when testing on another store)")
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    assert len(a.split) == len(a.name)
    out_dir = paths.SCORES / a.exp; out_dir.mkdir(parents=True, exist_ok=True)
    for sp, nm in zip(a.split, a.name):
        df = predict_split(a.exp, sp, a.feat, ckpt_name=a.ckpt)
        df.to_csv(out_dir / f"{nm}.csv", index=False)
        print(f"wrote {out_dir/nm}.csv  ({len(df)} rows)")


if __name__ == "__main__":
    main()
