"""Classical baseline: MFCC statistics + Logistic Regression / SVM.
MFCCs are derived from the stored log-mel (DCT-II), exactly how librosa defines them.
  python -m spoofdet.train.classical --feat mel64_w400 --protocol P0_official --clf svm --name p0_mfcc_svm
"""
import argparse

import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..common import paths
from ..common.constants import SCORE_COLUMNS
from ..common.io import read_manifest, save_yaml
from ..common.metrics_core import compute_eer
from ..data.features import FeatureStore


def mfcc_stats(store: FeatureStore, utt_ids, n_mfcc=20):
    X = []
    for u in utt_ids:
        m = dct(store.get(u), type=2, axis=0, norm="ortho")[:n_mfcc]        # (n_mfcc, T)
        d = np.diff(m, axis=1)
        X.append(np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]))
    return np.stack(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="mel64_w400")
    ap.add_argument("--protocol", default="P0_official")
    ap.add_argument("--clf", choices=["lr", "svm"], default="svm")
    ap.add_argument("--name", required=True, help="exp_id for the score folder")
    ap.add_argument("--tests", nargs="*", default=["test"], help="split csv names inside the protocol folder (without .csv)")
    ap.add_argument("--max-train", type=int, default=20000)
    a = ap.parse_args()
    store = FeatureStore(paths.FEATURES / a.feat)
    sd = paths.SPLITS / a.protocol
    tr = read_manifest(sd / "train.csv")
    if len(tr) > a.max_train:
        tr = tr.sample(a.max_train, random_state=1)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced") if a.clf == "lr" else \
        SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(mfcc_stats(store, tr.utt_id), tr.label)
    out = paths.SCORES / a.name; out.mkdir(parents=True, exist_ok=True)
    for name in ["val"] + a.tests:
        df = read_manifest(sd / f"{name}.csv")
        df["score"] = pipe.predict_proba(mfcc_stats(store, df.utt_id))[:, 1]
        out_name = "itw" if name == "test_itw" else name          # standard test name of the score-file contract
        df[SCORE_COLUMNS].to_csv(out / f"{out_name}.csv", index=False)
        print(f"{name:12s} n={len(df):6d} EER={compute_eer(df.score, df.label)[0]*100:6.2f}%")
    save_yaml(dict(exp_id=a.name, model=dict(name=f"mfcc_{a.clf}", kwargs={}), feat_id=a.feat, protocol=a.protocol,
                   norm="none", augment={}, train=dict(seed=1)), paths.CKPT / a.name / "config.yaml")   # read by eval.aggregate
    (out / "DONE").write_text("ok")


if __name__ == "__main__":
    main()
