"""STRETCH reference baseline: frozen wav2vec2 embeddings (mean-pooled) + logistic regression.
NOT exercised in the reference test run (needs internet to download the weights): try it on ~200 files first.
    pip install transformers
    python -m spoofdet.train.ssl_baseline --protocol P0_official --name p0_w2v2_lr --tests test --layer 8
"""
import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..common import paths
from ..common.constants import SCORE_COLUMNS, CLIP_SAMPLES
from ..common.io import read_manifest
from ..common.metrics_core import compute_eer
from ..data.audio import load_wave, fix_length


@torch.no_grad()
def embed(df, model, device, layer):
    out = []
    for p in tqdm(df["path"], desc="wav2vec2"):
        y = fix_length(load_wave(paths.RAW / p), CLIP_SAMPLES)
        y = (y - y.mean()) / (y.std() + 1e-7)
        hs = model(torch.from_numpy(y)[None].to(device), output_hidden_states=True).hidden_states[layer]
        out.append(hs.mean(1)[0].cpu().numpy())                       # (768,)
    return np.stack(out)


def main():
    from transformers import Wav2Vec2Model
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="P0_official"); ap.add_argument("--name", required=True)
    ap.add_argument("--tests", nargs="*", default=["test"]); ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=6000); ap.add_argument("--model", default="facebook/wav2vec2-base")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2Model.from_pretrained(a.model).to(device).eval()
    sd = paths.SPLITS / a.protocol
    tr = read_manifest(sd / "train.csv")
    tr = tr.sample(min(a.max_train, len(tr)), random_state=1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(embed(tr, model, device, a.layer), tr.label)
    out = paths.SCORES / a.name; out.mkdir(parents=True, exist_ok=True)
    for name in ["val"] + a.tests:
        df = read_manifest(sd / f"{name}.csv")
        df["score"] = clf.predict_proba(embed(df, model, device, a.layer))[:, 1]
        df[SCORE_COLUMNS].to_csv(out / f"{name}.csv", index=False)
        print(f"{name:12s} n={len(df):6d} EER={compute_eer(df.score, df.label)[0]*100:6.2f}%")


if __name__ == "__main__":
    main()
