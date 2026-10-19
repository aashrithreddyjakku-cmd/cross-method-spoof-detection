"""Params, FLOPs and CPU latency of each architecture.   python -m spoofdet.train.profile_model --n-mels 64"""
import argparse
import time

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from ..common import paths
from ..models import REGISTRY, build_model, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mels", type=int, default=64); ap.add_argument("--frames", type=int, default=401)
    ap.add_argument("--runs", type=int, default=30)
    a = ap.parse_args()
    torch.set_num_threads(1)
    x = torch.randn(1, 1, a.n_mels, a.frames); rows = []
    for name in REGISTRY:
        m = build_model(name).eval()
        with torch.no_grad():
            with FlopCounterMode(display=False) as fc:
                m(x)
            for _ in range(3): m(x)
            t0 = time.perf_counter()
            for _ in range(a.runs): m(x)
            ms = (time.perf_counter() - t0) / a.runs * 1000
        rows.append(dict(model=name, params=count_params(m), gflops=fc.get_total_flops() / 1e9, cpu_ms_per_4s_clip=ms))
    df = pd.DataFrame(rows); print(df.to_string(index=False))
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.TABLES / "model_profile.csv", index=False)


if __name__ == "__main__":
    main()
