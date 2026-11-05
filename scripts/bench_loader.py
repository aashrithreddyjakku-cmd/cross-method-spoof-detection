"""Task A.8: end-to-end Dataset/DataLoader check + loading speed (goal: >= 20 batches/s, num_workers=2, local SSD).
  python scripts/bench_loader.py --split splits/P1_lomo/A10/train.csv [--feat mel64_w400] [--plot]
"""
import argparse
import time

import numpy as np
from torch.utils.data import DataLoader

from spoofdet.common import paths
from spoofdet.data.dataset import SpoofDataset, compute_global_stats
from spoofdet.data.features import FeatureStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(paths.SPLITS / "P1_lomo" / "A10" / "train.csv"))
    ap.add_argument("--feat", default="mel64_w400")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--plot", action="store_true", help="save results/figures/data_specaugment_example.png")
    a = ap.parse_args()

    store = FeatureStore(paths.FEATURES / a.feat)
    aug = {"freq_mask": 8, "time_mask": 40, "n_masks": 2}
    ds = SpoofDataset(a.split, store, norm="utt", augment=aug)
    x, y = ds[0]
    print(f"item: {tuple(x.shape)} mean={x.mean().item():.3f} std={x.std().item():.3f} label={y}")
    stats = compute_global_stats(a.split, store, max_items=500)
    print("global stats:", stats["mean"].shape, stats["std"].shape)

    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                    persistent_workers=a.workers > 0)
    it = iter(dl); xb, yb = next(it)                         # warm-up (worker start)
    print(f"batch: {tuple(xb.shape)} spoof fraction={yb.float().mean().item():.2f}")
    n, t0 = 0, time.perf_counter()
    for _ in range(a.batches):
        try:
            next(it); n += 1
        except StopIteration:
            it = iter(dl)
    dt = time.perf_counter() - t0
    print(f"{n} batches in {dt:.2f}s -> {n / dt:.1f} batches/s ({'OK' if n / dt >= 20 else 'SLOW: copy the store to a local SSD'})")

    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        clean = SpoofDataset(a.split, store, norm="utt")[0][0]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        for k, (img, t) in enumerate(((clean, "utt-normalised"), (x, "with SpecAugment"))):
            ax[k].imshow(np.asarray(img[0]), origin="lower", aspect="auto", cmap="magma"); ax[k].set_title(t)
        paths.FIGURES.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(paths.FIGURES / "data_specaugment_example.png", dpi=150)
        print("saved", paths.FIGURES / "data_specaugment_example.png")


if __name__ == "__main__":      # required on Windows (DataLoader workers use 'spawn')
    main()
