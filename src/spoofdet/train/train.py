"""Train one experiment.   python -m spoofdet.train.train configs/experiments/<exp>.yaml [--set train.epochs=2 ...]"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..common import paths
from ..common.io import load_yaml, save_yaml
from ..common.metrics_core import compute_eer
from ..common.seed import set_seed
from ..data.dataset import SpoofDataset, compute_global_stats
from ..data.features import FeatureStore
from ..models import build_model, count_params, save_checkpoint


def set_by_path(cfg, dotted, value):
    keys = dotted.split("."); d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml_value(value)


def yaml_value(v):
    import yaml
    return yaml.safe_load(v)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval(); scores, labels = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        scores.append(p.cpu().numpy()); labels.append(y.numpy())
    s, l = np.concatenate(scores), np.concatenate(labels)
    return compute_eer(s, l)[0], s, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. train.epochs=2 train.seed=3")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    for item in a.set:
        k, v = item.split("=", 1); set_by_path(cfg, k, v)
    t, exp_id = cfg["train"], cfg["exp_id"]
    set_seed(t["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = paths.CKPT / exp_id; out.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out / "config.yaml")

    store = FeatureStore(paths.FEATURES / cfg["feat_id"])
    split_dir = paths.SPLITS / cfg["protocol"]
    norm, stats = cfg.get("norm", "utt"), None
    if norm == "global":
        stats = compute_global_stats(split_dir / "train.csv", store)
    tr_ds = SpoofDataset(split_dir / "train.csv", store, norm, stats, cfg.get("augment") or None)
    va_ds = SpoofDataset(split_dir / "val.csv", store, norm, stats, None)
    nw = t.get("num_workers", 2)
    tr = DataLoader(tr_ds, t["batch_size"], shuffle=True, num_workers=nw, drop_last=True, pin_memory=device == "cuda")
    va = DataLoader(va_ds, t["batch_size"] * 2, shuffle=False, num_workers=nw)

    mk = cfg["model"].get("kwargs") or {}
    model = build_model(cfg["model"]["name"], **mk).to(device)
    print(f"{exp_id}: {cfg['model']['name']} params={count_params(model):,} device={device} "
          f"train={len(tr_ds)} val={len(va_ds)}")

    counts = np.bincount(tr_ds.labels, minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    use_amp = bool(t.get("amp", True)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, bad, log = 1e9, 0, []
    for ep in range(1, t["epochs"] + 1):
        model.train(); t0 = time.time(); tot, n = 0.0, 0
        for x, y in tqdm(tr, desc=f"ep{ep}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        val_eer, _, _ = run_eval(model, va, device)
        log.append(dict(epoch=ep, train_loss=tot / n, val_eer=val_eer, lr=opt.param_groups[0]["lr"],
                        seconds=time.time() - t0))
        print(f"  epoch {ep:3d} loss {tot/n:.4f} val_EER {val_eer*100:6.2f}%  ({log[-1]['seconds']:.0f}s)")
        pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
        save_checkpoint(out / "last.pt", model, cfg["model"]["name"], mk, store.cfg, norm, stats, ep, val_eer, t["seed"])
        if val_eer < best:
            best, bad = val_eer, 0
            save_checkpoint(out / "best.pt", model, cfg["model"]["name"], mk, store.cfg, norm, stats, ep, val_eer, t["seed"])
        else:
            bad += 1
            if bad >= t.get("patience", 6):
                print("  early stop"); break
    print(f"best val EER {best*100:.2f}% -> {out/'best.pt'}")


if __name__ == "__main__":
    main()
