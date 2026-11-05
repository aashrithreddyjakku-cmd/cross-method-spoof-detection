"""E0 sanity checks before long runs (playbook B.4.1). Writes results/tables/b_sanity.csv.
  python scripts/sanity_checks.py [--only overfit shuffled determinism lr speed]
Uses temporary split and checkpoint folders, so real results are not touched.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from spoofdet.common import paths

SRC = paths.SPLITS / "P0_official"
TEMPLATE = paths.CONFIGS / "experiments" / "_template.yaml"


def train(tmp, exp, *over, threads=None):
    env = dict(os.environ, SPOOFDET_SPLITS=str(tmp / "splits"), SPOOFDET_CKPT=str(tmp / "ckpt"), SPOOFDET_DATA=str(paths.DATA))
    if threads:
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    r = subprocess.run([sys.executable, "-m", "spoofdet.train.train", str(TEMPLATE), "--set", f"exp_id={exp}", "train.num_workers=0", *over],
                       env=env, cwd=paths.ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        raise RuntimeError(f"{exp} failed:\n{r.stderr[-1500:]}")
    return pd.read_csv(tmp / "ckpt" / exp / "train_log.csv")


def split(tmp, name, tr, va):
    d = tmp / "splits" / name; d.mkdir(parents=True, exist_ok=True)
    tr.to_csv(d / "train.csv", index=False); va.to_csv(d / "val.csv", index=False)


def main():
    all_checks = ["overfit", "shuffled", "determinism", "lr", "speed"]
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=all_checks)
    a = ap.parse_args()
    tr, va = pd.read_csv(SRC / "train.csv"), pd.read_csv(SRC / "val.csv")
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    keys = {"overfit": "overfit", "shuffled": "shuffled", "determinism": "same seed", "lr": "learning rate", "speed": "seconds"}
    out = paths.TABLES / "b_sanity.csv"
    old = pd.read_csv(out, dtype=str, keep_default_na=False) if out.exists() else pd.DataFrame(columns=["check"])
    rerun = tuple(keys[k] for k in a.only)                 # keep the results of checks not rerun now
    rows = old[~old.check.str.startswith(rerun)].to_dict("records")

    def add(check, setting, value, criterion, passed):
        rows.append(dict(check=check, setting=setting, value=value, criterion=criterion, passed=passed))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(paths.TABLES / "b_sanity.csv", index=False)

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        split(tmp, "real", tr, va)
        if "overfit" in a.only:               # the model must be able to memorise a tiny set
            tiny = pd.concat(g.sample(128, random_state=1) for _, g in tr.groupby("label"))
            split(tmp, "tiny", tiny, tiny)
            lg = train(tmp, "overfit", "protocol=tiny", "augment={}", "model.name=smallcnn", "model.kwargs={dropout: 0.0}",
                       "train.epochs=40", "train.patience=1000", "train.weight_decay=0")
            loss = float(lg.train_loss.iloc[-1])
            add("overfit 256 clips", "smallcnn, no augmentation, 40 epochs", f"final train loss {loss:.4f}", "< 0.05", loss < 0.05)
        if "shuffled" in a.only:              # shuffled labels: nothing real to learn, so val EER must stay near 50%
            # One run can land off 50% in either direction by chance, so several shuffles must average near 50%.
            finals = []
            for seed in (1, 2, 3, 4):
                s = tr.copy(); s["label"] = s["label"].sample(frac=1, random_state=seed).values
                split(tmp, f"shuf{seed}", s, va)
                lg = train(tmp, f"shuffled{seed}", f"protocol=shuf{seed}", "model.name=smallcnn", "train.epochs=10", "train.patience=1000", f"train.seed={seed}")
                finals.append(100 * float(lg.val_eer.iloc[-1]))
                add("shuffled labels", f"smallcnn, 10 epochs, shuffle {seed}", f"final val EER {finals[-1]:.1f}%", "see mean", "")
            m = sum(finals) / len(finals)
            add("shuffled labels mean", f"{len(finals)} shuffles", f"mean val EER {m:.1f}%", "about 45-55%", 45 <= m <= 55)
        if "determinism" in a.only:
            x = [float(train(tmp, f"det{i}", "protocol=real", "model.name=smallcnn", "train.epochs=2").val_eer.iloc[-1]) for i in (1, 2)]
            d = 100 * abs(x[0] - x[1])
            add("same seed twice", "smallcnn, 2 epochs", f"val EER {100 * x[0]:.2f}% and {100 * x[1]:.2f}%", "difference < 0.1 pp", d < 0.1)
        if "lr" in a.only:                    # validation data only, so this does not touch any test file
            for model, lrs in (("smallcnn", ("0.0003", "0.001", "0.003")), ("lcnn", ("0.001", "0.003"))):
                for lr in lrs:
                    best = 100 * float(train(tmp, f"lr{model}{lr}", "protocol=real", f"model.name={model}", "train.epochs=5", f"train.lr={lr}").val_eer.min())
                    add("learning rate", f"{model}, lr {lr}, 5 epochs", f"best val EER {best:.2f}%", "lowest is best", "")
        if "speed" in a.only:
            for m in ("smallcnn", "lcnn", "vggish_cbam", "resnet18"):
                lg = train(tmp, f"speed_{m}", "protocol=real", f"model.name={m}", "train.epochs=1")
                add("seconds per epoch", f"{m}, P0 train (1,320 clips)", f"{lg.seconds.iloc[0]:.0f}", "(for planning)", "")
            for th in (8, 12):
                lg = train(tmp, f"speed_t{th}", "protocol=real", "model.name=lcnn", "train.epochs=1", threads=th)
                add("seconds per epoch", f"lcnn, {th} CPU threads", f"{lg.seconds.iloc[0]:.0f}", "(for planning)", "")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
