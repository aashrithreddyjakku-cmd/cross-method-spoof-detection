"""One command = train + score every relevant test file.
   python -m spoofdet.train.run_experiment configs/experiments/<exp>.yaml
Test files scored (whichever exist):  <protocol>/{val,test,test_seen,test_unseen}.csv and the cross-dataset sets
splits/P0_official/test_itw.csv, test_la21.csv, test_df21.csv.
"""
import subprocess
import sys

from ..common import paths
from ..common.io import load_yaml, read_manifest
from ..data.features import FeatureStore
from .predict import predict_split
from .train import set_by_path


def main():
    cfg_path, extra = sys.argv[1], sys.argv[2:]
    subprocess.run([sys.executable, "-m", "spoofdet.train.train", cfg_path] + (["--set"] + extra if extra else []), check=True)
    cfg = load_yaml(cfg_path)
    for item in extra:                                   # apply the same overrides train.py applied
        k, v = item.split("=", 1); set_by_path(cfg, k, v)
    exp_id = cfg["exp_id"]
    pdir = paths.SPLITS / cfg["protocol"]
    jobs = {n: pdir / f"{n}.csv" for n in ("val", "test", "test_seen", "test_unseen")}
    xd = paths.SPLITS / "P0_official"
    cross = {"itw": xd / "test_itw.csv", "la21": xd / "test_la21.csv", "df21": xd / "test_df21.csv"}
    store = FeatureStore(paths.FEATURES / cfg["feat_id"])
    for name, f in cross.items():
        if f.exists():
            if all(u in store for u in read_manifest(f).utt_id):
                jobs[name] = f
            else:                                        # ablation stores hold only the P1 union
                print(f"skip {name}: not in feature store {cfg['feat_id']}")
    out = paths.SCORES / exp_id; out.mkdir(parents=True, exist_ok=True)
    for name, f in jobs.items():
        if f.exists():
            df = predict_split(exp_id, f, workers=2)
            df.to_csv(out / f"{name}.csv", index=False); print(f"scored {name}: {len(df)}")
    (out / "DONE").write_text("ok")                       # marker used by scripts/run_queue.py


if __name__ == "__main__":
    main()
