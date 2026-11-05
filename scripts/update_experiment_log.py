"""Rebuild docs/EXPERIMENT_LOG.md from checkpoints/*/ (one row per run).
  python scripts/update_experiment_log.py
"""
import datetime as dt

import pandas as pd
import yaml

from spoofdet.common import paths

rows = []
for d in sorted(p for p in paths.CKPT.iterdir() if (p / "config.yaml").exists()):
    c = yaml.safe_load((d / "config.yaml").read_text())
    info = yaml.safe_load((d / "run_info.yaml").read_text()) if (d / "run_info.yaml").exists() else {}
    log = pd.read_csv(d / "train_log.csv") if (d / "train_log.csv").exists() else None
    done = (paths.SCORES / d.name / "DONE").exists()
    rows.append(dict(
        date=dt.date.fromtimestamp((d / "config.yaml").stat().st_mtime).isoformat(), exp_id=d.name, model=c["model"]["name"],
        feat_id=c["feat_id"], protocol=c["protocol"], norm=c.get("norm", "utt"), aug="yes" if c.get("augment") else "no",
        seed=c["train"]["seed"], epochs_run=len(log) if log is not None else "-",
        best_val_eer=f"{100 * log.val_eer.min():.2f}%" if log is not None else "-",
        train_min=f"{log.seconds.sum() / 60:.1f}" if log is not None else "-",
        where=f"{info.get('device', 'cpu')} ({'Kaggle' if info.get('device') == 'cuda' else 'local'})", commit=info.get("commit", "-"), scored="yes" if done else "NO"))
head = "# Experiment log  (Member B owns; rebuilt by scripts/update_experiment_log.py)\n\n"
out = paths.ROOT / "docs" / "EXPERIMENT_LOG.md"
out.write_text(head + pd.DataFrame(rows).to_markdown(index=False) + "\n", encoding="utf-8")
print(f"{len(rows)} runs -> {out}")
