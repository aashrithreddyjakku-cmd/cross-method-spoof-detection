"""Turn score CSVs into metric tables.
Operating-point rule: the decision threshold for "deployment" metrics comes from the experiment's own
validation scores (val.csv, seen methods only). Oracle metrics (threshold chosen on the test set itself) are
reported as well, so the gap between the two shows how badly the threshold transfers to unseen methods.

  python -m spoofdet.eval.evaluate --exp <exp_id>                 # every CSV in results/scores/<exp_id>/
  python -m spoofdet.eval.evaluate --all                          # every experiment -> results/tables/all_results.csv
Per score file it writes (under results/tables/<exp_id>/):  <name>_summary.json, <name>_per_method.csv, <name>_confusion.csv
"""
import argparse

import pandas as pd

from ..common import paths
from ..common.io import read_scores, save_json
from ..common.metrics_core import compute_eer
from .metrics import summarize, per_group_eer, bootstrap_eer_ci, metrics_at_threshold


def evaluate_scores(df, n_boot=300, val_thr=None):
    s = summarize(df.score, df.label)
    if val_thr is not None:
        m = metrics_at_threshold(df.score, df.label, val_thr)
        s.update(val_thr=val_thr, acc_at_val_thr=m["accuracy"], far_at_val_thr=m["far"], frr_at_val_thr=m["frr"],
                 tn_v=m["tn"], fp_v=m["fp"], fn_v=m["fn"], tp_v=m["tp"])
    s["eer_ci_lo"], s["eer_ci_hi"] = bootstrap_eer_ci(df.score, df.label, n_boot=n_boot)
    per_m = per_group_eer(df, "method_id") if (df.label == 1).any() else pd.DataFrame()
    return s, per_m


def eval_experiment(exp_id, n_boot=300):
    rows = []
    out = paths.TABLES / exp_id; out.mkdir(parents=True, exist_ok=True)
    val_thr = None
    vf = paths.SCORES / exp_id / "val.csv"
    if vf.exists():
        v = read_scores(vf)
        if v.label.nunique() == 2:
            val_thr = compute_eer(v.score, v.label)[1]
    for f in sorted((paths.SCORES / exp_id).glob("*.csv")):
        df = read_scores(f)
        if df.label.nunique() < 2:
            continue
        s, per_m = evaluate_scores(df, n_boot, val_thr)
        save_json(s, out / f"{f.stem}_summary.json")
        per_m.to_csv(out / f"{f.stem}_per_method.csv", index=False)
        k = ("tn_v", "fp_v", "fn_v", "tp_v") if val_thr is not None else ("tn", "fp", "fn", "tp")
        pd.DataFrame([[s[k[0]], s[k[1]]], [s[k[2]], s[k[3]]]], index=["true_bona", "true_spoof"],
                     columns=["pred_bona", "pred_spoof"]).to_csv(out / f"{f.stem}_confusion.csv")
        rows.append(dict(exp_id=exp_id, test_set=f.stem, **s))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    exps = [d.name for d in sorted(paths.SCORES.iterdir()) if d.is_dir()] if a.all else [a.exp]
    rows = []
    for e in exps:
        rows += eval_experiment(e, a.boot)
    res = pd.DataFrame(rows)
    paths.TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(paths.TABLES / ("all_results.csv" if a.all else f"{a.exp}_results.csv"), index=False)
    print(res[["exp_id", "test_set", "n", "eer", "auc", "acc_at_eer"]].to_string(index=False))


if __name__ == "__main__":
    main()
