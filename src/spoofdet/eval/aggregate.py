"""Combine all experiments into the tables used in the report.
   python -m spoofdet.eval.aggregate        (run `python -m spoofdet.eval.evaluate --all` first)
Writes results/tables/{experiments_meta,generalization_gap,lomo_by_method,ablations,cross_dataset}.csv (+ .md)
"""
import pandas as pd

from ..common import paths
from ..common.io import load_yaml


def load_meta(exp_ids):
    rows = []
    for e in exp_ids:
        p = paths.CONFIGS / "experiments" / f"{e}.yaml"
        if not p.exists():
            p = paths.CKPT / e / "config.yaml"
        if not p.exists():
            continue
        c = load_yaml(p)
        proto = c["protocol"]
        kind = proto.split("/")[0]
        rows.append(dict(exp_id=e, model=c["model"]["name"], feat_id=c["feat_id"], norm=c.get("norm", "utt"),
                         augment=bool(c.get("augment")), seed=c["train"]["seed"], protocol=proto, kind=kind,
                         held_out=proto.split("/")[1] if "/" in proto else ""))
    return pd.DataFrame(rows)


def ms(x):
    return f"{x.mean()*100:.2f} ± {x.std(ddof=0)*100:.2f}" if len(x) > 1 else f"{x.mean()*100:.2f}"


def main():
    res = pd.read_csv(paths.TABLES / "all_results.csv")
    meta = load_meta(res.exp_id.unique())
    meta.to_csv(paths.TABLES / "experiments_meta.csv", index=False)
    df = res.merge(meta, on="exp_id")

    # 1. generalisation gap: seen-method test vs unseen-method test (+ ITW)
    held = df[df.kind.isin(["P1_lomo", "P1_lofo"])]
    wide = held.pivot_table(index=["kind", "held_out", "model", "feat_id", "norm", "augment", "seed"],
                            columns="test_set", values="eer").reset_index()
    if "test_unseen" not in wide.columns:
        print("No leave-one-out experiments scored yet (no test_unseen files) - nothing to aggregate.")
        return
    if "test_seen" in wide.columns:
        wide["gap_unseen_minus_seen"] = wide["test_unseen"] - wide["test_seen"]
    wide.to_csv(paths.TABLES / "generalization_gap.csv", index=False)

    # 2. LOMO / LOFO table: rows = held-out method, columns = model, value = unseen EER mean ± std over seeds
    for kind in ("P1_lomo", "P1_lofo"):
        w = wide[(wide.kind == kind) & (wide.feat_id == "mel64_w400") & (wide.norm == "utt") & wide.augment]
        if len(w):
            t = w.groupby(["held_out", "model"])["test_unseen"].apply(ms).unstack()
            t.to_csv(paths.TABLES / f"{kind.lower()}_by_method.csv"); (paths.TABLES / f"{kind.lower()}_by_method.md").write_text(t.to_markdown())

    # 3. cross-dataset: any experiment scored on itw
    xd = df[df.test_set.isin(["itw", "la21", "df21"])].pivot_table(index=["exp_id", "model", "kind"], columns="test_set", values="eer").reset_index()
    xd.to_csv(paths.TABLES / "cross_dataset.csv", index=False)

    # 4. ablations: unseen EER mean over folds, per factor level
    ab = wide[wide["held_out"].notna()].copy()
    ab["factor_level"] = ab.apply(lambda r: f"{r.feat_id} | norm={r.norm} | aug={r.augment}", axis=1)
    t = ab.groupby("factor_level")["test_unseen"].agg(["mean", "std", "count"]).sort_values("mean")
    t.to_csv(paths.TABLES / "ablations.csv")
    print("wrote tables to", paths.TABLES)


if __name__ == "__main__":
    main()
