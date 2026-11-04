"""Combine evaluated experiment scores into report tables.

Run ``python -m spoofdet.eval.evaluate --all`` first, then
``python -m spoofdet.eval.aggregate``. EER values are written as fractions;
the LOMO/LOFO report cells are formatted as percentages.
"""
import pandas as pd

from ..common import paths
from ..common.io import load_yaml

LOO_KINDS = ("P1_lomo", "P1_lofo")
DATASET_TESTS = ("test", "itw", "la21", "df21")
META_COLUMNS = ["exp_id", "model", "feat_id", "norm", "augment", "seed", "protocol", "kind", "held_out"]
GAP_GROUP_COLUMNS = ["kind", "protocol", "held_out", "model", "feat_id", "norm", "augment"]


def load_meta(exp_ids):
    """Load the configuration metadata needed to group each evaluated run."""
    rows = []
    missing = []
    for exp_id in exp_ids:
        path = paths.CONFIGS / "experiments" / f"{exp_id}.yaml"
        if not path.exists():
            path = paths.CKPT / exp_id / "config.yaml"
        if not path.exists():
            missing.append(exp_id)
            continue

        config = load_yaml(path)
        protocol = config["protocol"]
        rows.append(dict(
            exp_id=exp_id,
            model=config["model"]["name"],
            feat_id=config["feat_id"],
            norm=config.get("norm", "utt"),
            augment=bool(config.get("augment")),
            seed=config["train"]["seed"],
            protocol=protocol,
            kind=protocol.split("/")[0],
            held_out=protocol.split("/", 1)[1] if "/" in protocol else "",
        ))
    if missing:
        raise FileNotFoundError(
            "missing experiment config for: " + ", ".join(map(str, missing))
        )
    return pd.DataFrame(rows, columns=META_COLUMNS)


def ms(values):
    """Format an EER series as mean +/- population SD, including seed count."""
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return ""
    mean = values.mean() * 100
    if len(values) == 1:
        return f"{mean:.2f} (n=1)"
    std = values.std(ddof=0) * 100
    return f"{mean:.2f} ± {std:.2f} (n={len(values)})"


def _mean_std_count(values, seed_values):
    values = pd.Series(values, dtype=float).dropna()
    if values.empty:
        return float("nan"), float("nan"), 0
    n_seeds = pd.Series(seed_values).loc[values.index].nunique()
    std = values.std(ddof=0) if len(values) > 1 else float("nan")
    return float(values.mean()), float(std), int(n_seeds)


def _summarize_by_seed(frame, group_columns, value_columns):
    """Summarize run-level values while keeping the number of seeds per metric."""
    records = []
    if frame.empty:
        columns = list(group_columns)
        for value in value_columns:
            columns.extend((value, f"{value}_std", f"{value}_n"))
        return pd.DataFrame(columns=columns)

    for key, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_columns, key))
        for value in value_columns:
            valid = group.dropna(subset=[value])
            mean, std, n_seeds = _mean_std_count(valid[value], valid["seed"])
            row[value] = mean
            row[f"{value}_std"] = std
            row[f"{value}_n"] = n_seeds
        records.append(row)
    return pd.DataFrame(records)


def _make_gap_tables(frame):
    is_ablation = frame["exp_id"].astype(str).str.startswith("abl_")
    selected = frame[frame.kind.isin(LOO_KINDS) & ~is_ablation]
    run_columns = ["exp_id", *GAP_GROUP_COLUMNS, "seed"]
    if selected.empty:
        by_seed = pd.DataFrame(columns=[*run_columns, "test_seen", "test_unseen", "gap_unseen_minus_seen"])
    else:
        by_seed = selected.pivot(index=run_columns, columns="test_set", values="eer").reset_index()
        by_seed.columns.name = None
        for name in ("test_seen", "test_unseen"):
            if name not in by_seed:
                by_seed[name] = float("nan")
        by_seed["gap_unseen_minus_seen"] = by_seed["test_unseen"] - by_seed["test_seen"]

    summary = _summarize_by_seed(
        by_seed,
        GAP_GROUP_COLUMNS,
        ["test_seen", "test_unseen", "gap_unseen_minus_seen"],
    )
    return by_seed, summary


def _make_method_tables(by_seed):
    tables = {}
    baseline = by_seed[
        (by_seed["feat_id"] == "mel64_w400")
        & (by_seed["norm"] == "utt")
        & (by_seed["augment"] == True)  # noqa: E712 - explicit comparison for pandas nullable values
        & by_seed["test_unseen"].notna()
    ]
    for kind in LOO_KINDS:
        current = baseline[baseline["kind"] == kind]
        if current.empty:
            tables[kind] = pd.DataFrame()
            continue
        tables[kind] = (
            current.groupby(["held_out", "model"], dropna=False)["test_unseen"]
            .apply(ms)
            .unstack("model")
            .sort_index()
        )
    return tables


def _make_cross_dataset_table(frame):
    grouping = ["model", "kind", "protocol", "held_out", "feat_id", "norm", "augment"]
    is_ablation = frame["exp_id"].astype(str).str.startswith("abl_")
    selected = frame[frame.test_set.isin(DATASET_TESTS) & ~is_ablation]
    if selected.empty:
        columns = grouping[:]
        for dataset in DATASET_TESTS:
            columns.extend((dataset, f"{dataset}_std", f"{dataset}_n"))
        return pd.DataFrame(columns=columns)

    wide = selected.pivot(index=[*grouping, "seed"], columns="test_set", values="eer").reset_index()
    wide.columns.name = None
    for name in DATASET_TESTS:
        if name not in wide:
            wide[name] = float("nan")
    return _summarize_by_seed(wide, grouping, list(DATASET_TESTS))


def _make_ablation_table(frame):
    # Matrix-generated ablations use the abl_ prefix. Ordinary LOMO/LOFO runs
    # are not ablation observations, even when they share the same settings.
    selected = frame[
        frame["exp_id"].astype(str).str.startswith("abl_")
        & frame["kind"].isin(LOO_KINDS)
        & frame["held_out"].fillna("").ne("")
        & frame["test_set"].eq("test_unseen")
    ].copy()
    columns = ["model", "kind", "factor_level", "mean", "std", "count", "n_folds", "n_seeds"]
    if selected.empty:
        return pd.DataFrame(columns=columns)

    selected["factor_level"] = selected.apply(
        lambda row: f"{row.feat_id} | norm={row.norm} | aug={row.augment}", axis=1
    )
    rows = []
    for (model, kind, level), group in selected.groupby(
        ["model", "kind", "factor_level"], dropna=False, sort=True
    ):
        values = group["eer"].dropna()
        rows.append({
            "model": model,
            "kind": kind,
            "factor_level": level,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)) if len(values) > 1 else float("nan"),
            "count": int(len(values)),
            "n_folds": int(group["held_out"].nunique()),
            "n_seeds": int(group["seed"].nunique()),
        })
    return pd.DataFrame(rows, columns=columns).sort_values("mean", na_position="last").reset_index(drop=True)


def aggregate_results(results, meta):
    """Build D4 tables from evaluated rows and their experiment metadata."""
    required = {"exp_id", "test_set", "eer"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"evaluation results are missing required columns: {missing}")
    if meta["exp_id"].duplicated().any():
        raise ValueError("experiment metadata must contain one row per exp_id")
    required_meta = set(META_COLUMNS)
    missing_meta = sorted(required_meta - set(meta.columns))
    if missing_meta:
        raise ValueError(f"experiment metadata are missing required columns: {missing_meta}")

    frame = results.merge(meta, on="exp_id", how="left", validate="many_to_one", indicator=True)
    unmatched = frame.loc[frame["_merge"] != "both", "exp_id"].drop_duplicates().tolist()
    if unmatched:
        raise ValueError("missing metadata for evaluated experiments: " + ", ".join(map(str, unmatched)))
    frame = frame.drop(columns="_merge")

    # Duplicate score outputs for one experiment are invalid. Separately,
    # standard runs must have one observation per config/seed/test cell.
    if frame.duplicated(["exp_id", "test_set"]).any():
        raise ValueError("evaluation results must contain at most one row per exp_id and test_set")
    is_ablation = frame["exp_id"].astype(str).str.startswith("abl_")
    standard = frame[~is_ablation]
    ablation_runs = frame[is_ablation]
    cell_key = ["kind", "protocol", "held_out", "model", "feat_id", "norm", "augment", "seed", "test_set"]
    for runs in (standard, ablation_runs):
        if not runs.duplicated(cell_key).any():
            continue
        duplicates = runs.loc[runs.duplicated(cell_key, keep=False), cell_key].drop_duplicates()
        example = duplicates.iloc[0].to_dict()
        raise ValueError(f"duplicate evaluation rows for one config/seed/test cell: {example}")

    gap_by_seed, gap_summary = _make_gap_tables(frame)
    method_tables = _make_method_tables(gap_by_seed)
    cross_dataset = _make_cross_dataset_table(frame)
    ablations = _make_ablation_table(frame)

    return {
        "generalization_gap_by_seed": gap_by_seed,
        "generalization_gap": gap_summary,
        "p1_lomo_by_method": method_tables["P1_lomo"],
        "p1_lofo_by_method": method_tables["P1_lofo"],
        "cross_dataset": cross_dataset,
        "ablations": ablations,
    }


def main():
    results_path = paths.TABLES / "all_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"run spoofdet.eval.evaluate --all first: {results_path}")
    results = pd.read_csv(results_path)
    meta = load_meta(results.exp_id.dropna().unique())
    tables = aggregate_results(results, meta)

    paths.TABLES.mkdir(parents=True, exist_ok=True)
    meta.to_csv(paths.TABLES / "experiments_meta.csv", index=False)
    tables["generalization_gap_by_seed"].to_csv(paths.TABLES / "generalization_gap_by_seed.csv", index=False)
    tables["generalization_gap"].to_csv(paths.TABLES / "generalization_gap.csv", index=False)
    for name in ("p1_lomo_by_method", "p1_lofo_by_method"):
        table = tables[name]
        table.to_csv(paths.TABLES / f"{name}.csv")
        (paths.TABLES / f"{name}.md").write_text(table.to_markdown(), encoding="utf-8")
    tables["cross_dataset"].to_csv(paths.TABLES / "cross_dataset.csv", index=False)
    tables["ablations"].to_csv(paths.TABLES / "ablations.csv", index=False)
    print("wrote D4 tables to", paths.TABLES)


if __name__ == "__main__":
    main()
