"""Generate D5 report figures from D2/D3 scores and D4 aggregate tables.

Run ``python -m spoofdet.eval.evaluate --all`` and
``python -m spoofdet.eval.aggregate`` first. Figures are saved as PNG files in
``results/figures/``. Missing inputs are skipped with a short message.
"""
from pathlib import Path

import pandas as pd

from spoofdet.common import paths
from spoofdet.common.io import read_scores
from spoofdet.eval.aggregate import load_meta
from spoofdet.eval.plots import (
    plot_ablation_comparison,
    plot_confusion,
    plot_generalization_gap,
    plot_heatmap,
    plot_per_method_bar,
    plot_roc_det,
)

BASELINE_FEAT = "mel64_w400"
BASELINE_NORM = "utt"
PLOT_TESTS = ("test", "test_seen", "test_unseen", "itw", "la21", "df21")
BASELINE_COLUMNS = {"feat_id", "norm", "augment"}


def _read_optional_csv(path):
    path = Path(path)
    if not path.is_file():
        print(f"skip missing input: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"skip unreadable input {path}: {exc}")
        return pd.DataFrame()


def _baseline_mask(frame):
    if not BASELINE_COLUMNS.issubset(frame.columns):
        return pd.Series(False, index=frame.index)
    augment = frame["augment"].map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes"}
    )
    return frame["feat_id"].eq(BASELINE_FEAT) & frame["norm"].eq(BASELINE_NORM) & augment


def _safe_stem(value):
    return str(value).replace("/", "_").replace("\\", "_")


def _generate_p0_roc(results, meta, scores_dir, figures_dir, generated):
    meta_required = {"exp_id", "model", "kind", "seed", *BASELINE_COLUMNS}
    if (
        results.empty
        or meta.empty
        or "test_set" not in results.columns
        or not meta_required.issubset(meta.columns)
    ):
        return
    p0 = meta[(meta["kind"] == "P0_official") & _baseline_mask(meta)].copy()
    if p0.empty:
        print("skip P0 ROC/DET: no canonical mel64_w400/utt/augmented experiments in metadata")
        return
    p0["_seed_order"] = pd.to_numeric(p0["seed"], errors="coerce")
    selected = p0.sort_values(["_seed_order", "exp_id"], na_position="last").groupby("model", sort=True).head(1)
    result_ids = set(results["exp_id"].dropna().astype(str))

    for test_set in ("test", "itw"):
        named_scores = {}
        eligible = set(results.loc[results["test_set"] == test_set, "exp_id"].dropna().astype(str))
        for row in selected.itertuples(index=False):
            exp_id = str(row.exp_id)
            score_path = Path(scores_dir) / exp_id / f"{test_set}.csv"
            if exp_id not in result_ids or exp_id not in eligible or not score_path.is_file():
                continue
            try:
                score_frame = read_scores(score_path)
                if score_frame["label"].nunique() != 2:
                    print(f"skip ROC/DET for one-class scores: {score_path}")
                    continue
                named_scores[f"{row.model} (seed={row.seed})"] = score_frame
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                print(f"skip unreadable ROC/DET scores {score_path}: {exc}")
        if not named_scores:
            continue
        output = Path(figures_dir) / f"roc_det_p0_{test_set}.png"
        try:
            plot_roc_det(named_scores, output)
            generated.append(output)
        except ValueError as exc:
            print(f"skip ROC/DET {test_set}: {exc}")


def _generate_per_run_figures(results, scores_dir, figures_dir, generated):
    required = {"exp_id", "test_set"}
    if results.empty or not required.issubset(results.columns):
        return
    selected = results[results["test_set"].isin(PLOT_TESTS)]
    confusion_keys = ("tn_v", "fp_v", "fn_v", "tp_v")
    oracle_keys = ("tn", "fp", "fn", "tp")

    for row in selected.to_dict(orient="records"):
        exp_id = str(row["exp_id"])
        test_set = str(row["test_set"])
        stem = f"{_safe_stem(exp_id)}_{_safe_stem(test_set)}"
        has_val_threshold = all(key in row and pd.notna(row[key]) for key in confusion_keys)
        counts = confusion_keys if has_val_threshold else oracle_keys
        if all(key in row and pd.notna(row[key]) for key in counts):
            if has_val_threshold and pd.notna(row.get("val_thr")):
                operating_point = f"validation threshold={float(row['val_thr']):.3f}"
            elif has_val_threshold:
                operating_point = "validation-derived threshold"
            else:
                operating_point = "test EER threshold (oracle)"
            title = f"{exp_id} | {test_set}\n{operating_point}"
            output = Path(figures_dir) / f"confusion_{stem}.png"
            try:
                plot_confusion(*(int(row[key]) for key in counts), output, title)
                generated.append(output)
            except (TypeError, ValueError, OverflowError) as exc:
                print(f"skip confusion matrix {exp_id}/{test_set}: {exc}")

        score_path = Path(scores_dir) / exp_id / f"{test_set}.csv"
        if not score_path.is_file():
            continue
        try:
            score_frame = read_scores(score_path)
            if score_frame["label"].nunique() != 2:
                continue
            output = Path(figures_dir) / f"per_method_{stem}.png"
            plot_per_method_bar(score_frame, output, f"{exp_id} | {test_set}")
            generated.append(output)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            print(f"skip per-method figure {score_path}: {exc}")


def _generate_d4_figures(gap, ablations, figures_dir, generated):
    if not gap.empty:
        required = {"kind", "model", "held_out", "test_unseen", "gap_unseen_minus_seen"}
        if required.issubset(gap.columns):
            baseline = gap[_baseline_mask(gap)]
            for kind in ("P1_lomo", "P1_lofo"):
                group = baseline[baseline["kind"] == kind]
                if group.empty:
                    continue
                kind_name = "LOMO" if kind == "P1_lomo" else "LOFO"
                gap_output = Path(figures_dir) / f"gap_{kind.lower()}.png"
                heatmap_output = Path(figures_dir) / f"{kind_name.lower()}_heatmap.png"
                try:
                    plot_generalization_gap(group, gap_output, f"{kind_name} generalization gap")
                    generated.append(gap_output)
                except ValueError as exc:
                    print(f"skip {kind_name} generalization-gap plot: {exc}")

                heatmap_rows = group.dropna(subset=["test_unseen"])
                if heatmap_rows.empty:
                    continue
                cell = ["model", "held_out"]
                if heatmap_rows.duplicated(cell).any():
                    print(f"skip {kind_name} heatmap: duplicate model/fold cells in D4 table")
                    continue
                matrix = heatmap_rows.pivot(index="model", columns="held_out", values="test_unseen") * 100
                try:
                    plot_heatmap(matrix, heatmap_output, f"{kind_name} unseen-method EER (%)")
                    generated.append(heatmap_output)
                except ValueError as exc:
                    print(f"skip {kind_name} heatmap: {exc}")
        else:
            print(f"skip D4 generalization figures: missing columns {sorted(required - set(gap.columns))}")
    else:
        print("skip D4 generalization figures: no generalization_gap.csv rows")

    if not ablations.empty:
        output = Path(figures_dir) / "ablation.png"
        try:
            plot_ablation_comparison(ablations, output)
            generated.append(output)
        except ValueError as exc:
            print(f"skip ablation figure: {exc}")
    else:
        print("skip ablation figure: no ablation results")


def generate_figures(results, meta, gap, ablations, scores_dir, figures_dir):
    """Generate all figures possible from supplied D2/D3/D4 outputs.

    The function writes only figures backed by available, valid score/table
    inputs and returns the paths it generated. It never creates example data.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    _generate_p0_roc(results, meta, scores_dir, figures_dir, generated)
    _generate_d4_figures(gap, ablations, figures_dir, generated)
    _generate_per_run_figures(results, scores_dir, figures_dir, generated)
    return generated


def main():
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    results = _read_optional_csv(paths.TABLES / "all_results.csv")
    meta = _read_optional_csv(paths.TABLES / "experiments_meta.csv")
    if meta.empty and not results.empty and "exp_id" in results:
        try:
            meta = load_meta(results["exp_id"].dropna().unique())
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            print(f"skip metadata-dependent figures: {exc}")
    gap = _read_optional_csv(paths.TABLES / "generalization_gap.csv")
    ablations = _read_optional_csv(paths.TABLES / "ablations.csv")

    generated = generate_figures(results, meta, gap, ablations, paths.SCORES, paths.FIGURES)
    if generated:
        print(f"wrote {len(generated)} figures to {paths.FIGURES}")
    else:
        print(f"no figures generated; input score files/tables are unavailable or incomplete ({paths.FIGURES})")


if __name__ == "__main__":
    main()
