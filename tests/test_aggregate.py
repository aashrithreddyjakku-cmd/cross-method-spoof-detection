import numpy as np
import pandas as pd
import pytest

from spoofdet.common import paths
from spoofdet.eval import aggregate


def _fixture_data():
    meta = []
    results = []

    def run(exp_id, protocol, seed, tests, model="lcnn", feat_id="mel64_w400", norm="utt", augment=True):
        kind, _, held_out = protocol.partition("/")
        meta.append({
            "exp_id": exp_id, "model": model, "feat_id": feat_id, "norm": norm,
            "augment": augment, "seed": seed, "protocol": protocol,
            "kind": kind, "held_out": held_out,
        })
        results.extend({"exp_id": exp_id, "test_set": test_set, "eer": eer}
                       for test_set, eer in tests.items())

    run("lomo_a01_s1", "P1_lomo/A01", 1, {"test_seen": 0.10, "test_unseen": 0.30})
    run("lomo_a01_s2", "P1_lomo/A01", 2, {"test_seen": 0.20, "test_unseen": 0.40})
    run("lofo_neural_s1", "P1_lofo/neural_vocoder", 1,
        {"test_seen": 0.15, "test_unseen": 0.35})

    run("abl_base_neural_lcnn", "P1_lofo/neural_vocoder", 1, {"test_unseen": 0.25})
    run("abl_base_source_lcnn", "P1_lofo/source_filter_vocoder", 1, {"test_unseen": 0.35})
    run("abl_feat_mel40_neural_lcnn", "P1_lofo/neural_vocoder", 1,
        {"test_unseen": 0.20}, feat_id="mel40_w400")
    # This regular run must not leak into the ablation summary.
    run("lomo_mel40_comparison", "P1_lomo/A04", 1,
        {"test_seen": 0.1, "test_unseen": 0.99}, feat_id="mel40_w400")

    for seed, offset in ((1, 0.0), (2, 0.1)):
        run(f"p0_smallcnn_s{seed}", "P0_official", seed, {
            "test": 0.10 + offset,
            "itw": 0.30 + offset,
            "la21": 0.50 + offset,
            "df21": 0.70 + offset,
        }, model="smallcnn")

    return pd.DataFrame(results), pd.DataFrame(meta)


def test_d4_builds_generalization_lomo_lofo_cross_dataset_and_ablation_tables():
    results, meta = _fixture_data()

    tables = aggregate.aggregate_results(results, meta)

    gap = tables["generalization_gap"].set_index(["kind", "held_out"])
    lomo = gap.loc[("P1_lomo", "A01")]
    assert lomo["test_seen"] == pytest.approx(0.15)
    assert lomo["test_seen_std"] == pytest.approx(0.05)
    assert lomo["test_seen_n"] == 2
    assert lomo["test_unseen"] == pytest.approx(0.35)
    assert lomo["gap_unseen_minus_seen"] == pytest.approx(0.20)
    assert lomo["gap_unseen_minus_seen_std"] == pytest.approx(0.0)
    assert lomo["gap_unseen_minus_seen_n"] == 2
    assert len(tables["generalization_gap_by_seed"]) == 4

    assert tables["p1_lomo_by_method"].loc["A01", "lcnn"] == "35.00 ± 5.00 (n=2)"
    assert tables["p1_lofo_by_method"].loc["neural_vocoder", "lcnn"] == "35.00 (n=1)"

    cross = tables["cross_dataset"].set_index(["model", "protocol"])
    p0 = cross.loc[("smallcnn", "P0_official")]
    assert p0["test"] == pytest.approx(0.15)
    assert p0["test_std"] == pytest.approx(0.05)
    assert p0["test_n"] == 2
    assert p0["itw"] == pytest.approx(0.35)
    assert p0["la21"] == pytest.approx(0.55)
    assert p0["df21"] == pytest.approx(0.75)

    ablations = tables["ablations"].set_index("factor_level")
    base_level = "mel64_w400 | norm=utt | aug=True"
    assert ablations.loc[base_level, "mean"] == pytest.approx(0.30)
    assert ablations.loc[base_level, "std"] == pytest.approx(0.05)
    assert ablations.loc[base_level, "count"] == 2
    assert ablations.loc[base_level, "n_folds"] == 2
    assert ablations.loc[base_level, "n_seeds"] == 1
    assert "mel40_w400 | norm=utt | aug=True" in ablations.index
    assert np.isnan(ablations.loc["mel40_w400 | norm=utt | aug=True", "std"])


def test_aggregation_keeps_independent_tables_when_no_lomo_or_lofo_results_exist():
    results, meta = _fixture_data()
    keep = meta["kind"] == "P0_official"
    results = results[results.exp_id.isin(meta.loc[keep, "exp_id"])]
    meta = meta[keep]

    tables = aggregate.aggregate_results(results, meta)

    assert tables["generalization_gap"].empty
    assert tables["p1_lomo_by_method"].empty
    assert tables["p1_lofo_by_method"].empty
    assert len(tables["cross_dataset"]) == 1
    assert tables["ablations"].empty


def test_aggregation_rejects_duplicate_config_seed_test_rows():
    results, meta = _fixture_data()
    duplicate = results.iloc[[0]].copy()
    duplicate["exp_id"] = "duplicate_id"
    duplicate_meta = meta.iloc[[0]].copy()
    duplicate_meta["exp_id"] = "duplicate_id"

    with pytest.raises(ValueError, match="duplicate evaluation rows"):
        aggregate.aggregate_results(pd.concat([results, duplicate]), pd.concat([meta, duplicate_meta]))


def test_aggregation_rejects_results_without_metadata():
    results, meta = _fixture_data()
    results.loc[results.index[0], "exp_id"] = "unconfigured"

    with pytest.raises(ValueError, match="missing metadata"):
        aggregate.aggregate_results(results, meta)


def test_main_writes_compatible_d4_table_names(tmp_path, monkeypatch):
    results, meta = _fixture_data()
    results.to_csv(tmp_path / "all_results.csv", index=False)
    monkeypatch.setattr(paths, "TABLES", tmp_path)
    monkeypatch.setattr(aggregate, "load_meta", lambda _: meta)

    aggregate.main()

    for filename in (
        "experiments_meta.csv", "generalization_gap.csv", "generalization_gap_by_seed.csv",
        "p1_lomo_by_method.csv", "p1_lomo_by_method.md", "p1_lofo_by_method.csv",
        "p1_lofo_by_method.md", "cross_dataset.csv", "ablations.csv",
    ):
        assert (tmp_path / filename).is_file()
    gap = pd.read_csv(tmp_path / "generalization_gap.csv")
    assert "gap_unseen_minus_seen" in gap.columns
    assert "gap_unseen_minus_seen_n" in gap.columns


def test_main_writes_cross_dataset_tables_without_lomo_or_lofo(tmp_path, monkeypatch):
    results, meta = _fixture_data()
    p0_ids = meta.loc[meta.kind == "P0_official", "exp_id"]
    results = results[results.exp_id.isin(p0_ids)]
    meta = meta[meta.exp_id.isin(p0_ids)]
    results.to_csv(tmp_path / "all_results.csv", index=False)
    monkeypatch.setattr(paths, "TABLES", tmp_path)
    monkeypatch.setattr(aggregate, "load_meta", lambda _: meta)

    aggregate.main()

    assert (tmp_path / "cross_dataset.csv").is_file()
    assert len(pd.read_csv(tmp_path / "cross_dataset.csv")) == 1
    assert (tmp_path / "ablations.csv").is_file()


def test_load_meta_fails_instead_of_silently_dropping_experiments(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "CONFIGS", tmp_path / "configs")
    monkeypatch.setattr(paths, "CKPT", tmp_path / "checkpoints")

    with pytest.raises(FileNotFoundError, match="missing experiment config"):
        aggregate.load_meta(["missing_config"])
