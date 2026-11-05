import numpy as np
import pandas as pd
import pytest

from spoofdet.common.constants import SCORE_COLUMNS
from spoofdet.eval.plots import (
    plot_ablation_comparison,
    plot_confusion,
    plot_generalization_gap,
    plot_heatmap,
    plot_per_method_bar,
    plot_roc_det,
)
from scripts.make_figures import generate_figures


def _scores(scores=(0.1, 0.3, 0.65, 0.9)):
    labels = [0, 0, 1, 1]
    return pd.DataFrame({
        "utt_id": [f"u{i}" for i in range(len(labels))],
        "score": scores,
        "label": labels,
        "dataset": ["la19"] * len(labels),
        "attack_id": ["-", "-", "A01", "A01"],
        "method_id": ["bonafide", "bonafide", "A01", "A01"],
        "method_family": ["-", "-", "neural_vocoder", "neural_vocoder"],
        "speaker_id": ["s1", "s2", "s3", "s4"],
    })[SCORE_COLUMNS]


def test_roc_det_and_per_method_plots_save_pngs(tmp_path):
    scores = _scores()
    roc_path = tmp_path / "curves" / "roc_det.png"
    method_path = tmp_path / "methods.png"

    plot_roc_det({"CNN (seed=1)": scores}, roc_path)
    plot_per_method_bar(scores, method_path, "P0 test — lcnn")

    assert roc_path.is_file() and roc_path.stat().st_size > 0
    assert method_path.is_file() and method_path.stat().st_size > 0


@pytest.mark.parametrize("scores", [(np.nan, 0.2, 0.8, 0.9), (-0.1, 0.2, 0.8, 0.9), (0.1, 0.2, 0.8, 1.1)])
def test_curve_plot_rejects_invalid_probability_scores(tmp_path, scores):
    with pytest.raises(ValueError):
        plot_roc_det({"invalid": _scores(scores)}, tmp_path / "invalid.png")


def test_curve_plot_rejects_single_class_inputs(tmp_path):
    frame = _scores().assign(label=[0, 0, 0, 0])
    with pytest.raises(ValueError, match="both bona-fide and spoof"):
        plot_roc_det({"one class": frame}, tmp_path / "one_class.png")


def test_confusion_heatmap_gap_and_ablation_plots_save_pngs(tmp_path):
    confusion = tmp_path / "confusion.png"
    heatmap = tmp_path / "heatmap.png"
    gap = tmp_path / "gap.png"
    ablation = tmp_path / "ablation.png"

    plot_confusion(15, 2, 4, 9, confusion, "p0_lcnn | test\nvalidation threshold=0.5")
    plot_heatmap(pd.DataFrame([[12.0, 24.0]], index=["lcnn"], columns=["A01", "A04"]), heatmap, "LOMO EER (%)")
    plot_generalization_gap(pd.DataFrame([{
        "model": "lcnn", "held_out": "A01", "gap_unseen_minus_seen": 0.08,
        "gap_unseen_minus_seen_std": 0.02,
    }]), gap, "LOMO generalization gap")
    plot_ablation_comparison(pd.DataFrame([{
        "model": "lcnn", "kind": "P1_lofo", "factor_level": "mel40 | norm=utt | aug=True",
        "mean": 0.25, "std": 0.03, "count": 3, "n_folds": 3, "n_seeds": 1,
    }]), ablation)

    for path in (confusion, heatmap, gap, ablation):
        assert path.is_file() and path.stat().st_size > 0


def _meta(exp_id, model, protocol, held_out="", seed=1):
    return {
        "exp_id": exp_id,
        "model": model,
        "feat_id": "mel64_w400",
        "norm": "utt",
        "augment": True,
        "seed": seed,
        "protocol": protocol,
        "kind": protocol.split("/")[0],
        "held_out": held_out,
    }


def _result(exp_id, test_set, eer):
    return {
        "exp_id": exp_id,
        "test_set": test_set,
        "eer": eer,
        "tn_v": 1,
        "fp_v": 1,
        "fn_v": 1,
        "tp_v": 1,
        "val_thr": 0.5,
        "tn": 2,
        "fp": 0,
        "fn": 0,
        "tp": 2,
    }


def test_generate_figures_uses_d3_scores_and_d4_tables(tmp_path):
    scores_root = tmp_path / "scores"
    figure_root = tmp_path / "figures"
    p0_id = "p0_lcnn_mel64_w400_s1"
    lomo_id = "lomo_A01_lcnn_mel64_w400_s1"
    lofo_id = "lofo_neural_lcnn_mel64_w400_s1"
    score_dir = scores_root / p0_id
    score_dir.mkdir(parents=True)
    _scores().to_csv(score_dir / "test.csv", index=False)
    _scores((0.2, 0.4, 0.6, 0.8)).to_csv(score_dir / "itw.csv", index=False)

    results = pd.DataFrame([
        _result(p0_id, "test", 0.0), _result(p0_id, "itw", 0.1),
        _result(lomo_id, "test_seen", 0.1), _result(lomo_id, "test_unseen", 0.2),
        _result(lofo_id, "test_seen", 0.1), _result(lofo_id, "test_unseen", 0.3),
    ])
    meta = pd.DataFrame([
        _meta(p0_id, "lcnn", "P0_official"),
        _meta(lomo_id, "lcnn", "P1_lomo/A01", "A01"),
        _meta(lofo_id, "lcnn", "P1_lofo/neural_vocoder", "neural_vocoder"),
    ])
    gap = pd.DataFrame([
        {"kind": "P1_lomo", "protocol": "P1_lomo/A01", "held_out": "A01", "model": "lcnn",
         "feat_id": "mel64_w400", "norm": "utt", "augment": True, "test_seen": 0.1,
         "test_unseen": 0.2, "gap_unseen_minus_seen": 0.1,
         "gap_unseen_minus_seen_std": np.nan, "gap_unseen_minus_seen_n": 1},
        {"kind": "P1_lofo", "protocol": "P1_lofo/neural_vocoder", "held_out": "neural_vocoder", "model": "lcnn",
         "feat_id": "mel64_w400", "norm": "utt", "augment": True, "test_seen": 0.1,
         "test_unseen": 0.3, "gap_unseen_minus_seen": 0.2,
         "gap_unseen_minus_seen_std": np.nan, "gap_unseen_minus_seen_n": 1},
    ])
    ablations = pd.DataFrame([{
        "model": "lcnn", "kind": "P1_lofo", "factor_level": "mel40_w400 | norm=utt | aug=True",
        "mean": 0.25, "std": np.nan, "count": 1, "n_folds": 1, "n_seeds": 1,
    }])

    generated = generate_figures(results, meta, gap, ablations, scores_root, figure_root)
    names = {path.name for path in generated}

    assert {
        "roc_det_p0_test.png", "roc_det_p0_itw.png", "gap_p1_lomo.png", "gap_p1_lofo.png",
        "lomo_heatmap.png", "lofo_heatmap.png", "ablation.png",
        f"confusion_{p0_id}_test.png", f"per_method_{p0_id}_test.png",
    }.issubset(names)
    assert all((figure_root / name).is_file() for name in names)


def test_generate_figures_skips_missing_inputs_without_failing(tmp_path):
    output_dir = tmp_path / "figures"
    generated = generate_figures(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        tmp_path / "missing_scores", output_dir,
    )

    assert generated == []
    assert output_dir.is_dir()
