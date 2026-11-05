import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from spoofdet.common.metrics_core import compute_eer
from spoofdet.eval.metrics import (summarize, per_group_eer, bootstrap_eer_ci, metrics_at_threshold,
                                   paired_bootstrap_eer_diff)
from spoofdet.eval.evaluate import eval_experiment, evaluate_scores
from spoofdet.common.constants import SCORE_COLUMNS
from spoofdet.common import paths


def gauss(d, n=20000, seed=0):
    r = np.random.RandomState(seed)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    return np.r_[r.randn(n), r.randn(n) + d], y


def test_eer_matches_theory():            # two unit Gaussians d apart -> EER = Phi(-d/2)
    for d in (1.0, 2.0, 3.0):
        s, y = gauss(d)
        assert abs(compute_eer(s, y)[0] - norm.cdf(-d / 2)) < 0.01


def test_worked_example():
    bona = [0.1, 0.2, 0.4, 0.7]; spoof = [0.3, 0.6, 0.8, 0.9]
    eer, thr = compute_eer(bona + spoof, [0] * 4 + [1] * 4)
    assert abs(eer - 0.25) < 1e-9


def test_summarize_and_threshold():
    s, y = gauss(2.0)
    r = summarize(s, y)
    assert abs(r["auc"] - norm.cdf(2 / np.sqrt(2))) < 0.01
    m = metrics_at_threshold(s, y, r["eer_thr"])
    assert abs(m["far"] - m["frr"]) < 0.01


def test_per_group_order():
    r = np.random.RandomState(1)
    rows = [dict(label=0, score=v, method_id="bonafide") for v in r.randn(2000)]
    for m, d in (("easy", 4.0), ("hard", 0.5)):
        rows += [dict(label=1, score=v, method_id=m) for v in r.randn(1000) + d]
    t = per_group_eer(pd.DataFrame(rows))
    assert list(t.group) == ["hard", "easy"]


def test_bootstrap_ci_contains_truth():
    s, y = gauss(2.0, n=3000)
    lo, hi = bootstrap_eer_ci(s, y, n_boot=200)
    assert lo < norm.cdf(-1) < hi


def test_paired_bootstrap():
    s1, y = gauss(3.0, n=2000); s2, _ = gauss(1.0, n=2000, seed=0)
    df = lambda s: pd.DataFrame(dict(utt_id=range(len(y)), label=y, score=s))
    mean, lo, hi, p = paired_bootstrap_eer_diff(df(s1), df(s2), n_boot=200)
    assert mean < 0 and hi < 0 and p > 0.99


def test_threshold_metrics_report_far_frr_and_confusion_counts():
    scores = [0.1, 0.7, 0.2, 0.8]
    labels = [0, 0, 1, 1]
    result = metrics_at_threshold(scores, labels, 0.5)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["accuracy"] == result["far"] == result["frr"] == 0.5


def test_summarize_reports_rates_and_custom_threshold_separately():
    result = summarize([0.1, 0.2, 0.7, 0.8], [0, 0, 1, 1], fixed_thr=0.75)
    assert result["acc_at_0p5"] == 1.0
    assert result["acc_at_fixed_thr"] == 0.75
    assert result["fixed_thr"] == 0.75
    assert "far_at_eer" in result and "frr_at_eer" in result


@pytest.mark.parametrize(
    "scores,labels,threshold",
    [([], [], 0.5), ([0.1], [], 0.5), ([[0.1]], [0], 0.5), ([np.nan], [0], 0.5),
     ([0.1], [2], 0.5), ([0.1], [0], np.inf), ([0.1], [0], "bad")],
)
def test_threshold_metrics_reject_invalid_inputs(scores, labels, threshold):
    with pytest.raises(ValueError):
        metrics_at_threshold(scores, labels, threshold)


def test_bootstrap_ci_is_stratified_for_rare_class():
    scores = np.linspace(0, 1, 100)
    labels = np.zeros(100, dtype=int)
    labels[-1] = 1
    lo, hi = bootstrap_eer_ci(scores, labels, n_boot=50, seed=7)
    assert 0 <= lo <= hi <= 1


@pytest.mark.parametrize("n_boot,alpha", [(0, 0.05), (-1, 0.05), (10, 0), (10, 1)])
def test_bootstrap_ci_rejects_invalid_parameters(n_boot, alpha):
    with pytest.raises(ValueError):
        bootstrap_eer_ci([0.1, 0.9], [0, 1], n_boot=n_boot, alpha=alpha)


def _score_frame(scores, labels, methods=None):
    n = len(labels)
    if methods is None:
        methods = ["bonafide" if label == 0 else "method_a" for label in labels]
    return pd.DataFrame({
        "utt_id": [f"u{i}" for i in range(n)], "score": scores, "label": labels,
        "dataset": ["la19"] * n, "attack_id": ["-" if y == 0 else "A01" for y in labels],
        "method_id": methods, "method_family": ["-" if y == 0 else "neural" for y in labels],
        "speaker_id": [f"s{i}" for i in range(n)],
    })[SCORE_COLUMNS]


def test_evaluate_scores_applies_validation_threshold_to_method_rates():
    frame = _score_frame([0.2, 0.7, 0.4, 0.9], [0, 0, 1, 1], ["bonafide", "bonafide", "method_a", "method_b"])
    summary, per_method = evaluate_scores(frame, n_boot=20, val_thr=0.5)
    assert summary["val_thr"] == 0.5
    assert summary["far_at_val_thr"] == 0.5
    rates = per_method.set_index("group")["detect_rate_at_ref_thr"]
    assert rates.to_dict() == {"method_a": 0.0, "method_b": 1.0}


def test_eval_experiment_uses_validation_threshold_for_test_confusion(tmp_path, monkeypatch):
    score_root = tmp_path / "scores"
    table_root = tmp_path / "tables"
    exp_dir = score_root / "sample"
    exp_dir.mkdir(parents=True)
    _score_frame([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]).to_csv(exp_dir / "val.csv", index=False)
    _score_frame([0.7, 0.1, 0.6, 0.9], [0, 0, 1, 1]).to_csv(exp_dir / "test_unseen.csv", index=False)
    monkeypatch.setattr(paths, "SCORES", score_root)
    monkeypatch.setattr(paths, "TABLES", table_root)

    rows = eval_experiment("sample", n_boot=20)

    test = next(row for row in rows if row["test_set"] == "test_unseen")
    assert test["val_thr"] == pytest.approx(0.8)
    assert test["fn_v"] == 1
    confusion = pd.read_csv(table_root / "sample" / "test_unseen_confusion.csv", index_col=0)
    assert confusion.loc["true_spoof", "pred_bona"] == 1


def test_eval_experiment_rejects_single_class_validation_scores(tmp_path, monkeypatch):
    score_root = tmp_path / "scores"
    exp_dir = score_root / "sample"
    exp_dir.mkdir(parents=True)
    _score_frame([0.1, 0.2], [0, 0]).to_csv(exp_dir / "val.csv", index=False)
    monkeypatch.setattr(paths, "SCORES", score_root)
    monkeypatch.setattr(paths, "TABLES", tmp_path / "tables")
    with pytest.raises(ValueError, match="both labels"):
        eval_experiment("sample")


def test_eval_experiment_reports_missing_score_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCORES", tmp_path / "scores")
    with pytest.raises(FileNotFoundError, match="score directory"):
        eval_experiment("missing")
