import numpy as np
import pandas as pd
from scipy.stats import norm

from spoofdet.common.metrics_core import compute_eer
from spoofdet.eval.metrics import (summarize, per_group_eer, bootstrap_eer_ci, metrics_at_threshold,
                                   paired_bootstrap_eer_diff)


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
