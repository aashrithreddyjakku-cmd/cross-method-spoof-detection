import numpy as np
import torch

from spoofdet.common.metrics_core import compute_eer
from spoofdet.data.audio import fix_length
from spoofdet.data.dataset import normalize, spec_augment
from spoofdet.data.features import wav_to_mel, n_frames, load_feat_cfg
from spoofdet.eval.metrics import summarize
from spoofdet.models import build_model, REGISTRY


def test_eer_perfect_and_random():
    y = np.r_[np.zeros(500), np.ones(500)]
    assert compute_eer(np.r_[np.zeros(500), np.ones(500)] + 0.0, y)[0] == 0.0
    rng = np.random.RandomState(0)
    assert abs(compute_eer(rng.rand(4000), rng.randint(0, 2, 4000))[0] - 0.5) < 0.05


def test_eer_direction():
    y = np.r_[np.zeros(100), np.ones(100)]
    assert compute_eer(-(y.astype(float)), y)[0] > 0.9      # inverted scores -> ~100% EER


def test_fix_length():
    assert len(fix_length(np.ones(10, np.float32), 64000)) == 64000
    assert len(fix_length(np.ones(70000, np.float32), 64000)) == 64000
    assert fix_length(np.zeros(0, np.float32), 5).shape == (5,)


def test_mel_shape():
    cfg = load_feat_cfg("mel64_w400")
    m = wav_to_mel(np.random.randn(30000).astype(np.float32) * .1, cfg)
    assert m.shape == (64, n_frames(cfg)) == (64, 401)


def test_normalize():
    x = np.random.randn(64, 401).astype(np.float32) * 5 + 3
    assert abs(normalize(x, "utt").mean()) < 1e-3
    assert np.allclose(normalize(x, "utt_band").mean(1), 0, atol=1e-3)


def test_specaugment_shape():
    assert spec_augment(torch.randn(1, 64, 401)).shape == (1, 64, 401)


def test_models_contract():
    for name in REGISTRY:
        for n_mels in (40, 64, 128):
            m = build_model(name).eval()
            out = m(torch.randn(2, 1, n_mels, 401))
            assert out.shape == (2, 2), (name, n_mels)
            assert hasattr(m, "gradcam_layer")


def test_summarize_keys():
    rng = np.random.RandomState(0); y = rng.randint(0, 2, 400)
    s = summarize(rng.rand(400) * .5 + y * .5, y)
    for k in ("eer", "auc", "acc_at_eer", "tn", "fp", "fn", "tp"):
        assert k in s
