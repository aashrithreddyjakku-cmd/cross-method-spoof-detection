from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from spoofdet.eval.gradcam import GradCAM, band_edges_hz, band_occlusion
from spoofdet.models import build_model
from spoofdet.models import save_checkpoint
from scripts import run_xai as xai_script
from scripts.run_xai import _select_samples, validate_unseen_split


@pytest.fixture(scope="module", autouse=True)
def _limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name,kwargs", [
    ("smallcnn", {"base": 4, "dropout": 0}),
    ("lcnn", {"dropout": 0}),
    ("resnet18", {"width": 4, "dropout": 0}),
    ("vggish_cbam", {"base": 4, "dropout": 0}),
])
def test_gradcam_works_for_registered_cnn_architectures(name, kwargs):
    model = build_model(name, **kwargs).eval()
    x = torch.randn(2, 1, 64, 64)
    explainer = GradCAM(model)
    try:
        maps, scores = explainer(x)
    finally:
        explainer.close()
    assert maps.shape == (2, 64, 64)
    assert scores.shape == (2,)
    assert np.isfinite(maps).all() and np.isfinite(scores).all()
    assert maps.min() >= 0 and maps.max() <= 1 + 1e-6
    assert ((scores >= 0) & (scores <= 1)).all()


def test_gradcam_validates_contract_and_model_device():
    model = build_model("smallcnn", base=4)
    cam = GradCAM(model)
    try:
        with pytest.raises(ValueError, match="shape"):
            cam(torch.randn(1, 64, 64))
        with pytest.raises(ValueError, match="target_class"):
            cam(torch.randn(1, 1, 64, 64), 2)
    finally:
        cam.close()


def test_band_occlusion_outputs_spoof_score_drops_and_validates_bands():
    model = build_model("smallcnn", base=4).eval()
    x = torch.randn(3, 1, 16, 20)
    values = band_occlusion(model, x, n_bands=4)
    assert values.shape == (3, 4)
    assert np.isfinite(values).all()
    with pytest.raises(ValueError, match="n_bands"):
        band_occlusion(model, x, n_bands=17)
    with pytest.raises(ValueError, match="fill"):
        band_occlusion(model, x, fill="zero")


def test_band_edges_cover_feature_frequency_range():
    edges = band_edges_hz({"n_mels": 64, "f_min": 20, "f_max": 8000}, 8)
    assert len(edges) == 8
    assert edges[0][0] == 20
    assert edges[-1][1] == 8000
    assert all(left[1] == pytest.approx(right[0]) for left, right in zip(edges, edges[1:]))


def _unseen_frame():
    return pd.DataFrame([
        {"utt_id": "b1", "label": 0, "method_id": "bonafide", "method_family": "bonafide"},
        {"utt_id": "s1", "label": 1, "method_id": "A10", "method_family": "neural_vocoder"},
        {"utt_id": "s2", "label": 1, "method_id": "A10", "method_family": "neural_vocoder"},
    ])


def test_xai_split_selector_accepts_only_correct_unseen_test_fold():
    path = Path("splits/P1_lomo/A10/test_unseen.csv")
    kind, held = validate_unseen_split(path, _unseen_frame())
    assert (kind, held) == ("P1_lomo", "A10")
    with pytest.raises(ValueError, match="requires splits"):
        validate_unseen_split(Path("splits/P1_lomo/A10/val.csv"), _unseen_frame())
    wrong_method = _unseen_frame()
    wrong_method.loc[wrong_method.label == 1, "method_id"] = "A01"
    with pytest.raises(ValueError, match="held-out method_id"):
        validate_unseen_split(path, wrong_method)


def test_xai_split_selector_validates_lofo_family_and_class_presence():
    path = Path("splits/P1_lofo/neural_vocoder/test_unseen.csv")
    assert validate_unseen_split(path, _unseen_frame()) == ("P1_lofo", "neural_vocoder")
    no_bona = _unseen_frame().query("label == 1")
    with pytest.raises(ValueError, match="bona fide and spoof"):
        validate_unseen_split(path, no_bona)


def test_xai_sampling_caps_each_spoof_method_and_control_group():
    frame = pd.concat([_unseen_frame(), _unseen_frame().assign(utt_id=lambda x: x.utt_id + "x",
                                                               method_id=lambda x: x.method_id.replace({"A10": "A11"}),
                                                               method_family=lambda x: x.method_family.replace({"neural_vocoder": "griffin_lim"}))],
                      ignore_index=True)
    selected = _select_samples(frame, n=1)
    assert selected[selected.label == 1].groupby("method_id").size().to_dict() == {"A10": 1, "A11": 1}
    assert (selected.label == 0).sum() == 1


def test_run_xai_saves_unseen_fold_tables_and_figures(tmp_path, monkeypatch):
    split = tmp_path / "splits" / "P1_lomo" / "A10" / "test_unseen.csv"
    split.parent.mkdir(parents=True)
    frame = pd.DataFrame([
        {"utt_id": "b1", "dataset": "la19", "partition": "eval", "path": "b1.wav", "label": 0,
         "attack_id": "bonafide", "method_id": "bonafide", "method_family": "bonafide",
         "speaker_id": "sp1", "duration_s": 4.0, "orig_sr": 16000},
        {"utt_id": "s1", "dataset": "la19", "partition": "eval", "path": "s1.wav", "label": 1,
         "attack_id": "A10", "method_id": "A10", "method_family": "neural_vocoder",
         "speaker_id": "sp2", "duration_s": 4.0, "orig_sr": 16000},
    ])
    frame.to_csv(split, index=False)
    ckpt_root = tmp_path / "checkpoints"
    table_root, figure_root = tmp_path / "tables", tmp_path / "figures"
    config_root, feature_root = tmp_path / "configs", tmp_path / "features"
    config_root.mkdir()
    monkeypatch.setattr(xai_script.paths, "CKPT", ckpt_root)
    monkeypatch.setattr(xai_script.paths, "TABLES", table_root)
    monkeypatch.setattr(xai_script.paths, "FIGURES", figure_root)
    monkeypatch.setattr(xai_script.paths, "CONFIGS", config_root)
    monkeypatch.setattr(xai_script.paths, "FEATURES", feature_root)
    exp = "lomo_A10_smallcnn_s1"
    exp_dir = ckpt_root / exp
    exp_dir.mkdir(parents=True)
    (exp_dir / "config.yaml").write_text("protocol: P1_lomo/A10\n", encoding="utf-8")
    feat_cfg = {"feat_id": "fixture", "n_mels": 8, "f_min": 20, "f_max": 8000}
    model = build_model("smallcnn", base=4, dropout=0)
    save_checkpoint(exp_dir / "best.pt", model, "smallcnn", {"base": 4, "dropout": 0},
                    feat_cfg, "utt", None, 1, 0.2, 1)

    features = {uid: np.random.default_rng(i).normal(size=(8, 32)).astype(np.float32)
                for i, uid in enumerate(frame.utt_id)}

    class FixtureFeatureStore:
        def __init__(self, _path):
            pass

        def __contains__(self, uid):
            return uid in features

        def get(self, uid):
            return features[uid]

    monkeypatch.setattr(xai_script, "FeatureStore", FixtureFeatureStore)
    outputs = xai_script.run_xai(exp, split, n=1, bands=4)
    assert len(outputs) == 5
    assert all(Path(p).is_file() for p in outputs)
    per_utterance = pd.read_csv(table_root / f"xai_{exp}_per_utterance.csv")
    assert set(per_utterance.label) == {0, 1}
    summary = pd.read_csv(table_root / f"xai_{exp}_band_occlusion.csv")
    assert "band0_drop_mean" in summary
