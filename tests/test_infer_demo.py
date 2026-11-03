from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from spoofdet.app.demo import build_app
from spoofdet.infer import Detector
from spoofdet.models import build_model, save_checkpoint


@pytest.fixture(scope="module", autouse=True)
def _limit_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def detector(tmp_path):
    model = build_model("smallcnn", base=4, dropout=0)
    feat_cfg = {
        "feat_id": "mel64_w400", "sr": 16000, "clip_seconds": 4.0,
        "n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 64,
        "f_min": 20, "f_max": 8000, "power": 2.0, "top_db": 80.0,
        "trim_silence": False,
    }
    checkpoint = tmp_path / "checkpoints" / "fixture" / "best.pt"
    save_checkpoint(checkpoint, model, "smallcnn", {"base": 4, "dropout": 0},
                    feat_cfg, "utt", None, 1, 0.2, 1)
    return Detector(checkpoint)


def test_missing_checkpoint_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Member B checkpoint"):
        Detector(tmp_path / "missing" / "best.pt")


def test_detector_uses_shared_resample_mono_and_four_second_mel_pipeline(detector, tmp_path):
    sr = 8000
    mono = np.sin(2 * np.pi * 440 * np.arange(sr, dtype=np.float32) / sr)
    stereo = np.column_stack([mono, mono * 0.5])
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, stereo, sr)
    mel = detector.mel(audio_path)
    assert mel.shape == (64, 401)
    assert np.isfinite(mel).all()
    x = detector.tensor(mel)
    assert x.shape == (1, 1, 64, 401)
    score = detector.score(audio_path)
    assert 0 <= score <= 1


def test_detector_accepts_waveform_and_gradio_numpy_tuple(detector):
    waveform = np.zeros(16000, dtype=np.float32)
    assert detector.mel(waveform).shape == (64, 401)
    assert detector.mel((8000, waveform[:8000])).shape == (64, 401)
    assert 0 <= detector.score(waveform, sr=16000) <= 1


def test_detector_reports_invalid_audio_and_incompatible_mel(detector, tmp_path):
    unsupported = tmp_path / "clip.ogg"
    unsupported.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="unsupported audio format"):
        detector.mel(unsupported)
    corrupt = tmp_path / "broken.wav"
    corrupt.write_bytes(b"not a wave")
    with pytest.raises(ValueError, match="could not decode"):
        detector.mel(corrupt)
    with pytest.raises(ValueError, match="empty"):
        detector.mel(np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        detector.tensor(np.zeros((40, 10), dtype=np.float32))


def test_demo_builds_with_explicit_missing_checkpoint_message():
    app = build_app(None)
    config = app.get_config_file()
    markdown = [component.get("props", {}).get("value", "")
                for component in config["components"] if component.get("type") == "markdown"]
    assert any("Trained checkpoint not found" in str(value) and "No prediction has been made" in str(value)
               for value in markdown)
    assert app.fns  # interface callbacks are registered, but inference remains disabled


def test_demo_rejects_missing_checkpoint_path_without_fake_prediction(tmp_path):
    app = build_app(tmp_path / "no" / "best.pt")
    config = app.get_config_file()
    markdown = [str(component.get("props", {}).get("value", ""))
                for component in config["components"] if component.get("type") == "markdown"]
    assert any("checkpoints/<exp_id>/best.pt" in value for value in markdown)
