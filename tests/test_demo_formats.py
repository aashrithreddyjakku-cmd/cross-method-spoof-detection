"""Demo input coverage (playbook C.13: tested with 5 kinds of audio) plus the threshold wiring."""
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from spoofdet.app.demo import build_app, validation_threshold
from spoofdet.infer import Detector
from spoofdet.models import build_model, save_checkpoint


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    cfg = {"feat_id": "mel64_w400", "sr": 16000, "clip_seconds": 4.0, "n_fft": 512, "win_length": 400, "hop_length": 160,
           "n_mels": 64, "f_min": 20, "f_max": 8000, "power": 2.0, "top_db": 80.0, "trim_silence": False}
    save_checkpoint(d / "best.pt", build_model("smallcnn", base=4, dropout=0), "smallcnn", {"base": 4, "dropout": 0}, cfg, "utt", None, 1, 0.2, 1)
    return Detector(d / "best.pt")


def tone(seconds, sr, channels=1):
    y = (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * sr)) / sr)).astype(np.float32)
    return np.column_stack([y] * channels) if channels > 1 else y


@pytest.mark.parametrize("name,seconds,sr,channels", [
    ("wav_16k.wav", 4, 16000, 1), ("flac_16k.flac", 3, 16000, 1), ("mp3_44k.mp3", 5, 44100, 2), ("stereo_48k.wav", 2, 48000, 2),
    ("phone_8k.wav", 3, 8000, 1), ("very_short.wav", 0.2, 16000, 1), ("long_30s.flac", 30, 16000, 1)])
def test_demo_audio_kinds(detector, tmp_path, name, seconds, sr, channels):
    p = tmp_path / name
    sf.write(p, tone(seconds, sr, channels), sr)
    assert detector.mel(p).shape == (64, 401)
    assert 0 <= detector.score(p) <= 1


def test_digital_silence_does_not_crash(detector, tmp_path):
    p = tmp_path / "silence.wav"
    sf.write(p, np.zeros(16000, np.float32), 16000)
    assert 0 <= detector.score(p) <= 1


def test_threshold_comes_from_validation_scores(tmp_path):
    root = tmp_path / "results" / "scores" / "x"; root.mkdir(parents=True)
    pd.DataFrame(dict(score=[0.1, 0.2, 0.8, 0.9], label=[0, 0, 1, 1])).to_csv(root / "val.csv", index=False)
    assert 0.2 <= validation_threshold(tmp_path / "checkpoints" / "x" / "best.pt", tmp_path / "results" / "scores") <= 0.8
    assert validation_threshold(tmp_path / "checkpoints" / "none" / "best.pt", tmp_path / "results" / "scores") is None


def test_app_builds_with_custom_threshold():
    assert build_app(None, threshold=0.12).fns
