"""Single-file inference using the exact audio and feature pipeline used in training."""

import os
from pathlib import Path

import librosa
import numpy as np
import torch

from .common.constants import SR
from .data.audio import load_wave
from .data.dataset import normalize
from .data.features import wav_to_mel
from .models import load_checkpoint

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3"}


class Detector:
    """Load one trained checkpoint and score clips as ``P(spoof)``.

    Audio paths support WAV, FLAC, and MP3. Array input is mono or sample-major
    multichannel audio; its sample rate can be supplied with ``sr``.
    """

    def __init__(self, ckpt_path, device="cpu"):
        self.ckpt_path = Path(ckpt_path).expanduser() if ckpt_path is not None else None
        if self.ckpt_path is None or not self.ckpt_path.is_file():
            shown = self.ckpt_path if self.ckpt_path is not None else "<not specified>"
            raise FileNotFoundError(
                f"trained checkpoint not found: {shown}. Expected a Member B checkpoint such as "
                "checkpoints/<exp_id>/best.pt. Train an experiment before running inference."
            )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested ({self.device}), but CUDA is unavailable")
        try:
            self.model, self.ck = load_checkpoint(self.ckpt_path, str(self.device))
        except Exception as exc:
            raise RuntimeError(f"could not load compatible checkpoint {self.ckpt_path}: {exc}") from exc
        self.cfg = self.ck.get("feat_cfg")
        self._validate_checkpoint()

    def _validate_checkpoint(self):
        if not isinstance(self.cfg, dict):
            raise ValueError(f"checkpoint {self.ckpt_path} is missing feat_cfg")
        required = {"feat_id", "sr", "clip_seconds", "n_mels", "n_fft", "win_length",
                    "hop_length", "f_min", "f_max", "power", "top_db"}
        missing = sorted(required - self.cfg.keys())
        if missing:
            raise ValueError(f"checkpoint {self.ckpt_path} has incomplete feat_cfg: missing {missing}")
        if int(self.cfg["sr"]) != SR or float(self.cfg["clip_seconds"]) != 4.0:
            raise ValueError(
                f"checkpoint audio contract must be {SR} Hz and 4 seconds; got "
                f"{self.cfg['sr']} Hz and {self.cfg['clip_seconds']} seconds"
            )
        if "norm" not in self.ck:
            raise ValueError(f"checkpoint {self.ckpt_path} is missing its normalization setting")
        if not hasattr(self.model, "gradcam_layer"):
            # Inference remains usable for compatible models without XAI support.
            self.gradcam_supported = False
        else:
            self.gradcam_supported = True

    def mel(self, path_or_wave, sr=None):
        """Return the project's raw log-mel features after shared preprocessing."""
        if isinstance(path_or_wave, (str, bytes, Path)) or hasattr(path_or_wave, "__fspath__"):
            path = Path(os.fsdecode(path_or_wave))
            if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
                allowed = ", ".join(sorted(suffix.lstrip(".") for suffix in SUPPORTED_AUDIO_SUFFIXES))
                raise ValueError(f"unsupported audio format {path.suffix or '<no extension>'}; use {allowed}")
            if not path.is_file():
                raise FileNotFoundError(f"audio file not found: {path}")
            try:
                waveform = load_wave(path, int(self.cfg["sr"]))
            except Exception as exc:
                raise ValueError(f"could not decode audio file {path}: {exc}") from exc
        else:
            source_sr = sr
            waveform = path_or_wave
            # Gradio's numpy audio form is (sample_rate, samples).
            if isinstance(path_or_wave, tuple) and len(path_or_wave) == 2:
                source_sr, waveform = path_or_wave
            try:
                waveform = np.asarray(waveform, dtype=np.float32)
            except (TypeError, ValueError) as exc:
                raise ValueError("audio must be a readable file path or numeric waveform") from exc
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            if waveform.ndim != 1:
                raise ValueError("audio waveform must be mono or sample-major multichannel audio")
            if waveform.size == 0:
                raise ValueError("audio waveform is empty")
            if not np.isfinite(waveform).all():
                raise ValueError("audio waveform contains NaN or infinite values")
            source_sr = int(source_sr or self.cfg["sr"])
            if source_sr <= 0:
                raise ValueError("audio sample rate must be a positive integer")
            if source_sr != int(self.cfg["sr"]):
                waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=int(self.cfg["sr"]))
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError("audio contains no valid finite samples")
        # wav_to_mel applies configured silence trimming, crop/repeat to 4 s,
        # and the exact same librosa log-mel settings used by feature extraction.
        return wav_to_mel(waveform, self.cfg)

    def tensor(self, mel):
        """Apply checkpoint normalization and move ``(1, M, T)`` input to its device."""
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim != 2 or mel.shape[0] != int(self.cfg["n_mels"]):
            raise ValueError(f"mel input must have shape ({self.cfg['n_mels']}, time)")
        if not np.isfinite(mel).all():
            raise ValueError("mel input contains NaN or infinite values")
        normalized = normalize(mel, self.ck["norm"], self.ck.get("norm_stats"))
        if not np.isfinite(normalized).all():
            raise ValueError("checkpoint normalization produced non-finite features")
        return torch.from_numpy(np.ascontiguousarray(normalized)).float()[None, None].to(self.device)

    @torch.no_grad()
    def score_tensor(self, x) -> float:
        """Score a normalized model input, returning the frozen P(spoof) contract."""
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 1:
            raise ValueError("model input must have shape (1, 1, mel, time)")
        logits = self.model(x.to(self.device))
        if not isinstance(logits, torch.Tensor) or logits.shape != (1, 2):
            raise ValueError("checkpoint model must return two logits: bona fide (0), spoof (1)")
        score = float(torch.softmax(logits, dim=1)[0, 1])
        if not np.isfinite(score):
            raise ValueError("model returned a non-finite spoof probability")
        return score

    def score(self, path_or_wave, sr=None) -> float:
        """Return P(spoof); higher values indicate a spoof prediction."""
        return self.score_tensor(self.tensor(self.mel(path_or_wave, sr=sr)))
