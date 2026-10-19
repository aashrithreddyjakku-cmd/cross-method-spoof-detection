"""Single-file inference used by the demo app:  wav/flac/mp3 path -> verdict + features."""
import numpy as np
import torch

from .data.audio import load_wave
from .data.dataset import normalize
from .data.features import wav_to_mel
from .models import load_checkpoint


class Detector:
    def __init__(self, ckpt_path, device="cpu"):
        self.device = device
        self.model, self.ck = load_checkpoint(ckpt_path, device)
        self.cfg = self.ck["feat_cfg"]

    def mel(self, path_or_wave, sr=None):
        y = load_wave(path_or_wave, self.cfg["sr"]) if isinstance(path_or_wave, (str, bytes)) or hasattr(path_or_wave, "__fspath__") else path_or_wave
        return wav_to_mel(y, self.cfg)

    def tensor(self, mel):
        m = normalize(mel, self.ck["norm"], self.ck["norm_stats"])
        return torch.from_numpy(np.ascontiguousarray(m)).float()[None, None].to(self.device)

    @torch.no_grad()
    def score(self, path_or_wave) -> float:
        x = self.tensor(self.mel(path_or_wave))
        return float(torch.softmax(self.model(x), 1)[0, 1])
