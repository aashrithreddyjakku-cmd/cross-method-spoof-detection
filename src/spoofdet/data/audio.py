"""Audio loading and fixed-length handling."""
import numpy as np
import soundfile as sf
import librosa

from ..common.constants import SR


def load_wave(path, sr: int = SR) -> np.ndarray:
    """Read any wav/flac -> mono float32 in [-1, 1] at `sr` Hz."""
    y, orig_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim == 2:                       # stereo -> mono
        y = y.mean(axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return np.ascontiguousarray(y, dtype=np.float32)


def fix_length(y: np.ndarray, n: int) -> np.ndarray:
    """Crop to the first n samples, or tile (repeat) if shorter. Empty audio -> zeros."""
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(y) >= n:
        return y[:n]
    reps = int(np.ceil(n / len(y)))
    return np.tile(y, reps)[:n]


def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Remove leading/trailing silence (energy based). Keeps y if everything would be removed."""
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    return yt if len(yt) > 0 else y
