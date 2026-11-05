"""Channel degradations for robustness test sets (stretch task S1)."""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa


def add_noise(y, snr_db, rng):
    p_sig = np.mean(y ** 2) + 1e-12
    noise = rng.randn(len(y)).astype(np.float32)
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / (np.mean(noise ** 2) + 1e-12))
    return y + noise


def telephony(y, sr=16000):
    """Down to 8 kHz (band-limits to <4 kHz) and back up."""
    lo = librosa.resample(y, orig_sr=sr, target_sr=8000)
    return librosa.resample(lo, orig_sr=8000, target_sr=sr)


def mp3_roundtrip(y, sr, bitrate="64k"):
    """Needs the ffmpeg command-line tool on PATH."""
    with tempfile.TemporaryDirectory() as d:
        src, mp3, back = Path(d, "a.wav"), Path(d, "a.mp3"), Path(d, "b.wav")
        sf.write(src, y, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", bitrate, str(mp3)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-ar", str(sr), str(back)], check=True)
        out, _ = sf.read(back, dtype="float32")
    return out[:, 0] if out.ndim == 2 else out


CONDITIONS = {
    "noise20": lambda y, sr, rng: add_noise(y, 20, rng),
    "noise10": lambda y, sr, rng: add_noise(y, 10, rng),
    "tel8k":   lambda y, sr, rng: telephony(y, sr),
    "mp3_64k": lambda y, sr, rng: mp3_roundtrip(y, sr, "64k"),
    "mp3_128k": lambda y, sr, rng: mp3_roundtrip(y, sr, "128k"),
}
