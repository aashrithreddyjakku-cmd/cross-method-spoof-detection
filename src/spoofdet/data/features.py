"""Waveform -> log-mel spectrogram, and the on-disk FeatureStore.

The store keeps RAW log-mel (dB). Normalisation + augmentation happen at load time
(see dataset.py), so one store serves many normalisation / augmentation experiments.
"""
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from ..common.io import load_yaml, save_yaml
from ..common import paths
from .audio import load_wave, fix_length, trim_silence


def load_feat_cfg(feat_id_or_path) -> dict:
    p = Path(feat_id_or_path)
    if not p.exists():
        p = paths.CONFIGS / "features" / f"{feat_id_or_path}.yaml"
    return load_yaml(p)


def n_frames(cfg: dict) -> int:
    n = int(round(cfg["sr"] * cfg["clip_seconds"]))
    return 1 + n // cfg["hop_length"]          # librosa center=True


def wav_to_mel(y: np.ndarray, cfg: dict) -> np.ndarray:
    """1-D float waveform (any length) -> log-mel in dB, shape (n_mels, T), float32."""
    if cfg.get("trim_silence", False):
        y = trim_silence(y, cfg.get("trim_top_db", 30.0))
    y = fix_length(y, int(round(cfg["sr"] * cfg["clip_seconds"])))
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg["sr"], n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
        win_length=cfg["win_length"], n_mels=cfg["n_mels"],
        fmin=cfg["f_min"], fmax=cfg["f_max"], power=cfg["power"], center=True)
    S_db = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=cfg["top_db"])
    return S_db.astype(np.float32)


def _work(args):
    path, cfg = args
    try:
        return wav_to_mel(load_wave(path, cfg["sr"]), cfg), None
    except Exception as e:                       # corrupt file: store zeros, report it
        return np.zeros((cfg["n_mels"], n_frames(cfg)), np.float32), repr(e)


class FeatureStore:
    """feats.npy  : float16 memmap (N, n_mels, T)
       index.csv  : utt_id,row
       feat_config.yaml : the config used"""

    def __init__(self, root):
        self.root = Path(root)
        self.cfg = load_yaml(self.root / "feat_config.yaml")
        idx = pd.read_csv(self.root / "index.csv", dtype={"utt_id": str})
        self.row = dict(zip(idx["utt_id"], idx["row"]))
        self._arr = None

    @property
    def arr(self):                               # lazy: each DataLoader worker opens its own memmap
        if self._arr is None:
            self._arr = np.load(self.root / "feats.npy", mmap_mode="r")
        return self._arr

    def __contains__(self, utt_id):
        return utt_id in self.row

    def __len__(self):
        return len(self.row)

    def get(self, utt_id) -> np.ndarray:
        return np.asarray(self.arr[self.row[utt_id]], dtype=np.float32)

    def __getstate__(self):
        d = self.__dict__.copy(); d["_arr"] = None
        return d

    @staticmethod
    def build(manifest_df: pd.DataFrame, cfg: dict, out_dir, raw_root=None,
              workers: int = 2, overwrite: bool = False):
        out = Path(out_dir)
        if (out / "feats.npy").exists() and not overwrite:
            print(f"[skip] {out} already exists (use overwrite=True / --overwrite)")
            return FeatureStore(out)
        out.mkdir(parents=True, exist_ok=True)
        raw_root = Path(raw_root or paths.RAW)
        df = manifest_df.drop_duplicates("utt_id").reset_index(drop=True)
        N, M, T = len(df), cfg["n_mels"], n_frames(cfg)
        arr = np.lib.format.open_memmap(out / "feats.npy", mode="w+",
                                        dtype=np.float16, shape=(N, M, T))
        jobs = [(str(raw_root / p), cfg) for p in df["path"]]
        failed = []
        with mp.Pool(workers) as pool:
            it = pool.imap(_work, jobs, chunksize=16)
            for i, (mel, err) in enumerate(tqdm(it, total=N, desc=cfg["feat_id"])):
                arr[i] = mel.astype(np.float16)
                if err:
                    failed.append((df.loc[i, "utt_id"], err))
        arr.flush()
        pd.DataFrame({"utt_id": df["utt_id"], "row": np.arange(N)}).to_csv(out / "index.csv", index=False)
        save_yaml(cfg, out / "feat_config.yaml")
        if failed:
            pd.DataFrame(failed, columns=["utt_id", "error"]).to_csv(out / "failed.csv", index=False)
            print(f"WARNING: {len(failed)} files failed -> {out/'failed.csv'}")
        return FeatureStore(out)
