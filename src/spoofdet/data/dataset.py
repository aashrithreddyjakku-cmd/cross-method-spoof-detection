"""PyTorch Dataset: split CSV + FeatureStore -> (1, n_mels, T) tensors, with normalisation + SpecAugment."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..common.io import read_manifest
from .features import FeatureStore


def compute_global_stats(split_csv, store: FeatureStore, max_items: int = 5000, seed: int = 0) -> dict:
    """Per-mel-band mean/std over (a sample of) the TRAIN split. Used when norm == 'global'."""
    df = read_manifest(split_csv)
    ids = df["utt_id"].sample(min(max_items, len(df)), random_state=seed).tolist()
    X = np.stack([store.get(u) for u in ids])               # (N, M, T)
    mean = X.mean(axis=(0, 2)); std = X.std(axis=(0, 2)) + 1e-5
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def normalize(mel: np.ndarray, norm: str, stats: dict = None) -> np.ndarray:
    """mel: (n_mels, T) dB.  norm in {'none','utt','utt_band','global'}"""
    if norm == "none":
        return mel
    if norm == "utt":                                       # one mean/std per utterance
        return (mel - mel.mean()) / (mel.std() + 1e-5)
    if norm == "utt_band":                                  # per-band CMVN over time
        return (mel - mel.mean(axis=1, keepdims=True)) / (mel.std(axis=1, keepdims=True) + 1e-5)
    if norm == "global":
        return (mel - stats["mean"][:, None]) / stats["std"][:, None]
    raise ValueError(f"unknown norm {norm}")


def spec_augment(x: torch.Tensor, freq_mask: int = 8, time_mask: int = 40, n_masks: int = 2) -> torch.Tensor:
    """x: (1, M, T). Fill random frequency/time stripes with the mean (0 after normalisation)."""
    _, M, T = x.shape
    x = x.clone()
    fill = x.mean()
    for _ in range(n_masks):
        f = np.random.randint(0, freq_mask + 1); f0 = np.random.randint(0, max(1, M - f))
        x[:, f0:f0 + f, :] = fill
        t = np.random.randint(0, time_mask + 1); t0 = np.random.randint(0, max(1, T - t))
        x[:, :, t0:t0 + t] = fill
    return x


class SpoofDataset(Dataset):
    def __init__(self, split_csv, store: FeatureStore, norm: str = "utt", stats: dict = None,
                 augment: dict = None):
        self.df = read_manifest(split_csv).reset_index(drop=True)
        missing = [u for u in self.df["utt_id"] if u not in store]
        if missing:
            raise KeyError(f"{len(missing)} utt_ids in {split_csv} are not in the feature store, "
                           f"e.g. {missing[:3]}. Re-run extract_features.py with a manifest that covers them.")
        self.store, self.norm, self.stats, self.augment = store, norm, stats, augment
        self.labels = self.df["label"].to_numpy()
        self.utt_ids = self.df["utt_id"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        mel = normalize(self.store.get(self.utt_ids[i]), self.norm, self.stats)
        x = torch.from_numpy(np.ascontiguousarray(mel)).unsqueeze(0)       # (1, M, T)
        if self.augment:
            x = spec_augment(x, **self.augment)
        return x, int(self.labels[i])
