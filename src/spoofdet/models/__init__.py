from pathlib import Path

import torch

from .archs import SmallCNN, LCNN, ResNet18, VGGishCBAM

REGISTRY = {"smallcnn": SmallCNN, "lcnn": LCNN, "resnet18": ResNet18, "vggish_cbam": VGGishCBAM}


def build_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name}; choose from {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, model_name, model_kwargs, feat_cfg, norm, norm_stats, epoch, val_eer, seed):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "model_name": model_name, "model_kwargs": model_kwargs,
        "feat_cfg": feat_cfg, "norm": norm,
        "norm_stats": None if norm_stats is None else {k: v.tolist() for k, v in norm_stats.items()},
        "epoch": epoch, "val_eer": val_eer, "seed": seed}, path)


def load_checkpoint(path, device="cpu"):
    """Returns (model in eval mode, ckpt dict). ckpt['norm_stats'] is converted back to numpy arrays."""
    import numpy as np
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_name"], **(ck["model_kwargs"] or {}))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    if ck.get("norm_stats") is not None:
        ck["norm_stats"] = {k: np.asarray(v, dtype=np.float32) for k, v in ck["norm_stats"].items()}
    return model, ck
