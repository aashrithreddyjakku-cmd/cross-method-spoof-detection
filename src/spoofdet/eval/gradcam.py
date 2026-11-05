"""Grad-CAM and mel-band occlusion for the project's two-logit CNN contract."""

import numpy as np
import torch
import torch.nn.functional as F
import librosa


class GradCAM:
    """Compute class-specific Grad-CAM maps from a model's declared final feature layer.

    Model inputs have shape ``(B, 1, mel, time)`` and model outputs are two logits,
    where class 1 is spoof. The layer must return a 4-D convolutional feature map.
    """

    def __init__(self, model):
        self.model = model
        self.acts = None
        self.grads = None
        layer = getattr(model, "gradcam_layer", None)
        if not isinstance(layer, torch.nn.Module):
            raise ValueError("model must expose gradcam_layer as a torch.nn.Module")
        self._forward_handle = layer.register_forward_hook(self._capture_activations)

    def _capture_activations(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("gradcam_layer must return a 4-D tensor (B, C, H, W)")
        self.acts = output
        self.grads = None
        if output.requires_grad:
            output.register_hook(self._capture_gradients)

    def _capture_gradients(self, gradients):
        self.grads = gradients

    def close(self):
        """Remove the hook when the explainer is no longer needed."""
        self._forward_handle.remove()

    def __call__(self, x: torch.Tensor, target_class: int = 1):
        """Return normalized CAM ``(B, mel, time)`` and P(spoof) scores.

        The input is expected to already be normalized using its checkpoint's
        feature normalization settings and placed on the model's device.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Grad-CAM input must have shape (B, 1, mel, time)")
        if x.shape[0] == 0:
            raise ValueError("Grad-CAM input batch must not be empty")
        if target_class not in (0, 1):
            raise ValueError("target_class must be 0 (bona fide) or 1 (spoof)")
        model_device = next(self.model.parameters()).device
        if x.device != model_device:
            raise ValueError(f"input is on {x.device}, but model is on {model_device}")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape != (x.shape[0], 2):
            raise ValueError("model must return logits with shape (B, 2)")
        logits[:, target_class].sum().backward()
        if self.acts is None or self.grads is None:
            raise RuntimeError("Grad-CAM gradients were not captured; check gradcam_layer and model graph")
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        maximum = cam.flatten(1).amax(dim=1).clamp_min(1e-8).view(-1, 1, 1)
        return (cam / maximum).detach().cpu().numpy(), torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()


@torch.no_grad()
def band_occlusion(model, x: torch.Tensor, n_bands: int = 8, fill: str = "mean"):
    """Return per-band decrease in P(spoof), positive when a band supports spoof.

    Bands are contiguous, non-overlapping mel-bin groups. Mean replacement uses
    each utterance's own normalized mean, which corresponds to neutral zero under
    the common utterance normalization.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("occlusion input must have shape (B, 1, mel, time)")
    if x.shape[0] == 0:
        raise ValueError("occlusion input batch must not be empty")
    if isinstance(n_bands, bool) or not isinstance(n_bands, int) or not 1 <= n_bands <= x.shape[2]:
        raise ValueError("n_bands must be an integer between 1 and the number of mel bins")
    if fill not in ("mean", "min"):
        raise ValueError("fill must be 'mean' or 'min'")
    device = next(model.parameters()).device
    if x.device != device:
        raise ValueError(f"input is on {x.device}, but model is on {device}")
    model.eval()
    logits = model(x)
    if logits.ndim != 2 or logits.shape != (x.shape[0], 2):
        raise ValueError("model must return logits with shape (B, 2)")
    base = torch.softmax(logits, dim=1)[:, 1]
    edges = np.linspace(0, x.shape[2], n_bands + 1, dtype=int)
    values = []
    for band in range(n_bands):
        masked = x.clone()
        mel_slice = slice(edges[band], edges[band + 1])
        if fill == "mean":
            replacement = x.mean(dim=(2, 3), keepdim=True)
        else:
            replacement = x.amin(dim=(2, 3), keepdim=True)
        masked[:, :, mel_slice, :] = replacement
        masked_score = torch.softmax(model(masked), dim=1)[:, 1]
        values.append((base - masked_score).cpu().numpy())
    return np.stack(values, axis=1)


def band_edges_hz(feat_cfg: dict, n_bands: int = 8):
    """Approximate Hz boundaries corresponding to equal groups of mel bins."""
    n_mels = int(feat_cfg["n_mels"])
    if isinstance(n_bands, bool) or not isinstance(n_bands, int) or not 1 <= n_bands <= n_mels:
        raise ValueError("n_bands must be an integer between 1 and n_mels")
    if "f_min" not in feat_cfg or "f_max" not in feat_cfg:
        raise ValueError("feature config must include f_min and f_max")
    centers = librosa.mel_frequencies(n_mels=n_mels, fmin=feat_cfg["f_min"], fmax=feat_cfg["f_max"])
    indices = np.linspace(0, n_mels, n_bands + 1, dtype=int)
    boundaries = np.empty(n_bands + 1, dtype=float)
    boundaries[0], boundaries[-1] = feat_cfg["f_min"], feat_cfg["f_max"]
    for i in range(1, n_bands):
        boundaries[i] = (centers[indices[i] - 1] + centers[indices[i]]) / 2
    return [(float(boundaries[i]), float(boundaries[i + 1])) for i in range(n_bands)]
