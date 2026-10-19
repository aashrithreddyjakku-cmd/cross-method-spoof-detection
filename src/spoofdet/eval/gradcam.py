"""Explainability: Grad-CAM and mel-band occlusion importance, for any model following the model contract."""
import numpy as np
import torch
import torch.nn.functional as F
import librosa


class GradCAM:
    def __init__(self, model):
        self.model, self.acts, self.grads = model, None, None
        layer = model.gradcam_layer
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x: torch.Tensor, target_class: int = 1):
        """x: (B,1,M,T) normalised input. Returns (cam (B,M,T) in [0,1], scores P(spoof) (B,))."""
        self.model.eval(); x = x.clone().requires_grad_(True)
        self.model.zero_grad()
        logits = self.model(x)
        logits[:, target_class].sum().backward()
        w = self.grads.mean((2, 3), keepdim=True)                     # channel weights
        cam = F.relu((w * self.acts).sum(1, keepdim=True))            # (B,1,h,w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        mx = cam.flatten(1).max(1)[0].clamp_min(1e-8).view(-1, 1, 1)
        return (cam / mx).detach().cpu().numpy(), torch.softmax(logits, 1)[:, 1].detach().cpu().numpy()


@torch.no_grad()
def band_occlusion(model, x: torch.Tensor, n_bands: int = 8, fill: str = "mean"):
    """Drop in P(spoof) when one contiguous group of mel bands is replaced by its mean.
    x: (B,1,M,T). Returns (B, n_bands) array: positive = band supports the 'spoof' decision."""
    model.eval()
    base = torch.softmax(model(x), 1)[:, 1]
    M = x.shape[2]; edges = np.linspace(0, M, n_bands + 1).astype(int)
    out = []
    for b in range(n_bands):
        xo = x.clone()
        sl = slice(edges[b], edges[b + 1])
        xo[:, :, sl, :] = xo.mean(dim=(2, 3), keepdim=True) if fill == "mean" else xo.min()
        out.append((base - torch.softmax(model(xo), 1)[:, 1]).cpu().numpy())
    return np.stack(out, axis=1)


def band_edges_hz(feat_cfg: dict, n_bands: int = 8):
    """Hz range covered by each occlusion band (for axis labels)."""
    freqs = librosa.mel_frequencies(n_mels=feat_cfg["n_mels"], fmin=feat_cfg["f_min"], fmax=feat_cfg["f_max"])
    edges = np.linspace(0, feat_cfg["n_mels"] - 1, n_bands + 1).astype(int)
    return [(float(freqs[edges[i]]), float(freqs[edges[i + 1]])) for i in range(n_bands)]
