import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, k=3, s=1, p=None):
    p = k // 2 if p is None else p
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MFM(nn.Module):
    """Max-Feature-Map: split channels in two halves and take the element-wise max (LCNN)."""
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel attention then spatial attention)."""
    def __init__(self, ch, reduction=8, k=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(ch, max(ch // reduction, 4)), nn.ReLU(inplace=True),
                                 nn.Linear(max(ch // reduction, 4), ch))
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        b, c, _, _ = x.shape
        att = torch.sigmoid(self.mlp(x.mean((2, 3))) + self.mlp(x.amax((2, 3)))).view(b, c, 1, 1)
        x = x * att
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(s))
