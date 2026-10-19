"""All CNN architectures. Contract for EVERY model:
   input  : float tensor (B, 1, n_mels, T)
   output : logits (B, 2)   [class 1 = spoof]
   attrs  : .gradcam_layer  = the nn.Module whose output is the last conv feature map (for Grad-CAM)
"""
import torch
import torch.nn as nn

from .blocks import conv_bn_relu, MFM, CBAM


class SmallCNN(nn.Module):
    def __init__(self, base=16, dropout=0.3, n_classes=2):
        super().__init__()
        ch = [1, base, base * 2, base * 4, base * 8]
        layers = []
        for i in range(4):
            layers += [conv_bn_relu(ch[i], ch[i + 1]), nn.MaxPool2d(2) if i < 3 else nn.Identity()]
        self.features = nn.Sequential(*layers)
        self.gradcam_layer = self.features[-2]            # last conv block (before the identity)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch[-1], n_classes))

    def forward(self, x):
        f = self.features(x)
        return self.head(f.mean((2, 3)))


class LCNN(nn.Module):
    """LCNN-style network with Max-Feature-Map activations (Wu et al. 2018; common ASVspoof baseline)."""
    def __init__(self, dropout=0.5, n_classes=2):
        super().__init__()
        def mfm_conv(cin, cout, k, p):
            return nn.Sequential(nn.Conv2d(cin, 2 * cout, k, 1, p), MFM())
        self.features = nn.Sequential(
            mfm_conv(1, 32, 5, 2), nn.MaxPool2d(2, 2),
            mfm_conv(32, 32, 1, 0), nn.BatchNorm2d(32),
            mfm_conv(32, 48, 3, 1), nn.MaxPool2d(2, 2), nn.BatchNorm2d(48),
            mfm_conv(48, 48, 1, 0), nn.BatchNorm2d(48),
            mfm_conv(48, 64, 3, 1), nn.MaxPool2d(2, 2),
            mfm_conv(64, 64, 1, 0), nn.BatchNorm2d(64),
            mfm_conv(64, 32, 3, 1), nn.BatchNorm2d(32),
            mfm_conv(32, 32, 1, 0), nn.BatchNorm2d(32),
            mfm_conv(32, 32, 3, 1), nn.MaxPool2d(2, 2),
        )
        self.gradcam_layer = self.features[-2]
        self.fc = nn.Sequential(nn.Linear(32, 160), MFM(), nn.BatchNorm1d(80), nn.Dropout(dropout),
                                nn.Linear(80, n_classes))

    def forward(self, x):
        return self.fc(self.features(x).mean((2, 3)))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.b2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.short = None if (stride == 1 and cin == cout) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = self.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return self.relu(o + (x if self.short is None else self.short(x)))


class ResNet18(nn.Module):
    """ResNet-18 layout [2,2,2,2], 1-channel stem, width multiplier (width=32 -> half-width, faster)."""
    def __init__(self, width=32, dropout=0.3, n_classes=2):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stem = nn.Sequential(nn.Conv2d(1, w[0], 3, 2, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True))
        blocks, cin = [], w[0]
        for i, cout in enumerate(w):
            for j in range(2):
                blocks.append(BasicBlock(cin, cout, 2 if (j == 0 and i > 0) else 1)); cin = cout
        self.layers = nn.Sequential(*blocks)
        self.gradcam_layer = self.layers[-1]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(w[-1], n_classes))

    def forward(self, x):
        return self.head(self.layers(self.stem(x)).mean((2, 3)))


class VGGishCBAM(nn.Module):
    """VGG-style stack (2 convs per block) + CBAM attention after the last block (cf. Kanwal et al. 2024)."""
    def __init__(self, base=16, dropout=0.4, n_classes=2):
        super().__init__()
        ch = [1, base, base * 2, base * 4, base * 8]
        layers = []
        for i in range(4):
            layers += [conv_bn_relu(ch[i], ch[i + 1]), conv_bn_relu(ch[i + 1], ch[i + 1])]
            if i < 3:
                layers.append(nn.MaxPool2d(2))
        layers.append(CBAM(ch[-1]))
        self.features = nn.Sequential(*layers)
        self.gradcam_layer = self.features[-1]
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(ch[-1], n_classes))

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))
