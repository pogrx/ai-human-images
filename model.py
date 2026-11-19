"""Neural network architectures for AI-generated vs. authentic image detection.

Every model takes a batch of RGB pixel tensors (N, 3, H, W) scaled to [0, 1] and
returns one raw logit per image. Apply ``torch.sigmoid`` to the logit to get the
confidence that the image is AI-generated (1.0 = certainly AI, 0.0 = certainly real).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HighPassResiduals(nn.Module):
    """Fixed (non-learned) high-pass filters, applied to each colour channel.

    Image generators leave their fingerprints mostly in fine, high-frequency
    detail (upsampling grids, unnatural noise statistics). Suppressing the
    image content and exposing these residuals up front makes them much easier
    for the network to pick up.
    """

    def __init__(self):
        super().__init__()
        k1 = torch.tensor([[0, 0, 0, 0, 0],
                           [0, -1, 2, -1, 0],
                           [0, 2, -4, 2, 0],
                           [0, -1, 2, -1, 0],
                           [0, 0, 0, 0, 0]], dtype=torch.float32) / 4.0
        k2 = torch.tensor([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
        k3 = torch.tensor([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 1, -2, 1, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]], dtype=torch.float32) / 2.0
        kernels = torch.stack([k1, k2, k3])                     # (3, 5, 5)
        # Apply each of the 3 kernels to each of the 3 RGB channels -> 9 maps.
        weight = kernels.unsqueeze(1).repeat(3, 1, 1, 1)         # (9, 1, 5, 5)
        self.register_buffer("weight", weight)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2, groups=3)


def conv_bn(c_in, c_out, stride=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ForensicCNN(nn.Module):
    """Compact, fully convolutional detector that trains reasonably on a CPU.

    The first layer runs at full resolution (no pooling) on the pixels plus
    their high-pass residuals, so pixel-level artifacts survive; later layers
    downsample with strided convolutions rather than pooling. Channel counts
    stay small while the feature maps are large, so a training step is cheaper
    than ResNet-18's on a CPU. Global average + max pooling at the end
    lets it accept any input size.
    """

    def __init__(self, width=16):
        super().__init__()
        w = width
        self.highpass = HighPassResiduals()
        self.features = nn.Sequential(
            conv_bn(3 + 9, w),                 # full resolution
            conv_bn(w, 2 * w, stride=2),       # 1/2
            conv_bn(2 * w, 2 * w),
            conv_bn(2 * w, 4 * w, stride=2),   # 1/4
            conv_bn(4 * w, 4 * w),
            conv_bn(4 * w, 8 * w, stride=2),   # 1/8
            conv_bn(8 * w, 8 * w),
            conv_bn(8 * w, 16 * w, stride=2),  # 1/16
            conv_bn(16 * w, 16 * w),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32 * w, 1),
        )

    def forward(self, x):
        x = (x - 0.5) / 0.25
        x = torch.cat([x, self.highpass(x)], dim=1)
        x = self.features(x)
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.head(pooled).squeeze(1)


class ResNetDetector(nn.Module):
    """ResNet-18 backbone, optionally starting from ImageNet weights.

    Slower on a CPU than ForensicCNN but usually generalizes better to
    generators it has not seen, given enough training images.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std).squeeze(1)


ARCHITECTURES = ("forensic", "resnet18")


def build_model(arch, pretrained=True):
    if arch == "forensic":
        return ForensicCNN()
    if arch == "resnet18":
        return ResNetDetector(pretrained=pretrained)
    raise ValueError(f"Unknown architecture {arch!r}; choose from {ARCHITECTURES}")
