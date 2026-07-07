import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import BasicBlock, PResNet, ResNet_cfg, donwload_url
from .common import get_activation, ConvNormLayer
from src.core import register

__all__ = ['PercepConv', 'PercepConvBasicBlock', 'PResNet_PercepConv']


class PercepConv(nn.Module):
    """Perceptual Convolution (PercepConv) Module

    Paper: "MSA-DETR: A Multi-Scale Attention Augmented Model for Small Object Detection in UAV Imagery"
    Authors: Li and Qi (2026)

    PercepConv is designed to enhance feature representation for small object detection in UAV imagery.
    It combines multi-scale convolution with perceptual attention mechanisms.

    Architecture:
    1. Multi-scale Depthwise Convolution Branch:
       - Uses 3x3, 5x5, 7x7 depthwise convolutions to capture features at different scales
       - Particularly effective for small objects with varying sizes in UAV images

    2. Perceptual Channel Attention (PCA):
       - Global Average Pooling -> FC -> ReLU -> FC -> Sigmoid
       - Emphasizes informative channels for small object features

    3. Perceptual Spatial Attention (PSA):
       - Channel-wise max and average pooling along spatial dimensions
       - Concat -> 7x7 conv -> Sigmoid
       - Highlights spatial locations of small objects

    4. Feature Fusion:
       - Weighted fusion of multi-scale features with attention maps

    Input/Output: (B, C, H, W) -> (B, C, H, W)
    """

    def __init__(self, dim, num_scales=3, reduction=4):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        mid_dim = max(dim // reduction, 8)

        kernel_sizes = [3, 5, 7]

        self.ms_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, ks, padding=ks // 2, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU(inplace=True),
            )
            for ks in kernel_sizes
        ])

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dim * num_scales, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, mid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, dim, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        ms_feats = [conv(x) for conv in self.ms_convs]
        ms_cat = torch.cat(ms_feats, dim=1)
        fused = self.fuse_conv(ms_cat)

        ca_weight = self.channel_attention(fused).view(B, C, 1, 1)
        ca_out = fused * ca_weight

        avg_out = torch.mean(ca_out, dim=1, keepdim=True)
        max_out, _ = torch.max(ca_out, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa_weight = self.spatial_attention(sa_input)
        sa_out = ca_out * sa_weight

        out = self.output_proj(sa_out)
        return out + x


class PercepConvBasicBlock(BasicBlock):
    """BasicBlock with PercepConv augmentation

    Integrates PercepConv module into ResNet BasicBlock to enhance
    feature extraction capability for small object detection.
    """
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.percepconv = PercepConv(ch_out)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.percepconv(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)
        return out


@register
class PResNet_PercepConv(PResNet):
    """ResNet with PercepConv augmentation

    Integrates PercepConv modules into specific stages of ResNet backbone
    to enhance multi-scale feature representation for small object detection
    in UAV imagery.

    Configuration:
    - Inserts PercepConv into stage 2 and stage 3 (where small object features are critical)
    - Maintains compatibility with pretrained weights (PercepConv params randomly initialized)
    - Supports depth 18 and 34 (BasicBlock-based architectures)
    """

    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=True, pretrained=False,
                 pretrained_source='paddle'):

        super().__init__(depth, variant, num_stages, return_idx, act,
                         freeze_at=-1, freeze_norm=False, pretrained=False,
                         pretrained_source=pretrained_source)

        assert depth in [18, 34], \
            f"PResNet_PercepConv only supports BasicBlock-based models (depth 18/34), got {depth}"

        self._insert_percepconv(act, variant)

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            if pretrained_source == 'torchvision':
                self._load_torchvision_pretrained(depth)
            else:
                state = torch.hub.load_state_dict_from_url(donwload_url[depth])
                missing, unexpected = self.load_state_dict(state, strict=False)
                print(f'Load PResNet_PercepConv{depth} state_dict from PaddlePaddle')
                if missing:
                    print(f'  PercepConv params randomly initialized: {len(missing)} keys')

    def _insert_percepconv(self, act, variant):
        for stage_idx in [1, 2]:
            blocks = self.res_layers[stage_idx].blocks
            old_block = blocks[1]

            new_block = PercepConvBasicBlock(
                ch_in=old_block.branch2a.conv.in_channels,
                ch_out=old_block.branch2a.conv.out_channels,
                stride=1,
                shortcut=True,
                act=act,
                variant=variant,
            )

            old_state = old_block.state_dict()
            new_block.load_state_dict(old_state, strict=False)

            blocks[1] = new_block