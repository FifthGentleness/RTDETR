import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import BasicBlock, PResNet, ResNet_cfg, donwload_url
from .common import ConvNormLayer
from src.core import register

__all__ = ['DSADCC_v3', 'DSADCCv3BasicBlock', 'PResNet_DSADCC_v3']


class DSADCC_v3(nn.Module):
    """Dual Spatial Aggregation & Dynamic Channel Calibration v3

    Parallel DSA and DCC branches with lightweight cross-interaction at fusion.

    Key design:
    1. Shared channel reduction: DSA and DCC share one channel_reduce,
       reducing parameters and ensuring both operate in the same feature space.
    2. Parallel computation: DSA and DCC compute independently from Xs,
       avoiding serial dependency and error propagation between branches.
    3. Spatial-adaptive DSA weights: per-pixel weight maps [B,4,H,W]
       for multi-scale spatial aggregation.
    4. Channel mixing: 1x1 Conv after weighted sum for cross-channel exchange.
    5. DCC scalar fusion: scalar alpha/beta for avg/max pooling fusion
       (strong regularization, avoids channel-wise overfitting).
    6. Lightweight cross-interaction at fusion: both branches' global signals
       jointly generate a channel-wise calibration weight that modulates the
       fused features, without altering either branch's independent output.

    Flow:
    1. Shared channel reduction: C -> Cr via 1x1 Conv -> Xs
    2. DSA (parallel): 4 DWConv -> spatial weights [B,4,H,W]
       -> weighted sum -> channel_mix -> Fdsa [B, Cr, H, W]
    3. DCC (parallel): GAP/GMP -> FC -> scalar alpha/beta
       -> combined -> Sigmoid -> Wc -> Xs * Wc -> Fdcc [B, Cr, H, W]
    4. Cross-interaction: GAP(Fdsa) + Wc -> cross_gate [B, Cr, 1, 1]
    5. Fusion: Concat(Fdsa, Fdcc) * cross_gate -> Conv1x1+BN+SiLU -> out [B, C, H, W]
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        cr = max(dim // reduction, 8)
        self.cr = cr
        cr_inner = max(cr // reduction, 8)
        self.cr_inner = cr_inner

        # --- Shared channel reduction ---
        self.channel_reduce = nn.Conv2d(dim, cr, 1, bias=False)

        # --- DSA: Multi-scale DWConv + channel mixing ---
        self.dwconv3 = nn.Conv2d(cr, cr, 3, padding=1, groups=cr, bias=False)
        self.dwconv5 = nn.Conv2d(cr, cr, 5, padding=2, groups=cr, bias=False)
        self.dwconv7 = nn.Conv2d(cr, cr, 7, padding=3, groups=cr, bias=False)
        self.dwconv_d4 = nn.Conv2d(cr, cr, 3, padding=4, dilation=4, groups=cr, bias=False)

        self.channel_mix = nn.Conv2d(cr, cr, 1, bias=False)

        # --- DSA: Spatial-adaptive weights ---
        self.weight_conv = nn.Conv2d(cr * 4, 4, 1, bias=False)

        # --- DCC: Channel attention ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.avg_fc = nn.Sequential(
            nn.Conv2d(cr, cr_inner, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.max_fc = nn.Sequential(
            nn.Conv2d(cr, cr_inner, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        self.fusion_conv = nn.Conv2d(cr_inner * 2, 2, 1, bias=False)

        self.channel_expand = nn.Sequential(
            nn.Conv2d(cr_inner, cr, 1, bias=False),
            nn.Sigmoid(),
        )

        # --- Lightweight cross-interaction at fusion ---
        self.cross_gate = nn.Sequential(
            nn.Conv2d(cr * 2, cr * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        # --- Fusion: Concat -> Conv1x1+BN+SiLU ---
        self.fusion = ConvNormLayer(cr * 2, dim, 1, 1, act='silu')

    def forward(self, x):
        # Shared channel reduction
        xs = self.channel_reduce(x)  # [B, Cr, H, W]

        # === DSA branch (parallel, independent) ===
        f1 = self.dwconv3(xs)
        f2 = self.dwconv5(xs)
        f3 = self.dwconv7(xs)
        f4 = self.dwconv_d4(xs)

        spatial_cat = torch.cat([f1, f2, f3, f4], dim=1)
        spatial_weights = F.softmax(self.weight_conv(spatial_cat), dim=1)  # [B, 4, H, W]

        fdsa = (spatial_weights[:, 0:1] * f1 +
                spatial_weights[:, 1:2] * f2 +
                spatial_weights[:, 2:3] * f3 +
                spatial_weights[:, 3:4] * f4)
        fdsa = self.channel_mix(fdsa)  # [B, Cr, H, W]

        # === DCC branch (parallel, independent) ===
        favg = self.gap(xs)
        fmax = self.gmp(xs)

        favg_reduced = self.avg_fc(favg)
        fmax_reduced = self.max_fc(fmax)

        fused = torch.cat([favg_reduced, fmax_reduced], dim=1)
        weights = F.softmax(self.fusion_conv(fused), dim=1)  # [B, 2, 1, 1]

        alpha = weights[:, 0:1]  # [B, 1, 1, 1] scalar weight
        beta = weights[:, 1:2]   # [B, 1, 1, 1] scalar weight

        combined = alpha * favg_reduced + beta * fmax_reduced
        wc = self.channel_expand(combined)  # [B, Cr, 1, 1]

        fdcc = xs * wc  # [B, Cr, H, W]

        # === Lightweight cross-interaction at fusion ===
        dsa_signal = self.gap(fdsa)                  # [B, Cr, 1, 1] spatial branch global signal
        cross_input = torch.cat([dsa_signal, wc], dim=1)  # [B, 2Cr, 1, 1]
        cross_weight = self.cross_gate(cross_input)  # [B, 2Cr, 1, 1]

        # === Fusion ===
        fused = torch.cat([fdsa, fdcc], dim=1)  # [B, 2Cr, H, W]
        fused = fused * cross_weight             # channel-wise calibration, broadcast to H×W
        out = self.fusion(fused)                 # [B, C, H, W]

        return out


class DSADCCv3BasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', reduction=4):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.dsadcc = DSADCC_v3(ch_out, reduction)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.dsadcc(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)
        return out


@register
class PResNet_DSADCC_v3(PResNet):
    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=True, pretrained=False,
                 pretrained_source='paddle', reduction=4):
        super().__init__(depth, variant, num_stages, return_idx, act,
                         freeze_at=-1, freeze_norm=False, pretrained=False,
                         pretrained_source=pretrained_source)

        assert depth in [18, 34], \
            f"PResNet_DSADCC_v3 only supports BasicBlock-based models (depth 18/34), got {depth}"

        self.reduction = reduction
        self._insert_dsadcc(act, variant, reduction)

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
                print(f'Load PResNet_DSADCC_v3{depth} state_dict from PaddlePaddle')
                if missing:
                    print(f'  DSADCC_v3 params randomly initialized: {len(missing)} keys')

    def _insert_dsadcc(self, act, variant, reduction):
        for stage_idx in [1, 2]:
            blocks = self.res_layers[stage_idx].blocks
            last_idx = len(blocks) - 1
            old_block = blocks[last_idx]

            new_block = DSADCCv3BasicBlock(
                ch_in=old_block.branch2a.conv.in_channels,
                ch_out=old_block.branch2a.conv.out_channels,
                stride=1,
                shortcut=True,
                act=act,
                variant=variant,
                reduction=reduction,
            )

            old_state = old_block.state_dict()
            new_block.load_state_dict(old_state, strict=False)

            blocks[last_idx] = new_block
