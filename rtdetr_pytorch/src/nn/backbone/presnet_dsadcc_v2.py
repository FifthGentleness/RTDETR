import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import BasicBlock, PResNet, ResNet_cfg, donwload_url
from .common import ConvNormLayer
from src.core import register

__all__ = ['DSA_v2', 'DCC_v2', 'DSADCC_v2', 'DSADCCv2BasicBlock', 'PResNet_DSADCC_v2']


class DSA_v2(nn.Module):
    """Dynamic Spatial Aggregation v2

    Multi-scale spatial feature aggregation with dynamic weighting.

    Improvements over v1:
    - Branch 4: 3x3 DWConv d=4 (equiv 9x9) instead of d=2 (equiv 5x5)
    - Per-branch GAP then Concat for weight generation
    - Internal residual connection with reduced input Xs

    Flow:
    1. Channel reduction: C -> Cr via 1x1 Conv -> Xs
    2. 4 parallel DWConv branches (3x3, 5x5, 7x7, 3x3-d4=9x9)
    3. Per-branch GAP -> Concat -> Conv1x1 -> Softmax -> dynamic scale weights
    4. Weighted sum of multi-scale features + Xs (internal residual) -> Fdsa
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        cr = max(dim // reduction, 8)
        self.cr = cr

        self.channel_reduce = nn.Conv2d(dim, cr, 1, bias=False)

        self.dwconv3 = nn.Conv2d(cr, cr, 3, padding=1, groups=cr, bias=False)
        self.dwconv5 = nn.Conv2d(cr, cr, 5, padding=2, groups=cr, bias=False)
        self.dwconv7 = nn.Conv2d(cr, cr, 7, padding=3, groups=cr, bias=False)
        self.dwconv_d4 = nn.Conv2d(cr, cr, 3, padding=4, dilation=4, groups=cr, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.weight_conv = nn.Conv2d(cr * 4, 4, 1, bias=False)

    def forward(self, x):
        xs = self.channel_reduce(x)

        f1 = self.dwconv3(xs)
        f2 = self.dwconv5(xs)
        f3 = self.dwconv7(xs)
        f4 = self.dwconv_d4(xs)

        d1 = self.gap(f1)
        d2 = self.gap(f2)
        d3 = self.gap(f3)
        d4 = self.gap(f4)

        cat = torch.cat([d1, d2, d3, d4], dim=1)
        weights = F.softmax(self.weight_conv(cat), dim=1)

        w1 = weights[:, 0:1]
        w2 = weights[:, 1:2]
        w3 = weights[:, 2:3]
        w4 = weights[:, 3:4]

        out = w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4
        out = out + xs

        return out


class DCC_v2(nn.Module):
    """Dynamic Channel Calibration v2

    Channel attention with dynamic statistic fusion of avg and max pooling.

    Improvement over v1:
    - Channel-wise alpha/beta instead of scalar alpha/beta
      Each channel independently decides how to mix avg and max pooling

    Flow:
    1. Channel reduction: C -> Cr via 1x1 Conv -> Xc
    2. GAP and GMP in parallel -> [B, Cr, 1, 1]
    3. Each through Conv1x1 + ReLU (Cr -> Cr/r)
    4. Concat -> Conv1x1 (2Cr/r -> 2Cr/r) -> reshape [B,2,Cr/r,1,1] -> Softmax
    5. Channel-wise alpha * Favg_reduced + beta * Fmax_reduced
    6. Conv1x1 (Cr/r -> Cr) + Sigmoid -> channel weights Wc
    7. Xc * Wc -> Fdcc [B, Cr, H, W]
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        cr = max(dim // reduction, 8)
        self.cr = cr
        cr_inner = max(cr // reduction, 8)
        self.cr_inner = cr_inner

        self.channel_reduce = nn.Conv2d(dim, cr, 1, bias=False)

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

        self.fusion_conv = nn.Conv2d(cr_inner * 2, cr_inner * 2, 1, bias=False)

        self.channel_expand = nn.Sequential(
            nn.Conv2d(cr_inner, cr, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xc = self.channel_reduce(x)

        favg = self.gap(xc)
        fmax = self.gmp(xc)

        favg_reduced = self.avg_fc(favg)
        fmax_reduced = self.max_fc(fmax)

        fused = torch.cat([favg_reduced, fmax_reduced], dim=1)
        logits = self.fusion_conv(fused)
        B = logits.shape[0]
        logits = logits.reshape(B, 2, self.cr_inner, 1, 1)
        weights = F.softmax(logits, dim=1)

        alpha = weights[:, 0]
        beta = weights[:, 1]

        combined = alpha * favg_reduced + beta * fmax_reduced
        wc = self.channel_expand(combined)

        return xc * wc


class DSADCC_v2(nn.Module):
    """Dual Spatial Aggregation & Dynamic Channel Calibration v2

    Combines DSA-v2 (spatial) and DCC-v2 (channel) branches with fusion.
    Includes residual scaling (init=0.1) for stable training.

    Flow:
    1. DSA-v2 branch -> Fdsa [B, Cr, H, W]
    2. DCC-v2 branch -> Fdcc [B, Cr, H, W]
    3. Concat -> Conv1x1 + BN + SiLU -> scale * out -> [B, C, H, W]
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dsa = DSA_v2(dim, reduction)
        self.dcc = DCC_v2(dim, reduction)

        cr = max(dim // reduction, 8)

        self.fusion = ConvNormLayer(cr * 2, dim, 1, 1, act='silu')
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        fdsa = self.dsa(x)
        fdcc = self.dcc(x)

        fused = torch.cat([fdsa, fdcc], dim=1)
        out = self.fusion(fused)
        out = out * self.scale

        return out


class DSADCCv2BasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', reduction=4):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.dsadcc = DSADCC_v2(ch_out, reduction)

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
class PResNet_DSADCC_v2(PResNet):
    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=True, pretrained=False,
                 pretrained_source='paddle', reduction=4):
        super().__init__(depth, variant, num_stages, return_idx, act,
                         freeze_at=-1, freeze_norm=False, pretrained=False,
                         pretrained_source=pretrained_source)

        assert depth in [18, 34], \
            f"PResNet_DSADCC_v2 only supports BasicBlock-based models (depth 18/34), got {depth}"

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
                print(f'Load PResNet_DSADCC_v2{depth} state_dict from PaddlePaddle')
                if missing:
                    print(f'  DSADCC_v2 params randomly initialized: {len(missing)} keys')

    def _insert_dsadcc(self, act, variant, reduction):
        for stage_idx in [1, 2]:
            blocks = self.res_layers[stage_idx].blocks
            last_idx = len(blocks) - 1
            old_block = blocks[last_idx]

            new_block = DSADCCv2BasicBlock(
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