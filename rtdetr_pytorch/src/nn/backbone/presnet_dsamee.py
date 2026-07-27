import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import BasicBlock, PResNet, ResNet_cfg, donwload_url
from .common import ConvNormLayer
from src.core import register

__all__ = ['DSAMEE', 'DSAMEEBasicBlock', 'PResNet_DSAMEE']


class MEE(nn.Module):
    """Multi-scale Edge Enhancement (from FA-DETR).

    Dual-path architecture: bypass branch for local context,
    enhancement branch for multi-scale edge extraction and gating.

    Flow:
    1. Channel expand + split -> X1 (bypass), X2 (enhancement)
    2. Bypass: X1 -> Conv3x3 -> Local
    3. Enhancement: for each scale s in {2,4,6,8}:
       - AdaptiveAvgPool(s) -> Conv1x1 -> DWConv3x3 -> T_s
       - Edge_s = T_s - AvgPool(T_s)           (high-frequency residual)
       - Enhanced_s = T_s + Sigmoid(Conv(Edge_s)) (gated edge enhancement)
       - Upsample to HxW
    4. Concat(Local, Enhanced_2, Enhanced_4, Enhanced_6, Enhanced_8)
       -> Conv1x1 -> out
    """

    def __init__(self, dim, scales=(2, 4, 6, 8)):
        super().__init__()
        self.scales = scales
        # After expand(dim→2*dim) + split, each chunk has dim channels
        ch = dim

        # Channel expand + split
        self.expand = nn.Conv2d(dim, dim * 2, 1, bias=False)

        # Bypass branch
        self.bypass_conv = ConvNormLayer(ch, ch, 3, 1, act='relu')

        # Enhancement branch: shared 1x1 reduction for all scales
        self.reduce = nn.Conv2d(ch, ch, 1, bias=False)

        # Per-scale DWConv + edge gate
        self.dwconvs = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
            for _ in scales
        ])
        self.edge_gates = nn.ModuleList([
            nn.Conv2d(ch, ch, 1, bias=False)
            for _ in scales
        ])

        # Fusion: Concat(bypass, 4 scales) → dim
        self.fuse = nn.Conv2d(ch * (1 + len(scales)), dim, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # Expand + split
        expanded = self.expand(x)  # [B, 2*Cr, H, W]
        x1, x2 = expanded.chunk(2, dim=1)  # each [B, Cr, H, W]

        # Bypass branch
        local = self.bypass_conv(x1)  # [B, Cr, H, W]

        # Enhancement branch
        x2_reduced = self.reduce(x2)  # [B, Cr, H, W]
        enhanced_list = []

        for i, s in enumerate(self.scales):
            # Pool to scale s
            pooled = F.adaptive_avg_pool2d(x2_reduced, (s, s))  # [B, Cr, s, s]

            # DWConv
            t = self.dwconvs[i](pooled)  # [B, Cr, s, s]

            # High-frequency edge extraction
            edge = t - F.avg_pool2d(t, kernel_size=3, stride=1, padding=1)

            # Gated edge enhancement + residual
            gate = torch.sigmoid(self.edge_gates[i](edge))
            enhanced = t + gate * edge  # [B, Cr, s, s]

            # Upsample back to HxW
            enhanced_up = F.interpolate(enhanced, size=(H, W), mode='bilinear', align_corners=False)
            enhanced_list.append(enhanced_up)

        # Fuse
        fused = torch.cat([local] + enhanced_list, dim=1)  # [B, Cr*(1+len(scales)), H, W]
        out = self.fuse(fused)  # [B, Cr, H, W]

        return out


class DSAMEE(nn.Module):
    """Dual Spatial Aggregation & Multi-scale Edge Enhancement.

    Parallel DSA and MEE branches with lightweight cross-interaction at fusion.

    Key design:
    1. Shared channel reduction: DSA and MEE share one channel_reduce,
       reducing parameters and ensuring both operate in the same feature space.
    2. Parallel computation: DSA (multi-scale spatial aggregation) and MEE
       (multi-scale edge enhancement) compute independently from Xs.
    3. DSA: per-pixel spatial weights [B,4,H,W] for adaptive scale selection,
       plus channel mixing for cross-channel exchange.
    4. MEE: explicit high-frequency edge extraction across multiple scales,
       with gated edge enhancement and bypass local context.
    5. Complementary: DSA covers mid-low frequency spatial patterns,
       MEE covers high-frequency edge/contour information.
    6. Lightweight cross-interaction at fusion: both branches' global signals
       jointly generate a channel-wise calibration weight.

    Flow:
    1. Shared channel reduction: C -> Cr via 1x1 Conv -> Xs
    2. DSA (parallel): 4 DWConv -> spatial weights [B,4,H,W]
       -> weighted sum -> channel_mix -> Fdsa [B, Cr, H, W]
    3. MEE (parallel): bypass + multi-scale edge enhancement -> Fmee [B, Cr, H, W]
    4. Cross-interaction: GAP(Fdsa) + GAP(Fmee) -> cross_gate [B, 2Cr, 1, 1]
    5. Fusion: Concat(Fdsa, Fmee) * cross_gate -> Conv1x1+BN+SiLU -> out [B, C, H, W]
    """

    def __init__(self, dim, reduction=4, mee_scales=(2, 4, 6, 8)):
        super().__init__()
        self.dim = dim
        cr = max(dim // reduction, 8)
        self.cr = cr

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

        # --- MEE: Multi-scale Edge Enhancement ---
        self.mee = MEE(cr, scales=mee_scales)

        # --- Lightweight cross-interaction at fusion ---
        self.gap = nn.AdaptiveAvgPool2d(1)
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

        # === MEE branch (parallel, independent) ===
        fmee = self.mee(xs)  # [B, Cr, H, W]

        # === Lightweight cross-interaction at fusion ===
        dsa_signal = self.gap(fdsa)   # [B, Cr, 1, 1]
        mee_signal = self.gap(fmee)   # [B, Cr, 1, 1]
        cross_input = torch.cat([dsa_signal, mee_signal], dim=1)  # [B, 2Cr, 1, 1]
        cross_weight = self.cross_gate(cross_input)  # [B, 2Cr, 1, 1]

        # === Fusion ===
        fused = torch.cat([fdsa, fmee], dim=1)  # [B, 2Cr, H, W]
        fused = fused * cross_weight             # channel-wise calibration, broadcast to HxW
        out = self.fusion(fused)                 # [B, C, H, W]

        return out


class DSAMEEBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', reduction=4):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.dsamee = DSAMEE(ch_out, reduction)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.dsamee(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)
        return out


@register
class PResNet_DSAMEE(PResNet):
    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=True, pretrained=False,
                 pretrained_source='paddle', reduction=4):
        super().__init__(depth, variant, num_stages, return_idx, act,
                         freeze_at=-1, freeze_norm=False, pretrained=False,
                         pretrained_source=pretrained_source)

        assert depth in [18, 34], \
            f"PResNet_DSAMEE only supports BasicBlock-based models (depth 18/34), got {depth}"

        self.reduction = reduction
        self._insert_dsamee(act, variant, reduction)

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
                print(f'Load PResNet_DSAMEE{depth} state_dict from PaddlePaddle')
                if missing:
                    print(f'  DSAMEE params randomly initialized: {len(missing)} keys')

    def _insert_dsamee(self, act, variant, reduction):
        for stage_idx in [1, 2]:
            blocks = self.res_layers[stage_idx].blocks
            last_idx = len(blocks) - 1
            old_block = blocks[last_idx]

            new_block = DSAMEEBasicBlock(
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
