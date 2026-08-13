import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import BasicBlock, PResNet, ResNet_cfg, donwload_url
from .common import ConvNormLayer
from src.core import register

__all__ = ['DSADOC_v4', 'DSADOCv4BasicBlock', 'PResNet_DSADOC_v4']


class HaarDWT(nn.Module):
    """Haar Discrete Wavelet Transform with odd-size padding.

    Decomposes input into 4 subbands: LL, LH, HL, HH.
    Each subband has spatial size (H//2, W//2).
    If H or W is odd, reflect-padding is applied before decomposition.
    """

    def forward(self, x):
        B, C, H, W = x.shape
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, [0, W % 2, 0, H % 2], mode='reflect')
            _, _, H, W = x.shape

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = (x00 + x10 + x01 + x11) * 0.5
        lh = (x00 - x10 + x01 - x11) * 0.5
        hl = (x00 + x10 - x01 - x11) * 0.5
        hh = (x00 - x10 - x01 + x11) * 0.5

        return ll, lh, hl, hh


class HaarIDWT(nn.Module):
    """Inverse Haar Discrete Wavelet Transform.

    Reconstructs full-resolution feature from 4 subbands.
    Output spatial size is (H_half * 2, W_half * 2).
    """

    def forward(self, ll, lh, hl, hh):
        x00 = ll + lh + hl + hh
        x01 = ll + lh - hl - hh
        x10 = ll - lh + hl - hh
        x11 = ll - lh - hl + hh

        B, C, H2, W2 = ll.shape
        out = torch.empty(B, C, H2 * 2, W2 * 2, device=ll.device, dtype=ll.dtype)
        out[:, :, 0::2, 0::2] = x00
        out[:, :, 0::2, 1::2] = x01
        out[:, :, 1::2, 0::2] = x10
        out[:, :, 1::2, 1::2] = x11
        return out


class ScharrEdge(nn.Module):
    """Scharr edge detection operator.

    Computes gradient magnitude from fixed Scharr kernels:
      Gx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
      Gy = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]]
    Magnitude = sqrt(Gx^2 + Gy^2 + eps) for numerical stability.
    Kernels are registered as buffers (non-learnable, move with model).
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        scharr_x = torch.tensor(
            [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / 16.0
        scharr_y = torch.tensor(
            [[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32
        ).reshape(1, 1, 3, 3) / 16.0

        self.register_buffer('kernel_x', scharr_x.repeat(channels, 1, 1, 1))
        self.register_buffer('kernel_y', scharr_y.repeat(channels, 1, 1, 1))

    def forward(self, x):
        gx = F.conv2d(x, self.kernel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.kernel_y, padding=1, groups=self.channels)
        magnitude = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return magnitude


class DSADOC_v4(nn.Module):
    """Dual Spatial Aggregation & Direction-Omnidirectional Channel calibration v4.

    Same module architecture as DSADOC_v3 (with cross_gate fusion).
    The difference is in PResNet_DSADOC_v4: DSADOC_v4 replaces the last
    convolution block in stages 2, 3, 4 (not stage 1), whereas v3 replaces
    all 4 stages and v2 only replaces stages 2-3.

    Architecture:
      DSA branch: 5-way spatial aggregation (4 DWConv + Scharr) with
                  spatial-adaptive weights + channel mixing.
      DCC branch: DWT -> subband calibration (global + spatial) -> IDWT ->
                  GroupNorm -> FiLM (zero-init scale) -> affine modulation.
      Cross-interaction: GAP(DSA) + GAP(DCC) -> cross_gate -> channel-wise calibration.
      Fusion:     Concat [fdsa, fdcc] * cross_weight -> Conv1x1+BN+SiLU.
    """

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.dim = dim
        cr = max(dim // reduction, 8)
        self.cr = cr

        # --- Shared channel reduction ---
        self.channel_reduce = nn.Conv2d(dim, cr, 1, bias=False)

        # --- DSA: Multi-scale DWConv ---
        self.dwconv3 = nn.Conv2d(cr, cr, 3, padding=1, groups=cr, bias=False)
        self.dwconv5 = nn.Conv2d(cr, cr, 5, padding=2, groups=cr, bias=False)
        self.dwconv7 = nn.Conv2d(cr, cr, 7, padding=3, groups=cr, bias=False)
        self.dwconv_d4 = nn.Conv2d(cr, cr, 3, padding=4, dilation=4, groups=cr, bias=False)

        # --- DSA: Scharr edge detection branch ---
        self.scharr = ScharrEdge(cr)

        # --- DSA: Channel mixing ---
        self.channel_mix = nn.Conv2d(cr, cr, 1, bias=False)

        # --- DSA: Spatial-adaptive weights (5-way) ---
        self.weight_conv = nn.Conv2d(cr * 5, 5, 1, bias=False)

        # --- DCC: Global pooling ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # --- DCC: DWT / IDWT ---
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        # --- DCC: Omnidirectional spatial feature for LL (all 4 subbands) ---
        self.spatial_mix_ll = nn.Sequential(
            nn.Conv2d(cr * 4, cr, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # --- DCC: High-freq direction-specific spatial features ---
        self.spatial_mix_lh = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.spatial_mix_hl = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.spatial_mix_hh = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # --- DCC: Global descriptor scalar fusion (alpha, beta, gamma) ---
        self.fusion_conv = nn.Conv2d(cr * 3, 3, 1, bias=False)

        # --- DCC: Wavelet-domain subband calibration ---
        self.cal_ll = nn.Conv2d(cr, cr, 1, bias=False)
        self.cal_lh = nn.Conv2d(cr, cr, 1, bias=False)
        self.cal_hl = nn.Conv2d(cr, cr, 1, bias=False)
        self.cal_hh = nn.Conv2d(cr, cr, 1, bias=False)

        # --- DCC: Normalization for IDWT output ---
        self.cal_norm = nn.GroupNorm(num_groups=1, num_channels=cr, eps=1e-5)

        # --- DCC: FiLM parameter generation with learnable scale ---
        self.film_gen = nn.Conv2d(cr, cr * 2, 1, bias=True)
        self.film_scale = nn.Parameter(torch.zeros(1))

        # --- Cross-interaction from DSADCC_v4 ---
        self.cross_gate = nn.Sequential(
            nn.Conv2d(cr * 2, cr * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        # --- Fusion: Concat -> Conv1x1+BN+SiLU ---
        self.fusion = ConvNormLayer(cr * 2, dim, 1, 1, act='silu')

    def forward(self, x):
        xs = self.channel_reduce(x)  # [B, Cr, H, W]

        # === DSA branch (parallel, independent) ===
        f1 = self.dwconv3(xs)
        f2 = self.dwconv5(xs)
        f3 = self.dwconv7(xs)
        f4 = self.dwconv_d4(xs)
        f_scharr = self.scharr(xs)
        f_scharr = f_scharr - f_scharr.mean(dim=[2, 3], keepdim=True)

        spatial_cat = torch.cat([f1, f2, f3, f4, f_scharr], dim=1)  # [B, 5*Cr, H, W]
        spatial_weights = F.softmax(self.weight_conv(spatial_cat), dim=1)  # [B, 5, H, W]

        fdsa = (spatial_weights[:, 0:1] * f1 +
                spatial_weights[:, 1:2] * f2 +
                spatial_weights[:, 2:3] * f3 +
                spatial_weights[:, 3:4] * f4 +
                spatial_weights[:, 4:5] * f_scharr)
        fdsa = self.channel_mix(fdsa)  # [B, Cr, H, W]

        # === DCC branch ===
        favg = self.gap(xs)   # [B, Cr, 1, 1]
        fmax = self.gmp(xs)   # [B, Cr, 1, 1]

        # DWT decomposition
        ll, lh, hl, hh = self.dwt(xs)  # each [B, Cr, H/2, W/2]

        # Global descriptor: GAP/GMP + DWT-LL
        dwt_ll = self.gap(ll)  # [B, Cr, 1, 1]
        global_cat = torch.cat([favg, fmax, dwt_ll], dim=1)  # [B, 3*Cr, 1, 1]
        weights = F.softmax(self.fusion_conv(global_cat), dim=1)  # [B, 3, 1, 1]
        alpha = weights[:, 0:1]  # [B, 1, 1, 1]
        beta = weights[:, 1:2]   # [B, 1, 1, 1]
        gamma = weights[:, 2:3]  # [B, 1, 1, 1]
        global_feat = alpha * favg + beta * fmax + gamma * dwt_ll  # [B, Cr, 1, 1]

        # Omnidirectional spatial feature for LL (all 4 subbands)
        all_cat = torch.cat([ll, lh, hl, hh], dim=1)  # [B, 4*Cr, H/2, W/2]
        spatial_ll = self.spatial_mix_ll(all_cat)       # [B, Cr, H/2, W/2]

        # Direction-specific spatial features for high-freq subbands
        high_cat = torch.cat([lh, hl, hh], dim=1)  # [B, 3*Cr, H/2, W/2]
        spatial_lh = self.spatial_mix_lh(high_cat)  # [B, Cr, H/2, W/2]
        spatial_hl = self.spatial_mix_hl(high_cat)  # [B, Cr, H/2, W/2]
        spatial_hh = self.spatial_mix_hh(high_cat)  # [B, Cr, H/2, W/2]

        # Wavelet-domain calibration (each subband with matched spatial feature)
        cal_ll = self.cal_ll(ll) + global_feat + spatial_ll   # [B, Cr, H/2, W/2]
        cal_lh = self.cal_lh(lh) + global_feat + spatial_lh  # [B, Cr, H/2, W/2]
        cal_hl = self.cal_hl(hl) + global_feat + spatial_hl  # [B, Cr, H/2, W/2]
        cal_hh = self.cal_hh(hh) + global_feat + spatial_hh  # [B, Cr, H/2, W/2]

        # IDWT reconstruction -> full-resolution calibration map
        cal_map = self.idwt(cal_ll, cal_lh, cal_hl, cal_hh)  # [B, Cr, H', W']

        # Crop to match original spatial size (IDWT may produce H+1/W+1 when input was odd)
        if cal_map.shape[2] != xs.shape[2] or cal_map.shape[3] != xs.shape[3]:
            cal_map = cal_map[:, :, :xs.shape[2], :xs.shape[3]]

        # Normalize calibration map for stable FiLM generation
        cal_map = self.cal_norm(cal_map)  # [B, Cr, H, W]

        # FiLM: residual-style with learnable scale (init=0 -> identity at start)
        film_params = self.film_gen(cal_map)  # [B, 2*Cr, H, W]
        gamma_raw, beta_raw = film_params.chunk(2, dim=1)  # each [B, Cr, H, W]

        scale = self.film_scale  # init=0, gradually learned
        film_gamma = 1.0 + scale * torch.tanh(gamma_raw)  # init: 1+0=1 (identity)
        film_beta = scale * torch.tanh(beta_raw)           # init: 0 (no shift)

        fdcc = xs * film_gamma + film_beta  # [B, Cr, H, W]

        # === Cross-interaction ===
        dsa_signal = self.gap(fdsa)                        # [B, Cr, 1, 1]
        dcc_signal = self.gap(fdcc)                        # [B, Cr, 1, 1]
        cross_input = torch.cat([dsa_signal, dcc_signal], dim=1)  # [B, 2Cr, 1, 1]
        cross_weight = self.cross_gate(cross_input)        # [B, 2Cr, 1, 1]

        # === Fusion ===
        fused = torch.cat([fdsa, fdcc], dim=1)  # [B, 2Cr, H, W]
        fused = fused * cross_weight             # channel-wise calibration, broadcast to HxW
        out = self.fusion(fused)                 # [B, C, H, W]

        return out


class DSADOCv4BasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', reduction=4):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.dsadcc = DSADOC_v4(ch_out, reduction)

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
class PResNet_DSADOC_v4(PResNet):
    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=True, pretrained=False,
                 pretrained_source='paddle', reduction=4):
        super().__init__(depth, variant, num_stages, return_idx, act,
                         freeze_at=-1, freeze_norm=False, pretrained=False,
                         pretrained_source=pretrained_source)

        assert depth in [18, 34], \
            f"PResNet_DSADOC_v4 only supports BasicBlock-based models (depth 18/34), got {depth}"

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
                print(f'Load PResNet_DSADOC_v4{depth} state_dict from PaddlePaddle')
                if missing:
                    print(f'  DSADOC_v4 params randomly initialized: {len(missing)} keys')

    def _insert_dsadcc(self, act, variant, reduction):
        for stage_idx in [1, 2, 3]:
            blocks = self.res_layers[stage_idx].blocks
            last_idx = len(blocks) - 1
            old_block = blocks[last_idx]

            new_block = DSADOCv4BasicBlock(
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