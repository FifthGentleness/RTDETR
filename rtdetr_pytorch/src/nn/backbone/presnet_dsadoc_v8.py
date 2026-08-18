import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import BasicBlock, PResNet, ResNet_cfg, donwload_url
from .common import ConvNormLayer
from src.core import register

__all__ = ['DSADOC_v8', 'DSADOCv8BasicBlock', 'PResNet_DSADOC_v8']


class HaarDWT(nn.Module):
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


# =========================================
# 类名: DSADOC_v8
# 类型: nn.Module 子类
# 代码逻辑链条中的具体职责: 在DSADOC_v2基础上添加BN+ReLU正则化
# 🔴 必须加: channel_reduce(BN+ReLU), channel_mix(BN+ReLU), cal_ll/lh/hl/hh(BN)
# 🟡 建议加: dwconv3/5/7/d4(BN), spatial_mix_ll/lh/hl/hh(补BN)
# 🟢 不加: weight_conv(→softmax), fusion_conv(→softmax), film_gen(零初始化), scharr(固定核)
# =========================================
class DSADOC_v8(nn.Module):
    """DSADOC_v8: DSADOC_v2 with BN+ReLU regularization for reduction=1 stability.

    Changes from DSADOC_v2:
    [MUST] channel_reduce: Conv1x1 → Conv1x1+BN+ReLU (入口归一化)
    [MUST] channel_mix:    Conv1x1 → Conv1x1+BN+ReLU (DSA分支非线性)
    [MUST] cal_ll/lh/hl/hh: Conv1x1 → Conv1x1+BN     (子带校准稳定化, 无ReLU保留正负)
    [SUGGEST] dwconv3/5/7/d4: DWConv → DWConv+BN      (5路输出同尺度, softmax选择更合理)
    [SUGGEST] spatial_mix_ll/lh/hl/hh: Conv1x1+ReLU → Conv1x1+BN+ReLU (数值范围可控)
    """

    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        cr = max(dim // reduction, 8)
        self.cr = cr

        # --- Shared channel reduction [MUST: +BN+ReLU] ---
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(dim, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )

        # --- DSA: Multi-scale DWConv [SUGGEST: +BN] ---
        self.dwconv3 = nn.Sequential(
            nn.Conv2d(cr, cr, 3, padding=1, groups=cr, bias=False),
            nn.BatchNorm2d(cr),
        )
        self.dwconv5 = nn.Sequential(
            nn.Conv2d(cr, cr, 5, padding=2, groups=cr, bias=False),
            nn.BatchNorm2d(cr),
        )
        self.dwconv7 = nn.Sequential(
            nn.Conv2d(cr, cr, 7, padding=3, groups=cr, bias=False),
            nn.BatchNorm2d(cr),
        )
        self.dwconv_d4 = nn.Sequential(
            nn.Conv2d(cr, cr, 3, padding=4, dilation=4, groups=cr, bias=False),
            nn.BatchNorm2d(cr),
        )

        # --- DSA: Scharr edge detection branch [NO CHANGE: fixed kernel] ---
        self.scharr = ScharrEdge(cr)

        # --- DSA: Channel mixing [MUST: +BN+ReLU] ---
        self.channel_mix = nn.Sequential(
            nn.Conv2d(cr, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )

        # --- DSA: Spatial-adaptive weights (5-way) [NO CHANGE: →softmax] ---
        self.weight_conv = nn.Conv2d(cr * 5, 5, 1, bias=False)

        # --- DCC: Global pooling ---
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # --- DCC: DWT / IDWT ---
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        # --- DCC: Omnidirectional spatial feature for LL [SUGGEST: +BN] ---
        self.spatial_mix_ll = nn.Sequential(
            nn.Conv2d(cr * 4, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )

        # --- DCC: High-freq direction-specific spatial features [SUGGEST: +BN] ---
        self.spatial_mix_lh = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )
        self.spatial_mix_hl = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )
        self.spatial_mix_hh = nn.Sequential(
            nn.Conv2d(cr * 3, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )

        # --- DCC: Global descriptor scalar fusion [NO CHANGE: →softmax] ---
        self.fusion_conv = nn.Conv2d(cr * 3, 3, 1, bias=False)

        # --- DCC: Wavelet-domain subband calibration [MUST: +BN, NO ReLU] ---
        self.cal_ll = nn.Sequential(
            nn.Conv2d(cr, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
        )
        self.cal_lh = nn.Sequential(
            nn.Conv2d(cr, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
        )
        self.cal_hl = nn.Sequential(
            nn.Conv2d(cr, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
        )
        self.cal_hh = nn.Sequential(
            nn.Conv2d(cr, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
        )

        # --- DCC: Normalization for IDWT output ---
        self.cal_norm = nn.GroupNorm(num_groups=1, num_channels=cr, eps=1e-5)

        # --- DCC: FiLM parameter generation [NO CHANGE: zero-init design] ---
        self.film_gen = nn.Conv2d(cr, cr * 2, 1, bias=True)
        self.film_scale = nn.Parameter(torch.zeros(1))

        # --- Cross-interaction ---
        self.cross_gate = nn.Sequential(
            nn.Conv2d(cr * 2, cr * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        # --- Fusion: Concat -> Conv1x1+BN+SiLU ---
        self.fusion = ConvNormLayer(cr * 2, dim, 1, 1, act='silu')

    def forward(self, x):
        # [B, dim, H, W] → [B, cr, H, W]
        xs = self.channel_reduce(x)

        # === DSA branch ===
        # [B, cr, H, W] each
        f1 = self.dwconv3(xs)
        f2 = self.dwconv5(xs)
        f3 = self.dwconv7(xs)
        f4 = self.dwconv_d4(xs)
        f_scharr = self.scharr(xs)
        f_scharr = f_scharr - f_scharr.mean(dim=[2, 3], keepdim=True)

        # [B, 5*cr, H, W] → [B, 5, H, W]
        spatial_cat = torch.cat([f1, f2, f3, f4, f_scharr], dim=1)
        spatial_weights = F.softmax(self.weight_conv(spatial_cat), dim=1)

        # [B, cr, H, W]
        fdsa = (spatial_weights[:, 0:1] * f1 +
                spatial_weights[:, 1:2] * f2 +
                spatial_weights[:, 2:3] * f3 +
                spatial_weights[:, 3:4] * f4 +
                spatial_weights[:, 4:5] * f_scharr)
        fdsa = self.channel_mix(fdsa)

        # === DCC branch ===
        favg = self.gap(xs)   # [B, cr, 1, 1]
        fmax = self.gmp(xs)   # [B, cr, 1, 1]

        ll, lh, hl, hh = self.dwt(xs)  # each [B, cr, H/2, W/2]

        dwt_ll = self.gap(ll)  # [B, cr, 1, 1]
        global_cat = torch.cat([favg, fmax, dwt_ll], dim=1)  # [B, 3*cr, 1, 1]
        weights = F.softmax(self.fusion_conv(global_cat), dim=1)  # [B, 3, 1, 1]
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        gamma = weights[:, 2:3]
        global_feat = alpha * favg + beta * fmax + gamma * dwt_ll  # [B, cr, 1, 1]

        all_cat = torch.cat([ll, lh, hl, hh], dim=1)  # [B, 4*cr, H/2, W/2]
        spatial_ll = self.spatial_mix_ll(all_cat)       # [B, cr, H/2, W/2]

        high_cat = torch.cat([lh, hl, hh], dim=1)  # [B, 3*cr, H/2, W/2]
        spatial_lh = self.spatial_mix_lh(high_cat)  # [B, cr, H/2, W/2]
        spatial_hl = self.spatial_mix_hl(high_cat)  # [B, cr, H/2, W/2]
        spatial_hh = self.spatial_mix_hh(high_cat)  # [B, cr, H/2, W/2]

        cal_ll = self.cal_ll(ll) + global_feat + spatial_ll   # [B, cr, H/2, W/2]
        cal_lh = self.cal_lh(lh) + global_feat + spatial_lh  # [B, cr, H/2, W/2]
        cal_hl = self.cal_hl(hl) + global_feat + spatial_hl  # [B, cr, H/2, W/2]
        cal_hh = self.cal_hh(hh) + global_feat + spatial_hh  # [B, cr, H/2, W/2]

        cal_map = self.idwt(cal_ll, cal_lh, cal_hl, cal_hh)  # [B, cr, H', W']

        if cal_map.shape[2] != xs.shape[2] or cal_map.shape[3] != xs.shape[3]:
            cal_map = cal_map[:, :, :xs.shape[2], :xs.shape[3]]

        cal_map = self.cal_norm(cal_map)  # [B, cr, H, W]

        film_params = self.film_gen(cal_map)  # [B, 2*cr, H, W]
        gamma_raw, beta_raw = film_params.chunk(2, dim=1)  # each [B, cr, H, W]

        scale = self.film_scale
        film_gamma = 1.0 + scale * torch.tanh(gamma_raw)
        film_beta = scale * torch.tanh(beta_raw)

        fdcc = xs * film_gamma + film_beta  # [B, cr, H, W]

        # === Cross-interaction ===
        dsa_signal = self.gap(fdsa)
        dcc_signal = self.gap(fdcc)
        cross_input = torch.cat([dsa_signal, dcc_signal], dim=1)  # [B, 2*cr, 1, 1]
        cross_weight = self.cross_gate(cross_input)               # [B, 2*cr, 1, 1]

        # === Fusion ===
        fused = torch.cat([fdsa, fdcc], dim=1)  # [B, 2*cr, H, W]
        fused = fused * cross_weight
        out = self.fusion(fused)                 # [B, dim, H, W]

        return out


class DSADOCv8BasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', reduction=1):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.dsadcc = DSADOC_v8(ch_out, reduction)

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
class PResNet_DSADOC_v8(PResNet):
    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=True, pretrained=False,
                 pretrained_source='paddle', reduction=1):
        super().__init__(depth, variant, num_stages, return_idx, act,
                         freeze_at=-1, freeze_norm=False, pretrained=False,
                         pretrained_source=pretrained_source)

        assert depth in [18, 34], \
            f"PResNet_DSADOC_v8 only supports BasicBlock-based models (depth 18/34), got {depth}"

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
                print(f'Load PResNet_DSADOC_v8{depth} state_dict from PaddlePaddle')
                if missing:
                    print(f'  DSADOC_v8 params randomly initialized: {len(missing)} keys')

    def _insert_dsadcc(self, act, variant, reduction):
        for stage_idx in [1, 2]:
            blocks = self.res_layers[stage_idx].blocks
            last_idx = len(blocks) - 1
            old_block = blocks[last_idx]

            new_block = DSADOCv8BasicBlock(
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