import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import BasicBlock, PResNet, ResNet_cfg, donwload_url
from .common import ConvNormLayer
from src.core import register

__all__ = ['DSADOC_v9', 'DSADOCv9BasicBlock', 'PResNet_DSADOC_v9']


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
# 类名: DSADOC_v9
# 类型: nn.Module 子类
# 代码逻辑链条中的具体职责: 基于DSADOC_v8，reduction默认设为2
# 在v8的BN+ReLU正则化基础上，通过reduction=2压缩瓶颈通道，
# 在v5(重)和v2(轻)之间取得参数量与性能的平衡
# v8: reduction=1, cr=dim, 参数量~22.4M (+11.3%)
# v9: reduction=2, cr=dim/2, 预估参数量~21.0M (+4.5%)
# v2: reduction=4, cr=dim/4, 参数量~20.3M (+1.0%)
# =========================================
class DSADOC_v9(nn.Module):
    """DSADOC_v9: DSADOC_v8 with reduction=2 for parameter-efficiency.

    Based on DSADOC_v8 (BN+ReLU regularization), changes default reduction from 1 to 2.
    This provides a balance between v5/v8 (reduction=1, heavy) and v2 (reduction=4, light).
    """

    def __init__(self, dim, reduction=2):
        super().__init__()
        self.dim = dim
        cr = max(dim // reduction, 8)
        self.cr = cr

        self.channel_reduce = nn.Sequential(
            nn.Conv2d(dim, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )

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

        self.scharr = ScharrEdge(cr)

        self.channel_mix = nn.Sequential(
            nn.Conv2d(cr, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )

        self.weight_conv = nn.Conv2d(cr * 5, 5, 1, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.dwt = HaarDWT()
        self.idwt = HaarIDWT()

        self.spatial_mix_ll = nn.Sequential(
            nn.Conv2d(cr * 4, cr, 1, bias=False),
            nn.BatchNorm2d(cr),
            nn.ReLU(inplace=True),
        )
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

        self.fusion_conv = nn.Conv2d(cr * 3, 3, 1, bias=False)

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

        self.cal_norm = nn.GroupNorm(num_groups=1, num_channels=cr, eps=1e-5)

        self.film_gen = nn.Conv2d(cr, cr * 2, 1, bias=True)
        self.film_scale = nn.Parameter(torch.zeros(1))

        self.cross_gate = nn.Sequential(
            nn.Conv2d(cr * 2, cr * 2, 1, bias=False),
            nn.Sigmoid(),
        )

        self.fusion = ConvNormLayer(cr * 2, dim, 1, 1, act='silu')

    def forward(self, x):
        xs = self.channel_reduce(x)

        f1 = self.dwconv3(xs)
        f2 = self.dwconv5(xs)
        f3 = self.dwconv7(xs)
        f4 = self.dwconv_d4(xs)
        f_scharr = self.scharr(xs)
        f_scharr = f_scharr - f_scharr.mean(dim=[2, 3], keepdim=True)

        spatial_cat = torch.cat([f1, f2, f3, f4, f_scharr], dim=1)
        spatial_weights = F.softmax(self.weight_conv(spatial_cat), dim=1)

        fdsa = (spatial_weights[:, 0:1] * f1 +
                spatial_weights[:, 1:2] * f2 +
                spatial_weights[:, 2:3] * f3 +
                spatial_weights[:, 3:4] * f4 +
                spatial_weights[:, 4:5] * f_scharr)
        fdsa = self.channel_mix(fdsa)

        favg = self.gap(xs)
        fmax = self.gmp(xs)

        ll, lh, hl, hh = self.dwt(xs)

        dwt_ll = self.gap(ll)
        global_cat = torch.cat([favg, fmax, dwt_ll], dim=1)
        weights = F.softmax(self.fusion_conv(global_cat), dim=1)
        alpha = weights[:, 0:1]
        beta = weights[:, 1:2]
        gamma = weights[:, 2:3]
        global_feat = alpha * favg + beta * fmax + gamma * dwt_ll

        all_cat = torch.cat([ll, lh, hl, hh], dim=1)
        spatial_ll = self.spatial_mix_ll(all_cat)

        high_cat = torch.cat([lh, hl, hh], dim=1)
        spatial_lh = self.spatial_mix_lh(high_cat)
        spatial_hl = self.spatial_mix_hl(high_cat)
        spatial_hh = self.spatial_mix_hh(high_cat)

        cal_ll = self.cal_ll(ll) + global_feat + spatial_ll
        cal_lh = self.cal_lh(lh) + global_feat + spatial_lh
        cal_hl = self.cal_hl(hl) + global_feat + spatial_hl
        cal_hh = self.cal_hh(hh) + global_feat + spatial_hh

        cal_map = self.idwt(cal_ll, cal_lh, cal_hl, cal_hh)

        if cal_map.shape[2] != xs.shape[2] or cal_map.shape[3] != xs.shape[3]:
            cal_map = cal_map[:, :, :xs.shape[2], :xs.shape[3]]

        cal_map = self.cal_norm(cal_map)

        film_params = self.film_gen(cal_map)
        gamma_raw, beta_raw = film_params.chunk(2, dim=1)

        scale = self.film_scale
        film_gamma = 1.0 + scale * torch.tanh(gamma_raw)
        film_beta = scale * torch.tanh(beta_raw)

        fdcc = xs * film_gamma + film_beta

        dsa_signal = self.gap(fdsa)
        dcc_signal = self.gap(fdcc)
        cross_input = torch.cat([dsa_signal, dcc_signal], dim=1)
        cross_weight = self.cross_gate(cross_input)

        fused = torch.cat([fdsa, fdcc], dim=1)
        fused = fused * cross_weight
        out = self.fusion(fused)

        return out


class DSADOCv9BasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b', reduction=2):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.dsadcc = DSADOC_v9(ch_out, reduction)

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
class PResNet_DSADOC_v9(PResNet):
    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=True, pretrained=False,
                 pretrained_source='paddle', reduction=2):
        super().__init__(depth, variant, num_stages, return_idx, act,
                         freeze_at=-1, freeze_norm=False, pretrained=False,
                         pretrained_source=pretrained_source)

        assert depth in [18, 34], \
            f"PResNet_DSADOC_v9 only supports BasicBlock-based models (depth 18/34), got {depth}"

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
                print(f'Load PResNet_DSADOC_v9{depth} state_dict from PaddlePaddle')
                if missing:
                    print(f'  DSADOC_v9 params randomly initialized: {len(missing)} keys')

    def _insert_dsadcc(self, act, variant, reduction):
        for stage_idx in [1, 2]:
            blocks = self.res_layers[stage_idx].blocks
            last_idx = len(blocks) - 1
            old_block = blocks[last_idx]

            new_block = DSADOCv9BasicBlock(
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