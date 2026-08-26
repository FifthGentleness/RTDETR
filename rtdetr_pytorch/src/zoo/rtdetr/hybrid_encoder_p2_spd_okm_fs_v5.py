# =========================================
# 文件说明：
# - 基于v4，两项改动：
#   (1) 去除Global分支的GMM（参数占比82.6%，性价比低，Large分支大核已提供空间覆盖）
#   (2) Global分支改为 FreqScale → SCA → FGM 串行（对齐OKNet原版BottleNect的FCA→SCA→FGM结构）
# - CCFF融合保持LE-DETR EFAM模式：
#       Concat[128,256,128]=512 → cv1(512→512)重混 → split[128,384]
#       → 创新(128)+identity(384) → cat[512] → cv2(512→512)融合 → CSPRep(512→256)降维
# - Global分支数据流:
#       x → FreqScale(频域动态滤波) → SCA(空间通道注意力) → FGM(频域门控) → global_out
#       对齐OKNet原版: FCA→SCA→FGM，但FCA升级为FreqScale(频率bin级动态滤波)
# =========================================
'''by lyuwenyu
'''

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from .hybrid_encoder import ConvNormLayer, RepVggBlock, CSPRepLayer, TransformerEncoderLayer, TransformerEncoder

from src.core import register


__all__ = ['HybridEncoderP2SPDOKMFSV5']


class SPDConv(nn.Module):
    def __init__(self, in_channels, out_channels, act='silu'):
        super().__init__()
        self.conv = ConvNormLayer(in_channels * 4, out_channels, 3, 1, act=act)

    def forward(self, x):
        x1 = x[:, :, 0::2, 0::2]
        x2 = x[:, :, 1::2, 0::2]
        x3 = x[:, :, 0::2, 1::2]
        x4 = x[:, :, 1::2, 1::2]
        x_spd = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x_spd)


class PDCConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, pdc_type='cv', theta=0.875):
        super().__init__()
        self.pdc_type = pdc_type
        self.theta = theta
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        if self.pdc_type == 'cv':
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        elif self.pdc_type == 'cd':
            weights_c = self.weight.sum(dim=[2, 3], keepdim=True) * self.theta
            yc = F.conv2d(x, weights_c, stride=self.stride, padding=0, groups=self.groups)
            y = F.conv2d(x, self.weight, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
            return y - yc
        else:
            raise ValueError(f'Unknown pdc_type: {self.pdc_type}')


class PDCBlock(nn.Module):
    def __init__(self, pdc_type, inplane, ouplane, stride=1, theta=0.875):
        super().__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)

        self.conv1 = nn.Sequential(
            PDCConv(inplane, inplane, kernel_size=3, padding=1, groups=inplane, pdc_type=pdc_type, theta=theta),
            nn.BatchNorm2d(inplane),
        )
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        identity = x
        if self.stride > 1:
            identity = self.pool(identity)
        y = self.conv1(x)
        y = self.relu2(y)
        y = self.conv2(y)
        if self.stride > 1:
            identity = self.shortcut(identity)
        y = y + identity
        return y


class DCFM(nn.Module):
    def __init__(self, channels, act='relu', theta=0.875):
        super().__init__()
        self.pdc_cv = PDCBlock(pdc_type='cv', inplane=channels, ouplane=channels, theta=theta)
        self.pdc_cd = PDCBlock(pdc_type='cd', inplane=channels, ouplane=channels, theta=theta)

        self.attention_fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 2, kernel_size=1)
        )

    def forward(self, x):
        diff_cv = self.pdc_cv(x)
        diff_cd = self.pdc_cd(x)

        diff_stack = torch.cat([diff_cv, diff_cd], dim=1)
        attention_weights = self.attention_fc(diff_stack)
        attention_weights = F.softmax(attention_weights, dim=1)

        diff_cv_weighted = diff_cv * attention_weights[:, 0:1, :, :]
        diff_cd_weighted = diff_cd * attention_weights[:, 1:2, :, :]

        fused_features = diff_cv_weighted + diff_cd_weighted
        return fused_features


class OKNetLargeKernel(nn.Module):
    def __init__(self, dim, large_kernel=31):
        super().__init__()
        pad = large_kernel // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.dw_1k = nn.Conv2d(dim, dim, kernel_size=(1, large_kernel), padding=(0, pad), stride=1, groups=dim)
        self.dw_k1 = nn.Conv2d(dim, dim, kernel_size=(large_kernel, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_kk = nn.Conv2d(dim, dim, kernel_size=large_kernel, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)
        out = self.dw_1k(out) + self.dw_k1(out) + self.dw_kk(out) + self.dw_11(out)
        out = self.norm(out)
        out = self.act(out)
        return out


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class FreqScale(nn.Module):
    def __init__(self, dim, group=8, num_filters=4, base_size=8,
                 reweight_ratio=0.25, init_scale=1e-5):
        super().__init__()
        assert dim % group == 0, f'FreqScale: dim({dim}) must be divisible by group({group})'
        self.dim = dim
        self.group = group
        self.num_filters = num_filters
        self.base_size = base_size
        self.filter_size = base_size // 2 + 1

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        reweight_hidden = max(1, int(reweight_ratio * dim))
        self.reweight = nn.Sequential(
            nn.Linear(dim, reweight_hidden, bias=False),
            StarReLU(),
            nn.Linear(reweight_hidden, group * num_filters, bias=False)
        )

        self.complex_weights = nn.Parameter(
            torch.empty(num_filters, dim // group, base_size, self.filter_size,
                        dtype=torch.float32)
        )
        nn.init.trunc_normal_(self.complex_weights, std=init_scale)

    def forward(self, x):
        B, C, H, W = x.shape

        x_in = self.in_conv(x)

        x_rfft = torch.fft.rfft2(x_in.to(torch.float32), dim=(2, 3), norm='ortho')
        _, _, RH, RW = x_rfft.shape

        x_perm = x_in.permute(0, 2, 3, 1)
        routing = self.reweight(x_perm.mean(dim=(1, 2)))
        routing = routing.view(B, self.group, self.num_filters).tanh_()

        weight = self.complex_weights
        if not weight.shape[2:4] == x_rfft.shape[2:4]:
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)
        weight = torch.einsum('bgf,fchw->bgchw', routing, weight)
        weight = weight.reshape(B, C, RH, RW)

        x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        out = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')

        return out


class SCA(nn.Module):
    """Spatial Channel Attention (from OKNet BottleNect).

    x → conv(GAP(x)) * x
    Channel-level spatial attention: each channel gets a scalar weight
    computed from global average pooling, then element-wise multiplied.
    """

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_att = self.conv(self.pool(x))
        return x_att * x


class FGM(nn.Module):
    """Frequency Gating Modulation.

    x → dwconv1(x) ⊙ fft2(dwconv2(x)) → ifft2 → |·| → α·out + β·x
    Spatial-frequency cross modulation with zero-initialized residual.
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='ortho')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(2, 3), norm='ortho')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta


class CCFFBlock(nn.Module):
    """Pure innovation block operating on split_channels.

    LE-DETR EFAM style: the split/identity logic is handled by the encoder,
    this block only processes the split_channels (e.g. 128ch) innovation path.
    No internal residual — the identity branch provides the skip connection.

    v5 changes vs v4:
    - Remove GMM (82.6% params, covered by Large branch)
    - Global: FreqScale → SCA → FGM (aligns with OKNet FCA→SCA→FGM structure)
    """

    def __init__(self, channels, large_kernel=31,
                 fs_group=16, fs_num_filters=4, fs_base_size=14,
                 dcfm_theta=0.875,
                 fs_reweight_ratio=0.25, fs_init_scale=1e-5):
        super().__init__()
        sc = channels

        self.dcfm = DCFM(sc, theta=dcfm_theta)
        self.small_fuse = ConvNormLayer(sc * 2, sc, 1, 1, act='silu')

        self.large_kernel_conv = OKNetLargeKernel(sc, large_kernel=large_kernel)

        fs_group_sc = max(1, fs_group)
        self.freq_scale = FreqScale(sc, group=fs_group_sc, num_filters=fs_num_filters,
                                    base_size=fs_base_size,
                                    reweight_ratio=fs_reweight_ratio, init_scale=fs_init_scale)
        self.sca = SCA(sc)
        self.fgm = FGM(sc)
        self.global_out = ConvNormLayer(sc, sc, 1, 1, act='silu')

        self.fuse_out = ConvNormLayer(sc, sc, 1, 1, act='silu')

    def forward(self, x):
        small_orig = x
        small_dcfm = self.dcfm(x)
        small_cat = torch.cat([small_orig, small_dcfm], dim=1)
        small_out = self.small_fuse(small_cat)

        large_out = self.large_kernel_conv(x)

        f_freq = self.freq_scale(x)
        f_sca = self.sca(f_freq)
        f_cross = self.fgm(f_sca)
        global_out = self.global_out(f_cross)

        out = small_out + large_out + global_out
        out = self.fuse_out(out)

        return out


@register
class HybridEncoderP2SPDOKMFSV5(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 256, 512],
                 feat_strides=[4, 8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[3],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 large_kernel=31,
                 fs_group=16,
                 fs_num_filters=4,
                 fs_base_size=14,
                 split_ratio=0.25,
                 dcfm_theta=0.875,
                 fs_reweight_ratio=0.25,
                 fs_init_scale=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels) - 1)]
        self.out_strides = feat_strides[1:]

        self.input_proj = nn.ModuleList()
        for i, in_channel in enumerate(in_channels):
            if i in [0, 1, 2]:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hidden_dim)
                    )
                )

        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()

        self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
        self.fpn_blocks.append(
            CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
        )
        self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))

        # CCFF: LE-DETR EFAM style
        # Concat[SPDConv(P2)=128, Upsample(Y4)=256, P3=128] = 512ch (不降维)
        # cv1(512→512) → split[128, 384] → 创新(128)+identity(384) → cat[512] → cv2(512→512) → CSPRep(512→256)
        self.ccff_spd_conv = SPDConv(in_channels[0], in_channels[1], act=act)
        ccff_concat_ch = in_channels[1] * 2 + hidden_dim
        self.split_channels = int(ccff_concat_ch * split_ratio)
        self.remaining_channels = ccff_concat_ch - self.split_channels

        self.ccff_cv1 = ConvNormLayer(ccff_concat_ch, ccff_concat_ch, 1, 1, act=act)
        self.ccff_innovation = CCFFBlock(self.split_channels,
                                         large_kernel=large_kernel,
                                         fs_group=fs_group,
                                         fs_num_filters=fs_num_filters,
                                         fs_base_size=fs_base_size,
                                         dcfm_theta=dcfm_theta,
                                         fs_reweight_ratio=fs_reweight_ratio,
                                         fs_init_scale=fs_init_scale)
        self.ccff_cv2 = ConvNormLayer(ccff_concat_ch, ccff_concat_ch, 1, 1, act=act)
        self.ccff_fuse_block = CSPRepLayer(ccff_concat_ch, hidden_dim,
                                           round(3 * depth_mult), act=act, expansion=expansion)

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()

        self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
        self.pan_blocks.append(
            CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
        )

        self.downsample_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act))
        self.pan_blocks.append(
            CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        y5 = self.lateral_convs[0](proj_feats[-1])
        upsample_feat = F.interpolate(y5, scale_factor=2., mode='nearest')
        p4_inner = self.fpn_blocks[0](torch.concat([upsample_feat, proj_feats[-2]], dim=1))
        y4 = self.lateral_convs[1](p4_inner)

        # CCFF: LE-DETR EFAM style
        p2_spd = self.ccff_spd_conv(proj_feats[0])
        y4_up = F.interpolate(y4, scale_factor=2., mode='nearest')
        ccff_input = torch.concat([p2_spd, y4_up, proj_feats[1]], dim=1)

        mixed = self.ccff_cv1(ccff_input)
        ok_branch, identity = torch.split(mixed, [self.split_channels, self.remaining_channels], dim=1)
        innovation_out = self.ccff_innovation(ok_branch)
        fused = torch.cat([innovation_out, identity], dim=1)
        fused = self.ccff_cv2(fused)
        f3 = self.ccff_fuse_block(fused)

        outs = [f3]
        downsample_feat = self.downsample_convs[0](f3)
        f4 = self.pan_blocks[0](torch.concat([downsample_feat, y4], dim=1))
        outs.append(f4)
        downsample_feat = self.downsample_convs[1](f4)
        f5 = self.pan_blocks[1](torch.concat([downsample_feat, y5], dim=1))
        outs.append(f5)

        return outs