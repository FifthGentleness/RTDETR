# =========================================
# 文件说明：
# - 该文件的作用：实现RT-DETR的改进混合编码器，融合MGDFIS的GMM和PiDiViT的DCFM，
#   以及OKNet的OmniKernel，构建三路并行CCFF融合块
#   CCFF融合流程：
#   concat[3C] → Conv1x1(3C→C) → OKM → 三路分岔(Small+DCFM / Large / Global(GMM∥FCA+FGSA)) → Add → Residual
#   - GMM(Global Mixing Module): 来自MGDFIS，通过水平/垂直方向的重排卷积实现全局空间混合
#   - OKM(OmniKernel): 来自OKNet，集成OKNetLargeKernel(全向大核分解卷积dw_1×K+dw_K×1+dw_K×K+dw_1×1)、FCA(频域通道注意力)、FGSA(频域门控空间注意力)
#   - DCFM(Difference-Calibrated Fusion Module): 来自PiDiViT，通过中心差分卷积提取边缘/纹理特征
#   三路分岔设计：
#   - Small分支: 原始小尺度特征 + DCFM差分增强 → Concat → Conv1x1(2C→C)
#   - Large分支: 全向大核空间特征(OKNetLargeKernel: dw_1×K + dw_K×1 + dw_K×K + dw_1×1)
#   - Global分支: GMM与FCA+FGSA并行 → Concat → Conv1x1(2C→C)
#   三路Element-wise Add + 1x1 Conv + Residual Add
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


__all__ = ['HybridEncoderP2SPDOKM']


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


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class GMM(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        self.channels = channels
        patch = 4
        self.C = int(channels / patch)
        self.proj_h = nn.Conv2d(H * self.C, self.C * H, (3, 3), stride=1, padding=(1, 1), groups=self.C, bias=True)
        self.proj_w = nn.Conv2d(W * self.C, self.C * W, (3, 3), stride=1, padding=(1, 1), groups=self.C, bias=True)

        self.fuse_h = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)
        self.fuse_w = nn.Conv2d(channels * 2, channels, (1, 1), (1, 1), bias=False)

        self.relate_pos_h = RelativePosition(channels, H)
        self.relate_pos_w = RelativePosition(channels, W)
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)

    def forward(self, x):
        N, C, H, W = x.shape
        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        C1 = int(C / self.C)

        x_h = x + pos_h
        x_h = x_h.view(N, C1, self.C, H, W)

        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C * W)
        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W)
        x_h = self.fuse_h(torch.cat([x_h, x], dim=1))

        x_h = self.activation(self.BN(x_h)) + pos_w
        x_w = self.proj_w(x_h.view(N, C1, H * self.C, W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = x_w.contiguous().view(N, C1, self.C, H, W).view(N, C, H, W)
        x = self.fuse_w(torch.cat([x, x_w], dim=1))
        return x


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
    def __init__(self, pdc_type, inplane, ouplane, stride=1):
        super().__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)

        self.conv1 = nn.Sequential(
            PDCConv(inplane, inplane, kernel_size=3, padding=1, groups=inplane, pdc_type=pdc_type),
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
    def __init__(self, channels, act='relu'):
        super().__init__()
        self.pdc_cv = PDCBlock(pdc_type='cv', inplane=channels, ouplane=channels)
        self.pdc_cd = PDCBlock(pdc_type='cd', inplane=channels, ouplane=channels)

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
    def __init__(self, dim, large_kernel=13):
        super().__init__()
        pad = large_kernel // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, large_kernel), padding=(0, pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(large_kernel, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=large_kernel, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.in_conv(x)
        out = self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out)
        out = self.norm(out)
        out = self.act(out)
        return out


class FGSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.randn(dim, 1, 1) * 0.1)
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))
        self.norm = nn.InstanceNorm2d(dim)

    def forward(self, x):
        res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm='backward')
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, s=fft_size, dim=(-2, -1), norm='backward')
        out = torch.abs(out)
        out = self.norm(out)
        return out * self.alpha + res * self.beta


class OKM(nn.Module):
    def __init__(self, dim, large_kernel=13, dropout=0.1):
        super().__init__()
        self.large_kernel_conv = OKNetLargeKernel(dim, large_kernel=large_kernel)

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)

        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(dim)

        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgsa = FGSA(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_intermediates=False):
        out = self.in_conv(x)

        x_fca = out
        x_att = self.fac_conv(self.fac_pool(x_fca))
        x_fft = torch.fft.fft2(x_fca, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, s=x_fft.size()[-2:], dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)

        x_att_sca = self.conv(self.pool(x_fca))
        x_sca = x_att_sca * x_fca
        fgsa_out = self.fgsa(x_sca)

        large_out = self.large_kernel_conv(out)

        result = x + large_out + fgsa_out
        result = self.norm(result)
        result = self.act(result)
        result = self.dropout(result)
        result = self.out_conv(result)

        if return_intermediates:
            return result, large_out, fgsa_out
        return result


class CCFFBlock(nn.Module):
    def __init__(self, channels, gmm_h, gmm_w, dropout=0.1):
        super().__init__()
        self.gmm = GMM(channels, gmm_h, gmm_w)
        self.okm = OKM(channels, dropout=dropout)

        self.dcfm = DCFM(channels)

        self.small_fuse = ConvNormLayer(channels * 2, channels, 1, 1, act='silu')
        self.global_fuse = ConvNormLayer(channels * 2, channels, 1, 1, act='silu')

        self.out_conv = ConvNormLayer(channels, channels, 1, 1, act='silu')

    def forward(self, x):
        identity = x

        x, large_out, fgsa_out = self.okm(x, return_intermediates=True)

        small_orig = x
        small_dcfm = self.dcfm(x)
        small_cat = torch.cat([small_orig, small_dcfm], dim=1)
        small_out = self.small_fuse(small_cat)

        gmm_out = self.gmm(x)
        global_out = self.global_fuse(torch.cat([fgsa_out, gmm_out], dim=1))

        out = small_out + large_out + global_out

        out = self.out_conv(out)

        out = out + identity

        return out


@register
class HybridEncoderP2SPDOKM(nn.Module):
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
                 gmm_h=80,
                 gmm_w=80):
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

        # channel projection (LE-DETR style: P2/P3 keep backbone raw channels, P4/P5 project to hidden_dim)
        self.input_proj = nn.ModuleList()
        for i, in_channel in enumerate(in_channels):
            if i in [0, 1]:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(hidden_dim)
                    )
                )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn (P5→P4 only)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()

        self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
        self.fpn_blocks.append(
            CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
        )
        self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))

        # CCFF: three-way parallel fusion of P2+P3+P4 at P3 scale
        # concat[3C] → Conv1x1(3C→C) → OKM → 三路分岔(Small+DCFM / Large / Global(GMM∥FCA+FGSA)) → Add → Residual
        self.ccff_spd_conv = SPDConv(in_channels[0], in_channels[1], act=act)
        ccff_concat_ch = in_channels[1] * 2 + hidden_dim
        self.ccff_channel_reduce = ConvNormLayer(ccff_concat_ch, hidden_dim, 1, 1, act=act)
        self.ccff_block = CCFFBlock(hidden_dim, gmm_h, gmm_w, dropout=dropout)

        # bottom-up pan (F3→F4→F5)
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

        # encoder
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

        # top-down fpn (P5→P4)
        y5 = self.lateral_convs[0](proj_feats[-1])
        upsample_feat = F.interpolate(y5, scale_factor=2., mode='nearest')
        p4_inner = self.fpn_blocks[0](torch.concat([upsample_feat, proj_feats[-2]], dim=1))
        y4 = self.lateral_convs[1](p4_inner)

        # CCFF: three-way parallel fusion P2+P3+P4 at P3 scale
        # concat[3C] → Conv1x1(3C→C) → OKM → 三路分岔(Small+DCFM / Large / Global(GMM∥FCA+FGSA)) → Add → Residual
        p2_spd = self.ccff_spd_conv(proj_feats[0])
        y4_up = F.interpolate(y4, scale_factor=2., mode='nearest')
        ccff_input = torch.concat([p2_spd, y4_up, proj_feats[1]], dim=1)
        f3 = self.ccff_channel_reduce(ccff_input)
        f3 = self.ccff_block(f3)

        # bottom-up pan (F3→F4→F5)
        outs = [f3]
        downsample_feat = self.downsample_convs[0](f3)
        f4 = self.pan_blocks[0](torch.concat([downsample_feat, y4], dim=1))
        outs.append(f4)
        downsample_feat = self.downsample_convs[1](f4)
        f5 = self.pan_blocks[1](torch.concat([downsample_feat, y5], dim=1))
        outs.append(f5)

        return outs