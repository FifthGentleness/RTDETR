# =========================================
# 文件说明：
# - 该文件的作用：实现RT-DETR的改进混合编码器v2，融合MGDFIS的GMM和PiDiViT的DCFM，
#   以及OKNet的OKNetLargeKernel和FDAM的FreqScale，构建三路并行CCFF融合块
#   v2与v1的区别：
#   - v1: concat[3C] → Conv1x1(3C→C) → CCFFBlock(C) → f3
#   - v2: concat[3C] → CCFFBlock(3C) → CSPRepLayer(3C→C) → f3
#   即v2不先降维，直接将concat的3C通道送入创新模块做全维度增强，
#   再用原版RT-DETR的CSPRepLayer融合块降维到hidden_dim
#   CCFF融合流程：
#   concat[3C] → 三路分岔(Small+DCFM / Large+OKNetLargeKernel / Global(GMM∥FreqScale)) → Add → Residual
#   → CSPRepLayer(3C→C)
#   - DCFM(Difference-Calibrated Fusion Module): 来自PiDiViT，通过中心差分卷积提取边缘/纹理特征
#   - OKNetLargeKernel: 来自OKNet，全向大核分解卷积(dw_1×K + dw_K×1 + dw_K×K + dw_1×1)
#   - GMM(Global Mixing Module): 来自MGDFIS，通过水平/垂直方向的重排卷积实现全局空间混合
#   - FreqScale(Frequency Scale): 来自FDAM的GroupDynamicScale，分组动态频谱调制
#   三路分岔设计：
#   - Small分支: 原始特征 + DCFM差分增强 → Concat → Conv1x1(2C→C)
#   - Large分支: OKNetLargeKernel全向大核空间特征(dw_1×K + dw_K×1 + dw_K×K + dw_1×1)
#   - Global分支: 左分支GMM(空间全局混合) ∥ 右分支FreqScale(分组动态频谱调制)
#                → 自适应融合 α_c·F_space + β_c·F_freq → Conv1x1+BN+Act
#   三路Element-wise Add + 1x1 Conv + Residual Add → CSPRepLayer(3C→C)
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


__all__ = ['HybridEncoderP2SPDOKMFSV2']


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
        range_vec_q = torch.arange(length_q, device=self.embeddings_table.device)
        range_vec_k = torch.arange(length_k, device=self.embeddings_table.device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class GMM(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        self.channels = channels
        self.H = H
        self.W = W
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

        if H != self.H or W != self.W:
            x = F.interpolate(x, size=(self.H, self.W), mode='bilinear', align_corners=False)

        pos_h = self.relate_pos_h(self.H, self.W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(self.H, self.W).unsqueeze(0).permute(0, 3, 1, 2)
        C1 = int(C / self.C)

        x_h = x + pos_h
        x_h = x_h.view(N, C1, self.C, self.H, self.W)

        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, self.H, self.C * self.W)
        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_h = x_h.view(N, C1, self.H, self.C, self.W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, self.H, self.W)
        x_h = self.fuse_h(torch.cat([x_h, x], dim=1))

        x_h = self.activation(self.BN(x_h)) + pos_w
        x_w = self.proj_w(x_h.view(N, C1, self.H * self.C, self.W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = x_w.contiguous().view(N, C1, self.C, self.H, self.W).view(N, C, self.H, self.W)
        out = self.fuse_w(torch.cat([x, x_w], dim=1))

        if H != self.H or W != self.W:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

        return out


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
                 reweight_ratio=0.0625, init_scale=1e-5):
        super().__init__()
        self.dim = dim
        self.group = group
        self.num_filters = num_filters
        self.base_size = base_size
        self.filter_size = base_size // 2 + 1

        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        reweight_hidden = int(reweight_ratio * dim)
        self.reweight = nn.Sequential(
            nn.Linear(dim, reweight_hidden, bias=False),
            StarReLU(),
            nn.Linear(reweight_hidden, group * num_filters, bias=False)
        )

        self.complex_weights = nn.Parameter(
            torch.randn(num_filters, dim // group, base_size, self.filter_size,
                        dtype=torch.float32) * init_scale
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


class CCFFBlock(nn.Module):
    def __init__(self, channels, gmm_h, gmm_w, large_kernel=31, dropout=0.1,
                 fs_group=16, fs_num_filters=4, fs_base_size=14):
        super().__init__()
        self.dcfm = DCFM(channels)
        self.small_fuse = ConvNormLayer(channels * 2, channels, 1, 1, act='silu')

        self.large_kernel_conv = OKNetLargeKernel(channels, large_kernel=large_kernel)

        self.gmm = GMM(channels, gmm_h, gmm_w)
        self.freq_scale = FreqScale(channels, group=fs_group, num_filters=fs_num_filters,
                                    base_size=fs_base_size)
        self.alpha = nn.Parameter(torch.full((channels, 1, 1), 0.5))
        self.beta = nn.Parameter(torch.full((channels, 1, 1), 0.5))
        self.global_out = ConvNormLayer(channels, channels, 1, 1, act='silu')

        self.out_conv = ConvNormLayer(channels, channels, 1, 1, act='silu')

    def forward(self, x):
        identity = x

        small_orig = x
        small_dcfm = self.dcfm(x)
        small_cat = torch.cat([small_orig, small_dcfm], dim=1)
        small_out = self.small_fuse(small_cat)

        large_out = self.large_kernel_conv(x)

        f_space = self.gmm(x)
        f_freq = self.freq_scale(x)
        global_fused = self.alpha * f_space + self.beta * f_freq
        global_out = self.global_out(global_fused)

        out = small_out + large_out + global_out

        out = self.out_conv(out)

        out = out + identity

        return out


@register
class HybridEncoderP2SPDOKMFSV2(nn.Module):
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
                 gmm_w=80,
                 large_kernel=31,
                 fs_group=16,
                 fs_num_filters=4,
                 fs_base_size=14):
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

        # CCFF v2: concat[3C] → CCFFBlock(3C) → CSPRepLayer(3C→C)
        self.ccff_spd_conv = SPDConv(in_channels[0], in_channels[1], act=act)
        ccff_concat_ch = in_channels[1] * 2 + hidden_dim
        self.ccff_block = CCFFBlock(ccff_concat_ch, gmm_h, gmm_w, large_kernel=large_kernel,
                                    dropout=dropout, fs_group=fs_group,
                                    fs_num_filters=fs_num_filters, fs_base_size=fs_base_size)
        self.ccff_fuse_block = CSPRepLayer(ccff_concat_ch, hidden_dim, round(3 * depth_mult),
                                           act=act, expansion=expansion)

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

        # CCFF v2: concat[3C] → CCFFBlock(3C) → CSPRepLayer(3C→C)
        p2_spd = self.ccff_spd_conv(proj_feats[0])
        y4_up = F.interpolate(y4, scale_factor=2., mode='nearest')
        ccff_input = torch.concat([p2_spd, y4_up, proj_feats[1]], dim=1)
        f3 = self.ccff_block(ccff_input)
        f3 = self.ccff_fuse_block(f3)

        outs = [f3]
        downsample_feat = self.downsample_convs[0](f3)
        f4 = self.pan_blocks[0](torch.concat([downsample_feat, y4], dim=1))
        outs.append(f4)
        downsample_feat = self.downsample_convs[1](f4)
        f5 = self.pan_blocks[1](torch.concat([downsample_feat, y5], dim=1))
        outs.append(f5)

        return outs