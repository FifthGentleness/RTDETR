# =========================================
# 文件说明：
# - 该文件的作用：实现RT-DETR的改进混合编码器，在基线HybridEncoder基础上做三项改进：
#   (1)添加P2尺度：Backbone输出从[C3,C4,C5]扩展为[C2,C3,C4,C5]
#   (2)P2上使用SPD-Conv：P2经SPD-Conv下采样到P3尺度参与CCFF融合
#   (3)P2/P3/P4采用CCFF三路并行融合：SPDConv(P2)+Upsample(Y4)+P3三路Concat→MFFM→DCFM，替代原始逐级FPN融合
#   P2和P3采用LE-DETR方式: 不经过input_proj, 保留backbone原始通道直接参与CCFF融合, 减少计算量
#   MFFM和DCFM严格对齐PiDiViT源码(PropagationLayer)的架构，作为独立模块串联:
#   - MFFM = PropagationLayer前半部分: 5路多尺度卷积 + scale_attention + conv_fusion
#   - DCFM = PropagationLayer后半部分: PDCBlock(cv) + PDCBlock(cd) + attention_fc加权融合
#   MFFM在DCFM之前独立执行，数据流: concat特征 → MFFM(多尺度融合) → DCFM(差分通道融合)
# - 在项目中的位置：模型定义 / 编码器(Encoder)模块
# - 与其他文件的关系：
#   - 被：RTDETR模型(rtdetr.py)作为编码器组件调用
#   - 依赖：src.core.register(模型注册), .utils.get_activation(激活函数获取)
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


__all__ = ['HybridEncoderP2SPDMFFMDCFM']


# =========================================
# 类名: SPDConv
# 类型: nn.Module 子类(Space-to-Depth卷积下采样模块)
# 代码逻辑链条中的具体职责: 实现SPD-Conv下采样，将空间像素重排到通道维度，
# 保留细粒度空间信息，避免stride-conv的信息丢失。在PAN路径中用于P2→P3的下采样
# 输入: [B, C, H, W] → 输出: [B, out_channels, H/2, W/2]
# =========================================
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


# =========================================
# 类名: PDCConv
# 类型: nn.Module 子类(像素差分卷积)
# 代码逻辑链条中的具体职责: 实现PiDiViT中的像素差分卷积(Pixel Difference Convolution)
# 支持cv(中心差分/普通卷积)和cd(中心差分)两种pdc_type
# 对齐PiDiViT源码: lib/ops_theta.py createConvFunc
# =========================================
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


# =========================================
# 类名: PDCBlock
# 类型: nn.Module 子类(像素差分卷积块)
# 代码逻辑链条中的具体职责: 对齐PiDiViT源码中的PDCBlock
# 架构: DWConv(PDC) → ReLU → Conv1x1 → + shortcut
# 对齐PiDiViT源码: lib/regionprop_update.py PDCBlock
# =========================================
class PDCBlock(nn.Module):
    def __init__(self, pdc_type, inplane, ouplane, stride=1):
        super().__init__()
        self.stride = stride
        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)

        self.conv1 = nn.Sequential(
            PDCConv(pdc_type, inplane, inplane, kernel_size=3, padding=1, groups=inplane, pdc_type=pdc_type),
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


# =========================================
# 类名: MFFM
# 类型: nn.Module 子类(Multi-Feature Fusion Module)
# 代码逻辑链条中的具体职责: 对齐PiDiViT源码PropagationLayer前半部分
# 5路多尺度卷积(1x1, 3x3, 5x5, 7x7, dilated 3x3) + scale_attention(SE-style) + conv_fusion
# 输入: [B, in_channels, H, W] → 输出: [B, out_channels, H, W]
# 对齐PiDiViT源码: lib/regionprop_update.py PropagationLayer (conv1x1~conv_dilated + scale_attention + conv_fusion)
# =========================================
class MFFM(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu'):
        super().__init__()
        _act = nn.ReLU()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv7x7 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv_dilated = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 5, out_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, 5, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_fusion = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x4 = self.conv7x7(x)
        x5 = self.conv_dilated(x)

        multi_scale_features = torch.cat([x1, x2, x3, x4, x5], dim=1)
        scale_weights = self.scale_attention(multi_scale_features)
        scale_weights = scale_weights.view(-1, 5, 1, 1, 1)

        x = (scale_weights[:, 0] * x1 +
             scale_weights[:, 1] * x2 +
             scale_weights[:, 2] * x3 +
             scale_weights[:, 3] * x4 +
             scale_weights[:, 4] * x5)

        x = self.conv_fusion(x)
        return x


# =========================================
# 类名: DCFM
# 类型: nn.Module 子类(Dual-Channel Fusion Module)
# 代码逻辑链条中的具体职责: 对齐PiDiViT源码PropagationLayer后半部分
# PDCBlock(cv) + PDCBlock(cd) + attention_fc(softmax加权融合)
# 输入: [B, channels, H, W] → 输出: [B, channels, H, W]
# 对齐PiDiViT源码: lib/regionprop_update.py PropagationLayer (pdc_cv + pdc_cd + attention_fc)
# =========================================
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


# =========================================
# 类名: MFFMDCFMBlock
# 类型: nn.Module 子类(MFFM+DCFM组合融合块)
# 代码逻辑链条中的具体职责: 将MFFM和DCFM串联组合，作为FPN/PAN中P2/P3/P4尺度的融合块
# 替代原始CSPRepLayer，严格对齐PiDiViT的PropagationLayer架构
# MFFM: 5路多尺度卷积 + scale_attention + conv_fusion
# DCFM: PDCBlock(cv) + PDCBlock(cd) + attention_fc加权融合
# 输入: [B, in_channels, H, W] → 输出: [B, out_channels, H, W]
# =========================================
class MFFMDCFMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu'):
        super().__init__()
        self.mffm = MFFM(in_channels, out_channels, act=act)
        self.dcfm = DCFM(out_channels, act=act)

    def forward(self, x):
        x = self.mffm(x)
        x = self.dcfm(x)
        return x


@register
# =========================================
# 类名: HybridEncoderP2SPDMFFMDCFM
# 类型: nn.Module 子类(改进RT-DETR混合编码器)
# 代码逻辑链条中的具体职责: 在HybridEncoder基线上做三项改进:
# (1)添加P2尺度: Backbone输出[C2,C3,C4,C5], strides=[4,8,16,32]
# (2)P2上使用SPD-Conv: P2经SPD-Conv下采样到P3尺度参与CCFF融合
# (3)P2/P3/P4采用CCFF三路并行融合: SPDConv(P2)+Upsample(Y4)+P3三路Concat→MFFM→DCFM
#    替代原始逐级FPN融合(P4→P3→P2), 对齐LE-DETR的CCFF方案
# P2和P3采用LE-DETR方式: 不经过input_proj, 保留backbone原始通道直接参与CCFF融合
# MFFM和DCFM作为独立模块串联，MFFM在DCFM之前，严格对齐PiDiViT源码PropagationLayer架构
# =========================================
class HybridEncoderP2SPDMFFMDCFM(nn.Module):
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
                 eval_spatial_size=None):
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
        # LE-DETR style: SPDConv outputs in_channels[1](=128) to align with LE-DETR
        # concat channels = in_channels[1](SPDConv_P2=128) + hidden_dim(Upsample_Y4=256) + in_channels[1](P3_raw=128) = 512
        # Plan B: MFFM在512维做特征提取 → Conv1x1降通道到256 → DCFM在256维做差分融合
        self.ccff_spd_conv = SPDConv(in_channels[0], in_channels[1], act=act)
        ccff_concat_ch = in_channels[1] * 2 + hidden_dim
        self.ccff_mffm = MFFM(ccff_concat_ch, ccff_concat_ch, act='relu')
        self.ccff_channel_reduce = ConvNormLayer(ccff_concat_ch, hidden_dim, 1, 1, act=act)
        self.ccff_dcfm = DCFM(hidden_dim, act='relu')

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
        p2_spd = self.ccff_spd_conv(proj_feats[0])
        y4_up = F.interpolate(y4, scale_factor=2., mode='nearest')
        ccff_input = torch.concat([p2_spd, y4_up, proj_feats[1]], dim=1)
        f3 = self.ccff_mffm(ccff_input)
        f3 = self.ccff_channel_reduce(f3)
        f3 = self.ccff_dcfm(f3)

        # bottom-up pan (F3→F4→F5)
        outs = [f3]
        downsample_feat = self.downsample_convs[0](f3)
        f4 = self.pan_blocks[0](torch.concat([downsample_feat, y4], dim=1))
        outs.append(f4)
        downsample_feat = self.downsample_convs[1](f4)
        f5 = self.pan_blocks[1](torch.concat([downsample_feat, y5], dim=1))
        outs.append(f5)

        return outs