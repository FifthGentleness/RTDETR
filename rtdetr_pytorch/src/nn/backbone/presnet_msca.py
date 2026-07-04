import torch
import torch.nn as nn
import torch.nn.functional as F

from .presnet import BasicBlock, PResNet, ResNet_cfg, donwload_url
from src.core import register

__all__ = ['CrossBranchInteraction', 'MSCCA', 'MSCSA', 'MSCA',
           'MSCABasicBlock', 'PResNet_MSCA']


class CrossBranchInteraction(nn.Module):
    """Cross-Branch Channel Attention (CBCA)

    Lightweight cross-branch interaction mechanism.
    Each branch's gate is computed from ALL branches' channel descriptors,
    creating bidirectional information flow before fusion.

    Flow:
    1. GAP each branch -> channel descriptors
    2. Concat all descriptors -> (B, S*C)
    3. Shared MLP -> (B, S*C) -> sigmoid gates
    4. Split gates and modulate each branch independently

    Innovation beyond original paper:
    - Original: simple Concat + 1x1 Fuse (no inter-branch communication)
    - Ours: cross-branch channel attention before fusion
      (each branch informed by what other branches focus on)
    """

    def __init__(self, dim, num_branches=3, reduction=4):
        super().__init__()
        self.num_branches = num_branches
        total_dim = dim * num_branches
        mid = max(total_dim // reduction, 8)

        self.cross_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(total_dim, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, total_dim, bias=False),
        )

    def forward(self, feats):
        cat = torch.cat(feats, dim=1)
        gates = torch.sigmoid(self.cross_attn(cat))
        gates = gates.chunk(self.num_branches, dim=1)
        return [f * g.unsqueeze(-1).unsqueeze(-1) for f, g in zip(feats, gates)]


class MSCCA(nn.Module):
    """Multi-scale Coupled Channel Attention (MSCCA)

    Paper: "Multi-scale coupled attention for visual object detection"
    Formulas: (3) CBS+RS, (7) Cross-covariance attention

    Input/Output: (B, C, H, W) -> (B, C, H, W)

    Flow:
    1. Multi-scale DWConv (3x3, 5x5, 7x7) + CBS channel alignment
    2. CrossBranchInteraction: cross-branch channel attention
    3. Concat + 1x1 Fuse (S*C -> C)
    4. Reshape to (B, N, C), compute Q, K, V
    5. Cross-covariance attention (formula 7):
       L2 norm Q/K -> softmax(Q^T K / sqrt(d)) -> V @ attn
    6. Channel attention: GAP -> sigmoid gating
    """

    def __init__(self, dim, num_scales=3, reduction=4):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        mid_dim = max(dim // reduction, 8)
        self.mid_dim = mid_dim
        self.scale = mid_dim ** -0.5

        kernel_sizes = [3, 5, 7]
        self.ms_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, ks, padding=ks // 2, groups=dim)
            for ks in kernel_sizes
        ])

        self.cbs_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 1, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU(inplace=True),
            )
            for _ in range(num_scales)
        ])

        self.cbi = CrossBranchInteraction(dim, num_scales, reduction)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * num_scales, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

        self.q_proj = nn.Linear(dim, mid_dim, bias=False)
        self.k_proj = nn.Linear(dim, mid_dim, bias=False)
        self.v_proj = nn.Linear(dim, mid_dim, bias=False)

        self.out_proj = nn.Linear(mid_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        ms_feats = []
        for conv, cbs in zip(self.ms_convs, self.cbs_list):
            ms_feats.append(cbs(conv(x)))

        ms_feats = self.cbi(ms_feats)

        ms_cat = torch.cat(ms_feats, dim=1)
        fused = self.fuse(ms_cat)

        X = fused.reshape(B, C, N).transpose(1, 2)

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Q_norm = F.normalize(Q, dim=1)
        K_norm = F.normalize(K, dim=1)
        attn = torch.bmm(Q_norm.transpose(1, 2), K_norm)
        attn = F.softmax(attn * self.scale, dim=-1)
        out = torch.bmm(V, attn)

        out = self.out_proj(out)
        out = self.norm(out + X)

        out = out.transpose(1, 2).reshape(B, C, H, W)

        ca = torch.sigmoid(out.mean(dim=[2, 3], keepdim=True))

        return x * ca


class MSCSA(nn.Module):
    """Multi-scale Coupled Spatial Attention (MSCSA)

    Paper: "Multi-scale coupled attention for visual object detection"
    Formulas: (12) NonLM, (13)-(14) Cross-Gram attention

    Input/Output: (B, C, H, W) -> (B, C, H, W)

    Flow:
    1. Multi-scale DWConv (3x3, 5x5, 7x7)
    2. CrossBranchInteraction: cross-branch channel attention
    3. Concat + 1x1 Fuse (S*C -> C)
    4. Reshape to (B, N, C), compute Q, K, V
    5. NonLM (formula 12): Linear -> BN1d -> ReLU -> Linear -> ELU+1
    6. Cross-Gram attention (formulas 13-14):
       sim = Q_hat @ K_hat^T, normalize, out = sim @ V
    7. Spatial attention: channel-wise pooling -> sigmoid gating
    """

    def __init__(self, dim, num_scales=3, reduction=4):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        mid_dim = max(dim // reduction, 8)
        self.mid_dim = mid_dim

        kernel_sizes = [3, 5, 7]
        self.ms_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, ks, padding=ks // 2, groups=dim)
            for ks in kernel_sizes
        ])

        self.cbi = CrossBranchInteraction(dim, num_scales, reduction)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * num_scales, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

        self.q_proj = nn.Linear(dim, mid_dim, bias=False)
        self.k_proj = nn.Linear(dim, mid_dim, bias=False)
        self.v_proj = nn.Linear(dim, mid_dim, bias=False)

        self.nonlm = nn.Sequential(
            nn.Linear(mid_dim, mid_dim, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, mid_dim, bias=False),
        )

        self.out_proj = nn.Linear(mid_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        ms_feats = [conv(x) for conv in self.ms_convs]

        ms_feats = self.cbi(ms_feats)

        ms_cat = torch.cat(ms_feats, dim=1)
        fused = self.fuse(ms_cat)

        X = fused.reshape(B, C, N).transpose(1, 2)

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Q_flat = Q.reshape(B * N, self.mid_dim)
        K_flat = K.reshape(B * N, self.mid_dim)
        Q_hat = F.elu(self.nonlm(Q_flat)) + 1
        K_hat = F.elu(self.nonlm(K_flat)) + 1
        Q_hat = Q_hat.reshape(B, N, self.mid_dim)
        K_hat = K_hat.reshape(B, N, self.mid_dim)

        sim = torch.bmm(Q_hat, K_hat.transpose(1, 2))
        sim = sim / (sim.sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.bmm(sim, V)

        out = self.out_proj(out)
        out = self.norm(out + X)

        out = out.transpose(1, 2).reshape(B, C, H, W)

        sa = torch.sigmoid(out.mean(dim=1, keepdim=True))

        return x * sa


class MSCA(nn.Module):
    """Multi-scale Coupled Attention = MSCCA + MSCSA

    Paper: "Multi-scale coupled attention for visual object detection"

    Coupled design:
    - MSCCA produces channel-attended features
    - MSCSA takes MSCCA output as input (sequential coupling)
    - Input/Output: (B, C, H, W) -> (B, C, H, W)
    """

    def __init__(self, dim, num_scales=3, reduction=4):
        super().__init__()
        self.mscca = MSCCA(dim, num_scales, reduction)
        self.mscsa = MSCSA(dim, num_scales, reduction)

    def forward(self, x):
        x_ca = self.mscca(x)
        x_sa = self.mscsa(x_ca)
        return x_sa


class MSCABasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.msca = MSCA(ch_out)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.msca(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)
        return out


@register
class PResNet_MSCA(PResNet):
    def __init__(self, depth, variant='d', num_stages=4, return_idx=[0, 1, 2, 3],
                 act='relu', freeze_at=-1, freeze_norm=True, pretrained=False,
                 pretrained_source='paddle'):
        super().__init__(depth, variant, num_stages, return_idx, act,
                         freeze_at=-1, freeze_norm=False, pretrained=False,
                         pretrained_source=pretrained_source)

        assert depth in [18, 34], \
            f"PResNet_MSCA only supports BasicBlock-based models (depth 18/34), got {depth}"

        self._insert_msca(act, variant)

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
                print(f'Load PResNet_MSCA{depth} state_dict from PaddlePaddle')
                if missing:
                    print(f'  MSCA params randomly initialized: {len(missing)} keys')

    def _insert_msca(self, act, variant):
        for stage_idx in [1, 2]:
            blocks = self.res_layers[stage_idx].blocks
            old_block = blocks[1]

            new_block = MSCABasicBlock(
                ch_in=old_block.branch2a.conv.in_channels,
                ch_out=old_block.branch2a.conv.out_channels,
                stride=1,
                shortcut=True,
                act=act,
                variant=variant,
            )

            old_state = old_block.state_dict()
            new_block.load_state_dict(old_state, strict=False)

            blocks[1] = new_block