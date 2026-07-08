import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .common import ConvNormLayer, FrozenBatchNorm2d
from src.core import register

__all__ = ['SFHF_FFN', 'TokenMixer_For_Local', 'SFHF_FourierUnit',
           'TokenMixer_For_Gloal', 'SFHF_Mixer', 'SFHF_Block',
           'CSFH_C2fBlock', 'CSFHNet']


class SFHF_FFN(nn.Module):
    """Feed-Forward Network with multi-scale depthwise convolutions.

    Splits expanded channels into 4 groups:
    - Group 0: identity (no processing)
    - Group 1: 3x3 DWConv
    - Group 2: 5x5 DWConv
    - Group 3: 7x7 DWConv
    """

    def __init__(self, dim):
        super(SFHF_FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )
        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)
        return x


class TokenMixer_For_Local(nn.Module):
    """Local token mixer using dilated depthwise convolutions.

    Splits channels in half and applies:
    - dilation=1 DWConv on first half
    - dilation=2 DWConv on second half
    """

    def __init__(self, dim):
        super(TokenMixer_For_Local, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1,
                                     padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1,
                                     padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)
        return x


class SFHF_FourierUnit(nn.Module):
    """Fourier domain processing unit with dynamic spectral weighting.

    Performs:
    1. 2D FFT to frequency domain
    2. BN + FPE (frequency position encoding via DWConv)
    3. Dynamic group-weighted spectral convolution
    4. Inverse FFT back to spatial domain
    """

    def __init__(self, in_channels, out_channels, groups=4):
        super(SFHF_FourierUnit, self).__init__()
        self.groups = groups
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.fdc = nn.Conv2d(in_channels=in_channels * 2,
                             out_channels=out_channels * 2 * self.groups,
                             kernel_size=1, stride=1, padding=0,
                             groups=self.groups, bias=True)
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=self.groups,
                      kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1)
        )
        self.fpe = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                             padding=1, stride=1, groups=in_channels * 2,
                             bias=True)

    def forward(self, x):
        batch, c, h, w = x.size()
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(ffted), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(ffted), dim=-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        ffted = rearrange(ffted, 'b c h w d -> b (c d) h w').contiguous()
        ffted = self.bn(ffted)
        ffted = self.fpe(ffted) + ffted
        dy_weight = self.weight(ffted)
        ffted = self.fdc(ffted).view(batch, self.groups, 2 * c, h, -1)
        ffted = torch.einsum('ijkml,ijml->ikml', ffted, dy_weight)
        ffted = F.gelu(ffted)
        ffted = rearrange(ffted, 'b (c d) h w -> b c h w d', d=2).contiguous()
        ffted = torch.view_as_complex(ffted)
        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        return output


class TokenMixer_For_Gloal(nn.Module):
    """Global token mixer using Fourier domain processing (FFC).

    Expands channels, applies SFHF_FourierUnit, then projects back
    with residual connection.
    """

    def __init__(self, dim):
        super(TokenMixer_For_Gloal, self).__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU()
        )
        self.FFC = SFHF_FourierUnit(self.dim * 2, self.dim * 2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.FFC(x)
        x = self.conv_fina(x + x0)
        return x


class SFHF_Mixer(nn.Module):
    """Cross-Spatial-Frequency Mixer combining local and global branches.

    1. Expand channels 2x via 1x1 conv
    2. Split into local and global branches
    3. Local: dilated DWConv (TokenMixer_For_Local)
    4. Global: Fourier domain processing (TokenMixer_For_Gloal)
    5. Concat + channel attention (SE-like) gating
    6. Project back to original channels
    """

    def __init__(self, dim):
        super(SFHF_Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = TokenMixer_For_Local(dim=self.dim)
        self.mixer_gloal = TokenMixer_For_Gloal(dim=self.dim)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim // 2, 2 * dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim, dim=1))
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.cat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)
        return x


class SFHF_Block(nn.Module):
    """Cross-Spatial-Frequency High-frequency Enhancement Block.

    Full block with:
    1. BN + SFHF_Mixer + learnable scale (beta)
    2. BN + SFHF_FFN + learnable scale (gamma)
    Both with residual connections.
    """

    def __init__(self, dim):
        super(SFHF_Block, self).__init__()
        self.dim = dim
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = SFHF_Mixer(dim=self.dim)
        self.ffn = SFHF_FFN(dim=self.dim)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x * self.gamma + copy
        return x


class CSFH_C2fBlock(nn.Module):
    """C2f-style block with SFHF_Block as inner module (faithful to paper).

    Equivalent to the original CSFPR-RTDETR code:
        class CSFH_Block(C2f):
            def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
                super().__init__(c1, c2, n, shortcut, g, e)
                self.m = nn.ModuleList(SFHF_Block(self.c) for _ in range(n))

    C2f structure:
    1. cv1: 1x1 Conv(c1 -> 2*c) with SiLU, c = c2 * expansion
    2. Split into 2 branches of c channels each
    3. Branch 0: identity
    4. Branch 1: pass through n SFHF_Block modules sequentially
    5. Concatenate all intermediate outputs: (2+n)*c channels
    6. cv2: 1x1 Conv((2+n)*c -> c2) with SiLU
    """

    def __init__(self, c1, c2, n=1, e=0.5, act='silu'):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvNormLayer(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = ConvNormLayer((2 + n) * self.c, c2, 1, 1, act=act)
        self.m = nn.ModuleList(SFHF_Block(self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


@register
class CSFHNet(nn.Module):
    """CSFH Feature Extraction Network from CSFPR-RTDETR paper.

    Faithful implementation of the Cross-Spatial-Frequency High-frequency
    enhancement backbone for small object detection in UAV images.

    Architecture (from paper YAML):
        Stem:  Conv(3->64, k=3, s=2)                              -> P1/2
        Stage0: Conv(64->128, s=2) + CSFH_C2fBlock(128, n=1)     -> P2/4
        Stage1: Conv(128->256, s=2) + CSFH_C2fBlock(256, n=1)    -> P3/8
        Stage2: Conv(256->384, s=2) + CSFH_C2fBlock(384, n=1)    -> P4/16
        Stage3: Conv(384->384, s=2) + CSFH_C2fBlock(384, n=3)    -> P5/32

    Args:
        channels_list: output channels for each stage [P2, P3, P4, P5]
        csfh_n: number of SFHF_Block modules inside each CSFH_C2fBlock
        expansion: C2f expansion ratio (hidden_channels = out_channels * expansion)
        return_idx: which stages to return (0=P2, 1=P3, 2=P4, 3=P5)
        freeze_at: freeze stages up to this index (-1 = no freeze)
        freeze_norm: whether to freeze BatchNorm layers
        act: activation function for Conv layers (default 'silu' matching original)
    """

    def __init__(self,
                 channels_list=[128, 256, 384, 384],
                 csfh_n=[1, 1, 1, 3],
                 expansion=0.5,
                 return_idx=[1, 2, 3],
                 freeze_at=-1,
                 freeze_norm=False,
                 act='silu'):
        super().__init__()

        self.return_idx = return_idx

        self.stem = ConvNormLayer(3, 64, 3, 2, act=act)

        in_ch = 64
        self.stages = nn.ModuleList()
        self.strides = []
        self.stage_channels = []

        stride = 4
        for i, (out_ch, n) in enumerate(zip(channels_list, csfh_n)):
            stage = nn.Sequential(
                ConvNormLayer(in_ch, out_ch, 3, 2, act=act),
                CSFH_C2fBlock(out_ch, out_ch, n=n, e=expansion, act=act),
            )
            self.stages.append(stage)
            self.stage_channels.append(out_ch)
            self.strides.append(stride)
            in_ch = out_ch
            stride *= 2

        self.out_channels = [self.stage_channels[i] for i in return_idx]
        self.out_strides = [self.strides[i] for i in return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at + 1, len(self.stages))):
                self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs

    @staticmethod
    def _freeze_parameters(m):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m):
        if isinstance(m, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(m.num_features)
            frozen.weight.data.copy_(m.weight.data)
            frozen.bias.data.copy_(m.bias.data)
            frozen.running_mean.data.copy_(m.running_mean.data)
            frozen.running_var.data.copy_(m.running_var.data)
            return frozen
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m