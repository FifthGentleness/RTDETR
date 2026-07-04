# =========================================
# 文件说明：
# - 该文件的作用：实现RT-DETR的混合编码器(HybridEncoder)，结合CNN和Transformer进行特征编码
# - 在项目中的位置：模型定义 / 编码器(Encoder)模块
# - 与其他文件的关系：
#   - 被：RTDETR模型(rtdetr.py)作为编码器组件调用
#   - 依赖：src.core.register(模型注册), .utils.get_activation(激活函数获取)
# - 核心功能：实现RT-DETR论文中的高效混合编码器，包含三个关键组件：
#   (1)通道投影层：将Backbone输出的多尺度特征统一到隐藏维度
#   (2)Transformer编码器：对最深层的特征进行全局注意力建模
#   (3)FPN+PAN特征融合：自顶向下和自底向上的多尺度特征融合
# =========================================
'''by lyuwenyu
'''

# 导入Python标准库copy模块,用于深拷贝对象以避免引用共享问题
import copy
# 导入PyTorch主库,提供张量运算、自动微分、GPU加速等核心深度学习功能
import torch
# 导入PyTorch神经网络模块,提供各种神经网络层(Conv2d,Linear,MultiheadAttention等)和损失函数
import torch.nn as nn
# 导入PyTorch函数式接口,提供F.pad、F.interpolate等函数式操作
import torch.nn.functional as F

# 从同目录的utils模块导入get_activation函数,用于根据字符串名称获取激活函数
from .utils import get_activation

# 从src.core模块导入register装饰器,用于将模型类注册到框架中实现自动发现
from src.core import register


# 定义模块的公共接口,指定哪些类可以被"from module import *"导入,这里只导出HybridEncoder类
__all__ = ['HybridEncoder']



# =========================================
# 类名: ConvNormLayer
# 类型: nn.Module 子类(卷积神经网络的基础构建单元)
# 代码逻辑链条中的具体职责: 将卷积层、批归一化层、激活函数封装为一个可复用的基础模块,简化CNN网络的构建过程。在HybridEncoder中用于构建通道投影层、FPN侧连接层、PAN下采样层等CNN组件
# =========================================
class ConvNormLayer(nn.Module):
    # 定义卷积归一化层类,继承自nn.Module基类,作为CNN的标准构建块使用
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        # 初始化方法,定义卷积层所需的所有参数
        super().__init__()
        # 调用父类nn.Module的初始化方法,确保正确初始化PyTorch模块的内部状态
        self.conv = nn.Conv2d(
            # 创建2D卷积层实例,作为类的成员变量存储,用于空间特征提取
            ch_in,
            # 输入通道数,对应上一层特征图的通道维度
            ch_out,
            # 输出通道数,决定本层输出特征图的通道维度
            kernel_size,
            # 卷积核的尺寸大小,可以是单个整数或(height,width)元组
            stride,
            # 卷积步长,控制输出特征图的空间尺寸下采样比例
            padding=(kernel_size-1)//2 if padding is None else padding,
            # 填充尺寸计算逻辑:若未指定padding则自动计算使输出尺寸不变的填充值
            bias=bias)
            # 是否添加可学习的偏置项,设置为False时依赖后续BatchNorm层提供
        self.norm = nn.BatchNorm2d(ch_out)
        # 创建2D批归一化层实例,对卷积输出的每个通道进行标准化处理
        self.act = nn.Identity() if act is None else get_activation(act)
        # 根据act参数选择激活函数,若act为None则使用恒等映射,否则调用工厂函数获取激活函数

    def forward(self, x):
        # 定义前向传播方法,实现ConvNormLayer的完整计算流程:卷积 -> 归一化 -> 激活
        return self.act(self.norm(self.conv(x)))
        # 依次执行三个操作:(1)self.conv(x)进行卷积特征提取,(2)self.norm()进行通道归一化,(3)self.act()应用非线性激活
        # 输入维度: x[B,C_in,H,W]
        # 输出维度: x[B,C_out,H',W']


# =========================================
# 类名: RepVggBlock
# 类型: nn.Module 子类(重参数化VGG块)
# 代码逻辑链条中的具体职责: 实现RepVGG风格的重参数化卷积块,训练时使用3x3和1x1卷积分支并行计算,推理时可融合为单个3x3卷积以提升推理速度。在CSPRepLayer中作为瓶颈层使用
# =========================================
class RepVggBlock(nn.Module):
    # 定义RepVggBlock类,继承自nn.Module,实现重参数化VGG块的训练和部署逻辑
    def __init__(self, ch_in, ch_out, act='relu'):
        # 初始化方法,接收输入输出通道数和激活函数类型参数
        super().__init__()
        # 调用父类nn.Module的初始化方法
        self.ch_in = ch_in
        # 保存输入通道数到实例属性,用于convert_to_deploy时创建融合卷积层
        self.ch_out = ch_out
        # 保存输出通道数到实例属性,用于convert_to_deploy时创建融合卷积层
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        # 创建3x3卷积层分支,步长为1,padding为1保持尺寸不变,不使用激活函数(为后续融合做准备)
        # 输入维度: [B,C_in,H,W], 输出维度: [B,C_out,H,W]
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        # 创建1x1卷积层分支,步长为1,不使用填充,不使用激活函数(为后续融合做准备)
        # 输入维度: [B,C_in,H,W], 输出维度: [B,C_out,H,W]
        self.act = nn.Identity() if act is None else get_activation(act)
        # 创建激活函数层,若act为None则使用恒等映射,否则调用工厂函数获取激活函数

    def forward(self, x):
        # 定义前向传播方法,根据是否已部署选择不同的计算路径
        if hasattr(self, 'conv'):
            # 检查是否存在融合后的conv属性,表示已完成部署转换
            y = self.conv(x)
            # 若已部署,使用融合后的单个卷积层进行推理(更快)
            # 输入维度: [B,C_in,H,W], 输出维度: [B,C_out,H,W]
        else:
            # 若未部署,使用训练模式的双分支并行计算
            y = self.conv1(x) + self.conv2(x)
            # 将3x3和1x1两个卷积分支的输出相加,实现多分支特征融合
            # conv1输入维度: [B,C_in,H,W], 输出: [B,C_out,H,W]
            # conv2输入维度: [B,C_in,H,W], 输出: [B,C_out,H,W]
            # 相加后维度: [B,C_out,H,W]

        return self.act(y)
        # 对融合后的卷积输出应用激活函数,引入非线性变换
        # 输入维度: [B,C_out,H,W], 输出维度: [B,C_out,H,W]

    def convert_to_deploy(self):
        # 定义部署转换方法,将训练时的多分支结构转换为推理时的单分支结构以提升速度
        if not hasattr(self, 'conv'):
            # 检查是否尚未创建融合卷积层
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
            # 创建融合后的3x3卷积层,参数与原始conv1相同

        kernel, bias = self.get_equivalent_kernel_bias()
        # 调用方法计算融合后的等效卷积核和偏置,将BN参数融入卷积
        self.conv.weight.data = kernel
        # 将融合后的卷积核数据赋值给融合卷积层
        self.conv.bias.data = bias
        # 将融合后的偏置数据赋值给融合卷积层

    def get_equivalent_kernel_bias(self):
        # 定义获取等效卷积核和偏置的方法,通过数学推导将两个卷积+BN分支融合为一个卷积
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        # 调用_fuse_bn_tensor方法融合3x3卷积分支的BatchNorm参数
        # 返回融合后的3x3卷积核和偏置
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        # 调用_fuse_bn_tensor方法融合1x1卷积分支的BatchNorm参数
        # 返回融合后的1x1卷积核和偏置
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1
        # 将1x1卷积核填充为3x3后与3x3卷积核相加,偏置直接相加,实现两个分支的融合

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        # 定义1x1卷积核填充为3x3的工具方法,便于与3x3卷积核相加
        if kernel1x1 is None:
            # 检查输入卷积核是否为None
            return 0
            # 若为None则返回0,避免后续运算出错
        else:
            # 若卷积核存在,进行填充操作
            return F.pad(kernel1x1, [1, 1, 1, 1])
            # 使用F.pad在四周各填充1个像素,将1x1卷积核变为3x3
            # 输入维度: [C_out,C_in,1,1], 输出维度: [C_out,C_in,3,3]

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        # 定义融合BatchNorm参数的静态方法,将Conv+BN融合为单个卷积
        if branch is None:
            # 检查分支是否存在
            return 0, 0
            # 若分支为None则返回0,表示该分支不存在融合
        kernel = branch.conv.weight
        # 获取卷积层的权重张量
        # shape: [C_out, C_in, K_h, K_w], 对于3x3卷积为[C_out,C_in,3,3]
        running_mean = branch.norm.running_mean
        # 获取BatchNorm的运行均值
        # shape: [C_out]
        running_var = branch.norm.running_var
        # 获取BatchNorm的运行方差
        # shape: [C_out]
        gamma = branch.norm.weight
        # 获取BatchNorm的缩放因子gamma
        # shape: [C_out]
        beta = branch.norm.bias
        # 获取BatchNorm的偏移因子beta
        # shape: [C_out]
        eps = branch.norm.eps
        # 获取BatchNorm的数值稳定性项,防止除零
        std = (running_var + eps).sqrt()
        # 计算标准差,公式: sqrt(var + eps)
        # shape: [C_out]
        t = (gamma / std).reshape(-1, 1, 1, 1)
        # 计算缩放因子t,用于融合BatchNorm的缩放到卷积权重
        # 公式: gamma / std, 然后reshape为[1,C_out,1,1]便于与卷积核广播
        # shape: [1, C_out, 1, 1]
        return kernel * t, beta - running_mean * gamma / std
        # 返回融合后的卷积核和偏置
        # 卷积核融合公式: kernel * t (每个卷积核乘以对应的缩放因子)
        # 偏置融合公式: beta - mean * gamma / std (将BatchNorm的均值和方差融合到偏置)


# =========================================
# 类名: CSPRepLayer
# 类型: nn.Module 子类(CSP重参数化层)
# 代码逻辑链条中的具体职责: 实现CSP(Cross Stage Partial)结构的重参数化层,通过分割特征、并行处理、跨阶段融合来提升特征表达能力并降低计算量。在HybridEncoder中作为FPN和PAN的特征融合块使用
# =========================================
class CSPRepLayer(nn.Module):
    # 定义CSPRepLayer类,继承自nn.Module,实现CSP结构的重参数化特征融合层
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        # 初始化方法,接收输入通道数、输出通道数、块数量、扩展因子、偏置和激活函数参数
        super(CSPRepLayer, self).__init__()
        # 调用父类nn.Module的初始化方法
        hidden_channels = int(out_channels * expansion)
        # 计算隐藏层通道数,通过扩展因子控制中间层的通道维度
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        # 创建第一个1x1卷积层,将输入通道映射到隐藏通道,带激活函数
        # 输入维度: [B,in_channels,H,W], 输出维度: [B,hidden_channels,H,W]
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        # 创建第二个1x1卷积层,同样将输入通道映射到隐藏通道(形成CSP分支)
        # 输入维度: [B,in_channels,H,W], 输出维度: [B,hidden_channels,H,W]
        self.bottlenecks = nn.Sequential(*[
            # 创建由多个RepVggBlock组成的序列容器作为瓶颈层
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
            # 循环创建num_blocks个RepVggBlock,每个块内部进行重参数化特征提取
        ])
        if hidden_channels != out_channels:
            # 检查隐藏通道数是否不等于输出通道数,需要通道维度变换
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
            # 创建1x1卷积层将隐藏通道映射到输出通道
            # 输入维度: [B,hidden_channels,H,W], 输出维度: [B,out_channels,H,W]
        else:
            # 若隐藏通道数等于输出通道数,不需要维度变换
            self.conv3 = nn.Identity()
            # 使用恒等映射,直接传递输入不进行任何变换

    def forward(self, x):
        # 定义前向传播方法,实现CSP结构的特征处理流程
        x_1 = self.conv1(x)
        # 将输入传入第一个卷积分支,进行通道映射和激活
        # 输入维度: [B,in_channels,H,W], 输出维度: [B,hidden_channels,H,W]
        x_1 = self.bottlenecks(x_1)
        # 将第一个分支的输出传入瓶颈层(多个RepVggBlock),进行深层特征提取
        # 输入维度: [B,hidden_channels,H,W], 输出维度: [B,hidden_channels,H,W]
        x_2 = self.conv2(x)
        # 将输入传入第二个卷积分支,作为跳跃连接直接传递原始特征
        # 输入维度: [B,in_channels,H,W], 输出维度: [B,hidden_channels,H,W]
        return self.conv3(x_1 + x_2)
        # 将两个分支的输出相加(CSP跨阶段融合),然后通过第三个卷积层映射到输出通道
        # x_1 + x_2 维度: [B,hidden_channels,H,W]
        # 最终输出维度: [B,out_channels,H,W]


# transformer
# 以下开始定义Transformer编码器相关组件

# =========================================
# 类名: TransformerEncoderLayer
# 类型: nn.Module 子类(Transformer编码器层)
# 代码逻辑链条中的具体职责: 实现Transformer编码器的基本层结构,包含多头自注意力和前馈神经网络两个子层,以及残差连接和层归一化。在HybridEncoder中对Backbone特征进行全局注意力建模,捕获长距离依赖关系
# =========================================
class TransformerEncoderLayer(nn.Module):
    # 定义TransformerEncoderLayer类,继承自nn.Module,实现Transformer编码器的基本层
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        # 初始化方法,接收模型维度、头数、前馈维度、dropout率、激活函数和归一化位置参数
        super().__init__()
        # 调用父类nn.Module的初始化方法
        self.normalize_before = normalize_before
        # 保存normalize_before参数,决定是Pre-LN还是Post-LN的归一化方式
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        # 创建多头自注意力模块,batch_first=True表示输入输出格式为[Batch,Seq,Features]
        # 参数说明: d_model(模型维度), nhead(注意力头数), dropout(dropout比例)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # 创建第一个线性层,将模型维度映射到前馈维度(扩展)
        # 输入维度: [B,Seq,d_model], 输出维度: [B,Seq,dim_feedforward]
        self.dropout = nn.Dropout(dropout)
        # 创建Dropout层,在训练时随机丢弃部分神经元以防止过拟合
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 创建第二个线性层,将前馈维度映射回模型维度(压缩)
        # 输入维度: [B,Seq,dim_feedforward], 输出维度: [B,Seq,d_model]

        self.norm1 = nn.LayerNorm(d_model)
        # 创建第一个LayerNorm层,对自注意力输出进行归一化
        self.norm2 = nn.LayerNorm(d_model)
        # 创建第二个LayerNorm层,对前馈网络输出进行归一化
        self.dropout1 = nn.Dropout(dropout)
        # 创建Dropout层,用于自注意力输出的残差连接
        self.dropout2 = nn.Dropout(dropout)
        # 创建Dropout层,用于前馈网络输出的残差连接
        self.activation = get_activation(activation)
        # 创建激活函数,用于前馈网络内部非线性变换

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        # 定义静态方法,将位置编码添加到输入张量中
        return tensor if pos_embed is None else tensor + pos_embed
        # 若位置编码为None则直接返回原张量,否则将位置编码添加到输入张量
        # tensor维度: [B,Seq,d_model], pos_embed维度: [1,Seq,d_model]或[1,H*W,d_model]
        # 输出维度: [B,Seq,d_model]

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        # 定义前向传播方法,实现Transformer编码器层的完整计算流程
        residual = src
        # 保存残差连接的原输入,用于后续残差加法
        if self.normalize_before:
            # 检查是否使用Pre-LN(先归一化)方式
            src = self.norm1(src)
            # 若normalize_before为True,则在自注意力前进行归一化(Pre-LN)
        q = k = self.with_pos_embed(src, pos_embed)
        # 将位置编码添加到查询和键向量,使得注意力具有位置感知能力
        # q和k维度: [B,Seq,d_model]
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        # 执行多头自注意力计算,返回注意力输出和注意力权重(权重被丢弃)
        # 输入q,k维度: [B,Seq,d_model], value维度: [B,Seq,d_model]
        # 输出src维度: [B,Seq,d_model]

        src = residual + self.dropout1(src)
        # 将原始输入与自注意力输出进行残差连接,并应用Dropout
        # residual维度: [B,Seq,d_model], src维度: [B,Seq,d_model]
        if not self.normalize_before:
            # 检查是否使用Post-LN(后归一化)方式
            src = self.norm1(src)
            # 若normalize_before为False,则在残差连接后进行归一化(Post-LN)

        residual = src
        # 更新残差变量,用于前馈网络的残差连接
        if self.normalize_before:
            # 检查是否使用Pre-LN方式
            src = self.norm2(src)
            # 若normalize_before为True,则在前馈网络前进行归一化
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # 执行前馈网络计算:线性投影->激活->Dropout->线性投影
        # linear1输入维度: [B,Seq,d_model], 输出: [B,Seq,dim_feedforward]
        # activation输出维度: [B,Seq,dim_feedforward]
        # dropout输出维度: [B,Seq,dim_feedforward]
        # linear2输出维度: [B,Seq,d_model]
        src = residual + self.dropout2(src)
        # 将原始输入与前馈网络输出进行残差连接,并应用Dropout
        if not self.normalize_before:
            # 检查是否使用Post-LN方式
            src = self.norm2(src)
            # 若normalize_before为False,则在残差连接后进行归一化
        return src
        # 返回Transformer编码器层的最终输出
        # 输出维度: [B,Seq,d_model]


# =========================================
# 类名: TransformerEncoder
# 类型: nn.Module 子类(Transformer编码器堆叠)
# 代码逻辑链条中的具体职责: 将多个TransformerEncoderLayer堆叠在一起形成完整的Transformer编码器,实现深层特征编码。在HybridEncoder中控制编码器深度,通常使用1-6层
# =========================================
class TransformerEncoder(nn.Module):
    # 定义TransformerEncoder类,继承自nn.Module,实现多层Transformer编码器的堆叠
    def __init__(self, encoder_layer, num_layers, norm=None):
        # 初始化方法,接收编码器层实例、层数和归一化层参数
        super(TransformerEncoder, self).__init__()
        # 调用父类nn.Module的初始化方法
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        # 创建ModuleList存储多个编码器层,使用深拷贝确保每个层独立
        # 深拷贝避免共享参数,每个层有独立的权重
        self.num_layers = num_layers
        # 保存编码器层数量到实例属性
        self.norm = norm
        # 保存最终归一化层,若为None则不进行最终归一化

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        # 定义前向传播方法,依次通过所有编码器层
        output = src
        # 初始化输出为输入src
        for layer in self.layers:
            # 遍历每一层编码器
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
            # 将当前输出传入下一层编码器,依次处理
            # 每层输出维度: [B,Seq,d_model]
        if self.norm is not None:
            # 检查是否存在最终归一化层
            output = self.norm(output)
            # 若存在归一化层,则对最终输出进行归一化
        return output
        # 返回经过所有编码器层处理后的最终输出


# 使用register装饰器注册HybridEncoder类,使其能被框架自动发现和管理
@register

# =========================================
# 类名: HybridEncoder
# 类型: nn.Module 子类(RT-DETR混合编码器核心组件)
# 代码逻辑链条中的具体职责: 实现RT-DETR的高效混合编码器,是论文的核心创新点之一。包含三个关键组件:
# (1)通道投影层:将Backbone多尺度特征统一到隐藏维度256;
# (2)Transformer编码器:对最深尺度特征(如stride=32)进行全局注意力建模;
# (3)FPN+PAN双向特征融合:自顶向下和自底向上融合多尺度特征,增强特征的语义信息和空间细节
# =========================================
class HybridEncoder(nn.Module):
    # 定义HybridEncoder类,继承自nn.Module,实现RT-DETR的高效混合编码器
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 # 输入通道数列表,对应Backbone输出的三个尺度特征图的通道数
                 # 默认值[512,1024,2048]对应ResNet50的C3/C4/C5输出通道数
                 feat_strides=[8, 16, 32],
                 # 特征图步长列表,表示每个尺度相对于输入图像的下采样倍数
                 # [8,16,32]对应1/8、1/16、1/32分辨率的特征图
                 hidden_dim=256,
                 # 隐藏维度,统一所有特征到这个维度,是Transformer的模型维度
                 nhead=8,
                 # 多头注意力的头数,用于Transformer编码器
                 dim_feedforward = 1024,
                 # 前馈网络的隐藏维度,Transformer编码器中FFN的中间层维度
                 dropout=0.0,
                 # Dropout比例,用于Transformer编码器的正则化
                 enc_act='gelu',
                 # 编码器激活函数类型,默认使用GELU(Transformer常用)
                 use_encoder_idx=[2],
                 # 使用Transformer编码器的特征层索引列表,默认只对最深尺度(C5)使用
                 # [2]表示只对stride=32的特征使用Transformer编码器
                 num_encoder_layers=1,
                 # Transformer编码器的层数,控制编码器深度
                 pe_temperature=10000,
                 # 位置编码的温度参数,用于2D sin-cos位置编码的频率计算
                 expansion=1.0,
                 # CSPRepLayer的扩展因子,控制中间层通道数
                 depth_mult=1.0,
                 # 深度乘数,用于调整CSPRepLayer中RepVggBlock的数量
                 act='silu',
                 # CNN部分使用的激活函数类型,默认使用SiLU
                 eval_spatial_size=None):
                # 评估时的空间尺寸,用于预计算位置编码,格式为[H,W]
        super().__init__()
        # 调用父类nn.Module的初始化方法
        self.in_channels = in_channels
        # 保存输入通道数列表到实例属性
        self.feat_strides = feat_strides
        # 保存特征图步长列表到实例属性
        self.hidden_dim = hidden_dim
        # 保存隐藏维度到实例属性
        self.use_encoder_idx = use_encoder_idx
        # 保存编码器使用的特征层索引到实例属性
        self.num_encoder_layers = num_encoder_layers
        # 保存编码器层数到实例属性
        self.pe_temperature = pe_temperature
        # 保存位置编码温度参数到实例属性
        self.eval_spatial_size = eval_spatial_size
        # 保存评估空间尺寸到实例属性

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        # 计算输出通道数列表,所有输出都统一到hidden_dim维度
        # 列表长度与输入通道数相同,每个元素都是hidden_dim
        self.out_strides = feat_strides
        # 保存输出步长列表,与输入步长相同(融合后不改变空间分辨率)
        
        # channel projection
        # 以下构建通道投影层,将Backbone输出的不同通道数统一到hidden_dim
        self.input_proj = nn.ModuleList()
        # 创建ModuleList存储多个投影层,每个尺度特征对应一个投影层
        for in_channel in in_channels:
            # 遍历每个尺度的输入通道数
            self.input_proj.append(
                # 将投影层添加到ModuleList
                nn.Sequential(
                    # 使用Sequential容器组合卷积和归一化
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    # 创建1x1卷积层,将输入通道映射到隐藏维度,不添加偏置(后续BN会处理)
                    # 输入维度: [B,in_channel,H,W], 输出维度: [B,hidden_dim,H,W]
                    nn.BatchNorm2d(hidden_dim)
                    # 创建批归一化层,对隐藏维度进行标准化,稳定训练
                )
            )

        # encoder transformer
        # 以下构建Transformer编码器,对选定尺度的特征进行全局注意力建模
        encoder_layer = TransformerEncoderLayer(
            # 创建单个Transformer编码器层实例
            hidden_dim,
            # 模型维度,与隐藏维度相同
            nhead=nhead,
            # 多头注意力的头数
            dim_feedforward=dim_feedforward,
            # 前馈网络维度
            dropout=dropout,
            # Dropout比例
            activation=enc_act)
            # 激活函数类型

        self.encoder = nn.ModuleList([
            # 创建ModuleList存储多个Transformer编码器
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
            # 为每个使用编码器的尺度创建一个独立的Transformer编码器
            # 深拷贝确保每个编码器有独立的参数
        ])

        # top-down fpn
        # 以下构建自顶向下的FPN(Feature Pyramid Network)特征融合路径
        self.lateral_convs = nn.ModuleList()
        # 创建ModuleList存储侧连接卷积层
        self.fpn_blocks = nn.ModuleList()
        # 创建ModuleList存储FPN融合块
        for _ in range(len(in_channels) - 1, 0, -1):
            # 倒序遍历,从最深尺度到次深尺度
            # 例如in_channels长度为3时,遍历range(2,0,-1),即[2,1]
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            # 添加1x1侧连接卷积层,从高分辨率特征提取语义信息
            # 不改变空间尺寸和通道数,仅进行通道调整
            # 输入维度: [B,hidden_dim,H,W], 输出维度: [B,hidden_dim,H,W]
            self.fpn_blocks.append(
                # 添加FPN融合块,使用CSPRepLayer实现跨尺度特征融合
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                # 输入维度: [B,hidden_dim*2,H,W](高分辨率特征上采样后与低分辨率特征拼接), 输出维度: [B,hidden_dim,H,W]
            )

        # bottom-up pan
        # 以下构建自底向上的PAN(Path Aggregation Network)特征融合路径
        self.downsample_convs = nn.ModuleList()
        # 创建ModuleList存储下采样卷积层
        self.pan_blocks = nn.ModuleList()
        # 创建ModuleList存储PAN融合块
        for _ in range(len(in_channels) - 1):
            # 正向遍历,从浅到深
            # 例如in_channels长度为3时,遍历range(2),即[0,1]
            self.downsample_convs.append(
                # 添加下采样卷积层,用于降低空间分辨率
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
                # 使用3x3卷积,步长2,实现2倍下采样
                # 输入维度: [B,hidden_dim,H,W], 输出维度: [B,hidden_dim,H/2,W/2]
            )
            self.pan_blocks.append(
                # 添加PAN融合块,与FPN块结构相同
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                # 输入维度: [B,hidden_dim*2,H,W](下采样特征与深尺度特征拼接), 输出维度: [B,hidden_dim,H,W]
            )

        self._reset_parameters()
        # 调用参数初始化方法,对所有参数进行初始化

    def _reset_parameters(self):
        # 定义参数初始化方法,用于预计算评估时的位置编码
        if self.eval_spatial_size:
            # 检查是否指定了评估空间尺寸
            for idx in self.use_encoder_idx:
                # 遍历所有使用编码器的特征层索引
                stride = self.feat_strides[idx]
                # 获取当前特征层的步长
                pos_embed = self.build_2d_sincos_position_embedding(
                    # 调用方法生成2D sin-cos位置编码
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    # 计算位置编码的宽度和高度(空间尺寸除以步长)
                    self.hidden_dim, self.pe_temperature)
                    # 传入隐藏维度和温度参数
                setattr(self, f'pos_embed{idx}', pos_embed)
                # 使用setattr将位置编码设置为实例属性,属性名为pos_embed{idx}

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        # 定义静态方法,生成2D正弦余弦位置编码
        # 用于为Transformer编码器提供空间位置信息
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        # 创建宽度方向的坐标张量,值为0到w-1的整数
        # shape: [w]
        grid_h = torch.arange(int(h), dtype=torch.float32)
        # 创建高度方向的坐标张量,值为0到h-1的整数
        # shape: [h]
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        # 创建2D坐标网格,使用meshgrid生成网格点坐标
        # grid_w维度: [w,h], grid_h维度: [w,h]
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
            # 断言embed_dim能被4整除,因为需要分别编码宽度和高度方向
        pos_dim = embed_dim // 4
        # 计算每个空间维度使用的编码维度,总维度除以4(宽度sin/cos + 高度sin/cos)
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        # 创建频率基础值,从0到1均匀分布
        # shape: [pos_dim]
        omega = 1. / (temperature ** omega)
        # 计算温度衰减的频率,温度越高低频成分越多
        # shape: [pos_dim]

        out_w = grid_w.flatten()[..., None] @ omega[None]
        # 计算宽度方向的频率编码
        # grid_w.flatten()维度: [w*h], [None]后: [w*h,1]
        # omega[None]维度: [1,pos_dim]
        # 矩阵乘法结果维度: [w*h,pos_dim]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        # 计算高度方向的频率编码
        # 与out_w相同的处理流程

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]
        # 拼接四个部分:宽度sin、宽度cos、高度sin、高度cos
        # torch.concat维度: [w*h, pos_dim*4] = [w*h, embed_dim/4*4] = [w*h, embed_dim]
        # [None,:,:]后维度: [1, w*h, embed_dim]
        # 最终返回2D位置编码,维度: [1, H*W, embed_dim]

    def forward(self, feats):
        # 定义前向传播方法,实现混合编码器的完整计算流程
        assert len(feats) == len(self.in_channels)
        # 断言输入特征数量与配置的通道数列表长度一致
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # 对所有尺度的特征进行通道投影,统一到hidden_dim
        # feats列表中每个元素维度: [B,in_channels[i],H_i,W_i]
        # proj_feats列表中每个元素维度: [B,hidden_dim,H_i,W_i]
        
        # encoder
        # 以下对选定尺度的特征进行Transformer编码
        if self.num_encoder_layers > 0:
            # 检查是否使用了Transformer编码器(num_encoder_layers > 0)
            for i, enc_ind in enumerate(self.use_encoder_idx):
                # 遍历所有使用编码器的特征层索引
                h, w = proj_feats[enc_ind].shape[2:]
                # 获取当前特征层的空间尺寸(高度和宽度)
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                # 将特征图展平并转换维度顺序为[Batch, Seq, Channel]
                # flatten(2)后维度: [B,hidden_dim,H*W]
                # permute(0,2,1)后维度: [B,H*W,hidden_dim]
                if self.training or self.eval_spatial_size is None:
                    # 检查是否在训练模式或未指定评估尺寸
                    pos_embed = self.build_2d_sincos_position_embedding(
                        # 动态生成位置编码
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                        # 传入宽度、高度、隐藏维度和温度参数,并将张量移到与src_flatten相同的设备
                else:
                    # 在评估模式且指定了评估尺寸
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)
                    # 获取预计算的位置编码,并移动到相同设备
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                # 将展平的特征和位置编码传入Transformer编码器
                # src_flatten维度: [B,H*W,hidden_dim], pos_embed维度: [1,H*W,hidden_dim]
                # memory维度: [B,H*W,hidden_dim]
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # 将编码后的特征重塑回原始空间格式
                # permute(0,2,1)后维度: [B,hidden_dim,H*W]
                # reshape(-1,hidden_dim,h,w)后维度: [B,hidden_dim,h,w]
                # contiguous()确保张量在内存中是连续的

        # broadcasting and fusion
        # 以下进行FPN+PAN双向特征融合
        inner_outs = [proj_feats[-1]]
        # 初始化内部输出列表,以最深尺度的特征作为起点
        # proj_feats[-1]维度: [B,hidden_dim,H_max,W_max]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # 倒序遍历,从深到浅进行FPN融合
            # 遍历范围: [2,1](假设有3个尺度特征)
            feat_high = inner_outs[0]
            # 获取上一轮融合后的高分辨率特征
            # feat_high维度: [B,hidden_dim,H,W]
            feat_low = proj_feats[idx - 1]
            # 获取当前层对应的低分辨率特征
            # feat_low维度: [B,hidden_dim,H*2,W*2]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            # 通过侧连接卷积调整高分辨率特征的通道数
            # lateral_convs索引计算: len(in_channels)-1-idx
            # 输入维度: [B,hidden_dim,H,W], 输出维度: [B,hidden_dim,H,W]
            inner_outs[0] = feat_high
            # 更新inner_outs[0]为调整后的高分辨率特征
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            # 对高分辨率特征进行上采样,恢复到与低分辨率特征相同的空间尺寸
            # 输入维度: [B,hidden_dim,H,W], 输出维度: [B,hidden_dim,H*2,W*2]
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            # 将上采样的高分辨率特征与低分辨率特征在通道维度拼接,然后通过FPN融合块
            # torch.concat维度: [B,hidden_dim*2,H*2,W*2]
            # FPN输出维度: [B,hidden_dim,H*2,W*2]
            inner_outs.insert(0, inner_out)
            # 将融合结果插入到inner_outs的开头,形成从深到浅的特征序列

        outs = [inner_outs[0]]
        # 初始化输出列表,以FPN最浅层的融合特征作为起点
        for idx in range(len(self.in_channels) - 1):
            # 正向遍历,从浅到深进行PAN融合
            # 遍历范围: [0,1](假设有3个尺度特征)
            feat_low = outs[-1]
            # 获取上一轮PAN融合后的低分辨率特征
            feat_high = inner_outs[idx + 1]
            # 获取对应的深尺度特征(来自FPN)
            downsample_feat = self.downsample_convs[idx](feat_low)
            # 通过下采样卷积降低低分辨率特征的空间尺寸
            # 输入维度: [B,hidden_dim,H,W], 输出维度: [B,hidden_dim,H/2,W/2]
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            # 将下采样的低分辨率特征与深尺度特征在通道维度拼接,然后通过PAN融合块
            # torch.concat维度: [B,hidden_dim*2,H/2,W/2]
            # PAN输出维度: [B,hidden_dim,H/2,W/2]
            outs.append(out)
            # 将PAN融合结果添加到输出列表

        return outs
        # 返回经过FPN+PAN双向融合后的多尺度特征列表
        # outs列表长度与输入特征数量相同,每个元素维度: [B,hidden_dim,H_i,W_i]