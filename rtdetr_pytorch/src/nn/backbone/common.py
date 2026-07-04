# =========================================
# 文件说明：
# - 该文件的作用：定义RT-DETR骨干网络(Backbone)的基础组件库
# - 在项目中的位置：模型定义 / 骨干网络底层模块
# - 与其他文件的关系：
#   - 依赖：被presnet.py(PResNet骨干网络)、hgnetv2.py(HGNetv2骨干网络)等调用
#   - 提供：ConvNormLayer(卷积归一化层)、FrozenBatchNorm2d(冻结批归一化层)、get_activation(激活函数工厂)
# - 核心功能：本文件是RT-DETR骨干网络的底层基础组件库，包含三个核心组件：
#   (1)ConvNormLayer封装卷积-归一化-激活的标准CNN结构；
#   (2)FrozenBatchNorm2d实现冻结批归一化用于迁移学习；
#   (3)get_activation激活函数工厂函数
# =========================================
'''by lyuwenyu
'''

# 导入PyTorch主库,提供张量运算、自动微分、GPU加速等核心深度学习功能
import torch
# 导入PyTorch神经网络模块,提供各种神经网络层(Conv2d,Linear,BatchNorm等)和损失函数
import torch.nn as nn


# =========================================
# 类名: ConvNormLayer
# 类型: nn.Module 子类(卷积神经网络的基础构建单元)
# 代码逻辑链条中的具体职责: 将卷积层、批归一化层、激活函数封装为一个可复用的基础模块,简化CNN网络的构建过程。在RT-DETR的Backbone(如PResNet)中作为基本卷积单元使用,负责特征提取和空间尺寸变换
# =========================================
class ConvNormLayer(nn.Module):
    # 定义卷积归一化层类,继承自nn.Module基类,作为CNN的标准构建块使用
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        # 初始化方法,定义卷积层所需的所有参数:输入通道数、输出通道数、卷积核大小、步长、填充、偏置、激活函数
        super().__init__()
        # 调用父类nn.Module的初始化方法,确保正确初始化PyTorch模块的内部状态(如parameters,buffers等)
        self.conv = nn.Conv2d(
            # 创建2D卷积层实例,作为类的成员变量存储,用于空间特征提取
            ch_in,
            # 输入通道数,对应上一层特征图的通道维度,决定输入特征的深度
            ch_out,
            # 输出通道数,决定本层输出特征图的通道维度,也决定了本层卷积核的数量
            kernel_size,
            # 卷积核的尺寸大小,可以是单个整数(正方形核)或(height,width)元组
            stride,
            # 卷积步长,控制输出特征图的空间尺寸下采样比例,stride=2时高宽各减半
            padding=(kernel_size-1)//2 if padding is None else padding,
            # 填充尺寸计算逻辑:若未指定padding则自动计算使输出尺寸不变的填充值,公式为(kernel_size-1)//2
            bias=bias)
            # 是否添加可学习的偏置项,设置为False时偏置由后续BatchNorm层提供,设为True则两者叠加
        self.norm = nn.BatchNorm2d(ch_out)
        # 创建2D批归一化层实例,对卷积输出的每个通道进行标准化处理,加速网络训练收敛
        self.act = nn.Identity() if act is None else get_activation(act)
        # 根据act参数选择激活函数,若act为None则使用恒等映射,否则调用工厂函数获取指定的激活函数

    def forward(self, x):
        # 定义前向传播方法,实现ConvNormLayer的完整计算流程:卷积 -> 归一化 -> 激活
        return self.act(self.norm(self.conv(x)))
        # 依次执行三个操作:(1)self.conv(x)进行卷积特征提取,(2)self.norm()进行通道归一化,(3)self.act()应用非线性激活
        # 输入维度: x[B,C_in,H,W]
        # 输出维度: x[B,C_out,H',W']


# =========================================
# 类名: FrozenBatchNorm2d
# 类型: nn.Module 子类(迁移学习专用归一化层)
# 代码逻辑链条中的具体职责: 实现冻结批归一化层,在迁移学习场景中固定预训练的统计量(均值、方差)和仿射参数(gamma、beta),避免微调时破坏预训练模型学到的特征分布
# =========================================
class FrozenBatchNorm2d(nn.Module):
    # 定义冻结批归一化层类,继承自nn.Module,用于推理阶段或迁移学习时固定归一化参数
    # 代码来源: copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    # 说明: BatchNorm2d where the batch statistics and the affine parameters are fixed
    # Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than torchvision.models.resnet[18,34,50,101] produce nans
    def __init__(self, num_features, eps=1e-5):
        # 初始化方法,接收特征通道数和数值稳定项参数
        super(FrozenBatchNorm2d, self).__init__()
        # 调用父类nn.Module的初始化方法,确保正确初始化PyTorch模块的内部状态
        n = num_features
        # 将num_features赋值给局部变量n,简写变量名便于后续使用
        self.register_buffer("weight", torch.ones(n))
        # 注册weight缓冲区为全1的张量,作为固定的缩放因子gamma,用于对归一化后的特征进行仿射变换
        # shape: (n,), 数值: 全1
        self.register_buffer("bias", torch.zeros(n))
        # 注册bias缓冲区为全0的张量,作为固定的偏移因子beta,用于调整归一化特征的中心位置
        # shape: (n,), 数值: 全0
        self.register_buffer("running_mean", torch.zeros(n))
        # 注册running_mean缓冲区存储预训练阶段计算得到的均值统计量,在推理时用于归一化
        # shape: (n,), 数值: 全0
        self.register_buffer("running_var", torch.ones(n))
        # 注册running_var缓冲区存储预训练阶段计算得到的方差统计量,在推理时用于归一化
        # shape: (n,), 数值: 全1
        self.eps = eps
        # 将eps赋值给实例属性,作为数值稳定性项,防止在计算方差倒数时出现除零错误
        # 默认值: 1e-5
        self.num_features = n
        # 将特征数赋值给实例属性,用于记录该层的通道维度信息

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 重写从state_dict加载预训练权重的方法,处理FrozenBatchNorm的特殊兼容性需求
        # 参数说明: state_dict(状态字典),prefix(键前缀),local_metadata(本地元数据),strict(严格检查),missing_keys(缺失键列表),unexpected_keys(意外键列表),error_msgs(错误信息列表)
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        # 拼接得到num_batches_tracked键的完整名称,用于标识标准BatchNorm跟踪的batch数量
        if num_batches_tracked_key in state_dict:
            # 检查state_dict中是否存在num_batches_tracked键
            del state_dict[num_batches_tracked_key]
            # 删除该键,因为FrozenBatchNorm不需要跟踪训练过程中的batch数量,删除可以避免加载时的兼容性问题

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            # 调用父类nn.Module的_load_from_state_dict方法,完成剩余状态字典项的加载
            state_dict, prefix, local_metadata, strict,
            # 将所有参数原样传递给父类方法
            missing_keys, unexpected_keys, error_msgs)
            # 继续传递缺失键、意外键和错误信息列表

    def forward(self, x):
        # 定义前向传播方法,使用预训练的固定统计量对输入进行归一化
        w = self.weight.reshape(1, -1, 1, 1)
        # 将weight从(n,)重塑为(1,n,1,1)以便与4D输入张量进行广播运算
        # 输入维度: (n,)
        # 输出维度: (1,n,1,1)
        b = self.bias.reshape(1, -1, 1, 1)
        # 将bias从(n,)重塑为(1,n,1,1)以便与4D输入张量进行广播运算
        # 输入维度: (n,)
        # 输出维度: (1,n,1,1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        # 将running_var从(n,)重塑为(1,n,1,1)以便与4D输入张量进行广播运算
        # 输入维度: (n,)
        # 输出维度: (1,n,1,1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        # 将running_mean从(n,)重塑为(1,n,1,1)以便与4D输入张量进行广播运算
        # 输入维度: (n,)
        # 输出维度: (1,n,1,1)
        scale = w * (rv + self.eps).rsqrt()
        # 计算缩放因子 scale = weight / sqrt(running_var + eps),rsqrt()是平方根倒数函数,比先sqrt再取倒数数值更稳定
        # 输入维度: (1,n,1,1)
        # 输出维度: (1,n,1,1)
        bias = b - rm * scale
        # 计算偏移量 bias = bias - running_mean * scale,对归一化后的特征进行中心调整
        # 输入维度: (1,n,1,1)
        # 输出维度: (1,n,1,1)
        return x * scale + bias
        # 应用仿射变换完成归一化: y = x * scale + bias,等价于标准的(x - mean) / sqrt(var + eps) * gamma + beta公式
        # 输入维度: x(B,C,H,W) 和 scale/bias(1,C,1,1)
        # 输出维度: x(B,C,H,W)

    def extra_repr(self):
        # 定义额外的字符串表示方法,用于自定义模块打印输出时的信息展示格式
        return (
            # 返回格式化的描述字符串
            "{num_features}, eps={eps}".format(**self.__dict__))
            # 使用format方法将num_features和eps值插入到模板字符串中,提供模块的关键参数信息
            # 返回示例: "128, eps=1e-5"


# =========================================
# 函数名: get_activation
# 类型: 普通函数(激活函数工厂函数)
# 代码逻辑链条中的具体职责: 根据字符串名称或模块实例动态创建对应的PyTorch激活函数模块,统一管理RT-DETR网络中使用的各种非线性激活操作
# =========================================
def get_activation(act: str, inpace: bool=True):
    # 定义激活函数工厂函数,根据字符串名称返回对应的PyTorch激活函数模块实例
    act = act.lower()
    # 将激活函数名称转换为小写字母,统一匹配格式,避免大小写不一致导致的判断失败
    # 输入示例: "SiLU"
    # 输出示例: "silu"
    
    if act == 'silu':
        # 判断是否为SiLU(Swish)激活函数,SiLU是RT-DETR的默认选择,平滑且非单调
        m = nn.SiLU()
        # 创建SiLU激活函数实例,数学公式为x * sigmoid(x),在负区间仍有梯度流动,避免神经元死亡问题

    elif act == 'relu':
        # 判断是否为ReLU激活函数,经典且计算高效的激活函数
        m = nn.ReLU()
        # 创建ReLU激活函数实例,数学公式为max(0,x),计算简单但可能导致部分神经元永久死亡

    elif act == 'leaky_relu':
        # 判断是否为LeakyReLU激活函数,改进版ReLU避免神经元死亡
        m = nn.LeakyReLU()
        # 创建LeakyReLU激活函数实例,数学公式为max(0.01x,x),在负区间保留小梯度保持信息流动

    elif act == 'silu':
        # 再次判断是否为SiLU(代码中存在重复判断逻辑,可优化但保持原样以确保功能正确)
        m = nn.SiLU()
        # 再次创建SiLU激活函数实例,与前面的判断处理相同
    
    elif act == 'gelu':
        # 判断是否为GELU激活函数,Transformer系列模型中广泛使用
        m = nn.GELU()
        # 创建GELU激活函数实例,数学公式为x * Φ(x),其中Φ是标准正态分布的累积分布函数,具有概率意义
        
    elif act is None:
        # 判断act参数是否为None,表示不需要激活函数
        m = nn.Identity()
        # 创建恒等映射模块,forward时直接返回输入不做任何变换,用于某些不需要非线性的层
    
    elif isinstance(act, nn.Module):
        # 判断act是否已经是nn.Module实例,提供直接传入模块的灵活性
        m = act
        # 直接使用传入的激活函数模块实例,无需创建新实例

    else:
        # 以上所有条件都不满足时的错误处理
        raise RuntimeError('')
        # 抛出运行时错误,表示传入的激活函数名称不支持,提示开发者检查输入参数

    if hasattr(m, 'inplace'):
        # 检查创建的激活函数模块是否有inplace属性(某些激活函数支持原地操作)
        m.inplace = inpace
        # 设置inplace属性为指定值,True时激活函数直接修改输入张量节省显存,False时创建新张量存储结果
    
    return m
    # 返回创建好的激活函数模块实例,供调用者使用
    # 返回类型: nn.Module子类实例