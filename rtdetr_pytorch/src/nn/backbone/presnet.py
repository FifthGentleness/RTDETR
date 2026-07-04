# =========================================
# 文件说明：
# - 该文件的作用：实现PResNet（Paddle ResNet）骨干网络，用于RT-DETR模型的特征提取
# - 在项目中的位置：模型定义 / 骨干网络（Backbone）
# - 与其他文件的关系：
#   - 依赖：common.py（ConvNormLayer、get_activation、FrozenBatchNorm2d）
#   - 被：rtdetr.py（RTDETR类）作为backbone组件调用
#   - 配置：configs/*.yaml（配置文件指定depth、variant等参数）
# =========================================

'''by lyuwenyu
'''
# 文件作者署名注释，说明代码由lyuwenyu编写

import torch
# 导入PyTorch深度学习框架核心库，提供张量计算和自动求导功能

import torch.nn as nn 
# 导入PyTorch神经网络模块，包含各种网络层（卷积、线性、池化等）

import torch.nn.functional as F 
# 导入PyTorch函数式接口，提供激活函数、损失函数、池化等函数式操作

from collections import OrderedDict
# 导入OrderedDict有序字典，用于构建有序的网络层序列（保证层的顺序）

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d
# 从同目录的common模块导入：
# - get_activation：获取激活函数的工厂函数
# - ConvNormLayer：卷积+批归一化+激活的组合层
# - FrozenBatchNorm2d：冻结的批归一化层（推理时使用，不更新统计量）

from src.core import register
# 从src.core模块导入register装饰器，用于将模型类注册到框架中


__all__ = ['PResNet']
# 定义模块的公共接口，指定哪些类可以被"from module import *"导入，这里只导出PResNet类


ResNet_cfg = {
    # 定义ResNet不同深度的配置字典，键为网络深度，值为每个stage的block数量
    18: [2, 2, 2, 2],
    # ResNet-18配置：4个stage，每个stage分别包含2、2、2、2个BasicBlock
    34: [3, 4, 6, 3],
    # ResNet-34配置：4个stage，每个stage分别包含3、4、6、3个BasicBlock
    50: [3, 4, 6, 3],
    # ResNet-50配置：4个stage，每个stage分别包含3、4、6、3个BottleNeck
    101: [3, 4, 23, 3],
    # ResNet-101配置：4个stage，每个stage分别包含3、4、23、3个BottleNeck
    # 152: [3, 8, 36, 3],
    # ResNet-152配置（注释掉）：4个stage，每个stage分别包含3、8、36、3个BottleNeck
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}

torchvision_model_fn = {
    18: 'resnet18',
    34: 'resnet34',
    50: 'resnet50',
    101: 'resnet101',
}


# =========================================
# 类的整体作用：BasicBlock是ResNet的基础残差块，用于浅层网络（ResNet-18/34）
# 核心方法说明：
# - __init__: 初始化残差块，构建两个3x3卷积层和可能的shortcut连接
# - forward: 执行前向传播，实现残差连接
# 是否为关键模块：✅ 是（ResNet的基本构建单元）
# =========================================
class BasicBlock(nn.Module):
    # 定义BasicBlock类，继承自nn.Module基类，这是ResNet-18和ResNet-34使用的基础残差块
    expansion = 1
    # 类变量，定义输出通道数的扩展倍数，BasicBlock的expansion为1（输出通道数=ch_out）

    # =========================================
    # 函数作用：初始化BasicBlock残差块，构建两个3x3卷积层和shortcut连接
    # 输入参数：
    # - ch_in: 输入通道数
    # - ch_out: 输出通道数
    # - stride: 第一个卷积层的步长，用于下采样
    # - shortcut: 是否使用shortcut连接（True表示直接相加，False表示需要1x1卷积调整维度）
    # - act: 激活函数类型，默认为'relu'
    # - variant: ResNet变体类型（'a', 'b', 'c', 'd'），影响shortcut的实现方式
    # 返回值：无（初始化方法）
    # 核心逻辑：
    # 1. 判断是否需要shortcut连接的1x1卷积
    # 2. 构建两个3x3卷积层（branch2a和branch2b）
    # 3. 设置激活函数
    # 调用关系：
    # - 被Blocks类的__init__调用
    # 重要性：核心函数（ResNet的基本构建单元）
    # =========================================
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        # 初始化方法，接收输入输出通道数、步长、shortcut标志、激活函数类型和变体类型
        super().__init__()
        # 调用父类nn.Module的初始化方法，完成PyTorch模块的必要初始化

        self.shortcut = shortcut
        # 保存shortcut标志，True表示输入输出维度相同可以直接相加，False需要1x1卷积调整

        if not shortcut:
            # 如果shortcut为False，说明输入输出维度不同，需要通过1x1卷积调整维度
            if variant == 'd' and stride == 2:
                # 如果是变体'd'且步长为2，使用平均池化+1x1卷积的shortcut（PaddlePaddle的ResNet-D变体）
                self.short = nn.Sequential(OrderedDict([
                    # 使用OrderedDict构建有序的序列模块，保证层的顺序
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    # 添加平均池化层，kernel_size=2, stride=2, padding=0, ceil_mode=True（向上取整）
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                    # 添加1x1卷积层，调整通道数，步长为1
                ]))
            else:
                # 如果不是变体'd'或步长不为2，直接使用1x1卷积调整维度
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)
                # 创建1x1卷积层，调整通道数为ch_out，步长为stride（用于下采样）

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        # 创建第一个3x3卷积层（branch2a），输入通道ch_in，输出通道ch_out，步长stride，带激活函数
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        # 创建第二个3x3卷积层（branch2b），输入输出通道均为ch_out，步长1，不带激活函数
        self.act = nn.Identity() if act is None else get_activation(act) 
        # 创建激活函数层，如果act为None则使用恒等映射（不改变输入），否则通过工厂函数获取激活函数


    # =========================================
    # 函数作用：执行BasicBlock的前向传播，实现残差连接
    # 输入参数：
    # - x: 输入特征张量，形状为[B, ch_in, H, W]
    # 返回值：
    # - out: 输出特征张量，形状为[B, ch_out, H/stride, W/stride]
    # 核心逻辑：
    # 1. 通过两个3x3卷积层处理输入
    # 2. 根据shortcut标志选择直接连接或卷积调整
    # 3. 将卷积输出与shortcut相加（残差连接）
    # 4. 应用激活函数
    # 调用关系：
    # - 被Blocks类的forward调用
    # 重要性：核心函数
    # =========================================
    def forward(self, x):
        # 前向传播方法，接收输入特征张量x
        out = self.branch2a(x)
        # 将输入x传入第一个3x3卷积层branch2a，提取特征
        out = self.branch2b(out)
        # 将branch2a的输出传入第二个3x3卷积层branch2b，进一步提取特征
        if self.shortcut:
            # 如果shortcut为True，说明输入输出维度相同
            short = x
            # 直接将输入x作为shortcut，无需调整维度
        else:
            # 如果shortcut为False，说明输入输出维度不同
            short = self.short(x)
            # 将输入x传入shortcut卷积层，调整维度以匹配输出
        
        out = out + short
        # 将卷积输出与shortcut相加，实现残差连接（这是ResNet的核心创新）
        out = self.act(out)
        # 对相加后的结果应用激活函数，引入非线性

        return out
        # 返回处理后的特征张量


# =========================================
# 类的整体作用：BottleNeck是ResNet的瓶颈残差块，用于深层网络（ResNet-50/101/152）
# 核心方法说明：
# - __init__: 初始化瓶颈块，构建1x1-3x3-1x1的卷积结构和shortcut连接
# - forward: 执行前向传播，实现残差连接
# 是否为关键模块：✅ 是（ResNet的基本构建单元）
# =========================================
class BottleNeck(nn.Module):
    # 定义BottleNeck类，继承自nn.Module基类，这是ResNet-50、101、152使用的瓶颈残差块
    expansion = 4
    # 类变量，定义输出通道数的扩展倍数，BottleNeck的expansion为4（输出通道数=ch_out*4）

    # =========================================
    # 函数作用：初始化BottleNeck瓶颈残差块，构建1x1-3x3-1x1的卷积结构和shortcut连接
    # 输入参数：
    # - ch_in: 输入通道数
    # - ch_out: 中间层通道数（输出通道数=ch_out*4）
    # - stride: 3x3卷积层的步长，用于下采样
    # - shortcut: 是否使用shortcut连接（True表示直接相加，False表示需要1x1卷积调整维度）
    # - act: 激活函数类型，默认为'relu'
    # - variant: ResNet变体类型（'a', 'b', 'c', 'd'），影响步长分配和shortcut的实现方式
    # 返回值：无（初始化方法）
    # 核心逻辑：
    # 1. 根据variant类型确定步长分配（变体'a'将步长放在第一个1x1卷积，其他放在3x3卷积）
    # 2. 构建1x1-3x3-1x1的卷积结构（降低维度-特征提取-提升维度）
    # 3. 判断是否需要shortcut连接的1x1卷积
    # 4. 设置激活函数
    # 调用关系：
    # - 被Blocks类的__init__调用
    # 重要性：核心函数（ResNet的基本构建单元）
    # =========================================
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        # 初始化方法，接收输入输出通道数、步长、shortcut标志、激活函数类型和变体类型
        super().__init__()
        # 调用父类nn.Module的初始化方法，完成PyTorch模块的必要初始化

        if variant == 'a':
            # 如果是变体'a'（PaddlePaddle的ResNet-A变体），将步长放在第一个1x1卷积
            stride1, stride2 = stride, 1
            # 第一个1x1卷积步长为stride，3x3卷积步长为1
        else:
            # 如果是其他变体（'b', 'c', 'd'），将步长放在3x3卷积
            stride1, stride2 = 1, stride
            # 第一个1x1卷积步长为1，3x3卷积步长为stride

        width = ch_out 
        # 保存中间层通道数（瓶颈宽度）

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        # 创建第一个1x1卷积层（branch2a），降低维度到width，步长为stride1，带激活函数
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        # 创建3x3卷积层（branch2b），保持维度为width，步长为stride2，带激活函数
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)
        # 创建第二个1x1卷积层（branch2c），提升维度到ch_out*expansion，步长为1，不带激活函数

        self.shortcut = shortcut
        # 保存shortcut标志，True表示输入输出维度相同可以直接相加，False需要1x1卷积调整维度
        if not shortcut:
            # 如果shortcut为False，说明输入输出维度不同，需要通过1x1卷积调整维度
            if variant == 'd' and stride == 2:
                # 如果是变体'd'且步长为2，使用平均池化+1x1卷积的shortcut（PaddlePaddle的ResNet-D变体）
                self.short = nn.Sequential(OrderedDict([
                    # 使用OrderedDict构建有序的序列模块，保证层的顺序
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    # 添加平均池化层，kernel_size=2, stride=2, padding=0, ceil_mode=True（向上取整）
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                    # 添加1x1卷积层，调整通道数为ch_out*expansion，步长为1
                ]))
            else:
                # 如果不是变体'd'或步长不为2，直接使用1x1卷积调整维度
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)
                # 创建1x1卷积层，调整通道数为ch_out*expansion，步长为stride（用于下采样）

        self.act = nn.Identity() if act is None else get_activation(act) 
        # 创建激活函数层，如果act为None则使用恒等映射（不改变输入），否则通过工厂函数获取激活函数

    # =========================================
    # 函数作用：执行BottleNeck的前向传播，实现残差连接
    # 输入参数：
    # - x: 输入特征张量，形状为[B, ch_in, H, W]
    # 返回值：
    # - out: 输出特征张量，形状为[B, ch_out*4, H/stride, W/stride]
    # 核心逻辑：
    # 1. 通过1x1-3x3-1x1卷积层处理输入（降维-特征提取-升维）
    # 2. 根据shortcut标志选择直接连接或卷积调整
    # 3. 将卷积输出与shortcut相加（残差连接）
    # 4. 应用激活函数
    # 调用关系：
    # - 被Blocks类的forward调用
    # 重要性：核心函数
    # =========================================
    def forward(self, x):
        # 前向传播方法，接收输入特征张量x
        out = self.branch2a(x)
        # 将输入x传入第一个1x1卷积层branch2a，降低维度
        out = self.branch2b(out)
        # 将branch2a的输出传入3x3卷积层branch2b，提取特征
        out = self.branch2c(out)
        # 将branch2b的输出传入第二个1x1卷积层branch2c，提升维度

        if self.shortcut:
            # 如果shortcut为True，说明输入输出维度相同
            short = x
            # 直接将输入x作为shortcut，无需调整维度
        else:
            # 如果shortcut为False，说明输入输出维度不同
            short = self.short(x)
            # 将输入x传入shortcut卷积层，调整维度以匹配输出

        out = out + short
        # 将卷积输出与shortcut相加，实现残差连接（这是ResNet的核心创新）
        out = self.act(out)
        # 对相加后的结果应用激活函数，引入非线性

        return out
        # 返回处理后的特征张量


# =========================================
# 类的整体作用：Blocks是ResNet的一个stage，由多个BasicBlock或BottleNeck组成
# 核心方法说明：
# - __init__: 初始化一个stage，包含多个残差块
# - forward: 执行stage的前向传播，依次通过所有残差块
# 是否为关键模块：✅ 是（ResNet的stage构建单元）
# =========================================
class Blocks(nn.Module):
    # 定义Blocks类，继承自nn.Module基类，这是ResNet的一个stage，包含多个残差块
    # =========================================
    # 函数作用：初始化ResNet的一个stage，包含多个残差块
    # 输入参数：
    # - block: 残差块类型（BasicBlock或BottleNeck）
    # - ch_in: 输入通道数
    # - ch_out: 输出通道数（中间层通道数）
    # - count: 该stage包含的残差块数量
    # - stage_num: stage编号（2, 3, 4, 5），用于判断是否需要下采样
    # - act: 激活函数类型，默认为'relu'
    # - variant: ResNet变体类型，影响残差块的实现
    # 返回值：无（初始化方法）
    # 核心逻辑：
    # 1. 创建指定数量的残差块
    # 2. 第一个残差块可能需要下采样（stride=2），其余残差块stride=1
    # 3. 第一个残差块需要shortcut卷积，其余直接连接
    # 调用关系：
    # - 被PResNet类的__init__调用
    # 重要性：核心函数
    # =========================================
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        # 初始化方法，接收残差块类型、输入输出通道数、块数量、stage编号等参数
        super().__init__()
        # 调用父类nn.Module的初始化方法，完成PyTorch模块的必要初始化

        self.blocks = nn.ModuleList()
        # 创建一个ModuleList，用于存储该stage的所有残差块
        for i in range(count):
            # 循环创建count个残差块
            self.blocks.append(
                block(
                    ch_in, 
                    # 输入通道数
                    ch_out,
                    # 输出通道数（中间层通道数）
                    stride=2 if i == 0 and stage_num != 2 else 1, 
                    # 步长：如果是第一个块且不是stage2，则stride=2（下采样），否则stride=1
                    shortcut=False if i == 0 else True,
                    # shortcut：第一个块需要shortcut卷积（False），其余直接连接（True）
                    variant=variant,
                    # ResNet变体类型
                    act=act)
                # 激活函数类型
            )

            if i == 0:
                # 如果是第一个残差块
                ch_in = ch_out * block.expansion
                # 更新输入通道数为输出通道数（考虑expansion因子），用于下一个残差块

    # =========================================
    # 函数作用：执行stage的前向传播，依次通过所有残差块
    # 输入参数：
    # - x: 输入特征张量，形状为[B, ch_in, H, W]
    # 返回值：
    # - out: 输出特征张量，形状为[B, ch_out*expansion, H/2, W/2]（如果下采样）
    # 核心逻辑：
    # 1. 依次通过所有残差块
    # 2. 每个残差块执行残差连接
    # 调用关系：
    # - 被PResNet类的forward调用
    # 重要性：核心函数
    # =========================================
    def forward(self, x):
        # 前向传播方法，接收输入特征张量x
        out = x
        # 将输入x赋值给out，作为初始输出
        for block in self.blocks:
            # 遍历该stage的所有残差块
            out = block(out)
            # 将当前输出传入下一个残差块，依次处理
        return out
        # 返回stage的最终输出


@register
# 使用register装饰器注册PResNet类，使其能被框架自动发现和管理
# =========================================
# 类的整体作用：PResNet是完整的ResNet骨干网络，用于提取图像的多尺度特征
# 核心方法说明：
# - __init__: 初始化ResNet网络，构建stem、4个stage和输出配置
# - forward: 执行前向传播，提取多尺度特征图
# - _freeze_parameters: 冻结模块参数，不参与训练
# - _freeze_norm: 冻结批归一化层，转换为FrozenBatchNorm2d
# 是否为关键模块：✅ 是（RT-DETR的backbone组件）
# =========================================
class PResNet(nn.Module):
    # 定义PResNet类，继承自nn.Module基类，这是完整的ResNet骨干网络实现
    # =========================================
    # 函数作用：初始化PResNet骨干网络，构建完整的ResNet架构
    # 输入参数：
    # - depth: ResNet深度（18, 34, 50, 101）
    # - variant: ResNet变体类型（'a', 'b', 'c', 'd'），影响stem和shortcut的实现
    # - num_stages: stage数量，默认为4
    # - return_idx: 返回哪些stage的输出，默认为[0, 1, 2, 3]（返回所有stage）
    # - act: 激活函数类型，默认为'relu'
    # - freeze_at: 冻结到哪个stage（-1表示不冻结，0表示冻结conv1，1表示冻结stage1等）
    # - freeze_norm: 是否冻结批归一化层，默认为True
    # - pretrained: 是否加载预训练权重，默认为False
    # 返回值：无（初始化方法）
    # 核心逻辑：
    # 1. 根据depth和variant构建stem（初始卷积层）
    # 2. 构建4个stage，每个stage包含多个残差块
    # 3. 配置输出通道数和步长
    # 4. 可选地冻结参数和批归一化层
    # 5. 可选地加载预训练权重
    # 调用关系：
    # - 被RTDETR类的__init__调用（作为backbone）
    # 重要性：核心函数
    # =========================================
    def __init__(
        self, 
        # 初始化方法开始
        depth, 
        # ResNet深度（18, 34, 50, 101）
        variant='d', 
        # ResNet变体类型，默认为'd'（PaddlePaddle的ResNet-D变体）
        num_stages=4, 
        # stage数量，默认为4
        return_idx=[0, 1, 2, 3], 
        # 返回哪些stage的输出，默认返回所有4个stage
        act='relu',
        # 激活函数类型，默认为'relu'
        freeze_at=-1, 
        # 冻结到哪个stage，-1表示不冻结
        freeze_norm=True, 
        pretrained=False,
        pretrained_source='paddle'):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        # 从配置字典中获取该深度ResNet每个stage的block数量
        ch_in = 64
        # 初始输入通道数为64
        if variant in ['c', 'd']:
            # 如果是变体'c'或'd'（PaddlePaddle的ResNet变体），使用3个3x3卷积作为stem
            conv_def = [
                # 定义stem的卷积配置列表
                [3, ch_in // 2, 3, 2, "conv1_1"],
                # 第一个卷积：输入3通道，输出32通道，3x3卷积，步长2，名称conv1_1
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                # 第二个卷积：输入32通道，输出32通道，3x3卷积，步长1，名称conv1_2
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
                # 第三个卷积：输入32通道，输出64通道，3x3卷积，步长1，名称conv1_3
            ]
        else:
            # 如果是变体'a'或'b'（原始ResNet），使用一个7x7卷积作为stem
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]
            # 定义stem的卷积配置：输入3通道，输出64通道，7x7卷积，步长2，名称conv1_1

        self.conv1 = nn.Sequential(OrderedDict([
            # 创建有序的序列模块作为stem（初始卷积层）
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
            # 使用列表推导式创建卷积层，每个配置创建一个ConvNormLayer
        ]))
        # 构建stem，根据variant类型可能是1个7x7卷积或3个3x3卷积

        ch_out_list = [64, 128, 256, 512]
        # 定义4个stage的输出通道数（中间层通道数）
        block = BottleNeck if depth >= 50 else BasicBlock
        # 根据深度选择残差块类型：深度>=50使用BottleNeck，否则使用BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        # 计算每个stage的实际输出通道数（中间层通道数*expansion）
        _out_strides = [4, 8, 16, 32]
        # 定义每个stage的输出步长（相对于输入图像的下采样倍数）

        self.res_layers = nn.ModuleList()
        # 创建ModuleList，用于存储4个stage
        for i in range(num_stages):
            # 循环创建num_stages个stage
            stage_num = i + 2
            # 计算stage编号（2, 3, 4, 5），用于判断是否需要下采样
            self.res_layers.append(
                # 将新创建的stage添加到ModuleList
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
                # 创建一个Blocks对象，包含block_nums[i]个残差块
            )
            ch_in = _out_channels[i]
            # 更新输入通道数为当前stage的输出通道数，用于下一个stage

        self.return_idx = return_idx
        # 保存需要返回的stage索引列表
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        # 根据return_idx计算实际输出的通道数列表
        self.out_strides = [_out_strides[_i] for _i in return_idx]
        # 根据return_idx计算实际输出的步长列表

        if freeze_at >= 0:
            # 如果freeze_at>=0，需要冻结部分参数
            self._freeze_parameters(self.conv1)
            # 冻结stem（conv1）的参数
            for i in range(min(freeze_at, num_stages)):
                # 循环冻结前freeze_at个stage
                self._freeze_parameters(self.res_layers[i])
                # 冻结第i个stage的参数

        if freeze_norm:
            # 如果需要冻结批归一化层
            self._freeze_norm(self)
            # 递归冻结所有批归一化层，转换为FrozenBatchNorm2d

        if pretrained:
            if pretrained_source == 'torchvision':
                self._load_torchvision_pretrained(depth)
            else:
                state = torch.hub.load_state_dict_from_url(donwload_url[depth])
                self.load_state_dict(state)
                print(f'Load PResNet{depth} state_dict from PaddlePaddle')

    @staticmethod
    def _build_torchvision_key_map(depth):
        key_map = {}
        key_map['conv1.weight'] = 'conv1.conv1_1.conv.weight'
        key_map['bn1.weight'] = 'conv1.conv1_1.norm.weight'
        key_map['bn1.bias'] = 'conv1.conv1_1.norm.bias'
        key_map['bn1.running_mean'] = 'conv1.conv1_1.norm.running_mean'
        key_map['bn1.running_var'] = 'conv1.conv1_1.norm.running_var'
        key_map['bn1.num_batches_tracked'] = 'conv1.conv1_1.norm.num_batches_tracked'

        for layer_idx in range(4):
            tv_prefix = f'layer{layer_idx + 1}'
            our_prefix = f'res_layers.{layer_idx}'
            num_blocks = ResNet_cfg[depth][layer_idx]
            for block_idx in range(num_blocks):
                tv_block = f'{tv_prefix}.{block_idx}'
                our_block = f'{our_prefix}.{block_idx}'
                key_map[f'{tv_block}.conv1.weight'] = f'{our_block}.branch2a.conv.weight'
                key_map[f'{tv_block}.bn1.weight'] = f'{our_block}.branch2a.norm.weight'
                key_map[f'{tv_block}.bn1.bias'] = f'{our_block}.branch2a.norm.bias'
                key_map[f'{tv_block}.bn1.running_mean'] = f'{our_block}.branch2a.norm.running_mean'
                key_map[f'{tv_block}.bn1.running_var'] = f'{our_block}.branch2a.norm.running_var'
                key_map[f'{tv_block}.bn1.num_batches_tracked'] = f'{our_block}.branch2a.norm.num_batches_tracked'
                key_map[f'{tv_block}.conv2.weight'] = f'{our_block}.branch2b.conv.weight'
                key_map[f'{tv_block}.bn2.weight'] = f'{our_block}.branch2b.norm.weight'
                key_map[f'{tv_block}.bn2.bias'] = f'{our_block}.branch2b.norm.bias'
                key_map[f'{tv_block}.bn2.running_mean'] = f'{our_block}.branch2b.norm.running_mean'
                key_map[f'{tv_block}.bn2.running_var'] = f'{our_block}.branch2b.norm.running_var'
                key_map[f'{tv_block}.bn2.num_batches_tracked'] = f'{our_block}.branch2b.norm.num_batches_tracked'
                if depth >= 50:
                    key_map[f'{tv_block}.conv3.weight'] = f'{our_block}.branch2c.conv.weight'
                    key_map[f'{tv_block}.bn3.weight'] = f'{our_block}.branch2c.norm.weight'
                    key_map[f'{tv_block}.bn3.bias'] = f'{our_block}.branch2c.norm.bias'
                    key_map[f'{tv_block}.bn3.running_mean'] = f'{our_block}.branch2c.norm.running_mean'
                    key_map[f'{tv_block}.bn3.running_var'] = f'{our_block}.branch2c.norm.running_var'
                    key_map[f'{tv_block}.bn3.num_batches_tracked'] = f'{our_block}.branch2c.norm.num_batches_tracked'
                key_map[f'{tv_block}.downsample.0.weight'] = f'{our_block}.short.conv.weight'
                key_map[f'{tv_block}.downsample.1.weight'] = f'{our_block}.short.norm.weight'
                key_map[f'{tv_block}.downsample.1.bias'] = f'{our_block}.short.norm.bias'
                key_map[f'{tv_block}.downsample.1.running_mean'] = f'{our_block}.short.norm.running_mean'
                key_map[f'{tv_block}.downsample.1.running_var'] = f'{our_block}.short.norm.running_var'
                key_map[f'{tv_block}.downsample.1.num_batches_tracked'] = f'{our_block}.short.norm.num_batches_tracked'
        return key_map

    def _load_torchvision_pretrained(self, depth):
        import torchvision.models as tv_models
        model_fn = getattr(tv_models, torchvision_model_fn[depth])
        tv_model = model_fn(weights='DEFAULT')
        tv_state = tv_model.state_dict()
        key_map = self._build_torchvision_key_map(depth)
        our_state = {}
        for tv_key, our_key in key_map.items():
            if tv_key in tv_state:
                our_state[our_key] = tv_state[tv_key]
        missing, unexpected = self.load_state_dict(our_state, strict=False)
        if unexpected:
            print(f'Warning: unexpected keys when loading torchvision weights: {unexpected}')
        if missing:
            print(f'Note: missing keys when loading torchvision weights (expected for shortcut=True blocks): {len(missing)} keys')
        print(f'Load PResNet{depth} state_dict from torchvision (ImageNet)')
            
    # =========================================
    # 函数作用：冻结模块的所有参数，使其不参与梯度更新
    # 输入参数：
    # - m: 需要冻结的模块
    # 返回值：无
    # 核心逻辑：
    # 1. 遍历模块的所有参数
    # 2. 将参数的requires_grad设置为False
    # 调用关系：
    # - 被__init__调用（冻结stem和stage参数）
    # 重要性：辅助函数
    # =========================================
    def _freeze_parameters(self, m: nn.Module):
        # 冻结参数方法，接收一个PyTorch模块
        for p in m.parameters():
            # 遍历模块的所有参数
            p.requires_grad = False
            # 将参数的requires_grad设置为False，使其不参与梯度更新

    # =========================================
    # 函数作用：递归冻结模块中的所有批归一化层，转换为FrozenBatchNorm2d
    # 输入参数：
    # - m: 需要处理的模块
    # 返回值：
    # - m: 处理后的模块（如果包含BatchNorm2d，则替换为FrozenBatchNorm2d）
    # 核心逻辑：
    # 1. 如果是BatchNorm2d，替换为FrozenBatchNorm2d
    # 2. 否则递归处理所有子模块
    # 调用关系：
    # - 被__init__调用（冻结所有批归一化层）
    # 重要性：辅助函数
    # =========================================
    def _freeze_norm(self, m: nn.Module):
        # 冻结批归一化层方法，接收一个PyTorch模块
        if isinstance(m, nn.BatchNorm2d):
            # 如果当前模块是BatchNorm2d
            m = FrozenBatchNorm2d(m.num_features)
            # 将其替换为FrozenBatchNorm2d（冻结的批归一化层，不更新统计量）
        else:
            # 如果不是BatchNorm2d
            for name, child in m.named_children():
                # 遍历所有子模块
                _child = self._freeze_norm(child)
                # 递归处理子模块
                if _child is not child:
                    # 如果子模块被替换了
                    setattr(m, name, _child)
                    # 更新父模块中的子模块引用
        return m
        # 返回处理后的模块

    # =========================================
    # 函数作用：执行PResNet的前向传播，提取多尺度特征图
    # 输入参数：
    # - x: 输入图像张量，形状为[B, 3, H, W]
    # 返回值：
    # - outs: 多尺度特征图列表，每个元素是一个stage的输出
    # 核心逻辑：
    # 1. 通过stem（初始卷积层）提取初步特征
    # 2. 通过最大池化下采样
    # 3. 依次通过4个stage，提取多尺度特征
    # 4. 根据return_idx返回指定的stage输出
    # 调用关系：
    # - 被RTDETR类的forward调用（作为backbone）
    # 重要性：核心函数
    # =========================================
    def forward(self, x):
        # 前向传播方法，接收输入图像张量x
        conv1 = self.conv1(x)
        # 将输入x传入stem（初始卷积层），提取初步特征
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        # 对stem的输出进行3x3最大池化，步长2，padding=1，进一步下采样
        outs = []
        # 创建空列表，用于存储需要返回的特征图
        for idx, stage in enumerate(self.res_layers):
            # 遍历所有stage，同时获取索引和stage对象
            x = stage(x)
            # 将特征传入当前stage，提取特征
            if idx in self.return_idx:
                # 如果当前stage的索引在return_idx中
                outs.append(x)
                # 将当前stage的输出添加到返回列表
        return outs
        # 返回多尺度特征图列表