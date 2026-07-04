# =========================================
# 文件说明：
# - 该文件的作用：定义RT-DETR（Real-Time DEtection TRansformer）目标检测模型的主类
# - 在项目中的位置：模型定义 / 核心架构
# - 与其他文件的关系：
#   - 依赖：backbone/*.py（骨干网络）、encoder/*.py（编码器）、decoder/*.py（解码器）
#   - 被：train.py（训练脚本）、infer.py（推理脚本）调用
#   - 配置：configs/*.yaml（配置文件指定具体组件实现）
# =========================================

"""by lyuwenyu
"""
# 文件作者署名注释，说明代码由lyuwenyu编写

import torch 
# 导入PyTorch深度学习框架核心库，提供张量计算和自动求导功能

import torch.nn as nn 
# 导入PyTorch神经网络模块，包含各种网络层（卷积、线性、池化等）

import torch.nn.functional as F 
# 导入PyTorch函数式接口，提供激活函数、损失函数、池化等函数式操作

import random 
# 导入Python标准库random模块，用于生成随机数（虽然代码中未直接使用，但可能用于其他场景）

import numpy as np 
# 导入NumPy数值计算库，用于数组操作和数学计算（用于多尺度训练的随机选择）

from src.core import register
# 从src.core模块导入register装饰器，用于将模型类注册到框架中，实现模块的自动发现和管理


__all__ = ['RTDETR', ]
# 定义模块的公共接口，指定哪些类可以被"from module import *"导入，这里只导出RTDETR类


@register
# 使用register装饰器装饰RTDETR类，将其注册到框架的模型库中，便于配置文件动态加载
class RTDETR(nn.Module):
    # =========================================
    # 类的整体作用：RT-DETR（Real-Time DEtection TRansformer）目标检测模型的主类
    # 核心方法说明：
    # - __init__: 初始化模型，组装backbone、encoder、decoder三个核心组件
    # - forward: 执行前向传播，将输入图像转换为检测结果
    # - deploy: 将模型转换为部署优化模式，提高推理速度
    # 是否为关键模块：✅ 是（这是整个RT-DETR模型的核心定义类）
    # =========================================
    # 定义RTDETR类，继承自nn.Module基类（所有PyTorch神经网络模块必须继承的基类）
    __inject__ = ['backbone', 'encoder', 'decoder', ]
    # 类变量，声明需要依赖注入的组件列表，框架会自动从配置中加载并注入这些模块

    # =========================================
    # 函数作用：初始化RT-DETR模型，组装backbone、encoder、decoder三个核心组件
    # 输入参数：
    # - backbone: nn.Module，骨干网络，负责从输入图像中提取多尺度特征
    # - encoder: 编码器模块，通常是基于Transformer的编码器，用于特征融合
    # - decoder: 解码器模块，基于Transformer的解码器，用于生成检测结果
    # - multi_scale=None: 可选参数，多尺度训练的尺寸列表，如[480, 512, 544, 576, 608, 640]
    # 返回值：无（初始化方法不返回值）
    # 核心逻辑：
    # 1. 调用父类nn.Module的初始化方法，完成PyTorch模块的基本设置
    # 2. 保存backbone实例，用于特征提取
    # 3. 保存encoder实例，用于特征编码和融合
    # 4. 保存decoder实例，用于检测生成
    # 5. 保存multi_scale参数，用于训练时的多尺度数据增强
    # 重要性：核心函数
    # =========================================
    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        # 初始化方法，接收backbone（骨干网络）、encoder（编码器）、decoder（解码器）三个必需组件，multi_scale为可选的多尺度训练参数
        super().__init__()
        # 调用父类nn.Module的__init__方法，完成PyTorch模块的必要初始化（如参数注册等）
        self.backbone = backbone
        # 保存backbone实例到实例变量，backbone用于从输入图像中提取多尺度特征图
        self.decoder = decoder
        # 保存decoder实例到实例变量，decoder负责将编码后的特征转换为最终的检测结果
        self.encoder = encoder
        # 保存encoder实例到实例变量，encoder用于对backbone提取的特征进行Transformer编码和融合
        self.multi_scale = multi_scale
        # 保存多尺度训练参数列表，训练时会从中随机选择尺寸进行数据增强，提高模型对不同尺寸目标的鲁棒性
        
    # =========================================
    # 函数作用：执行模型的前向传播，将输入图像转换为检测结果
    # 输入参数：
    # - x: 输入图像张量，形状为[batch_size, channels, height, width]，例如[8, 3, 640, 640]
    # - targets=None: 可选的目标标注信息，训练时用于计算loss，包含边界框、类别等
    # 返回值：模型的检测结果，通常包含边界框坐标、类别概率、目标置信度等
    # 核心逻辑：
    # 1. 多尺度数据增强（仅训练时）：从多尺度列表中随机选择尺寸，调整输入图像大小
    # 2. 特征提取：将输入传入backbone，提取多尺度特征图
    # 3. 特征编码：将特征传入encoder，使用Transformer自注意力机制融合特征
    # 4. 检测生成：将编码后的特征传入decoder，使用交叉注意力生成检测结果
    # 5. 返回结果：输出最终的检测结果
    # 调用关系：
    # - 被train.py调用（训练时）
    # - 被infer.py调用（推理时）
    # 重要性：核心函数
    # =========================================
    def forward(self, x, targets=None):
        # 前向传播方法，定义模型如何将输入转换为输出，x是输入图像张量[B,C,H,W]，targets是训练时的标注信息（可选）
        if self.multi_scale and self.training:
            # 判断条件：如果设置了多尺度训练参数且当前处于训练模式（self.training为True）
            sz = np.random.choice(self.multi_scale)
            # 从多尺度列表中随机选择一个尺寸，用于数据增强，这样可以让模型适应不同尺度的输入
            x = F.interpolate(x, size=[sz, sz])
            # 使用双线性插值将输入图像调整为随机选择的尺寸，F.interpolate是PyTorch的插值函数，mode默认为'nearest'但通常用'bilinear'
            
        x = self.backbone(x)
        # 将输入图像传入backbone骨干网络，backbone会提取多尺度特征图（通常是C3、C4、C5三个尺度的特征）
        x = self.encoder(x)        
        # 将backbone提取的多尺度特征传入encoder编码器，encoder使用Transformer的自注意力机制进行特征融合和增强
        x = self.decoder(x, targets)
        # 将编码后的特征传入decoder解码器，decoder使用Transformer的交叉注意力机制生成检测结果，训练时需要targets计算loss

        return x
        # 返回模型的输出结果，包含边界框坐标、类别概率、目标置信度等检测信息
    
    # =========================================
    # 函数作用：将模型转换为部署优化模式，提高推理速度
    # 输入参数：无
    # 返回值：转换后的模型实例（self），支持链式调用
    # 核心逻辑：
    # 1. 将模型设置为评估模式，关闭dropout、batchnorm的训练行为
    # 2. 遍历模型的所有子模块
    # 3. 对每个模块检查是否有convert_to_deploy方法
    # 4. 如果有，调用该方法进行部署优化（如融合Conv+BN层）
    # 5. 返回优化后的模型实例
    # 调用关系：
    # - 被infer.py调用（推理前）
    # 重要性：辅助函数（推理优化）
    # =========================================
    def deploy(self, ):
        # 部署方法，用于将模型转换为推理优化模式，提高推理速度和减少内存占用
        self.eval()
        # 将模型设置为评估模式，这会关闭dropout、batchnorm的训练模式行为，确保推理时行为一致
        for m in self.modules():
            # 遍历模型的所有子模块（包括backbone、encoder、decoder及其内部的所有层）
            if hasattr(m, 'convert_to_deploy'):
                # 检查当前模块是否有convert_to_deploy方法（通常用于融合卷积层和BN层，减少计算量）
                m.convert_to_deploy()
                # 调用该模块的部署转换方法，将训练时的结构转换为推理优化的结构（如Conv+BN融合为单个Conv）
        return self 
        # 返回转换后的模型实例，支持链式调用（model.deploy()返回model本身）