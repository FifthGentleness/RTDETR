"""by lyuwenyu
"""  # 模块文档字符串，说明代码作者

import math  # 导入Python标准库math模块，用于数学计算（如对数、指数等）
import torch  # 导入PyTorch深度学习框架，提供张量计算和自动求导功能
import torch.nn as nn  # 导入PyTorch神经网络模块，包含各种层和模型组件
import torch.nn.functional as F  # 导入PyTorch函数式接口，包含各种神经网络函数（如激活函数、卷积等）


def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:  # 定义反向sigmoid函数，接收张量x和极小值eps，返回反向sigmoid结果
    x = x.clip(min=0., max=1.)  # 将输入张量x的值裁剪到[0., 1.]范围内，确保数值在有效区间内，维度不变：[B, ...] -> [B, ...]
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))  # 计算log(x/(1-x))，即反向sigmoid，使用clip避免除零和log(0)，维度不变：[B, ...] -> [B, ...]


# =========================================================================
# 函数名: deformable_attention_core_func (可变形注意力核心函数)
# 类型: 函数
# 代码逻辑链条中的具体职责: 实现多尺度可变形注意力的核心计算，根据采样位置和注意力权重从多尺度特征图中采样特征并聚合
# =========================================================================
def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):  # 定义可变形注意力核心函数，接收特征值、空间形状、采样位置和注意力权重
    """  # 函数文档字符串开始
    Args:  # 参数说明开始
        value (Tensor): [bs, value_length, n_head, head_dim]  # 输入特征张量，维度：[batch_size, 特征长度, 注意力头数, head_dim]
        value_spatial_shapes (Tensor|List): [n_levels, 2]  # 各尺度特征图的空间形状，维度：[尺度数, 2]，每个元素是[高度, 宽度]
        value_level_start_index (Tensor|List): [n_levels]  # 各尺度特征图的起始索引（未在函数中使用）
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]  # 采样位置坐标，维度：[batch_size, 查询长度, 注意力头数, 尺度数, 采样点数, 2(坐标xy)]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]  # 采样点的注意力权重，维度：[batch_size, 查询长度, 注意力头数, 尺度数, 采样点数]

    Returns:  # 返回值说明开始
        output (Tensor): [bs, Length_{query}, n_head*head_dim]  # 输出特征张量，维度：[batch_size, 查询长度, 注意力头数*head_dim]
    """  # 函数文档字符串结束
    bs, _, n_head, c = value.shape  # 解包value张量的形状，获取batch_size、注意力头数和通道数，维度：[bs, value_length, n_head, c] -> bs, _, n_head, c
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape  # 解包采样位置张量的形状，获取查询长度、尺度数和采样点数，维度：[bs, Len_q, n_head, n_levels, n_points, 2] -> _, Len_q, _, n_levels, n_points, _

    split_shape = [h * w for h, w in value_spatial_shapes]  # 计算每个尺度特征图的像素数列表，用于分割value张量，例如：[[32,32],[16,16]] -> [1024, 256]
    value_list = value.split(split_shape, dim=1)  # 按照各尺度像素数沿维度1（特征长度维度）分割value张量，得到各尺度的特征列表，维度：[bs, value_length, n_head, c] -> [[bs, 1024, n_head, c], [bs, 256, n_head, c], ...]
    sampling_grids = 2 * sampling_locations - 1  # 将采样位置从[0,1]范围映射到[-1,1]范围，适配grid_sample函数的输入要求，维度：[bs, Len_q, n_head, n_levels, n_points, 2] -> [bs, Len_q, n_head, n_levels, n_points, 2]
    sampling_value_list = []  # 初始化空列表，用于存储各尺度的采样值
    for level, (h, w) in enumerate(value_spatial_shapes):  # 遍历每个尺度的特征图，获取尺度索引和对应的高度宽度
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(  # 将当前尺度特征重塑为4D张量，适配grid_sample输入，维度：[bs, h*w, n_head, c] -> [bs, h*w, n_head*c] -> [bs, n_head*c, h*w] -> [bs*n_head, c, h, w]
            0, 2, 1).reshape(bs * n_head, c, h, w)  # 继续reshape操作，将batch和head维度合并，维度：[bs, n_head*c, h*w] -> [bs*n_head, c, h, w]
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(  # 提取当前尺度的采样网格并调整维度顺序，维度：[bs, Len_q, n_head, n_points, 2] -> [bs, n_head, Len_q, n_points, 2]
            0, 2, 1, 3, 4).flatten(0, 1)  # 合并batch和head维度，维度：[bs, n_head, Len_q, n_points, 2] -> [bs*n_head, Len_q, n_points, 2]
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(  # 使用双线性插值从特征图中采样指定位置的值，维度：[bs*n_head, c, h, w], [bs*n_head, Len_q, n_points, 2] -> [bs*n_head, c, Len_q, n_points]
            value_l_,  # 输入特征图，维度：[bs*n_head, c, h, w]
            sampling_grid_l_,  # 采样网格坐标，维度：[bs*n_head, Len_q, n_points, 2]
            mode='bilinear',  # 使用双线性插值模式
            padding_mode='zeros',  # 超出边界的采样点填充零值
            align_corners=False)  # 不对齐角点，使用更精确的插值方式
        sampling_value_list.append(sampling_value_l_)  # 将当前尺度的采样值添加到列表中，维度：[bs*n_head, c, Len_q, n_points] -> list
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(  # 调整注意力权重的维度顺序并重塑，维度：[bs, Len_q, n_head, n_levels, n_points] -> [bs, n_head, Len_q, n_levels, n_points] -> [bs*n_head, 1, Len_q, n_levels*n_points]
        bs * n_head, 1, Len_q, n_levels * n_points)  # 继续reshape操作，合并batch和head维度，展平尺度和采样点维度，维度：[bs, n_head, Len_q, n_levels, n_points] -> [bs*n_head, 1, Len_q, n_levels*n_points]
    output = (torch.stack(  # 将各尺度的采样值堆叠并加权求和，维度：list of [bs*n_head, c, Len_q, n_points] -> [bs*n_head, c, Len_q, n_levels*n_points]
        sampling_value_list, dim=-2).flatten(-2) *  # 沿倒数第二维度堆叠并展平最后两维度，维度：[bs*n_head, c, Len_q, n_levels, n_points] -> [bs*n_head, c, Len_q, n_levels*n_points]
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)  # 与注意力权重相乘并沿采样点维度求和，最后重塑维度，维度：[bs*n_head, c, Len_q, n_levels*n_points] * [bs*n_head, 1, Len_q, n_levels*n_points] -> [bs*n_head, c, Len_q] -> [bs, n_head*c, Len_q]

    return output.permute(0, 2, 1)  # 调整输出维度顺序，使查询长度维度在前，维度：[bs, n_head*c, Len_q] -> [bs, Len_q, n_head*c]


import math  # 导入math模块（重复导入，用于bias初始化计算）
def bias_init_with_prob(prior_prob=0.01):  # 定义根据概率初始化偏置值的函数，接收先验概率参数，返回初始化的偏置值
    """initialize conv/fc bias value according to a given probability value."""  # 函数文档字符串，说明功能：根据给定概率值初始化卷积或全连接层的偏置
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))  # 计算偏置初始化值，使用反向sigmoid公式，使初始输出概率等于prior_prob
    return bias_init  # 返回计算得到的偏置初始化值



# =========================================================================
# 函数名: get_activation (获取激活函数)
# 类型: 函数
# 代码逻辑链条中的具体职责: 根据字符串名称或nn.Module对象获取对应的激活函数，支持多种常用激活函数
# =========================================================================
def get_activation(act: str, inpace: bool=True):  # 定义获取激活函数的函数，接收激活函数名称和是否原地操作标志，返回对应的激活函数模块
    '''get activation  # 函数文档字符串，说明功能：获取激活函数
    '''  # 文档字符串结束
    act = act.lower()  # 将激活函数名称转换为小写，实现大小写不敏感的匹配，例如：'ReLU' -> 'relu'
    
    if act == 'silu':  # 判断激活函数名称是否为'silu'（Swish激活函数）
        m = nn.SiLU()  # 创建SiLU激活函数模块，SiLU(x) = x * sigmoid(x)

    elif act == 'relu':  # 判断激活函数名称是否为'relu'（修正线性单元）
        m = nn.ReLU()  # 创建ReLU激活函数模块，ReLU(x) = max(0, x)

    elif act == 'leaky_relu':  # 判断激活函数名称是否为'leaky_relu'（带泄露的ReLU）
        m = nn.LeakyReLU()  # 创建LeakyReLU激活函数模块，允许小的负梯度通过

    elif act == 'silu':  # 重复判断'silu'（代码冗余，可能是复制粘贴错误）
        m = nn.SiLU()  # 再次创建SiLU激活函数模块（冗余代码）
    
    elif act == 'gelu':  # 判断激活函数名称是否为'gelu'（高斯误差线性单元）
        m = nn.GELU()  # 创建GELU激活函数模块，GELU在Transformer中常用
        
    elif act is None:  # 判断激活函数名称是否为None
        m = nn.Identity()  # 创建恒等映射模块，不进行任何变换
    
    elif isinstance(act, nn.Module):  # 判断act是否已经是nn.Module的实例
        m = act  # 直接使用传入的激活函数模块

    else:  # 以上条件都不满足时
        raise RuntimeError('')  # 抛出运行时错误（错误信息为空字符串，应该提供更有意义的错误信息）

    if hasattr(m, 'inplace'):  # 检查激活函数模块是否有inplace属性
        m.inplace = inpace  # 设置inplace属性，控制是否进行原地操作以节省内存
    
    return m  # 返回创建或配置好的激活函数模块

# =========================================================================
# 整体功能总结：
# 本文件实现了RT-DETR（Real-Time DEtection TRansformer）模型的核心工具函数集，主要包括：
# 1. inverse_sigmoid：实现反向sigmoid变换，用于将概率值转换为logits，常用于目标检测中的坐标编码
# 2. deformable_attention_core_func：实现多尺度可变形注意力的核心计算，这是RT-DETR的核心创新点，支持在多尺度特征图上进行灵活的特征采样和聚合
# 3. bias_init_with_prob：根据先验概率初始化偏置值，确保模型初始输出符合预期的概率分布
# 4. get_activation：统一的激活函数获取接口，支持多种常用激活函数，提供灵活的激活函数配置
# 这些工具函数为RT-DETR模型的基础架构提供了关键支持，特别是在多尺度特征融合和注意力机制方面。
# =========================================================================