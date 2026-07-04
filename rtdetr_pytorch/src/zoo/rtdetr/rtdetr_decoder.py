
"""by lyuwenyu
RT-DETR Transformer解码器实现
核心功能：实现RT-DETR的目标检测Transformer解码器，包含Encoder和Decoder两部分
- Encoder: 对Backbone特征进行编码
- Decoder: 使用可变形注意力机制进行目标检测
"""

# 导入Python标准库
import math  # 数学运算库，提供数学常量和函数（如pi）
import copy  # 深拷贝工具，用于复制对象
from collections import OrderedDict  # 有序字典，保持字典的插入顺序

# 导入PyTorch相关库
import torch  # PyTorch主库，提供张量运算和神经网络功能
import torch.nn as nn  # 神经网络模块，提供各种层结构
import torch.nn.functional as F  # 函数式接口，提供各种操作函数
import torch.nn.init as init  # 参数初始化工具

# 导入项目内部模块
from .denoising import get_contrastive_denoising_training_group  # 对比去噪训练组生成函数
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid  # 工具函数
from .utils import bias_init_with_prob  # 偏置初始化函数

# 导入注册器，用于模型注册
from src.core import register

# 定义模块导出列表
__all__ = ['RTDETRTransformer']  # 只导出RTDETRTransformer类


# =========================================================================
# 类名: MLP (多层感知机)
# 类型: nn.Module 子类
# 代码逻辑链条中的具体职责: 作为基础的前馈神经网络模块，用于 bbox 和 score 的预测。
# 在整个解码器中，MLP 被多次实例化用于生成边界框坐标和类别分数的预测头
# =========================================================================
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        # 初始化MLP的层结构
        # input_dim: 输入特征维度 [B, input_dim]
        # hidden_dim: 隐藏层维度
        # output_dim: 输出特征维度，通常为4（bbox坐标）或类别数
        # num_layers: 网络层数
        # act: 激活函数类型
        super().__init__()  # 调用父类初始化方法
        self.num_layers = num_layers  # 保存层数到实例属性
        h = [hidden_dim] * (num_layers - 1)  # 创建隐藏层维度列表 [hidden_dim, hidden_dim, ...]
        # 创建线性层列表，使用zip配对输入输出维度
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )  # layers: [Linear(input_dim, hidden_dim), Linear(hidden_dim, hidden_dim), ..., Linear(hidden_dim, output_dim)]
        self.act = nn.Identity() if act is None else get_activation(act)  # 根据act参数选择激活函数

    def forward(self, x):
        # 前向传播函数
        # x: 输入张量 [B, seq_len, input_dim]
        for i, layer in enumerate(self.layers):  # 遍历每一层
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
            # 隐藏层：线性层 + 激活函数
            # 输出层：仅线性层（不经过激活）
        return x  # 返回输出张量 [B, seq_len, output_dim]


# =========================================================================
# 类名: MSDeformableAttention (多尺度可变形注意力)
# 类型: nn.Module 子类
# 代码逻辑链条中的具体职责: 实现多尺度可变形注意力机制，是RT-DETR的核心创新点。
# 该模块能够在多个特征尺度上对参考点周围的特征进行采样和聚合，
# 大幅减少计算量的同时保持对任意形状目标的建模能力。
# =========================================================================
class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        # 初始化多尺度可变形注意力模块
        # embed_dim: 嵌入维度（查询/值的特征维度）
        # num_heads: 注意力头数
        # num_levels: 特征金字塔的层数（多尺度）
        # num_points: 每个参考点采样的点数
        super(MSDeformableAttention, self).__init__()  # 调用父类初始化方法
        self.embed_dim = embed_dim  # 保存嵌入维度 [B, query_len, embed_dim]
        self.num_heads = num_heads  # 保存注意力头数
        self.num_levels = num_levels  # 保存特征层数
        self.num_points = num_points  # 保存每层采样点数
        self.total_points = num_heads * num_levels * num_points  # 计算总采样点数

        self.head_dim = embed_dim // num_heads  # 计算每个头的维度
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"  # 断言维度整除

        # 定义可变形注意力的核心网络层
        # sampling_offsets: 生成采样偏移量，输出维度为 total_points * 2（x,y偏移）
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)  # [B, query_len, total_points*2]
        # attention_weights: 生成注意力权重
        self.attention_weights = nn.Linear(embed_dim, self.total_points)  # [B, query_len, total_points]
        # value_proj: 将value投影到多头空间
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # [B, value_len, embed_dim]
        # output_proj: 输出投影层
        self.output_proj = nn.Linear(embed_dim, embed_dim)  # [B, query_len, embed_dim]

        # 绑定可变形注意力的核心计算函数
        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()  # 调用参数初始化方法

    def _reset_parameters(self):
        # 初始化网络参数
        # 初始化sampling_offsets的权重为0
        init.constant_(self.sampling_offsets.weight, 0)  # sampling_offsets.weight: [total_points*2, embed_dim] → 全0
        # 计算注意力头的角度初始化值
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        # 例如 num_heads=8: thetas = [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
        # thetas: [num_heads]，表示每个头对应的角度
        # 创建网格初始化的基础向量
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)  # [num_heads, 2]
        # 例如: [[1.0, 0.0], [0.707, 0.707], [0.0, 1.0], ...]
        # 对初始向量进行归一化，使其绝对值最大为1
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values  # [num_heads, 2]
        # 扩展初始化向量到所有层和所有采样点
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        # [num_heads, num_levels, num_points, 2]
        # 计算每层的缩放因子
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        # scaling: [1, 1, num_points, 1]，值从1到num_points递增
        grid_init *= scaling  # 应用缩放，使不同采样点有不同的初始偏移幅度
        # 将初始化值赋值给sampling_offsets的偏置
        self.sampling_offsets.bias.data[...] = grid_init.flatten()  # [total_points*2]

        # 初始化attention_weights的权重和偏置为0
        init.constant_(self.attention_weights.weight, 0)  # attention_weights.weight: [total_points, embed_dim]
        init.constant_(self.attention_weights.bias, 0)  # attention_weights.bias: [total_points]

        # 使用Xavier均匀初始化value_proj的权重
        init.xavier_uniform_(self.value_proj.weight)  # value_proj.weight: [embed_dim, embed_dim]
        init.constant_(self.value_proj.bias, 0)  # value_proj.bias: [embed_dim]
        # 使用Xavier均匀初始化output_proj的权重
        init.xavier_uniform_(self.output_proj.weight)  # output_proj.weight: [embed_dim, embed_dim]
        init.constant_(self.output_proj.bias, 0)  # output_proj.bias: [embed_dim]

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        # 多尺度可变形注意力的前向传播
        # query: 查询张量 [bs, query_length, embed_dim]
        # reference_points: 参考点坐标 [bs, query_length, n_levels, 2]，值域[0,1]
        # value: 值张量 [bs, value_length, embed_dim]
        # value_spatial_shapes: 各特征层的空间形状列表 [(H_0, W_0), (H_1, W_1), ...]
        # value_mask: 有效区域的掩码 [bs, value_length]，True表示有效元素

        # 获取batch大小和查询长度
        bs, Len_q = query.shape[:2]  # bs: batch_size, Len_q: query序列长度
        Len_v = value.shape[1]  # Len_v: value序列长度

        # 对value进行投影
        value = self.value_proj(value)  # [bs, Len_v, embed_dim]
        # 应用有效掩码（如果有）
        if value_mask is not None:
            # 将掩码转换为value的数据类型，并扩展维度
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)  # [bs, Len_v, 1]
            value *= value_mask  # 将无效位置置0
        # 将value重塑为多头形式
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)
        # [bs, Len_v, num_heads, head_dim]

        # 生成采样偏移量
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        # [bs, Len_q, num_heads, num_levels, num_points, 2]

        # 生成注意力权重
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        # [bs, Len_q, num_heads, total_points]
        # 对注意力权重进行softmax归一化
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)
        # [bs, Len_q, num_heads, num_levels, num_points]

        # 根据参考点的维度计算采样位置
        if reference_points.shape[-1] == 2:
            # 2D坐标归一化参考点
            offset_normalizer = torch.tensor(value_spatial_shapes)  # [n_levels, 2]
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)  # [1, 1, 1, n_levels, 1, 2]，翻转[1,2]变为[W,H]
            # 计算最终采样位置：参考点 + 归一化偏移
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer  
            # [bs, Len_q, num_heads, num_levels, num_points, 2]
        elif reference_points.shape[-1] == 4:
            # bbox形式的参考点（x1, y1, x2, y2）
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
            # [bs, Len_q, 1, n_levels, n_points, 2]
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        # 调用核心可变形注意力计算函数
        output = self.ms_deformable_attn_core(
            value, value_spatial_shapes, sampling_locations, attention_weights)
        # value: [bs, Len_v, num_heads, head_dim]
        # value_spatial_shapes: 各层空间形状
        # sampling_locations: [bs, Len_q, num_heads, num_levels, num_points, 2]
        # attention_weights: [bs, Len_q, num_heads, num_levels, num_points]
        # output: [bs, Len_q, num_heads*head_dim = embed_dim]

        # 输出投影
        output = self.output_proj(output)  # [bs, Len_q, embed_dim]

        return output  # 返回注意力输出 [bs, query_length, embed_dim]


# =========================================================================
# 类名: TransformerDecoderLayer (Transformer解码器层)
# 类型: nn.Module 子类
# 代码逻辑链条中的具体职责: 实现Transformer解码器的基本结构，包含三个子层：
# 1. 自注意力层（Self-Attention）：查询自己的历史输出
# 2. 交叉注意力层（Cross-Attention）：从Encoder记忆中查询目标特征
# 3. 前馈网络（FFN）：进一步特征变换
# 每一层都包含残差连接和层归一化
# =========================================================================
class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,):
        # 初始化解码器层
        # d_model: 模型维度（特征维度）
        # n_head: 注意力头数
        # dim_feedforward: 前馈网络的隐藏层维度
        # dropout: Dropout比例
        # activation: 激活函数类型
        # n_levels: 特征金字塔层数
        # n_points: 每层采样点数
        super(TransformerDecoderLayer, self).__init__()  # 调用父类初始化方法

        # 自注意力层（Self-Attention）
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        # MultiheadAttention: 多头自注意力层，batch_first=True表示输入格式为[B, seq, dim]
        self.dropout1 = nn.Dropout(dropout)  # 自注意力的Dropout层
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后的层归一化

        # 交叉注意力层（Cross-Attention，可变形注意力）
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        # 可变形注意力，从Encoder记忆中查询
        self.dropout2 = nn.Dropout(dropout)  # 交叉注意力的Dropout层
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后的层归一化

        # 前馈网络（FFN）
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 第一个线性层 [B, seq, d_model] → [B, seq, dim_feedforward]
        self.activation = getattr(F, activation)  # 获取激活函数
        self.dropout3 = nn.Dropout(dropout)  # FFN第一层后的Dropout
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 第二个线性层 [B, seq, dim_feedforward] → [B, seq, d_model]
        self.dropout4 = nn.Dropout(dropout)  # FFN第二层后的Dropout
        self.norm3 = nn.LayerNorm(d_model)  # FFN后的层归一化

    def with_pos_embed(self, tensor, pos):
        # 将位置编码添加到张量
        # tensor: 原始张量 [B, seq, dim]
        # pos: 位置编码 [B, seq, dim] 或 None
        return tensor if pos is None else tensor + pos  # 无位置编码则返回原张量，否则相加

    def forward_ffn(self, tgt):
        # 前馈网络的前向传播
        # tgt: 输入张量 [B, seq, d_model]
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        # linear1: [B, seq, dim_feedforward]
        # activation: 非线性激活
        # dropout3: 随机失活
        # linear2: [B, seq, d_model]

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # 解码器层的前向传播
        # tgt: 目标序列（解码器输入）[B, tgt_len, d_model]
        # reference_points: 参考点坐标 [B, tgt_len, n_levels, 2] 或 [B, tgt_len, 4]
        # memory: Encoder输出 [B, memory_len, d_model]
        # memory_spatial_shapes: 内存的空间形状列表
        # memory_level_start_index: 各层的起始索引
        # attn_mask: 解码器自注意力的掩码
        # memory_mask: 解码器交叉注意力的掩码
        # query_pos_embed: 查询的位置编码

        # ============ 自注意力层 ============
        q = k = self.with_pos_embed(tgt, query_pos_embed)
        # q, k: [B, tgt_len, d_model]，添加位置编码后的查询和键

        # 自注意力计算
        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        # self_attn: [B, tgt_len, d_model]，计算查询与键的注意力并加权值
        tgt = tgt + self.dropout1(tgt2)  # 残差连接 + Dropout
        tgt = self.norm1(tgt)  # 层归一化 [B, tgt_len, d_model]

        # ============ 交叉注意力层 ============
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos_embed),  # 添加位置编码的查询
            reference_points,  # 参考点坐标
            memory,  # Encoder输出作为值
            memory_spatial_shapes,  # 空间形状
            memory_mask)  # 掩码
        # cross_attn: [B, tgt_len, d_model]，可变形注意力聚合多尺度特征
        tgt = tgt + self.dropout2(tgt2)  # 残差连接 + Dropout
        tgt = self.norm2(tgt)  # 层归一化 [B, tgt_len, d_model]

        # ============ 前馈网络 ============
        tgt2 = self.forward_ffn(tgt)  # FFN前向传播
        # forward_ffn: [B, tgt_len, d_model]
        tgt = tgt + self.dropout4(tgt2)  # 残差连接 + Dropout
        # 对数值进行钳制，防止数值溢出（fp16训练时尤为重要）
        tgt = self.norm3(tgt.clamp(min=-65504, max=65504))  # [B, tgt_len, d_model]

        return tgt  # 返回更新后的目标序列


# =========================================================================
# 类名: TransformerDecoder (Transformer解码器)
# 类型: nn.Module 子类
# 代码逻辑链条中的具体职责: 堆叠多个TransformerDecoderLayer形成完整的解码器。
# 管理多层解码器的顺序执行，并收集每层的输出用于辅助损失计算。
# 在RT-DETR中，Decoder负责迭代优化目标检测的边界框和类别预测。
# =========================================================================
class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        # 初始化Transformer解码器
        # hidden_dim: 隐藏层维度
        # decoder_layer: 解码器层的配置
        # num_layers: 解码器层数
        # eval_idx: 评估时使用的层索引，-1表示最后一层
        super(TransformerDecoder, self).__init__()  # 调用父类初始化方法
        # 使用深拷贝创建多个解码器层，确保参数不共享
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        # layers: 解码器层列表，长度为num_layers
        self.hidden_dim = hidden_dim  # 保存隐藏维度
        self.num_layers = num_layers  # 保存层数
        # 计算评估索引，支持负数索引
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        # 解码器的前向传播
        # tgt: 初始查询嵌入 [B, num_queries, hidden_dim]
        # ref_points_unact: 初始参考点（未激活）[B, num_queries, 4]
        # memory: Encoder输出 [B, memory_len, hidden_dim]
        # memory_spatial_shapes: 内存的空间形状
        # memory_level_start_index: 各层的起始索引
        # bbox_head: 边界框预测头列表
        # score_head: 分数预测头列表
        # query_pos_head: 查询位置编码头
        # attn_mask: 自注意力掩码
        # memory_mask: 交叉注意力掩码

        output = tgt  # 初始化输出为查询嵌入
        dec_out_bboxes = []  # 存储每层的边界框预测
        dec_out_logits = []  # 存储每层的类别预测

        # 对参考点进行Sigmoid激活并分离，用于后续bbox计算（避免梯度回传到参考点）
        ref_points_detach = F.sigmoid(ref_points_unact)
        # ref_points_detach: [B, num_queries, 4]，值域[0,1]

        for i, layer in enumerate(self.layers):
            # ============ 准备输入 ============
            # 为参考点添加一个维度，用于后续注意力计算
            ref_points_input = ref_points_detach.unsqueeze(2)
            # ref_points_input: [B, num_queries, 1, 4]
            # 生成查询位置编码
            query_pos_embed = query_pos_head(ref_points_detach)
            # query_pos_head: [B, num_queries, 4] → [B, num_queries, hidden_dim]

            # ============ 通过解码器层 ============
            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)
            # output: [B, num_queries, hidden_dim]

            # ============ bbox 预测 ============
            # 使用bbox_head预测bbox增量
            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            # bbox_head[i]: [B, num_queries, hidden_dim] → [B, num_queries, 4]
            # inverse_sigmoid: 将[0,1]的参考点转换回sigmoid逆函数空间
            # + : 增量与参考点相加
            # sigmoid: 激活到[0,1]范围
            # inter_ref_bbox: [B, num_queries, 4]

            # ============ 保存输出 ============
            if self.training:
                # 训练模式：保存所有层的分数预测
                dec_out_logits.append(score_head[i](output))
                # score_head[i]: [B, num_queries, hidden_dim] → [B, num_queries, num_classes]
                # 仅第一层保存bbox（后续层的bbox由参考点累积产生）
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    # 其他层使用上一层的参考点
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                # 评估模式：只保存指定层的输出
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break  # 提前退出循环

            # 更新参考点
            ref_points = inter_ref_bbox  # 用于下一层的计算
            # 根据训练/评估模式决定是否detach梯度
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        # 堆叠所有层的输出
        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)
        # dec_out_bboxes: [num_layers, B, num_queries, 4]
        # dec_out_logits: [num_layers, B, num_queries, num_classes]


# =========================================================================
# 类名: RTDETRTransformer (RT-DETR Transformer)
# 类型: nn.Module 子类（使用@register装饰器注册到模型库）
# 代码逻辑链条中的具体职责: RT-DETR的完整Transformer实现，整合了Encoder、Decoder、
# 特征投影、位置编码、预测头等所有组件。是RT-DETR检测器的核心模块，
# 负责从Backbone特征生成最终的目标检测结果。
# =========================================================================
@register  # 注册装饰器，将此类注册到模型库中
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']  # 共享参数配置，num_classes在所有子模块间共享

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True):
        # 初始化RT-DETR Transformer
        # num_classes: 目标类别数
        # hidden_dim: 隐藏层维度（所有层的统一维度）
        # num_queries: 查询向量数量（等于最大检测目标数）
        # position_embed_type: 位置编码类型 ('sine' 或 'learned')
        # feat_channels: Backbone特征通道列表 [C3, C4, C5]
        # feat_strides: Backbone特征步长列表 [8, 16, 32]
        # num_levels: 特征金字塔层数
        # num_decoder_points: 解码器每层采样点数
        # nhead: 注意力头数
        # num_decoder_layers: 解码器层数
        # dim_feedforward: 前馈网络维度
        # dropout: Dropout比例
        # activation: 激活函数类型
        # num_denoising: 去噪查询数量（用于DN-DETR训练）
        # label_noise_ratio: 标签噪声比例
        # box_noise_scale: 边界框噪声规模
        # learnt_init_query: 是否使用可学习的初始查询
        # eval_spatial_size: 评估时的空间尺寸
        # eval_idx: 评估时使用的解码器层索引
        # eps: 数值稳定性参数
        # aux_loss: 是否启用辅助损失

        super(RTDETRTransformer, self).__init__()  # 调用父类初始化方法

        # ============ 参数断言 ============
        # 检查位置编码类型是否支持
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        # 检查特征通道数是否不超过层级数
        assert len(feat_channels) <= num_levels
        # 检查特征通道和特征步长长度是否一致
        assert len(feat_strides) == len(feat_channels)
        # 如果层级数大于特征通道数，补充特征步长
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)  # 步长翻倍

        # ============ 保存配置参数 ============
        self.hidden_dim = hidden_dim  # 隐藏维度
        self.nhead = nhead  # 注意力头数
        self.feat_strides = feat_strides  # 特征步长列表
        self.num_levels = num_levels  # 特征层数
        self.num_classes = num_classes  # 类别数
        self.num_queries = num_queries  # 查询数
        self.eps = eps  # 数值稳定性参数
        self.num_decoder_layers = num_decoder_layers  # 解码器层数
        self.eval_spatial_size = eval_spatial_size  # 评估空间尺寸
        self.aux_loss = aux_loss  # 辅助损失开关

        # ============ 构建输入投影层 ============
        self._build_input_proj_layer(feat_channels)  # 构建Backbone到Transformer的特征投影

        # ============ 构建Transformer模块 ============
        # 创建解码器层配置
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        # 创建完整的解码器
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # ============ 去噪配置 ============
        self.num_denoising = num_denoising  # 去噪查询数量
        self.label_noise_ratio = label_noise_ratio  # 标签噪声比例
        self.box_noise_scale = box_noise_scale  # 边界框噪声规模

        # ============ 去噪模块 ============
        if num_denoising > 0:
            # 创建去噪类别嵌入（+1是为了额外的padding类别）
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)

        # ============ 解码器嵌入配置 ============
        self.learnt_init_query = learnt_init_query  # 是否学习初始查询
        if learnt_init_query:
            # 使用可学习的初始查询嵌入
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)  # [num_queries, hidden_dim]
        # 查询位置编码头：将bbox坐标转换为位置编码
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # ============ Encoder输出头 ============
        self.enc_output = nn.Sequential(
            # Encoder输出的后处理层
            nn.Linear(hidden_dim, hidden_dim),  # [B, seq, hidden_dim]
            nn.LayerNorm(hidden_dim,)  # 层归一化
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)  # Encoder分数预测头 [B, seq, num_classes]
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)  # Encoder bbox预测头
            # [B, seq, hidden_dim] → [B, seq, 4]
            
        # ============ Decoder预测头 ============
        # Decoder分数预测头（每层一个）
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        # Decoder bbox预测头（每层一个）
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # ============ 预计算评估锚点 ============
        if self.eval_spatial_size:
            # 如果指定了评估空间尺寸，预先生成锚点和有效掩码
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()  # 初始化所有参数

    def _reset_parameters(self):
        # 初始化所有可学习参数的权重
        bias = bias_init_with_prob(0.01)  # 计算偏置初始化值（基于先验概率）

        # ============ Encoder输出头初始化 ============
        init.constant_(self.enc_score_head.bias, bias)  # Encoder分数头的偏置
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)  # Encoder bbox头的最后一层权重
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)  # Encoder bbox头的最后一层偏置

        # ============ Decoder预测头初始化 ============
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # 遍历所有Decoder层
            init.constant_(cls_.bias, bias)  # 分数头偏置
            init.constant_(reg_.layers[-1].weight, 0)  # bbox头最后一层权重
            init.constant_(reg_.layers[-1].bias, 0)  # bbox头最后一层偏置

        # ============ 其他参数初始化 ============
        # 使用Xavier均匀初始化
        init.xavier_uniform_(self.enc_output[0].weight)  # Encoder输出层的线性权重
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)  # 可学习查询嵌入
        # 初始化查询位置编码头
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        # 构建输入投影层：将Backbone特征投影到Transformer维度
        self.input_proj = nn.ModuleList()  # 创建模块列表
        for in_channels in feat_channels:
            # 为每个Backbone输出创建一个投影层
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    # 1x1卷积进行通道变换
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                    # 批量归一化
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )
            # 输入: [B, in_channels, H, W]
            # 输出: [B, hidden_dim, H, W]

        # 如果num_levels大于特征通道数，需要额外添加投影层
        in_channels = feat_channels[-1]  # 使用最后一个Backbone特征的通道数

        for _ in range(self.num_levels - len(feat_channels)):
            # 添加额外的层级投影（使用3x3卷积+下采样）
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    # 3x3卷积，步长2，实现下采样
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            # 输入: [B, hidden_dim, H, W]
            # 输出: [B, hidden_dim, H/2, W/2]
            in_channels = self.hidden_dim  # 更新输入通道数

    def _get_encoder_input(self, feats):
        # 将Backbone特征转换为Encoder输入格式
        # feats: Backbone输出的特征列表 [P3, P4, P5]
        # feats[i]: [B, C_i, H_i, W_i]  # C_i: 通道数，H_i: 高度，W_i: 宽度

        # ============ 特征投影 ============
        # 对所有特征进行通道投影
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # proj_feats: 投影后的特征列表，每个元素 [B, hidden_dim, H_i, W_i]

        # 如果层级数大于投影后的特征数，需要额外处理
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)  # 当前投影特征数量
            for i in range(len_srcs, self.num_levels):
                # 对额外层级进行投影
                if i == len_srcs:
                    # 第一个额外层级：对原始特征进行投影
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    # 后续额外层级：对上一投影特征进行下采样
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # ============ 展平并拼接特征 ============
        feat_flatten = []  # 存储展平后的特征
        spatial_shapes = []  # 存储各层的空间形状
        level_start_index = [0, ]  # 各层的起始索引

        for i, feat in enumerate(proj_feats):
            # 获取当前特征的空间尺寸
            _, _, h, w = feat.shape  # feat: [B, hidden_dim, h, w]
            # 展平特征并转换维度顺序
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # flatten(2): [B, hidden_dim, h*w]
            # permute(0,2,1): [B, h*w, hidden_dim]
            # 保存空间形状
            spatial_shapes.append([h, w])  # [(H_0, W_0), (H_1, W_1), ...]
            # 计算并保存层起始索引
            level_start_index.append(h * w + level_start_index[-1])
            # level_start_index: [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]

        # 拼接所有层的特征
        feat_flatten = torch.concat(feat_flatten, 1)  # [B, H_0*W_0+H_1*W_1+..., hidden_dim]
        level_start_index.pop()  # 移除最后一个哨兵值

        return (feat_flatten, spatial_shapes, level_start_index)
        # feat_flatten: [B, total_len, hidden_dim]，所有层级特征拼接
        # spatial_shapes: [num_levels, 2]，各层空间形状
        # level_start_index: [num_levels]，各层起始索引

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        # 生成Encoder输出的参考锚点
        # spatial_shapes: 各特征层的空间形状
        # grid_size: 网格大小（作为初始WH的参考）
        # dtype: 数据类型
        # device: 设备

        if spatial_shapes is None:
            # 如果未指定空间形状，根据评估尺寸和步长计算
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
            # spatial_shapes: [[H_0, W_0], [H_1, W_1], ...]

        anchors = []  # 存储各层的锚点
        for lvl, (h, w) in enumerate(spatial_shapes):
            # 为每一层生成锚点
            # 创建网格坐标
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            # grid_y, grid_x: [H, W]
            # 合并x,y坐标
            grid_xy = torch.stack([grid_x, grid_y], -1)  # [H, W, 2]
            # 有效宽高
            valid_WH = torch.tensor([w, h]).to(dtype)  # [2]
            # 归一化中心点坐标到[0,1]
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            # grid_xy: [1, H, W, 2]，值域(0.5/W, 1-0.5/W)

            # 生成锚点宽高（与层级相关，层级越高WH越大）
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)  # [1, H, W, 2]
            # 拼接中心点和宽高
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
            # anchors[-1]: [1, H*W, 4] → [1, h*w, 4]

        # 拼接所有层的锚点
        anchors = torch.concat(anchors, 1).to(device)  # [1, total_anchors, 4]
        # total_anchors = H_0*W_0 + H_1*W_1 + ...

        # 生成有效掩码（锚点坐标在(eps, 1-eps)范围内）
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        # valid_mask: [1, total_anchors, 1]，True表示有效锚点

        # 对锚点进行对数变换（将[0,1]映射到实数空间）
        anchors = torch.log(anchors / (1 - anchors))
        # anchors: [1, total_anchors, 4]

        # 将无效锚点设置为无穷大
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask
        # anchors: [1, total_anchors, 4]，对数空间的锚点坐标
        # valid_mask: [1, total_anchors, 1]，有效锚点掩码

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        # 准备Decoder的输入
        # memory: Encoder输出 [B, memory_len, hidden_dim]
        # spatial_shapes: 空间形状列表
        # denoising_class: 去噪类别嵌入
        # denoising_bbox_unact: 去噪边界框（未激活）

        bs, _, _ = memory.shape  # 获取batch大小

        # ============ 获取锚点 ============
        if self.training or self.eval_spatial_size is None:
            # 训练模式或未指定评估尺寸：动态生成锚点
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            # 评估模式：使用预计算的锚点
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # ============ 应用有效掩码 ============
        memory = valid_mask.to(memory.dtype) * memory  # 将无效位置置0

        # ============ Encoder输出后处理 ============
        output_memory = self.enc_output(memory)  # [B, memory_len, hidden_dim]

        # Encoder预测
        enc_outputs_class = self.enc_score_head(output_memory)  # [B, memory_len, num_classes]
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors  # [B, memory_len, 4]

        # ============ 选择Top-K查询 ============
        # 选择分数最高的Top-K个查询
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        # enc_outputs_class.max(-1).values: [B, memory_len]，取每个位置的最高分数
        # topk_ind: [B, num_queries]，Top-K索引

        # 收集Top-K位置的坐标
        reference_points_unact = enc_outputs_coord_unact.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
        # topk_ind.unsqueeze(-1): [B, num_queries, 1]
        # repeat: [B, num_queries, 4]
        # gather: [B, num_queries, 4]

        # Sigmoid激活得到归一化坐标
        enc_topk_bboxes = F.sigmoid(reference_points_unact)  # [B, num_queries, 4]

        # ============ 添加去噪查询 ============
        if denoising_bbox_unact is not None:
            # 如果有去噪查询，将其添加到参考点
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
            # [B, num_denoising + num_queries, 4]

        # 收集Top-K位置的类别分数
        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))
        # enc_topk_logits: [B, num_queries, num_classes]

        # ============ 准备Decoder目标嵌入 ============
        if self.learnt_init_query:
            # 使用可学习的初始查询
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
            # tgt_embed.weight: [num_queries, hidden_dim]
            # target: [B, num_queries, hidden_dim]
        else:
            # 从Encoder输出中收集查询特征
            target = output_memory.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            # target: [B, num_queries, hidden_dim]
            target = target.detach()  # 分离梯度

        # 添加去噪类别嵌入
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)
            # [B, num_denoising + num_queries, hidden_dim]

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits
        # target: Decoder输入查询嵌入
        # reference_points_unact.detach(): 分离梯度后的参考点
        # enc_topk_bboxes: Encoder预测的Top-K bbox
        # enc_topk_logits: Encoder预测的Top-K logits

    def forward(self, feats, targets=None):
        # 完整的前向传播
        # feats: Backbone特征列表 [P3, P4, P5]
        # targets: 训练时的目标标签（字典列表）

        # ============ Encoder输入准备 ============
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        # memory: [B, total_len, hidden_dim] 
        # spatial_shapes: [num_levels, 2] 各层的空间形状
        # level_start_index: [num_levels]，每个层的起始索引，用于计算Top-K查询的索引

        # ============ 去噪训练准备 ============
        if self.training and self.num_denoising > 0:
            # 训练模式且启用去噪：生成去噪查询
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes,  # 类别数
                    self.num_queries,  # 查询数
                    self.denoising_class_embed,  # 类别嵌入
                    num_denoising=self.num_denoising,  # 去噪查询数量
                    label_noise_ratio=self.label_noise_ratio,  # 标签噪声比例
                    box_noise_scale=self.box_noise_scale,  # bbox噪声规模
                )
        else:
            # 评估模式或无去噪：设置为None
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None
        # denoising_class: [B, num_denoising, hidden_dim]，去噪类别嵌入
        # denoising_bbox_unact: [B, num_denoising, 4]，去噪bbox（未激活）
        # attn_mask: [B, num_denoising + num_queries, num_denoising + num_queries]，注意力掩码
        # dn_meta: 去噪元信息字典，包含去噪查询数量等信息，用于辅助损失计算 

        # ============ Decoder输入准备 ============
        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)
        # target: [B, num_queries, hidden_dim] 或 [B, num_denoising+num_queries, hidden_dim]
        # init_ref_points_unact: [B, num_queries, 4] 或 [B, num_denoising+num_queries, 4]


        # ============ Decoder前向传播 ============
        out_bboxes, out_logits = self.decoder(
            target,  # 查询嵌入
            init_ref_points_unact,  # 初始参考点
            memory,  # Encoder输出
            spatial_shapes,  # 空间形状
            level_start_index,  # 层起始索引
            self.dec_bbox_head,  # bbox预测头
            self.dec_score_head,  # 分数预测头
            self.query_pos_head,  # 查询位置编码头
            attn_mask=attn_mask)  # 注意力掩码
        # out_bboxes: [num_layers, B, num_queries, 4]
        # out_logits: [num_layers, B, num_queries, num_classes]

        # ============ 分离去噪输出 ============
        if self.training and dn_meta is not None:
            # 如果有去噪输出，分离主输出和去噪输出
            # dn_out_bboxes是去噪查询的预测结果，out_bboxes是主查询的预测结果
            dn_out_bboxes, out_bboxes = torch.split(
                out_bboxes, dn_meta['dn_num_split'], dim=2)
            # dn_num_split: [dn_num, query_num]
            dn_out_logits, out_logits = torch.split(
                out_logits, dn_meta['dn_num_split'], dim=2)

        # ============ 构建输出字典 ============
        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1]}
        # 只返回最后一层的预测结果
        # pred_logits: [B, num_queries, num_classes]
        # pred_boxes: [B, num_queries, 4]

        # ============ 添加辅助损失 ============
        if self.training and self.aux_loss:
            # 设置所有Decoder层的辅助输出
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            # 包含Encoder的输出
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            # 如果有去噪输出，添加去噪辅助输出
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out
        # 返回预测结果字典，包含主输出和辅助输出

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 为辅助损失设置输出格式
        # outputs_class: 类别预测列表
        # outputs_coord: 坐标预测列表
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
        # 返回字典列表，每个字典包含一个layer的输出
        # [{'pred_logits': tensor, 'pred_boxes': tensor}, ...]