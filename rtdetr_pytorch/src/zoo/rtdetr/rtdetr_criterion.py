"""
reference: 
https://github.com/facebookresearch/detr/blob/main/models/detr.py

by lyuwenyu
"""

import torch # 导入PyTorch核心库，提供张量操作和神经网络构建功能
import torch.nn as nn # 导入PyTorch神经网络模块，提供Module基类和各种层
import torch.nn.functional as F # 导入PyTorch函数式接口，包含各种损失函数和激活函数
import torchvision # 导入torchvision库，提供计算机视觉相关的模型和操作

# from torchvision.ops import box_convert, generalized_box_iou
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou # 从当前包的box_ops模块导入边界框操作函数：坐标转换、IoU计算和广义IoU计算

from src.misc.dist import get_world_size, is_dist_available_and_initialized # 从项目misc模块导入分布式训练工具函数：获取进程数和检查分布式初始化状态
from src.core import register # 从项目core模块导入注册器装饰器，用于将类注册到全局模块表中

# ============================================================================
# 函数/类级总结：
# 1. SetCriterion(nn.Module): RT-DETR的损失计算核心类，负责管理多种损失函数的计算流程，包括分类损失（支持交叉熵、BCE、Focal、VFL等多种变体）、边界框回归损失（L1和GIoU）、基数误差、掩码损失等。该类通过匈牙利匹配算法将模型预测与真实目标进行配对，并支持辅助损失和去噪训练（CDN）损失的计算。
# 2. accuracy(output, target, topk): 辅助函数，用于计算模型预测的top-k准确率，支持多k值输出，用于评估分类性能。
# 
# 整体功能总结：
# 这个文件实现了RT-DETR（Real-Time DEtection TRansformer）模型的损失计算模块。SetCriterion是核心组件，它协调整个损失计算流程：首先通过匈牙利匹配算法将模型的预测框与真实目标框进行最优配对，然后根据配置计算多种损失函数（分类损失、边界框损失、基数误差等），最后将各损失加权求和。该模块还支持多层辅助损失和去噪训练（Conditional Denoising）损失的独立计算，是RT-DETR模型端到端训练的关键组件，负责将模型预测与真实标签之间的差异转化为可优化的梯度信号。
# ============================================================================

@register # 使用register装饰器将SetCriterion类注册到全局模块注册表中，使其可以通过配置字符串动态创建
class SetCriterion(nn.Module): # 定义RT-DETR的损失计算器类，继承自nn.Module，是整个损失计算流程的核心控制器
    """ This class computes the loss for DETR. # 类的文档字符串，说明其用途是计算DETR系列模型的损失
    The process happens in two steps: # 损失计算分为两个主要步骤
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model # 第一步：使用匈牙利算法在预测框和真实框之间找到最优匹配
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box) # 第二步：对每个匹配对计算监督损失（分类损失和边界框损失）
    """
    __share__ = ['num_classes', ] # 类属性，定义需要在不同实例间共享的参数列表，这里共享类别数量配置
    __inject__ = ['matcher', ] # 类属性，定义需要依赖注入的模块，这里注入匈牙利匹配器用于预测与目标的配对

    def __init__(self, matcher, weight_dict, losses, alpha=0.2, gamma=2.0, eos_coef=1e-4, num_classes=80): # 初始化方法，创建损失计算器实例
        """ Create the criterion. # 初始化方法的文档字符串
        Parameters: # 参数说明部分开始
            num_classes: number of object categories, omitting the special no-object category # 目标类别数量（不含背景类）
            matcher: module able to compute a matching between targets and proposals # 匹配器模块，用于计算预测与真实目标的匹配关系
            weight_dict: dict containing as key the names of the losses and as values their relative weight. # 损失权重字典，键为损失名称，值为权重系数
            eos_coef: relative classification weight applied to the no-object category # 背景类的相对分类权重，用于处理正负样本不平衡
            losses: list of all the losses to be applied. See get_loss for list of available losses. # 要计算的损失类型列表
        """
        super().__init__() # 调用父类nn.Module的初始化方法，确保正确初始化PyTorch模块
        self.num_classes = num_classes # 将类别数保存为实例属性，用于后续损失计算和类别填充
        self.matcher = matcher # 保存匹配器引用，用于forward方法中的预测与目标配对
        self.weight_dict = weight_dict # 保存损失权重字典，用于对各项损失进行加权求和
        self.losses = losses # 保存要计算的损失类型列表，控制forward方法中调用哪些损失函数

        empty_weight = torch.ones(self.num_classes + 1) # 创建包含所有类别和背景类的权重张量，维度为[num_classes+1]（+1表示背景类）
        empty_weight[-1] = eos_coef # 将最后一个元素（背景类）的权重设置为eos_coef，用于降低背景类在交叉熵损失中的影响
        self.register_buffer('empty_weight', empty_weight) # 将权重张量注册为buffer，使其成为模块的一部分但不作为可学习参数，会随模型移动（如.to(cuda)）

        self.alpha = alpha # 保存Focal Loss和VFL损失中的alpha参数，用于调节正负样本的平衡权重
        self.gamma = gamma # 保存Focal Loss和VFL损失中的gamma参数，用于调节易分类和难分类样本的权重衰减


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True): # 定义分类损失函数，使用标准交叉熵损失
        """Classification loss (NLL) # 文档字符串，说明这是使用负对数似然的分类损失
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes] # 目标字典必须包含labels键，对应维度为[目标框数量]的张量
        """
        assert 'pred_logits' in outputs # 断言检查outputs字典中是否包含pred_logits键，确保模型输出包含分类预测
        src_logits = outputs['pred_logits'] # 从模型输出中提取分类logits，维度为[batch_size, num_queries, num_classes+1]

        idx = self._get_src_permutation_idx(indices) # 调用内部方法获取预测索引排列，用于从预测张量中提取与目标匹配的预测
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # 使用列表推导式和torch.cat拼接所有匹配的真实类别标签，维度为[匹配对数量]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, # 创建全填充张量，用背景类标签(num_classes)填充所有位置，维度[batch_size, num_queries]
                                    dtype=torch.int64, device=src_logits.device) # 指定数据类型为64位整型，并确保张量在与logits相同的设备上（CPU/CUDA）
        target_classes[idx] = target_classes_o # 根据匹配索引idx，将对应位置的标签替换为真实类别标签，未匹配位置保持为背景类

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight) # 计算加权交叉熵损失：先将logits转置从[bs,q,c+1]变为[bs,c+1,q]，然后与目标类别和背景权重计算损失，返回维度[]的标量
        losses = {'loss_ce': loss_ce} # 创建损失字典，将交叉熵损失以'loss_ce'为键存储

        if log: # 条件判断是否需要计算并记录分类误差
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # 计算分类错误率：用100减去accuracy函数返回的准确率百分比，用于训练监控，维度[]
        return losses # 返回包含所有分类相关损失的字典

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True): # 定义使用二元交叉熵的分类损失函数
        src_logits = outputs['pred_logits'] # 提取分类预测logits，维度[batch_size, num_queries, num_classes]
        idx = self._get_src_permutation_idx(indices) # 获取预测索引排列，用于索引匹配的预测
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # 拼接匹配的真实类别标签，维度[匹配对数量]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, # 创建全填充目标张量，填充背景类标签，维度[batch_size, num_queries]
                                    dtype=torch.int64, device=src_logits.device) # 设置数据类型和设备
        target_classes[idx] = target_classes_o # 将匹配位置的标签替换为真实类别标签

        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1] # 将类别标签转换为one-hot编码（包含背景类），然后去掉最后一个通道（背景类），维度[batch_size, num_queries, num_classes]
        loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction='none') # 计算二元交叉熵损失（逐元素计算不降维），维度[batch_size, num_queries, num_classes]
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes # 先对第1维（num_queries）求均值，再对第0维求和，最后乘以查询数并除以目标框数进行归一化，维度[]
        return {'loss_bce': loss} # 返回包含BCE损失的字典

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True): # 定义使用Focal Loss的分类损失函数
        assert 'pred_logits' in outputs # 断言检查模型输出包含分类预测
        src_logits = outputs['pred_logits'] # 提取分类logits，维度[batch_size, num_queries, num_classes]

        idx = self._get_src_permutation_idx(indices) # 获取预测索引排列
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # 拼接匹配的真实类别标签，维度[匹配对数量]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, # 创建全填充目标张量，维度[batch_size, num_queries]
                                    dtype=torch.int64, device=src_logits.device) # 设置数据类型和设备
        target_classes[idx] = target_classes_o # 替换匹配位置的标签为真实类别

        target = F.one_hot(target_classes, num_classes=self.num_classes+1)[..., :-1] # 转换为one-hot编码并去掉背景类，维度[batch_size, num_queries, num_classes]
        # ce_loss = F.binary_cross_entropy_with_logits(src_logits, target * 1., reduction="none") # 注释掉的BCE损失计算代码
        # prob = F.sigmoid(src_logits) # TODO .detach() # 注释掉的sigmoid概率计算
        # p_t = prob * target + (1 - prob) * (1 - target) # 注释掉的p_t计算（用于Focal Loss公式）
        # alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target) # 注释掉的alpha_t计算（Focal Loss的alpha因子）
        # loss = alpha_t * ce_loss * ((1 - p_t) ** self.gamma) # 注释掉的完整Focal Loss公式计算
        # loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes # 注释掉的损失归一化
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none') # 使用torchvision内置的sigmoid_focal_loss函数计算Focal损失，维度[batch_size, num_queries, num_classes]
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes # 对查询维度求均值，对batch维度求和，乘以查询数并除以目标框数归一化，维度[]

        return {'loss_focal': loss} # 返回包含Focal损失的字典

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True): # 定义使用Varifocal Loss的分类损失函数
        assert 'pred_boxes' in outputs # 断言检查模型输出包含边界框预测
        idx = self._get_src_permutation_idx(indices) # 获取预测索引排列

        src_boxes = outputs['pred_boxes'][idx] # 根据索引提取匹配的预测边界框，维度[匹配对数量, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # 拼接所有匹配的真实边界框，维度[匹配对数量, 4]
        ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)) # 计算预测框与真实框的IoU矩阵，需要先将中心坐标转换为xyxy格式，返回维度[匹配对数量, 匹配对数量]
        ious = torch.diag(ious).detach() # 提取对角线元素（即匹配对的IoU值），并分离梯度以避免不必要的梯度流，维度[匹配对数量]

        src_logits = outputs['pred_logits'] # 提取分类logits，维度[batch_size, num_queries, num_classes]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # 拼接匹配的真实类别标签，维度[匹配对数量]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, # 创建全填充目标张量，维度[batch_size, num_queries]
                                    dtype=torch.int64, device=src_logits.device) # 设置数据类型和设备
        target_classes[idx] = target_classes_o # 替换匹配位置的标签为真实类别
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1] # 转换为one-hot编码并去掉背景类，维度[batch_size, num_queries, num_classes]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype) # 创建与target_classes形状相同的零张量用于存储目标分数，维度[batch_size, num_queries]
        target_score_o[idx] = ious.to(target_score_o.dtype) # 将匹配位置的IoU值作为目标分数赋给对应位置，维度[batch_size, num_queries]
        target_score = target_score_o.unsqueeze(-1) * target # 将目标分数扩展维度后与one-hot标签相乘，生成加权目标分数，维度[batch_size, num_queries, num_classes]

        pred_score = F.sigmoid(src_logits).detach() # 计算预测的sigmoid分数并分离梯度，维度[batch_size, num_queries, num_classes]
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score # 计算VFL的加权因子：结合预测分数衰减、目标分数和Focal参数，维度[batch_size, num_queries, num_classes]
        
        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none') # 使用目标分数和加权因子计算加权BCE损失，维度[batch_size, num_queries, num_classes]
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes # 归一化处理得到最终损失值，维度[]
        return {'loss_vfl': loss} # 返回包含VFL损失的字典

    @torch.no_grad() # 装饰器，指定此函数在推断时不计算梯度，用于减少内存占用和计算量
    def loss_cardinality(self, outputs, targets, indices, num_boxes): # 定义基数误差损失函数（预测框数量与真实框数量的差异）
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes # 文档字符串说明此损失计算预测非空框数量的误差
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients # 强调此"损失"不用于反向传播，仅用于监控训练过程
        """
        pred_logits = outputs['pred_logits'] # 提取分类logits，维度[batch_size, num_queries, num_classes+1]
        device = pred_logits.device # 获取预测张量所在的设备（CPU/CUDA）
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device) # 计算每个样本的真实目标框数量并转换为张量，维度[batch_size]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1) # 对每个样本统计预测不为背景类（最后一个类别）的框数量，维度[batch_size]
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float()) # 计算预测框数与真实框数的L1损失，维度[]
        losses = {'cardinality_error': card_err} # 将基数误差存入损失字典
        return losses # 返回包含基数误差的字典

    def loss_boxes(self, outputs, targets, indices, num_boxes): # 定义边界框回归损失函数，计算L1损失和GIoU损失
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss # 文档字符串说明计算边界框相关的损失
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4] # 目标字典必须包含boxes键，对应维度为[目标框数, 4]的张量
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size. # 说明目标框格式为归一化的中心坐标+宽高格式
        """
        assert 'pred_boxes' in outputs # 断言检查模型输出包含边界框预测
        idx = self._get_src_permutation_idx(indices) # 获取预测索引排列
        src_boxes = outputs['pred_boxes'][idx] # 根据索引提取匹配的预测边界框，维度[匹配对数量, 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # 拼接所有匹配的真实边界框，维度[匹配对数量, 4]

        losses = {} # 初始化空损失字典

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none') # 计算L1回归损失（逐元素计算），比较预测框与真实框的坐标差异，维度[匹配对数量, 4]
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes # 对所有维度求和后除以目标框数进行归一化，维度[]

        loss_giou = 1 - torch.diag(generalized_box_iou( # 计算广义IoU损失：首先计算GIoU矩阵，然后提取对角线元素，最后用1减去GIoU得到损失
                box_cxcywh_to_xyxy(src_boxes), # 将预测框从中心坐标(cx,cy,w,h)转换为角点坐标(x1,y1,x2,y2)格式，维度[匹配对数量, 4]
                box_cxcywh_to_xyxy(target_boxes))) # 将目标框从中心坐标转换为角点坐标格式，维度[匹配对数量, 4]
        losses['loss_giou'] = loss_giou.sum() / num_boxes # 对所有匹配对的GIoU损失求和并归一化，维度[]
        return losses # 返回包含边界框损失的字典

    def loss_masks(self, outputs, targets, indices, num_boxes): # 定义分割掩码损失函数，计算Focal损失和Dice损失
        """Compute the losses related to the masks: the focal loss and the dice loss. # 文档字符串说明计算掩码相关的损失
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w] # 目标字典必须包含masks键，对应维度为[目标数, 高, 宽]的张量
        """
        assert "pred_masks" in outputs # 断言检查模型输出包含掩码预测

        src_idx = self._get_src_permutation_idx(indices) # 获取预测索引排列
        tgt_idx = self._get_tgt_permutation_idx(indices) # 获取目标索引排列（与预测索引的区别在于索引来源不同）
        src_masks = outputs["pred_masks"] # 提取预测掩码，维度[batch_size, num_queries, h, w]
        src_masks = src_masks[src_idx] # 根据预测索引提取匹配的预测掩码，维度[匹配对数量, h, w]
        masks = [t["masks"] for t in targets] # 从目标列表中提取所有掩码，生成列表
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose() # 将掩码列表转换为嵌套张量格式并进行分解，得到目标掩码和有效掩码，维度[总目标数, h, w]
        target_masks = target_masks.to(src_masks) # 将目标掩码转换到与预测掩码相同的数据类型和设备
        target_masks = target_masks[tgt_idx] # 根据目标索引提取匹配的目标掩码，维度[匹配对数量, h, w]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], # 对预测掩码上采样到目标掩码的尺寸：先在维度1处增加一个维度用于插值，维度[匹配对数量, 1, h, w]
                                mode="bilinear", align_corners=False) # 使用双线性插值方法，align_corners=False表示不进行角点对齐
        src_masks = src_masks[:, 0].flatten(1) # 去除新增的维度并展平空间维度，维度[匹配对数量, h*w]

        target_masks = target_masks.flatten(1) # 展平目标掩码的空间维度，维度[匹配对数量, h*w]
        target_masks = target_masks.view(src_masks.shape) # 调整目标掩码形状以匹配预测掩码的形状，维度[匹配对数量, h*w]
        losses = { # 创建损失字典
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes), # 计算sigmoid focal损失用于掩码分割，维度[]
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes), # 计算Dice损失用于掩码分割，维度[]
        }
        return losses # 返回包含掩码损失的字典

    def _get_src_permutation_idx(self, indices): # 定义私有辅助方法，用于获取预测索引的排列索引
        # permute predictions following indices # 注释说明此方法根据匹配索引对预测进行重排列
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)]) # 为每个样本的匹配索引创建批次标签：遍历indices，为每个样本i创建形状与src相同的全i张量，然后拼接，维度[总匹配数]
        src_idx = torch.cat([src for (src, _) in indices]) # 拼接所有预测索引（indices中的第一个元素），维度[总匹配数]
        return batch_idx, src_idx # 返回批次索引和预测索引的元组，可用于对预测张量进行批量索引

    def _get_tgt_permutation_idx(self, indices): # 定义私有辅助方法，用于获取目标索引的排列索引
        # permute targets following indices # 注释说明此方法根据匹配索引对目标进行重排列
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]) # 为每个样本的匹配索引创建批次标签：遍历indices，为每个样本i创建形状与tgt相同的全i张量，然后拼接，维度[总匹配数]
        tgt_idx = torch.cat([tgt for (_, tgt) in indices]) # 拼接所有目标索引（indices中的第二个元素），维度[总匹配数]
        return batch_idx, tgt_idx # 返回批次索引和目标索引的元组，可用于对目标张量进行批量索引

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs): # 定义损失获取方法，作为损失函数的分发器
        loss_map = { # 创建损失名称到损失函数的映射字典
            'labels': self.loss_labels, # 标准交叉熵分类损失
            'cardinality': self.loss_cardinality, # 基数误差损失
            'boxes': self.loss_boxes, # 边界框回归损失
            'masks': self.loss_masks, # 掩码分割损失

            'bce': self.loss_labels_bce, # 二元交叉熵分类损失
            'focal': self.loss_labels_focal, # Focal分类损失
            'vfl': self.loss_labels_vfl, # Varifocal分类损失
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?' # 断言检查请求的损失类型在映射中存在，如果不存在则抛出错误
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs) # 根据损失类型调用对应的损失计算方法并返回结果

    def forward(self, outputs, targets): # 定义前向传播方法，是损失计算的主入口
        """ This performs the loss computation. # 文档字符串说明此方法执行损失计算
        Parameters: # 参数说明开始
             outputs: dict of tensors, see the output specification of the model for the format # 模型输出的字典，格式参见模型输出规范
             targets: list of dicts, such that len(targets) == batch_size. # 目标字典列表，长度等于批次大小
                      The expected keys in each dict depends on the losses applied, see each loss' doc # 每个字典中的键取决于应用的损失类型
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if 'aux' not in k} # 过滤掉包含'aux'关键字的辅助输出，只保留主输出，生成新字典

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets) # 调用匹配器计算模型主输出与真实目标的匹配关系，返回匹配索引列表

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) # 计算当前批次中所有样本的真实目标框总数，使用生成器表达式累加标签数量
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device) # 将目标框总数转换为PyTorch张量，维度[1]，并指定与模型输出相同的设备
        if is_dist_available_and_initialized(): # 检查分布式训练是否已初始化
            torch.distributed.all_reduce(num_boxes) # 在所有分布式进程间执行全局归约操作，同步目标框总数
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item() # 计算平均目标框数（除以进程数），使用clamp确保最小值为1防止除零，最后转换为Python标量

        # Compute all the requested losses
        losses = {} # 初始化空损失字典
        for loss in self.losses: # 遍历所有需要计算的损失类型
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes) # 调用get_loss方法计算当前类型的损失，返回损失字典
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict} # 将损失值与权重字典中对应的权重相乘，生成加权损失字典
            losses.update(l_dict) # 将加权后的损失更新到总损失字典中

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs: # 检查模型输出中是否包含辅助输出（来自Transformer解码器的中间层）
            for i, aux_outputs in enumerate(outputs['aux_outputs']): # 遍历每个辅助输出及其索引
                indices = self.matcher(aux_outputs, targets) # 为当前辅助输出计算与真实目标的匹配关系
                for loss in self.losses: # 遍历所有损失类型
                    if loss == 'masks': # 判断是否为掩码损失
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue # 跳过掩码损失计算，因为中间层的掩码损失计算成本过高
                    kwargs = {} # 初始化关键字参数字典
                    if loss == 'labels': # 判断是否为分类损失
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False} # 设置log为False，只在最后一层记录分类误差

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs) # 计算辅助输出的各项损失
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict} # 应用损失权重
                    l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()} # 为损失键添加辅助层标识后缀，如loss_ce_aux_0
                    losses.update(l_dict) # 更新到总损失字典

        # In case of cdn auxiliary losses. For rtdetr
        if 'dn_aux_outputs' in outputs: # 检查是否存在去噪辅助输出（RT-DETR特有的条件去噪训练）
            assert 'dn_meta' in outputs, '' # 断言元数据中包含去噪相关信息
            indices = self.get_cdn_matched_indices(outputs['dn_meta'], targets) # 调用静态方法获取去噪匹配的索引
            num_boxes = num_boxes * outputs['dn_meta']['dn_num_group'] # 调整目标框数量，乘以去噪组数以适应去噪训练的批次大小

            for i, aux_outputs in enumerate(outputs['dn_aux_outputs']): # 遍历每个去噪辅助输出
                # indices = self.matcher(aux_outputs, targets) # 注释掉的代码，去噪匹配的索引已在上面统一计算
                for loss in self.losses: # 遍历所有损失类型
                    if loss == 'masks': # 判断是否为掩码损失
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue # 跳过掩码损失计算
                    kwargs = {} # 初始化关键字参数字典
                    if loss == 'labels': # 判断是否为分类损失
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False} # 设置log为False

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs) # 计算去噪辅助损失
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict} # 应用损失权重
                    l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()} # 为损失键添加去噪标识后缀，如loss_ce_dn_0
                    losses.update(l_dict) # 更新到总损失字典

        return losses # 返回包含所有损失的字典，用于模型的反向传播和参数更新

    @staticmethod # 静态方法装饰器，表示此方法不需要访问类或实例属性
    def get_cdn_matched_indices(dn_meta, targets): # 定义静态方法，用于获取去噪训练的匹配索引
        '''get_cdn_matched_indices # 方法文档字符串
        '''
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"] # 从去噪元数据中提取正样本索引列表和去噪组数
        num_gts = [len(t['labels']) for t in targets] # 计算每个样本的真实目标数量，生成列表
        device = targets[0]['labels'].device # 获取标签张量所在的设备
        
        dn_match_indices = [] # 初始化去噪匹配索引列表
        for i, num_gt in enumerate(num_gts): # 遍历每个样本及其索引
            if num_gt > 0: # 判断该样本是否包含真实目标
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device) # 创建从0到num_gt-1的索引张量，维度[num_gt]
                gt_idx = gt_idx.tile(dn_num_group) # 将索引张量按去噪组数重复，生成去噪目标索引，维度[num_gt*dn_num_group]
                assert len(dn_positive_idx[i]) == len(gt_idx) # 断言去噪正样本索引数量与生成的目标索引数量相等
                dn_match_indices.append((dn_positive_idx[i], gt_idx)) # 将正样本索引与目标索引组成元组添加到列表
            else: # 如果该样本没有真实目标
                dn_match_indices.append((torch.zeros(0, dtype=torch.int64, device=device),
                    torch.zeros(0, dtype=torch.int64, device=device))) # 目标和预测的空索引张量
        return dn_match_indices # 返回去噪匹配索引列表，每个元素为(预测索引, 目标索引)的元组


@torch.no_grad() # 装饰器指定此函数不计算梯度，用于推理阶段的准确率计算
def accuracy(output, target, topk=(1,)): # 定义准确率计算函数，支持多k值计算top-k准确率
    """Computes the precision@k for the specified values of k""" # 文档字符串说明计算指定k值的精确率
    if target.numel() == 0: # 检查目标张量中的元素数量是否为零（空目标）
        return [torch.zeros([], device=output.device)] # 返回零张量列表，设备与输出相同
    maxk = max(topk) # 获取topk元组中的最大值，用于确定需要保留多少个最大元素
    batch_size = target.size(0) # 获取批次大小，即第一个维度的大小

    _, pred = output.topk(maxk, 1, True, True) # 对输出张量执行topk操作，获取最大的maxk个值和它们的索引，返回(值, 索引)元组，维度[batch_size, maxk]
    pred = pred.t() # 转置预测结果，维度变为[maxk, batch_size]，便于后续与目标比较
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # 将目标张量重塑为[1, batch_size*...]后扩展到与pred相同形状，比较得到布尔张量表示预测是否正确，维度[maxk, batch_size]

    res = [] # 初始化结果列表
    for k in topk: # 遍历每个需要计算的k值
        correct_k = correct[:k].view(-1).float().sum(0) # 取前k个预测，将张量展平后求和得到正确预测的数量，维度[]
        res.append(correct_k.mul_(100.0 / batch_size)) # 计算准确率百分比并添加到结果列表，使用mul_进行原位乘法
    return res # 返回各k值对应的准确率列表