'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/box_ops.py
'''

import torch  # 导入PyTorch深度学习框架，用于张量计算和神经网络操作
from torchvision.ops.boxes import box_area  # 从torchvision导入box_area函数，用于计算边界框的面积


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)  # 将输入张量x在最后一个维度上解绑为四个变量：中心点x坐标、中心点y坐标、宽度、高度；维度从[...,4]变为[...]的四个独立张量
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),  # 计算左上角坐标：x_min = 中心x - 0.5*宽度，y_min = 中心y - 0.5*高度；维度保持不变
         (x_c + 0.5 * w), (y_c + 0.5 * h)]  # 计算右下角坐标：x_max = 中心x + 0.5*宽度，y_max = 中心y + 0.5*高度；维度保持不变
    return torch.stack(b, dim=-1)  # 将四个坐标分量在最后一个维度上堆叠，形成[...,4]格式的边界框张量；维度从[...]×4变为[...,4]


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)  # 将输入张量x在最后一个维度上解绑为四个变量：左上角x坐标、左上角y坐标、右下角x坐标、右下角y坐标；维度从[...,4]变为[...]的四个独立张量
    b = [(x0 + x1) / 2, (y0 + y1) / 2,  # 计算中心点坐标：x_c = (x0 + x1)/2，y_c = (y0 + y1)/2；维度保持不变
         (x1 - x0), (y1 - y0)]  # 计算宽度和高度：w = x1 - x0，h = y1 - y0；维度保持不变
    return torch.stack(b, dim=-1)  # 将四个分量在最后一个维度上堆叠，形成[...,4]格式的边界框张量；维度从[...]×4变为[...,4]


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)  # 计算第一组边界框的面积；输入维度[N,4]，输出维度[N]
    area2 = box_area(boxes2)  # 计算第二组边界框的面积；输入维度[M,4]，输出维度[M]

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # 计算两组边界框的左上角坐标的最大值，即交集的左上角；boxes1维度[N,1,2]，boxes2维度[M,2]，广播后维度[N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # 计算两组边界框的右下角坐标的最小值，即交集的右下角；boxes1维度[N,1,2]，boxes2维度[M,2]，广播后维度[N,M,2]

    wh = (rb - lt).clamp(min=0)  # 计算交集的宽度和高度，并将负值截断为0；维度[N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # 计算交集面积：宽度×高度；维度从[N,M,2]变为[N,M]

    union = area1[:, None] + area2 - inter  # 计算并集面积：area1 + area2 - 交集面积；area1[:,None]维度[N,1]，area2维度[M]，inter维度[N,M]，广播后维度[N,M]

    iou = inter / union  # 计算IoU值：交集面积/并集面积；维度[N,M]
    return iou, union  # 返回IoU矩阵和并集面积矩阵；IoU维度[N,M]，union维度[N,M]


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()  # 断言检查第一组边界框的有效性：右下角坐标必须大于等于左上角坐标；维度[N,2]比较
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()  # 断言检查第二组边界框的有效性：右下角坐标必须大于等于左上角坐标；维度[M,2]比较
    iou, union = box_iou(boxes1, boxes2)  # 调用box_iou函数计算IoU和并集面积；输入维度[N,4]和[M,4]，输出iou维度[N,M]，union维度[N,M]

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # 计算两组边界框左上角坐标的最小值，即最小外接矩形的左上角；boxes1维度[N,1,2]，boxes2维度[M,2]，广播后维度[N,M,2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # 计算两组边界框右下角坐标的最大值，即最小外接矩形的右下角；boxes1维度[N,1,2]，boxes2维度[M,2]，广播后维度[N,M,2]

    wh = (rb - lt).clamp(min=0)  # 计算最小外接矩形的宽度和高度；维度[N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]  # 计算最小外接矩形的面积；维度从[N,M,2]变为[N,M]

    return iou - (area - union) / area  # 计算Generalized IoU：IoU - (最小外接矩形面积 - 并集面积)/最小外接矩形面积；维度[N,M]


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:  # 检查输入mask张量是否为空；返回布尔值
        return torch.zeros((0, 4), device=masks.device)  # 如果mask为空，返回形状为(0,4)的零张量；维度[0,4]

    h, w = masks.shape[-2:]  # 获取mask的空间维度高度和宽度；从[N,H,W]中提取H和W

    y = torch.arange(0, h, dtype=torch.float)  # 创建从0到h-1的y坐标序列；维度[H]
    x = torch.arange(0, w, dtype=torch.float)  # 创建从0到w-1的x坐标序列；维度[W]
    y, x = torch.meshgrid(y, x)  # 生成二维网格坐标矩阵；y维度[H,W]，x维度[H,W]

    x_mask = (masks * x.unsqueeze(0))  # 将x坐标与mask相乘，得到mask区域内每个像素的x坐标；masks维度[N,H,W]，x.unsqueeze(0)维度[1,H,W]，广播后维度[N,H,W]
    x_max = x_mask.flatten(1).max(-1)[0]  # 计算每个mask中x坐标的最大值；flatten(1)将[N,H,W]展平为[N,H*W]，max(-1)在最后一维求最大值，输出维度[N]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]  # 将mask外的区域填充为大数值1e8，然后计算最小值得到mask的左边界；维度[N]

    y_mask = (masks * y.unsqueeze(0))  # 将y坐标与mask相乘，得到mask区域内每个像素的y坐标；masks维度[N,H,W]，y.unsqueeze(0)维度[1,H,W]，广播后维度[N,H,W]
    y_max = y_mask.flatten(1).max(-1)[0]  # 计算每个mask中y坐标的最大值；flatten(1)将[N,H,W]展平为[N,H*W]，max(-1)在最后一维求最大值，输出维度[N]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]  # 将mask外的区域填充为大数值1e8，然后计算最小值得到mask的上边界；维度[N]

    return torch.stack([x_min, y_min, x_max, y_max], 1)  # 将四个边界坐标在维度1上堆叠，形成[N,4]格式的边界框张量；从四个[N]维度张量堆叠为[N,4]


# 函数级总结：
# 1. box_cxcywh_to_xyxy(x): 将边界框从中心点坐标格式(cx,cy,w,h)转换为左上右下角坐标格式(x_min,y_min,x_max,y_max)，用于边界框格式转换
# 2. box_xyxy_to_cxcywh(x): 将边界框从左上右下角坐标格式(x_min,y_min,x_max,y_max)转换为中心点坐标格式(cx,cy,w,h)，用于边界框格式转换
# 3. box_iou(boxes1, boxes2): 计算两组边界框之间的交并比(IoU)和并集面积，用于目标检测中的边界框匹配和评估
# 4. generalized_box_iou(boxes1, boxes2): 计算两组边界框之间的广义交并比(GIoU)，解决了IoU在边界框不重叠时为0的问题，提供更好的梯度信息
# 5. masks_to_boxes(masks): 将分割mask转换为对应的边界框坐标，用于从实例分割结果中提取边界框

# 整体功能总结：
# 这个代码文件实现了目标检测和实例分割任务中边界框操作的核心功能，包括边界框格式转换、IoU计算、广义IoU计算以及从mask到边界框的转换。这些函数是DETR(Detection Transformer)模型的基础组件，用于处理边界框的坐标变换、相似度计算和损失函数计算，支持目标检测模型训练和推理过程中的边界框操作需求。