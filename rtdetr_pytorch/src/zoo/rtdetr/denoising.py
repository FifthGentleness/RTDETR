"""by lyuwenyu
""" # 文件头部注释，标识作者信息

import torch  # 导入PyTorch深度学习框架，用于张量计算和神经网络操作

from .utils import inverse_sigmoid  # 从本地utils模块导入inverse_sigmoid函数，用于逆sigmoid变换
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh  # 从本地box_ops模块导入边界框格式转换函数，支持中心宽高格式和左上右下格式的相互转换



def get_contrastive_denoising_training_group(targets,  # targets: 数据集中每个样本的真实标注列表，每个元素是字典包含'labels'（类别张量）和'boxes'（边界框张量）
                                             num_classes,  # num_classes: 分类任务的总类别数，用于创建填充值和随机标签的取值范围（随机标签范围为0到num_classes-1）
                                             num_queries,  # num_queries: 正常检测查询的数量，即模型中用于直接目标检测的可学习查询向量个数
                                             class_embed,  # class_embed: 类别嵌入网络层，将类别索引映射为密集的嵌入向量，用于后续的特征匹配和分类
                                             num_denoising=100,  # num_denoising: 对比去噪训练的去噪查询总数，控制每批次中生成的正负样本对数量，数值越大训练越充分但计算量增加
                                             label_noise_ratio=0.5,  # label_noise_ratio: 标签噪声比例参数，控制正样本中有多少比例的标签被替换为随机错误标签，用于增强模型对错误标签的鲁棒性
                                             box_noise_scale=1.0,):  # box_noise_scale: 边界框噪声缩放系数，控制正负样本边界框扰动的幅度，值越大边界框偏移越明显，用于训练模型识别不同重叠程度的边界框
    """Contrastive Denoising Training Group Generator.  # 函数文档字符串，说明此函数用于生成对比去噪训练组
    
    该函数实现RT-DETR中的对比去噪训练机制，通过构造正负样本对来增强检测器的判别能力：
    - 正样本：真实标注添加小扰动，模型应该正确匹配
    - 负样本：类别标签错误+边界框大偏移，模型应该拒绝匹配
    """  # 多行文档字符串，详细说明函数功能和设计原理
    if num_denoising <= 0:  # 判断去噪查询数量是否有效，若为0或负数则跳过去噪训练，直接返回None表示不启用此机制
        return None, None, None, None  # 返回四个None值作为占位符，后续代码会检查此返回值来判断是否进行去噪训练

    num_gts = [len(t['labels']) for t in targets]  # 遍历批次中每个样本的标注数据，提取其类别标签数量，得到每个样本包含的真实目标个数列表，维度：[bs]
    device = targets[0]['labels'].device  # 获取第一个样本的标签张量所在的计算设备（cuda或cpu），确保后续创建的所有张量都在同一设备上以避免设备不匹配错误
    
    max_gt_num = max(num_gts)  # 找出批次中真实目标数量的最大值，用于确定padding的列数，使得所有样本的标注可以对齐到相同形状
    if max_gt_num == 0:  # 检查批次中是否存在真实目标，若所有样本都没有目标则无法进行去噪训练（无法构造正负样本对）
        return None, None, None, None  # 返回四个None值，因为没有真实目标就无法构造去噪查询

    num_group = num_denoising // max_gt_num  # 计算每个样本的真实目标需要复制多少组
    num_group = 1 if num_group == 0 else num_group  # 当批次中目标数量超过指定的去噪查询数量时，num_group会为0，此时强制设置为1保证至少有一组正负样本对进行对比学习
    # pad gt to max_num of a batch # 注释说明：将真实目标padding到批次中的最大数量
    bs = len(num_gts) # 获取批次大小（batch size），即样本数量

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device) # 创建类别查询张量，形状为[batch_size, max_gt_num]，用num_classes填充，用于padding，维度：[bs, max_gt_num]
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device) # 创建边界框查询张量，形状为[batch_size, max_gt_num, 4]，初始化为0，用于padding，维度：[bs, max_gt_num, 4]
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device) # 创建padding掩码张量，形状为[batch_size, max_gt_num]，用于标记哪些位置是真实目标，维度：[bs, max_gt_num]

    for i in range(bs): # 遍历批次中的每个样本
        num_gt = num_gts[i] # 获取当前样本的真实目标数量
        if num_gt > 0: # 检查当前样本是否有真实目标
            input_query_class[i, :num_gt] = targets[i]['labels'] # 将当前样本的真实标签填充到类别查询张量的前num_gt个位置，维度：[bs, max_gt_num]
            input_query_bbox[i, :num_gt] = targets[i]['boxes'] # 将当前样本的真实边界框填充到边界框查询张量的前num_gt个位置，维度：[bs, max_gt_num, 4]
            pad_gt_mask[i, :num_gt] = 1 # 将padding掩码的前num_gt个位置设置为1，表示这些位置是真实目标，维度：[bs, max_gt_num]
    # each group has positive and negative queries. # 注释说明：每个组包含正样本和负样本查询；每个批次的总查询数=max_gt_num×2×num_group=num_denoising*2
    input_query_class = input_query_class.tile([1, 2 * num_group]) # 将类别查询张量在维度1上复制2*num_group次，生成正负样本对，维度：[bs, max_gt_num] -> [bs, max_gt_num * 2 * num_group]
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1]) # 将边界框查询张量在维度1上复制2*num_group次，维度2保持不变，维度：[bs, max_gt_num, 4] -> [bs, max_gt_num * 2 * num_group, 4]
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group]) # 将padding掩码张量在维度1上复制2*num_group次，维度：[bs, max_gt_num] -> [bs, max_gt_num * 2 * num_group]
    # positive and negative mask # 注释说明：创建正样本和负样本掩码
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device) # 创建负样本掩码张量，形状为[batch_size, max_gt_num * 2, 1]，初始化为0，维度：[bs, max_gt_num * 2, 1]
    negative_gt_mask[:, max_gt_num:] = 1 # 将负样本掩码的后半部分（max_gt_num到末尾）设置为1，标记为负样本，维度：[bs, max_gt_num * 2, 1]
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1]) # 将负样本掩码在维度1上复制num_group次，维度：[bs, max_gt_num * 2, 1] -> [bs, max_gt_num * 2 * num_group, 1]
    positive_gt_mask = 1 - negative_gt_mask # 通过1减去负样本掩码得到正样本掩码，维度：[bs, max_gt_num * 2 * num_group, 1]
    # contrastive denoising training positive index # 注释说明：计算对比去噪训练的正样本索引
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask # 压缩正样本掩码的最后一维，并与padding掩码相乘，得到最终的正样本掩码，维度：[bs, max_gt_num * 2 * num_group, 1] -> [bs, max_gt_num * 2 * num_group]
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1] # 找到正样本掩码中非零元素的索引，取第二列（列索引），得到正样本在查询序列中的位置，维度：[num_positive_samples, 2] -> [num_positive_samples]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts]) # 将正样本索引按样本分割，每个样本对应一个索引张量，返回元组
    # total denoising queries # 注释说明：计算总的去噪查询数量
    num_denoising = int(max_gt_num * 2 * num_group) # 计算总的去噪查询数量，即最大目标数乘以2（正负样本）再乘以组数

# 在正样本中添加标签噪声，增强模型对错误标签的鲁棒性
    if label_noise_ratio > 0: # 检查标签噪声比例是否大于0，如果是则添加标签噪声
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5) # 生成随机掩码，概率为label_noise_ratio * 0.5，用于选择哪些标签需要添加噪声，维度：[bs, max_gt_num * 2 * num_group]
        # randomly put a new one here # 注释说明：随机替换标签
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype) # 生成随机的新标签，范围从0到num_classes-1，维度：[bs, max_gt_num * 2 * num_group]
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class) # 根据掩码条件选择性地替换标签，只有当掩码为True且padding掩码为True时才替换，维度：[bs, max_gt_num * 2 * num_group]

    # if label_noise_ratio > 0:
    #     input_query_class = input_query_class.flatten()
    #     pad_gt_mask = pad_gt_mask.flatten()
    #     # half of bbox prob
    #     # mask = torch.rand(input_query_class.shape, device=device) < (label_noise_ratio * 0.5)
    #     mask = torch.rand_like(input_query_class) < (label_noise_ratio * 0.5)
    #     chosen_idx = torch.nonzero(mask * pad_gt_mask).squeeze(-1)
    #     # randomly put a new one here
    #     new_label = torch.randint_like(chosen_idx, 0, num_classes, dtype=input_query_class.dtype)
    #     # input_query_class.scatter_(dim=0, index=chosen_idx, value=new_label)
    #     input_query_class[chosen_idx] = new_label
    #     input_query_class = input_query_class.reshape(bs, num_denoising)
    #     pad_gt_mask = pad_gt_mask.reshape(bs, num_denoising)

    if box_noise_scale > 0: # 检查边界框噪声缩放因子是否大于0，如果是则添加边界框噪声
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox) # 将边界框从中心宽高格式(cxcywh)转换为左上右下格式(xyxy)，维度：[bs, max_gt_num * 2 * num_group, 4]
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale # 计算边界框的宽高的一半，并在最后一维复制2次，然后乘以噪声缩放因子，维度：[bs, max_gt_num * 2 * num_group, 4]
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0 # 生成随机符号，0或1，然后转换为-1或1，用于控制噪声的方向，维度：[bs, max_gt_num * 2 * num_group, 4]
        rand_part = torch.rand_like(input_query_bbox) # 生成随机噪声值，范围在[0, 1)之间，维度：[bs, max_gt_num * 2 * num_group, 4]
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask) # 对负样本的噪声值加1，使其范围在[1, 2)之间，正样本保持[0, 1)，维度：[bs, max_gt_num * 2 * num_group, 4]
        rand_part *= rand_sign # 将噪声值乘以随机符号，得到正负噪声，维度：[bs, max_gt_num * 2 * num_group, 4]
        known_bbox += rand_part * diff # 将噪声加到边界框上，得到加噪后的边界框，维度：[bs, max_gt_num * 2 * num_group, 4]
        known_bbox.clip_(min=0.0, max=1.0) # 将边界框坐标裁剪到[0, 1]范围内，确保坐标有效，维度：[bs, max_gt_num * 2 * num_group, 4]
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox) # 将边界框从左上右下格式(xyxy)转换回中心宽高格式(cxcywh)，维度：[bs, max_gt_num * 2 * num_group, 4]
        input_query_bbox = inverse_sigmoid(input_query_bbox) # 对边界框坐标进行逆sigmoid变换，将坐标从[0, 1]映射到(-inf, +inf)，维度：[bs, max_gt_num * 2 * num_group, 4]

    # class_embed = torch.concat([class_embed, torch.zeros([1, class_embed.shape[-1]], device=device)])
    # input_query_class = torch.gather(
    #     class_embed, input_query_class.flatten(),
    #     axis=0).reshape(bs, num_denoising, -1)
    # input_query_class = class_embed(input_query_class.flatten()).reshape(bs, num_denoising, -1)
    input_query_class = class_embed(input_query_class) # 使用类别嵌入函数将类别索引转换为类别嵌入向量，维度：[bs, max_gt_num * 2 * num_group] -> [bs, max_gt_num * 2 * num_group, embedding_dim]

    tgt_size = num_denoising + num_queries # 计算目标查询的总数量，即去噪查询数量加上正常查询数量
    # attn_mask = torch.ones([tgt_size, tgt_size], device=device) < 0
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device) # 创建注意力掩码张量，形状为[tgt_size, tgt_size]，初始化为False，表示所有查询都可以相互关注，维度：[tgt_size, tgt_size]
    # match query cannot see the reconstruction # 注释说明：匹配查询不能看到重建查询
    attn_mask[num_denoising:, :num_denoising] = True # 将注意力掩码的后半行（匹配查询）的前半列（去噪查询）设置为True，表示匹配查询不能关注去噪查询，维度：[tgt_size, tgt_size]
    
    # reconstruct cannot see each other # 注释说明：重建查询之间不能相互看到
    for i in range(num_group): # 遍历每个组
        if i == 0: # 如果是第一个组
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True # 设置第一个组不能看到后面的组，维度：[tgt_size, tgt_size]
        if i == num_group - 1: # 如果是最后一个组
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True # 设置最后一个组不能看到前面的组，维度：[tgt_size, tgt_size]
        else: # 如果是中间的组
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True # 设置当前组不能看到后面的组，维度：[tgt_size, tgt_size]
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True # 设置当前组不能看到前面的组，维度：[tgt_size, tgt_size]
    # 总结：通过上述循环设置注意力掩码，确保每个组内的去噪查询只能关注自己组内的查询，不能看到其他组的查询，从而实现组内对比学习的机制，维度：[tgt_size, tgt_size]
    # 匹配查询之间可以相互看到
    # 去噪查询之间根据组划分不能相互看到
    # 匹配查询不能看到去噪查询
    # 去噪查询不能看到匹配查询    
    
    dn_meta = { # 创建去噪元数据字典，存储去噪训练相关的元信息
        "dn_positive_idx": dn_positive_idx, # 存储正样本索引，用于后续计算损失
        "dn_num_group": num_group, # 存储组数，用于后续处理
        "dn_num_split": [num_denoising, num_queries] # 存储去噪查询和正常查询的数量分割，用于后续分离查询
    }

    # print(input_query_class.shape) # torch.Size([4, 196, 256])
    # print(input_query_bbox.shape) # torch.Size([4, 196, 4])
    # print(attn_mask.shape) # torch.Size([496, 496])
    
    return input_query_class, input_query_bbox, attn_mask, dn_meta # 返回去噪训练所需的四个元素：类别查询、边界框查询、注意力掩码和元数据

# 函数/类级总结：
# def get_contrastive_denoising_training_group: 该函数是RT-DETR中对比去噪训练的核心函数，负责生成用于去噪训练的查询组。它通过复制真实标注、添加标签噪声和边界框噪声，构建正负样本对，并生成相应的注意力掩码，实现对比学习机制。该函数接收真实目标数据、类别数、查询数、类别嵌入函数等参数，输出去噪查询的类别嵌入、边界框坐标、注意力掩码和元数据，为Transformer解码器提供训练所需的去噪输入。

# 整体功能总结：
# 该代码文件实现了RT-DETR（Real-Time DEtection TRansformer）中的对比去噪训练机制。主要功能包括：从真实标注数据中提取目标信息，通过复制和噪声注入生成去噪查询样本，构建正负样本对进行对比学习，生成注意力掩码控制查询间的可见性关系，最终输出用于Transformer解码器训练的去噪查询特征、边界框、注意力掩码和元数据。该机制通过引入噪声和对比学习，提升模型对目标检测任务的鲁棒性和准确性，使模型能够从有噪声的输入中恢复出正确的目标检测结果。