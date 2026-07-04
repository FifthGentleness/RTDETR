# RT-DETR configs 目录深度解读

本文档面向已经拿到完整项目代码的研发同学，目标是彻底讲清楚 configs 目录中每个文件的作用、配置继承关系、关键参数含义，以及如何安全修改。

---

## 1. 先看全局：configs 在项目里的定位

configs 是整个训练与推理流程的“声明式控制中心”。
你不需要改大量 Python 代码，就可以通过 YAML 组合出不同实验：

- 数据来源与数据增强
- 训练超参数
- 模型结构（backbone / encoder / decoder）
- 损失函数与匹配策略
- 运行时开关（AMP、EMA、SyncBN 等）

配置加载入口在 [src/core/yaml_config.py](../src/core/yaml_config.py) 与 [src/core/yaml_utils.py](../src/core/yaml_utils.py)。

---

## 2. configs 目录结构

- [configs/runtime.yml](runtime.yml)
- [configs/dataset/coco_detection.yml](dataset/coco_detection.yml)
- [configs/rtdetr/rtdetr_r18vd_6x_coco.yml](rtdetr/rtdetr_r18vd_6x_coco.yml)
- [configs/rtdetr/rtdetr_r34vd_6x_coco.yml](rtdetr/rtdetr_r34vd_6x_coco.yml)
- [configs/rtdetr/rtdetr_r50vd_6x_coco.yml](rtdetr/rtdetr_r50vd_6x_coco.yml)
- [configs/rtdetr/rtdetr_r50vd_m_6x_coco.yml](rtdetr/rtdetr_r50vd_m_6x_coco.yml)
- [configs/rtdetr/rtdetr_r101vd_6x_coco.yml](rtdetr/rtdetr_r101vd_6x_coco.yml)
- [configs/rtdetr/rtdetr_dla34_6x_coco.yml](rtdetr/rtdetr_dla34_6x_coco.yml)
- [configs/rtdetr/rtdetr_regnet_6x_coco.yml](rtdetr/rtdetr_regnet_6x_coco.yml)
- [configs/rtdetr/include/dataloader.yml](rtdetr/include/dataloader.yml)
- [configs/rtdetr/include/dataloader_regnet.yml](rtdetr/include/dataloader_regnet.yml)
- [configs/rtdetr/include/optimizer.yml](rtdetr/include/optimizer.yml)
- [configs/rtdetr/include/optimizer_regnet.yml](rtdetr/include/optimizer_regnet.yml)
- [configs/rtdetr/include/rtdetr_r50vd.yml](rtdetr/include/rtdetr_r50vd.yml)
- [configs/rtdetr/include/rtdetr_dla34.yml](rtdetr/include/rtdetr_dla34.yml)
- [configs/rtdetr/include/rtdetr_regnet.yml](rtdetr/include/rtdetr_regnet.yml)

---

## 3. 配置继承机制（最关键）

每个顶层实验文件通常用 __include__ 组合多个子配置。

示例：

- [configs/rtdetr/rtdetr_r50vd_6x_coco.yml](rtdetr/rtdetr_r50vd_6x_coco.yml)
  - 包含 [configs/dataset/coco_detection.yml](dataset/coco_detection.yml)
  - 包含 [configs/runtime.yml](runtime.yml)
  - 包含 [configs/rtdetr/include/dataloader.yml](rtdetr/include/dataloader.yml)
  - 包含 [configs/rtdetr/include/optimizer.yml](rtdetr/include/optimizer.yml)
  - 包含 [configs/rtdetr/include/rtdetr_r50vd.yml](rtdetr/include/rtdetr_r50vd.yml)

合并逻辑可以理解为：

1. 先加载 include 列表中的文件
2. 再应用当前文件中的字段覆盖
3. 同名字段后定义优先

这意味着顶层实验文件往往很短，但能决定最终实验的“组合结果”。

---

## 4. 分层理解每个文件

## 4.1 运行时基础层

### 4.1.1 runtime.yml
文件： [configs/runtime.yml](runtime.yml)

作用：定义训练运行机制开关，和具体模型结构无关。

核心字段：

- sync_bn: 是否开启同步 BN（多卡常用）
- find_unused_parameters: DDP 是否查找未使用参数
- use_amp: 是否启用混合精度
- scaler: GradScaler 配置
- use_ema: 是否启用 EMA
- ema: EMA 衰减率与 warmup

典型使用场景：

- 显存紧张、希望提速：开启 AMP
- 想要更稳定最终精度：开启 EMA
- 多卡训练遇到梯度同步问题：检查 find_unused_parameters

---

## 4.2 数据基础层

### 4.2.1 coco_detection.yml
文件： [configs/dataset/coco_detection.yml](dataset/coco_detection.yml)

作用：定义检测任务基础数据集与 dataloader 骨架。

核心字段：

- task: detection
- num_classes: 80
- remap_mscoco_category: 类别映射开关
- train_dataloader / val_dataloader:
  - dataset type: CocoDetection
  - img_folder / ann_file
  - transforms 框架（具体 ops 在 include/dataloader*.yml 里补齐）
  - batch_size / num_workers / shuffle / drop_last

典型使用场景：

- 更换数据集路径
- 更改类别数
- 调整 train/val 基础 batch 参数

---

## 4.3 数据增强与加载模板层

### 4.3.1 include/dataloader.yml
文件： [configs/rtdetr/include/dataloader.yml](rtdetr/include/dataloader.yml)

作用：标准 RT-DETR 数据增强模板（适用于大多数 ResNet/DLA 配置）。

训练增强链路（简化理解）：

- 光照扰动
- ZoomOut
- IoU 裁剪
- 无效框清理
- 随机翻转
- Resize 到 640x640
- ToImageTensor
- ConvertDtype
- ConvertBox 到 cxcywh 并归一化

这个文件还定义了：

- train/val 的 batch_size
- num_workers
- collate_fn

### 4.3.2 include/dataloader_regnet.yml
文件： [configs/rtdetr/include/dataloader_regnet.yml](rtdetr/include/dataloader_regnet.yml)

作用：RegNet 版本的数据加载模板。

与标准模板主要差异：

- batch_size 与 workers 设置不同（更偏 RegNet 配置）
- 数据增强主流程几乎一致

---

## 4.4 优化器与训练调度模板层

### 4.4.1 include/optimizer.yml
文件： [configs/rtdetr/include/optimizer.yml](rtdetr/include/optimizer.yml)

作用：标准训练超参数模板。

核心字段：

- use_ema / ema
- find_unused_parameters
- epoches: 72
- clip_max_norm: 0.1
- optimizer: AdamW
  - params: 用正则表达式做参数分组
  - backbone 单独学习率（更小）
  - encoder / decoder 中 norm/bias 不做 weight decay
- lr_scheduler: MultiStepLR

为什么参数分组重要：

- backbone 往往用更小 lr，避免破坏预训练特征
- norm/bias 去衰减有利于稳定训练

### 4.4.2 include/optimizer_regnet.yml
文件： [configs/rtdetr/include/optimizer_regnet.yml](rtdetr/include/optimizer_regnet.yml)

作用：RegNet 的优化器模板。

相对 optimizer.yml 的差异点：

- 参数分组规则略不同
- 适配 RegNet 主干命名结构

---

## 4.5 模型结构模板层

### 4.5.1 include/rtdetr_r50vd.yml
文件： [configs/rtdetr/include/rtdetr_r50vd.yml](rtdetr/include/rtdetr_r50vd.yml)

作用：最核心的 RT-DETR 基线结构模板。

定义内容包括：

- 顶层组件
  - model: RTDETR
  - criterion: SetCriterion
  - postprocessor: RTDETRPostProcessor
- Backbone
  - PResNet depth=50, variant=d
- Encoder
  - HybridEncoder 输入通道、隐藏维度、层数
- Decoder
  - RTDETRTransformer：query 数、decoder 层数、denoising 数
- Loss 与匹配
  - SetCriterion 权重（vfl、bbox、giou）
  - HungarianMatcher 成本权重

这是其他大部分变体的母版。

### 4.5.2 include/rtdetr_dla34.yml
文件： [configs/rtdetr/include/rtdetr_dla34.yml](rtdetr/include/rtdetr_dla34.yml)

作用：DLA34 主干版本的模型模板。

关键变化：

- backbone 从 PResNet 换成 DLANet
- encoder 的 in_channels 改为 DLA 输出通道
- 其他 RT-DETR 头部、损失逻辑大体保持

### 4.5.3 include/rtdetr_regnet.yml
文件： [configs/rtdetr/include/rtdetr_regnet.yml](rtdetr/include/rtdetr_regnet.yml)

作用：RegNet 主干版本的模型模板。

关键变化：

- backbone 改为 RegNet
- encoder in_channels 适配 RegNet 特征维度

---

## 4.6 顶层实验配置层（可直接传给 train.py）

下面这些文件都是“可执行实验入口”，负责把模板拼起来并做少量覆盖。

### 4.6.1 rtdetr_r50vd_6x_coco.yml
文件： [configs/rtdetr/rtdetr_r50vd_6x_coco.yml](rtdetr/rtdetr_r50vd_6x_coco.yml)

作用：R50vd 标准基线实验。

特点：

- 主要是 include 组合
- 给定 output_dir
- 几乎不改单项参数

### 4.6.2 rtdetr_r101vd_6x_coco.yml
文件： [configs/rtdetr/rtdetr_r101vd_6x_coco.yml](rtdetr/rtdetr_r101vd_6x_coco.yml)

作用：更大 backbone 的高精度版本。

相对 R50 基线的关键覆盖：

- PResNet depth 改为 101
- HybridEncoder hidden_dim 与 dim_feedforward 变大
- RTDETRTransformer feat_channels 跟随变大
- optimizer 里 backbone lr 更小

### 4.6.3 rtdetr_r18vd_6x_coco.yml
文件： [configs/rtdetr/rtdetr_r18vd_6x_coco.yml](rtdetr/rtdetr_r18vd_6x_coco.yml)

作用：轻量速度优先版本。

关键覆盖：

- backbone 改为 18 层
- freeze 行为调整
- HybridEncoder in_channels 与 expansion 调整
- decoder 层数减少为 3
- 优化器参数分组更细（针对 backbone norm/non-norm）

### 4.6.4 rtdetr_r34vd_6x_coco.yml
文件： [configs/rtdetr/rtdetr_r34vd_6x_coco.yml](rtdetr/rtdetr_r34vd_6x_coco.yml)

作用：中等规模折中版本。

关键覆盖：

- backbone 改为 34 层
- encoder 通道适配
- decoder 层数调整为 4
- 对应 optimizer 分组调整

### 4.6.5 rtdetr_r50vd_m_6x_coco.yml
文件： [configs/rtdetr/rtdetr_r50vd_m_6x_coco.yml](rtdetr/rtdetr_r50vd_m_6x_coco.yml)

作用：R50 的速度优化变体。

关键覆盖：

- HybridEncoder expansion 下调
- RTDETRTransformer eval_idx 调整为中间层（第 3 层）输出用于 eval

这个配置常用于“更低时延”场景。

### 4.6.6 rtdetr_dla34_6x_coco.yml
文件： [configs/rtdetr/rtdetr_dla34_6x_coco.yml](rtdetr/rtdetr_dla34_6x_coco.yml)

作用：DLA34 backbone 的完整实验入口。

特点：

- include 到 rtdetr_dla34 模板
- 数据与优化器沿用标准模板

### 4.6.7 rtdetr_regnet_6x_coco.yml
文件： [configs/rtdetr/rtdetr_regnet_6x_coco.yml](rtdetr/rtdetr_regnet_6x_coco.yml)

作用：RegNet backbone 的完整实验入口。

特点：

- include 使用 regnet 专用 dataloader + optimizer + model 模板
- output_dir 独立

---

## 5. 这些配置如何影响训练代码

可以从 [tools/train.py](../tools/train.py) 开始，核心流程是：

1. 读取你指定的顶层 yaml
2. 按 include 合并成最终配置
3. 构建 model / criterion / optimizer / dataloader
4. 进入 [src/solver/det_solver.py](../src/solver/det_solver.py) 与 [src/solver/det_engine.py](../src/solver/det_engine.py) 训练循环

因此，configs 的任何字段变化，都会直接影响最终训练行为。

---

## 6. 常见修改需求对应文件

- 改数据路径、类别数
  - [configs/dataset/coco_detection.yml](dataset/coco_detection.yml)

- 改数据增强策略
  - [configs/rtdetr/include/dataloader.yml](rtdetr/include/dataloader.yml)
  - [configs/rtdetr/include/dataloader_regnet.yml](rtdetr/include/dataloader_regnet.yml)

- 改优化器、学习率、训练轮数
  - [configs/rtdetr/include/optimizer.yml](rtdetr/include/optimizer.yml)
  - [configs/rtdetr/include/optimizer_regnet.yml](rtdetr/include/optimizer_regnet.yml)

- 改 backbone/encoder/decoder 结构
  - [configs/rtdetr/include/rtdetr_r50vd.yml](rtdetr/include/rtdetr_r50vd.yml)
  - [configs/rtdetr/include/rtdetr_dla34.yml](rtdetr/include/rtdetr_dla34.yml)
  - [configs/rtdetr/include/rtdetr_regnet.yml](rtdetr/include/rtdetr_regnet.yml)

- 建立新的实验组合
  - 复制某个顶层文件，如 [configs/rtdetr/rtdetr_r50vd_6x_coco.yml](rtdetr/rtdetr_r50vd_6x_coco.yml)
  - 修改 include 组合与 output_dir

---

## 7. 实战建议：如何新增一个自己的配置

建议步骤：

1. 复制一个最接近的顶层实验文件
2. 只修改你关心的字段（少量覆盖）
3. 保持 include 模板不动，先跑通
4. 再逐步调整 dataloader / optimizer / model 模板

最小风险原则：

- 一次只改一组变量
- output_dir 单独命名，避免覆盖旧实验
- 记录每次改动与结果指标

---

## 8. 关键理解总结

configs 的设计思想是“模板复用 + 顶层组装 + 局部覆盖”：

- runtime 与 dataset 提供公共底座
- include 提供可复用的模型/数据/优化策略模板
- 顶层实验文件负责拼装与差异化

这套体系让你能快速做消融实验，同时维持较强可维护性。

如果你要做自己的版本，通常只要新增 1 个顶层配置文件，必要时再新增 1 个 include 模板即可，不需要改核心训练代码。
