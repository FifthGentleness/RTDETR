# RT-DETR PyTorch 代码导航地图

快速定位你需要修改或学习的代码位置。

## 📍 快速定位表

### 核心模型文件

| 功能 | 文件位置 | 关键类/函数 | 说明 |
|-----|---------|-----------|------|
| **主模型入口** | `src/zoo/rtdetr/rtdetr.py` | `class RTDETR` | 完整模型定义，包含forward() |
| **骨干网络** | `src/nn/backbone/presnet.py` | `class PResNet` | ResNet18/34/50/101 backbone |
| **混合编码器** | `src/zoo/rtdetr/hybrid_encoder.py` | `class HybridEncoder` | transformer intra + CNN cross scale|
| **Transformer解码器** | `src/zoo/rtdetr/rtdetr_decoder.py` | `class RTDETRTransformer` | 6层可变形Transformer |
| **可变形注意** | `src/zoo/rtdetr/rtdetr_decoder.py` | `MSDeformableAttention` | 参考框周围采样，O(n)复杂度 |
| **二部图匹配** | `src/zoo/rtdetr/matcher.py` | `class HungarianMatcher` | 300查询→目标分配，匈牙利算法 |
| **损失函数** | `src/zoo/rtdetr/rtdetr_criterion.py` | `class SetCriterion` | focal + L1 + GIoU 损失 |
| **去噪查询** | `src/zoo/rtdetr/rtdetr_criterion.py` | `get_contrastive_denoising_training_group()` | 生成200个噪声框 |
| **后处理** | `src/zoo/rtdetr/rtdetr_postprocessor.py` | `class RTDETRPostProcessor` | 框格式转换、尺寸缩放、NMS |

### 数据处理

| 功能 | 文件位置 | 关键类/函数 | 说明 |
|-----|---------|-----------|------|
| **COCO数据集** | `src/data/coco/coco_dataset.py` | `class CocoDetection` | COCO标注加载 |
| **数据增强** | `src/data/transforms.py` | 多个Transform | RandomPhotometricDistort, IOU Crop等 |
| **框格式转换** | `src/data/transforms.py` | `ConvertBox` | XYXY→CXCYWH，规范化 |
| **数据加载器** | `src/data/dataloader.py` | 多个函数 | 构建train/val dataloader |

### 配置系统

| 功能 | 文件位置 | 关键类/函数 | 说明 |
|-----|---------|-----------|------|
| **YAML配置** | `src/core/yaml_config.py` | `class YAMLConfig` | 递归加载include，参数合并 |
| **基础配置** | `src/core/config.py` | `class BaseConfig` | 配置注册、动态创建 |
| **YAML工具** | `src/core/yaml_utils.py` | 多个函数 | 配置文件操作 |

### 训练和推理

| 功能 | 文件位置 | 关键类/函数 | 说明 |
|-----|---------|-----------|------|
| **主训练脚本** | `tools/train.py` | `main()` | 分布式初始化、求解器创建 |
| **求解器** | `src/solver/det_solver.py` | `class DetSolver` | 训练策略、学习率调度 |
| **训练循环** | `src/solver/det_engine.py` | `train_one_epoch()` | 单个epoch训练逻辑 |
| **评估** | `src/solver/det_engine.py` | `evaluate()` | COCO评估 |
| **推理脚本** | `tools/infer.py` | `main()` | 图像推理、结果可视化 |
| **导出脚本** | `tools/export_onnx.py` | - | 模型转ONNX格式 |

### 模型部署相关

| 功能 | 文件位置 | 关键类/函数 | 说明 |
|-----|---------|-----------|------|
| **RepVgg块** | `src/zoo/rtdetr/hybrid_encoder.py` | `class RepVggBlock` | 训练并联、推理融合 |
| **部署转换** | 各模块 | `convert_to_deploy()` | 去掉Dropout、BN融合 |
| **框转换工具** | `src/zoo/rtdetr/box_ops.py` | `box_convert()` 等 | cxcywh↔xyxy转换 |

---

## 🔍 按场景快速找代码

### 进阶学习路线

#### 1️⃣ 学习模型架构 (1-2天)
```
第一步: src/zoo/rtdetr/rtdetr.py (10min)
  ↓ 了解RTDETR的整体结构

第二步: src/nn/backbone/presnet.py → 理解特征提取 (30min)
第三步: src/zoo/rtdetr/hybrid_encoder.py → 理解多尺度融合 (1h)
第四步: src/zoo/rtdetr/rtdetr_decoder.py → 理解可变形注意 (2h)
```

#### 2️⃣ 学习数据处理 (2-3h)
```
src/data/coco/coco_dataset.py → COCO加载
       ↓
src/data/transforms.py → 增强策略
       ↓
src/data/dataloader.py → 批处理
```

#### 3️⃣ 学习训练流程 (2-3h)
```
tools/train.py (入口)
      ↓
src/core/yaml_config.py (配置加载) 
      ↓
src/solver/det_solver.py (训练管理)
      ↓
src/solver/det_engine.py (训练循环 + 评估)
```

#### 4️⃣ 学习推理流程 (1-2h)
```
tools/infer.py (图像推理)
      ↓
src/zoo/rtdetr/rtdetr_postprocessor.py (后处理)
      ↓
tools/export_onnx.py (模型导出)
```

---

## 💡 常见修改点

### 🔧 想修改什么？这里找

| 需求 | 修改位置 | 方法 |
|-----|---------|------|
| **增加新的backbone** | `src/nn/backbone/` 新建文件 | 继承 `nn.Module`, 在 `presnet.py` 里register |
| **改变encoder结构** | `src/zoo/rtdetr/hybrid_encoder.py` | 修改 `HybridEncoder.__init__()` 和 `forward()` |
| **改变decoder层数** | 配置文件或 `rtdetr_decoder.py` | num_decoder_layers参数 |
| **改变查询数** | 配置文件或 `rtdetr_decoder.py` | num_queries = 300 |
| **改变损失权重** | `src/zoo/rtdetr/rtdetr_criterion.py` | loss_dict中的系数 |
| **改变训练超参** | `configs/include/optimizer.yml` | 学习率、batch size等 |
| **增加数据增强** | `src/data/transforms.py` | 新增Transform类 |
| **改变去噪策略** | `src/zoo/rtdetr/rtdetr_criterion.py` | `get_contrastive_denoising_training_group()` |
| **调整推理后处理** | `src/zoo/rtdetr/rtdetr_postprocessor.py` | TopK选择、NMS阈值 |

---

## 📊 关键参数一览

### 模型超参数
```python
# src/zoo/rtdetr/rtdetr_decoder.py 或 config
num_queries = 300           # 检测查询数
num_decoder_layers = 6      # 解码层数
num_denoising = 100         # 去噪查询数
label_noise_ratio = 0.5     # 标签噪声比例
box_noise_scale = 1.0       # 框噪声规模
hidden_dim = 256            # 特征维度
```

### 训练超参数
```python
# configs/include/optimizer.yml
lr_backbone = 1e-4
lr_decoder = 2e-3
batch_size = 8              # 单GPU, 总batch = 64 (8cards)
num_epochs = 6              # 约COCO 36 epochs
```

### 数据增强参数
```python
# src/data/transforms.py
multi_scale_range = [480, 800]     # 多尺度颗粒度
IoU_crop_min = 0.1                 # 最小裁剪IoU
```

### 后处理参数
```python
# rtdetr_postprocessor.py
select_topk = 300           # TopK个数
nms_threshold = 0.5         # NMS阈值 (可选)
```

---

## 🎯 与其他项目的对应关系

### PyTorch vs Paddle版
| 功能 | PyTorch路径 | Paddle对应路径 |
|-----|-----------|---------------|
| 模型定义 | `src/zoo/rtdetr/` | `rtdetr_paddle/ppdet/modeling/` |
| 数据处理 | `src/data/` | `rtdetr_paddle/ppdet/data/` |
| 训练引擎 | `src/solver/` | `rtdetr_paddle/ppdet/engine/` |

### 重要差异
- **配置系统**: PyTorch使用自定义YAML系统，Paddle用PaddleDetection的配置
- **算子**: 某些专有操作(如RepVgg融合)在两版本略有差异
- **分布式**: PyTorch用torch.distributed, Paddle用paddle.distributed

---

## ✅ 编码约定

### 类命名
- 模型：`RT...` 前缀, 如 `RTDETRTransformer`
- 操作：`...Op` 后缀, 如 `MSDeformableAttention`
- 配置处理：基类 + 具体实现

### 目录结构约定
```
src/
├── nn/           # 神经网络基础模块
├── zoo/rtdetr/   # RT-DETR特定实现
├── solver/       # 训练和求解器
├── data/         # 数据加载和处理
├── optim/        # 优化器相关
└── misc/         # 工具函数
```

### 函数签名约定
```python
# 前向函数
def forward(self, x, targets=None, **kwargs)

# 训练/评估
train_one_epoch(model, criterion, data_loader, ...)
evaluate(model, criterion, postprocessor, data_loader, ...)

# 损失计算
def forward(self, outputs, targets)  # outputs, targets必须
```

---

## 🐛 调试文件快速访问

| 需求 | 查看 |
|-----|------|
| 看模型参数数量 | `tools/train.py` → `model` 定义后print stats |
| 看dataloader输出 | `src/data/dataloader.py` → 最后的demo代码 |
| 看模型前向过程 | 各模块 `forward()` 打debug print |
| 看损失变化 | `src/solver/det_engine.py` → `metric_logger` |
| 看推理输出 | `tools/infer.py` → 输出可视化 |
| 看配置是否加载 | `src/core/yaml_config.py` → 添加print |

