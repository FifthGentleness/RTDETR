---
name: "rtdetr-module-extractor"
description: "Extracts improvement modules from papers/PDFs and integrates them into RT-DETR. Invoke when user provides a paper or asks to add a new module/backbone to RT-DETR."
---

# RT-DETR Module Extractor

Extract core innovation modules from papers and integrate them into the RT-DETR project.

## Project Context

- **Codebase**: `e:\RT-DETR\rtdetr_pytorch\`
- **Architecture**: PaddlePaddle-to-PyTorch (RT-DETR)
- **Registration**: `@register` from `src.core.register` → auto-registers into `GLOBAL_CONFIG`
- **Config system**: YAML with `__include__` inheritance, loaded by `YAMLConfig`

## Key Paths

| Type | Path |
|------|------|
| Backbone | `src/nn/backbone/` (e.g. `presnet.py`, `presnet_msca.py`) |
| Modules | `src/nn/backbone/common.py` (`ConvNormLayer`, `FrozenBatchNorm2d`, `get_activation`) |
| Core | `src/core/yaml_utils.py` (`register`, `create`, `GLOBAL_CONFIG`) |
| Configs | `configs/rtdetr/` (e.g. `rtdetr_r18vd_6x_visdrone_msca.yml`) |
| Config includes | `configs/rtdetr/include/` (base configs like `rtdetr_r50vd.yml`) |
| Model zoo | `src/zoo/rtdetr/` (HybridEncoder, RTDETRTransformer) |

## Execution Pipeline

When user provides a paper (PDF) or asks to add a module, follow these steps **in order**:

### Step 1: Analyze Paper / Source

- Identify the **core innovation module** (attention mechanism, feature fusion, new block, etc.)
- Extract architecture details:
  - Input/output dimensions and tensor shapes
  - Layer structure (conv, attention, normalization, activation)
  - Hyperparameters (reduction ratio, num_heads, kernel_size, etc.)
  - Where in the network the module is inserted (backbone, encoder, decoder, neck)
- Clarify the **insertion point**: does it modify the backbone, replace a component, or add a new branch?

### Step 2: Implement Module

Create the PyTorch module following these conventions:

1. **File placement**:
   - Backbone variants → `src/nn/backbone/presnet_<variant>.py`
   - Standalone modules → `src/nn/backbone/<module_name>.py` (if used in backbone)
   - All backbone-related code goes under `src/nn/backbone/`

2. **Required imports**:
   ```python
   from src.core import register
   from .common import ConvNormLayer  # for conv+bn+act layers
   from .presnet import BasicBlock, PResNet, ResNet_cfg  # if extending PResNet
   ```

3. **Registration**: Every class that needs YAML config must use `@register`:
   ```python
   @register
   class MyNewModule(nn.Module):
       def __init__(self, dim, reduction=4, ...):
           ...
   ```

4. **`__all__` export**: List all public classes in `__all__`:
   ```python
   __all__ = ['MyNewModule', 'MyNewBlock', 'PResNet_MyVariant']
   ```

5. **Naming conventions**:
   - Backbone variants: `PResNet_<Variant>` (e.g. `PResNet_MSCA`)
   - Block variants: `<Module>BasicBlock` (e.g. `MSCABasicBlock`)
   - Attention modules: `<Name>Attention` or `<Name>Attn`
   - Use `ConvNormLayer` for all conv+bn+act layers (not raw `nn.Conv2d` + `nn.BatchNorm2d`)

6. **Weight compatibility**: New modules MUST support `strict=False` loading so pretrained weights can be partially loaded without errors.

### Step 3: Register in `__init__.py`

Add the new module's import to `src/nn/backbone/__init__.py`:
```python
from .presnet_<variant> import *
```

### Step 4: Create YAML Config

Create a new config file in `configs/rtdetr/`:

1. **Inherit base config** using `__include__`:
   ```yaml
   __include__: [
     '../dataset/<dataset>_detection.yml',
     '../runtime.yml',
     './include/dataloader.yml',
     './include/optimizer.yml',
     './include/rtdetr_r50vd.yml',  # base model config
   ]
   ```

2. **Override the backbone** and set module params:
   ```yaml
   output_dir: ./output/rtdetr_<variant>_<epochs>x_<dataset>

   RTDETR:
     backbone: PResNet_<Variant>

   PResNet_<Variant>:
     depth: 18
     variant: d
     return_idx: [1, 2, 3]
     freeze_at: -1
     freeze_norm: False
     pretrained: True
     # Add module-specific params here

   HybridEncoder:
     in_channels: [128, 256, 512]
     hidden_dim: 256
     expansion: 0.5

   RTDETRTransformer:
     eval_idx: -1
     num_decoder_layers: 3
     num_denoising: 100
   ```

3. **Naming convention**: `rtdetr_<variant>_<epochs>x_<dataset>.yml`

4. **Channel dimensions reference** (for depth=18, variant=d):
   - `return_idx=[1,2,3]` → `in_channels=[128, 256, 512]`
   - `return_idx=[2,3]` → `in_channels=[256, 512]`

### Step 5: Verify

Run these verification commands **in order**:

1. **Import test**:
   ```powershell
   python -c "from src.nn.backbone.presnet_<variant> import PResNet_<Variant>; print('Import OK')"
   ```
   Working directory: `e:\RT-DETR\rtdetr_pytorch`

2. **Registration test**:
   ```powershell
   python -c "from src.core import GLOBAL_CONFIG; from src.nn.backbone.presnet_<variant> import *; print('PResNet_<Variant>' in GLOBAL_CONFIG)"
   ```

3. **Forward pass test**:
   ```powershell
   python -c "import torch; from src.nn.backbone.presnet_<variant> import PResNet_<Variant>; m=PResNet_<Variant>(depth=18,variant='d',return_idx=[1,2,3]); x=torch.randn(1,3,640,640); outs=m(x); [print(o.shape) for o in outs]"
   ```

4. **Config validation**:
   ```powershell
   python -c "from src.core import YAMLConfig; cfg=YAMLConfig('configs/rtdetr/rtdetr_<variant>_<epochs>x_<dataset>.yml'); print('Config OK')"
   ```

## Reference Implementation

The best reference is `presnet_msca.py` — it shows the complete pattern:
- Extends `PResNet` with new blocks (`MSCABasicBlock`)
- Adds attention modules (`MSCSA`, `MSCA`, `CrossBranchInteraction`)
- Uses `@register` on `PResNet_MSCA`
- Has corresponding config `rtdetr_r18vd_6x_visdrone_msca.yml`

## Important Rules

- **NEVER modify existing registered classes** — only add new ones
- **NEVER change `__init__` signatures** of existing classes — breaks YAML config parsing
- **Always use `ConvNormLayer`** instead of raw conv+bn+act
- **Always add `@register`** to classes that need YAML config binding
- **Always update `__all__`** when adding new classes
- **Always support `strict=False`** for pretrained weight loading
- **Code comments in English** unless user specifies otherwise
---
name: "rtdetr-module-extractor"
description: "Extracts improvement modules from papers/PDFs and integrates them into RT-DETR. Invoke when user provides a paper or asks to add a new module/backbone to RT-DETR."
---

# RT-DETR Module Extractor

Extract core innovation modules from papers and integrate them into the RT-DETR project.

## Project Context

- **Codebase**: `e:\RT-DETR\rtdetr_pytorch\`
- **Architecture**: PaddlePaddle-to-PyTorch (RT-DETR)
- **Registration**: `@register` from `src.core.register` → auto-registers into `GLOBAL_CONFIG`
- **Config system**: YAML with `__include__` inheritance, loaded by `YAMLConfig`

## Key Paths

| Type | Path |
|------|------|
| Backbone | `src/nn/backbone/` (e.g. `presnet.py`, `presnet_msca.py`) |
| Modules | `src/nn/backbone/common.py` (`ConvNormLayer`, `FrozenBatchNorm2d`, `get_activation`) |
| Core | `src/core/yaml_utils.py` (`register`, `create`, `GLOBAL_CONFIG`) |
| Configs | `configs/rtdetr/` (e.g. `rtdetr_r18vd_6x_visdrone_msca.yml`) |
| Config includes | `configs/rtdetr/include/` (base configs like `rtdetr_r50vd.yml`) |
| Model zoo | `src/zoo/rtdetr/` (HybridEncoder, RTDETRTransformer) |

## Execution Pipeline

When user provides a paper (PDF) or asks to add a module, follow these steps **in order**:

### Step 1: Analyze Paper / Source

- Identify the **core innovation module** (attention mechanism, feature fusion, new block, etc.)
- Extract architecture details:
  - Input/output dimensions and tensor shapes
  - Layer structure (conv, attention, normalization, activation)
  - Hyperparameters (reduction ratio, num_heads, kernel_size, etc.)
  - Where in the network the module is inserted (backbone, encoder, decoder, neck)
- Clarify the **insertion point**: does it modify the backbone, replace a component, or add a new branch?

### Step 2: Implement Module

Create the PyTorch module following these conventions:

1. **File placement**:
   - Backbone variants → `src/nn/backbone/presnet_<variant>.py`
   - Standalone modules → `src/nn/backbone/<module_name>.py` (if used in backbone)
   - All backbone-related code goes under `src/nn/backbone/`

2. **Required imports**:
   ```python
   from src.core import register
   from .common import ConvNormLayer  # for conv+bn+act layers
   from .presnet import BasicBlock, PResNet, ResNet_cfg  # if extending PResNet
   ```

3. **Registration**: Every class that needs YAML config must use `@register`:
   ```python
   @register
   class MyNewModule(nn.Module):
       def __init__(self, dim, reduction=4, ...):
           ...
   ```

4. **`__all__` export**: List all public classes in `__all__`:
   ```python
   __all__ = ['MyNewModule', 'MyNewBlock', 'PResNet_MyVariant']
   ```

5. **Naming conventions**:
   - Backbone variants: `PResNet_<Variant>` (e.g. `PResNet_MSCA`)
   - Block variants: `<Module>BasicBlock` (e.g. `MSCABasicBlock`)
   - Attention modules: `<Name>Attention` or `<Name>Attn`
   - Use `ConvNormLayer` for all conv+bn+act layers (not raw `nn.Conv2d` + `nn.BatchNorm2d`)

6. **Weight compatibility**: New modules MUST support `strict=False` loading so pretrained weights can be partially loaded without errors.

### Step 3: Register in `__init__.py`

Add the new module's import to `src/nn/backbone/__init__.py`:
```python
from .presnet_<variant> import *
```

### Step 4: Create YAML Config

Create a new config file in `configs/rtdetr/`:

1. **Inherit base config** using `__include__`:
   ```yaml
   __include__: [
     '../dataset/<dataset>_detection.yml',
     '../runtime.yml',
     './include/dataloader.yml',
     './include/optimizer.yml',
     './include/rtdetr_r50vd.yml',  # base model config
   ]
   ```

2. **Override the backbone** and set module params:
   ```yaml
   output_dir: ./output/rtdetr_<variant>_<epochs>x_<dataset>

   RTDETR:
     backbone: PResNet_<Variant>

   PResNet_<Variant>:
     depth: 18
     variant: d
     return_idx: [1, 2, 3]
     freeze_at: -1
     freeze_norm: False
     pretrained: True
     # Add module-specific params here

   HybridEncoder:
     in_channels: [128, 256, 512]
     hidden_dim: 256
     expansion: 0.5

   RTDETRTransformer:
     eval_idx: -1
     num_decoder_layers: 3
     num_denoising: 100
   ```

3. **Naming convention**: `rtdetr_<variant>_<epochs>x_<dataset>.yml`

4. **Channel dimensions reference** (for depth=18, variant=d):
   - `return_idx=[1,2,3]` → `in_channels=[128, 256, 512]`
   - `return_idx=[2,3]` → `in_channels=[256, 512]`

### Step 5: Verify

Run these verification commands **in order**:

1. **Import test**:
   ```powershell
   python -c "from src.nn.backbone.presnet_<variant> import PResNet_<Variant>; print('Import OK')"
   ```
   Working directory: `e:\RT-DETR\rtdetr_pytorch`

2. **Registration test**:
   ```powershell
   python -c "from src.core import GLOBAL_CONFIG; from src.nn.backbone.presnet_<variant> import *; print('PResNet_<Variant>' in GLOBAL_CONFIG)"
   ```

3. **Forward pass test**:
   ```powershell
   python -c "import torch; from src.nn.backbone.presnet_<variant> import PResNet_<Variant>; m=PResNet_<Variant>(depth=18,variant='d',return_idx=[1,2,3]); x=torch.randn(1,3,640,640); outs=m(x); [print(o.shape) for o in outs]"
   ```

4. **Config validation**:
   ```powershell
   python -c "from src.core import YAMLConfig; cfg=YAMLConfig('configs/rtdetr/rtdetr_<variant>_<epochs>x_<dataset>.yml'); print('Config OK')"
   ```

## Reference Implementation

The best reference is `presnet_msca.py` — it shows the complete pattern:
- Extends `PResNet` with new blocks (`MSCABasicBlock`)
- Adds attention modules (`MSCSA`, `MSCA`, `CrossBranchInteraction`)
- Uses `@register` on `PResNet_MSCA`
- Has corresponding config `rtdetr_r18vd_6x_visdrone_msca.yml`

## Important Rules

- **NEVER modify existing registered classes** — only add new ones
- **NEVER change `__init__` signatures** of existing classes — breaks YAML config parsing
- **Always use `ConvNormLayer`** instead of raw conv+bn+act
- **Always add `@register`** to classes that need YAML config binding
- **Always update `__all__`** when adding new classes
- **Always support `strict=False`** for pretrained weight loading
- **Code comments in English** unless user specifies otherwise