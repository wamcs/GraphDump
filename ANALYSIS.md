# GraphDump 全模型 GE 图算子分析报告

## 一、有优化前后对比的模型（3 组）

### 1. ResNet50 — Conv-BN 融合 + fp16

优化方式: `fuse_conv_bn()` 把 53 层 BatchNorm 的参数数学折叠进前置 Conv，
加上 fp16 消除中间精度管理。

| 算子 | 原始 | 优化后 | 变化 | 原因 |
|---|---|---|---|---|
| **BNInfer** | 53 | 0 | **全消** | 推理时 BN 等价仿射变换 → W'=W·γ/σ(σ²+ε), b'=(b-μ)·γ/σ(σ²+ε)+β，预计算后 BN 不再需要 |
| **Cast** | 54 | 1 | -53 | fp32 需中间精度管理 Cast → fp16 全程一致后不需要 |
| **Shape/Fill** | 53/53 | 0/0 | **全消** | 为 BN 中间张量计算 shape 的辅助算子，随 BN 一起消失 |
| **Const** | 111 | 5 | -106 | 53 组 BN 的 mean/var/γ/β 常量被折叠进 Conv 权重 |
| **Data** | 268 | 109 | -159 | BN 参数不再作为独立输入占位 |
| Conv2D | 53 | 53 | 不变 | 数量不变，但每个 Conv 的 weight/bias 已含 BN 折叠值 |
| Relu | 49 | 49 | 不变 | 不受影响 |
| **NPU 总计** | **804** | **327** | **-59%** | |

**本质**: 代数恒等式 `BN(Conv(x)) ≡ Conv'(x)`，消除了推理时不必要的独立归一化步骤。

---

### 2. Wav2Vec2 — QKV 融合 + weight_norm 移除

优化方式: 每层 attention 的 3 个独立 Q/K/V Linear 合并为 1 个融合 QKV
Linear + chunk，移除 weight_norm 参数化。

| 算子 | 原始 | 优化后 | 变化 | 原因 |
|---|---|---|---|---|
| **MatMulV2** | 73 | 49 | -24 | 12 层 × 3 个 Linear → 12 个融合 Linear，少 12×2=24 次 matmul |
| **Transpose** | 124 | 52 | -72 | 原始对 Q/K/V 各做 BSHD↔BHSD 变换，融合后单次 reshape+split 即可 |
| **Reshape** | 212 | 104 | -108 | 同上，三个独立输出的逐一 reshape 消除 |
| Const | 543 | 363 | -180 | Q/K/V 权重从 3 个独立常量合并为 1 个 |
| Data | 213 | 164 | -49 | 同上 |
| **SplitV** | 0 | 12 | **+12** | 新增: 融合 QKV 输出需切回 Q/K/V |
| **Pack** | 0 | 12 | **+12** | 新增: attention 输出拼接 |
| FlashAttentionScore | 12 | 12 | 不变 | 注意力计算本身未变 |
| **NPU 总计** | **654** | **494** | **-24%** | |

**本质**: `W_q@x, W_k@x, W_v@x ≡ [W_q;W_k;W_v]@x` 再 split。Transpose/Reshape
的大量消除才是真正收益（原本为三个独立 Linear 各做一次 layout 变换）。

---

### 3. ESM2-650M — QKV 融合 + npu_gelu + npu_fusion_attention + npu_rotary

优化方式: 项目 `optimize.py` 的 R3_OPTS 全开——fuse_qkv + replace_gelu
+ npu_attention + npu_rotary + bypass_dropout。

| 算子 | 原始 | 优化后 | 变化 | 原因 |
|---|---|---|---|---|
| **MatMulV2** | 199 | 133 | -66 | 33 层 × 3 Linear → 33 融合（与 wav2vec2 同理） |
| **Transpose** | 333 | 267 | -66 | QKV layout 变换减少 |
| **Mul** | 436 | 137 | **-299** | **最大变化**: npu_rotary_mul 替换手动 rotary 计算（手动实现是 mul+cat+neg 的链） |
| **Add** | 368 | 200 | -168 | rotary 计算中的 add 链消除 + dropout 跳过后的 add 消除 |
| **Cast** | 249 | 43 | -206 | npu 原生算子不需要精度管理 Cast |
| **Erf** | 34 | 0 | **全消** | gelu 的 erf 核心被 `GeluV2`（GE 原生融合算子）替代 |
| **Neg** | 66 | 0 | **全消** | rotate_half 的 -x2 被 `RotaryMul` 替代 |
| **RealDiv** | 36 | 0 | **全消** | rotary 中的除法运算 |
| **StridedSliceV2** | 132 | 0 | **全消** | 手动 rotary 的切片操作被 `RotaryMul` 整体替代 |
| **LogicalNot** | 33 | 0 | **全消** | attention mask 的逻辑非运算消除 |
| **SplitV** | 0 | 33 | +33 | 新增: QKV 切分 |
| **Pack** | 0 | 33 | +33 | 新增: attention 输出拼接 |
| **RotaryMul** | 0 | 66 | **+66** | 新增: NPU 原生 rotary 融合算子 |
| **GeluV2** | 0 | 34 | **+34** | 新增: GE 原生 Gelu 融合算子 |
| **FlashAttentionScore** | 0 | 33 | **+33** | 新增: NPU 融合 attention（替换手动 sdpa） |
| LayerNormV4 | 68 | 68 | 不变 | 不受优化影响 |
| **NPU 总计** | **1627** | **1230** | **-24%** | |

**本质**: 多种优化叠加——QKV 融合减少矩阵乘和 layout 变换；`npu_rotary_mul`
把 10+ 个手动算子（mul/cat/neg/div/slice）折叠为 1 个融合算子；`npu_gelu`
消除了 erf+mul+add 链；`npu_fusion_attention` 替换整个 sdpa 展开。

---

## 二、单版本模型（无优化对比）

### 4. CLIP ViT-B/32（多模态双编码器）

| 算子 | 数量 | 说明 |
|---|---|---|
| NPU | 917 | 总计算算子（text_encoder + vision_encoder 两部分） |
| Reshape/Transpose | 229/232 | layout 变换占比高——ViT 的 patch embedding + attention 需要大量 reshape |
| Cast | 232 | fp16 混合精度管理 |
| MatMulV2 | 96 | 12 层 text (768) + 12 层 vision (1024) |
| Conv2D | 未统计（子图3内） | ViT 的 patch embedding 卷积 |
| GroupNorm | 少量 | ViT 的 pre-l norm 层 |

### 5. Stable Diffusion UNet（860M 扩散模型）

| 算子 | 数量 | 说明 |
|---|---|---|
| NPU | 1861 | 跨 54 个子图的总计算算子 |
| Reshape | 568 | ResBlock 的 up/downsample + attention 的 layout 变换 |
| Conv2D | 62 | 下采样/上采样/skip 连接卷积 |
| MatMulV2 | 56 | cross attention + self attention |
| Swish | 55 | GE 已把 silu 识别为 Swish 融合算子 |
| GroupNorm | 40 | UNet 特有的分组归一化 |
| LayerNormV4 | 48 | 部分 attention 后的 layer norm |

### 6. ProteinMPNN（图神经网络）

| 算子 | 数量 | 说明 |
|---|---|---|
| aten.slice | 335 | 消息传递中的边聚合 |
| aten.unsqueeze | 174 | 图结构维度扩展 |
| aten.view | 143 | shape 变换 |
| aten.sub/pow | 55/50 | 距离计算 + 归一化 |
| **GE** | ✗ | 被 `linalg_cross`（叉积）缺 converter 挡住 |

### 7. DINOv2（视觉 backbone）

FX 和 GE 均被 torch_npu 2.6 的 fake bicubic interpolate 限制挡住
（`NPU contiguous operator only supported contiguous memory format`），
只有 dispatch ground truth 可达:

| 算子 | 数量 |
|---|---|
| aten ops 总计 | 512 / 17 unique |
| aten.mul | 82 |
| aten.mm | 80 |
| aten.slice | 62 |
| aten.view | 53 |

---

## 三、总结

### 共性发现

1. **优化的本质是代数恒等式**——不管是 Conv-BN 折叠、QKV 融合还是 NPU 融合
   算子替换，都是在不改数学语义的前提下消除冗余计算步骤

2. **GE 图的变化集中在三类算子上**:
   - **被消除的**: BNInfer/Shape/Fill（BN 折叠）、Cast/Transpose/Reshape
     （fp16 统一 + QKV 融合）、Erf/Neg/RealDiv/StridedSlice（融合算子替换）
   - **新增的**: SplitV/Pack（QKV 切分代价）、RotaryMul/GeluV2/
     FlashAttentionScore（NPU 融合算子）
   - **不变的**: LayerNormV4、Conv2D、Relu（不受优化影响的基础层）

3. **优化效果与模型复杂度正相关**:
   - ResNet50（纯 CNN）: **-59%** NPU 算子
   - Wav2Vec2（Conv+Transformer）: **-24%**
   - ESM2（纯 Transformer 33 层）: **-24%**（绝对减量最大: -397）

4. **GE 图是优化验证的量化证据**——dispatch 的 ops_summary 提供 aten 级
   频次对比，GE 图提供 NPU 算子级对比，两层对拍构成完整的优化效果度量
