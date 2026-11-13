# GraphDump — 通用计算图 dump 工具

模型无关 / 设备无关（CPU / CUDA / NPU）/ **不要求模型能图编译**，一个入口
`dump()` 获取模型的各类计算图（含 NPU 算子视角的 GE 图），全程不做图编译。

## 一、能获取哪些图

```
Python 代码
   │ dynamo trace
   ▼
torch 层 FX 图           ← 【FX 图】F.linear / F.softmax 原样
   │ AOT 分解（可选, 默认关）
   ▼
core-aten 图             ← aot=True 可查看（已被 GE 直接捕获取代）
   │ torchair FX→GE converter
   ▼
GE 图（NPU 算子视角）    ← 【GE 图】Conv2D / SoftmaxV2 / Swish …
   │ GE 编译 + 内核选择（本工具不走到这一步）
   ▼
aclnn 内核
```

**两类真实图**——工具的产物就是它们的对比:

| 图 | 内容 | 典型用途 | 数值一致性 |
|---|---|---|---|
| FX 图 | torch 层计算图本体 | 模型结构分析、优化机会清单 | **逐位一致**（透传执行） |
| GE 图 | NPU 算子视角 | NPU 优化分析（最有价值——能看到 Swish、FlashAttentionScore 这类设备侧算子） | — |

**补充信息**:

| 信息 | 内容 | 用途 |
|---|---|---|
| dispatch | 运行时真实算子序列 + 形状 + 频次表 | ground truth、热点分析 |
| explain | 图断裂位置与原因 | 诊断为什么不能 fullgraph |

**Netron 可视化对两类图都生效**: FX 图经 `torch.onnx` 导出 `.onnx`（含
shape inference）; GE 图经 proto→ONNX 转换器导出 `.onnx`（含输入 shape、
真实权重数据、shape inference）。

## 二、具体用法

### 环境

```bash
cd /path/to/models
source <你的 set_env.sh>
export PYTHONPATH=$PWD:$PYTHONPATH
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # 必设
```

### 唯一入口 `dump()`

```python
from dump_tool import dump

dump(model, *args, **kwargs)                  # model(xxx) 原生形态
dump(model, inputs=make_inputs, ctx=ctx)      # 状态型输入 / 需要上下文
```

| 想要什么 | 参数 |
|---|---|
| FX + GE + dispatch + explain（默认全部） | `graphs=("fx","ge")` |
| 只要 FX 图 | `graphs=("fx",)` |
| 只要 GE 图 | `graphs=("ge",)` |
| 关闭补充信息 | `dispatch=False` / `explain=False` |
| 开启 AOT 中间层 | `aot=True`（可选，一般不需要） |
| Netron 可视化 | `netron=True`（默认开） |
| 缺 GE converter 的算子 | `ge_decompositions={op: 等价分解函数}` |
| KV cache 等被消费的输入 | `inputs=工厂函数`（每层重建） |
| autocast / inference_mode | `ctx=(lambda: torch.autocast("cuda", torch.bfloat16), torch.inference_mode)` |

### 输出目录结构

每个模型统一一个目录，内含固定子结构:

```
dump_graph/<model_name>/
├── dynamo/       FX 图（.py/.txt/.shapes.txt/.onnx）
├── ge/            GE 图（.air/ge_dump/逐子图.txt/.onnx）
├── dispatch/      运行时算子序列（ops.log + ops_summary.txt）
├── breaks.txt     断裂诊断
└── aot/           AOT 中间层（仅 aot=True 时）
```

### 接入方式（按手头信息选）

**已有模型对象**（最直接）:

```python
dump(model, input_ids, attention_mask=mask, use_cache=True)
```

**HF CausalLM**（标准格式全自动）:

```bash
python -m dump_tool dump --weights /path/to/hf/weights \
    --prompt "请介绍一下你自己。" --device npu --out-dir dump_graph/my_llm
```

**非 HF 模型**（复制 `examples/` 模板改三处: 加载/输入/ctx）:

```python
model = ...                        # ① 你的加载代码（GE 需 nn.Module）
def make_inputs(): return (x,), {} # ② model(xxx) 的 xxx
dump(model, inputs=make_inputs, out_dir="dump_graph/my_model")
```

### GE 图获取——两级策略

`graphs` 含 `"ge"` 时自动执行两级:

**第一级: 整图导出**（fullgraph + `.air`）。要求模型 fullgraph 可 trace。

**第二级: 整图失败自动降级逐子图**。原理: torchair 的 GE 图构造发生在
编译期——把它当 AOT fw_compiler 钩子，每个 dynamo 子图在钩子里构造 GE
图并 dump，执行期透传 eager 真值。图断裂不致命; 算子级容错（某子图缺
converter 时点名跳过，其余继续）。

两级都失败（栈级限制如 fake 执行不支持某 op）时，GE 不可达，FX/dispatch
继续执行不受影响。

### 结果判读

- `verify` 行 `max|diff|=0` → FX 捕获正确
- `结构不可比` → dynamo 重建空 ModelOutput 壳（已知，图本身有效，以 shapes.txt 为准）
- FX 层失败 + 自动补 dispatch → 栈级 fake-op 限制，用 ops.log 做 ground truth
- `GE 整图导出失败 → 降级逐子图` → 断裂或缺 converter

## 三、使用示例

`examples/` 下是各模型类型的完整可跑模板——不含机器绝对路径（根目录经
`DUMP_MODELS_ROOT` 环境变量参数化）:

| 模板 | 模型类型 | 演示要点 |
|---|---|---|
| `dump_sam3.py` | 自定义大模型 | 状态型输入 / autocast ctx / GE 分解表 |
| `dump_resnet50.py` | timm 卷积网络 | 自定义权重加载; GE 整图直达 |
| `dump_clip.py` | OpenAI CLIP（非 HF） | 双输入（image + text）; fp16 |
| `dump_wav2vec2.py` | HF 音频 | 断图模型 GE 逐子图降级实例 |
| `dump_dinov2.py` | HF 视觉 backbone | fake-op 栈限制下 dispatch 兜底实例 |
| `dump_esm2.py` | HF 蛋白质 MaskedLM | 大模型（650M）; transformers 5.x capture_outputs 修复 |
| `dump_proteinmpnn.py` | 自定义图神经网络 | torch.load 权重; GE 缺 converter 点名实例 |
| `dump_sd_unet.py` | Stable Diffusion UNet | 860M 扩散模型; pipeline → .unet 提取 |
| `dump_se3transformer.py` | NVIDIA SE3 Transformer | DGL 图输入（进阶） |

内置 demo（含数据依赖分支——fullgraph 编译必败的形态）:

```python
from dump_tool import dump
dump(my_model, my_input, out_dir="dump_graph/demo")
# dynamo: 2 个子图; verify max|diff|=0; breaks=1
# → "模型不能图编译, 计算图照样 dump"
```

## 环境要求

- Python ≥ 3.10, PyTorch ≥ 2.1（2.1 / 2.6 / 2.10 实测）
- **必设** `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`（aarch64 BLAS 段错误/挂死防护）
- 可选: `onnx`（.onnx 导出）; `transformers`（HF CLI）
- CANN 环境（GE 通道需要 `tbe`，PYTHONPATH 必须追加而非覆盖）

## 包结构

```
dump_tool/
├── __init__.py          唯一入口 dump() + 进阶 API
├── core/
│   ├── capture.py       FX 捕获 + dispatch + explain + AOT（可选）
│   ├── compat.py       dynamo trace 兼容 patch（5 项, 幂等, NPU 实测坑）
│   ├── netron_view.py   Netron 导出链: FX→ONNX + GE→ONNX（含 shape/权重）
│   └── ge_view.py       GE 两级策略（整图 + 逐子图）+ GE→Netron
├── entries/             喂料适配层（内部: HF 自动 / entry 契约 / 工厂形态）
├── examples/            9 个模型类型模板（路径参数化）
└── cli.py               python -m dump_tool dump
```

## 验证状态

| torch 版本 | 模型 | FX | GE | dispatch |
|---|---|---|---|---|
| 2.10 + CANN 9.2 | SAM3 / Qwen3-1.7B | ✓ | ✓ | ✓ |
| 2.6 + CANN 9.2 | resnet50 / CLIP / wav2vec2 | ✓ | ✓ | ✓ |
| 2.6 | ESM2-650M / ProteinMPNN / SD UNet | ✓ | ✓ / 部分 | ✓ |
| 2.6 | dinov2（fake bicubic 限制） | ✗ | ✗ | ✓ 兜底 |
| 2.1 | 兼容 patch 防御式跳过 | ✓ | — | ✓ |
