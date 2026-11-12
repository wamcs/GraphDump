# dump_tool — 通用计算图 dump 工具

模型无关 / 设备无关（CPU / CUDA / NPU）/ **不要求模型能图编译**，一个入口
`dump()` 获取模型的各类计算图（含 NPU 算子视角的 GE 图），全程不做图编译。

## 一、能获取哪些图

模型从 Python 调用到 NPU 内核要经过一条变换链，工具在链上不同位置取图：

```
Python 代码
   │ ① dynamo trace（字节码级符号执行）
   ▼
torch 层 FX 图           ← 【FX 图】F.linear / F.softmax 原样
   │ ② AOT 分解（linear→addmm、layer_norm→native_layer_norm …）
   ▼                                     ↑ aot=True 可选查看中间层
   core-aten 图              （已被 GE 直接捕获取代, 默认关）
   │ ③ torchair FX→GE converter（逐算子映射为 GE 算子）
   ▼
GE 图（NPU 算子视角）    ← 【GE 图】MatMulV2 / SoftmaxV2 / Swish …
   │ ④ GE 编译 + 内核选择（本工具不走到这一步）
   ▼
aclnn 内核
```

**两类真实图**——工具的产物就是它们的对比:

| 图 | 内容 | 典型用途 | 数值一致性 |
|---|---|---|---|
| FX 图 | torch 层计算图本体（`.py`/`.txt`/`.shapes.txt`） | 模型结构分析、优化机会清单 | **逐位一致**（透传执行） |
| GE 图 | NPU 算子视角（文本 proto / `.air`） | NPU 优化分析（最有价值——能看到 Swish、FlashAttentionScore 这类设备侧算子） | — |

**补充信息**（不是图, 但与图共同产出）:

| 信息 | 内容 | 用途 |
|---|---|---|
| dispatch | 运行时真实算子序列 + 形状 + 频次表 | ground truth、热点分析 |
| explain | 图断裂位置与原因（`breaks.txt`） | 诊断为什么不能 fullgraph |

**中间层（默认关闭）**: AOT core-aten 图（`aot=True` 可选）——FX→GE 之间
的分解形态, 原本的价值（GE 编译口径近似）已被 GE 直接捕获取代, 仅在无
CANN 环境又想看分解口径时可选。

**Netron 可视化对两类图都生效**: FX 图经 `torch.onnx` 导出 `.onnx`/`.pt`;
GE 图经 proto→ONNX 转换器导出 `.onnx`（常见 GE 算子映射为标准 ONNX op,
NPU 专属算子按原名透传——Netron 对未识别 op 类型仍渲染节点连线）。

## 二、具体用法

### 环境准备（每次必做）

```bash
cd /path/to/models                     # dump_tool 所在目录
source <你的 set_env.sh>               # CANN + venv（NPU 环境）
export PYTHONPATH=$PWD:$PYTHONPATH     # 追加（覆盖会丢 CANN 的 tbe）
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # 必设（aarch64 BLAS 段错误/挂死防护）
```

### 唯一入口 `dump()`

```python
from dump_tool import dump

dump(model, *args, **kwargs)                  # model(xxx) 原生形态
dump(model, inputs=make_inputs, ctx=ctx)      # 状态型输入/需要执行上下文时
```

| 想要什么 | 参数 |
|---|---|
| FX 图 + GE 图 + dispatch + explain | 默认（`graphs=("fx","ge")`, `dispatch=True`, `explain=True`） |
| 只要 FX 图 | `graphs=("fx",)` |
| 只要 GE 图 | `graphs=("ge",)` |
| 关闭补充信息 | `dispatch=False` / `explain=False` |
| 开启 AOT 中间层 | `aot=True`（进阶, 一般不需要） |
| Netron 可视化 | `netron=True`（默认开; FX 与 GE 都产出 `.onnx`/`.pt`） |
| 缺 GE converter 的算子 | `ge_decompositions={算子: 等价分解函数}` |
| KV cache 等被消费的输入 | `inputs=工厂函数`（每个捕获层各调用一次重建状态） |
| autocast / inference_mode | `ctx=(lambda: torch.autocast("cuda", torch.bfloat16), torch.inference_mode)` |

三种接入方式按手头信息选：

**已有模型对象**（最直接）——参数即 `model(xxx)` 的 `xxx`：

```python
dump(model, input_ids, attention_mask=mask, use_cache=True,
     out_dir="dump_graph/my_model")
```

**HF CausalLM**（标准格式可全自动）——CLI 一条命令：

```bash
python -m dump_tool dump --weights /path/to/hf/weights \
    --prompt "请介绍一下你自己。" --device npu --out-dir dump_graph/my_llm
# 常用: --entries prefill,decode | --graphs fx,ge | --no-netron | --demo
```

**非 HF 模型**（shell 传不了张量，加载方式只有模型作者知道）——复制
`examples/` 模板改三处：①`model = 你的加载代码`（GE 需要 nn.Module）
②`inputs()` 返回样例输入 ③可选 `ctx`。

### GE 图获取——两级策略

`graphs` 含 `"ge"` 时自动执行两级:

**第一级: 整图导出**（fullgraph + `.air`）。torchair 的"FX→GE 构造"与
"GE 编译（graph.load）"是两个独立阶段——export_mode 下构造完成后抛
`ExportSuccess` 优雅退出，编译永不执行。要求 fullgraph 可 trace。

**第二级: 整图失败自动降级逐子图捕获**（断裂容忍）。原理：torchair 的
GE 图构造发生在**编译期**（compiler 调用返回即就绪）——把它当 AOT
fw_compiler 钩子，每个 dynamo 子图在钩子里构造 GE 图并 dump，执行期
透传 eager 真值。图断裂不致命；算子级容错（某子图缺 converter 时点名
跳过，其余继续）。

前置条件：CANN 环境（`tbe` 在 CANN 的 PYTHONPATH，故 PYTHONPATH 必须追加）；
模型为 nn.Module。真实编译期的各优化阶段 GE IR 另可用 `DUMP_GE_GRAPH=1`。

### 结果判读

- `verify` 行 `max|diff|=0` → FX 捕获正确（透传执行，与 eager 逐位一致）
- `结构不可比` → dynamo 重建了空 ModelOutput 壳（已知现象，图本身有效，以 shapes.txt 为准）
- FX 层失败 + 自动补 dispatch → 栈级 fake-op 限制（如 bicubic interpolate），用 `ops.log` 做 ground truth
- `GE 整图导出失败 → 降级逐子图` → 断裂或缺 converter，看已拿到的 `ge_graph_*.txt` 与 `.onnx`

## 三、使用示例

`examples/` 下是各模型类型的完整可跑模板——**不含机器绝对路径**（根目录经
`DUMP_MODELS_ROOT` 环境变量参数化，默认相对定位），复制即可改造：

| 模板 | 模型类型 | 演示要点 |
|---|---|---|
| `dump_sam3.py` | 自定义大模型 | 状态型输入（`make_inputs` 每层重建）、bf16 autocast `ctx`、GE 分解表、fullgraph 前置环境变量 |
| `dump_dinov2.py` | HF 视觉 backbone | 图像预处理输入；fake-op 栈限制下 FX 失败自动补 dispatch 的实例 |
| `dump_wav2vec2.py` | HF 音频 | 音频特征输入；断图模型 GE 逐子图降级实例 |
| `dump_resnet50.py` | timm 卷积网络（非 transformers） | 自定义权重加载；GE 整图直达（无需降级） |

内置 demo（含数据依赖分支——fullgraph 编译必败的形态）的最简用法:

```python
from dump_tool import dump
dump(my_model, my_input, out_dir="dump_graph/demo")
```

```
[dump_tool] dynamo: 2 个子图 -> .../dynamo      # 断裂容忍: 每个子图独立落盘
[dump_tool] [demo/verify] dynamo vs eager: max|diff|=0.000e+00
[dump_tool] explain: graph_count=2, op_count=5, breaks=1
```

## 环境要求与已知建议

- Python ≥ 3.10，PyTorch ≥ 2.1（2.1 / 2.6 / 2.10 实测；兼容 patch 防御式跳过）
- **必设** `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`（见上）
- 可选依赖：`onnx`（.onnx 导出，缺失自动回退 `.pt`）；`transformers`（HF CLI）
- 大模型（28 层级）的 ONNX 导出单图可达 10+ 分钟

## 包结构

```
dump_tool/
├── __init__.py     唯一入口 dump() + 进阶 API
├── core/           捕获内核
│   ├── capture.py      FX 图捕获 + dispatch + explain + AOT（可选）+ 对拍
│   ├── compat.py       dynamo trace 兼容 patch（NPU 实测坑, 幂等）
│   ├── netron_view.py  Netron 导出链: FX→ONNX/PT + GE→ONNX
│   └── ge_view.py      GE 两级策略（整图导出 + 逐子图捕获）+ GE→Netron
├── entries/        喂料适配层（内部机制: CLI 的 HF 自动 + 工厂形态支撑）
├── examples/       各模型类型模板（路径参数化, 无机器绝对路径）
└── cli.py          python -m dump_tool dump（HF / --demo）
```

进阶 API（一般不需要）：`dump_graphs`、`capture_dynamo_graphs`（FX 单层）、
`capture_aot_graphs`（AOT 单层）、`capture_ge_graphs`（GE 逐子图）、
`trace_dispatch` / `explain_breaks`（补充信息单层）、
`export_ge_graph` / `export_netron_view` / `export_ge_netron`（格式转化单函数）。

## 验证状态

- torch 2.10 + torch_npu 2.10 + CANN 9.2：SAM3（自定义模型）与 Qwen3-1.7B
  （HF）——FX 与 GE 双图全链 dump，dynamo 层与 eager 逐位一致，ONNX 产物过
  `onnx.checker`（FX 路径）
- torch 2.6 + torch_npu 2.6：dinov2 / wav2vec2 / resnet50（HF 视觉、音频、
  timm 卷积）——两级 GE 策略全覆盖；GE→ONNX 转换器验证（resnet50 719 GE
  算子 → 339 ONNX 节点 + 268 输入 + 111 initializer）
- torch 2.1 + torch_npu 2.1：兼容 patch 防御式跳过，核心捕获路径可用
# GraphDump
