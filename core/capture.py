"""核心捕获层: dynamo FX / AOT core-aten / dispatch 运行时 / 断裂诊断。

核心解耦: dynamo 的"图捕获"与"后端编译"是两个独立阶段。本工具用透传
backend（拿到 FX 图后原样 eager 执行, 不做任何变换）做捕获:
    - 后端永不失败 → 模型能否被 torchair/GE 等图编译后端接受, 与能否
      dump 出计算图完全无关;
    - fullgraph=False 容忍图断裂 → 每个可捕获子图独立落盘, 断裂区自动
      回退 eager 执行; dynamo/dispatch 层透传原样执行, 输出与 eager
      逐位一致（aot 层因 core-aten 分解换算子形态, 仅供结构对照）;
    - dynamo 也 trace 不动的极端代码（强动态/数据依赖控制流）→ 用
      dispatch 模式兜底: 运行时逐算子记录, 对任何可运行模型恒成立。

捕获层级（对任意 callable: nn.Module / 普通函数 / bound method 均可）:
    capture_dynamo_graphs  dynamo 捕获的 FX 图（torch 层; 图断裂 → 多个
                           子图 + explain 断裂报告）
    capture_aot_graphs     AOTAutograd 前向图（core-aten 分解后——与
                           torchair cache_compile 送入 GE 的算子口径同源,
                           便于对齐"缺 converter 算子"与 GE 图）
    trace_dispatch         TorchDispatchMode 运行时逐算子记录（真实执行
                           序列 + 形状/dtype + 频次统计; 含数据依赖分支的
                           ground truth; 绝对兜底）
    dump_graphs            以上层级的一次性编排（单层失败不影响其它层）
"""

import collections
import os
import time

import torch
import torch._dynamo as dynamo
from torch.utils._python_dispatch import TorchDispatchMode

from .compat import (bump_dynamo_limits, patch_trace_compat,
                     restore_dynamo_limits)
from .netron_view import export_netron_view


# ---------- 元信息与落盘 ----------

def _meta(x):
    """值/张量的紧凑元信息（shape/dtype/device）, 用于图文件头与算子日志。"""
    if isinstance(x, torch.Tensor):
        return (f"T{tuple(x.shape)}.{str(x.dtype).replace('torch.', '')}"
                f"[{x.device.type}]")
    if isinstance(x, (tuple, list)):
        return "[" + ", ".join(_meta(v) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k}={_meta(v)}" for k, v in x.items()) + "}"
    if isinstance(x, float):
        return repr(x)  # 勿用 int(x) 美化: inf/nan 会抛 OverflowError
    text = repr(x)
    return text if len(text) <= 48 else text[:45] + "..."


def _node_output_meta(node):
    """节点输出张量的元信息（dynamo 用 example_value, aot 用 val）。"""
    meta = getattr(node, "meta", None) or {}
    for key in ("val", "example_value"):
        if key in meta:
            return _meta(meta[key])
    return "-"


def _save_fx_graph(gm, example_inputs, out_dir, prefix, note=""):
    """落盘一个 FX 图: 生成代码(.py) + textual IR(.txt) + 逐节点 shape 表
    (.shapes.txt), 头部带输入元信息。

    编号只统计同前缀 .py 文件（netron 导出的 .onnx/.pt 共享前缀, 不参
    与计数）→ 同一 out_dir 重复运行不覆盖、编号连续。
    """
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in os.listdir(out_dir)
                if f.startswith(prefix) and f.endswith(".py")]
    base = os.path.join(out_dir, f"{prefix}_{len(existing) + 1:02d}")
    header = (f"# note: {note}\n"
              f"# example_inputs: {_meta(example_inputs)}\n\n")
    try:
        code = gm.code
    except Exception:  # noqa: BLE001  极端图结构下 codegen 失败的兜底
        code = gm.print_readable(print_output=False)
    with open(base + ".py", "w", encoding="utf-8") as f:
        f.write(header)
        f.write(code)
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(header)
        f.write(str(gm.graph))
    # 逐节点输出 shape/dtype 表: dynamo 的 example_value / aot 的 val 都
    # 是 trace 期 FakeTensor 元数据, 不执行任何计算; 老版本 torch 无 meta
    # 时降级为 "-"
    with open(base + ".shapes.txt", "w", encoding="utf-8") as f:
        f.write(f"# 每个节点的输出 shape/dtype（placeholder 即输入形状）\n")
        f.write(f"# {note}\n\n")
        for node in gm.graph.nodes:
            if node.op == "output":
                continue
            target = str(node.target)
            f.write(f"{node.name:40s} {node.op:14s} "
                    f"{target[:56]:56s} -> {_node_output_meta(node)}\n")
    return base


# ---------- 层级 1: dynamo FX 图（透传 backend, 永不编译失败） ----------

def capture_dynamo_graphs(fn, args=(), kwargs=None, out_dir="graph_dump/dynamo",
                          fullgraph=False, dynamic=False, netron=False,
                          verbose=True):
    """用透传 backend 捕获 dynamo FX 图并原样执行。

    backend 只做两件事: 存图、返回 gm.forward——不引入任何编译/变换,
    因此数值与 eager 逐位一致, 且"模型能否被某个图编译后端接受"与本
    捕获无关（torchair/GE 编译不了照样能 dump）。fullgraph=False（默认）
    时图断裂不报错: 断裂区回退 eager, 每个捕获到的子图独立落盘; 重编译
    （新 shape 族）同样各落一盘。netron=True 时每个子图额外导出
    <base>.onnx/.pt（Netron 可视化, 见 netron_view.export_netron_view）。

    返回 (fn 的执行结果, 落盘图数)。
    """
    kwargs = kwargs or {}
    patch_trace_compat()
    dynamo.reset()
    saved = bump_dynamo_limits()
    n_graphs = [0]
    pending_exports = []

    def _pass_through(gm, example_inputs):
        n_graphs[0] += 1
        base = _save_fx_graph(gm, example_inputs, out_dir, "dynamo_graph",
                              note=f"fullgraph={fullgraph}, dynamic={dynamic}")
        if netron:
            # 【关键】不在 backend 回调里导出: onnx-dynamo 尝试内部会起
            # 嵌套 dynamo(torch.export), 在回调内执行会污染外层 eval-frame,
            # 异常从外层炸出且无法捕获 → 回调只暂存, 编译调用结束后
            # （dynamo 帧已退出）统一导出。
            pending_exports.append((gm, list(example_inputs), base))
        return gm.forward

    try:
        compiled = torch.compile(fn, backend=_pass_through,
                                 fullgraph=fullgraph, dynamic=dynamic)
        result = compiled(*args, **kwargs)
    finally:
        restore_dynamo_limits(saved)
    if netron:
        for gm, ex_inputs, base in pending_exports:
            got = export_netron_view(gm, ex_inputs, base, verbose)
            if got and verbose:
                print(f"[dump_tool] netron view -> {got}")
    if verbose:
        print(f"[dump_tool] dynamo: {n_graphs[0]} 个子图 -> {out_dir}")
    return result, n_graphs[0]


# ---------- 层级 2: AOT core-aten 图（与 GE 编译入口同源层级） ----------

def capture_aot_graphs(fn, args=(), kwargs=None, out_dir="graph_dump/aot",
                       dynamic=False, netron=False, verbose=True):
    """捕获 AOTAutograd 前向图（默认分解表作用后的 core-aten 图）。

    该层级即 torchair cache_compile 送入 GE 的算子口径（custom_
    decompositions 在此生效）, 便于与 GE 图 / "缺 converter 算子"清单
    对齐。fw_compiler 只存图并原样返回 gm.forward（同 aot_eager 语义,
    无任何设备侧编译）, 推理场景不挂 backward。图断裂容忍与 dynamo 层
    相同。netron=True 时每个图额外导出 <base>.onnx/.pt。

    注意: 与 dynamo 层不同, core-aten 分解会改变实际执行的算子形态
    （linear→addmm、sdpa 分解等）→ NPU 内核选择变化, 输出可能有轻微
    数值漂移（与真实 GE 编译同类现象）。该层用于算子口径/结构分析,
    数值基准以 dynamo 层（透传逐位一致）为准。

    返回 (fn 的执行结果, 落盘图数)。
    """
    kwargs = kwargs or {}
    from torch._dynamo.backends.common import aot_autograd

    patch_trace_compat()
    dynamo.reset()
    saved = bump_dynamo_limits()
    n_graphs = [0]
    pending_exports = []

    def _fw(gm, example_inputs):
        n_graphs[0] += 1
        base = _save_fx_graph(gm, example_inputs, out_dir, "aot_graph_fw",
                              note=f"dynamic={dynamic}")
        if netron:
            pending_exports.append((gm, list(example_inputs), base))
        return gm.forward

    try:
        backend = aot_autograd(fw_compiler=_fw)
        compiled = torch.compile(fn, backend=backend, fullgraph=False,
                                 dynamic=dynamic)
        result = compiled(*args, **kwargs)
    finally:
        restore_dynamo_limits(saved)
    if netron:
        for gm, ex_inputs, base in pending_exports:
            got = export_netron_view(gm, ex_inputs, base, verbose)
            if got and verbose:
                print(f"[dump_tool] netron view -> {got}")
    if verbose:
        print(f"[dump_tool] aot: {n_graphs[0]} 个前向图 -> {out_dir}")
    return result, n_graphs[0]


# ---------- 层级 3: 运行时逐算子记录（任何可运行模型恒成立的兜底） ----------

class _DispatchTraceMode(TorchDispatchMode):
    """dispatcher 层逐算子记录: 名称 + 输入形状/dtype, 原样透传执行。

    TorchDispatchMode 挂在 dispatch key 栈顶, 不参与任何 trace/编译 →
    对强动态控制流、数据依赖分支、dynamo 完全不可 trace 的代码全部成立,
    记录的是"真实执行的算子序列"（ground truth, 含实际走到的分支）。
    """

    def __init__(self, log_path):
        super().__init__()
        self._log_path = log_path
        self._log = None
        self.counts = collections.Counter()

    def __enter__(self):
        self._log = open(self._log_path, "w", encoding="utf-8")
        return super().__enter__()

    def __exit__(self, *exc_info):
        ret = super().__exit__(*exc_info)
        if self._log is not None:
            self._log.close()
            self._log = None
        return ret

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.counts[str(func)] += 1
        if self._log is not None:
            parts = [_meta(a) for a in args]
            parts += [f"{k}={_meta(v)}" for k, v in kwargs.items()]
            self._log.write(f"{func}({', '.join(parts)})\n")
        return func(*args, **kwargs)


def trace_dispatch(fn, args=(), kwargs=None, out_dir="graph_dump/dispatch",
                   topk=40, verbose=True):
    """运行时逐算子记录——不依赖任何 trace/编译, 对任何可运行模型恒成立。

    适合: 数据依赖控制流的真实执行路径、dynamo 完全不可 trace 的入口、
    以及与 FX 图对拍确认图断裂区里实际执行了什么算子。
    返回 (fn 的执行结果, Counter{算子名: 调用次数})。
    """
    kwargs = kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "ops.log")
    mode = _DispatchTraceMode(log_path)
    t0 = time.perf_counter()
    with mode:
        result = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    total = sum(mode.counts.values())
    lines = [f"# total={total} unique={len(mode.counts)} wall={wall:.3f}s"]
    lines += [f"{cnt:10d}  {name}" for name, cnt in mode.counts.most_common(topk)]
    with open(os.path.join(out_dir, "ops_summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if verbose:
        print(f"[dump_tool] dispatch: {total} ops / {len(mode.counts)} unique "
              f"-> {log_path}")
        print("\n".join(lines[: topk + 1]))
    return result, mode.counts


# ---------- 图断裂诊断 ----------

def explain_breaks(fn, args=(), kwargs=None, out_path=None, verbose=True):
    """dynamo.explain: 图断裂位置/原因（trace 级诊断, 不做真实设备计算）。"""
    kwargs = kwargs or {}
    patch_trace_compat()
    dynamo.reset()
    try:
        explained = dynamo.explain(fn)(*args, **kwargs)
    except Exception as e:  # noqa: BLE001  极端不可 trace 时给出兜底指引
        text = (f"explain 失败: {type(e).__name__}: {e}\n"
                "→ dynamo 无法 trace 该入口, 请改用 trace_dispatch 兜底。")
        summary = text.splitlines()[0]
    else:
        breaks = (getattr(explained, "graph_break_count", None)
                  if hasattr(explained, "graph_break_count") else None)
        if breaks is None:
            breaks = len(getattr(explained, "graph_breaks", []) or [])
        summary = (f"graph_count={explained.graph_count}, "
                   f"op_count={explained.op_count}, breaks={breaks}")
        text = summary + "\n\n" + str(explained)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if verbose:
        if out_path:
            print(f"[dump_tool] explain: {summary} -> {out_path}")
        else:
            print(f"[dump_tool] explain:\n{text}")
    return text


# ---------- 编排: 多层级一次性 dump ----------

def dump_graphs(fn, args=(), kwargs=None, out_dir="graph_dump",
                modes=("dynamo", "aot"), fullgraph=False, dynamic=False,
                netron=True, with_explain=True, verbose=True):
    """按 modes 逐层 dump（每层独立执行一遍 fn, 单层失败不影响其它层）。

    返回 dict: {mode: fn 执行结果}（失败层为 None; "explain" 为诊断文本）。
    netron 默认 True: dynamo/aot 层每个图额外导出 .onnx/.pt, Netron 可
    直接打开（FX 的 .txt/.py 是 PyTorch 内部表示, Netron 不识别）。仅
    需要文本产物时传 netron=False（大模型的 ONNX 导出可较慢）。
    注意:
      - 多层 = 多次执行——对有副作用的入口（原地改输入 state 的方法）
        传单层, 或先确认重复执行无害; 有 KV cache 等被消费状态的入口
        请按层单独调用 capture_* 并每层重建状态（参考 hf_llm_entry）;
      - dynamo / dispatch 层透传原样执行, 输出与 eager 逐位一致, 业务
        返回值以这两层为准;
      - aot 层图经 core-aten 分解, 实际执行的算子形态改变（linear→
        addmm、sdpa 分解等）→ NPU 内核选择变化, 输出可能有轻微数值
        漂移（与真实 GE 编译同类现象）; 该层用于算子口径/结构对照。
    """
    results = {}

    def _run(mode, fn_pass):
        try:
            results[mode] = fn_pass()
        except Exception as e:  # noqa: BLE001  单层失败降级, 不阻断其它层
            results[mode] = None
            print(f"[dump_tool] {mode} 层失败（其余层不受影响）: "
                  f"{type(e).__name__}: {e}")

    if "dynamo" in modes:
        _run("dynamo", lambda: capture_dynamo_graphs(
            fn, args, kwargs, os.path.join(out_dir, "dynamo"),
            fullgraph=fullgraph, dynamic=dynamic, netron=netron,
            verbose=verbose)[0])
    if "aot" in modes:
        _run("aot", lambda: capture_aot_graphs(
            fn, args, kwargs, os.path.join(out_dir, "aot"),
            dynamic=dynamic, netron=netron, verbose=verbose)[0])
    if "dispatch" in modes:
        _run("dispatch", lambda: trace_dispatch(
            fn, args, kwargs, os.path.join(out_dir, "dispatch"),
            verbose=verbose)[0])
    if with_explain:
        _run("explain", lambda: explain_breaks(
            fn, args, kwargs,
            out_path=os.path.join(out_dir, "breaks.txt"), verbose=verbose))
    return results


# ---------- model(xxx) 直配糖 + 通用对拍 ----------

def tensor_diff(a, b):
    """递归比较两个输出（tensor / dict / list / tuple）, 返回 max|diff|。

    结构不一致返回 -1.0; 无 tensor 时按值相等性返回 0.0 / -1.0。
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return -1.0
        return (a.float() - b.float()).abs().max().item()
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return -1.0
        return max((tensor_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return -1.0
        return max((tensor_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if a is None and b is None:
        return 0.0
    return 0.0 if a == b else -1.0
