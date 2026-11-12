"""dump_tool: 通用计算图 dump 工具（模型无关 / 设备无关 / 不要求能图编译）。

核心思想: dynamo 的"图捕获"与"后端编译"是两个独立阶段。用透传 backend
（拿到 FX 图后原样 eager 执行）做捕获, 后端永不失败——模型能否被
torchair/GE 等图编译后端接受, 与能否 dump 出计算图完全无关; dynamo
trace 不动的极端代码用 dispatch 模式（运行时逐算子记录）兜底。

唯一入口 dump()（其余均为进阶 API, 见 __all__）:

    from dump_tool import dump

    # 常规: 已有模型对象, 输入即 model(xxx) 的 xxx
    dump(model, input_ids, attention_mask=mask, use_cache=True,
         out_dir="graph_dump/my_model")

    # 输入含被消费状态（KV cache 等）或需要执行上下文（autocast 等）:
    def make_inputs():                       # 每个捕获层调用一次
        return (), dict(input_ids=..., use_cache=True)
    ctx = (lambda: torch.autocast("cuda", torch.bfloat16),
           torch.inference_mode)
    dump(model, inputs=make_inputs, ctx=ctx, out_dir="graph_dump/my_model")

    # GE 图（不做 GE 编译; 需 nn.Module + fullgraph 可 trace + CANN 环境）:
    dump(model, inputs=make_inputs, ctx=ctx, ge=True)

非 HF 模型一律用上面的 Python 形态（模板见 examples/dump_sam3.py）;
CLI（python -m dump_tool dump --weights <HF目录>）仅支持 HF CausalLM——
因为只有标准格式才能从权重自动推导加载与输入方式。

捕获层级与输出:
    dynamo/  FX 图（torch 层, 断裂容忍, 与 eager 逐位一致）
             .py 生成代码 | .txt textual IR | .shapes.txt 逐节点形状
             | .onnx/.pt Netron 可直接打开
    aot/     core-aten 分解图（torchair/GE 编译入口同源算子口径）
    dispatch/ops.log + ops_summary.txt  运行时真实算子序列与频次
    breaks.txt  图断裂诊断

已知环境建议:
    - netron 导出阶段在 aarch64 上建议 OPENBLAS_NUM_THREADS=1
      OMP_NUM_THREADS=1（多线程 BLAS 有死锁风险）;
    - .onnx 导出需要 pip 包 onnx（缺失自动回退 TorchScript 存档 .pt）;
    - GE 通道需要 CANN 环境（tbe 在 CANN 的 PYTHONPATH——自定义
      PYTHONPATH 必须追加而非覆盖）。

实测验证: torch 2.10 + torch_npu 2.10 + CANN 9.2 上完成 SAM3
(forward_grounding, Python 形态) 与 Qwen3-1.7B (prefill / decode, CLI 形态)
全链 dump——dynamo 层与 eager 逐位一致, ONNX 产物全部通过 onnx.checker;
两模型的 GE 图均以"无编译导出"通道捕获。
"""

import os

import torch

from .core import (DEFAULT_ATTEMPT_TIMEOUT, capture_aot_graphs,
                   capture_dynamo_graphs, dump_graphs, explain_breaks,
                   export_ge_graph, export_netron_view, patch_trace_compat,
                   tensor_diff, trace_dispatch)

__version__ = "3.0.0"


def dump(model, *args, inputs=None, ctx=None, graphs=("fx", "ge"),
         ge_decompositions=None, out_dir="graph_dump", netron=True,
         dispatch=True, explain=True, aot=False, verbose=True, **kwargs):
    """唯一入口: dump 一个模型入口的计算图。

    两类真实图: FX 图（torch 层）与 GE 图（NPU 算子层）。补充信息: dispatch
    （运行时算子序列+频次）与 explain（断裂诊断）。netron=True 时两类图
    都额外导出 Netron 可打开的 .onnx/.pt。aot 是 FX→GE 之间的分解中间层,
    已被直接 GE 捕获取代, 默认关闭（保留为进阶选项）。

    两种输入给法:
      直接传参（输入无状态, 各层可复用）: dump(model, x, mask=..., ...)
      工厂形态（输入含被消费状态, 如 KV cache）: dump(model, inputs=fn)
        —— fn 无参调用返回 (args, kwargs), 每个捕获层各调用一次重建状态。

    ctx: 执行上下文工厂元组（默认 (torch.no_grad,)）。
    graphs: ("fx", "ge") 子集——要哪类图。
    ge_decompositions: 缺 GE converter 算子的等价分解表。
    多层捕获 = 多次执行模型; 输出含 eager 基准对拍（FX 层应 max|diff|=0）。
    """
    contexts = ctx if ctx is not None else (torch.no_grad,)
    if callable(contexts):
        contexts = (contexts,)

    if inputs is None:
        def _make_inputs():
            return args, kwargs
    else:
        _make_inputs = inputs

    from .entries import _normalize_inputs

    if "ge" in graphs:
        if not isinstance(model, torch.nn.Module):
            if verbose:
                print(f"[dump_tool] GE 跳过: 要求 nn.Module, got "
                      f"{type(model).__name__}", flush=True)
        else:
            with _stack_contexts(contexts):
                call_args, call_kwargs = _normalize_inputs(_make_inputs())
                ge_dir = out_dir if "fx" not in graphs else out_dir + "_ge"
                # 两级 GE 策略: 整图优先（fullgraph + .air）; 失败自动降级
                # 逐子图捕获（断裂容忍 + 算子级容错）。
                try:
                    from .core import export_ge_graph
                    export_ge_graph(
                        model, call_args, call_kwargs, out_dir=ge_dir,
                        export_name="ge_graph", netron=netron,
                        custom_decompositions=ge_decompositions,
                        verbose=verbose)
                except Exception as e:  # noqa: BLE001  整图失败 → 降级
                    if verbose:
                        print(f"[dump_tool] GE 整图导出失败（{type(e).__name__}: "
                              f"{str(e)[:160]}）→ 降级为逐子图 GE 捕获",
                              flush=True)
                    from .core import capture_ge_graphs
                    capture_ge_graphs(
                        model, call_args, call_kwargs, out_dir=ge_dir,
                        custom_decompositions=ge_decompositions, netron=netron,
                        verbose=verbose)

    if "fx" in graphs:
        from .entries import DumpTarget, run_dump
        # name="" → 产物直接落在 out_dir 下（不额外套一层子目录）
        target = DumpTarget(name="", fn=model, make_inputs=_make_inputs,
                            contexts=contexts)
        modes = ["dynamo"]
        if aot:
            modes.append("aot")
        if dispatch:
            modes.append("dispatch")
        return run_dump([target], out_dir=out_dir, modes=tuple(modes),
                        netron=netron, with_explain=explain, verbose=verbose)

    return None


def _stack_contexts(contexts):
    import contextlib
    stack = contextlib.ExitStack()
    for factory in contexts:
        stack.enter_context(factory())
    return stack


__all__ = [
    "dump", "dump_graphs", "capture_dynamo_graphs", "capture_aot_graphs",
    "capture_ge_graphs", "trace_dispatch", "explain_breaks", "tensor_diff",
    "export_netron_view", "export_ge_netron", "export_ge_graph",
    "patch_trace_compat", "DEFAULT_ATTEMPT_TIMEOUT", "__version__",
]
