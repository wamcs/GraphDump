"""entries: 喂料适配层（内部机制——公共入口是 dump_tool.dump / CLI）。

职责: 把"权重 + 入口"变成可 dump 的目标, 支撑两处:
  - CLI: --weights <HF 目录>（标准格式可自动推导加载与输入）
  - dump(model, inputs=工厂, ctx=...) 的工厂形态（状态重建）

entry 文件契约（对任意本地可运行模型; 全部自行管理 import 与 sys.path）:

    # my_entry.py
    import os
    os.environ["MY_MODEL_FLAG"] = "1"   # 顶层: import 模型库之前需要的环境变量

    def load(weights, device=None, dtype=None):
        \"\"\"必需。加载并返回模型——nn.Module 或任意 callable;
        想用 GE 导出时必须返回 nn.Module（forward 即入口）。\"\"\"
        my_model = build_model(weights, device=device)
        return my_model

    def inputs(model):
        \"\"\"必需。返回一组样例输入 (args, kwargs)——**每次调用都会被执行**
        （每个捕获层各一次）, 消费型状态（KV cache 等）在此重建。\"\"\"
        x = make_sample_input()
        return (x,), {"flag": True}

    NAME = "my_model"                    # 可选: 输出子目录名（默认文件名）
    def contexts():                      # 可选: 调用包裹上下文（默认 no_grad）
        import contextlib, torch
        return contextlib.ExitStack()    # 例: autocast + inference_mode 嵌套
    def ge_decompositions():             # 可选: GE 导出的等价分解表
        return {...}
"""

import contextlib
import importlib.util
import os

import torch

from ..core import (capture_aot_graphs, capture_dynamo_graphs,
                    explain_breaks, export_ge_graph, tensor_diff,
                    trace_dispatch)


class DumpTarget:
    """一个可 dump 的目标: 入口 callable + 新鲜输入工厂 + 上下文。

    make_inputs: 每个捕获层调用一次（返回 (args, kwargs)）——KV cache
    等被消费的状态在这里重建, 保证层间互不污染。
    contexts: 长度任意的上下文工厂元组（如 no_grad / autocast /
    inference_mode）, runner 逐层嵌套包裹调用。
    """

    def __init__(self, name, fn, make_inputs,
                 contexts=(torch.no_grad,), ge_decompositions=None,
                 ge_module=None):
        self.name = name
        self.fn = fn
        self.make_inputs = make_inputs
        self.contexts = contexts
        self.ge_decompositions = ge_decompositions
        self.ge_module = ge_module


def load_entry(entry, weights, device=None, dtype=None, **entry_kwargs):
    """解析 entry 规格并构建 DumpTarget 列表。

    entry: "hf" / "demo"（内置）或 entry 文件路径。
    返回 list[DumpTarget]。
    """
    if entry == "hf":
        from .hf import build_targets
        return build_targets(weights, device=device, dtype=dtype,
                             **entry_kwargs)
    if entry == "demo":
        from .demo import build_targets
        return build_targets(device=device)
    if os.path.isfile(entry):
        from .script import build_targets
        return build_targets(entry, weights, device=device, dtype=dtype,
                             **entry_kwargs)
    raise FileNotFoundError(
        f"entry 不存在: {entry!r}（内置: hf / demo; 否则为 entry 文件路径）")


def _wrap_contexts(contexts):
    stack = contextlib.ExitStack()
    for factory in contexts:
        stack.enter_context(factory())
    return stack


def _safe(tag, fn, verbose=True):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001  单层失败不阻断其它层/其它目标
        if verbose:
            print(f"[dump_tool] {tag} 失败（其余不受影响）: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
        return None


def run_dump(targets, out_dir="graph_dump", modes=("dynamo", "aot", "dispatch"),
             netron=True, with_explain=True, verbose=True):
    """对目标列表逐个执行多层级 dump（含 eager 基准对拍）。"""
    report = {}
    for target in targets:
        tdir = os.path.join(out_dir, target.name)
        if verbose:
            print(f"[dump_tool] ===== dump 目标: {target.name} -> {tdir} =====",
                  flush=True)

        # eager 基准（对拍用）
        with _wrap_contexts(target.contexts):
            args, kwargs = _normalize_inputs(target.make_inputs())
            ref = target.fn(*args, **kwargs)

        entry_report = {}

        def _capture(mode, capture_fn):
            with _wrap_contexts(target.contexts):
                args, kwargs = _normalize_inputs(target.make_inputs())
                return _safe(
                    f"[{target.name}/{mode}]",
                    lambda: capture_fn(args, kwargs), verbose)

        if "dynamo" in modes:
            out = _capture("dynamo", lambda a, k: capture_dynamo_graphs(
                target.fn, a, k, os.path.join(tdir, "dynamo"), netron=netron,
                verbose=verbose)[0])
            entry_report["dynamo"] = out
            if out is not None and ref is not None:
                diff = _safe_verify(out, ref)
                if diff is not None:
                    if diff >= 0:
                        print(f"[dump_tool] [{target.name}/verify] dynamo vs "
                              f"eager: max|diff|={diff:.3e}（透传捕获, 应为 0）",
                              flush=True)
                    else:
                        # 结构不一致: 已知现象——dynamo 对部分 ModelOutput 重建
                        # 空壳（graph 本身有效, 输出在 FX 图内）。勿再打印"应为 0"。
                        print(f"[dump_tool] [{target.name}/verify] dynamo 输出"
                              "与 eager 结构不可比（如 ModelOutput 重建空壳; "
                              "图与 shapes.txt 本身有效）", flush=True)
            if out is None and "dispatch" not in modes:
                # trace 失败（栈级 fake-op 限制等）→ 自动补 dispatch 兜底:
                # 只要 eager 能跑就有 ground truth
                print(f"[dump_tool] [{target.name}] dynamo 层失败 → "
                      "自动补 dispatch 兜底（运行时逐算子记录）", flush=True)
                entry_report["dispatch"] = _capture(
                    "dispatch", lambda a, k: trace_dispatch(
                        target.fn, a, k, os.path.join(tdir, "dispatch"),
                        verbose=verbose)[0])
        if "aot" in modes:
            entry_report["aot"] = _capture("aot", lambda a, k: capture_aot_graphs(
                target.fn, a, k, os.path.join(tdir, "aot"), netron=netron,
                verbose=verbose)[0])
        if "dispatch" in modes:
            entry_report["dispatch"] = _capture(
                "dispatch", lambda a, k: trace_dispatch(
                    target.fn, a, k, os.path.join(tdir, "dispatch"),
                    verbose=verbose)[0])
        if with_explain:
            with _wrap_contexts(target.contexts):
                args, kwargs = _normalize_inputs(target.make_inputs())
                entry_report["explain"] = _safe(
                    f"[{target.name}/explain]",
                    lambda: explain_breaks(
                        target.fn, args, kwargs,
                        out_path=os.path.join(tdir, "breaks.txt"),
                        verbose=verbose), verbose)
        report[target.name] = entry_report
    return report


def _normalize_inputs(result):
    """兼容 inputs() 返回 (args, kwargs) / (args,) / 单值。"""
    if isinstance(result, tuple) and len(result) == 2 and (
            isinstance(result[1], dict) or result[1] is None):
        return result[0], (result[1] or {})
    if isinstance(result, tuple):
        return result, {}
    return (result,), {}


def _safe_verify(out, ref):
    try:
        return tensor_diff(out, ref)
    except Exception:  # noqa: BLE001  不可比较的输出跳过对拍
        return None


def run_ge(targets, out_dir="graph_dump/ge", dynamic=False, verbose=True):
    """对目标列表逐个做 GE 图导出（不做 GE 编译）。

    GE 通道要求 nn.Module（dynamo_export 接口 + 权重名映射）; entry 返回
    非 Module 的 callable 时跳过并说明。
    """
    report = {}
    for target in targets:
        module = target.ge_module or (
            target.fn if isinstance(target.fn, torch.nn.Module) else None)
        if module is None:
            if verbose:
                print(f"[dump_tool] [{target.name}] GE 跳过: 入口不是 "
                      "nn.Module（GE 通道要求 Module, 见 entries 文档）",
                      flush=True)
            report[target.name] = None
            continue
        with _wrap_contexts(target.contexts):
            args, kwargs = _normalize_inputs(target.make_inputs())
            report[target.name] = _safe(
                f"[{target.name}/ge]",
                lambda: export_ge_graph(
                    module, args, kwargs,
                    out_dir=os.path.join(out_dir, target.name),
                    export_name=target.name, dynamic=dynamic,
                    custom_decompositions=target.ge_decompositions,
                    verbose=verbose), verbose)
    return report
