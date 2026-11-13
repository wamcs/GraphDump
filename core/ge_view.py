"""不做 GE 图编译, 仅"构造 + 序列化"拿到 GE 图（torchair export 通道）。

原理: torchair 的流程里 "FX→GE 图构造" 与 "GE 编译(graph.load/build)" 是
两个阶段。export_mode 下, GeConcreteGraph 构造完成后直接抛 ExportSuccess
优雅退出——`self.graph.load()`（GE 编译）永不执行。产物为未编译的 GE 图
序列化, 供离线分析（或后续经 atc 编译）。

实测（torch_npu 2.10 + CANN 9.2, Qwen3-1.7B prefill）: 51s 完成 vs 真实
GE 编译数分钟; 构造期已有轻量图变换（silu→Swish、mask→MaskedFill）,
aten 层 285 个 _to_copy 在 GE 图膨胀为 686 个 Cast（层间损耗的直接证据）。

产物:
    <out_dir>/<export_name>.air   未编译 GE 图（AIR 格式, 含权重）
    <out_dir>/<weight files>      AIR 外挂权重（按参数名独立成文件）
    <out_dir>/ge_dump/*.txt       GE 图 GraphDef 文本 proto（人类可读,
                                  torchair debug.graph_dump 通道）

限制:
    - dynamo_export 强制 fullgraph=True——图断裂的模型不可用。先用
      dump_tool.explain_breaks 确认 breaks=0;
    - 需要 CANN 环境: GE 引擎初始化要 import tbe（CANN 的 PYTHONPATH
      组件）——在 source 过 set_env.sh 的 shell 里跑, 且自定义 PYTHONPATH
      必须**追加**（A:B:C）而非覆盖, 否则报 "No module named 'tbe'"。
"""

import os

import torch


def capture_ge_graphs(fn, args=(), kwargs=None,
                      out_dir="graph_dump/ge_subgraphs", dynamic=False,
                      custom_decompositions=None, netron=True, verbose=True):
    """断裂容忍的 GE 图捕获（逐子图, 无编译无 GE 执行）。

    原理: torchair 的 FX→GE 构造发生在**编译期**（compiler 调用返回时
    GeConcreteGraph 已就绪）。把 torchair 编译器当 AOT fw_compiler 钩子:
    每个可捕获子图在钩子里构造 GE 图并 dump, 然后返回 gm.forward 透传
    执行（eager 真值）——GE 编译/执行完全旁路。

    与 export_ge_graph（fullgraph 整图 + .air）的区别:
      - dynamo fullgraph=False: 图断裂不致命, 断裂区走 eager, 每个可
        捕获子图各自得到一张 GE 图（断裂区的算子不在 GE 图内, 可用
        dispatch 层的 ops.log 补齐 ground truth）;
      - 算子级容错: 某子图缺 GE converter 时点名报错并跳过, 其它子图
        继续不受传染;
      - 分解表与真实 torchair 编译同口径（默认表 + custom_decompositions）。

    产物: <out_dir>/ge_graph_NN.txt（GraphDef 文本 proto, 含 NPU 算子信息）
    返回 (fn 执行结果, GE 图数, 各子图失败原因列表)。
    """
    import time

    import torch._dynamo as dynamo
    import torch_npu  # noqa: F401
    import torch_npu.dynamo  # noqa: F401  懒注册 torchair
    from torch._dynamo.backends.common import aot_autograd
    from torchair import CompilerConfig
    from torchair.npu_fx_compiler import get_compiler

    from .compat import bump_dynamo_limits, patch_trace_compat, restore_dynamo_limits

    def _merge_decompositions(config, custom):
        """版本自适应的分解表合并（与真实 torchair 编译同口径）。"""
        try:
            from torchair.npu_fx_compiler import _get_merged_decompositions
            return _get_merged_decompositions(config, custom)  # 2.10: 含 patch
        except ImportError:
            # torch_npu 2.6 等旧版: 手动合并（default + custom + npu patch）
            from torchair._utils import add_npu_patch, get_npu_default_decompositions
            decomps = get_npu_default_decompositions()
            decomps.update(custom or {})
            add_npu_patch(decomps, config)
            return decomps

    kwargs = kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    config = CompilerConfig()  # mode=max-autotune → GeConcreteGraph 路径
    decomps = _merge_decompositions(config, custom_decompositions)
    compiler = get_compiler(config)

    state = {"n": 0, "fails": []}

    def _fw(gm, example_inputs):
        state["n"] += 1
        idx = state["n"]
        try:
            compiled = compiler(gm, list(example_inputs))
            ge_concrete = getattr(compiled, "runner", None)
            if ge_concrete is None or not hasattr(ge_concrete, "dump"):
                raise RuntimeError(
                    f"未取到 GeConcreteGraph（got {type(compiled).__name__}, "
                    "torchair 版本结构变化?）")
            path = os.path.join(out_dir, f"ge_graph_{idx:02d}.txt")
            ge_concrete.dump(path)
            if verbose:
                print(f"[dump_tool] ge subgraph {idx} -> {path}", flush=True)
        except Exception as e:  # noqa: BLE001  算子级容错: 跳过该子图
            reason = f"子图{idx}: {type(e).__name__}: {str(e)[:200]}"
            state["fails"].append(reason)
            if verbose:
                print(f"[dump_tool] ge subgraph {idx} 失败（跳过, 其余继续）: "
                      f"{reason}", flush=True)
        return gm.forward

    patch_trace_compat()
    dynamo.reset()
    saved = bump_dynamo_limits()
    t0 = time.perf_counter()
    try:
        backend = aot_autograd(fw_compiler=_fw, decompositions=decomps)
        result = torch.compile(fn, backend=backend, fullgraph=False,
                               dynamic=dynamic)(*args, **kwargs)
    finally:
        restore_dynamo_limits(saved)
    if verbose:
        print(f"[dump_tool] ge subgraphs: {state['n'] - len(state['fails'])}/"
              f"{state['n']} 成功, {time.perf_counter() - t0:.1f}s -> {out_dir}",
              flush=True)
    if netron:
        from .netron_view import export_ge_netron
        for i in range(1, state["n"] + 1):
            txt_path = os.path.join(out_dir, f"ge_graph_{i:02d}.txt")
            onnx_path = os.path.join(out_dir, f"ge_graph_{i:02d}.onnx")
            if os.path.exists(txt_path):
                if export_ge_netron(_load_ge_proto(txt_path), onnx_path,
                                    verbose) and verbose:
                    print(f"[dump_tool] ge netron view -> {onnx_path}",
                          flush=True)
    return result, state["n"], state["fails"]


def _load_ge_proto(txt_path):
    """从 dump 的文本 proto 反序列化为 GraphDef 对象。"""
    from google.protobuf import text_format
    from torchair._ge_concrete_graph import ge_ir_pb2

    gd = ge_ir_pb2.GraphDef()
    with open(txt_path, "r", encoding="utf-8") as f:
        text_format.Parse(f.read(), gd)
    return gd


def export_ge_graph(model, args=(), kwargs=None, out_dir="graph_dump/ge",
                    export_name="graph", dynamic=False, config=None,
                    custom_decompositions=None, netron=True, verbose=True):
    """对任意 nn.Module 不做 GE 编译地导出 GE 图（fullgraph 整图 + .air）。

    custom_decompositions: {OpOverload: 等价分解函数}——经 torchair
    get_npu_backend 的 custom_decompositions 通道注入。适用于含缺 GE
    converter 算子的模型（需先做等价分解, 如 SAM3 的 s302 分解表）;
    提供时走与 dynamo_export 同构的自建 backend 流程。

    返回 dict: {"air": .air 路径或 None, "ge_dump": txt proto 目录,
                "wall": 秒}。失败抛原始异常（缺 converter / fullgraph
    失败等——错误本身就是"该模型离 GE 编译还差什么"的答案）。
    """
    import time

    import torch_npu  # noqa: F401
    import torch_npu.dynamo  # noqa: F401  懒注册 torchair
    from torchair import CompilerConfig

    from .compat import patch_trace_compat

    kwargs = kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    if config is None:
        config = CompilerConfig()
    config.debug.graph_dump.type = "txt"
    config.debug.graph_dump.path = os.path.join(out_dir, "ge_dump")
    patch_trace_compat()

    if verbose:
        print(f"[dump_tool] ge export: {type(model).__name__} -> {out_dir} "
              f"（export_mode: 构造后抛 ExportSuccess, graph.load() 不执行）",
              flush=True)
    t0 = time.perf_counter()
    exported = False
    if custom_decompositions is None:
        from torchair import dynamo_export
        with torch.no_grad():
            try:
                dynamo_export(*args, model=model, export_path=out_dir,
                              export_name=export_name, dynamic=dynamic,
                              config=config, **kwargs)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "ExportSuccess":
                    exported = True
                else:
                    raise
    else:
        # 与 dynamo_export 同构（torchair/npu_export.py）, 仅注入自定义分解表
        from torchair import get_npu_backend
        from torchair._ge_concrete_graph.fx2ge_converter import ExportSuccess

        config.export.export_mode = True
        config.export.export_path_dir = out_dir
        config.export.export_name = export_name
        config.experimental_config.enable_ref_data = True
        weight_name = {}
        for param_name, param in model.named_buffers():
            weight_name[id(param)] = param_name
        for param_name, param in model.named_parameters():
            weight_name[id(param)] = param_name
        config.export.weight_name = weight_name

        import torch._dynamo as dynamo
        dynamo.reset()
        compiled = torch.compile(
            model, backend=get_npu_backend(
                compiler_config=config,
                custom_decompositions=custom_decompositions),
            fullgraph=True, dynamic=dynamic)
        with torch.no_grad():
            try:
                compiled(*args, **kwargs)
            except ExportSuccess:
                exported = True
    air_path = os.path.join(out_dir, export_name + ".air")
    result = {
        "air": air_path if os.path.exists(air_path) else None,
        "ge_dump": os.path.join(out_dir, "ge_dump"),
        "wall": time.perf_counter() - t0,
        "exported": exported or os.path.exists(air_path),
    }
    if verbose:
        print(f"[dump_tool] ge export done: {result['wall']:.1f}s, "
              f"air={'有' if result['air'] else '无'}, "
              f"txt={result['ge_dump']}", flush=True)
    # GE 图 → Netron 格式（ge_dump 下的 txt proto 解析回 GraphDef 再转 ONNX）
    if netron:
        import glob as _glob

        from .netron_view import export_ge_netron
        for txt in sorted(_glob.glob(
                os.path.join(result["ge_dump"], "*.txt"))):
            base = os.path.splitext(txt)[0]
            onnx_path = base + ".onnx"
            try:
                if export_ge_netron(_load_ge_proto(txt), onnx_path, verbose):
                    if verbose:
                        print(f"[dump_tool] ge netron view -> {onnx_path}",
                              flush=True)
            except Exception as e:  # noqa: BLE001  转换失败不影响已有产物
                if verbose:
                    print(f"[dump_tool] ge netron 转换失败（{os.path.basename(txt)}"
                          f"）: {type(e).__name__}: {str(e)[:120]}", flush=True)
    return result
