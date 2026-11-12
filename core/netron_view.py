"""图 → Netron 可打开格式（.onnx 优先 / .pt 兜底）的导出链。

两类图都可转 Netron 格式:
  FX 图   （dynamo 层捕获的 gm）→ export_netron_view（本文件主链）
  GE 图   （GraphDef proto）   → export_ge_netron（GE 算子 → ONNX 节点,
          常见算子映射为标准 ONNX op, NPU 专属算子按原名透传——Netron
          对未识别 op 类型仍能渲染节点与连线）

为什么需要这一步: FX 的 .txt/.py 是 PyTorch 内部表示（textual IR /
生成代码）, GE 的 .txt 是 CANN GraphDef proto——Netron 都不识别;
ONNX / TorchScript 存档才是交换格式。

实测踩坑全记录（torch 2.10 + torch_npu 2.10 + CANN 9.2, SAM3 与
Qwen3-1.7B 全链验证; 每条都已内置处理）:
  A. NPU 输入被 torch.onnx 的 C++ shape 推断以 device mismatch 拒绝
     → 导出统一在 CPU 副本上执行（图结构设备无关）。
  B. dynamo 在 dynamic=False 下把 device 烤成图常量（如 torch.arange(
     0, 21, device=device(type='npu', index=0))）——输入搬到哪都会现场
     造原设备张量 → deepcopy 图上重写全部非 CPU device 常量后 recompile
     （aot 层的 FakeTensor 断言同源: fake trace 中该类 op 造出真实
     NPU 张量）。
  C. dynamo/aot 会把小张量字面常量（tensor(4, device='npu') 等）直接
     嵌进节点 args → 随 device 常量一起重写。
  D. gm 内联 buffer 常量（RoPE inv_freq 等）多为未注册的平铺属性,
     nn.Module.to() 不搬运 → deepcopy 后逐 __dict__ 搬 CPU。
  E. aot 层 fw_compiler 的 example_inputs 是 FakeTensor → 按其 meta
     在 CPU 造零值张量（aot 图已泛函化, 无值依赖分支, 结构导出安全）;
     dynamo 层 example_inputs 是真实张量 → 搬 CPU。
  F. aten._pin_memory 无 ONNX 映射——pin 是宿主内存属性, 对图结构
     无语义, 恒等消除（输入直通）。
  G. bf16 autocast 下捕获的图依赖运行时隐式类型转换 → CPU 导出执行
     需包同款 torch.autocast(device_type="cpu", dtype=bf16) 变体尝试。
  H. onnxscript(dynamo=True) 导出器: 内部起嵌套 dynamo(torch.export)
     ——不得在 dynamo backend 回调内执行（污染外层 eval-frame, 异常
     从外层炸出且无法捕获）; 且可能产出未拓扑排序 / 无生产者输入的
     残缺模型 → onnx_sanitize 校验不过即弃用。
  I. aarch64 多线程 BLAS 在部分环境死锁（"Bad memory unallocation"
     后 linear 挂死, C 层信号不可中断）→ 运行导出时建议
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1; 工具内另有 per-attempt
     alarm 超时兜底。
  J. torch 2.10 的 legacy ONNX 导出器需要 pip 包 onnx 序列化;
     dynamo 导出器需要 onnxscript——缺失时自动走后续回退。

设计要点:
  - 全程在 deepcopy 副本上导出, 原图与捕获产物不受任何影响;
  - 导出必须发生在 capture 调用结束之后（见 capture.py 的
    pending_exports 机制）;
  - 产物过 onnx.checker 校验, 校验不过直接删除（不交付坏文件）。
"""

import copy
import os

import torch
import torch.fx  # noqa: F401  确保 torch.fx.Node 等子模块可用

# CPU 输入缓存: 同一模型的权重张量会在每个子图的 example_inputs 里
# 重复出现, 逐图做 D2H 拷贝是导出主要耗时——按 id(tensor) 缓存 CPU
# 副本并持有原张量强引用防止 id 复用后错命中。
# 【勿用 WeakKeyDictionary】其查找的相等探测会触发 Tensor.__eq__,
# 多元素张量直接 RuntimeError("Boolean value of Tensor is ambiguous")。
_NETRON_INPUT_CACHE = {}  # id(tensor) -> (tensor, cpu_copy)

DEFAULT_ATTEMPT_TIMEOUT = 1200.0  # 单次尝试秒数上限（大图合法导出可很慢）


def _materialize_input(x, fake_cls):
    if not isinstance(x, torch.Tensor):
        return x
    if fake_cls and isinstance(x, fake_cls):
        return torch.zeros(tuple(x.shape), dtype=x.dtype, device="cpu")
    entry = _NETRON_INPUT_CACHE.get(id(x))
    if entry is not None and entry[0] is x:
        return entry[1]
    cpu_copy = x.detach().to("cpu")
    _NETRON_INPUT_CACHE[id(x)] = (x, cpu_copy)
    return cpu_copy


def _args_to_cpu(value):
    if isinstance(value, torch.device) and value.type != "cpu":
        return torch.device("cpu")
    if isinstance(value, torch.Tensor):
        # 节点 args 里内联的小张量字面常量（tensor(4, device='npu') 等）
        return value.detach().to("cpu")
    if isinstance(value, (list, tuple)):
        return type(value)(_args_to_cpu(v) for v in value)
    return value


def _normalize_graph_for_export(module):
    """导出前的图归一化（deepcopy 副本上, 不动原图）。"""
    for node in module.graph.nodes:
        new_args = tuple(_args_to_cpu(a) for a in node.args)
        new_kwargs = {k: _args_to_cpu(v) for k, v in node.kwargs.items()}
        if new_args != node.args or new_kwargs != node.kwargs:
            node.args = new_args
            node.kwargs = new_kwargs
    pin_op = torch.ops.aten._pin_memory.default
    for node in list(module.graph.nodes):
        if node.op == "call_function" and node.target is pin_op:
            src = node.args[0] if node.args else None
            if isinstance(src, torch.fx.Node):
                node.replace_all_uses_with(src)
                module.graph.erase_node(node)
    module.recompile()


def export_netron_view(gm, example_inputs, base, verbose=True,
                       attempt_timeout=DEFAULT_ATTEMPT_TIMEOUT):
    """把捕获的 FX 图导出为 Netron 可打开的格式, 返回产物路径或 None。

    export_params=False: 只导图结构不带权重（大模型权重数 GB, 看结构
    不需要; 权重以输入节点呈现）。链式回退（成功即返回, 逐层记录原因）:
        onnx-legacy → jit-cpu → onnx-legacy-autocast → jit-cpu-autocast
        → jit-orig（原设备 TorchScript 存档）→ onnx-dynamo（onnxscript,
        最慢且产物需校验, 放最后）。
    """
    try:
        from torch._subclasses.fake_tensor import FakeTensor
    except Exception:  # noqa: BLE001  老版本 torch 无该类
        FakeTensor = ()  # type: ignore[assignment]

    cpu_inputs = tuple(_materialize_input(x, FakeTensor) for x in example_inputs)
    errors = []
    try:
        gm_cpu = copy.deepcopy(gm)
        gm_cpu.to("cpu")
        for mod in gm_cpu.modules():
            for name, val in list(mod.__dict__.items()):
                if isinstance(val, torch.Tensor):
                    mod.__dict__[name] = val.detach().to("cpu")
        _normalize_graph_for_export(gm_cpu)
    except Exception as e:  # noqa: BLE001  归一化失败: 记录原因, 原图继续
        errors.append(f"normalize: {type(e).__name__}: {str(e)[:120]}")
        gm_cpu = gm

    def _attempt(fn, tag):
        """带 alarm 超时的一次导出尝试（仅主线程生效; C 层挂死无法中断,
        由整体运行超时兜底——建议配合 OPENBLAS_NUM_THREADS=1 使用）。"""
        use_alarm = True
        try:
            import signal

            def _handler(signum, frame):  # noqa: ARG001
                raise TimeoutError(f"attempt 超过 {attempt_timeout}s 被中断")

            old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, attempt_timeout)
        except (ValueError, OSError):  # 非主线程 / 平台不支持 signal
            use_alarm = False
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  记录后继续回退
            errors.append(f"{tag}: {type(e).__name__}: {str(e)[:160]}")
            return None
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

    def _legacy_onnx(autocast):
        def run():
            if autocast:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    torch.onnx.export(gm_cpu, cpu_inputs, base + ".onnx",
                                      export_params=False, dynamo=False)
            else:
                torch.onnx.export(gm_cpu, cpu_inputs, base + ".onnx",
                                  export_params=False, dynamo=False)
            return base + ".onnx"
        return run

    def _dynamo_onnx():
        torch.onnx.export(gm_cpu, cpu_inputs, base + ".onnx",
                          export_params=False, dynamo=True)
        return base + ".onnx"

    def _cpu_jit(autocast):
        def run():
            if autocast:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    traced = torch.jit.trace(gm_cpu, cpu_inputs)
            else:
                traced = torch.jit.trace(gm_cpu, cpu_inputs)
            traced.save(base + ".pt")
            return base + ".pt"
        return run

    def _orig_jit():
        traced = torch.jit.trace(gm, tuple(example_inputs))
        traced.save(base + ".pt")
        return base + ".pt"

    # 【注意】元组里必须放函数对象——放调用结果会在构造期执行, 异常
    # 逃出 _attempt 的捕获范围（实测踩过）。
    attempts = ((_legacy_onnx(False), "onnx-legacy"),
                (_cpu_jit(False), "jit-cpu"),
                (_legacy_onnx(True), "onnx-legacy-autocast"),
                (_cpu_jit(True), "jit-cpu-autocast"),
                (_orig_jit, "jit-orig"),
                (_dynamo_onnx, "onnx-dynamo"))
    for fn, tag in attempts:
        got = _attempt(fn, tag)
        if got is not None:
            if got.endswith(".onnx") and not onnx_sanitize(got):
                errors.append(f"{tag}: onnx checker 校验未通过, 已弃用")
                continue
            if verbose and errors:
                print(f"[dump_tool] netron 回退链: {' | '.join(errors)}")
            return got
    if verbose:
        print(f"[dump_tool] netron 导出失败（fx 图不受影响）: "
              f"{' | '.join(errors)}")
    return None


def onnx_sanitize(path):
    """ONNX 产物规范性处理: Kahn 拓扑排序 + checker 校验。

    onnxscript 产物常见两类问题: 未排序（forward reference）、以及内部
    DFG 值名泄漏导致的无生产者输入（结构性残缺, 排序无法修复）。校验
    不过 → 删除文件返回 False, 调用方继续回退——避免交付坏文件。
    """
    try:
        import onnx
        model = onnx.load(path)
        graph = model.graph
        available = ({i.name for i in graph.input}
                     | {ini.name for ini in graph.initializer})
        pending = list(graph.node)
        ordered = []
        while pending:
            rest = []
            progressed = False
            for node in pending:
                deps = {name for name in node.input if name}
                if deps <= available:
                    ordered.append(node)
                    available.update(node.output)
                    progressed = True
                else:
                    rest.append(node)
            if not progressed:
                break
            pending = rest
        if len(ordered) == len(list(graph.node)):
            del graph.node[:]
            graph.node.extend(ordered)
            onnx.save(model, path)
        onnx.checker.check_model(onnx.load(path))
        return True
    except Exception:  # noqa: BLE001  校验不过视为该次导出失败
        try:
            os.remove(path)
        except OSError:
            pass
        return False


# ---------- GE 图 → ONNX（Netron 可视化） ----------

_GE_TO_ONNX_OP = {
    "Add": "Add", "Sub": "Sub", "Mul": "Mul", "RealDiv": "Div",
    "MatMulV2": "MatMul", "BatchMatMul": "MatMul", "BatchMatMulV2": "MatMul",
    "SoftmaxV2": "Softmax", "Relu": "Relu", "Sigmoid": "Sigmoid",
    "Tanh": "Tanh", "Gelu": "Gelu", "Exp": "Exp", "Pow": "Pow",
    "Rsqrt": "Rsqrt", "Sqrt": "Sqrt", "Neg": "Neg", "Abs": "Abs",
    "Floor": "Floor", "Ceil": "Ceil", "Round": "Round",
    "Cast": "Cast", "Reshape": "Reshape", "Transpose": "Transpose",
    "Squeeze": "Squeeze", "Unsqueeze": "Unsqueeze",
    "ConcatV2": "Concat", "SplitV": "Split", "Slice": "Slice",
    "StridedSliceV2": "Slice", "GatherV2": "Gather",
    "ReduceMean": "ReduceMean", "ReduceSum": "ReduceSum",
    "ReduceMax": "ReduceMax", "ReduceMin": "ReduceMin",
    "Conv2D": "Conv", "MaxPoolV1": "MaxPool", "AvgPoolV1": "AveragePool",
    "MaxPoolWithArgmaxV1": "MaxPool", "PadV2": "Pad", "Fill": "ConstantOfShape",
    "Shape": "Shape", "ZerosLike": "ConstantOfShape",
}

_GE_DTYPE = {
    "DT_FLOAT": 1, "DT_INT32": 6, "DT_INT64": 7, "DT_UINT8": 2,
    "DT_INT8": 3, "DT_UINT16": 4, "DT_INT16": 5, "DT_BOOL": 9,
    "DT_FLOAT16": 10, "DT_BFLOAT16": 16, "DT_DOUBLE": 11,
}


def _ge_attr_value(v):
    """从 GE AttrValue 取基础类型值。"""
    for field in ("i", "f", "b", "s"):
        if v.HasField(field):
            return getattr(v, field)
    if v.HasField("list"):
        items = [getattr(x) for x in []]  # 占位, 下面按字段展开
        del items
        lst = v.list
        if len(lst.i):
            return list(lst.i)
        if len(lst.f):
            return list(lst.f)
    return None


def _ge_tensor_to_onnx(tensor_desc, ge_tensor):
    """GE TensorProto → ONNX TensorProto（无数据时造同 shape 零值占位）。

    Netron 只看结构, 占位数据可接受; dtype 取不到时默认 float。
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    dims = [d for d in tensor_desc.shape.dim] if tensor_desc.HasField("shape") else []
    dtype_str = tensor_desc.dtype
    np_dtype = {1: np.float32, 6: np.int32, 7: np.int64, 2: np.uint8,
                9: np.bool_, 10: np.float16, 11: np.float64}.get(
        _GE_DTYPE.get(str(dtype_str), 1), np.float32)
    data = np.zeros(dims, dtype=np_dtype)
    return numpy_helper.from_array(data)


def export_ge_netron(graph_def, onnx_path, verbose=True):
    """GE GraphDef proto → ONNX（Netron 可直接打开）。

    Data → 图输入, NetOutput → 图输出, Const → initializer, 其余 op 按映射表
    转标准 ONNX op; 无映射的 NPU 专属算子（FlashAttentionScore / BNInfer /
    LayerNormV4 等）按原名透传——Netron 对未识别 op 类型仍渲染节点连线。
    返回 True/False。
    """
    try:
        import onnx
        from onnx import TensorProto, helper

        graph_inputs, graph_outputs, inits, nodes = [], [], [], []
        net_output_refs = []

        for op in graph_def.op:
            t = op.type
            if t == "Data":
                index = op.attr["index"].i if "index" in op.attr else len(graph_inputs)
                name = op.name
                # shape/dtype 从 attr 的 TensorDescriptor 里取（尽力而为）
                dims, dtype = None, TensorProto.FLOAT
                if "output" in op.attr and op.attr["output"].HasField("t"):
                    desc = op.attr["output"].t.desc
                    dims = [d for d in desc.shape.dim]
                    dtype = _GE_DTYPE.get(str(desc.dtype), TensorProto.FLOAT)
                vi = helper.make_tensor_value_info(
                    f"{name}:0", dtype, dims or [None])
                vi.name = f"{name}:0"
                graph_inputs.append(vi)
                continue
            if t == "NetOutput":
                net_output_refs = list(op.input)
                continue
            if t == "Const":
                if "value" in op.attr and op.attr["value"].HasField("t"):
                    try:
                        tensor = _ge_tensor_to_onnx(
                            op.attr["value"].t.desc, op.attr["value"].t)
                        tensor.name = f"{op.name}:0"
                        inits.append(tensor)
                        continue
                    except Exception:  # noqa: BLE001  数据取不到 → 当输入占位
                        pass
                vi = helper.make_tensor_value_info(f"{op.name}:0",
                                                   TensorProto.UNDEFINED, None)
                graph_inputs.append(vi)
                continue

            onnx_type = _GE_TO_ONNX_OP.get(t, t)  # 无映射按原名透传
            inputs = [x if ":" in x else f"{x}:0" for x in op.input]
            outputs = [f"{op.name}:{i}" for i in range(
                len(op.attr["_output_name_value"].list.i)
                if "_output_name_value" in op.attr else 1)]
            if not outputs:
                outputs = [f"{op.name}:0"]
            attrs = {}
            for key in ("axis", "axes", "perm", "keepdims", "transpose_x1",
                        "transpose_x2", "alpha", "beta", "mode"):
                if key in op.attr:
                    val = _ge_attr_value(op.attr[key])
                    if val is not None:
                        attrs[key if key not in ("transpose_x1", "transpose_x2")
                              else key.replace("transpose_x", "trans")] = val
            try:
                node = helper.make_node(onnx_type, inputs, outputs,
                                        name=op.name, **attrs)
            except Exception:  # noqa: BLE001  attrs 类型不兼容时去 attrs
                node = helper.make_node(onnx_type, inputs, outputs,
                                        name=op.name)
            nodes.append(node)

        # 输出名引用修正: NetOutput 引用 "name:0" 形式
        for name in net_output_refs:
            vi = helper.make_tensor_value_info(name, TensorProto.UNDEFINED,
                                               None)
            graph_outputs.append(vi)

        graph = helper.make_graph(nodes, graph_def.name or "ge_graph",
                                  graph_inputs, graph_outputs, inits)
        model = helper.make_model(graph, opset_imports=[
            helper.make_opsetid("", 17)])
        model.ir_version = 8
        onnx.save(model, onnx_path)
        return True
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[dump_tool] GE→ONNX 失败: {type(e).__name__}: "
                  f"{str(e)[:160]}", flush=True)
        try:
            os.remove(onnx_path)
        except OSError:
            pass
        return False
