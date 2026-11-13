"""dynamo trace 兼容 patch 集与限额管理（全部幂等, 不触碰数值路径）。

这些是"在真实 NPU + 大模型上做图捕获"实测踩出的框架层阻断, 逐个定位
并做最小化修复; 对 CUDA/CPU 环境行为不变, 对老版本 torch（相关内部
模块不存在时）防御式跳过。

实测环境:
  - torch 2.10.0 + torch_npu 2.10.0.post4 + CANN 9.2
    （SAM3 与 Qwen3-1.7B 全链验证）
  - torch 2.1.0 + torch_npu 2.1.0.post6
    （patch 目标模块不存在 → 防御式跳过, 核心捕获路径可用）

patch 清单:
  1) torch.utils._triton._device_supports_tma 在 NPU 上执行
     None >= (9, 0)（transfer_to_npu 后 torch.cuda.get_device_capability()
     返回 None）→ TypeError 中断 dynamo builder。补 None 防护（NPU 无
     CUDA TMA, 返回 False 语义正确）。
  2) aten._pin_memory.default/.out 在 fake 实现表中被注册为 nyi 桩——
     trace 域内任何 .pin_memory()（H2D 预取类优化代码常见）直接 NYI
     断言。恒等返回（pin 是宿主内存属性, fake tensor 无真实存储,
     元数据层恒等语义正确）。
  3) dynamo LazyVariableTracker.realize_all 的 memo 在递归返回后才
     写入——输入结构含环引用（如运行时缓存把容器挂回自身）时无限递归
     RecursionError。改为可变容器先登记占位再递归（对无环结构逐位
     不变）。
"""

import sys

import torch
import torch._dynamo as dynamo

_PATCHED_FLAG = "_dump_tool_safe"


def _patch_npu_triton_capability_check():
    try:
        from torch.utils import _triton
    except Exception:  # noqa: BLE001  老版本 torch 无该模块时无需 patch
        return
    original = getattr(_triton, "_device_supports_tma", None)
    if original is None:
        # torch 2.6 等中间版本: 模块存在但无该函数（2.7+ 引入）——
        # builder 不走该路径, 无需 patch。勿直接属性访问（实测会 AttributeError）。
        return
    if getattr(original, _PATCHED_FLAG, False):
        return

    def _safe_device_supports_tma():
        try:
            cap = torch.cuda.get_device_capability()
        except Exception:  # noqa: BLE001  无 cuda / 查询失败均视为不支持
            return False
        return cap is not None and bool(cap >= (9, 0)) and not torch.version.hip

    _safe_device_supports_tma.__dict__[_PATCHED_FLAG] = True
    _triton._device_supports_tma = _safe_device_supports_tma
    for name in ("has_triton_experimental_host_tma",
                 "has_triton_tensor_descriptor_host_tma", "has_triton_tma"):
        try:
            getattr(_triton, name).cache_clear()
        except Exception:  # noqa: BLE001  无 cache 属性时忽略
            pass


def _patch_pin_memory_fake_impl():
    try:
        from torch._subclasses import fake_impls as _fake_impls
        op_default = torch.ops.aten._pin_memory.default
        op_out = torch.ops.aten._pin_memory.out
        impls_dict = _fake_impls.op_implementations_dict
        nyi_fn = _fake_impls.nyi  # 属性结构随版本变化, 一并纳入防护
    except Exception:  # noqa: BLE001  老版本 torch 结构不同时跳过
        return

    def _pin_memory_fake(fake_mode, func, *args, **kwargs):  # noqa: ANN001,ARG001
        if func is op_out:
            out = kwargs.get("out")
            if out is None and len(args) > 1:
                out = args[1]
            return out if out is not None else args[0]
        return args[0]

    for op in (op_default, op_out):
        current = impls_dict.get(op)
        if current is not None and current is not nyi_fn:
            continue  # 已有真实实现（如新版 torch 补上后）, 不覆盖
        impls_dict[op] = _pin_memory_fake


def _patch_realize_all_cycles():
    try:
        import collections as _collections

        from torch._dynamo.variables import lazy as _lazy_mod

        cls = _lazy_mod.LazyVariableTracker
        if getattr(cls.realize_all.__func__, _PATCHED_FLAG, False):
            return
    except Exception:  # noqa: BLE001  老版本 torch 无该模块时跳过
        return

    def _realize_all(cls_, value, cache=None):
        if cache is None:
            cache = {}
        idx = id(value)
        if idx in cache:
            return cache[idx][0]
        value_cls = type(value)
        if issubclass(value_cls, _lazy_mod.LazyVariableTracker):
            result = cls_.realize_all(value.realize(), cache)
        elif issubclass(value_cls, _lazy_mod.VariableTracker):
            result = value
            cache[idx] = (result, value)  # 原地更新, 可先登记断环
            for key, item in value.__dict__.items():
                if key not in value._nonvar_fields:
                    value.__dict__[key] = cls_.realize_all(item, cache)
        elif value_cls is list:
            result = []
            cache[idx] = (result, value)  # 先登记, 断环回边
            for item in value:
                result.append(cls_.realize_all(item, cache))
        elif value_cls is tuple:
            result = tuple(cls_.realize_all(v, cache) for v in value)
        elif value_cls in (dict, _collections.OrderedDict):
            result = {}
            cache[idx] = (result, value)  # 先登记, 断环回边
            for k, v in list(value.items()):
                result[k] = cls_.realize_all(v, cache)
        else:
            result = value
        cache[idx] = (result, value)
        return result

    _realize_all.__dict__[_PATCHED_FLAG] = True
    cls.realize_all = classmethod(_realize_all)


def patch_trace_compat():
    """dynamo trace 前置兼容 patch 集（幂等, 不触碰数值路径）。"""
    _patch_npu_triton_capability_check()
    _patch_pin_memory_fake_impl()
    _patch_realize_all_cycles()
    _patch_transformers_capture_outputs()


def _patch_transformers_capture_outputs():
    """transformers 5.x 的 capture_outputs 装饰器兼容（幂等）。

    该装饰器包裹所有 HF 模型的 forward, 内部用 ContextVar + 动态 scope
    引用 torch——dynamo trace 时报 NameError: name 'torch' is not defined
    （torch 只在 TYPE_CHECKING 块里导入, 不在运行时 globals 里）。

    修复: 给 output_capturing 模块注入 torch 引用, 让 dynamo 能解析。
    capture_outputs 只影响 output_hidden_states/attentions 等可选捕获,
    不影响计算逻辑——对图 dump 无语义影响。
    """
    try:
        import transformers.utils.output_capturing as _oc
        if not hasattr(_oc, "torch"):
            _oc.torch = torch  # 注入缺失的 torch 引用
    except Exception:  # noqa: BLE001  非 transformers 5.x 时跳过
        pass


def bump_dynamo_limits():
    """临时放大 dynamo 重编译上限与 Python 递归上限（dump 场景专用）。

    大模型深嵌套的输入 dict/list 会让 dynamo 的 lazy realize_all 递归
    超过默认 recursionlimit(1000) → RecursionError 中断 trace; dump 场景
    还要求多 shape 族（重编译）图全部落盘。返回 (saved,) 供 restore。
    """
    cfg = dynamo.config
    saved = (cfg.cache_size_limit, cfg.accumulated_cache_size_limit,
             sys.getrecursionlimit())
    cfg.cache_size_limit = 64
    cfg.accumulated_cache_size_limit = 1024
    if sys.getrecursionlimit() < 10000:
        sys.setrecursionlimit(10000)
    return saved


def restore_dynamo_limits(saved):
    """恢复 bump_dynamo_limits 保存的限额。"""
    cfg = dynamo.config
    cfg.cache_size_limit, cfg.accumulated_cache_size_limit, recursion = saved
    sys.setrecursionlimit(recursion)
