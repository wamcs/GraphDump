"""entry 文件接入: 任意本地可运行模型经约定接口自动 dump。

entry 文件契约（详见 entries/__init__.py 模块 docstring）:
    load(weights, device=None, dtype=None) -> model   必需
    inputs(model) -> (args, kwargs) | (args,) | args   必需（每层重建）
    NAME = "..."                                       可选（输出子目录名）
    contexts() -> 上下文工厂元组                         可选（默认 no_grad）
    ge_decompositions() -> dict | None                  可选（GE 分解表）

entry 文件自行管理 sys.path / os.environ / import 顺序——工具对模型的
世界不做任何假设（包括"必须在 import 模型库前设的环境变量"）。
"""

import importlib.util
import os
import sys

import torch

from . import DumpTarget


def _load_entry_module(path):
    module_name = "dump_tool_user_entry"
    spec = importlib.util.spec_from_file_location(module_name,
                                                  os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_targets(path, weights, device=None, dtype=None, **entry_kwargs):
    """加载 entry 文件并构建单目标。返回 list[DumpTarget]。"""
    module = _load_entry_module(path)

    if not hasattr(module, "load") or not hasattr(module, "inputs"):
        raise AttributeError(
            f"entry 文件 {path} 缺少约定接口: 需要 load(weights, device, "
            "dtype) 与 inputs(model)（见 dump_tool/entries/__init__.py 的"
            "契约说明）")

    model = module.load(weights, device=device, dtype=dtype,
                        **entry_kwargs)
    name = getattr(module, "NAME", None) or os.path.splitext(
        os.path.basename(path))[0]
    contexts = getattr(module, "contexts", None)
    if callable(contexts):
        contexts = contexts()
    if not contexts:
        contexts = (torch.no_grad,)

    ge_decompositions = None
    if hasattr(module, "ge_decompositions"):
        got = module.ge_decompositions()
        if isinstance(got, dict):
            ge_decompositions = got

    def _make_inputs():
        return module.inputs(model)

    print(f"[dump_tool] script entry: {path} -> {type(model).__name__} "
          f"(target={name})", flush=True)
    return [DumpTarget(name=name, fn=model, make_inputs=_make_inputs,
                       contexts=contexts,
                       ge_decompositions=ge_decompositions)]
