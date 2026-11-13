"""CLI 入口: python -m dump_tool dump [options]。

唯一命令, 仅支持 HF CausalLM（--weights <HF 权重目录>）与自测（--demo）。
非 HF 模型请用 Python 形态（模板见 examples/dump_sam3.py）——shell 传不了
张量, 且加载/输入方式只有模型作者知道。
"""

import argparse
import os

import torch

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}


def main(argv=None):
    import sys
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "dump":
        argv = argv[1:]  # 兼容 "python -m dump_tool dump" 与 "... dump_tool"

    parser = argparse.ArgumentParser(
        prog="dump_tool",
        description="通用计算图 dump 工具（模型无关 / 设备无关 / 不要求能图编译）")
    parser.add_argument("--weights", default=None,
                        help="HF 权重目录（CLI 仅支持 HF CausalLM）")
    parser.add_argument("--demo", action="store_true",
                        help="自测小模型（代替 --weights）")
    parser.add_argument("--prompt", default="请介绍一下你自己。",
                        help="文本提示")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "npu"])
    parser.add_argument("--dtype", default="bfloat16", choices=list(_DTYPE))
    parser.add_argument("--attn", default="eager", choices=["eager", "sdpa"],
                        help="建议 eager: 纯 aten 路径, 图最显式")
    parser.add_argument("--entries", default="prefill,decode",
                        help="逗号分隔: prefill,decode")
    parser.add_argument("--modes", default="dynamo,aot,dispatch",
                        help="逗号分隔: dynamo,aot,dispatch")
    parser.add_argument("--netron", dest="netron", action="store_true",
                        default=True)
    parser.add_argument("--no-netron", dest="netron", action="store_false",
                        help="只要文本产物（大模型 ONNX 导出较慢）")
    parser.add_argument("--no-explain", dest="explain", action="store_false",
                        default=True)
    parser.add_argument("--ge", action="store_true", default=False,
                        help="GE 图导出（不做 GE 编译; 需要 CANN 环境, "
                             "入口须 nn.Module 且 fullgraph 可 trace）")
    parser.add_argument("--dynamic", action="store_true", default=False,
                        help="（ge）动态 shape")
    parser.add_argument("--out-dir", default="graph_dump")
    cli = parser.parse_args(argv)

    if not cli.demo and not cli.weights:
        parser.error(
            "需要 --weights <HF 权重目录>（或 --demo 自测）。"
            "非 HF 模型请用 Python 形态: from dump_tool import dump; "
            "模板见 dump_tool/examples/dump_sam3.py")

    device = cli.device
    if device == "auto":
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                device = "npu"
        except Exception:  # noqa: BLE001
            device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "npu":
        import torch_npu  # noqa: F401
        torch.npu.set_device(0)

    from .entries import load_entry
    if cli.demo:
        targets = load_entry("demo", None, device=cli.device)
    else:
        targets = load_entry(
            "hf", cli.weights, prompt=cli.prompt, device=cli.device,
            dtype=_DTYPE[cli.dtype], attn_implementation=cli.attn,
            entries=tuple(e for e in cli.entries.split(",") if e))

    if cli.ge:
        from .entries import run_dump, run_ge
        # --ge 模式: 同时跑 GE 图导出 + FX/dispatch 补充信息
        run_ge(targets, out_dir=cli.out_dir, dynamic=cli.dynamic)
        run_dump(targets, out_dir=cli.out_dir,
                 modes=("dynamo", "dispatch"), netron=cli.netron,
                 with_explain=cli.explain)
    else:
        from .entries import run_dump
        run_dump(targets, out_dir=cli.out_dir,
                 modes=tuple(m for m in cli.modes.split(",") if m),
                 netron=cli.netron, with_explain=cli.explain)

    _list_files(cli.out_dir)


def _list_files(out_dir):
    print("[dump_tool] 落盘清单:", flush=True)
    for root, _, files in sorted(os.walk(out_dir)):
        for name in sorted(files):
            print(f"  {os.path.join(root, name)}", flush=True)


if __name__ == "__main__":
    main()
