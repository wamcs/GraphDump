"""非 HF 模型（SAM3）的 dump 模板——自定义大模型完整接入示例。

演示要点:
  - 状态型输入: make_inputs 每次调用重建（eager 跑 set_image/set_text_prompt,
    留下 forward_grounding 所需的 backbone_out / geometric_prompt 状态）
  - bf16 autocast ctx（与官方 run_inference.py 的执行条件一致）
  - GE 分解表: 缺 GE converter 的 9 个算子的等价分解（s302 同款）
  - fullgraph 前置环境变量（SAM3_COMPILE=1 跳过断图 patch）

用法（source 过 CANN set_env.sh 的 shell 里, PYTHONPATH 追加 models/）:
    python dump_tool/examples/dump_sam3.py
"""
import os
import sys

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
SAM_ROOT = os.environ.get("SAM3_ROOT", os.path.join(MODELS_ROOT, "SAM_3_test"))
for path in (MODELS_ROOT, SAM_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
# 编译态 patch 集: S2-03 断图缓存跳过(否则 fullgraph 必断) + §15 pin 消除
os.environ.setdefault("SAM3_COMPILE", "1")

import torch  # noqa: E402
from dump_tool import dump  # noqa: E402
from run_inference import synthesize_image  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from scripts.npu_compat import *  # noqa: F401,F403,E402  NPU shim, 须在 sam3 前

BPE = os.path.join(SAM_ROOT, "original_script", "sam3", "assets",
                   "bpe_simple_vocab_16e6.txt.gz")
WEIGHTS = os.environ.get("SAM3_CKPT",
                         os.path.join(SAM_ROOT, "weights", "sam3.pt"))

_STATE = {}


class FgModule(torch.nn.Module):
    """dump 入口 = forward_grounding（GE 通道要求 nn.Module）。"""

    def __init__(self, sam3_model):
        super().__init__()
        self.model = sam3_model

    def forward(self, backbone_out, find_input, geometric_prompt):
        return self.model.forward_grounding(
            backbone_out=backbone_out, find_input=find_input,
            find_target=None, geometric_prompt=geometric_prompt)


# ① 模型加载
sam3 = build_sam3_image_model(bpe_path=BPE, checkpoint_path=WEIGHTS,
                              load_from_HF=False, device="cuda")
processor = Sam3Processor(sam3)
model = FgModule(sam3).eval()
_STATE["processor"] = processor


# ② 输入工厂: 每次调用重建状态（forward_grounding 消费 backbone_out）
def make_inputs():
    with torch.inference_mode(), torch.autocast(device_type="cuda",
                                                dtype=torch.bfloat16):
        fresh = processor.set_text_prompt(
            "circle", processor.set_image(synthesize_image()))
    return (), dict(backbone_out=fresh["backbone_out"],
                     find_input=processor.find_stage,
                     geometric_prompt=fresh["geometric_prompt"])


# ③ 执行上下文
ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16),
       torch.inference_mode)


def ge_decompositions():
    """GE 需要的等价分解表（缺 converter 算子, s302 同款九项）。"""
    from scripts.s302_compile import (
        _build_extra_decompositions,
        _build_sdpa_decompositions,
    )
    decomps = _build_sdpa_decompositions()
    decomps.update(_build_extra_decompositions())
    return decomps


if __name__ == "__main__":
    OUT = os.path.join(MODELS_ROOT, "dump_graph", "sam3")
    dump(model, inputs=make_inputs, ctx=ctx, out_dir=OUT,
         graphs=("fx", "ge"), ge_decompositions=ge_decompositions(),
         netron=True, verbose=True)
