"""非 HF 模型的 dump 模板（SAM3 实例, 实测可用）——复制本文件, 改三处。

工具唯一入口是 dump_tool.dump(); 它缺的信息就三条, 全部在下面声明:
  ① 模型怎么来        → model = ...（你的加载代码）
  ② 喂什么输入        → make_inputs()（model(xxx) 的 xxx; 状态型每层重建）
  ③ 执行上下文（可选） → ctx = (...)

用法（source 过 CANN set_env.sh 的 shell 里, PYTHONPATH 追加 models/）:
    python examples/dump_sam3.py            # dump + GE 图
"""

import os
import sys

# 模板不含任何机器绝对路径: 默认相对本文件定位（dump_tool/examples/ 的上上级
# 即模型仓库根目录）, 也可用环境变量 DUMP_MODELS_ROOT 显式指定。
MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
SAM_ROOT = os.environ.get("SAM3_ROOT", os.path.join(MODELS_ROOT, "SAM_3_test"))
for path in (MODELS_ROOT, SAM_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("SAM3_COMPILE", "1")  # 编译态 patch: fullgraph 干净

import torch  # noqa: E402
import torch_npu  # noqa: E402
from scripts.npu_compat import *  # noqa: F401,F403,E402  NPU shim, 须在 sam3 前

from run_inference import synthesize_image  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from dump_tool import dump  # noqa: E402

WEIGHTS = os.path.join(SAM_ROOT, "weights", "sam3.pt")
BPE = os.path.join(SAM_ROOT, "original_script", "sam3", "assets",
                   "bpe_simple_vocab_16e6.txt.gz")


class FgModule(torch.nn.Module):
    """dump 入口 = forward_grounding（GE 通道要求 nn.Module）。"""

    def __init__(self, sam3_model):
        super().__init__()
        self.model = sam3_model

    def forward(self, backbone_out, find_input, geometric_prompt):
        return self.model.forward_grounding(
            backbone_out=backbone_out, find_input=find_input,
            find_target=None, geometric_prompt=geometric_prompt)


# ① 模型怎么来
sam3 = build_sam3_image_model(bpe_path=BPE, checkpoint_path=WEIGHTS,
                              load_from_HF=False, device="cuda")
processor = Sam3Processor(sam3)
model = FgModule(sam3).eval()


# ② 喂什么输入（每个捕获层各调用一次; eager 准备 backbone_out 等状态）
def make_inputs():
    with torch.inference_mode(), torch.autocast(device_type="cuda",
                                                dtype=torch.bfloat16):
        fresh = processor.set_text_prompt(
            "circle", processor.set_image(synthesize_image()))
    return (), dict(backbone_out=fresh["backbone_out"],
                     find_input=processor.find_stage,
                     geometric_prompt=fresh["geometric_prompt"])


# ③ 执行上下文（与官方 run_inference.py 的执行条件一致）
ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16),
       torch.inference_mode)


def ge_decompositions():
    """GE 图需要: 缺 converter 算子的等价分解表（s302 同款九项）。"""
    from scripts.s302_compile import (_build_extra_decompositions,
                                      _build_sdpa_decompositions)
    decomps = _build_sdpa_decompositions()
    decomps.update(_build_extra_decompositions())
    return decomps


if __name__ == "__main__":
    do_dump = "--dump" in sys.argv or len(sys.argv) == 1
    do_ge = "--ge" in sys.argv or len(sys.argv) == 1
    if do_dump:
        dump(model, inputs=make_inputs, ctx=ctx,
             out_dir=os.path.join(SAM_ROOT, "graph_dump", "sam3_dump_tool"))
    if do_ge:
        dump(model, inputs=make_inputs, ctx=ctx, ge=True,
             ge_decompositions=ge_decompositions(),
             out_dir=os.path.join(SAM_ROOT, "graph_dump", "sam3_dump_tool_ge"))
