"""HF 视觉 backbone 模型（DINOv2）的 dump 模板。

演示要点: 图像预处理输入; fake-op 栈限制下 dynamo/aot 失败时自动降级
dispatch 兜底（ground truth 仍可得）。

用法（source 过 CANN set_env.sh 的 shell 里）:
    python dump_tool/examples/dump_dinov2.py
"""
import os
import sys

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DINO = os.path.join(MODELS_ROOT, "facebook_dinov2_base")
for path in (MODELS_ROOT, DINO):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
import torch_npu  # noqa: E402
from run_inference import load_model  # noqa: E402
from scripts.test_data import build_sample  # noqa: E402

from dump_tool import dump  # noqa: E402

torch.npu.set_device(0)
OUT = os.path.join(MODELS_ROOT, "dump_graph", "facebook_dinov2_base")

model = load_model("npu:0")
# 踩坑提示: 项目的样例构造返回 CPU 张量, 输入设备须自证
sample = build_sample("img_real_bs1").to("npu:0")
print(f"[dinov2] sample: {tuple(sample.shape)} on {sample.device}", flush=True)

dump(model, pixel_values=sample, out_dir=f"{OUT}/model",
     modes=("dynamo", "aot", "dispatch"), netron=True, verbose=True)
dump(model, pixel_values=sample, ge=True, out_dir=f"{OUT}/model_ge",
     verbose=True)
