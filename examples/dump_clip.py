"""CLIP ViT-B/32 多模态模型（OpenAI 原始实现, 非 HF）的 dump 模板。

演示要点: 非标准加载（OpenAI clip 包 + JIT 归档权重）; 双输入
（图像 + 文本 token）; fp16 混合精度模型。
"""
import os
import sys

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
CLIP_DIR = os.path.join(MODELS_ROOT, "openai_clip_vit_base_patch32")
sys.path.insert(0, MODELS_ROOT)
sys.path.insert(0, os.path.join(CLIP_DIR, "scripts", "openai_clip"))

import clip  # noqa: E402
import torch  # noqa: E402
from dump_tool import dump  # noqa: E402
from PIL import Image  # noqa: E402

torch.npu.set_device(0)
dev = "npu:0"
OUT = os.path.join(MODELS_ROOT, "dump_graph", "openai_clip_vit_base_patch32")

# 加载（照抄项目的 run_inference.py）
model, preprocess = clip.load(
    os.path.join(CLIP_DIR, "weights", "ViT-B-32.pt"), device=dev, jit=False)
model.eval()

image = preprocess(Image.open(os.path.join(CLIP_DIR, "data", "sample.jpg"))
                   ).unsqueeze(0).to(dev)
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(dev)
print(f"[clip] image={tuple(image.shape)} text={tuple(text.shape)}", flush=True)

# CLIP forward: 接受 image + text 两个输入
dump(model, image=image, text=text, out_dir=OUT, graphs=("fx", "ge"),
     netron=True, verbose=True)
