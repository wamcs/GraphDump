"""Stable Diffusion v1.4 UNet 的 dump 模板。

演示要点: 扩散模型 UNet（860M 参数, ResBlock+Attention+GroupNorm）;
latent 空间输入 + timestep 条件; 加载用 StableDiffusionPipeline → 取 .unet。
"""
import os
import sys

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
SD = os.path.join(MODELS_ROOT, "compvis_stable_diffusion_v1_4")
sys.path.insert(0, MODELS_ROOT)

import torch  # noqa: E402
from dump_tool import dump  # noqa: E402

torch.npu.set_device(0)
dev = "npu:0"
OUT = os.path.join(MODELS_ROOT, "dump_graph", "compvis_stable_diffusion_v1_4")

# 加载（照抄项目 inference.py: pipeline → 取 unet）
from diffusers import StableDiffusionPipeline  # noqa: E402

pipe = StableDiffusionPipeline.from_pretrained(
    os.path.join(SD, "weights"), torch_dtype=torch.float16, variant="fp16")
unet = pipe.unet.to(dev).eval()
del pipe.vae, pipe.safety_checker  # 释放不需要的组件

# 构造输入: latent + timestep + text embedding
torch.manual_seed(42)
text_embeds = torch.randn(1, 77, 768, device=dev, dtype=torch.float16)
latent = torch.randn(1, 4, 64, 64, device=dev, dtype=torch.float16)
timestep = torch.tensor([500], device=dev)

print(f"[sd_unet] latent={tuple(latent.shape)} t={timestep.item()} "
      f"text={tuple(text_embeds.shape)}", flush=True)

dump(unet, sample=latent, timestep=timestep,
     encoder_hidden_states=text_embeds,
     out_dir=OUT, graphs=("fx", "ge"), netron=True, verbose=True)
