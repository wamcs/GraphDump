"""timm 卷积网络（非 transformers）的 dump 模板。

演示要点: 自定义权重加载（timm + safetensors）; GE 整图直达
（纯卷积网络无缺 converter 算子, 无需降级）。

用法（source 过 CANN set_env.sh 的 shell 里）:
    python dump_tool/examples/dump_resnet50.py
"""
import os
import sys

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
R50 = os.path.join(MODELS_ROOT, "timm_resnet50_a1_in1k")
for path in (MODELS_ROOT, f"{R50}/script"):
    if path not in sys.path:
        sys.path.insert(0, path)

import timm  # noqa: E402
import torch  # noqa: E402
from dump_tool import dump  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torchvision import transforms  # noqa: E402

torch.npu.set_device(0)
dev = "npu:0"
OUT = os.path.join(MODELS_ROOT, "dump_graph", "timm_resnet50_a1_in1k")

# 自定义加载: timm 建结构 + safetensors 权重
model = timm.create_model("resnet50.a1_in1k", pretrained=False, num_classes=1000)
model.load_state_dict(load_file(os.path.join(R50, "weights", "model.safetensors")),
                      strict=True)
model = model.to(dev).eval()

tfm = transforms.Compose([
    transforms.Resize(235, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.open(os.path.join(R50, "script", "sample.jpg")).convert("RGB")
x = tfm(img).unsqueeze(0).to(dev)

dump(model, x, out_dir=OUT, graphs=("fx", "ge"), netron=True, verbose=True)
