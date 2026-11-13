"""ESM2-650M 蛋白质语言模型（HF MaskedLM）的 dump 模板。

演示要点: 大模型（650M, 33 层）; MaskedLM（非 CausalLM）——工具不关心
模型类型, 只要能 model(input_ids=...) 就能 dump。
"""
import os
import sys

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
ESM2 = os.path.join(MODELS_ROOT, "facebook_esm2_t33_650m_ur50d")
sys.path.insert(0, MODELS_ROOT)

import torch  # noqa: E402
from dump_tool import dump  # noqa: E402
from transformers import AutoModelForMaskedLM, AutoTokenizer  # noqa: E402

torch.npu.set_device(0)
dev = "npu:0"
OUT = os.path.join(MODELS_ROOT, "dump_graph", "facebook_esm2_t33_650m_ur50d")

tokenizer = AutoTokenizer.from_pretrained(os.path.join(ESM2, "weights"))
model = AutoModelForMaskedLM.from_pretrained(
    os.path.join(ESM2, "weights")).to(dev).eval()

# 蛋白质序列（照抄项目的测试数据模式）
seq = "MKTVLITLAGLLTVLAVLGVRVGGQAQAEAAVRLGADVAATGAGARVFAEVLADSGA"
inputs = tokenizer(seq, return_tensors="pt").to(dev)
print(f"[esm2] seq_len={inputs['input_ids'].shape[1]}", flush=True)

dump(model, **inputs, out_dir=OUT, graphs=("fx", "ge"),
     netron=True, verbose=True)
