"""HF 音频模型（Wav2Vec2）的 dump 模板。

演示要点: 音频特征输入; transformers v5 的 lazy import 会断图——
GE 整图导出失败时自动降级逐子图捕获的实例。

用法（source 过 CANN set_env.sh 的 shell 里）:
    python dump_tool/examples/dump_wav2vec2.py
"""
import os
import sys
import wave

import numpy as np

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
W2V2 = os.path.join(MODELS_ROOT, "facebook_wav2vec2_base")
for path in (MODELS_ROOT, f"{W2V2}/script"):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
from common import BENCHMARK_WAV, load_model  # noqa: E402
from dump_tool import dump  # noqa: E402


def load_wav_16k(path):
    """照抄项目 run.py 的音频读取方式。"""
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return arr


torch.npu.set_device(0)
OUT = os.path.join(MODELS_ROOT, "dump_graph", "facebook_wav2vec2_base")

audio = load_wav_16k(BENCHMARK_WAV)
print(f"[wav2vec2] audio: {len(audio)/16000:.2f}s", flush=True)

model, extractor = load_model("npu:0")
inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to("npu:0") for k, v in inputs.items()}

dump(model, **inputs, out_dir=OUT, graphs=("fx", "ge"),
     netron=True, verbose=True)
