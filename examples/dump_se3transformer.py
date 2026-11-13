"""SE3Transformer（NVIDIA 等变 Transformer, 非 HF）的 dump 模板。

演示要点: dgl 算子 patch 后的 NPU 兼容实现; 图结构 + 等变约束;
复杂的项目内 setup（se3_npu.compat / dgl_ops）。
"""
import os
import sys

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
SE3 = os.path.join(MODELS_ROOT, "SE3Transformer")
for path in (MODELS_ROOT, SE3, os.path.join(SE3, "nvidia_se3transformer")):
    if path not in sys.path:
        sys.path.insert(0, path)

# 项目自身的 setup（必须在 import 模型前）
from se3_npu.compat import setup_torch_compat  # noqa: E402

setup_torch_compat()
from se3_npu.dgl_ops import apply_dgl_op_patches  # noqa: E402

apply_dgl_op_patches()

import torch  # noqa: E402
from dump_tool import dump  # noqa: E402

torch.npu.set_device(0)
dev = "npu:0"
OUT = os.path.join(MODELS_ROOT, "dump_graph", "SE3Transformer")

# 加载模型（照抄 run_inference.py 的模型构建）
from se3_transformer.model.transformer import SE3Transformer  # noqa: E402

model = SE3Transformer(
    num_layers=7, num_heads=8, channels_div=2, num_degrees=4, num_channels=32,
    pooling="max", norm=True, use_layer_norm=True, tensor_cores=True).to(dev).eval()

# 构造输入（照抄项目测试数据的模式: 图节点特征 + 边邻接）
torch.manual_seed(42)
N = 50  # 图节点数
batch = {
    "feat": torch.randn(1, N, 1, 32, device=dev),       # (B, N, 1, C) 节点特征
    "edge_index": torch.randint(0, N, (2, N * 4), device=dev),  # 边索引
    "edge_feat": torch.randn(1, N * 4, 1, 32, device=dev),     # 边特征
}
print(f"[se3] N={N}, feat={tuple(batch['feat'].shape)}", flush=True)

dump(model, **batch, out_dir=OUT, graphs=("fx", "ge"),
     netron=True, verbose=True)
