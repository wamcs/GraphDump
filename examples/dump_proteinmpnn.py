"""ProteinMPNN（自定义图神经网络, 非 HF）的 dump 模板。

演示要点: 自定义架构 + torch.load 权重; 图结构输入; 直接 forward 调用
（model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)）。
"""
import os
import sys

MODELS_ROOT = os.environ.get(
    "DUMP_MODELS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
PMPNN = os.path.join(MODELS_ROOT, "dauparas_proteinmpnn")
for path in (MODELS_ROOT, os.path.join(PMPNN, "script")):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402
from dump_tool import dump  # noqa: E402
from protein_mpnn_utils import ProteinMPNN, parse_PDB, tied_featurize  # noqa: E402

torch.npu.set_device(0)
dev = "npu:0"
OUT = os.path.join(MODELS_ROOT, "dump_graph", "dauparas_proteinmpnn")

# 加载（照抄项目 run.py）
ckpt = torch.load(os.path.join(PMPNN, "weights", "v_48_020.pt"),
                  map_location="cpu")
model = ProteinMPNN(num_letters=21, node_features=128, edge_features=128,
                    hidden_dim=128, num_encoder_layers=3,
                    num_decoder_layers=3, augment_eps=0.0,
                    k_neighbors=ckpt.get("num_edges", 48))
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(dev).eval()

# 构造输入（照抄 run.py: PDB → featurize → 取需要的字段）
pdb_dict_list = parse_PDB(os.path.join(PMPNN, "script", "sample_data",
                                       "5L33.pdb"))
result = tied_featurize(pdb_dict_list, dev, chain_dict=None)
X, S, mask, lengths, chain_m, chain_encoding_all = result[:6]  # noqa: N816
randn = torch.randn(X.shape[0], X.shape[1], device=dev)
print(f"[proteinmpnn] X={tuple(X.shape)} S={tuple(S.shape)} "
      f"mask={tuple(mask.shape)}", flush=True)

# forward 签名: model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
residue_idx = result[12]
dump(model, X, S, mask, chain_m, residue_idx, chain_encoding_all, randn,
     out_dir=OUT, graphs=("fx", "ge"), netron=True, verbose=True)
