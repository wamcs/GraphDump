"""demo 自测入口: 含数据依赖控制流的小模型（fullgraph 编译必败的形态）。"""

import torch

from . import DumpTarget


class DemoModel(torch.nn.Module):
    """数据依赖分支（图断裂源）+ in-place + 字符串分支的复合形态。

    fullgraph 编译在该模型上必然失败（h.sum() > 0 是张量值分支）——用来
    验证"图编译不了, 计算图照样 dump"。
    """

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(64, 64)
        self.norm = torch.nn.LayerNorm(64)

    def forward(self, x, mode="a"):
        h = self.norm(self.proj(x))
        if h.sum() > 0:
            h = h * 1.1
        else:
            h = h * 0.9
        h.relu_()
        if mode == "a":
            h = h @ h.transpose(-1, -2)
        return h


def build_targets(device="auto"):
    if device and device != "auto":
        dev = device
    else:
        try:
            import torch_npu  # noqa: F401
            dev = "npu" if torch.npu.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            dev = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(0)
    model = DemoModel().to(dev).eval()
    x = torch.randn(4, 8, 64, device=dev)

    def _inputs():
        return (x, "a"), {}

    return [DumpTarget(name="demo", fn=model, make_inputs=_inputs,
                       contexts=(torch.no_grad, torch.inference_mode))]
