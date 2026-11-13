"""core: 与模型无关的捕获内核（dynamo/aot/dispatch/netron/GE）。"""

from .capture import (
                      capture_aot_graphs,
                      capture_dynamo_graphs,
                      dump_graphs,
                      explain_breaks,
                      tensor_diff,
                      trace_dispatch,
)
from .compat import bump_dynamo_limits, patch_trace_compat, restore_dynamo_limits
from .ge_view import capture_ge_graphs, export_ge_graph
from .netron_view import DEFAULT_ATTEMPT_TIMEOUT, export_ge_netron, export_netron_view

__all__ = [
    "capture_dynamo_graphs", "capture_aot_graphs", "capture_ge_graphs",
    "trace_dispatch", "explain_breaks", "dump_graphs", "tensor_diff",
    "export_netron_view", "export_ge_netron", "export_ge_graph",
    "patch_trace_compat", "bump_dynamo_limits", "restore_dynamo_limits",
    "DEFAULT_ATTEMPT_TIMEOUT",
]
