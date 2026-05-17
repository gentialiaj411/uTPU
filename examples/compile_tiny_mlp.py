import os
import sys

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from pytorch_compiler import compile_mlp_model


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def main():
    torch.manual_seed(0)
    model = TinyMLP().eval()
    x = torch.randn(1, 4)

    result = compile_mlp_model(model, x, target="cuda")

    print("PyTorch model compiler demo")
    print("===========================")
    print("FX graph:")
    print(result.fx_graph.graph)
    print()
    print("Graph IR ops:")
    for op in result.graph_ir.ops:
        out = result.graph_ir.values[op.outputs[0]]
        print(f"- {op.name}: {op.op} shape={out.shape} dtype={out.dtype}")
    print()
    print("Compile summary:")
    print(result.summary())
    print()
    print("Backend lowered ops:")
    for lowered in result.backend_ops:
        keys = sorted(k for k in lowered.lowering.keys() if k != "kernel_source" and k != "program")
        print(f"- {lowered.graph_op}: target={lowered.target} fused_activation={lowered.fused_activation}")
        print(f"  metadata_keys={keys}")

    with torch.no_grad():
        pytorch_out = model(x)
        compiled_out = result(x)
    max_abs = float(torch.max(torch.abs(compiled_out.detach().cpu() - pytorch_out.detach().cpu())).item())
    print()
    print("PyTorch output:")
    print(pytorch_out)
    print("Compiled output:")
    print(compiled_out)
    print(f"Max abs error: {max_abs:.8f}")


if __name__ == "__main__":
    main()
