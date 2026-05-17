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
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
            self.fc2.weight.copy_(torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def main():
    model = TinyMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])

    compiled = compile_mlp_model(model, x, target="cuda")
    with torch.no_grad():
        y_compiled = compiled(x, mode="compiled")
        y_pytorch = model(x)

    max_abs = float(torch.max(torch.abs(y_compiled.detach().cpu() - y_pytorch.detach().cpu())).item())

    print("Compiled tiny MLP execution")
    print("===========================")
    print(f"summary={compiled.summary()}")
    print(f"execution_report={compiled.execution_report()}")
    print(f"pytorch_output={y_pytorch.detach().cpu().tolist()}")
    print(f"compiled_output={y_compiled.detach().cpu().tolist()}")
    print(f"max_abs_error={max_abs:.8f}")


if __name__ == "__main__":
    main()
