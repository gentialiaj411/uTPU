import argparse
import os
import sys

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from compiler_introspection import (
    format_introspection_report,
    inspect_compiled_mlp,
    write_introspection_json,
)


class TinyIntegerMLP(nn.Module):
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
    parser = argparse.ArgumentParser(
        description="Inspect the scoped PyTorch FX -> Graph IR -> blocked-FC -> CUDA/uTPU compiler pipeline."
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(REPO_ROOT, "build", "reports", "compiler_introspection_tiny_mlp.json"),
        help="Path for the structured inspection report.",
    )
    args = parser.parse_args()

    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    report = inspect_compiled_mlp(model, x, array_size=16)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    write_introspection_json(report, args.output_json)
    print(format_introspection_report(report))
    print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
