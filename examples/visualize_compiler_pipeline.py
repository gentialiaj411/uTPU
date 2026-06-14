import argparse
import os
import sys

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from compiler_pipeline_visual import (
    build_compiler_pipeline_visual_report,
    format_visual_report_terminal,
    write_visual_report,
)


class TinyVisualMLP(nn.Module):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a visual report for the scoped uTPU compiler pipeline."
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(REPO_ROOT, "build", "reports", "compiler_pipeline_visual.json"),
        help="Path for the structured JSON report.",
    )
    parser.add_argument(
        "--output-html",
        default=os.path.join(REPO_ROOT, "build", "reports", "compiler_pipeline_visual.html"),
        help="Path for the visual HTML report.",
    )
    parser.add_argument(
        "--array-size",
        type=int,
        default=16,
        help="Array size used for blocked-FC planning and lowering.",
    )
    args = parser.parse_args()

    model = TinyVisualMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    report = build_compiler_pipeline_visual_report(model, x, array_size=args.array_size)
    write_visual_report(report, args.output_json, args.output_html)

    print(format_visual_report_terminal(report))
    print(f"\nwrote {args.output_json}")
    print(f"wrote {args.output_html}")


if __name__ == "__main__":
    main()
