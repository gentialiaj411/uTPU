import torch
import torch.nn as nn

from compiler_pipeline_visual import build_compiler_pipeline_visual_report


class TinyVisualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_visual_report_contains_pipeline_stages():
    model = TinyVisualMLP().eval()
    x = torch.randn(1, 4)
    report = build_compiler_pipeline_visual_report(model, x)

    assert report["title"] == "uTPU Compiler Pipeline Visual Report"
    assert [stage["stage"] for stage in report["stages"]] == [
        "PyTorch Module",
        "torch.fx Graph",
        "Graph IR Imported",
        "Pass Pipeline",
        "Runtime Plan",
        "CUDA Backend Lowering",
        "uTPU ISA Lowering",
        "Evidence / Limitations",
    ]
    assert "linear_relu" in report["summary"]["final_graph_ops"]
    assert "linear_relu_fusion" in report["summary"]["changed_passes"]
    assert report["stages"][4]["facts"]["memory_plan"]["physical_buffer_count"] >= 1
    assert report["stages"][6]["facts"]["utpu_instruction_words_total"] > 0
