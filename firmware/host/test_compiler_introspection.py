import os
import tempfile

import torch
import torch.nn as nn

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


def test_introspection_reports_pipeline_boundaries():
    report = inspect_compiled_mlp(TinyIntegerMLP().eval(), torch.tensor([[1.0, 2.0, 0.0, 0.0]]))

    assert report["cuda_summary"]["ok"] is True
    assert report["cuda_summary"]["fallback_ops"] == []
    assert report["cuda_summary"]["unsupported_ops"] == []
    assert [op["op"] for op in report["graph_ir"]["ops"]] == ["linear", "relu", "linear"]
    assert len(report["compile_plan"]["lowered_ops"]) == 2
    assert report["compile_plan"]["lowered_ops"][0]["fused_activation"] == "relu"
    assert len(report["cuda_backend_ops"]) == 2
    assert len(report["utpu_backend_ops"]) == 2
    assert report["derived"]["utpu_instruction_words_total"] > 0
    assert report["derived"]["utpu_all_lowered_ops_fit_instruction_bram"] is True


def test_introspection_text_is_honest_about_scope():
    report = inspect_compiled_mlp(TinyIntegerMLP().eval(), torch.tensor([[1.0, 2.0, 0.0, 0.0]]))
    text = format_introspection_report(report)

    assert "not_claimed=arbitrary PyTorch support" in text
    assert "not_claimed=transformer support" in text
    assert "fallback_ops=[]" in text
    assert "uTPU ISA footprint" in text


def test_introspection_json_roundtrip():
    report = inspect_compiled_mlp(TinyIntegerMLP().eval(), torch.tensor([[1.0, 2.0, 0.0, 0.0]]))
    path = os.path.join(tempfile.mkdtemp(prefix="utpu_introspection_"), "report.json")

    write_introspection_json(report, path)

    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def run_all():
    test_introspection_reports_pipeline_boundaries()
    test_introspection_text_is_honest_about_scope()
    test_introspection_json_roundtrip()
    print("test_compiler_introspection: PASS")


if __name__ == "__main__":
    run_all()
