import torch
import torch.nn as nn

from pytorch_compiler import compile_mlp_model, compile_model


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_compile_two_layer_mlp_end_to_end_import_and_lowering():
    model = TinyMLP()
    result = compile_mlp_model(model, torch.randn(1, 4), target="cuda")

    assert result.import_error is None
    assert result.graph_ir is not None
    assert [op.op for op in result.graph_ir.ops] == ["linear_relu", "linear"]
    assert result.plan is not None
    assert len(result.plan.lowered_ops) == 2
    assert result.plan.lowered_ops[0].fused_activation == "relu"
    assert len(result.backend_ops) == 2


def test_unsupported_model_reports_clear_import_error():
    class BadModel(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    result = compile_mlp_model(BadModel(), torch.randn(1, 4), target="cuda")

    assert result.ok is False
    assert result.import_error is not None
    assert "Unsupported call_function node" in result.import_error
    assert "sigmoid" in result.import_error


def test_cuda_target_routes_to_cuda_lowerer():
    result = compile_mlp_model(nn.Sequential(nn.Linear(4, 3), nn.ReLU()), torch.randn(1, 4), target="cuda")

    assert len(result.backend_ops) == 1
    lowered = result.backend_ops[0].lowering
    assert lowered["mode"] == "cuda_blocked_fc"
    assert lowered["kernel_name"] == "blocked_fc_int4_kernel"


def test_utpu_target_emits_instruction_program_path():
    result = compile_mlp_model(nn.Sequential(nn.Linear(4, 3), nn.ReLU()), torch.randn(1, 4), target="utpu")

    assert len(result.backend_ops) == 1
    lowered = result.backend_ops[0].lowering
    assert "program" in lowered
    assert lowered["program_instruction_words"] > 0
    assert lowered["array_size"] == 16
    assert lowered["executable_on_current_fpga_path"] is True


def test_compile_model_general_entrypoint_matches_wrapper():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
    x = torch.randn(1, 4)
    a = compile_model(model, x, target="cuda")
    b = compile_mlp_model(model, x, target="cuda")
    assert a.import_error == b.import_error
    assert a.runtime_plan is not None and b.runtime_plan is not None
    assert a.runtime_plan.unsupported_ops == b.runtime_plan.unsupported_ops


def run_all():
    test_compile_two_layer_mlp_end_to_end_import_and_lowering()
    test_unsupported_model_reports_clear_import_error()
    test_cuda_target_routes_to_cuda_lowerer()
    test_utpu_target_emits_instruction_program_path()
    test_compile_model_general_entrypoint_matches_wrapper()
    print("test_pytorch_compiler: PASS")


if __name__ == "__main__":
    run_all()
