import torch
import torch.nn as nn

from compiled_runtime import CompiledRuntimeError
from pytorch_compiler import compile_mlp_model


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


class TinyBiasMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, bias=True)
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
            self.fc1.bias.copy_(torch.tensor([1.0, -1.0, 1.0]))
            self.fc2.weight.copy_(torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))
            self.fc2.bias.copy_(torch.tensor([1.0, -1.0]))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_compile_returns_callable_object():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    assert callable(compiled)
    assert compiled.callable is True
    y = compiled(x)
    assert y.shape == (1, 2)


def test_compiled_mlp_uses_cuda_backend_for_linear_ops():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x, mode="compiled")
    report = compiled.execution_report()

    assert report["mode"] == "compiled"
    assert report["backend_linear_ops_executed"] == 2
    assert report["backend_elementwise_ops_executed"] == 1
    assert all(trace["engine"] != "pytorch_reference" for trace in report["op_traces"])
    assert any(trace["engine"] == "nvrtc_cuda_blocked_fc" for trace in report["op_traces"])
    assert any(trace["engine"] == "nvrtc_cuda_elementwise" for trace in report["op_traces"])


def test_no_silent_pytorch_linear_fallback():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x, mode="compiled")
    report = compiled.execution_report()

    assert report["pytorch_fallback_ops"] == 0
    assert report["fallback_ops"] == []


def test_execution_report_counts_backend_ops():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x)
    report = compiled.execution_report()

    assert report["backend_linear_ops_executed"] == 2
    assert report["backend_elementwise_ops_executed"] == 1
    assert report["adapter_ops"] == 0
    assert report["fallback_ops"] == []
    assert len(report["op_traces"]) == 3


def test_output_matches_pytorch_reference():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    with torch.no_grad():
        y_compiled = compiled(x, mode="compiled")
        y_ref = model(x)
    max_abs = float(torch.max(torch.abs(y_compiled.detach().cpu() - y_ref.detach().cpu())).item())

    assert max_abs == 0.0


def test_relu_executes_without_numpy_adapter():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x)
    report = compiled.execution_report()

    assert report["backend_elementwise_ops_executed"] == 1
    assert report["adapter_ops"] == 0
    assert all(trace["engine"] != "numpy_adapter" for trace in report["op_traces"])


def test_bias_executes_without_adapter():
    model = TinyBiasMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    with torch.no_grad():
        y_compiled = compiled(x)
        y_ref = model(x)
    report = compiled.execution_report()
    max_abs = float(torch.max(torch.abs(y_compiled.detach().cpu() - y_ref.detach().cpu())).item())

    assert max_abs == 0.0
    assert report["backend_linear_ops_executed"] == 2
    assert report["backend_elementwise_ops_executed"] == 2
    assert report["adapter_ops"] == 0
    assert report["fallback_ops"] == []
    assert all(trace["engine"] != "numpy_adapter" for trace in report["op_traces"])


def test_supported_mlp_has_zero_fallback_ops():
    model = TinyBiasMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x)
    report = compiled.execution_report()

    assert report["fallback_ops"] == []
    assert report["pytorch_fallback_ops"] == 0
    assert report["adapter_ops"] == 0


def test_no_silent_pytorch_or_numpy_fallback_in_compiled_mode():
    model = TinyBiasMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x, mode="compiled")
    report = compiled.execution_report()
    engines = [trace["engine"] for trace in report["op_traces"]]

    assert "pytorch_reference" not in engines
    assert "numpy_adapter" not in engines
    assert set(engines).issubset({"nvrtc_cuda_blocked_fc", "nvrtc_cuda_elementwise"})


def test_executor_cache_reused_across_calls():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x, mode="compiled")
    first_cache = compiled.runtime.cuda_executor.cache_stats()
    compiled(x, mode="compiled")
    second_cache = compiled.runtime.cuda_executor.cache_stats()

    assert first_cache["context_initialized"] is True
    assert second_cache["context_initialized"] is True
    assert second_cache["kernel_cache_entries"] == first_cache["kernel_cache_entries"]
    assert second_cache["buffer_cache_entries"] == first_cache["buffer_cache_entries"]


def test_second_call_avoids_recompile():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x, mode="compiled")
    first_report = compiled.execution_report()
    compiled(x, mode="compiled")
    second_report = compiled.execution_report()

    assert first_report["compile_time_ms"] > 0.0
    assert second_report["compile_time_ms"] == 0.0
    assert second_report["setup_time_ms"] <= first_report["setup_time_ms"]


def test_benchmark_reports_steady_state_latency():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    bench = compiled.benchmark(x, warmup=2, iters=5)

    assert bench["first_call_wall_ms"] > 0.0
    assert bench["steady_state_wall_ms"] > 0.0
    assert bench["compile_time_ms"] > 0.0
    assert bench["kernel_time_ms"] > 0.0
    assert bench["backend_linear_ops_executed"] == 2
    assert bench["backend_elementwise_ops_executed"] == 1
    assert bench["max_abs_error_vs_pytorch"] == 0.0


def test_resident_runtime_reports_single_transfer_pair():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    compiled(x, mode="compiled")
    report = compiled.execution_report()

    assert report["h2d_count"] == 1
    assert report["d2h_count"] == 1
    assert all("resident_graph_execution" in trace["notes"] for trace in report["op_traces"])


def test_quantized_reference_separates_backend_correctness_from_float_drift():
    model = nn.Sequential(
        nn.Linear(64, 32, bias=False),
        nn.ReLU(),
        nn.Linear(32, 16, bias=False),
    ).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    with torch.no_grad():
        model[0].weight.copy_(torch.randint(-2, 3, model[0].weight.shape, generator=generator, dtype=torch.float32))
        model[2].weight.copy_(torch.randint(-2, 3, model[2].weight.shape, generator=generator, dtype=torch.float32))
    x = torch.randint(-2, 3, (1, 64), generator=generator, dtype=torch.float32)
    compiled = compile_mlp_model(model, x, target="cuda")

    with torch.no_grad():
        y_compiled = compiled(x, mode="compiled")
        y_quant_ref = torch.as_tensor(compiled.runtime.quantized_reference(x), dtype=torch.float32)
        y_float_ref = model(x)

    compiled_vs_quant = float(torch.max(torch.abs(y_compiled.cpu() - y_quant_ref)).item())
    quant_vs_float = float(torch.max(torch.abs(y_quant_ref - y_float_ref.cpu())).item())

    assert compiled_vs_quant == 0.0
    assert quant_vs_float > 0.0


def test_reference_mode_still_available():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")

    y_ref_mode = compiled(x, mode="reference")
    with torch.no_grad():
        y_ref = model(x)
    report = compiled.execution_report()

    assert torch.allclose(y_ref_mode.cpu(), y_ref.cpu(), atol=1e-6)
    assert report["mode"] == "reference"
    assert report["pytorch_fallback_ops"] == 2
    assert report["backend_linear_ops_executed"] == 0


def test_relu_activation_is_applied_between_linear_layers():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")
    y_compiled = compiled(x)

    manual_relu = torch.relu(model.fc1(x))
    manual = model.fc2(manual_relu)
    no_relu = model.fc2(model.fc1(x))

    assert torch.allclose(y_compiled.cpu(), manual.cpu(), atol=1e-6)
    assert not torch.allclose(y_compiled.cpu(), no_relu.cpu(), atol=1e-6)


def test_unsupported_ops_fail_clearly():
    class BadModel(nn.Module):
        def forward(self, x):
            return torch.sigmoid(x)

    compiled = compile_mlp_model(BadModel(), torch.randn(1, 4), target="cuda")

    assert compiled.ok is False
    assert compiled.import_error is not None
    assert "Unsupported call_function node" in compiled.import_error
    try:
        compiled(torch.randn(1, 4))
        raise AssertionError("Expected unsupported compiled model call to fail")
    except Exception as e:
        assert "runtime" in str(e).lower() or "unsupported" in str(e).lower()


def test_utpu_target_emits_instruction_plan():
    model = nn.Sequential(nn.Linear(4, 3, bias=False), nn.ReLU(), nn.Linear(3, 2, bias=False)).eval()
    compiled = compile_mlp_model(model, torch.randn(1, 4), target="utpu")

    assert compiled.callable is False
    assert len(compiled.backend_ops) == 2
    assert all("program" in op.lowering for op in compiled.backend_ops)
    assert all(op.lowering["program_instruction_words"] > 0 for op in compiled.backend_ops)
    try:
        compiled(torch.randn(1, 4))
        raise AssertionError("Expected uTPU compiled call to fail without board runtime")
    except CompiledRuntimeError as e:
        assert "uTPU compilation emits instruction plans" in str(e)


def run_all():
    test_compile_returns_callable_object()
    test_compiled_mlp_uses_cuda_backend_for_linear_ops()
    test_no_silent_pytorch_linear_fallback()
    test_execution_report_counts_backend_ops()
    test_output_matches_pytorch_reference()
    test_relu_executes_without_numpy_adapter()
    test_bias_executes_without_adapter()
    test_supported_mlp_has_zero_fallback_ops()
    test_no_silent_pytorch_or_numpy_fallback_in_compiled_mode()
    test_executor_cache_reused_across_calls()
    test_second_call_avoids_recompile()
    test_benchmark_reports_steady_state_latency()
    test_resident_runtime_reports_single_transfer_pair()
    test_quantized_reference_separates_backend_correctness_from_float_drift()
    test_reference_mode_still_available()
    test_relu_activation_is_applied_between_linear_layers()
    test_unsupported_ops_fail_clearly()
    test_utpu_target_emits_instruction_plan()
    print("test_compiled_mlp_execution: PASS")


if __name__ == "__main__":
    run_all()
