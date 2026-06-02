import numpy as np
import pytest

from lowering_types import BlockedFCLoweringRequest
from backend_lowering import create_backend_lowerer
from cuda_blocked_fc_backend import CUDABlockedFCExecutor, detect_cuda_environment


def _sample_request():
    rng = np.random.default_rng(7)
    w = rng.integers(-8, 8, size=(10, 9), dtype=np.int8)
    x = rng.integers(-8, 8, size=(9,), dtype=np.int8)
    return BlockedFCLoweringRequest(
        weights_int4=w,
        activations_int4=x,
        out_features=10,
        in_features=9,
        array_size=16,
        apply_relu=False,
        apply_quant=True,
        weight_addr=0x080,
        input_addr=0x000,
        result_addr=0x100,
    )


def test_cuda_blocked_fc_lowering_metadata():
    req = _sample_request()
    lowerer = create_backend_lowerer("cuda")
    lowered = lowerer.lower_blocked_fc(req)
    assert lowered["mode"] == "cuda_blocked_fc"
    assert lowered["kernel_name"] == "blocked_fc_int4_kernel"
    assert lowered["memory_scopes"]["weights"] == "shared"
    assert lowered["memory_scopes"]["inputs"] == "shared"
    assert lowered["memory_scopes"]["accum"] == "register"


def test_cuda_blocked_fc_execute_or_skip():
    req = _sample_request()
    executor = CUDABlockedFCExecutor(verbose=False)
    result = executor.execute(req)
    env = detect_cuda_environment()
    if env.runtime_available:
        assert result["executed"] is True
        assert "output_unpadded" in result
        assert "max_abs_diff_vs_numpy_reference" in result
    else:
        assert result["executed"] is False
        assert "numpy_reference_output" in result
        assert "reason" in result


def test_smem_kernel_bit_exact_vs_naive():
    env = detect_cuda_environment()
    if not env.runtime_available:
        pytest.skip(env.reason or "CUDA runtime unavailable")

    req = _sample_request()
    executor = CUDABlockedFCExecutor(verbose=False)
    naive = executor.execute(req, schedule_params={"use_smem": False})
    smem = executor.execute(req, schedule_params={"use_smem": True})
    assert naive["executed"] is True
    assert smem["executed"] is True
    assert int(naive["max_abs_diff_vs_numpy_reference"]) == 0
    assert int(smem["max_abs_diff_vs_numpy_reference"]) == 0
    assert naive["output_unpadded"] == smem["output_unpadded"]
    assert smem.get("kernel_name") == "blocked_fc_int4_smem_kernel"


def run_all():
    test_cuda_blocked_fc_lowering_metadata()
    test_cuda_blocked_fc_execute_or_skip()
    env = detect_cuda_environment()
    if env.runtime_available:
        test_smem_kernel_bit_exact_vs_naive()
    print("test_cuda_backend: PASS")


if __name__ == "__main__":
    run_all()
