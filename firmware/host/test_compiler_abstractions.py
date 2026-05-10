import math

from compiler_abstractions import (
    MemoryScope,
    BlockedFCProblem,
    build_blocked_fc_schedule,
    utpu_target_desc,
    cuda_target_desc,
)


def test_blocked_schedule_utpu_9x196_array16():
    problem = BlockedFCProblem(out_features=9, in_features=196, array_size=16)
    schedule = build_blocked_fc_schedule(problem=problem, target=utpu_target_desc(array_size=16))

    assert schedule.out_blocks == math.ceil(9 / 16)
    assert schedule.in_blocks == math.ceil(196 / 16)
    assert schedule.out_padded == 16
    assert schedule.in_padded == 208
    assert schedule.weight_scope == MemoryScope.SHARED
    assert schedule.input_scope == MemoryScope.SHARED
    assert schedule.accum_scope == MemoryScope.REGISTER


def test_blocked_schedule_target_mismatch_raises():
    problem = BlockedFCProblem(out_features=10, in_features=9, array_size=16)
    try:
        build_blocked_fc_schedule(problem=problem, target=utpu_target_desc(array_size=8))
        raise AssertionError("Expected ValueError for array size mismatch")
    except ValueError as e:
        assert "does not match target" in str(e)


def test_cuda_target_desc_shape():
    target = cuda_target_desc(array_size=16)
    assert target.name == "cuda"
    assert target.array_size == 16
    assert target.shared_mem_bytes >= 48 * 1024
    assert target.warp_or_lane_width == 32


def run_all():
    test_blocked_schedule_utpu_9x196_array16()
    test_blocked_schedule_target_mismatch_raises()
    test_cuda_target_desc_shape()
    print("test_compiler_abstractions: PASS")


if __name__ == "__main__":
    run_all()
