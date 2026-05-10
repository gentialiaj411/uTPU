from dataclasses import dataclass
from enum import Enum
import math
from typing import Tuple


class MemoryScope(str, Enum):
    REGISTER = "register"
    SHARED = "shared"
    GLOBAL = "global"


@dataclass(frozen=True)
class TargetDesc:
    name: str
    array_size: int
    shared_mem_bytes: int
    register_bytes_per_thread: int
    warp_or_lane_width: int


@dataclass(frozen=True)
class BlockedFCSchedule:
    out_blocks: int
    in_blocks: int
    out_padded: int
    in_padded: int
    weight_scope: MemoryScope
    input_scope: MemoryScope
    accum_scope: MemoryScope


@dataclass(frozen=True)
class BlockedFCProblem:
    out_features: int
    in_features: int
    array_size: int

    def validate(self) -> None:
        if self.out_features <= 0 or self.in_features <= 0:
            raise ValueError("out_features and in_features must be positive")
        if self.array_size <= 0:
            raise ValueError("array_size must be positive")
        if self.array_size % 4 != 0:
            raise ValueError("array_size must be divisible by 4 for int4 packing")


def build_blocked_fc_schedule(problem: BlockedFCProblem, target: TargetDesc) -> BlockedFCSchedule:
    """
    Target-agnostic blocked FC schedule with explicit memory scopes.
    For uTPU, SHARED represents BRAM tile windows in the unified buffer.
    """
    problem.validate()
    if target.array_size != problem.array_size:
        raise ValueError(
            f"Schedule array_size={problem.array_size} does not match target array_size={target.array_size}"
        )

    out_blocks = math.ceil(problem.out_features / problem.array_size)
    in_blocks = math.ceil(problem.in_features / problem.array_size)
    out_padded = out_blocks * problem.array_size
    in_padded = in_blocks * problem.array_size

    return BlockedFCSchedule(
        out_blocks=out_blocks,
        in_blocks=in_blocks,
        out_padded=out_padded,
        in_padded=in_padded,
        weight_scope=MemoryScope.SHARED,
        input_scope=MemoryScope.SHARED,
        accum_scope=MemoryScope.REGISTER,
    )


def utpu_target_desc(array_size: int) -> TargetDesc:
    # 1KB unified buffer with 2-byte words; schedule scope mapping is conceptual.
    return TargetDesc(
        name="utpu",
        array_size=array_size,
        shared_mem_bytes=1024,
        register_bytes_per_thread=0,
        warp_or_lane_width=array_size,
    )


def cuda_target_desc(array_size: int = 16) -> TargetDesc:
    # Placeholder model for planning/metadata only in this phase.
    return TargetDesc(
        name="cuda",
        array_size=array_size,
        shared_mem_bytes=48 * 1024,
        register_bytes_per_thread=255 * 4,
        warp_or_lane_width=32,
    )
