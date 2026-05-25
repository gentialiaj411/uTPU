"""Phase 5 — Instruction scheduler + buffer-slot allocator.

This module adds a backend codegen pass that runs *before* ISA emission
for blocked-FC programs. It mirrors classic register allocation:

* :class:`BufferSlotAllocator` packs named ``Value``s with explicit
  liveness intervals into a fixed-capacity 16-bit-word unified buffer
  using a deterministic best-fit-by-finish-time policy. When the working
  set exceeds capacity it spills the value with the latest next use
  (Belady-style) and records a ``SpillEvent`` so the emitter knows it
  must re-issue the value's STORE block before its next consumer.

* :class:`BlockedFCScheduler` consumes a :class:`BlockedFCProblem` and
  emits a deterministic op stream that hoists per-input-block STOREs to
  a prelude (so each input block is host-stored exactly once instead of
  ``out_blocks`` times in the naive lowering). With spills the emitter
  faithfully re-stores spilled inputs before each consumer, preserving
  bit-exactness against the naive program.

* :func:`lower_blocked_fc_program_scheduled` is the public entry point:
  same signature as ``lowering_blocked_fc_utpu.lower_blocked_fc_program_utpu``
  plus an optional ``buffer_capacity`` knob. The returned program bytes
  are guaranteed to produce the same ``fetch_bytes`` as the naive
  emission for every shape that fits in the configured buffer.

Scope. This is sim-only and operates entirely on the host before bytes
hit the UART; it does not change RTL widths/sizes (Phase 4 owns those)
and never widens the ISA opcode space (Phase 5 is scoped to scheduling
+ allocation, not new instructions). All cycle / reload / utilization
numbers reported here come from :mod:`isa_simulator`, not silicon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from isa_encoder import ISAEncoder


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Value:
    """A named, sized buffer-resident value (in 16-bit words)."""

    name: str
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"value '{self.name}' has non-positive size {self.size}")


@dataclass(frozen=True)
class LiveInterval:
    """Liveness interval for a Value across the linear op stream.

    ``first_use_idx`` is the op index that first defines / loads the
    value; ``last_use_idx`` is the index of the final consumer. Both
    bounds are inclusive and zero-based against the scheduled op stream.
    """

    value: Value
    first_use_idx: int
    last_use_idx: int

    def __post_init__(self) -> None:
        if self.first_use_idx < 0 or self.last_use_idx < self.first_use_idx:
            raise ValueError(
                f"invalid liveness for {self.value.name}: "
                f"[{self.first_use_idx}, {self.last_use_idx}]"
            )


@dataclass(frozen=True)
class SpillEvent:
    """Record of a value evicted at a given op index.

    The emitter consults the spill log to know which value(s) must be
    re-stored before their next consumer. ``replaced_value`` is the
    incoming value whose insertion forced the eviction.
    """

    op_idx: int
    evicted_value: str
    replaced_value: str


@dataclass
class AllocationResult:
    """Outcome of running :class:`BufferSlotAllocator` on a workload."""

    layout: Dict[str, int] = field(default_factory=dict)
    spills: List[SpillEvent] = field(default_factory=list)
    capacity: int = 0
    peak_live_words: int = 0

    @property
    def spilled_values(self) -> List[str]:
        return sorted({s.evicted_value for s in self.spills})


# ---------------------------------------------------------------------------
# Buffer-slot allocator
# ---------------------------------------------------------------------------


class BufferAllocationError(RuntimeError):
    """Raised when the allocator cannot place a value even after spilling."""


class BufferSlotAllocator:
    """Deterministic buffer-slot allocator with Belady-style spilling.

    Allocator works on a fixed window ``[base_offset, base_offset + capacity)``
    of the unified buffer. Caller is responsible for partitioning the
    buffer into "scheduler-managed" and "fixed" regions: e.g. weight and
    result regions can be pre-pinned outside the window if desired.
    """

    def __init__(self, capacity: int, base_offset: int = 0):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if base_offset < 0:
            raise ValueError(f"base_offset must be non-negative, got {base_offset}")
        self.capacity = int(capacity)
        self.base_offset = int(base_offset)

    def allocate(
        self,
        intervals: List[LiveInterval],
        op_count: int,
        next_use_after: Optional[Dict[Tuple[str, int], int]] = None,
    ) -> AllocationResult:
        """Pack ``intervals`` into a single contiguous capacity window.

        Allocation proceeds in linear op order (0..op_count-1). At each
        op:

        1. Free intervals whose ``last_use_idx`` is strictly less than
           the current op index.
        2. Insert intervals whose ``first_use_idx`` is exactly the
           current op index. If the new interval doesn't fit, spill the
           live value with the latest next-use-after-op (Belady-style;
           ties broken by name for determinism).

        ``next_use_after`` is an optional pre-computed map
        ``(value_name, op_idx) -> next_use_idx``; when omitted the
        allocator falls back to "evict the value with the latest
        ``last_use_idx`` strictly greater than the current op". This is
        sufficient for blocked-FC where a value either has a single
        contiguous lifetime or is bookended by re-stores after spills.
        """
        # Sort by first_use, then by name for determinism.
        ordered = sorted(intervals, key=lambda iv: (iv.first_use_idx, iv.value.name))
        index_by_name = {iv.value.name: iv for iv in ordered}

        # Bucket intervals by first_use_idx for fast lookup.
        starts_by_idx: Dict[int, List[LiveInterval]] = {}
        for iv in ordered:
            starts_by_idx.setdefault(iv.first_use_idx, []).append(iv)

        live: Dict[str, Tuple[int, int]] = {}  # name -> (offset, size)
        layout: Dict[str, int] = {}
        spills: List[SpillEvent] = []
        peak = 0

        for op_idx in range(op_count + 1):
            for name in [n for n, (_, _) in live.items()
                         if index_by_name[n].last_use_idx < op_idx]:
                live.pop(name, None)

            for iv in sorted(
                starts_by_idx.get(op_idx, []),
                key=lambda x: x.value.name,
            ):
                placed = False
                while not placed:
                    offset = self._first_fit(live, iv.value.size)
                    if offset is not None:
                        live[iv.value.name] = (offset, iv.value.size)
                        layout[iv.value.name] = self.base_offset + offset
                        placed = True
                        break
                    victim = self._pick_spill_victim(
                        live=live,
                        index_by_name=index_by_name,
                        op_idx=op_idx,
                        incoming=iv.value.name,
                        next_use_after=next_use_after,
                    )
                    if victim is None:
                        raise BufferAllocationError(
                            f"cannot allocate value '{iv.value.name}' (size={iv.value.size}) "
                            f"at op_idx={op_idx} with capacity={self.capacity}: no live victim"
                        )
                    live.pop(victim, None)
                    spills.append(
                        SpillEvent(
                            op_idx=op_idx,
                            evicted_value=victim,
                            replaced_value=iv.value.name,
                        )
                    )
            occupied = sum(sz for (_, sz) in live.values())
            if occupied > peak:
                peak = occupied

        return AllocationResult(
            layout=layout,
            spills=spills,
            capacity=self.capacity,
            peak_live_words=peak,
        )

    def _first_fit(
        self,
        live: Dict[str, Tuple[int, int]],
        size: int,
    ) -> Optional[int]:
        if size > self.capacity:
            return None
        intervals = sorted((off, off + sz) for (off, sz) in live.values())
        cursor = 0
        for start, end in intervals:
            if start - cursor >= size:
                return cursor
            cursor = max(cursor, end)
        if self.capacity - cursor >= size:
            return cursor
        return None

    def _pick_spill_victim(
        self,
        live: Dict[str, Tuple[int, int]],
        index_by_name: Dict[str, LiveInterval],
        op_idx: int,
        incoming: str,
        next_use_after: Optional[Dict[Tuple[str, int], int]],
    ) -> Optional[str]:
        if not live:
            return None
        # Evict the live value whose next use is farthest in the future
        # (Belady). Falls back to last_use_idx when no next_use map.
        candidates: List[Tuple[int, str]] = []
        for name in live:
            if name == incoming:
                continue
            iv = index_by_name[name]
            if next_use_after is not None:
                key = (name, op_idx)
                next_use = next_use_after.get(key, iv.last_use_idx)
            else:
                next_use = iv.last_use_idx
            candidates.append((next_use, name))
        if not candidates:
            return None
        # Largest next_use first, name as deterministic tiebreaker.
        candidates.sort(key=lambda x: (-x[0], x[1]))
        return candidates[0][1]


# ---------------------------------------------------------------------------
# Blocked-FC scheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockedFCProblem:
    """Inputs needed to drive the scheduler over a blocked FC layer.

    ``array_size`` and ``items_per_word`` together determine the
    per-block buffer footprint:

    * weight block size (in words) = ``array_size * array_size // items_per_word``
    * input block size (in words)  = ``array_size // items_per_word``
    * result block size (in words) = ``array_size * array_size // items_per_word``
      (matches the RTL finalize footprint that pads to ``array_size``
      output rows).
    """

    out_blocks: int
    in_blocks: int
    array_size: int
    items_per_word: int = 4
    weight_addr: int = 256
    input_base_addr: int = 0
    result_addr: int = 320
    apply_quant: bool = True
    apply_relu: bool = False


@dataclass
class ScheduledProgram:
    """Output of the scheduler: program bytes plus measured layout."""

    program: bytes
    program_instruction_words: int
    layout: Dict[str, int]
    spills: List[SpillEvent]
    peak_live_words: int
    out_blocks: int
    in_blocks: int
    array_size: int
    apply_quant: bool
    apply_relu: bool
    blockers: List[str] = field(default_factory=list)


class BlockedFCScheduler:
    """Deterministic list-scheduler for blocked FC.

    The scheduler iterates the canonical ``(ob, ib)`` access pattern
    used by ``lower_blocked_fc_program_utpu`` but caches input blocks in
    stable buffer slots so they are STORE'd at most once per
    ``in_block``. Output bytes are bit-exact with the naive emission as
    long as the buffer capacity holds all in_blocks worth of inputs;
    when capacity is tight the allocator spills, and the emitter
    re-issues the spilled input's STOREs before its next use (no
    correctness loss, just lost reload savings).
    """

    def __init__(
        self,
        weights_int4: np.ndarray,
        activations_int4: np.ndarray,
        problem: BlockedFCProblem,
        buffer_capacity: int = 512,
    ):
        if buffer_capacity <= 0:
            raise ValueError(f"buffer_capacity must be positive, got {buffer_capacity}")
        self.problem = problem
        self.buffer_capacity = int(buffer_capacity)
        a = problem.array_size
        out_padded = problem.out_blocks * a
        in_padded = problem.in_blocks * a
        w = np.asarray(weights_int4, dtype=np.int8)
        x = np.asarray(activations_int4, dtype=np.int8).flatten()
        self._w_pad = np.zeros((out_padded, in_padded), dtype=np.int8)
        self._w_pad[:w.shape[0], :w.shape[1]] = w
        self._x_pad = np.zeros(in_padded, dtype=np.int8)
        self._x_pad[:x.shape[0]] = x

    # -- liveness model --------------------------------------------------

    def _build_intervals_and_ops(
        self,
    ) -> Tuple[List[LiveInterval], int, Dict[Tuple[str, int], int]]:
        """Build the abstract op stream and per-input-block liveness.

        The op stream is ordered as:

        ``[store_input(ib=0), store_input(ib=1), ...,``
         ``  for ob in 0..out_blocks-1:``
         ``    for ib in 0..in_blocks-1:``
         ``      [store_w, load_w, load_i, run_acc],``
         ``    [run_fin, fetch * a/items_per_word * 2]]``

        Each input block ``inp_<ib>`` is live from its initial store op
        through its final ``load_i`` (i.e. last consumer in the last
        ``ob`` iteration). Weight blocks have point-lifetime (one op
        between store and load) so we don't allocate slots for them
        through the allocator: they are statically pinned at
        ``weight_addr``. Result blocks are similarly pinned at
        ``result_addr + ob * result_block_size`` outside the
        scheduler-managed window.
        """
        a = self.problem.array_size
        items_per_word = self.problem.items_per_word
        in_block_words = a // items_per_word

        op_idx = 0
        intervals: List[LiveInterval] = []
        next_use_after: Dict[Tuple[str, int], int] = {}

        prelude_first_use: Dict[str, int] = {}
        for ib in range(self.problem.in_blocks):
            name = f"inp_{ib}"
            prelude_first_use[name] = op_idx
            op_idx += 1

        last_load_per_input: Dict[str, int] = {}
        load_op_indices: Dict[str, List[int]] = {n: [] for n in prelude_first_use}

        for ob in range(self.problem.out_blocks):
            for ib in range(self.problem.in_blocks):
                op_idx += 1  # store_w
                op_idx += 1  # load_w
                load_i_idx = op_idx
                op_idx += 1  # load_i
                op_idx += 1  # run_acc
                name = f"inp_{ib}"
                load_op_indices[name].append(load_i_idx)
                last_load_per_input[name] = load_i_idx
            op_idx += 1  # run_fin
            fetch_words = (a // items_per_word)
            op_idx += fetch_words * 2  # fetch * 2 (low + high byte)

        op_count = op_idx

        for name, first_use in prelude_first_use.items():
            intervals.append(
                LiveInterval(
                    value=Value(name=name, size=in_block_words),
                    first_use_idx=first_use,
                    last_use_idx=last_load_per_input[name],
                )
            )

        for name, loads in load_op_indices.items():
            sorted_loads = sorted(loads)
            for query_idx in range(op_count + 1):
                future = [li for li in sorted_loads if li >= query_idx]
                if future:
                    next_use_after[(name, query_idx)] = future[0]

        return intervals, op_count, next_use_after

    # -- emission --------------------------------------------------------

    def emit(self) -> ScheduledProgram:
        problem = self.problem
        a = problem.array_size
        items_per_word = problem.items_per_word

        intervals, op_count, next_use_after = self._build_intervals_and_ops()
        allocator = BufferSlotAllocator(
            capacity=self.buffer_capacity,
            base_offset=problem.input_base_addr,
        )
        alloc = allocator.allocate(
            intervals,
            op_count=op_count,
            next_use_after=next_use_after,
        )
        # Hoisting policy: hoist iff every input block fits without any
        # spill. With spills, address regions for hoisted slots and the
        # fallback "store-before-each-load" slot would collide and wreck
        # bit-exactness, so we cleanly fall back to fully-naive emission
        # (== bit-identical to the legacy lowering) and preserve the
        # allocator's telemetry (peak working set, spill events) for the
        # benchmark / claims artifact.
        all_hoisted = (not alloc.spills) and all(
            f"inp_{ib}" in alloc.layout for ib in range(problem.in_blocks)
        )
        hoisted: Dict[str, int] = dict(alloc.layout) if all_hoisted else {}

        encoder = ISAEncoder()
        layout: Dict[str, int] = {}
        if all_hoisted:
            for ib in range(problem.in_blocks):
                name = f"inp_{ib}"
                self._emit_store_input_block(encoder, hoisted[name], ib)
                layout[name] = hoisted[name]

        for ob in range(problem.out_blocks):
            out_base_addr = problem.result_addr + ob * (a // items_per_word)
            o0 = ob * a
            o1 = o0 + a
            for ib in range(problem.in_blocks):
                name = f"inp_{ib}"
                ib_offset = ib * a
                weight_block = self._w_pad[o0:o1, ib_offset:ib_offset + a]

                self._emit_store_weight_block(
                    encoder, problem.weight_addr, weight_block
                )
                encoder.loadWeights(problem.weight_addr)
                if name in hoisted:
                    encoder.loadInputs(hoisted[name])
                else:
                    fallback_addr = problem.input_base_addr
                    self._emit_store_input_block(encoder, fallback_addr, ib)
                    layout[name] = fallback_addr
                    encoder.loadInputs(fallback_addr)
                encoder.run(
                    out_base_addr,
                    compute=True,
                    quantize=False,
                    relu=False,
                    acc_clear=(ib == 0),
                )

            encoder.run(
                out_base_addr,
                compute=False,
                quantize=problem.apply_quant,
                relu=problem.apply_relu,
                acc_clear=False,
            )
            for widx in range(a // items_per_word):
                addr = out_base_addr + widx
                encoder.fetch(addr, top_half=False)
                encoder.fetch(addr, top_half=True)

        encoder.halt()
        program = encoder.getProgram()
        words = len(program) // 2

        blockers: List[str] = []
        if not problem.apply_quant:
            blockers.append(
                "Scheduled blocked-FC requires quantized finalize for the current "
                "host-visible output path; raw int32 fetch path is not exposed yet."
            )

        return ScheduledProgram(
            program=program,
            program_instruction_words=int(words),
            layout=layout,
            spills=list(alloc.spills),
            peak_live_words=int(alloc.peak_live_words),
            out_blocks=int(problem.out_blocks),
            in_blocks=int(problem.in_blocks),
            array_size=int(a),
            apply_quant=bool(problem.apply_quant),
            apply_relu=bool(problem.apply_relu),
            blockers=blockers,
        )

    def _emit_store_input_block(
        self,
        encoder: ISAEncoder,
        base_addr: int,
        ib: int,
    ) -> None:
        a = self.problem.array_size
        ib_offset = ib * a
        block = self._x_pad[ib_offset:ib_offset + a]
        self._emit_store_array(encoder, base_addr, block)

    def _emit_store_weight_block(
        self,
        encoder: ISAEncoder,
        base_addr: int,
        weight_block: np.ndarray,
    ) -> None:
        self._emit_store_array(encoder, base_addr, weight_block)

    def _emit_store_array(
        self,
        encoder: ISAEncoder,
        base_addr: int,
        data,
    ) -> None:
        flat = list(np.asarray(data).flatten())
        items_per_word = self.problem.items_per_word
        addr = base_addr
        for i in range(0, len(flat), items_per_word):
            chunk = flat[i:i + items_per_word]
            while len(chunk) < items_per_word:
                chunk.append(0)
            encoder.store(addr, chunk)
            addr += 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def lower_blocked_fc_program_scheduled(
    weights_int4,
    activations_int4,
    out_features: int,
    in_features: int,
    array_size: int,
    apply_relu: bool,
    apply_quant: bool,
    weight_addr: int,
    input_addr: int,
    result_addr: int,
    buffer_capacity: Optional[int] = None,
    items_per_word: int = 4,
) -> Dict[str, object]:
    """Scheduled lowering of a blocked-FC layer.

    Drop-in compatible with ``lowering_blocked_fc_utpu.lower_blocked_fc_program_utpu``
    (same positional args + same return shape) and produces a program
    that is bit-exact in ``fetch_bytes`` against the naive emission.
    Adds a ``buffer_capacity`` knob so tests can stress the spill path.
    """
    import math

    out_blocks = math.ceil(out_features / array_size)
    in_blocks = math.ceil(in_features / array_size)

    w = np.asarray(weights_int4, dtype=np.int8)
    x = np.asarray(activations_int4, dtype=np.int8).flatten()
    if w.shape != (out_features, in_features):
        raise ValueError(f"weights shape mismatch: expected {(out_features, in_features)}, got {w.shape}")
    if x.shape[0] != in_features:
        raise ValueError(f"activation length mismatch: expected {in_features}, got {x.shape[0]}")

    if buffer_capacity is None:
        # Default: use the gap [input_addr, weight_addr) for hoisting,
        # i.e. preserve the legacy section layout (Section A for inputs,
        # Section B for weights, Section C for results) verbatim. The
        # caller can pass an explicit ``buffer_capacity`` to stress the
        # spill path.
        if weight_addr > input_addr:
            buffer_capacity = int(weight_addr - input_addr)
        else:
            buffer_capacity = int(array_size // items_per_word)

    scheduler = BlockedFCScheduler(
        weights_int4=w,
        activations_int4=x,
        problem=BlockedFCProblem(
            out_blocks=out_blocks,
            in_blocks=in_blocks,
            array_size=array_size,
            items_per_word=items_per_word,
            weight_addr=weight_addr,
            input_base_addr=input_addr,
            result_addr=result_addr,
            apply_quant=apply_quant,
            apply_relu=apply_relu,
        ),
        buffer_capacity=buffer_capacity,
    )
    sp = scheduler.emit()

    executable = bool(sp.apply_quant)
    blockers = list(sp.blockers)
    if not sp.apply_quant:
        executable = False

    return {
        "program": sp.program,
        "program_instruction_words": int(sp.program_instruction_words),
        "fits_instruction_bram": bool(sp.program_instruction_words <= 1024),
        "array_size": int(sp.array_size),
        "out_blocks": int(sp.out_blocks),
        "in_blocks": int(sp.in_blocks),
        "block_ops": int(sp.out_blocks * sp.in_blocks),
        "executable_on_current_fpga_path": bool(executable),
        "int32_accumulation_supported": True,
        "quantize_after_accumulation_supported": True,
        "blockers": blockers,
        # Scheduler / allocator metadata (Phase 5):
        "scheduler": "blocked_fc_input_hoist_v1",
        "buffer_capacity_words": int(buffer_capacity),
        "buffer_layout": {k: int(v) for k, v in sp.layout.items()},
        "peak_live_words": int(sp.peak_live_words),
        "spill_events": [
            {
                "op_idx": int(s.op_idx),
                "evicted_value": s.evicted_value,
                "replaced_value": s.replaced_value,
            }
            for s in sp.spills
        ],
        "spill_count": int(len(sp.spills)),
    }
