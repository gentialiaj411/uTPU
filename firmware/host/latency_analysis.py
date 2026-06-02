"""Task 4 — Deterministic-Latency Static Analysis.

Exact, data-independent static cycle counting + data-independence prover
for compiled uTPU ISA programs (sequence of 16-bit instruction words).

This is **EXACT counting**, deliberately distinct from
``firmware/host/cost_model.py`` (the approximate GPU regression used for
schedule selection, ~9% median regression error). The uTPU ISA executes
on a deterministic in-order FSM: no caches, no branches on data, no
early-exit, only deterministic FIFO backpressure. So the cycle count of
a program is fully determined by its instruction stream — worst = best
= average — and can be computed statically without input data.

Two public surfaces:

* ``static_cycles_simulator(program, cfg=None)`` returns an ``int`` whose
  value matches ``isa_simulator.simulate_program_bytes(program,
  cfg=cfg).cycle_count_sequential`` exactly. The walker mirrors the
  simulator's opcode-decode + word-advance + cycle-cost accounting
  byte-for-byte. Verified in ``test_latency_analysis.py`` across the
  blocked-FC corpus (naive + scheduled lowerings) and on multiple
  shapes, plus by direct equality assertion in
  ``run_latency_determinism.py`` on every swept shape.
* ``prove_data_independent(program, allowed_opcodes=None, cfg=None)``
  walks the instruction stream and returns a ``DataIndependenceProof``
  describing which opcodes were observed, whether all are in the
  permitted data-independent set, and which (if any) are flagged as
  data-dependent. The default allowed-opcode set is the full current
  uTPU ISA — every opcode currently shipped is data-independent by
  construction (STORE/LOAD/RUN/FETCH/BSTORE/HALT/NOP and the D-type
  sub-ops BUFFER_XFER/BARRIER/ACC_ADD/PE_SELECT all transition on the
  instruction bits, never on stored data values). For the planted test
  case in ``test_latency_analysis.py``, the caller passes a *smaller*
  allowed-set and verifies a known opcode is correctly flagged.

This module deliberately does NOT model the absolute RTL FSM cycle
count. The Phase 7 remediation P4.1 cross-check has already established
that the simulator's 1-cycle-per-op accounting and the RTL FSM's
multi-cycle STORE/FETCH paths agree at the *percentage cycle reduction*
level (±2.0% per ``bench/results/scheduler_rtl_crosscheck.json``),
NOT at the absolute count level. The static model here matches the
simulator exactly; the simulator is RTL-corroborated at the percentage
level by P4.1; the *data-independence* claim — which is what Task 4
actually adds — is empirically RTL-validated by
``run_latency_determinism.py``'s shape × distribution sweep (RTL cycle
variance == 0 across M adversarial input distributions on the same
program).

Honest scope: the analysis is constant-time over the supported
data-independent op set; covers COMPUTE latency only; UART/IO framing
is bounded separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple  # noqa: F401

from isa_encoder import (  # noqa: F401  (re-export of opcode constants)
    DTYPE_SUBOP_ACC_ADD,
    DTYPE_SUBOP_BARRIER,
    DTYPE_SUBOP_BUFFER_XFER,
    DTYPE_SUBOP_PE_SELECT,
    DEFAULT_CFG,
    IsaConfig,
    OPCODE_BSTORE,
    OPCODE_DTYPE,
    OPCODE_FETCH,
    OPCODE_HALT,
    OPCODE_LOAD,
    OPCODE_NOP,
    OPCODE_RUN,
    OPCODE_STORE,
)


# ---------------------------------------------------------------------------
# Opcode names + the data-independent allowlist
# ---------------------------------------------------------------------------


OPCODE_NAMES: Dict[int, str] = {
    OPCODE_STORE: "STORE",
    OPCODE_FETCH: "FETCH",
    OPCODE_RUN: "RUN",
    OPCODE_LOAD: "LOAD",
    OPCODE_HALT: "HALT",
    OPCODE_NOP: "NOP",
    OPCODE_BSTORE: "BSTORE",
    OPCODE_DTYPE: "DTYPE",
}

DTYPE_SUBOP_NAMES: Dict[int, str] = {
    DTYPE_SUBOP_BUFFER_XFER: "BUFFER_XFER",
    DTYPE_SUBOP_BARRIER: "BARRIER",
    DTYPE_SUBOP_ACC_ADD: "ACC_ADD",
    DTYPE_SUBOP_PE_SELECT: "PE_SELECT",
}


def opcode_label(opcode: int, dtype_subop: Optional[int] = None) -> str:
    """Stable human-readable token for an opcode (or D-type sub-op).

    Used as the canonical token in the allowlist + in the prover's
    output. Keep these labels frozen — ``CLAIMS_MATRIX.md`` /
    ``latency_determinism.json`` schema references them by name.
    """
    if opcode == OPCODE_DTYPE:
        if dtype_subop is None:
            return "DTYPE"
        sub = DTYPE_SUBOP_NAMES.get(int(dtype_subop), f"DTYPE_SUBOP_0b{int(dtype_subop):02b}")
        return sub
    return OPCODE_NAMES.get(int(opcode), f"OPCODE_0b{int(opcode):03b}")


# The full set of opcode tokens currently shipped by the uTPU ISA. Every
# one of these is data-independent by inspection of the simulator's
# decode path (no branch on data values, no early-exit, no
# value-dependent iteration count). The prover's default allowlist is
# precisely this set; the planted test in ``test_latency_analysis.py``
# passes a smaller set to verify the prover correctly flags omitted
# opcodes.
DEFAULT_DATA_INDEPENDENT_OPS: Tuple[str, ...] = (
    "STORE",
    "LOAD",
    "RUN",
    "FETCH",
    "BSTORE",
    "HALT",
    "NOP",
    "BUFFER_XFER",
    "BARRIER",
    "ACC_ADD",
    "PE_SELECT",
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StaticCycleResult:
    """Result of ``static_cycles_simulator``.

    Attributes
    ----------
    total_cycles
        The static cycle count. Equal to
        ``isa_simulator.simulate_program_bytes(...).cycle_count_sequential``
        by construction (verified per-shape in
        ``test_latency_analysis.py``).
    per_opcode_cycles
        Sum of cycle contributions broken down by opcode token (e.g.
        ``{"STORE": 273, "BSTORE": 6, "LOAD": 21, "RUN": 8, ...}``).
        Sums to ``total_cycles`` exactly.
    per_opcode_counts
        Count of each opcode token encountered (e.g. ``{"STORE": 273,
        "BSTORE": 1, ...}``). Differs from ``per_opcode_cycles`` for
        opcodes with non-unit cycle costs (BSTORE, BUFFER_XFER).
    instructions_decoded
        Total number of distinct opcodes decoded (sum over
        ``per_opcode_counts``).
    halted
        Whether the program reached a ``HALT`` opcode within the walked
        instruction stream.
    """

    total_cycles: int
    per_opcode_cycles: Dict[str, int] = field(default_factory=dict)
    per_opcode_counts: Dict[str, int] = field(default_factory=dict)
    instructions_decoded: int = 0
    halted: bool = False


@dataclass
class DataIndependenceProof:
    """Result of ``prove_data_independent``.

    Attributes
    ----------
    is_proven
        True iff every opcode observed in the instruction stream is a
        member of ``allowed_opcodes``. Returns False if any opcode in
        ``data_dependent_ops_found`` is non-empty.
    allowed_opcodes
        Sorted snapshot of the allowlist the prover was given (used for
        the planted-defect test to record the narrowed allowlist
        verbatim).
    data_independent_ops_observed
        Sorted list of opcode tokens that were observed AND were in the
        allowlist (e.g. ``["BSTORE", "FETCH", "HALT", "LOAD", "NOP",
        "RUN", "STORE"]`` for a typical blocked-FC program).
    data_dependent_ops_found
        Sorted list of opcode tokens that were observed but were NOT
        in the allowlist. Empty for shipping uTPU programs (every
        opcode is data-independent); non-empty only if a future ISA
        adds an opcode like ``BRANCH_ON_NONZERO`` or if the caller
        narrowed the allowlist to test the prover (planted case).
    instructions_decoded
        Total instructions decoded by the prover (used as a sanity
        signal — a near-zero count implies the walker missed the
        program).
    """

    is_proven: bool
    allowed_opcodes: Tuple[str, ...]
    data_independent_ops_observed: Tuple[str, ...]
    data_dependent_ops_found: Tuple[str, ...]
    instructions_decoded: int


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------


def _bytes_to_words(program: bytes) -> List[int]:
    """Decode a uTPU compiled byte stream into 16-bit little-endian words.

    Matches ``isa_simulator.simulate_program_bytes``'s preamble exactly
    (lo byte then hi byte → 16-bit word). Raises on odd byte length.
    """
    if isinstance(program, (bytes, bytearray)):
        buf = bytes(program)
    else:
        # Caller passed a list of bytes / ints — coerce.
        buf = bytes(int(b) & 0xFF for b in program)
    if len(buf) % 2 != 0:
        raise ValueError(
            f"program byte stream length {len(buf)} is odd; cannot decode to 16b words"
        )
    return [int(buf[i]) | (int(buf[i + 1]) << 8) for i in range(0, len(buf), 2)]


def _walk_program(
    program: bytes,
    cfg: Optional[IsaConfig] = None,
    *,
    track_cycles: bool,
    max_steps: Optional[int] = None,
) -> Tuple[List[Tuple[str, int]], int, bool]:
    """Single shared walker for both cycle counting and the prover.

    Returns ``(decoded_tokens, total_cycles, halted)`` where
    ``decoded_tokens`` is a list of ``(opcode_token, cycle_cost)``
    pairs in instruction order. The token resolves D-type sub-ops to
    their sub-op name (e.g. ``"BUFFER_XFER"``) rather than ``"DTYPE"``,
    which the prover needs to be able to distinguish — a future
    data-dependent D-type sub-op (hypothetical) would be flagged
    individually, not by its umbrella opcode.

    The word-advance logic mirrors ``isa_simulator.py``'s decoder
    *exactly* so static cycles equal simulator cycles. If the
    simulator ever changes its word advancement, this walker MUST
    change in lockstep (locked by the test
    ``test_static_cycles_match_simulator_on_blocked_fc_corpus``).
    """
    cfg = cfg if cfg is not None else DEFAULT_CFG
    addr_mask = cfg.address_max
    extended_addr = cfg.extended_address

    words = _bytes_to_words(program)
    if max_steps is None:
        max_steps = len(words) * 8 + 1024

    decoded: List[Tuple[str, int]] = []
    pc = 0
    steps = 0
    total_cycles = 0
    halted = False

    def consume_addr_word() -> int:
        nonlocal pc
        if pc >= len(words):
            raise ValueError("truncated extended-address payload")
        value = words[pc] & addr_mask
        pc += 1
        return value

    while pc < len(words) and steps < max_steps:
        steps += 1
        instruction = words[pc] & 0xFFFF
        opcode = instruction & 0x7
        pc += 1

        if opcode == OPCODE_STORE:
            # STORE is multi-word in both legacy and extended layouts:
            # word2 = source / immediate, word3 = destination address.
            if pc + 1 >= len(words):
                raise ValueError("truncated STORE instruction")
            pc += 2
            cost = 1
            decoded.append(("STORE", cost))
            total_cycles += cost

        elif opcode == OPCODE_BSTORE:
            # Legacy: header has addr<<7, next word is count + payload.
            # Extended: header has no addr, next words are addr + count + payload.
            if extended_addr:
                _ = consume_addr_word()
            if pc >= len(words):
                raise ValueError("truncated BSTORE count")
            count = words[pc] & 0xFFFF
            pc += 1
            if pc + count > len(words):
                raise ValueError("truncated BSTORE payload")
            pc += count
            cost = 2 + int(count)
            decoded.append(("BSTORE", cost))
            total_cycles += cost

        elif opcode == OPCODE_LOAD:
            if extended_addr:
                _ = consume_addr_word()
            cost = 1
            decoded.append(("LOAD", cost))
            total_cycles += cost

        elif opcode == OPCODE_RUN:
            if extended_addr:
                _ = consume_addr_word()
            cost = 1
            decoded.append(("RUN", cost))
            total_cycles += cost

        elif opcode == OPCODE_FETCH:
            if extended_addr:
                _ = consume_addr_word()
            cost = 1
            decoded.append(("FETCH", cost))
            total_cycles += cost

        elif opcode == OPCODE_HALT:
            cost = 1
            decoded.append(("HALT", cost))
            total_cycles += cost
            halted = True
            break

        elif opcode == OPCODE_NOP:
            cost = 1
            decoded.append(("NOP", cost))
            total_cycles += cost

        elif opcode == OPCODE_DTYPE:
            subop = (instruction >> 5) & 0x3
            if subop == DTYPE_SUBOP_BUFFER_XFER:
                if extended_addr:
                    # extended: src_addr, dst_addr, count each own a word
                    if pc + 2 >= len(words):
                        raise ValueError("truncated extended BUFFER_XFER trailer")
                    _ = consume_addr_word()  # src_addr
                    _ = consume_addr_word()  # dst_addr
                    count = words[pc] & 0xFFFF
                    pc += 1
                else:
                    if pc >= len(words):
                        raise ValueError("truncated BUFFER_XFER trailer")
                    trailer = words[pc] & 0xFFFF
                    pc += 1
                    count = (trailer >> 9) & 0x7F
                cost = 2 + int(count)
                decoded.append(("BUFFER_XFER", cost))
                total_cycles += cost
            elif subop == DTYPE_SUBOP_BARRIER:
                if extended_addr:
                    _ = consume_addr_word()
                cost = 1
                decoded.append(("BARRIER", cost))
                total_cycles += cost
            elif subop == DTYPE_SUBOP_ACC_ADD:
                cost = 1
                decoded.append(("ACC_ADD", cost))
                total_cycles += cost
            elif subop == DTYPE_SUBOP_PE_SELECT:
                cost = 1
                decoded.append(("PE_SELECT", cost))
                total_cycles += cost
            else:
                raise ValueError(
                    f"unsupported D-type subop 0b{int(subop):02b} at pc={pc - 1}"
                )

        else:
            raise ValueError(f"unsupported opcode 0b{int(opcode):03b} at pc={pc - 1}")

    if steps >= max_steps and not halted:
        raise TimeoutError(f"static walker exceeded max_steps={max_steps}")

    if not track_cycles:
        total_cycles = 0  # caller is the prover and does not need the sum.
    return decoded, total_cycles, halted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def static_cycles_simulator(
    program: bytes, cfg: Optional[IsaConfig] = None
) -> StaticCycleResult:
    """Compute the exact static cycle count of a compiled uTPU program.

    Matches ``isa_simulator.simulate_program_bytes(program,
    cfg=cfg).cycle_count_sequential`` exactly by construction —
    locked at test time by
    ``test_static_cycles_match_simulator_on_blocked_fc_corpus``.

    No input data is consulted: the cycle count is a pure function of
    the instruction stream, which is the formal definition of
    data-independent latency for this ISA.
    """
    decoded, total_cycles, halted = _walk_program(
        program, cfg=cfg, track_cycles=True
    )
    per_op_cycles: Dict[str, int] = {}
    per_op_counts: Dict[str, int] = {}
    for token, cost in decoded:
        per_op_cycles[token] = per_op_cycles.get(token, 0) + int(cost)
        per_op_counts[token] = per_op_counts.get(token, 0) + 1
    return StaticCycleResult(
        total_cycles=int(total_cycles),
        per_opcode_cycles=per_op_cycles,
        per_opcode_counts=per_op_counts,
        instructions_decoded=sum(per_op_counts.values()),
        halted=bool(halted),
    )


def prove_data_independent(
    program: bytes,
    allowed_opcodes: Optional[Sequence[str]] = None,
    cfg: Optional[IsaConfig] = None,
) -> DataIndependenceProof:
    """Static data-independence proof over a compiled uTPU program.

    Walks the instruction stream and confirms every opcode (including
    D-type sub-ops, individually) is in ``allowed_opcodes``. Any opcode
    NOT in the allowlist is recorded in
    ``data_dependent_ops_found`` and ``is_proven`` is set False.

    By default ``allowed_opcodes`` is
    :data:`DEFAULT_DATA_INDEPENDENT_OPS` — the full current uTPU ISA,
    every opcode of which is data-independent by inspection of
    ``isa_simulator.py``'s decode path. The planted-defect test in
    ``test_latency_analysis.py`` passes a *narrower* allowlist (e.g.
    one that excludes ``RUN``) and verifies the prover correctly
    flags the excluded opcode in a known program. This proves the
    prover has teeth without requiring a hypothetical
    data-dependent opcode to be added to the encoder.
    """
    allowed: Set[str] = set(
        DEFAULT_DATA_INDEPENDENT_OPS if allowed_opcodes is None else allowed_opcodes
    )

    decoded, _cycles_ignored, _halted = _walk_program(
        program, cfg=cfg, track_cycles=False
    )

    observed: Set[str] = set()
    flagged: Set[str] = set()
    for token, _cost in decoded:
        observed.add(token)
        if token not in allowed:
            flagged.add(token)

    in_allowlist = sorted(t for t in observed if t in allowed)
    is_proven = len(flagged) == 0
    return DataIndependenceProof(
        is_proven=bool(is_proven),
        allowed_opcodes=tuple(sorted(allowed)),
        data_independent_ops_observed=tuple(in_allowlist),
        data_dependent_ops_found=tuple(sorted(flagged)),
        instructions_decoded=len(decoded),
    )


def analyze_program(
    program: bytes, cfg: Optional[IsaConfig] = None
) -> Dict[str, object]:
    """Bundle ``static_cycles_simulator`` + ``prove_data_independent`` for the harness."""
    cyc = static_cycles_simulator(program, cfg=cfg)
    proof = prove_data_independent(program, cfg=cfg)
    return {
        "static_cycles": int(cyc.total_cycles),
        "per_opcode_cycles": dict(cyc.per_opcode_cycles),
        "per_opcode_counts": dict(cyc.per_opcode_counts),
        "instructions_decoded": int(cyc.instructions_decoded),
        "halted": bool(cyc.halted),
        "data_independence_proven": bool(proof.is_proven),
        "data_independent_ops_observed": list(proof.data_independent_ops_observed),
        "data_dependent_ops_found": list(proof.data_dependent_ops_found),
        "allowed_opcodes": list(proof.allowed_opcodes),
    }


__all__ = [
    "DEFAULT_DATA_INDEPENDENT_OPS",
    "DTYPE_SUBOP_NAMES",
    "DataIndependenceProof",
    "OPCODE_NAMES",
    "StaticCycleResult",
    "analyze_program",
    "opcode_label",
    "prove_data_independent",
    "static_cycles_simulator",
]
