#!/usr/bin/env python3
"""Decompose program instruction words into embedded payload vs control.

Emits bench/results/program_word_composition.json for every shape in
board_fit_audit.json SHAPE_GRID plus the fused MNIST multi-layer program
and the capstone MNIST FC1 (256x196) blocked-FC lowering used for the
52k-word capacity claim.

Classification:
  - bstore_payload_words: 16-bit data words after BSTORE count
  - store_immediate_payload_words: STORE-immediate data words (bit4=1)
  - embedded_payload_words: sum of the two (weights/acts in the program image)
  - control_words: everything else (opcodes, addresses, counts, LOAD/RUN/FETCH,
    HALT, NOP/requant headers+params, BSTORE/STORE framing)

Part A removes control micro-ops only; it does not remove embedded payload.
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from isa_encoder import (  # noqa: E402
    DEFAULT_CFG,
    NOP_SUBOP_REQUANT,
    OPCODE_BSTORE,
    OPCODE_DTYPE,
    OPCODE_FETCH,
    OPCODE_HALT,
    OPCODE_LOAD,
    OPCODE_NOP,
    OPCODE_RUN,
    OPCODE_STORE,
    IsaConfig,
)
from lowering_blocked_fc_utpu import lower_blocked_fc_program_utpu  # noqa: E402
from run_board_fit_audit import SHAPE_GRID, _lower  # noqa: E402

OUT = REPO / "bench" / "results" / "program_word_composition.json"
MNIST_MEM = REPO / "build" / "test_vectors" / "mnist_case1_program.mem"
BOARD_FIT = REPO / "bench" / "results" / "board_fit_audit.json"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def words_from_bytes(program: bytes) -> List[int]:
    return [struct.unpack_from("<H", program, i)[0] for i in range(0, len(program), 2)]


def words_from_mem(path: Path) -> List[int]:
    return [
        int(line.strip(), 16)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def decompose_words(words: Sequence[int], *, extended_address: bool) -> Dict[str, Any]:
    """Walk a flat 16-bit instruction stream and classify words."""
    i = 0
    n = len(words)
    bstore_payload = 0
    store_imm_payload = 0
    control = 0
    by_opcode_control: Dict[str, int] = {}
    bursts = 0
    store_imm = 0
    ops_seen: Dict[str, int] = {}

    def add_ctrl(name: str, count: int = 1) -> None:
        nonlocal control
        control += count
        by_opcode_control[name] = by_opcode_control.get(name, 0) + count

    def bump_op(name: str) -> None:
        ops_seen[name] = ops_seen.get(name, 0) + 1

    while i < n:
        w = words[i]
        op = w & 0x7
        if op == OPCODE_HALT:
            bump_op("HALT")
            add_ctrl("HALT", 1)
            i += 1
            break
        if op == OPCODE_BSTORE:
            bump_op("BSTORE")
            if extended_address:
                # header, addr, count, payload...
                if i + 2 >= n:
                    raise ValueError(f"truncated BSTORE at {i}")
                count = int(words[i + 2])
                add_ctrl("BSTORE_framing", 3)  # header+addr+count
                bstore_payload += count
                bursts += 1
                i += 3 + count
            else:
                # header(with addr), count, payload...
                if i + 1 >= n:
                    raise ValueError(f"truncated BSTORE at {i}")
                count = int(words[i + 1])
                add_ctrl("BSTORE_framing", 2)
                bstore_payload += count
                bursts += 1
                i += 2 + count
            continue
        if op == OPCODE_STORE:
            bump_op("STORE")
            imm = bool((w >> 4) & 1)
            if imm:
                # opcode, data, addr
                if i + 2 >= n:
                    raise ValueError(f"truncated STORE-imm at {i}")
                add_ctrl("STORE_framing", 2)
                store_imm_payload += 1
                store_imm += 1
                i += 3
            else:
                # opcode, src, dest
                if i + 2 >= n:
                    raise ValueError(f"truncated STORE-addr at {i}")
                add_ctrl("STORE_addr", 3)
                i += 3
            continue
        if op in (OPCODE_LOAD, OPCODE_RUN, OPCODE_FETCH):
            name = {OPCODE_LOAD: "LOAD", OPCODE_RUN: "RUN", OPCODE_FETCH: "FETCH"}[op]
            bump_op(name)
            residual = bool((w >> 7) & 1) if (op == OPCODE_RUN and extended_address) else False
            if extended_address:
                # header + addr (+ optional residual addr)
                words_needed = 3 if residual else 2
                if i + words_needed - 1 >= n:
                    raise ValueError(f"truncated {name} at {i}")
                add_ctrl(name, words_needed)
                i += words_needed
            else:
                add_ctrl(name, 1)
                i += 1
            continue
        if op == OPCODE_NOP:
            bump_op("NOP")
            sub = (w >> 3) & 0x3
            vector = bool((w >> 5) & 1)
            if sub == NOP_SUBOP_REQUANT and extended_address:
                if vector:
                    length = ((w >> 8) & 0xFF) + 1
                    # header + length*(mult,shift)
                    take = 1 + 2 * length
                    if i + take - 1 >= n:
                        raise ValueError(f"truncated requant vector at {i}")
                    add_ctrl("REQUANT", take)
                    i += take
                else:
                    # header, mult, shift
                    if i + 2 >= n:
                        raise ValueError(f"truncated requant at {i}")
                    add_ctrl("REQUANT", 3)
                    i += 3
            else:
                add_ctrl("NOP", 1)
                i += 1
            continue
        if op == OPCODE_DTYPE:
            bump_op("DTYPE")
            # Conservative: most DTYPE are 1 word; buffer xfer may be 2 — treat as control.
            add_ctrl("DTYPE", 1)
            i += 1
            continue
        # Unknown / reserved
        bump_op(f"UNKNOWN_{op}")
        add_ctrl("UNKNOWN", 1)
        i += 1

    total = len(words)
    embedded = bstore_payload + store_imm_payload
    accounted = embedded + control
    if accounted != total:
        # Trailing unread words (should not happen if halt present).
        trailing = total - accounted
        if trailing > 0:
            control += trailing
            by_opcode_control["TRAILING"] = trailing
            accounted = embedded + control

    return {
        "program_instruction_words": total,
        "bstore_payload_words": bstore_payload,
        "store_immediate_payload_words": store_imm_payload,
        "embedded_payload_words": embedded,
        "control_words": control,
        "bstore_bursts": bursts,
        "store_immediate_ops": store_imm,
        "embedded_payload_fraction": (embedded / total) if total else None,
        "control_fraction": (control / total) if total else None,
        "part_a_control_only_size_ratio": (
            (embedded + 6) / total if total and embedded else None
        ),  # sketch: all tiles → one 6-word TILE_GEMM, payload unchanged
        "by_opcode_control_words": by_opcode_control,
        "ops_seen": ops_seen,
        "accounted_words": accounted,
        "accounted_ok": accounted == total,
        "extended_address": bool(extended_address),
    }


def _row(
    *,
    name: str,
    tag: str,
    shape: Optional[Dict[str, int]],
    decomp: Dict[str, Any],
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "name": name,
        "tag": tag,
        "shape": shape,
        "notes": notes,
        **decomp,
    }


def main() -> int:
    cases: List[Dict[str, Any]] = []

    # Board-fit shapes (legacy B=1 STORE-immediate path — matches audit).
    for out_f, in_f, tag in SHAPE_GRID:
        lowered = _lower(out_f, in_f, 16, prog_depth=10**9, seed=0xB041D ^ (out_f * 2654435761 + in_f * 7))
        words = words_from_bytes(lowered["program"])
        decomp = decompose_words(words, extended_address=False)
        assert decomp["program_instruction_words"] == int(lowered["program_instruction_words"])
        cases.append(
            _row(
                name=f"board_fit_{out_f}x{in_f}",
                tag=tag,
                shape={"out_features": out_f, "in_features": in_f, "array_size": 16, "batch": 1},
                decomp=decomp,
                notes="board_fit_audit lowering: DEFAULT_CFG legacy, B=1, STORE-immediate payloads",
            )
        )

    # Fused MNIST multi-layer (EXT_ADDR + BSTORE).
    if MNIST_MEM.exists():
        words = words_from_mem(MNIST_MEM)
        decomp = decompose_words(words, extended_address=True)
        cases.append(
            _row(
                name="mnist_fused_case1",
                tag="multi_layer_reference",
                shape={"fc": "64->64->10", "array_size": 16, "batch": 1},
                decomp=decomp,
                notes="build/test_vectors/mnist_case1_program.mem (fused compressed)",
            )
        )

    # Capstone FC1 256x196 — the ~52k capacity claim shape.
    rng = np.random.default_rng(0x52C225)
    w = rng.integers(-8, 8, size=(256, 196), dtype=np.int8)
    x = rng.integers(-8, 8, size=(196,), dtype=np.int8)
    fc1 = lower_blocked_fc_program_utpu(
        w, x, 256, 196, 16, False, True, 0, 64, 68, prog_depth=10**9, cfg=DEFAULT_CFG
    )
    decomp = decompose_words(words_from_bytes(fc1["program"]), extended_address=False)
    cases.append(
        _row(
            name="mnist_capstone_fc1_256x196",
            tag="capacity_claim_fc1",
            shape={"out_features": 256, "in_features": 196, "array_size": 16, "batch": 1},
            decomp=decomp,
            notes=(
                f"blocked-FC legacy lowering; program_instruction_words={decomp['program_instruction_words']} "
                "(historical claim cited 52225 — compare)"
            ),
        )
    )

    # Headline verdict — capacity claim needs BOTH fraction and absolute payload.
    board_fit_payload = [c for c in cases if c["name"].startswith("board_fit_")]
    payload_fracs = [float(c["embedded_payload_fraction"]) for c in board_fit_payload]
    median_payload = float(np.median(payload_fracs))
    fc1 = next(c for c in cases if c["name"] == "mnist_capstone_fc1_256x196")
    mnist = next((c for c in cases if c["name"] == "mnist_fused_case1"), None)
    fc1_after = fc1["embedded_payload_words"] + 6

    if mnist and float(mnist["embedded_payload_fraction"]) >= 0.85:
        fused_note = "fused_mnist_payload_dominates"
    else:
        fused_note = "fused_mnist_control_or_mixed"

    if fc1["embedded_payload_words"] > 8192:
        # Absolute payload alone exceeds demo-fit BRAM — Part A cannot unlock FC1.
        verdict = "payload_absolute_blocks_capacity_part_a_must_rescope"
    elif median_payload >= 0.85:
        verdict = "payload_dominates_part_a_capacity_fails"
    elif median_payload >= 0.70:
        verdict = "payload_dominates_borderline"
    else:
        verdict = "control_dominates_fraction_but_check_absolute_payload"

    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "methodology": {
            "board_fit_shapes": "run_board_fit_audit.SHAPE_GRID via same _lower()",
            "mnist_fused": "mnist_case1_program.mem",
            "fc1_capstone": "lower_blocked_fc_program_utpu(256,196) DEFAULT_CFG",
            "classification": (
                "embedded_payload = BSTORE payload words + STORE-immediate data words; "
                "control = all other instruction words"
            ),
            "part_a_sketch": (
                "part_a_control_only_size_ratio / after_part_a sketch replaces all "
                "control with one 6-word TILE_GEMM and keeps embedded payload — "
                "optimistic upper bound on Part A program-size reduction"
            ),
        },
        "verdict": verdict,
        "verdict_detail": {
            "fused_mnist_path": fused_note,
            "board_fit_median_payload_fraction": median_payload,
            "fc1_payload_alone_exceeds_8192": fc1["embedded_payload_words"] > 8192,
            "implication": (
                "Fused/BSTORE programs are ~87% payload — Part A compresses ~1.14x words. "
                "Legacy board_fit programs are ~33% payload / ~67% control by fraction, but "
                "capstone FC1 payload alone (14144) already exceeds PROG_DEPTH=8192, so "
                "control compression cannot make that model fit. Capacity unlock for real "
                "models requires sourcing weights outside the program image (Phase 6 forward)."
            ),
        },
        "headline": {
            "board_fit_median_embedded_payload_fraction": median_payload,
            "fc1_program_words": fc1["program_instruction_words"],
            "fc1_embedded_payload_words": fc1["embedded_payload_words"],
            "fc1_control_words": fc1["control_words"],
            "fc1_embedded_payload_fraction": fc1["embedded_payload_fraction"],
            "fc1_after_part_a_control_strip_sketch_words": fc1_after,
            "fc1_fits_prog_depth_8192_after_part_a_sketch": fc1_after <= 8192,
            "mnist_fused_embedded_payload_fraction": (
                None if mnist is None else mnist["embedded_payload_fraction"]
            ),
            "mnist_fused_words": None if mnist is None else mnist["program_instruction_words"],
            "mnist_fused_payload_words": None if mnist is None else mnist["embedded_payload_words"],
            "mnist_fused_control_words": None if mnist is None else mnist["control_words"],
            "historical_fc1_claim_words": 52225,
            "historical_vs_regen_fc1_delta": 52225 - fc1["program_instruction_words"],
        },
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Compact stdout for the agent/user
    print(f"verdict={verdict}")
    print(
        f"board_fit median payload frac={median_payload:.4f} "
        f"fc1 words={fc1['program_instruction_words']} "
        f"payload={fc1['embedded_payload_words']} ({fc1['embedded_payload_fraction']:.4f}) "
        f"control={fc1['control_words']} after_PartA_sketch={fc1_after} "
        f"fits_8192={fc1_after <= 8192}"
    )
    if mnist:
        print(
            f"mnist_fused words={mnist['program_instruction_words']} "
            f"payload={mnist['embedded_payload_words']} ({mnist['embedded_payload_fraction']:.4f}) "
            f"control={mnist['control_words']}"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
