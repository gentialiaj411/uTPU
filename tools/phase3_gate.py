"""Executable, fail-closed Phase-3 RTL vector-unit completion gate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "bench/results/tiny_transformer_vec_rtl.json"
SYNTH = ROOT / "bench/results/boolean_vec_unit_synth.json"
DECODE_CLASSES = {
    "zero_length", "truncated_descriptor", "reserved_subop", "unknown_param_kind",
    "reserved_record_bits", "invalid_signed_range", "pointer_extent", "v1_shape_violation",
    "illegal_flags", "bad_record_kind", "prohibited_overlap", "softmax_head_id",
    "softmax_noncausal",
}
LEGACY_OPS = {
    "FETCH top_half=1", "FETCH top_half=0", "LOAD weights", "LOAD inputs", "RUN",
    "residual RUN", "BSTORE multiword", "STORE", "NOP", "requant NOP bit3",
    "EXT_ADDR FETCH two-word", "EXT_ADDR LOAD two-word", "EXT_ADDR RUN two-word",
    "EXT_ADDR BSTORE two-word",
}
TRANSITIONS = {f"{op}:{pattern}" for op in LEGACY_OPS
               for pattern in ("X->VEC", "VEC->X", "X->VEC->X")}
TRANSITIONS |= {"VEC->VEC", "VEC first after upload", "VEC last before HALT"}


def validate_artifacts(evidence: dict, synth: dict | None) -> list[str]:
    bad: list[str] = []
    per_op = evidence.get("per_op", {})
    vadd = per_op.get("vadd_rq_exhaustive", {})
    tuples = vadd.get("exponent_tuples", [])
    tuple_cases = vadd.get("tuple_cases", [])
    if (vadd.get("cases", 0) < 65536 * len(tuples) or len(tuples) < 5
            or len(tuple_cases) != len(tuples)
            or any(item.get("cases", 0) < 65536 for item in tuple_cases)):
        bad.append("VADD_RQ exhaustive minimum/tuple coverage")
    if vadd.get("mismatches", 1) != 0 or vadd.get("passed") is not True:
        bad.append("VADD_RQ mismatches")
    for key, minimum in (("vpos_add_full_table", 1536), ("rmsnorm_2000_rows", 2000),
                         ("softmax_16000_rows", 16000)):
        item = per_op.get(key, {})
        if item.get("cases", 0) < minimum or item.get("mismatches", 1) != 0 or item.get("passed") is not True:
            bad.append(key + " minimum/mismatches")
    if per_op.get("rmsnorm_2000_rows", {}).get("rsqrt_lut_bin_coverage") != 192:
        bad.append("RMSNorm LUT bins != 192/192")
    if set(per_op.get("softmax_16000_rows", {}).get("row_lengths", [])) != set(range(1, 33)):
        bad.append("softmax row lengths incomplete")
    if evidence.get("divider", {}).get("cases", 0) < 2048 or evidence.get("divider", {}).get("mismatches", 1) != 0:
        bad.append("divider minimum/mismatches")
    traces = evidence.get("trace_full_top", {})
    if traces.get("cases", 0) < 8 or traces.get("mismatches", 1) != 0:
        bad.append("full-top traces minimum/mismatches")
    stages = traces.get("stage_totals", {})
    if len(stages) != 9 or any(item.get("cases", 0) < 8 or item.get("mismatches", 1) != 0 for item in stages.values()):
        bad.append("full-top trace stages incomplete")

    decode = evidence.get("decode_errors", {}).get("classes", {})
    if not DECODE_CLASSES.issubset(decode):
        bad.append("decode error classes missing: " + ",".join(sorted(DECODE_CLASSES - set(decode))))
    for name, item in decode.items():
        if item.get("passed") is not True or item.get("mismatches", 1) != 0:
            bad.append("decode error failure " + name)
    absent = decode.get("vector_unit_absent_escape", {})
    if absent.get("passed") is not True or not absent.get("descriptor_would_be_legacy_bstore"):
        bad.append("VEC=0 absent escape case missing/failing")

    matrix = evidence.get("transition_matrix", {})
    rows = matrix.get("matrix", {})
    if not TRANSITIONS.issubset(rows):
        bad.append("transition matrix incomplete: " + ",".join(sorted(TRANSITIONS - set(rows))))
    for name in TRANSITIONS & set(rows):
        item = rows[name]
        if item.get("cases", 0) < 1 or item.get("passed") is not True or item.get("mismatches", 1) != 0:
            bad.append("transition row failed: " + name)
    if matrix.get("mismatches", 1) != 0 or matrix.get("passed") is not True:
        bad.append("transition matrix mismatch")
    if matrix.get("legacy_buffer_snapshot_bytes_compared", 0) <= 0 or matrix.get("vector_bytes_compared", 0) <= 0:
        bad.append("transition compared-byte minimum")
    mixed = evidence.get("mixed", {})
    if mixed.get("legacy_bytes", 0) < 256 or mixed.get("vec_bytes", 0) <= 0 or mixed.get("mismatches", 1) != 0:
        bad.append("mixed full-top program outputs")
    if mixed.get("non_covered", {}).get("TILE_GEMM") != "RTL sequencer not implemented (Phase 4 decision)":
        bad.append("TILE_GEMM non-coverage decision missing")

    dual = evidence.get("dual_mode_regression", {})
    globbed = {Path(x).stem for x in dual.get("globbed_testbenches", [])}
    results = dual.get("results", {})
    if not globbed or not globbed.issubset(results):
        bad.append("dual-mode glob inventory/results differ")
    for module, item in results.items():
        if item.get("exempted"):
            if not item.get("reason"):
                bad.append("dual-mode exemption lacks reason: " + module)
        elif item.get("passed") is not True or not all(item.get("modes", {}).get(m, {}).get("passed") for m in ("0", "1")):
            bad.append("dual-mode failure: " + module)

    if synth is None:
        bad.append("artifact missing: run tools/run_boolean_synth.ps1")
    else:
        if synth.get("status") != "ok": bad.append("synthesis status")
        if synth.get("part") != "xc7s50csga324-1": bad.append("synthesis wrong part")
        if synth.get("core_clock_hz") != 50000000: bad.append("synthesis core clock is not 50 MHz")
        if synth.get("top") != "top_board": bad.append("synthesis top is not top_board")
        modes = synth.get("modes", {})
        if set(modes) != {"0", "1"}: bad.append("synthesis mode reports missing")
        else:
            one = modes["1"]
            for key, limit in (("lut", 32600), ("dsp", 120), ("bram", 75)):
                if one.get(key, limit + 1) > limit: bad.append(f"synthesis {key} limit")
            if synth.get("delta", {}).get("lut", 999999) > 7400: bad.append("synthesis LUT delta limit")
            if one.get("wns_ns", -999) < 0: bad.append("synthesis WNS < 0")
            if one.get("whs_ns") is None or modes["0"].get("whs_ns") is None:
                bad.append("synthesis WHS missing")
            if not one.get("worst_path", {}).get("startpoint") or not one.get("worst_path", {}).get("endpoint"):
                bad.append("synthesis VEC=1 worst path endpoints missing")
        ooc = synth.get("vector_unit_ooc", {})
        if ooc.get("wns_ns", -999) < 1.0: bad.append("vector OOC WNS < +1.0 ns")
    return bad


def run(command: list[str], label: str, bad: list[str]) -> bool:
    print(f"RUN: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode:
        bad.append(label)
        return False
    return True


def main() -> int:
    bad: list[str] = []
    py = sys.executable
    for gate in ("tools/phase1_gate.py", "tools/phase2_gate.py"):
        run([py, gate], gate, bad)
    for command, label in (([py, "tools/gen_tiny_transformer_luts.py", "--check"], "LUT fixture"),
                           ([py, "tools/gen_vector_lut_mems.py", "--check"], "LUT mem files"),
                           ([py, "-m", "pytest", "firmware/host/test_vector_rtl_structure.py", "-q", "-p", "no:cacheprovider"], "structural vector unit test")):
        run(command, label, bad)

    # Scalar arithmetic evidence is fingerprint-bound to HEAD. Re-run every
    # fragile full-top boundary suite live on every gate invocation.
    for suite in ("decode", "mixed", "transition_matrix", "trace_full_top"):
        run([py, "tools/run_phase3_evidence.py", "--only", suite], "evidence " + suite, bad)
    run([py, "tools/run_phase3_host_suite.py"], "host suite new failures", bad)

    evidence = None
    if RESULT.is_file():
        try:
            evidence = json.loads(RESULT.read_text(encoding="utf-8"))
        except Exception as exc:
            bad.append("invalid Phase-3 evidence JSON: " + str(exc))
    else:
        bad.append("Phase-3 evidence missing")
    if evidence is not None:
        expected_tree = {p: subprocess.check_output(["git", "rev-parse", f"HEAD:{p}"], cwd=ROOT, text=True).strip()
                         for p in ("firmware/host", "rtl", "tools")}
        if evidence.get("source_tree") != expected_tree:
            bad.append("Phase-3 evidence fingerprint differs from HEAD")
        bad.extend(validate_artifacts(evidence, json.loads(SYNTH.read_text(encoding="utf-8")) if SYNTH.is_file() else None))
    summary = ROOT / "build/phase3_host_suite/summary.json"
    if not summary.is_file() or json.loads(summary.read_text(encoding="utf-8")).get("status") != "passed":
        bad.append("host suite new failures")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "firmware/host", "rtl", "tools", "scripts",
                                     "bench/results/tiny_transformer_vec_rtl.json", "bench/results/boolean_vec_unit_synth.json"],
                                    cwd=ROOT, text=True).strip()
    if dirty: bad.append("uncommitted scoped changes")
    for item in bad: print("UNMET:", item)
    return int(bool(bad))


if __name__ == "__main__":
    raise SystemExit(main())
