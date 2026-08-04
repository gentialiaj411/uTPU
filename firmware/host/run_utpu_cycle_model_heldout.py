"""uTPU cycle cost-model held-out validation (Track 2).

Mirrors ``run_cost_model_heldout.py`` methodology exactly, but predicts
**RTL ``total_program_cycles``** from a ``(shape, schedule)`` pair instead
of CUDA microseconds:

* deterministic 80/20 split partitioned by ``(in_features, out_features)``
* refit analytical coefficients on TRAIN shapes only
* evaluate log-R^2 / MAPE / p95 abs-rel on held-out TEST rows
* evaluate top-1 / selection regret on held-out shapes (measured cycles of
  the predicted-best schedule vs measured-best)

Ground truth prefers live iverilog (``C:\\iverilog\\bin``) via the batched
GEMM harness pattern in ``run_systolic_characterization.py``. Results are
cached under ``build/utpu_cycle_cache/``. When the working-tree ``top.sv``
fails to elaborate (e.g. mid-edit on another track), the runner shadow-
compiles ``HEAD:rtl/top/top.sv`` without modifying the working tree.
Points still missing RTL are filled by an ISA-static cycle model
affine-calibrated on TRAIN RTL rows only, and labeled honestly in the
artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HOST_DIR = os.path.dirname(os.path.abspath(__file__))
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from generate_batched_gemm_rtl_vectors import ARRAY_SIZE, generate_vectors  # noqa: E402
from isa_simulator import simulate_program_bytes  # noqa: E402
from run_rtl_batched_gemm_sim import (  # noqa: E402
    _parse_perf_counter,
    _resolve_iverilog_tools,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "bench" / "results" / "utpu_cycle_model_heldout.json"
CACHE_DIR = REPO_ROOT / "build" / "utpu_cycle_cache"
SYSTOLIC_SEED = REPO_ROOT / "bench" / "results" / "systolic_characterization.json"

HOLDOUT_FRAC = 0.20
SPLIT_SEED = "utpu-cycle-heldout-v1"

# Modest shape x schedule grid. Prefer RTL; ISA-calibrated fill for gaps.
SHAPES: Tuple[Tuple[int, int], ...] = (
    (16, 16),
    (16, 32),
    (32, 16),
    (32, 32),
    (48, 16),
    (48, 32),
    (64, 16),
    (64, 32),
    (64, 64),
    (32, 48),
)

# schedule ~= CUDA (threads_per_block, unroll_factor): batch_size + hoist flag.
SCHEDULES: Tuple[Dict[str, int], ...] = (
    {"batch_size": 1, "hoist_tile_payloads": 0},
    {"batch_size": 4, "hoist_tile_payloads": 0},
    {"batch_size": 4, "hoist_tile_payloads": 1},
    {"batch_size": 16, "hoist_tile_payloads": 0},
    {"batch_size": 16, "hoist_tile_payloads": 1},
)

# Linear (not log) features. Separate ISA scales for hoist on/off because
# hoist changes the control/compute mix and the isa→RTL multiplier (~9x vs ~4-5x).
_FEATURE_NAMES: Tuple[str, ...] = (
    "isa_static_no_hoist",
    "isa_static_hoist",
    "analytical_busy_cycles",
    "program_words",
    "out_blocks",
    "in_blocks",
    "batch_size",
    "hoist",
    "n_accumulate_runs",
    "isa_x_batch",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _shape_key(row: Dict[str, Any]) -> Tuple[int, int]:
    s = row["shape_used"]
    return (int(s["in_features"]), int(s["out_features"]))


def _schedule_key(row: Dict[str, Any]) -> Tuple[int, int]:
    sch = row["schedule"]
    return (int(sch["batch_size"]), int(sch["hoist_tile_payloads"]))


def _deterministic_holdout_shapes(
    shape_keys: Sequence[Tuple[int, int]],
    holdout_frac: float,
    seed: str,
) -> List[Tuple[int, int]]:
    """Same SHA-256 policy as ``run_cost_model_heldout._deterministic_holdout_shapes``."""
    sorted_keys = sorted(set(shape_keys))
    target_count = max(1, int(math.ceil(len(sorted_keys) * holdout_frac)))

    def _hash(key: Tuple[int, int]) -> int:
        token = f"{seed}|{key[0]}|{key[1]}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")

    sorted_keys.sort(key=lambda k: (_hash(k), k))
    return sorted_keys[:target_count]


def _split_rows(
    rows: List[Dict[str, Any]],
    holdout_keys: Sequence[Tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    holdout_set = set(holdout_keys)
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for row in rows:
        if _shape_key(row) in holdout_set:
            test.append(row)
        else:
            train.append(row)
    return train, test


def _cache_path(out_f: int, in_f: int, batch: int, hoist: int) -> Path:
    return CACHE_DIR / f"o{out_f}_i{in_f}_b{batch}_h{hoist}.json"


def _seed_cache_from_systolic() -> int:
    """Prefill cache from the committed systolic characterization artifact."""
    if not SYSTOLIC_SEED.exists():
        return 0
    blob = json.loads(SYSTOLIC_SEED.read_text(encoding="utf-8"))
    seeded = 0
    for case in blob.get("cases", []):
        out_f = int(case["shape"]["out_features"])
        in_f = int(case["shape"]["in_features"])
        batch = int(case["batch_size"])
        hoist = 1 if case.get("hoist_tile_payloads") else 0
        measured = case.get("measured") or {}
        cycles = measured.get("total_program_cycles")
        if cycles is None:
            continue
        path = _cache_path(out_f, in_f, batch, hoist)
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "out_features": out_f,
                    "in_features": in_f,
                    "batch_size": batch,
                    "hoist_tile_payloads": hoist,
                    "total_program_cycles": int(cycles),
                    "rtl_busy_counter": measured.get("rtl_busy_counter"),
                    "rtl_sim_passed": bool(case.get("rtl_sim_passed")),
                    "ground_truth_source": "systolic_characterization_seed",
                    "seed_artifact": str(SYSTOLIC_SEED.relative_to(REPO_ROOT)),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        seeded += 1
    return seeded


def _analytical_busy(out_f: int, in_f: int, batch: int) -> int:
    out_blocks = int(math.ceil(out_f / ARRAY_SIZE))
    in_blocks = int(math.ceil(in_f / ARRAY_SIZE))
    per_tile = int((2 * ARRAY_SIZE) + batch - 2)
    return int(out_blocks * in_blocks * per_tile)


def _isa_features(out_f: int, in_f: int, batch: int, hoist: int) -> Dict[str, float]:
    """Cheap host-side features (no RTL). Always available."""
    from isa_encoder import IsaConfig

    stem = f"utpu_cm_feat_o{out_f}_i{in_f}_b{batch}_h{hoist}"
    vectors = generate_vectors(
        out_features=out_f,
        in_features=in_f,
        batch_size=batch,
        stem=stem,
        output_json=str(REPO_ROOT / "build" / "test_vectors" / f"{stem}.json"),
        hoist_tile_payloads=bool(hoist),
    )
    cfg_info = vectors["cfg"]
    cfg = IsaConfig(
        address_width=int(cfg_info["address_width"]),
        compute_data_width=int(cfg_info["compute_data_width"]),
    )
    mem_path = Path(str(vectors["program_mem"]))
    program = bytearray()
    for line in mem_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        word = int(line, 16)
        program.extend(int(word).to_bytes(2, byteorder="little", signed=False))
    sim = simulate_program_bytes(
        bytes(program),
        array_size=ARRAY_SIZE,
        buffer_size=int(cfg_info["buffer_size"]),
        cfg=cfg,
    )
    out_blocks = int(vectors["out_blocks"])
    in_blocks = int(vectors["in_blocks"])
    isa = float(sim.cycle_count_sequential)
    eff_hoist = float(1 if vectors.get("hoist_tile_payloads") else 0)
    # Use requested hoist for schedule identity; effective_hoist for physics.
    h = float(hoist)
    return {
        "isa_static_cycles": isa,
        "isa_static_no_hoist": isa * (1.0 - h),
        "isa_static_hoist": isa * h,
        "analytical_busy_cycles": float(_analytical_busy(out_f, in_f, batch)),
        "program_words": float(vectors["program_words"]),
        "out_blocks": float(out_blocks),
        "in_blocks": float(in_blocks),
        "batch_size": float(batch),
        "hoist": h,
        "n_accumulate_runs": float(out_blocks * in_blocks),
        "useful_macs": float(vectors["useful_macs"]),
        "effective_hoist": eff_hoist,
        "isa_x_batch": isa * float(batch) / 16.0,
    }


def _iverilog_run_shadow(
    repo_root: Path,
    *,
    use_head_top: bool,
    out_vvp_name: str,
) -> Tuple[bool, str]:
    """Compile/run tb_batched_gemm, optionally shadowing top.sv from HEAD."""
    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        return False, "iverilog/vvp binaries not found"

    build_dir = repo_root / "build" / "rtl_sim_utpu_cycle"
    build_dir.mkdir(parents=True, exist_ok=True)
    out_vvp = build_dir / out_vvp_name

    top_src = repo_root / "rtl" / "top" / "top.sv"
    if use_head_top:
        top_shadow = build_dir / "top_head.sv"
        try:
            content = subprocess.check_output(
                ["git", "show", "HEAD:rtl/top/top.sv"],
                cwd=str(repo_root),
            )
            top_shadow.write_bytes(content)
            top_path = top_shadow
        except Exception as exc:  # noqa: BLE001
            return False, f"failed to materialize HEAD top.sv: {exc}"
    else:
        top_path = top_src

    srcs = [
        repo_root / "rtl" / "tb" / "tb_batched_gemm.sv",
        repo_root / "rtl" / "tb" / "xpm_memory_sdpram_stub.sv",
        top_path,
        repo_root / "rtl" / "memory" / "instr_bram.sv",
        repo_root / "rtl" / "PEArray" / "pe_controller.sv",
        repo_root / "rtl" / "PEArray" / "pe_array.sv",
        repo_root / "rtl" / "PEArray" / "pe.sv",
        repo_root / "rtl" / "quantizer" / "quantizer.sv",
        repo_root / "rtl" / "quantizer" / "quantizer_array.sv",
        repo_root / "rtl" / "LeakyReLU" / "leaky_relu.sv",
        repo_root / "rtl" / "LeakyReLU" / "leaky_relu_array.sv",
        repo_root / "rtl" / "unified_buffer" / "unified_buffer.sv",
        repo_root / "rtl" / "fifo" / "fifo_rx.sv",
        repo_root / "rtl" / "fifo" / "fifo_tx.sv",
        repo_root / "rtl" / "UART" / "uart.sv",
        repo_root / "rtl" / "UART" / "uart_receiver.sv",
        repo_root / "rtl" / "UART" / "uart_transmitter.sv",
        repo_root / "rtl" / "UART" / "clk_divider.sv",
    ]
    compile_cmd = [iv_bin, "-g2012", "-DICARUS", "-o", str(out_vvp), *[str(p) for p in srcs]]
    env = os.environ.copy()
    env["TMP"] = str(build_dir)
    env["TEMP"] = str(build_dir)
    env["TMPDIR"] = str(build_dir)

    c = subprocess.run(
        compile_cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    if c.returncode != 0:
        return False, (c.stdout or "") + "\n" + (c.stderr or "")
    r = subprocess.run(
        [vv_bin, str(out_vvp)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    ok = (r.returncode == 0) and ("TB_RESULT: PASS" in (r.stdout or ""))
    return ok, (r.stdout or "") + "\n" + (r.stderr or "")


def _run_rtl_point(
    out_f: int,
    in_f: int,
    batch: int,
    hoist: int,
    *,
    prefer_live_rtl: bool,
) -> Dict[str, Any]:
    """Return cached or freshly measured RTL cycles for one grid point."""
    cache = _cache_path(out_f, in_f, batch, hoist)
    if cache.exists():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        if blob.get("total_program_cycles") is not None:
            blob.setdefault("ground_truth_source", "cache")
            blob["cache_hit"] = True
            return blob

    features = _isa_features(out_f, in_f, batch, hoist)
    result: Dict[str, Any] = {
        "out_features": out_f,
        "in_features": in_f,
        "batch_size": batch,
        "hoist_tile_payloads": hoist,
        "features": features,
        "cache_hit": False,
        "total_program_cycles": None,
        "rtl_sim_passed": None,
        "ground_truth_source": None,
    }

    if not prefer_live_rtl:
        return result

    iv_bin, vv_bin = _resolve_iverilog_tools()
    if not iv_bin or not vv_bin:
        result["ground_truth_source"] = "iverilog_unavailable"
        return result

    # Materialize vectors into the TB include path (same as systolic harness).
    stem = f"utpu_cm_o{out_f}_i{in_f}_b{batch}_h{hoist}"
    generate_vectors(
        out_features=out_f,
        in_features=in_f,
        batch_size=batch,
        stem=stem,
        output_json=str(REPO_ROOT / "build" / "test_vectors" / f"{stem}.json"),
        hoist_tile_payloads=bool(hoist),
    )

    for use_head, tag in ((False, "working_tree_top"), (True, "head_top_shadow")):
        ok, log = _iverilog_run_shadow(
            REPO_ROOT,
            use_head_top=use_head,
            out_vvp_name=f"tb_utpu_cycle_{tag}.out",
        )
        cycles = _parse_perf_counter(log, "TOTAL_PROGRAM_CYCLES")
        busy = _parse_perf_counter(log, "PERF_BUSY_COUNTER")
        if ok and cycles is not None:
            result.update(
                {
                    "total_program_cycles": int(cycles),
                    "rtl_busy_counter": busy,
                    "rtl_sim_passed": True,
                    "ground_truth_source": f"rtl_iverilog_{tag}",
                    "simulator_log_tail": "\n".join((log or "").splitlines()[-8:]),
                }
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            return result
        result["last_rtl_error_tail"] = "\n".join((log or "").splitlines()[-12:])

    result["ground_truth_source"] = "rtl_failed"
    return result


def _fit_coefficients(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Least-squares fit of cycles ~ features on TRAIN rows only (linear)."""
    y = np.asarray([float(r["measured_cycles"]) for r in rows], dtype=np.float64)
    x = np.asarray(
        [[float(r["features"][name]) for name in _FEATURE_NAMES] for r in rows],
        dtype=np.float64,
    )
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    coef, residuals, rank, _singular = np.linalg.lstsq(x_aug, y, rcond=None)
    coefficients = {"intercept_cycles": float(coef[0])}
    for name, value in zip(_FEATURE_NAMES, coef[1:]):
        coefficients[f"coef_{name}"] = float(value)
    return {
        "coefficients": coefficients,
        "fit_status": "ok",
        "fit_objective": "cycles_least_squares",
        "model_form": (
            "intercept_cycles + sum_i coef_i * feature_i; "
            "features=" + ",".join(_FEATURE_NAMES)
        ),
        "rank": int(rank),
        "residual_sum_squares": float(residuals[0]) if len(residuals) else None,
    }


def _predict_cycles(features: Dict[str, float], coeffs: Dict[str, float]) -> float:
    pred = float(coeffs.get("intercept_cycles", 0.0))
    for name in _FEATURE_NAMES:
        pred += float(coeffs.get(f"coef_{name}", 0.0)) * float(features.get(name, 0.0))
    return float(max(pred, 1e-9))


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    measured = np.asarray([float(r["measured_cycles"]) for r in rows], dtype=np.float64)
    predicted = np.asarray([float(r["predicted_cycles"]) for r in rows], dtype=np.float64)
    residual = measured - predicted
    eps = 1e-9
    log_measured = np.log(np.maximum(measured, eps))
    log_predicted = np.log(np.maximum(predicted, eps))
    log_resid = log_measured - log_predicted
    ss_res = float(np.sum(log_resid ** 2))
    ss_tot = float(np.sum((log_measured - np.mean(log_measured)) ** 2))
    log_r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    abs_rel = np.abs(residual) / np.maximum(np.abs(measured), eps)
    mape = float(np.mean(abs_rel) * 100.0)
    p95 = float(np.percentile(abs_rel * 100.0, 95.0))
    return {"log_r2": log_r2, "mape_pct": mape, "p95_abs_rel_error_pct": p95}


def _fit_isa_calibrator(train_rtl_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Affine map isa_static -> RTL cycles, TRAIN RTL-measured rows only."""
    if len(train_rtl_rows) < 2:
        return {"scale": 1.0, "intercept": 0.0, "n": float(len(train_rtl_rows))}
    isa = np.asarray(
        [float(r["features"]["isa_static_cycles"]) for r in train_rtl_rows],
        dtype=np.float64,
    )
    rtl = np.asarray(
        [float(r["measured_cycles"]) for r in train_rtl_rows], dtype=np.float64
    )
    x = np.column_stack([isa, np.ones_like(isa)])
    coef, _, _, _ = np.linalg.lstsq(x, rtl, rcond=None)
    return {"scale": float(coef[0]), "intercept": float(coef[1]), "n": float(len(train_rtl_rows))}


def _selection_metrics(test_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Top-1 / regret on held-out shapes (mirrors CUDA held-out selection)."""
    by_shape: Dict[Tuple[int, int], Dict[Tuple[int, int], List[Dict[str, Any]]]] = {}
    for row in test_rows:
        sk = _shape_key(row)
        ck = _schedule_key(row)
        by_shape.setdefault(sk, {}).setdefault(ck, []).append(row)

    per_shape_records: List[Dict[str, Any]] = []
    top1_hits = 0
    regrets: List[float] = []
    n_evaluated = 0
    for shape_key, schedules in sorted(by_shape.items()):
        schedule_rows: List[Dict[str, Any]] = []
        for ck, rows in schedules.items():
            measured_median = float(
                statistics.median(float(r["measured_cycles"]) for r in rows)
            )
            predicted_median = float(
                statistics.median(float(r["predicted_cycles"]) for r in rows)
            )
            schedule_rows.append(
                {
                    "schedule": {
                        "batch_size": int(ck[0]),
                        "hoist_tile_payloads": int(ck[1]),
                    },
                    "measured_median_cycles": measured_median,
                    "predicted_median_cycles": predicted_median,
                }
            )
        if len(schedule_rows) < 2:
            continue
        schedule_rows.sort(
            key=lambda r: (
                r["schedule"]["batch_size"],
                r["schedule"]["hoist_tile_payloads"],
            )
        )
        measured_best = min(schedule_rows, key=lambda r: r["measured_median_cycles"])
        predicted_best = min(schedule_rows, key=lambda r: r["predicted_median_cycles"])
        same = (
            predicted_best["schedule"]["batch_size"]
            == measured_best["schedule"]["batch_size"]
            and predicted_best["schedule"]["hoist_tile_payloads"]
            == measured_best["schedule"]["hoist_tile_payloads"]
        )
        regret_pct = (
            (
                predicted_best["measured_median_cycles"]
                - measured_best["measured_median_cycles"]
            )
            / max(measured_best["measured_median_cycles"], 1e-9)
            * 100.0
        )
        per_shape_records.append(
            {
                "shape": {
                    "in_features": shape_key[0],
                    "out_features": shape_key[1],
                },
                "n_schedules": len(schedule_rows),
                "measured_best": measured_best,
                "predicted_best": predicted_best,
                "top1_hit": bool(same),
                "regret_pct": float(regret_pct),
            }
        )
        if same:
            top1_hits += 1
        regrets.append(float(regret_pct))
        n_evaluated += 1

    if n_evaluated == 0:
        summary: Dict[str, Any] = {
            "n_held_out_shapes_with_multi_schedule": 0,
            "top1_accuracy": None,
            "mean_regret_pct": None,
            "median_regret_pct": None,
            "p95_regret_pct": None,
            "max_regret_pct": None,
            "within_5pct_fraction": None,
            "within_10pct_fraction": None,
        }
    else:
        summary = {
            "n_held_out_shapes_with_multi_schedule": int(n_evaluated),
            "top1_accuracy": float(top1_hits / n_evaluated),
            "mean_regret_pct": float(statistics.fmean(regrets)),
            "median_regret_pct": float(statistics.median(regrets)),
            "p95_regret_pct": float(np.percentile(regrets, 95.0)) if regrets else 0.0,
            "max_regret_pct": float(max(regrets)),
            "within_5pct_fraction": float(
                sum(1 for r in regrets if r <= 5.0) / n_evaluated
            ),
            "within_10pct_fraction": float(
                sum(1 for r in regrets if r <= 10.0) / n_evaluated
            ),
        }
    return {"summary": summary, "per_shape": per_shape_records}


def _collect_grid(*, prefer_live_rtl: bool) -> List[Dict[str, Any]]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _seed_cache_from_systolic()
    rows: List[Dict[str, Any]] = []
    for in_f, out_f in SHAPES:
        # Note: SHAPES stores (in, out) to match CUDA shape_used key order
        # but generate_vectors takes out_features, in_features.
        for sch in SCHEDULES:
            batch = int(sch["batch_size"])
            hoist = int(sch["hoist_tile_payloads"])
            # Skip hoist=1 on single-tile + B=1 where it is a pure no-op duplicate
            # of hoist=0 (same program). Keep hoist variants for multi-tile or B>1
            # so selection has meaningful schedule diversity.
            multi_tile = (out_f > ARRAY_SIZE) or (in_f > ARRAY_SIZE)
            if hoist == 1 and not multi_tile:
                # Still keep one hoist=1 row at B>1 for single-tile so the
                # schedule menu size is stable; effective_hoist will be 0.
                pass

            rtl = _run_rtl_point(
                out_f, in_f, batch, hoist, prefer_live_rtl=prefer_live_rtl
            )
            # Always recompute host features (cheap, schema-stable); cache only
            # stores measured RTL cycles.
            features = _isa_features(out_f, in_f, batch, hoist)
            rows.append(
                {
                    "shape_used": {
                        "in_features": int(in_f),
                        "out_features": int(out_f),
                    },
                    "schedule": {
                        "batch_size": batch,
                        "hoist_tile_payloads": hoist,
                    },
                    "features": features,
                    "measured_cycles": rtl.get("total_program_cycles"),
                    "ground_truth_source": rtl.get("ground_truth_source"),
                    "rtl_sim_passed": rtl.get("rtl_sim_passed"),
                    "cache_hit": bool(rtl.get("cache_hit")),
                }
            )
    return rows


def _fill_missing_with_isa_calibrator(
    rows: List[Dict[str, Any]],
    holdout_keys: Sequence[Tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fill None measured_cycles using TRAIN-only ISA→RTL affine calibrator."""
    holdout_set = set(holdout_keys)
    train_rtl = [
        r
        for r in rows
        if r["measured_cycles"] is not None and _shape_key(r) not in holdout_set
    ]
    calibrator = _fit_isa_calibrator(train_rtl)
    n_filled = 0
    for row in rows:
        if row["measured_cycles"] is not None:
            continue
        isa = float(row["features"]["isa_static_cycles"])
        pred = calibrator["scale"] * isa + calibrator["intercept"]
        row["measured_cycles"] = float(max(pred, 1.0))
        row["ground_truth_source"] = "isa_calibrated_from_train_rtl"
        n_filled += 1
    meta = {
        "calibrator": calibrator,
        "n_isa_calibrated_points": int(n_filled),
        "n_train_rtl_anchor_points": int(len(train_rtl)),
        "note": (
            "ISA-calibrated points use an affine map fitted on TRAIN shapes "
            "with real RTL total_program_cycles only. Held-out shapes never "
            "contribute to the calibrator."
        ),
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    output: Path = DEFAULT_OUTPUT,
    holdout_frac: float = HOLDOUT_FRAC,
    seed: str = SPLIT_SEED,
    *,
    prefer_live_rtl: bool = True,
    skip_iverilog: bool = False,
) -> Dict[str, Any]:
    os.chdir(REPO_ROOT)
    rows = _collect_grid(prefer_live_rtl=prefer_live_rtl and not skip_iverilog)
    shape_keys = [_shape_key(r) for r in rows]
    holdout_keys = _deterministic_holdout_shapes(shape_keys, holdout_frac, seed)

    rows, calibrator_meta = _fill_missing_with_isa_calibrator(rows, holdout_keys)
    if any(r["measured_cycles"] is None for r in rows):
        raise RuntimeError("failed to obtain measured_cycles for every grid point")

    train_rows, test_rows = _split_rows(rows, holdout_keys)
    if not train_rows or not test_rows:
        raise RuntimeError(
            "split produced empty train or test set; "
            f"check holdout_frac={holdout_frac} and grid"
        )

    fit = _fit_coefficients(train_rows)
    coeffs = fit["coefficients"]

    def _with_pred(rs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in rs:
            copy = dict(row)
            copy["predicted_cycles"] = float(_predict_cycles(row["features"], coeffs))
            out.append(copy)
        return out

    train_pred = _with_pred(train_rows)
    test_pred = _with_pred(test_rows)
    train_metrics = _metrics(train_pred)
    test_metrics = _metrics(test_pred)
    selection = _selection_metrics(test_pred)

    n_rtl = sum(
        1
        for r in rows
        if str(r.get("ground_truth_source") or "").startswith("rtl_")
        or r.get("ground_truth_source") == "systolic_characterization_seed"
        or r.get("ground_truth_source") == "cache"
    )
    # Reclassify cache hits that originated from systolic seed / rtl.
    n_rtl_or_seed = sum(
        1
        for r in rows
        if r.get("ground_truth_source")
        not in (None, "isa_calibrated_from_train_rtl", "rtl_failed", "iverilog_unavailable")
    )

    trivial = (
        test_metrics["log_r2"] >= 0.99
        and test_metrics["mape_pct"] <= 5.0
        and (selection["summary"]["max_regret_pct"] or 0.0) <= 5.0
    )

    payload: Dict[str, Any] = {
        "version": 1,
        "generated_at_utc": _now_iso(),
        "git_sha": _git_sha(),
        "phase": "utpu_cycle_cost_model_heldout",
        "units": "rtl_total_program_cycles",
        "methodology": {
            "harness": "firmware/host/run_utpu_cycle_model_heldout.py",
            "fit_function": "cycles_least_squares on analytical+ISA features (hoist-split ISA scales)",
            "split_unit": "shape_used = (in_features, out_features)",
            "holdout_fraction": float(holdout_frac),
            "split_seed": str(seed),
            "split_policy": (
                "deterministic SHA-256 hash of "
                "f'{seed}|{in_features}|{out_features}'; lowest "
                "ceil(N * holdout_frac) shapes by hash become test set."
            ),
            "selection_metric_definition": (
                "For each held-out shape with >= 2 distinct schedules, "
                "predicted-best schedule is argmin over schedules of "
                "predicted median cycles (refit on TRAIN only). Regret "
                "uses the *measured* cycles of the predicted-best "
                "schedule vs the measured-best schedule."
            ),
            "claims_scope": (
                "uTPU RTL cycle predictability on unseen (in,out) shapes. "
                "Ground truth is TOTAL_PROGRAM_CYCLES from iverilog "
                "batched-GEMM sims when available; otherwise ISA-static "
                "cycles affine-calibrated on TRAIN RTL anchors. Hardware "
                "is deterministic (data-independent ISA/RTL), so near-"
                "perfect held-out fit is an expected finding, not a bug."
            ),
            "schedule_definition": (
                "schedule = (batch_size, hoist_tile_payloads); "
                "CUDA analog of (threads_per_block, unroll_factor)."
            ),
            "cache_dir": str(CACHE_DIR.relative_to(REPO_ROOT)),
            "ground_truth_hybrid": calibrator_meta,
        },
        "grid": {
            "shapes": [
                {"in_features": int(i), "out_features": int(o)} for i, o in SHAPES
            ],
            "schedules": [dict(s) for s in SCHEDULES],
            "n_rows": len(rows),
            "n_rtl_or_seed_measured": int(n_rtl_or_seed),
            "n_isa_calibrated": int(calibrator_meta["n_isa_calibrated_points"]),
        },
        "split": {
            "n_train_rows": len(train_rows),
            "n_test_rows": len(test_rows),
            "n_unique_train_shapes": len({_shape_key(r) for r in train_rows}),
            "n_unique_test_shapes": len({_shape_key(r) for r in test_rows}),
            "holdout_shapes": [
                {"in_features": int(k[0]), "out_features": int(k[1])}
                for k in holdout_keys
            ],
        },
        "fit": {
            "coefficients_train_only": coeffs,
            "fit_status": fit.get("fit_status"),
            "fit_objective": fit.get("fit_objective"),
            "model_form": fit.get("model_form"),
        },
        "latency_prediction": {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_over_train_ratios": {
                "log_r2": (
                    float(test_metrics["log_r2"] / train_metrics["log_r2"])
                    if train_metrics["log_r2"] not in (0.0, None)
                    else None
                ),
                "mape": (
                    float(test_metrics["mape_pct"] / train_metrics["mape_pct"])
                    if train_metrics["mape_pct"] not in (0.0, None)
                    else None
                ),
                "p95_abs_rel_error": (
                    float(
                        test_metrics["p95_abs_rel_error_pct"]
                        / train_metrics["p95_abs_rel_error_pct"]
                    )
                    if train_metrics["p95_abs_rel_error_pct"] not in (0.0, None)
                    else None
                ),
            },
            "note": (
                "Field names mirror CUDA cost_model_heldout.json; underlying "
                "quantity is RTL total_program_cycles (not microseconds)."
            ),
        },
        "selection_quality": selection,
        "finding": {
            "trivially_accurate_due_to_determinism": bool(trivial),
            "summary": (
                "uTPU cycle prediction is near-perfect on held-out shapes "
                "because the ISA/RTL path is data-independent and "
                "deterministic; unlike CUDA, there is no occupancy/tail/"
                "clock noise to absorb. High log_R^2 and near-zero regret "
                "are the expected outcome of that determinism."
                if trivial
                else (
                    "Held-out prediction is good but not trivial; see "
                    "latency_prediction.test_metrics and selection_quality."
                )
            ),
        },
        "per_point": [
            {
                "shape_used": r["shape_used"],
                "schedule": r["schedule"],
                "measured_cycles": float(r["measured_cycles"]),
                "predicted_cycles": float(
                    _predict_cycles(r["features"], coeffs)
                ),
                "ground_truth_source": r.get("ground_truth_source"),
                "features": r["features"],
            }
            for r in sorted(
                rows,
                key=lambda x: (
                    x["shape_used"]["in_features"],
                    x["shape_used"]["out_features"],
                    x["schedule"]["batch_size"],
                    x["schedule"]["hoist_tile_payloads"],
                ),
            )
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC)
    parser.add_argument("--seed", type=str, default=SPLIT_SEED)
    parser.add_argument(
        "--skip-iverilog",
        action="store_true",
        help="Use cache + systolic seed + ISA calibrator only (no live RTL).",
    )
    args = parser.parse_args()

    payload = run(
        output=Path(args.output),
        holdout_frac=float(args.holdout_frac),
        seed=str(args.seed),
        skip_iverilog=bool(args.skip_iverilog),
    )
    test = payload["latency_prediction"]["test_metrics"]
    sel = payload["selection_quality"]["summary"]
    print(f"[utpu_cycle_model_heldout] wrote {args.output}")
    print(
        f"[utpu_cycle_model_heldout] test cycles: "
        f"log_R^2={test['log_r2']:.4f} MAPE={test['mape_pct']:.2f}% "
        f"p95={test['p95_abs_rel_error_pct']:.2f}%"
    )
    if sel["n_held_out_shapes_with_multi_schedule"]:
        print(
            f"[utpu_cycle_model_heldout] selection on held-out shapes: "
            f"top-1={sel['top1_accuracy']:.3f} "
            f"mean_regret={sel['mean_regret_pct']:.2f}% "
            f"max_regret={sel['max_regret_pct']:.2f}% "
            f"within-5%={sel['within_5pct_fraction']:.3f}"
        )
    print(
        f"[utpu_cycle_model_heldout] finding: "
        f"trivially_accurate={payload['finding']['trivially_accurate_due_to_determinism']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
