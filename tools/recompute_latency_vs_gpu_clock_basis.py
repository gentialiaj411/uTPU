#!/usr/bin/env python3
"""Recompute latency_determinism_vs_gpu.json clock basis without re-timing GPU.

Shipping clock: 12 ns / ~83.333 MHz (WNS=+0.271, thin) — headline.
Ceiling clock: 10 ns / 100 MHz (WNS=+0.012, marginal) — must quote WNS.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "bench" / "results" / "latency_determinism_vs_gpu.json"
PLOT = REPO / "docs" / "latency_determinism_vs_gpu_logx.png"

SHIPPING_MHZ = 1000.0 / 12.0  # 83.333...
SHIPPING_PERIOD_NS = 12.0
SHIPPING_WNS = 0.271
SHIPPING_MARGIN = "thin"
SHIPPING_SOURCE = (
    "bench/results/design_space_sweep.json::shipping_point "
    "N=8 INT8 MAX_BATCH_COUNT=48 @ 12 ns (WNS=+0.271, margin_class=thin)"
)

CEILING_MHZ = 100.0
CEILING_PERIOD_NS = 10.0
CEILING_WNS = 0.012
CEILING_MARGIN = "marginal"
CEILING_SOURCE = (
    "bench/results/design_space_sweep.json::demonstrated_fmax_ceiling "
    "N=8 INT8 mb48 @ 10 ns (WNS=+0.012, margin_class=marginal). "
    "Any 100 MHz claim must quote WNS=+0.012 ns inline."
)


def _pct(sorted_vals, p):
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = min(f + 1, n - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def wall_stats(cycles, mhz):
    samples = [c * (1000.0 / mhz) for c in cycles]
    s = sorted(samples)
    return {
        "samples_ns": samples,
        "p50_ns": _pct(s, 50),
        "p90_ns": _pct(s, 90),
        "p99_ns": _pct(s, 99),
        "p99_9_ns": _pct(s, 99.9),
        "max_ns": float(max(s)) if s else None,
        "min_ns": float(min(s)) if s else None,
        "mean_ns": float(statistics.mean(s)) if s else None,
        "stddev_ns": float(statistics.pstdev(s)) if len(s) > 1 else 0.0,
    }


def main() -> int:
    data = json.loads(ART.read_text(encoding="utf-8"))
    fpga = data["fpga_arm"]
    cycles = [int(c) for c in fpga["rtl_cycles_observed"]]
    assert cycles and max(cycles) == min(cycles)

    ship = wall_stats(cycles, SHIPPING_MHZ)
    ceil = wall_stats(cycles, CEILING_MHZ)

    fpga["clock_mhz"] = SHIPPING_MHZ
    fpga["clock_period_ns"] = SHIPPING_PERIOD_NS
    fpga["clock_source"] = SHIPPING_SOURCE
    fpga["wns_ns"] = SHIPPING_WNS
    fpga["margin_class"] = SHIPPING_MARGIN
    fpga["conversion"] = (
        f"wall_ns = cycles * (1000 / clock_mhz)  # {SHIPPING_PERIOD_NS} ns/cycle "
        f"at shipping ~{SHIPPING_MHZ:.3f} MHz"
    )
    fpga["stats"] = {
        "p50_ns": ship["p50_ns"],
        "p90_ns": ship["p90_ns"],
        "p99_ns": ship["p99_ns"],
        "p99_9_ns": ship["p99_9_ns"],
        "max_ns": ship["max_ns"],
        "min_ns": ship["min_ns"],
        "mean_ns": ship["mean_ns"],
        "stddev_ns": ship["stddev_ns"],
    }
    for k in ("p50_ns", "p90_ns", "p99_ns", "p99_9_ns", "max_ns", "stddev_ns"):
        fpga[k] = ship[k]
    fpga["jitter_ns"] = 0.0
    fpga["samples_ns"] = ship["samples_ns"]
    fpga["shipping_clock"] = {
        "clock_mhz": SHIPPING_MHZ,
        "clock_period_ns": SHIPPING_PERIOD_NS,
        "wns_ns": SHIPPING_WNS,
        "margin_class": SHIPPING_MARGIN,
        "source": SHIPPING_SOURCE,
        "p50_ns": ship["p50_ns"],
        "role": "headline_shipping_default",
    }
    fpga["demonstrated_ceiling_clock"] = {
        "clock_mhz": CEILING_MHZ,
        "clock_period_ns": CEILING_PERIOD_NS,
        "wns_ns": CEILING_WNS,
        "margin_class": CEILING_MARGIN,
        "source": CEILING_SOURCE,
        "p50_ns": ceil["p50_ns"],
        "role": "demonstrated_ceiling_not_shipping",
        "claim_rule": (
            "Any claim citing 100 MHz must carry WNS=+0.012 ns inline "
            "(margin_class=marginal). Shipping default remains 12 ns / ~83 MHz."
        ),
    }

    gpu = data["gpu_arm"]
    gpu_p50 = float((gpu.get("stats") or {}).get("p50_ns") or gpu.get("p50_ns"))
    ship_p50 = float(ship["p50_ns"])
    ceil_p50 = float(ceil["p50_ns"])

    comparison = data["comparison"]
    comparison["fpga_tail_from_cycle_conversion"] = {
        "clock_basis": "shipping_12ns",
        "clock_mhz": SHIPPING_MHZ,
        "wns_ns": SHIPPING_WNS,
        "margin_class": SHIPPING_MARGIN,
        "conversion": f"wall_ns = cycles * {SHIPPING_PERIOD_NS}  # shipping",
        "p50_ns": ship["p50_ns"],
        "p90_ns": ship["p90_ns"],
        "p99_ns": ship["p99_ns"],
        "p99_9_ns": ship["p99_9_ns"],
        "max_ns": ship["max_ns"],
        "stddev_ns": ship["stddev_ns"],
    }
    comparison["fpga_tail_at_100mhz_ceiling"] = {
        "clock_basis": "demonstrated_ceiling_10ns",
        "clock_mhz": CEILING_MHZ,
        "wns_ns": CEILING_WNS,
        "margin_class": CEILING_MARGIN,
        "conversion": "wall_ns = cycles * 10  # ceiling only",
        "p50_ns": ceil["p50_ns"],
        "p90_ns": ceil["p90_ns"],
        "p99_ns": ceil["p99_ns"],
        "p99_9_ns": ceil["p99_9_ns"],
        "max_ns": ceil["max_ns"],
        "stddev_ns": ceil["stddev_ns"],
        "claim_rule": fpga["demonstrated_ceiling_clock"]["claim_rule"],
    }
    comparison["median_latency_loss"] = {
        "definition": (
            "median_latency_loss_factor = fpga_p50_ns / gpu_p50_ns at the "
            "SHIPPING clock (12 ns / ~83.333 MHz, WNS=+0.271). "
            "Values >1 mean FPGA median is slower. Does NOT claim FPGA faster."
        ),
        "clock_basis": "shipping_12ns",
        "clock_mhz": SHIPPING_MHZ,
        "wns_ns": SHIPPING_WNS,
        "margin_class": SHIPPING_MARGIN,
        "fpga_p50_ns": ship_p50,
        "gpu_p50_ns": gpu_p50,
        "median_latency_loss_factor": ship_p50 / gpu_p50,
        "median_latency_loss_ns": ship_p50 - gpu_p50,
        "populated": True,
        "speedup_claim_forbidden": True,
        "reason": (
            "Headline uses shipping 12 ns / ~83 MHz (thin). "
            "100 MHz is the demonstrated ceiling only (see "
            "median_latency_loss_at_100mhz_ceiling)."
        ),
    }
    comparison["median_latency_loss_at_100mhz_ceiling"] = {
        "clock_basis": "demonstrated_ceiling_10ns",
        "clock_mhz": CEILING_MHZ,
        "wns_ns": CEILING_WNS,
        "margin_class": CEILING_MARGIN,
        "fpga_p50_ns": ceil_p50,
        "gpu_p50_ns": gpu_p50,
        "median_latency_loss_factor": ceil_p50 / gpu_p50,
        "median_latency_loss_ns": ceil_p50 - gpu_p50,
        "populated": True,
        "speedup_claim_forbidden": True,
        "claim_rule": fpga["demonstrated_ceiling_clock"]["claim_rule"],
        "note": "Not the headline; shipping default is 12 ns / ~83 MHz.",
    }
    comparison["claim_text"] = (
        "FPGA/RTL end-to-end inference latency is data-independent "
        "(cycle variance == 0) . Headline wall-clock uses the shipping "
        f"{SHIPPING_MHZ:.3f} MHz close (12 ns, WNS=+{SHIPPING_WNS} ns, thin). "
        f"100 MHz is the demonstrated ceiling (WNS=+{CEILING_WNS} ns, marginal) "
        "and must quote WNS if cited. Claim is bounded jitter + median-latency "
        "loss, NOT speedup."
    )
    comparison["fpga_jitter_ns"] = 0.0
    data["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    data["clock_basis_note"] = (
        "Reconciled with design-space shipping point (12 ns) and "
        "demonstrated 100 MHz ceiling (WNS=+0.012)."
    )

    ART.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        "updated",
        ART,
        "ship_factor",
        round(ship_p50 / gpu_p50, 4),
        "ceil_factor",
        round(ceil_p50 / gpu_p50, 4),
    )

    # Regen plot at shipping ns.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("plot skipped", exc)
        return 0

    gpu_samples = (gpu.get("samples_ns") or [])[:5000]
    if not gpu_samples and (gpu.get("stats") or {}).get("p50_ns"):
        # Fall back: reconstruct approximate from percentiles only for axes.
        gpu_samples = []
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    if gpu.get("status") == "ok":
        # Prefer samples file if present
        sp = REPO / "bench" / "results" / "_latency_vs_gpu_cuda_subprocess.samples.json"
        if sp.exists():
            raw = json.loads(sp.read_text(encoding="utf-8"))
            gpu_samples = [float(x) for x in (raw.get("samples_ns") or raw if isinstance(raw, list) else [])]
        if gpu_samples:
            ax.hist(
                gpu_samples,
                bins=np.logspace(np.log10(max(min(gpu_samples), 1e2)), np.log10(max(gpu_samples)), 60),
                alpha=0.55,
                label="GPU kernel (N≥10000)",
                color="#ff7f0e",
            )
    ax.axvline(ship_p50, color="#1f77b4", lw=2, label=f"FPGA p50 @ ~83 MHz shipping ({ship_p50:.0f} ns)")
    ax.axvline(
        ceil_p50,
        color="#1f77b4",
        lw=1.5,
        ls="--",
        label=f"FPGA p50 @ 100 MHz ceiling WNS=+0.012 ({ceil_p50:.0f} ns)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Latency (ns, log)")
    ax.set_ylabel("Count")
    ax.set_title("Bounded jitter vs GPU tail (shipping 83 MHz headline)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT, dpi=140)
    plt.close(fig)
    print("wrote", PLOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
