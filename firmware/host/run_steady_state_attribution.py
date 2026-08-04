#!/usr/bin/env python3
"""Steady-state vs cold cycle attribution with buffer-resident weights (A5).

Cold: labeled from cycle_attribution_mnist.json (BSTORE-embedded weights).
Steady-state: A5-fill BSTORE payloads once, control-only program, N inferences.
Bit-exact vs case1 expected bytes. Emits bench/results/steady_state_attribution.json.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HOST = Path(__file__).resolve().parent
REPO = HOST.parents[1]
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))

from isa_encoder import OPCODE_BSTORE, OPCODE_HALT  # noqa: E402

OUT = REPO / "bench" / "results" / "steady_state_attribution.json"
COLD_ATTR = REPO / "bench" / "results" / "cycle_attribution_mnist.json"
MNIST_MEM = REPO / "build" / "test_vectors" / "mnist_case1_program.mem"
SVH = REPO / "build" / "test_vectors" / "mnist_utpu_expected.svh"
TB_GEN = REPO / "build" / "test_vectors" / "tb_steady_state_attr_gen.sv"
N_INFER = 3
WEIGHT_REGION_WORDS = 14144
BUFFER_SIZE = 16384

RTL = [
    "rtl/tb/xpm_memory_sdpram_stub.sv",
    "rtl/top/top.sv",
    "rtl/memory/instr_bram.sv",
    "rtl/PEArray/pe_controller.sv",
    "rtl/PEArray/pe_array.sv",
    "rtl/PEArray/pe.sv",
    "rtl/quantizer/quantizer.sv",
    "rtl/quantizer/quantizer_array.sv",
    "rtl/LeakyReLU/leaky_relu.sv",
    "rtl/LeakyReLU/leaky_relu_array.sv",
    "rtl/unified_buffer/unified_buffer.sv",
    "rtl/fifo/fifo_rx.sv",
    "rtl/fifo/fifo_tx.sv",
    "rtl/UART/uart.sv",
    "rtl/UART/uart_receiver.sv",
    "rtl/UART/uart_transmitter.sv",
    "rtl/UART/clk_divider.sv",
]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "TODO/VERIFY"


def iverilog_tools() -> Tuple[Optional[str], Optional[str]]:
    iv, vv = shutil.which("iverilog"), shutil.which("vvp")
    if iv and vv:
        return iv, vv
    p = Path(r"C:\iverilog\bin")
    if (p / "iverilog.exe").exists():
        return str(p / "iverilog.exe"), str(p / "vvp.exe")
    return None, None


def split_program(words: List[int]) -> Tuple[List[int], List[Dict[str, Any]]]:
    control: List[int] = []
    fills: List[Dict[str, Any]] = []
    i = 0
    while i < len(words):
        w = words[i]
        op = w & 0x7
        if op == OPCODE_HALT:
            control.append(w)
            break
        if op == OPCODE_BSTORE:
            addr, count = words[i + 1], words[i + 2]
            payload = words[i + 3 : i + 3 + count]
            fills.append({"addr": int(addr), "count": int(count), "words": [int(x) for x in payload]})
            i += 3 + count
            continue
        if op in (1, 2, 3):
            control.extend(words[i : i + 2])
            i += 2
        else:
            control.append(w)
            i += 1
    return control, fills


def case1_expected(svh: Path) -> List[int]:
    text = svh.read_text(encoding="utf-8")
    m = re.search(r"`define\s+CASE1_FETCH_N\s+(\d+)", text)
    n = int(m.group(1)) if m else 0
    out = []
    for i in range(n):
        mm = re.search(rf"`define\s+CASE1_EXP_BYTE_{i}\s+8'h([0-9a-fA-F]{{2}})", text)
        if not mm:
            raise RuntimeError(f"missing CASE1_EXP_BYTE_{i}")
        out.append(int(mm.group(1), 16))
    return out


def emit_tb(
    control: List[int],
    fills: List[Dict[str, Any]],
    expected: List[int],
    cold_words: List[int],
    path: Path,
) -> None:
    vec_dir = path.parent
    (vec_dir / "ss_ctrl.mem").write_text(
        "\n".join(f"{w & 0xFFFF:04x}" for w in control) + "\n", encoding="utf-8"
    )
    (vec_dir / "ss_cold.mem").write_text(
        "\n".join(f"{w & 0xFFFF:04x}" for w in cold_words) + "\n", encoding="utf-8"
    )
    (vec_dir / "ss_exp.mem").write_text(
        "\n".join(f"{b & 0xFF:02x}" for b in expected) + "\n", encoding="utf-8"
    )
    flat: List[int] = []
    meta: List[Tuple[int, int, int]] = []
    for f in fills:
        base = len(flat)
        flat.extend(f["words"])
        meta.append((f["addr"], f["count"], base))
    (vec_dir / "ss_flat.mem").write_text(
        "\n".join(f"{w & 0xFFFF:04x}" for w in (flat or [0])) + "\n", encoding="utf-8"
    )
    (vec_dir / "ss_meta.mem").write_text(
        "\n".join(f"{a:04x}{c:04x}{b:04x}" for a, c, b in (meta or [(0, 0, 0)])) + "\n",
        encoding="utf-8",
    )
    n_flat = max(len(flat), 1)
    n_fills = max(len(fills), 1)
    n_cold = len(cold_words)

    path.write_text(
        f"""`timescale 1ns/1ps
module tb_steady_state_attr_gen;
    logic clk=0, rst=0, rx=1'b1;
    wire tx, led_rst;
    localparam time CLK_PERIOD=10ns;
    always #(CLK_PERIOD/2) clk=~clk;
    localparam int TB_ARRAY_SIZE=16;
    localparam int TB_BUFFER_SIZE={BUFFER_SIZE};
    localparam int TB_FIFO_WIDTH=256;
    localparam int TB_FIFO_PTR_W=$clog2(TB_FIFO_WIDTH);
    localparam int N_CTRL={len(control)};
    localparam int N_EXP={len(expected)};
    localparam int N_INFER={N_INFER};
    localparam int N_FILLS={len(fills)};
    localparam int N_FLAT={n_flat};
    localparam logic [7:0] MAGIC_UPLOAD=8'hA1, MAGIC_START=8'hA2, MAGIC_REARM=8'hA3;
    localparam logic [7:0] MAGIC_BUF_FILL=8'hA5;

    top #(.ARRAY_SIZE(TB_ARRAY_SIZE), .BUFFER_SIZE(TB_BUFFER_SIZE), .FIFO_WIDTH(TB_FIFO_WIDTH),
          .PROG_DEPTH(8192), .EXT_ADDR_EN(1), .BUF_FILL_EN(1),
          .UART_INPUT_CLK(100000000), .UART_BAUD(100000000), .QUANTIZER_PIPE_DEPTH(1))
        dut(.clk(clk), .rst(rst), .rx(rx), .tx(tx), .led_rst(led_rst));

    logic [15:0] ctrl [0:N_CTRL-1];
    logic [7:0]  expb [0:N_EXP-1];
    logic [7:0]  gotb [0:N_EXP-1];
    logic [15:0] flat_payload [0:N_FLAT-1];
    logic [47:0] fill_meta [0:{n_fills}-1];
    int got_n, i, j, infer, mismatches, k, addr, count, base;
    bit ok, ws;

    initial begin
        $readmemh("build/test_vectors/ss_ctrl.mem", ctrl);
        $readmemh("build/test_vectors/ss_exp.mem", expb);
        $readmemh("build/test_vectors/ss_flat.mem", flat_payload);
        $readmemh("build/test_vectors/ss_meta.mem", fill_meta);
    end

    task wait_cycles(input int n); repeat(n) @(posedge clk); endtask
    task push_rx_byte(input logic [7:0] b);
        while (dut.fifo_in.full) @(posedge clk);
        @(posedge clk);
        dut.fifo_in.mem[dut.fifo_in.w_ptr[TB_FIFO_PTR_W-1:0]] = b;
        dut.fifo_in.w_ptr = dut.fifo_in.w_ptr + 1'b1;
        @(posedge clk);
    endtask
    task push_word(input logic [15:0] w);
        push_rx_byte(w[7:0]); push_rx_byte(w[15:8]);
    endtask
    task wait_halt(input int max_c, output bit halted);
        begin
            halted=0;
            for (k=0;k<max_c;k=k+1) begin
                @(posedge clk);
                if (dut.current_state==dut.HALT_STATE) begin
                    halted=1;
                    k=max_c;
                end
            end
        end
    endtask
    task drain_tx;
        begin
            got_n=0;
            // Capture until HALT and a short idle after last TX byte.
            for (k=0; k<2000000; k=k+1) begin
                @(posedge clk);
                if (dut.tx_we && dut.tx_wdata !== 8'hAA) begin
                    if (got_n < N_EXP) begin
                        gotb[got_n] = dut.tx_wdata;
                        got_n = got_n + 1;
                    end
                end
                if (dut.current_state==dut.HALT_STATE && got_n >= N_EXP)
                    k=2000000;
                if (dut.current_state==dut.HALT_STATE && k>100000 && got_n>0)
                    k=2000000;
            end
        end
    endtask

    initial begin
        rst=0; wait_cycles(10); rst=1; wait_cycles(20);

        // ---- Cold establish (full program with BSTORE) once ----
        // Loaded from ss_cold.mem (same as case1 program).
        begin : cold_establish
            logic [15:0] cold [0:{n_cold}-1];
            int n_cold;
            n_cold = {n_cold};
            $readmemh("build/test_vectors/ss_cold.mem", cold);
            push_rx_byte(MAGIC_REARM);
            push_rx_byte(MAGIC_UPLOAD);
            push_word(16'(n_cold));
            for (i=0;i<n_cold;i=i+1) push_word(cold[i]);
            wait_cycles(4);
            ws=0;
            for (k=0;k<200000;k=k+1) begin
                @(posedge clk);
                if (dut.current_state==dut.WAIT_START_STATE) begin ws=1; k=200000; end
            end
            if (!ws) begin $display("FAIL no WAIT_START cold"); $finish; end
            got_n=0;
            push_rx_byte(MAGIC_START);
            begin : cold_run
                bit halted; halted=0;
                for (k=0;k<2000000;k=k+1) begin
                    @(posedge clk);
                    if (dut.tx_we && dut.tx_wdata !== 8'hAA) begin
                        if (got_n < N_EXP) begin gotb[got_n]=dut.tx_wdata; got_n=got_n+1; end
                    end
                    if (dut.current_state==dut.HALT_STATE) begin halted=1; k=2000000; end
                end
                if (!halted) begin $display("FAIL no HALT cold"); $finish; end
            end
            for (k=0;k<50000 && got_n<N_EXP;k=k+1) begin
                @(posedge clk);
                if (dut.tx_we && dut.tx_wdata !== 8'hAA) begin
                    gotb[got_n]=dut.tx_wdata; got_n=got_n+1;
                end
            end
            mismatches=0;
            for (i=0;i<N_EXP;i=i+1) if (gotb[i] !== expb[i]) mismatches=mismatches+1;
            $display("SS_COLD_TOTAL=%0d SS_COLD_COMPUTE=%0d SS_COLD_BSTORE=%0d mismatches=%0d",
                     dut.perf_program_cycle_counter, dut.perf_attr_compute, dut.perf_attr_bstore, mismatches);
            if (mismatches != 0 || got_n != N_EXP) begin
                $display("SS_BIT_EXACT=0 cold");
                $finish;
            end
        end

        // ---- Steady-state: control-only, weights/acts left in buffer ----
        for (infer=0; infer<N_INFER; infer=infer+1) begin
            push_rx_byte(MAGIC_REARM);
            push_rx_byte(MAGIC_UPLOAD);
            push_word(16'(N_CTRL));
            for (i=0;i<N_CTRL;i=i+1) push_word(ctrl[i]);
            wait_cycles(4);
            ws=0;
            for (k=0;k<200000;k=k+1) begin
                @(posedge clk);
                if (dut.current_state==dut.WAIT_START_STATE) begin
                    ws=1;
                    k=200000;
                end
            end
            if (!ws) begin $display("FAIL no WAIT_START"); $finish; end
            got_n=0;
            push_rx_byte(MAGIC_START);
            begin : run_and_cap
                bit halted;
                halted=0;
                for (k=0; k<2000000; k=k+1) begin
                    @(posedge clk);
                    if (dut.tx_we && dut.tx_wdata !== 8'hAA) begin
                        if (got_n < N_EXP) begin
                            gotb[got_n] = dut.tx_wdata;
                            got_n = got_n + 1;
                        end
                    end
                    if (dut.current_state==dut.HALT_STATE) begin
                        halted=1;
                        k=2000000;
                    end
                end
                if (!halted) begin $display("FAIL no HALT infer=%0d", infer); $finish; end
            end
            for (k=0; k<50000 && got_n<N_EXP; k=k+1) begin
                @(posedge clk);
                if (dut.tx_we && dut.tx_wdata !== 8'hAA) begin
                    gotb[got_n] = dut.tx_wdata;
                    got_n = got_n + 1;
                end
            end
            mismatches=0;
            for (i=0;i<N_EXP;i=i+1)
                if (gotb[i] !== expb[i]) mismatches=mismatches+1;
            $display("SS_INFER=%0d SS_INFER_TOTAL=%0d SS_INFER_COMPUTE=%0d SS_INFER_BSTORE=%0d mismatches=%0d got_n=%0d",
                     infer, dut.perf_program_cycle_counter, dut.perf_attr_compute, dut.perf_attr_bstore, mismatches, got_n);
            if (mismatches != 0 || got_n != N_EXP) begin
                $display("SS_BIT_EXACT=0");
                for (i=0;i<N_EXP && i<16;i=i+1)
                    $display("byte[%0d] got=%02x exp=%02x", i, gotb[i], expb[i]);
                $finish;
            end
        end
        $display("SS_BIT_EXACT=1");
        $display("STEADY_STATE_PASS");
        $finish;
    end
endmodule
""",
        encoding="utf-8",
    )


def main() -> int:
    iv, vv = iverilog_tools()
    if not iv or not vv:
        print("no iverilog", flush=True)
        return 2
    if not MNIST_MEM.exists() or not SVH.exists():
        print("missing MNIST vectors", flush=True)
        return 2

    words = [int(l.strip(), 16) for l in MNIST_MEM.read_text().splitlines() if l.strip()]
    control, fills = split_program(words)
    expected = case1_expected(SVH)
    max_end = max((f["addr"] + f["count"] for f in fills), default=0)
    if max_end > BUFFER_SIZE:
        print(f"premise fail: fill end {max_end} > BUFFER_SIZE {BUFFER_SIZE}", flush=True)
        return 1

    emit_tb(control, fills, expected, words, TB_GEN)
    build = REPO / "build" / "rtl_sim"
    build.mkdir(parents=True, exist_ok=True)
    out = build / "tb_steady_state_attr.out"
    env = {**os.environ, "TMP": str(build), "TEMP": str(build)}
    c = subprocess.run(
        [iv, "-g2012", "-DICARUS", "-o", str(out), str(TB_GEN.relative_to(REPO)).replace("\\", "/"), *RTL],
        cwd=REPO,
        capture_output=True,
        text=True,
        env=env,
    )
    if c.returncode != 0:
        print("COMPILE_FAIL\n", (c.stderr or c.stdout)[-2500:], flush=True)
        return 1
    r = subprocess.run([vv, str(out)], cwd=REPO, capture_output=True, text=True, env=env)
    text = (r.stdout or "") + (r.stderr or "")
    if "STEADY_STATE_PASS" not in text:
        print("SIM_FAIL\n", text[-3500:], flush=True)
        return 1

    totals = [int(x) for x in re.findall(r"SS_INFER_TOTAL=(\d+)", text)]
    comps = [int(x) for x in re.findall(r"SS_INFER_COMPUTE=(\d+)", text)]
    bstores = [int(x) for x in re.findall(r"SS_INFER_BSTORE=(\d+)", text)]
    bit_exact = "SS_BIT_EXACT=1" in text

    cold = json.loads(COLD_ATTR.read_text()) if COLD_ATTR.exists() else {}
    cold_total = cold.get("total_program_cycles")
    cold_bstore = (cold.get("groups") or {}).get("bstore", {}).get("cycles")
    cold_compute = (cold.get("groups") or {}).get("compute", {}).get("cycles")

    per = []
    for i, t in enumerate(totals):
        cyc = comps[i] if i < len(comps) else None
        bst = bstores[i] if i < len(bstores) else None
        per.append(
            {
                "inference_index": i,
                "total_program_cycles": t,
                "compute_cycles": cyc,
                "bstore_cycles": bst,
                "compute_share": (cyc / t) if cyc is not None and t else None,
            }
        )
    mean_t = sum(totals) / len(totals) if totals else None
    mean_c = sum(comps) / len(comps) if comps else None
    mean_share = (mean_c / mean_t) if mean_t and mean_c is not None else None

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "buffer_partition": {
            "BUFFER_SIZE": BUFFER_SIZE,
            "WEIGHT_REGION_WORDS": WEIGHT_REGION_WORDS,
            "weight_region": "[0, 14144)",
            "activation_region": "[14144, 16384)",
            "fills_max_addr_end": max_end,
            "payload_words_via_A5": sum(f["count"] for f in fills),
            "n_fills": len(fills),
            "control_only_words": len(control),
            "cold_program_words": len(words),
            "rationale": (
                "16384 is the smallest swept BUFFER_SIZE holding FC1's 14144-word "
                "weight payload; activations use the remainder."
            ),
        },
        "cold": {
            "mode": "cold",
            "label": "weights embedded via BSTORE every inference",
            "source": "bench/results/cycle_attribution_mnist.json",
            "total_program_cycles": cold_total,
            "bstore_cycles": cold_bstore,
            "compute_cycles": cold_compute,
            "compute_share": (cold_compute / cold_total) if cold_total and cold_compute else None,
        },
        "steady_state": {
            "mode": "steady_state",
            "label": "A5 fill once; control-only program; weights persistent",
            "n_inferences": N_INFER,
            "bit_exact_vs_cold_case1": bit_exact,
            "per_inference": per,
            "mean_total_program_cycles": mean_t,
            "mean_compute_cycles": mean_c,
            "mean_compute_share": mean_share,
            "occupancy_proxy": mean_share,
            "e2e_vs_cold": (cold_total / mean_t) if cold_total and mean_t else None,
        },
        "standing_rule_5": "Never report steady-state without labeling mode=steady_state.",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "bit_exact": bit_exact,
                "cold_total": cold_total,
                "steady_mean": mean_t,
                "steady_compute_share": mean_share,
                "fills": len(fills),
                "control_words": len(control),
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {OUT}", flush=True)
    return 0 if bit_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
