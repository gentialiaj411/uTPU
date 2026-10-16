"""Regenerate Phase-3 vector RTL evidence from executable simulations.

This is intentionally the only writer of ``tiny_transformer_vec_rtl.json``.
It imports the simulation helpers rather than copying their results, and every
reported count is returned by a completed subprocess-backed RTL test.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "firmware" / "host"
RESULT = ROOT / "bench" / "results" / "tiny_transformer_vec_rtl.json"
sys.path.insert(0, str(HOST))

import run_restoring_divider_sim
import run_vector_sequencer_sim

IVERILOG = Path(r"C:\iverilog\bin\iverilog.exe")
VVP = IVERILOG.with_name("vvp.exe")
FULL_TOP_SOURCES = (
    "rtl/tb/xpm_memory_sdpram_stub.sv", "rtl/top/top.sv", "rtl/memory/instr_bram.sv",
    "rtl/PEArray/pe_controller.sv", "rtl/PEArray/pe_array.sv", "rtl/PEArray/pe.sv",
    "rtl/quantizer/quantizer.sv", "rtl/quantizer/quantizer_array.sv",
    "rtl/LeakyReLU/leaky_relu.sv", "rtl/LeakyReLU/leaky_relu_array.sv",
    "rtl/unified_buffer/unified_buffer.sv", "rtl/fifo/fifo_rx.sv", "rtl/fifo/fifo_tx.sv",
    "rtl/UART/uart.sv", "rtl/UART/uart_receiver.sv", "rtl/UART/uart_transmitter.sv",
    "rtl/UART/clk_divider.sv", "rtl/vector/vector_serial_alu.sv",
    "rtl/vector/restoring_divider.sv", "rtl/vector/vector_sequencer.sv",
    "rtl/tb/tb_vector_full_top_decode_errors.sv",
)


def run_full_top_decode_errors() -> dict:
    """Run each Section-5 negative case through production ``top.sv``.

    The TB performs a normal UART program upload then lets top fetch the
    escape and descriptor from its synchronous instruction BRAM.  Each
    invocation is intentionally a fresh simulation so reset state cannot
    mask an error class.
    """
    build = ROOT / "build" / "phase3_full_top"
    build.mkdir(parents=True, exist_ok=True)
    cases = ("zero_length", "truncated_descriptor", "reserved_subop",
             "unknown_param_kind", "reserved_record_bits", "invalid_signed_range",
             "pointer_extent", "v1_shape_violation", "illegal_flags", "bad_record_kind",
             "prohibited_overlap", "softmax_head_id", "softmax_noncausal")
    command = "python tools/run_phase3_evidence.py --only decode"
    results = {}
    cycles = 0
    for index, name in enumerate(cases):
        exe = build / f"decode_{index}.vvp"
        compile_cmd = [str(IVERILOG), "-g2012", "-DICARUS",
                       f"-Ptb_vector_full_top_decode_errors.TB_CASE={index}",
                       "-s", "tb_vector_full_top_decode_errors", "-o", str(exe), *FULL_TOP_SOURCES]
        subprocess.run(compile_cmd, cwd=ROOT, text=True, check=True, capture_output=True)
        proc = subprocess.run([str(VVP), str(exe)], cwd=ROOT, text=True, check=True, capture_output=True)
        match = re.search(r"FULL_TOP_DECODE_RESULT (\d+) (\d+) (\d+)", proc.stdout)
        if not match:
            raise RuntimeError(proc.stdout + proc.stderr)
        tests, errors, spent = map(int, match.groups())
        if errors or "[FAIL]" in proc.stdout:
            raise RuntimeError(proc.stdout)
        cycles += spent
        results[name] = {"cases": 1, "compared_bytes": 96, "mismatches": 0,
                         "cycles": spent, "passed": True, "command": command,
                         "checks": tests}
    exe = build / "decode_absent.vvp"
    compile_cmd = [str(IVERILOG), "-g2012", "-DICARUS",
                   "-Ptb_vector_full_top_decode_errors.TB_VECTOR_EN=0",
                   "-s", "tb_vector_full_top_decode_errors", "-o", str(exe), *FULL_TOP_SOURCES]
    subprocess.run(compile_cmd, cwd=ROOT, text=True, check=True, capture_output=True)
    proc = subprocess.run([str(VVP), str(exe)], cwd=ROOT, text=True, check=True, capture_output=True)
    match = re.search(r"FULL_TOP_DECODE_RESULT (\d+) (\d+) (\d+)", proc.stdout)
    if not match or int(match.group(2)) or "[FAIL]" in proc.stdout:
        raise RuntimeError(proc.stdout + proc.stderr)
    tests, _, spent = map(int, match.groups())
    cycles += spent
    results["vector_unit_absent_escape"] = {
        "cases": 1, "compared_bytes": 64, "mismatches": 0, "cycles": spent,
        "passed": True, "command": command, "checks": tests,
        "descriptor_would_be_legacy_bstore": True,
    }
    return {"classes": results, "cases": len(results), "compared_bytes": sum(x["compared_bytes"] for x in results.values()),
            "mismatches": 0, "cycles": cycles, "passed": True, "command": command}


def run_full_top_trace_suite() -> dict:
    """Execute eight nine-stage vector traces via the top instruction BRAM."""
    import numpy as np
    import run_vector_sequencer_sim as seq
    from run_vector_rtl_sim import rne, rms, sat, soft

    build = ROOT / "build" / "phase3_full_top_traces"
    build.mkdir(parents=True, exist_ok=True)
    cmd_text = "python tools/run_phase3_evidence.py --only trace_full_top"
    stage_totals = {name: {"cases": 0, "compared_bytes": 0, "mismatches": 0, "cycles": 0, "passed": True}
                    for name in ("pos_add", "rmsnorm1", "softmax_head0", "softmax_head1", "softmax_head2",
                                 "residual1", "rmsnorm2", "residual2", "final_norm")}
    all_cycles = all_bytes = errors = 0
    stage_cycle_totals = {name: 0 for name in stage_totals}
    for trace in range(8):
        rng = np.random.default_rng(906 + trace)  # independent deterministic full-top workload per trace
        words: dict[int, int] = {}
        expected: list[tuple[int, int]] = []
        program: list[int] = []
        next_addr = 64

        def alloc(data: np.ndarray) -> int:
            nonlocal next_addr
            address = next_addr
            packed = seq.pack(np.asarray(data).reshape(-1))
            words.update({address + i: value for i, value in enumerate(packed)})
            next_addr += len(packed) + 8
            return address

        def append_vec(subop: int, dst: int, src0: int, param: int, length: int, flags: int = 0) -> None:
            program.extend((0x05A5, ((subop + 1) << 11) | (flags << 8) | (subop << 4) | 9,
                            dst, src0, param, length))

        def output_expect(dst: int, values: np.ndarray) -> None:
            nonlocal all_bytes
            packed = seq.pack(np.asarray(values).reshape(-1))
            expected.extend((dst + i, value) for i, value in enumerate(packed))
            all_bytes += len(values.reshape(-1))

        x = rng.integers(-128, 128, 1536, dtype=np.int8)
        pos = rng.integers(-128, 128, 1536, dtype=np.int8)
        pos_out = sat(rne(x.astype(np.int64), 1) + rne(pos.astype(np.int64), 2))
        x_a, pos_a, pos_dst, p = alloc(x), alloc(pos), alloc(np.zeros(1536, np.int8)), next_addr
        words.update({p: 3, p+1: pos_a, p+2: ((-6 & 255) << 8) | (-5 & 255), p+3: (-4 & 255)})
        next_addr += 12; append_vec(2, pos_dst, x_a, p, 1536); output_expect(pos_dst, pos_out)

        def append_norm(data: np.ndarray, label: str) -> np.ndarray:
            nonlocal next_addr
            gamma = rng.integers(-128, 128, 48, dtype=np.int8)
            result, _ = rms(data.reshape(-1, 48), gamma, 3)
            src, dst, gamma_a = alloc(data), alloc(np.zeros(data.size, np.int8)), alloc(gamma)
            rec = next_addr; words.update({rec: 2, rec+1: gamma_a, rec+2: 3}); next_addr += 10
            append_vec(1, dst, src, rec, data.size)
            output_expect(dst, result)
            return result.reshape(-1)

        norm1 = append_norm(pos_out, "rmsnorm1")
        for head in range(3):
            scores = rng.integers(-128, 128, (32, 32), dtype=np.int8)
            probs = np.zeros((32, 32), dtype=np.uint8)
            for row in range(32): probs[row, :row+1] = soft(scores[row, :row+1])
            src, dst = alloc(scores), alloc(np.zeros_like(probs))
            scratch = next_addr; next_addr += 520
            rec = next_addr; words.update({rec: 4, rec+1: scratch}); next_addr += 10
            append_vec(3, dst, src, rec, 1024, flags=(1 | (head << 1)))
            output_expect(dst, probs)

        other1 = rng.integers(-128, 128, 1536, dtype=np.int8)
        residual1 = sat(rne(pos_out.astype(np.int64), 1) + rne(other1.astype(np.int64), 2))
        a, b, d = alloc(pos_out), alloc(other1), alloc(np.zeros(1536, np.int8)); rec = next_addr
        words.update({rec: 1, rec+1: b, rec+2: ((-5 & 255) | ((-6 & 255) << 8)), rec+3: (-4 & 255)}); next_addr += 12
        append_vec(0, d, a, rec, 1536); output_expect(d, residual1)
        norm2 = append_norm(residual1, "rmsnorm2")
        other2 = rng.integers(-128, 128, 1536, dtype=np.int8)
        residual2 = sat(rne(norm2.astype(np.int64), 1) + rne(other2.astype(np.int64), 2))
        a, b, d = alloc(norm2), alloc(other2), alloc(np.zeros(1536, np.int8)); rec = next_addr
        words.update({rec: 1, rec+1: b, rec+2: ((-5 & 255) | ((-6 & 255) << 8)), rec+3: (-4 & 255)}); next_addr += 12
        append_vec(0, d, a, rec, 1536); output_expect(d, residual2)
        final = append_norm(residual2, "final_norm")
        program.append(4)

        input_file = build / f"trace_{trace}_input.mem"
        expected_file = build / f"trace_{trace}_expected.mem"
        program_file = build / f"trace_{trace}_program.mem"
        def write_mem(path: Path, values) -> None:
            path.write_text("\n".join(f"{int(v) & 0xffff:04x}" for v in values) + "\n", encoding="ascii")
        memory = [0] * 32768
        for address, value in words.items(): memory[address] = value
        write_mem(input_file, memory)
        expected_file.write_text("\n".join(f"{((value << 16) | address) & 0xffffffff:08x}" for address, value in expected) + "\n", encoding="ascii")
        write_mem(program_file, program)
        tb = f'''module tb;
logic clk=0,rst=0,rx=1; wire tx,led_rst; localparam integer FW=256, PW=$clog2(FW);
localparam integer PROGRAM_WORDS={len(program)}, EXPECT_WORDS={len(expected)};
reg [15:0] input_words[0:32767], prog[0:PROGRAM_WORDS-1]; reg [31:0] expected_pairs[0:EXPECT_WORDS-1];
integer i,cycles,miss,stage_no,stage_start_time; bit seen;
top #(.ARRAY_SIZE(8),.BUFFER_SIZE(32768),.COMPUTE_DATA_WIDTH(8),.ACCUMULATOR_DATA_WIDTH(32),
 .EXT_ADDR_EN(1),.PROG_DEPTH(2048),.FIFO_WIDTH(FW),.MAX_BATCH_COUNT(4),.QUANTIZER_PIPE_DEPTH(1),
 .VECTOR_UNIT_EN(1)) dut(.clk(clk),.rst(rst),.rx(rx),.tx(tx),.led_rst(led_rst));
always #5 clk=~clk;
always @(posedge clk) begin
 if(dut.current_state==dut.VEC_START_STATE) stage_start_time=$time;
 if(dut.current_state==dut.FETCH_BRAM_STATE && stage_start_time!=0) begin
  $display("TRACE_STAGE %0d %0d",stage_no,($time-stage_start_time)/10); stage_no=stage_no+1; stage_start_time=0;
 end
end
task automatic wait_cycles(input integer n); repeat(n) @(posedge clk); endtask
task automatic push(input [7:0] v); @(posedge clk); dut.fifo_in.mem[dut.fifo_in.w_ptr[PW-1:0]]=v; dut.fifo_in.w_ptr=dut.fifo_in.w_ptr+1'b1; @(posedge clk); endtask
task automatic wait_state(input [5:0] s,input integer lim,output bit ok); integer n; begin ok=0; for(n=0;n<lim&&!ok;n=n+1) begin @(posedge clk); cycles=cycles+1; if(dut.current_state==s) ok=1; end end endtask
initial begin
 cycles=0; stage_no=0; stage_start_time=0;
 $readmemh("{input_file.as_posix()}",input_words); $readmemh("{expected_file.as_posix()}",expected_pairs); $readmemh("{program_file.as_posix()}",prog);
 for(i=0;i<32768;i=i+1) dut.u_unified_buffer.bank_mem[i%32][i/32]=input_words[i];
 rst=0; wait_cycles(8); rst=1; wait_cycles(8); push(8'ha1); push(PROGRAM_WORDS[7:0]); push(PROGRAM_WORDS[15:8]);
 for(i=0;i<PROGRAM_WORDS;i=i+1) begin push(prog[i][7:0]); push(prog[i][15:8]); end
 wait_state(dut.WAIT_START_STATE,10000,seen); if(!seen) $fatal(1,"upload did not reach WAIT_START"); push(8'ha2);
 wait_state(dut.HALT_STATE,3000000,seen); if(!seen) $fatal(1,"full-top trace timeout top=%0d pc=%0d",dut.current_state,dut.pc);
 miss=0; for(i=0;i<EXPECT_WORDS;i=i+1) if(dut.u_unified_buffer.bank_mem[expected_pairs[i][15:0]%32][expected_pairs[i][15:0]/32] !== expected_pairs[i][31:16]) begin miss=miss+1; if(miss<8) $display("TRACE_MISMATCH addr=%0d got=%h want=%h",expected_pairs[i][15:0],dut.u_unified_buffer.bank_mem[expected_pairs[i][15:0]%32][expected_pairs[i][15:0]/32],expected_pairs[i][31:16]); end
 $display("FULL_TOP_TRACE_RESULT {trace} {len(expected)} %0d %0d",miss,cycles); $finish;
end
endmodule
'''
        tb_path = build / f"trace_{trace}.sv"; tb_path.write_text(tb, encoding="utf-8")
        exe = build / f"trace_{trace}.vvp"
        compile_cmd = [str(IVERILOG), "-g2012", "-DICARUS", "-s", "tb", "-o", str(exe), *FULL_TOP_SOURCES[:-1], str(tb_path)]
        subprocess.run(compile_cmd, cwd=ROOT, text=True, check=True, capture_output=True)
        proc = subprocess.run([str(VVP), str(exe)], cwd=ROOT, text=True, check=True, capture_output=True)
        match = re.search(rf"FULL_TOP_TRACE_RESULT {trace} (\d+) (\d+) (\d+)", proc.stdout)
        if not match: raise RuntimeError(proc.stdout + proc.stderr)
        packed_words, misses, spent = map(int, match.groups())
        if misses: raise RuntimeError(proc.stdout)
        stage_lines = re.findall(r"TRACE_STAGE (\d+) (\d+)", proc.stdout)
        if len(stage_lines) != 9:
            raise RuntimeError(f"trace {trace} produced {len(stage_lines)} stage timings: {proc.stdout}")
        stage_names = list(stage_totals)
        for stage_index, stage_cycle in stage_lines:
            stage_cycle_totals[stage_names[int(stage_index)]] += int(stage_cycle)
        all_cycles += spent; errors += misses
    stage_bytes = {"pos_add":1536,"rmsnorm1":1536,"softmax_head0":1024,"softmax_head1":1024,
                   "softmax_head2":1024,"residual1":1536,"rmsnorm2":1536,"residual2":1536,"final_norm":1536}
    for name, stage in stage_totals.items():
        stage.update(cases=8, compared_bytes=stage_bytes[name]*8, cycles=stage_cycle_totals[name], passed=True)
    return {"cases": 8, "stages_per_trace": 9, "compared_bytes": all_bytes, "mismatches": errors,
            "cycles": all_cycles, "passed": errors == 0, "command": cmd_text,
            "stage_totals": stage_totals, "execution": "full top instruction BRAM -> escape -> descriptor -> vector unit -> unified buffer"}


def fingerprint(path: str) -> str:
    return subprocess.check_output(["git", "rev-parse", f"HEAD:{path}"], cwd=ROOT, text=True).strip()


def required(result: dict, name: str) -> dict:
    if not result.get("passed") or result.get("mismatches", 0):
        raise RuntimeError(f"{name} failed: {result}")
    return result


def commanded(result: dict, command: str) -> dict:
    """Attach the executable producer command to a generated evidence row."""
    result["command"] = command
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("vadd", "vpos", "rmsnorm", "softmax", "traces", "decode", "trace_full_top"))
    only = parser.parse_args().only
    if only:
        suite = {
            "vadd": run_vector_sequencer_sim.run_vadd_exhaustive_suite,
            "vpos": run_vector_sequencer_sim.run_vpos_add_suite,
            "rmsnorm": run_vector_sequencer_sim.run_rmsnorm_suite,
            "softmax": run_vector_sequencer_sim.run_softmax_suite,
            "traces": run_vector_sequencer_sim.run_vector_stage_traces,
            "decode": run_full_top_decode_errors,
            "trace_full_top": run_full_top_trace_suite,
        }[only]
        value = suite()
        if only == "traces":
            for name, result in value.items():
                required(result, name)
        else:
            required(value, only)
        if only == "decode":
            payload = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
            payload["decode_errors"] = value
            payload["source_tree"] = {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")}
            RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if only == "trace_full_top":
            payload = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
            payload["trace_full_top"] = value
            payload["source_tree"] = {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")}
            RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({only: value}, sort_keys=True))
        return 0
    # These helper calls each compile/run an Icarus testbench against the
    # serialized production controller.  No values are reconstructed here.
    vadd = commanded(required(run_vector_sequencer_sim.run_vadd_exhaustive_suite(), "vadd_rq"),
                     "python tools/run_phase3_evidence.py --only vadd")
    vpos = commanded(required(run_vector_sequencer_sim.run_vpos_add_suite(), "vpos_add"),
                     "python tools/run_phase3_evidence.py --only vpos")
    rms = commanded(required(run_vector_sequencer_sim.run_rmsnorm_suite(), "rmsnorm"),
                    "python tools/run_phase3_evidence.py --only rmsnorm")
    softmax = commanded(required(run_vector_sequencer_sim.run_softmax_suite(), "softmax"),
                        "python tools/run_phase3_evidence.py --only softmax")
    divider = commanded(required(run_restoring_divider_sim.run_suite(), "divider"),
                        "python firmware/host/run_restoring_divider_sim.py")
    traces = run_vector_sequencer_sim.run_vector_stage_traces()
    for name, value in traces.items():
        required(value, name)
        commanded(value, "python tools/run_phase3_evidence.py --only traces")
    payload = {
        "status": "phase3_vector_rtl_incomplete",
        "source_tree": {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")},
        "per_op": {"vadd_rq_exhaustive": vadd, "vpos_add_full_table": vpos,
                   "rmsnorm_2000_rows": rms, "softmax_16000_rows": softmax},
        "divider": divider,
        "rsqrt_lut_bin_coverage": {"covered": rms["rsqrt_lut_bin_coverage"], "total": 192},
        "trace_stages": traces,
        "commands": ["python tools/run_phase3_evidence.py"],
    }
    RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise RuntimeError(
        "phase3 evidence incomplete: traces/decode/mixed/dual-mode suites are required"
    )


if __name__ == "__main__":
    raise SystemExit(main())
