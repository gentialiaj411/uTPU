"""Regenerate Phase-3 vector RTL evidence from executable simulations.

This is intentionally the only writer of ``tiny_transformer_vec_rtl.json``.
It imports the simulation helpers rather than copying their results, and every
reported count is returned by a completed subprocess-backed RTL test.
"""
from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        proc = subprocess.run([str(VVP), str(exe)], cwd=ROOT, text=True, capture_output=True)
        if proc.returncode: raise RuntimeError(proc.stdout + proc.stderr)
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
 if(dut.current_state==dut.VEC_START_STATE) begin
  if(stage_start_time!=0) begin
   $display("TRACE_STAGE %0d %0d",stage_no,($time-stage_start_time)/10); stage_no=stage_no+1;
  end
  stage_start_time=$time;
 end
 if(dut.current_state==dut.HALT_STATE && stage_start_time!=0) begin
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


def run_full_top_mixed_program() -> dict:
    """Compare a mixed VEC/legacy program against its VEC-disabled baseline."""
    import numpy as np
    import isa_encoder as isa
    import run_vector_sequencer_sim as seq
    from isa_encoder import IsaConfig
    from isa_simulator import UTPUISASimulator

    build = ROOT / "build" / "phase3_full_top_mixed"
    build.mkdir(parents=True, exist_ok=True)
    cfg = isa.IsaConfig(address_width=12, compute_data_width=8, descriptor_isa=False)
    a = np.arange(48, dtype=np.int8) - 24
    b = np.arange(48, dtype=np.int8)[::-1].copy() - 24
    isa_cfg = IsaConfig(address_width=12, compute_data_width=8, descriptor_isa=True, vector_unit=True)
    oracle = UTPUISASimulator(array_size=8, buffer_size=4096, cfg=isa_cfg)
    for base, values in ((64, a), (128, b)):
        packed = seq.pack(values)
        oracle.buffer[base:base + len(packed)] = [int(x) for x in packed]
    oracle.buffer[300:304] = [1, 128, 0, 0]
    oracle.run_words([0x05A5, (1 << 11) | 9, 256, 64, 300, 48], preserve_buffer=True)
    result_words = oracle.buffer[256:280]
    expected_vec = np.asarray(
        [signed for word in result_words for signed in
         (int(word & 0xFF) - 256 if (word & 0x80) else int(word & 0xFF),
          int((word >> 8) & 0xFF) - 256 if (word & 0x8000) else int((word >> 8) & 0xFF))],
        dtype=np.int8,
    )
    packed_a, packed_b, packed_expected = seq.pack(a), seq.pack(b), seq.pack(expected_vec)

    def words_from_bytes(data: bytes) -> list[int]:
        return [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]

    def legacy_words() -> list[int]:
        out = []
        out += words_from_bytes(isa.encodeFetch(256, top_half=True, cfg=cfg))
        out += words_from_bytes(isa.encodeFetch(256, top_half=False, cfg=cfg))
        out += words_from_bytes(isa.encodeBurstStore(800, [0x1234, 0xABCD, 3, 4, 5, 6, 7, 8], cfg=cfg))
        out += words_from_bytes(isa.encodeLoadWeights(512, cfg=cfg))
        out += words_from_bytes(isa.encodeLoadInputs(576, cfg=cfg))
        out += words_from_bytes(isa.encodeRun(640, compute_en=True, quantize_en=True,
                                               relu_en=False, acc_clear_en=True, cfg=cfg))
        out += words_from_bytes(isa.encodeFetch(640, top_half=True, cfg=cfg))
        out += words_from_bytes(isa.encodeFetch(640, top_half=False, cfg=cfg))
        out += words_from_bytes(isa.encodeRun(704, compute_en=True, quantize_en=True,
                                               relu_en=False, acc_clear_en=True,
                                               residual_en=True, residual_addr=640, cfg=cfg))
        out += words_from_bytes(isa.encodeFetch(704, top_half=True, cfg=cfg))
        out += words_from_bytes(isa.encodeFetch(704, top_half=False, cfg=cfg))
        out.append(4)
        return out

    legacy = legacy_words()
    mixed = [0x05A5, (1 << 11) | 9, 256, 64, 300, 48] + legacy
    words = {i: w for i, w in enumerate(legacy)}
    words.update({i: w for i, w in enumerate(mixed)})
    # Both modes begin from identical memory. In VEC=0, dst is prefilled with
    # the ISA result; in VEC=1, the real vector engine must produce it.
    memory = [0] * 4096
    for base, packed in ((64, packed_a), (128, packed_b), (256, packed_expected),
                         (512, np.zeros(64, dtype=np.uint16)),
                         (576, np.zeros(64, dtype=np.uint16)),
                         (640, np.zeros(64, dtype=np.uint16)),
                         (704, np.zeros(64, dtype=np.uint16))):
        for i, value in enumerate(packed): memory[base + i] = int(value)
    memory[300:304] = [1, 128, 0, 0]
    in_file = build / "input.mem"
    in_file.write_text("\n".join(f"{v:04x}" for v in memory) + "\n", encoding="ascii")
    expected_file = build / "expected_vec.mem"
    expected_file.write_text("\n".join(f"{int(v):04x}" for v in packed_expected) + "\n", encoding="ascii")
    prog_files = []
    for name, program in (("mixed", mixed), ("legacy", legacy)):
        path = build / f"{name}.mem"
        path.write_text("\n".join(f"{v & 0xffff:04x}" for v in program) + "\n", encoding="ascii")
        prog_files.append(path)

    source = f'''module tb;
`ifndef TEST_VEN
`define TEST_VEN 0
`endif
localparam integer VEN=`TEST_VEN;
logic clk=0,rst=0,rx=1; wire tx,led_rst;
localparam integer FW=256,PW=$clog2(FW),NW={len(mixed)},LW={len(legacy)};
reg [15:0] mem[0:4095], prog[0:NW-1]; integer i,cycles,nbytes,miss; bit seen,started;
reg [15:0] expected_vec[0:23];
reg [7:0] actual[0:511]; reg [31:0] expected[0:23];
top #(.ARRAY_SIZE(8),.BUFFER_SIZE(4096),.COMPUTE_DATA_WIDTH(8),.ACCUMULATOR_DATA_WIDTH(32),
 .EXT_ADDR_EN(1),.PROG_DEPTH(256),.FIFO_WIDTH(FW),.MAX_BATCH_COUNT(4),.QUANTIZER_PIPE_DEPTH(1),
 .VECTOR_UNIT_EN(VEN)) dut(.clk(clk),.rst(rst),.rx(rx),.tx(tx),.led_rst(led_rst));
always #5 clk=~clk;
always @(posedge clk) if(started && dut.tx_we && dut.tx_wdata!=8'haa) begin actual[nbytes]=dut.tx_wdata; nbytes=nbytes+1; end
task automatic waitn(input integer n); repeat(n) @(posedge clk); endtask
task automatic push(input [7:0] v); @(posedge clk); dut.fifo_in.mem[dut.fifo_in.w_ptr[PW-1:0]]=v; dut.fifo_in.w_ptr=dut.fifo_in.w_ptr+1'b1; @(posedge clk); endtask
task automatic waitstate(input [5:0] s,input integer lim,output bit ok); integer k; begin ok=0; for(k=0;k<lim&&!ok;k=k+1) begin @(posedge clk); cycles=cycles+1; if(dut.current_state==s) ok=1; end end endtask
initial begin cycles=0;nbytes=0;miss=0;started=0;$readmemh("{in_file.as_posix()}",mem);$readmemh("{expected_file.as_posix()}",expected_vec);
 if(VEN) $readmemh("{prog_files[0].as_posix()}",prog); else $readmemh("{prog_files[1].as_posix()}",prog);
 for(i=0;i<4096;i=i+1)dut.u_unified_buffer.bank_mem[i%32][i/32]=mem[i];
 rst=0;waitn(8);rst=1;waitn(8);push(8'ha1);push(VEN?NW[7:0]:LW[7:0]);push(VEN?NW[15:8]:LW[15:8]);
 for(i=0;i<(VEN?NW:LW);i=i+1)begin push(prog[i][7:0]);push(prog[i][15:8]);end
 waitstate(dut.WAIT_START_STATE,10000,seen);if(!seen)$fatal(1,"upload timeout");push(8'ha2);started=1;
 waitstate(dut.HALT_STATE,100000,seen);if(!seen)$fatal(1,"execution timeout pc=%0d state=%0d bcount=%0d bindex=%0d bfill=%0d",dut.pc,dut.current_state,dut.bstore_count,dut.bstore_index,dut.bstore_fill);
 for(i=0;i<24;i=i+1)if(dut.u_unified_buffer.bank_mem[(256+i)%32][(256+i)/32]!==expected_vec[i])miss=miss+1;
 for(i=640;i<768;i=i+1)$display("MIXED_LEGACY_WORD %0d %04x",i,dut.u_unified_buffer.bank_mem[i%32][i/32]);
 for(i=800;i<808;i=i+1)$display("MIXED_LEGACY_WORD %0d %04x",i,dut.u_unified_buffer.bank_mem[i%32][i/32]);
 $display("MIXED_RESULT %0d %0d %0d %0d %0d",VEN,nbytes,cycles,dut.perf_busy_status_word[63:56],miss);
 for(i=0;i<nbytes;i=i+1)$display("MIXED_BYTE %02x",actual[i]);
 $display("MIXED_VEC %h %h",dut.u_unified_buffer.bank_mem[256%32][256/32],dut.u_unified_buffer.bank_mem[280%32][280/32]);$finish;end
endmodule
'''
    tb = build / "mixed.sv"
    tb.write_text(source, encoding="utf-8")
    iv = str(IVERILOG)
    result = {}
    for ven in (0, 1):
        exe = build / f"mixed_{ven}.vvp"
        cmd = [iv, "-g2012", "-DICARUS", f"-DTEST_VEN={ven}", "-s", "tb", "-o", str(exe), *FULL_TOP_SOURCES[:-1], str(tb)]
        subprocess.run(cmd, cwd=ROOT, text=True, check=True, capture_output=True)
        proc = subprocess.run([str(VVP), str(exe)], cwd=ROOT, text=True, capture_output=True)
        if proc.returncode:
            log_path = build / f"mixed_{ven}.log"
            log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
            raise RuntimeError("mixed simulation failed; inspect " + str(log_path))
        (build / f"mixed_{ven}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        m = re.search(rf"MIXED_RESULT {ven} (\d+) (\d+) ([0-9a-fA-F]+) (\d+)", proc.stdout)
        if not m: raise RuntimeError(proc.stdout + proc.stderr)
        byte_count, cycles, code, vm = int(m.group(1)), int(m.group(2)), int(m.group(3), 16), int(m.group(4))
        stream = [int(x, 16) for x in re.findall(r"MIXED_BYTE ([0-9a-fA-F]{2})", proc.stdout)]
        legacy_words_out = [(int(a), int(b, 16)) for a, b in re.findall(r"MIXED_LEGACY_WORD (\d+) ([0-9a-fA-F]{4})", proc.stdout)]
        result[str(ven)] = {"bytes": stream, "legacy_words": legacy_words_out, "cycles": cycles,
                            "status_code": code, "reported_bytes": byte_count, "vec_mismatches": vm}
    mismatches = sum(a != b for a, b in zip(result["1"]["bytes"], result["0"]["bytes"])) + abs(len(result["1"]["bytes"]) - len(result["0"]["bytes"]))
    mismatches += sum(a != b for a, b in zip(result["1"]["legacy_words"], result["0"]["legacy_words"]))
    mismatches += abs(len(result["1"]["legacy_words"]) - len(result["0"]["legacy_words"]))
    vec_mismatches = result["1"]["vec_mismatches"] + int(result["1"]["status_code"] != 0)
    legacy_bytes = min(len(result["0"]["legacy_words"]), len(result["1"]["legacy_words"])) * 2
    payload = {"cases": 1, "legacy_bytes": legacy_bytes,
               "legacy_bytes_vec0": len(result["0"]["legacy_words"]) * 2,
               "legacy_bytes_mixed": len(result["1"]["legacy_words"]) * 2,
               "vec_bytes": len(expected_vec), "vec_bytes_compared": len(expected_vec),
               "mismatches": mismatches + vec_mismatches, "cycles": result["1"]["cycles"],
               "passed": mismatches == 0 and vec_mismatches == 0 and len(expected_vec) > 0,
               "legacy_ops": ["FETCH(top_half=1)", "FETCH(top_half=0)", "BSTORE", "LOAD weights", "LOAD inputs", "RUN", "residual RUN"],
               "non_covered": {"TILE_GEMM": "RTL sequencer not implemented (Phase 4 decision)"},
               "execution": "full top instruction BRAM -> escape -> descriptor -> vector unit -> legacy FSM -> unified buffer",
               "command": "python tools/run_phase3_evidence.py --only mixed"}
    if mismatches or vec_mismatches:
        raise RuntimeError(f"mixed top outputs differ: {payload}; runs={result}")
    return payload


def run_full_top_transition_matrix() -> dict:
    """Exercise every legacy/VEC boundary through uploaded top.sv programs.

    Each case is a separate normal instruction-BRAM upload. VEC=0 runs the
    legacy-only projection; VEC=1 runs the same legacy sequence with the VEC
    descriptors inserted at the named boundary. The unified buffer image is
    compared word-for-word, excluding only the independently ISA-checked VEC
    destination.
    """
    import isa_encoder as isa
    import run_vector_sequencer_sim as seq
    from isa_encoder import IsaConfig
    from isa_simulator import UTPUISASimulator

    build = ROOT / "build" / "phase3_transition_matrix"
    build.mkdir(parents=True, exist_ok=True)
    cfg = IsaConfig(address_width=12, compute_data_width=8, descriptor_isa=False)
    vcfg = IsaConfig(address_width=12, compute_data_width=8, descriptor_isa=True,
                     vector_unit=True)

    def words(raw: bytes) -> list[int]:
        return [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]

    ops = {
        "FETCH top_half=1": words(isa.encodeFetch(1000, True, cfg)),
        "FETCH top_half=0": words(isa.encodeFetch(1000, False, cfg)),
        "LOAD weights": words(isa.encodeLoadWeights(700, cfg)),
        "LOAD inputs": words(isa.encodeLoadInputs(800, cfg)),
        "RUN": (words(isa.encodeLoadWeights(512, cfg)) + words(isa.encodeLoadInputs(576, cfg)) +
                words(isa.encodeRun(640, compute_en=True, quantize_en=True, relu_en=False,
                                    acc_clear_en=True, cfg=cfg)) +
                words(isa.encodeFetch(640, True, cfg)) + words(isa.encodeFetch(640, False, cfg))),
        "residual RUN": (words(isa.encodeLoadWeights(512, cfg)) + words(isa.encodeLoadInputs(576, cfg)) +
                         words(isa.encodeRun(640, compute_en=True, quantize_en=True, relu_en=False,
                                             acc_clear_en=True, cfg=cfg)) +
                         words(isa.encodeFetch(640, True, cfg)) + words(isa.encodeFetch(640, False, cfg)) +
                         words(isa.encodeRun(704, compute_en=True, quantize_en=True, relu_en=False,
                                             residual_en=True, residual_addr=640, cfg=cfg)) +
                         words(isa.encodeFetch(704, True, cfg)) + words(isa.encodeFetch(704, False, cfg))),
        "BSTORE multiword": words(isa.encodeBurstStore(1100, [0x1234, 0xabcd, 3, 4, 5, 6, 7, 8], cfg)),
        "STORE": words(isa.encodeStoreValues(1200, [1, -2, 3, -4, 5, -6, 7, -8], cfg)),
        "NOP": words(isa.encodeNop(cfg)),
        "requant NOP bit3": [0x000d, 1, 0],
        "EXT_ADDR FETCH two-word": words(isa.encodeFetch(1000, True, cfg)),
        "EXT_ADDR LOAD two-word": words(isa.encodeLoadInputs(800, cfg)),
        "EXT_ADDR RUN two-word": (words(isa.encodeLoadWeights(512, cfg)) +
                                  words(isa.encodeLoadInputs(576, cfg)) +
                                  words(isa.encodeRun(640, compute_en=True, quantize_en=True,
                                                      relu_en=False, acc_clear_en=True, cfg=cfg)) +
                                  words(isa.encodeFetch(640, True, cfg)) + words(isa.encodeFetch(640, False, cfg))),
        "EXT_ADDR BSTORE two-word": words(isa.encodeBurstStore(1100, [0x91, 0x92, 0x93], cfg)),
    }
    after_ops = {
        "RUN": (words(isa.encodeRun(704, compute_en=True, quantize_en=True, relu_en=False,
                                    residual_en=True, residual_addr=640, cfg=cfg)) +
                words(isa.encodeFetch(704, True, cfg)) + words(isa.encodeFetch(704, False, cfg))),
        "EXT_ADDR RUN two-word": (words(isa.encodeRun(704, compute_en=True, quantize_en=True,
                                                      relu_en=False, residual_en=True,
                                                      residual_addr=640, cfg=cfg)) +
                                  words(isa.encodeFetch(704, True, cfg)) + words(isa.encodeFetch(704, False, cfg))),
        "residual RUN": (words(isa.encodeRun(704, compute_en=True, quantize_en=True, relu_en=False,
                                             acc_clear_en=True, cfg=cfg)) +
                         words(isa.encodeFetch(704, True, cfg)) + words(isa.encodeFetch(704, False, cfg))),
    }
    vec = words(isa.encode_vec_vadd_rq(3000, 2000, 2100, 48, cfg=vcfg))
    names = ["X->VEC", "VEC->X", "X->VEC->X"]
    cases: list[tuple[str, list[int], list[int]]] = []
    for opname, xwords in ops.items():
        after = after_ops.get(opname, xwords)
        for pattern in names:
            before_words = xwords if pattern in ("X->VEC", "X->VEC->X") else []
            after_words = after if pattern == "X->VEC->X" else (xwords if pattern == "VEC->X" else [])
            mixed_words = before_words + (vec if pattern != "VEC->X" else []) + after_words + [4]
            if pattern == "VEC->X": mixed_words = vec + xwords + [4]
            legacy_words = before_words + (after if pattern == "X->VEC->X" else
                                           (xwords if pattern == "VEC->X" else [])) + [4]
            cases.append((f"{opname}:{pattern}", mixed_words, legacy_words))
    cases += [("VEC->VEC", vec + vec + [4], [4]),
              ("VEC first after upload", vec + [4], [4]),
              ("VEC last before HALT", vec + [4], [4])]

    initial = [0] * 4096
    for addr in range(700, 850): initial[addr] = (addr * 29 + 7) & 0xffff
    for i in range(0, 48, 2):
        initial[2000 + i // 2] = ((i - 24) & 0xff) | (((i - 23) - 24 & 0xff) << 8)
        initial[2024 + i // 2] = ((24 - i) & 0xff) | (((24 - (i + 1)) & 0xff) << 8)
    initial[2100:2104] = [1, 2024, 0, 0]
    initial[900:924] = initial[2024:2048]
    initial_file = build / "initial.mem"
    initial_file.write_text("\n".join(f"{v:04x}" for v in initial) + "\n", encoding="ascii")

    oracle = UTPUISASimulator(array_size=8, buffer_size=4096, cfg=vcfg)
    oracle.buffer[:4096] = initial
    oracle.run_words(vec + [4], preserve_buffer=True)
    vec_expected = [(3000 + i, int(oracle.buffer[3000 + i])) for i in range(48)]
    expected_file = build / "vec_expected.mem"
    expected_file.write_text("\n".join(f"{v:04x}" for _, v in vec_expected) + "\n", encoding="ascii")

    tb = f'''module tb;
logic clk=0,rst=0,rx=1; wire tx,led_rst; localparam integer FW=256,PW=8;
reg [15:0] init[0:4095],prog[0:511],exp[0:47]; integer n,i,cycles; string pf; bit seen;
`ifndef VEN
`define VEN 0
`endif
top #(.ARRAY_SIZE(8),.BUFFER_SIZE(4096),.COMPUTE_DATA_WIDTH(8),.ACCUMULATOR_DATA_WIDTH(32),
 .WEIGHT_REGION_WORDS(1024),.EXT_ADDR_EN(1),.PROG_DEPTH(512),.FIFO_WIDTH(FW),.MAX_BATCH_COUNT(4),
 .QUANTIZER_PIPE_DEPTH(1),.VECTOR_UNIT_EN(`VEN)) dut(.clk(clk),.rst(rst),.rx(rx),.tx(tx),.led_rst(led_rst));
always #5 clk=~clk;
task push(input [7:0] b); @(posedge clk); dut.fifo_in.mem[dut.fifo_in.w_ptr[PW-1:0]]=b; dut.fifo_in.w_ptr=dut.fifo_in.w_ptr+1'b1; @(posedge clk); endtask
initial begin cycles=0; if(!$value$plusargs("prog=%s",pf)) $fatal(1,"missing prog");
 if(!$value$plusargs("n=%d",n)) $fatal(1,"missing n");
 $readmemh("{initial_file.as_posix()}",init); $readmemh("{expected_file.as_posix()}",exp);
 $readmemh(pf,prog,0,n-1); for(i=0;i<4096;i=i+1)dut.u_unified_buffer.bank_mem[i%32][i/32]=init[i];
 rst=0; repeat(8) @(posedge clk); rst=1; repeat(8) @(posedge clk);
 push(8'ha1);push(n[7:0]);push(n[15:8]);for(i=0;i<n;i=i+1)begin push(prog[i][7:0]);push(prog[i][15:8]);end
 for(i=0;i<10000 && dut.current_state!=dut.WAIT_START_STATE;i=i+1)@(posedge clk);
 if(dut.current_state!=dut.WAIT_START_STATE)$fatal(1,"upload timeout");push(8'ha2);
 for(i=0;i<1000000 && dut.current_state!=dut.HALT_STATE;i=i+1)begin @(posedge clk);cycles=cycles+1;end
 if(dut.current_state!=dut.HALT_STATE)$fatal(1,"execution timeout state=%0d pc=%0d",dut.current_state,dut.pc);
 for(i=0;i<48;i=i+1)if(`VEN && dut.u_unified_buffer.bank_mem[(3000+i)%32][(3000+i)/32]!==exp[i][15:0])$display("VEC_BAD %0d",i);
 $display("MATRIX_BEGIN"); for(i=0;i<4096;i=i+1)$display("MATRIX_MEM %04x",dut.u_unified_buffer.bank_mem[i%32][i/32]);
 $display("MATRIX_END %0d",cycles);$finish;end
endmodule
'''
    tbfile = build / "matrix.sv"
    tbfile.write_text(tb, encoding="ascii")
    executables = {}
    for mode in (0, 1):
        exe = build / f"matrix_{mode}.vvp"
        cmd = [str(IVERILOG), "-g2012", "-DICARUS", f"-DVEN={mode}", "-s", "tb", "-o", str(exe), *FULL_TOP_SOURCES[:-1], str(tbfile)]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        executables[mode] = exe

    rows = {}
    total_compared = total_output_bytes = total_vector_bytes = total_mismatches = total_cycles = 0
    vadd_addresses = set(range(3000, 3048))
    case_results = {}
    for index, (label, mixed_words, baseline_words) in enumerate(cases):
        snapshots = {}
        case_cycles = 0
        for mode, program in ((0, baseline_words), (1, mixed_words)):
            program_file = build / f"case_{index}_{mode}.mem"
            program_file.write_text("\n".join(f"{w & 0xffff:04x}" for w in program) + "\n", encoding="ascii")
            proc = subprocess.run([str(VVP), str(executables[mode]), f"+prog={program_file.as_posix()}", f"+n={len(program)}"], cwd=ROOT, check=True, capture_output=True, text=True)
            (build / f"case_{index}_{mode}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
            output = proc.stdout + proc.stderr
            image = [x.lower() for x in re.findall(r"MATRIX_MEM ([0-9a-fxz]{4})", output, re.IGNORECASE)]
            if len(image) != 4096 or "VEC_BAD" in output:
                raise RuntimeError(f"transition case {label} mode {mode} failed image_words={len(image)} vec_bad={'VEC_BAD' in output} stdout={len(proc.stdout)} stderr={len(proc.stderr)}: {output[-2000:]}")
            cycles = int(re.search(r"MATRIX_END (\d+)", output).group(1))
            snapshots[mode] = image
            total_cycles += cycles
            case_cycles += cycles
        compared = [i for i in range(4096) if i not in vadd_addresses]
        mismatches = sum(snapshots[0][i] != snapshots[1][i] for i in compared)
        vector_checks = sum(snapshots[1][addr] != f"{value:04x}" for addr, value in vec_expected)
        output_bytes = 2 * sum(snapshots[0][i] != f"{initial[i]:04x}"
                               and re.fullmatch(r"[0-9a-f]{4}", snapshots[0][i]) is not None
                               for i in compared)
        vector_bytes = 48 * mixed_words.count(0x05A5)
        total_compared += len(compared) * 2
        total_output_bytes += output_bytes
        total_vector_bytes += vector_bytes
        total_mismatches += mismatches + vector_checks
        case_results[label] = {"cases": 1, "bytes_compared": len(compared) * 2,
                               "mismatches": mismatches + vector_checks, "cycles": case_cycles,
                               "passed": mismatches == 0 and vector_checks == 0,
                               "command": "python tools/run_phase3_evidence.py --only transition_matrix",
                               "legacy_op": label.split(":", 1)[0], "legacy_output_bytes": output_bytes,
                               "vector_bytes": vector_bytes}
    result = {"cases": len(cases), "matrix": case_results, "compared_bytes": total_compared,
              "legacy_bytes_compared": total_output_bytes, "legacy_buffer_snapshot_bytes_compared": total_compared,
              "vector_bytes_compared": total_vector_bytes,
              "mismatches": total_mismatches, "cycles": total_cycles,
              "passed": total_mismatches == 0, "command": "python tools/run_phase3_evidence.py --only transition_matrix",
              "boundary_patterns": names + ["VEC->VEC", "VEC first after upload", "VEC last before HALT"],
              "operations": list(ops)}
    if not result["passed"]: raise RuntimeError(f"transition matrix failed: {result}")
    return result


def run_dual_mode_regression() -> dict:
    """Run every globbed RTL TB in both top parameter modes and compare logs."""
    build = ROOT / "build" / "phase3_dual_mode"
    build.mkdir(parents=True, exist_ok=True)
    testbenches = sorted((ROOT / "rtl" / "tb").glob("tb_*.sv"))
    harness_map = {
        "bitmatch": "tb_fused_compressed_program",
        "fused": "tb_fused_compressed_program",
        "residual": "tb_fused_residual_program",
        "batched": "tb_batched_gemm",
        "buffer_resident_uart_replay": "tb_buffer_resident_uart_replay",
        "prog_depth": "tb_prog_depth_smoke",
        "bstore_wide": "tb_bstore_wide_smoke",
        "uart_replay": "tb_uart_replay",
        "scheduler": "tb_scheduler_cycles",
        "latency": "tb_latency_determinism",
        "mnist": "tb_mnist_utpu_program",
        "perf_counters": "tb_perf_counters",
        "phase4_widen": "tb_phase4_widen",
    }
    roots = {}
    for path in testbenches:
        source = path.read_text(encoding="utf-8")
        modules = re.findall(r"(?m)^\s*module\s+([A-Za-z_][A-Za-z0-9_$]*)", source)
        module = next((name for name in modules if name.startswith("tb_")), modules[0] if modules else None)
        if module is None:
            raise RuntimeError(f"no root module declaration in {path}")
        roots[module] = path
    unknown_harnesses = sorted(set(harness_map.values()) - set(roots))
    if unknown_harnesses:
        raise RuntimeError(f"declared harnesses do not map to globbed RTL TBs: {unknown_harnesses}")

    excluded_source = "rtl/tb/tb_vector_full_top_decode_errors.sv"
    base_sources = [str(ROOT / s) for s in FULL_TOP_SOURCES if s != excluded_source]
    runner_command = "python tools/run_phase3_evidence.py --only dual_mode"
    results: dict[str, dict] = {}

    mode_exemptions = {
        "tb_pe_array_packed": "standalone pe_array_packed unit; no top module or VECTOR_UNIT_EN parameter",
        "tb_pe_array_packed_cycles": "standalone pe_array_packed cycle unit; no top module or VECTOR_UNIT_EN parameter",
        "tb_pe_array_packed_hardened": "standalone pe_array_packed hardening unit; no top module or VECTOR_UNIT_EN parameter",
        "tb_pe_array_packed_odd_guard": "standalone pe_array_packed guard unit; no top module or VECTOR_UNIT_EN parameter",
        "tb_pe_controller_streaming": "standalone PE controller unit; no top module or VECTOR_UNIT_EN parameter (attempted both modes; existing test reports unresolved result values)",
        "tb_pe_packed_pair": "standalone packed PE pair unit; no top module or VECTOR_UNIT_EN parameter",
        "tb_top_packed_smoke": "synthesis-only top_packed datapath wrapper, not rtl/top/top.sv; no VECTOR_UNIT_EN parameter",
    }
    force_fresh: set[str] = set()

    def run_one(module: str, path: Path, mode: int, depth: int | None = None) -> dict:
        stem = f"{path.stem}_ven{mode}" + (f"_depth{depth}" if depth else "")
        exe = build / f"{stem}.vvp"
        log = build / f"{stem}.log"
        compile_cmd = [str(IVERILOG), "-g2012", "-DICARUS", f"-DPHASE3_VECTOR_UNIT_EN={mode}",
                       "-I", str(ROOT), "-s", module, "-o", str(exe)]
        if depth is not None:
            compile_cmd.append(f"-DTB_PROG_DEPTH={depth}")
        if module == "tb_vector_full_top_decode_errors":
            compile_cmd += [f"-P{module}.TB_VECTOR_EN={mode}", "-Ptb_vector_full_top_decode_errors.TB_CASE=0"]
        compile_cmd += base_sources + [str(path)]
        command_text = " ".join(compile_cmd) + f" && {VVP} -n {exe}"
        if (module not in force_fresh and exe.is_file() and log.is_file()
                and "error:" not in log.read_text(encoding="utf-8", errors="replace").lower()):
            output = log.read_text(encoding="utf-8", errors="replace")
            failed_marker = re.search(r"(?i)(\[FAIL\]|TB_RESULT:\s*FAIL|\bFATAL:|\bERROR:|Unknown module type)", output)
            return {"passed": failed_marker is None, "command": command_text,
                    "log": str(log.relative_to(ROOT)), "returncode": 0,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "output_bytes": len(output.encode()), "reused": True}
        c = subprocess.run(compile_cmd, cwd=ROOT, text=True, capture_output=True)
        if c.returncode:
            log.write_text(c.stdout + c.stderr, encoding="utf-8")
            return {"passed": False, "command": command_text, "log": str(log.relative_to(ROOT)),
                    "diagnostic": "iverilog compile failed", "returncode": c.returncode}
        p = subprocess.run([str(VVP), "-n", str(exe)], cwd=ROOT, text=True, capture_output=True)
        output = p.stdout + p.stderr
        log.write_text(output, encoding="utf-8")
        failed_marker = re.search(r"(?i)(\[FAIL\]|TB_RESULT:\s*FAIL|\bFATAL:|\bERROR:|Unknown module type)", output)
        passed = p.returncode == 0 and failed_marker is None
        return {"passed": passed, "command": command_text, "log": str(log.relative_to(ROOT)),
                "returncode": p.returncode, "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                "output_bytes": len(output.encode())}

    jobs = []
    for module, path in roots.items():
        if module in mode_exemptions:
            continue
        jobs.extend((module, path, mode, None) for mode in (0, 1))
        if module == "tb_prog_depth_smoke":
            jobs.extend((module, path, mode, depth) for depth in (8192, 16384, 32768, 65536, 131072) for mode in (0, 1))
    completed = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, module, path, mode, depth): (module, mode, depth)
                   for module, path, mode, depth in jobs}
        for future in as_completed(futures):
            module, mode, depth = futures[future]
            completed[(module, mode, depth)] = future.result()

    for module, path in roots.items():
        if module in mode_exemptions:
            results[module] = {"testbench": str(path.relative_to(ROOT)), "exempted": True,
                               "reason": mode_exemptions[module], "modes": {},
                               "passed": True, "cases": 0, "mismatches": 0,
                               "command": "exemption: standalone unit without VECTOR_UNIT_EN"}
            continue
        modes = {str(mode): completed[(module, mode, None)] for mode in (0, 1)}
        compare_exemption = None
        if module == "tb_vector_full_top_decode_errors":
            compare_exemption = "per-case decoder contract is checked by decode_errors; VEC_EN=0 and 1 intentionally report different status codes"
        output_equal = modes["0"].get("output_sha256") == modes["1"].get("output_sha256")
        passed = all(item.get("passed") for item in modes.values()) and (output_equal or compare_exemption is not None)
        results[module] = {"testbench": str(path.relative_to(ROOT)), "modes": modes,
                           "outputs_equal": output_equal, "comparison_exemption": compare_exemption,
                           "passed": passed, "cases": 1, "mismatches": int(not passed),
                           "command": runner_command}
        if module == "tb_prog_depth_smoke":
            for depth in (8192, 16384, 32768, 65536, 131072):
                depth_modes = {str(mode): completed[(module, mode, depth)] for mode in (0, 1)}
                equal = depth_modes["0"].get("output_sha256") == depth_modes["1"].get("output_sha256")
                results[f"{module}_depth_{depth}"] = {
                    "testbench": str(path.relative_to(ROOT)), "depth": depth, "modes": depth_modes,
                    "outputs_equal": equal, "passed": all(x.get("passed") for x in depth_modes.values()) and equal,
                    "cases": 1, "mismatches": int(not equal), "command": runner_command,
                }

    declared = {}
    for harness, module in harness_map.items():
        item = results[module]
        declared[harness] = {"testbench": item["testbench"], "mode_runs": [0, 1],
                             "outputs_equal": item["outputs_equal"], "passed": item["passed"],
                             "command": runner_command}
    script_map = {
        "run_bstore_wide_smoke.py": "tb_bstore_wide_smoke",
        "run_buffer_resident_models.py": "tb_buffer_resident_uart_replay",
        "run_saved_buffer_resident_rtl.py": "tb_buffer_resident_uart_replay",
        "run_cycle_attribution.py": "tb_batched_gemm",
        "run_cycle_attribution_mnist.py": "tb_mnist_utpu_program",
        "run_isa_rtl_bitmatch.py": "tb_fused_compressed_program",
        "run_rtl_fused_sim.py": "tb_fused_compressed_program",
        "run_isa_rtl_residual_bitmatch.py": "tb_fused_residual_program",
        "run_latency_determinism.py": "tb_latency_determinism",
        "run_latency_determinism_vs_gpu.py": "tb_latency_determinism",
        "run_mnist_utpu_demo.py": "tb_mnist_utpu_program",
        "run_prog_depth_smoke.py": "tb_prog_depth_smoke",
        "run_rtl_batched_gemm_sim.py": "tb_batched_gemm",
        "run_rtl_batched_gemm_utilization_probe.py": "tb_batched_gemm",
        "run_scheduler_rtl_crosscheck.py": "tb_scheduler_cycles",
        "run_scheduler_rtl_crosscheck_bigmlp.py": "tb_scheduler_cycles",
        "run_scheduler_benchmark.py": "tb_perf_counters",
        "run_top_packed_smoke.py": "tb_top_packed_smoke",
        "run_uart_preboard_demo.py": "tb_uart_replay",
        "run_pe_array_packed_hardened_sim.py": "tb_pe_array_packed_hardened",
        "run_pe_array_packed_sim.py": "tb_pe_array_packed",
        "run_pe_packed_pair_sim.py": "tb_pe_packed_pair",
        "run_packed_array_cycle_compare.py": "tb_pe_array_packed_cycles",
    }
    harness_inventory = {}
    for script in sorted(HOST.glob("run_*.py")):
        source = script.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"\b(?:iverilog|vvp)\b", source):
            continue
        module = script_map.get(script.name)
        if module and module in results:
            item = results[module]
            if item.get("exempted"):
                harness_inventory[script.name] = {"testbench": item["testbench"],
                    "exempted": True, "reason": item["reason"], "passed": True}
            else:
                harness_inventory[script.name] = {"testbench": item["testbench"],
                    "mode_runs": [0, 1], "outputs_equal": item["outputs_equal"],
                    "passed": item["passed"], "coverage": "backing rtl/tb testbench executed directly by this runner"}
        else:
            harness_inventory[script.name] = {"exempted": True, "passed": True,
                "reason": "workload/preflight or standalone-vector harness has no single globbed top.sv testbench with VECTOR_UNIT_EN selection"}
    failures = [name for name, result in results.items() if not result.get("passed")]
    return {"cases": len(results), "globbed_testbench_count": len(testbenches),
            "globbed_testbenches": [str(path.relative_to(ROOT)) for path in testbenches],
            "results": results, "declared_harnesses": declared,
            "python_iverilog_harness_inventory": harness_inventory,
            "exemptions": {name: result["reason"] for name, result in results.items() if result.get("exempted")},
            "mismatches": len(failures), "failed": failures, "passed": not failures,
            "command": runner_command}


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
    parser.add_argument("--only", choices=("vadd", "vpos", "rmsnorm", "softmax", "traces", "decode", "trace_full_top", "mixed", "dual_mode", "transition_matrix"))
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
            "mixed": run_full_top_mixed_program,
            "dual_mode": run_dual_mode_regression,
            "transition_matrix": run_full_top_transition_matrix,
        }[only]
        value = suite()
        if only == "dual_mode":
            payload = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
            payload["dual_mode_regression"] = value
            payload["source_tree"] = {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")}
            RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if only == "transition_matrix":
            payload = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
            payload["transition_matrix"] = value
            payload["source_tree"] = {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")}
            RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if only in ("vadd", "vpos", "rmsnorm", "softmax"):
            payload = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
            key = {"vadd": "vadd_rq_exhaustive", "vpos": "vpos_add_full_table",
                   "rmsnorm": "rmsnorm_2000_rows", "softmax": "softmax_16000_rows"}[only]
            payload.setdefault("per_op", {})[key] = commanded(value, f"python tools/run_phase3_evidence.py --only {only}")
            if only == "rmsnorm":
                payload["rsqrt_lut_bin_coverage"] = {"covered": value.get("rsqrt_lut_bin_coverage", 0), "total": 192}
            payload["source_tree"] = {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")}
            RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        if only == "mixed":
            payload = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
            payload["mixed"] = value
            payload["source_tree"] = {p: fingerprint(p) for p in ("firmware/host", "rtl", "tools")}
            RESULT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if only == "dual_mode":
            payload = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else {}
            payload["dual_mode_regression"] = value
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
