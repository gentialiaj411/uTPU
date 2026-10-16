"""Exercise the serialized buffer controller with Icarus and ISA arithmetic.

This is intentionally separate from the wide combinational arithmetic test:
it checks real buffer addresses, record validation, causal row traversal, and
the write-side protocol used by ``top``.
"""
from __future__ import annotations
import os, re
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "phase3_sequencer"
IVERILOG = Path(r"C:\iverilog\bin\iverilog.exe")
VVP = IVERILOG.with_name("vvp.exe")
from run_vector_rtl_sim import rne, rms, sat, soft

SELECTED_OPS = set(os.environ.get("VECTOR_SEQUENCER_OPS", "vadd,norm,pos,soft").split(","))


def pack(values: np.ndarray) -> list[int]:
    x = np.asarray(values).astype(np.uint8).reshape(-1)
    return [int(x[i]) | (int(x[i + 1]) << 8) for i in range(0, len(x), 2)]


def run_case(name: str, subop: int, dst: int, src: int, record: int, length: int,
             words: dict[int, int], expected: list[int]) -> dict:
    # Exhaustive variants share the operation selector with their base op.
    selector = "vadd" if name.startswith("vadd") else ("pos" if name.startswith("pos") else name)
    if selector not in SELECTED_OPS:
        return {"cases": 0, "compared_bytes": 0, "mismatches": 0, "cycles": 0, "passed": True, "skipped": True}
    BUILD.mkdir(parents=True, exist_ok=True)
    init = "\n".join(f"mem[{a}] = 16'h{v:04x};" for a, v in sorted(words.items()))
    checks = "\n".join(f"if (mem[{dst+i}] !== 16'h{v:04x}) begin miss=miss+1; $display(\"SEQ_MISMATCH {name} {i} %h {v:04x}\",mem[{dst+i}]); end" for i, v in enumerate(expected))
    tb = f"""module tb;
reg clk=0,rst=1,start=0; reg [3:0] subop=4'd{subop}; reg [15:0] dst=16'd{dst},src0=16'd{src},param=16'd{record},length=16'd{length};
wire busy,done,error,buf_we,buf_re,buf_store_en; wire [15:0] buf_addr,buf_wdata; reg [15:0] buf_rdata; reg [15:0] mem[0:32767]; integer i,cycles,miss;
vector_sequencer #(.ADDRESS_SIZE(16),.BUFFER_WORDS(32768)) dut(.clk(clk),.rst(rst),.start(start),.subop(subop),.dst(dst),.src0(src0),.param(param),.length(length),.buf_rdata(buf_rdata),.busy(busy),.done(done),.error(error),.buf_we(buf_we),.buf_re(buf_re),.buf_store_en(buf_store_en),.buf_addr(buf_addr),.buf_wdata(buf_wdata));
always #5 clk=~clk;
always @(posedge clk) begin if(buf_re) buf_rdata<=mem[buf_addr]; if(buf_we) mem[buf_addr]<=buf_wdata; end
initial begin for(i=0;i<32768;i=i+1) mem[i]=0; buf_rdata=0; {init}
  repeat(3) @(posedge clk); rst=0; @(negedge clk); start=1; repeat(2) @(posedge clk); @(negedge clk); start=0; cycles=0;
  while(!done && cycles<20000) begin @(posedge clk); cycles=cycles+1; end
  miss=0; if(error || !done) miss=miss+1; {checks}
  $display("SEQ_RESULT {name} %0d %0d",cycles,miss); $finish; end
endmodule
"""
    tb_path = BUILD / f"{name}.sv"
    tb_path.write_text(tb, encoding="utf-8")
    exe = BUILD / f"{name}.vvp"
    for cmd in ([str(IVERILOG), "-g2012", "-s", "tb", "-o", str(exe),
                 "rtl/vector/vector_serial_alu.sv", "rtl/vector/restoring_divider.sv", "rtl/vector/vector_sequencer.sv", str(tb_path)],
                [str(VVP), str(exe)]):
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if p.returncode:
            raise RuntimeError(p.stdout + p.stderr)
    line = next(x for x in p.stdout.splitlines() if x.startswith("SEQ_RESULT"))
    _, _, cycles, misses = line.split()
    return {"cases": 1, "compared_bytes": 2 * len(expected), "mismatches": int(misses),
            "cycles": int(cycles), "passed": int(misses) == 0}


def run_rmsnorm_suite() -> dict:
    """Run the contractual 2,000-row/all-bin RMSNorm set through the port FSM."""
    if "norm" not in SELECTED_OPS:
        return {"cases": 0, "compared_bytes": 0, "mismatches": 0, "cycles": 0, "passed": True, "skipped": True}
    rng = np.random.default_rng(304)
    found = {}
    while len(found) < 192:
        n = int(rng.integers(1, 49)); lim = int(rng.integers(1, 129)); row = np.zeros(48, np.int8)
        row[:n] = rng.integers(-lim, lim + 1, n, dtype=np.int16).astype(np.int8); rng.shuffle(row)
        _, bins = rms(np.asarray([row]), np.ones(48, np.int8), 0); found.setdefault(bins[0], row)
    rows = np.empty((2000, 48), np.int8)
    rows[:3] = [np.zeros(48, np.int8), np.full(48, -128, np.int8), np.full(48, 127, np.int8)]
    rows[3:195] = np.asarray([found[k] for k in range(192)], np.int8)
    rows[195:] = rng.integers(-128, 128, (1805, 48), dtype=np.int8)
    gamma = rng.integers(-128, 128, 48, dtype=np.int8)
    expected, bins = rms(rows, gamma, 3)
    BUILD.mkdir(parents=True, exist_ok=True)
    def write_bytes(path: Path, values: np.ndarray) -> None:
        path.write_text("\n".join(f"{int(v) & 255:02x}" for v in values.reshape(-1)) + "\n", encoding="ascii")
    paths = {"rows": BUILD / "rms_suite_rows.mem", "gamma": BUILD / "rms_suite_gamma.mem", "expected": BUILD / "rms_suite_expected.mem"}
    write_bytes(paths["rows"], rows); write_bytes(paths["gamma"], gamma); write_bytes(paths["expected"], expected)
    tb = f'''module tb;
reg clk=0,rst=1,start=0; reg [15:0] buf_rdata; wire busy,done,error,buf_we,buf_re,buf_store_en; wire [15:0] buf_addr,buf_wdata;
reg [15:0] mem[0:32767]; reg [7:0] rows[0:95999],gamma[0:47],expected_mem[0:95999]; integer i,row,cycles,miss;
vector_sequencer #(.ADDRESS_SIZE(16),.BUFFER_WORDS(32768)) dut(.clk(clk),.rst(rst),.start(start),.subop(4'd1),.dst(16'd200),.src0(16'd100),.param(16'd800),.length(16'd48),.buf_rdata(buf_rdata),.busy(busy),.done(done),.error(error),.buf_we(buf_we),.buf_re(buf_re),.buf_store_en(buf_store_en),.buf_addr(buf_addr),.buf_wdata(buf_wdata));
always #5 clk=~clk;
always @(posedge clk) begin if(buf_re) buf_rdata<=mem[buf_addr]; if(buf_we) mem[buf_addr]<=buf_wdata; end
initial begin for(i=0;i<32768;i=i+1) mem[i]=0; buf_rdata=0; $readmemh("{paths['rows'].as_posix()}",rows); $readmemh("{paths['gamma'].as_posix()}",gamma); $readmemh("{paths['expected'].as_posix()}",expected_mem); mem[800]=2;mem[801]=600;mem[802]=3; miss=0; cycles=0;
repeat(3) @(posedge clk); rst=0;
for(row=0;row<2000;row=row+1) begin
  for(i=0;i<24;i=i+1) mem[100+i]={{rows[row*48+i*2+1],rows[row*48+i*2]}};
  for(i=0;i<24;i=i+1) mem[600+i]={{gamma[i*2+1],gamma[i*2]}};
  @(negedge clk); start=1; @(negedge clk); start=0;
  while(!done && cycles<1200000) begin @(posedge clk); cycles=cycles+1; end
  if(error || !done) miss=miss+1;
  for(i=0;i<24;i=i+1) if(mem[200+i] !== {{expected_mem[row*48+i*2+1],expected_mem[row*48+i*2]}}) miss=miss+1;
end
$display("RMS_SUITE_RESULT 2000 %0d %0d",cycles,miss); $finish; end endmodule
'''
    tb_path = BUILD / "rms_suite.sv"; tb_path.write_text(tb, encoding="utf-8")
    exe = BUILD / "rms_suite.vvp"
    for cmd in ([str(IVERILOG), "-g2012", "-s", "tb", "-o", str(exe), "rtl/vector/vector_serial_alu.sv", "rtl/vector/restoring_divider.sv", "rtl/vector/vector_sequencer.sv", str(tb_path)], [str(VVP), str(exe)]):
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.returncode: raise RuntimeError(proc.stdout + proc.stderr)
    _, cases, cycles, misses = next(line for line in proc.stdout.splitlines() if line.startswith("RMS_SUITE_RESULT")).split()
    return {"cases": int(cases), "compared_bytes": 96000, "mismatches": int(misses), "cycles": int(cycles),
            "rsqrt_lut_bin_coverage": len(set(bins)), "passed": int(misses) == 0}


def run_vadd_exhaustive_suite() -> dict:
    """Exercise every INT8 pair for each exported exponent tuple.

    This deliberately runs the *serialized* controller, rather than merely
    checking the old combinational compatibility shell.  Five 1,536-code
    descriptors cover 7,680 pair values per outer iteration; the SV harness
    loops over all 65,536 pairs for every exponent tuple without recompiling.
    """
    if "vadd" not in SELECTED_OPS:
        return {"cases": 0, "compared_bytes": 0, "mismatches": 0, "cycles": 0, "passed": True, "skipped": True}
    tuples = [(-5, -5, -4), (-4, -1, 0), (-31, -31, -30), (-31, 31, 31), (30, 30, 31)]
    pairs = np.arange(-128, 128, dtype=np.int8)
    aa, bb = np.meshgrid(pairs, pairs, indexing="ij")
    aa, bb = aa.reshape(-1), bb.reshape(-1)
    # A descriptor is limited to 1,536 codes.  Run each maximum-sized
    # descriptor independently: this keeps reset/start sequencing exactly the
    # same as the production top integration and avoids a testbench shortcut.
    cases = mismatches = cycles = 0
    chunk = 1536
    for tuple_index, (ea, eb, eo) in enumerate(tuples):
        expected = sat(rne(aa.astype(np.int64), eo - ea) + rne(bb.astype(np.int64), eo - eb))
        for offset in range(0, len(aa), chunk):
            valid = min(chunk, len(aa) - offset)
            a = np.zeros(chunk, dtype=np.int8); b = np.zeros(chunk, dtype=np.int8); out = np.zeros(chunk, dtype=np.int8)
            a[:valid], b[:valid], out[:valid] = aa[offset:offset + valid], bb[offset:offset + valid], expected[offset:offset + valid]
            words = ({100 + i: v for i, v in enumerate(pack(a))} |
                     {1000 + i: v for i, v in enumerate(pack(b))} |
                     {5000: 1, 5001: 1000, 5002: (ea & 255) | ((eb & 255) << 8), 5003: eo & 255})
            result = run_case(f"vadd_exhaustive_{tuple_index}_{offset}", 0, 2000, 100, 5000, chunk, words, pack(out))
            cases += valid; mismatches += result["mismatches"]; cycles += result["cycles"]
    return {"cases": cases, "compared_bytes": cases, "mismatches": mismatches, "cycles": cycles, "passed": mismatches == 0,
            "exponent_tuples": [list(x) for x in tuples]}


def run_vpos_add_suite() -> dict:
    """Run the complete 32x48 positional table through one VPOS descriptor."""
    rng = np.random.default_rng(904)
    x = rng.integers(-128, 128, 1536, dtype=np.int8)
    pos = rng.integers(-128, 128, 1536, dtype=np.int8)
    expected = sat(rne(x.astype(np.int64), 1) + rne(pos.astype(np.int64), 2))
    words = ({2000 + i: v for i, v in enumerate(pack(x))} |
             {1000 + i: v for i, v in enumerate(pack(pos))} |
             {5000: 3, 5001: 1000, 5002: ((-6 & 255) << 8) | (-5 & 255), 5003: -4 & 255})
    result = run_case("pos", 2, 3000, 2000, 5000, 1536, words, pack(expected))
    result["cases"] = 1536
    return result


def run_softmax_suite() -> dict:
    """Run 500 causal heads (16,000 rows) through the serialized FSM."""
    if "soft" not in SELECTED_OPS:
        return {"cases": 0, "compared_bytes": 0, "mismatches": 0, "cycles": 0, "passed": True, "skipped": True}
    rng = np.random.default_rng(905)
    heads = rng.integers(-128, 128, (500, 32, 32), dtype=np.int8)
    # Include adversarial extrema while retaining every causal row length in
    # every descriptor.
    heads[0, :, :] = 127; heads[1, :, :] = -128
    expected = np.zeros_like(heads, dtype=np.uint8)
    for h in range(len(heads)):
        for row in range(32):
            expected[h, row, :row + 1] = soft(heads[h, row, :row + 1])
    BUILD.mkdir(parents=True, exist_ok=True)
    def write_bytes(path: Path, values: np.ndarray) -> None:
        path.write_text("\n".join(f"{int(v) & 255:02x}" for v in values.reshape(-1)) + "\n", encoding="ascii")
    score_file, expected_file = BUILD / "softmax_suite_scores.mem", BUILD / "softmax_suite_expected.mem"
    write_bytes(score_file, heads); write_bytes(expected_file, expected)
    tb = f'''module tb;
reg clk=0,rst=1,start=0; reg [15:0] buf_rdata; wire busy,done,error,buf_we,buf_re,buf_store_en; wire [15:0] buf_addr,buf_wdata;
reg [15:0] mem[0:32767]; reg [7:0] scores[0:511999], expected_mem[0:511999]; integer i,h,cycles,miss;
vector_sequencer #(.ADDRESS_SIZE(16),.BUFFER_WORDS(32768)) dut(.clk(clk),.rst(rst),.start(start),.subop(4'd3),.dst(16'd3000),.src0(16'd2000),.param(16'd5000),.length(16'd1024),.buf_rdata(buf_rdata),.busy(busy),.done(done),.error(error),.buf_we(buf_we),.buf_re(buf_re),.buf_store_en(buf_store_en),.buf_addr(buf_addr),.buf_wdata(buf_wdata));
always #5 clk=~clk; always @(posedge clk) begin if(buf_re) buf_rdata<=mem[buf_addr]; if(buf_we) mem[buf_addr]<=buf_wdata; end
initial begin for(i=0;i<32768;i=i+1) mem[i]=0; buf_rdata=0; $readmemh("{score_file.as_posix()}",scores); $readmemh("{expected_file.as_posix()}",expected_mem); mem[5000]=4;mem[5001]=7000;miss=0;cycles=0;
repeat(3) @(posedge clk); rst=0;
for(h=0;h<500;h=h+1) begin
  for(i=0;i<512;i=i+1) mem[2000+i]={{scores[h*1024+i*2+1],scores[h*1024+i*2]}};
  @(negedge clk);start=1;@(negedge clk);start=0;while(!done && cycles<11000000) begin @(posedge clk);cycles=cycles+1;end
  if(error || !done) miss=miss+1;
  for(i=0;i<512;i=i+1) if(mem[3000+i] !== {{expected_mem[h*1024+i*2+1],expected_mem[h*1024+i*2]}}) miss=miss+1;
end
$display("SOFTMAX_SUITE_RESULT 16000 %0d %0d",cycles,miss);$finish;end endmodule
'''
    tb_path = BUILD / "softmax_suite.sv"; tb_path.write_text(tb, encoding="utf-8")
    exe = BUILD / "softmax_suite.vvp"
    for cmd in ([str(IVERILOG), "-g2012", "-s", "tb", "-o", str(exe), "rtl/vector/vector_serial_alu.sv", "rtl/vector/restoring_divider.sv", "rtl/vector/vector_sequencer.sv", str(tb_path)], [str(VVP), str(exe)]):
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
    _, cases, cycles, misses = next(line for line in proc.stdout.splitlines() if line.startswith("SOFTMAX_SUITE_RESULT")).split()
    return {"cases": int(cases), "compared_bytes": int(cases) * 32, "mismatches": int(misses), "cycles": int(cycles), "passed": int(misses) == 0,
            "row_lengths": list(range(1, 33))}


def main() -> int:
    rng = np.random.default_rng(903)
    results = {}
    results["vadd_rq"] = run_vadd_exhaustive_suite()

    results["rmsnorm"] = run_rmsnorm_suite()

    results["vpos_add"] = run_vpos_add_suite()
    results["softmax"] = run_softmax_suite()
    print(results)
    return int(any(not r["passed"] for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
