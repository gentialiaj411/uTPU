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
    if name not in SELECTED_OPS:
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
                 "rtl/vector/vector_serial_alu.sv", "rtl/vector/vector_sequencer.sv", str(tb_path)],
                [str(VVP), str(exe)]):
        p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if p.returncode:
            raise RuntimeError(p.stdout + p.stderr)
    line = next(x for x in p.stdout.splitlines() if x.startswith("SEQ_RESULT"))
    _, _, cycles, misses = line.split()
    return {"cases": 1, "compared_bytes": 2 * len(expected), "mismatches": int(misses),
            "cycles": int(cycles), "passed": int(misses) == 0}


def main() -> int:
    rng = np.random.default_rng(903)
    results = {}
    a = rng.integers(-128, 128, 48, dtype=np.int8); b = rng.integers(-128, 128, 48, dtype=np.int8)
    expect = pack(sat(rne(a.astype(np.int64), 1) + rne(b.astype(np.int64), 2)))
    w = {100+i:v for i,v in enumerate(pack(a))} | {150+i:v for i,v in enumerate(pack(b))}
    w |= {800:1,801:150,802:(-5 & 255) | ((-6 & 255)<<8),803:(-4 & 255)}
    results["vadd_rq"] = run_case("vadd",0,200,100,800,48,w,expect)

    x = rng.integers(-128,128,48,dtype=np.int8); g = rng.integers(-128,128,48,dtype=np.int8)
    out,_ = rms(np.asarray([x]),g,3)
    w = {100+i:v for i,v in enumerate(pack(x))} | {600+i:v for i,v in enumerate(pack(g))}
    w |= {800:2,801:600,802:3}
    results["rmsnorm"] = run_case("norm",1,200,100,800,48,w,pack(out[0]))

    x = rng.integers(-128,128,1536,dtype=np.int8); p = rng.integers(-128,128,1536,dtype=np.int8)
    out = sat(rne(x.astype(np.int64),1)+rne(p.astype(np.int64),2))
    w = {2000+i:v for i,v in enumerate(pack(x))} | {1000+i:v for i,v in enumerate(pack(p))}
    w |= {5000:3,5001:1000,5002:(-5 & 255) | ((-6 & 255)<<8),5003:(-4 & 255)}
    results["vpos_add"] = run_case("pos",2,3000,2000,5000,1536,w,pack(out))

    x = rng.integers(-128,128,(32,32),dtype=np.int8); out=np.zeros((32,32),np.uint8)
    for row in range(32): out[row,:row+1]=soft(x[row,:row+1])
    w={2000+i:v for i,v in enumerate(pack(x))}; w|={5000:4,5001:7000}
    results["softmax"] = run_case("soft",3,3000,2000,5000,1024,w,pack(out))
    print(results)
    return int(any(not r["passed"] for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
