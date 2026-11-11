"""Differential test the serialized VSOFTMAX restoring divider."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "phase3_divider"
IVERILOG = Path(r"C:\iverilog\bin\iverilog.exe")
VVP = IVERILOG.with_name("vvp.exe")


def write_words(path: Path, values: np.ndarray) -> None:
    path.write_text("\n".join(f"{int(v):06x}" for v in values) + "\n", encoding="ascii")


def run_suite() -> dict:
    rng = np.random.default_rng(1337)
    denominator = np.concatenate((np.array([1, 1, (1 << 22) - 1, (1 << 22) - 1], np.uint32), rng.integers(1, 1 << 22, 2044, dtype=np.uint32)))
    numerator = np.concatenate((np.array([0, (1 << 24) - 1, 0, (1 << 24) - 1], np.uint32), rng.integers(0, 1 << 24, 2044, dtype=np.uint32)))
    expected = numerator // denominator
    BUILD.mkdir(parents=True, exist_ok=True)
    paths = {name: BUILD / f"{name}.mem" for name in ("numerator", "denominator", "expected")}
    for name, values in (("numerator", numerator), ("denominator", denominator), ("expected", expected)):
        write_words(paths[name], values)
    count = len(numerator)
    tb = f'''module tb;
reg clk=0,rst=1,start=0; reg [23:0] numerator,denominator; wire busy,done; wire [23:0] quotient;
reg [23:0] nums[0:{count - 1}],dens[0:{count - 1}],expected[0:{count - 1}]; integer i,cycles,miss;
restoring_divider dut(.clk(clk),.rst(rst),.start(start),.numerator(numerator),.denominator(denominator[21:0]),.busy(busy),.done(done),.quotient(quotient));
always #5 clk=~clk;
initial begin $readmemh("{paths['numerator'].as_posix()}",nums);$readmemh("{paths['denominator'].as_posix()}",dens);$readmemh("{paths['expected'].as_posix()}",expected); numerator=0;denominator=0;miss=0;cycles=0;repeat(3) @(posedge clk);rst=0;
for(i=0;i<{count};i=i+1) begin numerator=nums[i];denominator=dens[i];@(negedge clk);start=1;@(negedge clk);start=0;while(!done && cycles<60000) begin @(posedge clk);cycles=cycles+1;end if(!done || quotient!==expected[i]) miss=miss+1; end
$display("DIVIDER_RESULT {count} %0d %0d",cycles,miss);$finish;end endmodule
'''
    tb_path = BUILD / "divider_tb.sv"; tb_path.write_text(tb, encoding="utf-8")
    exe = BUILD / "divider.vvp"
    for cmd in ([str(IVERILOG), "-g2012", "-s", "tb", "-o", str(exe), "rtl/vector/restoring_divider.sv", str(tb_path)], [str(VVP), str(exe)]):
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
    _, cases, cycles, mismatches = next(line for line in proc.stdout.splitlines() if line.startswith("DIVIDER_RESULT")).split()
    result = {"cases": int(cases), "cycles": int(cycles), "mismatches": int(mismatches), "passed": int(mismatches) == 0}
    return result


def main() -> int:
    result = run_suite()
    print(result)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
