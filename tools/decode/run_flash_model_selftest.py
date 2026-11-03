"""Runs tools/decode/tb/test_flash_model_ddr.sv (hand-driven self-test of qspi_flash_model.sv: EDh DDR read, LC-dependent
dummy counts, error checks) under Icarus and writes build/flash_model_selftest.json. Exit 0 only if every check passed and
the number of model errors seen equals the number the test expects, also cross-checked against the FLASH-DDR lines in the log."""
from __future__ import annotations
import json, random, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
IV = Path(r"C:\iverilog\bin\iverilog.exe"); VVP = IV.with_name("vvp.exe")

def main():
    b = ROOT / "build"; b.mkdir(exist_ok=True)
    rng = random.Random(25)
    (b / "flash_selftest_img.hex").write_text("\n".join(f"{rng.randrange(256):02x}" for _ in range(4096)) + "\n")
    vvp = b / "flash_model_selftest.vvp"
    srcs = [ROOT / "tools/decode/tb/qspi_flash_model.sv", ROOT / "tools/decode/tb/test_flash_model_ddr.sv"]
    c = subprocess.run([str(IV), "-g2012", "-s", "test_flash_model_ddr", "-o", str(vvp), *map(str, srcs)], capture_output=True, text=True, cwd=ROOT, timeout=600)
    if c.returncode: print(c.stdout, c.stderr); return 1
    r = subprocess.run([str(VVP), str(vvp)], capture_output=True, text=True, cwd=ROOT, timeout=600)
    log = r.stdout + r.stderr
    (b / "flash_model_selftest.log").write_text(log)
    m = re.search(r"SUMMARY checks=(\d+) errors_expected=(\d+) errors_seen=(\d+) fails=(\d+)", log)
    if not m: print(log[-2000:]); print("NO SUMMARY"); return 1
    checks, exp, seen, fails = map(int, m.groups())
    lines = len(re.findall(r"FLASH-DDR:", log))
    ok = r.returncode == 0 and fails == 0 and exp == seen == lines and "FAIL" not in log and "FLASH:" not in log
    o = dict(status="pass" if ok else "fail", checks=checks, errors_expected=exp, errors_seen=seen, flash_ddr_error_lines=lines, fails=fails,
             iverilog=subprocess.run([str(IV), "-V"], capture_output=True, text=True).stdout.splitlines()[0])
    (b / "flash_model_selftest.json").write_text(json.dumps(o, indent=2) + "\n")
    print(m.group(0), f"log_error_lines={lines}", o["status"])
    return 0 if ok else 1
if __name__ == "__main__": sys.exit(main())
