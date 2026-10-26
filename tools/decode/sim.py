"""Small iverilog/vvp compile-and-run helper for decode unit tests."""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "decode"

def compile_and_run(sources, top, tb, plusargs=()):
    BUILD.mkdir(parents=True, exist_ok=True)
    exe = BUILD / f"{top}.vvp"
    srcs = [str(Path(s).resolve()) for s in sources]
    tb_path = str(Path(tb).resolve())
    subprocess.run(["iverilog", "-g2012", "-o", str(exe), "-s", top, *srcs, tb_path],
                   cwd=ROOT, check=True, capture_output=True, text=True)
    run = subprocess.run(["vvp", str(exe), *list(plusargs)], cwd=ROOT,
                         check=False, capture_output=True, text=True)
    return run.returncode, run.stdout
