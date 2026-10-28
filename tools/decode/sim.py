"""Small iverilog/vvp compile-and-run helper for decode unit tests."""
from __future__ import annotations
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "decode"
IVERILOG = Path(r"C:\iverilog\bin\iverilog.exe")
VVP = IVERILOG.with_name("vvp.exe")
DEFAULT_TIMEOUT_S = 6 * 3600       # wall-clock cap for one vvp run; a hang must fail, never wait forever
TIMEOUT_RC = 124                   # returned (like coreutils timeout) when vvp is killed by the wall-clock cap


def _text(x):
    return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")


def compile_and_run(sources, top, tb, plusargs=(), defines=(), timeout=DEFAULT_TIMEOUT_S, compile_timeout=3600):
    """Compile with iverilog and run with vvp. Returns (returncode, stdout).

    A vvp run that exceeds `timeout` seconds is killed and reported as (TIMEOUT_RC, stdout + a TIMEOUT line) so the
    caller's status logic sees a failure (no SUMMARY line, nonzero rc) instead of hanging.
    """
    BUILD.mkdir(parents=True, exist_ok=True)
    exe = BUILD / f"{Path(tb).stem}{'_' + '_'.join(defines) if defines else ''}.vvp"
    srcs = [str(Path(s).resolve()) for s in sources]
    tb_path = str(Path(tb).resolve())
    flags=[f"-D{x}" for x in defines]
    subprocess.run([str(IVERILOG), "-g2012", *flags, "-o", str(exe), "-s", top, *srcs, tb_path],
                   cwd=ROOT, check=True, capture_output=True, text=True, timeout=compile_timeout)
    try:
        run = subprocess.run([str(VVP), str(exe), *list(plusargs)], cwd=ROOT,
                             check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return TIMEOUT_RC, _text(e.stdout) + f"\nTIMEOUT wall-clock {timeout}s exceeded\n"
    return run.returncode, run.stdout
