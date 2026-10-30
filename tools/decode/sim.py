"""Shared Icarus/Verilator compile-and-run helper for decode tests."""
from __future__ import annotations
import subprocess
import os, sys, re, shlex, hashlib, json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "decode"
IVERILOG = Path(r"C:\iverilog\bin\iverilog.exe")
VVP = IVERILOG.with_name("vvp.exe")
DEFAULT_TIMEOUT_S = 6 * 3600       # wall-clock cap for one vvp run; a hang must fail, never wait forever
SIMULATOR = os.environ.get("UTPU_SIM", "icarus")
if "--sim" in sys.argv:
    i = sys.argv.index("--sim")
    SIMULATOR = sys.argv[i+1]
    del sys.argv[i:i+2]
for i, arg in enumerate(sys.argv):
    if arg.startswith('--sim='):
        SIMULATOR = arg.split('=',1)[1]
        del sys.argv[i]
        break
if SIMULATOR not in ("icarus", "verilator"):
    raise ValueError("--sim must be icarus or verilator")
os.environ["UTPU_SIM"] = SIMULATOR
VERILATOR_TOOLS = Path(os.environ.get("UTPU_VERILATOR_TOOLS", str(ROOT.parent / "uTPU-research/research/data/tools/usr")))

def linux_path(path):
    text = str(Path(path).resolve()).replace(chr(92), "/")
    return re.sub(r"^([A-Za-z]):", lambda m: "/mnt/"+m[1].lower(), text)

@lru_cache(maxsize=8)
def _version(executable, digest, linux):
    cmd = ['wsl','-d','Ubuntu','--',linux_path(executable),'--version'] if linux else [str(executable),'-V']
    r = subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    if r.returncode: raise RuntimeError('simulator version query failed')
    return (r.stdout+r.stderr).strip()

def simulator_identity():
    if SIMULATOR == "icarus":
        exe, compiler = VVP, IVERILOG
        version = _version(exe, hashlib.sha256(exe.read_bytes()).hexdigest(), False)
    else:
        exe = VERILATOR_TOOLS / "bin/verilator_bin"
        compiler = exe
        version = _version(exe, hashlib.sha256(exe.read_bytes()).hexdigest(), True)
    return {"name": SIMULATOR, "version": version, "executable": str(exe), "executable_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(), "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest()}

def _cleanup_verilator_build(directory):
    script = "import sys,shutil;from pathlib import Path;p=Path(sys.argv[1]);root=Path('/tmp/utpu-perf');assert p.parent.resolve()==root.resolve() and p.name.endswith('_obj');shutil.rmtree(p) if p.exists() else None"
    subprocess.run(['wsl','-d','Ubuntu','--','python3','-c',script,directory],capture_output=True,timeout=120,check=True)

def run_verilator(sources, top, tb, plusargs, defines, timeout, compile_timeout):
    tag = Path(tb).stem + ("_"+"_".join(defines) if defines else "")
    converted = BUILD / (tag+"_verilator.sv")
    text = Path(tb).read_text()
    text = re.sub(r"([A-Za-z]):/", lambda m: "/mnt/"+m[1].lower()+"/", text)
    converted.write_text(text)
    obj = BUILD / (tag+"_obj")
    obj_linux = "/tmp/utpu-perf/"+tag+"_obj"
    args = [linux_path(VERILATOR_TOOLS/"bin/verilator_bin"), "--binary", "--timing", "-Wno-fatal", "-j", "1", "--top-module", top, "--Mdir", obj_linux, *["-D"+x for x in defines], *[linux_path(x) for x in sources], linux_path(converted)]
    command = "mkdir -p /tmp/utpu-perf; export VERILATOR_ROOT="+shlex.quote(linux_path(VERILATOR_TOOLS/"share/verilator"))+"; "+shlex.join(args)
    result = subprocess.run(["wsl","-d","Ubuntu","--","bash","-lc",command], cwd=ROOT, capture_output=True,text=True,timeout=compile_timeout)
    (BUILD/(tag+"_compile.log")).write_text(result.stdout+result.stderr)
    if result.returncode:
        _cleanup_verilator_build(obj_linux)
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout, stderr=result.stderr)
    try:
        r = subprocess.run(["wsl","-d","Ubuntu","--",obj_linux+"/V"+top,*map(str,plusargs)],cwd=ROOT,capture_output=True,text=True,timeout=timeout)
        _cleanup_verilator_build(obj_linux)
        return r.returncode,r.stdout+r.stderr
    except subprocess.TimeoutExpired as e:
        _cleanup_verilator_build(obj_linux)
        return TIMEOUT_RC,_text(e.stdout)+"\nTIMEOUT Verilator wall cap exceeded\n"

TIMEOUT_RC = 124                   # returned (like coreutils timeout) when vvp is killed by the wall-clock cap


def _text(x):
    return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")


def compile_and_run(sources, top, tb, plusargs=(), defines=(), timeout=DEFAULT_TIMEOUT_S, compile_timeout=3600):
    """Compile with iverilog and run with vvp. Returns (returncode, stdout).

    A vvp run that exceeds `timeout` seconds is killed and reported as (TIMEOUT_RC, stdout + a TIMEOUT line) so the
    caller's status logic sees a failure (no SUMMARY line, nonzero rc) instead of hanging.
    """
    BUILD.mkdir(parents=True, exist_ok=True)
    testbench = Path(tb)
    text = testbench.read_text()
    if "`timescale" not in text and "timeunit" not in text:
        testbench.write_text("`timescale 1ns/1ps\n" + text)
    if SIMULATOR == "verilator":
        return run_verilator(sources, top, tb, plusargs, defines, timeout, compile_timeout)
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
