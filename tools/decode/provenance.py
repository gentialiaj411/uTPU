"""Source provenance for decode verification artifacts (stdlib only; shared by runners and tools/decode_gate.py).

An artifact records `provenance` = {repo-relative path: sha256, 'tool:iverilog': sha256(version line)} for every
input its run depended on. The gate recomputes the hashes; any difference means the artifact is stale.

ARTIFACT_SOURCES is the single place that lists what each artifact type depends on (glob patterns, repo-relative).
"""
from __future__ import annotations
import glob, hashlib, os, subprocess, sys, platform
from functools import lru_cache
import numpy as np
from sim import simulator_identity
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IVERILOG = Path(r"C:\iverilog\bin\iverilog.exe")

D = 'rtl/decode/'
TOOLS = ['tools/decode/sim.py', 'tools/decode/harness.py', 'tools/decode/provenance.py']
NUMERICS = ['firmware/host/numerics_v2/runq_runtime.py', 'firmware/host/numerics_v2/export.py',
            'firmware/host/numerics_v2/fixed.py', 'firmware/host/numerics_v2/calibration_260k.json',
            'firmware/host/numerics_v2/luts/*', 'tools/llama2c/reference.py']
LUTS = ['tools/decode/gen_luts.py', 'rtl/decode/luts/*.mem']        # generated from firmware luts.json by gen_luts.py
M260 = ['models/llama2c/stories260K/stories260K.bin']
M15 = ['models/numerics_v2/gptq_qkv_wo_w2_int8_g32.npz']
TOP_RTL = [D + f for f in ('decode_pkg.sv', 'seq_mul.sv', 'mul_pipe.sv', 'div_const_pipe.sv', 'gemv_unit.sv', 'act_quant.sv',
                           'rmsnorm.sv', 'vec_alu.sv', 'attention.sv', 'top_decode.sv')]
QSPI_RTL = [D + 'qspi_reader.sv', 'rtl/board/qspi_pad_regs.sv', 'tools/decode/tb/qspi_flash_model.sv']
BOARD_RTL = ['rtl/board/*.sv']
GOLD = lambda *n: [f'tools/decode/golden_{x}.py' for x in n]

# artifact name (file stem under bench/results) -> dependency globs. The runner script itself is the TB template source.
ARTIFACT_SOURCES = {
    'decode_gemv_rtl': ['tools/decode/run_gemv_rtl.py', D + 'gemv_unit.sv', *GOLD('gemv'), *TOOLS, *NUMERICS, *M260],
    'decode_act_quant_rtl': ['tools/decode/run_act_quant_rtl.py', D + 'act_quant.sv', *GOLD('act_quant'), *TOOLS, *NUMERICS, *M260, *M15],
    'decode_rms_rtl': ['tools/decode/run_rms_rtl.py', D + 'decode_pkg.sv', D + 'seq_mul.sv', D + 'rmsnorm.sv',
                       *GOLD('rms'), *TOOLS, *NUMERICS, *M260, *M15],
    'decode_vec_alu_rtl': ['tools/decode/run_vec_rtl.py', D + 'decode_pkg.sv', D + 'vec_alu.sv', *GOLD('vec'), *TOOLS,
                           *NUMERICS, *LUTS, *M260, *M15],
    'decode_attention_rtl': ['tools/decode/run_attn_rtl.py', D + 'decode_pkg.sv', D + 'mul_pipe.sv', D + 'div_const_pipe.sv',
                             D + 'attention.sv', *GOLD('attn'), *TOOLS, *NUMERICS, *LUTS, *M260],
    'decode_gemv15_rtl': ['tools/decode/run_gemv_rtl.py', D + 'gemv_unit.sv', *GOLD('gemv15', 'gemv'), 'tools/decode/run_decode_rtl.py',
                          *TOOLS, *NUMERICS, *M15],
    'decode_attention15_rtl': ['tools/decode/run_attn15_rtl.py', D + 'decode_pkg.sv', D + 'mul_pipe.sv', D + 'div_const_pipe.sv',
                               D + 'attention.sv', *GOLD('attn15', 'attn'), 'tools/decode/run_decode_rtl.py', *TOOLS, *NUMERICS, *LUTS, *M15],
    'decode_qspi_rtl': ['tools/decode/run_qspi_rtl.py', *QSPI_RTL, *TOOLS],
    'decode_qspi_ddr_rtl': ['tools/decode/run_qspi_rtl.py', *QSPI_RTL, *TOOLS],      # same sources, run with --ddr (EDh, CR1 LC=11)
    # M5 board top. The UART TB builds boolean_core (every rtl/board and rtl/decode source) around the flash model and plays
    # the host tool's frames against the 260K golden (run_decode_rtl.golden + pack_decode_image).
    'boolean_uart_rtl': ['tools/board/run_uart_rtl.py', 'tools/board/boolean_host.py', 'tools/board/tb/*.sv', *BOARD_RTL,
                         'tools/decode/run_decode_rtl.py', 'tools/decode/pack_decode_image.py',
                         *QSPI_RTL, *TOP_RTL, *TOOLS, *NUMERICS, *LUTS, *M260],
    # DDR variant of the UART TB (--ddr): boolean_core DDR=1, flash model powered up at LC=00 so the host procedure must switch it to LC=11
    'boolean_uart_ddr_rtl': ['tools/board/run_uart_rtl.py', 'tools/board/boolean_host.py', 'tools/board/tb/*.sv', *BOARD_RTL,
                             'tools/decode/run_decode_rtl.py', 'tools/decode/pack_decode_image.py',
                             *QSPI_RTL, *TOP_RTL, *TOOLS, *NUMERICS, *LUTS, *M260],
    # pin check: the XDC is parsed; the placed ports come from the implementation run, so its artifact is an input
    'boolean_pin_check': ['tools/board/pin_check.py', 'constraints/boolean_xc7s50.xdc', 'tools/board/impl_top.py',
                          'bench/results/boolean_top_impl.json', 'tools/decode/provenance.py'],
    # full implementation of boolean_top: every source in impl_top.SRCS, the XDC, and the 15M image parameters checked
    'boolean_top_impl': ['tools/board/impl_top.py', 'constraints/boolean_xc7s50.xdc', *BOARD_RTL, *TOP_RTL, D + 'qspi_reader.sv',
                         'tools/decode/pack_decode_image.py', 'tools/decode/provenance.py', *M15,
                         'build/decode/decode_image_15m/params.json'],
    # the same implementation with boolean_top DDR=1 (impl_top.py --ddr); the pin check keeps consuming the default boolean_top_impl
    'boolean_top_impl_ddr': ['tools/board/impl_top.py', 'constraints/boolean_xc7s50.xdc', 'constraints/boolean_xc7s50_qspi_ddr.xdc', *BOARD_RTL, *TOP_RTL, D + 'qspi_reader.sv',
                             'tools/decode/pack_decode_image.py', 'tools/decode/provenance.py', *M15,
                             'build/decode/decode_image_15m/params.json'],
}
# out-of-context synthesis records (decode_synth_ooc.json) by entry name; the 15M wrappers are generated into build/ from the 15M image
# parameters by gen_15m_wrap.py / gen_qspi_wrap.py, so the generated wrapper, its generators and the packer are inputs too
_W15 = ['build/decode/top_decode_15m_wrap.sv', 'tools/decode/gen_15m_wrap.py', 'tools/decode/pack_decode_image.py']
SYNTH_SOURCES = {
    'gemv_unit': [D + 'gemv_unit.sv', 'tools/decode/synth_ooc.py'],
    'act_quant': [D + 'act_quant.sv', 'tools/decode/synth_ooc.py'],
    'rmsnorm': [D + 'decode_pkg.sv', D + 'seq_mul.sv', D + 'rmsnorm.sv', 'tools/decode/synth_ooc.py'],
    'vec_alu': [D + 'decode_pkg.sv', D + 'vec_alu.sv', 'tools/decode/synth_ooc.py'],
    'attention': [D + 'decode_pkg.sv', D + 'mul_pipe.sv', D + 'div_const_pipe.sv', D + 'attention.sv', 'tools/decode/synth_ooc.py'],
    'top_decode_260k': [*TOP_RTL, 'tools/decode/synth_ooc.py'],
    'top_decode_15m': [*TOP_RTL, *_W15, 'build/decode/decode_image_15m/params.json', 'tools/decode/synth_ooc.py'],
    'qspi_reader': [D + 'qspi_reader.sv', 'tools/decode/synth_ooc.py'],
    'top_decode_15m_qspi': [*TOP_RTL, D + 'qspi_reader.sv', *_W15, 'build/decode/top_decode_15m_qspi_wrap.sv', 'build/decode/decode_image_15m/params.json',
                            'tools/decode/gen_qspi_wrap.py', 'tools/decode/synth_ooc.py'],
    # DDR=1 reader: qspi_reader alone (synth_ooc -GDDR=1) and the 15M top with the DDR reader (gen_qspi_wrap.py --ddr)
    'qspi_reader_ddr': [D + 'qspi_reader.sv', 'tools/decode/synth_ooc.py'],
    'top_decode_15m_qspi_ddr': [*TOP_RTL, D + 'qspi_reader.sv', *_W15, 'build/decode/top_decode_15m_qspi_ddr_wrap.sv', 'build/decode/decode_image_15m/params.json',
                                'tools/decode/gen_qspi_wrap.py', 'tools/decode/synth_ooc.py'],
}
# fault-injection results are per unit; they depend on every RTL file the faults can touch plus the injector and runners
FAULT_SOURCES = [*NUMERICS, *M260, *M15, *LUTS, 'tools/decode/pack_decode_image.py', 'tools/decode/golden_*.py', D + '*.sv', 'rtl/board/*.sv', 'tools/decode/tb/*.sv', 'tools/decode/fault_inject.py', 'tools/decode/run_*_rtl.py',
                 'tools/board/run_uart_rtl.py', 'tools/board/tb/*.sv', 'tools/board/boolean_host.py', *TOOLS]


def e2e_sources(model: str, flash: str) -> list:
    """model: 'stories260K' | 'stories15M'; flash: 'ideal' | 'qspi' | 'qspi-ddr'."""
    return ['tools/decode/run_decode_rtl.py', 'tools/decode/pack_decode_image.py', *TOP_RTL, *(QSPI_RTL if flash in ('qspi', 'qspi-ddr') else []),
            *TOOLS, *NUMERICS, *LUTS, *(M15 if model == 'stories15M' else M260)]


def iverilog_version() -> str:
    return _iverilog_version_cached(sha256_file(IVERILOG) if IVERILOG.exists() else 'missing')

@lru_cache(maxsize=8)
def _iverilog_version_cached(executable_digest) -> str:
    try:
        r = subprocess.run([str(IVERILOG), '-V'], capture_output=True, text=True, timeout=30)
        return (r.stdout.splitlines() or ['unknown'])[0].strip()
    except Exception as e:                                    # noqa: BLE001
        return f'unavailable:{type(e).__name__}'


def sha256_file(p) -> str:
    """Text files (no NUL in the first 8 KiB) hash with CRLF normalized to LF, so CRLF and LF checkouts of the same commit
    agree; binary files hash raw."""
    data = Path(p).read_bytes()
    if b'\0' not in data[:8192]:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def rel(p) -> str:
    """Repo-relative key without following links: a junction/symlink inside the repo (e.g. models/) keeps its repo path."""
    p = Path(p)
    p = Path(os.path.normpath(os.path.abspath(p if p.is_absolute() else ROOT / p)))
    for base in (ROOT, Path(os.path.abspath(ROOT))):
        try:
            return p.relative_to(base).as_posix()
        except ValueError:
            pass
    return p.as_posix()


def expand(patterns) -> list:
    out = []
    for pat in patterns:
        hits = sorted(glob.glob(str(ROOT / pat)))
        if not hits:
            raise FileNotFoundError(f'provenance source missing: {pat}')
        out += [rel(h) for h in hits if Path(h).is_file()]
    return sorted(set(out))


def provenance(paths, overrides=None) -> dict:
    """paths: patterns/paths. overrides: {default relpath: actual path} when a runner swaps in a mutant file."""
    files = expand(paths)
    ov = {rel(k): Path(v) for k, v in (overrides or {}).items()}
    res = {}
    for f in files:
        if f in ov:
            res[rel(ov[f])] = sha256_file(ov[f])
        else:
            res[f] = sha256_file(ROOT / f)
    res['tool:iverilog'] = hashlib.sha256(iverilog_version().encode()).hexdigest()
    res['tool:simulator'] = simulator_identity()
    res['runtime:python'] = hashlib.sha256(sys.version.encode()).hexdigest()
    res['runtime:numpy'] = hashlib.sha256(np.__version__.encode()).hexdigest()
    return res


def stale(prov: dict, name: str) -> list:
    """Messages for each recorded file whose hash no longer matches."""
    msgs = []
    for k, h in prov.items():
        if k == 'tool:simulator':
            import sim
            selected = sim.SIMULATOR
            try:
                sim.SIMULATOR = h['name']
                if simulator_identity() != h: msgs.append(f'stale {name}: simulator identity changed')
            except Exception as exc:
                msgs.append(f'stale {name}: simulator identity unavailable: {exc}')
            finally: sim.SIMULATOR = selected
            continue
        if k.startswith('runtime:'):
            value = sys.version if k == 'runtime:python' else np.__version__
            if hashlib.sha256(value.encode()).hexdigest() != h: msgs.append(f'stale {name}: {k} changed')
            continue
        if k == 'tool:iverilog':
            if IVERILOG.exists() and hashlib.sha256(iverilog_version().encode()).hexdigest() != h:
                msgs.append(f'stale {name}: iverilog version changed')
            continue
        p = ROOT / k
        if not p.is_file():
            msgs.append(f'stale {name}: {k} missing')
        elif sha256_file(p) != h:
            msgs.append(f'stale {name}: {k} changed')
    return msgs


def missing_coverage(prov: dict, patterns, name: str) -> list:
    """Files the artifact type depends on that its provenance block does not list."""
    try:
        need = expand(patterns)
    except FileNotFoundError as e:
        return [f'{name}: {e}']
    return [f'provenance of {name} lacks {f}' for f in [*need, 'tool:simulator', 'runtime:python', 'runtime:numpy'] if f not in prov]
