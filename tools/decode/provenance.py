"""Source provenance for decode verification artifacts (stdlib only; shared by runners and tools/decode_gate.py).

An artifact records `provenance` = {repo-relative path: sha256, 'tool:iverilog': sha256(version line)} for every
input its run depended on. The gate recomputes the hashes; any difference means the artifact is stale.

ARTIFACT_SOURCES is the single place that lists what each artifact type depends on (glob patterns, repo-relative).
"""
from __future__ import annotations
import glob, hashlib, subprocess
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
}
# fault-injection results are per unit; they depend on every RTL file the faults can touch plus the injector and runners
FAULT_SOURCES = [D + '*.sv', 'rtl/board/*.sv', 'tools/decode/tb/*.sv', 'tools/decode/fault_inject.py', 'tools/decode/run_*_rtl.py',
                 'tools/board/run_uart_rtl.py', *TOOLS]


def e2e_sources(model: str, flash: str) -> list:
    """model: 'stories260K' | 'stories15M'; flash: 'ideal' | 'qspi'."""
    return ['tools/decode/run_decode_rtl.py', 'tools/decode/pack_decode_image.py', *TOP_RTL, *(QSPI_RTL if flash == 'qspi' else []),
            *TOOLS, *NUMERICS, *LUTS, *(M15 if model == 'stories15M' else M260)]


def iverilog_version() -> str:
    try:
        r = subprocess.run([str(IVERILOG), '-V'], capture_output=True, text=True, timeout=30)
        return (r.stdout.splitlines() or ['unknown'])[0].strip()
    except Exception as e:                                    # noqa: BLE001
        return f'unavailable:{type(e).__name__}'


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def rel(p) -> str:
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


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
    return res


def stale(prov: dict, name: str) -> list:
    """Messages for each recorded file whose hash no longer matches."""
    msgs = []
    for k, h in prov.items():
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
    return [f'provenance of {name} lacks {f}' for f in need if f not in prov]
