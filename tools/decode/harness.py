"""Shared helpers for decode-unit differential tests (golden capture, mem files, sim, artifacts)."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import compile_and_run, ROOT, VVP
from provenance import provenance, ARTIFACT_SOURCES, e2e_sources  # noqa: F401  (re-exported for runners)

sys.path.insert(0, str(ROOT))
BUILD = ROOT / 'build/decode'


def load_runtime_260k(kv_bits=8):
    """The stories260K export/runtime used by every golden capture (matches golden_gemv.py)."""
    from tools.llama2c.reference import Llama2CModel
    from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
    from firmware.host.numerics_v2.export import export_runq_model
    ref = Llama2CModel(ROOT / 'models/llama2c/stories260K/stories260K.bin')
    cal = json.loads((ROOT / 'firmware/host/numerics_v2/calibration_260k.json').read_text())
    exp = export_runq_model(ref, cal['activation_scales'], 4, 32, improved=True,
                            mixed_precision={'qkv': 8, 'wo': 8, 'w2': 8})
    exp.pop('dynamic_base_scales')
    return exp, RunQIntegerLlama(exp, ROOT / 'firmware/host/numerics_v2/luts', kv_bits=kv_bits)


def run_tokens(runtime, tokens=(1, 2, 3)):
    for t in tokens:
        runtime.forward_argmax(t, len(runtime.tokens), 4096)


def save_cases(name, meta, cases):
    """cases: list of dicts of ints / int lists. Returns (path, sha256)."""
    out = BUILD / f'golden_{name}'
    out.mkdir(parents=True, exist_ok=True)
    blob = json.dumps({'meta': meta, 'cases': cases}, separators=(',', ':')).encode()
    path = out / 'cases.json'
    path.write_bytes(blob)
    return path, hashlib.sha256(blob).hexdigest()


def load_cases(name):
    path = BUILD / f'golden_{name}' / 'cases.json'
    blob = path.read_bytes()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


def write_mem(name, values, width):
    path = BUILD / f'{name}.mem'
    mask = (1 << width) - 1
    path.write_text(''.join(f'{int(v) & mask:0{(width + 3) // 4}x}\n' for v in values))
    return path.as_posix()


def parse_summary(out):
    line = next((x for x in out.splitlines() if x.startswith('SUMMARY')), '')
    return line, (dict(t.split('=') for t in line.split()[1:]) if line else {})


def finish(o, path, print_keys=None):
    """Write artifact `o` to `path`, print a summary line, exit 0 only if o['status'] == 'pass' (F12)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(o, indent=2) + '\n')
    print(json.dumps({k: o[k] for k in (print_keys or ('status',)) if k in o}))
    sys.exit(0 if o.get('status') == 'pass' else 1)


def run_differential(unit, rtl_sources, tb_text, mutate_define, expected_mutation, meta, sha, output, extra=None,
                     write=True, overrides=None, sources=None):
    """Compile+run normal and mutated sims, build the artifact (written unless write=False), return it.

    Status is 'pass' only if the clean run saw SUMMARY errors=0, the mutation run caught exactly expected_mutation
    (which must be > 0), and the provenance block was recorded."""
    tb = BUILD / f'{unit}_tb.sv'
    tb.write_text(tb_text)
    rc, out = compile_and_run(rtl_sources, f'{unit}_tb', tb)
    line, kv = parse_summary(out)
    mrc, mout = compile_and_run(rtl_sources, f'{unit}_tb', tb, defines=(mutate_define,))
    mline, mkv = parse_summary(mout)
    errors, merrors = int(kv.get('errors', -1)), int(mkv.get('errors', -1))
    o = {
        'status': 'pass' if (rc == 0 and errors == 0 and expected_mutation > 0 and merrors == expected_mutation
                             and sources is not None) else 'fail',
        'golden_sha256': sha, **meta,
        'mismatch_count': errors, 'real_mismatch_count': int(kv.get('real_errors', -1)),
        'first_mismatches': [x for x in out.splitlines() if x.startswith('MISMATCH')][:8],
        'mutation_mismatch_count': merrors, 'mutation_expected': expected_mutation,
        'cycles_total': int(kv.get('cycles', 0)),
        'vvp_command': f'{VVP} build/decode/{unit}_tb.vvp',
        'sim_stdout_sha256': hashlib.sha256(out.encode()).hexdigest(),
        'mutation_stdout_sha256': hashlib.sha256(mout.encode()).hexdigest(),
        'summary': line, 'mutation_summary': mline, **(extra or {}),
    }
    if sources is not None:                      # recorded last, after the sims: hashes of what is on disk when the run finished
        o['provenance'] = provenance(sources, overrides)
    if write:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(o, indent=2) + '\n')
    return o


def tie_rng(seed):
    return np.random.default_rng(seed)
