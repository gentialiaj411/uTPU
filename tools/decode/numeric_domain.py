"""Numeric-domain characterization for the integer decode runtime (audit N07/F10/F11 support).

The integer runtime (firmware/host/numerics_v2/runq_runtime.py + fixed.py) is the spec; nothing here changes a
result. This module
  * loads the 260K and 15M runtimes and real eval-text token windows (shared with golden_*.py),
  * instruments every fixed.div_const_rne call site (max |numerator|, wide-path hits, observed error vs exact RNE),
  * instruments runq_runtime._quantize (input domain, INT32_MIN),
  * measures div_const_rne's error vs exact RNE over each call site's characterized numerator range, and
  * writes build/decode/numeric_domain.json (consumed by docs/numeric_domain.md and test_numerics_v2.py).

Usage: python tools/decode/numeric_domain.py [--prompts N] [--length L] [--samples S] [--skip-15m]
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT, BUILD
from firmware.host.numerics_v2 import fixed, runq_runtime as rr
from firmware.host.numerics_v2.fixed import div_rne

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
NARROW_LIMIT = 1 << 31            # fixed.div_const_rne: |n| <= 2^31 -> exact path, above -> "wide" approximate path


# ---- loaders -----------------------------------------------------------------------------------------
def load_260k(kv_bits=8):
    from harness import load_runtime_260k
    return load_runtime_260k(kv_bits)


def load_15m(kv_bits=8):
    from run_decode_rtl import load_model
    return load_model('15m', kv_bits)


def eval_ids(model, minimum=50_000):
    """Token ids of the eval slice exactly as firmware/host/numerics_v2/run_{260k,15m}.py tokenize it (cached)."""
    model = model.lower()
    cache = BUILD / f'eval_ids_{model}.json'
    if cache.exists():
        return json.loads(cache.read_text())
    from tools.llama2c.metrics import tokenize_eval
    from tools.llama2c.tokenizer import Tokenizer
    base = ROOT / 'models/llama2c'
    if model == '260k':
        tok = Tokenizer(base / 'stories260K/tok512.bin', 512)
    else:
        tok = Tokenizer(base / 'tokenizer.bin', 32000)
    ids = tokenize_eval(tok, base / 'eval_slice.txt', minimum=minimum)[:minimum + 1]
    BUILD.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids))
    return ids


def eval_windows(model, count, length, minimum=50_000):
    """The eval prompts' start offsets (run_260k.prompts_from: 64 evenly spaced windows) extended to `length`
    real-text tokens (teacher forced), so activations see real text rather than 8-token prompts only."""
    ids = eval_ids(model, minimum)
    span = len(ids) - 8 - 128
    total = 64
    starts = [(i * span) // total for i in range(total)]
    pick = [starts[(i * total) // count] for i in range(count)]
    return [ids[s:s + length] for s in pick]


# ---- exact reference ---------------------------------------------------------------------------------
def exact_rne_div(n, d):
    """Exact round-half-even n/d on Python ints (sign-symmetric like fixed.div_rne)."""
    mag = abs(int(n))
    q, r = divmod(mag, d)
    if 2 * r > d or (2 * r == d and (q & 1)):
        q += 1
    return -q if n < 0 else q


def exact_rne_div_array(n, d):
    """Vectorised exact RNE (int64 inputs, |n| < 2^62)."""
    return div_rne(np.asarray(n, dtype=np.int64), d)


# ---- instrumentation ---------------------------------------------------------------------------------
def _new_site():
    return {'calls': 0, 'values': 0, 'max_abs_n': 0, 'wide_values': 0, 'max_err': 0, 'max_err_n': 0}


class Probe:
    """Context manager recording div_const_rne call sites and _quantize inputs while a runtime executes."""

    def __init__(self):
        self.sites = {}
        self.quant = {'groups': 0, 'min_raw': 0, 'max_raw': 0, 'int32_min_hits': 0, 'calls': {}}

    def __enter__(self):
        self.orig_fixed, self.orig_rr, self.orig_q = fixed.div_const_rne, rr.div_const_rne, rr._quantize
        probe = self

        def wrapped(value, denominator):
            f = sys._getframe(1)
            if f.f_code.co_name == 'div127' and f.f_globals.get('__name__', '').endswith('fixed'):
                f = f.f_back
            site = f"{f.f_code.co_name}:{f.f_lineno}"
            out = probe.orig_fixed(value, denominator)
            n = np.asarray(value, dtype=np.int64)
            mag = np.abs(n)
            rec = probe.sites.setdefault((site, int(denominator)), _new_site())
            rec['calls'] += 1
            rec['values'] += int(n.size)
            if n.size:
                rec['max_abs_n'] = max(rec['max_abs_n'], int(mag.max()))
            wide = mag > NARROW_LIMIT
            if wide.any():
                rec['wide_values'] += int(wide.sum())
                flat_n = n.reshape(-1)[wide.reshape(-1)]
                flat_o = np.asarray(out).reshape(-1)[wide.reshape(-1)]
                for x, y in zip(flat_n.tolist(), flat_o.tolist()):
                    e = abs(int(y) - exact_rne_div(x, int(denominator)))
                    if e > rec['max_err']:
                        rec['max_err'], rec['max_err_n'] = e, abs(x)
            return out

        def quantize(raw, group_size=rr.ACTIVATION_GROUP, qmax=127):
            v = np.asarray(raw, dtype=np.int64)
            q = probe.quant
            q['groups'] += int(math.ceil(v.shape[-1] / group_size)) * int(np.prod(v.shape[:-1], dtype=np.int64))
            q['min_raw'], q['max_raw'] = min(q['min_raw'], int(v.min())), max(q['max_raw'], int(v.max()))
            q['int32_min_hits'] += int((v == I32_MIN).sum())
            f = sys._getframe(1)
            c = q['calls'].setdefault(f"{f.f_code.co_name}:{f.f_lineno}", {'min': 0, 'max': 0})
            c['min'], c['max'] = min(c['min'], int(v.min())), max(c['max'], int(v.max()))
            return probe.orig_q(raw, group_size, qmax)

        def watch(owner, name, tag, arg=None):
            orig = getattr(owner, name)
            setattr(owner, name, orig)
            probe._restore.append((owner, name, orig))

            def w(*args, **kw):
                out = orig(*args, **kw)
                for label, arr in ((tag, out),) + (((tag + '_in', args[arg]),) if arg is not None else ()):
                    m = int(np.max(np.abs(np.asarray(arr, dtype=np.int64)))) if np.size(arr) else 0
                    probe.ops[label] = max(probe.ops.get(label, 0), m)
                return out
            setattr(owner, name, w)

        self._restore, self.ops = [], {}
        cls = rr.RunQIntegerLlama
        watch(rr, '_linear_high', 'linear_high_out')
        watch(rr, '_attention_value', 'attention_value_out')
        watch(rr, '_rms_high', 'rms_out', arg=0)
        watch(cls, '_score', 'score_out')
        watch(cls, '_silu_high', 'silu_out')
        watch(cls, '_hidden_high', 'hidden_out')
        watch(cls, '_rope_high', 'rope_out')
        watch(cls, '_embedding_high', 'embedding_out')
        fixed.div_const_rne = rr.div_const_rne = wrapped
        rr._quantize = quantize
        return self

    def __exit__(self, *exc):
        fixed.div_const_rne, rr.div_const_rne, rr._quantize = self.orig_fixed, self.orig_rr, self.orig_q
        for owner, name, orig in self._restore:
            setattr(owner, name, orig)


def run_windows(rt, windows, full_every=0):
    """Teacher-forced decode of each window; the classifier runs on every `full_every`-th step (0 = never)."""
    for w in windows:
        rt.reset()
        for pos, tok in enumerate(w):
            if full_every and pos % full_every == 0:
                rt.forward(tok, pos)
            else:
                rt._forward_hidden(tok, pos)


def probe_model(model, kv_bits, count, length, full_every, loaded=None):
    exp, rt = loaded or (load_260k if model == '260k' else load_15m)(kv_bits)
    windows = eval_windows(model, count, length)
    with Probe() as p:
        t0 = time.time()
        run_windows(rt, windows, full_every)
    return p, {'model': model, 'kv_bits': kv_bits, 'prompts': count, 'tokens_per_prompt': length,
               'classifier_every': full_every, 'seconds': round(time.time() - t0, 1)}, exp, rt


# ---- error characterization ---------------------------------------------------------------------------
def error_profile(d, hi, samples, rng):
    """Max |div_const_rne(n, d) - exact RNE| over the wide range (2^31, hi]: boundaries, tie neighbourhoods,
    log-uniform and uniform-near-top random samples. Calls the real fixed.div_const_rne."""
    lo = NARROW_LIMIT + 1
    pts = {lo, lo + 1, 2 * lo, hi, hi - 1}
    pts |= {(1 << k) + o for k in range(32, 63) for o in (-1, 0, 1) if lo <= (1 << k) + o <= hi}
    for base in (hi, hi // 2, hi // 8, 1 << 33):
        m = (base // d) * d
        pts |= {m, m + d // 2, m + (d + 1) // 2, m - d // 2}
    pts = {p for p in pts if lo <= p <= hi}
    logs = np.exp(rng.uniform(math.log(lo), math.log(max(hi, lo + 1)), size=samples)).astype(np.int64)
    top = rng.integers(max(lo, hi - max(hi // 64, 1)), hi + 1, size=samples, dtype=np.int64)
    pts_arr = np.concatenate([np.array(sorted(pts), dtype=np.int64), np.clip(logs, lo, hi), top])
    got = fixed.div_const_rne(pts_arr, d)
    err = np.abs(got - exact_rne_div_array(pts_arr, d))
    i = int(np.argmax(err))
    R = {7: fixed.INV7_Q32, 127: fixed.INV127_Q32}.get(d, int(round((1 << 32) / d)))
    delta = abs(R - (1 << 32) / d)
    return {'d': d, 'range_hi': int(hi), 'samples': int(pts_arr.size), 'max_err': int(err[i]), 'max_err_at_n': int(pts_arr[i]),
            'frac_wrong': float(np.mean(err > 0)), 'recip_q32': int(R), 'recip_delta': delta,
            'first_order_err_at_hi': max(0.0, hi * delta / (1 << 32) - 1.0)}


def narrow_profile(d, samples, rng, exhaustive_to=1 << 22):
    """Narrow path |n| <= 2^31 vs exact RNE: exhaustive on [-exhaustive_to, exhaustive_to], random + boundaries above."""
    ex = np.arange(-exhaustive_to, exhaustive_to + 1, dtype=np.int64)
    rnd = rng.integers(-NARROW_LIMIT, NARROW_LIMIT + 1, size=samples, dtype=np.int64)
    edge = np.array([NARROW_LIMIT, -NARROW_LIMIT, NARROW_LIMIT - 1, 1 - NARROW_LIMIT], dtype=np.int64)
    m = (NARROW_LIMIT // d) * d
    tie = np.array([m, m - d, m + d // 2, m - d // 2, m + (d + 1) // 2], dtype=np.int64)
    allv = np.concatenate([ex, rnd, edge, tie[np.abs(tie) <= NARROW_LIMIT]])
    bad = int(np.sum(fixed.div_const_rne(allv, d) != exact_rne_div_array(allv, d)))
    return {'d': d, 'values': int(allv.size), 'exhaustive_abs_n': exhaustive_to, 'mismatches': bad}


# ---- characterized domain (the contract the tests assert) ---------------------------------------------------
# Per call site: the largest |numerator| the error characterization covers. Derivations are in
# docs/numeric_domain.md; the observed maxima over eval windows must stay at or below these.
RMS_SS_HI = 1 << 40                      # RMSNorm sum of squares: ~30x above the largest observed (see docs)
ATTN_VALUE_HI = (16 * 32767 * 127 * (1 << 31)) >> 15


def _score_div_lines():
    import inspect
    src, first = inspect.getsourcelines(rr.RunQIntegerLlama._score)
    return [first + i for i, t in enumerate(src) if 'div127(score)' in t]


def site_hi(site, d, kv_bits):
    """(characterized numerator bound, theoretical-basis text) for a call site; model-independent parts only."""
    name, line = site.split(':')[0], int(site.split(':')[1])
    qk = 127 if kv_bits == 8 else 7
    if name == '_rms_high':
        return RMS_SS_HI, f'ss = sum of D={d} int32 squares: operand-width bound D*2^62 exceeds int64 (np.sum wraps); contract ss <= 2^40'
    if name == '_to_base':
        return d * (1 << 31), f'|code| <= qmax={d} times block peak <= 2^31'
    if name == '_attention_value':
        return ATTN_VALUE_HI, 'accum <= 16*32767*127 (asserted) times v peak <= 2^31, >> 15'
    if name == '_score':
        if line == min(_score_div_lines()):
            return 127 * qk * (1 << 31) + 127 * qk, (f'product of two peaks and accum may wrap int64; any score that passes the int32 '
                                                    f'OverflowError check has |n| <= 127*{qk}*2^31')
        return qk * (1 << 31) + qk, f'second stage; any score passing the int32 check has |n| <= {qk}*2^31'
    if name == '_linear_high':
        return None, 'per-group accum <= 64*127*wmax times activation peak <= 2^31 times row multiplier >> (shift-16); model specific'
    return None, ''


def linear_hi(rt):
    """Operand-width bound for _linear_high's div127 numerator, from the model's actual multipliers and shifts."""
    best = 0
    recs = [r for layer in rt.layers for r in layer.values() if isinstance(r, dict) and 'weight' in r] + [rt.classifier]
    for r in recs:
        small = ((r['multiplier'].astype(np.int64) + (1 << 15)) >> 16).reshape(-1).tolist()
        sh = (r['shift'].astype(np.int64) - 16).reshape(-1).tolist()
        acc = 64 * 127 * int(r['qmax']) * (1 << 31)
        best = max(best, max((acc * m) >> max(s, 0) for m, s in zip(small, sh)))
    return best


def characterize(a):
    rng = np.random.default_rng(1001)
    t_all = time.time()
    models = [('260k', 8, a.prompts, a.length), ('260k', 4, a.prompts, a.length)]
    if not a.skip_15m:
        models += [('15m', 8, a.prompts_15m, a.length_15m), ('15m', 4, a.prompts_15m, a.length_15m)]
    ops = {}
    runs, sites, quant = [], {}, {'groups': 0, 'min_raw': 0, 'max_raw': 0, 'int32_min_hits': 0}
    for model, kv, count, length in models:
        p, info, exp, rt = probe_model(model, kv, count, length, 8)
        runs.append(info)
        lin = linear_hi(rt)
        for (site, d), r in p.sites.items():
            hi, why = site_hi(site, d, kv)
            if hi is None:
                hi = lin
            sites[f'{model}/kv{kv}/{site}/d{d}'] = {**r, 'model': model, 'kv_bits': kv, 'site': site, 'd': d, 'characterized_hi': int(hi),
                                                  'within_characterized': r['max_abs_n'] <= hi, 'basis': why}
        for k, v in p.ops.items():
            ops[k] = max(ops.get(k, 0), v)
        q = p.quant
        quant['groups'] += q['groups']
        quant['min_raw'], quant['max_raw'] = min(quant['min_raw'], q['min_raw']), max(quant['max_raw'], q['max_raw'])
        quant['int32_min_hits'] += q['int32_min_hits']
    profiles = {}
    for r in sites.values():
        if r['characterized_hi'] > NARROW_LIMIT:
            key = f"d{r['d']}/hi_2^{math.log2(r['characterized_hi']):.2f}"
            if key not in profiles:
                profiles[key] = error_profile(r['d'], r['characterized_hi'], a.samples, rng)
            r['profile'] = key
            r['error_bound'] = profiles[key]['max_err']
    narrow = [narrow_profile(d, a.samples, rng) for d in sorted({r['d'] for r in sites.values()})]
    return {'runs': runs, 'sites': sites, 'error_profiles': profiles, 'narrow_profiles': narrow, 'quantize_domain': quant, 'op_output_max_abs': ops,
            'rms_ss_hi': RMS_SS_HI, 'seconds': round(time.time() - t_all, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompts', type=int, default=32)
    ap.add_argument('--length', type=int, default=64)
    ap.add_argument('--prompts-15m', type=int, default=12)
    ap.add_argument('--length-15m', type=int, default=48)
    ap.add_argument('--samples', type=int, default=200_000)
    ap.add_argument('--skip-15m', action='store_true')
    ap.add_argument('--output', default=str(BUILD / 'numeric_domain.json'))
    a = ap.parse_args()
    out = characterize(a)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=2, default=str) + '\n')
    for k, r in out['sites'].items():
        print(f"{k:48s} max|n|=2^{math.log2(max(r['max_abs_n'], 1)):5.1f} wide={r['wide_values']:6d} obs_err={r['max_err']} "
              f"hi=2^{math.log2(max(r['characterized_hi'], 1)):5.1f} ok={r['within_characterized']}")
    for k, r in out['error_profiles'].items():
        print(k, 'max_err', r['max_err'], 'at', r['max_err_at_n'], 'first-order', round(r['first_order_err_at_hi'], 2))
    print('ops', {k: f'2^{math.log2(max(v, 1)):.1f}' for k, v in out['op_output_max_abs'].items()})
    print('narrow', [(n['d'], n['mismatches']) for n in out['narrow_profiles']], 'quant', out['quantize_domain'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
