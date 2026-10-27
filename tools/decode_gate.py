"""R3 decode-engine gate: every unit and the end-to-end engine are bit-exact in RTL simulation.

Exits non-zero unless all of the following hold (artifacts under bench/results/):
  * each unit artifact: status pass, 0 mismatches, mutation build caught on every expected element;
  * fault injection: every fault detected, or carries a written equivalence argument;
  * OOC place-and-route on xc7s50 at 50 MHz: timing met for every recorded block;
  * top_decode_15m (stories15M parameters, MAX_SEQ 128, KV4): timing met, <= 68 BRAM36, <= 85% LUT;
  * 15M-shape unit coverage (attention HS=48/KV4/pos<=127, GEMV K=288/768) passes with mutation caught;
  * end-to-end top_decode runs (KV8 and KV4, throttled flash) reproduce every logit and token;
  * M4 QSPI: reader-vs-flash-model unit test over all configurations with its nibble-swap mutation caught, qspi fault set,
    qspi_reader OOC (0 BRAM) and top_decode_15m + reader OOC (same BRAM/LUT/timing limits), and the e2e run through
    qspi_reader + the flash model with 0 logit/token mismatches.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'bench/results'
UNITS = ['decode_gemv_rtl', 'decode_act_quant_rtl', 'decode_rms_rtl', 'decode_vec_alu_rtl', 'decode_attention_rtl',
         'decode_gemv15_rtl', 'decode_attention15_rtl',
         'decode_qspi_rtl']   # last two: stories15M shapes (K=288/768; HS=48, KV4, positions to 127)
FAULT_UNITS = ['gemv', 'actq', 'rms', 'vec', 'attn', 'attn15', 'top', 'qspi']
SYNTH = ['gemv_unit', 'act_quant', 'rmsnorm', 'vec_alu', 'attention', 'top_decode_260k', 'top_decode_15m',
         'qspi_reader', 'top_decode_15m_qspi']   # M4: QSPI flash reader alone and 15M top_decode + reader
BRAM36_LIMIT_15M = 68        # leaves room for the board top + QSPI + UART on the 75-BRAM36 xc7s50
LUT_LIMIT_15M = 0.85 * 32600
E2E = ['decode_e2e_rtl_kv8_p1', 'decode_e2e_rtl_kv4_p1', 'decode_e2e_rtl_kv8_p2', 'decode_e2e_rtl_kv8_p4', 'decode_e2e_rtl_15m_kv4_p1',
       'decode_e2e_rtl_kv8_p1_qspi']   # M4: top_decode + qspi_reader + independent flash model


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def main():
    unmet, report = [], {}
    for u in UNITS:
        o = load(u)
        if o is None:
            unmet.append(f'missing {u}'); continue
        ok = o.get('status') == 'pass' and o.get('mismatch_count') == 0 and o.get('mutation_mismatch_count') == o.get('mutation_expected')
        if u == 'decode_qspi_rtl':
            ok = ok and bool(o.get('variants')) and all(v.get('ok') for v in o['variants'])
        report[u] = 'pass' if ok else 'FAIL'
        if not ok:
            unmet.append(f'{u} not pass')
    faults = load('decode_fault_injection') or {}
    for u in FAULT_UNITS:
        if u not in faults:
            unmet.append(f'fault injection missing for {u}'); continue
        bad = [f for f, r in faults[u].items() if not r.get('ok')]
        report[f'faults_{u}'] = f"{sum(r['detected'] for r in faults[u].values())} detected, " \
                                f"{sum('equivalent_reason' in r for r in faults[u].values())} equivalent"
        if bad:
            unmet.append(f'fault injection {u}: {bad}')
    synth = load('decode_synth_ooc') or {}
    for s in SYNTH:
        r = synth.get(s)
        if r is None:
            unmet.append(f'synth missing {s}'); continue
        report[f'synth_{s}'] = f"WNS {r['wns_ns']:+.3f} ns, {int(r['lut'])} LUT, {int(r['dsp'])} DSP, {r['bram36_equiv']} BRAM36"
        if not r['timing_met']:
            unmet.append(f'synth {s} timing not met ({r["wns_ns"]} ns)')
        if s == 'qspi_reader' and r['bram36_equiv'] != 0:
            unmet.append(f'qspi_reader uses {r["bram36_equiv"]} BRAM36 (must be 0)')
        if s in ('top_decode_15m', 'top_decode_15m_qspi'):
            if r['bram36_equiv'] > BRAM36_LIMIT_15M:
                unmet.append(f'synth {s} BRAM36 {r["bram36_equiv"]} > {BRAM36_LIMIT_15M}')
            if r['lut'] > LUT_LIMIT_15M:
                unmet.append(f'synth {s} LUT {int(r["lut"])} > {int(LUT_LIMIT_15M)} (85%)')
    for e in E2E:
        o = load(e)
        if o is None:
            unmet.append(f'missing {e}'); continue
        ok = o['status'] == 'pass' and o['logit_mismatches'] == 0 and o['token_mismatches'] == 0
        if e.endswith('_qspi') and o.get('flash') != 'qspi':
            ok = False
        report[e] = f"{'pass' if ok else 'FAIL'}: {o['steps']} steps, {o['logits_compared']} logits, " \
                    f"{o['cycles_per_token_mean']:.0f} cycles/token at {o['flash_bytes_per_cycle']} B/cycle"
        if not ok:
            unmet.append(f'{e} not pass')
    print(json.dumps({'exit_code': int(bool(unmet)), 'report': report, 'unmet': unmet}, indent=2))
    sys.exit(1 if unmet else 0)


if __name__ == '__main__':
    main()
