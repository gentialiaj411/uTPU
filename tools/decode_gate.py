"""R3 decode-engine gate: every unit and the end-to-end engine are bit-exact in RTL simulation.

Exits non-zero unless all of the following hold (artifacts under bench/results/, or --results-dir):
  * each unit artifact: status pass, 0 mismatches, mutation build caught on every expected element;
  * fault injection: every fault detected, or carries a written equivalence argument;
  * OOC place-and-route on xc7s50 at 50 MHz: timing met for every recorded block;
  * top_decode_15m (stories15M parameters, MAX_SEQ 128, KV4): timing met, <= 68 BRAM36, <= 85% LUT;
  * 15M-shape unit coverage (attention HS=48/KV4/pos<=127, GEMV K=288/768) passes with mutation caught;
  * end-to-end top_decode runs (KV8 and KV4, throttled flash) reproduce every logit and token;
  * M4 QSPI: reader-vs-flash-model unit test over all configurations with its nibble-swap mutation caught, qspi fault set,
    qspi_reader OOC (0 BRAM) and top_decode_15m + reader OOC (same BRAM/LUT/timing limits), and the e2e run through
    qspi_reader + the flash model with 0 logit/token mismatches;
  * M6: every QSPI IO path (CS#, DQ in/out, setup and hold) has slack >= 0 under the flash AC-parameter constraints, and the
    board_ctrl/UART fault unit is detected or proven equivalent;
  * DDR (EDh) plumbing: decode_qspi_ddr_rtl (run_qspi_rtl.py --ddr: every DDR variant ok, DDR negatives detected, MEASURED streaming
    rate >= 0.49 byte/core cycle), its fault unit 'qspi_ddr' (never passes while fault_inject.FAULTS['qspi_ddr'] is empty), the DDR e2e
    runs (260K and 15M through the DDR reader, flash 'qspi-ddr', 0.5 B/cycle), qspi_reader_ddr / top_decode_15m_qspi_ddr OOC, the UART
    run against a DDR build (V=2, flash switched from LC=00 to LC=11 by the host procedure, wrong LC caught) and boolean_top_impl_ddr;
  * integrity (audit F02/F03): every artifact carries a `provenance` block (sha256 of each RTL/TB/golden/LUT/model source it
    depended on, tools/decode/provenance.py) that is recomputed here ("stale <artifact>: <file> changed"); an artifact without
    one fails ("no provenance"); results must be non-vacuous (steps > 0, logits_compared == steps*VOCAB,
    index_order_ok, element counts > 0, every fault unit holds exactly the names in fault_inject.FAULTS, each ok);
  * M5 board top: UART-driven simulation of boolean_core + flash model (tokens, probed logits, flash Quad-enable procedure,
    mutation caught, QUAD=0 read trips the flash model), boolean_pin_check all pins agree, and the full implementation of
    boolean_top (xc7s50, constraints/boolean_xc7s50.xdc, no bitstream): timing met, <= 68 BRAM36, <= 85% LUT, 0 DRC errors.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'bench/results'
sys.path.insert(0, str(ROOT / 'tools/decode'))
import provenance as prov_mod                           # noqa: E402  (stdlib only; one source list per artifact type)
from provenance import ARTIFACT_SOURCES, FAULT_SOURCES, SYNTH_SOURCES, e2e_sources   # noqa: E402
import fault_inject                                     # noqa: E402  (FAULTS is the single source of expected fault names)

UNITS = ['decode_gemv_rtl', 'decode_act_quant_rtl', 'decode_rms_rtl', 'decode_vec_alu_rtl', 'decode_attention_rtl',
         'decode_gemv15_rtl', 'decode_attention15_rtl',
         'decode_qspi_rtl',   # last two: stories15M shapes (K=288/768; HS=48, KV4, positions to 127)
         'decode_qspi_ddr_rtl']   # EDh DDR reader (run_qspi_rtl.py --ddr): same strictness plus the measured streaming rate
FAULT_UNITS = ['gemv', 'actq', 'rms', 'vec', 'attn', 'attn15', 'top', 'qspi', 'board', 'qspi_ddr', 'board_ddr']
EXPECTED_FAULTS = {u: set(fault_inject.FAULTS[u][2]) for u in FAULT_UNITS}   # import-time snapshot; run_gate recomputes it so it cannot drift
DDR_MIN_BYTES_PER_CYCLE = 0.49          # EDh at the 25 MHz SCK delivers 0.5 byte per 50 MHz core cycle; measured in decode_qspi_ddr_rtl
# unit artifact -> numeric fields whose sum (output elements / bytes compared) must be > 0
UNIT_COUNTS = {'decode_gemv_rtl': ['real_output_elements', 'random_output_elements'],
               'decode_act_quant_rtl': ['real_elements', 'random_directed_elements'],
               'decode_rms_rtl': ['elements'], 'decode_vec_alu_rtl': ['beats'],
               'decode_attention_rtl': ['output_elements'],
               'decode_gemv15_rtl': ['real_output_elements', 'random_output_elements'],
               'decode_attention15_rtl': ['output_elements'], 'decode_qspi_rtl': ['bytes_checked'],
               'decode_qspi_ddr_rtl': ['bytes_checked']}
VOCAB = {'stories260K': 512, 'stories15M': 32000}
SYNTH = ['gemv_unit', 'act_quant', 'rmsnorm', 'vec_alu', 'attention', 'top_decode_260k', 'top_decode_15m',
         'qspi_reader', 'top_decode_15m_qspi',   # M4: QSPI flash reader alone and 15M top_decode + reader
         'qspi_reader_ddr', 'top_decode_15m_qspi_ddr']   # the same two with the DDR=1 (EDh) reader
BRAM36_LIMIT_15M = 68        # leaves room for the board top + QSPI + UART on the 75-BRAM36 xc7s50
LUT_LIMIT_15M = 0.85 * 32600
E2E = ['decode_e2e_rtl_kv8_p1', 'decode_e2e_rtl_kv4_p1', 'decode_e2e_rtl_kv8_p2', 'decode_e2e_rtl_kv8_p4', 'decode_e2e_rtl_15m_kv4_p1',
       'decode_e2e_rtl_kv8_p1_qspi',   # M4: top_decode + qspi_reader + independent flash model
       'decode_e2e_rtl_kv8_p1_qspiddr', 'decode_e2e_rtl_15m_kv4_p1_qspiddr']   # the same through the DDR=1 reader (flash model CR1 LC=11), 260K and 15M


def run_gate(res_dir=RES):
    """Returns (unmet, report). `res_dir` holds the artifact JSONs."""
    unmet, report = [], {}

    def load(name):
        p = Path(res_dir) / f'{name}.json'
        return json.loads(p.read_text()) if p.exists() else None

    def check_prov(name, pv, patterns=None):
        """Provenance must exist, cover the artifact type's sources, and match the tree."""
        if not pv or not [k for k in pv if not k.startswith('tool:')]:
            unmet.append(f'no provenance in {name}')
            return
        unmet.extend(prov_mod.stale(pv, name))
        if patterns is not None:
            unmet.extend(prov_mod.missing_coverage(pv, patterns, name))

    for u in UNITS:
        o = load(u)
        if o is None:
            unmet.append(f'missing {u}'); continue
        ok = o.get('status') == 'pass' and o.get('mismatch_count') == 0 and o.get('mutation_mismatch_count') == o.get('mutation_expected')
        ok = ok and (o.get('mutation_expected') or 0) > 0 and sum(o.get(k) or 0 for k in UNIT_COUNTS[u]) > 0   # non-vacuous
        if u in ('decode_qspi_rtl', 'decode_qspi_ddr_rtl'):
            import run_qspi_rtl                       # lazy: pulls numpy
            expected_variants = run_qspi_rtl.DDR_VARIANTS if u == 'decode_qspi_ddr_rtl' else run_qspi_rtl.VARIANTS
            ok = ok and bool(o.get('variants')) and all(v.get('ok') for v in o['variants']) \
                and {v.get('variant') for v in o['variants']} == {v['name'] for v in expected_variants}
        if u == 'decode_qspi_ddr_rtl':
            import run_qspi_rtl
            bpc = o.get('measured_bytes_per_cycle')
            board = [v for v in o.get('variants', []) if v.get('variant') == run_qspi_rtl.BOARD_DDR_VARIANT]
            ok = ok and o.get('ddr') == 1 and isinstance(bpc, (int, float)) and bpc >= DDR_MIN_BYTES_PER_CYCLE \
                and len(board) == 1 and board[0].get('stream_bytes', 0) >= run_qspi_rtl.STREAM_BYTES \
                and all(v.get('flash_lc') == run_qspi_rtl.DDR_CR1_LC for v in o.get('variants', [])) \
                and len(o.get('negative_tests') or []) == len(run_qspi_rtl.DDR_NEGATIVES) and all(n.get('detected') for n in o['negative_tests']) \
                and bool(o.get('rejected_parameters')) and all(x.get('ok') for x in o['rejected_parameters'])
            report['decode_qspi_ddr_rtl_bytes_per_cycle'] = bpc
            if not (isinstance(bpc, (int, float)) and bpc >= DDR_MIN_BYTES_PER_CYCLE):
                unmet.append(f'decode_qspi_ddr_rtl measured bytes/cycle {bpc} < {DDR_MIN_BYTES_PER_CYCLE}')
        check_prov(u, o.get('provenance'), ARTIFACT_SOURCES[u])
        report[u] = 'pass' if ok else 'FAIL'
        if not ok:
            unmet.append(f'{u} not pass')
    faults = load('decode_fault_injection') or {}
    expected_faults = {u: set(fault_inject.FAULTS[u][2]) for u in FAULT_UNITS}   # fresh: FAULTS is the single source of expected names
    for u in FAULT_UNITS:
        if not expected_faults[u]:
            unmet.append(f'fault injection {u}: no faults defined in fault_inject.FAULTS (placeholder unit: the fault list is still empty)')
        if u not in faults:
            unmet.append(f'fault injection missing for {u}'); continue
        if not faults[u]:
            unmet.append(f'fault injection {u}: result is empty (a unit with no recorded faults never passes)')
        names = set(faults[u])
        if names != expected_faults[u]:
            unmet.append(f'fault injection {u}: missing {sorted(expected_faults[u] - names)} unexpected {sorted(names - expected_faults[u])}')
        bad = [f for f, r in faults[u].items() if not r.get('ok') or 'error' in r]
        check_prov(f'decode_fault_injection[{u}]', (faults.get('provenance') or {}).get(u), FAULT_SOURCES)
        report[f'faults_{u}'] = f"{sum(bool(r.get('detected')) for r in faults[u].values())} detected, " \
                                f"{sum('equivalent_reason' in r for r in faults[u].values())} equivalent"
        if bad:
            unmet.append(f'fault injection {u}: {bad}')
    synth = load('decode_synth_ooc') or {}
    for s in SYNTH:
        r = synth.get(s)
        if r is None:
            unmet.append(f'synth missing {s}'); continue
        check_prov(f'decode_synth_ooc[{s}]', r.get('provenance'), SYNTH_SOURCES[s])
        report[f'synth_{s}'] = f"WNS {r['wns_ns']:+.3f} ns, {int(r['lut'])} LUT, {int(r['dsp'])} DSP, {r['bram36_equiv']} BRAM36"
        if not r['timing_met']:
            unmet.append(f'synth {s} timing not met ({r["wns_ns"]} ns)')
        if s in ('qspi_reader', 'qspi_reader_ddr') and r['bram36_equiv'] != 0:
            unmet.append(f'{s} uses {r["bram36_equiv"]} BRAM36 (must be 0)')
        if s == 'qspi_reader_ddr' and (r.get('generics') or {}).get('DDR') != '1':
            unmet.append(f'synth {s} was not run with -GDDR=1 (generics {r.get("generics")})')
        if s in ('top_decode_15m', 'top_decode_15m_qspi', 'top_decode_15m_qspi_ddr'):
            if r['bram36_equiv'] > BRAM36_LIMIT_15M:
                unmet.append(f'synth {s} BRAM36 {r["bram36_equiv"]} > {BRAM36_LIMIT_15M}')
            if r['lut'] > LUT_LIMIT_15M:
                unmet.append(f'synth {s} LUT {int(r["lut"])} > {int(LUT_LIMIT_15M)} (85%)')
    for e in E2E:
        o = load(e)
        if o is None:
            unmet.append(f'missing {e}'); continue
        ok = o['status'] == 'pass' and o['logit_mismatches'] == 0 and o['token_mismatches'] == 0
        # non-vacuity: the step count follows from the prompts, the logit count from the model's vocabulary
        n_steps = sum(len(p) + o.get('generated_tokens_per_prompt', 0) - 1 for p in o.get('prompts', []))
        vocab = VOCAB.get(o.get('model'))
        ok = ok and o['steps'] > 0 and o['steps'] == n_steps and vocab is not None and o['logits_compared'] == o['steps'] * vocab
        ok = ok and o.get('index_order_ok') is True and o.get('index_errors') == 0 and o.get('argmax_errors') == 0
        if e.endswith('_qspi') and o.get('flash') != 'qspi':
            ok = False
        if e.endswith('_qspiddr') and (o.get('flash') != 'qspi-ddr' or o.get('flash_bytes_per_cycle') != 0.5):
            ok = False
        check_prov(e, o.get('provenance'), e2e_sources(o.get('model', 'stories260K'), o.get('flash', 'ideal')))
        report[e] = f"{'pass' if ok else 'FAIL'}: {o['steps']} steps, {o['logits_compared']} logits, " \
                    f"{o['cycles_per_token_mean']:.0f} cycles/token at {o['flash_bytes_per_cycle']} B/cycle"
        if not ok:
            unmet.append(f'{e} not pass')
    # M5 board top: UART path in simulation, pin verification, full implementation of boolean_top
    for un in ('boolean_uart_rtl', 'boolean_uart_ddr_rtl'):
        u = load(un)
        if u is None:
            unmet.append(f'missing {un}')
            continue
        ok = (u.get('status') == 'pass' and u.get('mismatch_count') == 0 and u.get('mutation_mismatch_count', 0) > 0
              and u.get('negative_quad0_read_trips_flash_model') is True and u.get('quad_enable_via_uart', {}).get('wrr_commands_applied', 0) >= 1)
        if un == 'boolean_uart_ddr_rtl':   # DDR build: V = 2, flash powered up at LC=00 and switched to {LC=11, QUAD=1} by the host procedure, wrong LC caught
            ok = ok and u.get('ddr') == 1 and u.get('build_version') == 2 and u.get('flash_model_cr1_powerup') == 0 \
                and u.get('flash_cr1_after_host_procedure') == 0xC2 and u.get('negative_lc00_under_ddr_build_detected') is True
        check_prov(un, u.get('provenance'), ARTIFACT_SOURCES[un])
        report[un] = f"{'pass' if ok else 'FAIL'}: {u.get('steps')} steps, {u.get('checks')} checks, {u.get('mismatch_count')} mismatches"
        if not ok:
            unmet.append(f'{un} not pass')
    pc = load('boolean_pin_check')
    if pc is None:
        unmet.append('missing boolean_pin_check')
    else:
        check_prov('boolean_pin_check', pc.get('provenance'), ARTIFACT_SOURCES['boolean_pin_check'])
        report['boolean_pin_check'] = f"{len(pc['pins'])} pins, all_agree={pc['all_agree']}"
        if not pc['all_agree'] or pc.get('disagreements'):
            unmet.append(f"boolean pin check disagreements: {pc.get('disagreements')}")
    for bn in ("boolean_top_impl", "boolean_top_impl_ddr"):   # default build and the DDR=1 build (impl_top.py --ddr): identical requirements
        bi = load(bn)
        if bi is None or bi.get('status') != 'ok':
            unmet.append(f'missing or failed {bn}')
        else:
            check_prov(bn, bi.get('provenance'), ARTIFACT_SOURCES[bn])
            if bn.endswith('_ddr') and (bi.get('ddr') != 1 or (bi.get('generics') or {}).get('DDR') != '1'):
                unmet.append(f'{bn} was not implemented with boolean_top DDR=1')
            report[bn] = f"WNS {bi['wns_ns']:+.3f} ns, WHS {bi['whs_ns']:+.3f} ns, {int(bi['lut'])} LUT, {bi['bram36_equiv']} BRAM36, DRC errors {len(bi['drc_error_rules'])}"
            if not bi['timing_met']:
                unmet.append(f'{bn} timing not met')
            if bi['bram36_equiv'] > BRAM36_LIMIT_15M:
                unmet.append(f"{bn} BRAM36 {bi['bram36_equiv']} > {BRAM36_LIMIT_15M}")
            if bi['lut'] > LUT_LIMIT_15M:
                unmet.append(f"{bn} LUT {int(bi['lut'])} > {int(LUT_LIMIT_15M)} (85%)")
            if bi['drc_error_rules']:
                unmet.append(f"{bn} DRC errors {bi['drc_error_rules']}")
            # M6: every QSPI IO path (constrained against the flash AC numbers) must have non-negative setup and hold slack
            io = bi.get('qspi_io_slack')
            if not io:
                unmet.append(f'{bn} has no per-path QSPI IO slack')
            else:
                need = [('qspi_cs_n', 'out_setup'), ('qspi_cs_n', 'out_hold')] + [(f'qspi_dq[{i}]', k) for i in range(4) for k in ('in_setup', 'in_hold')] + \
                       [(f'qspi_dq[{i}]', k) for i in (0, 2, 3) for k in ('out_setup', 'out_hold')]
                worst = []
                for port, kind in need:
                    sl = (io.get(port, {}).get(kind) or {}).get('slack_ns')
                    if sl is None:
                        unmet.append(f'{bn} missing QSPI IO slack {port} {kind}')
                    else:
                        worst.append(sl)
                        if sl < 0:
                            unmet.append(f'{bn} QSPI IO slack {port} {kind} = {sl} ns < 0')
                report['boolean_top_qspi_io_min_slack' + ('_ddr' if bn.endswith('_ddr') else '')] = f"{min(worst):+.3f} ns over {len(worst)} paths" if worst else 'none'

    return unmet, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=str(RES), help='directory holding the artifact JSONs (default bench/results)')
    a = ap.parse_args()
    unmet, report = run_gate(Path(a.results_dir))
    print(json.dumps({'exit_code': int(bool(unmet)), 'report': report, 'unmet': unmet}, indent=2))
    sys.exit(1 if unmet else 0)


if __name__ == '__main__':
    main()
