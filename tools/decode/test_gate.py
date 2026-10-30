"""Self-test for tools/decode_gate.py: a valid synthetic artifact set passes; each integrity failure mode fails.

Provenance blocks are built from the real tree (needs the generated LUTs and the models/ directory, as the runners do);
the test skips if those inputs are absent.
"""
from __future__ import annotations
import copy, json, shutil, sys
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import decode_gate as gate                                 # noqa: E402
import fault_inject                                        # noqa: E402
import provenance as pv                                    # noqa: E402
import run_qspi_rtl                                        # noqa: E402

TINY_SRC = ['rtl/decode/gemv_unit.sv']


def build_valid(d: Path):
    d.mkdir(parents=True, exist_ok=True)

    def put(name, obj):
        (d / f'{name}.json').write_text(json.dumps(obj))

    for u in gate.UNITS:
        o = {'status': 'pass', 'mismatch_count': 0, 'mutation_mismatch_count': 7, 'mutation_expected': 7,
             'provenance': pv.provenance(pv.ARTIFACT_SOURCES[u])}
        for k in gate.UNIT_COUNTS[u]:
            o[k] = 100
        if u == 'decode_qspi_rtl':
            o['variants'] = [{'variant': v['name'], 'ok': True} for v in run_qspi_rtl.VARIANTS]
        put(u, o)
    faults = {'provenance': {}}
    for unit in gate.FAULT_UNITS:
        faults[unit] = {}
        for name, (_, _, _, *why) in fault_inject.FAULTS[unit][2].items():
            r = {'mismatches': 0 if why else 3, 'real_mismatches': 0 if why else 3, 'detected': not why, 'ok': True}
            if why:
                r['equivalent_reason'] = why[0]
            faults[unit][name] = r
        faults['provenance'][unit] = pv.provenance(pv.FAULT_SOURCES)
    put('decode_fault_injection', faults)
    put('decode_synth_ooc', {s: {'wns_ns': 0.5, 'lut': 100.0, 'dsp': 1.0, 'bram36_equiv': 0.0 if s == 'qspi_reader' else 10.0,
                                 'timing_met': True, 'sources': TINY_SRC,
                                 'provenance': pv.provenance(pv.SYNTH_SOURCES[s])} for s in gate.SYNTH})
    for e in gate.E2E:
        model = 'stories15M' if '15m' in e else 'stories260K'
        flash = 'qspi' if e.endswith('_qspi') else 'ideal'
        steps = 6
        put(e, {'status': 'pass', 'model': model, 'flash': flash, 'prompts': [[1, 2, 3]], 'generated_tokens_per_prompt': 4,
                'steps': steps, 'logits_compared': steps * gate.VOCAB[model], 'vocab': gate.VOCAB[model],
                'logit_mismatches': 0, 'token_mismatches': 0, 'index_errors': 0, 'argmax_errors': 0, 'index_order_ok': True,
                'cycles_per_token_mean': 1000.0, 'flash_bytes_per_cycle': 1.0,
                'provenance': pv.provenance(pv.e2e_sources(model, flash))})
    put('boolean_uart_rtl', {'status': 'pass', 'mismatch_count': 0, 'mutation_mismatch_count': 3, 'steps': 1, 'checks': 1,
                             'negative_quad0_read_trips_flash_model': True, 'quad_enable_via_uart': {'wrr_commands_applied': 1},
                             'provenance': pv.provenance(pv.ARTIFACT_SOURCES['boolean_uart_rtl'])})
    put('boolean_pin_check', {'pins': ['a'], 'all_agree': True, 'provenance': pv.provenance(pv.ARTIFACT_SOURCES['boolean_pin_check'])})
    io = {'qspi_cs_n': {'out_setup': {'slack_ns': 1.0}, 'out_hold': {'slack_ns': 1.0}}}
    for i in range(4):
        io[f'qspi_dq[{i}]'] = {k: {'slack_ns': 1.0} for k in ('in_setup', 'in_hold', 'out_setup', 'out_hold')}
    put('boolean_top_impl', {'status': 'ok', 'wns_ns': 0.1, 'whs_ns': 0.1, 'lut': 1000, 'bram36_equiv': 10, 'drc_error_rules': [],
                             'timing_met': True, 'qspi_io_slack': io,
                             'provenance': pv.provenance(pv.ARTIFACT_SOURCES['boolean_top_impl'])})


@pytest.fixture(scope='module')
def valid_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp('valid')
    try:
        build_valid(d)
    except FileNotFoundError as e:
        pytest.skip(f'inputs for provenance missing: {e}')
    return d


@pytest.fixture()
def work(valid_dir, tmp_path):
    d = tmp_path / 'r'
    shutil.copytree(valid_dir, d)
    return d


def edit(d, name, fn):
    p = d / f'{name}.json'
    o = json.loads(p.read_text())
    fn(o)
    p.write_text(json.dumps(o))


def unmet(d):
    return gate.run_gate(d)[0]


def test_valid_set_passes(work):
    assert unmet(work) == []


def test_empty_steps_fails(work):
    edit(work, 'decode_e2e_rtl_kv8_p1', lambda o: o.update(steps=0, logits_compared=0, prompts=[]))
    assert any('decode_e2e_rtl_kv8_p1 not pass' in m for m in unmet(work))


def test_logit_count_mismatch_fails(work):
    edit(work, 'decode_e2e_rtl_kv8_p1', lambda o: o.update(logits_compared=o['logits_compared'] - 1))
    assert any('decode_e2e_rtl_kv8_p1 not pass' in m for m in unmet(work))


def test_index_order_false_fails(work):
    edit(work, 'decode_e2e_rtl_kv8_p1', lambda o: o.update(index_order_ok=False))
    assert any('decode_e2e_rtl_kv8_p1 not pass' in m for m in unmet(work))


def test_index_order_missing_fails(work):
    edit(work, 'decode_e2e_rtl_kv8_p1', lambda o: o.pop('index_order_ok'))
    assert any('decode_e2e_rtl_kv8_p1 not pass' in m for m in unmet(work))


def test_empty_fault_dict_fails(work):
    edit(work, 'decode_fault_injection', lambda o: o.update(top={}))
    assert any(m.startswith('fault injection top: missing') for m in unmet(work))


def test_missing_fault_name_fails(work):
    edit(work, 'decode_fault_injection', lambda o: o['top'].pop('top_duplicate_idx0'))
    assert any("top_duplicate_idx0" in m for m in unmet(work))


def test_fault_not_ok_fails(work):
    edit(work, 'decode_fault_injection', lambda o: o['top']['top_late_pos'].update(ok=False, detected=False))
    assert any('top_late_pos' in m for m in unmet(work))


def test_fault_error_entry_fails(work):
    edit(work, 'decode_fault_injection', lambda o: o['gemv']['ties_half_up'].update(error='crash'))
    assert any('ties_half_up' in m for m in unmet(work))


def test_unit_zero_elements_fails(work):
    edit(work, 'decode_rms_rtl', lambda o: o.update(elements=0))
    assert any('decode_rms_rtl not pass' in m for m in unmet(work))


def test_stale_hash_fails(work):
    def corrupt(o):
        k = 'rtl/decode/gemv_unit.sv'
        o['provenance'][k] = '0' * 64
    edit(work, 'decode_gemv_rtl', corrupt)
    assert 'stale decode_gemv_rtl: rtl/decode/gemv_unit.sv changed' in unmet(work)


def test_stale_fault_unit_fails(work):
    edit(work, 'decode_fault_injection', lambda o: o['provenance']['gemv'].update({'rtl/decode/gemv_unit.sv': '1' * 64}))
    assert any(m.startswith('stale decode_fault_injection[gemv]: rtl/decode/gemv_unit.sv') for m in unmet(work))


def test_stale_synth_fails(work):
    edit(work, 'decode_synth_ooc', lambda o: o['qspi_reader']['provenance'].update({TINY_SRC[0]: '2' * 64}))
    assert any(m.startswith('stale decode_synth_ooc[qspi_reader]') for m in unmet(work))


def test_missing_provenance_fails(work):
    edit(work, 'decode_rms_rtl', lambda o: o.pop('provenance'))
    assert 'no provenance in decode_rms_rtl' in unmet(work)


@pytest.mark.parametrize('name', ['boolean_uart_rtl', 'boolean_pin_check', 'boolean_top_impl'])
def test_board_artifacts_need_provenance_coverage_and_freshness(work, name):
    edit(work, name, lambda o: o.pop('provenance'))
    assert f'no provenance in {name}' in unmet(work)
    edit(work, name, lambda o: o.update(provenance={'tools/decode/provenance.py': pv.sha256_file(pv.ROOT / 'tools/decode/provenance.py')}))
    assert any(m.startswith(f'provenance of {name} lacks') for m in unmet(work))
    edit(work, name, lambda o: o.update(provenance={**pv.provenance(pv.ARTIFACT_SOURCES[name]), 'constraints/boolean_xc7s50.xdc': '4' * 64}
                                        if name != 'boolean_uart_rtl' else {**pv.provenance(pv.ARTIFACT_SOURCES[name]), 'rtl/board/board_ctrl.sv': '4' * 64}))
    assert any(m.startswith(f'stale {name}') for m in unmet(work))


def test_synth_entry_must_cover_its_rtl(work):
    edit(work, 'decode_synth_ooc', lambda o: o['top_decode_260k'].update(provenance=pv.provenance(pv.SYNTH_SOURCES['gemv_unit']), sources=pv.SYNTH_SOURCES['gemv_unit']))
    assert any(m.startswith('provenance of decode_synth_ooc[top_decode_260k] lacks') for m in unmet(work))


def test_incomplete_provenance_fails(work):
    edit(work, 'decode_rms_rtl', lambda o: o['provenance'].pop('rtl/decode/rmsnorm.sv'))
    assert any('lacks rtl/decode/rmsnorm.sv' in m for m in unmet(work))


def test_stale_gemv_names_the_file(work):
    edit(work, 'decode_gemv_rtl', lambda o: o['provenance'].update({'rtl/decode/gemv_unit.sv': '3' * 64}))
    assert any(m.startswith('stale decode_gemv_rtl') for m in unmet(work))
@pytest.mark.parametrize('key', ['runtime:python','runtime:numpy'])
def test_runtime_identity_changes_invalidate(work,key):
    edit(work,'decode_gemv_rtl',lambda o:o['provenance'].update({key:'0'*64}))
    assert any(key+' changed' in x for x in unmet(work))

@pytest.mark.parametrize('field',['version','executable_sha256','compiler_sha256'])
def test_simulator_identity_changes_invalidate(work,field):
    edit(work,'decode_gemv_rtl',lambda o:o['provenance']['tool:simulator'].update({field:'changed'}))
    assert any('simulator identity changed' in x for x in unmet(work))

@pytest.mark.parametrize('path', [*pv.M260,*pv.M15,*pv.NUMERICS,'tools/decode/pack_decode_image.py','tools/decode/golden_*.py'])
def test_fault_dependencies_invalidate(work,path):
    expanded=pv.expand([path])
    edit(work,'decode_fault_injection',lambda o:o['provenance']['gemv'].update({expanded[0]:'0'*64}))
    assert any('stale decode_fault_injection[gemv]' in x for x in unmet(work))

@pytest.mark.parametrize('key',['tool:simulator','runtime:python','runtime:numpy'])
def test_new_identity_inputs_required(work,key):
    edit(work,'decode_gemv_rtl',lambda o:o['provenance'].pop(key))
    assert any('lacks '+key in x for x in unmet(work))
@pytest.mark.parametrize('which',['python','numpy'])
def test_actual_runtime_change_invalidates(work,monkeypatch,which):
    if which=='numpy':monkeypatch.setattr(pv.np,'__version__',pv.np.__version__+'+sprint-mutation')
    else:monkeypatch.setattr(pv.sys,'version',pv.sys.version+' sprint-mutation')
    assert any('runtime:'+which+' changed' in x for x in unmet(work))

def test_actual_simulator_digest_change_invalidates(work,monkeypatch):
    original=pv.simulator_identity
    def changed():
        result=dict(original());result['executable_sha256']='changed executable bytes';return result
    monkeypatch.setattr(pv,'simulator_identity',changed)
    assert any('simulator identity changed' in x for x in unmet(work))
