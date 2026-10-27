"""Full implementation (synthesis + place + route, NO write_bitstream) of rtl/board/boolean_top.sv on xc7s50csga324-1
with constraints/boolean_xc7s50.xdc. Records utilization, timing and DRC in bench/results/boolean_top_impl.json.
Also checks that boolean_core.sv's default parameters equal the stories15M image parameters (params.json).
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIVADO = r'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat'
PART = 'xc7s50csga324-1'
SRCS = ['rtl/decode/decode_pkg.sv', 'rtl/decode/seq_mul.sv', 'rtl/decode/mul_pipe.sv', 'rtl/decode/div_const_pipe.sv', 'rtl/decode/gemv_unit.sv',
        'rtl/decode/act_quant.sv', 'rtl/decode/rmsnorm.sv', 'rtl/decode/vec_alu.sv', 'rtl/decode/attention.sv', 'rtl/decode/top_decode.sv',
        'rtl/decode/qspi_reader.sv', 'rtl/board/uart_rx.sv', 'rtl/board/uart_tx.sv', 'rtl/board/board_ctrl.sv', 'rtl/board/boolean_core.sv',
        'rtl/board/qspi_pads.sv', 'rtl/board/boolean_top.sv']
XDC = 'constraints/boolean_xc7s50.xdc'


def check_params():
    """boolean_core defaults vs the 15M image parameters used by the verified build wrapper."""
    pj = json.loads((ROOT / 'build/decode/decode_image_15m/params.json').read_text())
    text = (ROOT / 'rtl/board/boolean_core.sv').read_text()
    bad = []
    for k, v in pj.items():
        m = re.search(r'\b' + re.escape(k) + r'\s*=\s*(-?\d+)', text)
        if m is None or int(m.group(1)) != int(v):
            bad.append((k, v, m.group(1) if m else None))
    return bad


def main():
    bad = check_params()
    if bad:
        print('boolean_core defaults differ from params.json:', bad); sys.exit(2)
    out = ROOT / 'build/decode/synth/boolean_top'
    out.mkdir(parents=True, exist_ok=True)
    rp = lambda n: (out / n).as_posix()
    tcl = out / 'run.tcl'
    tcl.write_text('\n'.join([*(f'read_verilog -sv {Path(s).as_posix()}' for s in SRCS), f'read_xdc {XDC}',
        f'synth_design -top boolean_top -part {PART}', f'report_utilization -file {rp("synth_util.rpt")}',
        'opt_design', 'place_design', 'route_design',
        f'report_utilization -file {rp("util.rpt")}', f'report_utilization -hierarchical -hierarchical_depth 2 -file {rp("util_hier.rpt")}',
        f'report_timing_summary -file {rp("timing.rpt")}', f'report_drc -file {rp("drc.rpt")}', f'report_io -file {rp("io.rpt")}',
        f'report_clock_utilization -file {rp("clock_util.rpt")}', '']))
    reparse = '--reparse' in sys.argv      # re-read the reports of the last run without running Vivado
    log = out / 'vivado.log'
    for f in ([] if reparse else out.glob('*.rpt')):
        f.unlink()
    for attempt in range(0 if reparse else 2):
        with log.open('w') as fh:
            subprocess.run(['cmd', '/c', VIVADO, '-mode', 'batch', '-nojournal', '-nolog', '-source', str(tcl)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if (out / 'timing.rpt').exists():
            break
    res_path = ROOT / 'bench/results/boolean_top_impl.json'
    if not (out / 'timing.rpt').exists():
        errs = [l for l in log.read_text(errors='replace').splitlines() if 'ERROR' in l][:6]
        res_path.write_text(json.dumps({'status': 'failed', 'errors': errs, 'part': PART}, indent=2) + '\n')
        print('FAILED', errs); sys.exit(1)
    util, timing, drc = (out / 'util.rpt').read_text(), (out / 'timing.rpt').read_text(), (out / 'drc.rpt').read_text()

    def cell(label):
        m = re.search(r'^\|\s*' + re.escape(label) + r'\*?\s*\|\s*([\d.]+)', util, re.M)
        return float(m.group(1)) if m else None

    m = re.search(r'WNS\(ns\)\s+TNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)', timing, re.S)
    wns, tns, _, _, whs, ths = (float(x) for x in m.groups()[:6])
    drc_sum = re.search(r'Checks found:\s*(\d+)', drc)
    errs_drc = re.findall(r'^\|\s*(\S+)\s*\|\s*(?:Error|Critical Warning)\s*\|', drc, re.M)
    rec = {'status': 'ok', 'part': PART, 'top': 'boolean_top', 'constraints': XDC, 'bitstream_generated': False, 'sources': SRCS,
           'lut': cell('Slice LUTs'), 'lut_pct': None, 'ff': cell('Slice Registers'), 'dsp': cell('DSPs'), 'bram36_equiv': cell('Block RAM Tile'),
           'wns_ns': wns, 'tns_ns': tns, 'whs_ns': whs, 'ths_ns': ths, 'timing_met': wns >= 0 and whs >= 0 and tns == 0 and ths == 0,
           'drc_checks_found': int(drc_sum.group(1)) if drc_sum else None, 'drc_summary_rules': re.findall(r'^\|\s*(\S+)\s*\|\s*(\w+(?: \w+)?)\s*\|[^|]*\|\s*(\d+)\s*\|', drc, re.M), 'drc_error_rules': errs_drc, 'bitstream_note': 'write_bitstream is not in the Tcl; no bitstream was generated',
           'reports': {n: f'build/decode/synth/boolean_top/{n}' for n in ('util.rpt', 'timing.rpt', 'drc.rpt', 'io.rpt', 'clock_util.rpt')}}
    rec['lut_pct'] = round(100 * rec['lut'] / 32600, 1) if rec['lut'] else None
    res_path.write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps({k: rec[k] for k in ('lut', 'lut_pct', 'ff', 'dsp', 'bram36_equiv', 'wns_ns', 'whs_ns', 'timing_met', 'drc_checks_found', 'drc_error_rules')}))


if __name__ == '__main__':
    main()
