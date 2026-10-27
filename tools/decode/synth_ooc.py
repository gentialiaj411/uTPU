"""Out-of-context Vivado place+route of one decode unit on xc7s50 at 50 MHz.

Usage: python tools/decode/synth_ooc.py <name> <top> <src.sv> [<src.sv> ...]
Merges LUT/FF/DSP/BRAM/WNS into bench/results/decode_synth_ooc.json.
"""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIVADO = r'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat'
PART, PERIOD = 'xc7s50csga324-1', 20.0


def main():
    name, top, srcs = sys.argv[1], sys.argv[2], sys.argv[3:]
    out = ROOT / 'build/decode/synth' / name
    out.mkdir(parents=True, exist_ok=True)
    tcl = out / 'run.tcl'
    tcl.write_text('\n'.join([*(f'read_verilog -sv {Path(s).as_posix()}' for s in srcs),
        f'synth_design -top {top} -part {PART} -mode out_of_context',
        f'create_clock -period {PERIOD:.3f} -name clk [get_ports clk]',
        f'report_utilization -hierarchical -hierarchical_depth 3 -file {(out / "synth_util_hier.rpt").as_posix()}',
        f'report_utilization -file {(out / "synth_util.rpt").as_posix()}',
        'opt_design', 'place_design', 'route_design',
        f'report_utilization -hierarchical -hierarchical_depth 3 -file {(out / "util_hier.rpt").as_posix()}',
        f'report_utilization -file {(out / "util.rpt").as_posix()}',
        f'report_timing_summary -file {(out / "timing.rpt").as_posix()}', '']))
    for f in ('util.rpt', 'timing.rpt', 'synth_util.rpt', 'synth_util_hier.rpt', 'util_hier.rpt'):
        (out / f).unlink(missing_ok=True)
    log = out / 'vivado.log'
    for attempt in range(2):   # Vivado 2025.2 opt_design occasionally fails with an empty Synth 20-411
        with log.open('w') as fh:
            subprocess.run(['cmd', '/c', VIVADO, '-mode', 'batch', '-nojournal', '-nolog', '-source', str(tcl)],
                           cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if (out / 'timing.rpt').exists():
            break
    if not (out / 'timing.rpt').exists():
        errs = [l for l in log.read_text(errors='replace').splitlines() if 'ERROR' in l]
        rec = {'part': PART, 'period_ns': PERIOD, 'top': top, 'sources': srcs, 'status': 'failed', 'timing_met': False,
               'errors': errs[:5], 'wns_ns': None, 'lut': 0.0, 'ff': 0.0, 'dsp': 0.0, 'bram36_equiv': 0.0}
        if (out / 'synth_util.rpt').exists():   # a failed place/route still records the post-synthesis numbers
            su = (out / 'synth_util.rpt').read_text()
            for k, label in (('lut', 'Slice LUTs'), ('ff', 'Slice Registers'), ('dsp', 'DSPs'), ('bram36_equiv', 'Block RAM Tile')):
                m = re.search(r'^\|\s*' + re.escape(label) + r'\*?\s*\|\s*([\d.]+)', su, re.M)
                rec[k] = float(m.group(1)) if m else 0.0
            rec['numbers_are'] = 'post-synthesis (place/route failed)'
        res_path = ROOT / 'bench/results/decode_synth_ooc.json'
        results = json.loads(res_path.read_text()) if res_path.exists() else {}
        results[name] = rec
        res_path.write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps({'name': name, 'status': 'failed', 'errors': errs[:5], **{k: rec[k] for k in ('lut', 'ff', 'dsp', 'bram36_equiv')}}))
        sys.exit(1)
    util = (out / 'util.rpt').read_text()
    timing = (out / 'timing.rpt').read_text()

    def cell(label):
        m = re.search(r'^\|\s*' + re.escape(label) + r'\*?\s*\|\s*([\d.]+)', util, re.M)
        return float(m.group(1)) if m else 0.0

    wns = float(re.search(r'WNS\(ns\).*?\n.*?\n\s*(-?[\d.]+)', timing, re.S).group(1))
    rec = {'part': PART, 'period_ns': PERIOD, 'top': top, 'sources': srcs,
           'lut': cell('Slice LUTs'), 'ff': cell('Slice Registers'), 'dsp': cell('DSPs'),
           'bram36_equiv': cell('Block RAM Tile'), 'wns_ns': wns, 'timing_met': wns >= 0}
    res_path = ROOT / 'bench/results/decode_synth_ooc.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    results[name] = rec
    res_path.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({'name': name, **{k: rec[k] for k in ('lut', 'ff', 'dsp', 'bram36_equiv', 'wns_ns')}}))


if __name__ == '__main__':
    main()
