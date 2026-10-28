"""Verify every pin the Boolean top uses against the device database and the board documents.

The constraints file itself is parsed (constraints/boolean_xc7s50.xdc): every port's PACKAGE_PIN and IOSTANDARD in the XDC is
compared with the reviewed pin map below (pin from Real Digital's master constraints file where it lists one, otherwise from the
schematic net read by eye from the rasterized sheets 10, 11 and 13 of the Boolean schematic; the PDF text layer is scrambled, so the
'schematic' fields are transcriptions, not machine output) and with the device's own PIN_FUNC / BANK from the installed Vivado
(queried live with link_design -part xc7s50csga324-1). A port agrees when XDC pin == reviewed pin, XDC IOSTANDARD == the standard the
bank VCCO needs, and the device function matches what the signal needs. The XDC must constrain exactly the reviewed ports (no
missing, no extra). If an implementation has been run (tools/board/impl_top.py writes build/decode/synth/boolean_top/ports_placed.txt),
the placed PACKAGE_PIN / IOSTANDARD of every port of the routed design is compared too. The schematic file's sha256 and revision are
recorded. Writes bench/results/boolean_pin_check.json; exits non-zero if anything disagrees.
"""
from __future__ import annotations
import hashlib, json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIVADO = r'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat'
PART = 'xc7s50csga324-1'
SCH = 'Boolean schematic KT-000-001-001-001 V2I1 (realdigital.org/downloads/63f9a8205ebd9c2e8c2d265ad25097dc.pdf)'
SCH_REVISION = 'KT-000-001-001-001 V2I1 (revision string as transcribed by the reviewer from the title block; the PDF text layer is scrambled)'
SCH_CANDIDATES = [os.environ.get('BOOLEAN_SCHEMATIC', ''), str(ROOT / 'docs/boolean_schematic.pdf'),
                  str(ROOT.parent / 'uTPU-audit-scratch/boolean.pdf'), r'C:\Users\bhask\Documents\PROJECTS\uTPU-audit-scratch\boolean.pdf']
XDC = 'Real Digital Boolean master constraints (realdigital.org/downloads/8d5c167add28c014173edcf51db78bb9.txt)'
XDC_PATH = ROOT / 'constraints/boolean_xc7s50.xdc'
PLACED = ROOT / 'build/decode/synth/boolean_top/ports_placed.txt'
IOSTD_FOR_VCCO = {3.3: 'LVCMOS33'}

# signal, pin, device PIN_FUNC substring required, xdc port (None: no port, driven through STARTUPE2), schematic net/sheet, bank VCCO per schematic sheet 13
USED = [
    ('clk (100 MHz)',  'F14', 'MRCC_15',   'clk',       ('UCLK, driven by IC9 ASEM1-100.000MHZ', 11),        3.3),
    ('uart_rxd (FPGA RX)', 'V12', '_14',   'UART_rxd',  ('UART_TXD_IN (USB-UART bridge TX into the FPGA)', 11), 3.3),
    ('uart_txd (FPGA TX)', 'U11', '_14',   'UART_txd',  ('UART_RXD_OUT (FPGA TX to the USB-UART bridge)', 11), 3.3),
    ('qspi_cs_n',      'M13', 'FCS_B_14',  'qspi_cs_n', ('QSPI_CS to IC8 pin 7 (CS)', 11),                   3.3),
    ('qspi_dq[0]',     'K17', 'D00_MOSI_14', 'qspi_dq[0]', ('QSPI_DQ0 to IC8 pin 15 (SDI/DQ0)', 11),        3.3),
    ('qspi_dq[1]',     'K18', 'D01_DIN_14', 'qspi_dq[1]', ('QSPI_DQ1 to IC8 pin 8 (SDO/DQ1)', 11),          3.3),
    ('qspi_dq[2]',     'L14', 'D02_14',    'qspi_dq[2]', ('QSPI_DQ2 to IC8 pin 9 (WP#/DQ2), 1.8k pull-up', 11), 3.3),
    ('qspi_dq[3]',     'M15', 'D03_14',    'qspi_dq[3]', ('QSPI_DQ3 to IC8 pin 1 (HOLD#/DQ3), 1.8k pull-up', 11), 3.3),
    ('flash SCLK (CCLK_0, via STARTUPE2, no port)', 'C8', 'CCLK_0', None, ('QSPI_SCK to IC8 pin 16 (SCK)', 10), 3.3),
]
# pins another source (the coordinator's recollection of the Arty S7 pinout) proposed; checked and rejected
RECALLED = [('qspi_cs_n', 'L13', 'FCS_B'), ('qspi_dq[3]', 'M14', 'D03')]


def parse_xdc(path):
    """{port: {'pin':..., 'iostandard':...}} from set_property PACKAGE_PIN / IOSTANDARD lines targeting get_ports (comments stripped)."""
    ports = {}
    text = '\n'.join(l.split('#', 1)[0] for l in path.read_text().splitlines())
    for m in re.finditer(r'set_property\s+(-dict\s*\{([^}]*)\}|(\w+)\s+(\S+))\s+\[get_ports\s+(\{[^}]*\}|\S+?)\s*\]', text):
        port = m.group(5).strip('{} ')
        pairs = dict(zip(*[iter(m.group(2).split())] * 2)) if m.group(2) else {m.group(3): m.group(4)}
        for k, v in pairs.items():
            if k in ('PACKAGE_PIN', 'IOSTANDARD'):
                ports.setdefault(port, {})['pin' if k == 'PACKAGE_PIN' else 'iostandard'] = v
    return ports


def query():
    out = ROOT / 'build/decode'
    out.mkdir(parents=True, exist_ok=True)
    raw = out / 'pins_raw.txt'
    tcl = out / 'pins.tcl'
    tcl.write_text(f'link_design -part {PART}\nset fh [open {raw.as_posix()} w]\n'
                   'foreach p [get_package_pins] { puts $fh "$p|[get_property PIN_FUNC $p]|[get_property BANK $p]" }\nclose $fh\n')
    raw.unlink(missing_ok=True)
    subprocess.run(['cmd', '/c', VIVADO, '-mode', 'batch', '-nojournal', '-nolog', '-source', str(tcl)], cwd=ROOT, capture_output=True)
    pins = {}
    for line in raw.read_text().splitlines():
        p, f, b = line.split('|')
        pins[p] = (f, b)
    return pins


def schematic_record():
    for c in SCH_CANDIDATES:
        if c and Path(c).is_file():
            data = Path(c).read_bytes()
            return {'file': Path(c).name, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'revision': SCH_REVISION, 'source_url': SCH}
    return {'file': None, 'sha256': None, 'revision': SCH_REVISION, 'source_url': SCH, 'note': 'schematic PDF not found at any candidate path (set BOOLEAN_SCHEMATIC)'}


def main():
    pins = query()
    xdc = parse_xdc(XDC_PATH)
    placed = {}
    if PLACED.exists():
        for line in PLACED.read_text().splitlines():
            n, pin, std, d = line.split('|')
            placed[n] = {'pin': pin, 'iostandard': std}
    rows, bad = [], []
    reviewed_ports = {u[3] for u in USED if u[3]}
    for extra in sorted(set(xdc) - reviewed_ports):
        bad.append(f'XDC constrains port {extra} which is not in the reviewed pin map')
    for sig, pin, need, xdc_port, (net, sheet), vcco in USED:
        func, bank = pins.get(pin, ('<not in package>', '?'))
        want_std = IOSTD_FOR_VCCO.get(vcco, '?')
        problems = []
        if need not in func:
            problems.append(f'device function {func} lacks {need}')
        xp = xdc.get(xdc_port) if xdc_port else None
        if xdc_port:
            if xp is None or 'pin' not in xp:
                problems.append('port has no PACKAGE_PIN in the XDC')
            else:
                if xp['pin'] != pin: problems.append(f"XDC pin {xp['pin']} != reviewed {pin}")
                if xp.get('iostandard') != want_std: problems.append(f"XDC IOSTANDARD {xp.get('iostandard')} != {want_std}")
            if placed:
                pl = placed.get(xdc_port)
                if pl is None: problems.append('port missing from the placed design')
                else:
                    if pl['pin'] != pin: problems.append(f"placed pin {pl['pin']} != reviewed {pin}")
                    if pl['iostandard'] != want_std: problems.append(f"placed IOSTANDARD {pl['iostandard']} != {want_std}")
        rows.append({'signal': sig, 'pin': pin, 'device_pin_func': func, 'device_bank': bank, 'required_function': need,
                     'xdc_port': xdc_port, 'xdc_pin': xp.get('pin') if xp else None, 'xdc_iostandard': xp.get('iostandard') if xp else None,
                     'placed': placed.get(xdc_port) if xdc_port else None,
                     'master_xdc': (f'{XDC}: {xdc_port} on {pin}' if sig.startswith(('clk', 'uart')) else 'not listed in the master XDC'),
                     'schematic': f'{SCH} sheet {sheet}: {net}', 'schematic_bank_vcco_v': vcco, 'required_iostandard': want_std,
                     'agrees': not problems, 'problems': problems})
        bad += [f'{sig}: {p}' for p in problems]
    rej = []
    for sig, pin, func in RECALLED:
        f, b = pins.get(pin, ('<not in package>', '?'))
        rej.append({'signal': sig, 'recalled_pin': pin, 'recalled_as': func, 'device_pin_func': f, 'accepted': func in f})
    o = {'part': PART, 'vivado': VIVADO, 'package_pins_queried': len(pins), 'xdc_file': str(XDC_PATH.relative_to(ROOT)),
         'xdc_sha256': hashlib.sha256(XDC_PATH.read_bytes()).hexdigest(), 'xdc_ports_parsed': xdc,
         'placed_design_checked': bool(placed), 'schematic': schematic_record(), 'pins': rows, 'recalled_pins_checked': rej,
         'cfgbvs': 'sheet 10: CFGBVS_0 (V9) tied to VCCp3V3, XDC sets CFGBVS VCCO / CONFIG_VOLTAGE 3.3',
         'vcco': 'sheet 13: VCCO_0, VCCO_14 and VCCO_15 are all VCCp3V3',
         'all_agree': not bad, 'disagreements': bad}
    # TODO(integration): provenance
    (ROOT / 'bench/results/boolean_pin_check.json').write_text(json.dumps(o, indent=2) + '\n')
    for r in rows:
        print(f"{r['signal']:52s} {r['pin']:4s} {r['device_pin_func']:34s} xdc={r['xdc_pin']}/{r['xdc_iostandard']} agrees={r['agrees']}")
    for r in rej:
        print('recalled', r)
    print('placed design checked:', bool(placed), '| schematic sha256:', o['schematic']['sha256'])
    for b in bad:
        print('DISAGREE:', b)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
