"""Verify every pin the Boolean top uses against the device database and the board documents.

For each signal: the pin from Real Digital's master constraints file (where it lists one), the schematic net at that
pin (read by eye from the rasterized sheets 10, 11 and 13 of the Boolean schematic, because the PDF text layer is
scrambled: the 'schematic' fields below are transcriptions, not machine output), and the device's own PIN_FUNC / BANK
from the installed Vivado (queried live here with link_design -part xc7s50csga324-1). A pin agrees when the device
function matches what the signal needs and the schematic net is the signal. Writes bench/results/boolean_pin_check.json;
exits non-zero if any pin used by the design disagrees.
"""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIVADO = r'C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat'
PART = 'xc7s50csga324-1'
SCH = 'Boolean schematic KT-000-001-001-001 V2I1 (realdigital.org/downloads/63f9a8205ebd9c2e8c2d265ad25097dc.pdf)'
XDC = 'Real Digital Boolean master constraints (realdigital.org/downloads/8d5c167add28c014173edcf51db78bb9.txt)'

# signal, pin, device PIN_FUNC substring required, xdc port (or None), schematic net/sheet, bank VCCO per schematic sheet 13
USED = [
    ('clk (100 MHz)',  'F14', 'MRCC_15',   'clk',       ('UCLK, driven by IC9 ASEM1-100.000MHZ', 11),        3.3),
    ('uart_rxd (FPGA RX)', 'V12', '_14',   'UART_rxd',  ('UART_TXD_IN (USB-UART bridge TX into the FPGA)', 11), 3.3),
    ('uart_txd (FPGA TX)', 'U11', '_14',   'UART_txd',  ('UART_RXD_OUT (FPGA TX to the USB-UART bridge)', 11), 3.3),
    ('qspi_cs_n',      'M13', 'FCS_B_14',  None,        ('QSPI_CS to IC8 pin 7 (CS)', 11),                   3.3),
    ('qspi_dq[0]',     'K17', 'D00_MOSI_14', None,      ('QSPI_DQ0 to IC8 pin 15 (SDI/DQ0)', 11),            3.3),
    ('qspi_dq[1]',     'K18', 'D01_DIN_14', None,       ('QSPI_DQ1 to IC8 pin 8 (SDO/DQ1)', 11),             3.3),
    ('qspi_dq[2]',     'L14', 'D02_14',    None,        ('QSPI_DQ2 to IC8 pin 9 (WP#/DQ2), 1.8k pull-up', 11), 3.3),
    ('qspi_dq[3]',     'M15', 'D03_14',    None,        ('QSPI_DQ3 to IC8 pin 1 (HOLD#/DQ3), 1.8k pull-up', 11), 3.3),
    ('flash SCLK (CCLK_0, via STARTUPE2, no constraint)', 'C8', 'CCLK_0', None, ('QSPI_SCK to IC8 pin 16 (SCK)', 10), 3.3),
]
# pins another source (the coordinator's recollection of the Arty S7 pinout) proposed; checked and rejected
RECALLED = [('qspi_cs_n', 'L13', 'FCS_B'), ('qspi_dq[3]', 'M14', 'D03')]


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


def main():
    pins = query()
    rows, bad = [], []
    for sig, pin, need, xdc_port, (net, sheet), vcco in USED:
        func, bank = pins.get(pin, ('<not in package>', '?'))
        ok = need in func
        rows.append({'signal': sig, 'pin': pin, 'device_pin_func': func, 'device_bank': bank, 'required_function': need,
                     'master_xdc': (f'{XDC}: {xdc_port} on {pin}' if xdc_port else 'not listed in the master XDC'),
                     'schematic': f'{SCH} sheet {sheet}: {net}', 'schematic_bank_vcco_v': vcco,
                     'iostandard': 'LVCMOS33' if vcco == 3.3 else '?', 'agrees': ok})
        if not ok:
            bad.append(sig)
    rej = []
    for sig, pin, func in RECALLED:
        f, b = pins.get(pin, ('<not in package>', '?'))
        rej.append({'signal': sig, 'recalled_pin': pin, 'recalled_as': func, 'device_pin_func': f, 'accepted': func in f})
    o = {'part': PART, 'vivado': VIVADO, 'package_pins_queried': len(pins), 'pins': rows,
         'recalled_pins_checked': rej,
         'cfgbvs': 'sheet 10: CFGBVS_0 (V9) tied to VCCp3V3, XDC sets CFGBVS VCCO / CONFIG_VOLTAGE 3.3',
         'vcco': 'sheet 13: VCCO_0, VCCO_14 and VCCO_15 are all VCCp3V3',
         'all_agree': not bad, 'disagreements': bad}
    (ROOT / 'bench/results/boolean_pin_check.json').write_text(json.dumps(o, indent=2) + '\n')
    for r in rows:
        print(f"{r['signal']:52s} {r['pin']:4s} {r['device_pin_func']:34s} agrees={r['agrees']}")
    for r in rej:
        print('recalled', r)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
