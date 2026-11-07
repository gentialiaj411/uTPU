"""Program the Boolean board over JTAG with Vivado Hardware Manager (tools/board/program_flash.tcl).

  program_flash.py build/board/sdr/boolean_top.mcs            # write bitstream + weights to flash, verify, reboot from flash
  program_flash.py build/board/sdr/boolean_top.bit --jtag     # volatile FPGA load only (weights must already be in flash)
  add --dry-run to print the Tcl without touching hardware.
"""
from pathlib import Path
import argparse, subprocess

ROOT = Path(__file__).resolve().parents[2]
VIVADO = r"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat"


def make_tcl(target: Path, jtag: bool) -> str:
    if jtag:
        if target.suffix != ".bit": raise ValueError(f"--jtag needs a .bit file, got {target.name}")
        head = f"set ::mode jtag\nset ::bit_file {{{target.as_posix()}}}\nset ::mcs_file {{}}\nset ::prm_file {{}}\n"
    else:
        if target.suffix != ".mcs": raise ValueError(f"flash programming needs the .mcs from make_flash_image.py, got {target.name}")
        prm = target.with_suffix(".prm")
        if not prm.exists(): raise FileNotFoundError(f"{prm} (write_cfgmem writes it next to the .mcs)")
        head = f"set ::mode flash\nset ::bit_file {{}}\nset ::mcs_file {{{target.as_posix()}}}\nset ::prm_file {{{prm.as_posix()}}}\n"
    return head + (ROOT / "tools/board/program_flash.tcl").read_text()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help=".mcs (flash) or .bit (with --jtag)")
    ap.add_argument("--jtag", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--vivado", default=VIVADO)
    a = ap.parse_args(argv)
    target = a.target.resolve()
    if not target.exists(): raise FileNotFoundError(target)
    tcl = make_tcl(target, a.jtag)
    generated = ROOT / "build/board/program_flash_generated.tcl"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text(tcl)
    if a.dry_run:
        print(tcl); return 0
    subprocess.run(["cmd", "/c", a.vivado, "-mode", "batch", "-nojournal", "-nolog", "-source", str(generated)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
