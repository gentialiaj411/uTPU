"""Run or dry-run the Vivado Hardware Manager flash-programming Tcl."""
from pathlib import Path
import argparse, subprocess
ROOT = Path(__file__).resolve().parents[2]
def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("mcs", type=Path); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--vivado", default=r"C:\AMDDesignTools\2025.2\Vivado\bin\vivado.bat")
    a=ap.parse_args(argv); mcs=a.mcs.resolve()
    if not mcs.exists(): raise FileNotFoundError(mcs)
    tcl=(ROOT/"tools/board/program_flash.tcl").read_text(); tcl=tcl.replace("$::mcs_file", "{" + mcs.as_posix() + "}")
    generated=ROOT/"build/board/program_flash_generated.tcl"; generated.parent.mkdir(parents=True, exist_ok=True); generated.write_text(tcl)
    if a.dry_run: print(tcl); return 0
    subprocess.run(["cmd","/c",a.vivado,"-mode","batch","-nojournal","-nolog","-source",str(generated)],cwd=ROOT,check=True); return 0
if __name__ == "__main__": raise SystemExit(main())
