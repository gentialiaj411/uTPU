"""The RTL ROM images must be a mechanical rendering of the frozen fixture."""
from pathlib import Path
import subprocess
import sys


def test_vector_lut_mem_images_match_frozen_fixture():
    root = Path(__file__).resolve().parents[2]
    assert subprocess.run([sys.executable, "tools/gen_vector_lut_mems.py", "--check"], cwd=root).returncode == 0
