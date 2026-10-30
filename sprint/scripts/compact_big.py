"""Isolated post-compaction regression; no production source edits."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/decode'));sys.path.insert(0,str(ROOT))
sys.argv=['tools/decode/run_decode_rtl.py','--model','15m','--kv-bits','4','--prompts','8','--gen','32','--sim','verilator','--define','SPRINT_COMPACT_BIG','--timeout','4500','--output',str(ROOT/'sprint/data/15m_big_compact.json')]
import harness
isolated=ROOT/'build/decode/compact_big';isolated.mkdir(parents=True,exist_ok=True);harness.BUILD=isolated
import run_decode_rtl as runner
runner.BUILD=isolated
sources=runner.e2e_sources
runner.e2e_sources=lambda model,flash:[*sources(model,flash),'sprint/scripts/compact_big.py']
runner.main()
