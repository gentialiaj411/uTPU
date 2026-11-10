"""Provenance source lists for the K=8192 vocab-pruned artifacts.

Kept out of tools/decode/provenance.py on purpose: that file is itself a recorded source of every decode artifact, so adding
entries there would stale all of them. tools/decode_gate.py merges PRUNED_SOURCES into ARTIFACT_SOURCES; the pruned runners
(tools/vocabprune/run_decode_rtl.py, hardware.py, finalize.py, tools/board/impl_top.py --pruned) record provenance from here.
This file is covered by the tools/vocabprune/*.py glob, so editing it stales the pruned artifacts only.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'decode'))
from provenance import BOARD_RTL, D, LUTS, M15, NUMERICS, QSPI_RTL, TOOLS, TOP_RTL   # noqa: E402

PRUNED_COMMON = ['tools/vocabprune/*.py', 'tools/vocabprune/idmaps/idmap_8192.json', 'tools/decode/pack_decode_image.py']

PRUNED_SOURCES = {
    'vocab_prune_rtl': [*PRUNED_COMMON, 'tools/decode/run_decode_rtl.py', *TOP_RTL, *TOOLS, *NUMERICS, *LUTS, *M15],
    'vocab_prune_ddr_rtl': [*PRUNED_COMMON, 'tools/decode/run_decode_rtl.py', *TOP_RTL, *QSPI_RTL, *TOOLS, *NUMERICS, *LUTS, *M15],
    'vocab_prune_study': ['tools/vocabprune/*.py', 'tools/vocabprune/idmaps/idmap_*.json', 'tools/decode/pack_decode_image.py',
                          'bench/results/decode_e2e_rtl_15m_kv4_p1_qspiddr.json', *NUMERICS, *M15],
    'pruned_ddr_impl': [*PRUNED_COMMON, 'tools/board/impl_top.py', 'tools/board/build_bitstream.py', 'tools/board/make_flash_image.py',
                        'constraints/boolean_xc7s50.xdc', 'constraints/boolean_xc7s50_qspi_ddr.xdc', *BOARD_RTL, *TOP_RTL,
                        D + 'qspi_reader.sv', 'tools/decode/provenance.py', *M15],
}
