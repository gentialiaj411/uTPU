"""Structural guardrail for the serialized Phase-3 vector datapath."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def test_vector_datapath_has_no_combinational_element_loops():
    # Bit-slice generate loops are permitted; a loop in an always_comb block
    # is not.  The vector unit must advance element state in clocked FSMs.
    for source in (ROOT / "rtl" / "vector").glob("*.sv"):
        text = source.read_text(encoding="utf-8")
        for match in re.finditer(r"always_comb\s+begin(?P<body>.*?)\bend\b", text, re.S):
            assert not re.search(r"\bfor\s*\(", match.group("body")), source
