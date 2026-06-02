"""Committed minimized fuzzer repros + their regression tests.

Empty in v1 — no real miscompiles were discovered during development. The
deliverable is the corpus + coverage + the planted-bug proof-of-teeth in
`firmware/host/test_fuzzer.py`. Any bug found by a future run that
minimizes cleanly will be committed here as `bug_<id>.py` + a generated
pytest `test_bug_<id>.py`, and listed in `bench/results/fuzzer_report.json`
under `bugs[]`.
"""
