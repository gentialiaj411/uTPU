"""Metamorphic differential fuzzer package (Task 2 / `utpu_upgrade_plan.md` §4).

A generator + metamorphic + differential harness that finds compiler
divergences across the existing GraphIR → backend pipeline:

* `graph_generator` — sample random VALID GraphIR programs against the
  compiler's own legality predicates (`graph_passes.supported_ops_for_backend`
  + shape inference rules). Deterministic given a seed.
* `differential_oracle` — thin wrapper over `diff_oracle.run_all_backends`
  that auto-includes `cuda_megakernel` (Task 1) when CUDA is available
  and a graph contains exactly one fusable region.
* `metamorphic` — semantics-preserving variants of the SAME graph that
  must agree (fusion on/off, region-fused vs op-by-op, schedule selected
  vs alternative, DCE on/off, tiling A vs B). A mismatch is a compiler
  bug with no external oracle needed.
* `minimizer` — delta-debugging (`ddmin`) shrinker that reduces a
  failing graph to the minimal set of ops + tightest shapes that still
  trigger the divergence.

Honesty contract (inherited from `CLAUDE.md` / `AGENTS.md`):
- The fuzzer reports the bugs IT actually catches. If it caught zero
  during a run, the artifact says zero — no fabricated bug count.
- The planted-bug test (in `test_fuzzer.py`) is the proof-of-teeth:
  an injected wrong rewrite MUST be caught by the metamorphic harness,
  even if no real bug is found in the live compiler.
- Integer-fragment relations are bit-exact. Float relations use
  documented rtol/atol from `diff_oracle`.
"""

from __future__ import annotations
