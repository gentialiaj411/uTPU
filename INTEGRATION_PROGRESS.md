# Integration progress

## Chunk status
- [x] C1 Review code diffs and finish provenance wiring.
- [x] C2 Cross-branch consistency, required pytest, and code commit.
- [ ] C3 Fast artifact regeneration.
- [ ] C4 Owner runner and handoff for long/Vivado jobs.
- [ ] C5 Owner results, strict gate, stale-artifact demo.
- [ ] C6 Documentation and artifact/docs commits.

## Commands and results
- `Get-Content INTEGRATION_PROGRESS.md` — exit 1; file did not exist at start of continuation.
- `git status --short --branch` — exit 0; branch `audit-fixes`; inherited modifications present in code, results, and fault_inject staged.
- `git diff -- ...` — exit 0; began review of inherited code diff (output truncated; review not yet complete).
- `Get-Content INTEGRATION_PROGRESS.md; git status --short --branch` — exit 0; resumed from C1, code/result modifications unchanged.
- `git show-ref --verify ...` — exit 0 for all four supplied branch refs.
- Python AST comparison via `git show <ref>:tools/decode/fault_inject.py` — exit 0; missing fault names: none across the four refs. Current unit counts: rms 7, vec 8, attn 11, actq 5, top 10, qspi 24, board 29, attn15 1, gemv 5.
- `python -m pytest tools/decode/test_gate.py firmware/host/numerics_v2/test_numerics_v2.py firmware/host/test_claim_drift.py -q` — exit 0; 85 passed in 119.12s (repeated after provenance update).
- `git diff --check` — exit 0.
- `rg -n "TODO(integration)" tools docs` — exit 1/no matches.
- QSPI inventory Python check — exit 0; runner has 58 variants, QSPI has 24 faults; gate checks these from their canonical sources.
- Added `build/decode/decode_image_15m/params.json` to board implementation and 15M synthesis provenance source lists; this is used to generate/validate those results.
- Code commit: pending in this turn.

## Owner commands required
- Pending C4; list Vivado and jobs exceeding 30 minutes here.
