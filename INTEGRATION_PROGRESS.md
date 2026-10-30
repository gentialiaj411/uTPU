# Integration progress

## Chunk status
- [x] C1 Review code diffs and finish provenance wiring.
- [x] C2 Cross-branch consistency, required pytest, and code commit.
- [x] C3 Fast decode unit artifact regeneration; `pin_check` moved into owner sequence because it requires Vivado.
- [x] C4 Owner runner and handoff for long/Vivado jobs prepared; owner execution pending.
- [x] C5 Owner results, strict gate, stale-artifact demo (owner script stopped at synth job; fixed top name, owner rerun required).
- [x] C6 Documentation and artifact/docs commits.

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
- `python tools/decode/run_gemv_rtl.py` — exit 0; status pass, mismatch 0, mutation mismatches/expected 1006, cycles/tile 101.2654.
- `python tools/decode/run_gemv_rtl.py --golden build/decode/golden_gemv15/cases.json --output bench/results/decode_gemv15_rtl.json` — exit 1; CLI expects a golden directory, not `cases.json`; no artifact was written by this failed invocation. Retry with `--golden build/decode/golden_gemv15`.
- `python tools/decode/run_gemv_rtl.py --golden build/decode/golden_gemv15 --output bench/results/decode_gemv15_rtl.json` — exit 0; status pass, mismatch 0, mutation mismatches/expected 2448, cycles/tile 365.1176.
- `python tools/decode/run_act_quant_rtl.py` — exit 0; status pass, mismatch 0, mutation mismatches/expected 5447, cycles/group 137.6328.
- `python tools/decode/run_rms_rtl.py` — exit 0; status pass, mismatch 0, mutation mismatches/expected 68576.
- `python tools/decode/run_vec_rtl.py` — exit 0; status pass, mismatch 0, mutation mismatches/expected 75044, cycles total 1123868.
- `python tools/decode/run_attn_rtl.py` — exit 0; status pass, mismatch 0, mutation mismatches/expected 28960, cycles/head call 2336.2627; kv8 and kv4 each 1810 calls, 0 errors.
- `python tools/decode/run_attn15_rtl.py` — exit 0; status pass, mismatch 0, mutation mismatches/expected 41184, cycles/head call 21121.3438; KV4 summary 858 calls, 0 errors. Started 2026-10-28 12:53:58 local.
- `python tools/decode/run_qspi_rtl.py` — exit 0; status pass across 58 variants, negative model checks, and parameter rejection checks. First variants had clean errors=0 and expected mutation mismatches.
- Poll session 29153 — exit 0; final runner output `{"status":"pass"}`.
- `Get-Content INTEGRATION_PROGRESS.md; git status --short --branch` — exit 0; resumed after attention15/QSPI completion; generated artifacts remain uncommitted.
- `python -c ... inspect e2e artifacts` — exit 0; confirmed gate outputs encode period 1/2/4 as p1/p2/p4; current p1 elapsed was 3506.7s, so these e2e runs belong in the owner sequence.
- Created `OWNER_RUN.ps1`: sequential fail-fast jobs, each output logged under `build/owner_logs`; includes all nine fault units, six strict-gate e2e configs (including 0.5 and 0.25 B/cycle via periods 2 and 4), UART TB, nine synth OOC entries, implementation, pin check, 15M KV4 e2e, then strict gate. No owner jobs run yet.
- PowerShell parser validation of `OWNER_RUN.ps1` — exit 0; `OWNER_RUN.ps1 parse OK`.
- Owner reported first `OWNER_RUN.ps1` invocation stopped at line 10 with a Python 3.11.9 interactive banner / `NativeCommandError`; no successful job result was provided. Updated `Invoke-Logged` to use explicit `-CommandArgs` (avoid PowerShell automatic `$Args` name), capture native output to a per-job temporary file, preserve exit status, then append output and exit code to the named log. Renamed loop-local `$args` to `$cmdline`.
- PowerShell parse + `Invoke-Logged` argument-forwarding smoke (`python -c ... arg-one arg-two`) — exit 0; Python received both args; helper output captured; syntax valid.
- Owner reran `powershell -ExecutionPolicy Bypass -File OWNER_RUN.ps1` on 2026-10-29. The script redirects each command's output to `build/owner_logs/<job>.log`; console silence is expected. Snapshot at 12:28:48 PM: `fault_gemv.log` exit 0, `fault_actq.log` exit 0, `fault_rms.log` exit 0, `fault_vec.log` exit 0; `fault_attn` started with empty temporary output. This confirms progress; not blocked.
- Later owner logs show all fault suites and five e2e runs completed before 15:59, then UART passed (4 steps, 178 checks, 0 mismatches, mutation 1; 3,795,247 cycles). QSPI e2e passed (10 steps, 0 logit/token/index/argmax errors; 930,680.6 cycles/token mean). Script stopped at first OOC synth job: `synth_gemv_unit` logged `module 'gemv_unit' not found`, exit 1. RTL declares `decode_gemv_unit`; corrected that top name in OWNER_RUN.ps1. Earlier stages are complete in this run and their artifacts/logs exist; rerunning script currently reruns the suites from the start.
- Process check after synth failure — no python/vvp/vivado processes visible; owner script is stopped.
- Read-only docs review — `docs/r3_progress.md` already has Claim scope and audit sections through board; final integration table must be appended after owner artifacts are verified.
- PowerShell data-shape check for nested `$e2e` and `$synthJobs` arrays in `OWNER_RUN.ps1` — exit 0; nested job arrays remain arrays and expand into the expected command/argument sequences.
- C3 `pin_check` was NOT RUN here; it invokes Vivado and is included after implementation in `OWNER_RUN.ps1`.
- `Get-Process python,vvp ...` during attention15 — exit 0; active VVP process confirmed, so run was left intact.
- Continuation polling session 1553 — exit 0; attention15 final result above.
- `python tools/decode/run_gemv_rtl.py --help` (and other unit runner `--help` commands) — exit 0; established runner options; C3 still active.
- Code commit `0d84601` — exit 0; provenance/gate code committed, results remain uncommitted.
- `git diff --check` — exit 0; no whitespace errors.
- `rg -n "TODO(integration)" tools docs` — exit 1/no matches.

## Owner commands required
- Owner must rerun from this worktree after this fix: `powershell -ExecutionPolicy Bypass -File OWNER_RUN.ps1`. It runs sequentially and stops on first failure. It includes full fault suites and stories15M e2e (>30 minutes) and Vivado synth/implementation/pin checks. Read `build/owner_logs/<name>.log` after each run; all logs are created by the script.

## Sprint closure 2026-10-30T12:36:15.581653
All owner log tails EXIT 0 verified in build/sprint_logs/owner_log_audit.csv. Strict gate passed; comment-only gemv mutation failed naming stale gemv_unit.sv; original bytes restored and gate passed again. Artifacts generated by owner tools, not hand-edited.
