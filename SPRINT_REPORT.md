# uTPU engineering sprint report
Generated 2026-10-30T17:11:43.686764. Worktree uTPU-perf, branch perf, based on integration e8d9282.

P4 RTL remains EXPERIMENTAL pending complete fault and physical closure; measured simulation results are not board performance claims.

## Summary
1. P0 DONE: integration closed at e8d9282; all owner log tails EXIT 0, fresh strict gate PASS, stale-comment FAIL, restored gate PASS (MEASURED; sprint/data/p0_*).
2. P1 DONE: 11/11 completed backend comparisons identical and passing (MEASURED; crosscheck_comparison.csv).
3. P1 15M small: pass, 11 steps; wall 207.37 s (MEASURED; 15m_timing.json).
4. P1 large: pass, 302 steps, 8 actual prompts; errors logit/index/argmax = 0/0/0 (MEASURED; 15m_big.json).
5. P2 DONE code/tests: simulator executable/compiler identity, Python/NumPy identity and fault model/golden dependencies tracked; see test_gate_final.log (MEASURED).
6. P3 NO-GO for hardware from these bounds; MEASURED max mean row skip 0.000000 over the final screening variants; best screening estimate 1.0741 tok/s at 0.25 B/clk (ESTIMATED), versus baseline bandwidth ceiling 1.0745 (DERIVED).
7. P4 TIMEBOXED; large regression and unit suites pass, complete fault acceptance unfinished; 15M QSPI bytes/token 10,684,413, clocks/token 42,856,333.5 (MEASURED); 1.16669 tok/s at 50MHz (DERIVED).
8. P5 TIMEBOXED design-only: existing autonomous prefetch already overlaps serial compute until FIFO full; no extra RTL feature retained.
9. P6 TIMEBOXED until physical regeneration: strict gate kept strict; OWNER_RUN2.ps1 finishes timeboxed fault sweeps, then runs sequential logged Vivado jobs and final gate. No push or hardware programming.

## Bugs and verification findings first
Confirmed long-run instrumentation bug: run_decode_rtl.py declares its total cycles as a signed 32-bit SystemVerilog integer. The compact 302-step regression prints cycles=-519156838 after overflow. Its per-step counters and numerical comparisons remain valid; do not use that SUMMARY total as a measured elapsed-cycle value. The compact total is DERIVED as round(302 * 12502683.635761589) = 3775810458 clocks; its signed-32 representation exactly matches the erroneous SUMMARY. The baseline also overflows. Evidence: cycle_counter_overflow.json and cycle_counter_audit.py; the Python mean comes from parsed per-step counts. Widening this reporting counter and adding a >2^31-clock directed check is deferred, because changing the producer now would invalidate the completed long-run provenance. This is a testbench reporting bug, not a numerical DUT failure.

- Unit TB timing was unspecified: GEMV used #0.1 without timescale. Verilator initially reported 12,754 mismatches and stale done behavior. Explicit 1ns/1ps is now supplied to every TB lacking a time declaration for BOTH backends. Golden arithmetic and scoreboard remain unchanged. The clean backends agree on 100.2654 cycles/tile versus historical 101.2654; this difference is testbench sampling, not a compute speedup. Evidence gemv_*.json and SPRINT_LOG.md.
- The 15M e2e runner silently truncated --prompts 8 to its two-entry prompt bank. It now extends the bank from TinyStories validation and rejects context overflow rather than silently under-testing. Existing default prompts are preserved.
- Provenance previously missed simulator executable identity, Python/NumPy versions and fault model/numeric inputs. New inputs are required by coverage, so historical artifacts fail until regenerated.
- P3 self-review caught a software screening error: maximum per-row bounds cannot be reconstructed from only groupwise block-header maxima. The initial CSV was explicitly archived INVALID and excluded. Final coefficient results actually use the declared groupwise block header.

Additional confirmed verification findings: sparse QSPI used multiple stateful RNG calls in task arguments, whose evaluation order differed across simulators; explicit ordered sampling now preserves the full stimulus set while making it deterministic. UART queue polling raced active-edge monitor updates; post-edge sampling removes the four-cycle summary skew without changing checks. The initial QSPI/UART compile failures were also traced to a full 7.7GB tmpfs and zero-byte makefiles, not RTL errors. Failed outputs and disk evidence are retained. Finally, the original research 50-prompt record list contained only nine distinct eight-token prefixes; P3 now requires 50 distinct prefixes and records them explicitly.

## Evidence rules and environment
MEASURED means executed here, with command/artifact. DERIVED is arithmetic from named measurements or explicitly named historical inputs. ESTIMATED is a model with assumptions; never a board result. Numerical dimensions and test counts are configurations. No bench/results JSON was hand-edited. Tools: Windows Python/NumPy identities are recorded in each newly generated provenance; Verilator 5.032 runs under WSL Ubuntu using tools extracted in ../uTPU-research/research/data/tools. Icarus uses C:/iverilog/bin. No new Vivado run was attempted.

## P0 integration closure
Commands: strict gate python tools/decode_gate.py; stale demonstration script ../uTPU-integ/build/sprint_logs/close.py appends a comment to GEMV inside try/finally, checks the stale failure, restores exact bytes and reruns the gate. Owner logs were audited by final-line EXIT 0, saved as sprint/data/p0_owner_log_audit.csv. Copies of before/stale/restored gate output are p0_gate_*.log. Integration docs and owner-generated artifacts committed e8d9282; untracked owner clockInfo.txt was preserved.

## P1 portable simulator backend
tools/decode/sim.py implements --sim icarus|verilator (default icarus), shared by runners and inherited through UTPU_SIM for fault subprocesses. Windows memory paths are mapped to WSL only in the generated Verilator TB copy. Compile logs go to build/decode/*_compile.log. Generated C++/objects build under isolated /tmp/utpu-perf directories, which are removed after every completed run/failure to avoid filling the 7.7GB tmpfs. The initial exhaustion failure and recovery logs are preserved. The old vvp_command field is retained for artifact compatibility; mandatory tool:simulator provenance identifies the actual backend. Reproduce using the exact Python producer commands, not that legacy field. Build failures preserve CalledProcessError so existing parameter-rejection tests remain meaningful. CLI help added to each runner. Explicit TB time precision applies identically to both backends. No expected outputs or tolerances were changed.

Commands: python sprint/scripts/crosscheck.py; python sprint/scripts/compare.py; python sprint/scripts/regress15.py. Crosscheck compiles clean and mutation variants through the original runners. Comparison includes every summary, mismatch count, cycle counter and variant recursively; stdout hashes and wall times are intentionally not asserted equal. Original output/logs are sprint/data/<unit>_<backend>.{json,log}.

| Unit | Icarus status | Verilator status | Identical relevant fields MEASURED | Differences |
| --- | --- | --- | --- | --- |
| actq | pass | pass | True | {} |
| attn15 | pass | pass | True | {} |
| attn | pass | pass | True | {} |
| e2e_ideal | pass | pass | True | {} |
| e2e_qspi | pass | pass | True | {} |
| gemv15 | pass | pass | True | {} |
| gemv | pass | pass | True | {} |
| qspi | pass | pass | True | {} |
| rms | pass | pass | True | {} |
| uart | pass | pass | True | {} |
| vec | pass | pass | True | {} |

| Runner | Icarus wall s MEASURED | Verilator wall s MEASURED | Speedup DERIVED I/V |
| --- | --- | --- | --- |
| gemv | 8.016 | 18.182 | 0.441 |
| gemv15 | 75.999 | 102.070 | 0.745 |
| actq | 8.826 | 88.075 | 0.100 |
| rms | 52.859 | 16.347 | 3.234 |
| vec | 8.500 | 27.605 | 0.308 |
| attn | 280.993 | 34.673 | 8.104 |
| attn15 | 1120.026 | 73.758 | 15.185 |
| qspi | 93.289 | 2161.307 | 0.043 |
| uart | 608.884 | 119.072 | 5.114 |
| e2e_ideal | 146.170 | 12.958 | 11.280 |
| e2e_qspi | 382.588 | 17.791 | 21.504 |

These runner times include compilation, golden preparation and provenance; they are not simulation-only speedups. The earlier 87x was a small precompiled workload, not a guaranteed full 15M speedup. The measured 15M small and big wall times replace that extrapolation. Concurrent screening/regressions affect wall time.

| 15M workload | Actual prompts MEASURED | Steps MEASURED | Status | Logit mismatches MEASURED | Index / argmax MEASURED |
| --- | --- | --- | --- | --- | --- |
| small baseline | 1 | 11 | pass | 0 | 0 / 0 |
| 8x32 baseline | 8 | 302 | pass | 0 | 0 / 0 |
| 8x32 compact image | 8 | 302 | pass | 0 | 0 / 0 |

Final model-compatibility repair: after compaction changed reset timing, Verilator retained an old flash array-data nibble during a later RDID despite correct instruction/register state, whereas Icarus passed all reset checks. Setting only the inactive flash_o value to zero when flash_oe=0 removes redundant internal tri-state encoding. The external bus remains Z under its explicit enable, active X/tV windows and all protocol/timing checks are unchanged. The single-timed-process prototype failed and was not adopted. Evidence: uart_compact_failure.json, uart_reset_detail_trace.log, uart_reset_icarus_trace.log and uart_reset_idle_data_trace.log. Full final-model revalidation commands/status are in engineering_status.json.

## P2 provenance
Simulator provenance records name, version, executable path/SHA256 and compiler SHA256. Version strings are cached only by executable content digest; stale checks recompute binary digests. Python sys.version and NumPy __version__ hashes are mandatory virtual dependencies. Fault provenance includes M260, M15, NUMERICS and LUTS in addition to fault runner and RTL dependencies. Tests cover missing identities, changed recorded identities, actual mocked runtime changes and actual mocked simulator digest changes, plus each model/numeric source digest. Command python -m pytest tools/decode/test_gate.py -q; test_gate_final.log. New virtual dependencies also make physical artifacts stale; OWNER_RUN2 regenerates their provenance rather than accepting historical results.

## P3 exact bounded classifier argmax
Command python sprint/scripts/argmax_bound.py. Input model/text/prompt/numeric hashes are in sprint_input_hashes.json. Dataset: 50 distinct eight-token prefixes from TinyStories validation (argmax_prompts.json); a diversity audit found the original research 50-record prompt list had only nine unique prefixes, so nine complete measurements were reused and 41 distinct prompts were added. Eight prompt tokens plus 120 generated tokens, KV4, full sequence length128; the last classifier evaluation forwards input position126, and predicts the final generated token at index127. The reference integer model computes all logits as an independent checker; the selection simulation reads only admitted blocks and never uses skipped logits to set the running best. This is a screening measurement of reads and decisions, NOT a Python inference speedup. The current reproduction script always evaluates all distinct prompts from scratch; the one-off diversity-repair cache is not required on a fresh checkout, and partial/timeboxed runs return nonzero with actual position counts.

### Bound proof
For a weight group g, write d_rg=sum_i x_i*w_ri; a_g is its positive activation scale and m_rg=RNE(M_rg/2^16)>=0; s_rg=shift_rg-16>=0. L1 gives |d|<=max|x|*sum|w|; Cauchy gives |d|<=ceil(sqrt(sum x^2 * sum w^2)). Integer square comparisons correct the sqrt proposal. Multiplying by positive a*m preserves the bound; signed RNE is odd and monotone, so |RNE(d*a*m/2^s)|<=RNE(D*a*m/2^s). For the pinned classifier, bound_domain_proof.json certifies the actual shift range lies in the hardware six-bit shift domain and the worst legal INT32 activation-scale L1 product fits INT64. Per-input checks also reject overflow rather than silently wrap. Coefficient products use the limb construction described below.

div127 uses Q32 reciprocal R=33818641, followed by at most one unit correction. For |n|<=N, a safe envelope is floor((N*R+2^31)/2^32)+1; this also covers the wide approximate division path and does not incorrectly assume exact division there. Summing per-group envelopes bounds the final signed logit. Reciprocal products are evaluated in high/low limbs to prevent overflow. Runtime checks confirm every reference logit lies below its bound.

Coefficient variant: precompute c_rg=ceil(L1_rg*m_rg*2^32/2^s_rg). Store C_bg=max_rows_in_block c_rg separately for every group. For each activation, N_bg=ceil(C_bg*a_g*max|x_g|/2^32)+1 bounds the rounded scaled magnitude of every row in that block. Apply the reciprocal envelope and sum. Measured model coefficients fit u32; the script asserts this. Groupwise block maxima are used even though they are looser than max of activation-specific per-row sums. For full-range arithmetic, c*a is split into high/low Q32 limbs before multiplying by max|x|: ceil(c*a*x/2^32)=high*x+ceil(low*x/2^32), avoiding a potentially overflowing three-factor product. An independent arbitrary-precision oracle checked 10000 input triples (bound_math_property.json). This post-measurement arithmetic hardening is algebraically identical on every measured input, whose original product-overflow assertions all passed; the measured script digest is retained separately.

Skip a block only if U<best, or U==best and every row index in the block is greater than the current best index. RTL and integer lexsort choose the lowest index on ties. Independent reference checks require identical argmax AND maximum logit on every sampled position. No probability or quality tolerance is used. Because no sampled block was skipped, empirical equality does not exercise a successful pruning branch; the skip safety argument is mathematical, not evidence of useful pruning.

| Bound | Block rows | Ordering | Positions MEASURED | Mean skipped MEASURED | p10/p50/p90 MEASURED | Metadata B DERIVED | Mean B/token DERIVED | tok/s .25 ESTIMATED | tok/s .5 ESTIMATED | All identical MEASURED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coeff | 16 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 72000 | 11705661.0 | 1.0679 | 2.1357 | True |
| coeff | 16 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 72000 | 11705661.0 | 1.0645 | 2.1223 | True |
| coeff | 64 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 18000 | 11651661.0 | 1.0728 | 2.1456 | True |
| coeff | 64 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 18000 | 11651661.0 | 1.0719 | 2.1420 | True |
| coeff | 256 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 4500 | 11638161.0 | 1.0741 | 2.1481 | True |
| coeff | 256 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 4500 | 11638161.0 | 1.0738 | 2.1472 | True |
| l1 | 16 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9158 | 1.8315 | True |
| l1 | 16 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9139 | 1.8239 | True |
| l1 | 64 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9158 | 1.8315 | True |
| l1 | 64 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9153 | 1.8296 | True |
| l1 | 256 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9158 | 1.8315 | True |
| l1 | 256 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9156 | 1.8310 | True |
| l2 | 16 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9158 | 1.8315 | True |
| l2 | 16 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9144 | 1.8259 | True |
| l2 | 64 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9158 | 1.8315 | True |
| l2 | 64 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9155 | 1.8303 | True |
| l2 | 256 | natural | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9158 | 1.8315 | True |
| l2 | 256 | upper_first | 6000 | 0.00000 | 0.0000/0.0000/0.0000 | 2016000 | 13649661.0 | 0.9157 | 1.8313 | True |

Traffic model uses pre-P4 format: 144 code bytes +45 scale bytes per row (189B). L1/L2 screening streams all per-row norm+scale metadata (9 groups*(2+5)*32000=2,016,000 B), then rereads admitted blocks including scales; this deliberately pays both passes. Coefficient headers cost ceil(32000/block)*9*4 bytes: block16 72,000B, block64 18,000B, block256 4,500B. Against eight BRAM36=36,864B nominal capacity, only block64/256 header caches fit before port/fragmentation/control costs. Upper-first ordering needs header storage and sort/control; hardware cost is NOT MEASURED. No headers are assumed free.

QSPI model counts one seek for each non-contiguous admitted block: 6Bh command8+address24+dummy8=40 SCLK cycles=80 core clocks at HALF=1. Stored CSV predictions are optimistic protocol-only ceilings: cycles=(11,633,661-6,048,000+classifier_bytes)/rate+80*seeks; tok/s=50,000,000/cycles. CS-high/capture/pad/control latency and compute are additional; therefore these are upper bounds on architectural benefit, not measured end-to-end tok/s. Full non-contiguous reads preserve aligned packed tiles; blocks16/64/256 are all multiples of16.

P3 decision: NO-GO for hardware from these bounds. A conditional screening advantage does not authorize RTL. Required follow-up: measured controller/metadata bandwidth and seek latency, then an independently checked hardware prototype. A failure of these norm bounds does not prove every exact bounded argmax scheme impossible.

## P4 compact metadata
Disposition: TIMEBOXED; large regression and unit suites pass, complete fault acceptance unfinished. Prepared implementation sprint/scripts/implement_metadata.py is only applied if all baseline backend comparisons pass, protecting an unvalidated production format from depending on a failing simulator. If smoke/proof checks fail, exact pre-change source bytes are restored. See engineering_status.json for whether it actually ran.

If applied: packer stores q=RNE(original_multiplier,16) as u16 plus original shift u8 for GEMV tensors; embedding scale metadata stays u32+u8. RTL G_SCALES consumes three bytes, expands {q,16b0}; embedding E_SCALES still consumes five. Classifier tile offsets and board/default image addresses are regenerated from the packer. Math proof: RNE((q<<16)/2^16)=q, exactly the old GEMV effective multiplier; subsequent products, shifts and div127 are unchanged. Packer conservatively requires q<32768 although stored in u16, because the unchanged GEMV interprets the expanded 32-bit multiplier as signed. The measured model range fits this constraint; accepting arbitrary u16 would be unsound. Generic flash models consume the new packed image automatically, rather than supplying hand-authored expected data. New top fault omits <<16 and must be detected.

Post-compaction large regression command: python sprint/scripts/compact_big.py. It uses an isolated generated-input directory and an unused compile tag to prevent collisions with standard regeneration; the producer records this wrapper as an extra provenance dependency. Artifact 15m_big_compact.json.

Validation commands: python -m pytest sprint/scripts/test_compact_metadata.py -q compares the integer golden on original versus independently decoded compact metadata over generated tie and random cases; original clean/mutation runners and touched-unit fault suites are then rerun. No change to runq_runtime.py or fixed.py. Full acceptance requires every listed producer/fault to pass, not just the smoke.

| Config | Before B/token MEASURED | After B/token MEASURED | Before cycles/token MEASURED | After cycles/token MEASURED | 50MHz tok/s DERIVED |
| --- | --- | --- | --- | --- | --- |
| 260K QSPI | 232554 | 215786 | 930680.6 | 863608.6 | 57.89660 |
| 15M QSPI | 11633661 | 10684413 | 46653325.5 | 42856333.5 | 1.16669 |

The before/after 15M QSPI microbenchmark uses the same four positions (--gen 1 with the original four-token prompt). The separate 15m_small.json is IDEAL 1 B/clk and is not substituted as a QSPI baseline. Historical 11.63MB/token and estimated 1.07 tok/s are not substituted for a fresh QSPI measurement. Research predicted 949,248B saving/pass; only actual producer output above supports a new measured claim.

Independent readonly preflight (MEASURED): python sprint/scripts/metadata_preflight.py intercepts the planned packer write, loads the original packer from integration commit e8d9282, decodes the actual compact bytes independently, verifies all packed weight-code payloads are unchanged, and compares all effective multipliers, shifts and two integer golden forwards per tensor. This does not modify production files or substitute for RTL validation. Artifacts metadata_preflight.csv and planned_packer_normalized_sha256.txt.
| Model | Tensors MEASURED | Serialized bytes saved MEASURED | All metadata / forwards equal MEASURED |
| --- | --- | --- | --- |
| 260k | 36 | 16768 | True |
| 15m | 43 | 949248 | True |

Static RTL-kind fault-target audit: python sprint/scripts/fault_needle_audit.py confirms every RTL-kind mutation needle still matches the current source after compaction. This is a source-integrity check, not a substitute for executing mutations; data/fault_needle_audit.csv.

## P5 serial overlap
TIMEBOXED design-only. Independent pf_active/pf_left prefetch and st_ready already run during RMS/attention/SwiGLU/quant until FIFO fills. Thus the research inference that all those phase clocks imply idle flash was too strong. No unmeasured enable parameter was added. See sprint/P5_PREFETCH_REVIEW.md. FIFO sizing and occupancy instrumentation remain next experiments, not a delivered speedup.

Independent overlap limit DERIVED for the measured short 260K QSPI workload: bytes/rate=215786/0.25=863144.0 clocks/token versus measured 863608.6. Even eliminating every non-bandwidth clock would improve throughput by at most 0.0538 percent. Source e2e_qspi_icarus.json, command in crosscheck.py. This does not bound long-context or faster-stream cases; it explains why another prefetch enable is a poor short-case optimization.
## P6 regeneration and strict gate
Engineering commands and return codes: sprint/data/engineering_status.json, with timestamped command lines in SPRINT_LOG.md. Every artifact is produced by its runner. The full QSPI unit producer output from the final P1 cross-check may be promoted byte-for-byte after full provenance freshness/coverage checks, since metadata format changes do not touch that unit or its inputs; this avoids rerunning its identical complete protocol suite. Promotion verifies identical SHA256 and never patches an artifact hash or field. Check the final full strict gate log sprint/data/strict_gate_closed.log (the earlier orchestration snapshot is strict_gate_final.log); any unmet item remains unmet. Physical results are NOT RUN until OWNER_RUN2.ps1 executes. No alternate relaxed gate was introduced.

| Job | Exit MEASURED | Recorded interval seconds MEASURED | Exact command |
| --- | --- | --- | --- |
| compare_final | 1 | 0.08 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/compare.py |
| 15m_qspi_before_metadata | 0 | 250.80 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --flash qspi --gen 1 --sim verilator --output sprint/data/15m_qspi_before_metadata.json |
| metadata_apply | 0 | 1.04 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/implement_metadata.py |
| metadata_python_exactness | 0 | 0.47 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe -m pytest sprint/scripts/test_compact_metadata.py -q |
| 260k_metadata | 0 | 16.98 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 4 --flash qspi --gen 8 --sim verilator --output sprint/data/260k_metadata.json |
| 15m_metadata | 0 | 228.71 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --flash qspi --gen 1 --sim verilator --output sprint/data/15m_metadata.json |
| regen_gemv | 0 | 11.79 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_gemv_rtl.py --sim verilator |
| regen_gemv15 | 0 | 17.60 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_gemv_rtl.py --golden build/decode/golden_gemv15 --output bench/results/decode_gemv15_rtl.json --sim verilator |
| regen_actq | 0 | 34.53 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_act_quant_rtl.py --sim verilator |
| regen_rms | 0 | 13.09 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_rms_rtl.py --sim verilator |
| regen_vec | 0 | 22.61 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_vec_rtl.py --sim verilator |
| regen_attn | 0 | 28.45 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_attn_rtl.py --sim verilator |
| regen_attn15 | 0 | 49.68 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_attn15_rtl.py --sim verilator |
| regen_qspi | 0 | 0.31 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/promote_fresh_qspi.py |
| regen_uart | 0 | 144.57 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/board/run_uart_rtl.py --sim verilator |
| fault_gemv | 0 | 61.81 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py gemv |
| fault_top | 0 | 145.24 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py top |
| fault_board | 0 | 957.66 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py board |
| regen_e2e_kv8_p1 | 0 | 12.48 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv8_p1.json |
| regen_e2e_kv4_p1 | 0 | 12.28 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 4 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv4_p1.json |
| regen_e2e_kv8_p2 | 0 | 12.95 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --period 2 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv8_p2.json |
| regen_e2e_kv8_p4 | 0 | 15.41 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --period 4 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv8_p4.json |
| regen_e2e_kv8_p1_qspi | 0 | 18.35 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --flash qspi --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv8_p1_qspi.json |
| regen_e2e_15m_kv4_p1 | 0 | 158.41 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_15m_kv4_p1.json |
| final_uart_icarus | 0 | 637.15 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/board/run_uart_rtl.py --sim icarus --output sprint/data/uart_icarus.json |
| fault_qspi | 124 | NOT RECORDED | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py qspi |
| final_15m_qspi | 0 | 234.54 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --flash qspi --gen 1 --sim verilator --output sprint/data/15m_metadata.json |
| final_qspi_verilator | 0 | 2675.94 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/final_qspi_validation.py verilator bench/results/decode_qspi_rtl.json |
| final_qspi_icarus | 0 | 2675.94 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/final_qspi_validation.py icarus sprint/data/qspi_icarus.json |
| postfix_decode_icarus | 0 | 2675.94 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/postfix_decode_pair.py icarus |
| postfix_decode_verilator | 0 | 2675.94 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/postfix_decode_pair.py verilator |
| strict_gate_final | 1 | 1.53 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode_gate.py |
| fault_qspi_prior_attempt | 4294967295 | 476.58 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py qspi |
| fault_actq | 0 | 42.31 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py actq |
| fault_rms | 124 | NOT RECORDED | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py rms |
| fault_vec | NOT RUN | NOT RECORDED | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py vec |
| fault_attn | NOT RUN | NOT RECORDED | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py attn |
| fault_attn15 | NOT RUN | NOT RECORDED | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py attn15 |
| strict_gate_closed | 1 | 4.09 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode_gate.py |
| fault_actq_verilator_timeboxed | 0 | 42.31 | C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py actq |

Parallel jobs record launch-to-result-collection intervals, including collection delay; these are not isolated simulator runtimes. Sequential jobs record runner wall time. Final remaining fault sweeps use UTPU_SIM=icarus (fast_fault_recovery.json), after identical full backend protocol summaries were established. The redundant Verilator QSPI fault sweep was TIMEBOXED; its failed/interrupted attempt remains visible, and passing Icarus producer results replace its acceptance evidence.

| Unmet artifact category | Dependency/gate reasons DERIVED |
| --- | --- |
| boolean_pin_check | 4 |
| boolean_top_impl | 8 |
| decode_fault_injection[attn15] | 53 |
| decode_fault_injection[attn] | 53 |
| decode_fault_injection[qspi] | 53 |
| decode_fault_injection[rms] | 53 |
| decode_fault_injection[vec] | 53 |
| decode_synth_ooc[act_quant] | 3 |
| decode_synth_ooc[attention] | 3 |
| decode_synth_ooc[gemv_unit] | 3 |
| decode_synth_ooc[qspi_reader] | 3 |
| decode_synth_ooc[rmsnorm] | 3 |
| decode_synth_ooc[top_decode_15m] | 8 |
| decode_synth_ooc[top_decode_15m_qspi] | 9 |
| decode_synth_ooc[top_decode_260k] | 4 |
| decode_synth_ooc[vec_alu] | 3 |

These counts describe strict-gate unmet dependency reasons, not numerical mismatches. The raw complete list is in strict_gate_closed.log. Full fault sweeps that did not finish remain stale; they are queued in OWNER_RUN2 before the physical jobs.

## Diff and numbers versus audit-fixes
```text
 OWNER_RUN2.ps1                                         |     43 +
 SPRINT_LOG.md                                          |    191 +
 SPRINT_REPORT.md                                       |    480 +
 SPRINT_STATE.md                                        |     10 +
 bench/results/boolean_uart_rtl.json                    |     45 +-
 bench/results/decode_act_quant_rtl.json                |     21 +-
 bench/results/decode_attention15_rtl.json              |     21 +-
 bench/results/decode_attention_rtl.json                |     21 +-
 bench/results/decode_e2e_rtl_15m_kv4_p1.json           |     51 +-
 bench/results/decode_e2e_rtl_kv4_p1.json               |     51 +-
 bench/results/decode_e2e_rtl_kv8_p1.json               |     51 +-
 bench/results/decode_e2e_rtl_kv8_p1_qspi.json          |     53 +-
 bench/results/decode_e2e_rtl_kv8_p2.json               |     51 +-
 bench/results/decode_e2e_rtl_kv8_p4.json               |     51 +-
 bench/results/decode_fault_injection.json              |    331 +-
 bench/results/decode_gemv15_rtl.json                   |     31 +-
 bench/results/decode_gemv_rtl.json                     |     29 +-
 bench/results/decode_qspi_rtl.json                     |    168 +-
 bench/results/decode_rms_rtl.json                      |     21 +-
 bench/results/decode_vec_alu_rtl.json                  |     59 +-
 docs/PERF_SPRINT.md                                    |     40 +
 rtl/board/boolean_core.sv                              |      4 +-
 rtl/decode/top_decode.sv                               |     12 +-
 sprint/.gitignore                                      |      3 +
 sprint/P5_PREFETCH_REVIEW.md                           |      6 +
 sprint/data/15m_big.json                               |    503 +
 sprint/data/15m_big_compact.json                       |    506 +
 sprint/data/15m_metadata.json                          |    130 +
 sprint/data/15m_qspi_before_metadata.json              |    130 +
 sprint/data/15m_small.json                             |    134 +
 sprint/data/15m_timing.json                            |     42 +
 sprint/data/260k_metadata.json                         |    135 +
 sprint/data/actq_icarus.json                           |     55 +
 sprint/data/actq_verilator.json                        |     55 +
 sprint/data/argmax_progress.json                       |      1 +
 sprint/data/argmax_prompts.json                        |      1 +
 sprint/data/argmax_sweep.csv                           | 216001 +++++++++++++++++++++++++++++++++++++++++++++++++++++++
 sprint/data/argmax_unique_complete.json                |      1 +
 sprint/data/attn15_icarus.json                         |     71 +
 sprint/data/attn15_verilator.json                      |     71 +
 sprint/data/attn_icarus.json                           |     71 +
 sprint/data/attn_verilator.json                        |     71 +
 sprint/data/backend_recovery_complete.json             |     12 +
 sprint/data/bound_domain_proof.json                    |     11 +
 sprint/data/bound_math_property.json                   |      6 +
 sprint/data/close_sprint_complete.json                 |     23 +
 sprint/data/compact_big_job.json                       |      3 +
 sprint/data/completion_status.json                     |      1 +
 sprint/data/crosscheck.csv                             |     28 +
 sprint/data/crosscheck_comparison.csv                  |     12 +
 sprint/data/crosscheck_progress.json                   |    137 +
 sprint/data/cycle_counter_overflow.json                |     16 +
 sprint/data/deterministic_recovery_complete.json       |     22 +
 sprint/data/deterministic_recovery_progress.json       |     22 +
 sprint/data/e2e_ideal_icarus.json                      |    133 +
 sprint/data/e2e_ideal_verilator.json                   |    133 +
 sprint/data/e2e_qspi_icarus.json                       |    136 +
 sprint/data/e2e_qspi_verilator.json                    |    136 +
 sprint/data/engineering_pre_flashfix_status.json       |    326 +
 sprint/data/engineering_status.json                    |    489 +
 sprint/data/fault_needle_audit.csv                     |     93 +
 sprint/data/fault_needle_missing.json                  |      1 +
 sprint/data/final_flash_repair_complete.json           |      1 +
 sprint/data/final_image_default_check.json             |      7 +
 sprint/data/final_summary.txt                          |      9 +
 sprint/data/finalization_complete.json                 |      1 +
 sprint/data/gate_unmet_categories.json                 |     18 +
 sprint/data/gemv15_icarus.json                         |     58 +
 sprint/data/gemv15_verilator.json                      |     58 +
 sprint/data/gemv_icarus.json                           |     60 +
 sprint/data/gemv_verilator.json                        |     60 +
 sprint/data/golden_semantics_audit.json                |      5 +
 sprint/data/golden_semantics_final.diff                |      0
 sprint/data/measured_bound_script_sha256.txt           |      1 +
 sprint/data/metadata_baseline.json                     |      1 +
 sprint/data/metadata_implementation.diff               |    122 +
 sprint/data/metadata_preflight.csv                     |     80 +
 sprint/data/p0_owner_log_audit.csv                     |     29 +
 sprint/data/packer_plan_identity.json                  |      5 +
 sprint/data/planned_packer_normalized_sha256.txt       |      1 +
 sprint/data/postfix_pair_icarus.json                   |     14 +
 sprint/data/postfix_pair_verilator.json                |     14 +
 sprint/data/qspi_before_rng_fix_icarus_capture.json    |   2256 +
 sprint/data/qspi_before_rng_fix_verilator_capture.json |   2256 +
 sprint/data/qspi_flash_model_idle_data.sv              |    186 +
 sprint/data/qspi_flash_model_single_process.sv         |    188 +
 sprint/data/qspi_icarus.json                           |   2257 +
 sprint/data/qspi_verilator.json                        |   2257 +
 sprint/data/qspi_verilator_tmpfull.json                |   2134 +
 sprint/data/remaining_fault_recovery.json              |     12 +
 sprint/data/restarted_jobs.json                        |      4 +
 sprint/data/rms_icarus.json                            |     62 +
 sprint/data/rms_verilator.json                         |     62 +
 sprint/data/sprint_input_hashes.json                   |      9 +
 sprint/data/uart_compact_failure.json                  |    134 +
 sprint/data/uart_icarus.json                           |    127 +
 sprint/data/uart_reset_diagnostic.json                 |     11 +
 sprint/data/uart_reset_diagnostic_initial.json         |     11 +
 sprint/data/uart_reset_icarus_timing.json              |     30 +
 sprint/data/uart_verilator.json                        |    127 +
 sprint/data/vec_icarus.json                            |    245 +
 sprint/data/vec_verilator.json                         |    245 +
 sprint/data/wall_deadline.json                         |      1 +
 sprint/scripts/add_metadata_check.py                   |      3 +
 sprint/scripts/adopt_big.py                            |     11 +
 sprint/scripts/argmax_bound.py                         |     81 +
 sprint/scripts/baseline_packer_reproduction.py         |      6 +
 sprint/scripts/bound_domain_proof.py                   |     12 +
 sprint/scripts/bound_math.py                           |      8 +
 sprint/scripts/bound_math_property.py                  |     12 +
 sprint/scripts/cache_versions.py                       |     19 +
 sprint/scripts/capture_reset_trace.py                  |      9 +
 sprint/scripts/check_image_defaults.py                 |      7 +
 sprint/scripts/clean_linux_builds.py                   |     12 +
 sprint/scripts/close_sprint.py                         |     20 +
 sprint/scripts/compact_big.py                          |     13 +
 sprint/scripts/compare.py                              |     23 +
 sprint/scripts/compile_error_contract.py               |      2 +
 sprint/scripts/complete_driver.py                      |     10 +
 sprint/scripts/crosscheck.py                           |     22 +
 sprint/scripts/cycle_counter_audit.py                  |      7 +
 sprint/scripts/deterministic_stimulus.py               |      7 +
 sprint/scripts/engineering_driver.py                   |     72 +
 sprint/scripts/engineering_launcher.py                 |      4 +
 sprint/scripts/experimental_scope.py                   |      2 +
 sprint/scripts/extend_repair_deadline.py               |      3 +
 sprint/scripts/fair_metadata_benchmark.py              |     10 +
 sprint/scripts/fast_fault_recovery.py                  |     13 +
 sprint/scripts/fault_dependency_review.py              |      4 +
 sprint/scripts/fault_needle_audit.py                   |     16 +
 sprint/scripts/final_findings_review.py                |      7 +
 sprint/scripts/final_flash_repair.py                   |     40 +
 sprint/scripts/final_qspi_validation.py                |     16 +
 sprint/scripts/final_report_refinement.py              |      5 +
 sprint/scripts/finalize.py                             |     32 +
 sprint/scripts/fix_bound_metadata.py                   |     10 +
 sprint/scripts/fix_preflight_path.py                   |      2 +
 sprint/scripts/fix_timescale.py                        |      2 +
 sprint/scripts/fix_wall_deadline.py                    |     15 +
 sprint/scripts/fresh_bound_reproduction.py             |     14 +
 sprint/scripts/golden_semantics_audit.py               |      6 +
 sprint/scripts/harden_bound_arithmetic.py              |     10 +
 sprint/scripts/implement_backend.py                    |     12 +
 sprint/scripts/implement_metadata.py                   |     25 +
 sprint/scripts/input_manifest.py                       |      5 +
 sprint/scripts/isolated_postfix_decode.py              |     15 +
 sprint/scripts/linux_build_cache.py                    |      9 +
 sprint/scripts/metadata_headline.py                    |      9 +
 sprint/scripts/metadata_preflight.py                   |     42 +
 sprint/scripts/overlap_bound_review.py                 |      8 +
 sprint/scripts/owner_fault_completion.py               |      4 +
 sprint/scripts/owner_final_adjustment.py               |      6 +
 sprint/scripts/owner_script.py                         |      7 +
 sprint/scripts/package_report_stats.py                 |     22 +
 sprint/scripts/postfix_decode_pair.py                  |      9 +
 sprint/scripts/precise_report_scope.py                 |     11 +
 sprint/scripts/prioritize_final_validation.py          |     10 +
 sprint/scripts/promote_fresh_qspi.py                   |     10 +
 sprint/scripts/prototype_flash_output_fix.py           |     14 +
 sprint/scripts/prototype_idle_data.py                  |      4 +
 sprint/scripts/qspi_failure.py                         |     13 +
 sprint/scripts/record_bound_domain.py                  |      5 +
 sprint/scripts/recover_backends.py                     |     10 +
 sprint/scripts/recover_determinism.py                  |     12 +
 sprint/scripts/recovery_barrier.py                     |      4 +
 sprint/scripts/regress15.py                            |     13 +
 sprint/scripts/remaining_fault_recovery.py             |      9 +
 sprint/scripts/report.py                               |    151 +
 sprint/scripts/report_gate_scope.py                    |      5 +
 sprint/scripts/report_review.py                        |      6 +
 sprint/scripts/report_skip_headline.py                 |      3 +
 sprint/scripts/restart_finalizer.py                    |      3 +
 sprint/scripts/runner_cli.py                           |     24 +
 sprint/scripts/signed_expansion_guard.py               |      5 +
 sprint/scripts/state_repair_complete.py                |      2 +
 sprint/scripts/temp_cleanup_fix.py                     |     12 +
 sprint/scripts/test_compact_metadata.py                |     21 +
 sprint/scripts/timebox_close.py                        |     33 +
 sprint/scripts/uart_reset_detail.py                    |     14 +
 sprint/scripts/uart_reset_diagnostic.py                |     21 +
 sprint/scripts/uart_reset_icarus.py                    |     11 +
 sprint/scripts/unique_prompt_repair.py                 |     30 +
 sprint/scripts/watch_compact_big.py                    |      7 +
 sprint/scripts/watch_final_repair.py                   |      2 +
 sprint/scripts/watch_unique.py                         |      9 +
 tools/board/run_uart_rtl.py                            |     19 +-
 tools/decode/fault_inject.py                           |      1 +
 tools/decode/pack_decode_image.py                      |     13 +-
 tools/decode/provenance.py                             |     30 +-
 tools/decode/run_act_quant_rtl.py                      |      1 +
 tools/decode/run_attn15_rtl.py                         |      1 +
 tools/decode/run_attn_rtl.py                           |      1 +
 tools/decode/run_decode_rtl.py                         |     16 +
 tools/decode/run_gemv_rtl.py                           |      1 +
 tools/decode/run_qspi_rtl.py                           |      7 +-
 tools/decode/run_rms_rtl.py                            |      1 +
 tools/decode/run_vec_rtl.py                            |      1 +
 tools/decode/sim.py                                    |     73 +-
 tools/decode/tb/qspi_flash_model.sv                    |      4 +-
 tools/decode/test_gate.py                              |     32 +
 200 files changed, 235744 insertions(+), 391 deletions(-)
```
Before: implicit unit TB time precision and compiler-version-only provenance. After: shared backend, explicit precision, mandatory runtime/tool identities. Representative GEMV cycle difference 101.2654 ->100.2654 is a TB sampling correction and both new backends must agree; not a hardware performance gain. Production arithmetic source diff must stay empty for firmware/host/numerics_v2/runq_runtime.py and fixed.py. New metadata measurements, if complete, are separated from research estimates above.

Per-package source diff counts below are DERIVED by summing git diff --numstat e8d9282 for the listed groups; package scripts/data are separately visible in the complete stat.
| Package | Files DERIVED | Lines added DERIVED | Lines removed DERIVED |
| --- | --- | --- | --- |
| P1 | 10 | 108 | 13 |
| P2 | 2 | 59 | 3 |
| P3 | 1 | 81 | 0 |
| P4 | 4 | 16 | 14 |
| P5 | 1 | 6 | 0 |
| P6 | 17 | 739 | 359 |

## OWNER_RUN2.ps1
NOT RUN: `powershell -ExecutionPolicy Bypass -File OWNER_RUN2.ps1`. Requires Vivado2025.2, the models junction, generated LUTs, matching final packed image, and successful final fast artifact regeneration. All jobs sequential; build/owner_logs2/<job>.log records START/EXIT, first failure throws, strict gate is last. No bitstream generation.

```powershell
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$logDir = Join-Path $root 'build/owner_logs2'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Logged([string]$Name, [string]$Exe, [string[]]$CommandArgs) {
    $log = Join-Path $logDir ($Name + '.log')
    $raw = Join-Path $logDir ($Name + '.output.tmp')
    "[$(Get-Date -Format o)] START $Exe $($CommandArgs -join ' ')" | Set-Content -LiteralPath $log
    Remove-Item -LiteralPath $raw -Force -ErrorAction SilentlyContinue
    & $Exe @CommandArgs *> $raw
    $rc = $LASTEXITCODE
    if (Test-Path -LiteralPath $raw) { Get-Content -LiteralPath $raw | Add-Content -LiteralPath $log; Remove-Item -LiteralPath $raw -Force }
    "[$(Get-Date -Format o)] EXIT $rc" | Add-Content -LiteralPath $log
    if ($rc -ne 0) { throw "$Name failed with exit code $rc. See $log" }
}

$py = 'python'
Invoke-Logged -Name 'pack_15m' -Exe $py -CommandArgs @('tools/decode/pack_decode_image.py','--model','15m')
Invoke-Logged -Name 'wrap_15m' -Exe $py -CommandArgs @('tools/decode/gen_15m_wrap.py')
Invoke-Logged -Name 'wrap_qspi' -Exe $py -CommandArgs @('tools/decode/gen_qspi_wrap.py')
# Finish timeboxed fault-provenance regeneration before physical closure.
$env:UTPU_SIM = 'icarus'
foreach ($unit in @('qspi','rms','vec','attn','attn15')) {
    Invoke-Logged -Name ('fault_' + $unit) -Exe $py -CommandArgs @('tools/decode/fault_inject.py',$unit)
}
$synthJobs = @(
    @('gemv_unit','decode_gemv_unit','rtl/decode/gemv_unit.sv'),
    @('act_quant','act_quant','rtl/decode/act_quant.sv'),
    @('rmsnorm','rmsnorm','rtl/decode/decode_pkg.sv','rtl/decode/seq_mul.sv','rtl/decode/rmsnorm.sv'),
    @('vec_alu','vec_alu','rtl/decode/decode_pkg.sv','rtl/decode/vec_alu.sv'),
    @('attention','attention','rtl/decode/decode_pkg.sv','rtl/decode/mul_pipe.sv','rtl/decode/div_const_pipe.sv','rtl/decode/attention.sv'),
    @('top_decode_260k','top_decode','rtl/decode/decode_pkg.sv','rtl/decode/seq_mul.sv','rtl/decode/mul_pipe.sv','rtl/decode/div_const_pipe.sv','rtl/decode/gemv_unit.sv','rtl/decode/act_quant.sv','rtl/decode/rmsnorm.sv','rtl/decode/vec_alu.sv','rtl/decode/attention.sv','rtl/decode/top_decode.sv'),
    @('top_decode_15m','top_decode_15m','rtl/decode/decode_pkg.sv','rtl/decode/seq_mul.sv','rtl/decode/mul_pipe.sv','rtl/decode/div_const_pipe.sv','rtl/decode/gemv_unit.sv','rtl/decode/act_quant.sv','rtl/decode/rmsnorm.sv','rtl/decode/vec_alu.sv','rtl/decode/attention.sv','rtl/decode/top_decode.sv','build/decode/top_decode_15m_wrap.sv'),
    @('qspi_reader','qspi_reader','rtl/decode/qspi_reader.sv'),
    @('top_decode_15m_qspi','top_decode_15m_qspi','rtl/decode/decode_pkg.sv','rtl/decode/seq_mul.sv','rtl/decode/mul_pipe.sv','rtl/decode/div_const_pipe.sv','rtl/decode/gemv_unit.sv','rtl/decode/act_quant.sv','rtl/decode/rmsnorm.sv','rtl/decode/vec_alu.sv','rtl/decode/attention.sv','rtl/decode/top_decode.sv','rtl/decode/qspi_reader.sv','build/decode/top_decode_15m_qspi_wrap.sv')
)
foreach ($j in $synthJobs) { $cmdline=@('tools/decode/synth_ooc.py') + [string[]]$j; Invoke-Logged -Name "synth_$($j[0])" -Exe $py -CommandArgs $cmdline }
Invoke-Logged -Name 'boolean_top_impl' -Exe $py -CommandArgs @('tools/board/impl_top.py')
Invoke-Logged -Name 'boolean_pin_check' -Exe $py -CommandArgs @('tools/board/pin_check.py')
Invoke-Logged -Name 'strict_gate' -Exe $py -CommandArgs @('tools/decode_gate.py')
Write-Output 'OWNER_RUN2 complete'
```
## Next three items
| Priority | Work | Concrete acceptance criterion | Agent-hours ESTIMATED |
| --- | --- | --- | --- |
| 1 | Complete physical closure of compact metadata | OWNER_RUN2 finishes with strict unmet[]; routed timing and resources fit; new image and matching RTL used together | 2-4, plus tool runtime |
| 2 | Widen total-cycle instrumentation and preserve long-run evidence | A directed >2^31-clock counter test passes, total equals sum of STEP counters, and affected artifacts regenerate without output changes | 1-2 |
| 3 | Screen tighter progressive classifier bounds before more RTL | Actually skip blocks on the same 50x120 dataset, preserve exact argmax/max-logit and improve byte/cycle estimates after metadata and seek costs | 4-8 |

## NOT RUN / incomplete list
- OWNER_RUN2 Vivado and final physical gate: exact command above.
- If P1/P4 acceptance is incomplete, metadata hardware optimization is not declared complete merely because a patch script exists.
- Full baud/reset/context boundary expansion, hardware bounded argmax, FIFO-depth A/B prefetch experiment: NOT RUN; no feature or result claimed.
- `python tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --flash qspi --gen 8 --sim icarus --output sprint/data/15m_qspi_icarus.json`: NOT RUN (long Icarus experiment).
