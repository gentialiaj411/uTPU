# Sprint log
- Integration owner logs audited, all EXIT 0. Gate PASS, comment mutation stale FAIL, restore PASS. Committed audit-fixes e8d9282; perf worktree created from that commit. Owner artifacts preserved.
- Decision: no new Vivado execution; owner script will include synthesis and implementation with strict gate last.
- Backend finding: unit TB omitted timescale; Verilator initially produced 12,754 mismatches and stale-done/23 cycles per tile. Explicit 1ns/1ps precision for BOTH backends fixes the delta-cycle race: GEMV now passes both with 100.2654 cycles/tile and identical mutation count1006. Historical101.2654 differs because TB timing was unspecified. No arithmetic/tolerance changed.
- Version queries cached by executable content digest, not merely path or mtime. Executable hashes still recomputed for stale checks. Restarted gate tests after avoiding repeated slow compiler version invocations.
- P3 self-review caught a modeling bug: max(row bounds) cannot be computed using only groupwise block-header maxima. Archived initial CSV as INVALID; restarted with actual groupwise maximum coefficients computed from the declared block metadata. L1/L2 variants include all row norm/scale metadata traffic. No invalid CSV will support a performance claim.
- Budget boundary: no further speculative features. Existing baseline, bounded-argmax and engineering validation jobs will finish or timebox; final report will mark unmet acceptance explicitly. P5 is design-only because autonomous prefetch already exists. No numbers from the invalid initial P3 archive will be reported.
- P4 proof review: expanded metadata is consumed by an existing signed 32-bit multiplier cast. Therefore q<<16 is safe only for q<32768. Tightened the planned packer assertion to this range (still stored u16); accepting arbitrary u16 would be unsound. No arithmetic change required for the measured model range.
- Final dependency review also added packed-image producer and golden capture script sources to fault provenance; models and arithmetic sources alone did not cover their adapters. Final gate tests will be rerun after all production source changes settle.
- Build throughput finding: compiling each Verilator variant on /mnt/c dominated the QSPI sweep. Moved generated C++/objects to isolated /tmp/utpu-perf while keeping source and raw compile logs in this worktree. Stopped only own active QSPI runner and crosscheck orchestration, archived partial log, and resumed completed measurements unchanged; all QSPI variants rerun from scratch. No checks removed.
- P3 final screening completed all 50x120 positions, 216000 variant/rate records, exact argmax and maximum-logit checks passed. Norm and coefficient bounds skipped zero rows in every tested variant; no-go for implementing these bounds in hardware. This does not rule out tighter progressive or geometric bounds.
- Big 15M regression was still executing near its wrapper one-hour cap. Preserved the existing child simulation and replaced only its wrapper with an adoption waiter (45 minute additional cap), avoiding a wasteful restart. No stimulus/check changed; its runner-measured wall_seconds will be authoritative.
- Confirmed tool infrastructure failure: /tmp is a 7.7GB tmpfs and variant object/PCH directories exhausted it. Verilator silently wrote zero-byte makefiles, so negative QSPI and UART compilation failed with make No targets, correctly causing runner FAIL. Preserved failed outputs; cleaned only completed owned /tmp/utpu-perf/*_obj directories after checking active executable paths. Backend now removes each generated object directory after run/failure while preserving raw compile logs. Full QSPI/UART Verilator suites rerunning; no failure treated as a pass.
- Confirmed research data-quality hole: 50 prompt records represented only 9 distinct eight-token prefixes. Fixed P3 sampling to require 50 distinct prefixes; reused the nine distinct completed measurements and launched the remaining 41 prompts. The repeated-prefix CSV is archived and excluded from final evidence. All new positions remain <=127. P3 final no-go will be re-evaluated from the repaired sweep.

- 2026-10-30T14:22:17.306576 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/compare.py -> sprint/data/compare_final.log

- 2026-10-30T14:22:17.427267 compare_final EXIT 0

- 2026-10-30T14:22:17.443146 P4 BLOCKED by P1 full backend equivalence; arithmetic and packed format preserved.

- 2026-10-30T14:22:17.443146 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_gemv_rtl.py --sim icarus -> sprint/data/regen_gemv.log

- 2026-10-30T14:22:45.992823 regen_gemv EXIT 0

- 2026-10-30T14:22:45.992823 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_gemv_rtl.py --golden build/decode/golden_gemv15 --output bench/results/decode_gemv15_rtl.json --sim icarus -> sprint/data/regen_gemv15.log

- 2026-10-30T14:25:02.597677 regen_gemv15 EXIT 0

- 2026-10-30T14:25:02.597677 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_act_quant_rtl.py --sim icarus -> sprint/data/regen_actq.log
- P1 found real cross-simulator TB gaps after infrastructure recovery: sparse QSPI stimulus used multiple stateful RNG calls in the same task argument list (unspecified evaluation order); UART queue polling shared the active clock-edge region with monitor updates, producing a four-cycle summary skew. Split every multi-RNG argument into ordered assignments and sample UART task polls 0.1ns after the clock on BOTH backends. No stimulus cases, checks or tolerances removed; full QSPI/UART comparison rerunning. Paused downstream engineering until this acceptance is resolved.

- 2026-10-30T14:25:52.069456 Finalization: all launched jobs ended or were explicitly TIMEBOXED at wall deadline. timedout=False
- Final plan preserves a fair four-position 15M QSPI before/after benchmark. P6 may promote the complete final P1 QSPI producer output unchanged only after checking all current dependency hashes, required identity coverage and byte-identical SHA256; otherwise it reruns. P4 does not change the QSPI unit or its inputs. This saves redundant compilation without removing any test or editing an artifact.

- 2026-10-30T15:09:41.504788 Independent work alongside simulations: readonly compact-packer preview independently decoded all 79 model tensors and matched both effective multiplier arrays and integer forwards (two activations/tensor). Measured serialized 15M saving 949248 bytes. No production source was changed by this preflight. Artifact sprint/data/metadata_preflight.csv. Refreshed provenance tests and hashed model/text/prompt/numeric inputs.
- Validation scheduling decision: run touched-unit fault suites first (GEMV, top, board, QSPI), then remaining suites. Reserved final regeneration time also runs any not-yet-started fault suites before the unchanged strict gate; no failed suite is silently skipped or retried as a pass.

- 2026-10-30T15:15:30.045644 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/compare.py -> sprint/data/compare_final.log

- 2026-10-30T15:15:30.098550 compare_final EXIT 0

- 2026-10-30T15:15:30.103595 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --flash qspi --gen 1 --sim verilator --output sprint/data/15m_qspi_before_metadata.json -> sprint/data/15m_qspi_before_metadata.log

- 2026-10-30T15:19:40.906510 15m_qspi_before_metadata EXIT 0

- 2026-10-30T15:19:40.906510 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/implement_metadata.py -> sprint/data/metadata_apply.log

- 2026-10-30T15:19:41.943373 metadata_apply EXIT 0

- 2026-10-30T15:19:41.943373 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe -m pytest sprint/scripts/test_compact_metadata.py -q -> sprint/data/metadata_python_exactness.log

- 2026-10-30T15:19:42.412434 metadata_python_exactness EXIT 0

- 2026-10-30T15:19:42.412434 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 4 --flash qspi --gen 8 --sim verilator --output sprint/data/260k_metadata.json -> sprint/data/260k_metadata.log

- 2026-10-30T15:19:59.394172 260k_metadata EXIT 0

- 2026-10-30T15:19:59.395172 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --flash qspi --gen 1 --sim verilator --output sprint/data/15m_metadata.json -> sprint/data/15m_metadata.log
- Independent bound-proof review hardened coefficient arithmetic for the full declared input range: split c*a into Q32 limbs before multiplying by max-code, rather than forming a potentially 71-bit product in int64. Verified 10000 triples against Python arbitrary-precision ceil arithmetic. This is algebraically identical on the measured inputs, all of which passed the prior overflow guards; retained the measured script digest. No new skip or throughput result claimed.

- 2026-10-30T15:23:48.105939 15m_metadata EXIT 0

- 2026-10-30T15:23:48.106942 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_gemv_rtl.py --sim verilator -> sprint/data/regen_gemv.log

- 2026-10-30T15:23:59.896035 regen_gemv EXIT 0

- 2026-10-30T15:23:59.896035 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_gemv_rtl.py --golden build/decode/golden_gemv15 --output bench/results/decode_gemv15_rtl.json --sim verilator -> sprint/data/regen_gemv15.log

- 2026-10-30T15:24:17.498069 regen_gemv15 EXIT 0

- 2026-10-30T15:24:17.498069 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_act_quant_rtl.py --sim verilator -> sprint/data/regen_actq.log

- 2026-10-30T15:24:52.025782 regen_actq EXIT 0

- 2026-10-30T15:24:52.025782 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_rms_rtl.py --sim verilator -> sprint/data/regen_rms.log

- 2026-10-30T15:25:05.116019 regen_rms EXIT 0

- 2026-10-30T15:25:05.116019 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_vec_rtl.py --sim verilator -> sprint/data/regen_vec.log

- 2026-10-30T15:25:27.725979 regen_vec EXIT 0

- 2026-10-30T15:25:27.726978 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_attn_rtl.py --sim verilator -> sprint/data/regen_attn.log

- 2026-10-30T15:25:56.178571 regen_attn EXIT 0

- 2026-10-30T15:25:56.179572 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_attn15_rtl.py --sim verilator -> sprint/data/regen_attn15.log

- 2026-10-30T15:26:45.860314 regen_attn15 EXIT 0

- 2026-10-30T15:26:45.860314 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe sprint/scripts/promote_fresh_qspi.py -> sprint/data/regen_qspi.log

- 2026-10-30T15:26:46.172348 regen_qspi EXIT 0

- 2026-10-30T15:26:46.173354 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/board/run_uart_rtl.py --sim verilator -> sprint/data/regen_uart.log

- 2026-10-30T15:28:43.516943 regen_uart EXIT 1

- 2026-10-30T15:28:43.517929 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py gemv -> sprint/data/fault_gemv.log

- 2026-10-30T15:29:42.319396 fault_gemv EXIT 0

- 2026-10-30T15:29:42.319396 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py top -> sprint/data/fault_top.log
- Started the same 8-prompt x32-token 15M regression again after compact metadata was applied, with isolated generated inputs and an unused compile tag. The existing producer records the wrapper as an extra provenance input. This is required validation of the new image ABI, not a new numerical experiment; all golden comparisons remain unchanged.

- 2026-10-30T15:32:00.554116 fault_top EXIT 0

- 2026-10-30T15:32:00.554116 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py board -> sprint/data/fault_board.log

- 2026-10-30T15:47:13.439711 fault_board EXIT 1

- 2026-10-30T15:47:13.439711 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv8_p1.json -> sprint/data/regen_e2e_kv8_p1.log

- 2026-10-30T15:47:25.923164 regen_e2e_kv8_p1 EXIT 0

- 2026-10-30T15:47:25.924164 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 4 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv4_p1.json -> sprint/data/regen_e2e_kv4_p1.log

- 2026-10-30T15:47:38.203511 regen_e2e_kv4_p1 EXIT 0

- 2026-10-30T15:47:38.203511 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --period 2 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv8_p2.json -> sprint/data/regen_e2e_kv8_p2.log

- 2026-10-30T15:47:51.151281 regen_e2e_kv8_p2 EXIT 0

- 2026-10-30T15:47:51.151281 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --period 4 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv8_p4.json -> sprint/data/regen_e2e_kv8_p4.log

- 2026-10-30T15:48:06.562422 regen_e2e_kv8_p4 EXIT 0

- 2026-10-30T15:48:06.563420 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 260k --kv-bits 8 --flash qspi --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_kv8_p1_qspi.json -> sprint/data/regen_e2e_kv8_p1_qspi.log

- 2026-10-30T15:48:23.722310 regen_e2e_kv8_p1_qspi EXIT 0

- 2026-10-30T15:48:23.722310 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/run_decode_rtl.py --model 15m --kv-bits 4 --gen 8 --sim verilator --output bench/results/decode_e2e_rtl_15m_kv4_p1.json -> sprint/data/regen_e2e_15m_kv4_p1.log

- 2026-10-30T15:51:02.132996 regen_e2e_15m_kv4_p1 EXIT 0

- 2026-10-30T15:51:02.132996 RUN C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/decode/fault_inject.py qspi -> sprint/data/fault_qspi.log
- Validation found a real unresolved failure after compaction: UART main clean/mutation tests pass, but reset-mid-decode robustness returns six FF bytes for subsequent RDID; a previously declared equivalent watchdog-reader mutation is now detected. These failures remain failures. Running a minimal diagnostic with read-only flash/control traces; no stimulus/check was removed from the acceptance suite and no equivalence waiver was changed.
- Reset diagnosis narrowed the failure to simulation-model behavior: Icarus passes the identical robust scenario; under Verilator the flash instruction is correctly 9F and regbyte is correct, but flash_o retains an old array-data nibble through the register read. Prototyping a single timed output process for mutually exclusive register/array phases; all setup/hold/command checks and delays remain unchanged.
- Combining timed model writers did not resolve the Verilator reset failure; that prototype is NOT adopted. Testing the smaller interface correction: when flash_oe=0, drive a known inactive data value instead of a second redundant Z on flash_o. The external bus remains tri-stated by flash_oe, and every driven-X transition, timing check and data value is preserved.

- 2026-10-30T16:18:28.617218 FINAL REPAIR C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/board/run_uart_rtl.py --sim verilator
- Confirmed minimal fix: replacing only the two inactive flash_o=Z assignments (both paired with flash_oe=0) by a known inactive value makes Verilator pass the identical 113-check reset robustness scenario, matching Icarus. The external bus remains identically Z when disabled, and driven X windows/timing checks are untouched. Adopted this two-line model-interface correction, not the failed timed-process refactor. Regenerating both simulator protocol suites, UART cross-checks, touched and remaining faults, and QSPI e2e under the frozen final model. Extended validation budget by 30 minutes within the owner five-hour limit to finish this required repair.

- 2026-10-30T16:20:53.191726 regen_uart EXIT 0

- 2026-10-30T16:20:53.198890 FINAL REPAIR C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe tools/board/run_uart_rtl.py --sim icarus --output sprint/data/uart_icarus.json
