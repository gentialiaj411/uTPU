# stories15M mixed-precision INT4 study

Fixed text SHA-256: `c8191cfe5a461bc825b3e3326a2921dfe978a63878b4c14547033b7217adea17` (MEASURED). Group size 32.

| config | int bpb (MEASURED) | delta | unpruned B/token (DERIVED) | pruned B/token (DERIVED) | pruned tok/s (DERIVED) | wall s |
|---|---:|---:|---:|---:|---:|---:|
| measured_baseline_integer | 0.562313190 | +0.000% | 10684413 | 6613245 | n/a | n/a |
| everything | 0.589832671 | +4.894% | 9025533 | 4954365 | 5.046 | 3112.7 |
| qkv | 0.572675311 | +1.843% | 9937917 | 5866749 | 4.261 | 4526.8 |
| wo | 0.570646606 | +1.482% | 10435581 | 6364413 | 3.928 | 3680.7 |
| w2 | 0.571220956 | +1.584% | 10020861 | 5949693 | 4.202 | 3138.6 |
| all_attention | 0.580138257 | +3.170% | 9689085 | 5617917 | 4.450 | 3490.2 |
| w2_qkv | 0.581749927 | +3.457% | 9274365 | 5203197 | 4.805 | 3417.4 |

Improvement method (MEASURED): non-GPTQ improved scale/clipping search (`improved=True`, `gptq=False`) was worse: everything 0.607202815 bpb; wo 0.570795902 bpb.
Pruned+INT4 everything (MEASURED): 0.591851090 bpb, restricted K=8192 softmax, 1450.4 s. The pruned-only integer reference is 0.564346 bpb (MEASURED in vocab-prune study).

Decision: **NO-GO**. Everything passes the speed leg at 1.335x pruned-only (DERIVED), but fails unpruned and pruned 1% quality thresholds.

INT4 does not reduce GEMV cycles in the existing RTL datapath; its throughput benefit here is stream bandwidth only (DERIVED from RTL parameters).

Reproduction (scripts on branch `int4`): `python tools/int4/build_artifacts.py`; `python tools/int4/integer_eval.py ...`; `python tools/int4/pruned_eval.py ...`; `python tools/int4/finalize_study.py`.

NOT RUN: GO-only packer option, byte-identical proof, golden, and new-image RTL simulation; exact skipped simulation command is recorded in `bench/results/int4_study.json`.
