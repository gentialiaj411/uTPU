# Sprint state
P0 DONE: integration e8d9282, all owner logs EXIT 0; gate/stale/restore verified.
P1 IN-PROGRESS: backend and CLI implemented; explicit TB timescale fixes a cross-simulator race. Baseline cross-check sweep and 15M regressions running.
P2 DONE code/tests: runtime and simulator identities plus fault model/golden inputs; 36 gate tests pass. Artifact regeneration deferred P6.
P3 DONE / NO-GO for tested bounds: repaired sweep contains 50 distinct prefixes and 6000 generated positions, with zero rows skipped and exact argmax/max-logit agreement. The original repeated-prefix data is excluded.
P4 TODO: implementation script prepared but deliberately not applied until baseline P1 runs finish; production arithmetic unchanged.
P5 TIMEBOXED: integrated RTL already independently prefetches through serial phases. No extra overlap RTL retained; design review in sprint/P5_PREFETCH_REVIEW.md.
P6 TODO: owner Vivado script drafted, no Vivado executed.
