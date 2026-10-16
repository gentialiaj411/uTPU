# Phase 3 serialized vector microarchitecture plan
1. VADD_RQ states: READ_A, READ_B, ALIGN_A, ALIGN_B, ADD, RNE, SAT_WRITE; one packed word per pass, zero multipliers.
2. VPOS_ADD uses the VADD_RQ pipeline with POSITION_READ in place of READ_B; one packed word per pass, zero multipliers.
3. VRMSNORM states: SUMSQ, MEAN, LOG_INIT, LOG_SCAN, LUT_ADDR, LUT_READ, NEWTON_T, NEWTON_R, SCALE_MUL, SCALE_RNE, SCALE_WRITE. LOG_SCAN retires one even-exponent threshold comparison per cycle; it is not a priority-encoder loop.
4. VRMSNORM consumes one packed word in SUMSQ and SCALE passes; each multiply has registered inputs and result.
5. VRMSNORM uses one shared multiplier, with at most one multiply active in every registered FSM state.
6. VSOFTMAX states: MAX, EXP_ADDR, EXP_READ, SUM, DIV_START, DIV_WAIT, DIV_WRITE for each causal row.
7. VSOFTMAX consumes one score element in MAX and EXP/SUM; divider emits one quotient bit per cycle.
8. The iterative divider has registered numerator/remainder state, 24 iterations, and no multiplier.
9. Only one VEC descriptor runs at once, so the RMSNorm multiplier is shared across all vector suboperations.
