# Numeric domain of the integer decode runtime

The integer runtime (`firmware/host/numerics_v2/runq_runtime.py`, `fixed.py`) is the spec and the decode RTL matches it
bit for bit. This note does not change any result. It records where that equivalence is exact by construction, where it
is exact only on a measured reachable range, and where the RTL and the runtime do different things on inputs that no
real run produces.

Everything below is produced by `tools/decode/numeric_domain.py` (JSON: `build/decode/numeric_domain.json`, gitignored)
and asserted in `firmware/host/numerics_v2/test_numerics_v2.py` (`test_div_const_rne_*`, `test_reachable_div_const_ranges_*`,
`test_quantize_rejects_int32_min_and_accepts_int32_max`).

Method. Every `fixed.div_const_rne` call site and every `_quantize` input is instrumented while the runtime decodes real
eval text: the 64 evenly spaced eval-prompt start offsets of `run_260k.prompts_from`, extended to real-text windows
(teacher forced; classifier run every 8th step). Run recorded here: stories260K, 32 windows x 64 tokens; stories15M, 12 windows
x 48 tokens; each at KV8 and KV4 (about 770k quantizer groups in total). Observed
maxima are lower bounds on the reachable range; the "operand-width" bounds are upper bounds.

## F10: `div_const_rne` is exact RNE only for |n| <= 2^31

`div_const_rne(n, d)` multiplies by a Q32 reciprocal `R` of `d` and applies a bounded +-1 correction.

* Narrow path, |n| <= 2^31: exactly round-half-to-even. Checked against exact integer RNE on [-2^22, 2^22] exhaustively plus
  400k random values over the whole narrow range plus the 2^31 boundary and tie neighbourhoods, for d in {7, 64, 127, 288}:
  0 mismatches.
* Wide path, |n| > 2^31: the estimate is rounded half up and drifts from `n/d` by `|n| * |R/2^32 - 1/d|`. One +-1 correction
  cannot absorb that, so the result is approximate. Auditor example: the mean of 288 squares of 65536 has numerator
  `288 * 2^32`; `div_const_rne` returns 4294967327, exact RNE is 4294967296 (error 31). Measured max error follows the
  first-order model `max(0, hi * |R/2^32 - 1/d| - 1)` to within 1 (table below). The prior docstring and the comment in
  `_rms_high` called this exact and claimed the runtime asserts |n| <= 2^31; neither is true and both are fixed.
  The wide path is not changed (the RTL matches it bit for bit and every artifact depends on it).

### Reachable ranges per call site

`max |n|` is the observed maximum over the runs above (the larger of KV8/KV4 and model). "Operand-width bound" is the
worst case from operand widths (activation peaks <= 2^31, accumulators as asserted); "max error" is measured with the real
`fixed.div_const_rne` against exact RNE over the whole wide range (2^31, bound]: boundaries, tie neighbourhoods, 400k
log-uniform and 400k near-top random samples per row.

| Call site (denominator) | Observed max \|n\| (260K / 15M) | Observed wide-path values | Operand-width bound (what the error row covers) | Max error vs exact RNE at that bound | Verdict |
|---|---|---|---|---|---|
| `_rms_high` mean, D=64 (260K) | 2^33.5 / - | 6313 (KV8) | ss = sum of D int32 squares: D*2^62, exceeds int64 (np.sum wraps); contract ss <= 2^40 | 0 (64 is a power of two, R exact) | exact; wide path reached but error-free |
| `_rms_high` mean, D=288 (15M) | - / 2^35.4 | 2896 (KV8) | same; contract ss <= 2^40 | 28 (first-order model 27.4) | wide path reached in real decode; observed error 0 (KV8) and 1 (KV4); bound 28 on a mean of about 2^32 (relative 6e-9) |
| `_to_base` (qmax=127, KV8) | 2^22.6 / 2^23.1 | 0 | code <= 127 times peak <= 2^31 = 2^38.0 | 55 | unreachable; 2^8 below the narrow limit |
| `_to_base` (qmax=7, KV4) | 2^18.5 / 2^19.0 | 0 | 7 * 2^31 = 2^33.8 | 1 | unreachable |
| `_score` first div127 (KV8 / KV4) | 2^27.3 / 2^24.0 (KV4: 2^23.2 / 2^19.8) | 0 | any score that passes the int32 OverflowError check: 127*127*2^31 = 2^45.0 (KV4: 127*7*2^31 = 2^40.8) | 7048 / 388 | unreachable; at least 2^3.7 below the narrow limit |
| `_score` second div (d=127, KV8 / d=7, KV4) | 2^20.3 / 2^17.0 (KV4: 2^16.2 / 2^12.9) | 0 | 127*2^31 = 2^38.0 / 7*2^31 = 2^33.8 | 55 / 1 | unreachable |
| `_attention_value` div127 | 2^22.3 / 2^22.5 | 0 | 16*32767*127 * 2^31 >> 15 = 2^42.0 | 887 | unreachable |
| `_linear_high` div127 (every linear layer and classifier) | 2^22.4 / 2^23.2 | 0 | per 64-wide group: accum <= 64*127*wmax, activation peak <= 2^31, model's largest multiplier/shift ratio: 2^44.7 (260K), 2^45.5 (15M) | 5821 / 9890 | unreachable; at least 2^7.8 below the narrow limit |

Reading the table. Only the RMSNorm mean ever reaches the wide path in real decode (both models at their real D), and
there the error is at most 1 observed and at most 28 over the contract range ss <= 2^40, which is 2^4.6 above the largest
sum of squares seen. All other sites stay at least 2^3.7 below 2^31 on every observed numerator, so they are exact RNE in
practice; their worst-case rows show what would happen only if activations were pushed far outside anything observed
(the `_linear_high`/`_score` rows are loose upper bounds because the int64 operand products can wrap before the check).
The wide-path error of the RMSNorm mean feeds `isqrt(mean << 30)` and a Q24 reciprocal; an error of <= 28 on a mean of
about 2^32 is a relative perturbation of about 6e-9 of the mean. Its effect on logits was not separately measured; the
RTL reproduces it exactly, so it is a property of the spec, not a mismatch.

The contract the tests assert: observed `max |n|` <= the bound above for every site; every non-RMSNorm site stays on the
exact narrow path (`wide_values == 0`); the RMSNorm wide-path error stays within the first-order model plus 1.

## F11: `_quantize` input domain

`_quantize` computes `peak = max(1, max|x|).astype(np.int32)`. At `x == -2^31` the peak is 2^31, the cast wraps to a
negative number, and `reciprocal_rne` raises `ValueError("denominator must be positive")`. So the runtime's `_quantize`
domain is `[-(2^31 - 1), 2^31 - 1]`.

Reachability. No producer clamps to that symmetric range:

* `_linear_high` (q, k, v, h1, h3, wo, w2 outputs) and `_attention_value` check `raw in [-2^31, 2^31 - 1]`, which admits
  exactly `-2^31`.
* `_rms_high` and `_hidden_high` end in `astype(np.int32)`, which wraps silently.

So `-2^31` is not provably unreachable. It is unreachable in practice: over the runs above the smallest quantizer input
is -183449 (2^17.5) and the largest 291735, zero elements equal INT32_MIN, and every op output is below 2^19 (see the
overflow table), about 12 bits from INT32_MIN. Contract: activations entering `_quantize` lie in
`[-(2^31 - 1), 2^31 - 1]`; `test_reachable_div_const_ranges_*` asserts the observed range is within +-2^28.

RTL vs golden at INT32_MIN. `act_quant.sv` tracks `|x|` as an unsigned 32-bit magnitude (`32'(-raw_in)`), so
`-2^31` gives peak 2^31, `scale_out = 0x80000000`, `recip = div_rne(127 << 24, 2^31) = 1` and code `clamp(rne(x/2^24))`.
The golden raises. The directed RTL cases (4 groups in `golden_act_quant.py`: INT32_MIN with qmax 127 in a 64-group, with
qmax 7 in a 48-group, INT32_MAX-with-INT32_MIN neighbour, and a 32-group) compare the RTL against the runtime's own
math with the peak kept in int64 (an extension of the domain, not a runtime behaviour). They pass: the RTL defines the
input as peak 2^31 and produces codes -127 / +127 for the +-2^31 elements. This divergence is unreachable and is a contract
edge, not a bug.

## Where the RTL keeps low bits and the runtime raises or casts

"Observed" is the largest absolute output of that operation over the runs above (260K and 15M, KV8 and KV4).

| Operation | Golden behaviour | RTL behaviour | Observed max abs | Verdict |
|---|---|---|---|---|
| `_quantize` peak cast | `ValueError` at x = -2^31 | unsigned 32-bit magnitude, peak 2^31, `scale_out = 0x80000000` | min input -2^17.5 | contract edge, unreachable; directed RTL case above |
| `checked_matmul` int32 dot product | `assert` on the static bound `k*127*wmax <= 2^31 - 1` (<= 64*127*127 = 2^20) | 32-bit `acc` in `gemv_unit` wraps | static proof, data independent | identical; cannot overflow |
| `_linear_high` scaling product `accum * peak * small` | int64, wraps silently | 64-bit registers, wraps identically | n/a (worst case 2^66 > int64 only with out-of-range peaks) | bit-exact including wrap |
| `_linear_high` result `raw` | `OverflowError` outside int32 | `out_acc` is 64-bit, `result_int` takes bits [31:0] | linear_high output 2^16.9 | diverges only where the golden would raise; 14 bits of margin |
| `_rms_high` `np.sum(x*x)` | int64, wraps | 64-bit `ss`, wraps identically | ss 2^35.4; wrap needs ss >= 2^63 | bit-exact including wrap |
| `_rms_high` negative wrapped mean | `math.isqrt` raises `ValueError` | sign applied to the quotient, `sq_n` taken as unsigned (47-bit root) | unreachable (needs ss wrap) | diverges only after an int64 wrap; unreachable |
| `_rms_high` output `astype(int32)` | wraps | low 32 bits | rms out 2^17.6 | identical |
| `_score` product chain and result | int64 wraps; `OverflowError` outside int32 | 64-bit chain, `sbuf` stores `p[31:0]` | score out 2^13.4 | diverges only where the golden would raise; 17 bits of margin |
| `_attention_value` accumulator `assert` and result | `assert` accum <= 2^31 - 1 (static, 16*32767*127); `OverflowError` outside int32 | 32-bit `acc`, 64-bit `raw`, out = `raw[31:0]` | attention out 2^15.6 | static proof; diverges only where the golden would raise |
| residual add `(x + y).astype(int32)` | wraps | 32-bit `a + b` | rms input 2^17.2 | identical (wrap tested in the vec runner, both geometries) |
| `_rope_high` `astype(int32)` | wraps | `32'(rshift_rne(...))` low 32 | rope out 2^15.4 | identical |
| `_silu_high`, `_hidden_high` products and `astype(int32)` | int64 wraps (silu * h3 * small can exceed 2^63), then wraps to int32 | low-64 emulated from split 32-bit products (`low64 = t2 + (t3 << 32)`), low 32 | silu 2^16.7, hidden 2^18.2 | bit-exact including the int64 wrap (random SwiGLU cases with 2^30 magnitudes) |
| `_embedding_high` `astype(int32)` | wraps | low 32 | embedding out 2^12.8 | identical |
| `div_const_rne` wide path | approximate (F10) | same approximation, bit for bit | rms only, see F10 | exact match to the golden; both differ from exact RNE |

## Geometry coverage of the unit tests (N07)

`vec_alu` was only unit-tested at the 260K RoPE half-width of 4 (HS=8), so a fault that corrupts output only when
`rope_half == 24` (15M, HS=48, ROM `rope_cs_q15_h48_p128.mem`, positions up to 127) could not be seen. `golden_vec.py` and
`run_vec_rtl.py` now run both geometries (`vec` and `vec15`): real captures from stories15M at RoPE positions 0, 1, 2, 3, 4,
7, 8, 15, 16, 31, 32, 63, 64, 100, 126, 127 (all layers, q and k), SwiGLU and embedding at positions 0, 1, 2, 127, plus directed
edges (int32 extremes at positions 0, 1, 64, 126, 127, RoPE rounding ties, wrap-around adds, mul-shift ties). The fault
`rope_h48_only_bitflip` in `fault_inject.py` flips an output bit only when `rope_half == 24`; it is invisible to the 260K
set (0 mismatches) and caught by the 15M set. `golden_rms.py` gained stories15M real RMSNorm captures (D=288, 15M gamma,
multiplier, shift and epsilon; 6 decoded tokens) and `golden_act_quant.py` stories15M real quantizer groups (lengths 64, 32 and 48
at 15M magnitudes, KV8 and KV4) plus the INT32_MIN directed groups above.
