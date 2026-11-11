# QSPI fast-clock study: Real Digital Boolean

Date: 2026-11-09  
Scope: `xc7s50csga324-1`, on-board `S25FL128S`, 50 MHz decode core. No Vivado run,
board operation, or long end-to-end simulation was performed for this study.

## Executive result

The only credible path to approximately 1 byte/core-cycle is **EDh DDR at a
50 MHz SCK**, with a real capture phase and per-lane input delay calibration.
The current core-edge capture scheme cannot simply be doubled: the worst-case
fixed-edge budget is `10 - 7.5 - 8.0 - 0.5 = -6.0 ns`. A calibrated capture
point can instead use the data eye; using the current conservative numbers gives
an eye of roughly `10 + 2.0 - 8.0 - 0.5 = 3.5 ns` before FPGA setup/routing
margin. That is plausible, but it is not timing closure.

Recommendation: **GO for a small hardware characterization/prototype, not for
an RTL merge yet**. The first de-risk measurement is a scope/logic-analyzer
capture of CCLK, all four DQ lanes, and the FPGA-side capture clock while sweeping
capture phase/IDELAY. No scope is strictly required: an on-FPGA self-test bitstream can read a known flash
region (or the DLP pattern) at every MMCM phase / IDELAY tap, compare against the known
bytes in hardware, and report a per-lane pass/fail map over UART. Accept only if every lane has a repeatable eye at hot/cold
operating points and the eventual Vivado report closes both edges. No prototype
was implemented because this requires vendor SelectIO/MMCM timing and cannot be
validated by the permitted short RTL test alone.

## Facts and assumptions

The existing reader is EDh DDR at 25 MHz SCK (`HALF=1` from a 50 MHz core),
LC=00, QUAD=1, DLP disabled, and reports 0.500063 B/core-cycle. The current
implementation has DDR WNS +1.484 ns. The board XDC models S25FL128S `tV(max) =
8.0 ns`, `tHO(min) = 2.0 ns`, trace allowance 0.5 ns, and STARTUPE2
`TUSRCCLKO = 0.50..7.50 ns`.

The Rev. *T S25FL128S table lists EDh DDR Quad I/O Read at a maximum 80 MHz
command clock. Its DDR latency table gives, for EDh, mode=1 and dummy=3 at
LC=11 / <=50 MHz, and mode=1 and dummy=6 at LC=00 / <=66 MHz. Therefore 50 MHz
EDh is within the part limit. The exact ordering suffix still matters for the
HPLC/EHPLC table selection, so the board JEDEC ID and CR1 must be recorded.

The current LC=00 policy is the safer boot choice. It is already valid for 50 MHz
EDh under the <=66 MHz row and avoids changing the flash's power-up/default
latency contract. QUAD must remain set. VDLR/NVDLR are both 00h on the shipped
part in the current model, so no DLP is inserted. A volatile VDLR training value
could be written after configuration, but programming NVDLR is OTP and should
not be part of this work. If DLP is enabled, the datasheet requires a latency
code with at least five dummy cycles; LC=00 provides six at this clock rate.

Bank 14 is an HR bank on Spartan-7 and supports 3.3 V LVCMOS. UG471 documents
IDELAYE2 and ISERDESE2 in the SelectIO input path. Thus the four DQ pins are
eligible for IDELAY-based capture; a bank-level IDELAYCTRL/reference clock and
appropriate placement constraints are still required. The board's dedicated
CCLK path remains through STARTUPE2/USRCCLKO, with the first three user clock
cycles consumed by the existing startup-pulse mechanism.

Sources: [Infineon S25FL128S/S25FL256S Rev. *T datasheet](https://www.infineon.com/assets/row/public/documents/10/49/infineon-s25fl128s-s25fl256s-128-mb-16-mb-256-mb-32-mb-fl-s-flash-spi-multi-io-3-v-datasheet-en.pdf?fileId=8ac78c8c7d0d8da4017d0ecfb6a64a17), [AMD UG470](https://docs.amd.com/api/khub/documents/FOs3lXmlcWxBhTIFxVKyGA/content), [AMD DS189](https://docs.amd.com/r/en-US/ds189-spartan-7-data-sheet), [AMD UG471](https://docs.amd.com/api/khub/documents/IbGcnPFe6eF19mah_Y~IA/content), and the [Real Digital Boolean reference manual](https://realdigital.org/doc/02013cd17602c8af749f00561f88ae21).

## Options

### (a) EDh DDR, 50 MHz SCK: target 1 B/core-cycle

**Feasible? Conditional GO for characterization; NO-GO for immediate production RTL.**

At 50 MHz SCK, each 10 ns edge interval carries one nibble, or one byte every
20 ns / 1 byte per 20 ns of core time only if the core and SCK edge schedule are
properly decoupled. In throughput terms this is 1 B per 50 MHz core cycle. A
plain sample on the next core edge is not valid under worst-case STARTUPE2 plus
flash delay. A phase-shifted MMCM capture clock, IDELAYE2 on each DQ, and either
ISERDESE2 or a dual-edge input sampler can center the sample in the approximately
3.5 ns conservative eye. The capture clock must be constrained to the actual
USRCCLKO-generated SCK, not treated as an unrelated ideal clock.

The DLP is useful as a calibration pattern, but it changes the dummy protocol.
The first implementation should retain LC=00, QUAD=1, VDLR=00, and use a known
data-training command or an explicitly enabled volatile DLP only during a
calibration phase. Retraining after temperature/voltage changes is advisable;
there is no reason to touch OTP NVDLR. CR1 LC=11 is not required at 50 MHz and
should not be used merely to save six dummy cycles.

Risk is medium-high: source-clock delay, DQ skew, IOB/IDELAY placement, bus
turnaround, capture-clock phase, and asynchronous transfer all need measured or
implemented evidence. The existing +1.484 ns WNS does not cover this path.

Likely change size: medium RTL change (capture primitive wrapper, calibration
state, byte assembly, CDC FIFO, and a mode parameter), medium/large XDC change
(MMCM generated/phase clocks, IDELAYCTRL, per-edge input delays, and false-path/
CDC exceptions). Keep the current reader as the default.

Hardware de-risk: probe CCLK and DQ0..DQ3 at the flash/FPGA breakout, sweep
MMCM phase and IDELAY taps, and log the error-free tap window at room, hot, and
cold conditions. A good result is a stable multi-tap window on every lane,
not merely one passing setting.

### (b) 6Bh SDR, 50 MHz SCK

**NO-GO as a bandwidth improvement.** 6Bh SDR still returns one nibble per SCK
cycle, so at 50 MHz SCK it remains 0.5 B/core-cycle. It may be useful as a
separate signal-integrity experiment, but it does not address the target.

Risk is low-to-medium; change size is small-to-medium (clock generation and
output/input timing), but projected throughput is unchanged. The measurement
would be a normal SDR eye/scope check, not a performance de-risk.

### (c) Separate 66--80 MHz DDR clock domain plus async FIFO

**66 MHz: NO-GO for the first implementation; conditional future GO. 80 MHz:
NO-GO on this board without unusually strong measured evidence.**

The flash command limit permits EDh to 80 MHz, but the board timing becomes the
constraint. With the same conservative `tV=8 ns`, `tHO=2 ns`, and 0.5 ns skew
allowance, the nominal data eye is approximately:

| SCK | edge interval | eye estimate before FPGA margin | ideal stream rate |
|---:|---:|---:|---:|
| 50 MHz | 10.000 ns | 3.5 ns | 1.00 B/core-cycle |
| 66 MHz | 7.576 ns | 1.08 ns | 1.32 B/core-cycle |
| 80 MHz | 6.250 ns | -0.25 ns | 1.60 B/core-cycle |

66 MHz needs the same phase/IDELAY/ISERDES machinery plus an MMCM-generated
source clock, async FIFO, clean stop/backpressure behavior, and CDC timing
exceptions. 80 MHz has no positive eye under these conservative numbers even
though the flash data-sheet frequency maximum is 80 MHz. The realistic board
gain is therefore not worth making 66/80 the first target. The de-risk
measurement is the same phase/tap sweep, now with a required margin after adding
package, routing, jitter, voltage, and temperature effects.

Likely change size: large RTL/XDC change. The FIFO must absorb burst rate and
the reader must stop the source clock only at safe byte boundaries. Projected
throughput is compute-bound in the supplied model, so the extra nominal link
rate does not improve the headline result beyond 50 MHz DDR.

### (d) Other on-board bandwidth sources

**NO-GO.** The Boolean board reference material lists one 16 MB QSPI flash and
does not list a second flash or microSD/eMMC device. The other peripherals
(HDMI, optional BLE, ADC, LEDs, Pmods, and display) are not an on-board weight
store with a comparable local bus. An external Pmod memory would be a board
extension, not an alternate on-board source, and would add wiring/protocol and
signal-integrity work. No realistic second source exceeds the existing QSPI
path within the requested scope.

## Throughput projection

The cycle model is the existing measured/projection model: for each token,
streamed bytes consume `bytes / sustained B-per-core-cycle`, seek overhead is
retained, and the result is bounded below by the measured compute floor of
12,338,619 core cycles/token. It is not a new board measurement.

Using the supplied current 0.5 B/core-cycle flash-bound result:

| workload | current 0.5 B/cycle | projected 1.0 B/cycle stream | bounded result | projected tok/s at 50 MHz |
|---|---:|---:|---:|---:|
| 15M | 21,744,985 cycles/token, measured model, 2.30 tok/s | about 10,872,493 cycles/token before compute | 12,338,619 cycles/token | **4.05** |
| pruned K=8192 | 6,613,245 stream B/token = 13,226,490 stream cycles/token, 3.78 tok/s DERIVED | 6,613,245 stream cycles/token | 7,734,699 cycles/token (pruned ideal-stream MEASURED, docs/VOCAB_PRUNE_STUDY.md) | **6.46** |

Both values are projections (`50,000,000 / 12,338,619` and `50,000,000 / 7,734,699`)
and are compute-floor limited. The pruned image has its own, lower compute floor
(the classifier GEMV shrinks to 8,192 rows), so 1 B/cycle moves it from flash-bound
to compute-bound: 2.30 -> 6.46 tok/s (2.8x) for pruning plus 50 MHz DDR together.
A 66 or 80 MHz stream cannot raise either number unless the compute floor also changes. Existing ideal-stream simulations around 3.76 tok/s and
the 25 MHz DDR board estimate must not be described as hardware measurements.

## Draft constraints for a future 50 MHz DDR prototype

This is intentionally a draft, not installed in the active XDC:

```tcl
# Names are placeholders until the MMCM/IDELAY wrapper is implemented.
create_generated_clock -name qspi_sck_50 \
  -source [get_pins u_qspi_clk/mmcm/CLKOUT0] \
  -divide_by 1 [get_pins u_qspi_clk/startup/USRCCLKO]
create_generated_clock -name qspi_capture_100 \
  -source [get_pins u_qspi_clk/mmcm/CLKOUT1] [get_pins u_qspi_capture/CLK]
set_clock_latency -source -early 0.5 [get_clocks qspi_sck_50]
set_clock_latency -source -late  7.5 [get_clocks qspi_sck_50]
set_input_delay -clock qspi_sck_50 -max 8.5 [get_ports {qspi_dq[*]}]
set_input_delay -clock qspi_sck_50 -min 2.0 [get_ports {qspi_dq[*]}]
# Replace with the actual IDELAY/ISERDES and CDC paths after placement.
set_false_path -from [get_clocks qspi_sck_50] -to [get_clocks sys_clk]
set_false_path -from [get_clocks sys_clk] -to [get_clocks qspi_capture_100]
```

The `set_false_path` lines are placeholders only: they must not hide the
source-synchronous DQ capture path or the FIFO synchronizers. The final XDC
needs explicit rise/fall checks, IDELAYCTRL timing, clock uncertainty/jitter,
and a reviewed CDC report. The existing XDC's output/input delay assumptions
should be retained as the starting electrical limits, not copied blindly to a
dual-edge interface.

## Final verdict

Pursue 50 MHz EDh DDR with calibrated capture as the next experiment. Do not
merge a fast reader until the phase/tap eye measurement and implemented timing
report pass across temperature/voltage. Keep the current 25 MHz EDh reader,
LC=00, QUAD=1 default unchanged. Do not spend the first iteration on 66/80 MHz,
SDR 6Bh, or an alternate memory source.

