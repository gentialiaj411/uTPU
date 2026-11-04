# Boolean board bring-up

This is the first-power-on runbook for `xc7s50csga324-1` with the S25FL128S 16 MiB flash. It assumes a Windows host and the repository root shown below. Commands are intentionally copy/pasteable; Vivado operations are owner-run and were **NOT RUN** in this preparation.

1. **Install and inspect.** Install Vivado 2025.2 (or Vivado Lab 2025.2 with Hardware Manager), the board’s JTAG/USB driver, and Python 3.11. Real Digital documents one USB connection for power, UART/COM, and programming, but the public board page does not identify the USB-UART bridge IC; driver identity is **UNKNOWN—VERIFY from the board schematic/device manager**. Connect the PROG/UART micro-USB, do not attach the other uTPU worktrees, and note the COM port.

2. **Create the packed model image.**

   ```powershell
   $py='C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe'
   & $py tools/decode/pack_decode_image.py --model 15m
   ```

   Expected artifact: `build/decode/decode_image_15m/image.bin` and `params.json`.

3. **Build SDR (recommended first).**

   ```powershell
   & $py tools/board/build_bitstream.py
   & $py tools/board/make_flash_image.py
   ```

   `build_bitstream.py` runs `impl_top.py` once (including its check that `boolean_core` parameters match the image's `params.json`), refuses to continue unless timing is met and DRC has no Error/Critical Warning, then writes the bitstream from the routed checkpoint. `make_flash_image.py` runs `write_cfgmem`, then parses the `.mcs` back and checks that every weight byte at `0x280000` equals `image.bin` and that the bitstream's configuration data is intact.

   **Measured on 2026-11-04 (Vivado 2025.2, compressed):**
   - SDR `.bit`: 1,538,349 bytes, WNS +0.98 ns, WHS +0.015 ns, DRC clean.
   - The `.mcs` maps the bitstream to `0x000000..0x17792d` and the weights to `0x280000..0xe10040` (12,124,224 bytes).
   - All weight bytes were verified against `image.bin`, with no bit swapping.

4. **Program and boot from flash.** With the board powered by PROG/UART USB and the JTAG target visible:

   ```powershell
   & $py tools/board/program_flash.py build/board/sdr/boolean_top.mcs --dry-run
   & $py tools/board/program_flash.py build/board/sdr/boolean_top.mcs
   ```

   Inspect the dry-run Tcl before running it. The script loads Vivado's indirect-programming core, then erases, programs and verifies the flash, then reboots the FPGA from flash (`boot_hw_device`). A power cycle must also boot it.

   The cfgmem part `s25fl128sxxxxxx0-spi-x1_x2_x4` exists in the installed Vivado 2025.2. So do all the Hardware Manager commands the Tcl uses; this was checked with `get_cfgmem_parts` and `info commands` on 2026-11-04. Vivado also lists a `...xxxxxx1-...` variant, which most likely differs in sector architecture.

   **VERIFY:** read the board's JEDEC ID first (step 5, `id`). Its 5th byte is the sector architecture, and the part must match it. Do not program with a mismatched sector size.

   For a quick FPGA-only test after the flash already holds the weights:

   ```powershell
   & $py tools/board/program_flash.py build/board/sdr/boolean_top.bit --jtag
   ```

   This load is volatile and is lost at power-off.

5. **Check UART and flash.** Replace `COM<n>` with the port found in Device Manager:

   ```powershell
   & $py tools/board/boolean_host.py --port COM<n> ping
   & $py tools/board/boolean_host.py --port COM<n> info
   & $py tools/board/boolean_host.py --port COM<n> id
   & $py tools/board/boolean_host.py --port COM<n> cr1
   & $py tools/board/boolean_host.py --port COM<n> enable-quad
   & $py tools/board/boolean_host.py --port COM<n> info
   ```

   `ping` must report version `1` for SDR (CR1 LC=`00`) or `2` for DDR (LC=`11`). The JEDEC prefix must be `01 20 18`; the full expected response is `01 20 18 4d 00 80`. `info` must end in `COMPATIBLE`.

6. **First verified decode.** `utpu_chat.py --verify --gen 1` compares token and selected logit with the software golden; this is the first measured result:

   ```powershell
   & $py tools/board/host/utpu_chat.py --port COM<n> --prompt 'Once upon a' --verify --gen 1
   ```

   Then run a short story and retain the printed per-step wall time and FPGA cycle count:

   ```powershell
   & $py tools/board/host/utpu_chat.py --port COM<n> --prompt 'Once upon a time' --gen 8
   ```

   Record throughput two ways: wall tok/s is generated board steps divided by the sum of per-step wall seconds (the client prints this); FPGA tok/s is `50,000,000 / mean FPGA cycles per step` (the client also prints the cycle-derived rate). Include prompt-token steps in the per-step ledger and distinguish generated-token count from total board steps.

7. **DDR, only after SDR works.**

   ```powershell
   & $py tools/board/build_bitstream.py --ddr
   & $py tools/board/make_flash_image.py --ddr
   & $py tools/board/program_flash.py build/board/ddr/boolean_top.mcs --dry-run
   & $py tools/board/program_flash.py build/board/ddr/boolean_top.mcs
   ```

   Power-cycle, then repeat steps 5–6. If DDR is timing-sensitive or fails, reflash the SDR MCS and record the fallback.

   > **Boot-safety warning, until branch `ddr-lc0` is merged.** The current DDR build makes the host write the non-volatile CR1 latency code to LC=`11`. At LC=`11` the flash's 0Bh/6Bh reads have 0 dummy cycles, while the FPGA's configuration logic expects the default count. After a DDR session, the board will most likely **fail to boot from flash** at the next power-on.
   >
   > Recovery: load the bitstream over JTAG (`program_flash.py <bit> --jtag`), then run `boolean_host.py enable-quad` against an SDR build, which rewrites LC=`00`.
   >
   > The fix (DDR EDh at LC=`00` with 6 dummy cycles, allowed up to 66 MHz by datasheet Table 27) is in progress on `ddr-lc0`. After it lands, both builds use LC=`00`, QUAD=1, and this warning goes away.

8. **Record the result.** Do not add generated bench output. Create `bench/results/board_run_<sdr|ddr>.json` only after hardware testing, with this schema:

   ```json
   {"variant":"sdr","git_commit":"...","bitstream_sha256":"...","image_sha256":"...","board":"Real Digital Boolean","port":"COM<n>","version":1,"jedec_id":"0120184d0080","cr1":2,"tokens":8,"per_step":[{"pos":0,"token":0,"cycles":0,"wall_s":0.0}],"wall_s":0.0,"wall_tok_s":0.0,"fpga_tok_s":0.0,"verify":true,"notes":""}
   ```

## Troubleshooting

| Symptom | Action |
|---|---|
| No UART response | Confirm PROG/UART cable, COM port, 115200 8-N-1, USB driver, FPGA boot, and TX/RX pins; retry `ping`. |
| Wrong version byte | Bitstream/variant mismatch or stale JTAG image; power-cycle and reflash matching SDR/DDR MCS. Unknown byte is a host refusal. |
| JEDEC ID wrong | Check board/flash wiring and CS#; expected prefix is `01 20 18`. Stop before any register write. |
| Flash reads all `FF`/`00` | Check MCS address map, flash programming verify, power-cycle boot, QSPI pins, and CR1 QUAD/LC. |
| E01–E05 | E01 token out of range; E02 position out of range; E03 skipped position; E04 watchdog; E05 invalid SPI frame. Retry in order after fixing the host request. |
| Verify mismatch at step 0 | Wrong image/bitstream/tokenizer, stale flash, wrong variant LC, or wiring. Rebuild/reflash and check hashes/`info`. |
| Verify mismatch only later | Sequence/token position or KV state issue; start a new `X` sequence and retry, then capture the first failing position. |
| Watchdog E04 | Measure cycles; check flash reads and QSPI timing. Fall back to SDR if DDR. |
| DDR timing-sensitive failures | Use the SDR build/MCS, record the failure, and scope CCLK/DQ before changing timing constraints. |

Sources: [Real Digital Boolean board](https://www.realdigital.org/hardware/boolean), [AMD UG908 7-Series bitstream settings](https://docs.amd.com/r/en-US/ug908-vivado-programming-debugging/7-Series-Bitstream-Settings), [AMD supported cfgmem parts](https://docs.amd.com/api/khub/documents/dVQ5aJuXFoRjIyXUvHAuxg/content), and [AMD UG470 SPI configuration](https://docs.amd.com/api/khub/documents/yTp0C97l4DN6t6VsQFIBsw/content). `SPI_BUSWIDTH=4` selects x4; `SPI_32BIT_ADDR=No` follows the 16 MiB/128 Mbit user contract (32-bit addressing is not needed for this image); and `CONFIGRATE=26` is the nearest supported conservative 7-Series CCLK setting to the 25 MHz user-clocked QSPI reader. UG908 documents each bitstream property and UG470 documents the SPI address/bus-width behavior. 
