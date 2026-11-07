# Boolean board host client

`tools/board/host/utpu_chat.py` turns prompt text into token IDs with the repository's
`tools/llama2c/tokenizer.py` implementation, sends the Boolean UART protocol, and
prints each generated piece plus host wall time and the cycle count returned by the FPGA.
It targets the model the board bitstream is built for, stories15M (`rtl/board/boolean_core.sv:18-19`:
D=288, NL=6, VOCAB=32000, MAX_SEQ=128, KV4=1). Prompts are tokenized with the 32000-token llama2.c tokenizer
`models/llama2c/tokenizer.bin`. The `--verify` path runs the stories15M KV4 golden (`run_decode_rtl.load_model("15m", 4)`,
the same golden as `run_decode_rtl.py --model 15m --kv-bits 4`) and checks each returned argmax token and the probed
logit. `tests/test_utpu_chat.py::test_client_model_matches_board_build` fails if the client's golden or tokenizer
drifts from the `boolean_core` parameters.

## Connect to a board

The RTL UART is 8N1 with `CPB=434` at the 50 MHz core clock, i.e. 115200 baud
(`rtl/board/uart_rx.sv:1-2`, `rtl/board/uart_tx.sv:1-2`,
`docs/r3_progress.md:391-393`). On Windows, identify the board's USB-UART COM port in
Device Manager, then run from the repository root:

```powershell
python tools/board/host/utpu_chat.py --port COM5 --baud 115200 --prompt "Once upon a" --gen 8
```

If the flash has not yet been configured for the reader, request the CR1 QUAD/LC
write first:

```powershell
python tools/board/host/utpu_chat.py --port COM5 --baud 115200 --quad-enable --prompt "Once upon a" --gen 8
```

`--verify` checks every step bit-exactly against the stories15M golden (token and probed logit):

```powershell
python tools/board/host/utpu_chat.py --port COM5 --baud 115200 --prompt "Once upon a" --gen 8 --verify
```

The summary reports board steps per second from wall time (host golden time excluded) and from the FPGA cycle
counter at 50 MHz.

## UART framing

All byte order and response framing below comes from the protocol comment at the top
of `rtl/board/board_ctrl.sv:1-12`. Multi-byte values are little endian. The T command
is six bytes: ASCII `T`, token low, token high, one position byte, probe index low,
probe index high. The R response is eleven bytes: ASCII `R`, argmax token low/high,
four cycle bytes least-significant first, then the signed 32-bit probed logit least-
significant byte first (`board_ctrl.sv:3-5,126-131`). The position is one byte in the
RTL (`fb[2]`, `board_ctrl.sv:79,113`); it has no pair of position bytes to swap.

`S` is ASCII `S`, transmit count, receive count, then transmit bytes; the response is
ASCII `s`, returned count, then received bytes (`board_ctrl.sv:6-7,99-106,156-162`).
`V` is one byte and returns `v 01`; `X` is one byte and returns `x 01`
(`board_ctrl.sv:9-10,83-84`). The host must wait for each response before sending the
next request; bytes arriving during a decode or SPI transaction are ignored
(`board_ctrl.sv:12`).

Rejected requests return ASCII `E`, code, next expected position. E01 means token is
at least VOCAB; E02 means position is at least MAX_SEQ; E03 means position skips the
next expected position; E04 means the decode watchdog expired and the decode engine
and flash reader were reset; E05 means an S transmit or receive count exceeded 8
(`board_ctrl.sv:11,110-112,136,99-106`). The client turns these into an error message
containing the code and next position. Serial short reads report a timeout with the
number of bytes received and expected.

## Quad enable sequence

The client uses the same register operations as `boolean_host.Board.enable_quad`
(`tools/board/boolean_host.py:128-149`), which `run_uart_rtl.py` injects as the
host-generated `info -> enable-quad -> info` frame sequence (`tools/board/run_uart_rtl.py:221-228,261`).
It reads SR1 (05h) and CR1 (35h), computes CR1 with LC=00 and QUAD=1 while preserving
other CR1 bits, sends WREN (06h), checks WEL, sends WRR (01h) with SR1 bits 1:0
cleared and the new CR1, polls SR1.WIP, then reads SR1/CR1 back. Those SPI opcode and
register values are the operations in the existing host implementation at
`tools/board/boolean_host.py:128-149`; no board is programmed or accessed by tests.

Both SDR and DDR builds deliberately target the same non-volatile state: `LC=00`,
`QUAD=1`. The DDR EDh read uses the S25FL128S Table 27/29 row for 25 MHz (one mode
cycle plus six dummy cycles), preserving the configuration-flash latency needed by
the 7-series FPGA on the next boot.

The board-side protocol errors and software loopback are tested with:

```powershell
python -m pytest tests/test_utpu_chat.py -q
```
