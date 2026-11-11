"""Phase 7 remediation P3 — board-level instruction BRAM configuration.

Until P3 landed, the host always hard-coded `prog_depth = 1024` and the
lowering reported `fits_instruction_bram = (words <= 1024)`. That gave
every realistic uTPU workload `fits_instruction_bram = False` (the
smallest workload in `bench/results/tiling_correctness.json` lowers to
~106k instruction words), so "board execution" was structurally
blocked.

This module gives the host an explicit board model so the
fit-against-BRAM check is parameter-driven (the underlying RTL
parameter, `top.sv::PROG_DEPTH`, was already a parameter — only the
host was hard-coded). Three reference boards are pre-defined:

- ``pynqz2_baseline``: current RTL default (`PROG_DEPTH = 1024`).
  Only single-block-op-class demos (M=16/K<=64, M=32/K=32, etc.) fit;
  this is the smallest honest "the bitstream-as-shipped runs" config.
- ``pynqz2_bram_max``: bump `PROG_DEPTH = 8192` (16 KiB of BRAM —
  fits comfortably on a Pynq-Z2 that ships with 280 KiB of BRAM).
  Covers single-tile MLPs in the `(M, K) <= (64, 128)` range.
- ``vu13p_uram``: bump `PROG_DEPTH = 131072` (256 KiB; well within
  the UltraScale+ URAM budget). Covers the existing
  tiling_correctness.json shape grid in a single tile.

A `BoardConfig` is consumed by:
- ``firmware/host/lowering_blocked_fc_utpu.py::lower_blocked_fc_program_utpu``
  (the per-tile `fits_instruction_bram` flag now uses the configured
  depth);
- ``firmware/host/run_board_fit_audit.py`` (sweeps each board × each
  shape and emits `bench/results/board_fit_audit.json`).

The default kept across legacy call sites is
``BoardConfig.pynqz2_baseline()`` so existing artifacts and tests do
not move; the new fit reports are *additive*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BoardConfig:
    """RTL+host configuration for instruction memory + datapath sizing.

    Only fields the host actually consumes are present here. `prog_depth`
    is the only one that changes the fit check; the rest are recorded for
    artifact provenance so the audit JSON is self-describing.
    """

    name: str
    prog_depth: int
    buffer_size: int = 1024
    array_size: int = 16
    data_width_bits: int = 8
    notes: str = ""
    weight_region_words: int | None = None
    max_batch_count: int | None = None

    @classmethod
    def pynqz2_baseline(cls) -> "BoardConfig":
        return cls(
            name="pynqz2_baseline",
            prog_depth=1024,
            buffer_size=1024,
            array_size=16,
            data_width_bits=8,
            notes=(
                "Current RTL default (top.sv::PROG_DEPTH=1024 = 2 KiB). "
                "Only block_ops <= 4 workloads (e.g. 16x16, 16x32, 16x64, 32x32) "
                "fit; larger shapes overflow at ISA-encode time."
            ),
        )

    @classmethod
    def pynqz2_bram_max(cls) -> "BoardConfig":
        return cls(
            name="pynqz2_bram_max",
            prog_depth=8192,
            buffer_size=1024,
            array_size=16,
            data_width_bits=8,
            notes=(
                "Pynq-Z2 with PROG_DEPTH bumped to 8192 (16 KiB BRAM "
                "for the instruction memory; Pynq-Z2 ships with ~280 KiB total). "
                "Synthesis change only (top.sv::PROG_DEPTH override); "
                "no board re-spin needed beyond a fresh bitstream."
            ),
        )

    @classmethod
    def vu13p_uram(cls) -> "BoardConfig":
        return cls(
            name="vu13p_uram",
            prog_depth=131072,
            buffer_size=4096,
            array_size=16,
            data_width_bits=8,
            notes=(
                "Larger Xilinx Ultrascale+ part with the instruction memory "
                "placed in URAM (PROG_DEPTH=131072 = 256 KiB) and the "
                "unified buffer at 4096 words. Covers the full Phase 3 "
                "tiling-correctness shape grid in a single tile."
            ),
        )

    @classmethod
    def artix_a7100t_bram_max(cls, prog_depth: int = 65536) -> "BoardConfig":
        """Artix-7 A7-100T with instruction BRAM grown into free BRAM36.

        Default PROG_DEPTH=65536 — closes after the UPLOAD_LEN_MAX fix
        (`bench/results/prog_depth_sweep.json`). Two-byte UART length still
        caps uploadable words at 65535 (=DEPTH−1); that is enough for FC1 and
        board-fit 256×256. Datapath matches shipping synth (N=8 INT8, buffer 4096).
        """
        return cls(
            name="artix_a7100t_bram_max",
            prog_depth=int(prog_depth),
            buffer_size=16384,
            array_size=8,
            data_width_bits=8,
            notes=(
                f"Artix-7 xc7a100tcsg324-1 with PROG_DEPTH={int(prog_depth)} "
                "and BUFFER_SIZE=16384 (smallest swept size that holds FC1 "
                "14144-word weight payload; activations in [14144, 16384)). "
                "Requires UPLOAD_LEN_MAX fix (never PROG_DEPTH[15:0]). "
                "UART length field still maxes at 65535 words. A5 buf-fill enabled."
            ),
        )

    @classmethod
    def boolean_xc7s50(cls) -> "BoardConfig":
        """Real Digital Boolean (xc7s50csga324-1) fit configuration.

        The MAX_BATCH_COUNT setting belongs to the RTL build, rather than the
        ISA program.  The board runner records it alongside this host model so
        the uploaded program and bitstream cannot be confused with the former
        Artix shipping configuration.
        """
        return cls(
            name="boolean_xc7s50",
            # The INT8 196->256->10 network needs 27,648 tiled weight words:
            # 25,600 for 200x256 FC1 and 2,048 for padded 16x256 FC2.
            # The 32,768-word buffer leaves 5,120 words for activation scratch.
            # The measured transformer program is 25,551 words, so the
            # Boolean profile is intentionally fixed at the 32K gate.
            prog_depth=32768,
            buffer_size=32768,
            array_size=8,
            data_width_bits=8,
            weight_region_words=27648,
            max_batch_count=4,
            notes=(
                "Real Digital Boolean, xc7s50csga324-1: N=8 INT8, "
                "BUFFER_SIZE=32768, WEIGHT_REGION_WORDS=27648, "
                "PROG_DEPTH=32768, MAX_BATCH_COUNT=4. Existing Artix sweeps "
                "show 33 BRAM36 at PROG_DEPTH=32768/BUFFER_SIZE=4096; "
                "memory-bit extrapolation estimates about 46/75 BRAM36 "
                "(29-tile margin) for this point. "
                "Estimate only; Boolean synthesis is required for measured fit."
            ),
        )

    @classmethod
    def reference_set(cls, artix_prog_depth: int | None = None) -> List["BoardConfig"]:
        boards = [
            cls.pynqz2_baseline(),
            cls.pynqz2_bram_max(),
            cls.vu13p_uram(),
        ]
        if artix_prog_depth is not None:
            boards.append(cls.artix_a7100t_bram_max(artix_prog_depth))
        else:
            boards.append(cls.artix_a7100t_bram_max())
        boards.append(cls.boolean_xc7s50())
        return boards

    def fits(self, program_instruction_words: int) -> bool:
        return int(program_instruction_words) <= int(self.prog_depth)

    def as_dict(self) -> Dict[str, object]:
        result = {
            "name": self.name,
            "prog_depth": int(self.prog_depth),
            "buffer_size": int(self.buffer_size),
            "array_size": int(self.array_size),
            "data_width_bits": int(self.data_width_bits),
            "notes": self.notes,
        }
        if self.weight_region_words is not None:
            result["weight_region_words"] = int(self.weight_region_words)
        if self.max_batch_count is not None:
            result["max_batch_count"] = int(self.max_batch_count)
        return result
