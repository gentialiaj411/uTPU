"""Bit-level Python simulator for the RTL-observable uTPU ISA.

Phase 4 Stage A widening: the simulator now accepts an optional
``IsaConfig`` (default ``DEFAULT_CFG``) describing the target's
``address_width`` and ``compute_data_width``. With the default config
the simulator's behaviour is byte-identical to pre-Phase-4 (INT4 / 9-bit
address / single-word LOAD/RUN/FETCH/BSTORE/DTYPE-with-addr).

Behaviour with a non-default config:
* When ``address_width > 9`` the simulator decodes the 2-word
  extended-address form for LOAD/RUN/FETCH/BSTORE and the
  address-bearing D-type sub-ops (matches ``isa_encoder._emit_with_addr``).
* When ``compute_data_width != 4`` the simulator generalises the
  per-word packing factor (``items_per_word = 16 // compute_data_width``)
  and clipping range, so weight/input loading and RUN finalize work for
  INT8 (``compute_data_width=8``) as well as INT4.
* ``accumulator_data_width`` is currently used only for clip-range
  selection during the finalize step (defaults to 16 to match the legacy
  16-bit accumulator).

Public API ``simulate_program_bytes`` / ``simulate_words`` /
``simulate_mem_file`` keep their existing positional arguments
(``array_size=16, buffer_size=512``) and gain optional keyword args for
the new config dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from isa_encoder import (
    DEFAULT_CFG,
    DTYPE_SUBOP_ACC_ADD,
    DTYPE_SUBOP_BARRIER,
    DTYPE_SUBOP_BUFFER_XFER,
    DTYPE_SUBOP_PE_SELECT,
    IsaConfig,
    NOP_SUBOP_REQUANT,
    OPCODE_DTYPE,
    OPCODE_TILE_GEMM,
    OPCODE_VEC,
    VEC_ESCAPE_WORD,
    TILE_GEMM_WORD_COUNT,
    VEC_WORD_COUNT,
    VEC_SUBOP_VADD_RQ,
    VEC_SUBOP_VRMSNORM,
    VEC_SUBOP_VPOS_ADD,
    VEC_SUBOP_VSOFTMAX,
    VEC_SUBOP_VMATMUL,
    VEC_SUBOP_VLAYOUT,
    VEC_PARAM_KIND_VADD_RQ,
    VEC_PARAM_KIND_VRMSNORM,
    VEC_PARAM_KIND_VPOS_ADD,
    VEC_PARAM_KIND_VSOFTMAX,
    VEC_PARAM_KIND_VMATMUL,
    VEC_PARAM_KIND_VLAYOUT,
    decodeTileGemmWords,
    decodeVecWords,
)
from requantization import requantize_value


OPCODE_STORE = 0b000
OPCODE_FETCH = 0b001
OPCODE_RUN = 0b010
OPCODE_LOAD = 0b011
OPCODE_HALT = 0b100
OPCODE_NOP = 0b101
OPCODE_BSTORE = 0b110


_LUT_FIXTURE = Path(__file__).parent / "fixtures" / "tiny_transformer_integer_luts.json"
_LUT_DOCUMENT = json.loads(_LUT_FIXTURE.read_text(encoding="utf-8"))
_LUT_PAYLOAD = {key: _LUT_DOCUMENT[key] for key in ("exp_q15", "rsqrt_q14")}
_LUT_BYTES = json.dumps(_LUT_PAYLOAD, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
if _LUT_DOCUMENT.get("payload_sha256") != hashlib.sha256(_LUT_BYTES).hexdigest():
    raise RuntimeError("tiny transformer LUT fixture hash mismatch")
EXP_Q15 = np.asarray(_LUT_PAYLOAD["exp_q15"], dtype=np.uint16)
RSQRT_Q14 = np.asarray(_LUT_PAYLOAD["rsqrt_q14"], dtype=np.uint16)


def _sign_extend(value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    sign_bit = 1 << (width - 1)
    return value - (1 << width) if value & sign_bit else value


# Legacy specialisations preserved for callers that import them.
def _sign_extend_int4(x: int) -> int:
    return _sign_extend(x, 4)


def _pack_int4(values: Iterable[int]) -> int:
    word = 0
    vals = list(values)
    while len(vals) < 4:
        vals.append(0)
    for i, value in enumerate(vals[:4]):
        word |= (int(value) & 0xF) << (4 * i)
    return word & 0xFFFF


def _unpack_int4_word(word: int) -> List[int]:
    return [_sign_extend_int4((word >> (4 * i)) & 0xF) for i in range(4)]


def _pack_compute_word(values: Iterable[int], cfg: IsaConfig) -> int:
    width = cfg.compute_data_width
    items = cfg.items_per_word
    mask = (1 << width) - 1
    word = 0
    vals = list(values)
    while len(vals) < items:
        vals.append(0)
    for i, value in enumerate(vals[:items]):
        word |= (int(value) & mask) << (width * i)
    return word & ((1 << cfg.instruction_width) - 1)


def _unpack_compute_word(word: int, cfg: IsaConfig) -> List[int]:
    width = cfg.compute_data_width
    items = cfg.items_per_word
    mask = (1 << width) - 1
    return [_sign_extend((word >> (width * i)) & mask, width) for i in range(items)]


def _clip_int4(x: int) -> int:
    if x > 7:
        return 7
    if x < -8:
        return -8
    return int(x)


def _clip_to_width(x: int, width: int) -> int:
    hi = (1 << (width - 1)) - 1
    lo = -(1 << (width - 1))
    if x > hi:
        return hi
    if x < lo:
        return lo
    return int(x)


@dataclass
class ISASimulationResult:
    halted: bool
    pc: int
    fetch_bytes: List[int]
    buffer_words: Dict[int, int]
    instruction_count: int
    executed_ops: Dict[str, int] = field(default_factory=dict)
    cycle_count_sequential: int = 0
    cycle_count_parallel_estimate: int = 0
    num_pe: int = 1
    # Phase 5 measurement hooks: host-emitted store traffic, redundant
    # reload bytes (same value re-stored to the same address), MAC count
    # and array utilization. All sim-only; intended for naive vs scheduled
    # comparison before ISA emission. Not a hardware claim.
    store_bytes_total: int = 0
    redundant_store_bytes: int = 0
    compute_runs: int = 0
    total_macs: int = 0
    cycles_per_mac: float = 0.0
    array_utilization: float = 0.0


@dataclass
class _PEState:
    buffer: List[int]
    weights: List[List[int]]
    weight_shadow: List[List[int]]
    inputs: List[int]
    input_matrix: List[List[int]]
    acc_partial_sums: List[int]
    acc_partial_matrix: List[List[int]]

    @classmethod
    def create(
        cls,
        array_size: int,
        buffer_size: int,
        max_batch_size: int,
        buffer: Optional[List[int]] = None,
    ) -> "_PEState":
        zeros = [[0 for _ in range(array_size)] for _ in range(array_size)]
        return cls(
            buffer=buffer if buffer is not None else [0] * buffer_size,
            weights=[row[:] for row in zeros],
            weight_shadow=[row[:] for row in zeros],
            inputs=[0 for _ in range(array_size)],
            acc_partial_sums=[0 for _ in range(array_size)],
            input_matrix=[[0 for _ in range(max_batch_size)] for _ in range(array_size)],
            acc_partial_matrix=[[0 for _ in range(max_batch_size)] for _ in range(array_size)],
        )


class UTPUISASimulator:
    """Bit-level Python simulator for the RTL-observable uTPU ISA behavior.

    Supports optional 2-PE simulation with per-PE program-counter semantics
    via PE tags (bit 12) and D-type barrier / buffer-xfer / acc-add /
    pe-select instructions.

    Phase 4: ``cfg`` carries ``address_width`` and ``compute_data_width``.
    The simulator preserves legacy behaviour exactly when ``cfg is None``
    (resolves to ``isa_encoder.DEFAULT_CFG`` = INT4 / 9-bit / single-word).
    """

    def __init__(
        self,
        array_size: int = 16,
        buffer_size: int = 512,
        alpha_shift: int = 2,
        num_pe: int = 1,
        accumulator_data_width: int = 16,
        cfg: Optional[IsaConfig] = None,
        max_batch_size: int = 64,
    ):
        cfg = cfg if cfg is not None else DEFAULT_CFG
        if array_size <= 0:
            raise ValueError("array_size must be positive")
        if array_size % cfg.items_per_word != 0:
            raise ValueError(
                f"array_size must be a positive multiple of items_per_word={cfg.items_per_word}; "
                f"got {array_size}"
            )
        if num_pe not in (1, 2):
            raise ValueError("num_pe must be 1 or 2")
        if accumulator_data_width <= cfg.compute_data_width:
            raise ValueError(
                "accumulator_data_width must exceed compute_data_width"
            )
        self.array_size = int(array_size)
        self.buffer_size = int(buffer_size)
        self.alpha_shift = int(alpha_shift)
        self.num_pe = int(num_pe)
        self.cfg = cfg
        self.accumulator_data_width = int(accumulator_data_width)
        self.max_batch_size = int(max_batch_size)
        self.reset()

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------

    def reset(self, *, preserve_buffer: bool = False) -> None:
        preserved_buffers = (
            [pe.buffer for pe in self.pes]
            if preserve_buffer and hasattr(self, "pes")
            else None
        )
        self.pes = [
            _PEState.create(
                self.array_size,
                self.buffer_size,
                self.max_batch_size,
                buffer=(preserved_buffers[index] if preserved_buffers is not None else None),
            )
            for index in range(self.num_pe)
        ]
        self.fetch_bytes: List[int] = []
        self.executed_ops = {
            "store": 0,
            "fetch": 0,
            "run": 0,
            "residual_add": 0,
            "load": 0,
            "load_weights": 0,
            "load_inputs": 0,
            "halt": 0,
            "nop": 0,
            "bstore": 0,
            "buffer_xfer": 0,
            "barrier": 0,
            "acc_add": 0,
            "pe_select": 0,
        }
        self._barrier_hits: Dict[int, int] = {}
        self._section_pe_cycles: List[Dict[int, int]] = []
        self._current_section: Dict[int, int] = {pe: 0 for pe in range(self.num_pe)}
        self._cycle_count_sequential = 0
        self._active_pe = 0
        # Phase 5 measurement hooks: tracked only on host-emitted writes
        # (STORE / BSTORE), not on internal data movement (RUN finalize,
        # BUFFER_XFER) which represents on-chip traffic the host did not
        # have to issue.
        self._store_bytes_total = 0
        self._redundant_store_bytes = 0
        self._last_store_value: Dict[Tuple[int, int], int] = {}
        self._compute_runs = 0
        self._total_macs = 0
        self._loaded_input_batch_count = 1
        self._weight_overlap_active = False
        self._requant_enable = False
        self._requant_multipliers = [1 for _ in range(self.array_size)]
        self._requant_right_shifts = [0 for _ in range(self.array_size)]

    @property
    def buffer(self) -> List[int]:
        return self.pes[0].buffer

    def _pe(self, pe_id: int) -> _PEState:
        if not 0 <= pe_id < self.num_pe:
            raise ValueError(f"invalid pe_id {pe_id} for num_pe={self.num_pe}")
        return self.pes[pe_id]

    def _legacy_addr(self, word: int) -> int:
        """Decode 9-bit address packed into bits 7..15 of an instruction word."""
        return (word >> 7) & 0x1FF

    def _check_addr(self, addr: int) -> None:
        if not 0 <= addr < self.buffer_size:
            raise ValueError(f"buffer address out of range: {addr}")

    def _read_word(self, pe_id: int, addr: int) -> int:
        self._check_addr(addr)
        return self.pes[pe_id].buffer[addr] & 0xFFFF

    def _write_word(self, pe_id: int, addr: int, value: int) -> None:
        self._check_addr(addr)
        self.pes[pe_id].buffer[addr] = int(value) & 0xFFFF

    # ------------------------------------------------------------------
    # weight / input loading + compute (parameter-driven on cfg)
    # ------------------------------------------------------------------

    def _load_weights(self, pe_id: int, addr: int, *, to_shadow_only: bool = False) -> None:
        pe = self._pe(pe_id)
        items_per_word = self.cfg.items_per_word
        words_needed = (self.array_size * self.array_size) // items_per_word
        flat: List[int] = []
        for offset in range(words_needed):
            flat.extend(_unpack_compute_word(self._read_word(pe_id, addr + offset), self.cfg))
        for row in range(self.array_size):
            start = row * self.array_size
            pe.weight_shadow[row] = flat[start:start + self.array_size]
            if not to_shadow_only:
                pe.weights[row] = pe.weight_shadow[row][:]

    def _commit_weights(self, pe_id: int) -> None:
        pe = self._pe(pe_id)
        for row in range(self.array_size):
            pe.weights[row] = pe.weight_shadow[row][:]

    def _load_inputs(self, pe_id: int, addr: int, batch_count: int = 1) -> None:
        pe = self._pe(pe_id)
        items_per_word = self.cfg.items_per_word
        if int(batch_count) > int(self.max_batch_size):
            raise ValueError(
                f"batch_count={batch_count} exceeds simulator max_batch_size={self.max_batch_size}"
            )
        if int(batch_count) == 1:
            words_needed = self.array_size // items_per_word
        else:
            words_needed = (self.array_size * int(batch_count)) // items_per_word
        flat: List[int] = []
        for offset in range(words_needed):
            flat.extend(_unpack_compute_word(self._read_word(pe_id, addr + offset), self.cfg))
        pe.inputs = flat[: self.array_size]
        for row in range(self.array_size):
            for col in range(self.max_batch_size):
                pe.input_matrix[row][col] = 0
        if int(batch_count) == 1:
            for row in range(self.array_size):
                pe.input_matrix[row][0] = pe.inputs[row]
        else:
            expected = self.array_size * int(batch_count)
            padded = flat[:expected] + [0] * max(0, expected - len(flat))
            for col in range(int(batch_count)):
                for row in range(self.array_size):
                    pe.input_matrix[row][col] = int(padded[col * self.array_size + row])
        self._loaded_input_batch_count = int(batch_count)

    def _run_accumulate(self, pe_id: int, acc_clear: bool, batch_count: int = 1) -> None:
        pe = self._pe(pe_id)
        batch = int(batch_count)
        if batch == 1:
            for row in range(self.array_size):
                lane_val = 0
                for k in range(self.array_size):
                    lane_val += pe.weights[row][k] * pe.inputs[k]
                if acc_clear:
                    pe.acc_partial_sums[row] = lane_val
                    pe.acc_partial_matrix[row][0] = lane_val
                else:
                    pe.acc_partial_sums[row] += lane_val
                    pe.acc_partial_matrix[row][0] += lane_val
            for row in range(self.array_size):
                for col in range(1, self.max_batch_size):
                    pe.acc_partial_matrix[row][col] = 0
        else:
            for row in range(self.array_size):
                for col in range(batch):
                    lane_val = 0
                    for k in range(self.array_size):
                        lane_val += pe.weights[row][k] * pe.input_matrix[k][col]
                    if acc_clear:
                        pe.acc_partial_matrix[row][col] = lane_val
                    else:
                        pe.acc_partial_matrix[row][col] += lane_val
                for col in range(batch, self.max_batch_size):
                    if acc_clear:
                        pe.acc_partial_matrix[row][col] = 0
            for row in range(self.array_size):
                pe.acc_partial_sums[row] = pe.acc_partial_matrix[row][0]
        self._compute_runs += 1
        self._total_macs += int(self.array_size) * int(self.array_size) * int(batch)

    def _record_host_store(self, pe_id: int, addr: int, value: int) -> None:
        """Record a host-emitted STORE/BSTORE write for reload accounting.

        ``store_bytes_total`` is incremented by 2 (1 word == 2 bytes).
        A write is counted as ``redundant`` when the most recent host-store
        to the same (pe, addr) wrote the same 16-bit value; this is the
        signal a scheduler / allocator can eliminate by caching the value
        in-place across consumers (i.e. not re-emitting the STORE for
        unchanged input blocks across output blocks in blocked-FC).
        """
        key = (int(pe_id), int(addr))
        masked = int(value) & 0xFFFF
        self._store_bytes_total += 2
        last = self._last_store_value.get(key)
        if last is not None and last == masked:
            self._redundant_store_bytes += 2
        self._last_store_value[key] = masked

    def _run_finalize(
        self,
        pe_id: int,
        result_addr: int,
        quantize: bool,
        relu: bool,
        residual_addr: Optional[int] = None,
        batch_count: int = 1,
    ) -> None:
        pe = self._pe(pe_id)
        batch = int(batch_count)
        if batch > 1 and residual_addr is not None:
            raise ValueError("batched finalize does not currently support residual add")
        residual_values = [0 for _ in range(self.array_size)]
        if residual_addr is not None:
            items_per_word = self.cfg.items_per_word
            words_needed = self.array_size // items_per_word
            flat: List[int] = []
            for offset in range(words_needed):
                flat.extend(_unpack_compute_word(self._read_word(pe_id, residual_addr + offset), self.cfg))
            residual_values = flat[: self.array_size]
        outputs = [0 for _ in range(self.array_size * self.array_size)]
        if batch == 1:
            for i, acc in enumerate(pe.acc_partial_sums):
                combined = int(acc) + int(residual_values[i])
                if quantize:
                    if self._requant_enable:
                        multiplier, right_shift = self._requant_for_row(i)
                        q = requantize_value(
                            combined,
                            multiplier=multiplier,
                            right_shift=right_shift,
                            out_width=self.cfg.compute_data_width,
                        )
                    else:
                        q = _clip_to_width(combined, self.cfg.compute_data_width)
                else:
                    q = int(combined)
                if relu and q < 0:
                    q = q >> self.alpha_shift
                outputs[i] = q
        else:
            padded_cols = ((batch + self.array_size - 1) // self.array_size) * self.array_size
            outputs = [0 for _ in range(self.array_size * padded_cols)]
            for col in range(batch):
                for row in range(self.array_size):
                    combined = int(pe.acc_partial_matrix[row][col])
                    if quantize:
                        if self._requant_enable:
                            multiplier, right_shift = self._requant_for_row(row)
                            q = requantize_value(
                                combined,
                                multiplier=multiplier,
                                right_shift=right_shift,
                                out_width=self.cfg.compute_data_width,
                            )
                        else:
                            q = _clip_to_width(combined, self.cfg.compute_data_width)
                    else:
                        q = int(combined)
                    if relu and q < 0:
                        q = q >> self.alpha_shift
                    outputs[col * self.array_size + row] = q

        items_per_word = self.cfg.items_per_word
        padded = list(outputs)
        for word_idx in range(len(padded) // items_per_word):
            chunk = padded[word_idx * items_per_word : (word_idx + 1) * items_per_word]
            self._write_word(pe_id, result_addr + word_idx, _pack_compute_word(chunk, self.cfg))

    def _acc_add(self, dst_pe: int, src_pe: int) -> None:
        dst = self._pe(dst_pe)
        src = self._pe(src_pe)
        for i in range(self.array_size):
            dst.acc_partial_sums[i] += int(src.acc_partial_sums[i])

    def _buffer_xfer(self, src_pe: int, dst_pe: int, src_addr: int, dst_addr: int, count: int) -> None:
        for offset in range(int(count)):
            value = self._read_word(src_pe, src_addr + offset)
            self._write_word(dst_pe, dst_addr + offset, value)

    # ------------------------------------------------------------------
    # Phase-2 VEC descriptor (independent from tiny_transformer_reference)
    # ------------------------------------------------------------------

    @staticmethod
    def _rshift_rne(value: np.ndarray | int, shift: int | np.ndarray) -> np.ndarray:
        a = np.asarray(value, dtype=np.int64)
        s = np.asarray(shift, dtype=np.int64)
        if np.any(s < 0):
            raise ValueError("VEC RNE shift must be non-negative")
        safe = np.maximum(s, 1)
        mag = np.abs(a)
        quotient = np.right_shift(mag, safe)
        remainder = mag - np.left_shift(quotient, safe)
        half = np.left_shift(np.ones_like(safe, dtype=np.int64), safe - 1)
        rounded = quotient + ((remainder > half) | ((remainder == half) & ((quotient & 1) == 1)))
        return np.where(s == 0, a, np.where(a < 0, -rounded, rounded))

    def _vec_check_range(self, base: int, elements: int, label: str) -> tuple[int, int]:
        if elements <= 0 or elements % 2:
            raise ValueError(f"{label} must have a positive even code length")
        words = elements // 2
        self._check_addr(base)
        self._check_addr(base + words - 1)
        return int(base), int(base + words)

    def _vec_read_codes(self, pe_id: int, base: int, elements: int, *, unsigned: bool = False) -> np.ndarray:
        self._vec_check_range(base, elements, "VEC operand")
        out = np.empty(elements, dtype=np.int64)
        for offset in range(elements // 2):
            word = self._read_word(pe_id, base + offset)
            lo, hi = word & 0xFF, (word >> 8) & 0xFF
            if not unsigned:
                lo = _sign_extend(lo, 8)
                hi = _sign_extend(hi, 8)
            out[2 * offset], out[2 * offset + 1] = lo, hi
        return out

    def _vec_write_codes(self, pe_id: int, base: int, values: np.ndarray, *, unsigned: bool = False) -> None:
        codes = np.asarray(values, dtype=np.int64).reshape(-1)
        self._vec_check_range(base, len(codes), "VEC destination")
        lo, hi = (0, 255) if unsigned else (-128, 127)
        if np.any(codes < lo) or np.any(codes > hi):
            raise ValueError("VEC output outside destination code range")
        for offset in range(len(codes) // 2):
            word = (int(codes[2 * offset]) & 0xFF) | ((int(codes[2 * offset + 1]) & 0xFF) << 8)
            self._write_word(pe_id, base + offset, word)

    @staticmethod
    def _vec_no_partial_overlap(left: tuple[int, int], right: tuple[int, int], *, allow_exact: bool, label: str) -> None:
        overlaps = left[0] < right[1] and right[0] < left[1]
        if overlaps and not (allow_exact and left == right):
            raise ValueError(f"illegal VEC overlap: {label}")

    def _vec_record(self, pe_id: int, address: int, kind: int) -> list[int]:
        self._check_addr(address)
        self._check_addr(address + 7)
        words = [self._read_word(pe_id, address + i) for i in range(8)]
        if words[0] != kind:
            raise ValueError(f"VEC parameter record kind {words[0] & 0xF} != {kind}")
        return words

    @staticmethod
    def _vec_i8(value: int) -> int:
        return _sign_extend(int(value) & 0xFF, 8)

    def _vec_code_at(self, pe_id: int, base: int, element: int, *, unsigned: bool = False) -> int:
        """Read one packed VEC code without imposing an even-length range."""
        if element < 0:
            raise ValueError("negative VEC element index")
        word = self._read_word(pe_id, int(base) + (int(element) >> 1))
        value = (word >> (8 * (int(element) & 1))) & 0xFF
        return value if unsigned else _sign_extend(value, 8)

    def _vec_set_code_at(self, pe_id: int, base: int, element: int, value: int) -> None:
        if element < 0 or not -128 <= int(value) <= 255:
            raise ValueError("VEC byte output outside representable range")
        address = int(base) + (int(element) >> 1)
        previous = self._read_word(pe_id, address)
        shift = 8 * (int(element) & 1)
        self._write_word(pe_id, address, (previous & ~(0xFF << shift)) | ((int(value) & 0xFF) << shift))

    def _vec_matmul(self, pe_id: int, desc: dict) -> None:
        """Subop 4: serialized activation matmul, independent of PE state."""
        record = self._vec_record(pe_id, int(desc["param_record"]), VEC_PARAM_KIND_VMATMUL)
        if record[7] or (record[4] & ~0x7) or (record[6] & ~0x1F):
            raise ValueError("VMATMUL parameter record has nonzero reserved bits")
        m = (record[2] & 0x1F) + 1
        k = ((record[2] >> 5) & 0x1F) + 1
        n = ((record[2] >> 10) & 0x1F) + 1
        transpose_b = bool(record[2] & 0x8000)
        a_stride, b_stride = record[3] & 0xFF, (record[3] >> 8) & 0xFF
        a_unsigned, b_unsigned, causal_record = bool(record[4] & 1), bool(record[4] & 2), bool(record[4] & 4)
        causal = bool(desc["causal"])
        if causal != causal_record:
            raise ValueError("VMATMUL descriptor/record causal mismatch")
        if a_stride < k or b_stride < (k if transpose_b else n) or (a_stride & 1) or (b_stride & 1):
            raise ValueError("VMATMUL invalid row strides")
        if int(desc["length"]) != m * n or (m * n) & 1:
            raise ValueError("VMATMUL length must equal an even M*N")
        if causal and not ((transpose_b and m == n) or (a_unsigned and not transpose_b and m == k)):
            raise ValueError("illegal VMATMUL causal/transpose/signedness combination")
        b_base, dst, src0, param = int(record[1]), int(desc["dst"]), int(desc["src0"]), int(desc["param_record"])
        a_elements, b_rows = m * a_stride, (n if transpose_b else k)
        b_elements = b_rows * b_stride
        c_elements = m * n
        a_range = self._vec_check_range(src0, a_elements if not (a_elements & 1) else a_elements + 1, "VMATMUL A")
        b_range = self._vec_check_range(b_base, b_elements if not (b_elements & 1) else b_elements + 1, "VMATMUL B")
        c_range = self._vec_check_range(dst, c_elements, "VMATMUL C")
        record_range = (param, param + 8)
        for left, right, label in ((a_range, b_range, "A/B"), (a_range, c_range, "A/C"),
                                   (b_range, c_range, "B/C"), (a_range, record_range, "A/record"),
                                   (b_range, record_range, "B/record"), (c_range, record_range, "C/record")):
            self._vec_no_partial_overlap(left, right, allow_exact=False, label="VMATMUL " + label)
        out = np.zeros((m, n), dtype=np.int64)
        for row in range(m):
            for col in range(n):
                if causal and transpose_b and col > row:
                    continue
                limit = row + 1 if causal and a_unsigned and not transpose_b else k
                acc = 0
                for inner in range(limit):
                    av = self._vec_code_at(pe_id, src0, row * a_stride + inner, unsigned=a_unsigned)
                    b_index = col * b_stride + inner if transpose_b else inner * b_stride + col
                    bv = self._vec_code_at(pe_id, b_base, b_index, unsigned=b_unsigned)
                    acc += av * bv
                out[row, col] = requantize_value(acc, multiplier=int(record[5]),
                                                  right_shift=int(record[6] & 0x1F), out_width=8)
        self._vec_write_codes(pe_id, dst, out.reshape(-1))

    def _vec_layout(self, pe_id: int, desc: dict) -> None:
        """Subop 5: scalar, byte-exact permutations between frozen layouts."""
        record = self._vec_record(pe_id, int(desc["param_record"]), VEC_PARAM_KIND_VLAYOUT)
        if any(record[index] for index in range(4, 8)) or (record[1] & 0xE000) or (record[2] & ~0x7):
            raise ValueError("VLAYOUT parameter record has nonzero reserved bits")
        rows, cols = (record[1] & 0x1F) + 1, ((record[1] >> 5) & 0xFF) + 1
        mode, src_stride, dst_stride = record[2] & 0x7, record[3] & 0xFF, (record[3] >> 8) & 0xFF
        if mode > 3 or src_stride < cols or dst_stride < cols or (src_stride & 1) or (dst_stride & 1):
            raise ValueError("VLAYOUT invalid mode or stride")
        if int(desc["length"]) != rows * cols or (rows * cols) & 1:
            raise ValueError("VLAYOUT length must equal even R*C")
        src0, dst, param = int(desc["src0"]), int(desc["dst"]), int(desc["param_record"])
        groups4, groups8 = (rows + 3) // 4, (rows + 7) // 8
        col8 = (cols + 7) // 8
        def rm(r: int, c: int, stride: int) -> int: return r * stride + c
        def lin(r: int, c: int) -> int: return ((r // 4) * cols + c) * 4 + (r % 4)
        def lfin(r: int, c: int) -> int: return ((c // 8) * rows + r) * 8 + (c % 8)
        def wt(r: int, c: int) -> int: return ((r // 8) * cols + c) * 8 + (r % 8)
        source_extent = {0: rows * src_stride, 1: col8 * rows * 8, 2: rows * src_stride, 3: groups4 * cols * 4}[mode]
        dest_extent = {0: groups4 * cols * 4, 1: rows * dst_stride, 2: groups8 * cols * 8, 3: rows * dst_stride}[mode]
        src_range = self._vec_check_range(src0, source_extent if not (source_extent & 1) else source_extent + 1, "VLAYOUT source")
        dst_range = self._vec_check_range(dst, dest_extent if not (dest_extent & 1) else dest_extent + 1, "VLAYOUT destination")
        self._vec_no_partial_overlap(src_range, dst_range, allow_exact=False, label="VLAYOUT source/destination")
        self._vec_no_partial_overlap(src_range, (param, param + 8), allow_exact=False, label="VLAYOUT source/record")
        self._vec_no_partial_overlap(dst_range, (param, param + 8), allow_exact=False, label="VLAYOUT destination/record")
        for index in range(dest_extent):
            self._vec_set_code_at(pe_id, dst, index, 0)
        for r in range(rows):
            for c in range(cols):
                src_index, dst_index = ((rm(r, c, src_stride), lin(r, c)) if mode == 0 else
                                        (lfin(r, c), rm(r, c, dst_stride)) if mode == 1 else
                                        (rm(r, c, src_stride), wt(r, c)) if mode == 2 else
                                        (lin(r, c), rm(r, c, dst_stride)))
                self._vec_set_code_at(pe_id, dst, dst_index, self._vec_code_at(pe_id, src0, src_index))

    def _execute_vec(self, pe_id: int, desc: dict) -> None:
        if self.cfg.compute_data_width != 8:
            raise ValueError("VEC v1 requires compute_data_width=8")
        subop, length = int(desc["subop"]), int(desc["length"])
        dst, src0, record_addr = int(desc["dst"]), int(desc["src0"]), int(desc["param_record"])
        if length % 2:
            raise ValueError("VEC v1 length must be even")
        if subop == VEC_SUBOP_VMATMUL:
            self._vec_matmul(pe_id, desc)
            return
        if subop == VEC_SUBOP_VLAYOUT:
            self._vec_layout(pe_id, desc)
            return
        if subop in (VEC_SUBOP_VADD_RQ, VEC_SUBOP_VRMSNORM) and not (48 <= length <= 1536 and length % 48 == 0):
            raise ValueError("VEC v1 add/norm length must be 48*n, 1<=n<=32")
        if subop == VEC_SUBOP_VPOS_ADD and length != 1536:
            raise ValueError("VPOS_ADD v1 length must be 1536")
        if subop == VEC_SUBOP_VSOFTMAX and length != 1024:
            raise ValueError("VSOFTMAX v1 length must be 1024")
        dst_range, src0_range = self._vec_check_range(dst, length, "dst"), self._vec_check_range(src0, length, "src0")
        self._vec_no_partial_overlap(dst_range, src0_range, allow_exact=True, label="dst/src0")
        record_range = (record_addr, record_addr + 8)
        self._vec_no_partial_overlap(dst_range, record_range, allow_exact=False, label="dst/parameter-record")
        self._vec_no_partial_overlap(src0_range, record_range, allow_exact=False, label="src0/parameter-record")

        if subop == VEC_SUBOP_VADD_RQ:
            record = self._vec_record(pe_id, record_addr, VEC_PARAM_KIND_VADD_RQ)
            if any(record[index] for index in (4, 5, 6, 7)) or (record[3] & 0xFF00):
                raise ValueError("VADD_RQ parameter record has nonzero reserved bits")
            src1, ea, eb, eo = record[1], self._vec_i8(record[2]), self._vec_i8(record[2] >> 8), self._vec_i8(record[3])
            if not all(-31 <= value <= 31 for value in (ea, eb, eo)) or eo < max(ea, eb):
                raise ValueError("invalid VADD_RQ exponent record")
            src1_range = self._vec_check_range(src1, length, "src1")
            self._vec_no_partial_overlap(dst_range, src1_range, allow_exact=True, label="dst/src1")
            a, b = self._vec_read_codes(pe_id, src0, length), self._vec_read_codes(pe_id, src1, length)
            result = self._rshift_rne(a, eo - ea) + self._rshift_rne(b, eo - eb)
            self._vec_write_codes(pe_id, dst, np.clip(result, -128, 127))
        elif subop == VEC_SUBOP_VRMSNORM:
            record = self._vec_record(pe_id, record_addr, VEC_PARAM_KIND_VRMSNORM)
            if any(record[index] for index in (3, 4, 5, 6, 7)) or (record[2] & 0xFF00):
                raise ValueError("VRMSNORM parameter record has nonzero reserved bits")
            gamma_base, gamma_shift = record[1], self._vec_i8(record[2])
            if not 0 <= gamma_shift <= 31:
                raise ValueError("VRMSNORM gamma_shift outside [0,31]")
            gamma_range = self._vec_check_range(gamma_base, 48, "gamma")
            self._vec_no_partial_overlap(dst_range, gamma_range, allow_exact=False, label="dst/gamma")
            self._vec_no_partial_overlap(src0_range, gamma_range, allow_exact=False, label="src0/gamma")
            gamma = self._vec_read_codes(pe_id, gamma_base, 48)
            source = self._vec_read_codes(pe_id, src0, length)
            out = np.empty(length, dtype=np.int64)
            for row_start in range(0, length, 48):
                row = source[row_start:row_start + 48]
                square_sum = int(np.sum(row * row))
                q0, rem = divmod(square_sum, 48)
                mean = q0 + int(2 * rem > 48 or (2 * rem == 48 and (q0 & 1))) + 1
                exponent = (mean.bit_length() - 1) & ~1
                m_q14 = min(65535, max(16384, int(self._rshift_rne(mean << 14, exponent))))
                r0 = int(RSQRT_Q14[min(191, (m_q14 - 16384) >> 8)])
                t = int(self._rshift_rne(m_q14 * r0 * r0, 28))
                r1 = int(self._rshift_rne(r0 * (3 * (1 << 14) - t), 15))
                out[row_start:row_start + 48] = self._rshift_rne(row * gamma * r1, 14 + gamma_shift + exponent // 2)
            self._vec_write_codes(pe_id, dst, np.clip(out, -128, 127))
        elif subop == VEC_SUBOP_VPOS_ADD:
            record = self._vec_record(pe_id, record_addr, VEC_PARAM_KIND_VPOS_ADD)
            if any(record[index] for index in (4, 5, 6, 7)) or (record[3] & 0xFF00):
                raise ValueError("VPOS_ADD parameter record has nonzero reserved bits")
            pos_base, ea, eb, eo = record[1], self._vec_i8(record[2]), self._vec_i8(record[2] >> 8), self._vec_i8(record[3])
            if not all(-31 <= value <= 31 for value in (ea, eb, eo)) or eo < max(ea, eb):
                raise ValueError("invalid VPOS_ADD exponent record")
            pos_range = self._vec_check_range(pos_base, 1536, "position table")
            self._vec_no_partial_overlap(dst_range, pos_range, allow_exact=False, label="dst/position")
            self._vec_no_partial_overlap(src0_range, pos_range, allow_exact=False, label="src0/position")
            a, b = self._vec_read_codes(pe_id, src0, length), self._vec_read_codes(pe_id, pos_base, length)
            self._vec_write_codes(pe_id, dst, np.clip(self._rshift_rne(a, eo - ea) + self._rshift_rne(b, eo - eb), -128, 127))
        elif subop == VEC_SUBOP_VSOFTMAX:
            record = self._vec_record(pe_id, record_addr, VEC_PARAM_KIND_VSOFTMAX)
            if any(record[index] for index in range(2, 8)):
                raise ValueError("VSOFTMAX parameter record has nonzero reserved words")
            scratch_range = self._vec_check_range(record[1], 1024, "softmax scratch")
            self._vec_no_partial_overlap(dst_range, scratch_range, allow_exact=False, label="dst/scratch")
            self._vec_no_partial_overlap(src0_range, scratch_range, allow_exact=False, label="src0/scratch")
            scores, out = self._vec_read_codes(pe_id, src0, length), np.zeros(length, dtype=np.int64)
            for row in range(32):
                start, count = row * 32, row + 1
                values = scores[start:start + count]
                rel = np.clip(values - int(np.max(values)), -128, 0)
                exp = EXP_Q15[(-rel).astype(np.int16)].astype(np.int64)
                total = int(exp.sum())
                if total <= 0:
                    raise ValueError("VSOFTMAX causal row has zero denominator")
                out[start:start + count] = (255 * exp + total // 2) // total
            self._vec_write_codes(pe_id, dst, out, unsigned=True)
        else:
            raise ValueError(f"unknown/reserved VEC subop {subop}")

    def _execute_tile_gemm(self, pe_id: int, desc: dict) -> None:
        """Golden-model expansion of TILE_GEMM into LOAD/RUN semantics.

        Requires hoist_payloads=True for Part A v1 (payloads already in buffer).
        Weight-overlap flag is parked and ignored here.
        """
        if int(desc["array_size"]) != int(self.array_size):
            raise ValueError(
                f"TILE_GEMM array_size={desc['array_size']} != simulator array_size={self.array_size}"
            )
        if int(desc["dtype_code"]) != int(self.cfg.compute_data_width):
            raise ValueError(
                f"TILE_GEMM dtype_code={desc['dtype_code']} != compute_data_width={self.cfg.compute_data_width}"
            )
        if not desc["hoist_payloads"]:
            raise ValueError(
                "TILE_GEMM golden model v1 requires hoist_payloads=1 "
                "(non-hoist payload stores remain host BSTORE/STORE)"
            )
        if desc.get("weight_overlap"):
            # Parked: Approach A concurrency not part of Part A golden expansion.
            pass

        n = int(self.array_size)
        batch = int(desc["batch_size"])
        out_blocks = int(desc["out_blocks"])
        in_blocks = int(desc["in_blocks"])
        items = int(self.cfg.items_per_word)
        w_words = (n * n) // items
        x_words = (n * batch) // items
        # Match lowering_blocked_fc_utpu hoist result stride (padded batch cols to N).
        out_chunks = max(1, (batch + n - 1) // n)
        y_words = ((n * n) // items) * out_chunks
        weight_base = int(desc["weight_base"])
        input_base = int(desc["input_base"])
        result_base = int(desc["result_base"])

        for ob in range(out_blocks):
            out_addr = result_base + ob * y_words
            for ib in range(in_blocks):
                w_addr = weight_base + (ob * in_blocks + ib) * w_words
                x_addr = input_base + ib * x_words
                self._load_weights(pe_id, w_addr, to_shadow_only=False)
                self._load_inputs(pe_id, x_addr, batch_count=batch)
                acc_clear = bool(desc["acc_clear_first"]) and (ib == 0)
                self._run_accumulate(pe_id, acc_clear=acc_clear, batch_count=batch)
            if desc["finalize_last"]:
                # Part A v1: quantize+writeback only. ReLU is not in the frozen
                # TILE_GEMM flags yet (matches blocked-FC calls with apply_relu=False).
                self._run_finalize(
                    pe_id,
                    out_addr,
                    quantize=True,
                    relu=False,
                    residual_addr=None,
                    batch_count=batch,
                )

    def _record_cycle(self, pe_id: int, cost: int = 1) -> None:
        self._cycle_count_sequential += int(cost)
        self._current_section[pe_id] = self._current_section.get(pe_id, 0) + int(cost)

    def _requant_for_row(self, row: int) -> Tuple[int, int]:
        return int(self._requant_multipliers[row]), int(self._requant_right_shifts[row])

    def _close_barrier_section(self, barrier_id: int) -> None:
        self._barrier_hits[barrier_id] = self._barrier_hits.get(barrier_id, 0) + 1
        self._section_pe_cycles.append(dict(self._current_section))
        self._current_section = {pe: 0 for pe in range(self.num_pe)}

    def _parallel_cycle_estimate(self) -> int:
        total = 0
        for section in self._section_pe_cycles:
            if section:
                total += max(section.values())
            else:
                total += 0
        return int(total)

    # ------------------------------------------------------------------
    # main interpreter loop
    # ------------------------------------------------------------------

    def run_words(
        self,
        words: List[int],
        max_steps: Optional[int] = None,
        *,
        preserve_buffer: bool = False,
    ) -> ISASimulationResult:
        """Execute one program; optionally retain preloaded unified-buffer words.

        ``preserve_buffer=True`` models A5 host fills between invocations while
        resetting all PE/accumulator/perf state. It is intended for replaying a
        control-only image with a fixed weight image and changing input region.
        """
        self.reset(preserve_buffer=preserve_buffer)
        cfg = self.cfg
        pc = 0
        steps = 0
        halted = False
        max_steps = max_steps or (len(words) * 8 + 1024)
        fetch_pe = 0

        addr_mask = cfg.address_max
        extended_addr = cfg.extended_address

        def consume_addr_word() -> int:
            """Pop the address word that follows an extended-format header."""
            nonlocal pc
            if pc >= len(words):
                raise ValueError("truncated extended-address payload")
            value = words[pc] & addr_mask
            pc += 1
            return value

        while pc < len(words) and steps < max_steps:
            steps += 1
            instruction = words[pc] & 0xFFFF
            pe_id = int(self._active_pe)
            # Descriptor space is opt-in. Do not reinterpret legacy FETCH with
            # top-half set (low nibble 0x9) unless descriptor ISA is enabled.
            low4 = instruction & 0xF
            if self.cfg.descriptor_isa and low4 == OPCODE_TILE_GEMM:
                opcode = OPCODE_TILE_GEMM
            elif self.cfg.descriptor_isa and self.cfg.vector_unit and instruction == VEC_ESCAPE_WORD:
                opcode = OPCODE_VEC
            elif self.cfg.descriptor_isa and not self.cfg.vector_unit and instruction == VEC_ESCAPE_WORD:
                raise ValueError("VEC requires IsaConfig(descriptor_isa=True, vector_unit=True)")
            else:
                opcode = instruction & 0x7
            pc += 1

            if opcode == OPCODE_TILE_GEMM:
                if not self.cfg.descriptor_isa:
                    raise ValueError("TILE_GEMM requires IsaConfig(descriptor_isa=True)")
                # Header already consumed; need words[pc-1] plus 5 trailers.
                desc_words = [instruction] + list(words[pc : pc + TILE_GEMM_WORD_COUNT - 1])
                if len(desc_words) < TILE_GEMM_WORD_COUNT:
                    raise ValueError("truncated TILE_GEMM descriptor")
                pc += TILE_GEMM_WORD_COUNT - 1
                desc = decodeTileGemmWords(desc_words, cfg=self.cfg)
                self._execute_tile_gemm(pe_id, desc)
                self.executed_ops.setdefault("tile_gemm", 0)
                self.executed_ops["tile_gemm"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_VEC:
                desc_words = [instruction] + list(words[pc : pc + VEC_WORD_COUNT - 1])
                if len(desc_words) < VEC_WORD_COUNT:
                    raise ValueError("truncated VEC descriptor")
                pc += VEC_WORD_COUNT - 1
                desc = decodeVecWords(desc_words, cfg=self.cfg)
                self._execute_vec(pe_id, desc)
                self.executed_ops.setdefault("vec", 0)
                self.executed_ops["vec"] += 1
                name = f"vec_{desc['subop']}"
                self.executed_ops[name] = self.executed_ops.get(name, 0) + 1
                self._record_cycle(pe_id, cost=max(1, int(desc["length"]) // 2))

            elif opcode == OPCODE_STORE:
                # STORE is multi-word in both legacy and extended layouts.
                # word2 = source / immediate; word3 = destination address
                # (low ``address_width`` bits in extended mode, low 9 bits
                # in legacy mode for byte-identical decoding).
                if pc + 1 >= len(words):
                    raise ValueError("truncated STORE instruction")
                source = words[pc] & 0xFFFF
                if extended_addr:
                    dest = words[pc + 1] & addr_mask
                else:
                    dest = words[pc + 1] & 0x1FF
                pc += 2
                immediate_mode = bool((instruction >> 4) & 0x1)
                if immediate_mode:
                    src_value = source
                else:
                    src_addr = source & (addr_mask if extended_addr else 0x1FF)
                    src_value = self._read_word(pe_id, src_addr)
                self._write_word(pe_id, dest, src_value)
                self._record_host_store(pe_id, dest, src_value)
                self.executed_ops["store"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_BSTORE:
                # Legacy: header has addr<<7, next word is count.
                # Extended: header has no addr, next words are addr and count.
                if extended_addr:
                    base = consume_addr_word()
                else:
                    base = self._legacy_addr(instruction)
                if pc >= len(words):
                    raise ValueError("truncated BSTORE count")
                count = words[pc] & 0xFFFF
                pc += 1
                if pc + count > len(words):
                    raise ValueError("truncated BSTORE payload")
                for i in range(count):
                    payload = words[pc + i]
                    self._write_word(pe_id, base + i, payload)
                    self._record_host_store(pe_id, base + i, payload)
                pc += count
                self.executed_ops["bstore"] += 1
                self._record_cycle(pe_id, cost=2 + int(count))

            elif opcode == OPCODE_LOAD:
                if extended_addr:
                    addr = consume_addr_word()
                else:
                    addr = self._legacy_addr(instruction)
                load_batch_count = (((instruction >> 8) & 0xFF) + 1) if extended_addr and (((instruction >> 3) & 0x1) == 0) else 1
                is_weights = bool((instruction >> 3) & 0x1)
                commit_after = bool((instruction >> 4) & 0x1) and is_weights
                if is_weights:
                    if commit_after or self._weight_overlap_active:
                        self._weight_overlap_active = True
                        self._load_weights(pe_id, addr, to_shadow_only=True)
                        if commit_after:
                            self._commit_weights(pe_id)
                    else:
                        self._load_weights(pe_id, addr, to_shadow_only=False)
                else:
                    self._load_inputs(pe_id, addr, batch_count=load_batch_count)
                self.executed_ops["load"] += 1
                if is_weights:
                    self.executed_ops.setdefault("load_weights", 0)
                    self.executed_ops["load_weights"] += 1
                else:
                    self.executed_ops.setdefault("load_inputs", 0)
                    self.executed_ops["load_inputs"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_RUN:
                if extended_addr:
                    result_addr = consume_addr_word()
                else:
                    result_addr = self._legacy_addr(instruction)
                compute = bool((instruction >> 3) & 0x1)
                quantize = bool((instruction >> 4) & 0x1)
                relu = bool((instruction >> 5) & 0x1)
                acc_clear = bool((instruction >> 6) & 0x1)
                residual_en = bool((instruction >> 7) & 0x1) if extended_addr else False
                run_batch_count = (((instruction >> 8) & 0xFF) + 1) if extended_addr else 1
                if run_batch_count != int(self._loaded_input_batch_count):
                    raise ValueError(
                        f"RUN batch_count={run_batch_count} does not match last LOAD_INPUTS "
                        f"batch_count={self._loaded_input_batch_count}"
                    )
                residual_addr = None
                if residual_en:
                    residual_addr = consume_addr_word()
                if compute and not quantize and not relu:
                    self._run_accumulate(pe_id, acc_clear=acc_clear, batch_count=run_batch_count)
                    if self._weight_overlap_active:
                        self._commit_weights(pe_id)
                elif (not compute) and quantize:
                    self._run_finalize(
                        pe_id,
                        result_addr,
                        quantize=quantize,
                        relu=relu,
                        residual_addr=residual_addr,
                        batch_count=run_batch_count,
                    )
                    if residual_en:
                        self.executed_ops["residual_add"] += 1
                elif compute and quantize:
                    self._run_accumulate(pe_id, acc_clear=True, batch_count=run_batch_count)
                    self._run_finalize(
                        pe_id,
                        result_addr,
                        quantize=quantize,
                        relu=relu,
                        residual_addr=residual_addr,
                        batch_count=run_batch_count,
                    )
                    if residual_en:
                        self.executed_ops["residual_add"] += 1
                else:
                    raise ValueError(
                        "unsupported RUN mode: "
                        f"compute={compute} quantize={quantize} relu={relu}"
                    )
                self.executed_ops["run"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_FETCH:
                if extended_addr:
                    addr = consume_addr_word()
                else:
                    addr = self._legacy_addr(instruction)
                word = self._read_word(pe_id, addr)
                high_byte = bool((instruction >> 3) & 0x1)
                self.fetch_bytes.append((word >> 8) & 0xFF if high_byte else word & 0xFF)
                fetch_pe = pe_id
                self.executed_ops["fetch"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_HALT:
                self.executed_ops["halt"] += 1
                self._record_cycle(pe_id)
                halted = True
                break

            elif opcode == OPCODE_NOP:
                if extended_addr and ((instruction >> 3) & 0x1) == NOP_SUBOP_REQUANT:
                    self._requant_enable = bool((instruction >> 4) & 0x1)
                    if (instruction >> 5) & 0x1:
                        count = ((instruction >> 8) & 0xFF) + 1
                        if count > self.array_size:
                            raise ValueError(
                                f"per-channel requant vector length {count} exceeds array_size={self.array_size}"
                            )
                        if pc + (2 * count) - 1 >= len(words):
                            raise ValueError("truncated per-channel REQUANT trailer")
                        self._requant_multipliers = [1 for _ in range(self.array_size)]
                        self._requant_right_shifts = [0 for _ in range(self.array_size)]
                        for idx in range(count):
                            self._requant_multipliers[idx] = words[pc] & 0xFFFF
                            self._requant_right_shifts[idx] = words[pc + 1] & 0xFFFF
                            pc += 2
                    else:
                        if pc + 1 >= len(words):
                            raise ValueError("truncated REQUANT trailer")
                        multiplier = words[pc] & 0xFFFF
                        right_shift = words[pc + 1] & 0xFFFF
                        self._requant_multipliers = [multiplier for _ in range(self.array_size)]
                        self._requant_right_shifts = [right_shift for _ in range(self.array_size)]
                        pc += 2
                self.executed_ops["nop"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_DTYPE:
                subop = (instruction >> 5) & 0x3
                if subop == DTYPE_SUBOP_BUFFER_XFER:
                    if extended_addr:
                        # Extended layout: src_addr, dst_addr, count each
                        # consume their own 16-bit word.
                        if pc + 2 >= len(words):
                            raise ValueError("truncated extended BUFFER_XFER trailer")
                        src_addr = consume_addr_word()
                        dst_addr = consume_addr_word()
                        count = words[pc] & 0xFFFF
                        pc += 1
                    else:
                        if pc >= len(words):
                            raise ValueError("truncated BUFFER_XFER trailer")
                        src_addr = self._legacy_addr(instruction)
                        trailer = words[pc] & 0xFFFF
                        pc += 1
                        dst_addr = trailer & 0x1FF
                        count = (trailer >> 9) & 0x7F
                    dst_pe = (instruction >> 3) & 0x1
                    src_pe = (instruction >> 4) & 0x1
                    self._buffer_xfer(src_pe, dst_pe, src_addr, dst_addr, count)
                    self.executed_ops["buffer_xfer"] += 1
                    self._record_cycle(pe_id, cost=2 + int(count))
                elif subop == DTYPE_SUBOP_BARRIER:
                    if extended_addr:
                        barrier_id = consume_addr_word()
                    else:
                        barrier_id = self._legacy_addr(instruction)
                    self.executed_ops["barrier"] += 1
                    self._record_cycle(pe_id)
                    self._close_barrier_section(barrier_id)
                elif subop == DTYPE_SUBOP_ACC_ADD:
                    dst_pe = (instruction >> 3) & 0x1
                    src_pe = (instruction >> 4) & 0x1
                    self._acc_add(dst_pe, src_pe)
                    self.executed_ops["acc_add"] += 1
                    self._record_cycle(dst_pe)
                elif subop == DTYPE_SUBOP_PE_SELECT:
                    selected = (instruction >> 3) & 0x1
                    if selected >= self.num_pe:
                        raise ValueError(
                            f"PE_SELECT targets pe_id={selected} but num_pe={self.num_pe}"
                        )
                    self._active_pe = int(selected)
                    self.executed_ops["pe_select"] += 1
                    self._record_cycle(self._active_pe)
                else:
                    raise ValueError(f"unsupported D-type subop {subop} at pc={pc - 1}")

            else:
                raise ValueError(f"unsupported opcode {opcode} at pc={pc - 1}")

        if steps >= max_steps and not halted:
            raise TimeoutError(f"ISA simulation exceeded max_steps={max_steps}")

        nonzero_buffer = {i: w for i, w in enumerate(self.pes[fetch_pe].buffer) if w != 0}
        parallel_estimate = self._parallel_cycle_estimate() + sum(self._current_section.values())
        total_macs = int(self._total_macs)
        cycles_total = int(self._cycle_count_sequential)
        cycles_per_mac = (cycles_total / total_macs) if total_macs > 0 else 0.0
        denom = cycles_total * int(self.array_size) * int(self.array_size)
        array_utilization = (total_macs / denom) if denom > 0 else 0.0
        return ISASimulationResult(
            halted=halted,
            pc=pc,
            fetch_bytes=list(self.fetch_bytes),
            buffer_words=nonzero_buffer,
            instruction_count=len(words),
            executed_ops=dict(self.executed_ops),
            cycle_count_sequential=cycles_total,
            cycle_count_parallel_estimate=int(parallel_estimate if self.num_pe > 1 else cycles_total),
            num_pe=int(self.num_pe),
            store_bytes_total=int(self._store_bytes_total),
            redundant_store_bytes=int(self._redundant_store_bytes),
            compute_runs=int(self._compute_runs),
            total_macs=int(total_macs),
            cycles_per_mac=float(cycles_per_mac),
            array_utilization=float(array_utilization),
        )


# ---------------------------------------------------------------------------
# Module-level convenience entry points
# ---------------------------------------------------------------------------


def words_from_bytes(program: bytes) -> List[int]:
    if len(program) % 2 != 0:
        raise ValueError("program bytes must contain complete 16-bit words")
    return [
        int.from_bytes(program[i : i + 2], byteorder="little", signed=False)
        for i in range(0, len(program), 2)
    ]


def words_from_mem(path: str) -> List[int]:
    words: List[int] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        words.append(int(stripped, 16) & 0xFFFF)
    return words


def program_uses_multi_pe(words: List[int]) -> bool:
    for word in words:
        if (word & 0x7) == OPCODE_DTYPE:
            subop = (word >> 5) & 0x3
            if subop in (DTYPE_SUBOP_BUFFER_XFER, DTYPE_SUBOP_BARRIER, DTYPE_SUBOP_ACC_ADD, DTYPE_SUBOP_PE_SELECT):
                return True
    return False


def simulate_words(
    words: List[int],
    array_size: int = 16,
    buffer_size: int = 512,
    num_pe: Optional[int] = None,
    cfg: Optional[IsaConfig] = None,
    accumulator_data_width: int = 16,
) -> ISASimulationResult:
    resolved_num_pe = int(num_pe) if num_pe is not None else (2 if program_uses_multi_pe(words) else 1)
    return UTPUISASimulator(
        array_size=array_size,
        buffer_size=buffer_size,
        num_pe=resolved_num_pe,
        cfg=cfg,
        accumulator_data_width=accumulator_data_width,
    ).run_words(words)


def simulate_program_bytes(
    program: bytes,
    array_size: int = 16,
    buffer_size: int = 512,
    num_pe: Optional[int] = None,
    cfg: Optional[IsaConfig] = None,
    accumulator_data_width: int = 16,
) -> ISASimulationResult:
    return simulate_words(
        words_from_bytes(program),
        array_size=array_size,
        buffer_size=buffer_size,
        num_pe=num_pe,
        cfg=cfg,
        accumulator_data_width=accumulator_data_width,
    )


def simulate_mem_file(
    path: str,
    array_size: int = 16,
    buffer_size: int = 512,
    num_pe: Optional[int] = None,
    cfg: Optional[IsaConfig] = None,
    accumulator_data_width: int = 16,
) -> ISASimulationResult:
    return simulate_words(
        words_from_mem(path),
        array_size=array_size,
        buffer_size=buffer_size,
        num_pe=num_pe,
        cfg=cfg,
        accumulator_data_width=accumulator_data_width,
    )
