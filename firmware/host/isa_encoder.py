"""uTPU ISA encoder.

Phase 4 Stage A widening (additive only): the encoder now carries an
``IsaConfig`` describing the target's ``instruction_width``,
``address_width``, ``compute_data_width`` and ``items_per_word``.

* ``IsaConfig.legacy()`` (== the module-level ``DEFAULT_CFG``) reproduces
  today's INT4 / 9-bit-address single-word LOAD/RUN/FETCH/BSTORE/DTYPE
  encoding **byte-identically**.
* When ``address_width > 9`` (i.e. the target buffer needs more than the
  9 high bits of a 16-bit instruction word), LOAD/RUN/FETCH/BSTORE and
  the address-bearing D-type sub-ops are emitted in a 2-word
  ``extended-address`` form: word1 carries opcode + flags only, word2
  carries the full address in its low ``address_width`` bits. STORE is
  already multi-word and stores its address in word3 — when
  ``address_width > 9``, word3 simply holds a wider field.

This keeps existing programs / tests / RTL bitmatch artefacts byte-
identical (they all run with ``DEFAULT_CFG``) while exposing the wider
buffer / dtype configuration that Phase 4's RTL widen will exercise in
follow-up sim work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import struct


OPCODE_STORE = 0b000  # 0 - store data to buffer
OPCODE_FETCH = 0b001  # 1 - fetch data from buffer
OPCODE_RUN = 0b010    # 2 - execute
OPCODE_LOAD = 0b011   # 3 - load data into PE array
OPCODE_HALT = 0b100   # 4 - stop execution
OPCODE_NOP = 0b101    # 5 - no operation
OPCODE_BSTORE = 0b110 # 6 - burst store sequential 16-bit words to buffer
OPCODE_DTYPE = 0b111  # 7 - multi-PE / dataflow extensions (sim-validated; not in current RTL)

DTYPE_SUBOP_BUFFER_XFER = 0b00
DTYPE_SUBOP_BARRIER = 0b01
DTYPE_SUBOP_ACC_ADD = 0b10
DTYPE_SUBOP_PE_SELECT = 0b11

PE_ID_SHIFT = 12

# Legacy constants kept for callers that import them by name.
INSTRUCTION_WIDTH = 16
ADDRESS_WIDTH = 9


@dataclass(frozen=True)
class IsaConfig:
    """Target-shape parameters that change the encoder's bit-layout.

    Attributes
    ----------
    instruction_width
        Bits per instruction word. Currently locked to 16 (matches RTL
        ``BUFFER_WORD_SIZE``); future widening would touch the FSM and
        UART upload protocol so it stays at 16 here.
    address_width
        Bits required to address the unified buffer
        (``$clog2(BUFFER_SIZE)`` on the RTL side). When this exceeds the
        9 high bits of a 16-bit instruction word (i.e. > 9), the encoder
        emits a 2-word extended-address format for LOAD/RUN/FETCH/BSTORE
        and the address-bearing D-type sub-ops.
    compute_data_width
        Bits per compute element (4 for INT4, 8 for INT8). Drives how
        many elements pack into a 16-bit ``BUFFER_WORD_SIZE`` slot.
    items_per_word
        ``instruction_width // compute_data_width``. Cached for clarity.
    """

    instruction_width: int = 16
    address_width: int = 9
    compute_data_width: int = 4

    @property
    def items_per_word(self) -> int:
        return self.instruction_width // self.compute_data_width

    @property
    def address_max(self) -> int:
        return (1 << self.address_width) - 1

    @property
    def extended_address(self) -> bool:
        return self.address_width > 9

    def __post_init__(self) -> None:
        if self.instruction_width != 16:
            raise ValueError(
                f"instruction_width must be 16 (RTL FSM lock); got {self.instruction_width}"
            )
        if self.compute_data_width <= 0 or self.compute_data_width > 16:
            raise ValueError(
                f"compute_data_width must be in (0, 16]; got {self.compute_data_width}"
            )
        if 16 % self.compute_data_width != 0:
            raise ValueError(
                "compute_data_width must evenly divide instruction_width=16"
            )
        if self.address_width < 9 or self.address_width > 16:
            raise ValueError(
                f"address_width must be in [9, 16]; got {self.address_width}"
            )


# Legacy default — byte-identical to pre-Phase-4 behaviour. Every
# existing call site that does not pass ``cfg=`` resolves to this.
DEFAULT_CFG = IsaConfig(instruction_width=16, address_width=9, compute_data_width=4)


def _resolve_cfg(cfg: Optional[IsaConfig]) -> IsaConfig:
    return cfg if cfg is not None else DEFAULT_CFG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tag_pe(instruction: int, pe_id: int) -> int:
    """Tag an encoded 16-bit instruction word with target PE id (0 or 1)."""
    if pe_id not in (0, 1):
        raise ValueError(f"pe_id must be 0 or 1, got {pe_id}")
    return int(instruction) & ~(1 << PE_ID_SHIFT) | ((int(pe_id) & 1) << PE_ID_SHIFT)


def pe_id_from_word(word: int) -> int:
    return (int(word) >> PE_ID_SHIFT) & 1


def int4To16(values: List[int]) -> int:
    """Pack up to 4 signed INT4 values into a 16-bit word (little-end nibble first)."""
    while len(values) < 4:
        values = values + [0]

    result = 0
    for i, val in enumerate(values):
        nibble = int(val) & 0xF
        result |= (nibble << (i * 4))
    return result


def pack_values_to_word(values: List[int], cfg: Optional[IsaConfig] = None) -> int:
    """Generalisation of ``int4To16`` for arbitrary ``compute_data_width``.

    ``compute_data_width=4`` reproduces ``int4To16`` byte-identically.
    """
    cfg = _resolve_cfg(cfg)
    width = cfg.compute_data_width
    items = cfg.items_per_word
    mask = (1 << width) - 1
    while len(values) < items:
        values = values + [0]
    word = 0
    for i, val in enumerate(values[:items]):
        word |= (int(val) & mask) << (i * width)
    return word & ((1 << cfg.instruction_width) - 1)


def encodeAddress(addr: int, cfg: Optional[IsaConfig] = None) -> int:
    cfg = _resolve_cfg(cfg)
    if not 0 <= addr <= cfg.address_max:
        raise ValueError(
            f"Address {addr} out of range (0-{cfg.address_max}) for "
            f"address_width={cfg.address_width}"
        )
    return int(addr)


def instructionToBytes(instruction: int) -> bytes:
    return struct.pack("<H", instruction & 0xFFFF)


# ---------------------------------------------------------------------------
# STORE family (already multi-word; address always lives in word3)
# ---------------------------------------------------------------------------


def encodeStoreValues(addr: int, values: List[int], cfg: Optional[IsaConfig] = None) -> bytes:
    """Encode a STORE-immediate: 3 words.

    Layout (unchanged):
        word1 = OPCODE_STORE | (1<<4)       # immediate-mode flag
        word2 = packed INT4 / INT8 values   # `pack_values_to_word`
        word3 = destination address (low ``address_width`` bits)
    """
    cfg = _resolve_cfg(cfg)
    addr = encodeAddress(addr, cfg)
    word1 = OPCODE_STORE | (1 << 4)
    word2 = pack_values_to_word(values, cfg)
    word3 = addr
    return instructionToBytes(word1) + instructionToBytes(word2) + instructionToBytes(word3)


def encodeStoreAddress(destAddr: int, srcAddr: int, cfg: Optional[IsaConfig] = None) -> bytes:
    """Encode a STORE-from-buffer-address: 3 words.

    word1 = OPCODE_STORE  (bit 4 cleared = address mode)
    word2 = source address (low ``address_width`` bits)
    word3 = destination address (low ``address_width`` bits)
    """
    cfg = _resolve_cfg(cfg)
    destAddr = encodeAddress(destAddr, cfg)
    srcAddr = encodeAddress(srcAddr, cfg)
    word1 = OPCODE_STORE  # bit 4 = 0 -> address-mode
    word2 = srcAddr
    word3 = destAddr
    return instructionToBytes(word1) + instructionToBytes(word2) + instructionToBytes(word3)


# ---------------------------------------------------------------------------
# Address-bearing single-flag ops: LOAD / RUN / FETCH / BSTORE / DTYPE-with-addr
#
# Legacy 9-bit format (``cfg.address_width == 9``) packs the address into
# bits 7..15 of the same instruction word: byte-identical to pre-Phase-4.
# Extended format (``cfg.address_width > 9``) emits two words:
#   word1 = opcode | flags                 (no address)
#   word2 = address (low ``address_width`` bits)
# ---------------------------------------------------------------------------


def _emit_with_addr(header_no_addr: int, addr: int, cfg: IsaConfig) -> bytes:
    if cfg.extended_address:
        return instructionToBytes(header_no_addr) + instructionToBytes(addr & 0xFFFF)
    # Legacy: pack address into bits 7..15 of header word.
    return instructionToBytes(header_no_addr | (addr << 7))


def encodeLoad(addr: int, is_weights: bool, cfg: Optional[IsaConfig] = None) -> bytes:
    cfg = _resolve_cfg(cfg)
    addr = encodeAddress(addr, cfg)
    header = OPCODE_LOAD | ((1 if is_weights else 0) << 3)
    return _emit_with_addr(header, addr, cfg)


def encodeLoadWeights(addr: int, cfg: Optional[IsaConfig] = None) -> bytes:
    return encodeLoad(addr, is_weights=True, cfg=cfg)


def encodeLoadInputs(addr: int, cfg: Optional[IsaConfig] = None) -> bytes:
    return encodeLoad(addr, is_weights=False, cfg=cfg)


def encodeRun(
    result_addr: int,
    compute_en: bool = True,
    quantize_en: bool = True,
    relu_en: bool = True,
    acc_clear_en: bool = False,
    cfg: Optional[IsaConfig] = None,
) -> bytes:
    """RUN encoding.

    Legacy 9-bit format (cfg.address_width == 9):
        bits 0-2 : OPCODE
        bit 3    : compute_en
        bit 4    : quantize_en
        bit 5    : relu_en
        bit 6    : acc_clear_en
        bits 7-15: result address

    Extended format (cfg.address_width > 9):
        word1 = opcode | flags (bits 3..6); bits 7..15 unused (zero)
        word2 = result address (low ``address_width`` bits)
    """
    cfg = _resolve_cfg(cfg)
    result_addr = encodeAddress(result_addr, cfg)
    header = (
        OPCODE_RUN
        | ((1 if compute_en else 0) << 3)
        | ((1 if quantize_en else 0) << 4)
        | ((1 if relu_en else 0) << 5)
        | ((1 if acc_clear_en else 0) << 6)
    )
    return _emit_with_addr(header, result_addr, cfg)


def encodeFetch(addr: int, top_half: bool = True, cfg: Optional[IsaConfig] = None) -> bytes:
    """FETCH encoding.

    Legacy 9-bit format:
        bits 0-2 : OPCODE
        bit 3    : top/bottom selector
        bits 4-6 : reserved
        bits 7-15: address

    Extended format mirrors RUN: opcode|flags + address word.
    """
    cfg = _resolve_cfg(cfg)
    addr = encodeAddress(addr, cfg)
    header = OPCODE_FETCH | ((1 if top_half else 0) << 3)
    return _emit_with_addr(header, addr, cfg)


def encodeHalt(cfg: Optional[IsaConfig] = None) -> bytes:
    return instructionToBytes(OPCODE_HALT)


def encodeNop(cfg: Optional[IsaConfig] = None) -> bytes:
    return instructionToBytes(OPCODE_NOP)


# ---------------------------------------------------------------------------
# D-type extensions (multi-PE)
# ---------------------------------------------------------------------------


def encodePeSelect(pe_id: int, cfg: Optional[IsaConfig] = None) -> bytes:
    if pe_id not in (0, 1):
        raise ValueError(f"pe_id must be 0 or 1, got {pe_id}")
    header = OPCODE_DTYPE | (DTYPE_SUBOP_PE_SELECT << 5) | ((int(pe_id) & 1) << 3)
    return instructionToBytes(header)


def encodeDType(subop: int, pe_id: int = 0) -> int:
    if subop not in (DTYPE_SUBOP_BUFFER_XFER, DTYPE_SUBOP_BARRIER, DTYPE_SUBOP_ACC_ADD, DTYPE_SUBOP_PE_SELECT):
        raise ValueError(f"unsupported D-type subop: {subop}")
    return OPCODE_DTYPE | ((int(subop) & 0x3) << 5)


def encodeBufferXfer(
    src_addr: int,
    dst_addr: int,
    count: int,
    src_pe: int = 0,
    dst_pe: int = 1,
    cfg: Optional[IsaConfig] = None,
) -> bytes:
    """Copy ``count`` buffer words from src_pe[src_addr:] to dst_pe[dst_addr:].

    Legacy layout (cfg.address_width == 9):
        word0 = [111][dst_pe:bit3][src_pe:bit4][00 at bits5-6][src_addr<<7]
        word1 = [dst_addr:bits0-8][count:bits9-15]    (count limited to 7 bits)

    Extended layout (cfg.address_width > 9):
        word0 = [111][dst_pe:bit3][src_pe:bit4][00 at bits5-6]
        word1 = src_addr (full)
        word2 = dst_addr (full)
        word3 = count
    The wider count field also lifts the 7-bit cap.
    """
    cfg = _resolve_cfg(cfg)
    src_addr = encodeAddress(src_addr, cfg)
    dst_addr = encodeAddress(dst_addr, cfg)
    if cfg.extended_address:
        if count <= 0 or count > 0xFFFF:
            raise ValueError(f"buffer xfer count must be in 1..65535, got {count}")
        header = encodeDType(DTYPE_SUBOP_BUFFER_XFER, pe_id=0)
        header |= (int(dst_pe) & 1) << 3
        header |= (int(src_pe) & 1) << 4
        return (
            instructionToBytes(header)
            + instructionToBytes(src_addr & 0xFFFF)
            + instructionToBytes(dst_addr & 0xFFFF)
            + instructionToBytes(int(count) & 0xFFFF)
        )
    if count <= 0 or count > 0x7F:
        raise ValueError(f"buffer xfer count must be in 1..127, got {count}")
    header = encodeDType(DTYPE_SUBOP_BUFFER_XFER, pe_id=0)
    header |= (int(dst_pe) & 1) << 3
    header |= (int(src_pe) & 1) << 4
    header |= (src_addr << 7)
    trailer = (dst_addr & 0x1FF) | ((int(count) & 0x7F) << 9)
    return instructionToBytes(header) + instructionToBytes(trailer)


def encodeBarrier(barrier_id: int = 0, cfg: Optional[IsaConfig] = None) -> bytes:
    cfg = _resolve_cfg(cfg)
    barrier_id = encodeAddress(barrier_id, cfg)
    header = encodeDType(DTYPE_SUBOP_BARRIER)
    return _emit_with_addr(header, barrier_id, cfg)


def encodeAccAdd(src_pe: int, dst_pe: int = 0, cfg: Optional[IsaConfig] = None) -> bytes:
    if src_pe == dst_pe:
        raise ValueError("ACC_ADD requires distinct src_pe and dst_pe")
    header = encodeDType(DTYPE_SUBOP_ACC_ADD)
    header |= (int(dst_pe) & 1) << 3
    header |= (int(src_pe) & 1) << 4
    return instructionToBytes(header)


def encodeBurstStore(addr: int, words: List[int], cfg: Optional[IsaConfig] = None) -> bytes:
    cfg = _resolve_cfg(cfg)
    addr = encodeAddress(addr, cfg)
    if len(words) == 0:
        raise ValueError("BURST_STORE requires at least one word")
    if len(words) > 0xFFFF:
        raise ValueError("BURST_STORE count exceeds 16-bit field")
    if cfg.extended_address:
        header = OPCODE_BSTORE
        payload = [
            instructionToBytes(header),
            instructionToBytes(addr & 0xFFFF),
            instructionToBytes(len(words) & 0xFFFF),
        ]
    else:
        header = OPCODE_BSTORE | (addr << 7)
        payload = [instructionToBytes(header), instructionToBytes(len(words))]
    for w in words:
        payload.append(instructionToBytes(int(w) & 0xFFFF))
    return b"".join(payload)


# ---------------------------------------------------------------------------
# ISAEncoder convenience class
# ---------------------------------------------------------------------------


class ISAEncoder:
    """Stateful encoder; default ``cfg`` is the legacy INT4 / 9-bit layout."""

    def __init__(self, cfg: Optional[IsaConfig] = None):
        self.cfg = _resolve_cfg(cfg)
        self.instructions: List[bytes] = []

    def store(self, addr: int, values: List[int]) -> "ISAEncoder":
        self.instructions.append(encodeStoreValues(addr, values, cfg=self.cfg))
        return self

    def loadWeights(self, addr: int) -> "ISAEncoder":
        self.instructions.append(encodeLoadWeights(addr, cfg=self.cfg))
        return self

    def loadInputs(self, addr: int) -> "ISAEncoder":
        self.instructions.append(encodeLoadInputs(addr, cfg=self.cfg))
        return self

    def run(
        self,
        result_addr: int,
        compute: bool = True,
        quantize: bool = True,
        relu: bool = True,
        acc_clear: bool = False,
    ) -> "ISAEncoder":
        self.instructions.append(
            encodeRun(result_addr, compute, quantize, relu, acc_clear, cfg=self.cfg)
        )
        return self

    def fetch(self, addr: int, top_half: bool = True) -> "ISAEncoder":
        self.instructions.append(encodeFetch(addr, top_half, cfg=self.cfg))
        return self

    def halt(self) -> "ISAEncoder":
        self.instructions.append(encodeHalt(cfg=self.cfg))
        return self

    def nop(self) -> "ISAEncoder":
        self.instructions.append(encodeNop(cfg=self.cfg))
        return self

    def burst_store(self, addr: int, words: List[int]) -> "ISAEncoder":
        self.instructions.append(encodeBurstStore(addr, words, cfg=self.cfg))
        return self

    def buffer_xfer(
        self,
        src_addr: int,
        dst_addr: int,
        count: int,
        src_pe: int = 0,
        dst_pe: int = 1,
    ) -> "ISAEncoder":
        self.instructions.append(
            encodeBufferXfer(src_addr, dst_addr, count, src_pe=src_pe, dst_pe=dst_pe, cfg=self.cfg)
        )
        return self

    def barrier(self, barrier_id: int = 0) -> "ISAEncoder":
        self.instructions.append(encodeBarrier(barrier_id, cfg=self.cfg))
        return self

    def acc_add(self, src_pe: int, dst_pe: int = 0) -> "ISAEncoder":
        self.instructions.append(encodeAccAdd(src_pe, dst_pe=dst_pe, cfg=self.cfg))
        return self

    def pe_select(self, pe_id: int) -> "ISAEncoder":
        self.instructions.append(encodePeSelect(pe_id, cfg=self.cfg))
        return self

    def getProgram(self) -> bytes:
        return b"".join(self.instructions)

    def getInstructionCount(self) -> int:
        return len(self.instructions)

    def clear(self) -> None:
        self.instructions = []


if __name__ == "__main__":
    print("ISA Encoder Test")
    print("=" * 50)
    print("\nIndividual instruction tests (legacy DEFAULT_CFG):")
    store_bytes = encodeStoreValues(0x080, [1, 2, 3, 4])
    print(f"STORE 0x080, [1,2,3,4] (3 words): {store_bytes.hex()}")
    load_w_bytes = encodeLoadWeights(0x080)
    print(f"LOADWEI 0x080: {load_w_bytes.hex()}")
    load_i_bytes = encodeLoadInputs(0x000)
    print(f"LOADIN 0x000: {load_i_bytes.hex()}")
    run_bytes = encodeRun(0x100, True, True, True)
    print(f"RUN 0x100 (all enabled): {run_bytes.hex()}")
    fetch_bytes = encodeFetch(0x100, top_half=True)
    print(f"FETCH 0x100 (top): {fetch_bytes.hex()}")
    halt_bytes = encodeHalt()
    print(f"HALT: {halt_bytes.hex()}")
    print("\n" + "=" * 50)
    print("Encoder class test (legacy):")
    encoder = ISAEncoder()
    encoder.store(0x080, [5, 6, 7, 8])
    encoder.loadWeights(0x080)
    encoder.store(0x000, [1, 2, 0, 0])
    encoder.loadInputs(0x000)
    encoder.run(0x100)
    encoder.fetch(0x100)
    encoder.halt()
    program = encoder.getProgram()
    print(f"Program size: {len(program)} bytes")
    print(f"Instructions: {encoder.getInstructionCount()}")
    print(f"Program hex: {program.hex()}")
    print("\n" + "=" * 50)
    print("Phase 4 extended-address smoke (address_width=14, INT8):")
    cfg = IsaConfig(address_width=14, compute_data_width=8)
    enc2 = ISAEncoder(cfg)
    enc2.store(0x1234, [10, -7])
    enc2.loadWeights(0x1234)
    enc2.run(0x2000)
    enc2.fetch(0x2000)
    enc2.halt()
    p2 = enc2.getProgram()
    print(f"Extended program size: {len(p2)} bytes")
    print(f"Extended hex: {p2.hex()}")
