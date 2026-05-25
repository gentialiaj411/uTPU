from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from isa_encoder import (
    DTYPE_SUBOP_ACC_ADD,
    DTYPE_SUBOP_BARRIER,
    DTYPE_SUBOP_BUFFER_XFER,
    DTYPE_SUBOP_PE_SELECT,
    OPCODE_DTYPE,
)


OPCODE_STORE = 0b000
OPCODE_FETCH = 0b001
OPCODE_RUN = 0b010
OPCODE_LOAD = 0b011
OPCODE_HALT = 0b100
OPCODE_NOP = 0b101
OPCODE_BSTORE = 0b110


def _sign_extend_int4(x: int) -> int:
    x &= 0xF
    return x - 16 if x >= 8 else x


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


def _clip_int4(x: int) -> int:
    if x > 7:
        return 7
    if x < -8:
        return -8
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


@dataclass
class _PEState:
    buffer: List[int]
    weights: List[List[int]]
    inputs: List[int]
    acc_partial_sums: List[int]

    @classmethod
    def create(cls, array_size: int, buffer_size: int) -> "_PEState":
        return cls(
            buffer=[0] * buffer_size,
            weights=[[0 for _ in range(array_size)] for _ in range(array_size)],
            inputs=[0 for _ in range(array_size)],
            acc_partial_sums=[0 for _ in range(array_size)],
        )


class UTPUISASimulator:
    """Bit-level Python simulator for the RTL-observable uTPU ISA behavior.

    Supports optional 2-PE simulation with per-PE program-counter semantics via
    PE tags (bit 12) and D-type barrier / buffer-xfer / acc-add instructions.
    """

    def __init__(self, array_size: int = 16, buffer_size: int = 512, alpha_shift: int = 2, num_pe: int = 1):
        if array_size <= 0 or array_size % 4 != 0:
            raise ValueError("array_size must be a positive multiple of 4")
        if num_pe not in (1, 2):
            raise ValueError("num_pe must be 1 or 2")
        self.array_size = int(array_size)
        self.buffer_size = int(buffer_size)
        self.alpha_shift = int(alpha_shift)
        self.num_pe = int(num_pe)
        self.reset()

    def reset(self) -> None:
        self.pes = [_PEState.create(self.array_size, self.buffer_size) for _ in range(self.num_pe)]
        self.fetch_bytes: List[int] = []
        self.executed_ops = {
            "store": 0,
            "fetch": 0,
            "run": 0,
            "load": 0,
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

    @property
    def buffer(self) -> List[int]:
        return self.pes[0].buffer

    def _pe(self, pe_id: int) -> _PEState:
        if not 0 <= pe_id < self.num_pe:
            raise ValueError(f"invalid pe_id {pe_id} for num_pe={self.num_pe}")
        return self.pes[pe_id]

    def _addr(self, word: int) -> int:
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

    def _load_weights(self, pe_id: int, addr: int) -> None:
        pe = self._pe(pe_id)
        words_needed = (self.array_size * self.array_size) // 4
        flat: List[int] = []
        for offset in range(words_needed):
            flat.extend(_unpack_int4_word(self._read_word(pe_id, addr + offset)))
        for row in range(self.array_size):
            start = row * self.array_size
            pe.weights[row] = flat[start:start + self.array_size]

    def _load_inputs(self, pe_id: int, addr: int) -> None:
        pe = self._pe(pe_id)
        words_needed = self.array_size // 4
        flat: List[int] = []
        for offset in range(words_needed):
            flat.extend(_unpack_int4_word(self._read_word(pe_id, addr + offset)))
        pe.inputs = flat[: self.array_size]

    def _run_accumulate(self, pe_id: int, acc_clear: bool) -> None:
        pe = self._pe(pe_id)
        for row in range(self.array_size):
            lane_val = 0
            for k in range(self.array_size):
                lane_val += pe.weights[row][k] * pe.inputs[k]
            if acc_clear:
                pe.acc_partial_sums[row] = lane_val
            else:
                pe.acc_partial_sums[row] += lane_val

    def _run_finalize(self, pe_id: int, result_addr: int, quantize: bool, relu: bool) -> None:
        pe = self._pe(pe_id)
        outputs = [0 for _ in range(self.array_size)]
        for i, acc in enumerate(pe.acc_partial_sums):
            q = _clip_int4(acc) if quantize else int(acc)
            if relu and q < 0:
                q = q >> self.alpha_shift
            outputs[i] = q

        padded = outputs + [0] * (self.array_size * self.array_size - self.array_size)
        for word_idx in range((self.array_size * self.array_size) // 4):
            chunk = padded[word_idx * 4 : (word_idx + 1) * 4]
            self._write_word(pe_id, result_addr + word_idx, _pack_int4(chunk))

    def _acc_add(self, dst_pe: int, src_pe: int) -> None:
        dst = self._pe(dst_pe)
        src = self._pe(src_pe)
        for i in range(self.array_size):
            dst.acc_partial_sums[i] += int(src.acc_partial_sums[i])

    def _buffer_xfer(self, src_pe: int, dst_pe: int, src_addr: int, dst_addr: int, count: int) -> None:
        for offset in range(int(count)):
            value = self._read_word(src_pe, src_addr + offset)
            self._write_word(dst_pe, dst_addr + offset, value)

    def _record_cycle(self, pe_id: int, cost: int = 1) -> None:
        self._cycle_count_sequential += int(cost)
        self._current_section[pe_id] = self._current_section.get(pe_id, 0) + int(cost)

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

    def run_words(self, words: List[int], max_steps: Optional[int] = None) -> ISASimulationResult:
        self.reset()
        pc = 0
        steps = 0
        halted = False
        max_steps = max_steps or (len(words) * 8 + 1024)
        fetch_pe = 0

        while pc < len(words) and steps < max_steps:
            steps += 1
            instruction = words[pc] & 0xFFFF
            pe_id = int(self._active_pe)
            opcode = instruction & 0x7
            pc += 1

            if opcode == OPCODE_STORE:
                if pc + 1 >= len(words):
                    raise ValueError("truncated STORE instruction")
                source = words[pc] & 0xFFFF
                dest = words[pc + 1] & 0x1FF
                pc += 2
                immediate_mode = bool((instruction >> 4) & 0x1)
                self._write_word(
                    pe_id,
                    dest,
                    source if immediate_mode else self._read_word(pe_id, source & 0x1FF),
                )
                self.executed_ops["store"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_BSTORE:
                if pc >= len(words):
                    raise ValueError("truncated BSTORE count")
                base = self._addr(instruction)
                count = words[pc] & 0xFFFF
                pc += 1
                if pc + count > len(words):
                    raise ValueError("truncated BSTORE payload")
                for i in range(count):
                    self._write_word(pe_id, base + i, words[pc + i])
                pc += count
                self.executed_ops["bstore"] += 1
                self._record_cycle(pe_id, cost=2 + int(count))

            elif opcode == OPCODE_LOAD:
                addr = self._addr(instruction)
                if (instruction >> 3) & 0x1:
                    self._load_weights(pe_id, addr)
                else:
                    self._load_inputs(pe_id, addr)
                self.executed_ops["load"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_RUN:
                result_addr = self._addr(instruction)
                compute = bool((instruction >> 3) & 0x1)
                quantize = bool((instruction >> 4) & 0x1)
                relu = bool((instruction >> 5) & 0x1)
                acc_clear = bool((instruction >> 6) & 0x1)
                if compute and not quantize and not relu:
                    self._run_accumulate(pe_id, acc_clear=acc_clear)
                elif (not compute) and quantize:
                    self._run_finalize(pe_id, result_addr, quantize=quantize, relu=relu)
                elif compute and quantize:
                    self._run_accumulate(pe_id, acc_clear=True)
                    self._run_finalize(pe_id, result_addr, quantize=quantize, relu=relu)
                else:
                    raise ValueError(
                        "unsupported RUN mode: "
                        f"compute={compute} quantize={quantize} relu={relu}"
                    )
                self.executed_ops["run"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_FETCH:
                addr = self._addr(instruction)
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
                self.executed_ops["nop"] += 1
                self._record_cycle(pe_id)

            elif opcode == OPCODE_DTYPE:
                subop = (instruction >> 5) & 0x3
                if subop == DTYPE_SUBOP_BUFFER_XFER:
                    if pc >= len(words):
                        raise ValueError("truncated BUFFER_XFER trailer")
                    src_addr = self._addr(instruction)
                    dst_pe = (instruction >> 3) & 0x1
                    src_pe = (instruction >> 4) & 0x1
                    trailer = words[pc] & 0xFFFF
                    pc += 1
                    dst_addr = trailer & 0x1FF
                    count = (trailer >> 9) & 0x7F
                    self._buffer_xfer(src_pe, dst_pe, src_addr, dst_addr, count)
                    self.executed_ops["buffer_xfer"] += 1
                    self._record_cycle(pe_id, cost=2 + int(count))
                elif subop == DTYPE_SUBOP_BARRIER:
                    barrier_id = self._addr(instruction)
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
                        raise ValueError(f"PE_SELECT targets pe_id={selected} but num_pe={self.num_pe}")
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
        return ISASimulationResult(
            halted=halted,
            pc=pc,
            fetch_bytes=list(self.fetch_bytes),
            buffer_words=nonzero_buffer,
            instruction_count=len(words),
            executed_ops=dict(self.executed_ops),
            cycle_count_sequential=int(self._cycle_count_sequential),
            cycle_count_parallel_estimate=int(parallel_estimate if self.num_pe > 1 else self._cycle_count_sequential),
            num_pe=int(self.num_pe),
        )


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
) -> ISASimulationResult:
    resolved_num_pe = int(num_pe) if num_pe is not None else (2 if program_uses_multi_pe(words) else 1)
    return UTPUISASimulator(array_size=array_size, buffer_size=buffer_size, num_pe=resolved_num_pe).run_words(words)


def simulate_program_bytes(
    program: bytes,
    array_size: int = 16,
    buffer_size: int = 512,
    num_pe: Optional[int] = None,
) -> ISASimulationResult:
    return simulate_words(words_from_bytes(program), array_size=array_size, buffer_size=buffer_size, num_pe=num_pe)


def simulate_mem_file(path: str, array_size: int = 16, buffer_size: int = 512, num_pe: Optional[int] = None) -> ISASimulationResult:
    return simulate_words(words_from_mem(path), array_size=array_size, buffer_size=buffer_size, num_pe=num_pe)
