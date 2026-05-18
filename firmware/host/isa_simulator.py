from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


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


class UTPUISASimulator:
    """Bit-level Python simulator for the RTL-observable uTPU ISA behavior.

    The simulator follows the autonomous RTL path in rtl/top/top.sv for the
    compressed blocked-FC programs: STORE/BSTORE update unified-buffer words,
    LOAD snapshots a 16x16 weight tile or 16-lane input vector, RUN implements
    accumulate/finalize modes, and FETCH emits low/high bytes.
    """

    def __init__(self, array_size: int = 16, buffer_size: int = 512, alpha_shift: int = 2):
        if array_size <= 0 or array_size % 4 != 0:
            raise ValueError("array_size must be a positive multiple of 4")
        self.array_size = int(array_size)
        self.buffer_size = int(buffer_size)
        self.alpha_shift = int(alpha_shift)
        self.reset()

    def reset(self) -> None:
        self.buffer = [0] * self.buffer_size
        self.weights = [[0 for _ in range(self.array_size)] for _ in range(self.array_size)]
        self.inputs = [0 for _ in range(self.array_size)]
        self.acc_partial_sums = [0 for _ in range(self.array_size)]
        self.fetch_bytes: List[int] = []
        self.executed_ops = {
            "store": 0,
            "fetch": 0,
            "run": 0,
            "load": 0,
            "halt": 0,
            "nop": 0,
            "bstore": 0,
        }

    def _addr(self, word: int) -> int:
        return (word >> 7) & 0x1FF

    def _check_addr(self, addr: int) -> None:
        if not 0 <= addr < self.buffer_size:
            raise ValueError(f"buffer address out of range: {addr}")

    def _read_word(self, addr: int) -> int:
        self._check_addr(addr)
        return self.buffer[addr] & 0xFFFF

    def _write_word(self, addr: int, value: int) -> None:
        self._check_addr(addr)
        self.buffer[addr] = int(value) & 0xFFFF

    def _load_weights(self, addr: int) -> None:
        words_needed = (self.array_size * self.array_size) // 4
        flat: List[int] = []
        for offset in range(words_needed):
            flat.extend(_unpack_int4_word(self._read_word(addr + offset)))
        for row in range(self.array_size):
            start = row * self.array_size
            self.weights[row] = flat[start:start + self.array_size]

    def _load_inputs(self, addr: int) -> None:
        words_needed = self.array_size // 4
        flat: List[int] = []
        for offset in range(words_needed):
            flat.extend(_unpack_int4_word(self._read_word(addr + offset)))
        self.inputs = flat[:self.array_size]

    def _run_accumulate(self, acc_clear: bool) -> None:
        for row in range(self.array_size):
            lane_val = 0
            for k in range(self.array_size):
                lane_val += self.weights[row][k] * self.inputs[k]
            if acc_clear:
                self.acc_partial_sums[row] = lane_val
            else:
                self.acc_partial_sums[row] += lane_val

    def _run_finalize(self, result_addr: int, quantize: bool, relu: bool) -> None:
        outputs = [0 for _ in range(self.array_size)]
        for i, acc in enumerate(self.acc_partial_sums):
            q = _clip_int4(acc) if quantize else int(acc)
            if relu and q < 0:
                q = q >> self.alpha_shift
            outputs[i] = q

        # RTL compute writeback writes ARRAY_SIZE*ARRAY_SIZE lanes as packed
        # int4 words. Only the first ARRAY_SIZE lanes carry finalized outputs.
        padded = outputs + [0] * (self.array_size * self.array_size - self.array_size)
        for word_idx in range((self.array_size * self.array_size) // 4):
            chunk = padded[word_idx * 4:(word_idx + 1) * 4]
            self._write_word(result_addr + word_idx, _pack_int4(chunk))

    def run_words(self, words: List[int], max_steps: Optional[int] = None) -> ISASimulationResult:
        self.reset()
        pc = 0
        steps = 0
        halted = False
        max_steps = max_steps or (len(words) * 8 + 1024)

        while pc < len(words) and steps < max_steps:
            steps += 1
            instruction = words[pc] & 0xFFFF
            pc += 1
            opcode = instruction & 0x7

            if opcode == OPCODE_STORE:
                if pc + 1 >= len(words):
                    raise ValueError("truncated STORE instruction")
                source = words[pc] & 0xFFFF
                dest = words[pc + 1] & 0x1FF
                pc += 2
                immediate_mode = bool((instruction >> 4) & 0x1)
                self._write_word(dest, source if immediate_mode else self._read_word(source & 0x1FF))
                self.executed_ops["store"] += 1

            elif opcode == OPCODE_BSTORE:
                if pc >= len(words):
                    raise ValueError("truncated BSTORE count")
                base = self._addr(instruction)
                count = words[pc] & 0xFFFF
                pc += 1
                if pc + count > len(words):
                    raise ValueError("truncated BSTORE payload")
                for i in range(count):
                    self._write_word(base + i, words[pc + i])
                pc += count
                self.executed_ops["bstore"] += 1

            elif opcode == OPCODE_LOAD:
                addr = self._addr(instruction)
                if (instruction >> 3) & 0x1:
                    self._load_weights(addr)
                else:
                    self._load_inputs(addr)
                self.executed_ops["load"] += 1

            elif opcode == OPCODE_RUN:
                result_addr = self._addr(instruction)
                compute = bool((instruction >> 3) & 0x1)
                quantize = bool((instruction >> 4) & 0x1)
                relu = bool((instruction >> 5) & 0x1)
                acc_clear = bool((instruction >> 6) & 0x1)
                if compute and not quantize and not relu:
                    self._run_accumulate(acc_clear=acc_clear)
                elif (not compute) and quantize:
                    self._run_finalize(result_addr, quantize=quantize, relu=relu)
                elif compute and quantize:
                    self._run_accumulate(acc_clear=True)
                    self._run_finalize(result_addr, quantize=quantize, relu=relu)
                else:
                    raise ValueError(
                        "unsupported RUN mode: "
                        f"compute={compute} quantize={quantize} relu={relu}"
                    )
                self.executed_ops["run"] += 1

            elif opcode == OPCODE_FETCH:
                addr = self._addr(instruction)
                word = self._read_word(addr)
                high_byte = bool((instruction >> 3) & 0x1)
                self.fetch_bytes.append((word >> 8) & 0xFF if high_byte else word & 0xFF)
                self.executed_ops["fetch"] += 1

            elif opcode == OPCODE_HALT:
                self.executed_ops["halt"] += 1
                halted = True
                break

            elif opcode == OPCODE_NOP:
                self.executed_ops["nop"] += 1

            else:
                raise ValueError(f"unsupported opcode {opcode} at pc={pc - 1}")

        if steps >= max_steps and not halted:
            raise TimeoutError(f"ISA simulation exceeded max_steps={max_steps}")

        nonzero_buffer = {i: w for i, w in enumerate(self.buffer) if w != 0}
        return ISASimulationResult(
            halted=halted,
            pc=pc,
            fetch_bytes=list(self.fetch_bytes),
            buffer_words=nonzero_buffer,
            instruction_count=len(words),
            executed_ops=dict(self.executed_ops),
        )


def words_from_bytes(program: bytes) -> List[int]:
    if len(program) % 2 != 0:
        raise ValueError("program bytes must contain complete 16-bit words")
    return [
        int.from_bytes(program[i:i + 2], byteorder="little", signed=False)
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


def simulate_words(words: List[int], array_size: int = 16, buffer_size: int = 512) -> ISASimulationResult:
    return UTPUISASimulator(array_size=array_size, buffer_size=buffer_size).run_words(words)


def simulate_program_bytes(program: bytes, array_size: int = 16, buffer_size: int = 512) -> ISASimulationResult:
    return simulate_words(words_from_bytes(program), array_size=array_size, buffer_size=buffer_size)


def simulate_mem_file(path: str, array_size: int = 16, buffer_size: int = 512) -> ISASimulationResult:
    return simulate_words(words_from_mem(path), array_size=array_size, buffer_size=buffer_size)
