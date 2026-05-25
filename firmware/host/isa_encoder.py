from typing import List, Tuple
import struct


OPCODE_STORE = 0b000 #0 - store data to buffer
OPCODE_FETCH = 0b001 #1 - fetch data from buffer
OPCODE_RUN = 0b010 #2 - execute
OPCODE_LOAD = 0b011 #3 - load data into PE array
OPCODE_HALT = 0b100 #4 - stop execution
OPCODE_NOP = 0b101 #7 - no operation
OPCODE_BSTORE = 0b110 #6 - burst store sequential 16-bit words to buffer
OPCODE_DTYPE = 0b111 #7 - multi-PE / dataflow extensions (sim-validated; not in current RTL)

DTYPE_SUBOP_BUFFER_XFER = 0b00
DTYPE_SUBOP_BARRIER = 0b01
DTYPE_SUBOP_ACC_ADD = 0b10
DTYPE_SUBOP_PE_SELECT = 0b11

# Legacy no-op: PE routing uses D-type PE_SELECT, not address bits.
PE_ID_SHIFT = 12

INSTRUCTION_WIDTH = 16
ADDRESS_WIDTH = 9


def tag_pe(instruction: int, pe_id: int) -> int:
  """Tag an encoded 16-bit instruction word with target PE id (0 or 1)."""
  if pe_id not in (0, 1):
    raise ValueError(f"pe_id must be 0 or 1, got {pe_id}")
  return int(instruction) & ~(1 << PE_ID_SHIFT) | ((int(pe_id) & 1) << PE_ID_SHIFT)


def pe_id_from_word(word: int) -> int:
  return (int(word) >> PE_ID_SHIFT) & 1

#convert list of [-8,7] ints into 16 bit value
def int4To16(values: List[int]) -> int:
    while len(values) < 4:
        values = values + [0]

    result = 0
    for i, val in enumerate(values):
        nibble = int(val) & 0xF
        result |= (nibble << (i*4))
    return result

#check if valid address
def encodeAddress(addr: int) -> int:
    if not 0 <= addr <= 511: 
        raise ValueError(f"Address {addr} out of range (0-511)")
    return addr

#convert 16-bit instructino to bytes
def instructionToBytes(instruction: int) -> bytes:
    return struct.pack("<H", instruction & 0xFFFF)


#encode STORE intruction with values
def encodeStoreValues(addr: int, values: List[int]) -> bytes:
    addr = encodeAddress(addr)

    word1 = OPCODE_STORE #bits 0-2
    word1 |= (1 <<4) #bit 4: immediate mode
    # word1 |= (addr << 7) # REMOVED: address is now in word 3
    
    word2 = int4To16(values)
    
    word3 = addr # Destination address

    return instructionToBytes(word1) + instructionToBytes(word2) + instructionToBytes(word3)

#encode STORE instruction that copies from one address to another
def encodeStoreAddress(destAddr: int, srcAddr: int) -> bytes:
    destAddr = encodeAddress(destAddr)
    srcAddr = encodeAddress(srcAddr)

    word1 = OPCODE_STORE #bits 0-2
    word1 |= (0 << 4) #bit 4: address mode (not immediate)
    # word1 |= (destAddr << 7) # REMOVED: destination address is now in word 3

    word2 = srcAddr
    
    word3 = destAddr

    return instructionToBytes(word1) + instructionToBytes(word2) + instructionToBytes(word3)

#encode LOAD instruction
def encodeLoad(addr: int, is_weights:bool) -> bytes:
    addr = encodeAddress(addr)
    instruction = OPCODE_LOAD #bits 0-2
    instruction |= (1 if is_weights else 0) << 3 #bit 3
    instruction |= (addr << 7) #bits 7-15
    return instructionToBytes(instruction)

#load weights from address
def encodeLoadWeights(addr: int) -> bytes:
    return encodeLoad(addr, is_weights=True)

#load inputs from address
def encodeLoadInputs(addr: int) -> bytes:
    return encodeLoad(addr, is_weights=False)

def encodeRun(
    result_addr: int,
    compute_en: bool = True,
    quantize_en: bool = True,
    relu_en: bool = True,
    acc_clear_en: bool = False
) -> bytes:

    """
    bits 0-2: OPCODO
    bit 3: compute_en
    bit 4: quantize_en 
    bit 5: relu_en
    bit 6: acc_clear_en (used in accumulate mode)
    bits 7-15: result address
    """
    result_addr = encodeAddress(result_addr)
    instruction = OPCODE_RUN #bits 0-2
    instruction |= (1 if compute_en else 0) << 3 #bit 3
    instruction |= (1 if quantize_en else 0) << 4 #bit 4
    instruction |= (1 if relu_en else 0) << 5 #bit 5
    instruction |= (1 if acc_clear_en else 0) << 6 #bit 6
    instruction |= (result_addr << 7) #bits 7-15
    return instructionToBytes(instruction)

#encode FETCH instruction
def encodeFetch(addr: int, top_half: bool = True) -> bytes:

    """
    bits 0-2: OPCODE
    bit 3: top/bottom selector
    bits 4-6: not used
    bits 7-15: address
    """
    addr = encodeAddress(addr)
    instruction = OPCODE_FETCH #bits 0-2
    instruction |= (1 if top_half else 0) << 3 #bit 3
    instruction |= (addr << 7) #bits 7-15
    return instructionToBytes(instruction)

#encode HALT instruction
def encodeHalt() -> bytes: 
    """
    bits 0-2: OPCODE
    bits 3-15: not used
    """
    instruction = OPCODE_HALT 
    return instructionToBytes(instruction)

#encode a NOP instruction
def encodeNop() -> bytes:
    instruction = OPCODE_NOP
    return instructionToBytes(instruction)

def encodePeSelect(pe_id: int) -> bytes:
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
) -> bytes:
  """Copy count buffer words from src_pe[src_addr:] to dst_pe[dst_addr:]."""
  src_addr = encodeAddress(src_addr)
  dst_addr = encodeAddress(dst_addr)
  if count <= 0 or count > 0x7F:
    raise ValueError(f"buffer xfer count must be in 1..127, got {count}")
  header = encodeDType(DTYPE_SUBOP_BUFFER_XFER, pe_id=0)
  header |= (int(dst_pe) & 1) << 3
  header |= (int(src_pe) & 1) << 4
  header |= (src_addr << 7)
  trailer = (dst_addr & 0x1FF) | ((int(count) & 0x7F) << 9)
  return instructionToBytes(header) + instructionToBytes(trailer)


def encodeBarrier(barrier_id: int = 0) -> bytes:
  barrier_id = encodeAddress(barrier_id)
  header = encodeDType(DTYPE_SUBOP_BARRIER)
  header |= (barrier_id << 7)
  return instructionToBytes(header)


def encodeAccAdd(src_pe: int, dst_pe: int = 0) -> bytes:
  """Add src_pe accumulator lanes into dst_pe accumulator (sim merge primitive)."""
  if src_pe == dst_pe:
    raise ValueError("ACC_ADD requires distinct src_pe and dst_pe")
  header = encodeDType(DTYPE_SUBOP_ACC_ADD)
  header |= (int(dst_pe) & 1) << 3
  header |= (int(src_pe) & 1) << 4
  return instructionToBytes(header)


def encodeBurstStore(addr: int, words: List[int]) -> bytes:
    addr = encodeAddress(addr)
    if len(words) == 0:
        raise ValueError("BURST_STORE requires at least one word")
    if len(words) > 0xFFFF:
        raise ValueError("BURST_STORE count exceeds 16-bit field")
    header = OPCODE_BSTORE | (addr << 7)
    payload = [instructionToBytes(header), instructionToBytes(len(words))]
    for w in words:
        payload.append(instructionToBytes(int(w) & 0xFFFF))
    return b"".join(payload)

#encoder class that tracks instructions
class ISAEncoder:
    def __init__(self):
        self.instructions = []

    #add STORE instruction
    def store(self, addr: int, values: List[int]) -> "ISAEncoder":
        self.instructions.append(encodeStoreValues(addr, values))
        return self
    
    #add LOADWEI instruction
    def loadWeights(self, addr: int) -> "ISAEncoder":
        self.instructions.append(encodeLoadWeights(addr))
        return self
    
    #add LOADIN instruction
    def loadInputs(self, addr: int) -> "ISAEncoder":
        self.instructions.append(encodeLoadInputs(addr))
        return self

    
    #add RUN instruction
    def run(
        self,
        result_addr: int,
        compute: bool = True,
        quantize: bool = True,
        relu: bool = True,
        acc_clear: bool = False
    ) -> "ISAEncoder":
        self.instructions.append(encodeRun(result_addr, compute, quantize, relu, acc_clear))
        return self
    
    #add FETCH instruction
    def fetch(self, addr: int, top_half: bool=True) -> "ISAEncoder":
        self.instructions.append(encodeFetch(addr, top_half))
        return self
    
    #add HALT instruction
    def halt(self) -> "ISAEncoder":
        self.instructions.append(encodeHalt())
        return self
    
    #add NOP instruction
    def nop(self) -> "ISAEncoder":
        self.instructions.append(encodeNop())
        return self

    # add BURST_STORE instruction sequence
    def burst_store(self, addr: int, words: List[int]) -> "ISAEncoder":
        self.instructions.append(encodeBurstStore(addr, words))
        return self

    def buffer_xfer(self, src_addr: int, dst_addr: int, count: int, src_pe: int = 0, dst_pe: int = 1) -> "ISAEncoder":
        self.instructions.append(encodeBufferXfer(src_addr, dst_addr, count, src_pe=src_pe, dst_pe=dst_pe))
        return self

    def barrier(self, barrier_id: int = 0) -> "ISAEncoder":
        self.instructions.append(encodeBarrier(barrier_id))
        return self

    def acc_add(self, src_pe: int, dst_pe: int = 0) -> "ISAEncoder":
        self.instructions.append(encodeAccAdd(src_pe, dst_pe=dst_pe))
        return self

    def pe_select(self, pe_id: int) -> "ISAEncoder":
        self.instructions.append(encodePeSelect(pe_id))
        return self
    
    #get program as bytes
    def getProgram(self) -> bytes:
        return b''.join(self.instructions)
    
    #get num of instructions in program
    def getInstructionCount(self) -> int:
        return len(self.instructions)
    
    #clear instructions
    def clear(self) -> None:
        self.instructions = []

if __name__ == "__main__":
    print("ISA Encoder Test")
    print("=" * 50)
    print("\nIndividual instruction tests:")
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
    print("Encoder class test:")
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
