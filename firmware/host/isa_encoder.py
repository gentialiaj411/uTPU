from typing import List, Tuple
import struct


OPCODE_STORE = 0b000 #0 - store data to buffer
OPCODE_FETCH = 0b001 #1 - fetch data from buffer
OPCODE_RUN = 0b010 #2 - execute
OPCODE_LOAD = 0b011 #3 - load data into PE array
OPCODE_HALT = 0b100 #4 - stop execution
OPCODE_NOP = 0b111 #7 - no operation

INSTRUCTION_WIDTH = 16
ADDRESS_WIDTH = 9

#convert list of [-8,7] ints into 16 bit value
def int4To16(values: List[int]) -> int:
    while len(values) < 4:
        values = values + [0]

    result = 0
    for i, val in enumerate(values):
        nibble = val & 0xF
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

def encodeRun(result_addr: int, compute_en: bool = True,  quantize_en: bool = True, relu_en: bool = True) -> bytes:

    """
    bits 0-2: OPCODO
    bit 3: compute_en
    bit 4: quantize_en 
    bit 5: relu_en
    bit 6: not used
    bits 7-15: result address
    """
    result_addr = encodeAddress(result_addr)
    instruction = OPCODE_RUN #bits 0-2
    instruction |= (1 if compute_en else 0) << 3 #bit 3
    instruction |= (1 if quantize_en else 0) << 4 #bit 4
    instruction |= (1 if relu_en else 0) << 5 #bit 5
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
    def run(self, result_addr: int, compute: bool=True, quantize: bool=True, relu: bool=True) -> "ISAEncoder":
        self.instructions.append(encodeRun(result_addr, compute, quantize, relu))
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
