import numpy as np
import time
from typing import List, Optional
from uart_driver import UARTDriver
from isa_encoder import (
    ISAEncoder,
    encodeStoreValues,
    encodeLoadWeights,
    encodeLoadInputs,
    encodeRun,
    encodeFetch,
    encodeHalt,
    int4To16
)


# loads programs and data onto tpu
class ProgramLoader:
    BUFFER_SECTION_A = 0x000  # 0x000-0x07F: Section A
    BUFFER_SECTION_B = 0x080  # 0x080-0x0FF: Section B
    BUFFER_SECTION_C = 0x100  # 0x100-0x17F: Section C
    BUFFER_SECTION_D = 0x180  # 0x180-0x1FF: Section D

    def __init__(self, uart, verbose):
        self.uart = uart
        self.verbose = verbose
        self.encoder = ISAEncoder()

    def _log(self, message):
        if self.verbose:
            print(f"[ProgramLoader] {message}")

    # send bytes to chip
    def sendBytes(self, data):
        self.uart.sendBytes(data)
        self._log(f"Sent {len(data)} bytes")

    # send single encoded instruction
    def sendInstructions(self, instruction_bytes):
        self.uart.sendBytes(instruction_bytes)
        time.sleep(0.001)

    # send program to chip
    def sendProgram(self, program):
        self._log(f"Sending program: {len(program)} bytes")
        chunk_size = 128

        for i in range(0, len(program), chunk_size):
            chunk = program[i:i + chunk_size]
            self.uart.sendBytes(chunk)
            time.sleep(0.015)

        self._log("Program sent successfully")

    # load array to unified buffer
    def loadInt4ArrayToBuffer(self, base_addr, data):
        flatData = list(data.flatten())

        self._log(f"Loading {len(flatData)} int4 values to address 0x{base_addr:03X}")

        addr = base_addr
        for i in range(0, len(flatData), 4):
            chunk = flatData[i:i + 4]
            while len(chunk) < 4:
                chunk.append(0)

            storeBytes = encodeStoreValues(addr, chunk)
            self.sendInstructions(storeBytes)

            addr += 1  # each store fills one 16-bit word

        self._log(f"Loaded to address 0x{base_addr:03X} - 0x{addr - 1:03X}")

    # load weight matrix into buffer
    def loadWeightsToBuffer(self, base_addr, weights):
        self._log(f"Loading weights {weights.shape} to 0x{base_addr:03X}")
        self.loadInt4ArrayToBuffer(base_addr, weights)

    # load input activations into buffer
    def loadInputToBuffer(self, base_addr, inputs):
        self._log(f"Loading inputs {inputs.shape} to 0x{base_addr:03X}")
        self.loadInt4ArrayToBuffer(base_addr, inputs)

    def readResults(self, base_addr, count):
        self._log(f"Reading {count} values from 0x{base_addr:03X}")
        program = b''
        numWords = (count + 3) // 4

        for i in range(numWords):
            addr = base_addr + i
            program += encodeFetch(addr, top_half=False)
            program += encodeFetch(addr, top_half=True)

        self.sendProgram(program)
        time.sleep(0.1)

        numBytes = numWords * 2
        received = self.uart.receiveBytes(numBytes)
        self._log(f"Received {len(received)} bytes")

        results = []
        for byte in received:
            lowNibble = byte & 0x0F
            highNibble = (byte >> 4) & 0x0F

            if lowNibble >= 8:
                lowNibble -= 16
            if highNibble >= 8:
                highNibble -= 16

            results.append(lowNibble)
            results.append(highNibble)

        return results[:count]

    # execute 2x2 matrix multiply on the chip
    def execute2x2MatMul(self, weights, inputs, weight_addr, input_addr, result_addr):
        self._log("Executing 2x2 matmul")
        self.encoder.clear()

        self.encoder.store(weight_addr, weights)
        self.encoder.loadWeights(weight_addr)

        inputPadded = inputs + [0] * (4 - len(inputs))
        self.encoder.store(input_addr, inputPadded)
        self.encoder.loadInputs(input_addr)

        self.encoder.run(result_addr, compute=True, quantize=True, relu=True)
        self.encoder.fetch(result_addr, top_half=False)
        self.encoder.fetch(result_addr, top_half=True)
        self.encoder.halt()

        program = self.encoder.getProgram()
        self.sendProgram(program)

        time.sleep(0.05)
        received = self.uart.receiveBytes(2)

        results = []
        for byte in received:
            low = byte & 0x0F
            if low >= 8:
                low -= 16
            results.append(low)

        return results[:2]

    # sends reset sequence to chip
    def resetChip(self):
        self._log("Resetting chip...")
        self.uart.flushInput()
        for _ in range(5):
            self.sendInstructions(encodeHalt())
        time.sleep(0.1)
        self.uart.flushInput()
        self._log("Chip reset complete")


if __name__ == "__main__":
    import sys

    port = sys.argv[1] if len(sys.argv) > 1 else "COM3"
    print("ProgramLoader Test")
    print("=" * 50)

    try:
        uart = UARTDriver(port, baud=115200)
        loader = ProgramLoader(uart, verbose=True)

        loader.resetChip()

        print("\nTesting 2x2 matmul...")
        weights = [1, 2, 3, 4]
        inputs = [1, 1]

        results = loader.execute2x2MatMul(
            weights,
            inputs,
            ProgramLoader.BUFFER_SECTION_B,
            ProgramLoader.BUFFER_SECTION_A,
            ProgramLoader.BUFFER_SECTION_C
        )

        print(f"Weights: {weights}")
        print(f"Inputs: {inputs}")
        print(f"Results: {results}")

        uart.close()

    except Exception as e:
        print(f"Error: {e}")
