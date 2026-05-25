import struct
import numpy as np
import time
import os
import re
import math
from typing import List, Optional, Dict, Any
try:
    from uart_driver import UARTDriver
except ImportError:
    UARTDriver = None
from isa_encoder import (
    ISAEncoder,
    IsaConfig,
    DEFAULT_CFG,
    encodeStoreValues,
    encodeLoadWeights,
    encodeLoadInputs,
    encodeRun,
    encodeFetch,
    encodeHalt,
    int4To16,
    pack_values_to_word,
)
from lowering_types import BlockedFCLoweringRequest
from backend_lowering import create_backend_lowerer
from cuda_blocked_fc_backend import CUDABlockedFCExecutor

# Upload protocol magic bytes (must match rtl/top/top.sv)
MAGIC_UPLOAD = 0xA1   # begin program upload
MAGIC_START  = 0xA2   # start execution
MAGIC_REARM  = 0xA3   # re-arm from HALT (optional, requires hardware reset otherwise)
MAGIC_READ_PERF = 0xA4  # read 3x64-bit perf counters (cycle, busy, program_count)


class ProgramLoader:
    BUFFER_SECTION_A = 0x000  # 0x000-0x07F
    BUFFER_SECTION_B = 0x080  # 0x080-0x0FF
    BUFFER_SECTION_C = 0x100  # 0x100-0x17F
    BUFFER_SECTION_D = 0x180  # 0x180-0x1FF

    def __init__(self, uart, verbose, backend: str = "utpu", cfg: Optional[IsaConfig] = None):
        # ``cfg`` lets callers opt into widened (Phase 4) ISA encodings — INT8
        # datapath, extended 2-word address layout, etc. The default keeps the
        # legacy 9-bit-address / INT4 byte-for-byte representation so all
        # pre-Phase-4 host code paths and the hardware FPGA build remain
        # unchanged when ``cfg`` is omitted.
        self.cfg = cfg or DEFAULT_CFG
        self.uart = uart
        self.verbose = verbose
        self.encoder = ISAEncoder(cfg=self.cfg)
        self.default_array_size = self._resolve_array_size()
        self.backend_name = backend
        self.backend_lowerer = create_backend_lowerer(backend)
        self.cuda_executor = CUDABlockedFCExecutor(verbose=verbose) if backend.strip().lower() == "cuda" else None

    def _log(self, message):
        if self.verbose:
            print(f"[ProgramLoader] {message}")

    def _resolve_array_size(self) -> int:
        host_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(host_dir, "..", ".."))
        top_sv_path = os.path.join(repo_root, "rtl", "top", "top.sv")
        try:
            with open(top_sv_path, "r", encoding="utf-8") as f:
                txt = f.read()
            m = re.search(r"parameter\s+ARRAY_SIZE\s*=\s*(\d+)", txt)
            if m:
                return int(m.group(1))
        except OSError:
            pass
        return 16

    def sendBytes(self, data):
        self.uart.send_bytes_to_chip(data)
        self._log(f"Sent {len(data)} bytes")

    def _send_chunked(self, data: bytes, chunk_size: int = 128):
        """Send bytes in chunks with inter-chunk pacing to avoid RX FIFO overrun."""
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            self.uart.send_bytes_to_chip(chunk)
            time.sleep(0.015)

    # ------------------------------------------------------------------
    # Autonomous upload protocol
    # ------------------------------------------------------------------

    def uploadProgram(self, program: bytes):
        """
        Upload a complete program into the on-chip instruction BRAM.

        Protocol (bytes sent over UART):
          1. MAGIC_UPLOAD (0xA1)
          2. program length in 16-bit instruction words, little-endian (2 bytes)
          3. program bytes (n_words * 2 bytes), little-endian per word

        After this call the chip sits in WAIT_START_STATE.
        Call start() to begin execution.
        """
        if len(program) % 2 != 0:
            raise ValueError("Program length must be even (16-bit instructions)")
        n_words = len(program) // 2
        if n_words > 1024:
            raise ValueError(f"Program too large: {n_words} words (max 1024)")

        self._log(f"Uploading {n_words} instructions ({len(program)} bytes)")

        # 0xA3 (re-arm) is sent first so this call works whether the chip is
        # sitting in HALT_STATE or already in UPLOAD_HEADER_STATE.
        # In UPLOAD_HEADER_STATE the chip ignores every byte except 0xA1, so
        # the extra 0xA3 is harmlessly discarded.
        header = bytes([MAGIC_REARM, MAGIC_UPLOAD]) + struct.pack('<H', n_words)
        self.uart.send_bytes_to_chip(header)
        time.sleep(0.005)

        self._send_chunked(program)
        self._log("Upload complete, waiting for start")

    def start(self):
        """Send start signal; chip begins executing from PC=0."""
        self.uart.send_bytes_to_chip(bytes([MAGIC_START]))
        self._log("Start signal sent")

    def rearm(self):
        """Re-arm chip from HALT state so a new uploadProgram() can run.
        Requires RTL support (HALT_STATE → UPLOAD_HEADER on 0xA3).
        Alternative: assert hardware reset line."""
        self.uart.send_bytes_to_chip(bytes([MAGIC_REARM]))
        self._log("Re-arm signal sent")

    def readPerfCounters(self, timeout: float = 0.5) -> Dict[str, int]:
        """Read cycle, busy, and program counters over UART."""
        self.uart.flush_input()
        self.uart.send_bytes_to_chip(bytes([MAGIC_READ_PERF]))
        payload = self.uart.receive_exact(24, timeout=timeout)
        if len(payload) != 24:
            raise RuntimeError(f"Expected 24 perf bytes, received {len(payload)}")
        return {
            "cycle_counter": int.from_bytes(payload[0:8], byteorder="big", signed=False),
            "busy_counter": int.from_bytes(payload[8:16], byteorder="big", signed=False),
            "program_count": int.from_bytes(payload[16:24], byteorder="big", signed=False),
        }

    # ------------------------------------------------------------------
    # Array helpers (unchanged)
    # ------------------------------------------------------------------

    def loadInt4ArrayToBuffer(self, base_addr, data):
        """Build STORE instructions that load int4 array into unified buffer."""
        flatData = list(data.flatten())
        self._log(f"Loading {len(flatData)} int4 values to 0x{base_addr:03X}")
        addr = base_addr
        for i in range(0, len(flatData), 4):
            chunk = flatData[i:i + 4]
            while len(chunk) < 4:
                chunk.append(0)
            self.encoder.store(addr, chunk)
            addr += 1
        self._log(f"Encoded stores for 0x{base_addr:03X}–0x{addr - 1:03X}")

    def loadWeightsToBuffer(self, base_addr, weights):
        self._log(f"Encoding weights {weights.shape} to 0x{base_addr:03X}")
        self.loadInt4ArrayToBuffer(base_addr, weights)

    def loadInputToBuffer(self, base_addr, inputs):
        self._log(f"Encoding inputs to 0x{base_addr:03X}")
        self.loadInt4ArrayToBuffer(base_addr, np.array(inputs))

    # ------------------------------------------------------------------
    # 2×2 matrix multiply — autonomous single-program version
    # ------------------------------------------------------------------

    def execute2x2MatMul(self, weights, inputs, weight_addr, input_addr, result_addr,
                         quantize: bool = True, relu: bool = True,
                         timeout: float = 0.5) -> List[int]:
        """
        Upload and autonomously execute a 2×2 matmul.

        The complete program (store weights, load weights, store inputs,
        load inputs, run, fetch lo, fetch hi, halt) is uploaded once,
        then the chip executes without further host involvement.

        Returns two int4 result values.
        """
        self._log("Building 2x2 matmul program")
        self.encoder.clear()

        # store weights
        self.encoder.store(weight_addr, weights)
        self.encoder.loadWeights(weight_addr)

        # store inputs (pad to 4 slots)
        input_padded = list(inputs) + [0] * (4 - len(inputs))
        self.encoder.store(input_addr, input_padded)
        self.encoder.loadInputs(input_addr)

        # compute
        self.encoder.run(result_addr, compute=True, quantize=quantize, relu=relu)

        # fetch both halves of the result word so the host can read them back
        self.encoder.fetch(result_addr, top_half=False)
        self.encoder.fetch(result_addr, top_half=True)

        # halt — chip stops here; host reads the 2 bytes the fetches pushed to TX FIFO
        self.encoder.halt()

        program = self.encoder.getProgram()

        self.uart.flush_input()
        self.uploadProgram(program)
        self.start()

        # wait for 2 result bytes (one per fetch)
        deadline = time.time() + timeout
        while self.uart.bytes_waiting() < 2 and time.time() < deadline:
            time.sleep(0.002)

        remaining = max(0.05, deadline - time.time())
        received = self.uart.receive_exact(2, timeout=remaining)

        results = []
        for byte in received:
            low = byte & 0x0F
            if low >= 8:
                low -= 16
            results.append(low)

        return results[:2]

    def estimate_fc_layer_instruction_words(
        self,
        out_features: int,
        in_features: int,
        array_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Estimate instruction words for legacy per-tile RPC and ARRAY_SIZE block programs.
        This is an ISA-level estimate using current op encoding:
          - STORE immediate = 3 words (header + data + addr)
          - LOAD / RUN / FETCH / HALT = 1 word each
        """
        a = array_size or self.default_array_size
        if a <= 0:
            raise ValueError(f"Invalid array_size={a}")

        # Legacy 2x2 mode: each tile RPC builds one 12-word program.
        legacy_tile_runs = ((out_features + (out_features % 2)) // 2) * ((in_features + (in_features % 2)) // 2)
        legacy_words = legacy_tile_runs * 12

        out_blocks = math.ceil(out_features / a)
        in_blocks = math.ceil(in_features / a)
        block_ops = out_blocks * in_blocks

        # Current ISA schedule per ARRAY_SIZE block op:
        # weight stores + LOADWEI + input stores + LOADIN + RUN + fetch all output words
        # We must fetch 4 packed words for 16 outputs -> 8 FETCH ops (lo/hi per word).
        weight_store_ops = (a * a) // 4
        input_store_ops = a // 4
        output_words = a // 4
        fetch_ops = output_words * 2

        words_per_acc_block_op = (
            (weight_store_ops * 3)
            + 1
            + (input_store_ops * 3)
            + 1
            + 1
        )
        words_per_finalize_block = 1 + fetch_ops
        block_program_words = (
            block_ops * words_per_acc_block_op
            + out_blocks * words_per_finalize_block
            + 1  # HALT
        )

        return {
            "array_size": int(a),
            "out_features": int(out_features),
            "in_features": int(in_features),
            "legacy_2x2_tile_runs": int(legacy_tile_runs),
            "legacy_2x2_instruction_words": int(legacy_words),
            "out_blocks": int(out_blocks),
            "in_blocks": int(in_blocks),
            "block_ops": int(block_ops),
            "words_per_acc_block_op": int(words_per_acc_block_op),
            "words_per_finalize_block": int(words_per_finalize_block),
            "block_program_instruction_words": int(block_program_words),
            "instruction_bram_words": 1024,
            "fits_instruction_bram": bool(block_program_words <= 1024),
        }

    def build_fc_layer_block_program(
        self,
        weights_int4,
        activations_int4,
        out_features: int,
        in_features: int,
        array_size: Optional[int] = None,
        apply_relu: bool = False,
        apply_quant: bool = True,
        weight_addr: int = BUFFER_SECTION_B,
        input_addr: int = BUFFER_SECTION_A,
        result_addr: int = BUFFER_SECTION_C,
    ) -> Dict[str, Any]:
        """
        Build a block-scheduled FC program using current ISA instructions.
        Returns program bytes plus metadata and executability status.
        """
        a = array_size or self.default_array_size
        return self.backend_lowerer.lower_blocked_fc(
            BlockedFCLoweringRequest(
                weights_int4=weights_int4,
                activations_int4=activations_int4,
                out_features=out_features,
                in_features=in_features,
                array_size=a,
                apply_relu=apply_relu,
                apply_quant=apply_quant,
                weight_addr=weight_addr,
                input_addr=input_addr,
                result_addr=result_addr,
            )
        )

    def _encode_accumulate_op(self, weight_block, input_block, out_base_addr: int, acc_clear: bool) -> bytes:
        e = ISAEncoder()
        self._encode_store_tensor_ops(e, self.BUFFER_SECTION_B, weight_block)
        e.loadWeights(self.BUFFER_SECTION_B)
        self._encode_store_tensor_ops(e, self.BUFFER_SECTION_A, input_block)
        e.loadInputs(self.BUFFER_SECTION_A)
        e.run(out_base_addr, compute=True, quantize=False, relu=False, acc_clear=acc_clear)
        return e.getProgram()

    def _pack_int4_words(self, data) -> List[int]:
        arr = np.asarray(data).flatten().astype(np.int8)
        words = []
        for i in range(0, len(arr), 4):
            chunk = arr[i:i+4].tolist()
            while len(chunk) < 4:
                chunk.append(0)
            words.append(int4To16(chunk))
        return words

    def _encode_accumulate_op_compressed(self, weight_block, input_block, out_base_addr: int, acc_clear: bool) -> bytes:
        e = ISAEncoder()
        w_words = self._pack_int4_words(weight_block)
        i_words = self._pack_int4_words(input_block)
        e.burst_store(self.BUFFER_SECTION_B, w_words)
        e.loadWeights(self.BUFFER_SECTION_B)
        e.burst_store(self.BUFFER_SECTION_A, i_words)
        e.loadInputs(self.BUFFER_SECTION_A)
        e.run(out_base_addr, compute=True, quantize=False, relu=False, acc_clear=acc_clear)
        return e.getProgram()

    def _encode_finalize_fetch_op(self, out_base_addr: int, array_size: int, apply_quant: bool, apply_relu: bool) -> bytes:
        e = ISAEncoder()
        e.run(out_base_addr, compute=False, quantize=apply_quant, relu=apply_relu, acc_clear=False)
        for widx in range(array_size // 4):
            addr = out_base_addr + widx
            e.fetch(addr, top_half=False)
            e.fetch(addr, top_half=True)
        return e.getProgram()

    def _encode_finalize_no_fetch_op(self, out_base_addr: int, apply_quant: bool, apply_relu: bool) -> bytes:
        e = ISAEncoder()
        e.run(out_base_addr, compute=False, quantize=apply_quant, relu=apply_relu, acc_clear=False)
        return e.getProgram()

    def _encode_store_tensor_ops(self, encoder: ISAEncoder, base_addr: int, data) -> None:
        flat = list(np.asarray(data).flatten())
        addr = base_addr
        for i in range(0, len(flat), 4):
            chunk = flat[i:i+4]
            while len(chunk) < 4:
                chunk.append(0)
            encoder.store(addr, chunk)
            addr += 1

    def _encode_accumulate_op_compressed_from_buffer(
        self,
        weight_block,
        input_base_addr: int,
        out_base_addr: int,
        acc_clear: bool
    ) -> bytes:
        e = ISAEncoder()
        w_words = self._pack_int4_words(weight_block)
        e.burst_store(self.BUFFER_SECTION_B, w_words)
        e.loadWeights(self.BUFFER_SECTION_B)
        e.loadInputs(input_base_addr)
        e.run(out_base_addr, compute=True, quantize=False, relu=False, acc_clear=acc_clear)
        return e.getProgram()

    def build_fc_layer_block_program_segmented(
        self,
        weights_int4,
        activations_int4,
        out_features: int,
        in_features: int,
        array_size: Optional[int] = None,
        apply_relu: bool = False,
        apply_quant: bool = True,
        max_words_per_segment: int = 1024,
        result_addr: int = BUFFER_SECTION_C,
    ) -> Dict[str, Any]:
        a = array_size or self.default_array_size
        if max_words_per_segment <= 1:
            raise ValueError("max_words_per_segment must be > 1")

        w = np.asarray(weights_int4, dtype=np.int8)
        x = np.asarray(activations_int4, dtype=np.int8).flatten()
        if w.shape != (out_features, in_features):
            raise ValueError(f"weights shape mismatch: expected {(out_features, in_features)}, got {w.shape}")
        if x.shape[0] != in_features:
            raise ValueError(f"activation length mismatch: expected {in_features}, got {x.shape[0]}")

        out_blocks = math.ceil(out_features / a)
        in_blocks = math.ceil(in_features / a)
        out_padded = out_blocks * a
        in_padded = in_blocks * a

        w_pad = np.zeros((out_padded, in_padded), dtype=np.int8)
        w_pad[:out_features, :in_features] = w
        x_pad = np.zeros(in_padded, dtype=np.int8)
        x_pad[:in_features] = x

        ops = []
        for ob in range(out_blocks):
            out_base_addr = result_addr + ob * (a // 4)
            o0 = ob * a
            o1 = o0 + a
            for ib in range(in_blocks):
                i0 = ib * a
                i1 = i0 + a
                op_bytes = self._encode_accumulate_op(
                    w_pad[o0:o1, i0:i1],
                    x_pad[i0:i1],
                    out_base_addr,
                    acc_clear=(ib == 0),
                )
                ops.append({
                    "kind": "accumulate",
                    "ob": ob,
                    "ib": ib,
                    "acc_clear": bool(ib == 0),
                    "bytes": op_bytes,
                    "words": len(op_bytes) // 2,
                })
            fin_bytes = self._encode_finalize_fetch_op(out_base_addr, a, apply_quant=apply_quant, apply_relu=apply_relu)
            ops.append({
                "kind": "finalize_fetch",
                "ob": ob,
                "bytes": fin_bytes,
                "words": len(fin_bytes) // 2,
            })

        segments = []
        cur_ops = []
        cur_words = 0
        for op in ops:
            op_words = op["words"]
            if op_words + 1 > max_words_per_segment:
                raise ValueError(f"Single op too large for segment budget: {op_words}+HALT > {max_words_per_segment}")
            if cur_words + op_words + 1 > max_words_per_segment:
                seg_enc = ISAEncoder()
                for sop in cur_ops:
                    seg_enc.instructions.append(sop["bytes"])
                seg_enc.halt()
                seg_program = seg_enc.getProgram()
                segments.append({
                    "program": seg_program,
                    "words": len(seg_program)//2,
                    "ops": cur_ops,
                })
                cur_ops = []
                cur_words = 0
            cur_ops.append(op)
            cur_words += op_words

        if cur_ops:
            seg_enc = ISAEncoder()
            for sop in cur_ops:
                seg_enc.instructions.append(sop["bytes"])
            seg_enc.halt()
            seg_program = seg_enc.getProgram()
            segments.append({
                "program": seg_program,
                "words": len(seg_program)//2,
                "ops": cur_ops,
            })

        words_per_segment = [s["words"] for s in segments]
        first_seg_acc_clear = any(op.get("acc_clear", False) for op in segments[0]["ops"]) if segments else False
        later_seg_acc_clear = any(
            op.get("acc_clear", False)
            for seg in segments[1:]
            for op in seg["ops"]
        )
        final_seg_finalize = any(op["kind"] == "finalize_fetch" for op in segments[-1]["ops"]) if segments else False
        earlier_finalize = any(
            op["kind"] == "finalize_fetch"
            for seg in segments[:-1]
            for op in seg["ops"]
        )

        output_words = a // 4
        output_fetch_bytes = output_words * 2
        return {
            "segments": [s["program"] for s in segments],
            "segment_words": words_per_segment,
            "segment_op_kinds": [[op["kind"] for op in s["ops"]] for s in segments],
            "segment_acc_clear_flags": [[bool(op.get("acc_clear", False)) for op in s["ops"]] for s in segments],
            "segment_count": len(segments),
            "max_segment_words": max_words_per_segment,
            "fits_each_segment": all(w <= max_words_per_segment for w in words_per_segment),
            "accumulator_persistence_required": len(segments) > 1,
            "first_segment_has_acc_clear": first_seg_acc_clear,
            "later_segments_have_acc_clear": later_seg_acc_clear,
            "final_segment_has_finalize_fetch": final_seg_finalize,
            "earlier_segments_have_finalize_fetch": earlier_finalize,
            "array_size": int(a),
            "out_blocks": int(out_blocks),
            "in_blocks": int(in_blocks),
            "block_ops": int(out_blocks * in_blocks),
            "output_fetch_bytes": int(output_fetch_bytes),
            "output_nibbles": int(a),
            "executable_on_current_fpga_path": bool(apply_quant),
            "int32_accumulation_supported": True,
            "quantize_after_accumulation_supported": bool(apply_quant),
            "blockers": [] if apply_quant else ["Final quantize/fetch path is required for current runtime output."],
        }

    def _decode_int4_bytes(self, received: bytes, count: int) -> List[int]:
        out = []
        for byte in received:
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            if lo >= 8:
                lo -= 16
            if hi >= 8:
                hi -= 16
            out.append(lo)
            out.append(hi)
        return out[:count]

    def execute_fc_layer_blocked(
        self,
        weights_int4,
        activations_int4,
        out_features: int,
        in_features: int,
        array_size: Optional[int] = None,
        apply_relu: bool = False,
        apply_quant: bool = True,
        allow_segmentation: bool = False,
        max_words_per_segment: int = 1024,
        timeout: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Build (and if legal, execute) an ARRAY_SIZE-block FC program.
        Legacy 2x2 path remains unchanged in execute2x2MatMul().
        """
        build = self.build_fc_layer_block_program(
            weights_int4=weights_int4,
            activations_int4=activations_int4,
            out_features=out_features,
            in_features=in_features,
            array_size=array_size,
            apply_relu=apply_relu,
            apply_quant=apply_quant,
        )
        if self.backend_name.strip().lower() == "cuda":
            req_array_size = array_size or self.default_array_size
            exec_result = self.cuda_executor.execute(
                BlockedFCLoweringRequest(
                    weights_int4=weights_int4,
                    activations_int4=activations_int4,
                    out_features=out_features,
                    in_features=in_features,
                    array_size=req_array_size,
                    apply_relu=apply_relu,
                    apply_quant=apply_quant,
                    weight_addr=self.BUFFER_SECTION_B,
                    input_addr=self.BUFFER_SECTION_A,
                    result_addr=self.BUFFER_SECTION_C,
                )
            )
            return {
                **build,
                **exec_result,
            }
        if allow_segmentation and not build["fits_instruction_bram"]:
            return self.execute_fc_layer_blocked_segmented(
                weights_int4=weights_int4,
                activations_int4=activations_int4,
                out_features=out_features,
                in_features=in_features,
                array_size=array_size,
                apply_relu=apply_relu,
                apply_quant=apply_quant,
                max_words_per_segment=max_words_per_segment,
                timeout=timeout,
            )
        if not build["executable_on_current_fpga_path"]:
            return {
                **build,
                "executed": False,
                "reason": "Program generation succeeded, but execution is blocked by current RTL/ISA semantics.",
            }
        if self.uart is None:
            return {
                **build,
                "executed": False,
                "reason": "No UART attached to ProgramLoader.",
            }

        self.uart.flush_input()
        self.uploadProgram(build["program"])
        self.start()
        time.sleep(timeout)
        return {
            **build,
            "executed": True,
        }

    def execute_fc_layer_blocked_segmented(
        self,
        weights_int4,
        activations_int4,
        out_features: int,
        in_features: int,
        array_size: Optional[int] = None,
        apply_relu: bool = False,
        apply_quant: bool = True,
        max_words_per_segment: int = 1024,
        timeout: float = 0.5,
    ) -> Dict[str, Any]:
        seg = self.build_fc_layer_block_program_segmented(
            weights_int4=weights_int4,
            activations_int4=activations_int4,
            out_features=out_features,
            in_features=in_features,
            array_size=array_size,
            apply_relu=apply_relu,
            apply_quant=apply_quant,
            max_words_per_segment=max_words_per_segment,
        )
        if not seg["executable_on_current_fpga_path"]:
            return {
                **seg,
                "executed": False,
                "reason": "Segmented program generation succeeded, but execution is blocked by current RTL/ISA semantics.",
            }

        if self.uart is None:
            return {
                **seg,
                "executed": False,
                "reason": "No UART attached to ProgramLoader.",
                "estimated_uart_upload_start_phases": seg["segment_count"],
            }

        upload_time_s = 0.0
        exec_wait_s = 0.0
        fetch_time_s = 0.0
        bytes_sent = 0
        bytes_received = 0

        decoded = []
        for i, program in enumerate(seg["segments"]):
            self.uart.flush_input()

            t0 = time.perf_counter()
            self.uploadProgram(program)
            t1 = time.perf_counter()
            upload_time_s += (t1 - t0)

            # header: REARM+UPLOAD+len(2) plus program bytes
            bytes_sent += 4 + len(program)

            t2 = time.perf_counter()
            self.start()
            time.sleep(timeout)
            t3 = time.perf_counter()
            exec_wait_s += (t3 - t2)
            bytes_sent += 1  # start byte

            # Only the final segment emits finalize+fetch bytes.
            if i == seg["segment_count"] - 1:
                t4 = time.perf_counter()
                expected = seg["output_fetch_bytes"]
                received = self.uart.receive_exact(expected, timeout=max(0.1, timeout))
                t5 = time.perf_counter()
                fetch_time_s += (t5 - t4)
                bytes_received += len(received)
                decoded = self._decode_int4_bytes(received, seg["output_nibbles"])

            self._log(f"Segment {i+1}/{seg['segment_count']} executed")

        return {
            **seg,
            "executed": True,
            "estimated_uart_upload_start_phases": seg["segment_count"],
            "output_int4_padded": decoded,
            "timing": {
                "upload_time_ms": upload_time_s * 1000.0,
                "execution_wait_time_ms": exec_wait_s * 1000.0,
                "fetch_time_ms": fetch_time_s * 1000.0,
                "total_wall_time_ms": (upload_time_s + exec_wait_s + fetch_time_s) * 1000.0,
            },
            "io": {
                "bytes_sent": int(bytes_sent),
                "bytes_received": int(bytes_received),
            },
        }

    def build_fc_layer_block_program_compressed(
        self,
        weights_int4,
        activations_int4,
        out_features: int,
        in_features: int,
        array_size: Optional[int] = None,
        apply_relu: bool = False,
        apply_quant: bool = True,
        result_addr: int = BUFFER_SECTION_C,
    ) -> Dict[str, Any]:
        a = array_size or self.default_array_size
        w = np.asarray(weights_int4, dtype=np.int8)
        x = np.asarray(activations_int4, dtype=np.int8).flatten()
        if w.shape != (out_features, in_features):
            raise ValueError(f"weights shape mismatch: expected {(out_features, in_features)}, got {w.shape}")
        if x.shape[0] != in_features:
            raise ValueError(f"activation length mismatch: expected {in_features}, got {x.shape[0]}")

        out_blocks = math.ceil(out_features / a)
        in_blocks = math.ceil(in_features / a)
        out_padded = out_blocks * a
        in_padded = in_blocks * a
        w_pad = np.zeros((out_padded, in_padded), dtype=np.int8)
        w_pad[:out_features, :in_features] = w
        x_pad = np.zeros(in_padded, dtype=np.int8)
        x_pad[:in_features] = x

        e = ISAEncoder()
        for ob in range(out_blocks):
            out_base_addr = result_addr + ob * (a // 4)
            o0 = ob * a
            o1 = o0 + a
            for ib in range(in_blocks):
                i0 = ib * a
                i1 = i0 + a
                # append pre-encoded op chunk bytes directly
                e.instructions.append(
                    self._encode_accumulate_op_compressed(
                        w_pad[o0:o1, i0:i1], x_pad[i0:i1], out_base_addr, acc_clear=(ib == 0)
                    )
                )
            e.instructions.append(
                self._encode_finalize_fetch_op(
                    out_base_addr, a, apply_quant=apply_quant, apply_relu=apply_relu
                )
            )
        e.halt()
        program = e.getProgram()
        words = len(program) // 2
        return {
            "program": program,
            "program_instruction_words": int(words),
            "fits_instruction_bram": bool(words <= 1024),
            "array_size": int(a),
            "out_blocks": int(out_blocks),
            "in_blocks": int(in_blocks),
            "block_ops": int(out_blocks * in_blocks),
            "mode": "compressed",
            "executable_on_current_fpga_path": bool(apply_quant),
        }

    def build_full_inference_program_compressed_fused(
        self,
        fc1_weights_int4,
        fc2_weights_int4,
        input_activations_int4,
        array_size: Optional[int] = None,
        fc1_apply_relu: bool = True,
        fc2_apply_relu: bool = False,
        apply_quant: bool = True,
        result_addr: int = BUFFER_SECTION_C,
        num_pe: int = 1,
    ) -> Dict[str, Any]:
        """
        Build one fused compressed program:
        FC1 accumulate/finalize (no host fetch) -> FC2 consume FC1 output buffer -> final fetch -> HALT.

        ``num_pe=2`` emits a dual-PE FC1 K-split schedule when FC1 has >=2 K-blocks.
        """
        if int(num_pe) == 2:
            return self._build_full_inference_program_compressed_fused_2pe(
                fc1_weights_int4=fc1_weights_int4,
                fc2_weights_int4=fc2_weights_int4,
                input_activations_int4=input_activations_int4,
                array_size=array_size,
                fc1_apply_relu=fc1_apply_relu,
                fc2_apply_relu=fc2_apply_relu,
                apply_quant=apply_quant,
                result_addr=result_addr,
            )
        return self._build_full_inference_program_compressed_fused_1pe(
            fc1_weights_int4=fc1_weights_int4,
            fc2_weights_int4=fc2_weights_int4,
            input_activations_int4=input_activations_int4,
            array_size=array_size,
            fc1_apply_relu=fc1_apply_relu,
            fc2_apply_relu=fc2_apply_relu,
            apply_quant=apply_quant,
            result_addr=result_addr,
        )

    def _build_full_inference_program_compressed_fused_1pe(
        self,
        fc1_weights_int4,
        fc2_weights_int4,
        input_activations_int4,
        array_size: Optional[int] = None,
        fc1_apply_relu: bool = True,
        fc2_apply_relu: bool = False,
        apply_quant: bool = True,
        result_addr: int = BUFFER_SECTION_C,
    ) -> Dict[str, Any]:
        """
        Build one fused compressed program:
        FC1 accumulate/finalize (no host fetch) -> FC2 consume FC1 output buffer -> final fetch -> HALT.
        """
        a = array_size or self.default_array_size
        fc1_w = np.asarray(fc1_weights_int4, dtype=np.int8)
        fc2_w = np.asarray(fc2_weights_int4, dtype=np.int8)
        x = np.asarray(input_activations_int4, dtype=np.int8).flatten()

        fc1_out, fc1_in = fc1_w.shape
        fc2_out, fc2_in = fc2_w.shape
        if x.shape[0] != fc1_in:
            raise ValueError(f"input activation length mismatch: expected {fc1_in}, got {x.shape[0]}")
        if fc2_in > a:
            raise ValueError(f"FC2 input dim {fc2_in} exceeds array_size {a}; fused path assumes single FC2 input block")

        # FC1 padded views
        fc1_out_blocks = math.ceil(fc1_out / a)
        fc1_in_blocks = math.ceil(fc1_in / a)
        fc1_w_pad = np.zeros((fc1_out_blocks * a, fc1_in_blocks * a), dtype=np.int8)
        fc1_w_pad[:fc1_out, :fc1_in] = fc1_w
        x_pad = np.zeros(fc1_in_blocks * a, dtype=np.int8)
        x_pad[:fc1_in] = x

        # FC2 padded views (current model: one out block, one in block)
        fc2_out_blocks = math.ceil(fc2_out / a)
        fc2_in_blocks = math.ceil(fc2_in / a)
        fc2_w_pad = np.zeros((fc2_out_blocks * a, fc2_in_blocks * a), dtype=np.int8)
        fc2_w_pad[:fc2_out, :fc2_in] = fc2_w

        e = ISAEncoder()
        breakdown = {
            "fc1_weight_bstore_words": 0,
            "fc1_activation_bstore_words": 0,
            "fc1_load_words": 0,
            "fc1_run_words": 0,
            "fc1_fetch_words": 0,
            "fc1_halt_words": 0,
            "fc2_weight_bstore_words": 0,
            "fc2_activation_bstore_words": 0,
            "fc2_load_words": 0,
            "fc2_run_words": 0,
            "fc2_fetch_words": 0,
            "final_halt_words": 0,
        }

        # FC1: preload all input activation K-block words once (13 blocks * 4 words).
        fc1_input_words = self._pack_int4_words(x_pad)
        e.burst_store(self.BUFFER_SECTION_A, fc1_input_words)
        breakdown["fc1_activation_bstore_words"] += 2 + len(fc1_input_words)

        # FC1: group two K-block weight tiles per burst to reduce BSTORE headers.
        for ob in range(fc1_out_blocks):
            out_base_addr = result_addr + ob * (a // 4)
            o0 = ob * a
            o1 = o0 + a
            ib = 0
            while ib < fc1_in_blocks:
                # Burst up to two weight blocks into BUFFER_SECTION_B contiguous windows.
                group = min(2, fc1_in_blocks - ib)
                weight_words = []
                for g in range(group):
                    gi = ib + g
                    i0 = gi * a
                    i1 = i0 + a
                    weight_words.extend(self._pack_int4_words(fc1_w_pad[o0:o1, i0:i1]))
                e.burst_store(self.BUFFER_SECTION_B, weight_words)
                breakdown["fc1_weight_bstore_words"] += 2 + len(weight_words)

                for g in range(group):
                    gi = ib + g
                    # weight window start for this sub-block in BUFFER_SECTION_B
                    w_addr = self.BUFFER_SECTION_B + (64 * g)
                    # activation window start for this sub-block in BUFFER_SECTION_A
                    i_addr = self.BUFFER_SECTION_A + (4 * gi)
                    e.loadWeights(w_addr)
                    e.loadInputs(i_addr)
                    e.run(
                        out_base_addr,
                        compute=True,
                        quantize=False,
                        relu=False,
                        acc_clear=(gi == 0)
                    )
                    breakdown["fc1_load_words"] += 2
                    breakdown["fc1_run_words"] += 1
                ib += group
            # finalize FC1 but do not fetch to host
            e.instructions.append(self._encode_finalize_no_fetch_op(out_base_addr, apply_quant=apply_quant, apply_relu=fc1_apply_relu))
            breakdown["fc1_run_words"] += 1

        # FC2 consume FC1 output directly from buffer C (no FC2 activation BSTORE)
        for ob in range(fc2_out_blocks):
            out_base_addr = result_addr + ob * (a // 4)
            o0 = ob * a
            o1 = o0 + a
            for ib in range(fc2_in_blocks):
                i0 = ib * a
                i1 = i0 + a
                op_chunk = self._encode_accumulate_op_compressed_from_buffer(
                    fc2_w_pad[o0:o1, i0:i1],
                    input_base_addr=result_addr,  # FC1 finalized output lives here
                    out_base_addr=out_base_addr,
                    acc_clear=(ib == 0)
                )
                e.instructions.append(op_chunk)
                # weight BSTORE (2+64) + LOADWEI + LOADIN + RUN
                breakdown["fc2_weight_bstore_words"] += 66
                breakdown["fc2_load_words"] += 2
                breakdown["fc2_run_words"] += 1

            # finalize and fetch only final logits
            # Finalize FC2 and fetch only words required for true output dim.
            e.run(out_base_addr, compute=False, quantize=apply_quant, relu=fc2_apply_relu, acc_clear=False)
            needed_words = math.ceil(fc2_out / 4)
            for widx in range(needed_words):
                addr = out_base_addr + widx
                e.fetch(addr, top_half=False)
                e.fetch(addr, top_half=True)
            breakdown["fc2_run_words"] += 1
            breakdown["fc2_fetch_words"] += needed_words * 2

        e.halt()
        breakdown["final_halt_words"] = 1

        program = e.getProgram()
        words = len(program) // 2

        return {
            "program": program,
            "program_instruction_words": int(words),
            "fits_instruction_bram": bool(words <= 1024),
            "array_size": int(a),
            "fc1_out_blocks": int(fc1_out_blocks),
            "fc1_in_blocks": int(fc1_in_blocks),
            "fc2_out_blocks": int(fc2_out_blocks),
            "fc2_in_blocks": int(fc2_in_blocks),
            "mode": "compressed_fused",
            "num_pe": 1,
            "breakdown": breakdown,
            "executable_on_current_fpga_path": bool(apply_quant),
        }

    def _build_full_inference_program_compressed_fused_2pe(
        self,
        fc1_weights_int4,
        fc2_weights_int4,
        input_activations_int4,
        array_size: Optional[int] = None,
        fc1_apply_relu: bool = True,
        fc2_apply_relu: bool = False,
        apply_quant: bool = True,
        result_addr: int = BUFFER_SECTION_C,
    ) -> Dict[str, Any]:
        base = self._build_full_inference_program_compressed_fused_1pe(
            fc1_weights_int4=fc1_weights_int4,
            fc2_weights_int4=fc2_weights_int4,
            input_activations_int4=input_activations_int4,
            array_size=array_size,
            fc1_apply_relu=fc1_apply_relu,
            fc2_apply_relu=fc2_apply_relu,
            apply_quant=apply_quant,
            result_addr=result_addr,
        )
        a = array_size or self.default_array_size
        fc1_w = np.asarray(fc1_weights_int4, dtype=np.int8)
        fc2_w = np.asarray(fc2_weights_int4, dtype=np.int8)
        x = np.asarray(input_activations_int4, dtype=np.int8).flatten()
        fc1_out, fc1_in = fc1_w.shape
        fc2_out, fc2_in = fc2_w.shape
        fc1_in_blocks = math.ceil(fc1_in / a)
        if fc1_in_blocks < 2:
            return {
                **base,
                "num_pe": 1,
                "mode": "compressed_fused",
                "multi_pe_requested": True,
                "multi_pe_schedule_emitted": False,
                "multi_pe_fallback_reason": "FC1 has fewer than two K-blocks; 2-PE schedule requires K-split",
            }

        fc1_out_blocks = math.ceil(fc1_out / a)
        fc2_out_blocks = math.ceil(fc2_out / a)
        fc2_in_blocks = math.ceil(fc2_in / a)
        fc1_w_pad = np.zeros((fc1_out_blocks * a, fc1_in_blocks * a), dtype=np.int8)
        fc1_w_pad[:fc1_out, :fc1_in] = fc1_w
        x_pad = np.zeros(fc1_in_blocks * a, dtype=np.int8)
        x_pad[:fc1_in] = x
        fc2_w_pad = np.zeros((fc2_out_blocks * a, fc2_in_blocks * a), dtype=np.int8)
        fc2_w_pad[:fc2_out, :fc2_in] = fc2_w

        split = fc1_in_blocks // 2
        pe0_k_blocks = list(range(0, split if split > 0 else 1))
        pe1_k_blocks = list(range(split if split > 0 else 1, fc1_in_blocks))
        if not pe0_k_blocks or not pe1_k_blocks:
            return {
                **base,
                "num_pe": 1,
                "mode": "compressed_fused",
                "multi_pe_requested": True,
                "multi_pe_schedule_emitted": False,
                "multi_pe_fallback_reason": "unable to partition FC1 K-blocks across two PEs",
            }

        e = ISAEncoder()
        fc1_input_words = self._pack_int4_words(x_pad)
        e.burst_store(self.BUFFER_SECTION_A, fc1_input_words)
        e.buffer_xfer(
            src_addr=self.BUFFER_SECTION_A,
            dst_addr=self.BUFFER_SECTION_A,
            count=len(fc1_input_words),
            src_pe=0,
            dst_pe=1,
        )

        def _emit_fc1_k_blocks(pe_id: int, k_blocks: List[int]) -> None:
            e.pe_select(pe_id)
            for ob in range(fc1_out_blocks):
                out_base_addr = result_addr + ob * (a // 4)
                o0 = ob * a
                o1 = o0 + a
                for ib in k_blocks:
                    i0 = ib * a
                    i1 = i0 + a
                    weight_words = self._pack_int4_words(fc1_w_pad[o0:o1, i0:i1])
                    e.burst_store(self.BUFFER_SECTION_B, weight_words)
                    e.loadWeights(self.BUFFER_SECTION_B)
                    e.loadInputs(self.BUFFER_SECTION_A + (4 * ib))
                    e.run(
                        out_base_addr,
                        compute=True,
                        quantize=False,
                        relu=False,
                        acc_clear=(ib == k_blocks[0]),
                    )

        _emit_fc1_k_blocks(0, pe0_k_blocks)
        _emit_fc1_k_blocks(1, pe1_k_blocks)
        e.pe_select(0)
        e.barrier(barrier_id=0)
        e.acc_add(src_pe=1, dst_pe=0)

        for ob in range(fc1_out_blocks):
            out_base_addr = result_addr + ob * (a // 4)
            e.instructions.append(
                self._encode_finalize_no_fetch_op(out_base_addr, apply_quant=apply_quant, apply_relu=fc1_apply_relu)
            )

        for ob in range(fc2_out_blocks):
            out_base_addr = result_addr + ob * (a // 4)
            o0 = ob * a
            o1 = o0 + a
            for ib in range(fc2_in_blocks):
                i0 = ib * a
                i1 = i0 + a
                e.instructions.append(
                    self._encode_accumulate_op_compressed_from_buffer(
                        fc2_w_pad[o0:o1, i0:i1],
                        input_base_addr=result_addr,
                        out_base_addr=out_base_addr,
                        acc_clear=(ib == 0),
                    )
                )
            e.run(out_base_addr, compute=False, quantize=apply_quant, relu=fc2_apply_relu, acc_clear=False)
            needed_words = math.ceil(fc2_out / 4)
            for widx in range(needed_words):
                addr = out_base_addr + widx
                e.fetch(addr, top_half=False)
                e.fetch(addr, top_half=True)

        e.halt()
        program = e.getProgram()
        words = len(program) // 2
        return {
            "program": program,
            "program_instruction_words": int(words),
            "fits_instruction_bram": bool(words <= 1024),
            "array_size": int(a),
            "fc1_out_blocks": int(fc1_out_blocks),
            "fc1_in_blocks": int(fc1_in_blocks),
            "fc2_out_blocks": int(fc2_out_blocks),
            "fc2_in_blocks": int(fc2_in_blocks),
            "mode": "compressed_fused_2pe",
            "num_pe": 2,
            "multi_pe_requested": True,
            "multi_pe_schedule_emitted": True,
            "multi_pe_fc1_k_split": {
                "pe0_k_blocks": pe0_k_blocks,
                "pe1_k_blocks": pe1_k_blocks,
            },
            "executable_on_current_fpga_path": False,
            "multi_pe_sim_only": True,
            "blockers": [
                "2-PE schedule is simulator-validated only; current RTL is single-PE.",
            ],
        }

    # ------------------------------------------------------------------
    # readResults: upload a fetch program, run it, collect bytes
    # ------------------------------------------------------------------

    def readResults(self, base_addr: int, count: int) -> List[int]:
        self._log(f"Reading {count} values from 0x{base_addr:03X}")
        self.encoder.clear()
        n_words = (count + 3) // 4
        for i in range(n_words):
            addr = base_addr + i
            self.encoder.fetch(addr, top_half=False)
            self.encoder.fetch(addr, top_half=True)
        self.encoder.halt()

        program = self.encoder.getProgram()
        self.uart.flush_input()
        self.uploadProgram(program)
        self.start()

        time.sleep(0.1)
        n_bytes = n_words * 2
        received = self.uart.receive_exact(n_bytes, timeout=0.2)
        self._log(f"Received {len(received)} bytes")

        results = []
        for byte in received:
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            if lo >= 8: lo -= 16
            if hi >= 8: hi -= 16
            results.append(lo)
            results.append(hi)

        return results[:count]

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def resetChip(self):
        """Flush host-side UART buffers. Use hardware reset line to restart the FSM."""
        self._log("Flushing host-side UART buffers")
        self.uart.flush_input()
        time.sleep(0.05)
        self.uart.flush_input()
        self._log("Host-side flush complete")


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
        inputs  = [1, 1]

        results = loader.execute2x2MatMul(
            weights,
            inputs,
            ProgramLoader.BUFFER_SECTION_B,
            ProgramLoader.BUFFER_SECTION_A,
            ProgramLoader.BUFFER_SECTION_C
        )

        print(f"Weights: {weights}")
        print(f"Inputs:  {inputs}")
        print(f"Results: {results}")

        uart.close()

    except Exception as e:
        print(f"Error: {e}")
