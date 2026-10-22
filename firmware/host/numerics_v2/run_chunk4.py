"""Chunk-4 sensitivity and diagonal-GPTQ experiments for stories15M."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from tools.llama2c.metrics import tokenize_eval
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer

from .export import accumulator_bounds, export_runq_model
from .run_15m import LOGIT_CHUNK, teacher_forced_metrics, hardware_budget
from .runq_runtime import RunQIntegerLlama

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def evaluate(reference, scales, hessian, ids, bits, group, name, mixed=None, gptq=False):
    exported = export_runq_model(reference, scales, bits, group, improved=False,
                                 hessian_diag=hessian if gptq else None,
                                 mixed_precision=mixed, gptq=gptq)
    base = exported.pop("dynamic_base_scales")
    model = RunQIntegerLlama(exported, HERE / "luts")
    row = {"name": name, "weight_bits": bits, "group_size": group,
           "gptq": gptq, "mixed_precision": mixed or {},
           "accumulator_bounds": accumulator_bounds(reference, bits, group),
           "hardware_budget": hardware_budget(exported, bits)}
    row.update(teacher_forced_metrics(reference, model, ids, base["logits"], "cpu"))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--config", action="append")
    ap.add_argument("--output", type=Path, default=ROOT / "bench/results/numerics_v2_chunk4.json")
    args = ap.parse_args()
    base = ROOT / "models/llama2c"
    reference = Llama2CModel(base / "stories15M.bin")
    tokenizer = Tokenizer(base / "tokenizer.bin", reference.config.vocab_size)
    ids = tokenize_eval(tokenizer, base / "eval_slice.txt", minimum=args.tokens)[:args.tokens + 1]
    calibration = json.loads((HERE / "calibration_15m.json").read_text(encoding="utf-8"))
    scales, hessian = calibration["activation_scales"], calibration["hessian_diag"]
    all_specs = [("baseline_int4_g32", 4, 32, None, False),
             ("classifier_embedding_int8", 4, 32, {"classifier": 8}, False),
             ("qkv_int8", 4, 32, {"qkv": 8}, False),
             ("wo_int8", 4, 32, {"wo": 8}, False),
             ("w13_int8", 4, 32, {"w13": 8}, False),
             ("w2_int8", 4, 32, {"w2": 8}, False),
             ("gptq_g32", 4, 32, None, True),
             ("gptq_g64", 4, 64, None, True),
             ("gptq_classifier_embedding_int8_g32", 4, 32, {"classifier": 8}, True),
             ("gptq_qkv_int8_g32", 4, 32, {"qkv": 8}, True),
             ("gptq_qkv_classifier_int8_g32", 4, 32, {"qkv": 8, "classifier": 8}, True)]
    specs = all_specs if not args.config else [item for item in all_specs if item[0] in args.config]
    rows = []
    for name, bits, group, mixed, gptq in specs:
        print(f"starting {name}", flush=True)
        rows.append(evaluate(reference, scales, hessian, ids, bits, group, name, mixed, gptq))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema_version": 1, "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                           "model": "stories15M", "eval_tokens": args.tokens,
                                           "calibration": calibration["metadata"], "configs": rows}, indent=2) + "\n",
                                encoding="utf-8")
    print(json.dumps([(r["name"], r["perplexity"], r["teacher_forced_top1_agreement_pct"],
                       r["hardware_budget"]["weight_bytes_including_scales"]) for r in rows], indent=2))


if __name__ == "__main__":
    main()
