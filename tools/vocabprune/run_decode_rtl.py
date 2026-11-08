"""Run the existing ideal-stream 15M RTL testbench against one pruned image."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decode.harness import BUILD, parse_summary, write_mem
from tools.decode.pack_decode_image import pack, load_export_15m
from tools.decode.run_decode_rtl import FLASH_IDEAL, SRCS, TB
from tools.decode.sim import VVP, compile_and_run
from tools.vocabprune.core import prune_export


def ensure_rtl_luts() -> None:
    """Materialize the packed ROMs expected by RTL's relative $readmemh paths."""
    src = ROOT / "firmware/host/numerics_v2/luts"
    dst = ROOT / "rtl/decode/luts"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("sigmoid_q15.mem", "exp_q15.mem"):
        target = dst / name
        if not target.exists():
            target.write_bytes((src / name).read_bytes())
    for suffix in ("h8", "h48"):
        target = dst / f"rope_cs_q15_{suffix}{'_p128' if suffix == 'h48' else ''}.mem"
        if not target.exists():
            cos = (src / f"rope_cos_q15_{suffix}.mem").read_text().splitlines()
            sin = (src / f"rope_sin_q15_{suffix}.mem").read_text().splitlines()
            target.write_text("".join(f"{c.strip()}{s.strip()}\n" for c, s in zip(cos, sin)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-keep", type=Path, required=True)
    ap.add_argument("--model", choices=("15m", "260k"), default="15m")
    ap.add_argument("--prompts", type=int, default=1)
    ap.add_argument("--gen", type=int, default=2)
    ap.add_argument("--output", type=Path, default=ROOT / "bench/results/vocab_prune_rtl.json")
    args = ap.parse_args()
    ensure_rtl_luts()
    idmap = json.loads(args.vocab_keep.read_text(encoding="utf-8"))
    if args.model == "260k":
        from tools.decode.harness import load_runtime_260k
        source_export, _ = load_runtime_260k(kv_bits=4)
    else:
        source_export = load_export_15m()
    export = prune_export(source_export, idmap["kept_original_ids"])
    from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama
    runtime = RunQIntegerLlama(export, ROOT / "firmware/host/numerics_v2/luts", kv_bits=4)
    keep = int(idmap["keep_size"])
    # The existing runner's 15M prompt is retained when all IDs are kept; for a pruned
    # image each prompt ID must already be in K.  This deterministic prompt uses only rows
    # guaranteed by the map, then exercises two generated steps.
    prompts = [[idmap["original_to_compact"][1], idmap["original_to_compact"][3]]]
    if args.prompts != 1:
        raise ValueError("this bounded pruned-image smoke accepts exactly one prompt")
    seqs, logits = [], []
    for prompt in prompts:
        runtime.reset()
        tokens, outs, rows = [], [], []
        tok = prompt[0]
        for pos in range(len(prompt) + args.gen - 1):
            logit = runtime.forward(tok, pos).astype(np.int64)
            nxt = int(np.argmax(logit))
            tokens.append(tok); outs.append(nxt); rows.append(logit)
            tok = prompt[pos + 1] if pos + 1 < len(prompt) else nxt
        seqs.append((tokens, outs))
        logits.extend(rows)
    image, params = pack(export)
    tag = f"vocab{keep}"
    files = dict(img=write_mem(f"e2e_img_{tag}", image, 8),
                 steps=write_mem(f"e2e_steps_{tag}", [(t << 32) | (o << 16) | p
                                                       for (tokens, outs) in seqs for p, (t, o) in enumerate(zip(tokens, outs))], 48),
                 logits=write_mem(f"e2e_logits_{tag}", [int(x) for row in logits for x in row], 32))
    keys = ['D', 'H', 'NL', 'NH', 'NKV', 'HS', 'VOCAB', 'CLS4', 'WQ4', 'WK4', 'WV4', 'WO4', 'W14', 'W34', 'W24',
            'CLS_BASE', 'EMB_SCALE_BASE', 'LAYERS_BASE', 'IMAGE_BYTES', 'CLS_TILE_BYTES', 'CLS_KB', 'CLS_G', 'RECIP_D', 'NORM_MULT',
            'EPS', 'GFB', 'NORM_SHIFT', 'SILU_MULT', 'SWIGLU_MULT', 'SCORE_MULT', 'SILU_SHIFT', 'SWIGLU_SHIFT', 'SCORE_SHIFT']
    plist = ', '.join(f'.{k}({params[k]})' for k in keys) + f', .MAX_SEQ({128 if args.model == "15m" else 64}), .KV4(1)'
    if args.model == "15m":
        plist += ', .ROPE_CS("rtl/decode/luts/rope_cs_q15_h48_p128.mem"), .ROPE_DEPTH(3072)'
    steps = sum(len(x[0]) for x in seqs)
    tb = BUILD / f"e2e_{tag}_tb.sv"
    tb.write_text(TB % dict(flash=FLASH_IDEAL, ns=steps, vocab=keep, nbytes=len(image), period=1,
                            params=plist, pw=7, limit=80_000_000, **files))
    srcs = [ROOT / "rtl/decode" / s for s in SRCS]
    t0 = time.time()
    rc, out = compile_and_run(srcs, "e2e_tb", tb, timeout=3600)
    line, kv = parse_summary(out)
    result = {"status": "pass" if rc == 0 and kv.get("errors") == "0" else "fail",
              "vocab": keep, "prompts": len(prompts), "generated_tokens_per_prompt": args.gen,
              "image_bytes": len(image), "image_sha256": hashlib.sha256(image).hexdigest(),
              "logit_mismatches": int(kv.get("errors", -1)), "token_mismatches": int(kv.get("token_errors", -1)),
              "argmax_errors": int(kv.get("argmax_errors", -1)), "summary": line or out[-1000:],
              "first_mismatches": [x for x in out.splitlines() if x.startswith(("MISMATCH", "TOKEN", "INDEX"))][:8],
              "vvp_command": f"{VVP} build/decode/e2e_tb.vvp", "wall_seconds": round(time.time() - t0, 1),
              "model": args.model,
              "command": f"python tools/vocabprune/run_decode_rtl.py --model {args.model} --vocab-keep {args.vocab_keep} --prompts 1 --gen {args.gen}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
