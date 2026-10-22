"""Three identical-prompt float versus best-fitting integer GPTQ samples."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from tools.llama2c.metrics import tokenize_eval
from tools.llama2c.reference import Llama2CModel
from tools.llama2c.tokenizer import Tokenizer
from .export import export_runq_model
from .run_260k import prompts_from
from .runq_runtime import RunQIntegerLlama

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def main():
    base = ROOT / "models/llama2c"
    ref = Llama2CModel(base / "stories15M.bin")
    tok = Tokenizer(base / "tokenizer.bin", ref.config.vocab_size)
    ids = tokenize_eval(tok, base / "eval_slice.txt", minimum=512)
    prompts = prompts_from(ids, count=3, length=8)
    cal = json.loads((HERE / "calibration_15m.json").read_text(encoding="utf-8"))
    mixed = {"qkv": 8, "wo": 8, "w2": 8}
    exported = export_runq_model(ref, cal["activation_scales"], 4, 32,
                                 improved=True, gptq=False,
                                 mixed_precision=mixed)
    exported.pop("dynamic_base_scales")
    integer = RunQIntegerLlama(exported, HERE / "luts")
    rows = []
    for prompt in prompts:
        ref.reset(); integer.reset()
        float_ids = list(prompt); int_ids = list(prompt)
        f_logits = None
        i_next = None
        for pos, token in enumerate(prompt):
            f_logits = ref.forward(int(token), pos)
            i_next = integer.forward_argmax(int(token), pos, 4096)
        for _ in range(56):
            f_token = int(np.argmax(f_logits)); float_ids.append(f_token)
            f_logits = ref.forward(f_token, len(float_ids) - 1)
            i_token = int(i_next)
            int_ids.append(i_token)
            i_next = integer.forward_argmax(i_token, len(int_ids) - 1, 4096)
        rows.append({"prompt": tok.decode(prompt), "float": tok.decode(float_ids), "integer": tok.decode(int_ids)})
    output = ROOT / "bench/results/chunk4_samples.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
