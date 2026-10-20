# llama2.c float golden

Pinned revisions and SHA-256 digests are in `manifest.json`. Run from repo root:

```text
python tools/llama2c/download.py
python -m tools.llama2c.verify models/llama2c/stories260K/stories260K.bin
python -m tools.llama2c.metrics
python -m pytest tools/llama2c/test_llama2c.py -q
```

`reference.py` loads llama2.c v0 checkpoints and implements float32 RMSNorm, dynamic
RoPE, GQA causal attention, SwiGLU, an explicit KV cache, and single-token decode.
`tokenizer.py` ports `run.c`'s score-ordered BPE merge and byte fallback. The vendored
`run.c` and `model.py` are pinned to the `llama2c_revision` in the manifest.
PPL uses the pinned PyTorch model as a batched evaluator only after the artifact's
64-token NumPy parity gate passes; this avoids an impractical scalar 32K-vocabulary
projection for every 15M-model eval token. `metrics.greedy_agreement` performs real
stateful candidate-vs-float comparisons for the integer implementations in later chunks.

The official `karpathy/tinyllamas` repository does not contain `tokenizer.bin`; its 15M
model uses llama2.c's standard 32K tokenizer, so the downloader obtains that file from
the pinned official `karpathy/llama2.c` revision. The manifest records this exception.
