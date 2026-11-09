"""Fixed-text K=8192 byte-fallback evaluation for a local INT4 artifact."""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.llama2c.tokenizer import Tokenizer
from tools.int4.vocabprune.core import encode_with_fallback, prune_export
from firmware.host.numerics_v2.runq_runtime import RunQIntegerLlama, _linear_high

def nll(logits, targets, scale):
    v = np.asarray(logits, dtype=np.float64) * scale
    m = np.max(v, axis=1)
    return float(np.sum(m + np.log(np.sum(np.exp(v - m[:, None]), axis=1)) - v[np.arange(targets.size), targets]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True); ap.add_argument('--idmap', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True); ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--text', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); t0 = time.time()
    payload = np.load(a.artifact, allow_pickle=True)['payload'][0]
    scale = float(payload['dynamic_base_scales']['logits']); tok = Tokenizer(a.tokenizer, 32000)
    idmap = json.loads(a.idmap.read_text()); keep = int(idmap['keep_size'])
    raw = a.text.read_bytes(); docs = [x for x in raw.decode('utf-8').split('<|endoftext|>') if x]
    full = RunQIntegerLlama(payload, ROOT/'firmware/host/numerics_v2/luts', kv_bits=4)
    pruned = prune_export(payload, idmap['kept_original_ids']); classifier = pruned['classifier']
    nll_sum = 0.0; count = 0; fallback = 0; expanded = 0
    for doc in docs:
        compact, sequence, fb, original_count = encode_with_fallback(tok, doc, idmap)
        fallback += fb; expanded += len(sequence)
        for start in range(0, len(sequence)-1, 128):
            window = sequence[start:min(start+129, len(sequence))]
            if len(window) < 2: continue
            hidden = full._forward_sequence_hidden(np.asarray(window[:-1], dtype=np.int64))
            logits = _linear_high(hidden, classifier)
            targets = np.asarray(compact[start+1:start+len(window)], dtype=np.int64)
            nll_sum += nll(logits, targets, scale); count += int(targets.size)
    out = {'status':'measured','artifact':str(a.artifact),'artifact_sha256':hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
           'idmap':str(a.idmap),'idmap_sha256':hashlib.sha256(a.idmap.read_bytes()).hexdigest(),'keep_size':keep,
           'eval_text':str(a.text),'eval_text_sha256':hashlib.sha256(raw).hexdigest(),'eval_tokens':count,
           'expanded_input_tokens':expanded,'fallback_tokens':fallback,'validation_bytes_scored':len(raw),
           'nll_nats':nll_sum,'bits_per_byte':nll_sum/math.log(2)/len(raw),'kv_bits':4,'softmax':'restricted to K=8192 kept rows',
           'protocol':'fixed eval_text.txt; each document byte-fallback encoded with idmap; non-overlapping 128-token windows reset per document',
           'elapsed_seconds':time.time()-t0}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2)+'\n'); print(json.dumps(out, indent=2))
if __name__ == '__main__': main()
