from __future__ import annotations

import argparse
import heapq
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PY = Path(r"C:\Users\bhask\AppData\Local\Programs\Python\Python311\python.exe")
FULL = Path(r"C:\Users\bhask\Documents\PROJECTS\uTPU\build\decode\decode_image_15m\image.bin")
FULL_PARAMS = FULL.with_name("params.json")
PRUNED = Path(r"C:\Users\bhask\Documents\PROJECTS\uTPU-vocabprune\build\vocabprune\vocab_8192.image.bin")
PRUNED_PARAMS = PRUNED.with_name("vocab_8192.image.params.json")

ALIGN = 16
TABLE_BYTES = 256                 # one canonical code-length byte per symbol
BLOCK_HEADER_BYTES = 4            # compressed-byte count, needed for sequential tile stepping
SEEK_INDEX_BYTES = 4              # one address/length entry per random-access classifier row


def ceildiv(a, b):
    return (a + b - 1) // b


def align(a, n=ALIGN):
    return ceildiv(a, n) * n


def huffman_lengths(data: bytes) -> list[int]:
    counts = Counter(data)
    if not counts:
        return [0] * 256
    heap = [(n, i, [i]) for i, n in counts.items()]
    heapq.heapify(heap)
    next_id = 256
    lengths = [0] * 256
    while len(heap) > 1:
        a_n, _, a_syms = heapq.heappop(heap)
        b_n, _, b_syms = heapq.heappop(heap)
        for s in a_syms + b_syms:
            lengths[s] += 1
        heapq.heappush(heap, (a_n + b_n, next_id, a_syms + b_syms))
        next_id += 1
    return lengths


def bounded_lengths(data: bytes, max_len: int) -> list[int]:
    """Length-limited canonical-Huffman lengths.

    The first pass is ordinary Huffman.  The standard overflow repair moves
    long leaves upward while preserving Kraft equality; symbols are then
    assigned longest codes in ascending-frequency order.
    """
    raw = huffman_lengths(data)
    # Reconstruct symbol/frequency order explicitly (Counter.values() order is not enough).
    freq = Counter(data)
    syms = sorted(freq, key=lambda s: (freq[s], s))
    counts = Counter(x for x in raw if x)
    overflow = sum(n for l, n in counts.items() if l > max_len)
    if overflow:
        counts[max_len] += 0
        # Each repair replaces one node at depth b by two at b+1 and removes one
        # node at the current maximum depth. This is the zlib/JPEG-style repair.
        while overflow > 0:
            b = max_len - 1
            while b > 0 and counts[b] == 0:
                b -= 1
            if b == 0:
                break
            counts[b] -= 1
            counts[b + 1] += 2
            counts[max_len] -= 1
            overflow -= 2
        # Any remaining long leaves are represented by max_len leaves.
        for l in list(counts):
            if l > max_len:
                counts[max_len] += counts[l]
                del counts[l]
    # Make the requested number of leaves exact if the repair overshot by one.
    nused = len(syms)
    total = sum(counts.values())
    if total != nused:
        counts = Counter({max_len: nused})
        # Split leaves until Kraft equality and leaf count match.
        counts.clear()
        counts[max_len] = nused
        while sum(counts.values()) < nused:
            l = max(k for k, v in counts.items() if v and k > 1)
            counts[l] -= 1
            counts[l - 1] += 2
    out = [0] * 256
    pos = 0
    for l in range(max_len, 0, -1):
        for _ in range(counts.get(l, 0)):
            if pos >= len(syms):
                break
            out[syms[pos]] = l
            pos += 1
    if pos != len(syms):
        # Safe fallback: a complete fixed-length table is always valid for 256 symbols.
        out = [max_len if i in freq else 0 for i in range(256)]
    kraft = sum(2.0 ** -x for x in out if x)
    if kraft > 1.0000001:
        raise RuntimeError((max_len, kraft, counts))
    return out


def huff_payload(data: bytes, max_len: int) -> tuple[int, int, int]:
    lengths = bounded_lengths(data, max_len)
    bits = sum(lengths[x] for x in data)
    return ceildiv(bits, 8), bits, TABLE_BYTES


def zero_rle_bytes(data: bytes) -> int:
    # Exact hardware-simple format: [tag,value/length] per literal or zero run.
    # tag=0 literal, tag=1 run; run length is one byte (split at 255).
    i = out = 0
    while i < len(data):
        if data[i] == 0:
            j = i
            while j < len(data) and data[j] == 0:
                j += 1
            out += 2 * ceildiv(j - i, 255)
            i = j
        else:
            out += 2
            i += 1
    return out


def rice_bits_signed(data: bytes, k: int = 2) -> int:
    # Signed-byte zigzag, unary quotient plus k-bit remainder. Lossless.
    total = 0
    for x in data:
        s = x if x < 128 else x - 256
        z = (s << 1) if s >= 0 else ((-s << 1) - 1)
        total += (z >> k) + 1 + k
    return total


def canonical_encode(data: bytes, max_len: int = 15) -> tuple[list[int], bytes, int]:
    lengths = bounded_lengths(data, max_len)
    code = 0; prev = 0; codes = {}
    for sym, ln in sorted(((s, l) for s, l in enumerate(lengths) if l), key=lambda x: (x[1], x[0])):
        code <<= ln - prev; codes[sym] = (code, ln); code += 1; prev = ln
    out = bytearray(); acc = nbits = 0
    for sym in data:
        c, ln = codes[sym]; acc = (acc << ln) | c; nbits += ln
        while nbits >= 8:
            nbits -= 8; out.append((acc >> nbits) & 0xff); acc &= (1 << nbits) - 1 if nbits else 0
    if nbits: out.append((acc << (8 - nbits)) & 0xff)
    return lengths, bytes(out), sum(lengths[x] for x in data)


def canonical_decode(lengths: list[int], payload: bytes, bit_count: int) -> bytes:
    code = 0; prev = 0; table = {}
    for sym, ln in sorted(((s, l) for s, l in enumerate(lengths) if l), key=lambda x: (x[1], x[0])):
        code <<= ln - prev; table[(ln, code)] = sym; code += 1; prev = ln
    out = bytearray(); cur = n = 0
    for i in range(bit_count):
        cur = (cur << 1) | ((payload[i // 8] >> (7 - i % 8)) & 1); n += 1
        sym = table.get((n, cur))
        if sym is not None: out.append(sym); cur = n = 0
    if n: raise ValueError("truncated canonical Huffman payload")
    return bytes(out)


def tensor_shape(rows, k, int4):
    groups = ceildiv(k, 32)
    kb = ceildiv(k, 2) if int4 else k
    tiles = ceildiv(rows, 16)
    tile = groups * 16 * 3 + 16 * kb
    return groups, kb, tiles, tile


def parse_image(image: bytes, p: dict):
    """Return logical tensor blocks and the two streamed classifier views."""
    tensors = []
    pos = p["CLS_BASE"]
    def add(name, rows, k, int4, kind="weight"):
        nonlocal pos
        groups, kb, tiles, tile_bytes = tensor_shape(rows, k, int4)
        blocks = []
        for t in range(tiles):
            b = image[pos:pos + tile_bytes]
            meta_len = groups * 16 * 3
            meta = b[:meta_len]
            codes = b[meta_len:]
            row_codes = [codes[r * kb:(r + 1) * kb] for r in range(16)]
            blocks.append({"meta": meta, "codes": codes, "rows": row_codes})
            pos += tile_bytes
        tensors.append({"name": name, "rows": rows, "k": k, "int4": int4, "groups": groups,
                        "kb": kb, "blocks": blocks, "kind": kind, "raw_bytes": tiles * tile_bytes})
    add("classifier", p["VOCAB"], p["D"], bool(p["CLS4"]))
    cls_end = p["EMB_SCALE_BASE"]
    if pos != cls_end:
        raise RuntimeError(f"classifier parse ended at {pos}, expected {cls_end}")
    emb_scales = image[p["EMB_SCALE_BASE"]:p["LAYERS_BASE"]]
    pos = p["LAYERS_BASE"]
    for li in range(p["NL"]):
        gamma = image[pos:pos + p["D"] * 2]; pos += p["D"] * 2
        tensors.append({"name": f"layer{li}.gamma_att", "kind": "metadata", "raw": gamma, "raw_bytes": len(gamma)})
        for name, rows, k, flag in [("wq", p["D"], p["D"], p["WQ4"]), ("wk", p["D"], p["D"], p["WK4"]),
                                     ("wv", p["D"], p["D"], p["WV4"]), ("wo", p["D"], p["D"], p["WO4"])]:
            add(f"layer{li}.{name}", rows, k, bool(flag))
        gamma = image[pos:pos + p["D"] * 2]; pos += p["D"] * 2
        tensors.append({"name": f"layer{li}.gamma_ffn", "kind": "metadata", "raw": gamma, "raw_bytes": len(gamma)})
        for name, rows, k, flag in [("w1", p["H"], p["D"], p["W14"]), ("w3", p["H"], p["D"], p["W34"]),
                                     ("w2", p["D"], p["H"], p["W24"])]:
            add(f"layer{li}.{name}", rows, k, bool(flag))
    gamma = image[pos:pos + p["D"] * 2]; pos += p["D"] * 2
    tensors.append({"name": "gamma_final", "kind": "metadata", "raw": gamma, "raw_bytes": len(gamma)})
    if pos != len(image):
        raise RuntimeError(f"image parse ended at {pos}, file is {len(image)}")
    return tensors, emb_scales


def block_size(block, max_len, codec):
    meta = len(block["meta"])
    codes = block["codes"]
    if codec == "huffman":
        payload, bits, _ = huff_payload(codes, max_len)
        return align(meta + BLOCK_HEADER_BYTES + payload), bits
    if codec == "zero_rle":
        payload = zero_rle_bytes(codes)
        return align(meta + BLOCK_HEADER_BYTES + payload), payload * 8
    if codec == "rice":
        bits = rice_bits_signed(codes, 2)
        return align(meta + BLOCK_HEADER_BYTES + ceildiv(bits, 8)), bits
    return align(meta + BLOCK_HEADER_BYTES + len(codes)), len(codes) * 8


def summarize_model(image_path, params_path):
    image = image_path.read_bytes(); p = json.loads(params_path.read_text())
    tensors, emb_scales = parse_image(image, p)
    weight_tensors = [x for x in tensors if x.get("kind") == "weight"]
    codes_all = b"".join(b["codes"] for t in weight_tensors for b in t["blocks"])
    meta_all = b"".join(b["meta"] for t in weight_tensors for b in t["blocks"]) + b"".join(x.get("raw", b"") for x in tensors if x.get("kind") == "metadata") + emb_scales
    entropy = lambda d: (sum(n * (-math.log2(n / len(d))) for n in Counter(d).values()) if d else 0.0)
    codecs = {}
    for codec in ["huffman12", "huffman13", "huffman14", "huffman15", "zero_rle", "rice", "raw"]:
        hlen = int(codec[-2:]) if codec.startswith("huffman") else 15
        layer_size = 0; cls_full = 0; cls_row_sizes = []
        table_count = 0; block_count = 0; bits = 0
        for t in tensors:
            if t.get("kind") == "metadata":
                layer_size += len(t["raw"])
                continue
            table_count += 1
            for bi, b in enumerate(t["blocks"]):
                sz, bbits = block_size(b, hlen, "huffman" if codec.startswith("huffman") else codec)
                block_count += 1; bits += bbits
                if t["name"] == "classifier":
                    cls_full += sz
                    for row in b["rows"]:
                        if codec.startswith("huffman"):
                            pb, _, _ = huff_payload(row, hlen); rs = align(BLOCK_HEADER_BYTES + pb)
                        elif codec == "zero_rle": rs = align(BLOCK_HEADER_BYTES + zero_rle_bytes(row))
                        elif codec == "rice": rs = align(BLOCK_HEADER_BYTES + ceildiv(rice_bits_signed(row, 2), 8))
                        else: rs = align(BLOCK_HEADER_BYTES + len(row))
                        cls_row_sizes.append(rs)
                else:
                    layer_size += sz
        table_bytes = table_count * TABLE_BYTES if codec.startswith("huffman") else 0
        # Random classifier row access needs a fixed directory. The full classifier
        # stream already contains all row blocks; embedding pays one more row block.
        row_index = p["VOCAB"] * SEEK_INDEX_BYTES
        # The image stores the complete embedding-scale table for random row
        # access, but one token streams only its CLS_G entries.
        emb_scale_row = p["CLS_G"] * 5
        compressed_stream_avg = layer_size + cls_full + emb_scale_row + row_index + sum(cls_row_sizes) / len(cls_row_sizes)
        compressed_stream_max = layer_size + cls_full + emb_scale_row + row_index + max(cls_row_sizes)
        total_image = layer_size + cls_full + len(emb_scales) + table_bytes + row_index
        codecs[codec] = {"compressed_image_bytes": int(math.ceil(total_image)),
                         "stream_bytes_per_token_avg": compressed_stream_avg + table_bytes,
                         "stream_bytes_per_token_max": compressed_stream_max + table_bytes,
                         "payload_bits": int(bits), "table_bytes": table_bytes,
                         "seek_index_bytes": row_index, "block_count": block_count}
    stream_raw = p["CLS_KB"] + p["CLS_G"] * 5 + (p["IMAGE_BYTES"] - p["LAYERS_BASE"]) + (p["EMB_SCALE_BASE"] - p["CLS_BASE"])
    # The expression above includes the full classifier and layer tail, and the
    # per-token row+scale prefix. It is equivalent to run_decode_rtl.py:206.
    region_map = {}
    for t in tensors:
        key = ("classifier" if t.get("kind") == "weight" and t["name"] == "classifier"
               else (t["name"].split(".")[-1] if t.get("kind") == "metadata" else t["name"].split(".")[-1]))
        q = region_map.setdefault(key, {"raw_bytes": 0, "code_bytes": 0, "metadata_bytes": 0, "entropy_sum": 0.0})
        if t.get("kind") == "metadata":
            q["raw_bytes"] += t["raw_bytes"]; q["metadata_bytes"] += t["raw_bytes"]
        else:
            cb = b"".join(b["codes"] for b in t["blocks"]); mb = b"".join(b["meta"] for b in t["blocks"])
            q["raw_bytes"] += t["raw_bytes"]; q["code_bytes"] += len(cb); q["metadata_bytes"] += len(mb)
            q["entropy_sum"] += entropy(cb)
    region_map["embedding_scales"] = {"raw_bytes": len(emb_scales), "code_bytes": 0, "metadata_bytes": len(emb_scales), "entropy_sum": entropy(emb_scales)}
    for q in region_map.values():
        q["order0_entropy_bits_per_code_byte"] = q["entropy_sum"] / q["code_bytes"] if q["code_bytes"] else None
        del q["entropy_sum"]
    return {"image": str(image_path), "image_bytes": len(image), "params": p,
            "stream_raw_bytes_per_token": stream_raw, "order0_entropy_bits_per_byte": entropy(codes_all) / len(codes_all),
            "order0_entropy_code_bytes": len(codes_all), "metadata_bytes": len(meta_all),
            "tensors": [{"name": t["name"], "raw_bytes": t.get("raw_bytes", 0),
                         "code_bytes": sum(len(b["codes"]) for b in t.get("blocks", [])),
                         "metadata_bytes": sum(len(b["meta"]) for b in t.get("blocks", [])) if t.get("kind") == "weight" else t.get("raw_bytes", 0),
                         "rows": t.get("rows"), "int4": t.get("int4")} for t in tensors], "regions": region_map, "codecs": codecs}


def projections(results):
    # These are the repo's measured/claimed decode floors. The user-provided
    # pruned 7,734,699 figure is treated as its compute floor because 50e6/3.78
    # is flash-bound at about 13.23M cycles, not 7.73M.
    floors = {"15m": 12338619, "vocab8192": 7734699}
    observed = {"15m": {"stream": 10684413, "cycles": 21744984.545454547},
                "vocab8192": {"stream": 6613245, "cycles": 2 * 6613245 + 6159}}
    out = {}
    for key, r in results.items():
        name = "15m" if key == "15m" else "vocab8192"
        floor = floors[name]; obs = observed[name]
        out[name] = {"compute_floor_cycles": floor, "baseline_cycles": obs["cycles"], "baseline_tok_s": 50_000_000 / obs["cycles"], "codecs": {}}
        for codec, c in r["codecs"].items():
            cyc = max(floor, c["stream_bytes_per_token_avg"] / 0.5 + (obs["cycles"] - 2 * obs["stream"]))
            out[name]["codecs"][codec] = {"cycles_per_token": cyc, "tok_s": 50_000_000 / cyc,
                                           "gain_pct": (obs["cycles"] / cyc - 1) * 100}
    return out


def make_doc(payload):
    r15, r8 = payload["models"]["15m"], payload["models"]["vocab8192"]
    lines = ["# Lossless weight compression study", "", "Generated by `tools/wcomp/weight_compression_study.py`.", "",
             "## Scope and measured stream", "", "The packer is `tools/decode/pack_decode_image.py`; the RTL prefetcher in `rtl/decode/top_decode.sv` performs four seeks per token: embedding code row, embedding scale row, the layer tail, and the classifier region. The classifier row seek makes row-granular random access mandatory. `rtl/decode/qspi_reader.sv` then emits the selected range in address order; it has no decompression or random seek inside a compressed block.", "",
             "The measured raw streams are **%d B/token** for 15M and **%d B/token** for pruned K=8192. The 15M number agrees with `run_decode_rtl.py`'s `stream_bytes_per_token` formula and the recorded DDR result." % (r15["stream_raw_bytes_per_token"], r8["stream_raw_bytes_per_token"]), "",
             "## Entropy and codec sizes", "", "Sizes include raw scale/gamma metadata, 16-byte block alignment, 4-byte block lengths, one 4-byte classifier row index entry per vocabulary row, and one 256-byte canonical code-length table per weight tensor for Huffman. Huffman is measured per 16-row tile; classifier code rows are separately seekable. `zero_rle` is a two-byte literal/run format; `rice` is signed-zigzag Rice(k=2); `raw` is aligned blocks with headers.", "",
             "| image | order-0 code entropy | codec | image bytes | avg streamed B/token | max streamed B/token |", "|---|---:|---|---:|---:|---:|"]
    for name, rr in [("15M", r15), ("K=8192", r8)]:
        for codec, c in rr["codecs"].items():
            lines.append("| %s | %.3f bit/B | %s | %d | %.1f | %.1f |" % (name, rr["order0_entropy_bits_per_byte"], codec, c["compressed_image_bytes"], c["stream_bytes_per_token_avg"], c["stream_bytes_per_token_max"]))
    lines += ["", "The code-only entropy is reported separately because scale and gamma bytes are structured metadata rather than int8 weight symbols. Nibble packing is already present for INT4 tensors in the baseline image; a generic nibble/bit-plane transpose is size-neutral, so it is not counted as a gain. It can still be useful as a wiring/layout choice.", "", "## Projection", "", "The projection uses 50 MHz and 0.5 flash B/cycle, with each model's measured/reference DDR residual held constant. The compute floors are 12,338,619 cycles/token for 15M and 7,734,699 for K=8192. The latter is interpreted as the compute floor: 50 MHz / 3.78 tok/s is approximately 13.23M cycles/token, so treating 7.73M as total cycles would be internally inconsistent.", "", "| image | codec | projected cycles/token | projected tok/s | gain vs DDR baseline |", "|---|---|---:|---:|---:|"]
    for name, pp in payload["projections"].items():
        for codec, c in pp["codecs"].items():
            lines.append("| %s | %s | %.0f | %.3f | %.1f%% |" % (name, codec, c["cycles_per_token"], c["tok_s"], c["gain_pct"]))
    lines += ["", "## Hardware feasibility and verdict", "", "A canonical Huffman decoder belongs between the QSPI byte stream and the existing FIFO/weight consumer. A 12–15-bit two-level table (prefix table plus suffix table) can produce one symbol/cycle; two replicated lookup paths can produce two symbols/cycle. For a 15M stream, the compute side consumes roughly 0.87 B/cycle at the compute floor, so one decoded byte/cycle is sufficient with a small FIFO. Two symbols/cycle is the safer implementation point for code-length skew and backpressure.", "", "A practical estimate is 1–2 BRAM36 for code tables/FIFO state and roughly 500–1,500 LUTs for a one/two-symbol canonical decoder, plus a few hundred LUTs for the block directory and refill FSM. Against the recorded xc7s50 budget (~20.8k/32.6k LUT, 67/75 BRAM36, 73/120 DSP), the LUT cost is acceptable but BRAM headroom is only about eight BRAM36 and timing/placement are material risks. Metadata remains raw and must bypass the decoder. The decoder adds a block-start latency of approximately 16–40 cycles, hidden by the existing 4 KiB prefetch FIFO except at short embedding seeks.", "", "The legacy 15M image conditionally clears the 10% bar (+11.4% projected with Huffman-15), but the planned K=8192 default reaches only +5.7%. Overall verdict: **NO-GO for the planned default**; conditional GO for legacy 15M only if that image remains a target. The script includes bit-exact reference encode/decode round trips; no RTL was started because the default verdict is NO-GO.", "", "## Reproduction", "", "`python tools/wcomp/weight_compression_study.py` regenerates this JSON and this document from the two read-only images and their parameter files.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, default=ROOT / "bench/results/weight_compression_study.json")
    a = ap.parse_args()
    models = {"15m": summarize_model(FULL, FULL_PARAMS), "vocab8192": summarize_model(PRUNED, PRUNED_PARAMS)}
    round_trips = []
    for name, path, pp in [("15m", FULL, FULL_PARAMS), ("vocab8192", PRUNED, PRUNED_PARAMS)]:
        img = path.read_bytes(); ts, _ = parse_image(img, json.loads(pp.read_text()))
        samples = [b["codes"] for t in ts if t.get("kind") == "weight" for b in t["blocks"]]
        for i in (0, len(samples) // 2, len(samples) - 1):
            lengths, enc, nbits = canonical_encode(samples[i], 15)
            round_trips.append({"model": name, "sample": i, "bytes": len(samples[i]), "pass": canonical_decode(lengths, enc, nbits) == samples[i]})
    payload = {"study": "lossless_weight_compression", "script": "tools/wcomp/weight_compression_study.py", "models": models, "projections": projections(models), "reference_round_trip": round_trips, "verdict": "NO-GO"}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (ROOT / "docs/WEIGHT_COMPRESSION_STUDY.md").write_text(make_doc(payload), encoding="utf-8")
    print(json.dumps({"output": str(a.output), "verdict": payload["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
