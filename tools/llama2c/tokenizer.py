"""Direct Python port of llama2.c run.c's tokenizer encode/decode."""

from __future__ import annotations

import re
import struct
from pathlib import Path


class Tokenizer:
    def __init__(self, path: str | Path, vocab_size: int):
        with Path(path).open("rb") as f:
            (self.max_token_length,) = struct.unpack("<I", f.read(4))
            self.scores, self.vocab = [], []
            for _ in range(vocab_size):
                score, size = struct.unpack("<fI", f.read(8))
                self.scores.append(score)
                self.vocab.append(f.read(size))
            if f.read(1):
                raise ValueError("tokenizer has trailing data (wrong vocab_size?)")
        self.lookup = {piece: i for i, piece in enumerate(self.vocab)}

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        raw = text.encode("utf-8")
        tokens = ([1] if bos else []) + ([self.lookup[b" "]] if raw else [])
        # Python iteration over decoded characters exactly groups valid UTF-8 codepoints.
        for char in text:
            piece = char.encode("utf-8")
            token = self.lookup.get(piece)
            tokens.extend([b + 3 for b in piece] if token is None else [token])
        while True:
            best_score, best_id, best_idx = float("-inf"), -1, -1
            for i in range(len(tokens) - 1):
                merged = self.lookup.get(self.vocab[tokens[i]] + self.vocab[tokens[i + 1]])
                if merged is not None and self.scores[merged] > best_score:
                    best_score, best_id, best_idx = self.scores[merged], merged, i
            if best_idx < 0:
                break
            tokens[best_idx : best_idx + 2] = [best_id]
        if eos:
            tokens.append(2)
        return tokens

    def decode_piece(self, previous: int, token: int) -> bytes:
        piece = self.vocab[token]
        if previous == 1 and piece.startswith(b" "):
            piece = piece[1:]
        match = re.fullmatch(br"<0x([0-9A-Fa-f]{2})>", piece)
        return bytes([int(match.group(1), 16)]) if match else piece

    def decode(self, tokens: list[int]) -> str:
        out, previous = bytearray(), 1
        for token in tokens:
            out += self.decode_piece(previous, token)
            previous = token
        return out.decode("utf-8", errors="replace")
