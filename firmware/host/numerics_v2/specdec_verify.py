"""Lossless speculative verification helpers for RunQIntegerLlama.

The runtime is deliberately left untouched.  A pass consumes the last accepted
token and k proposed tokens, returning the accepted proposals plus the greedy
fallback token.  On a rejection the runtime is restored to the state after the
accepted prefix, so the caller can append the fallback exactly as ordinary
greedy decoding would.
"""
from __future__ import annotations

import numpy as np


def _snapshot(rt):
    return (list(rt.tokens), rt.key_code.copy(), rt.key_scale.copy(),
            rt.value_code.copy(), rt.value_scale.copy())


def _restore(rt, snap):
    tokens, kc, ks, vc, vs = snap
    rt.tokens[:] = tokens
    rt.key_code[...] = kc
    rt.key_scale[...] = ks
    rt.value_code[...] = vc
    rt.value_scale[...] = vs


def verify_k(rt, position: int, drafts: list[int] | tuple[int, ...], token: int | None = None):
    """Verify up to ``k`` draft tokens and return an exact greedy result.

    ``rt`` must already contain the accepted context except for the pending
    input token, and ``position`` must be ``len(rt.tokens)``.  Pass that pending
    token explicitly in ``token`` (or use the last consumed token by default).
    The return value is a dict with ``accepted`` (draft
    tokens matching the integer model), ``token`` (the next greedy token), and
    ``output`` (accepted tokens followed by that token).  The runtime is left
    after consuming only the accepted draft tokens; this is the state needed
    before the returned fallback is appended by the caller.
    """
    if int(position) != len(rt.tokens):
        raise ValueError("position must equal len(runtime.tokens)")
    drafts = [int(x) for x in drafts]
    vocab = int(rt.cfg["vocab_size"])
    if any(x < 0 or x >= vocab for x in drafts):
        raise ValueError("draft token outside model vocabulary")
    snap = _snapshot(rt)
    prev = int(token if token is not None else (rt.tokens[-1] if rt.tokens else 1))
    accepted = []
    predicted = None
    rejected = False
    for i, draft in enumerate(drafts):
        predicted = int(rt.forward_argmax(prev, int(position) + i, 4096))
        if predicted != draft:
            rejected = True
            break
        accepted.append(draft)
        prev = draft
    else:
        # The final prediction is the greedy token following all k drafts.
        predicted = int(rt.forward_argmax(prev, int(position) + len(drafts), 4096))

    # Rejected speculative work is not part of the committed sequence.  Replay
    # only accepted proposals from the pre-pass state, preserving exact KV bits.
    if rejected:
        _restore(rt, snap)
        prev = int(token if token is not None else (rt.tokens[-1] if rt.tokens else 1))
        for i in range(len(accepted) + 1):
            # Consume the pending context token and each accepted token except
            # the last one; this leaves the state at the accepted prefix.
            if i >= len(accepted):
                break
            rt.forward_argmax(prev, int(position) + i, 4096)
            prev = accepted[i]
    return {"accepted": accepted, "token": predicted,
            "output": accepted + [predicted], "k": len(drafts)}


def greedy_step(rt, token: int, position: int) -> int:
    """Consume one token and return its bit-exact greedy successor."""
    return int(rt.forward_argmax(int(token), int(position), 4096))

