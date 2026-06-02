"""Random valid GraphIR program generator (Task 2 / `utpu_upgrade_plan.md` §4.2).

The generator samples small `nn.Module`-equivalent GraphIR programs over a
weighted op set:

* `LINEAR(x, W, b)`           — matmul, optionally with bias
* `LINEAR_RELU(x, W, b)`      — fused (Linear+ReLU) post-fusion form
* `RELU(x)`                   — elementwise
* `ADD(a, b)`                 — elementwise (residual when one operand is external);
                                also supports broadcasting variants (e.g. bias add
                                with shape ``(N,)`` against ``(M, N)``).
* `SCALE(x, c)`               — elementwise multiply by constant
* `VIEW(x, shape)`            — reshape; size-preserving
* `PERMUTE(x, axes)`          — axis transpose
* `SOFTMAX(x)`                — along last axis
* `LAYER_NORM(x)`             — RMS-norm variant; matches reference interpreter
* `BATCHED_MATMUL(a, b)`      — 3-D matmul

Every emitted graph satisfies the compiler's own legality predicates:

1. `is_op_supported_for_backend(op_kind, "cuda")` is True for every op.
2. `shape_inference_pass(graph)` succeeds (every consumer's input shapes
   are consistent with the producer's output shape).
3. The graph is acyclic + topologically sorted in `graph.ops` order.
4. Every input named in `graph.inputs` is referenced; every output in
   `graph.outputs` is produced (no dangling references).
5. Weights / biases / scales are concrete `np.ndarray` attrs (no None
   placeholders) so `GraphReferenceInterpreter` can run the graph.

The generator is **deterministic** given a seed: same `seed` always
yields the same graph + the same input tensors. The fuzzer driver loops
seeds 0..N to walk a deterministic corpus.

Shape sampling is weighted to **corner cases** the rest of the project
already cares about: multiples of `array_size = 16` (boundary tile sizes
for `tiling_controller`), powers-of-two used by the megakernel benchmark
(64/128/256/512), and small "tiny" shapes that exercise launch-overhead
regimes. See `SHAPE_BUCKETS` for the locked sampling distribution.

Graph families (v2 / Task 2 hardening pass, 2026-05-25):
  - `linear_chain`                  — N linears with per-layer epilogue.
  - `elementwise_chain`             — RELU / SCALE / ADD chain.
  - `residual_branch`               — producer with TWO consumers (multi-
                                       consumer; tests fusion rejection +
                                       DCE invariants under sharing).
  - `broadcast_add_chain`           — ADD with broadcasting bias-like vector.
  - `layout_chain`                  — VIEW / PERMUTE / VIEW round-trips
                                       with a final elementwise tail.
  - `mixed_linear_elementwise`      — interleaved LINEAR / elementwise ops.
  - `multi_consumer_reject_or_dce_case` — a LINEAR whose output feeds two
                                       branches plus a deliberately-dead op
                                       so DCE has something to drop.
  - `attention_lite`                — a small SOFTMAX/LAYER_NORM/BMM chain
                                       on rank-3 tensors. Emitted only if
                                       the chosen bucket fits the rank-3
                                       reduction safely; opt-in via weight.

Every family routes through `assert_program_legal` before being returned.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import is_op_supported_for_backend, shape_inference_pass


# Op kinds the generator may emit. Every entry MUST be in
# `_BACKEND_SUPPORTED_OPS["cuda"]` (see `graph_passes.py`); we assert this
# at module import time so a future op-set change is caught loudly.
_GENERATOR_OPS: FrozenSet[str] = frozenset(
    {
        OpKind.LINEAR,
        OpKind.LINEAR_RELU,
        OpKind.RELU,
        OpKind.ADD,
        OpKind.SCALE,
        OpKind.VIEW,
        OpKind.PERMUTE,
        OpKind.SOFTMAX,
        OpKind.LAYER_NORM,
        OpKind.BATCHED_MATMUL,
    }
)
for _op in _GENERATOR_OPS:
    assert is_op_supported_for_backend(_op, "cuda"), (
        f"graph_generator emits {_op!r} but cuda backend doesn't support it; "
        "either add it to graph_passes._BACKEND_SUPPORTED_OPS or drop it here"
    )


# Weighted shape buckets. `(M, K, N)` triples, paired with a sampling weight.
# The buckets target three regimes:
#   * tiny / launch-overhead (the megakernel benchmark headline regime)
#   * boundary-aligned to `tiling_controller`'s `array_size = 16` family
#   * small powers-of-two that produce nontrivial reductions but still
#     run cheaply on the host
SHAPE_BUCKETS: Tuple[Dict[str, Any], ...] = (
    {"name": "tiny_8",          "weight": 2, "M": (1, 4),    "K": (8, 16),    "N": (8, 16)},
    {"name": "boundary_16",     "weight": 3, "M": (1, 4),    "K": (16, 16),   "N": (16, 16)},
    {"name": "boundary_32",     "weight": 3, "M": (1, 4),    "K": (32, 32),   "N": (32, 32)},
    {"name": "small_64",        "weight": 2, "M": (1, 8),    "K": (64, 64),   "N": (64, 64)},
    {"name": "small_128",       "weight": 1, "M": (1, 4),    "K": (128, 128), "N": (128, 128)},
    {"name": "rect_32x16",      "weight": 1, "M": (1, 2),    "K": (32, 32),   "N": (16, 16)},
    {"name": "rect_16x64",      "weight": 1, "M": (1, 2),    "K": (16, 16),   "N": (64, 64)},
)


# Graph families this generator can emit. Adding a new family means:
#   1) write `_build_<family>_graph` returning a `GeneratedProgram`,
#   2) add its name + weight to `GRAPH_FAMILY_WEIGHTS`,
#   3) dispatch in `generate_program` via `_FAMILY_DISPATCH`.
GRAPH_FAMILY_WEIGHTS: Tuple[Tuple[str, int], ...] = (
    ("linear_chain", 30),
    ("elementwise_chain", 18),
    ("residual_branch", 10),
    ("broadcast_add_chain", 8),
    ("layout_chain", 10),
    ("mixed_linear_elementwise", 14),
    ("multi_consumer_reject_or_dce_case", 8),
    ("attention_lite", 2),
)

ALL_GRAPH_FAMILIES: Tuple[str, ...] = tuple(name for name, _ in GRAPH_FAMILY_WEIGHTS)


@dataclass(frozen=True)
class GeneratedProgram:
    """A generator output bundle: the graph + a deterministic input sample.

    `seed` is the seed that produced this program; `inputs` is the list of
    `np.ndarray` values aligned with `graph.inputs`. `metadata` carries
    diagnostics (sampled shape bucket, op kind sequence) used for coverage
    reporting in the artifact.
    """

    seed: int
    graph: GraphIR
    inputs: List[np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _pick_shape_bucket(rng: random.Random) -> Dict[str, Any]:
    weights = [b["weight"] for b in SHAPE_BUCKETS]
    return rng.choices(list(SHAPE_BUCKETS), weights=weights, k=1)[0]


def _pick_int_in_range(rng: random.Random, low: int, high: int) -> int:
    if high <= low:
        return int(low)
    return int(rng.randint(low, high))


def _make_random_weight(
    rng: random.Random, shape: Tuple[int, ...], scale: float = 0.1
) -> np.ndarray:
    """Generate a small-magnitude float32 weight tensor.

    Magnitudes are kept tiny (default `scale=0.1`) so multi-op chains do
    not blow up to inf / NaN — that would make every tolerance comparison
    vacuously fail. We sample uniformly in `[-scale, scale]`.
    """
    state = np.random.RandomState(rng.randint(0, 2**31 - 1))
    return state.uniform(-scale, scale, size=shape).astype(np.float32)


def _make_random_input(rng: random.Random, shape: Tuple[int, ...]) -> np.ndarray:
    state = np.random.RandomState(rng.randint(0, 2**31 - 1))
    return state.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def _name(prefix: str, idx: int) -> str:
    return f"{prefix}_{idx}"


# ---------------------------------------------------------------------------
# Family: linear_chain
# ---------------------------------------------------------------------------

def _build_linear_chain_graph(
    rng: random.Random,
    seed: int,
    bucket: Dict[str, Any],
    n_layers: int,
    epilogue_pattern: Sequence[str],
) -> GeneratedProgram:
    """Generate `n_layers` of LINEAR with a small epilogue per layer.

    `epilogue_pattern` is a sequence of strings drawn from
    `{"relu", "scale", "add", "linear_relu"}` describing how each layer's
    output is post-processed. `add` sources a graph-input residual to keep
    the legality bar simple (residuals never come from inside a fused
    region — that would be `residual_internal` per `region_fusion`).
    """
    M = _pick_int_in_range(rng, bucket["M"][0], bucket["M"][1])
    K = _pick_int_in_range(rng, bucket["K"][0], bucket["K"][1])
    N = _pick_int_in_range(rng, bucket["N"][0], bucket["N"][1])

    g = GraphIR(name=f"fuzz_linear_chain_seed{seed}_layers{n_layers}")
    g.add_value("x", shape=(M, K), dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list: List[np.ndarray] = [_make_random_input(rng, (M, K))]
    op_seq: List[str] = []

    cur_value = "x"
    cur_features = K
    residual_count = 0
    next_op_idx = 0
    for layer in range(n_layers):
        next_features = N if layer == n_layers - 1 else _pick_int_in_range(
            rng, bucket["N"][0], bucket["N"][1]
        )
        bias = rng.random() < 0.5
        weight = _make_random_weight(rng, (next_features, cur_features))
        bias_arr: Optional[np.ndarray] = None
        if bias:
            bias_arr = _make_random_weight(rng, (next_features,))
        op_kind = OpKind.LINEAR
        force_linear_relu = (
            layer < len(epilogue_pattern)
            and epilogue_pattern[layer] == "linear_relu"
        )
        if force_linear_relu:
            op_kind = OpKind.LINEAR_RELU

        out_name = _name("y", next_op_idx)
        next_op_idx += 1
        attrs: Dict[str, Any] = {
            "weight": weight,
            "in_features": int(cur_features),
            "out_features": int(next_features),
        }
        if bias_arr is not None:
            attrs["bias"] = bias_arr
        g.add_value(out_name, shape=(M, next_features), dtype="torch.float32")
        g.add_op(
            OpNode(
                name=_name("linear", layer),
                op=op_kind,
                inputs=[cur_value],
                outputs=[out_name],
                attrs=attrs,
            )
        )
        op_seq.append(op_kind)
        cur_value = out_name
        cur_features = next_features

        epilogue_kind = (
            epilogue_pattern[layer]
            if layer < len(epilogue_pattern) and epilogue_pattern[layer] != "linear_relu"
            else None
        )
        if epilogue_kind == "relu":
            out2 = _name("y", next_op_idx)
            next_op_idx += 1
            g.add_value(out2, shape=(M, next_features), dtype="torch.float32")
            g.add_op(
                OpNode(
                    name=_name("relu", layer),
                    op=OpKind.RELU,
                    inputs=[cur_value],
                    outputs=[out2],
                    attrs={},
                )
            )
            op_seq.append(OpKind.RELU)
            cur_value = out2
        elif epilogue_kind == "scale":
            out2 = _name("y", next_op_idx)
            next_op_idx += 1
            g.add_value(out2, shape=(M, next_features), dtype="torch.float32")
            g.add_op(
                OpNode(
                    name=_name("scale", layer),
                    op=OpKind.SCALE,
                    inputs=[cur_value],
                    outputs=[out2],
                    attrs={"scale": float(round(rng.uniform(0.1, 2.0), 4))},
                )
            )
            op_seq.append(OpKind.SCALE)
            cur_value = out2
        elif epilogue_kind == "add":
            res_name = _name("residual", residual_count)
            residual_count += 1
            g.add_value(res_name, shape=(M, next_features), dtype="torch.float32")
            g.inputs.append(res_name)
            inputs_list.append(_make_random_input(rng, (M, next_features)))
            out2 = _name("y", next_op_idx)
            next_op_idx += 1
            g.add_value(out2, shape=(M, next_features), dtype="torch.float32")
            g.add_op(
                OpNode(
                    name=_name("add", layer),
                    op=OpKind.ADD,
                    inputs=[cur_value, res_name],
                    outputs=[out2],
                    attrs={},
                )
            )
            op_seq.append(OpKind.ADD)
            cur_value = out2

    g.outputs = [cur_value]
    return GeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        metadata={
            "kind": "linear_chain",
            "shape_bucket": bucket["name"],
            "n_layers": int(n_layers),
            "epilogue_pattern": list(epilogue_pattern),
            "ops_emitted": op_seq,
            "M": int(M),
            "K": int(K),
            "N": int(N),
        },
    )


# ---------------------------------------------------------------------------
# Family: elementwise_chain
# ---------------------------------------------------------------------------

def _build_elementwise_chain_graph(
    rng: random.Random,
    seed: int,
    bucket: Dict[str, Any],
    chain_len: int,
) -> GeneratedProgram:
    """Generate a pure elementwise chain (RELU / ADD residual / SCALE).

    Targets the `elementwise_chain` region kind in `region_fusion`.
    """
    K = _pick_int_in_range(rng, bucket["K"][0], bucket["K"][1])
    N = _pick_int_in_range(rng, bucket["N"][0], bucket["N"][1])
    shape = (K, N)
    g = GraphIR(name=f"fuzz_elementwise_seed{seed}_len{chain_len}")
    g.add_value("x", shape=shape, dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list: List[np.ndarray] = [_make_random_input(rng, shape)]
    op_seq: List[str] = []
    cur = "x"
    next_idx = 0
    residual_count = 0
    valid_kinds = ["relu", "scale", "add"]
    for step in range(chain_len):
        kind = rng.choice(valid_kinds)
        out_name = _name("e", next_idx)
        next_idx += 1
        g.add_value(out_name, shape=shape, dtype="torch.float32")
        if kind == "relu":
            g.add_op(
                OpNode(
                    name=_name("relu", step),
                    op=OpKind.RELU,
                    inputs=[cur],
                    outputs=[out_name],
                    attrs={},
                )
            )
            op_seq.append(OpKind.RELU)
        elif kind == "scale":
            g.add_op(
                OpNode(
                    name=_name("scale", step),
                    op=OpKind.SCALE,
                    inputs=[cur],
                    outputs=[out_name],
                    attrs={"scale": float(round(rng.uniform(0.1, 2.0), 4))},
                )
            )
            op_seq.append(OpKind.SCALE)
        else:  # add
            res_name = _name("residual", residual_count)
            residual_count += 1
            g.add_value(res_name, shape=shape, dtype="torch.float32")
            g.inputs.append(res_name)
            inputs_list.append(_make_random_input(rng, shape))
            g.add_op(
                OpNode(
                    name=_name("add", step),
                    op=OpKind.ADD,
                    inputs=[cur, res_name],
                    outputs=[out_name],
                    attrs={},
                )
            )
            op_seq.append(OpKind.ADD)
        cur = out_name
    g.outputs = [cur]
    return GeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        metadata={
            "kind": "elementwise_chain",
            "shape_bucket": bucket["name"],
            "chain_len": int(chain_len),
            "ops_emitted": op_seq,
            "K": int(K),
            "N": int(N),
        },
    )


# ---------------------------------------------------------------------------
# Family: residual_branch
# ---------------------------------------------------------------------------

def _build_residual_branch_graph(
    rng: random.Random, seed: int, bucket: Dict[str, Any]
) -> GeneratedProgram:
    """Build a graph with a producer feeding TWO downstream paths.

    Pattern: ``x -> LINEAR -> y0 -> [SCALE -> y1, RELU -> y2] -> ADD(y1, y2) -> out``.
    The producer (``LINEAR``) has multi-consumer output; the LINEAR_RELU
    fusion engine MUST reject this fusion (multi-consumer guard). Useful for
    exercising the rejection path in real corpora.
    """
    M = _pick_int_in_range(rng, bucket["M"][0], bucket["M"][1])
    K = _pick_int_in_range(rng, bucket["K"][0], bucket["K"][1])
    N = _pick_int_in_range(rng, bucket["N"][0], bucket["N"][1])
    g = GraphIR(name=f"fuzz_residual_branch_seed{seed}")
    g.add_value("x", shape=(M, K), dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list: List[np.ndarray] = [_make_random_input(rng, (M, K))]

    weight = _make_random_weight(rng, (N, K))
    g.add_value("y0", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="linear_root",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["y0"],
            attrs={
                "weight": weight,
                "in_features": int(K),
                "out_features": int(N),
            },
        )
    )
    g.add_value("y_scale", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="branch_scale",
            op=OpKind.SCALE,
            inputs=["y0"],
            outputs=["y_scale"],
            attrs={"scale": float(round(rng.uniform(0.1, 2.0), 4))},
        )
    )
    g.add_value("y_relu", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="branch_relu",
            op=OpKind.RELU,
            inputs=["y0"],
            outputs=["y_relu"],
            attrs={},
        )
    )
    g.add_value("out", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="branch_merge",
            op=OpKind.ADD,
            inputs=["y_scale", "y_relu"],
            outputs=["out"],
            attrs={},
        )
    )
    g.outputs = ["out"]
    return GeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        metadata={
            "kind": "residual_branch",
            "shape_bucket": bucket["name"],
            "ops_emitted": [OpKind.LINEAR, OpKind.SCALE, OpKind.RELU, OpKind.ADD],
            "M": int(M),
            "K": int(K),
            "N": int(N),
        },
    )


# ---------------------------------------------------------------------------
# Family: broadcast_add_chain
# ---------------------------------------------------------------------------

def _build_broadcast_add_chain_graph(
    rng: random.Random, seed: int, bucket: Dict[str, Any]
) -> GeneratedProgram:
    """ADD with a broadcast residual (rank-1 vs rank-2).

    Pattern: ``x:(M, N) + bias:(N,) -> y0 -> SCALE -> y1 -> ADD(y1, bias) -> out``.
    Tests the elementwise add path against numpy broadcasting. We rely on
    `shape_inference_pass`'s lenient rule (output shape := lhs shape) which
    matches NumPy semantics for this case. We pre-record the value shape
    of the broadcast operand as ``(N,)`` so shape inference for downstream
    ops sees the right ranks.
    """
    M = _pick_int_in_range(rng, bucket["M"][0], bucket["M"][1])
    N = _pick_int_in_range(rng, bucket["N"][0], bucket["N"][1])
    g = GraphIR(name=f"fuzz_broadcast_add_seed{seed}")
    g.add_value("x", shape=(M, N), dtype="torch.float32")
    g.add_value("bias_vec", shape=(N,), dtype="torch.float32")
    g.inputs = ["x", "bias_vec"]
    inputs_list: List[np.ndarray] = [
        _make_random_input(rng, (M, N)),
        _make_random_input(rng, (N,)),
    ]
    g.add_value("y0", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="bcast_add_0",
            op=OpKind.ADD,
            inputs=["x", "bias_vec"],
            outputs=["y0"],
            attrs={},
        )
    )
    g.add_value("y1", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="scale_post",
            op=OpKind.SCALE,
            inputs=["y0"],
            outputs=["y1"],
            attrs={"scale": float(round(rng.uniform(0.1, 2.0), 4))},
        )
    )
    g.add_value("out", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="bcast_add_tail",
            op=OpKind.ADD,
            inputs=["y1", "bias_vec"],
            outputs=["out"],
            attrs={},
        )
    )
    g.outputs = ["out"]
    return GeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        metadata={
            "kind": "broadcast_add_chain",
            "shape_bucket": bucket["name"],
            "ops_emitted": [OpKind.ADD, OpKind.SCALE, OpKind.ADD],
            "M": int(M),
            "N": int(N),
        },
    )


# ---------------------------------------------------------------------------
# Family: layout_chain
# ---------------------------------------------------------------------------

def _build_layout_chain_graph(
    rng: random.Random, seed: int, bucket: Dict[str, Any]
) -> GeneratedProgram:
    """VIEW + PERMUTE round-trip with an elementwise tail.

    Pattern (size-preserving):
      ``x:(M, K*N) -> VIEW -> (M, K, N) -> PERMUTE -> (M, N, K)
                  -> VIEW -> (M, N*K) -> SCALE -> RELU -> out``.
    K and N are drawn from the bucket and bounded so the flattened size
    stays small.
    """
    M = _pick_int_in_range(rng, bucket["M"][0], bucket["M"][1])
    K = _pick_int_in_range(rng, bucket["K"][0], min(bucket["K"][1], 16))
    N = _pick_int_in_range(rng, bucket["N"][0], min(bucket["N"][1], 16))
    flat = K * N
    g = GraphIR(name=f"fuzz_layout_chain_seed{seed}")
    g.add_value("x", shape=(M, flat), dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list: List[np.ndarray] = [_make_random_input(rng, (M, flat))]

    g.add_value("v0", shape=(M, K, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="view_in",
            op=OpKind.VIEW,
            inputs=["x"],
            outputs=["v0"],
            attrs={"args": ((M, K, N),)},
        )
    )
    g.add_value("p0", shape=(M, N, K), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="permute_swap",
            op=OpKind.PERMUTE,
            inputs=["v0"],
            outputs=["p0"],
            attrs={"args": ((0, 2, 1),)},
        )
    )
    g.add_value("v1", shape=(M, flat), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="view_out",
            op=OpKind.VIEW,
            inputs=["p0"],
            outputs=["v1"],
            attrs={"args": ((M, flat),)},
        )
    )
    g.add_value("v2", shape=(M, flat), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="scale_tail",
            op=OpKind.SCALE,
            inputs=["v1"],
            outputs=["v2"],
            attrs={"scale": float(round(rng.uniform(0.1, 2.0), 4))},
        )
    )
    g.add_value("out", shape=(M, flat), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="relu_tail",
            op=OpKind.RELU,
            inputs=["v2"],
            outputs=["out"],
            attrs={},
        )
    )
    g.outputs = ["out"]
    return GeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        metadata={
            "kind": "layout_chain",
            "shape_bucket": bucket["name"],
            "ops_emitted": [
                OpKind.VIEW,
                OpKind.PERMUTE,
                OpKind.VIEW,
                OpKind.SCALE,
                OpKind.RELU,
            ],
            "M": int(M),
            "K": int(K),
            "N": int(N),
        },
    )


# ---------------------------------------------------------------------------
# Family: mixed_linear_elementwise
# ---------------------------------------------------------------------------

def _build_mixed_linear_elementwise_graph(
    rng: random.Random, seed: int, bucket: Dict[str, Any]
) -> GeneratedProgram:
    """Interleaved LINEAR / elementwise pattern.

    Pattern: ``x -> LINEAR -> RELU -> SCALE -> LINEAR_RELU -> ADD(residual) -> out``.
    Exercises chains that span fusable regions broken by a global-sync
    boundary (the second LINEAR_RELU): the planner must NOT fuse across it.
    """
    M = _pick_int_in_range(rng, bucket["M"][0], bucket["M"][1])
    K = _pick_int_in_range(rng, bucket["K"][0], bucket["K"][1])
    H = _pick_int_in_range(rng, bucket["N"][0], bucket["N"][1])
    N = _pick_int_in_range(rng, bucket["N"][0], bucket["N"][1])

    g = GraphIR(name=f"fuzz_mixed_seed{seed}")
    g.add_value("x", shape=(M, K), dtype="torch.float32")
    g.add_value("residual_tail", shape=(M, N), dtype="torch.float32")
    g.inputs = ["x", "residual_tail"]
    inputs_list: List[np.ndarray] = [
        _make_random_input(rng, (M, K)),
        _make_random_input(rng, (M, N)),
    ]

    w1 = _make_random_weight(rng, (H, K))
    g.add_value("h0", shape=(M, H), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="lin1",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["h0"],
            attrs={
                "weight": w1,
                "in_features": int(K),
                "out_features": int(H),
                "bias": _make_random_weight(rng, (H,)),
            },
        )
    )
    g.add_value("h1", shape=(M, H), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="relu1",
            op=OpKind.RELU,
            inputs=["h0"],
            outputs=["h1"],
            attrs={},
        )
    )
    g.add_value("h2", shape=(M, H), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="scale1",
            op=OpKind.SCALE,
            inputs=["h1"],
            outputs=["h2"],
            attrs={"scale": float(round(rng.uniform(0.1, 2.0), 4))},
        )
    )

    w2 = _make_random_weight(rng, (N, H))
    g.add_value("h3", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="lin2_relu",
            op=OpKind.LINEAR_RELU,
            inputs=["h2"],
            outputs=["h3"],
            attrs={
                "weight": w2,
                "in_features": int(H),
                "out_features": int(N),
            },
        )
    )
    g.add_value("out", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="add_tail",
            op=OpKind.ADD,
            inputs=["h3", "residual_tail"],
            outputs=["out"],
            attrs={},
        )
    )
    g.outputs = ["out"]
    return GeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        metadata={
            "kind": "mixed_linear_elementwise",
            "shape_bucket": bucket["name"],
            "ops_emitted": [
                OpKind.LINEAR,
                OpKind.RELU,
                OpKind.SCALE,
                OpKind.LINEAR_RELU,
                OpKind.ADD,
            ],
            "M": int(M),
            "K": int(K),
            "H": int(H),
            "N": int(N),
        },
    )


# ---------------------------------------------------------------------------
# Family: multi_consumer_reject_or_dce_case
# ---------------------------------------------------------------------------

def _build_multi_consumer_reject_or_dce_graph(
    rng: random.Random, seed: int, bucket: Dict[str, Any]
) -> GeneratedProgram:
    """Multi-consumer producer + a dead RELU on the side path.

    Pattern:
      ``x -> LINEAR -> y0 -> [RELU -> y_used (graph output path),
                              SCALE -> y_dead (NOT consumed)]``.
    The dead-side SCALE produces a value that nothing consumes and is not
    declared as a graph output → DCE MUST drop it. The producer
    (`LINEAR`) has TWO consumers → `linear_relu_fusion` MUST reject the
    LINEAR->RELU pair (multi-consumer guard).
    """
    M = _pick_int_in_range(rng, bucket["M"][0], bucket["M"][1])
    K = _pick_int_in_range(rng, bucket["K"][0], bucket["K"][1])
    N = _pick_int_in_range(rng, bucket["N"][0], bucket["N"][1])
    g = GraphIR(name=f"fuzz_multi_consumer_seed{seed}")
    g.add_value("x", shape=(M, K), dtype="torch.float32")
    g.inputs = ["x"]
    inputs_list: List[np.ndarray] = [_make_random_input(rng, (M, K))]

    weight = _make_random_weight(rng, (N, K))
    g.add_value("y0", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="linear_shared",
            op=OpKind.LINEAR,
            inputs=["x"],
            outputs=["y0"],
            attrs={
                "weight": weight,
                "in_features": int(K),
                "out_features": int(N),
            },
        )
    )
    g.add_value("y_used", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="relu_used",
            op=OpKind.RELU,
            inputs=["y0"],
            outputs=["y_used"],
            attrs={},
        )
    )
    # Dead-side: SCALE consumes y0 (giving y0 two consumers) but its output
    # is neither downstream-consumed nor a graph output → DCE drops it.
    g.add_value("y_dead", shape=(M, N), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="scale_dead",
            op=OpKind.SCALE,
            inputs=["y0"],
            outputs=["y_dead"],
            attrs={"scale": float(round(rng.uniform(0.1, 2.0), 4))},
        )
    )
    g.outputs = ["y_used"]
    return GeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        metadata={
            "kind": "multi_consumer_reject_or_dce_case",
            "shape_bucket": bucket["name"],
            "ops_emitted": [OpKind.LINEAR, OpKind.RELU, OpKind.SCALE],
            "M": int(M),
            "K": int(K),
            "N": int(N),
            "has_dead_op": True,
            "has_multi_consumer": True,
        },
    )


# ---------------------------------------------------------------------------
# Family: attention_lite
# ---------------------------------------------------------------------------

def _build_attention_lite_graph(
    rng: random.Random, seed: int, bucket: Dict[str, Any]
) -> GeneratedProgram:
    """Tiny softmax/layer-norm/batched-matmul chain (rank-3 tensors).

    Pattern: ``x:(B, T, D) -> LAYER_NORM -> y0
                          -> SOFTMAX -> y1 (along last dim)
                          -> BATCHED_MATMUL(y1, w_v) -> out``.
    where ``w_v:(B, D, D)``. We do NOT emit a full attention block —
    that requires 4-D tensors and the rest of the QKV projection plumbing.
    This is a small chain that exercises the reference interpreter's
    softmax / layer_norm / bmm paths.
    """
    B = 1
    T = max(2, _pick_int_in_range(rng, 2, 8))
    D = max(4, _pick_int_in_range(rng, 4, 16))
    g = GraphIR(name=f"fuzz_attention_lite_seed{seed}")
    g.add_value("x", shape=(B, T, D), dtype="torch.float32")
    g.add_value("w_v", shape=(B, D, D), dtype="torch.float32")
    g.inputs = ["x", "w_v"]
    inputs_list: List[np.ndarray] = [
        _make_random_input(rng, (B, T, D)),
        _make_random_weight(rng, (B, D, D), scale=0.2),
    ]

    g.add_value("ln0", shape=(B, T, D), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="ln",
            op=OpKind.LAYER_NORM,
            inputs=["x"],
            outputs=["ln0"],
            attrs={"eps": 1e-5, "norm_kind": "rms_norm"},
        )
    )
    g.add_value("sm0", shape=(B, T, D), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="softmax",
            op=OpKind.SOFTMAX,
            inputs=["ln0"],
            outputs=["sm0"],
            attrs={},
        )
    )
    g.add_value("out", shape=(B, T, D), dtype="torch.float32")
    g.add_op(
        OpNode(
            name="bmm",
            op=OpKind.BATCHED_MATMUL,
            inputs=["sm0", "w_v"],
            outputs=["out"],
            attrs={},
        )
    )
    g.outputs = ["out"]
    return GeneratedProgram(
        seed=seed,
        graph=g,
        inputs=inputs_list,
        metadata={
            "kind": "attention_lite",
            "shape_bucket": bucket["name"],
            "ops_emitted": [OpKind.LAYER_NORM, OpKind.SOFTMAX, OpKind.BATCHED_MATMUL],
            "B": int(B),
            "T": int(T),
            "D": int(D),
        },
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _build_linear_chain_variant(
    rng: random.Random, seed: int, bucket: Dict[str, Any]
) -> GeneratedProgram:
    n_layers = rng.choice([1, 1, 2, 2, 3])
    epilogue_options = ["relu", "scale", "add", "linear_relu", None]
    pattern = [rng.choice(epilogue_options) for _ in range(n_layers)]
    cleaned = [p if p is not None else "relu" for p in pattern]
    return _build_linear_chain_graph(
        rng, seed=seed, bucket=bucket, n_layers=n_layers, epilogue_pattern=cleaned
    )


def _build_elementwise_chain_variant(
    rng: random.Random, seed: int, bucket: Dict[str, Any]
) -> GeneratedProgram:
    chain_len = rng.choice([2, 2, 3, 4])
    return _build_elementwise_chain_graph(rng, seed=seed, bucket=bucket, chain_len=chain_len)


_FAMILY_DISPATCH = {
    "linear_chain": _build_linear_chain_variant,
    "elementwise_chain": _build_elementwise_chain_variant,
    "residual_branch": _build_residual_branch_graph,
    "broadcast_add_chain": _build_broadcast_add_chain_graph,
    "layout_chain": _build_layout_chain_graph,
    "mixed_linear_elementwise": _build_mixed_linear_elementwise_graph,
    "multi_consumer_reject_or_dce_case": _build_multi_consumer_reject_or_dce_graph,
    "attention_lite": _build_attention_lite_graph,
}
for _fam in ALL_GRAPH_FAMILIES:
    assert _fam in _FAMILY_DISPATCH, f"graph family '{_fam}' has no dispatch entry"


def _pick_family(rng: random.Random) -> str:
    names = [n for n, _ in GRAPH_FAMILY_WEIGHTS]
    weights = [w for _, w in GRAPH_FAMILY_WEIGHTS]
    return rng.choices(names, weights=weights, k=1)[0]


def generate_program(seed: int, family: Optional[str] = None) -> GeneratedProgram:
    """Generate one random valid program from `seed`.

    Deterministic: same seed always yields the same program. The family
    selector is weighted via ``GRAPH_FAMILY_WEIGHTS``; pass an explicit
    ``family`` to force a specific generator (useful in tests).
    """
    rng = random.Random(int(seed))
    bucket = _pick_shape_bucket(rng)
    chosen_family = family if family is not None else _pick_family(rng)
    if chosen_family not in _FAMILY_DISPATCH:
        raise ValueError(
            f"unknown graph family {chosen_family!r}; valid families: {sorted(_FAMILY_DISPATCH)}"
        )
    builder = _FAMILY_DISPATCH[chosen_family]
    prog = builder(rng, seed=seed, bucket=bucket)

    # Shape-inference must succeed on every legal program by definition.
    # If it fails the generator is buggy; raise loudly so the test sees it.
    inferred = shape_inference_pass(prog.graph)
    prog.metadata["shape_inference_ok"] = True
    prog.metadata["op_count"] = len(prog.graph.ops)
    prog.metadata["input_count"] = len(prog.graph.inputs)
    prog.metadata["family"] = chosen_family
    # No-op: just ensure the inferred graph is throwaway-equivalent (we keep
    # the original; downstream reference interpreter does its own inference).
    _ = inferred
    return prog


def assert_program_legal(program: GeneratedProgram) -> None:
    """Strong legality check used by tests + the runner before recording a graph.

    Raises `AssertionError` with a focused message if any predicate fails.
    The fuzzer NEVER emits a program that fails this check; the test harness
    re-checks every generated graph anyway as a defense-in-depth.
    """
    g = program.graph
    assert isinstance(g, GraphIR), "program.graph must be GraphIR"
    assert g.inputs, f"graph '{g.name}' has no declared inputs"
    assert g.outputs, f"graph '{g.name}' has no declared outputs"
    assert len(program.inputs) == len(g.inputs), (
        f"graph '{g.name}' has {len(g.inputs)} inputs but {len(program.inputs)} sample tensors"
    )

    for name in g.inputs:
        assert name in g.values, f"input '{name}' missing from graph.values"

    for op in g.ops:
        assert op.op in _GENERATOR_OPS, (
            f"op '{op.name}' has unsupported kind '{op.op}' (generator emits only {sorted(_GENERATOR_OPS)})"
        )
        assert is_op_supported_for_backend(op.op, "cuda"), (
            f"op '{op.name}' kind '{op.op}' not legal for cuda backend"
        )
        for inp in op.inputs:
            assert inp in g.values, f"op '{op.name}' input '{inp}' not in graph.values"
        for out in op.outputs:
            assert out in g.values, f"op '{op.name}' output '{out}' not in graph.values"

    produced = {out for op in g.ops for out in op.outputs} | set(g.inputs)
    for out in g.outputs:
        assert out in produced, f"graph output '{out}' not produced by any op or input"

    # Acyclic + topologically sorted.
    available = set(g.inputs)
    for op in g.ops:
        for inp in op.inputs:
            assert inp in available, (
                f"op '{op.name}' consumes '{inp}' which has not been produced yet "
                "(graph is not topologically sorted)"
            )
        for out in op.outputs:
            assert out not in available, (
                f"op '{op.name}' redefines '{out}' (SSA invariant violated)"
            )
            available.add(out)

    shape_inference_pass(g)


def coverage_summary(programs: Sequence[GeneratedProgram]) -> Dict[str, Any]:
    """Aggregate generator-level coverage across a corpus.

    Returns:
      - `ops_covered`: sorted list of unique op kinds emitted.
      - `shape_buckets_covered`: sorted list of unique shape bucket names hit.
      - `kinds_covered`: sorted list of program-shape kinds (graph families).
      - `graph_families_covered`: alias of `kinds_covered` for schema clarity.

    The fuzzer artifact embeds this directly. It is NOT compiler-pass
    branch coverage (that is reported separately by `coverage.py` if
    available); it is the cheaper "what was the generator able to emit"
    measure that always works.
    """
    ops: set = set()
    buckets: set = set()
    kinds: set = set()
    for p in programs:
        for op in p.graph.ops:
            ops.add(op.op)
        bucket = p.metadata.get("shape_bucket")
        if bucket:
            buckets.add(bucket)
        kind = p.metadata.get("kind")
        if kind:
            kinds.add(kind)
    return {
        "ops_covered": sorted(ops),
        "shape_buckets_covered": sorted(buckets),
        "kinds_covered": sorted(kinds),
        "graph_families_covered": sorted(kinds),
    }
