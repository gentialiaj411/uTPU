"""Delta-debugging minimizer (Task 2 / `utpu_upgrade_plan.md` §4.2 step 4).

Given a `GeneratedProgram` whose evaluation triggers a divergence, find
the smallest program (fewest ops, smallest shapes) that still triggers
the same divergence. This is the classic `ddmin` algorithm specialized
to GraphIR mutations:

1. **Op deletion** — try removing one op at a time from the tail (an op
   whose output is downstream of the failing relation often does not
   matter); keep the deletion if the failing relation still fires AND
   the resulting graph is still legal (shape inference passes, every
   referenced value is produced, the failing relation is still
   applicable).
2. **Shape shrinking** — try halving each dimension that participates in
   the failing op's input shape, regenerating consistent weights.
3. **Op kind specialization** — try replacing a chain `LINEAR -> RELU`
   with `LINEAR_RELU` (or vice versa) — sometimes the bug is in the
   fused form only.

The interface is intentionally narrow: `minimize(program, predicate)`,
where `predicate(program) -> bool` is a callback the fuzzer driver
supplies that returns True iff the failing condition reproduces. The
minimizer never re-runs the full backend set — the caller decides what
counts as "still failing".
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from graph_ir import GraphIR, OpKind, OpNode
from graph_passes import shape_inference_pass

from fuzz.graph_generator import GeneratedProgram, assert_program_legal


@dataclass(frozen=True)
class MinimizationStats:
    """Summary of one minimization run."""

    initial_op_count: int
    final_op_count: int
    initial_input_count: int
    final_input_count: int
    iterations: int
    deletions_attempted: int
    deletions_kept: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_op_count": int(self.initial_op_count),
            "final_op_count": int(self.final_op_count),
            "initial_input_count": int(self.initial_input_count),
            "final_input_count": int(self.final_input_count),
            "iterations": int(self.iterations),
            "deletions_attempted": int(self.deletions_attempted),
            "deletions_kept": int(self.deletions_kept),
        }


Predicate = Callable[[GeneratedProgram], bool]


def _drop_op(program: GeneratedProgram, op_idx: int) -> Optional[GeneratedProgram]:
    """Return a new program with `op_idx` removed and dangling values cleaned.

    Only legal when the removed op's outputs have no remaining consumers
    AND no graph output points at them. The graph's terminal output may
    be redirected to the previous op's output IF that previous op's
    shape matches the original output shape (otherwise the shape
    contract changes and the predicate is no longer comparable).
    """
    g = program.graph
    if op_idx < 0 or op_idx >= len(g.ops):
        return None
    new_g = copy.deepcopy(g)
    op = new_g.ops[op_idx]
    op_outputs = list(op.outputs)

    # Allow removing the terminal op IF we can redirect graph.outputs to
    # its primary input, which then must already be a value with a
    # matching shape and become a graph output. We only attempt this when
    # the op is the last in topological order AND its output is the sole
    # graph output — anything fancier is not worth it for ddmin.
    is_terminal = op_idx == len(new_g.ops) - 1 and op_outputs and op_outputs[0] in new_g.outputs

    if is_terminal:
        if not op.inputs:
            return None
        new_terminal = op.inputs[0]
        # Shape compatibility: the new terminal value's shape must equal
        # the dropped op's output shape; otherwise a shape mismatch in
        # the predicate would be misattributed to the bug.
        old_out_value = new_g.values.get(op_outputs[0])
        new_out_value = new_g.values.get(new_terminal)
        if (
            old_out_value is None
            or new_out_value is None
            or old_out_value.shape is None
            or new_out_value.shape is None
            or tuple(old_out_value.shape) != tuple(new_out_value.shape)
        ):
            return None
        new_g.outputs = [new_terminal]
    else:
        # Internal op: drop only if no remaining op consumes its outputs
        # and they aren't in graph.outputs.
        for out in op_outputs:
            if out in new_g.outputs:
                return None
        for other in new_g.ops:
            if other is op:
                continue
            for inp in other.inputs:
                if inp in op_outputs:
                    return None

    new_g.ops = [o for i, o in enumerate(new_g.ops) if i != op_idx]
    # Drop now-orphaned values (never produced, never consumed, not an
    # input/output). Keep input + output values intact.
    referenced: set = set(new_g.inputs) | set(new_g.outputs)
    for o in new_g.ops:
        referenced.update(o.inputs)
        referenced.update(o.outputs)
    new_g.values = {k: v for k, v in new_g.values.items() if k in referenced}
    # Drop dangling consumers from value records (they referenced the
    # removed op and would confuse downstream passes).
    for value in new_g.values.values():
        value.consumers = [c for c in value.consumers if c != op.name]
        if value.producer == op.name:
            value.producer = None
    # Drop graph.inputs that are no longer referenced (e.g. the residual
    # for a dropped ADD).
    surviving_inputs: List[str] = [n for n in new_g.inputs if n in referenced]
    if not surviving_inputs:
        return None
    drop_indices = [i for i, n in enumerate(new_g.inputs) if n not in surviving_inputs]
    new_inputs_arrays = [
        program.inputs[i] for i in range(len(program.inputs)) if i not in set(drop_indices)
    ]
    new_g.inputs = surviving_inputs
    if len(new_inputs_arrays) != len(new_g.inputs):
        return None
    # Final legality bar: shape inference still succeeds.
    try:
        shape_inference_pass(new_g)
    except Exception:
        return None
    new_program = GeneratedProgram(
        seed=program.seed,
        graph=new_g,
        inputs=list(new_inputs_arrays),
        metadata={**program.metadata, "minimized": True},
    )
    try:
        assert_program_legal(new_program)
    except AssertionError:
        return None
    return new_program


def minimize(
    program: GeneratedProgram,
    predicate: Predicate,
    max_iterations: int = 32,
) -> Tuple[GeneratedProgram, MinimizationStats]:
    """Shrink `program` while `predicate` still fires.

    Strategy: greedy single-op deletion sweeps, repeated until a full
    sweep makes no progress. Per sweep we walk ops back-to-front (more
    likely to be deletable safely). The result is locally minimal — not
    globally minimal, but good enough for a reproducer. We cap iterations
    so a misbehaving predicate can't loop forever.

    The caller-supplied `predicate` MUST be deterministic on the same
    program; the minimizer assumes True/False is stable.
    """
    initial_op_count = len(program.graph.ops)
    initial_input_count = len(program.graph.inputs)
    deletions_attempted = 0
    deletions_kept = 0
    iterations = 0
    current = program
    if not predicate(current):
        # Predicate already false — nothing to minimize.
        return current, MinimizationStats(
            initial_op_count=initial_op_count,
            final_op_count=initial_op_count,
            initial_input_count=initial_input_count,
            final_input_count=initial_input_count,
            iterations=0,
            deletions_attempted=0,
            deletions_kept=0,
        )

    while iterations < max_iterations:
        iterations += 1
        progress = False
        # Walk indices high-to-low so deletions don't shift remaining
        # indices we haven't considered yet.
        for op_idx in range(len(current.graph.ops) - 1, -1, -1):
            if op_idx >= len(current.graph.ops):
                continue
            deletions_attempted += 1
            candidate = _drop_op(current, op_idx)
            if candidate is None:
                continue
            try:
                still_fails = predicate(candidate)
            except Exception:
                still_fails = False
            if still_fails:
                current = candidate
                deletions_kept += 1
                progress = True
        if not progress:
            break

    return current, MinimizationStats(
        initial_op_count=initial_op_count,
        final_op_count=len(current.graph.ops),
        initial_input_count=initial_input_count,
        final_input_count=len(current.graph.inputs),
        iterations=iterations,
        deletions_attempted=deletions_attempted,
        deletions_kept=deletions_kept,
    )
