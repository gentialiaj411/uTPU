import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from cuda_blocked_fc_backend import CUDABlockedFCExecutor, CUDAGraphOpExecutor
from cuda_autotuner import DEFAULT_CACHE_PATH, lookup_best_schedule
from graph_ir import GraphIR, OpKind
from graph_runtime_plan import GraphRuntimePlan, RuntimeOpPlan
from lowering_types import BlockedFCLoweringRequest


class CompiledRuntimeError(RuntimeError):
    pass


@dataclass
class RuntimeOpTrace:
    graph_op: str
    op: str
    engine: str
    latency_ms: float
    notes: List[str] = field(default_factory=list)


@dataclass
class RuntimeExecutionStats:
    target: str
    device: str
    mode: str
    backend_linear_ops_executed: int = 0
    backend_elementwise_ops_executed: int = 0
    pytorch_fallback_ops: int = 0
    adapter_ops: int = 0
    fallback_ops: List[str] = field(default_factory=list)
    op_traces: List[RuntimeOpTrace] = field(default_factory=list)
    max_abs_error_vs_pytorch: Optional[float] = None
    compile_time_ms: float = 0.0
    setup_time_ms: float = 0.0
    h2d_time_ms: float = 0.0
    kernel_time_ms: float = 0.0
    d2h_time_ms: float = 0.0
    h2d_count: int = 0
    d2h_count: int = 0
    adapter_time_ms: float = 0.0
    wall_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "device": self.device,
            "mode": self.mode,
            "backend_linear_ops_executed": self.backend_linear_ops_executed,
            "backend_elementwise_ops_executed": self.backend_elementwise_ops_executed,
            "pytorch_fallback_ops": self.pytorch_fallback_ops,
            "adapter_ops": self.adapter_ops,
            "fallback_ops": list(self.fallback_ops),
            "max_abs_error_vs_pytorch": self.max_abs_error_vs_pytorch,
            "compile_time_ms": self.compile_time_ms,
            "setup_time_ms": self.setup_time_ms,
            "h2d_time_ms": self.h2d_time_ms,
            "kernel_time_ms": self.kernel_time_ms,
            "d2h_time_ms": self.d2h_time_ms,
            "h2d_count": self.h2d_count,
            "d2h_count": self.d2h_count,
            "adapter_time_ms": self.adapter_time_ms,
            "wall_time_ms": self.wall_time_ms,
            "op_traces": [
                {
                    "graph_op": trace.graph_op,
                    "op": trace.op,
                    "engine": trace.engine,
                    "latency_ms": trace.latency_ms,
                    "notes": list(trace.notes),
                }
                for trace in self.op_traces
            ],
        }


def _as_int4_array(data: Any) -> np.ndarray:
    return np.clip(np.rint(np.asarray(data)), -8, 7).astype(np.int8)


def _as_float_array(data: Any) -> np.ndarray:
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy().astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def _quantize_int4(data: Any) -> np.ndarray:
    return np.clip(np.rint(np.asarray(data, dtype=np.float32)), -8, 7).astype(np.int8)


def _quantized_linear_output(x_int4: np.ndarray, w_int4: np.ndarray) -> np.ndarray:
    accum = w_int4.astype(np.int32) @ x_int4.astype(np.int32).reshape(-1)
    return np.clip(accum, -8, 7).astype(np.int8)


class CompiledMLPRuntime:
    def __init__(
        self,
        graph: GraphIR,
        runtime_plan: GraphRuntimePlan,
        target: str,
        reference_model: Optional[Any] = None,
        array_size: int = 16,
        use_tuned_schedule: bool = False,
        autotune_cache_path: str = DEFAULT_CACHE_PATH,
    ):
        self.graph = graph
        self.runtime_plan = runtime_plan
        self.target = (target or "cuda").strip().lower()
        self.reference_model = reference_model
        self.array_size = int(array_size)
        self.use_tuned_schedule = bool(use_tuned_schedule)
        self.autotune_cache_path = autotune_cache_path or DEFAULT_CACHE_PATH
        self.last_stats: Optional[RuntimeExecutionStats] = None
        self._torch = self._require_torch()
        self.device = self._resolve_device()
        self.params = self._materialize_parameters()
        self.cuda_executor = CUDABlockedFCExecutor(verbose=False) if self.target == "cuda" else None
        self.graph_op_executor = CUDAGraphOpExecutor(device=str(self.device)) if self.target == "cuda" else None

    def _require_torch(self):
        try:
            import torch
        except Exception as e:
            raise CompiledRuntimeError("PyTorch is required to execute compiled MLP runtime") from e
        return torch

    def _resolve_device(self):
        torch = self._torch
        if self.target == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.target == "utpu":
            return torch.device("cpu")
        raise CompiledRuntimeError(f"Unknown runtime target '{self.target}'. Expected 'cuda' or 'utpu'.")

    def _materialize_parameters(self) -> Dict[str, np.ndarray]:
        params: Dict[str, np.ndarray] = {}
        for op in self.graph.ops:
            if "weight" in op.attrs:
                params[f"{op.name}.weight"] = np.asarray(op.attrs["weight"], dtype=np.float32)
            if op.attrs.get("bias") is not None:
                params[f"{op.name}.bias"] = np.asarray(op.attrs["bias"], dtype=np.float32)
        return params

    def _new_stats(self, mode: str) -> RuntimeExecutionStats:
        return RuntimeExecutionStats(
            target=self.target,
            device=str(self.device),
            mode=mode,
        )

    def _normalize_args(self, args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = tuple(args[0])
        if len(args) != len(self.graph.inputs):
            raise CompiledRuntimeError(
                f"Expected {len(self.graph.inputs)} input tensors, got {len(args)}"
            )
        return args

    def _to_torch_output(self, value: np.ndarray, like: Optional[Any]):
        torch = self._torch
        device = like.device if like is not None and hasattr(like, "device") else self.device
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def execute_linear_cuda(self, op: RuntimeOpPlan, x: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
        if self.cuda_executor is None:
            raise CompiledRuntimeError("CUDA backend executor is not available for this runtime")
        if op.weight_buffer is None:
            raise CompiledRuntimeError(f"Linear op '{op.graph_op}' has no weight buffer")

        w = _as_int4_array(self.params[op.weight_buffer])
        x_int4 = _as_int4_array(x).reshape(-1)
        request = BlockedFCLoweringRequest(
            weights_int4=w,
            activations_int4=x_int4,
            out_features=int(w.shape[0]),
            in_features=int(w.shape[1]),
            array_size=self.array_size,
            apply_relu=False,
            apply_quant=True,
            weight_addr=0x080,
            input_addr=0x000,
            result_addr=0x100,
        )
        schedule_params = None
        if self.use_tuned_schedule:
            schedule_params = lookup_best_schedule(
                out_features=int(w.shape[0]),
                in_features=int(w.shape[1]),
                array_size=self.array_size,
                path=self.autotune_cache_path,
            )
        result = self.cuda_executor.execute(request, schedule_params=schedule_params)
        if not result.get("executed", False):
            raise CompiledRuntimeError(
                f"CUDA backend execution failed for '{op.graph_op}': {result.get('reason', 'unknown reason')}"
            )
        return np.asarray(result["output_unpadded"], dtype=np.float32).reshape(1, -1), result

    def execute_elementwise_cuda(
        self,
        op: RuntimeOpPlan,
        y: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if self.cuda_executor is None:
            raise CompiledRuntimeError("CUDA backend executor is not available for this runtime")
        bias = None
        if op.bias_buffer is not None:
            bias = _as_int4_array(self.params[op.bias_buffer])
        result = self.cuda_executor.execute_elementwise_int4(
            values_int4=_as_int4_array(y),
            bias_int4=bias,
            apply_relu=op.apply_relu,
        )
        if not result.get("executed", False):
            raise CompiledRuntimeError(
                f"CUDA elementwise execution failed for '{op.graph_op}': {result.get('reason', 'unknown reason')}"
            )
        return np.asarray(result["output"], dtype=np.float32).reshape(1, -1), result

    def quantized_reference(self, *args) -> np.ndarray:
        args = self._normalize_args(args)
        buffers: Dict[str, np.ndarray] = {}
        for name, value in zip(self.graph.inputs, args):
            buffers[name] = _quantize_int4(_as_float_array(value)).reshape(-1)

        for op in self.runtime_plan.ops:
            if op.op != "linear":
                raise CompiledRuntimeError(f"Runtime op '{op.op}' is not supported by quantized reference")
            w = _quantize_int4(self.params[op.weight_buffer])
            y = _quantized_linear_output(buffers[op.inputs[0]], w)
            if op.bias_buffer is not None:
                y = np.clip(y.astype(np.int32) + _quantize_int4(self.params[op.bias_buffer]).astype(np.int32), -8, 7).astype(np.int8)
            if op.apply_relu:
                y = np.where(y >= 0, y, 0).astype(np.int8)
            buffers[op.output] = y.reshape(-1)

        outputs = [buffers[name] for name in self.graph.outputs]
        out = outputs[0] if len(outputs) == 1 else tuple(outputs)
        return out.reshape(1, -1) if not isinstance(out, tuple) else tuple(v.reshape(1, -1) for v in out)

    def _schedule_params_for_weight(self, w: np.ndarray) -> Optional[Dict[str, int]]:
        if not self.use_tuned_schedule:
            return None
        return lookup_best_schedule(
            out_features=int(w.shape[0]),
            in_features=int(w.shape[1]),
            array_size=self.array_size,
            path=self.autotune_cache_path,
        )

    def _execute_compiled_resident(self, args) -> Any:
        call_t0 = time.perf_counter()
        if self.cuda_executor is None:
            raise CompiledRuntimeError("CUDA backend executor is not available for this runtime")

        input_name = self.graph.inputs[0]
        input_int4 = _as_int4_array(args[0])
        graph_ops = []
        for op in self.runtime_plan.ops:
            if op.op != "linear":
                raise CompiledRuntimeError(f"Runtime op '{op.op}' is not executable in compiled mode")
            w = _as_int4_array(self.params[op.weight_buffer])
            bias = _as_int4_array(self.params[op.bias_buffer]) if op.bias_buffer is not None else None
            graph_ops.append(
                {
                    "name": op.graph_op,
                    "weights_int4": w,
                    "bias_int4": bias,
                    "apply_relu": bool(op.apply_relu),
                    "schedule_params": self._schedule_params_for_weight(w),
                }
            )

        result = self.cuda_executor.execute_graph_resident_int4(
            ops=graph_ops,
            input_int4=input_int4,
            array_size=self.array_size,
        )
        if not result.get("executed", False):
            raise CompiledRuntimeError(
                f"CUDA resident graph execution failed: {result.get('reason', 'unknown reason')}"
            )

        stats = self._new_stats(mode="compiled")
        stats.backend_linear_ops_executed = int(result.get("backend_linear_ops_executed", 0))
        stats.backend_elementwise_ops_executed = int(result.get("backend_elementwise_ops_executed", 0))
        stats.compile_time_ms = float(result.get("compile_time_ms", 0.0) or 0.0)
        stats.setup_time_ms = float(result.get("setup_time_ms", 0.0) or 0.0)
        stats.h2d_time_ms = float(result.get("h2d_time_ms", 0.0) or 0.0)
        stats.kernel_time_ms = float(result.get("kernel_time_ms", 0.0) or 0.0)
        stats.d2h_time_ms = float(result.get("d2h_time_ms", 0.0) or 0.0)
        stats.h2d_count = int(result.get("h2d_count", 0) or 0)
        stats.d2h_count = int(result.get("d2h_count", 0) or 0)
        for op_result in result.get("op_results", []):
            op_name = str(op_result.get("name"))
            is_elementwise = op_result.get("op") == "bias_relu"
            engine = "nvrtc_cuda_elementwise"
            if not is_elementwise:
                engine = "nvrtc_cuda_blocked_fc_tuned" if self.use_tuned_schedule else "nvrtc_cuda_blocked_fc"
            stats.op_traces.append(
                RuntimeOpTrace(
                    graph_op=op_name,
                    op=str(op_result.get("op")),
                    engine=engine,
                    latency_ms=float(op_result.get("kernel_time_ms", 0.0)),
                    notes=[
                        "resident_graph_execution",
                        f"kernel_time_ms={op_result.get('kernel_time_ms')}",
                        f"compile_time_ms={op_result.get('compile_time_ms')}",
                        f"setup_time_ms={op_result.get('setup_time_ms')}",
                        f"kernel_cache_hit={op_result.get('kernel_cache_hit')}",
                        f"schedule_params={op_result.get('schedule_params')}",
                    ],
                )
            )

        stats.wall_time_ms = (time.perf_counter() - call_t0) * 1000.0
        self.last_stats = stats
        out = np.asarray(result["output_unpadded"], dtype=np.float32).reshape(1, -1)
        return self._to_torch_output(out, args[0])

    def _execute_compiled(self, args) -> Any:
        if self.target != "cuda":
            raise CompiledRuntimeError(
                "Callable compiled execution is currently implemented for target='cuda'. "
                "uTPU compilation emits instruction plans but does not execute without board runtime."
            )
        if not self.runtime_plan.executable:
            raise CompiledRuntimeError(
                "Compiled graph is not executable: " + "; ".join(self.runtime_plan.unsupported_ops)
            )

        if self._can_use_resident_linear_path():
            return self._execute_compiled_resident(args)

        return self._execute_compiled_graph_ops(args)

    def _can_use_resident_linear_path(self) -> bool:
        if len(self.runtime_plan.ops) == 0 or len(self.graph.inputs) != 1:
            return False
        return all(op.op == OpKind.LINEAR for op in self.runtime_plan.ops)

    def _execute_compiled_graph_ops(self, args) -> Any:
        call_t0 = time.perf_counter()
        if self.graph_op_executor is None:
            raise CompiledRuntimeError("CUDA graph-op executor is not available for this runtime")
        self._validate_graph_ops(args)
        result = self.graph_op_executor.run(self.graph, *args)
        if not result.get("executed", False):
            raise CompiledRuntimeError(
                f"CUDA graph-op execution failed: {result.get('reason', 'unknown reason')}"
            )

        stats = self._new_stats(mode="compiled")
        for op in self.runtime_plan.ops:
            engine = "cuda_graph_ops"
            if op.op == OpKind.LINEAR:
                stats.backend_linear_ops_executed += 1
            else:
                stats.backend_elementwise_ops_executed += 1
            stats.op_traces.append(
                RuntimeOpTrace(
                    graph_op=op.graph_op,
                    op=op.op,
                    engine=engine,
                    latency_ms=0.0,
                    notes=["graph-op execution path"],
                )
            )

        stats.wall_time_ms = (time.perf_counter() - call_t0) * 1000.0
        self.last_stats = stats
        outputs = result["outputs"]
        if isinstance(outputs, list):
            return tuple(self._to_torch_output(np.asarray(v, dtype=np.float32), args[0]) for v in outputs)
        return self._to_torch_output(np.asarray(outputs, dtype=np.float32), args[0])

    def _shape_for_runtime_value(self, value: Any):
        if hasattr(value, "shape"):
            return tuple(int(d) for d in value.shape)
        arr = np.asarray(value)
        return tuple(int(d) for d in arr.shape)

    def _validate_graph_ops(self, args) -> None:
        shapes: Dict[str, tuple[int, ...]] = {}
        op_by_name = {op.name: op for op in self.graph.ops}
        for name, value in zip(self.graph.inputs, args):
            shapes[name] = self._shape_for_runtime_value(value)

        for op in self.runtime_plan.ops:
            in_shape = shapes.get(op.inputs[0])
            out_shape = self.graph.values.get(op.output).shape if op.output in self.graph.values else None

            if op.op == OpKind.VIEW:
                if in_shape is not None and out_shape is not None:
                    in_elems = int(np.prod(in_shape))
                    out_elems = int(np.prod(out_shape))
                    if in_elems != out_elems:
                        raise CompiledRuntimeError(
                            f"Invalid view '{op.graph_op}': input elements {in_elems} != output elements {out_elems}"
                        )
            elif op.op == OpKind.PERMUTE:
                if in_shape is not None:
                    rank = len(in_shape)
                    graph_op = op_by_name.get(op.graph_op)
                    axes = tuple(graph_op.attrs.get("args", ())) if graph_op is not None else ()
                    if len(axes) != rank:
                        raise CompiledRuntimeError(
                            f"Invalid permute '{op.graph_op}': axes rank {len(axes)} != input rank {rank}"
                        )
                    if sorted(int(a) for a in axes) != list(range(rank)):
                        raise CompiledRuntimeError(
                            f"Invalid permute '{op.graph_op}': axes must be a permutation of 0..{rank - 1}"
                        )
            elif op.op == OpKind.LAYER_NORM:
                if in_shape is not None and len(in_shape) < 2:
                    raise CompiledRuntimeError(
                        f"Invalid layer_norm '{op.graph_op}': expected rank >= 2, got shape {in_shape}"
                    )
            elif op.op == OpKind.SCALED_DOT_PRODUCT_ATTENTION:
                if len(op.inputs) < 3:
                    raise CompiledRuntimeError(
                        f"Invalid attention '{op.graph_op}': expected q/k/v inputs"
                    )
                q_shape = shapes.get(op.inputs[0])
                k_shape = shapes.get(op.inputs[1])
                v_shape = shapes.get(op.inputs[2])
                if q_shape is not None and k_shape is not None and v_shape is not None:
                    if not (len(q_shape) == len(k_shape) == len(v_shape) == 4):
                        raise CompiledRuntimeError(
                            f"Invalid attention '{op.graph_op}': expected rank-4 q/k/v, "
                            f"got q={q_shape}, k={k_shape}, v={v_shape}"
                        )
                    if q_shape[0] != k_shape[0] or q_shape[0] != v_shape[0]:
                        raise CompiledRuntimeError(
                            f"Invalid attention '{op.graph_op}': batch mismatch q={q_shape}, k={k_shape}, v={v_shape}"
                        )
                    if q_shape[1] != k_shape[1] or q_shape[1] != v_shape[1]:
                        raise CompiledRuntimeError(
                            f"Invalid attention '{op.graph_op}': head mismatch q={q_shape}, k={k_shape}, v={v_shape}"
                        )
                    if q_shape[-1] != k_shape[-1]:
                        raise CompiledRuntimeError(
                            f"Invalid attention '{op.graph_op}': q/k head dim mismatch q={q_shape}, k={k_shape}"
                        )
                    if k_shape[-2] != v_shape[-2]:
                        raise CompiledRuntimeError(
                            f"Invalid attention '{op.graph_op}': k/v seq mismatch k={k_shape}, v={v_shape}"
                        )
                if len(op.inputs) > 3:
                    mask_shape = shapes.get(op.inputs[3])
                    if mask_shape is not None and len(mask_shape) not in {2, 3, 4}:
                        raise CompiledRuntimeError(
                            f"Invalid attention '{op.graph_op}': mask rank must be 2/3/4, got {mask_shape}"
                        )
                    if (
                        mask_shape is not None
                        and q_shape is not None
                        and k_shape is not None
                        and len(mask_shape) == 4
                    ):
                        if mask_shape[-2] != q_shape[-2] or mask_shape[-1] != k_shape[-2]:
                            raise CompiledRuntimeError(
                                f"Invalid attention '{op.graph_op}': mask shape {mask_shape} "
                                f"must end with (q_seq={q_shape[-2]}, k_seq={k_shape[-2]})"
                            )

            if out_shape is not None:
                shapes[op.output] = tuple(int(d) for d in out_shape)

    def _execute_reference(self, args) -> Any:
        torch = self._torch
        call_t0 = time.perf_counter()
        stats = self._new_stats(mode="reference")
        buffers: Dict[str, Any] = {}
        for name, value in zip(self.graph.inputs, args):
            buffers[name] = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        for op in self.runtime_plan.ops:
            t0 = time.perf_counter()
            x = buffers[op.inputs[0]]
            w = torch.as_tensor(self.params[op.weight_buffer], dtype=torch.float32, device=self.device)
            y = x.matmul(w.t())
            if op.bias_buffer is not None:
                b = torch.as_tensor(self.params[op.bias_buffer], dtype=torch.float32, device=self.device)
                y = y + b
            if op.apply_relu:
                y = torch.relu(y)
            t1 = time.perf_counter()
            buffers[op.output] = y
            stats.pytorch_fallback_ops += 1
            stats.fallback_ops.append(op.graph_op)
            stats.op_traces.append(
                RuntimeOpTrace(
                    graph_op=op.graph_op,
                    op=op.op,
                    engine="pytorch_reference",
                    latency_ms=(t1 - t0) * 1000.0,
                    notes=["explicit reference mode"],
                )
            )

        outputs = [buffers[name] for name in self.graph.outputs]
        stats.wall_time_ms = (time.perf_counter() - call_t0) * 1000.0
        self.last_stats = stats
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def __call__(self, *args, mode: str = "compiled"):
        args = self._normalize_args(args)
        mode_name = (mode or "compiled").strip().lower()
        if mode_name == "compiled":
            return self._execute_compiled(args)
        if mode_name == "reference":
            return self._execute_reference(args)
        if mode_name == "fallback":
            return self._execute_reference(args)
        raise CompiledRuntimeError("mode must be one of: compiled, reference, fallback")

    def compare_with_pytorch(self, *args) -> Dict[str, Any]:
        if self.reference_model is None:
            raise CompiledRuntimeError("No reference PyTorch model is attached to this compiled runtime")
        torch = self._torch
        with torch.no_grad():
            compiled_out = self(*args, mode="compiled")
            ref_out = self.reference_model(*args)
        max_abs = float(torch.max(torch.abs(compiled_out.detach().cpu() - ref_out.detach().cpu())).item())
        if self.last_stats is not None:
            self.last_stats.max_abs_error_vs_pytorch = max_abs
        return {
            "max_abs_error": max_abs,
            "compiled_output": compiled_out,
            "pytorch_output": ref_out,
        }

    def execution_report(self) -> Dict[str, Any]:
        if self.last_stats is None:
            return self._new_stats(mode="not_run").to_dict()
        return self.last_stats.to_dict()

    def benchmark(self, *args, warmup: int = 5, iters: int = 50) -> Dict[str, Any]:
        args = self._normalize_args(args)
        if warmup < 0 or iters <= 0:
            raise CompiledRuntimeError("benchmark requires warmup >= 0 and iters > 0")

        first_out = self(*args, mode="compiled")
        first_report = self.execution_report()

        for _ in range(warmup):
            self(*args, mode="compiled")

        reports = []
        for _ in range(iters):
            self(*args, mode="compiled")
            reports.append(self.execution_report())

        def avg(field: str) -> float:
            return float(sum(r[field] for r in reports) / len(reports))

        max_abs = None
        if self.reference_model is not None:
            torch = self._torch
            with torch.no_grad():
                ref = self.reference_model(*args)
            max_abs = float(torch.max(torch.abs(first_out.detach().cpu() - ref.detach().cpu())).item())

        steady_state_wall_ms = avg("wall_time_ms")
        return {
            "target": self.target,
            "device": str(self.device),
            "warmup": int(warmup),
            "iters": int(iters),
            "first_call_wall_ms": float(first_report["wall_time_ms"]),
            "steady_state_wall_ms": steady_state_wall_ms,
            "compile_time_ms": float(first_report["compile_time_ms"]),
            "setup_time_ms": float(first_report["setup_time_ms"]),
            "h2d_time_ms": avg("h2d_time_ms"),
            "kernel_time_ms": avg("kernel_time_ms"),
            "d2h_time_ms": avg("d2h_time_ms"),
            "h2d_count": int(round(avg("h2d_count"))),
            "d2h_count": int(round(avg("d2h_count"))),
            "adapter_time_ms": avg("adapter_time_ms"),
            "backend_linear_ops_executed": int(reports[-1]["backend_linear_ops_executed"]),
            "backend_elementwise_ops_executed": int(reports[-1]["backend_elementwise_ops_executed"]),
            "adapter_ops": int(reports[-1]["adapter_ops"]),
            "fallback_ops": list(reports[-1]["fallback_ops"]),
            "max_abs_error_vs_pytorch": max_abs,
            "last_op_traces": reports[-1]["op_traces"],
        }
