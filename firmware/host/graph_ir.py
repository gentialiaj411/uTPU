from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class OpKind:
    LINEAR = "linear"
    RELU = "relu"
    ADD = "add"
    VIEW = "view"


@dataclass
class TensorValue:
    name: str
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    producer: Optional[str] = None
    consumers: List[str] = field(default_factory=list)


@dataclass
class OpNode:
    name: str
    op: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphIR:
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    values: Dict[str, TensorValue] = field(default_factory=dict)
    ops: List[OpNode] = field(default_factory=list)

    def add_value(
        self,
        name: str,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[str] = None,
        producer: Optional[str] = None,
    ) -> TensorValue:
        existing = self.values.get(name)
        if existing is not None:
            if shape is not None:
                existing.shape = shape
            if dtype is not None:
                existing.dtype = dtype
            if producer is not None:
                existing.producer = producer
            return existing

        value = TensorValue(name=name, shape=shape, dtype=dtype, producer=producer)
        self.values[name] = value
        return value

    def add_op(self, op: OpNode) -> None:
        self.ops.append(op)
        for output in op.outputs:
            self.add_value(output, producer=op.name)
        for input_name in op.inputs:
            value = self.add_value(input_name)
            if op.name not in value.consumers:
                value.consumers.append(op.name)

    def get_value(self, name: str) -> TensorValue:
        try:
            return self.values[name]
        except KeyError as e:
            raise KeyError(f"Unknown graph value '{name}'") from e


GraphModuleIR = GraphIR
