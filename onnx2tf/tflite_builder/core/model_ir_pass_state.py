from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.passes import (
    OrderedPassManager,
    PassInvariantError,
    PassResult,
    PassSpec,
)
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR


@dataclass
class ModelIRPassState:
    """Shared indexed/layout state for ordered ModelIR pass groups."""

    model_ir: ModelIR
    layout_state: Optional[LayoutState] = None
    graph_index: ModelIRGraphIndex = field(init=False)
    _constant_digest_cache: Dict[int, Tuple[Any, str]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.graph_index = ModelIRGraphIndex(self.model_ir)
        if self.layout_state is None:
            self.layout_state = LayoutState.from_model_ir(self.model_ir)
        else:
            self.layout_state.sync_from_model_ir(self.model_ir)

    def _freeze_constant_buffers(self, model_ir: ModelIR) -> None:
        for tensor in model_ir.tensors.values():
            if isinstance(tensor.data, np.ndarray) and tensor.data.flags.writeable:
                tensor.data.flags.writeable = False
        for subgraph in model_ir.subgraphs:
            self._freeze_constant_buffers(subgraph)

    def _constant_digest(self, value: Any) -> str:
        cache_key = id(value)
        cached = self._constant_digest_cache.get(cache_key)
        if cached is not None and cached[0] is value:
            return cached[1]
        if isinstance(value, bytes):
            payload = value
        else:
            array = np.asarray(value)
            if array.dtype.hasobject:
                payload = json.dumps(
                    self._normalize_value(array.tolist()),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            else:
                payload = np.ascontiguousarray(array).tobytes(order="C")
        digest = hashlib.sha256(payload).hexdigest()
        self._constant_digest_cache[cache_key] = (value, digest)
        return digest

    def _normalize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if math.isnan(value):
                return {"float": "nan"}
            if math.isinf(value):
                return {"float": "inf" if value > 0 else "-inf"}
            return {"float": value.hex()}
        if isinstance(value, np.generic):
            return self._normalize_value(value.item())
        if isinstance(value, np.ndarray):
            if value.flags.writeable:
                value.flags.writeable = False
            return {
                "ndarray": {
                    "dtype": value.dtype.str,
                    "shape": [int(dim) for dim in value.shape],
                    "digest": self._constant_digest(value),
                }
            }
        if isinstance(value, bytes):
            return {
                "bytes": {
                    "length": len(value),
                    "digest": self._constant_digest(value),
                }
            }
        if is_dataclass(value) and not isinstance(value, type):
            return self._normalize_value(asdict(value))
        if isinstance(value, Mapping):
            return {
                str(key): self._normalize_value(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [self._normalize_value(item) for item in value]
            return sorted(
                normalized,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            )
        if isinstance(value, np.dtype):
            return {"dtype": value.str}
        raise TypeError(
            "unsupported ModelIR fingerprint value: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

    def _model_payload(self, model_ir: ModelIR) -> Dict[str, Any]:
        tensors = []
        for name in sorted(model_ir.tensors):
            tensor = model_ir.tensors[name]
            data = tensor.data
            tensors.append(
                {
                    "name": str(name),
                    "dtype": str(tensor.dtype),
                    "shape": [int(dim) for dim in tensor.shape],
                    "shape_signature": (
                        None
                        if tensor.shape_signature is None
                        else [int(dim) for dim in tensor.shape_signature]
                    ),
                    "data": None if data is None else self._normalize_value(data),
                    "is_variable": bool(tensor.is_variable),
                    "quantization": self._normalize_value(tensor.quantization),
                    "logical_layout": str(tensor.logical_layout),
                    "physical_layout": str(tensor.physical_layout),
                    "onnx_tensor_name": tensor.onnx_tensor_name,
                }
            )
        operators = [
            {
                "op_type": str(operator.op_type),
                "inputs": [str(name) for name in operator.inputs],
                "outputs": [str(name) for name in operator.outputs],
                "options": self._normalize_value(operator.options),
                "axis_semantics": self._normalize_value(operator.axis_semantics),
                "version": int(operator.version),
                "onnx_node_name": operator.onnx_node_name,
                "onnx_op_type": operator.onnx_op_type,
            }
            for operator in model_ir.operators
        ]
        return {
            "name": str(model_ir.name),
            "description": str(model_ir.description),
            "inputs": [str(name) for name in model_ir.inputs],
            "outputs": [str(name) for name in model_ir.outputs],
            "tensors": tensors,
            "operators": operators,
            "subgraphs": [self._model_payload(subgraph) for subgraph in model_ir.subgraphs],
        }

    def fingerprint(self) -> bytes:
        assert self.layout_state is not None
        self._freeze_constant_buffers(self.model_ir)
        payload = {
            "model_ir": self._model_payload(self.model_ir),
            "layout_state": {
                "logical": dict(sorted(self.layout_state.logical.items())),
                "physical": dict(sorted(self.layout_state.physical.items())),
            },
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    def validate(self) -> List[str]:
        assert self.layout_state is not None
        return validate_model_ir_invariants(
            self.model_ir,
            graph_index=self.graph_index,
        ) + self.layout_state.validate_against_model_ir(self.model_ir)

    def snapshot(self) -> ModelIR:
        return copy.deepcopy(self.model_ir)

    def restore(self, snapshot: ModelIR) -> None:
        target = self.model_ir
        target.name = str(snapshot.name)
        target.description = str(snapshot.description)
        target.tensors = copy.deepcopy(snapshot.tensors)
        target.operators = copy.deepcopy(snapshot.operators)
        target.inputs = list(snapshot.inputs)
        target.outputs = list(snapshot.outputs)
        target.subgraphs = copy.deepcopy(snapshot.subgraphs)
        target.metadata = copy.deepcopy(snapshot.metadata)
        self._freeze_constant_buffers(target)
        self.graph_index.refresh()
        assert self.layout_state is not None
        self.layout_state.sync_from_model_ir(target)

    def create_ordered_manager(self) -> OrderedPassManager["ModelIRPassState"]:
        return OrderedPassManager[ModelIRPassState](
            fingerprint=lambda state: state.fingerprint(),
            validator=lambda state: state.validate(),
            clone=lambda state: state.snapshot(),
            restore=lambda state, snapshot: state.restore(snapshot),
        )


def run_model_ir_pass_group(
    model_ir: ModelIR,
    *,
    specs: Iterable[PassSpec[ModelIRPassState]],
    layout_state: Optional[LayoutState] = None,
    default_details: Optional[Mapping[str, Any]] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[PassResult]]:
    """Run ordered ModelIR specs with shared state and normalized diagnostics."""

    state = ModelIRPassState(model_ir, layout_state=layout_state)
    manager = state.create_ordered_manager()
    for spec in specs:
        manager.register(spec)
    try:
        results = manager.run(state)
    except PassInvariantError as error:
        if diagnostics is not None:
            diagnostics.append(
                {
                    "stage": "model_ir_pass",
                    "code": error.pass_id,
                    "message": "invariant validation failed; transaction rolled back",
                    "phase": error.phase,
                    "status": "failed",
                    "iterations": error.iterations,
                    "changed": False,
                    "stopped_by_cycle": False,
                    "skipped_by_precondition": False,
                    "problems": list(error.problems),
                }
            )
        raise
    if diagnostics is not None:
        for result in results:
            skipped = bool(result.details.get("skipped_by_precondition", False))
            status = (
                "skipped"
                if skipped
                else "cycle_stopped"
                if result.stopped_by_cycle
                else "changed"
                if result.changed
                else "unchanged"
            )
            diagnostics.append(
                {
                    "stage": "model_ir_pass",
                    "code": str(result.pass_id),
                    "message": f"model ir pass {status}",
                    "phase": str(result.phase),
                    "status": status,
                    "iterations": int(result.iterations),
                    "changed": bool(result.changed),
                    "stopped_by_cycle": bool(result.stopped_by_cycle),
                    "skipped_by_precondition": skipped,
                }
            )
    details: Dict[str, Any] = dict(default_details or {})
    for result in results:
        for key, value in result.details.items():
            if key not in {"changed", "skipped_by_precondition"}:
                details[str(key)] = value
    return details, results
