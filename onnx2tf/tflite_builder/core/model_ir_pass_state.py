from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

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
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


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
    _prepared_pass_data: Dict[str, Any] = field(
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

    def set_prepared_pass_data(self, key: str, value: Any) -> None:
        """Store data prepared for the next callback in this pass session."""

        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("prepared pass data key must not be empty")
        self._prepared_pass_data[normalized_key] = value

    def take_prepared_pass_data(self, key: str) -> Any:
        """Consume callback data prepared in this pass session, if present."""

        return self._prepared_pass_data.pop(str(key), None)

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
        self._prepared_pass_data.clear()
        self._freeze_constant_buffers(target)
        self.graph_index.refresh()
        assert self.layout_state is not None
        self.layout_state.sync_from_model_ir(target)

    @classmethod
    def create_ordered_manager(cls) -> OrderedPassManager["ModelIRPassState"]:
        return OrderedPassManager[ModelIRPassState](
            fingerprint=lambda state: state.fingerprint(),
            validator=lambda state: state.validate(),
            clone=lambda state: state.snapshot(),
            restore=lambda state, snapshot: state.restore(snapshot),
        )


@dataclass(frozen=True)
class ModelIRPreflightResult:
    """Cheap model-only candidate scan result used before pass state creation."""

    matched: bool
    operators_visited: int


@dataclass
class ModelIRPassStateScope:
    """Lazily share pass state across adjacent groups with no raw mutation."""

    model_ir: ModelIR
    layout_state: Optional[LayoutState] = None
    _state: Optional[ModelIRPassState] = field(
        init=False,
        default=None,
        repr=False,
    )

    def acquire(
        self,
        *,
        model_ir: ModelIR,
        layout_state: Optional[LayoutState],
    ) -> Tuple[ModelIRPassState, bool]:
        if model_ir is not self.model_ir:
            raise ValueError("ModelIRPassStateScope cannot cross ModelIR instances")
        if layout_state is not self.layout_state:
            raise ValueError("ModelIRPassStateScope cannot cross LayoutState instances")
        if self._state is None:
            self._state = ModelIRPassState(
                self.model_ir,
                layout_state=self.layout_state,
            )
            return self._state, True
        return self._state, False


def preflight_any_operator(
    model_ir: ModelIR,
    predicate: Callable[[OperatorIR], bool],
) -> ModelIRPreflightResult:
    for visited, operator in enumerate(model_ir.operators, start=1):
        if predicate(operator):
            return ModelIRPreflightResult(True, visited)
    return ModelIRPreflightResult(False, len(model_ir.operators))


def preflight_required_op_types(
    model_ir: ModelIR,
    required_op_types: Iterable[str],
) -> ModelIRPreflightResult:
    missing = {str(op_type) for op_type in required_op_types}
    if len(missing) == 0:
        return ModelIRPreflightResult(True, 0)
    for visited, operator in enumerate(model_ir.operators, start=1):
        missing.discard(str(operator.op_type))
        if len(missing) == 0:
            return ModelIRPreflightResult(True, visited)
    return ModelIRPreflightResult(False, len(model_ir.operators))


def summarize_model_ir_pass_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    events = [
        dict(event)
        for event in diagnostics
        if str(event.get("stage", "")) == "model_ir_pass"
    ]
    status_counts: Dict[str, int] = {}
    totals = {
        "preflight_operators_visited": 0,
        "state_build_count": 0,
        "snapshot_count": 0,
        "fingerprint_count": 0,
    }
    by_pass: Dict[str, Dict[str, Any]] = {}
    by_group: Dict[int, Dict[str, Any]] = {}
    for event in events:
        status = str(event.get("status", "unknown"))
        status_counts[status] = int(status_counts.get(status, 0)) + 1
        metrics = event.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        visited = int(metrics.get("preflight_operators_visited", 0))
        state_built = bool(metrics.get("state_built", False))
        snapshots = int(metrics.get("snapshot_count", 0))
        fingerprints = int(metrics.get("fingerprint_count", 0))
        totals["snapshot_count"] += snapshots
        totals["fingerprint_count"] += fingerprints

        group_sequence = int(event.get("group_sequence", 0))
        group_summary = by_group.get(group_sequence)
        if group_summary is None:
            group_summary = {
                "pass_ids": [],
                "preflight_operators_visited": visited,
                "state_built": state_built,
            }
            by_group[group_sequence] = group_summary
            totals["preflight_operators_visited"] += visited
            totals["state_build_count"] += int(state_built)

        code = str(event.get("code", ""))
        group_summary["pass_ids"].append(code)
        pass_summary = by_pass.setdefault(
            code,
            {
                "event_count": 0,
                "changed_count": 0,
                "skipped_count": 0,
                "snapshot_count": 0,
                "fingerprint_count": 0,
            },
        )
        pass_summary["event_count"] += 1
        pass_summary["changed_count"] += int(bool(event.get("changed", False)))
        pass_summary["skipped_count"] += int(
            bool(event.get("skipped_by_precondition", False))
        )
        pass_summary["snapshot_count"] += snapshots
        pass_summary["fingerprint_count"] += fingerprints

    return {
        "schema_version": 2,
        "event_count": len(events),
        "status_counts": dict(sorted(status_counts.items())),
        "totals": totals,
        "groups": {
            str(group_sequence): by_group[group_sequence]
            for group_sequence in sorted(by_group)
        },
        "by_pass": {code: by_pass[code] for code in sorted(by_pass)},
    }


def run_model_ir_pass_group(
    model_ir: ModelIR,
    *,
    specs: Iterable[PassSpec[ModelIRPassState]],
    layout_state: Optional[LayoutState] = None,
    default_details: Optional[Mapping[str, Any]] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
    preflight: Optional[
        Callable[[ModelIR], bool | ModelIRPreflightResult]
    ] = None,
) -> Tuple[Dict[str, Any], List[PassResult]]:
    """Run ordered ModelIR specs with shared state and normalized diagnostics."""

    preflight_operators_visited = 0
    state_built = False
    diagnostic_sequence = 0
    diagnostic_invocations: Dict[str, int] = {}
    maximum_group_sequence: Optional[int] = None
    if diagnostics is not None:
        for event in diagnostics:
            if str(event.get("stage", "")) != "model_ir_pass":
                continue
            diagnostic_sequence += 1
            code = str(event.get("code", ""))
            diagnostic_invocations[code] = int(
                diagnostic_invocations.get(code, 0) + 1
            )
            event_group_sequence = int(event.get("group_sequence", 0))
            if (
                maximum_group_sequence is None
                or event_group_sequence > maximum_group_sequence
            ):
                maximum_group_sequence = event_group_sequence
    group_sequence = int(
        1 + (maximum_group_sequence if maximum_group_sequence is not None else 0)
    )

    def _record_event(
        event: Dict[str, Any],
        *,
        snapshot_count: int = 0,
        fingerprint_count: int = 0,
    ) -> None:
        nonlocal diagnostic_sequence
        if diagnostics is None:
            return
        code = str(event.get("code", ""))
        diagnostic_sequence += 1
        invocation = int(diagnostic_invocations.get(code, 0) + 1)
        diagnostic_invocations[code] = invocation
        diagnostics.append(
            {
                **event,
                "sequence": diagnostic_sequence,
                "invocation": invocation,
                "group_sequence": group_sequence,
                "metrics": {
                    "preflight_operators_visited": preflight_operators_visited,
                    "state_built": state_built,
                    "snapshot_count": int(snapshot_count),
                    "fingerprint_count": int(fingerprint_count),
                },
            }
        )

    manager = ModelIRPassState.create_ordered_manager()
    for spec in list(specs):
        manager.register(spec)
    preflight_matched = True
    if preflight is not None:
        raw_preflight = preflight(model_ir)
        if isinstance(raw_preflight, ModelIRPreflightResult):
            preflight_matched = bool(raw_preflight.matched)
            preflight_operators_visited = int(raw_preflight.operators_visited)
        else:
            preflight_matched = bool(raw_preflight)
    if not preflight_matched:
        results = [
            PassResult(
                pass_id=spec.pass_id,
                phase=spec.phase.name.lower(),
                iterations=0,
                changed=False,
                stopped_by_cycle=False,
                details={"skipped_by_precondition": True},
            )
            for spec in manager.ordered_specs()
        ]
    else:
        if state_scope is None:
            state = ModelIRPassState(model_ir, layout_state=layout_state)
            state_built = True
        else:
            state, state_built = state_scope.acquire(
                model_ir=model_ir,
                layout_state=layout_state,
            )
        try:
            results = manager.run(state)
        except PassInvariantError as error:
            _record_event(
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
                },
                snapshot_count=error.snapshot_count,
                fingerprint_count=error.fingerprint_count,
            )
            raise
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
        _record_event(
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
            },
            snapshot_count=result.snapshot_count,
            fingerprint_count=result.fingerprint_count,
        )
    details: Dict[str, Any] = dict(default_details or {})
    for result in results:
        for key, value in result.details.items():
            if key not in {"changed", "skipped_by_precondition"}:
                details[str(key)] = value
    return details, results
