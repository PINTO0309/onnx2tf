from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

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

    def __post_init__(self) -> None:
        self.graph_index = ModelIRGraphIndex(self.model_ir)
        if self.layout_state is None:
            self.layout_state = LayoutState.from_model_ir(self.model_ir)
        else:
            self.layout_state.sync_from_model_ir(self.model_ir)

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
        self.graph_index.refresh()
        assert self.layout_state is not None
        self.layout_state.sync_from_model_ir(target)

    def create_ordered_manager(self) -> OrderedPassManager["ModelIRPassState"]:
        return OrderedPassManager[ModelIRPassState](
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
