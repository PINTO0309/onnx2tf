from __future__ import annotations

from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    preflight_any_operator,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR


def _optimize_boundary_input_layout_transposes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Elide synthetic input-boundary layout adapters inserted as:
      input --TRANSPOSE--> input_onnx_ncx_internal

    Safety:
    - Matches only synthetic wrapper outputs named '*_onnx_ncx_internal'.
    - Input tensor must be a model input and not a model output.
    - Never mutates model-input metadata (public I/O contract).
    - Rewrites only when input/internal tensor metadata already match exactly.
    """
    removed = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)
        consumers = graph_index.consumers
        changed = False

        for op_idx, op in enumerate(model_ir.operators):
            if str(op.op_type) != "TRANSPOSE":
                continue
            if len(op.inputs) < 2 or len(op.outputs) != 1:
                continue

            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            if input_name not in model_inputs:
                continue
            if input_name in model_outputs or output_name in model_outputs:
                continue
            if not str(output_name).endswith("_onnx_ncx_internal"):
                continue
            # Ensure this is the boundary adapter form [N,C,*] internalization.
            perm = _read_transpose_perm(model_ir, op)
            if perm is None:
                continue
            rank = len(perm)
            if rank == 3 and perm != [0, 2, 1]:
                continue
            if rank == 4 and perm != [0, 3, 1, 2]:
                continue
            if rank == 5 and perm != [0, 4, 1, 2, 3]:
                continue
            if rank not in {3, 4, 5}:
                continue

            internal_tensor = model_ir.tensors.get(output_name, None)
            input_tensor = model_ir.tensors.get(input_name, None)
            if internal_tensor is None or input_tensor is None:
                continue

            input_shape = [int(v) for v in list(input_tensor.shape)]
            internal_shape = [int(v) for v in list(internal_tensor.shape)]
            if input_shape != internal_shape:
                continue
            input_signature = (
                [int(v) for v in list(input_tensor.shape_signature)]
                if input_tensor.shape_signature is not None
                else input_shape
            )
            internal_signature = (
                [int(v) for v in list(internal_tensor.shape_signature)]
                if internal_tensor.shape_signature is not None
                else internal_shape
            )
            if input_signature != internal_signature:
                continue
            if str(input_tensor.dtype) != str(internal_tensor.dtype):
                continue
            if input_tensor.quantization != internal_tensor.quantization:
                continue
            consumer_op_types = {
                str(model_ir.operators[int(consumer_idx)].op_type)
                for consumer_idx in consumers.get(str(output_name), [])
                if int(consumer_idx) != int(op_idx)
            }
            if len(consumer_op_types & {"GATHER", "GATHER_ND", "SLICE", "STRIDED_SLICE"}) > 0:
                continue

            # Redirect all internal users to model input without changing
            # external input tensor metadata.
            _replace_tensor_inputs(
                model_ir,
                output_name,
                input_name,
                graph_index=graph_index,
            )

            graph_index.remove_operator(op_idx)
            removed += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"removed_boundary_input_layout_transpose": int(removed)}


def run_boundary_input_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run guarded boundary-adapter removal as an ordered layout pass."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        model_inputs = set(str(name) for name in candidate_model.inputs)
        return preflight_any_operator(
            candidate_model,
            lambda op: (
                str(op.op_type) == "TRANSPOSE"
                and len(op.inputs) >= 2
                and len(op.outputs) == 1
                and str(op.inputs[0]) in model_inputs
                and str(op.outputs[0]).endswith("_onnx_ncx_internal")
            ),
        )

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        return _preflight(pass_state.model_ir)

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_boundary_input_layout_transposes(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("removed_boundary_input_layout_transpose", 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.boundary_input_adapter",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"removed_boundary_input_layout_transpose": 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
