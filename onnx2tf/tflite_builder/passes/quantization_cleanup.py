from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _rename_tensor_globally,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR


def _quantized_tensors_share_exact_grid(
    model_ir: ModelIR,
    lhs_name: str,
    rhs_name: str,
) -> bool:
    """Return whether two quantized tensors use exactly the same grid."""

    lhs = model_ir.tensors.get(str(lhs_name), None)
    rhs = model_ir.tensors.get(str(rhs_name), None)
    if lhs is None or rhs is None:
        return False
    if str(lhs.dtype).upper() != str(rhs.dtype).upper():
        return False
    lhs_q = lhs.quantization
    rhs_q = rhs.quantization
    if lhs_q is None or rhs_q is None:
        return False
    if int(lhs_q.quantized_dimension) != int(rhs_q.quantized_dimension):
        return False
    return bool(
        np.array_equal(
            np.asarray(lhs_q.scale, dtype=np.float64),
            np.asarray(rhs_q.scale, dtype=np.float64),
        )
        and np.array_equal(
            np.asarray(lhs_q.zero_point, dtype=np.int64),
            np.asarray(rhs_q.zero_point, dtype=np.int64),
        )
    )


def _optimize_terminal_quantize_dequantize(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Remove a terminal Q/DQ round trip only across an exact quant grid."""

    removed_pairs = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers

        for q_idx, q_op in enumerate(model_ir.operators):
            if (
                str(q_op.op_type) != "QUANTIZE"
                or len(q_op.inputs) != 1
                or len(q_op.outputs) != 1
            ):
                continue

            float_input_name = str(q_op.inputs[0])
            quantized_name = str(q_op.outputs[0])
            if float_input_name in model_ir.inputs:
                continue

            quantized_users = consumers.get(quantized_name, [])
            if len(quantized_users) != 1:
                continue
            dq_idx = int(quantized_users[0])
            if dq_idx == q_idx:
                continue
            dq_op = model_ir.operators[dq_idx]
            if (
                str(dq_op.op_type) != "DEQUANTIZE"
                or len(dq_op.inputs) != 1
                or len(dq_op.outputs) != 1
                or str(dq_op.inputs[0]) != quantized_name
            ):
                continue

            float_output_name = str(dq_op.outputs[0])
            if float_output_name not in model_ir.outputs:
                continue
            if len(consumers.get(float_output_name, [])) > 0:
                continue

            float_input_users = consumers.get(float_input_name, [])
            if len(float_input_users) != 1 or int(float_input_users[0]) != q_idx:
                continue
            if float_input_name not in producers:
                continue
            if float_input_name in model_ir.outputs:
                continue
            float_producer = model_ir.operators[int(producers[float_input_name])]
            if (
                str(float_producer.op_type) != "DEQUANTIZE"
                or len(float_producer.inputs) != 1
                or len(float_producer.outputs) != 1
                or str(float_producer.outputs[0]) != float_input_name
            ):
                continue
            if not _quantized_tensors_share_exact_grid(
                model_ir,
                str(float_producer.inputs[0]),
                quantized_name,
            ):
                continue

            _rename_tensor_globally(
                model_ir=model_ir,
                old_name=float_input_name,
                new_name=float_output_name,
                layout_state=layout_state,
                graph_index=graph_index,
            )
            for remove_idx in sorted([q_idx, dq_idx], reverse=True):
                graph_index.remove_operator(remove_idx)
            removed_pairs += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {
        "removed_terminal_quantize_dequantize_pairs": int(removed_pairs),
    }


def run_terminal_quantize_dequantize_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run exact-grid terminal Q/DQ cleanup transactionally."""

    def _preflight(candidate_model: ModelIR) -> bool:
        return any(
            str(op.op_type) == "QUANTIZE"
            and len(op.inputs) == 1
            and len(op.outputs) == 1
            for op in candidate_model.operators
        )

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        return _preflight(pass_state.model_ir)

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_terminal_quantize_dequantize(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("removed_terminal_quantize_dequantize_pairs", 0)
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="cleanup.terminal_quantize_dequantize",
                phase=PassPhase.POST_LOWERING_CLEANUP,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"removed_terminal_quantize_dequantize_pairs": 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
