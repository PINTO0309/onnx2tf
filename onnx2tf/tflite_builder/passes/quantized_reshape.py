from __future__ import annotations

from typing import Any, Dict, List

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _clone_quantization,
    _is_same_per_tensor_quantization,
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR

def _optimize_dequant_reshape_quantize_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Fold DEQUANTIZE->RESHAPE->QUANTIZE into quantized RESHAPE.

    Target pattern:
      Xq --DEQUANTIZE--> Xf --RESHAPE(shape)--> Yf --QUANTIZE--> Yq

    Rewritten:
      Xq --RESHAPE(shape)--> Yq

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - input/output quantized tensors use equivalent per-tensor quantization
    - input/output quantized dtypes are identical
    """
    folded = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers

        for dq_idx, dq_op in enumerate(model_ir.operators):
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            q_in_name = str(dq_op.inputs[0])
            f_in_name = str(dq_op.outputs[0])

            reshape_users = consumers.get(f_in_name, [])
            if len(reshape_users) != 1:
                continue
            reshape_idx = int(reshape_users[0])
            reshape_op = model_ir.operators[reshape_idx]
            if str(reshape_op.op_type) != "RESHAPE" or len(reshape_op.inputs) < 1 or len(reshape_op.outputs) != 1:
                continue
            if str(reshape_op.inputs[0]) != f_in_name:
                continue
            f_out_name = str(reshape_op.outputs[0])

            q_users = consumers.get(f_out_name, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != f_out_name:
                continue
            q_out_name = str(q_op.outputs[0])

            if f_in_name in model_ir.outputs or f_out_name in model_ir.outputs:
                continue

            q_in_tensor = model_ir.tensors.get(q_in_name, None)
            q_out_tensor = model_ir.tensors.get(q_out_name, None)
            f_out_tensor = model_ir.tensors.get(f_out_name, None)
            if q_in_tensor is None or q_out_tensor is None:
                continue
            if not _all_per_tensor_quantized([q_in_tensor, q_out_tensor]):
                continue
            if not _is_same_per_tensor_quantization(
                q_in_tensor.quantization,
                q_out_tensor.quantization,
            ):
                continue

            q_in_dtype = str(q_in_tensor.dtype)
            q_out_dtype = str(q_out_tensor.dtype)
            if q_in_dtype != q_out_dtype:
                continue
            if q_in_dtype in {"FLOAT16", "FLOAT32", "FLOAT64", "BOOL", "STRING"}:
                continue

            _replace_operator_input_at(
                model_ir=model_ir,
                op=reshape_op,
                input_index=0,
                new_input_name=q_in_name,
                graph_index=graph_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=reshape_op,
                new_outputs=[q_out_name],
                graph_index=graph_index,
            )

            q_out_tensor.dtype = q_in_dtype
            q_out_tensor.quantization = _clone_quantization(q_in_tensor.quantization)
            if f_out_tensor is not None:
                q_out_tensor.shape = [int(v) for v in list(f_out_tensor.shape)]
                if f_out_tensor.shape_signature is not None:
                    q_out_tensor.shape_signature = [
                        int(v) for v in list(f_out_tensor.shape_signature)
                    ]
                else:
                    q_out_tensor.shape_signature = [int(v) for v in list(f_out_tensor.shape)]

            for remove_idx in sorted([dq_idx, q_idx], reverse=True):
                graph_index.remove_operator(remove_idx)
            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if folded > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"folded_dequant_reshape_quantize_chains": int(folded)}


def run_quantized_reshape_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run quantization-preserving Reshape fusion as an ordered layout pass."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"DEQUANTIZE", "RESHAPE", "QUANTIZE"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if len(required) == 0:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        model_outputs = set(str(value) for value in candidate_model.outputs)
        for dq_op in candidate_model.operators:
            if (
                str(dq_op.op_type) != "DEQUANTIZE"
                or len(dq_op.inputs) != 1
                or len(dq_op.outputs) != 1
            ):
                continue
            f_in_name = str(dq_op.outputs[0])
            reshape_users = pass_state.graph_index.consumer_indices(f_in_name)
            if len(reshape_users) != 1:
                continue
            reshape_op = candidate_model.operators[int(reshape_users[0])]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 1
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != f_in_name
            ):
                continue
            f_out_name = str(reshape_op.outputs[0])
            quantize_users = pass_state.graph_index.consumer_indices(f_out_name)
            if len(quantize_users) != 1:
                continue
            quantize_op = candidate_model.operators[int(quantize_users[0])]
            if (
                str(quantize_op.op_type) != "QUANTIZE"
                or len(quantize_op.inputs) != 1
                or len(quantize_op.outputs) != 1
                or str(quantize_op.inputs[0]) != f_out_name
                or f_in_name in model_outputs
                or f_out_name in model_outputs
            ):
                continue
            q_in = candidate_model.tensors.get(str(dq_op.inputs[0]))
            q_out = candidate_model.tensors.get(str(quantize_op.outputs[0]))
            if q_in is None or q_out is None:
                continue
            if not _all_per_tensor_quantized([q_in, q_out]):
                continue
            if not _is_same_per_tensor_quantization(
                q_in.quantization,
                q_out.quantization,
            ):
                continue
            dtype = str(q_in.dtype)
            if (
                str(q_out.dtype) == dtype
                and dtype not in {"FLOAT16", "FLOAT32", "FLOAT64", "BOOL", "STRING"}
            ):
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_dequant_reshape_quantize_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("folded_dequant_reshape_quantize_chains", 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.dequant_reshape_quantize",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"folded_dequant_reshape_quantize_chains": 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
