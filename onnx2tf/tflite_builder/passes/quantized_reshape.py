from __future__ import annotations

from typing import Dict

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _build_tensor_consumer_map,
    _clone_quantization,
    _is_same_per_tensor_quantization,
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR

def _optimize_dequant_reshape_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
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

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

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
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=reshape_op,
                new_outputs=[q_out_name],
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
                del model_ir.operators[remove_idx]
            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"folded_dequant_reshape_quantize_chains": int(folded)}
