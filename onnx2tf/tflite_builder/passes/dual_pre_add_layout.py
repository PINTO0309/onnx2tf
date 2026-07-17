from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR

def optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Reduce dual NHWC->NCHW adapters feeding one ADD into a single post-ADD adapter.

    Target:
      a_nhwc --T(0,3,1,2)--> a_nchw
      b_nhwc --T(0,3,1,2)--> b_nchw
      ADD(a_nchw, b_nchw) -> y_nchw

    Rewrite:
      ADD(a_nhwc, b_nhwc) -> y_nhwc
      y_nhwc --T(0,3,1,2)--> y_nchw

    Safety:
    - Both pre-transpose outputs are consumed only by the target ADD.
    - ADD output is rank-4 and is not a graph output.
    - Skip when ADD already fans out to NCHW->NHWC post-transpose consumers.
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_const_name = "__nhwc_to_nchw_perm_rank4__"

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _op_index(op_ref: OperatorIR) -> Optional[int]:
        for idx, op in enumerate(model_ir.operators):
            if id(op) == id(op_ref):
                return int(idx)
        return None

    if perm_const_name not in model_ir.tensors:
        model_ir.tensors[perm_const_name] = TensorIR(
            name=perm_const_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for add_idx, add_op in enumerate(model_ir.operators):
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue

            add_out_name = str(add_op.outputs[0])
            if add_out_name in model_outputs:
                continue

            add_out_users = [int(v) for v in consumers.get(add_out_name, []) if int(v) != int(add_idx)]
            if any(
                str(model_ir.operators[int(user_idx)].op_type) == "TRANSPOSE"
                and _read_transpose_perm(model_ir, model_ir.operators[int(user_idx)]) == perm_nchw_to_nhwc
                for user_idx in add_out_users
            ):
                continue

            pre_indices: List[int] = []
            new_add_inputs: List[str] = []
            rewritable = True
            for add_input_name in [str(v) for v in list(add_op.inputs)]:
                pre_idx = producers.get(add_input_name, None)
                if pre_idx is None:
                    rewritable = False
                    break
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != add_input_name
                    or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                    or str(add_input_name) in model_outputs
                ):
                    rewritable = False
                    break
                if set(int(v) for v in consumers.get(add_input_name, [])) != {int(add_idx)}:
                    rewritable = False
                    break
                pre_input_name = str(pre_op.inputs[0])
                pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
                if pre_input_tensor is None or len(list(pre_input_tensor.shape)) != 4:
                    rewritable = False
                    break
                pre_indices.append(int(pre_idx))
                new_add_inputs.append(pre_input_name)
            if not rewritable:
                continue

            add_out_tensor = model_ir.tensors.get(add_out_name, None)
            if add_out_tensor is None or len(list(add_out_tensor.shape)) != 4:
                continue
            add_out_shape = [int(v) for v in list(add_out_tensor.shape)]
            add_out_signature = (
                [int(v) for v in list(add_out_tensor.shape_signature)]
                if add_out_tensor.shape_signature is not None
                else [int(v) for v in list(add_out_tensor.shape)]
            )
            add_out_nhwc_shape = _permute_shape(add_out_shape, perm_nchw_to_nhwc)
            add_out_nhwc_signature = _permute_shape(add_out_signature, perm_nchw_to_nhwc)
            if add_out_nhwc_shape is None or add_out_nhwc_signature is None:
                continue

            add_out_nhwc_name = _unique_tensor_name(f"{add_out_name}_nhwc")
            model_ir.tensors[add_out_nhwc_name] = TensorIR(
                name=add_out_nhwc_name,
                dtype=str(add_out_tensor.dtype),
                shape=[int(v) for v in list(add_out_nhwc_shape)],
                shape_signature=[int(v) for v in list(add_out_nhwc_signature)],
                data=None,
                is_variable=False,
                quantization=_clone_quantization(add_out_tensor.quantization),
            )

            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=[str(v) for v in list(new_add_inputs)],
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=add_op,
                new_outputs=[str(add_out_nhwc_name)],
            )

            for remove_idx in sorted(list(set(int(v) for v in pre_indices)), reverse=True):
                del model_ir.operators[int(remove_idx)]

            add_new_idx = _op_index(add_op)
            if add_new_idx is None:
                changed = False
                break
            model_ir.operators.insert(
                int(add_new_idx) + 1,
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[str(add_out_nhwc_name), str(perm_const_name)],
                    outputs=[str(add_out_name)],
                    options={},
                ),
            )

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains": int(optimized)}
