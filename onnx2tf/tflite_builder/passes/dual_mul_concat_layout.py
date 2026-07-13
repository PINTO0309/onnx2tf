from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _optimize_transpose_dual_mul_concat_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around dual-MUL concat blocks.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      MUL(x_nchw, c0_nchw) -> m0_nchw
      MUL(x_nchw, c1_nchw) -> m1_nchw
      CONCAT(axis=1, [m0_nchw, m1_nchw]) -> c_nchw
      c_nchw --TRANSPOSE(0,2,3,1)--> y_nhwc

    Rewrite:
      MUL(x_nhwc, c0_nhwc) -> m0_nhwc
      MUL(x_nhwc, c1_nhwc) -> m1_nhwc
      CONCAT(axis=3, [m0_nhwc, m1_nhwc]) -> y_nhwc
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _clone_or_rewrite_mul_const_to_nhwc(
        *,
        mul_idx: int,
        side_input_index: int,
        side_input_name: str,
        target_shape_nhwc: Optional[List[int]],
        chain_indices: set[int],
        consumers: Dict[str, List[int]],
    ) -> bool:
        side_tensor = model_ir.tensors.get(str(side_input_name), None)
        if side_tensor is None or side_tensor.data is None:
            return False
        side_data = np.asarray(side_tensor.data)
        if int(side_data.size) == 1:
            return True
        if side_data.ndim != 4:
            return False

        target_shape = (
            [int(v) for v in list(target_shape_nhwc)]
            if _is_fully_known_positive_shape(target_shape_nhwc)
            else None
        )
        nhwc_data: Optional[np.ndarray] = None
        if target_shape is not None:
            side_shape = [int(v) for v in list(side_data.shape)]
            if _broadcast_static_shapes(target_shape, side_shape) is not None:
                nhwc_data = np.asarray(side_data)
            else:
                rotated = np.transpose(side_data, perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
                rotated_shape = [int(v) for v in list(rotated.shape)]
                if _broadcast_static_shapes(target_shape, rotated_shape) is None:
                    return False
                nhwc_data = np.asarray(rotated)
        else:
            side_shape = [int(v) for v in list(side_data.shape)]
            if (
                len(side_shape) == 4
                and int(side_shape[0]) == 1
                and int(side_shape[1]) > 0
                and int(side_shape[2]) == 1
                and int(side_shape[3]) == 1
            ):
                nhwc_data = np.transpose(side_data, perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
            else:
                nhwc_data = np.asarray(side_data)
        if nhwc_data is None:
            return False

        side_users = [int(v) for v in consumers.get(str(side_input_name), [])]
        shared_outside_chain = any(int(u) not in chain_indices for u in side_users)
        rewritten_side_name = str(side_input_name)
        if shared_outside_chain:
            rewritten_side_name = _unique_tensor_name(f"{side_input_name}_nhwc")
            model_ir.tensors[rewritten_side_name] = TensorIR(
                name=rewritten_side_name,
                dtype=str(side_tensor.dtype),
                shape=[int(v) for v in list(nhwc_data.shape)],
                shape_signature=[int(v) for v in list(nhwc_data.shape)],
                data=np.asarray(nhwc_data),
                is_variable=False,
                quantization=_clone_quantization(side_tensor.quantization),
            )
        else:
            side_tensor.data = np.asarray(nhwc_data)
            side_tensor.shape = [int(v) for v in list(nhwc_data.shape)]
            side_tensor.shape_signature = [int(v) for v in list(nhwc_data.shape)]

        _replace_operator_input_at(
            model_ir=model_ir,
            op=model_ir.operators[int(mul_idx)],
            input_index=int(side_input_index),
            new_input_name=str(rewritten_side_name),
        )
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue
            concat_out_name = str(post_op.inputs[0])
            post_out_name = str(post_op.outputs[0])
            if concat_out_name in model_outputs or post_out_name in model_outputs:
                continue

            concat_idx = producers.get(str(concat_out_name), None)
            if concat_idx is None:
                continue
            concat_op = model_ir.operators[int(concat_idx)]
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.inputs) != 2 or len(concat_op.outputs) != 1:
                continue
            axis = int(concat_op.options.get("axis", 1))
            if axis < 0:
                axis += 4
            if axis != 1:
                continue
            if set(int(v) for v in consumers.get(str(concat_out_name), [])) != {int(post_idx)}:
                continue

            mul_indices: List[int] = []
            mul_plans: List[Tuple[int, int, str, str]] = []
            shared_data_name: Optional[str] = None
            rewritable = True
            for concat_input_name in [str(v) for v in list(concat_op.inputs)]:
                mul_idx = producers.get(str(concat_input_name), None)
                if mul_idx is None:
                    rewritable = False
                    break
                mul_op = model_ir.operators[int(mul_idx)]
                if (
                    str(mul_op.op_type) != "MUL"
                    or len(mul_op.inputs) != 2
                    or len(mul_op.outputs) != 1
                    or str(mul_op.outputs[0]) != str(concat_input_name)
                ):
                    rewritable = False
                    break
                if set(int(v) for v in consumers.get(str(concat_input_name), [])) != {int(concat_idx)}:
                    rewritable = False
                    break
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                data_input_index = None
                side_input_index = None
                if shared_data_name is None:
                    for candidate_data_index in [0, 1]:
                        candidate_side_index = 1 - int(candidate_data_index)
                        candidate_data_name = str(mul_inputs[int(candidate_data_index)])
                        candidate_prod_idx = producers.get(candidate_data_name, None)
                        if candidate_prod_idx is None:
                            continue
                        candidate_prod_op = model_ir.operators[int(candidate_prod_idx)]
                        if (
                            str(candidate_prod_op.op_type) == "TRANSPOSE"
                            and len(candidate_prod_op.inputs) >= 2
                            and len(candidate_prod_op.outputs) == 1
                            and str(candidate_prod_op.outputs[0]) == candidate_data_name
                            and _read_transpose_perm(model_ir, candidate_prod_op) == perm_nhwc_to_nchw
                        ):
                            shared_data_name = str(candidate_data_name)
                            data_input_index = int(candidate_data_index)
                            side_input_index = int(candidate_side_index)
                            break
                else:
                    if mul_inputs[0] == str(shared_data_name):
                        data_input_index = 0
                        side_input_index = 1
                    elif mul_inputs[1] == str(shared_data_name):
                        data_input_index = 1
                        side_input_index = 0
                if data_input_index is None or side_input_index is None:
                    rewritable = False
                    break
                side_input_name = str(mul_inputs[int(side_input_index)])
                side_tensor = model_ir.tensors.get(str(side_input_name), None)
                if side_tensor is None or side_tensor.data is None:
                    rewritable = False
                    break

                mul_indices.append(int(mul_idx))
                mul_plans.append(
                    (
                        int(mul_idx),
                        int(side_input_index),
                        str(side_input_name),
                        str(mul_inputs[int(data_input_index)]),
                    )
                )

            if not rewritable or shared_data_name is None:
                continue

            pre_idx = producers.get(str(shared_data_name), None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != str(shared_data_name)
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            ):
                continue
            if shared_data_name in model_outputs or str(pre_op.inputs[0]) in model_outputs:
                continue
            if set(int(v) for v in consumers.get(str(shared_data_name), [])) != set(int(v) for v in mul_indices):
                continue

            chain_indices = {int(pre_idx), int(concat_idx), int(post_idx)} | set(int(v) for v in mul_indices)
            target_shape_nhwc = (
                [int(v) for v in list(model_ir.tensors[str(pre_op.inputs[0])].shape)]
                if str(pre_op.inputs[0]) in model_ir.tensors
                else None
            )
            for mul_idx, side_input_index, side_input_name, _ in mul_plans:
                if not _clone_or_rewrite_mul_const_to_nhwc(
                    mul_idx=int(mul_idx),
                    side_input_index=int(side_input_index),
                    side_input_name=str(side_input_name),
                    target_shape_nhwc=target_shape_nhwc,
                    chain_indices=chain_indices,
                    consumers=consumers,
                ):
                    rewritable = False
                    break
            if not rewritable:
                continue

            for mul_idx in mul_indices:
                mul_op = model_ir.operators[int(mul_idx)]
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                for input_index, input_name in enumerate(list(mul_inputs)):
                    if str(input_name) == str(shared_data_name):
                        mul_inputs[int(input_index)] = str(pre_op.inputs[0])
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=mul_inputs,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(mul_op.outputs[0]), None),
                    perm_nchw_to_nhwc,
                )

            concat_op.options["axis"] = 3
            _set_operator_outputs(
                model_ir=model_ir,
                op=concat_op,
                new_outputs=[post_out_name],
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(concat_out_name), None),
                perm_nchw_to_nhwc,
            )

            old_concat_tensor = model_ir.tensors.get(str(concat_out_name), None)
            post_out_tensor = model_ir.tensors.get(str(post_out_name), None)
            if old_concat_tensor is not None and post_out_tensor is not None:
                post_out_tensor.dtype = str(old_concat_tensor.dtype)
                post_out_tensor.quantization = _clone_quantization(old_concat_tensor.quantization)
                post_out_tensor.shape = [int(v) for v in list(old_concat_tensor.shape)]
                post_out_tensor.shape_signature = (
                    [int(v) for v in list(old_concat_tensor.shape_signature)]
                    if old_concat_tensor.shape_signature is not None
                    else [int(v) for v in list(old_concat_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_out_tensor,
                    perm_nchw_to_nhwc,
                )

            for remove_idx in sorted([int(post_idx), int(pre_idx)], reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_dual_mul_concat_prepost_nhwc_chains": int(rewritten)}

