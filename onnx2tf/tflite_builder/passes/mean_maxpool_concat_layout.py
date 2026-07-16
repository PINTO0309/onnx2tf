from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _prune_unused_tensors,
    _quant_scale_count,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _optimize_transpose_mean_maxpool_concat_conv_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate redundant NCHW/NHWC transpose chains around
    DEQUANTIZE->MEAN and MAX_POOL_2D branches before CONCATENATION->CONV.

    Target pattern (rank-4):
      q_raw_nhwc --TRANSPOSE(0,3,1,2)--> q_nchw --DEQUANTIZE--> dq_mean --MEAN(axes=[2,3])--> mean_nchw
      q_raw_nhwc --DEQUANTIZE--> dq_pool --MAX_POOL_2D--> pool_nhwc --TRANSPOSE(0,3,1,2)--> pool_nchw
      CONCAT(axis=1, [mean_nchw, pool_nchw]) -> QUANTIZE -> q_cat_nchw --TRANSPOSE(0,2,3,1)--> q_cat_nhwc -> CONV

    Rewritten:
      q_raw_nhwc --DEQUANTIZE--> dq_mean --MEAN(axes=[1,2])--> mean_nhwc
      q_raw_nhwc --DEQUANTIZE--> dq_pool --MAX_POOL_2D--> pool_nhwc
      CONCAT(axis=3, [mean_nhwc, pool_nhwc]) -> QUANTIZE -> q_cat_nhwc -> CONV
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    old_to_new_axis = perm_nhwc_to_nchw

    def _normalize_axis(axis: int, rank: int) -> Optional[int]:
        a = int(axis)
        if a < 0:
            a += int(rank)
        if a < 0 or a >= int(rank):
            return None
        return int(a)

    def _tensor_shape_signature(tensor: TensorIR) -> List[int]:
        if tensor.shape_signature is not None:
            return [int(v) for v in list(tensor.shape_signature)]
        return [int(v) for v in list(tensor.shape)]

    def _rank4_shape_and_signature(
        tensor: Optional[TensorIR],
    ) -> Optional[Tuple[List[int], List[int]]]:
        if tensor is None:
            return None
        shape = [int(v) for v in list(tensor.shape)]
        signature = _tensor_shape_signature(tensor)
        if len(shape) != 4 or len(signature) != 4:
            return None
        return shape, signature

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            q_raw_name = str(pre_op.inputs[0])
            q_nchw_name = str(pre_op.outputs[0])
            if q_raw_name in model_ir.outputs or q_nchw_name in model_ir.outputs:
                continue

            q_raw_tensor = model_ir.tensors.get(q_raw_name, None)
            q_nchw_tensor = model_ir.tensors.get(q_nchw_name, None)
            if q_raw_tensor is None or q_nchw_tensor is None:
                continue
            q_raw_shape = [int(v) for v in list(q_raw_tensor.shape)]
            if len(q_raw_shape) != 4:
                continue

            q_nchw_users = [int(v) for v in consumers.get(q_nchw_name, [])]
            if len(q_nchw_users) != 1:
                continue
            dq_mean_idx = int(q_nchw_users[0])
            dq_mean_op = model_ir.operators[dq_mean_idx]
            if str(dq_mean_op.op_type) != "DEQUANTIZE" or len(dq_mean_op.inputs) != 1 or len(dq_mean_op.outputs) != 1:
                continue
            dq_mean_out_name = str(dq_mean_op.outputs[0])
            if dq_mean_out_name in model_ir.outputs:
                continue

            dq_mean_out_users = [int(v) for v in consumers.get(dq_mean_out_name, [])]
            if len(dq_mean_out_users) != 1:
                continue
            mean_idx = int(dq_mean_out_users[0])
            mean_op = model_ir.operators[mean_idx]
            if str(mean_op.op_type) != "MEAN" or len(mean_op.inputs) < 2 or len(mean_op.outputs) != 1:
                continue
            if not bool(mean_op.options.get("keepDims", True)):
                continue
            mean_out_name = str(mean_op.outputs[0])
            if mean_out_name in model_ir.outputs:
                continue

            mean_axes_name = str(mean_op.inputs[1])
            mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
            if (
                mean_axes_tensor is None
                or str(mean_axes_tensor.dtype).upper() != "INT32"
                or bool(mean_axes_tensor.is_variable)
                or mean_axes_tensor.quantization is not None
                or mean_axes_name in model_ir.inputs
                or mean_axes_name in model_ir.outputs
                or [int(v) for v in consumers.get(mean_axes_name, [])]
                != [int(mean_idx)]
            ):
                continue
            try:
                mean_axes_data = np.asarray(mean_axes_tensor.data)
            except Exception:
                continue
            if mean_axes_data.dtype != np.dtype(np.int32):
                continue
            mean_axes_raw = _read_const_ints_from_tensor(mean_axes_tensor)
            if mean_axes_raw is None:
                continue
            mean_axes_old: List[int] = []
            axes_ok = True
            for axis in mean_axes_raw:
                norm_axis = _normalize_axis(int(axis), 4)
                if norm_axis is None:
                    axes_ok = False
                    break
                mean_axes_old.append(int(norm_axis))
            if not axes_ok:
                continue
            if sorted(mean_axes_old) != [2, 3]:
                continue

            q_raw_users = [int(v) for v in consumers.get(q_raw_name, [])]
            q_raw_users_wo_pre = [int(v) for v in q_raw_users if int(v) != int(pre_idx)]
            if len(q_raw_users_wo_pre) != 1:
                continue
            dq_pool_idx = int(q_raw_users_wo_pre[0])
            dq_pool_op = model_ir.operators[dq_pool_idx]
            if str(dq_pool_op.op_type) != "DEQUANTIZE" or len(dq_pool_op.inputs) != 1 or len(dq_pool_op.outputs) != 1:
                continue
            if str(dq_pool_op.inputs[0]) != q_raw_name:
                continue
            dq_pool_out_name = str(dq_pool_op.outputs[0])
            if dq_pool_out_name in model_ir.outputs:
                continue

            dq_pool_out_users = [int(v) for v in consumers.get(dq_pool_out_name, [])]
            if len(dq_pool_out_users) != 1:
                continue
            pool_idx = int(dq_pool_out_users[0])
            pool_op = model_ir.operators[pool_idx]
            if str(pool_op.op_type) != "MAX_POOL_2D" or len(pool_op.inputs) != 1 or len(pool_op.outputs) != 1:
                continue
            if str(pool_op.inputs[0]) != dq_pool_out_name:
                continue
            pool_out_name = str(pool_op.outputs[0])
            if pool_out_name in model_ir.outputs:
                continue

            pool_out_users = [int(v) for v in consumers.get(pool_out_name, [])]
            if len(pool_out_users) != 1:
                continue
            pool_post_idx = int(pool_out_users[0])
            pool_post_op = model_ir.operators[pool_post_idx]
            if str(pool_post_op.op_type) != "TRANSPOSE" or len(pool_post_op.inputs) < 2 or len(pool_post_op.outputs) != 1:
                continue
            if str(pool_post_op.inputs[0]) != pool_out_name:
                continue
            if _read_transpose_perm(model_ir, pool_post_op) != perm_nhwc_to_nchw:
                continue
            pool_nchw_name = str(pool_post_op.outputs[0])
            if pool_nchw_name in model_ir.outputs:
                continue

            mean_users = [int(v) for v in consumers.get(mean_out_name, [])]
            pool_nchw_users = [int(v) for v in consumers.get(pool_nchw_name, [])]
            if len(mean_users) != 1 or len(pool_nchw_users) != 1:
                continue
            concat_idx = int(mean_users[0])
            if int(pool_nchw_users[0]) != int(concat_idx):
                continue
            concat_op = model_ir.operators[concat_idx]
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue
            concat_out_name = str(concat_op.outputs[0])
            if concat_out_name in model_ir.outputs:
                continue
            concat_axis_old = _normalize_axis(int(concat_op.options.get("axis", 1)), 4)
            if concat_axis_old is None or int(concat_axis_old) != 1:
                continue
            if mean_out_name not in set(str(v) for v in concat_op.inputs):
                continue
            if pool_nchw_name not in set(str(v) for v in concat_op.inputs):
                continue

            concat_out_users = [int(v) for v in consumers.get(concat_out_name, [])]
            if len(concat_out_users) != 1:
                continue
            q_cat_idx = int(concat_out_users[0])
            q_cat_op = model_ir.operators[q_cat_idx]
            if str(q_cat_op.op_type) != "QUANTIZE" or len(q_cat_op.inputs) != 1 or len(q_cat_op.outputs) != 1:
                continue
            if str(q_cat_op.inputs[0]) != concat_out_name:
                continue
            q_cat_name = str(q_cat_op.outputs[0])
            if q_cat_name in model_ir.outputs:
                continue

            post_users = [int(v) for v in consumers.get(q_cat_name, [])]
            if len(post_users) == 0:
                continue
            removable_post_indices: List[int] = []
            valid_posts = True
            for post_idx in post_users:
                post_op = model_ir.operators[post_idx]
                if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                    valid_posts = False
                    break
                if str(post_op.inputs[0]) != q_cat_name:
                    valid_posts = False
                    break
                if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                    valid_posts = False
                    break
                post_out_name = str(post_op.outputs[0])
                if post_out_name in model_ir.outputs:
                    valid_posts = False
                    break
                removable_post_indices.append(int(post_idx))
            if not valid_posts:
                continue

            planned_concat_inputs = [
                (
                    str(pool_out_name)
                    if str(input_name) == str(pool_nchw_name)
                    else str(input_name)
                )
                for input_name in list(concat_op.inputs)
            ]
            rank4_tensor_names = [
                q_raw_name,
                q_nchw_name,
                dq_mean_out_name,
                mean_out_name,
                dq_pool_out_name,
                pool_out_name,
                pool_nchw_name,
                concat_out_name,
                q_cat_name,
            ] + list(planned_concat_inputs)
            post_output_names = [
                str(model_ir.operators[int(post_idx)].outputs[0])
                for post_idx in removable_post_indices
            ]
            rank4_tensor_names.extend(post_output_names)
            tensor_plans: Dict[
                str,
                Tuple[TensorIR, List[int], List[int]],
            ] = {}
            metadata_valid = True
            for tensor_name in rank4_tensor_names:
                if tensor_name in tensor_plans:
                    continue
                tensor = model_ir.tensors.get(str(tensor_name), None)
                shape_and_signature = _rank4_shape_and_signature(tensor)
                if tensor is None or shape_and_signature is None:
                    metadata_valid = False
                    break
                tensor_plans[str(tensor_name)] = (
                    tensor,
                    list(shape_and_signature[0]),
                    list(shape_and_signature[1]),
                )
            if not metadata_valid:
                continue

            dq_mean_out_tensor = tensor_plans[dq_mean_out_name][0]
            mean_out_tensor = tensor_plans[mean_out_name][0]
            concat_out_tensor = tensor_plans[concat_out_name][0]
            q_cat_tensor = tensor_plans[q_cat_name][0]

            q_cat_has_per_axis_quantization = (
                q_cat_tensor.quantization is not None
                and _quant_scale_count(q_cat_tensor.quantization) > 1
            )
            if (
                q_cat_has_per_axis_quantization
                and int(q_cat_tensor.quantization.quantized_dimension) != 1
            ):
                continue

            q_raw_shape_plan = list(tensor_plans[q_raw_name][1])
            q_raw_signature_plan = list(tensor_plans[q_raw_name][2])
            mean_axes_new = [
                int(old_to_new_axis[int(axis)]) for axis in mean_axes_old
            ]
            mean_shape_plan = list(q_raw_shape_plan)
            mean_signature_plan = list(q_raw_signature_plan)
            for axis in mean_axes_new:
                mean_shape_plan[int(axis)] = 1
                mean_signature_plan[int(axis)] = 1

            concat_input_metadata: List[Tuple[List[int], List[int]]] = []
            for tensor_name in planned_concat_inputs:
                if str(tensor_name) == str(mean_out_name):
                    concat_input_metadata.append(
                        (list(mean_shape_plan), list(mean_signature_plan))
                    )
                else:
                    concat_input_metadata.append(
                        (
                            list(tensor_plans[str(tensor_name)][1]),
                            list(tensor_plans[str(tensor_name)][2]),
                        )
                    )
            if len(concat_input_metadata) == 0:
                continue
            concat_shape_plan = list(concat_input_metadata[0][0])
            for shape_i, _ in concat_input_metadata[1:]:
                concat_shape_plan[3] += int(shape_i[3])
            concat_signature_plan = list(concat_input_metadata[0][1])
            if any(
                int(signature_i[3]) < 0
                for _, signature_i in concat_input_metadata
            ):
                concat_signature_plan[3] = -1
            else:
                concat_signature_plan[3] = int(
                    sum(
                        int(signature_i[3])
                        for _, signature_i in concat_input_metadata
                    )
                )

            remove_indices = sorted(
                set(
                    [int(pre_idx), int(pool_post_idx)]
                    + [int(v) for v in removable_post_indices]
                ),
                reverse=True,
            )

            # Commit only after every edge, tensor, signature, and axis is valid.
            _set_operator_inputs(
                model_ir=model_ir,
                op=dq_mean_op,
                new_inputs=[q_raw_name],
            )
            dq_mean_out_tensor.shape = list(q_raw_shape_plan)
            dq_mean_out_tensor.shape_signature = list(q_raw_signature_plan)
            _write_const_ints_to_tensor(mean_axes_tensor, mean_axes_new)
            mean_out_tensor.shape = list(mean_shape_plan)
            mean_out_tensor.shape_signature = list(mean_signature_plan)
            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=list(planned_concat_inputs),
            )
            concat_op.options["axis"] = 3
            concat_out_tensor.shape = list(concat_shape_plan)
            concat_out_tensor.shape_signature = list(concat_signature_plan)
            q_cat_tensor.shape = list(concat_shape_plan)
            q_cat_tensor.shape_signature = list(concat_signature_plan)
            if q_cat_has_per_axis_quantization:
                q_cat_tensor.quantization.quantized_dimension = 3

            # Remove post-quantize transpose adapters and reconnect their consumers.
            for post_idx in removable_post_indices:
                post_op = model_ir.operators[int(post_idx)]
                post_out_name = str(post_op.outputs[0])
                _replace_tensor_inputs(model_ir, post_out_name, q_cat_name)

            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {
        "optimized_transpose_mean_maxpool_concat_conv_chains": int(optimized),
    }
