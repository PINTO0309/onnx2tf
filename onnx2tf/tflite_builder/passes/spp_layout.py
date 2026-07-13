from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


def _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Collapse SPP-style NCHW resize/add/concat/affine bridges back to NHWC.

    Strict target:
      base_nhwc --T--> base_nchw
      resize_i_nhwc --T--> resize_i_nchw
      ADD(base_nchw, resize_i_nchw) -> add_i_nchw (4 branches)
      CONCAT(axis=1, add_i_nchw...) -> cat0_nchw
      MUL(cat0_nchw, const_nchw) -> mul0_nchw --T--> mul0_nhwc
      ADD(mul0_nhwc, bias_nhwc) -> ... -> CONV -> conv0_nhwc --T--> conv0_nchw
      CONCAT(axis=1, base_nchw, conv0_nchw) -> cat1_nchw
      MUL(cat1_nchw, const_nchw) -> mul1_nchw --T--> mul1_nhwc
      ADD(mul1_nhwc, bias_nhwc) -> ... -> CONV

    Rewrite:
      Keep both concat islands in NHWC, rewrite CONCAT axis to 3,
      permute channelwise MUL constants to NHWC, and remove the transposes.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _channelwise_const_name(binary_op: OperatorIR, tensor_name_to_skip: str) -> Optional[str]:
        for input_name in [str(v) for v in list(binary_op.inputs)]:
            if str(input_name) == str(tensor_name_to_skip):
                continue
            tensor = model_ir.tensors.get(str(input_name), None)
            if tensor is None or tensor.data is None:
                continue
            const_arr = np.asarray(tensor.data)
            if const_arr.ndim == 4:
                return str(input_name)
        return None

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat0_idx, concat0_op in enumerate(model_ir.operators):
            if str(concat0_op.op_type) != "CONCATENATION" or len(concat0_op.inputs) != 4 or len(concat0_op.outputs) != 1:
                continue
            axis0 = int(concat0_op.options.get("axis", 1))
            if axis0 < 0:
                axis0 += 4
            if axis0 != 1:
                continue

            concat0_out_name = str(concat0_op.outputs[0])
            if concat0_out_name in model_outputs:
                continue

            branch_add_indices: List[int] = []
            base_nchw_name: Optional[str] = None
            resize_post_indices: List[int] = []
            resize_nhwc_names: List[str] = []
            valid = True

            for add_out_name in [str(v) for v in list(concat0_op.inputs)]:
                add_idx = producers.get(add_out_name, None)
                if add_idx is None:
                    valid = False
                    break
                add_op = model_ir.operators[int(add_idx)]
                if (
                    str(add_op.op_type) != "ADD"
                    or len(add_op.inputs) != 2
                    or len(add_op.outputs) != 1
                    or str(add_op.outputs[0]) != str(add_out_name)
                    or set(int(v) for v in consumers.get(str(add_out_name), [])) != {int(concat0_idx)}
                ):
                    valid = False
                    break

                add_inputs = [str(v) for v in list(add_op.inputs)]
                transpose_inputs: List[Tuple[int, str, str]] = []
                for input_name in add_inputs:
                    pre_idx = producers.get(str(input_name), None)
                    if pre_idx is None:
                        continue
                    pre_op = model_ir.operators[int(pre_idx)]
                    if (
                        str(pre_op.op_type) == "TRANSPOSE"
                        and len(pre_op.inputs) >= 2
                        and len(pre_op.outputs) == 1
                        and str(pre_op.outputs[0]) == str(input_name)
                        and _read_transpose_perm(model_ir, pre_op) == perm_nhwc_to_nchw
                        and str(pre_op.outputs[0]) not in model_outputs
                    ):
                        transpose_inputs.append((int(pre_idx), str(pre_op.inputs[0]), str(pre_op.outputs[0])))
                if len(transpose_inputs) != 2:
                    valid = False
                    break

                branch_add_indices.append(int(add_idx))
                if base_nchw_name is None:
                    names = [name for _, _, name in transpose_inputs]
                    for candidate_name in names:
                        if all(
                            candidate_name in [str(v) for v in list(model_ir.operators[int(producers[str(other_out)])].inputs)]
                            for other_out in list(concat0_op.inputs)
                            if producers.get(str(other_out), None) is not None
                        ):
                            base_nchw_name = str(candidate_name)
                            break
                if base_nchw_name is None:
                    valid = False
                    break

                branch_info = None
                for pre_idx, nhwc_name, nchw_name in transpose_inputs:
                    if str(nchw_name) == str(base_nchw_name):
                        continue
                    branch_prod_idx = producers.get(str(nhwc_name), None)
                    if branch_prod_idx is None:
                        continue
                    branch_prod = model_ir.operators[int(branch_prod_idx)]
                    if str(branch_prod.op_type) == "RESIZE_BILINEAR":
                        branch_info = (int(pre_idx), str(nhwc_name), str(nchw_name))
                        break
                if branch_info is None:
                    valid = False
                    break
                resize_post_indices.append(int(branch_info[0]))
                resize_nhwc_names.append(str(branch_info[1]))

            if not valid or base_nchw_name is None:
                continue

            base_pre_idx = producers.get(str(base_nchw_name), None)
            if base_pre_idx is None:
                continue
            base_pre_op = model_ir.operators[int(base_pre_idx)]
            if (
                str(base_pre_op.op_type) != "TRANSPOSE"
                or len(base_pre_op.inputs) < 2
                or len(base_pre_op.outputs) != 1
                or _read_transpose_perm(model_ir, base_pre_op) != perm_nhwc_to_nchw
            ):
                continue
            base_nhwc_name = str(base_pre_op.inputs[0])

            concat0_users = [int(v) for v in consumers.get(str(concat0_out_name), [])]
            if len(concat0_users) != 1:
                continue
            mul0_idx = int(concat0_users[0])
            mul0_op = model_ir.operators[int(mul0_idx)]
            if str(mul0_op.op_type) != "MUL" or len(mul0_op.inputs) != 2 or len(mul0_op.outputs) != 1:
                continue
            mul0_out_name = str(mul0_op.outputs[0])
            mul0_const_name = _channelwise_const_name(mul0_op, concat0_out_name)
            if mul0_const_name is None:
                continue

            mul0_users = [int(v) for v in consumers.get(str(mul0_out_name), [])]
            if len(mul0_users) != 1:
                continue
            post0_idx = int(mul0_users[0])
            post0_op = model_ir.operators[int(post0_idx)]
            if (
                str(post0_op.op_type) != "TRANSPOSE"
                or len(post0_op.inputs) < 2
                or len(post0_op.outputs) != 1
                or str(post0_op.inputs[0]) != str(mul0_out_name)
                or _read_transpose_perm(model_ir, post0_op) != perm_nchw_to_nhwc
            ):
                continue
            post0_out_name = str(post0_op.outputs[0])

            post0_users = [int(v) for v in consumers.get(str(post0_out_name), [])]
            if len(post0_users) != 1:
                continue
            add0_idx = int(post0_users[0])
            add0_op = model_ir.operators[int(add0_idx)]
            if str(add0_op.op_type) != "ADD" or len(add0_op.inputs) != 2 or len(add0_op.outputs) != 1:
                continue
            add0_out_name = str(add0_op.outputs[0])

            add0_users = [int(v) for v in consumers.get(str(add0_out_name), [])]
            if len(add0_users) != 1:
                continue
            conv0_idx = int(add0_users[0])
            conv0_op = model_ir.operators[int(conv0_idx)]
            if str(conv0_op.op_type) not in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                continue
            conv0_out_name = str(conv0_op.outputs[0])

            conv0_users = [int(v) for v in consumers.get(str(conv0_out_name), [])]
            if len(conv0_users) != 1:
                continue
            conv0_post_idx = int(conv0_users[0])
            conv0_post_op = model_ir.operators[int(conv0_post_idx)]
            if (
                str(conv0_post_op.op_type) != "TRANSPOSE"
                or len(conv0_post_op.inputs) < 2
                or len(conv0_post_op.outputs) != 1
                or str(conv0_post_op.inputs[0]) != str(conv0_out_name)
                or _read_transpose_perm(model_ir, conv0_post_op) != perm_nhwc_to_nchw
            ):
                continue
            conv0_nchw_name = str(conv0_post_op.outputs[0])

            conv0_nchw_users = [int(v) for v in consumers.get(str(conv0_nchw_name), [])]
            base_nchw_users = [int(v) for v in consumers.get(str(base_nchw_name), [])]
            if len(conv0_nchw_users) != 1 or int(conv0_nchw_users[0]) not in set(int(v) for v in base_nchw_users):
                continue

            concat1_idx = int(conv0_nchw_users[0])
            concat1_op = model_ir.operators[int(concat1_idx)]
            if (
                str(concat1_op.op_type) != "CONCATENATION"
                or len(concat1_op.inputs) != 2
                or len(concat1_op.outputs) != 1
            ):
                continue
            axis1 = int(concat1_op.options.get("axis", 1))
            if axis1 < 0:
                axis1 += 4
            if axis1 != 1:
                continue
            concat1_inputs = [str(v) for v in list(concat1_op.inputs)]
            if set(concat1_inputs) != {str(base_nchw_name), str(conv0_nchw_name)}:
                continue

            concat1_out_name = str(concat1_op.outputs[0])
            concat1_users = [int(v) for v in consumers.get(str(concat1_out_name), [])]
            if len(concat1_users) != 1:
                continue
            mul1_idx = int(concat1_users[0])
            mul1_op = model_ir.operators[int(mul1_idx)]
            if str(mul1_op.op_type) != "MUL" or len(mul1_op.inputs) != 2 or len(mul1_op.outputs) != 1:
                continue
            mul1_out_name = str(mul1_op.outputs[0])
            mul1_const_name = _channelwise_const_name(mul1_op, concat1_out_name)
            if mul1_const_name is None:
                continue

            mul1_users = [int(v) for v in consumers.get(str(mul1_out_name), [])]
            if len(mul1_users) != 1:
                continue
            post1_idx = int(mul1_users[0])
            post1_op = model_ir.operators[int(post1_idx)]
            if (
                str(post1_op.op_type) != "TRANSPOSE"
                or len(post1_op.inputs) < 2
                or len(post1_op.outputs) != 1
                or str(post1_op.inputs[0]) != str(mul1_out_name)
                or _read_transpose_perm(model_ir, post1_op) != perm_nchw_to_nhwc
            ):
                continue
            post1_out_name = str(post1_op.outputs[0])

            post1_users = [int(v) for v in consumers.get(str(post1_out_name), [])]
            if len(post1_users) != 1:
                continue
            add1_idx = int(post1_users[0])
            add1_op = model_ir.operators[int(add1_idx)]
            if str(add1_op.op_type) != "ADD" or len(add1_op.inputs) != 2 or len(add1_op.outputs) != 1:
                continue
            add1_out_name = str(add1_op.outputs[0])

            add1_users = [int(v) for v in consumers.get(str(add1_out_name), [])]
            if len(add1_users) != 1:
                continue
            conv1_op = model_ir.operators[int(add1_users[0])]
            if str(conv1_op.op_type) not in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                continue

            for add_idx, resize_nhwc_name in zip(branch_add_indices, resize_nhwc_names):
                add_op = model_ir.operators[int(add_idx)]
                add_inputs = [
                    str(base_nhwc_name) if str(input_name) == str(base_nchw_name)
                    else str(resize_nhwc_name) if str(input_name) != str(base_nchw_name)
                    else str(input_name)
                    for input_name in list(add_op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=add_op,
                    new_inputs=add_inputs,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(add_op.outputs[0]), None),
                    perm_nchw_to_nhwc,
                )

            concat0_op.options["axis"] = 3
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(concat0_out_name), None),
                perm_nchw_to_nhwc,
            )

            mul0_const_tensor = model_ir.tensors.get(str(mul0_const_name), None)
            if mul0_const_tensor is not None and mul0_const_tensor.data is not None:
                const_arr = np.asarray(mul0_const_tensor.data)
                if const_arr.ndim == 4:
                    const_arr = np.transpose(const_arr, axes=perm_nchw_to_nhwc)
                    mul0_const_tensor.data = np.asarray(const_arr)
                    mul0_const_tensor.shape = [int(v) for v in list(const_arr.shape)]
                    mul0_const_tensor.shape_signature = [int(v) for v in list(const_arr.shape)]
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(mul0_out_name), None),
                perm_nchw_to_nhwc,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=add0_op,
                new_inputs=[
                    str(mul0_out_name) if str(input_name) == str(post0_out_name) else str(input_name)
                    for input_name in list(add0_op.inputs)
                ],
            )

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat1_op,
                new_inputs=[
                    str(base_nhwc_name) if str(input_name) == str(base_nchw_name)
                    else str(conv0_out_name) if str(input_name) == str(conv0_nchw_name)
                    else str(input_name)
                    for input_name in list(concat1_op.inputs)
                ],
            )
            concat1_op.options["axis"] = 3
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(concat1_out_name), None),
                perm_nchw_to_nhwc,
            )

            mul1_const_tensor = model_ir.tensors.get(str(mul1_const_name), None)
            if mul1_const_tensor is not None and mul1_const_tensor.data is not None:
                const_arr = np.asarray(mul1_const_tensor.data)
                if const_arr.ndim == 4:
                    const_arr = np.transpose(const_arr, axes=perm_nchw_to_nhwc)
                    mul1_const_tensor.data = np.asarray(const_arr)
                    mul1_const_tensor.shape = [int(v) for v in list(const_arr.shape)]
                    mul1_const_tensor.shape_signature = [int(v) for v in list(const_arr.shape)]
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(mul1_out_name), None),
                perm_nchw_to_nhwc,
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=add1_op,
                new_inputs=[
                    str(mul1_out_name) if str(input_name) == str(post1_out_name) else str(input_name)
                    for input_name in list(add1_op.inputs)
                ],
            )

            remove_indices = {
                int(base_pre_idx),
                int(post0_idx),
                int(conv0_post_idx),
                int(post1_idx),
                *[int(v) for v in resize_post_indices],
            }
            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains": int(rewritten)}
