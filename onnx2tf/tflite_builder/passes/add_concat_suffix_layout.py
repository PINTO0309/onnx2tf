from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


def _optimize_transpose_add_concat_const_suffix_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around ADD fan-in concat blocks with const suffixes.

    Target:
      (x_i_nhwc --T(0,3,1,2)--> x_i_nchw)
      (b_nhwc  --T(0,3,1,2)--> b_nchw; shared)
      ADD(x_i_nchw, b_nchw) -> a_i_nchw
      CONCAT(axis=1, [a_i_nchw...]) -> c_nchw
      c_nchw --(MUL|ADD with const)*--> z_nchw
      z_nchw --T(0,2,3,1)--> z_nhwc

    Rewrite:
      ADD(x_i_nhwc, b_nhwc) -> a_i_nhwc
      CONCAT(axis=3, [a_i_nhwc...]) -> c_nhwc
      c_nhwc --(MUL|ADD with const_nhwc)*--> z_nhwc
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue
            concat_out_name = str(concat_op.outputs[0])
            if concat_out_name in model_outputs:
                continue
            concat_axis = int(concat_op.options.get("axis", 1))
            if concat_axis < 0:
                concat_axis += 4
            if concat_axis != 1:
                continue

            concat_inputs = [str(v) for v in list(concat_op.inputs)]
            if len(concat_inputs) < 2:
                continue

            add_rewrites: List[Tuple[int, List[str]]] = []
            branch_pre_remove_indices: List[int] = []
            base_pre_idx: Optional[int] = None
            base_pre_input_name: Optional[str] = None
            rewritable = True

            for add_out_name in concat_inputs:
                add_idx = producers.get(str(add_out_name), None)
                if add_idx is None:
                    rewritable = False
                    break
                add_op = model_ir.operators[int(add_idx)]
                if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                    rewritable = False
                    break
                if str(add_op.outputs[0]) != str(add_out_name):
                    rewritable = False
                    break
                if set(int(v) for v in consumers.get(str(add_out_name), [])) != {int(concat_idx)}:
                    rewritable = False
                    break

                add_inputs = [str(v) for v in list(add_op.inputs)]
                transpose_inputs: List[Tuple[int, int, str, bool]] = []
                for input_index, add_input_name in enumerate(add_inputs):
                    pre_idx = producers.get(str(add_input_name), None)
                    if pre_idx is None:
                        continue
                    pre_op = model_ir.operators[int(pre_idx)]
                    if (
                        str(pre_op.op_type) != "TRANSPOSE"
                        or len(pre_op.inputs) < 2
                        or len(pre_op.outputs) != 1
                        or str(pre_op.outputs[0]) != str(add_input_name)
                        or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                        or str(add_input_name) in model_outputs
                    ):
                        continue
                    direct_only = set(int(v) for v in consumers.get(str(add_input_name), [])) == {int(add_idx)}
                    transpose_inputs.append((int(input_index), int(pre_idx), str(pre_op.inputs[0]), bool(direct_only)))

                if len(transpose_inputs) < 2:
                    rewritable = False
                    break

                branch_plan = None
                base_plan = None
                for plan in transpose_inputs:
                    if bool(plan[3]):
                        branch_plan = plan
                        break
                if branch_plan is None:
                    rewritable = False
                    break
                for plan in transpose_inputs:
                    if int(plan[1]) != int(branch_plan[1]):
                        base_plan = plan
                        break
                if base_plan is None:
                    rewritable = False
                    break

                if base_pre_idx is None:
                    base_pre_idx = int(base_plan[1])
                    base_pre_input_name = str(base_plan[2])
                elif int(base_pre_idx) != int(base_plan[1]):
                    rewritable = False
                    break

                new_add_inputs = list(add_inputs)
                new_add_inputs[int(branch_plan[0])] = str(branch_plan[2])
                new_add_inputs[int(base_plan[0])] = str(base_plan[2])
                add_rewrites.append((int(add_idx), [str(v) for v in new_add_inputs]))
                branch_pre_remove_indices.append(int(branch_plan[1]))

            if not rewritable or base_pre_idx is None or base_pre_input_name is None:
                continue

            # Strict suffix: CONCAT -> MUL(const) -> ADD(const) -> TRANSPOSE(NCHW->NHWC)
            concat_users = [int(v) for v in consumers.get(concat_out_name, [])]
            if len(concat_users) != 1:
                continue
            mul_idx = int(concat_users[0])
            mul_op = model_ir.operators[mul_idx]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_in0 = str(mul_op.inputs[0])
            mul_in1 = str(mul_op.inputs[1])
            if mul_in0 == concat_out_name:
                mul_side_index = 1
                mul_side_name = mul_in1
            elif mul_in1 == concat_out_name:
                mul_side_index = 0
                mul_side_name = mul_in0
            else:
                continue
            mul_side_tensor = model_ir.tensors.get(mul_side_name, None)
            if mul_side_tensor is None or mul_side_tensor.data is None:
                continue
            mul_out_name = str(mul_op.outputs[0])
            if mul_out_name in model_outputs:
                continue

            mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
            if len(mul_users) != 1:
                continue
            add2_idx = int(mul_users[0])
            add2_op = model_ir.operators[add2_idx]
            if str(add2_op.op_type) != "ADD" or len(add2_op.inputs) != 2 or len(add2_op.outputs) != 1:
                continue
            add2_in0 = str(add2_op.inputs[0])
            add2_in1 = str(add2_op.inputs[1])
            if add2_in0 == mul_out_name:
                add2_side_index = 1
                add2_side_name = add2_in1
            elif add2_in1 == mul_out_name:
                add2_side_index = 0
                add2_side_name = add2_in0
            else:
                continue
            add2_side_tensor = model_ir.tensors.get(add2_side_name, None)
            if add2_side_tensor is None or add2_side_tensor.data is None:
                continue
            add2_out_name = str(add2_op.outputs[0])
            if add2_out_name in model_outputs:
                continue

            add2_users = [int(v) for v in consumers.get(add2_out_name, [])]
            if len(add2_users) != 1:
                continue
            post_idx = int(add2_users[0])
            post_op = model_ir.operators[post_idx]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != add2_out_name
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue
            post_out_name = str(post_op.outputs[0])
            if post_out_name in model_outputs:
                continue

            # Apply rewrites atomically after full validation.
            for add_idx, new_inputs in add_rewrites:
                add_op = model_ir.operators[int(add_idx)]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=add_op,
                    new_inputs=[str(v) for v in new_inputs],
                )
                if len(add_op.outputs) == 1:
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(add_op.outputs[0]), None),
                        perm_nchw_to_nhwc,
                    )

            if np.asarray(mul_side_tensor.data).ndim == 4:
                mul_nhwc = np.transpose(np.asarray(mul_side_tensor.data), perm_nchw_to_nhwc).astype(
                    np.asarray(mul_side_tensor.data).dtype, copy=False
                )
                mul_side_tensor.data = np.asarray(mul_nhwc)
                mul_side_tensor.shape = [int(v) for v in list(mul_nhwc.shape)]
                mul_side_tensor.shape_signature = [int(v) for v in list(mul_nhwc.shape)]
            if np.asarray(add2_side_tensor.data).ndim == 4:
                add2_nhwc = np.transpose(np.asarray(add2_side_tensor.data), perm_nchw_to_nhwc).astype(
                    np.asarray(add2_side_tensor.data).dtype, copy=False
                )
                add2_side_tensor.data = np.asarray(add2_nhwc)
                add2_side_tensor.shape = [int(v) for v in list(add2_nhwc.shape)]
                add2_side_tensor.shape_signature = [int(v) for v in list(add2_nhwc.shape)]

            concat_op.options["axis"] = 3
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(concat_out_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(mul_out_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(add2_out_name, None),
                perm_nchw_to_nhwc,
            )

            _set_operator_outputs(
                model_ir=model_ir,
                op=add2_op,
                new_outputs=[post_out_name],
            )

            old_add2_tensor = model_ir.tensors.get(add2_out_name, None)
            post_tensor = model_ir.tensors.get(post_out_name, None)
            if old_add2_tensor is not None and post_tensor is not None:
                post_tensor.dtype = str(old_add2_tensor.dtype)
                post_tensor.quantization = _clone_quantization(old_add2_tensor.quantization)
                post_tensor.shape = [int(v) for v in list(old_add2_tensor.shape)]
                post_tensor.shape_signature = (
                    [int(v) for v in list(old_add2_tensor.shape_signature)]
                    if old_add2_tensor.shape_signature is not None
                    else [int(v) for v in list(old_add2_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = set(int(v) for v in branch_pre_remove_indices)
            remove_indices.add(int(post_idx))
            # Remove base transpose only when it became dead.
            base_nchw_name = str(model_ir.operators[int(base_pre_idx)].outputs[0])
            remaining_users = [int(v) for v in consumers.get(base_nchw_name, [])]
            if set(int(v) for v in remaining_users).issubset(set(int(idx) for idx, _ in add_rewrites)):
                remove_indices.add(int(base_pre_idx))

            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_add_concat_const_suffix_nhwc_chains": int(optimized)}

