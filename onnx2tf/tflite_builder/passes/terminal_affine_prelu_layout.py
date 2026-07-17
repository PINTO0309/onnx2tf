from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Terminal rewrite for:
      TRANSPOSE(NHWC->NCHW) -> MUL(const) -> ADD(const) -> PRELU(const) -> TRANSPOSE(NCHW->NHWC)

    This pass is intentionally strict and runs only in the terminal stage to avoid
    interacting with earlier graph-wide transpose optimizers.
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

    def _ensure_const_nhwc_channel_last(
        *,
        tensor_name: str,
        chain_index_set: set[int],
        consumers: Dict[str, List[int]],
    ) -> Optional[str]:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None or tensor.data is None:
            return None
        data = np.asarray(tensor.data)
        if int(data.size) == 1:
            return str(tensor_name)
        if data.ndim != 4:
            return None
        shape = [int(v) for v in list(data.shape)]
        nhwc_data: Optional[np.ndarray] = None
        if (
            int(shape[0]) == 1
            and int(shape[1]) == 1
            and int(shape[2]) == 1
            and int(shape[3]) > 0
        ):
            nhwc_data = np.asarray(data)
        elif (
            int(shape[0]) == 1
            and int(shape[1]) > 0
            and int(shape[2]) == 1
            and int(shape[3]) == 1
        ):
            nhwc_data = np.transpose(data, perm_nchw_to_nhwc).astype(data.dtype, copy=False)
        else:
            return None

        side_users = [int(v) for v in consumers.get(str(tensor_name), [])]
        shared_outside_chain = any(int(u) not in chain_index_set for u in side_users)
        if shared_outside_chain:
            cloned_name = _unique_tensor_name(f"{tensor_name}_nhwc")
            model_ir.tensors[cloned_name] = TensorIR(
                name=cloned_name,
                dtype=str(tensor.dtype),
                shape=[int(v) for v in list(nhwc_data.shape)],
                shape_signature=[int(v) for v in list(nhwc_data.shape)],
                data=np.asarray(nhwc_data),
                is_variable=False,
                quantization=_clone_quantization(tensor.quantization),
            )
            return str(cloned_name)

        tensor.data = np.asarray(nhwc_data)
        tensor.shape = [int(v) for v in list(nhwc_data.shape)]
        tensor.shape_signature = [int(v) for v in list(nhwc_data.shape)]
        return str(tensor_name)

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)  # noqa: F841
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue
            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_input_name in model_outputs or pre_output_name in model_outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) == 0:
                continue
            for mul_idx in pre_users:
                mul_op = model_ir.operators[int(mul_idx)]
                if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                    continue
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                if mul_inputs[0] == pre_output_name:
                    mul_side_input_index = 1
                    mul_side_name = str(mul_inputs[1])
                elif mul_inputs[1] == pre_output_name:
                    mul_side_input_index = 0
                    mul_side_name = str(mul_inputs[0])
                else:
                    continue

                mul_out_name = str(mul_op.outputs[0])
                if mul_out_name in model_outputs:
                    continue
                mul_out_users = [int(v) for v in consumers.get(mul_out_name, []) if int(v) != int(mul_idx)]
                if len(mul_out_users) != 1:
                    continue
                add_idx = int(mul_out_users[0])
                add_op = model_ir.operators[int(add_idx)]
                if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                    continue
                add_inputs = [str(v) for v in list(add_op.inputs)]
                if add_inputs[0] == mul_out_name:
                    add_side_input_index = 1
                    add_side_name = str(add_inputs[1])
                elif add_inputs[1] == mul_out_name:
                    add_side_input_index = 0
                    add_side_name = str(add_inputs[0])
                else:
                    continue

                add_out_name = str(add_op.outputs[0])
                if add_out_name in model_outputs:
                    continue
                add_out_users = [int(v) for v in consumers.get(add_out_name, []) if int(v) != int(add_idx)]
                if len(add_out_users) != 1:
                    continue
                prelu_idx = int(add_out_users[0])
                prelu_op = model_ir.operators[int(prelu_idx)]
                if (
                    str(prelu_op.op_type) != "PRELU"
                    or len(prelu_op.inputs) != 2
                    or len(prelu_op.outputs) != 1
                    or str(prelu_op.inputs[0]) != add_out_name
                ):
                    continue
                prelu_side_name = str(prelu_op.inputs[1])
                prelu_out_name = str(prelu_op.outputs[0])
                if prelu_out_name in model_outputs:
                    continue

                prelu_users = [int(v) for v in consumers.get(prelu_out_name, []) if int(v) != int(prelu_idx)]
                if len(prelu_users) == 0:
                    continue
                post_indices: List[int] = []
                post_output_names: List[str] = []
                legacy_users: List[int] = []
                for user_idx in prelu_users:
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) == "TRANSPOSE"
                        and len(user_op.inputs) >= 2
                        and len(user_op.outputs) == 1
                        and str(user_op.inputs[0]) == prelu_out_name
                        and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                        and str(user_op.outputs[0]) not in model_outputs
                    ):
                        post_indices.append(int(user_idx))
                        post_output_names.append(str(user_op.outputs[0]))
                    else:
                        legacy_users.append(int(user_idx))
                if len(post_indices) == 0:
                    continue

                chain_index_set = {int(mul_idx), int(add_idx), int(prelu_idx)}
                mul_side_for_op = _ensure_const_nhwc_channel_last(
                    tensor_name=str(mul_side_name),
                    chain_index_set=chain_index_set,
                    consumers=consumers,
                )
                if mul_side_for_op is None:
                    continue
                add_side_for_op = _ensure_const_nhwc_channel_last(
                    tensor_name=str(add_side_name),
                    chain_index_set=chain_index_set,
                    consumers=consumers,
                )
                if add_side_for_op is None:
                    continue
                prelu_side_for_op = _ensure_const_nhwc_channel_last(
                    tensor_name=str(prelu_side_name),
                    chain_index_set=chain_index_set,
                    consumers=consumers,
                )
                if prelu_side_for_op is None:
                    continue

                mul_inputs[int(mul_side_input_index)] = str(mul_side_for_op)
                if mul_inputs[0] == pre_output_name:
                    mul_inputs[0] = str(pre_input_name)
                elif mul_inputs[1] == pre_output_name:
                    mul_inputs[1] = str(pre_input_name)
                else:
                    continue
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=mul_inputs,
                )
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=add_op,
                    input_index=int(add_side_input_index),
                    new_input_name=str(add_side_for_op),
                )
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=prelu_op,
                    input_index=1,
                    new_input_name=str(prelu_side_for_op),
                )

                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(mul_out_name, None),
                    perm_nchw_to_nhwc,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(add_out_name, None),
                    perm_nchw_to_nhwc,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(prelu_out_name, None),
                    perm_nchw_to_nhwc,
                )

                canonical_post_output_name = str(post_output_names[0])
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=prelu_op,
                    new_outputs=[canonical_post_output_name],
                )
                for alias_name in post_output_names[1:]:
                    _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name)

                old_prelu_tensor = model_ir.tensors.get(prelu_out_name, None)
                canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
                if old_prelu_tensor is not None and canonical_tensor is not None:
                    canonical_tensor.dtype = str(old_prelu_tensor.dtype)
                    canonical_tensor.quantization = _clone_quantization(old_prelu_tensor.quantization)
                    canonical_tensor.shape = [int(v) for v in list(old_prelu_tensor.shape)]
                    canonical_tensor.shape_signature = (
                        [int(v) for v in list(old_prelu_tensor.shape_signature)]
                        if old_prelu_tensor.shape_signature is not None
                        else [int(v) for v in list(old_prelu_tensor.shape)]
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        canonical_tensor,
                        perm_nchw_to_nhwc,
                    )

                if len(legacy_users) > 0:
                    keep_post_idx = int(post_indices[0])
                    keep_post_op = model_ir.operators[int(keep_post_idx)]
                    keep_perm_name = str(keep_post_op.inputs[1])
                    keep_perm_tensor = model_ir.tensors.get(keep_perm_name, None)
                    if keep_perm_tensor is not None:
                        keep_perm_tensor.data = np.asarray(perm_nhwc_to_nchw, dtype=np.int32)
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=keep_post_op,
                        new_inputs=[canonical_post_output_name, keep_perm_name],
                    )
                    _set_operator_outputs(
                        model_ir=model_ir,
                        op=keep_post_op,
                        new_outputs=[prelu_out_name],
                    )
                    post_remove_indices = [int(v) for v in post_indices[1:]]
                else:
                    post_remove_indices = [int(v) for v in post_indices]

                remove_indices = set(int(v) for v in post_remove_indices)
                pre_remaining_users = [int(v) for v in pre_users if int(v) != int(mul_idx)]
                if len(pre_remaining_users) == 0:
                    remove_indices.add(int(pre_idx))
                for remove_idx in sorted(list(remove_indices), reverse=True):
                    del model_ir.operators[int(remove_idx)]

                rewritten += 1
                changed = True
                break

            if changed:
                break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains": int(rewritten)}
