from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _invert_perm,
    _is_per_tensor_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.layout_transpose import _is_inverse_perm


def optimize_transpose_quant_dequant_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose-quantize/dequantize-transpose bridges when both transposes cancel.

    Target patterns:
      X --Transpose(P)--> A --QUANTIZE--> B --Transpose(inv(P))--> Y
      X --Transpose(P)--> A --DEQUANTIZE--> B --Transpose(inv(P))--> Y

    This is safe for per-tensor quantization only.
    """
    removed_bridge_pairs = 0
    rewritten_add_qdq_residual_bridges = 0
    rewritten_mixed_add_qdq_residual_bridges = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        # Pattern A:
        #   X -T(P)-> A -Q-> B -DQ-> C -T(invP)-> Y
        # -> X -Q-> B -DQ-> Y
        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            bridge_a = pre_op.outputs[0]
            q_users = consumers.get(bridge_a, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if q_op.inputs[0] != bridge_a:
                continue

            bridge_b = q_op.outputs[0]
            dq_users = consumers.get(bridge_b, [])
            if len(dq_users) != 1:
                continue
            dq_idx = int(dq_users[0])
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if dq_op.inputs[0] != bridge_b:
                continue

            bridge_c = dq_op.outputs[0]
            post_users = consumers.get(bridge_c, [])
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if post_op.inputs[0] != bridge_c:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_pre is None or perm_post is None:
                continue
            if not _is_inverse_perm(perm_pre, perm_post):
                continue

            q_out_tensor = model_ir.tensors.get(bridge_b, None)
            if q_out_tensor is None or not _is_per_tensor_quantization(q_out_tensor.quantization):
                continue

            if bridge_a in model_ir.outputs or bridge_b in model_ir.outputs or bridge_c in model_ir.outputs:
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=q_op,
                new_inputs=[pre_op.inputs[0]],
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=dq_op,
                new_outputs=[post_op.outputs[0]],
            )

            new_dq_out = model_ir.tensors.get(post_op.outputs[0], None)
            old_dq_out = model_ir.tensors.get(bridge_c, None)
            if new_dq_out is not None and old_dq_out is not None:
                new_dq_out.dtype = str(old_dq_out.dtype)

            for remove_idx in sorted([pre_idx, post_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_bridge_pairs += 1
            changed = True
            break

        if changed:
            continue

        # Pattern B:
        #   X -T(P)-> A -(Q|DQ)-> B -T(invP)-> Y
        # -> X -(Q|DQ)-> Y
        for mid_idx, mid_op in enumerate(model_ir.operators):
            if str(mid_op.op_type) not in {"QUANTIZE", "DEQUANTIZE"}:
                continue
            if len(mid_op.inputs) != 1 or len(mid_op.outputs) != 1:
                continue

            bridge_in = mid_op.inputs[0]
            bridge_out = mid_op.outputs[0]

            pre_idx = producers.get(bridge_in, None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[pre_idx]
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.outputs) != 1 or len(pre_op.inputs) < 2:
                continue
            if pre_op.outputs[0] != bridge_in:
                continue
            in_users = set(consumers.get(bridge_in, []))
            if int(mid_idx) not in in_users:
                continue
            can_remove_pre = in_users == {int(mid_idx)}

            post_users = [int(v) for v in consumers.get(bridge_out, []) if int(v) != int(mid_idx)]
            if len(post_users) == 0:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            valid_posts = True
            for post_idx in post_users:
                post_op = model_ir.operators[int(post_idx)]
                if (
                    str(post_op.op_type) != "TRANSPOSE"
                    or len(post_op.inputs) < 2
                    or len(post_op.outputs) != 1
                    or post_op.inputs[0] != bridge_out
                ):
                    valid_posts = False
                    break
                perm_post = _read_transpose_perm(model_ir, post_op)
                if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                    valid_posts = False
                    break
                post_indices.append(int(post_idx))
                post_output_names.append(str(post_op.outputs[0]))
            if not valid_posts or len(post_indices) == 0:
                continue

            # Keep visible output names stable.
            if bridge_out in model_ir.outputs:
                continue
            if can_remove_pre and bridge_in in model_ir.outputs:
                continue
            single_post = len(post_indices) == 1
            if (not single_post) and any(
                post_output_name in model_ir.outputs for post_output_name in post_output_names
            ):
                continue

            pre_input_name = pre_op.inputs[0]
            representative_post_output_name = post_output_names[0]

            if str(mid_op.op_type) == "QUANTIZE":
                bridge_out_tensor = model_ir.tensors.get(bridge_out, None)
                if bridge_out_tensor is None:
                    continue
                if not _is_per_tensor_quantization(bridge_out_tensor.quantization):
                    continue
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mid_op,
                    new_inputs=[pre_input_name],
                )
                if single_post:
                    post_output_name = representative_post_output_name
                    _set_operator_outputs(
                        model_ir=model_ir,
                        op=mid_op,
                        new_outputs=[post_output_name],
                    )
                    post_output_tensor = model_ir.tensors.get(post_output_name, None)
                    if post_output_tensor is not None:
                        post_output_tensor.dtype = str(bridge_out_tensor.dtype)
                        post_output_tensor.quantization = _clone_quantization(
                            bridge_out_tensor.quantization
                        )
                else:
                    rep_post_tensor = model_ir.tensors.get(representative_post_output_name, None)
                    if rep_post_tensor is not None:
                        bridge_out_tensor.shape = list(rep_post_tensor.shape)
                        bridge_out_tensor.shape_signature = (
                            list(rep_post_tensor.shape_signature)
                            if rep_post_tensor.shape_signature is not None
                            else list(rep_post_tensor.shape)
                        )
                        bridge_out_tensor.dtype = str(rep_post_tensor.dtype)
                    bridge_out_tensor.quantization = _clone_quantization(
                        bridge_out_tensor.quantization
                    )
            else:
                bridge_in_tensor = model_ir.tensors.get(bridge_in, None)
                if bridge_in_tensor is None:
                    continue
                if not _is_per_tensor_quantization(bridge_in_tensor.quantization):
                    continue
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mid_op,
                    new_inputs=[pre_input_name],
                )
                if single_post:
                    post_output_name = representative_post_output_name
                    _set_operator_outputs(
                        model_ir=model_ir,
                        op=mid_op,
                        new_outputs=[post_output_name],
                    )
                    pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
                    if pre_input_tensor is not None and pre_input_tensor.quantization is None:
                        pre_input_tensor.quantization = _clone_quantization(
                            bridge_in_tensor.quantization
                        )
                    post_output_tensor = model_ir.tensors.get(post_output_name, None)
                    bridge_out_tensor = model_ir.tensors.get(bridge_out, None)
                    if post_output_tensor is not None and bridge_out_tensor is not None:
                        post_output_tensor.dtype = str(bridge_out_tensor.dtype)
                else:
                    pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
                    if pre_input_tensor is not None and pre_input_tensor.quantization is None:
                        pre_input_tensor.quantization = _clone_quantization(
                            bridge_in_tensor.quantization
                        )
                    bridge_out_tensor = model_ir.tensors.get(bridge_out, None)
                    rep_post_tensor = model_ir.tensors.get(representative_post_output_name, None)
                    if bridge_out_tensor is not None and rep_post_tensor is not None:
                        bridge_out_tensor.shape = list(rep_post_tensor.shape)
                        bridge_out_tensor.shape_signature = (
                            list(rep_post_tensor.shape_signature)
                            if rep_post_tensor.shape_signature is not None
                            else list(rep_post_tensor.shape)
                        )
                        bridge_out_tensor.dtype = str(rep_post_tensor.dtype)

            if not single_post:
                for post_output_name in post_output_names:
                    _replace_tensor_inputs(model_ir, post_output_name, bridge_out)

            remove_indices = list(post_indices)
            if can_remove_pre:
                remove_indices.append(pre_idx)
            for remove_idx in sorted(remove_indices, reverse=True):
                del model_ir.operators[remove_idx]
            removed_bridge_pairs += int(1 if single_post else max(1, len(post_indices)))
            changed = True
            break

        if changed:
            continue

        # Pattern C:
        #   a_nhwc -T(n2c)-> a_nchw -Q-> aq -DQ-> adq -(optional T(c2n))*->
        #   b_nhwc -T(n2c)-> b_nchw -Q-> bq -DQ-> bdq
        #   ADD(adq, bdq) -> sum_nchw -Q-> sum_q -DQ-> sum_dq -T(c2n)-> sum_nhwc
        # -> Remove the NCHW adapters by keeping Q/DQ and ADD in NHWC.
        #
        # Notes:
        # - This intentionally targets strict residual branches where ADD inputs are
        #   Q/DQ-dequantized tensors produced from pre-transpose wrappers.
        # - Side fanout from branch DQ outputs is allowed only via inverse post-transpose.
        model_outputs = set(str(v) for v in model_ir.outputs)
        for add_idx, add_op in enumerate(model_ir.operators):
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue

            add_out_name = str(add_op.outputs[0])
            if add_out_name in model_outputs:
                continue

            branch_specs: List[Dict[str, Any]] = []
            common_perm_post: Optional[List[int]] = None
            valid_add_branches = True

            for add_input_name in [str(v) for v in list(add_op.inputs)]:
                if add_input_name in model_outputs:
                    valid_add_branches = False
                    break

                dq_idx = producers.get(add_input_name, None)
                if dq_idx is None:
                    valid_add_branches = False
                    break
                dq_op = model_ir.operators[int(dq_idx)]
                if (
                    str(dq_op.op_type) != "DEQUANTIZE"
                    or len(dq_op.inputs) != 1
                    or len(dq_op.outputs) != 1
                    or str(dq_op.outputs[0]) != add_input_name
                ):
                    valid_add_branches = False
                    break

                q_out_name = str(dq_op.inputs[0])
                q_idx = producers.get(q_out_name, None)
                if q_idx is None:
                    valid_add_branches = False
                    break
                q_op = model_ir.operators[int(q_idx)]
                if (
                    str(q_op.op_type) != "QUANTIZE"
                    or len(q_op.inputs) != 1
                    or len(q_op.outputs) != 1
                    or str(q_op.outputs[0]) != q_out_name
                ):
                    valid_add_branches = False
                    break

                q_out_tensor = model_ir.tensors.get(q_out_name, None)
                if q_out_tensor is None or not _is_per_tensor_quantization(q_out_tensor.quantization):
                    valid_add_branches = False
                    break

                pre_in_name = str(q_op.inputs[0])
                pre_idx = producers.get(pre_in_name, None)
                if pre_idx is None:
                    valid_add_branches = False
                    break
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != pre_in_name
                ):
                    valid_add_branches = False
                    break

                perm_pre = _read_transpose_perm(model_ir, pre_op)
                if perm_pre is None:
                    valid_add_branches = False
                    break
                perm_post_expected = _invert_perm(perm_pre)
                if perm_post_expected is None:
                    valid_add_branches = False
                    break
                if common_perm_post is None:
                    common_perm_post = [int(v) for v in list(perm_post_expected)]
                elif list(common_perm_post) != list(perm_post_expected):
                    valid_add_branches = False
                    break

                pre_output_name = str(pre_op.outputs[0])
                pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
                can_remove_pre = set(pre_users) == {int(q_idx)}

                branch_post_indices: List[int] = []
                branch_post_output_names: List[str] = []
                branch_other_users = [
                    int(v)
                    for v in consumers.get(add_input_name, [])
                    if int(v) != int(add_idx)
                ]
                for user_idx in branch_other_users:
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) != "TRANSPOSE"
                        or len(user_op.inputs) < 2
                        or len(user_op.outputs) != 1
                        or str(user_op.inputs[0]) != add_input_name
                    ):
                        valid_add_branches = False
                        break
                    user_perm = _read_transpose_perm(model_ir, user_op)
                    if user_perm is None or list(user_perm) != list(perm_post_expected):
                        valid_add_branches = False
                        break
                    branch_post_out_name = str(user_op.outputs[0])
                    if branch_post_out_name in model_outputs:
                        valid_add_branches = False
                        break
                    branch_post_indices.append(int(user_idx))
                    branch_post_output_names.append(branch_post_out_name)
                if not valid_add_branches:
                    break

                branch_specs.append(
                    {
                        "pre_idx": int(pre_idx),
                        "q_idx": int(q_idx),
                        "dq_idx": int(dq_idx),
                        "pre_input_name": str(pre_op.inputs[0]),
                        "q_out_name": str(q_out_name),
                        "dq_out_name": str(add_input_name),
                        "perm_post": [int(v) for v in list(perm_post_expected)],
                        "post_indices": [int(v) for v in branch_post_indices],
                        "post_output_names": [str(v) for v in branch_post_output_names],
                        "can_remove_pre": bool(can_remove_pre),
                    }
                )

            if not valid_add_branches or len(branch_specs) != 2 or common_perm_post is None:
                continue

            add_users = [
                int(v)
                for v in consumers.get(add_out_name, [])
                if int(v) != int(add_idx)
            ]
            if len(add_users) != 1:
                continue

            out_q_idx = int(add_users[0])
            out_q_op = model_ir.operators[int(out_q_idx)]
            if str(out_q_op.op_type) != "QUANTIZE" or len(out_q_op.inputs) != 1 or len(out_q_op.outputs) != 1:
                continue
            if str(out_q_op.inputs[0]) != add_out_name:
                continue

            out_q_name = str(out_q_op.outputs[0])
            out_q_tensor = model_ir.tensors.get(out_q_name, None)
            if out_q_tensor is None or not _is_per_tensor_quantization(out_q_tensor.quantization):
                continue
            if out_q_name in model_outputs:
                continue

            out_q_users = [int(v) for v in consumers.get(out_q_name, []) if int(v) != int(out_q_idx)]
            if len(out_q_users) != 1:
                continue

            out_dq_idx = int(out_q_users[0])
            out_dq_op = model_ir.operators[int(out_dq_idx)]
            if (
                str(out_dq_op.op_type) != "DEQUANTIZE"
                or len(out_dq_op.inputs) != 1
                or len(out_dq_op.outputs) != 1
                or str(out_dq_op.inputs[0]) != out_q_name
            ):
                continue

            out_dq_name = str(out_dq_op.outputs[0])
            if out_dq_name in model_outputs:
                continue

            out_post_indices: List[int] = []
            out_post_output_names: List[str] = []
            out_legacy_user_indices: List[int] = []
            invalid_output_tail = False
            out_dq_users = [
                int(v)
                for v in consumers.get(out_dq_name, [])
                if int(v) != int(out_dq_idx)
            ]
            for user_idx in out_dq_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == out_dq_name
                ):
                    user_perm = _read_transpose_perm(model_ir, user_op)
                    if user_perm is not None and list(user_perm) == list(common_perm_post):
                        out_post_name = str(user_op.outputs[0])
                        if out_post_name in model_outputs:
                            invalid_output_tail = True
                            break
                        out_post_indices.append(int(user_idx))
                        out_post_output_names.append(out_post_name)
                        continue
                out_legacy_user_indices.append(int(user_idx))

            if invalid_output_tail:
                continue
            if len(out_legacy_user_indices) > 0 and len(out_post_indices) == 0:
                continue

            remove_indices: set[int] = set()
            removed_transposes_local = 0
            out_dq_tensor = model_ir.tensors.get(out_dq_name, None)
            out_dq_shape_before_permute = (
                [int(v) for v in list(out_dq_tensor.shape)]
                if out_dq_tensor is not None and out_dq_tensor.shape is not None
                else None
            )
            out_dq_signature_before_permute = (
                [int(v) for v in list(out_dq_tensor.shape_signature)]
                if out_dq_tensor is not None and out_dq_tensor.shape_signature is not None
                else (
                    [int(v) for v in list(out_dq_tensor.shape)]
                    if out_dq_tensor is not None and out_dq_tensor.shape is not None
                    else None
                )
            )

            for branch_spec in branch_specs:
                q_op = model_ir.operators[int(branch_spec["q_idx"])]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=q_op,
                    new_inputs=[str(branch_spec["pre_input_name"])],
                )

                q_tensor = model_ir.tensors.get(str(branch_spec["q_out_name"]), None)
                dq_tensor = model_ir.tensors.get(str(branch_spec["dq_out_name"]), None)
                post_output_names = [str(v) for v in list(branch_spec["post_output_names"])]
                if len(post_output_names) > 0:
                    representative_post_tensor = model_ir.tensors.get(post_output_names[0], None)
                    if dq_tensor is not None and representative_post_tensor is not None:
                        dq_tensor.shape = [int(v) for v in list(representative_post_tensor.shape)]
                        dq_tensor.shape_signature = (
                            [int(v) for v in list(representative_post_tensor.shape_signature)]
                            if representative_post_tensor.shape_signature is not None
                            else [int(v) for v in list(representative_post_tensor.shape)]
                        )
                    if q_tensor is not None and dq_tensor is not None:
                        q_tensor.shape = [int(v) for v in list(dq_tensor.shape)]
                        q_tensor.shape_signature = (
                            [int(v) for v in list(dq_tensor.shape_signature)]
                            if dq_tensor.shape_signature is not None
                            else [int(v) for v in list(dq_tensor.shape)]
                        )
                else:
                    _permute_tensor_metadata_if_rank_matches(
                        q_tensor,
                        [int(v) for v in list(branch_spec["perm_post"])],
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        dq_tensor,
                        [int(v) for v in list(branch_spec["perm_post"])],
                    )

                for post_output_name in post_output_names:
                    _replace_tensor_inputs(model_ir, post_output_name, str(branch_spec["dq_out_name"]))
                for post_idx in [int(v) for v in list(branch_spec["post_indices"])]:
                    remove_indices.add(int(post_idx))
                    removed_transposes_local += 1

                if bool(branch_spec["can_remove_pre"]):
                    remove_indices.add(int(branch_spec["pre_idx"]))
                    removed_transposes_local += 1

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(add_out_name, None),
                [int(v) for v in list(common_perm_post)],
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(out_q_name, None),
                [int(v) for v in list(common_perm_post)],
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(out_dq_name, None),
                [int(v) for v in list(common_perm_post)],
            )

            keep_post_idx: Optional[int] = None
            keep_post_output_name: Optional[str] = None
            perm_nhwc_to_nchw = _invert_perm(list(common_perm_post))
            if len(out_legacy_user_indices) > 0:
                keep_post_idx = int(out_post_indices[0])
                keep_post_op = model_ir.operators[int(keep_post_idx)]
                keep_post_output_name = str(keep_post_op.outputs[0])
                if perm_nhwc_to_nchw is None:
                    keep_post_idx = None
                    keep_post_output_name = None
                else:
                    keep_perm_name = str(keep_post_op.inputs[1])
                    keep_perm_tensor = model_ir.tensors.get(keep_perm_name, None)
                    if keep_perm_tensor is not None:
                        keep_perm_tensor.data = np.asarray(perm_nhwc_to_nchw, dtype=np.int32)
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=keep_post_op,
                        new_inputs=[out_dq_name, keep_perm_name],
                    )
                    _set_operator_outputs(
                        model_ir=model_ir,
                        op=keep_post_op,
                        new_outputs=[keep_post_output_name],
                    )
                    if keep_post_output_name is not None:
                        keep_post_tensor = model_ir.tensors.get(keep_post_output_name, None)
                        if (
                            keep_post_tensor is not None
                            and out_dq_shape_before_permute is not None
                            and out_dq_signature_before_permute is not None
                        ):
                            keep_post_tensor.shape = [int(v) for v in list(out_dq_shape_before_permute)]
                            keep_post_tensor.shape_signature = [
                                int(v) for v in list(out_dq_signature_before_permute)
                            ]
            if len(out_legacy_user_indices) > 0 and keep_post_output_name is None:
                continue

            for out_post_output_name in out_post_output_names:
                _replace_tensor_inputs(model_ir, out_post_output_name, out_dq_name)
            for legacy_user_idx in out_legacy_user_indices:
                if keep_post_output_name is None:
                    continue
                legacy_user_op = model_ir.operators[int(legacy_user_idx)]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=legacy_user_op,
                    new_inputs=[
                        keep_post_output_name if str(inp) == out_dq_name else str(inp)
                        for inp in list(legacy_user_op.inputs)
                    ],
                )
            for out_post_idx in out_post_indices:
                if keep_post_idx is not None and int(out_post_idx) == int(keep_post_idx):
                    continue
                remove_indices.add(int(out_post_idx))
                removed_transposes_local += 1

            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            removed_bridge_pairs += int(max(1, removed_transposes_local // 2))
            rewritten_add_qdq_residual_bridges += 1
            changed = True
            break

        if changed:
            continue

        # Pattern D:
        #   skip_nhwc --T(P)------------------------------> skip_nchw
        #   proj_nhwc --T(P)--> QUANTIZE -> DEQUANTIZE --> proj_nchw
        #   ADD(proj_nchw, skip_nchw) -> y_nchw -> QUANTIZE -> DEQUANTIZE -> T(invP) -> y_nhwc
        # -> Keep Q/DQ and ADD in NHWC by bypassing skip/proj pre-transposes and final post-transpose.
        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            post_input_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if post_input_name in model_ir.outputs:
                continue

            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None:
                continue
            perm_pre_expected = _invert_perm(perm_post)
            if perm_pre_expected is None:
                continue

            dq_out_name = post_input_name
            dq_idx = producers.get(dq_out_name, None)
            if dq_idx is None:
                continue
            dq_op = model_ir.operators[int(dq_idx)]
            if (
                str(dq_op.op_type) != "DEQUANTIZE"
                or len(dq_op.inputs) != 1
                or len(dq_op.outputs) != 1
                or str(dq_op.outputs[0]) != dq_out_name
            ):
                continue

            dq_out_users = [int(v) for v in consumers.get(dq_out_name, []) if int(v) != int(dq_idx)]
            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_user_indices: List[int] = []
            valid_post_fanout = True
            for user_idx in dq_out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == dq_out_name
                ):
                    user_perm = _read_transpose_perm(model_ir, user_op)
                    if user_perm is not None and list(user_perm) == list(perm_post):
                        post_indices.append(int(user_idx))
                        post_output_names.append(str(user_op.outputs[0]))
                        continue
                legacy_user_indices.append(int(user_idx))
            if not valid_post_fanout:
                continue
            if int(post_idx) not in set(int(v) for v in post_indices):
                continue
            if len(post_indices) == 0:
                continue

            out_q_name = str(dq_op.inputs[0])
            out_q_idx = producers.get(out_q_name, None)
            if out_q_idx is None:
                continue
            out_q_op = model_ir.operators[int(out_q_idx)]
            if (
                str(out_q_op.op_type) != "QUANTIZE"
                or len(out_q_op.inputs) != 1
                or len(out_q_op.outputs) != 1
                or str(out_q_op.outputs[0]) != out_q_name
            ):
                continue

            out_q_tensor = model_ir.tensors.get(out_q_name, None)
            if out_q_tensor is None or not _is_per_tensor_quantization(out_q_tensor.quantization):
                continue
            add_out_name = str(out_q_op.inputs[0])
            if add_out_name in model_ir.outputs:
                continue

            add_idx = producers.get(add_out_name, None)
            if add_idx is None:
                continue
            add_op = model_ir.operators[int(add_idx)]
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            if str(add_op.outputs[0]) != add_out_name:
                continue

            proj_branch: Optional[Dict[str, Any]] = None
            skip_branch: Optional[Dict[str, Any]] = None
            valid_inputs = True

            for add_input_name in [str(v) for v in list(add_op.inputs)]:
                add_input_prod_idx = producers.get(add_input_name, None)
                if add_input_prod_idx is None:
                    valid_inputs = False
                    break
                add_input_prod_op = model_ir.operators[int(add_input_prod_idx)]
                add_input_prod_type = str(add_input_prod_op.op_type)

                if add_input_prod_type == "DEQUANTIZE":
                    if (
                        len(add_input_prod_op.inputs) != 1
                        or len(add_input_prod_op.outputs) != 1
                        or str(add_input_prod_op.outputs[0]) != add_input_name
                    ):
                        valid_inputs = False
                        break
                    in_q_name = str(add_input_prod_op.inputs[0])
                    in_q_idx = producers.get(in_q_name, None)
                    if in_q_idx is None:
                        valid_inputs = False
                        break
                    in_q_op = model_ir.operators[int(in_q_idx)]
                    if (
                        str(in_q_op.op_type) != "QUANTIZE"
                        or len(in_q_op.inputs) != 1
                        or len(in_q_op.outputs) != 1
                        or str(in_q_op.outputs[0]) != in_q_name
                    ):
                        valid_inputs = False
                        break
                    in_q_tensor = model_ir.tensors.get(in_q_name, None)
                    if in_q_tensor is None or not _is_per_tensor_quantization(in_q_tensor.quantization):
                        valid_inputs = False
                        break

                    pre_name = str(in_q_op.inputs[0])
                    pre_idx = producers.get(pre_name, None)
                    if pre_idx is None:
                        valid_inputs = False
                        break
                    pre_op = model_ir.operators[int(pre_idx)]
                    if (
                        str(pre_op.op_type) != "TRANSPOSE"
                        or len(pre_op.inputs) < 2
                        or len(pre_op.outputs) != 1
                        or str(pre_op.outputs[0]) != pre_name
                    ):
                        valid_inputs = False
                        break
                    pre_perm = _read_transpose_perm(model_ir, pre_op)
                    if pre_perm is None or list(pre_perm) != list(perm_pre_expected):
                        valid_inputs = False
                        break

                    dq_input_users = [int(v) for v in consumers.get(add_input_name, [])]
                    if set(dq_input_users) != {int(add_idx)}:
                        valid_inputs = False
                        break

                    pre_output_name = str(pre_op.outputs[0])
                    pre_output_users = [int(v) for v in consumers.get(pre_output_name, [])]
                    can_remove_pre = set(pre_output_users) == {int(in_q_idx)}

                    if proj_branch is not None:
                        valid_inputs = False
                        break
                    proj_branch = {
                        "pre_idx": int(pre_idx),
                        "q_idx": int(in_q_idx),
                        "dq_name": str(add_input_name),
                        "pre_input_name": str(pre_op.inputs[0]),
                        "can_remove_pre": bool(can_remove_pre),
                    }
                    continue

                if add_input_prod_type == "TRANSPOSE":
                    if (
                        len(add_input_prod_op.inputs) < 2
                        or len(add_input_prod_op.outputs) != 1
                        or str(add_input_prod_op.outputs[0]) != add_input_name
                    ):
                        valid_inputs = False
                        break
                    pre_perm = _read_transpose_perm(model_ir, add_input_prod_op)
                    if pre_perm is None or list(pre_perm) != list(perm_pre_expected):
                        valid_inputs = False
                        break

                    skip_users = [int(v) for v in consumers.get(add_input_name, [])]
                    if set(skip_users) != {int(add_idx)}:
                        valid_inputs = False
                        break

                    if skip_branch is not None:
                        valid_inputs = False
                        break
                    skip_branch = {
                        "pre_idx": int(add_input_prod_idx),
                        "pre_input_name": str(add_input_prod_op.inputs[0]),
                        "skip_name": str(add_input_name),
                    }
                    continue

                valid_inputs = False
                break

            if (not valid_inputs) or proj_branch is None or skip_branch is None:
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=model_ir.operators[int(proj_branch["q_idx"])],
                new_inputs=[str(proj_branch["pre_input_name"])],
            )

            new_add_inputs = [
                str(skip_branch["pre_input_name"]) if str(v) == str(skip_branch["skip_name"]) else str(v)
                for v in list(add_op.inputs)
            ]
            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=new_add_inputs,
            )

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(proj_branch["dq_name"]), None),
                list(perm_post),
            )
            proj_q_op = model_ir.operators[int(proj_branch["q_idx"])]
            if len(proj_q_op.outputs) == 1:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(proj_q_op.outputs[0]), None),
                    list(perm_post),
                )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(add_out_name, None),
                list(perm_post),
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(out_q_name, None),
                list(perm_post),
            )
            canonical_post_idx = int(post_idx)
            canonical_post_output_name = str(post_output_name)

            _set_operator_outputs(
                model_ir=model_ir,
                op=dq_op,
                new_outputs=[canonical_post_output_name],
            )
            new_dq_out_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            old_dq_out_tensor = model_ir.tensors.get(dq_out_name, None)
            if new_dq_out_tensor is not None and old_dq_out_tensor is not None:
                new_dq_out_tensor.dtype = str(old_dq_out_tensor.dtype)

            remove_indices = {int(skip_branch["pre_idx"])}
            if bool(proj_branch["can_remove_pre"]):
                remove_indices.add(int(proj_branch["pre_idx"]))
            if len(legacy_user_indices) > 0:
                keep_post_op = model_ir.operators[int(canonical_post_idx)]
                keep_perm_name = str(keep_post_op.inputs[1])
                keep_perm_tensor = model_ir.tensors.get(keep_perm_name, None)
                if keep_perm_tensor is not None:
                    keep_perm_tensor.data = np.asarray(list(perm_pre_expected), dtype=np.int32)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_inputs=[canonical_post_output_name, keep_perm_name],
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_outputs=[dq_out_name],
                )
                for post_output in post_output_names:
                    if str(post_output) == canonical_post_output_name:
                        continue
                    _replace_tensor_inputs(model_ir, str(post_output), canonical_post_output_name)
                for idx_rm in post_indices:
                    if int(idx_rm) == int(canonical_post_idx):
                        continue
                    remove_indices.add(int(idx_rm))
            else:
                for post_output in post_output_names:
                    if str(post_output) == canonical_post_output_name:
                        continue
                    _replace_tensor_inputs(model_ir, str(post_output), canonical_post_output_name)
                for idx_rm in post_indices:
                    remove_indices.add(int(idx_rm))

            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            rewritten_mixed_add_qdq_residual_bridges += 1
            removed_bridge_pairs += int(1)
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_quantize_dequantize_bridges": int(removed_bridge_pairs),
        "rewritten_add_qdq_residual_transpose_bridges": int(rewritten_add_qdq_residual_bridges),
        "rewritten_mixed_add_qdq_residual_transpose_bridges": int(
            rewritten_mixed_add_qdq_residual_bridges
        ),
    }
