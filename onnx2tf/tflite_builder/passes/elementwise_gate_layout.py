from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _invert_perm,
    _is_scalar_like_tensor,
    _is_singleton_constant_tensor,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR

def _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around SUM/LOGISTIC/SUB + MUL + ADD merge blocks.

    Target:
      a_nhwc --T(0,3,1,2)--> a_nchw
      b_nhwc --T(0,3,1,2)--> b_nchw
      c_nhwc --T(0,3,1,2)--> c_nchw --SUM(axis=1, keepDims=True)--> s_nchw --LOGISTIC--> sig_nchw
      one-sigmoid branch: SUB(one, sig_nchw) -> sub_nchw (optional)
      MUL(sub_or_sig_nchw, a_nchw) -> m0_nchw
      MUL(sig_nchw, b_nchw)        -> m1_nchw
      ADD(m0_nchw, m1_nchw) -> y_nchw
      y_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      - Bypass all three pre-transposes (a/b/c) to NHWC.
      - Remap SUM reduction axes from NCHW to NHWC.
      - Keep SUM/LOGISTIC/SUB/MUL/ADD in NHWC.
      - Remove post-transpose by making ADD produce NHWC output directly.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        def _extract_mul_pre_gate(
            mul_idx: int,
        ) -> Optional[Tuple[int, str, str, str]]:
            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                return None
            pre_idx: Optional[int] = None
            pre_input_name: Optional[str] = None
            pre_output_name: Optional[str] = None
            gate_name: Optional[str] = None
            for input_name in list(mul_op.inputs):
                prod_idx = producers.get(str(input_name), None)
                if prod_idx is not None:
                    prod_op = model_ir.operators[int(prod_idx)]
                    if (
                        str(prod_op.op_type) == "TRANSPOSE"
                        and len(prod_op.inputs) >= 2
                        and len(prod_op.outputs) == 1
                        and str(prod_op.outputs[0]) == str(input_name)
                        and _read_transpose_perm(model_ir, prod_op) == perm_nhwc_to_nchw
                        and str(prod_op.inputs[0]) not in model_outputs
                        and str(prod_op.outputs[0]) not in model_outputs
                    ):
                        if pre_idx is not None:
                            return None
                        pre_idx = int(prod_idx)
                        pre_input_name = str(prod_op.inputs[0])
                        pre_output_name = str(prod_op.outputs[0])
                        continue
                if gate_name is not None:
                    return None
                gate_name = str(input_name)
            if pre_idx is None or pre_input_name is None or pre_output_name is None or gate_name is None:
                return None
            return int(pre_idx), str(pre_input_name), str(pre_output_name), str(gate_name)

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue
            add_output_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if add_output_name in model_outputs or post_output_name in model_outputs:
                continue

            add_idx = producers.get(add_output_name, None)
            if add_idx is None:
                continue
            add_op = model_ir.operators[int(add_idx)]
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            if str(add_op.outputs[0]) != add_output_name:
                continue

            add_inputs = [str(v) for v in list(add_op.inputs)]
            mul0_idx = producers.get(add_inputs[0], None)
            mul1_idx = producers.get(add_inputs[1], None)
            if mul0_idx is None or mul1_idx is None:
                continue
            if int(mul0_idx) == int(mul1_idx):
                continue

            mul0_info = _extract_mul_pre_gate(int(mul0_idx))
            mul1_info = _extract_mul_pre_gate(int(mul1_idx))
            if mul0_info is None or mul1_info is None:
                continue

            pre0_idx, pre0_in_name, pre0_out_name, gate0_name = mul0_info
            pre1_idx, pre1_in_name, pre1_out_name, gate1_name = mul1_info
            if int(pre0_idx) == int(pre1_idx):
                continue

            # Identify LOGISTIC gate and optional SUB(one, sigmoid) gate.
            logistic_idx: Optional[int] = None
            logistic_out_name: Optional[str] = None
            sub_idx: Optional[int] = None
            sub_out_name: Optional[str] = None

            def _try_resolve_gate_pair(
                g_logistic: str,
                g_other: str,
            ) -> Optional[Tuple[int, str, Optional[int], Optional[str]]]:
                g_logistic_prod = producers.get(str(g_logistic), None)
                if g_logistic_prod is None:
                    return None
                g_logistic_op = model_ir.operators[int(g_logistic_prod)]
                if str(g_logistic_op.op_type) != "LOGISTIC" or len(g_logistic_op.outputs) != 1:
                    return None
                if str(g_logistic_op.outputs[0]) != str(g_logistic):
                    return None
                if str(g_other) == str(g_logistic):
                    return int(g_logistic_prod), str(g_logistic), None, None
                g_other_prod = producers.get(str(g_other), None)
                if g_other_prod is None:
                    return None
                g_other_op = model_ir.operators[int(g_other_prod)]
                if str(g_other_op.op_type) != "SUB" or len(g_other_op.inputs) != 2 or len(g_other_op.outputs) != 1:
                    return None
                if str(g_other_op.outputs[0]) != str(g_other):
                    return None
                if str(g_logistic) not in {str(v) for v in list(g_other_op.inputs)}:
                    return None
                return int(g_logistic_prod), str(g_logistic), int(g_other_prod), str(g_other)

            resolved = _try_resolve_gate_pair(gate0_name, gate1_name)
            if resolved is None:
                resolved = _try_resolve_gate_pair(gate1_name, gate0_name)
            if resolved is None:
                continue
            logistic_idx, logistic_out_name, sub_idx, sub_out_name = resolved

            logistic_op = model_ir.operators[int(logistic_idx)]
            if len(logistic_op.inputs) != 1:
                continue
            sum_output_name = str(logistic_op.inputs[0])
            if sum_output_name in model_outputs:
                continue

            sum_idx = producers.get(sum_output_name, None)
            if sum_idx is None:
                continue
            sum_op = model_ir.operators[int(sum_idx)]
            if str(sum_op.op_type) != "SUM" or len(sum_op.inputs) < 2 or len(sum_op.outputs) != 1:
                continue
            if str(sum_op.outputs[0]) != sum_output_name:
                continue
            if str(sum_op.inputs[0]) in model_outputs:
                continue

            pre2_out_name = str(sum_op.inputs[0])
            pre2_idx = producers.get(pre2_out_name, None)
            if pre2_idx is None:
                continue
            pre2_op = model_ir.operators[int(pre2_idx)]
            if (
                str(pre2_op.op_type) != "TRANSPOSE"
                or len(pre2_op.inputs) < 2
                or len(pre2_op.outputs) != 1
                or str(pre2_op.outputs[0]) != pre2_out_name
                or _read_transpose_perm(model_ir, pre2_op) != perm_nhwc_to_nchw
            ):
                continue
            pre2_in_name = str(pre2_op.inputs[0])
            if pre2_in_name in model_outputs or pre2_out_name in model_outputs:
                continue

            # Pre-transpose outputs must be local to this chain.
            if set(int(v) for v in consumers.get(pre0_out_name, [])) != {int(mul0_idx)} and set(
                int(v) for v in consumers.get(pre0_out_name, [])
            ) != {int(mul1_idx)}:
                continue
            if set(int(v) for v in consumers.get(pre1_out_name, [])) != {int(mul0_idx)} and set(
                int(v) for v in consumers.get(pre1_out_name, [])
            ) != {int(mul1_idx)}:
                continue
            if set(int(v) for v in consumers.get(pre2_out_name, [])) != {int(sum_idx)}:
                continue

            # ADD output must fan out only to inverse post-transposes.
            add_out_users = [int(v) for v in consumers.get(add_output_name, [])]
            if len(add_out_users) == 0:
                continue
            post_indices: List[int] = []
            post_output_names: List[str] = []
            valid_posts = True
            for user_idx in add_out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == add_output_name
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                else:
                    valid_posts = False
                    break
            if not valid_posts or len(post_indices) == 0:
                continue

            # Remap SUM axes from NCHW to NHWC.
            sum_axes_name = str(sum_op.inputs[1])
            sum_axes_tensor = model_ir.tensors.get(sum_axes_name, None)
            sum_axes_vals = _read_const_ints_from_tensor(sum_axes_tensor)
            if sum_axes_vals is None or len(sum_axes_vals) == 0:
                continue
            rank = 4
            pre2_in_tensor = model_ir.tensors.get(pre2_in_name, None)
            if pre2_in_tensor is not None and pre2_in_tensor.shape is not None and len(pre2_in_tensor.shape) > 0:
                rank = int(len(pre2_in_tensor.shape))
            if rank != 4:
                continue
            old_to_new_axis = _invert_perm(perm_nchw_to_nhwc)
            if old_to_new_axis is None:
                continue
            mapped_axes: List[int] = []
            valid_axes = True
            for axis in sum_axes_vals:
                a = int(axis)
                if a < 0:
                    a += int(rank)
                if a < 0 or a >= int(rank):
                    valid_axes = False
                    break
                mapped_axes.append(int(old_to_new_axis[int(a)]))
            if not valid_axes or len(mapped_axes) == 0:
                continue
            _write_const_ints_to_tensor(sum_axes_tensor, [int(v) for v in mapped_axes])

            # Rewire SUM to NHWC source.
            sum_inputs = [str(v) for v in list(sum_op.inputs)]
            sum_inputs[0] = str(pre2_in_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=sum_op,
                new_inputs=sum_inputs,
                graph_index=graph_index,
            )

            # Rewire both MULs from transpose outputs to transpose inputs.
            for mul_idx, pre_out_name, pre_in_name in [
                (int(mul0_idx), str(pre0_out_name), str(pre0_in_name)),
                (int(mul1_idx), str(pre1_out_name), str(pre1_in_name)),
            ]:
                mul_op = model_ir.operators[int(mul_idx)]
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                replaced = False
                for in_idx, in_name in enumerate(list(mul_inputs)):
                    if str(in_name) == str(pre_out_name):
                        mul_inputs[in_idx] = str(pre_in_name)
                        replaced = True
                if not replaced:
                    valid_posts = False
                    break
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=mul_inputs,
                    graph_index=graph_index,
                )
            if not valid_posts:
                continue

            # Update metadata to NHWC across the rewritten chain.
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(sum_output_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(logistic_out_name), None),
                perm_nchw_to_nhwc,
            )
            if sub_out_name is not None:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(sub_out_name), None),
                    perm_nchw_to_nhwc,
                )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(model_ir.operators[int(mul0_idx)].outputs[0]), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(model_ir.operators[int(mul1_idx)].outputs[0]), None),
                perm_nchw_to_nhwc,
            )

            # Canonicalize ADD output to post-transpose output tensor.
            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=add_op,
                new_outputs=[canonical_post_output_name],
                graph_index=graph_index,
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name, graph_index=graph_index)

            old_add_out_tensor = model_ir.tensors.get(add_output_name, None)
            canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_add_out_tensor is not None and canonical_tensor is not None:
                canonical_tensor.dtype = str(old_add_out_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(old_add_out_tensor.quantization)
                canonical_tensor.shape = [int(v) for v in list(old_add_out_tensor.shape)]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(old_add_out_tensor.shape_signature)]
                    if old_add_out_tensor.shape_signature is not None
                    else [int(v) for v in list(old_add_out_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    canonical_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = {
                int(pre0_idx),
                int(pre1_idx),
                int(pre2_idx),
                *[int(v) for v in post_indices],
            }
            for remove_idx in sorted(remove_indices, reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_sum_logistic_muladd_prepost_nhwc_chains": int(rewritten)}

def _optimize_transpose_weighted_add_swish_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around weighted ADD + Swish chains.

    Target:
      a_nhwc --T(0,3,1,2)--> a_nchw --MUL(w0_scalar_like)--> wa_nchw
      b_nhwc --T(0,3,1,2)--> b_nchw --MUL(w1_scalar_like)--> wb_nchw
      ADD(wa_nchw, wb_nchw) -> s_nchw --LOGISTIC--> sig_nchw
      MUL(s_nchw, sig_nchw) -> y_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      - Bypass both pre-transposes to NHWC.
      - Keep ADD/LOGISTIC/MUL in NHWC.
      - Remove inverse post-transpose by making MUL emit NHWC directly.
      - Keep one adapter transpose only when legacy NCHW users exist.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        def _extract_weighted_mul(
            *,
            mul_idx: int,
            expected_output_name: str,
            add_idx: int,
        ) -> Optional[Tuple[int, str, str, str, str]]:
            mul_op = model_ir.operators[int(mul_idx)]
            if (
                str(mul_op.op_type) != "MUL"
                or len(mul_op.inputs) != 2
                or len(mul_op.outputs) != 1
                or str(mul_op.outputs[0]) != str(expected_output_name)
            ):
                return None

            inputs = [str(v) for v in list(mul_op.inputs)]
            pre_idx: Optional[int] = None
            pre_in_name: Optional[str] = None
            pre_out_name: Optional[str] = None
            side_name: Optional[str] = None
            for input_name in inputs:
                prod_idx = producers.get(str(input_name), None)
                if prod_idx is None:
                    if side_name is not None:
                        return None
                    side_name = str(input_name)
                    continue
                prod_op = model_ir.operators[int(prod_idx)]
                if (
                    str(prod_op.op_type) == "TRANSPOSE"
                    and len(prod_op.inputs) >= 2
                    and len(prod_op.outputs) == 1
                    and str(prod_op.outputs[0]) == str(input_name)
                    and _read_transpose_perm(model_ir, prod_op) == perm_nhwc_to_nchw
                    and str(prod_op.inputs[0]) not in model_outputs
                    and str(prod_op.outputs[0]) not in model_outputs
                ):
                    if pre_idx is not None:
                        return None
                    pre_idx = int(prod_idx)
                    pre_in_name = str(prod_op.inputs[0])
                    pre_out_name = str(prod_op.outputs[0])
                    continue
                if side_name is not None:
                    return None
                side_name = str(input_name)

            if (
                pre_idx is None
                or pre_in_name is None
                or pre_out_name is None
                or side_name is None
                or not _is_scalar_like_tensor(model_ir, str(side_name))
            ):
                return None

            if set(int(v) for v in consumers.get(str(expected_output_name), [])) != {int(add_idx)}:
                return None
            if int(mul_idx) not in set(int(v) for v in consumers.get(str(pre_out_name), [])):
                return None

            return int(pre_idx), str(pre_in_name), str(pre_out_name), str(side_name), str(expected_output_name)

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue

            swish_out_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if swish_out_name in model_outputs or post_output_name in model_outputs:
                continue

            swish_mul_idx = producers.get(swish_out_name, None)
            if swish_mul_idx is None:
                continue
            swish_mul_op = model_ir.operators[int(swish_mul_idx)]
            if (
                str(swish_mul_op.op_type) != "MUL"
                or len(swish_mul_op.inputs) != 2
                or len(swish_mul_op.outputs) != 1
                or str(swish_mul_op.outputs[0]) != swish_out_name
            ):
                continue

            swish_inputs = [str(v) for v in list(swish_mul_op.inputs)]
            logistic_idx: Optional[int] = None
            logistic_out_name: Optional[str] = None
            add_idx: Optional[int] = None
            add_out_name: Optional[str] = None

            for input_name in swish_inputs:
                prod_idx = producers.get(str(input_name), None)
                if prod_idx is None:
                    continue
                prod_op = model_ir.operators[int(prod_idx)]
                if (
                    str(prod_op.op_type) == "LOGISTIC"
                    and len(prod_op.inputs) == 1
                    and len(prod_op.outputs) == 1
                    and str(prod_op.outputs[0]) == str(input_name)
                ):
                    if logistic_idx is not None:
                        logistic_idx = None
                        break
                    logistic_idx = int(prod_idx)
                    logistic_out_name = str(input_name)
                    add_out_name = str(prod_op.inputs[0])
                else:
                    add_out_name = str(input_name)

            if logistic_idx is None or logistic_out_name is None or add_out_name is None:
                continue
            if add_out_name in model_outputs:
                continue

            add_idx = producers.get(add_out_name, None)
            if add_idx is None:
                continue
            add_op = model_ir.operators[int(add_idx)]
            if (
                str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
                or str(add_op.outputs[0]) != str(add_out_name)
            ):
                continue

            # Strict swish topology only.
            if set(int(v) for v in consumers.get(str(logistic_out_name), [])) != {int(swish_mul_idx)}:
                continue
            add_out_users = set(int(v) for v in consumers.get(str(add_out_name), []))
            if add_out_users != {int(logistic_idx), int(swish_mul_idx)}:
                continue

            add_inputs = [str(v) for v in list(add_op.inputs)]
            mul0_idx = producers.get(add_inputs[0], None)
            mul1_idx = producers.get(add_inputs[1], None)
            if mul0_idx is None or mul1_idx is None or int(mul0_idx) == int(mul1_idx):
                continue

            mul0_info = _extract_weighted_mul(
                mul_idx=int(mul0_idx),
                expected_output_name=str(add_inputs[0]),
                add_idx=int(add_idx),
            )
            mul1_info = _extract_weighted_mul(
                mul_idx=int(mul1_idx),
                expected_output_name=str(add_inputs[1]),
                add_idx=int(add_idx),
            )
            if mul0_info is None or mul1_info is None:
                continue

            pre0_idx, pre0_in_name, pre0_out_name, _side0_name, mul0_out_name = mul0_info
            pre1_idx, pre1_in_name, pre1_out_name, _side1_name, mul1_out_name = mul1_info

            if int(pre0_idx) == int(pre1_idx):
                continue

            swish_out_users = [int(v) for v in consumers.get(swish_out_name, []) if int(v) != int(swish_mul_idx)]
            if len(swish_out_users) == 0:
                continue
            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_users: List[int] = []
            valid_posts = True
            for user_idx in swish_out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == str(swish_out_name)
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                else:
                    legacy_users.append(int(user_idx))
            if not valid_posts or len(post_indices) == 0:
                continue

            # Rewire weighted MULs to NHWC sources.
            mul0_op = model_ir.operators[int(mul0_idx)]
            mul0_inputs = [str(v) for v in list(mul0_op.inputs)]
            for idx, input_name in enumerate(list(mul0_inputs)):
                if str(input_name) == str(pre0_out_name):
                    mul0_inputs[idx] = str(pre0_in_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=mul0_op,
                new_inputs=mul0_inputs,
                graph_index=graph_index,
            )

            mul1_op = model_ir.operators[int(mul1_idx)]
            mul1_inputs = [str(v) for v in list(mul1_op.inputs)]
            for idx, input_name in enumerate(list(mul1_inputs)):
                if str(input_name) == str(pre1_out_name):
                    mul1_inputs[idx] = str(pre1_in_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=mul1_op,
                new_inputs=mul1_inputs,
                graph_index=graph_index,
            )

            # LOGISTIC is layout-agnostic; keep its input/output names and update metadata.
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(logistic_out_name), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(mul0_out_name), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(mul1_out_name), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(add_out_name), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(swish_out_name), None),
                perm_nchw_to_nhwc,
            )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=swish_mul_op,
                new_outputs=[canonical_post_output_name],
                graph_index=graph_index,
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name, graph_index=graph_index)

            old_swish_out_tensor = model_ir.tensors.get(str(swish_out_name), None)
            canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_swish_out_tensor is not None and canonical_tensor is not None:
                canonical_tensor.dtype = str(old_swish_out_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(old_swish_out_tensor.quantization)
                canonical_tensor.shape = [int(v) for v in list(old_swish_out_tensor.shape)]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(old_swish_out_tensor.shape_signature)]
                    if old_swish_out_tensor.shape_signature is not None
                    else [int(v) for v in list(old_swish_out_tensor.shape)]
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
                    graph_index=graph_index,
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_outputs=[swish_out_name],
                    graph_index=graph_index,
                )
                post_remove_indices = [int(v) for v in post_indices[1:]]
            else:
                post_remove_indices = [int(v) for v in post_indices]

            remove_indices = set(int(v) for v in post_remove_indices)
            pre0_remaining_users = [int(v) for v in consumers.get(str(pre0_out_name), []) if int(v) != int(mul0_idx)]
            if len(pre0_remaining_users) == 0:
                remove_indices.add(int(pre0_idx))
            pre1_remaining_users = [int(v) for v in consumers.get(str(pre1_out_name), []) if int(v) != int(mul1_idx)]
            if len(pre1_remaining_users) == 0:
                remove_indices.add(int(pre1_idx))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_weighted_add_swish_prepost_nhwc_chains": int(rewritten)}

def _optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around nested weighted-ADD trees followed by Swish.

    Target (constrained):
      leaves: MUL(w_scalar_like, T(0,3,1,2)(x_nhwc)) ...
      nested ADD tree over leaves -> sum_nchw
      LOGISTIC(sum_nchw) + MUL(sum_nchw, sigmoid) -> y_nchw
      y_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      - Rewire each leaf MUL to consume NHWC tensor directly.
      - Keep nested ADD + LOGISTIC + MUL in NHWC.
      - Remove post inverse transpose (or keep one adapter for legacy users).
      - Remove now-unused leaf pre-transposes.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue

            swish_out_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if swish_out_name in model_outputs or post_output_name in model_outputs:
                continue

            swish_mul_idx = producers.get(swish_out_name, None)
            if swish_mul_idx is None:
                continue
            swish_mul_op = model_ir.operators[int(swish_mul_idx)]
            if (
                str(swish_mul_op.op_type) != "MUL"
                or len(swish_mul_op.inputs) != 2
                or len(swish_mul_op.outputs) != 1
                or str(swish_mul_op.outputs[0]) != str(swish_out_name)
            ):
                continue

            swish_inputs = [str(v) for v in list(swish_mul_op.inputs)]
            logistic_idx: Optional[int] = None
            logistic_out_name: Optional[str] = None
            root_add_out_name: Optional[str] = None
            for input_name in swish_inputs:
                prod_idx = producers.get(str(input_name), None)
                if prod_idx is None:
                    continue
                prod_op = model_ir.operators[int(prod_idx)]
                if (
                    str(prod_op.op_type) == "LOGISTIC"
                    and len(prod_op.inputs) == 1
                    and len(prod_op.outputs) == 1
                    and str(prod_op.outputs[0]) == str(input_name)
                ):
                    if logistic_idx is not None:
                        logistic_idx = None
                        break
                    logistic_idx = int(prod_idx)
                    logistic_out_name = str(input_name)
                    root_add_out_name = str(prod_op.inputs[0])
                else:
                    root_add_out_name = str(input_name)
            if logistic_idx is None or logistic_out_name is None or root_add_out_name is None:
                continue
            if root_add_out_name in model_outputs:
                continue
            if set(int(v) for v in consumers.get(str(logistic_out_name), [])) != {int(swish_mul_idx)}:
                continue
            if set(int(v) for v in consumers.get(str(root_add_out_name), [])) != {int(logistic_idx), int(swish_mul_idx)}:
                continue

            root_add_idx = producers.get(str(root_add_out_name), None)
            if root_add_idx is None:
                continue

            leaf_rewrites: List[Tuple[int, str, str]] = []
            leaf_pre_indices: List[int] = []
            add_indices: List[int] = []
            add_output_names: List[str] = []
            valid_tree = True

            visited_add_outputs: set[str] = set()

            def _collect_tree(tensor_name: str, parent_add_idx: Optional[int]) -> bool:
                nonlocal valid_tree
                prod_idx = producers.get(str(tensor_name), None)
                if prod_idx is None:
                    return False
                prod_op = model_ir.operators[int(prod_idx)]
                prod_type = str(prod_op.op_type)

                if (
                    prod_type == "ADD"
                    and len(prod_op.inputs) == 2
                    and len(prod_op.outputs) == 1
                    and str(prod_op.outputs[0]) == str(tensor_name)
                    and str(tensor_name) not in model_outputs
                ):
                    if str(tensor_name) in visited_add_outputs:
                        return False
                    visited_add_outputs.add(str(tensor_name))
                    add_indices.append(int(prod_idx))
                    add_output_names.append(str(tensor_name))

                    users = set(int(v) for v in consumers.get(str(tensor_name), []))
                    if parent_add_idx is None:
                        if users != {int(logistic_idx), int(swish_mul_idx)}:
                            return False
                    else:
                        if users != {int(parent_add_idx)}:
                            return False

                    in0 = str(prod_op.inputs[0])
                    in1 = str(prod_op.inputs[1])
                    if not _collect_tree(in0, int(prod_idx)):
                        return False
                    if not _collect_tree(in1, int(prod_idx)):
                        return False
                    return True

                if (
                    prod_type == "MUL"
                    and len(prod_op.inputs) == 2
                    and len(prod_op.outputs) == 1
                    and str(prod_op.outputs[0]) == str(tensor_name)
                    and str(tensor_name) not in model_outputs
                ):
                    mul_inputs = [str(v) for v in list(prod_op.inputs)]
                    scalar_indices = [
                        int(i)
                        for i, name in enumerate(mul_inputs)
                        if _is_scalar_like_tensor(model_ir, str(name))
                    ]
                    if len(scalar_indices) != 1:
                        return False
                    data_name = str(mul_inputs[1 - int(scalar_indices[0])])
                    pre_idx = producers.get(str(data_name), None)
                    if pre_idx is None:
                        return False
                    pre_op = model_ir.operators[int(pre_idx)]
                    if (
                        str(pre_op.op_type) != "TRANSPOSE"
                        or len(pre_op.inputs) < 2
                        or len(pre_op.outputs) != 1
                        or str(pre_op.outputs[0]) != str(data_name)
                        or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                    ):
                        return False
                    pre_in_name = str(pre_op.inputs[0])
                    pre_out_name = str(pre_op.outputs[0])
                    if pre_in_name in model_outputs or pre_out_name in model_outputs:
                        return False
                    if set(int(v) for v in consumers.get(pre_out_name, [])) != {int(prod_idx)}:
                        return False

                    leaf_rewrites.append((int(prod_idx), str(pre_out_name), str(pre_in_name)))
                    leaf_pre_indices.append(int(pre_idx))
                    return True

                return False

            valid_tree = _collect_tree(str(root_add_out_name), None)
            if not valid_tree:
                continue
            if len(leaf_rewrites) < 3:
                # Keep this pass focused on nested trees; flat 2-leaf pattern is handled elsewhere.
                continue

            swish_out_users = [int(v) for v in consumers.get(swish_out_name, []) if int(v) != int(swish_mul_idx)]
            if len(swish_out_users) == 0:
                continue
            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_users: List[int] = []
            for user_idx in swish_out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == str(swish_out_name)
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                else:
                    legacy_users.append(int(user_idx))
            if len(post_indices) == 0:
                continue

            for mul_idx, pre_out_name, pre_in_name in list(leaf_rewrites):
                mul_op = model_ir.operators[int(mul_idx)]
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                replaced = False
                for idx, name in enumerate(list(mul_inputs)):
                    if str(name) == str(pre_out_name):
                        mul_inputs[idx] = str(pre_in_name)
                        replaced = True
                if not replaced:
                    valid_tree = False
                    break
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=mul_inputs,
                    graph_index=graph_index,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(mul_op.outputs[0]), None),
                    perm_nchw_to_nhwc,
                )
            if not valid_tree:
                continue

            for add_output_name in list(dict.fromkeys(add_output_names)):
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(add_output_name), None),
                    perm_nchw_to_nhwc,
                )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(logistic_out_name), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(swish_out_name), None),
                perm_nchw_to_nhwc,
            )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=swish_mul_op,
                new_outputs=[canonical_post_output_name],
                graph_index=graph_index,
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name, graph_index=graph_index)

            old_swish_out_tensor = model_ir.tensors.get(str(swish_out_name), None)
            canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_swish_out_tensor is not None and canonical_tensor is not None:
                canonical_tensor.dtype = str(old_swish_out_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(old_swish_out_tensor.quantization)
                canonical_tensor.shape = [int(v) for v in list(old_swish_out_tensor.shape)]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(old_swish_out_tensor.shape_signature)]
                    if old_swish_out_tensor.shape_signature is not None
                    else [int(v) for v in list(old_swish_out_tensor.shape)]
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
                    graph_index=graph_index,
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_outputs=[swish_out_name],
                    graph_index=graph_index,
                )
                post_remove_indices = [int(v) for v in post_indices[1:]]
            else:
                post_remove_indices = [int(v) for v in post_indices]

            remove_indices = set(int(v) for v in post_remove_indices)
            remove_indices.update(int(v) for v in leaf_pre_indices)
            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_nested_weighted_add_swish_prepost_nhwc_chains": int(rewritten)}

def _optimize_transpose_logistic_muladd_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around LOGISTIC + MUL + ADD merge blocks.

    Target:
      skip_nhwc --T(0,3,1,2)--> skip_nchw
      data_nhwc --T(0,3,1,2)--> data_nchw
      gate_nhwc --T(0,3,1,2)--> gate_nchw --LOGISTIC--> sig_nchw
      MUL(data_nchw, sig_nchw) -> mul_nchw
      ADD(mul_nchw, skip_nchw) -> y_nchw
      y_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      - Bypass three pre-transposes (skip/data/gate) to NHWC.
      - Keep LOGISTIC/MUL/ADD in NHWC.
      - Remove inverse post-transpose by making ADD emit NHWC directly.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue

            add_output_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if add_output_name in model_outputs or post_output_name in model_outputs:
                continue

            add_idx = producers.get(add_output_name, None)
            if add_idx is None:
                continue
            add_op = model_ir.operators[int(add_idx)]
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            if str(add_op.outputs[0]) != add_output_name:
                continue

            add_inputs = [str(v) for v in list(add_op.inputs)]
            mul_input_side: Optional[int] = None
            mul_idx: Optional[int] = None
            for i, input_name in enumerate(add_inputs):
                producer_idx = producers.get(str(input_name), None)
                if producer_idx is None:
                    continue
                producer_op = model_ir.operators[int(producer_idx)]
                if (
                    str(producer_op.op_type) == "MUL"
                    and len(producer_op.outputs) == 1
                    and str(producer_op.outputs[0]) == str(input_name)
                ):
                    if mul_idx is not None:
                        mul_idx = None
                        break
                    mul_input_side = int(i)
                    mul_idx = int(producer_idx)
            if mul_idx is None or mul_input_side is None:
                continue

            skip_input_name = str(add_inputs[1 - int(mul_input_side)])
            skip_pre_idx = producers.get(skip_input_name, None)
            if skip_pre_idx is None:
                continue
            skip_pre_op = model_ir.operators[int(skip_pre_idx)]
            if (
                str(skip_pre_op.op_type) != "TRANSPOSE"
                or len(skip_pre_op.inputs) < 2
                or len(skip_pre_op.outputs) != 1
                or str(skip_pre_op.outputs[0]) != skip_input_name
                or _read_transpose_perm(model_ir, skip_pre_op) != perm_nhwc_to_nchw
            ):
                continue
            skip_nhwc_name = str(skip_pre_op.inputs[0])
            if skip_nhwc_name in model_outputs or skip_input_name in model_outputs:
                continue

            mul_op = model_ir.operators[int(mul_idx)]
            if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_output_name = str(mul_op.outputs[0])
            if mul_output_name in model_outputs:
                continue
            # Optional affine tail:
            #   MUL(data, logistic(gate)) -> MUL(const) -> ADD(skip, ...)
            # Unwrap one scalar-const MUL so this pass can still match and remove
            # surrounding layout transposes.
            core_mul_idx = int(mul_idx)
            core_mul_op = mul_op
            core_mul_output_name = str(mul_output_name)
            if str(core_mul_op.op_type) == "MUL":
                core_mul_inputs = [str(v) for v in list(core_mul_op.inputs)]
                scalar_const_indices = [
                    int(i)
                    for i, input_name in enumerate(core_mul_inputs)
                    if _is_singleton_constant_tensor(model_ir, str(input_name))
                ]
                if len(scalar_const_indices) == 1:
                    main_input_name = str(core_mul_inputs[1 - int(scalar_const_indices[0])])
                    upstream_mul_idx = producers.get(main_input_name, None)
                    if upstream_mul_idx is not None:
                        upstream_mul_op = model_ir.operators[int(upstream_mul_idx)]
                        if (
                            str(upstream_mul_op.op_type) == "MUL"
                            and len(upstream_mul_op.inputs) == 2
                            and len(upstream_mul_op.outputs) == 1
                            and str(upstream_mul_op.outputs[0]) == str(main_input_name)
                        ):
                            core_mul_idx = int(upstream_mul_idx)
                            core_mul_op = upstream_mul_op
                            core_mul_output_name = str(main_input_name)

            pre_data_idx: Optional[int] = None
            pre_data_in_name: Optional[str] = None
            pre_data_out_name: Optional[str] = None
            logistic_idx: Optional[int] = None
            logistic_out_name: Optional[str] = None
            for input_name in [str(v) for v in list(core_mul_op.inputs)]:
                input_producer_idx = producers.get(str(input_name), None)
                if input_producer_idx is None:
                    continue
                input_producer_op = model_ir.operators[int(input_producer_idx)]
                if (
                    str(input_producer_op.op_type) == "TRANSPOSE"
                    and len(input_producer_op.inputs) >= 2
                    and len(input_producer_op.outputs) == 1
                    and str(input_producer_op.outputs[0]) == str(input_name)
                    and _read_transpose_perm(model_ir, input_producer_op) == perm_nhwc_to_nchw
                ):
                    if pre_data_idx is not None:
                        pre_data_idx = None
                        break
                    pre_data_idx = int(input_producer_idx)
                    pre_data_in_name = str(input_producer_op.inputs[0])
                    pre_data_out_name = str(input_producer_op.outputs[0])
                    continue
                if (
                    str(input_producer_op.op_type) == "LOGISTIC"
                    and len(input_producer_op.outputs) == 1
                    and str(input_producer_op.outputs[0]) == str(input_name)
                ):
                    if logistic_idx is not None:
                        logistic_idx = None
                        break
                    logistic_idx = int(input_producer_idx)
                    logistic_out_name = str(input_name)
            if (
                pre_data_idx is None
                or pre_data_in_name is None
                or pre_data_out_name is None
                or logistic_idx is None
                or logistic_out_name is None
            ):
                continue

            if pre_data_in_name in model_outputs or pre_data_out_name in model_outputs:
                continue

            logistic_op = model_ir.operators[int(logistic_idx)]
            if str(logistic_op.op_type) != "LOGISTIC" or len(logistic_op.inputs) != 1 or len(logistic_op.outputs) != 1:
                continue
            if str(logistic_op.outputs[0]) != logistic_out_name:
                continue
            gate_pre_out_name = str(logistic_op.inputs[0])
            if gate_pre_out_name in model_outputs:
                continue

            gate_pre_idx = producers.get(gate_pre_out_name, None)
            if gate_pre_idx is None:
                continue
            gate_pre_op = model_ir.operators[int(gate_pre_idx)]
            if (
                str(gate_pre_op.op_type) != "TRANSPOSE"
                or len(gate_pre_op.inputs) < 2
                or len(gate_pre_op.outputs) != 1
                or str(gate_pre_op.outputs[0]) != gate_pre_out_name
                or _read_transpose_perm(model_ir, gate_pre_op) != perm_nhwc_to_nchw
            ):
                continue
            gate_nhwc_name = str(gate_pre_op.inputs[0])
            if gate_nhwc_name in model_outputs or gate_pre_out_name in model_outputs:
                continue

            # The chain user must exist; extra legacy users are allowed and preserved.
            skip_pre_users = [int(v) for v in consumers.get(skip_input_name, [])]
            if int(add_idx) not in set(skip_pre_users):
                continue
            data_pre_users = [int(v) for v in consumers.get(pre_data_out_name, [])]
            if int(core_mul_idx) not in set(data_pre_users):
                continue
            gate_pre_users = [int(v) for v in consumers.get(gate_pre_out_name, [])]
            if int(logistic_idx) not in set(gate_pre_users):
                continue

            # ADD output must have at least one inverse post-transpose. Non-post users
            # are kept via one retained adapter transpose.
            add_out_users = [int(v) for v in consumers.get(add_output_name, [])]
            if len(add_out_users) == 0:
                continue
            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_users: List[int] = []
            for user_idx in add_out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == add_output_name
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                else:
                    legacy_users.append(int(user_idx))
            if len(post_indices) == 0:
                continue

            # Rewire LOGISTIC/MUL/ADD to NHWC sources.
            _set_operator_inputs(
                model_ir=model_ir,
                op=logistic_op,
                new_inputs=[str(gate_nhwc_name)],
                graph_index=graph_index,
            )

            mul_inputs = [str(v) for v in list(core_mul_op.inputs)]
            mul_replaced = False
            for i, input_name in enumerate(list(mul_inputs)):
                if str(input_name) == str(pre_data_out_name):
                    mul_inputs[i] = str(pre_data_in_name)
                    mul_replaced = True
            if not mul_replaced:
                continue
            _set_operator_inputs(
                model_ir=model_ir,
                op=core_mul_op,
                new_inputs=mul_inputs,
                graph_index=graph_index,
            )

            add_inputs_rewritten = [str(v) for v in list(add_op.inputs)]
            add_replaced = False
            for i, input_name in enumerate(list(add_inputs_rewritten)):
                if str(input_name) == str(skip_input_name):
                    add_inputs_rewritten[i] = str(skip_nhwc_name)
                    add_replaced = True
            if not add_replaced:
                continue
            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=add_inputs_rewritten,
                graph_index=graph_index,
            )

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(logistic_out_name), None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(core_mul_output_name), None),
                perm_nchw_to_nhwc,
            )
            if str(core_mul_output_name) != str(mul_output_name):
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(mul_output_name), None),
                    perm_nchw_to_nhwc,
                )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(add_output_name), None),
                perm_nchw_to_nhwc,
            )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=add_op,
                new_outputs=[canonical_post_output_name],
                graph_index=graph_index,
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, str(alias_name), canonical_post_output_name, graph_index=graph_index)

            old_add_out_tensor = model_ir.tensors.get(add_output_name, None)
            canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_add_out_tensor is not None and canonical_tensor is not None:
                canonical_tensor.dtype = str(old_add_out_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(old_add_out_tensor.quantization)
                canonical_tensor.shape = [int(v) for v in list(old_add_out_tensor.shape)]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(old_add_out_tensor.shape_signature)]
                    if old_add_out_tensor.shape_signature is not None
                    else [int(v) for v in list(old_add_out_tensor.shape)]
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
                    graph_index=graph_index,
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=keep_post_op,
                    new_outputs=[add_output_name],
                    graph_index=graph_index,
                )
                post_remove_indices = [int(v) for v in post_indices[1:]]
            else:
                post_remove_indices = [int(v) for v in post_indices]

            remove_indices = set(int(v) for v in post_remove_indices)
            # Remove pre-transposes only when no remaining users need the NCHW view.
            if len([int(v) for v in skip_pre_users if int(v) != int(add_idx)]) == 0:
                remove_indices.add(int(skip_pre_idx))
            if len([int(v) for v in data_pre_users if int(v) != int(core_mul_idx)]) == 0:
                remove_indices.add(int(pre_data_idx))
            if len([int(v) for v in gate_pre_users if int(v) != int(logistic_idx)]) == 0:
                remove_indices.add(int(gate_pre_idx))
            for remove_idx in sorted(remove_indices, reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_logistic_muladd_prepost_nhwc_chains": int(rewritten)}


def _inverse_post_source_ops(
    pass_state: ModelIRPassState,
    source_type: str,
) -> List[Any]:
    model_ir = pass_state.model_ir
    model_outputs = {str(name) for name in model_ir.outputs}
    sources: List[Any] = []
    for post_op in model_ir.operators:
        if (
            str(post_op.op_type) != "TRANSPOSE"
            or len(post_op.inputs) < 2
            or len(post_op.outputs) != 1
            or _read_transpose_perm(model_ir, post_op) != [0, 2, 3, 1]
            or str(post_op.inputs[0]) in model_outputs
            or str(post_op.outputs[0]) in model_outputs
        ):
            continue
        source_op = pass_state.graph_index.producer(str(post_op.inputs[0]))
        if source_op is not None and str(source_op.op_type) == source_type:
            sources.append(source_op)
    return sources


def _is_nhwc_to_nchw_source(
    pass_state: ModelIRPassState,
    tensor_name: str,
) -> bool:
    producer = pass_state.graph_index.producer(tensor_name)
    return bool(
        producer is not None
        and str(producer.op_type) == "TRANSPOSE"
        and len(producer.inputs) >= 2
        and len(producer.outputs) == 1
        and str(producer.outputs[0]) == tensor_name
        and _read_transpose_perm(pass_state.model_ir, producer) == [0, 3, 1, 2]
    )


def _has_sum_logistic_muladd_candidate(pass_state: ModelIRPassState) -> bool:
    model_ir = pass_state.model_ir
    model_outputs = {str(name) for name in model_ir.outputs}
    for add_op in _inverse_post_source_ops(pass_state, "ADD"):
        if len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
            continue
        mul_ops = [pass_state.graph_index.producer(str(name)) for name in add_op.inputs]
        if any(
            op is None or str(op.op_type) != "MUL" or len(op.inputs) != 2
            for op in mul_ops
        ):
            continue
        gate_names: List[str] = []
        data_sources = 0
        for mul_op in mul_ops:
            assert mul_op is not None
            for input_name in mul_op.inputs:
                name = str(input_name)
                if _is_nhwc_to_nchw_source(pass_state, name):
                    data_sources += 1
                else:
                    gate_names.append(name)
        if data_sources != 2:
            continue
        for gate_name in gate_names:
            gate_op = pass_state.graph_index.producer(gate_name)
            if gate_op is not None and str(gate_op.op_type) == "SUB":
                logistic_names = [
                    str(name)
                    for name in gate_op.inputs
                    if (
                        (producer := pass_state.graph_index.producer(str(name)))
                        is not None
                        and str(producer.op_type) == "LOGISTIC"
                    )
                ]
            else:
                logistic_names = [gate_name]
            for logistic_name in logistic_names:
                logistic_op = pass_state.graph_index.producer(logistic_name)
                if (
                    logistic_op is None
                    or str(logistic_op.op_type) != "LOGISTIC"
                    or len(logistic_op.inputs) != 1
                ):
                    continue
                sum_op = pass_state.graph_index.producer(
                    str(logistic_op.inputs[0])
                )
                if (
                    sum_op is not None
                    and str(sum_op.op_type) == "SUM"
                    and len(sum_op.inputs) >= 2
                    and len(sum_op.outputs) == 1
                    and str(sum_op.outputs[0]) not in model_outputs
                    and _is_nhwc_to_nchw_source(
                        pass_state,
                        str(sum_op.inputs[0]),
                    )
                ):
                    sum_idx = pass_state.graph_index.operator_index(sum_op)
                    if sum_idx is not None and set(
                        pass_state.graph_index.consumer_indices(
                            str(sum_op.inputs[0])
                        )
                    ) == {sum_idx}:
                        return True
    return False


def _swish_root_add(
    pass_state: ModelIRPassState,
    swish_op: Any,
) -> Any | None:
    if len(swish_op.inputs) != 2 or len(swish_op.outputs) != 1:
        return None
    for input_name in swish_op.inputs:
        logistic_op = pass_state.graph_index.producer(str(input_name))
        if (
            logistic_op is None
            or str(logistic_op.op_type) != "LOGISTIC"
            or len(logistic_op.inputs) != 1
        ):
            continue
        root_name = str(logistic_op.inputs[0])
        if root_name not in {str(name) for name in swish_op.inputs}:
            continue
        root_op = pass_state.graph_index.producer(root_name)
        if root_op is not None and str(root_op.op_type) == "ADD":
            return root_op
    return None


def _has_weighted_swish_candidate(
    pass_state: ModelIRPassState,
    *,
    nested: bool,
) -> bool:
    for swish_op in _inverse_post_source_ops(pass_state, "MUL"):
        root_add = _swish_root_add(pass_state, swish_op)
        if root_add is None or len(root_add.inputs) != 2:
            continue
        child_ops = [
            pass_state.graph_index.producer(str(name)) for name in root_add.inputs
        ]
        has_nested_add = any(
            child is not None and str(child.op_type) == "ADD"
            for child in child_ops
        )
        if nested != has_nested_add:
            continue
        if nested:
            return True
        valid_leaves = True
        for child in child_ops:
            if child is None or str(child.op_type) != "MUL" or len(child.inputs) != 2:
                valid_leaves = False
                break
            if not any(
                _is_nhwc_to_nchw_source(pass_state, str(name))
                for name in child.inputs
            ):
                valid_leaves = False
                break
        if valid_leaves:
            return True
    return False


def _has_logistic_muladd_candidate(pass_state: ModelIRPassState) -> bool:
    for add_op in _inverse_post_source_ops(pass_state, "ADD"):
        if len(add_op.inputs) != 2:
            continue
        input_ops = [
            pass_state.graph_index.producer(str(name)) for name in add_op.inputs
        ]
        mul_ops = [op for op in input_ops if op is not None and str(op.op_type) == "MUL"]
        if len(mul_ops) != 1:
            continue
        skip_names = [
            str(name)
            for name, producer in zip(add_op.inputs, input_ops)
            if producer is not mul_ops[0]
        ]
        if len(skip_names) != 1 or not _is_nhwc_to_nchw_source(
            pass_state,
            skip_names[0],
        ):
            continue
        mul_op = mul_ops[0]
        if len(mul_op.inputs) != 2:
            continue
        data_sources = [
            str(name)
            for name in mul_op.inputs
            if _is_nhwc_to_nchw_source(pass_state, str(name))
        ]
        logistic_ops = [
            pass_state.graph_index.producer(str(name))
            for name in mul_op.inputs
            if not _is_nhwc_to_nchw_source(pass_state, str(name))
        ]
        if len(data_sources) != 1 or len(logistic_ops) != 1:
            continue
        logistic_op = logistic_ops[0]
        if (
            logistic_op is not None
            and str(logistic_op.op_type) == "LOGISTIC"
            and len(logistic_op.inputs) == 1
            and _is_nhwc_to_nchw_source(
                pass_state,
                str(logistic_op.inputs[0]),
            )
        ):
            return True
    return False


def run_elementwise_gate_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Run ordered rank-four elementwise gate layout propagation."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "MUL", "ADD", "LOGISTIC"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    callbacks = [
        (
            "layout.sum_logistic_muladd_nhwc",
            10,
            _has_sum_logistic_muladd_candidate,
            _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains,
            "optimized_transpose_sum_logistic_muladd_prepost_nhwc_chains",
        ),
        (
            "layout.weighted_add_swish_nhwc",
            20,
            lambda state: _has_weighted_swish_candidate(state, nested=False),
            _optimize_transpose_weighted_add_swish_prepost_nhwc_chains,
            "optimized_transpose_weighted_add_swish_prepost_nhwc_chains",
        ),
        (
            "layout.nested_weighted_add_swish_nhwc",
            30,
            lambda state: _has_weighted_swish_candidate(state, nested=True),
            _optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains,
            "optimized_transpose_nested_weighted_add_swish_prepost_nhwc_chains",
        ),
        (
            "layout.logistic_muladd_nhwc",
            40,
            _has_logistic_muladd_candidate,
            _optimize_transpose_logistic_muladd_prepost_nhwc_chains,
            "optimized_transpose_logistic_muladd_prepost_nhwc_chains",
        ),
    ]
    specs: List[PassSpec] = []
    defaults: Dict[str, int] = {}
    for pass_id, priority, precondition, optimizer, stats_key in callbacks:
        defaults[stats_key] = 0

        def _run(
            pass_state: ModelIRPassState,
            *,
            optimizer: Any = optimizer,
            stats_key: str = stats_key,
        ) -> Dict[str, int | bool]:
            stats = optimizer(
                pass_state.model_ir,
                graph_index=pass_state.graph_index,
                layout_state=pass_state.layout_state,
            )
            return {**stats, "changed": bool(stats.get(stats_key, 0))}

        specs.append(
            PassSpec(
                pass_id=pass_id,
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=precondition,
                priority=priority,
                transactional=True,
            )
        )

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details=defaults,
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
