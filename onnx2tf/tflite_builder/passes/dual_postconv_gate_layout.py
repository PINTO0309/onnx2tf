from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
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
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR

def _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around a logistic/sub-gated dual-MUL/ADD block feeding two convs.

    Target:
      g_nhwc --T(0,3,1,2)--> g_nchw --LOGISTIC--> sig_nchw --SUB(1, sig)->sub_nchw
      a_nhwc --T(0,3,1,2)--> a_nchw
      b_nhwc --T(0,3,1,2)--> b_nchw
      MUL(sig_nchw, a_nchw) -> m0_nchw
      MUL(sub_nchw, b_nchw) -> m1_nchw
      ADD(b_nchw, m0_nchw)  -> y0_nchw --T(0,2,3,1)--> y0_nhwc -> CONV
      ADD(m1_nchw, a_nchw)  -> y1_nchw --T(0,2,3,1)--> y1_nhwc -> CONV

    Rewrite:
      - Bypass all three pre-transposes to NHWC.
      - Keep LOGISTIC/SUB/MUL/ADD in NHWC.
      - Remove both post-transposes and feed CONV inputs directly.
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

        for gate_pre_idx, gate_pre_op in enumerate(model_ir.operators):
            if str(gate_pre_op.op_type) != "TRANSPOSE" or len(gate_pre_op.inputs) < 2 or len(gate_pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, gate_pre_op) != perm_nhwc_to_nchw:
                continue

            gate_pre_in_name = str(gate_pre_op.inputs[0])
            gate_pre_out_name = str(gate_pre_op.outputs[0])
            if gate_pre_in_name in model_outputs or gate_pre_out_name in model_outputs:
                continue

            gate_users = [int(v) for v in consumers.get(gate_pre_out_name, [])]
            if len(gate_users) != 1:
                continue
            logistic_idx = int(gate_users[0])
            logistic_op = model_ir.operators[int(logistic_idx)]
            if str(logistic_op.op_type) != "LOGISTIC" or len(logistic_op.inputs) != 1 or len(logistic_op.outputs) != 1:
                continue
            if str(logistic_op.inputs[0]) != gate_pre_out_name:
                continue

            logistic_out_name = str(logistic_op.outputs[0])
            if logistic_out_name in model_outputs:
                continue
            logistic_users = [int(v) for v in consumers.get(logistic_out_name, [])]
            if len(logistic_users) != 2:
                continue

            sub_idx: Optional[int] = None
            mul_sig_idx: Optional[int] = None
            for user_idx in logistic_users:
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) == "SUB":
                    if sub_idx is not None:
                        sub_idx = None
                        break
                    sub_idx = int(user_idx)
                elif str(user_op.op_type) == "MUL":
                    if mul_sig_idx is not None:
                        mul_sig_idx = None
                        break
                    mul_sig_idx = int(user_idx)
            if sub_idx is None or mul_sig_idx is None:
                continue

            sub_op = model_ir.operators[int(sub_idx)]
            if len(sub_op.inputs) != 2 or len(sub_op.outputs) != 1:
                continue
            if str(sub_op.outputs[0]) in model_outputs:
                continue
            if str(logistic_out_name) not in {str(v) for v in list(sub_op.inputs)}:
                continue
            sub_out_name = str(sub_op.outputs[0])
            sub_users = [int(v) for v in consumers.get(sub_out_name, [])]
            if len(sub_users) != 1:
                continue
            mul_sub_idx = int(sub_users[0])
            mul_sub_op = model_ir.operators[int(mul_sub_idx)]
            if str(mul_sub_op.op_type) != "MUL" or len(mul_sub_op.inputs) != 2 or len(mul_sub_op.outputs) != 1:
                continue

            mul_sig_op = model_ir.operators[int(mul_sig_idx)]
            if str(mul_sig_op.op_type) != "MUL" or len(mul_sig_op.inputs) != 2 or len(mul_sig_op.outputs) != 1:
                continue

            def _resolve_pre_for_mul(
                mul_op: OperatorIR,
                gate_tensor_candidates: List[str],
            ) -> Optional[Tuple[int, str, str]]:
                pre_idx_local: Optional[int] = None
                pre_in_name_local: Optional[str] = None
                pre_out_name_local: Optional[str] = None
                for input_name in list(mul_op.inputs):
                    if str(input_name) in set(str(v) for v in gate_tensor_candidates):
                        continue
                    prod_idx = producers.get(str(input_name), None)
                    if prod_idx is None:
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
                        pre_idx_local = int(prod_idx)
                        pre_in_name_local = str(prod_op.inputs[0])
                        pre_out_name_local = str(prod_op.outputs[0])
                        break
                if pre_idx_local is None or pre_in_name_local is None or pre_out_name_local is None:
                    return None
                return int(pre_idx_local), str(pre_in_name_local), str(pre_out_name_local)

            sig_pre_info = _resolve_pre_for_mul(
                mul_sig_op,
                gate_tensor_candidates=[str(logistic_out_name)],
            )
            sub_pre_info = _resolve_pre_for_mul(
                mul_sub_op,
                gate_tensor_candidates=[str(sub_out_name)],
            )
            if sig_pre_info is None or sub_pre_info is None:
                continue

            sig_pre_idx, sig_pre_in_name, sig_pre_out_name = sig_pre_info
            sub_pre_idx, sub_pre_in_name, sub_pre_out_name = sub_pre_info
            if int(sig_pre_idx) == int(sub_pre_idx):
                continue

            mul_sig_out_name = str(mul_sig_op.outputs[0])
            mul_sub_out_name = str(mul_sub_op.outputs[0])
            if mul_sig_out_name in model_outputs or mul_sub_out_name in model_outputs:
                continue
            mul_sig_users = [int(v) for v in consumers.get(mul_sig_out_name, [])]
            mul_sub_users = [int(v) for v in consumers.get(mul_sub_out_name, [])]
            if len(mul_sig_users) != 1 or len(mul_sub_users) != 1:
                continue
            add_sig_idx = int(mul_sig_users[0])
            add_sub_idx = int(mul_sub_users[0])
            if int(add_sig_idx) == int(add_sub_idx):
                continue
            add_sig_op = model_ir.operators[int(add_sig_idx)]
            add_sub_op = model_ir.operators[int(add_sub_idx)]
            if str(add_sig_op.op_type) != "ADD" or len(add_sig_op.inputs) != 2 or len(add_sig_op.outputs) != 1:
                continue
            if str(add_sub_op.op_type) != "ADD" or len(add_sub_op.inputs) != 2 or len(add_sub_op.outputs) != 1:
                continue

            add_sig_inputs = [str(v) for v in list(add_sig_op.inputs)]
            add_sub_inputs = [str(v) for v in list(add_sub_op.inputs)]
            if str(mul_sig_out_name) not in set(add_sig_inputs):
                continue
            if str(mul_sub_out_name) not in set(add_sub_inputs):
                continue
            add_sig_other = (
                add_sig_inputs[1] if add_sig_inputs[0] == str(mul_sig_out_name) else add_sig_inputs[0]
            )
            add_sub_other = (
                add_sub_inputs[1] if add_sub_inputs[0] == str(mul_sub_out_name) else add_sub_inputs[0]
            )
            if set([str(add_sig_other), str(add_sub_other)]) != set([str(sig_pre_out_name), str(sub_pre_out_name)]):
                continue

            add_sig_out_name = str(add_sig_op.outputs[0])
            add_sub_out_name = str(add_sub_op.outputs[0])
            if add_sig_out_name in model_outputs or add_sub_out_name in model_outputs:
                continue

            def _collect_inverse_posts(add_out_name: str) -> Optional[Tuple[List[int], List[str]]]:
                out_users = [int(v) for v in consumers.get(str(add_out_name), [])]
                if len(out_users) == 0:
                    return None
                post_indices_local: List[int] = []
                post_output_names_local: List[str] = []
                for u_idx in out_users:
                    u_op = model_ir.operators[int(u_idx)]
                    if (
                        str(u_op.op_type) == "TRANSPOSE"
                        and len(u_op.inputs) >= 2
                        and len(u_op.outputs) == 1
                        and str(u_op.inputs[0]) == str(add_out_name)
                        and _read_transpose_perm(model_ir, u_op) == perm_nchw_to_nhwc
                        and str(u_op.outputs[0]) not in model_outputs
                    ):
                        post_indices_local.append(int(u_idx))
                        post_output_names_local.append(str(u_op.outputs[0]))
                    else:
                        return None
                return post_indices_local, post_output_names_local

            sig_post_info = _collect_inverse_posts(add_sig_out_name)
            sub_post_info = _collect_inverse_posts(add_sub_out_name)
            if sig_post_info is None or sub_post_info is None:
                continue
            sig_post_indices, sig_post_output_names = sig_post_info
            sub_post_indices, sub_post_output_names = sub_post_info

            # Ensure pre-transpose outputs are local to this chain.
            allowed_users = {
                int(mul_sig_idx),
                int(mul_sub_idx),
                int(add_sig_idx),
                int(add_sub_idx),
            }
            if not set(int(v) for v in consumers.get(str(sig_pre_out_name), [])).issubset(allowed_users):
                continue
            if not set(int(v) for v in consumers.get(str(sub_pre_out_name), [])).issubset(allowed_users):
                continue

            # 1) Bypass gate pre-transpose.
            _set_operator_inputs(
                model_ir=model_ir,
                op=logistic_op,
                new_inputs=[str(gate_pre_in_name)],
                graph_index=graph_index,
            )

            # 2) Bypass the two data pre-transposes in MUL and ADD.
            for mul_op, pre_out_name, pre_in_name in [
                (mul_sig_op, sig_pre_out_name, sig_pre_in_name),
                (mul_sub_op, sub_pre_out_name, sub_pre_in_name),
            ]:
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                mul_inputs = [
                    str(pre_in_name) if str(v) == str(pre_out_name) else str(v)
                    for v in mul_inputs
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=mul_inputs,
                    graph_index=graph_index,
                )

            for add_op, pre_out_name, pre_in_name in [
                (add_sig_op, sig_pre_out_name, sig_pre_in_name),
                (add_sig_op, sub_pre_out_name, sub_pre_in_name),
                (add_sub_op, sig_pre_out_name, sig_pre_in_name),
                (add_sub_op, sub_pre_out_name, sub_pre_in_name),
            ]:
                add_inputs = [str(v) for v in list(add_op.inputs)]
                rewritten_inputs = [
                    str(pre_in_name) if str(v) == str(pre_out_name) else str(v)
                    for v in add_inputs
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=add_op,
                    new_inputs=rewritten_inputs,
                    graph_index=graph_index,
                )

            # 3) Mark rewritten intermediates as NHWC.
            for tensor_name in [
                str(logistic_out_name),
                str(sub_out_name),
                str(mul_sig_out_name),
                str(mul_sub_out_name),
                str(add_sig_out_name),
                str(add_sub_out_name),
            ]:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            # 4) Bypass both post-transpose groups.
            for add_op, old_add_out_name, post_output_names in [
                (add_sig_op, add_sig_out_name, sig_post_output_names),
                (add_sub_op, add_sub_out_name, sub_post_output_names),
            ]:
                canonical_post_output_name = str(post_output_names[0])
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=add_op,
                    new_outputs=[canonical_post_output_name],
                    graph_index=graph_index,
                )
                for alias_name in list(post_output_names[1:]):
                    _replace_tensor_inputs(
                        model_ir,
                        str(alias_name),
                        canonical_post_output_name,
                        graph_index=graph_index,
                    )

                old_add_out_tensor = model_ir.tensors.get(str(old_add_out_name), None)
                canonical_tensor = model_ir.tensors.get(str(canonical_post_output_name), None)
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
                int(gate_pre_idx),
                int(sig_pre_idx),
                int(sub_pre_idx),
                *[int(v) for v in sig_post_indices],
                *[int(v) for v in sub_post_indices],
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
    return {"optimized_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains": int(rewritten)}


def _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around logistic/sub-gated dual-MUL branches whose MUL
    outputs are immediately transposed back to NHWC before layout-agnostic consumers.

    Target:
      g_nhwc --T(0,3,1,2)--> g_nchw --LOGISTIC--> sig_nchw --SUB(1, sig)->sub_nchw
      a_nhwc --T(0,3,1,2)--> a_nchw
      b_nhwc --T(0,3,1,2)--> b_nchw
      MUL(sig_nchw, a_nchw) -> ma_nchw --T(0,2,3,1)--> ma_nhwc
      MUL(sub_nchw, b_nchw) -> mb_nchw --T(0,2,3,1)--> mb_nhwc
      ma_nhwc / mb_nhwc then feed NHWC-only consumers (e.g. ADD -> CONV)

    Rewrite:
      - Bypass all three pre-transposes to NHWC.
      - Keep LOGISTIC/SUB/MUL in NHWC.
      - Remove both post-transposes and make the MULs emit NHWC directly.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _collect_inverse_posts(tensor_name: str) -> Optional[Tuple[List[int], List[str]]]:
        out_users = [int(v) for v in consumers.get(str(tensor_name), [])]
        if len(out_users) == 0:
            return None
        post_indices_local: List[int] = []
        post_output_names_local: List[str] = []
        for user_idx in out_users:
            user_op = model_ir.operators[int(user_idx)]
            if (
                str(user_op.op_type) != "TRANSPOSE"
                or len(user_op.inputs) < 2
                or len(user_op.outputs) != 1
                or str(user_op.inputs[0]) != str(tensor_name)
                or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                or str(user_op.outputs[0]) in model_outputs
            ):
                return None
            post_indices_local.append(int(user_idx))
            post_output_names_local.append(str(user_op.outputs[0]))
        return post_indices_local, post_output_names_local

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for gate_pre_idx, gate_pre_op in enumerate(model_ir.operators):
            if str(gate_pre_op.op_type) != "TRANSPOSE" or len(gate_pre_op.inputs) < 2 or len(gate_pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, gate_pre_op) != perm_nhwc_to_nchw:
                continue

            gate_pre_in_name = str(gate_pre_op.inputs[0])
            gate_pre_out_name = str(gate_pre_op.outputs[0])
            if gate_pre_in_name in model_outputs or gate_pre_out_name in model_outputs:
                continue

            gate_users = [int(v) for v in consumers.get(gate_pre_out_name, [])]
            if len(gate_users) != 1:
                continue
            logistic_idx = int(gate_users[0])
            logistic_op = model_ir.operators[int(logistic_idx)]
            if (
                str(logistic_op.op_type) != "LOGISTIC"
                or len(logistic_op.inputs) != 1
                or len(logistic_op.outputs) != 1
                or str(logistic_op.inputs[0]) != gate_pre_out_name
            ):
                continue

            logistic_out_name = str(logistic_op.outputs[0])
            if logistic_out_name in model_outputs:
                continue
            logistic_users = [int(v) for v in consumers.get(logistic_out_name, [])]
            if len(logistic_users) != 2:
                continue

            sub_idx: Optional[int] = None
            mul_sig_idx: Optional[int] = None
            for user_idx in logistic_users:
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) == "SUB":
                    if sub_idx is not None:
                        sub_idx = None
                        break
                    sub_idx = int(user_idx)
                elif str(user_op.op_type) == "MUL":
                    if mul_sig_idx is not None:
                        mul_sig_idx = None
                        break
                    mul_sig_idx = int(user_idx)
            if sub_idx is None or mul_sig_idx is None:
                continue

            sub_op = model_ir.operators[int(sub_idx)]
            if len(sub_op.inputs) != 2 or len(sub_op.outputs) != 1 or str(sub_op.outputs[0]) in model_outputs:
                continue
            if str(logistic_out_name) not in {str(v) for v in list(sub_op.inputs)}:
                continue
            sub_out_name = str(sub_op.outputs[0])
            sub_users = [int(v) for v in consumers.get(sub_out_name, [])]
            if len(sub_users) != 1:
                continue
            mul_sub_idx = int(sub_users[0])
            if int(mul_sub_idx) == int(mul_sig_idx):
                continue

            mul_sig_op = model_ir.operators[int(mul_sig_idx)]
            mul_sub_op = model_ir.operators[int(mul_sub_idx)]
            if (
                str(mul_sig_op.op_type) != "MUL"
                or len(mul_sig_op.inputs) != 2
                or len(mul_sig_op.outputs) != 1
                or str(mul_sub_op.op_type) != "MUL"
                or len(mul_sub_op.inputs) != 2
                or len(mul_sub_op.outputs) != 1
            ):
                continue

            def _resolve_pre_for_mul(
                mul_op: OperatorIR,
                gate_tensor_candidates: List[str],
            ) -> Optional[Tuple[int, str, str]]:
                pre_idx_local: Optional[int] = None
                pre_in_name_local: Optional[str] = None
                pre_out_name_local: Optional[str] = None
                gate_tensor_set = set(str(v) for v in gate_tensor_candidates)
                for input_name in list(mul_op.inputs):
                    if str(input_name) in gate_tensor_set:
                        continue
                    prod_idx = producers.get(str(input_name), None)
                    if prod_idx is None:
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
                        pre_idx_local = int(prod_idx)
                        pre_in_name_local = str(prod_op.inputs[0])
                        pre_out_name_local = str(prod_op.outputs[0])
                        break
                if pre_idx_local is None or pre_in_name_local is None or pre_out_name_local is None:
                    return None
                return int(pre_idx_local), str(pre_in_name_local), str(pre_out_name_local)

            sig_pre_info = _resolve_pre_for_mul(
                mul_sig_op,
                gate_tensor_candidates=[str(logistic_out_name)],
            )
            sub_pre_info = _resolve_pre_for_mul(
                mul_sub_op,
                gate_tensor_candidates=[str(sub_out_name)],
            )
            if sig_pre_info is None or sub_pre_info is None:
                continue

            sig_pre_idx, sig_pre_in_name, sig_pre_out_name = sig_pre_info
            sub_pre_idx, sub_pre_in_name, sub_pre_out_name = sub_pre_info
            if int(sig_pre_idx) == int(sub_pre_idx):
                continue

            if set(int(v) for v in consumers.get(str(gate_pre_out_name), [])) != {int(logistic_idx)}:
                continue
            if set(int(v) for v in consumers.get(str(sig_pre_out_name), [])) != {int(mul_sig_idx)}:
                continue
            if set(int(v) for v in consumers.get(str(sub_pre_out_name), [])) != {int(mul_sub_idx)}:
                continue

            mul_sig_out_name = str(mul_sig_op.outputs[0])
            mul_sub_out_name = str(mul_sub_op.outputs[0])
            if mul_sig_out_name in model_outputs or mul_sub_out_name in model_outputs:
                continue

            sig_post_info = _collect_inverse_posts(mul_sig_out_name)
            sub_post_info = _collect_inverse_posts(mul_sub_out_name)
            if sig_post_info is None or sub_post_info is None:
                continue
            sig_post_indices, sig_post_output_names = sig_post_info
            sub_post_indices, sub_post_output_names = sub_post_info

            _set_operator_inputs(
                model_ir=model_ir,
                op=logistic_op,
                new_inputs=[str(gate_pre_in_name)],
                graph_index=graph_index,
            )

            for mul_op, pre_out_name, pre_in_name in [
                (mul_sig_op, sig_pre_out_name, sig_pre_in_name),
                (mul_sub_op, sub_pre_out_name, sub_pre_in_name),
            ]:
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                replaced = False
                for input_idx, input_name in enumerate(list(mul_inputs)):
                    if str(input_name) == str(pre_out_name):
                        mul_inputs[input_idx] = str(pre_in_name)
                        replaced = True
                if not replaced:
                    break
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=mul_inputs,
                    graph_index=graph_index,
                )
            else:
                for tensor_name in [
                    str(logistic_out_name),
                    str(sub_out_name),
                    str(mul_sig_out_name),
                    str(mul_sub_out_name),
                ]:
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(tensor_name), None),
                        perm_nchw_to_nhwc,
                    )

                for mul_op, old_mul_out_name, post_output_names in [
                    (mul_sig_op, mul_sig_out_name, sig_post_output_names),
                    (mul_sub_op, mul_sub_out_name, sub_post_output_names),
                ]:
                    canonical_post_output_name = str(post_output_names[0])
                    _set_operator_outputs(
                        model_ir=model_ir,
                        op=mul_op,
                        new_outputs=[canonical_post_output_name],
                        graph_index=graph_index,
                    )
                    for alias_name in post_output_names[1:]:
                        _replace_tensor_inputs(
                            model_ir,
                            str(alias_name),
                            canonical_post_output_name,
                            graph_index=graph_index,
                        )

                    old_mul_out_tensor = model_ir.tensors.get(str(old_mul_out_name), None)
                    canonical_tensor = model_ir.tensors.get(str(canonical_post_output_name), None)
                    if old_mul_out_tensor is not None and canonical_tensor is not None:
                        canonical_tensor.dtype = str(old_mul_out_tensor.dtype)
                        canonical_tensor.quantization = _clone_quantization(old_mul_out_tensor.quantization)
                        canonical_tensor.shape = [int(v) for v in list(old_mul_out_tensor.shape)]
                        canonical_tensor.shape_signature = (
                            [int(v) for v in list(old_mul_out_tensor.shape_signature)]
                            if old_mul_out_tensor.shape_signature is not None
                            else [int(v) for v in list(old_mul_out_tensor.shape)]
                        )
                        _permute_tensor_metadata_if_rank_matches(
                            canonical_tensor,
                            perm_nchw_to_nhwc,
                        )

                remove_indices = {
                    int(gate_pre_idx),
                    int(sig_pre_idx),
                    int(sub_pre_idx),
                    *[int(v) for v in sig_post_indices],
                    *[int(v) for v in sub_post_indices],
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
    return {"optimized_transpose_logistic_sub_mul_postadd_nhwc_chains": int(rewritten)}


def _is_indexed_transpose(
    model_ir: ModelIR,
    op: OperatorIR | None,
    perm: List[int],
) -> bool:
    return bool(
        op is not None
        and str(op.op_type) == "TRANSPOSE"
        and len(op.inputs) >= 2
        and len(op.outputs) == 1
        and _read_transpose_perm(model_ir, op) == perm
    )


def _complementary_gate_prefixes(
    pass_state: ModelIRPassState,
) -> List[Dict[str, Any]]:
    model_ir = pass_state.model_ir
    graph_index = pass_state.graph_index
    model_outputs = {str(name) for name in model_ir.outputs}

    def _data_pre_for_mul(
        mul_op: OperatorIR,
        gate_name: str,
    ) -> Optional[Tuple[OperatorIR, int, str]]:
        if (
            str(mul_op.op_type) != "MUL"
            or len(mul_op.inputs) != 2
            or len(mul_op.outputs) != 1
            or gate_name not in {str(name) for name in mul_op.inputs}
        ):
            return None
        data_names = [
            str(name) for name in mul_op.inputs if str(name) != gate_name
        ]
        if len(data_names) != 1:
            return None
        pre_op = graph_index.producer(data_names[0])
        if (
            not _is_indexed_transpose(
                model_ir,
                pre_op,
                [0, 3, 1, 2],
            )
            or pre_op is None
            or str(pre_op.inputs[0]) in model_outputs
            or data_names[0] in model_outputs
        ):
            return None
        pre_index = graph_index.operator_index(pre_op)
        if pre_index is None:
            return None
        return pre_op, pre_index, data_names[0]

    prefixes: List[Dict[str, Any]] = []
    for gate_pre_op in model_ir.operators:
        if not _is_indexed_transpose(
            model_ir,
            gate_pre_op,
            [0, 3, 1, 2],
        ):
            continue
        gate_pre_in = str(gate_pre_op.inputs[0])
        gate_pre_out = str(gate_pre_op.outputs[0])
        if gate_pre_in in model_outputs or gate_pre_out in model_outputs:
            continue
        gate_users = graph_index.consumers_of(gate_pre_out)
        if len(gate_users) != 1:
            continue
        logistic_op = gate_users[0]
        if (
            str(logistic_op.op_type) != "LOGISTIC"
            or len(logistic_op.inputs) != 1
            or len(logistic_op.outputs) != 1
            or str(logistic_op.inputs[0]) != gate_pre_out
        ):
            continue
        logistic_out = str(logistic_op.outputs[0])
        if logistic_out in model_outputs:
            continue
        logistic_users = graph_index.consumers_of(logistic_out)
        sub_ops = [op for op in logistic_users if str(op.op_type) == "SUB"]
        mul_sig_ops = [op for op in logistic_users if str(op.op_type) == "MUL"]
        if (
            len(logistic_users) != 2
            or len(sub_ops) != 1
            or len(mul_sig_ops) != 1
        ):
            continue
        sub_op = sub_ops[0]
        mul_sig_op = mul_sig_ops[0]
        if (
            len(sub_op.inputs) != 2
            or len(sub_op.outputs) != 1
            or logistic_out not in {str(name) for name in sub_op.inputs}
        ):
            continue
        sub_out = str(sub_op.outputs[0])
        if sub_out in model_outputs:
            continue
        sub_users = graph_index.consumers_of(sub_out)
        if len(sub_users) != 1 or str(sub_users[0].op_type) != "MUL":
            continue
        mul_sub_op = sub_users[0]
        sig_pre = _data_pre_for_mul(mul_sig_op, logistic_out)
        sub_pre = _data_pre_for_mul(mul_sub_op, sub_out)
        if sig_pre is None or sub_pre is None or sig_pre[1] == sub_pre[1]:
            continue
        mul_sig_index = graph_index.operator_index(mul_sig_op)
        mul_sub_index = graph_index.operator_index(mul_sub_op)
        if mul_sig_index is None or mul_sub_index is None:
            continue
        mul_sig_out = str(mul_sig_op.outputs[0])
        mul_sub_out = str(mul_sub_op.outputs[0])
        if mul_sig_out in model_outputs or mul_sub_out in model_outputs:
            continue
        prefixes.append(
            {
                "sig_pre": sig_pre,
                "sub_pre": sub_pre,
                "mul_sig_index": mul_sig_index,
                "mul_sub_index": mul_sub_index,
                "mul_sig_out": mul_sig_out,
                "mul_sub_out": mul_sub_out,
            }
        )
    return prefixes


def _inverse_posts_only(
    pass_state: ModelIRPassState,
    tensor_name: str,
) -> bool:
    model_ir = pass_state.model_ir
    model_outputs = {str(name) for name in model_ir.outputs}
    users = pass_state.graph_index.consumers_of(tensor_name)
    return bool(users) and all(
        _is_indexed_transpose(model_ir, user, [0, 2, 3, 1])
        and str(user.outputs[0]) not in model_outputs
        for user in users
    )


def _has_dual_postconv_gate_candidate(pass_state: ModelIRPassState) -> bool:
    model_ir = pass_state.model_ir
    graph_index = pass_state.graph_index
    model_outputs = {str(name) for name in model_ir.outputs}
    for prefix in _complementary_gate_prefixes(pass_state):
        mul_sig_out = str(prefix["mul_sig_out"])
        mul_sub_out = str(prefix["mul_sub_out"])
        add_sig_users = graph_index.consumers_of(mul_sig_out)
        add_sub_users = graph_index.consumers_of(mul_sub_out)
        if len(add_sig_users) != 1 or len(add_sub_users) != 1:
            continue
        add_sig_op = add_sig_users[0]
        add_sub_op = add_sub_users[0]
        add_sig_index = graph_index.operator_index(add_sig_op)
        add_sub_index = graph_index.operator_index(add_sub_op)
        if (
            add_sig_index is None
            or add_sub_index is None
            or add_sig_index == add_sub_index
            or str(add_sig_op.op_type) != "ADD"
            or str(add_sub_op.op_type) != "ADD"
            or len(add_sig_op.inputs) != 2
            or len(add_sub_op.inputs) != 2
            or len(add_sig_op.outputs) != 1
            or len(add_sub_op.outputs) != 1
        ):
            continue
        add_sig_other = [
            str(name)
            for name in add_sig_op.inputs
            if str(name) != mul_sig_out
        ]
        add_sub_other = [
            str(name)
            for name in add_sub_op.inputs
            if str(name) != mul_sub_out
        ]
        sig_pre = prefix["sig_pre"]
        sub_pre = prefix["sub_pre"]
        if (
            len(add_sig_other) != 1
            or len(add_sub_other) != 1
            or {add_sig_other[0], add_sub_other[0]}
            != {sig_pre[2], sub_pre[2]}
        ):
            continue
        add_sig_out = str(add_sig_op.outputs[0])
        add_sub_out = str(add_sub_op.outputs[0])
        if (
            add_sig_out in model_outputs
            or add_sub_out in model_outputs
            or not _inverse_posts_only(pass_state, add_sig_out)
            or not _inverse_posts_only(pass_state, add_sub_out)
        ):
            continue
        allowed_users = {
            int(prefix["mul_sig_index"]),
            int(prefix["mul_sub_index"]),
            add_sig_index,
            add_sub_index,
        }
        if not set(graph_index.consumer_indices(sig_pre[2])).issubset(
            allowed_users
        ):
            continue
        if not set(graph_index.consumer_indices(sub_pre[2])).issubset(
            allowed_users
        ):
            continue
        return True
    return False


def _has_postadd_gate_candidate(pass_state: ModelIRPassState) -> bool:
    graph_index = pass_state.graph_index
    for prefix in _complementary_gate_prefixes(pass_state):
        sig_pre = prefix["sig_pre"]
        sub_pre = prefix["sub_pre"]
        if set(graph_index.consumer_indices(sig_pre[2])) != {
            int(prefix["mul_sig_index"])
        }:
            continue
        if set(graph_index.consumer_indices(sub_pre[2])) != {
            int(prefix["mul_sub_index"])
        }:
            continue
        if not _inverse_posts_only(pass_state, str(prefix["mul_sig_out"])):
            continue
        if not _inverse_posts_only(pass_state, str(prefix["mul_sub_out"])):
            continue
        return True
    return False

def run_dual_postconv_gate_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Propagate NHWC through guarded complementary-gate output branches."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "LOGISTIC", "SUB", "MUL", "ADD"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    callbacks = [
        (
            "layout.dual_postconv_complementary_gate_nhwc",
            10,
            _has_dual_postconv_gate_candidate,
            _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains,
            "optimized_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains",
        ),
        (
            "layout.postadd_complementary_gate_nhwc",
            20,
            _has_postadd_gate_candidate,
            _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains,
            "optimized_transpose_logistic_sub_mul_postadd_nhwc_chains",
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
