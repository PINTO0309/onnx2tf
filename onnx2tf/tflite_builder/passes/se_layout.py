from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_singleton_constant_tensor,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
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
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR

def _optimize_transpose_se_conv_mul_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips in EfficientNet-style SE conv gating blocks.

    Target (constrained):
      x_nhwc --T(0,3,1,2)--> x_nchw --LOGISTIC--> s1
      MUL(x_nchw, s1) -> sw_nchw --MEAN(axes=[2,3],keepDims=1)--> m_nchw --T(0,2,3,1)--> m_nhwc
      m_nhwc -> ... -> g_nhwc --T(0,3,1,2)--> g_nchw --GATE--> s2
      MUL(sw_nchw, s2) -> y_nchw --T(0,2,3,1)--> y_nhwc

    GATE can be either:
      - LOGISTIC(g_nchw), or
      - ADD(g_nchw, const) -> MUL|DIV(const)

    Rewrite:
      - Bypass the two NHWC->NCHW pre-transposes and NHWC inverse post-transposes.
      - Keep LOGISTIC/ADD/MUL/DIV in NHWC (layout-agnostic).
      - Remap MEAN axes from NCHW to NHWC.
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

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_output_name, [])]
            if len(pre_users) != 2:
                continue

            log1_idx: Optional[int] = None
            mul1_idx: Optional[int] = None
            for user_idx in pre_users:
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if (
                    user_type == "LOGISTIC"
                    and len(user_op.inputs) == 1
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == pre_output_name
                ):
                    log1_idx = int(user_idx)
                elif (
                    user_type == "MUL"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and pre_output_name in {str(v) for v in list(user_op.inputs)}
                ):
                    mul1_idx = int(user_idx)
            if log1_idx is None or mul1_idx is None:
                continue

            log1_op = model_ir.operators[int(log1_idx)]
            log1_out_name = str(log1_op.outputs[0])
            if set(int(v) for v in consumers.get(log1_out_name, [])) != {int(mul1_idx)}:
                continue

            mul1_op = model_ir.operators[int(mul1_idx)]
            mul1_inputs = [str(v) for v in list(mul1_op.inputs)]
            if pre_output_name not in mul1_inputs or log1_out_name not in mul1_inputs:
                continue
            mul1_data_input_index = int(0 if mul1_inputs[0] == pre_output_name else 1)
            mul1_out_name = str(mul1_op.outputs[0])
            if mul1_out_name in model_outputs:
                continue

            mul1_users = [int(v) for v in consumers.get(mul1_out_name, []) if int(v) != int(mul1_idx)]
            if len(mul1_users) != 2:
                continue

            mean_idx: Optional[int] = None
            mul2_idx: Optional[int] = None
            for user_idx in mul1_users:
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if (
                    user_type == "MEAN"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == mul1_out_name
                    and bool(user_op.options.get("keepDims", False))
                ):
                    mean_idx = int(user_idx)
                elif (
                    user_type == "MUL"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and mul1_out_name in {str(v) for v in list(user_op.inputs)}
                ):
                    mul2_idx = int(user_idx)
            if mean_idx is None or mul2_idx is None:
                continue

            mean_op = model_ir.operators[int(mean_idx)]
            mean_out_name = str(mean_op.outputs[0])
            if mean_out_name in model_outputs:
                continue
            mean_axes_name = str(mean_op.inputs[1])
            mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
            mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
            if mean_axes_vals is None or len(mean_axes_vals) == 0:
                continue

            x_tensor = model_ir.tensors.get(pre_input_name, None)
            x_shape = list(x_tensor.shape) if x_tensor is not None else []
            rank = int(len(x_shape)) if len(x_shape) > 0 else 4
            if rank != 4:
                continue
            normalized_axes: List[int] = []
            valid_axes = True
            for axis in mean_axes_vals:
                a = int(axis)
                if a < 0:
                    a += int(rank)
                if a < 0 or a >= int(rank):
                    valid_axes = False
                    break
                normalized_axes.append(int(a))
            if not valid_axes or sorted(normalized_axes) != [2, 3]:
                continue
            mapped_axes = [int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes]
            if sorted(mapped_axes) != [1, 2]:
                continue

            mean_users = [int(v) for v in consumers.get(mean_out_name, []) if int(v) != int(mean_idx)]
            if len(mean_users) == 0:
                continue
            post_mean_indices: List[int] = []
            post_mean_outputs: List[str] = []
            mean_passthrough_via_squeeze = False
            mean_squeeze_idx: Optional[int] = None
            mean_squeeze_op: Optional[OperatorIR] = None
            # Legacy path: MEAN output is wrapped by NCHW->NHWC transpose adapter(s).
            valid_post_mean = True
            for user_idx in mean_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) != "TRANSPOSE"
                    or len(user_op.inputs) < 2
                    or len(user_op.outputs) != 1
                    or str(user_op.inputs[0]) != mean_out_name
                    or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                    or str(user_op.outputs[0]) in model_outputs
                ):
                    valid_post_mean = False
                    break
                post_mean_indices.append(int(user_idx))
                post_mean_outputs.append(str(user_op.outputs[0]))
            # EfficientDet-like variant: MEAN output is consumed by SQUEEZE directly
            # and converted back to NHWC with RESHAPE (no explicit post-transpose).
            if not valid_post_mean or len(post_mean_indices) == 0:
                post_mean_indices = []
                post_mean_outputs = []
                if len(mean_users) != 1:
                    continue
                candidate_idx = int(mean_users[0])
                candidate_op = model_ir.operators[int(candidate_idx)]
                if (
                    str(candidate_op.op_type) != "SQUEEZE"
                    or len(candidate_op.inputs) != 1
                    or len(candidate_op.outputs) != 1
                    or str(candidate_op.inputs[0]) != mean_out_name
                    or str(candidate_op.outputs[0]) in model_outputs
                ):
                    continue
                mean_passthrough_via_squeeze = True
                mean_squeeze_idx = int(candidate_idx)
                mean_squeeze_op = candidate_op

            mul2_op = model_ir.operators[int(mul2_idx)]
            mul2_inputs = [str(v) for v in list(mul2_op.inputs)]
            if mul2_inputs[0] == mul1_out_name:
                gate_out_name = str(mul2_inputs[1])
            elif mul2_inputs[1] == mul1_out_name:
                gate_out_name = str(mul2_inputs[0])
            else:
                continue

            gate_pre_idx: Optional[int] = None
            gate_pre_op: Optional[OperatorIR] = None
            gate_rewrite_op: Optional[OperatorIR] = None
            gate_rewrite_input_index: Optional[int] = None
            gate_nchw_tensor_names: List[str] = []
            gate_pre_needs_removal = False
            gate_pre_is_reshape = False

            if set(int(v) for v in consumers.get(gate_out_name, [])) != {int(mul2_idx)}:
                continue

            gate_prod_idx = producers.get(gate_out_name, None)
            if gate_prod_idx is None:
                continue
            gate_prod_op = model_ir.operators[int(gate_prod_idx)]
            gate_prod_type = str(gate_prod_op.op_type)

            if (
                gate_prod_type == "LOGISTIC"
                and len(gate_prod_op.inputs) == 1
                and len(gate_prod_op.outputs) == 1
                and str(gate_prod_op.outputs[0]) == gate_out_name
            ):
                gate_pre_out_name = str(gate_prod_op.inputs[0])
                gate_pre_idx = producers.get(gate_pre_out_name, None)
                if gate_pre_idx is None:
                    continue
                gate_pre_op = model_ir.operators[int(gate_pre_idx)]
                # Legacy path: gate LOGISTIC input is wrapped by NHWC->NCHW transpose.
                if (
                    str(gate_pre_op.op_type) != "TRANSPOSE"
                    or len(gate_pre_op.inputs) < 2
                    or len(gate_pre_op.outputs) != 1
                    or str(gate_pre_op.outputs[0]) != gate_pre_out_name
                    or _read_transpose_perm(model_ir, gate_pre_op) != perm_nhwc_to_nchw
                    or gate_pre_out_name in model_outputs
                ):
                    # EfficientDet-like path: gate LOGISTIC input is RESHAPE to NCHW
                    # from NHWC tensor. Keep RESHAPE op and rewrite shape to NHWC.
                    if (
                        str(gate_pre_op.op_type) != "RESHAPE"
                        or len(gate_pre_op.inputs) < 2
                        or len(gate_pre_op.outputs) != 1
                        or str(gate_pre_op.outputs[0]) != gate_pre_out_name
                        or gate_pre_out_name in model_outputs
                    ):
                        continue
                    gate_shape_tensor = model_ir.tensors.get(str(gate_pre_op.inputs[1]), None)
                    gate_shape_vals = _read_const_ints_from_tensor(gate_shape_tensor)
                    if gate_shape_vals is None or len(gate_shape_vals) != 4:
                        continue
                    if set(int(v) for v in consumers.get(gate_pre_out_name, [])) != {int(gate_prod_idx)}:
                        continue
                    swapped_shape = [
                        int(gate_shape_vals[0]),
                        int(gate_shape_vals[2]),
                        int(gate_shape_vals[3]),
                        int(gate_shape_vals[1]),
                    ]
                    _write_const_ints_to_tensor(gate_shape_tensor, swapped_shape)
                    gate_pre_is_reshape = True
                if set(int(v) for v in consumers.get(gate_pre_out_name, [])) != {int(gate_prod_idx)}:
                    continue

                gate_rewrite_op = gate_prod_op
                if not gate_pre_is_reshape:
                    gate_rewrite_input_index = 0
                    gate_pre_needs_removal = True
                gate_nchw_tensor_names = [str(gate_pre_out_name), str(gate_out_name)]

            elif (
                gate_prod_type in {"MUL", "DIV"}
                and len(gate_prod_op.inputs) == 2
                and len(gate_prod_op.outputs) == 1
                and str(gate_prod_op.outputs[0]) == gate_out_name
            ):
                gate_scale_inputs = [str(v) for v in list(gate_prod_op.inputs)]
                gate_add_idx: Optional[int] = None
                gate_add_out_name: Optional[str] = None
                gate_scale_side_name: Optional[str] = None
                for gate_scale_input_name in gate_scale_inputs:
                    gate_scale_input_prod = producers.get(str(gate_scale_input_name), None)
                    if gate_scale_input_prod is not None:
                        gate_scale_input_op = model_ir.operators[int(gate_scale_input_prod)]
                        if (
                            str(gate_scale_input_op.op_type) == "ADD"
                            and len(gate_scale_input_op.inputs) == 2
                            and len(gate_scale_input_op.outputs) == 1
                            and str(gate_scale_input_op.outputs[0]) == str(gate_scale_input_name)
                        ):
                            gate_add_idx = int(gate_scale_input_prod)
                            gate_add_out_name = str(gate_scale_input_name)
                            continue
                    gate_scale_side_name = str(gate_scale_input_name)
                if (
                    gate_add_idx is None
                    or gate_add_out_name is None
                    or gate_scale_side_name is None
                    or not _is_singleton_constant_tensor(model_ir, gate_scale_side_name)
                ):
                    continue
                if set(int(v) for v in consumers.get(gate_add_out_name, [])) != {int(gate_prod_idx)}:
                    continue

                gate_add_op = model_ir.operators[int(gate_add_idx)]
                gate_add_inputs = [str(v) for v in list(gate_add_op.inputs)]
                gate_add_data_input_index: Optional[int] = None
                gate_pre_out_name: Optional[str] = None
                for gate_add_input_index, gate_add_input_name in enumerate(gate_add_inputs):
                    gate_add_input_prod = producers.get(str(gate_add_input_name), None)
                    if gate_add_input_prod is not None:
                        gate_add_input_op = model_ir.operators[int(gate_add_input_prod)]
                        if (
                            str(gate_add_input_op.op_type) == "TRANSPOSE"
                            and len(gate_add_input_op.inputs) >= 2
                            and len(gate_add_input_op.outputs) == 1
                            and str(gate_add_input_op.outputs[0]) == str(gate_add_input_name)
                            and _read_transpose_perm(model_ir, gate_add_input_op) == perm_nhwc_to_nchw
                            and str(gate_add_input_name) not in model_outputs
                        ):
                            gate_pre_idx = int(gate_add_input_prod)
                            gate_pre_op = gate_add_input_op
                            gate_pre_out_name = str(gate_add_input_name)
                            gate_add_data_input_index = int(gate_add_input_index)
                            continue
                    if not _is_singleton_constant_tensor(model_ir, str(gate_add_input_name)):
                        gate_add_data_input_index = None
                        break
                if (
                    gate_pre_idx is None
                    or gate_pre_op is None
                    or gate_pre_out_name is None
                    or gate_add_data_input_index is None
                ):
                    continue
                if set(int(v) for v in consumers.get(gate_pre_out_name, [])) != {int(gate_add_idx)}:
                    continue

                gate_rewrite_op = gate_add_op
                gate_rewrite_input_index = int(gate_add_data_input_index)
                gate_nchw_tensor_names = [str(gate_add_out_name), str(gate_out_name)]
                gate_pre_needs_removal = True

            else:
                continue

            if (
                gate_pre_idx is None
                or gate_pre_op is None
                or gate_rewrite_op is None
                or (gate_rewrite_input_index is None and not gate_pre_is_reshape)
            ):
                continue

            mul2_out_name = str(mul2_op.outputs[0])
            if mul2_out_name in model_outputs:
                continue
            mul2_users = [int(v) for v in consumers.get(mul2_out_name, []) if int(v) != int(mul2_idx)]
            if len(mul2_users) == 0:
                continue
            post_out_indices: List[int] = []
            post_out_outputs: List[str] = []
            valid_post_out = True
            for user_idx in mul2_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) != "TRANSPOSE"
                    or len(user_op.inputs) < 2
                    or len(user_op.outputs) != 1
                    or str(user_op.inputs[0]) != mul2_out_name
                    or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                    or str(user_op.outputs[0]) in model_outputs
                ):
                    valid_post_out = False
                    break
                post_out_indices.append(int(user_idx))
                post_out_outputs.append(str(user_op.outputs[0]))
            if not valid_post_out or len(post_out_indices) == 0:
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=log1_op,
                new_inputs=[pre_input_name],
                graph_index=graph_index,
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=mul1_op,
                input_index=int(mul1_data_input_index),
                new_input_name=pre_input_name,
                graph_index=graph_index,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(log1_out_name, None),
                perm_nchw_to_nhwc,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(mul1_out_name, None),
                perm_nchw_to_nhwc,
            )

            _write_const_ints_to_tensor(mean_axes_tensor, [int(v) for v in mapped_axes])
            if isinstance(mean_op.options, dict):
                mean_options = dict(mean_op.options)
                for key in ["axis", "axes", "onnxRawAxes"]:
                    value = mean_options.get(key, None)
                    if isinstance(value, list) and len(value) == len(mapped_axes):
                        mean_options[key] = [int(v) for v in mapped_axes]
                mean_op.options = mean_options
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(mean_out_name, None),
                perm_nchw_to_nhwc,
            )
            for post_mean_out_name in post_mean_outputs:
                _replace_tensor_inputs(
                    model_ir,
                    post_mean_out_name,
                    mean_out_name,
                    graph_index=graph_index,
                )
            if mean_passthrough_via_squeeze and mean_squeeze_op is not None:
                squeeze_dims = list(mean_squeeze_op.options.get("squeezeDims", []))
                remapped_dims: List[int] = []
                rank4 = 4
                for axis in squeeze_dims:
                    a = int(axis)
                    if a < 0:
                        a += int(rank4)
                    if a < 0 or a >= int(rank4):
                        remapped_dims = []
                        break
                    remapped_dims.append(int(perm_nhwc_to_nchw[int(a)]))
                if len(remapped_dims) > 0:
                    mean_squeeze_op.options = dict(mean_squeeze_op.options)
                    mean_squeeze_op.options["squeezeDims"] = sorted([int(v) for v in remapped_dims])
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(mean_squeeze_op.outputs[0]), None),
                    perm_nchw_to_nhwc,
                )

            if gate_rewrite_input_index is not None:
                gate_pre_input_name = str(gate_pre_op.inputs[0])
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=gate_rewrite_op,
                    input_index=int(gate_rewrite_input_index),
                    new_input_name=gate_pre_input_name,
                    graph_index=graph_index,
                )
            for gate_nchw_tensor_name in gate_nchw_tensor_names:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(gate_nchw_tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            representative_output_name = str(post_out_outputs[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=mul2_op,
                new_outputs=[representative_output_name],
                graph_index=graph_index,
            )
            for alias_name in post_out_outputs[1:]:
                _replace_tensor_inputs(
                    model_ir,
                    alias_name,
                    representative_output_name,
                    graph_index=graph_index,
                )

            old_mul2_tensor = model_ir.tensors.get(mul2_out_name, None)
            representative_tensor = model_ir.tensors.get(representative_output_name, None)
            if old_mul2_tensor is not None and representative_tensor is not None:
                representative_tensor.dtype = str(old_mul2_tensor.dtype)
                representative_tensor.quantization = _clone_quantization(old_mul2_tensor.quantization)
                representative_tensor.shape = [int(v) for v in list(old_mul2_tensor.shape)]
                representative_tensor.shape_signature = (
                    [int(v) for v in list(old_mul2_tensor.shape_signature)]
                    if old_mul2_tensor.shape_signature is not None
                    else [int(v) for v in list(old_mul2_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    representative_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = set([int(pre_idx)])
            if gate_pre_needs_removal and gate_pre_idx is not None:
                remove_indices.add(int(gate_pre_idx))
            remove_indices.update(int(v) for v in post_mean_indices)
            remove_indices.update(int(v) for v in post_out_indices)
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
    return {"optimized_transpose_se_conv_mul_prepost_nhwc_chains": int(rewritten)}


def run_se_conv_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Propagate NHWC through guarded convolutional SE gates."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "LOGISTIC", "MUL", "MEAN"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        graph_index = pass_state.graph_index
        model_outputs = {str(name) for name in candidate_model.outputs}
        for pre_op in candidate_model.operators:
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or _read_transpose_perm(candidate_model, pre_op)
                != [0, 3, 1, 2]
            ):
                continue
            pre_input_name = str(pre_op.inputs[0])
            pre_output_name = str(pre_op.outputs[0])
            if pre_output_name in model_outputs:
                continue
            pre_users = sorted(set(graph_index.consumer_indices(pre_output_name)))
            if len(pre_users) != 2:
                continue
            log1_idx: Optional[int] = None
            mul1_idx: Optional[int] = None
            for user_idx in pre_users:
                user_op = candidate_model.operators[user_idx]
                if (
                    str(user_op.op_type) == "LOGISTIC"
                    and len(user_op.inputs) == 1
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == pre_output_name
                ):
                    log1_idx = user_idx
                elif (
                    str(user_op.op_type) == "MUL"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and pre_output_name in {str(name) for name in user_op.inputs}
                ):
                    mul1_idx = user_idx
            if log1_idx is None or mul1_idx is None:
                continue
            log1_op = candidate_model.operators[log1_idx]
            log1_output_name = str(log1_op.outputs[0])
            if set(graph_index.consumer_indices(log1_output_name)) != {mul1_idx}:
                continue
            mul1_op = candidate_model.operators[mul1_idx]
            if {str(name) for name in mul1_op.inputs} != {
                pre_output_name,
                log1_output_name,
            }:
                continue
            mul1_output_name = str(mul1_op.outputs[0])
            if mul1_output_name in model_outputs:
                continue
            mul1_users = sorted(set(graph_index.consumer_indices(mul1_output_name)))
            mean_idx: Optional[int] = None
            mul2_idx: Optional[int] = None
            for user_idx in mul1_users:
                user_op = candidate_model.operators[user_idx]
                if (
                    str(user_op.op_type) == "MEAN"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == mul1_output_name
                    and bool(user_op.options.get("keepDims", False))
                ):
                    mean_idx = user_idx
                elif (
                    str(user_op.op_type) == "MUL"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and mul1_output_name in {str(name) for name in user_op.inputs}
                ):
                    mul2_idx = user_idx
            if mean_idx is None or mul2_idx is None or len(mul1_users) != 2:
                continue
            mean_op = candidate_model.operators[mean_idx]
            axes = _read_const_ints_from_tensor(
                candidate_model.tensors.get(str(mean_op.inputs[1]))
            )
            input_tensor = candidate_model.tensors.get(pre_input_name)
            rank = len(input_tensor.shape) if input_tensor is not None else 4
            if axes is None or rank != 4:
                continue
            normalized_axes = [
                int(axis) + rank if int(axis) < 0 else int(axis)
                for axis in axes
            ]
            if sorted(normalized_axes) != [2, 3]:
                continue
            mean_output_name = str(mean_op.outputs[0])
            if mean_output_name in model_outputs:
                continue
            mean_users = sorted(set(graph_index.consumer_indices(mean_output_name)))
            if not mean_users:
                continue
            inverse_posts = all(
                str(candidate_model.operators[user_idx].op_type) == "TRANSPOSE"
                and len(candidate_model.operators[user_idx].inputs) >= 2
                and len(candidate_model.operators[user_idx].outputs) == 1
                and str(candidate_model.operators[user_idx].inputs[0])
                == mean_output_name
                and _read_transpose_perm(
                    candidate_model,
                    candidate_model.operators[user_idx],
                ) == [0, 2, 3, 1]
                and str(candidate_model.operators[user_idx].outputs[0])
                not in model_outputs
                for user_idx in mean_users
            )
            squeeze_post = (
                len(mean_users) == 1
                and str(candidate_model.operators[mean_users[0]].op_type)
                == "SQUEEZE"
                and len(candidate_model.operators[mean_users[0]].inputs) == 1
                and len(candidate_model.operators[mean_users[0]].outputs) == 1
                and str(candidate_model.operators[mean_users[0]].inputs[0])
                == mean_output_name
                and str(candidate_model.operators[mean_users[0]].outputs[0])
                not in model_outputs
            )
            if not inverse_posts and not squeeze_post:
                continue
            mul2_op = candidate_model.operators[mul2_idx]
            gate_names = [
                str(name)
                for name in mul2_op.inputs
                if str(name) != mul1_output_name
            ]
            if len(gate_names) != 1:
                continue
            gate_name = gate_names[0]
            if set(graph_index.consumer_indices(gate_name)) != {mul2_idx}:
                continue
            gate_op = graph_index.producer(gate_name)
            if gate_op is None or str(gate_op.op_type) not in {
                "LOGISTIC",
                "MUL",
                "DIV",
            }:
                continue
            mul2_output_name = str(mul2_op.outputs[0])
            if mul2_output_name in model_outputs:
                continue
            mul2_users = sorted(set(graph_index.consumer_indices(mul2_output_name)))
            if mul2_users and all(
                str(candidate_model.operators[user_idx].op_type) == "TRANSPOSE"
                and len(candidate_model.operators[user_idx].inputs) >= 2
                and len(candidate_model.operators[user_idx].outputs) == 1
                and str(candidate_model.operators[user_idx].inputs[0])
                == mul2_output_name
                and _read_transpose_perm(
                    candidate_model,
                    candidate_model.operators[user_idx],
                ) == [0, 2, 3, 1]
                and str(candidate_model.operators[user_idx].outputs[0])
                not in model_outputs
                for user_idx in mul2_users
            ):
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_se_conv_mul_prepost_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_se_conv_mul_prepost_nhwc_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.se_conv_gate_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_transpose_se_conv_mul_prepost_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}

def _optimize_transpose_se_fc_mul_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips in SE-like gating blocks.

    Target:
      x_nhwc --T(0,3,1,2)--> x_nchw
      x_nhwc --AVG_POOL_2D--> p_nhwc --T(0,3,1,2)--> p_nchw
      p_nchw --RESHAPE--> r --FULLY_CONNECTED--> f --PRELU--> g --RESHAPE--> s_nchw
      MUL(x_nchw, s_nchw) -> m_nchw --T(0,2,3,1)--> m_nhwc

    Rewrite:
      - Bypass both pre-transposes and the post-transpose.
      - Keep AVG_POOL/FC/PRELU chain as-is, but rewrite final gate reshape to NHWC.
      - Feed MUL directly with x_nhwc and gate_nhwc.
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    gate_head_passthrough_unary = {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "LOGISTIC",
        "TANH",
        "HARD_SWISH",
    }
    output_bridge_passthrough_unary = {
        "GELU",
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "LOGISTIC",
        "TANH",
        "HARD_SWISH",
    }

    def _clone_const_tensor_for_operator_input(
        *,
        op: OperatorIR,
        input_index: int,
        tag: str,
    ) -> str:
        input_name = str(op.inputs[int(input_index)])
        tensor = model_ir.tensors.get(input_name, None)
        if tensor is None or tensor.data is None:
            return input_name
        cloned_name = f"{input_name}{tag}"
        suffix = 1
        while cloned_name in model_ir.tensors:
            cloned_name = f"{input_name}{tag}_{suffix}"
            suffix += 1
        model_ir.tensors[cloned_name] = TensorIR(
            name=cloned_name,
            dtype=str(tensor.dtype),
            shape=[int(v) for v in list(tensor.shape)] if tensor.shape is not None else [],
            shape_signature=(
                [int(v) for v in list(tensor.shape_signature)]
                if tensor.shape_signature is not None
                else (
                    [int(v) for v in list(tensor.shape)]
                    if tensor.shape is not None
                    else []
                )
            ),
            data=np.array(tensor.data, copy=True),
            is_variable=bool(tensor.is_variable),
            quantization=_clone_quantization(tensor.quantization),
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=op,
            input_index=int(input_index),
            new_input_name=cloned_name,
        )
        return str(cloned_name)

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_mul_idx, pre_mul_op in enumerate(model_ir.operators):
            if str(pre_mul_op.op_type) != "TRANSPOSE" or len(pre_mul_op.inputs) < 2 or len(pre_mul_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_mul_op) != perm_nhwc_to_nchw:
                continue

            x_nhwc_name = str(pre_mul_op.inputs[0])
            x_nchw_name = str(pre_mul_op.outputs[0])
            if x_nchw_name in model_outputs:
                continue

            x_nchw_users = [int(v) for v in consumers.get(x_nchw_name, [])]
            if len(x_nchw_users) == 0:
                continue

            for mul_idx in x_nchw_users:
                mul_op = model_ir.operators[int(mul_idx)]
                if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                    continue

                mul_in0 = str(mul_op.inputs[0])
                mul_in1 = str(mul_op.inputs[1])
                if mul_in0 == x_nchw_name:
                    gate_name = str(mul_in1)
                    x_side_index = 0
                elif mul_in1 == x_nchw_name:
                    gate_name = str(mul_in0)
                    x_side_index = 1
                else:
                    continue

                gate_post_name = str(gate_name)
                gate_post_idx = producers.get(gate_post_name, None)
                if gate_post_idx is None:
                    continue
                gate_post_op = model_ir.operators[int(gate_post_idx)]
                gate_reshape_idx = int(gate_post_idx)
                gate_reshape_op = gate_post_op
                gate_requires_post_unary = False
                if str(gate_reshape_op.op_type) != "RESHAPE":
                    if (
                        str(gate_post_op.op_type) in gate_head_passthrough_unary
                        and len(gate_post_op.inputs) == 1
                        and len(gate_post_op.outputs) == 1
                        and str(gate_post_op.outputs[0]) == gate_post_name
                    ):
                        gate_reshape_input_name = str(gate_post_op.inputs[0])
                        gate_reshape_idx_candidate = producers.get(gate_reshape_input_name, None)
                        if gate_reshape_idx_candidate is None:
                            continue
                        gate_reshape_op = model_ir.operators[int(gate_reshape_idx_candidate)]
                        if (
                            str(gate_reshape_op.op_type) != "RESHAPE"
                            or len(gate_reshape_op.inputs) < 1
                            or len(gate_reshape_op.outputs) != 1
                            or str(gate_reshape_op.outputs[0]) != gate_reshape_input_name
                        ):
                            continue
                        gate_reshape_idx = int(gate_reshape_idx_candidate)
                        gate_requires_post_unary = True
                    else:
                        continue
                if gate_post_name in model_outputs:
                    continue

                # Allow stacked gate heads:
                #   RESHAPE -> ((FULLY_CONNECTED | 1x1 CONV_2D | 1x1 DEPTHWISE_CONV_2D) -> PRELU/UNARY)* -> RESHAPE(gate)
                gate_mlp_input_name = str(gate_reshape_op.inputs[0])
                flat_idx: Optional[int] = None
                flat_op: Optional[OperatorIR] = None
                saw_fc = False
                max_hops = 16
                hop = 0
                while hop < max_hops:
                    producer_idx = producers.get(gate_mlp_input_name, None)
                    if producer_idx is None:
                        break
                    producer_op = model_ir.operators[int(producer_idx)]
                    producer_type = str(producer_op.op_type)
                    if (
                        producer_type == "RESHAPE"
                        and len(producer_op.inputs) >= 1
                        and len(producer_op.outputs) == 1
                        and str(producer_op.outputs[0]) == gate_mlp_input_name
                    ):
                        flat_idx = int(producer_idx)
                        flat_op = producer_op
                        break
                    if (
                        producer_type == "PRELU"
                        and len(producer_op.inputs) == 2
                        and len(producer_op.outputs) == 1
                        and str(producer_op.outputs[0]) == gate_mlp_input_name
                    ):
                        gate_mlp_input_name = str(producer_op.inputs[0])
                        hop += 1
                        continue
                    if (
                        producer_type in gate_head_passthrough_unary
                        and len(producer_op.inputs) == 1
                        and len(producer_op.outputs) == 1
                        and str(producer_op.outputs[0]) == gate_mlp_input_name
                    ):
                        gate_mlp_input_name = str(producer_op.inputs[0])
                        hop += 1
                        continue
                    if (
                        producer_type == "FULLY_CONNECTED"
                        and len(producer_op.inputs) >= 2
                        and len(producer_op.outputs) == 1
                        and str(producer_op.outputs[0]) == gate_mlp_input_name
                    ):
                        saw_fc = True
                        gate_mlp_input_name = str(producer_op.inputs[0])
                        hop += 1
                        continue
                    if (
                        producer_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
                        and len(producer_op.inputs) >= 2
                        and len(producer_op.outputs) == 1
                        and str(producer_op.outputs[0]) == gate_mlp_input_name
                    ):
                        conv_kernel_tensor = model_ir.tensors.get(str(producer_op.inputs[1]), None)
                        conv_kernel_shape = list(conv_kernel_tensor.shape) if conv_kernel_tensor is not None else []
                        is_1x1_conv = (
                            producer_type == "CONV_2D"
                            and len(conv_kernel_shape) == 4
                            and int(conv_kernel_shape[1]) == 1
                            and int(conv_kernel_shape[2]) == 1
                        )
                        is_1x1_dwconv = (
                            producer_type == "DEPTHWISE_CONV_2D"
                            and len(conv_kernel_shape) == 4
                            and int(conv_kernel_shape[1]) == 1
                            and int(conv_kernel_shape[2]) == 1
                        )
                        if not (is_1x1_conv or is_1x1_dwconv):
                            break
                        saw_fc = True
                        gate_mlp_input_name = str(producer_op.inputs[0])
                        hop += 1
                        continue
                    break
                if flat_idx is None or flat_op is None or not saw_fc:
                    continue

                pre_gate_name = str(flat_op.inputs[0])
                pre_gate_idx = producers.get(pre_gate_name, None)
                if pre_gate_idx is None:
                    continue
                pre_gate_op = model_ir.operators[int(pre_gate_idx)]

                pool_out_name: Optional[str] = None
                pool_idx: Optional[int] = None
                removable_pre_gate_idx: Optional[int] = None
                rewrite_pool_mean_to_nhwc = False
                mapped_mean_axes: Optional[List[int]] = None

                # Legacy path: pooled output is wrapped by NHWC->NCHW transpose.
                if (
                    str(pre_gate_op.op_type) == "TRANSPOSE"
                    and len(pre_gate_op.inputs) >= 2
                    and len(pre_gate_op.outputs) == 1
                    and str(pre_gate_op.outputs[0]) == pre_gate_name
                    and _read_transpose_perm(model_ir, pre_gate_op) == perm_nhwc_to_nchw
                    and pre_gate_name not in model_outputs
                ):
                    pool_out_name = str(pre_gate_op.inputs[0])
                    pool_idx = producers.get(pool_out_name, None)
                    removable_pre_gate_idx = int(pre_gate_idx)
                # Already optimized path: flatten consumes pooled NHWC output directly.
                elif (
                    str(pre_gate_op.op_type) in {"AVERAGE_POOL_2D", "MEAN"}
                    and len(pre_gate_op.inputs) >= 1
                    and len(pre_gate_op.outputs) == 1
                    and str(pre_gate_op.outputs[0]) == pre_gate_name
                ):
                    pool_out_name = str(pre_gate_name)
                    pool_idx = int(pre_gate_idx)
                    removable_pre_gate_idx = None
                else:
                    continue

                if pool_out_name is None or pool_idx is None:
                    continue
                pool_op = model_ir.operators[int(pool_idx)]
                pool_op_type = str(pool_op.op_type)
                if pool_op_type == "AVERAGE_POOL_2D":
                    if len(pool_op.inputs) != 1 or len(pool_op.outputs) != 1:
                        continue
                    if str(pool_op.outputs[0]) != str(pool_out_name):
                        continue
                    if str(pool_op.inputs[0]) != x_nhwc_name:
                        continue
                elif pool_op_type == "MEAN":
                    if len(pool_op.inputs) < 2 or len(pool_op.outputs) != 1:
                        continue
                    if str(pool_op.outputs[0]) != str(pool_out_name):
                        continue
                    pool_keep_dims = bool(
                        pool_op.options.get(
                            "keepDims",
                            pool_op.options.get(
                                "keep_dims",
                                pool_op.options.get("keepdims", False),
                            ),
                        )
                    )
                    if not pool_keep_dims:
                        continue
                    pool_mean_input_name = str(pool_op.inputs[0])
                    mean_axes_name = str(pool_op.inputs[1])
                    mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                    mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
                    if mean_axes_vals is None or len(mean_axes_vals) == 0:
                        continue
                    normalized_axes: List[int] = []
                    axes_valid = True
                    for axis in mean_axes_vals:
                        a = int(axis)
                        if a < 0:
                            a += 4
                        if a < 0 or a >= 4:
                            axes_valid = False
                            break
                        normalized_axes.append(int(a))
                    if not axes_valid:
                        continue
                    if pool_mean_input_name == x_nhwc_name:
                        if sorted(normalized_axes) != [1, 2]:
                            continue
                    elif pool_mean_input_name == x_nchw_name:
                        if sorted(normalized_axes) != [2, 3]:
                            continue
                        mapped_mean_axes = [int(perm_nhwc_to_nchw[int(v)]) for v in normalized_axes]
                        if sorted(mapped_mean_axes) != [1, 2]:
                            continue
                        rewrite_pool_mean_to_nhwc = True
                    else:
                        continue
                else:
                    continue

                mul_out_name = str(mul_op.outputs[0])
                if mul_out_name in model_outputs:
                    continue
                mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
                if len(mul_users) == 0:
                    continue

                post_indices: List[int] = []
                post_output_names: List[str] = []
                post_bridge_op: OperatorIR = mul_op
                post_bridge_output_name = str(mul_out_name)
                post_bridge_via_unary = False
                post_bridge_unary_idx: Optional[int] = None
                valid_posts = True
                for user_idx in mul_users:
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) != "TRANSPOSE"
                    ):
                        if (
                            str(user_op.op_type) in output_bridge_passthrough_unary
                            and len(user_op.inputs) == 1
                            and len(user_op.outputs) == 1
                            and str(user_op.inputs[0]) == mul_out_name
                            and post_bridge_unary_idx is None
                        ):
                            unary_out_name = str(user_op.outputs[0])
                            if unary_out_name in model_outputs:
                                valid_posts = False
                                break
                            unary_users = [int(v) for v in consumers.get(unary_out_name, [])]
                            if len(unary_users) == 0:
                                valid_posts = False
                                break
                            valid_unary_tail = True
                            for unary_user_idx in unary_users:
                                unary_user_op = model_ir.operators[int(unary_user_idx)]
                                if (
                                    str(unary_user_op.op_type) != "TRANSPOSE"
                                    or len(unary_user_op.inputs) < 2
                                    or len(unary_user_op.outputs) != 1
                                    or str(unary_user_op.inputs[0]) != unary_out_name
                                    or _read_transpose_perm(model_ir, unary_user_op) != perm_nchw_to_nhwc
                                    or str(unary_user_op.outputs[0]) in model_outputs
                                ):
                                    valid_unary_tail = False
                                    break
                            if not valid_unary_tail:
                                valid_posts = False
                                break
                            post_bridge_via_unary = True
                            post_bridge_unary_idx = int(user_idx)
                            post_bridge_op = user_op
                            post_bridge_output_name = str(unary_out_name)
                            post_indices.extend([int(v) for v in unary_users])
                            post_output_names.extend(
                                [str(model_ir.operators[int(v)].outputs[0]) for v in unary_users]
                            )
                            continue
                        valid_posts = False
                        break
                    if (
                        len(user_op.inputs) < 2
                        or len(user_op.outputs) != 1
                        or str(user_op.inputs[0]) != mul_out_name
                        or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                        or str(user_op.outputs[0]) in model_outputs
                    ):
                        valid_posts = False
                        break
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                if not valid_posts or len(post_indices) == 0:
                    continue

                # Rewrite gate reshape shape [N,C,H,W] -> [N,H,W,C].
                shape_rewritten = False
                if len(gate_reshape_op.inputs) >= 2:
                    gate_shape_name = str(gate_reshape_op.inputs[1])
                    gate_shape_tensor = model_ir.tensors.get(gate_shape_name, None)
                    gate_shape_vals = _read_const_ints_from_tensor(gate_shape_tensor)
                    if gate_shape_vals is not None and len(gate_shape_vals) == 4:
                        swapped_shape = [
                            int(gate_shape_vals[0]),
                            int(gate_shape_vals[2]),
                            int(gate_shape_vals[3]),
                            int(gate_shape_vals[1]),
                        ]
                        _write_const_ints_to_tensor(gate_shape_tensor, swapped_shape)
                        shape_rewritten = True

                if isinstance(gate_reshape_op.options, dict):
                    opts = dict(gate_reshape_op.options)
                    options_changed = False
                    for key in ["newShape", "onnxRawNewShape"]:
                        value = opts.get(key, None)
                        if not isinstance(value, list) or len(value) != 4:
                            continue
                        swapped_value = [
                            int(value[0]),
                            int(value[2]),
                            int(value[3]),
                            int(value[1]),
                        ]
                        if [int(v) for v in list(value)] != swapped_value:
                            opts[key] = swapped_value
                            options_changed = True
                            shape_rewritten = True
                    if options_changed:
                        gate_reshape_op.options = opts

                if not shape_rewritten:
                    continue

                # Bypass pre-gate transpose into flatten reshape when present.
                if removable_pre_gate_idx is not None:
                    _replace_operator_input_at(
                        model_ir=model_ir,
                        op=flat_op,
                        input_index=0,
                        new_input_name=pool_out_name,
                    )
                if rewrite_pool_mean_to_nhwc:
                    pool_inputs = [str(v) for v in list(pool_op.inputs)]
                    pool_inputs[0] = str(x_nhwc_name)
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=pool_op,
                        new_inputs=pool_inputs,
                    )
                    if mapped_mean_axes is not None:
                        mean_axes_name = _clone_const_tensor_for_operator_input(
                            op=pool_op,
                            input_index=1,
                            tag="__se_fc_nhwc_axes",
                        )
                        mean_axes_tensor = model_ir.tensors.get(str(mean_axes_name), None)
                        if not _write_const_ints_to_tensor(mean_axes_tensor, [int(v) for v in mapped_mean_axes]):
                            continue
                        if isinstance(pool_op.options, dict):
                            mean_opts = dict(pool_op.options)
                            for key in ["axis", "axes", "onnxRawAxes"]:
                                value = mean_opts.get(key, None)
                                if isinstance(value, list) and len(value) == len(mapped_mean_axes):
                                    mean_opts[key] = [int(v) for v in mapped_mean_axes]
                            pool_op.options = mean_opts

                # Bypass pre-mul transpose into MUL.
                mul_inputs = [str(v) for v in list(mul_op.inputs)]
                mul_inputs[int(x_side_index)] = str(x_nhwc_name)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=mul_inputs,
                )

                # Gate/MUL metadata: convert from NCHW to NHWC.
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(gate_post_name, None),
                    perm_nchw_to_nhwc,
                )
                if gate_requires_post_unary:
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(gate_reshape_op.outputs[0]), None),
                        perm_nchw_to_nhwc,
                    )
                if rewrite_pool_mean_to_nhwc:
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(pool_out_name), None),
                        perm_nchw_to_nhwc,
                    )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(mul_out_name, None),
                    perm_nchw_to_nhwc,
                )
                if post_bridge_via_unary and post_bridge_unary_idx is not None:
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(post_bridge_output_name), None),
                        perm_nchw_to_nhwc,
                    )

                # Remove post transpose by letting bridge producer emit NHWC directly.
                representative_output_name = str(post_output_names[0])
                _set_operator_outputs(model_ir=model_ir, op=post_bridge_op, new_outputs=[representative_output_name])
                for alias_name in post_output_names[1:]:
                    _replace_tensor_inputs(model_ir, alias_name, representative_output_name)

                old_mul_tensor = model_ir.tensors.get(
                    str(post_bridge_output_name),
                    None,
                )
                representative_tensor = model_ir.tensors.get(representative_output_name, None)
                if old_mul_tensor is not None and representative_tensor is not None:
                    representative_tensor.dtype = str(old_mul_tensor.dtype)
                    representative_tensor.quantization = _clone_quantization(old_mul_tensor.quantization)
                    representative_tensor.shape = [int(v) for v in list(old_mul_tensor.shape)]
                    representative_tensor.shape_signature = (
                        [int(v) for v in list(old_mul_tensor.shape_signature)]
                        if old_mul_tensor.shape_signature is not None
                        else [int(v) for v in list(old_mul_tensor.shape)]
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        representative_tensor,
                        perm_nchw_to_nhwc,
                    )

                remove_indices = set(int(v) for v in post_indices)
                pre_mul_exempt_users = {int(mul_idx)}
                if rewrite_pool_mean_to_nhwc and str(pool_op.op_type) == "MEAN":
                    pre_mul_exempt_users.add(int(pool_idx))
                pre_mul_remaining_users = [
                    int(v)
                    for v in x_nchw_users
                    if int(v) not in pre_mul_exempt_users
                ]
                if len(pre_mul_remaining_users) == 0:
                    remove_indices.add(int(pre_mul_idx))

                if removable_pre_gate_idx is not None:
                    pre_gate_users = [int(v) for v in consumers.get(pre_gate_name, [])]
                    pre_gate_remaining_users = [int(v) for v in pre_gate_users if int(v) != int(flat_idx)]
                    if len(pre_gate_remaining_users) == 0:
                        remove_indices.add(int(removable_pre_gate_idx))

                for remove_idx in sorted(list(remove_indices), reverse=True):
                    del model_ir.operators[int(remove_idx)]

                rewritten += 1
                changed = True
                break

            # Alternate float SE-like path:
            #   x_nhwc --T--> x_nchw --ADD(c)--> a --(MUL|DIV)(c)--> b --MUL(x_nchw,b)--> xh_nchw
            #   xh_nchw --MEAN--> m_nchw --T--> m_nhwc --CONV--CONV--> g_nhwc --T--> g_nchw
            #   g_nchw --MUL(alpha)--ADD(beta)--RELU_0_TO_1--> h_nchw
            #   MUL(xh_nchw, h_nchw) -> y_nchw --T--> y_nhwc
            if not changed:
                add0_idx: Optional[int] = None
                main_mul_idx: Optional[int] = None
                add0_out_name: Optional[str] = None
                main_mul_gate_name: Optional[str] = None

                for user_idx in x_nchw_users:
                    user_op = model_ir.operators[int(user_idx)]
                    user_type = str(user_op.op_type)
                    if user_type == "ADD":
                        if len(user_op.inputs) != 2 or len(user_op.outputs) != 1:
                            continue
                        add_inputs = [str(v) for v in list(user_op.inputs)]
                        if x_nchw_name == add_inputs[0]:
                            side_name = str(add_inputs[1])
                        elif x_nchw_name == add_inputs[1]:
                            side_name = str(add_inputs[0])
                        else:
                            continue
                        if not _is_singleton_constant_tensor(model_ir, side_name):
                            continue
                        if add0_idx is not None:
                            add0_idx = None
                            break
                        add0_idx = int(user_idx)
                        add0_out_name = str(user_op.outputs[0])
                    elif user_type == "MUL":
                        if len(user_op.inputs) != 2 or len(user_op.outputs) != 1:
                            continue
                        mul_inputs = [str(v) for v in list(user_op.inputs)]
                        if x_nchw_name == mul_inputs[0]:
                            other_name = str(mul_inputs[1])
                        elif x_nchw_name == mul_inputs[1]:
                            other_name = str(mul_inputs[0])
                        else:
                            continue
                        if main_mul_idx is not None:
                            main_mul_idx = None
                            break
                        main_mul_idx = int(user_idx)
                        main_mul_gate_name = str(other_name)

                if (
                    add0_idx is not None
                    and main_mul_idx is not None
                    and add0_out_name is not None
                    and main_mul_gate_name is not None
                ):
                    add0_op = model_ir.operators[int(add0_idx)]
                    main_mul_op = model_ir.operators[int(main_mul_idx)]
                    main_mul_out_name = str(main_mul_op.outputs[0])
                    if main_mul_out_name not in model_outputs:
                        scale_idx = producers.get(str(main_mul_gate_name), None)
                        if scale_idx is not None:
                            scale_op = model_ir.operators[int(scale_idx)]
                            scale_type = str(scale_op.op_type)
                            if (
                                scale_type in {"MUL", "DIV"}
                                and len(scale_op.inputs) == 2
                                and len(scale_op.outputs) == 1
                                and str(scale_op.outputs[0]) == str(main_mul_gate_name)
                            ):
                                scale_inputs = [str(v) for v in list(scale_op.inputs)]
                                if str(add0_out_name) == scale_inputs[0]:
                                    scale_side_name = str(scale_inputs[1])
                                elif str(add0_out_name) == scale_inputs[1]:
                                    scale_side_name = str(scale_inputs[0])
                                else:
                                    scale_side_name = ""
                                if (
                                    len(scale_side_name) > 0
                                    and _is_singleton_constant_tensor(model_ir, scale_side_name)
                                ):
                                    main_out_users = [int(v) for v in consumers.get(main_mul_out_name, [])]
                                    mean_idx: Optional[int] = None
                                    out_mul_idx: Optional[int] = None
                                    for user_idx in main_out_users:
                                        user_op = model_ir.operators[int(user_idx)]
                                        user_type = str(user_op.op_type)
                                        if (
                                            user_type == "MEAN"
                                            and len(user_op.inputs) >= 2
                                            and len(user_op.outputs) == 1
                                            and str(user_op.inputs[0]) == str(main_mul_out_name)
                                            and bool(user_op.options.get("keepDims", False))
                                        ):
                                            if mean_idx is not None:
                                                mean_idx = None
                                                break
                                            mean_idx = int(user_idx)
                                        elif (
                                            user_type == "MUL"
                                            and len(user_op.inputs) == 2
                                            and len(user_op.outputs) == 1
                                            and str(main_mul_out_name) in {str(v) for v in list(user_op.inputs)}
                                        ):
                                            if out_mul_idx is not None:
                                                out_mul_idx = None
                                                break
                                            out_mul_idx = int(user_idx)
                                    if mean_idx is not None and out_mul_idx is not None:
                                        mean_op = model_ir.operators[int(mean_idx)]
                                        mean_out_name = str(mean_op.outputs[0])
                                        if mean_out_name not in model_outputs:
                                            mean_axes_name = str(mean_op.inputs[1])
                                            mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                                            mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
                                            if mean_axes_vals is not None and len(mean_axes_vals) > 0:
                                                mean_users = [int(v) for v in consumers.get(mean_out_name, [])]
                                                if len(mean_users) == 1:
                                                    post_mean_idx = int(mean_users[0])
                                                    post_mean_op = model_ir.operators[int(post_mean_idx)]
                                                    if (
                                                        str(post_mean_op.op_type) == "TRANSPOSE"
                                                        and len(post_mean_op.inputs) >= 2
                                                        and len(post_mean_op.outputs) == 1
                                                        and str(post_mean_op.inputs[0]) == str(mean_out_name)
                                                        and _read_transpose_perm(model_ir, post_mean_op) == perm_nchw_to_nhwc
                                                    ):
                                                        post_mean_out_name = str(post_mean_op.outputs[0])
                                                        post_mean_out_users = [int(v) for v in consumers.get(post_mean_out_name, [])]
                                                        if len(post_mean_out_users) == 1:
                                                            conv1_idx = int(post_mean_out_users[0])
                                                            conv1_op = model_ir.operators[int(conv1_idx)]
                                                            if (
                                                                str(conv1_op.op_type) in {"CONV_2D", "DEPTHWISE_CONV_2D"}
                                                                and len(conv1_op.inputs) >= 1
                                                                and len(conv1_op.outputs) == 1
                                                                and str(conv1_op.inputs[0]) == str(post_mean_out_name)
                                                            ):
                                                                conv1_out_name = str(conv1_op.outputs[0])
                                                                conv1_out_users = [int(v) for v in consumers.get(conv1_out_name, [])]
                                                                if len(conv1_out_users) == 1:
                                                                    conv2_idx = int(conv1_out_users[0])
                                                                    conv2_op = model_ir.operators[int(conv2_idx)]
                                                                    if (
                                                                        str(conv2_op.op_type) in {"CONV_2D", "DEPTHWISE_CONV_2D"}
                                                                        and len(conv2_op.inputs) >= 1
                                                                        and len(conv2_op.outputs) == 1
                                                                        and str(conv2_op.inputs[0]) == str(conv1_out_name)
                                                                    ):
                                                                        conv2_out_name = str(conv2_op.outputs[0])
                                                                        conv2_out_users = [int(v) for v in consumers.get(conv2_out_name, [])]
                                                                        if len(conv2_out_users) == 1:
                                                                            pre_gate_idx = int(conv2_out_users[0])
                                                                            pre_gate_op = model_ir.operators[int(pre_gate_idx)]
                                                                            if (
                                                                                str(pre_gate_op.op_type) == "TRANSPOSE"
                                                                                and len(pre_gate_op.inputs) >= 2
                                                                                and len(pre_gate_op.outputs) == 1
                                                                                and str(pre_gate_op.inputs[0]) == str(conv2_out_name)
                                                                                and _read_transpose_perm(model_ir, pre_gate_op) == perm_nhwc_to_nchw
                                                                            ):
                                                                                pre_gate_out_name = str(pre_gate_op.outputs[0])
                                                                                pre_gate_users = [int(v) for v in consumers.get(pre_gate_out_name, [])]
                                                                                if len(pre_gate_users) == 1:
                                                                                    hs_mul_idx = int(pre_gate_users[0])
                                                                                    hs_mul_op = model_ir.operators[int(hs_mul_idx)]
                                                                                    if (
                                                                                        str(hs_mul_op.op_type) == "MUL"
                                                                                        and len(hs_mul_op.inputs) == 2
                                                                                        and len(hs_mul_op.outputs) == 1
                                                                                    ):
                                                                                        hs_mul_inputs = [str(v) for v in list(hs_mul_op.inputs)]
                                                                                        if pre_gate_out_name == hs_mul_inputs[0]:
                                                                                            hs_mul_side_name = str(hs_mul_inputs[1])
                                                                                        elif pre_gate_out_name == hs_mul_inputs[1]:
                                                                                            hs_mul_side_name = str(hs_mul_inputs[0])
                                                                                        else:
                                                                                            hs_mul_side_name = ""
                                                                                        hs_mul_out_name = str(hs_mul_op.outputs[0])
                                                                                        if (
                                                                                            len(hs_mul_side_name) > 0
                                                                                            and _is_singleton_constant_tensor(model_ir, hs_mul_side_name)
                                                                                        ):
                                                                                            hs_mul_users = [int(v) for v in consumers.get(hs_mul_out_name, [])]
                                                                                            if len(hs_mul_users) == 1:
                                                                                                hs_add_idx = int(hs_mul_users[0])
                                                                                                hs_add_op = model_ir.operators[int(hs_add_idx)]
                                                                                                if (
                                                                                                    str(hs_add_op.op_type) == "ADD"
                                                                                                    and len(hs_add_op.inputs) == 2
                                                                                                    and len(hs_add_op.outputs) == 1
                                                                                                ):
                                                                                                    hs_add_inputs = [str(v) for v in list(hs_add_op.inputs)]
                                                                                                    if hs_mul_out_name == hs_add_inputs[0]:
                                                                                                        hs_add_side_name = str(hs_add_inputs[1])
                                                                                                    elif hs_mul_out_name == hs_add_inputs[1]:
                                                                                                        hs_add_side_name = str(hs_add_inputs[0])
                                                                                                    else:
                                                                                                        hs_add_side_name = ""
                                                                                                    hs_add_out_name = str(hs_add_op.outputs[0])
                                                                                                    if (
                                                                                                        len(hs_add_side_name) > 0
                                                                                                        and _is_singleton_constant_tensor(model_ir, hs_add_side_name)
                                                                                                    ):
                                                                                                        hs_add_users = [int(v) for v in consumers.get(hs_add_out_name, [])]
                                                                                                        if len(hs_add_users) == 1:
                                                                                                            hs_out_name: Optional[str] = None
                                                                                                            hs_gate_op = model_ir.operators[int(hs_add_users[0])]
                                                                                                            if (
                                                                                                                str(hs_gate_op.op_type) == "RELU_0_TO_1"
                                                                                                                and len(hs_gate_op.inputs) == 1
                                                                                                                and len(hs_gate_op.outputs) == 1
                                                                                                                and str(hs_gate_op.inputs[0]) == str(hs_add_out_name)
                                                                                                            ):
                                                                                                                hs_out_name = str(hs_gate_op.outputs[0])
                                                                                                            elif (
                                                                                                                str(hs_gate_op.op_type) == "MAXIMUM"
                                                                                                                and len(hs_gate_op.inputs) == 2
                                                                                                                and len(hs_gate_op.outputs) == 1
                                                                                                            ):
                                                                                                                hs_max_inputs = [str(v) for v in list(hs_gate_op.inputs)]
                                                                                                                if hs_add_out_name == hs_max_inputs[0]:
                                                                                                                    hs_max_side_name = str(hs_max_inputs[1])
                                                                                                                elif hs_add_out_name == hs_max_inputs[1]:
                                                                                                                    hs_max_side_name = str(hs_max_inputs[0])
                                                                                                                else:
                                                                                                                    hs_max_side_name = ""
                                                                                                                hs_max_out_name = str(hs_gate_op.outputs[0])
                                                                                                                if (
                                                                                                                    len(hs_max_side_name) > 0
                                                                                                                    and _is_singleton_constant_tensor(model_ir, hs_max_side_name)
                                                                                                                ):
                                                                                                                    hs_max_users = [int(v) for v in consumers.get(hs_max_out_name, [])]
                                                                                                                    if len(hs_max_users) == 1:
                                                                                                                        hs_min_op = model_ir.operators[int(hs_max_users[0])]
                                                                                                                        if (
                                                                                                                            str(hs_min_op.op_type) == "MINIMUM"
                                                                                                                            and len(hs_min_op.inputs) == 2
                                                                                                                            and len(hs_min_op.outputs) == 1
                                                                                                                        ):
                                                                                                                            hs_min_inputs = [str(v) for v in list(hs_min_op.inputs)]
                                                                                                                            if hs_max_out_name == hs_min_inputs[0]:
                                                                                                                                hs_min_side_name = str(hs_min_inputs[1])
                                                                                                                            elif hs_max_out_name == hs_min_inputs[1]:
                                                                                                                                hs_min_side_name = str(hs_min_inputs[0])
                                                                                                                            else:
                                                                                                                                hs_min_side_name = ""
                                                                                                                            if (
                                                                                                                                len(hs_min_side_name) > 0
                                                                                                                                and _is_singleton_constant_tensor(model_ir, hs_min_side_name)
                                                                                                                            ):
                                                                                                                                hs_out_name = str(hs_min_op.outputs[0])
                                                                                                            if hs_out_name is not None:
                                                                                                                out_mul_op = model_ir.operators[int(out_mul_idx)]
                                                                                                                out_mul_inputs = [str(v) for v in list(out_mul_op.inputs)]
                                                                                                                if (
                                                                                                                    str(main_mul_out_name) in out_mul_inputs
                                                                                                                    and str(hs_out_name) in out_mul_inputs
                                                                                                                ):
                                                                                                                    out_mul_out_name = str(out_mul_op.outputs[0])
                                                                                                                    if out_mul_out_name not in model_outputs:
                                                                                                                        out_mul_users = [int(v) for v in consumers.get(out_mul_out_name, [])]
                                                                                                                        post_indices: List[int] = []
                                                                                                                        post_output_names: List[str] = []
                                                                                                                        valid_posts = True
                                                                                                                        for user_idx in out_mul_users:
                                                                                                                            user_op = model_ir.operators[int(user_idx)]
                                                                                                                            if (
                                                                                                                                str(user_op.op_type) != "TRANSPOSE"
                                                                                                                                or len(user_op.inputs) < 2
                                                                                                                                or len(user_op.outputs) != 1
                                                                                                                                or str(user_op.inputs[0]) != str(out_mul_out_name)
                                                                                                                                or _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc
                                                                                                                                or str(user_op.outputs[0]) in model_outputs
                                                                                                                            ):
                                                                                                                                valid_posts = False
                                                                                                                                break
                                                                                                                            post_indices.append(int(user_idx))
                                                                                                                            post_output_names.append(str(user_op.outputs[0]))
                                                                                                                        if valid_posts and len(post_indices) > 0:
                                                                                                                            rank = 4
                                                                                                                            x_nhwc_tensor = model_ir.tensors.get(x_nhwc_name, None)
                                                                                                                            if (
                                                                                                                                x_nhwc_tensor is not None
                                                                                                                                and x_nhwc_tensor.shape is not None
                                                                                                                                and len(x_nhwc_tensor.shape) > 0
                                                                                                                            ):
                                                                                                                                rank = int(len(list(x_nhwc_tensor.shape)))
                                                                                                                            if rank == 4:
                                                                                                                                normalized_axes: List[int] = []
                                                                                                                                valid_axes = True
                                                                                                                                for axis in mean_axes_vals:
                                                                                                                                    a = int(axis)
                                                                                                                                    if a < 0:
                                                                                                                                        a += int(rank)
                                                                                                                                    if a < 0 or a >= int(rank):
                                                                                                                                        valid_axes = False
                                                                                                                                        break
                                                                                                                                    normalized_axes.append(int(a))
                                                                                                                                if valid_axes and len(normalized_axes) > 0:
                                                                                                                                    mapped_axes = [
                                                                                                                                        int(perm_nhwc_to_nchw[int(axis)])
                                                                                                                                        for axis in normalized_axes
                                                                                                                                    ]
                                                                                                                                    if (
                                                                                                                                        _write_const_ints_to_tensor(
                                                                                                                                            mean_axes_tensor,
                                                                                                                                            [int(v) for v in mapped_axes],
                                                                                                                                        )
                                                                                                                                        or _read_const_ints_from_tensor(mean_axes_tensor)
                                                                                                                                        == [int(v) for v in mapped_axes]
                                                                                                                                    ):
                                                                                                                                        add0_inputs = [str(v) for v in list(add0_op.inputs)]
                                                                                                                                        add0_inputs = [
                                                                                                                                            str(x_nhwc_name) if str(v) == str(x_nchw_name) else str(v)
                                                                                                                                            for v in add0_inputs
                                                                                                                                        ]
                                                                                                                                        _set_operator_inputs(
                                                                                                                                            model_ir=model_ir,
                                                                                                                                            op=add0_op,
                                                                                                                                            new_inputs=add0_inputs,
                                                                                                                                        )

                                                                                                                                        main_mul_inputs = [str(v) for v in list(main_mul_op.inputs)]
                                                                                                                                        main_mul_inputs = [
                                                                                                                                            str(x_nhwc_name) if str(v) == str(x_nchw_name) else str(v)
                                                                                                                                            for v in main_mul_inputs
                                                                                                                                        ]
                                                                                                                                        _set_operator_inputs(
                                                                                                                                            model_ir=model_ir,
                                                                                                                                            op=main_mul_op,
                                                                                                                                            new_inputs=main_mul_inputs,
                                                                                                                                        )

                                                                                                                                        conv1_inputs = [str(v) for v in list(conv1_op.inputs)]
                                                                                                                                        conv1_inputs = [
                                                                                                                                            str(mean_out_name) if str(v) == str(post_mean_out_name) else str(v)
                                                                                                                                            for v in conv1_inputs
                                                                                                                                        ]
                                                                                                                                        _set_operator_inputs(
                                                                                                                                            model_ir=model_ir,
                                                                                                                                            op=conv1_op,
                                                                                                                                            new_inputs=conv1_inputs,
                                                                                                                                        )

                                                                                                                                        hs_mul_inputs_updated = [str(v) for v in list(hs_mul_op.inputs)]
                                                                                                                                        hs_mul_inputs_updated = [
                                                                                                                                            str(conv2_out_name) if str(v) == str(pre_gate_out_name) else str(v)
                                                                                                                                            for v in hs_mul_inputs_updated
                                                                                                                                        ]
                                                                                                                                        _set_operator_inputs(
                                                                                                                                            model_ir=model_ir,
                                                                                                                                            op=hs_mul_op,
                                                                                                                                            new_inputs=hs_mul_inputs_updated,
                                                                                                                                        )

                                                                                                                                        for tensor_name in [
                                                                                                                                            str(add0_out_name),
                                                                                                                                            str(main_mul_gate_name),
                                                                                                                                            str(main_mul_out_name),
                                                                                                                                            str(mean_out_name),
                                                                                                                                            str(hs_mul_out_name),
                                                                                                                                            str(hs_add_out_name),
                                                                                                                                            str(hs_out_name),
                                                                                                                                            str(out_mul_out_name),
                                                                                                                                        ]:
                                                                                                                                            _permute_tensor_metadata_if_rank_matches(
                                                                                                                                                model_ir.tensors.get(tensor_name, None),
                                                                                                                                                perm_nchw_to_nhwc,
                                                                                                                                            )

                                                                                                                                        canonical_post_output_name = str(post_output_names[0])
                                                                                                                                        _set_operator_outputs(
                                                                                                                                            model_ir=model_ir,
                                                                                                                                            op=out_mul_op,
                                                                                                                                            new_outputs=[canonical_post_output_name],
                                                                                                                                        )
                                                                                                                                        for alias_name in post_output_names[1:]:
                                                                                                                                            _replace_tensor_inputs(
                                                                                                                                                model_ir,
                                                                                                                                                str(alias_name),
                                                                                                                                                canonical_post_output_name,
                                                                                                                                            )

                                                                                                                                        old_out_mul_tensor = model_ir.tensors.get(out_mul_out_name, None)
                                                                                                                                        canonical_tensor = model_ir.tensors.get(canonical_post_output_name, None)
                                                                                                                                        if old_out_mul_tensor is not None and canonical_tensor is not None:
                                                                                                                                            canonical_tensor.dtype = str(old_out_mul_tensor.dtype)
                                                                                                                                            canonical_tensor.quantization = _clone_quantization(old_out_mul_tensor.quantization)
                                                                                                                                            canonical_tensor.shape = [int(v) for v in list(old_out_mul_tensor.shape)]
                                                                                                                                            canonical_tensor.shape_signature = (
                                                                                                                                                [int(v) for v in list(old_out_mul_tensor.shape_signature)]
                                                                                                                                                if old_out_mul_tensor.shape_signature is not None
                                                                                                                                                else [int(v) for v in list(old_out_mul_tensor.shape)]
                                                                                                                                            )
                                                                                                                                            _permute_tensor_metadata_if_rank_matches(
                                                                                                                                                canonical_tensor,
                                                                                                                                                perm_nchw_to_nhwc,
                                                                                                                                            )

                                                                                                                                        remove_indices = set(int(v) for v in post_indices)
                                                                                                                                        remove_indices.add(int(post_mean_idx))
                                                                                                                                        remove_indices.add(int(pre_gate_idx))
                                                                                                                                        pre_mul_remaining_users = [
                                                                                                                                            int(v)
                                                                                                                                            for v in x_nchw_users
                                                                                                                                            if int(v) not in {int(add0_idx), int(main_mul_idx)}
                                                                                                                                        ]
                                                                                                                                        if len(pre_mul_remaining_users) == 0:
                                                                                                                                            remove_indices.add(int(pre_mul_idx))
                                                                                                                                        for remove_idx in sorted(list(remove_indices), reverse=True):
                                                                                                                                            del model_ir.operators[int(remove_idx)]

                                                                                                                                        rewritten += 1
                                                                                                                                        changed = True

            if changed:
                break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_se_fc_mul_prepost_nhwc_chains": int(rewritten)}
