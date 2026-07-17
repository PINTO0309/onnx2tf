from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_onnx_shape,
)
from onnx2tf.tflite_builder.passes.pre_add_direct_unary_layout import (
    optimize_transpose_pre_add_direct_unary_nhwc_chains as _optimize_transpose_pre_add_direct_unary_nhwc_chains_pass,
)

def optimize_transpose_pre_add_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Convert NCHW Add blocks back to NHWC when wrapped by transpose adapters.

    Target:
      a_nhwc -> (optional swish-wrapped NCHW adapters) -> a_nchw
      b_nhwc -> (optional swish-wrapped NCHW adapters) -> b_nchw
      ADD(a_nchw, b_nchw) -> y_nchw
      (optional unary RELU/RELU6/LOGISTIC/TANH) -> y2_nchw
      y_or_y2_nchw -> TRANSPOSE(0,2,3,1) -> y_nhwc

    Rewrite:
      ADD(a_nhwc, b_nhwc) -> y_nhwc
      (if unary exists, keep unary and bridge its output instead of ADD output directly)

    Notes:
    - If y_nchw has legacy consumers, keep one adapter TRANSPOSE(0,3,1,2): y_nhwc -> y_nchw.
    - Swish-wrapped NCHW inputs are recognized as:
        pre_nhwc --T(0,3,1,2)--> x_nchw --LOGISTIC--> s --MUL(x_nchw,s)--> x_swish_nchw
    """
    indexed_stats = _optimize_transpose_pre_add_direct_unary_nhwc_chains_pass(
        model_ir,
        layout_state=layout_state,
    )
    optimized = int(
        indexed_stats.get(
            "optimized_transpose_pre_add_direct_unary_nhwc_chains",
            0,
        )
    )
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    unary_passthrough_ops = {"RELU", "RELU6", "LOGISTIC", "TANH", "GELU", "HARD_SWISH", "LEAKY_RELU"}
    skip_add_activation_fuse_marker = "__skip_add_activation_fuse__"
    optimized_add_marker = "__transpose_pre_add_nhwc_optimized__"

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _rewrite_prelu_alpha_to_nhwc(
        *,
        prelu_idx: int,
        prelu_op: OperatorIR,
        target_nhwc_shape: Optional[List[int]],
        consumers: Dict[str, List[int]],
    ) -> Optional[str]:
        alpha_name = str(prelu_op.inputs[1]) if len(prelu_op.inputs) >= 2 else ""
        alpha_tensor = model_ir.tensors.get(alpha_name, None)
        if alpha_tensor is None or alpha_tensor.data is None:
            return None

        alpha_data = np.asarray(alpha_tensor.data)
        alpha_candidates: List[np.ndarray] = []
        if int(alpha_data.ndim) == int(len(perm_nchw_to_nhwc)):
            # Prefer layout-remapped alpha first when both can broadcast.
            alpha_candidates.append(
                np.transpose(alpha_data, axes=perm_nchw_to_nhwc).astype(alpha_data.dtype, copy=False)
            )
        alpha_candidates.append(np.asarray(alpha_data))
        if int(alpha_data.ndim) == 3:
            alpha_candidates.append(
                np.transpose(alpha_data, axes=[1, 2, 0]).astype(alpha_data.dtype, copy=False)
            )

        def _broadcastable_to_target(candidate: np.ndarray) -> bool:
            if target_nhwc_shape is None or not _is_fully_known_positive_shape(target_nhwc_shape):
                return True
            return (
                _broadcast_static_shapes(
                    [int(v) for v in list(target_nhwc_shape)],
                    [int(v) for v in list(candidate.shape)],
                )
                is not None
            )

        selected_alpha: Optional[np.ndarray] = None
        for candidate in alpha_candidates:
            if _broadcastable_to_target(candidate):
                selected_alpha = np.asarray(candidate)
                break
        if selected_alpha is None:
            return None

        selected_alpha_name = str(alpha_name)
        alpha_users = [int(v) for v in consumers.get(alpha_name, [])]
        alpha_needs_rewrite = (
            selected_alpha.shape != alpha_data.shape
            or not np.array_equal(selected_alpha, alpha_data)
        )
        if alpha_needs_rewrite:
            shape, signature = normalize_onnx_shape(list(selected_alpha.shape))
            if len(alpha_users) == 1 and int(alpha_users[0]) == int(prelu_idx):
                alpha_tensor.data = np.asarray(selected_alpha)
                alpha_tensor.shape = [int(v) for v in list(shape)]
                alpha_tensor.shape_signature = [int(v) for v in list(signature)]
            else:
                selected_alpha_name = _unique_tensor_name(f"{alpha_name}_nhwc")
                model_ir.tensors[selected_alpha_name] = TensorIR(
                    name=selected_alpha_name,
                    dtype=str(alpha_tensor.dtype),
                    shape=[int(v) for v in list(shape)],
                    shape_signature=[int(v) for v in list(signature)],
                    data=np.asarray(selected_alpha),
                    is_variable=False,
                    quantization=_clone_quantization(alpha_tensor.quantization),
                )
        return str(selected_alpha_name)

    def _analyze_swish_input_to_nhwc(
        *,
        input_name: str,
        consumer_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        mul_idx = producers.get(str(input_name), None)
        if mul_idx is None:
            return None
        mul_op = model_ir.operators[int(mul_idx)]
        if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
            return None
        if str(mul_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(consumer_idx)}:
            return None

        in0_name = str(mul_op.inputs[0])
        in1_name = str(mul_op.inputs[1])
        in0_prod_idx = producers.get(in0_name, None)
        in1_prod_idx = producers.get(in1_name, None)

        log_idx: Optional[int] = None
        log_out_name: Optional[str] = None
        pre_nchw_name: Optional[str] = None
        mul_data_input_index: Optional[int] = None

        if in0_prod_idx is not None:
            in0_prod_op = model_ir.operators[int(in0_prod_idx)]
            if (
                str(in0_prod_op.op_type) == "LOGISTIC"
                and len(in0_prod_op.inputs) == 1
                and len(in0_prod_op.outputs) == 1
                and str(in0_prod_op.outputs[0]) == in0_name
            ):
                log_idx = int(in0_prod_idx)
                log_out_name = in0_name
                pre_nchw_name = str(in0_prod_op.inputs[0])
                mul_data_input_index = 1

        if log_idx is None and in1_prod_idx is not None:
            in1_prod_op = model_ir.operators[int(in1_prod_idx)]
            if (
                str(in1_prod_op.op_type) == "LOGISTIC"
                and len(in1_prod_op.inputs) == 1
                and len(in1_prod_op.outputs) == 1
                and str(in1_prod_op.outputs[0]) == in1_name
            ):
                log_idx = int(in1_prod_idx)
                log_out_name = in1_name
                pre_nchw_name = str(in1_prod_op.inputs[0])
                mul_data_input_index = 0

        if log_idx is None or log_out_name is None or pre_nchw_name is None or mul_data_input_index is None:
            return None
        if str(mul_op.inputs[int(mul_data_input_index)]) != str(pre_nchw_name):
            return None

        pre_idx = producers.get(str(pre_nchw_name), None)
        if pre_idx is None:
            return None
        pre_op = model_ir.operators[int(pre_idx)]
        if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
            return None
        if str(pre_op.outputs[0]) != str(pre_nchw_name):
            return None
        if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
            return None
        if str(pre_nchw_name) in model_outputs:
            return None

        if set(int(v) for v in consumers.get(str(pre_nchw_name), [])) != {int(log_idx), int(mul_idx)}:
            return None
        if set(int(v) for v in consumers.get(str(log_out_name), [])) != {int(mul_idx)}:
            return None

        return {
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_op.inputs[0]),
            "log_idx": int(log_idx),
            "mul_idx": int(mul_idx),
            "mul_data_input_index": int(mul_data_input_index),
            "log_out_name": str(log_out_name),
            "mul_out_name": str(input_name),
        }

    def _apply_swish_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        pre_idx = int(plan["pre_idx"])
        pre_input_name = str(plan["pre_input_name"])
        log_idx = int(plan["log_idx"])
        mul_idx = int(plan["mul_idx"])
        mul_data_input_index = int(plan["mul_data_input_index"])
        log_out_name = str(plan["log_out_name"])
        mul_out_name = str(plan["mul_out_name"])

        mul_op = model_ir.operators[int(mul_idx)]
        _set_operator_inputs(
            model_ir=model_ir,
            op=model_ir.operators[int(log_idx)],
            new_inputs=[pre_input_name],
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=mul_op,
            input_index=int(mul_data_input_index),
            new_input_name=pre_input_name,
        )

        for tensor_name in [log_out_name, mul_out_name]:
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(tensor_name, None),
                perm_nchw_to_nhwc,
            )
        return [int(pre_idx)]

    def _analyze_unary_input_to_nhwc(
        *,
        input_name: str,
        consumer_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        unary_idx = producers.get(str(input_name), None)
        if unary_idx is None:
            return None
        unary_op = model_ir.operators[int(unary_idx)]
        if str(unary_op.op_type) not in unary_passthrough_ops:
            return None
        if len(unary_op.inputs) != 1 or len(unary_op.outputs) != 1:
            return None
        if str(unary_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(consumer_idx)}:
            return None

        pre_nchw_name = str(unary_op.inputs[0])
        pre_idx = producers.get(pre_nchw_name, None)
        if pre_idx is None:
            return None
        pre_op = model_ir.operators[int(pre_idx)]
        if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
            return None
        if str(pre_op.outputs[0]) != pre_nchw_name:
            return None
        if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
            return None
        if str(pre_nchw_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(pre_nchw_name), [])) != {int(unary_idx)}:
            return None

        return {
            "input_name": str(input_name),
            "unary_idx": int(unary_idx),
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_op.inputs[0]),
        }

    def _apply_unary_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        input_name = str(plan["input_name"])
        unary_idx = int(plan["unary_idx"])
        pre_idx = int(plan["pre_idx"])
        pre_input_name = str(plan["pre_input_name"])

        _set_operator_inputs(
            model_ir=model_ir,
            op=model_ir.operators[int(unary_idx)],
            new_inputs=[pre_input_name],
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(input_name, None),
            perm_nchw_to_nhwc,
        )
        return [int(pre_idx)]

    def _analyze_mul_const_input_to_nhwc(
        *,
        input_name: str,
        consumer_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        mul_idx = producers.get(str(input_name), None)
        if mul_idx is None:
            return None
        mul_op = model_ir.operators[int(mul_idx)]
        if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
            return None
        if str(mul_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(consumer_idx)}:
            return None

        data_input_index: Optional[int] = None
        side_input_index: Optional[int] = None
        side_input_name: Optional[str] = None
        pre_idx: Optional[int] = None
        pre_input_name: Optional[str] = None
        pre_output_name: Optional[str] = None

        for candidate_data_index, candidate_side_index in [(0, 1), (1, 0)]:
            data_name = str(mul_op.inputs[int(candidate_data_index)])
            side_name = str(mul_op.inputs[int(candidate_side_index)])
            side_tensor = model_ir.tensors.get(side_name, None)
            if side_tensor is None or side_tensor.data is None:
                continue

            data_prod_idx = producers.get(data_name, None)
            if data_prod_idx is None:
                continue
            data_prod_op = model_ir.operators[int(data_prod_idx)]
            if (
                str(data_prod_op.op_type) != "TRANSPOSE"
                or len(data_prod_op.inputs) < 2
                or len(data_prod_op.outputs) != 1
                or str(data_prod_op.outputs[0]) != data_name
                or _read_transpose_perm(model_ir, data_prod_op) != perm_nhwc_to_nchw
                or str(data_name) in model_outputs
            ):
                continue
            data_users = set(int(v) for v in consumers.get(data_name, []))
            if int(mul_idx) not in data_users:
                continue

            data_tensor = model_ir.tensors.get(data_name, None)
            pre_in_name = str(data_prod_op.inputs[0])
            pre_in_tensor = model_ir.tensors.get(pre_in_name, None)
            if data_tensor is None or pre_in_tensor is None:
                continue
            if (
                not _is_fully_known_positive_shape(list(data_tensor.shape))
                or not _is_fully_known_positive_shape(list(pre_in_tensor.shape))
                or len(list(data_tensor.shape)) != 4
                or len(list(pre_in_tensor.shape)) != 4
            ):
                continue

            side_data = np.asarray(side_tensor.data)
            if int(side_data.size) != 1 and side_data.ndim not in {3, 4}:
                continue
            if int(side_data.size) != 1:
                if _broadcast_static_shapes(list(data_tensor.shape), [int(v) for v in list(side_data.shape)]) is None:
                    continue

            data_input_index = int(candidate_data_index)
            side_input_index = int(candidate_side_index)
            side_input_name = str(side_name)
            pre_idx = int(data_prod_idx)
            pre_input_name = pre_in_name
            pre_output_name = data_name
            break

        if (
            data_input_index is None
            or side_input_index is None
            or side_input_name is None
            or pre_idx is None
            or pre_input_name is None
            or pre_output_name is None
        ):
            return None

        side_tensor = model_ir.tensors.get(side_input_name, None)
        if side_tensor is None or side_tensor.data is None:
            return None
        side_data = np.asarray(side_tensor.data)
        nhwc_side_data: Optional[np.ndarray] = None
        side_needs_update = False
        if int(side_data.size) != 1:
            target_nhwc_shape = [int(v) for v in list(model_ir.tensors[str(pre_input_name)].shape)]
            side_shape = [int(v) for v in list(side_data.shape)]
            # Prefer semantic channel remap for NCHW channel-wise constants even
            # when static shapes are broadcast-ambiguous (e.g. [1,64,64,64]).
            is_nchw_channelwise = (
                int(side_data.ndim) == 4
                and len(side_shape) == 4
                and int(side_shape[0]) == 1
                and int(side_shape[1]) > 0
                and int(side_shape[2]) == 1
                and int(side_shape[3]) == 1
            )
            if is_nchw_channelwise:
                rotated = np.transpose(side_data, axes=perm_nchw_to_nhwc).astype(side_data.dtype, copy=False)
                rotated_shape = [int(v) for v in list(rotated.shape)]
                if _broadcast_static_shapes(target_nhwc_shape, rotated_shape) is None:
                    return None
                nhwc_side_data = np.asarray(rotated)
                side_needs_update = not np.array_equal(nhwc_side_data, side_data)
            elif _broadcast_static_shapes(target_nhwc_shape, side_shape) is not None:
                nhwc_side_data = np.asarray(side_data)
                side_needs_update = False
            else:
                rotated = np.asarray(side_data)
                found = False
                transpose_perm = (
                    perm_nchw_to_nhwc
                    if int(rotated.ndim) == 4
                    else [1, 2, 0]
                )
                max_rotate = 3 if int(rotated.ndim) == 3 else 1
                for _ in range(int(max_rotate)):
                    rotated = np.transpose(rotated, transpose_perm).astype(side_data.dtype, copy=False)
                    rotated_shape = [int(v) for v in list(rotated.shape)]
                    if _broadcast_static_shapes(target_nhwc_shape, rotated_shape) is not None:
                        nhwc_side_data = np.asarray(rotated)
                        side_needs_update = True
                        found = True
                        break
                if not found:
                    return None

        shared_outside_mul = any(
            int(v) != int(mul_idx)
            for v in consumers.get(str(side_input_name), [])
        )

        return {
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_input_name),
            "mul_idx": int(mul_idx),
            "mul_data_input_index": int(data_input_index),
            "mul_side_input_index": int(side_input_index),
            "mul_side_input_name": str(side_input_name),
            "mul_out_name": str(input_name),
            "pre_removable": bool(
                set(int(v) for v in consumers.get(str(pre_output_name), [])) == {int(mul_idx)}
            ),
            "side_needs_update": bool(side_needs_update),
            "side_shared_outside_mul": bool(shared_outside_mul),
            "side_nhwc_data": (None if nhwc_side_data is None else np.asarray(nhwc_side_data)),
        }

    def _apply_mul_const_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        pre_idx = int(plan["pre_idx"])
        pre_input_name = str(plan["pre_input_name"])
        mul_idx = int(plan["mul_idx"])
        mul_data_input_index = int(plan["mul_data_input_index"])
        mul_side_input_index = int(plan["mul_side_input_index"])
        mul_side_input_name = str(plan["mul_side_input_name"])
        mul_out_name = str(plan["mul_out_name"])

        side_input_name_for_mul = str(mul_side_input_name)
        if bool(plan.get("side_needs_update", False)):
            nhwc_data = np.asarray(plan.get("side_nhwc_data"))
            side_tensor = model_ir.tensors.get(str(mul_side_input_name), None)
            if side_tensor is None:
                return []
            if bool(plan.get("side_shared_outside_mul", False)):
                side_input_name_for_mul = _unique_tensor_name(f"{mul_side_input_name}_nhwc")
                model_ir.tensors[side_input_name_for_mul] = TensorIR(
                    name=side_input_name_for_mul,
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

        mul_op = model_ir.operators[int(mul_idx)]
        mul_inputs = [str(v) for v in list(mul_op.inputs)]
        if int(mul_data_input_index) < 0 or int(mul_data_input_index) >= len(mul_inputs):
            return []
        if int(mul_side_input_index) < 0 or int(mul_side_input_index) >= len(mul_inputs):
            return []
        mul_inputs[int(mul_data_input_index)] = str(pre_input_name)
        mul_inputs[int(mul_side_input_index)] = str(side_input_name_for_mul)
        _set_operator_inputs(
            model_ir=model_ir,
            op=mul_op,
            new_inputs=mul_inputs,
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(mul_out_name, None),
            perm_nchw_to_nhwc,
        )
        return [int(pre_idx)] if bool(plan.get("pre_removable", False)) else []

    def _analyze_gather_input_to_nhwc(
        *,
        input_name: str,
        consumer_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        gather_idx = producers.get(str(input_name), None)
        if gather_idx is None:
            return None
        gather_op = model_ir.operators[int(gather_idx)]
        if str(gather_op.op_type) != "GATHER" or len(gather_op.inputs) < 2 or len(gather_op.outputs) != 1:
            return None
        if str(gather_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(consumer_idx)}:
            return None

        gather_options = dict(gather_op.options) if isinstance(gather_op.options, dict) else {}
        if int(gather_options.get("batchDims", 0)) != 0:
            return None

        gather_data_name = str(gather_op.inputs[0])
        pre_idx = producers.get(gather_data_name, None)
        if pre_idx is None:
            return None
        pre_op = model_ir.operators[int(pre_idx)]
        if (
            str(pre_op.op_type) != "TRANSPOSE"
            or len(pre_op.inputs) < 2
            or len(pre_op.outputs) != 1
            or str(pre_op.outputs[0]) != gather_data_name
            or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            or str(gather_data_name) in model_outputs
        ):
            return None

        gather_data_tensor = model_ir.tensors.get(gather_data_name, None)
        if gather_data_tensor is None or len(list(gather_data_tensor.shape)) != 4:
            return None
        axis = int(gather_options.get("axis", 0))
        if axis < 0:
            axis += 4
        if axis < 0 or axis >= 4:
            return None
        remapped_axis = int(perm_nhwc_to_nchw[axis])

        pre_users = set(int(v) for v in consumers.get(gather_data_name, []))
        return {
            "gather_idx": int(gather_idx),
            "gather_in_name": str(gather_data_name),
            "gather_out_name": str(input_name),
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_op.inputs[0]),
            "remapped_axis": int(remapped_axis),
            "pre_removable": bool(pre_users == {int(gather_idx)}),
        }

    def _apply_gather_input_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        gather_idx = int(plan["gather_idx"])
        gather_op = model_ir.operators[int(gather_idx)]
        gather_inputs = [str(v) for v in list(gather_op.inputs)]
        if len(gather_inputs) < 2:
            return []
        gather_inputs[0] = str(plan["pre_input_name"])
        _set_operator_inputs(
            model_ir=model_ir,
            op=gather_op,
            new_inputs=gather_inputs,
        )
        gather_options = dict(gather_op.options) if isinstance(gather_op.options, dict) else {}
        gather_options["axis"] = int(plan["remapped_axis"])
        gather_options["batchDims"] = 0
        gather_op.options = gather_options
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(str(plan["gather_out_name"]), None),
            perm_nchw_to_nhwc,
        )
        return [int(plan["pre_idx"])] if bool(plan.get("pre_removable", False)) else []

    def _analyze_mul_sub_const_input_to_nhwc(
        *,
        input_name: str,
        consumer_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        mul_idx = producers.get(str(input_name), None)
        if mul_idx is None:
            return None
        mul_op = model_ir.operators[int(mul_idx)]
        if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
            return None
        if str(mul_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(consumer_idx)}:
            return None

        mul_data_input_index: Optional[int] = None
        mul_side_input_index: Optional[int] = None
        mul_side_input_name: Optional[str] = None
        sub_idx: Optional[int] = None
        sub_out_name: Optional[str] = None
        sub_input_rewrites: List[Dict[str, Any]] = []
        removable_pre_indices: List[int] = []
        target_nhwc_shape: Optional[List[int]] = None

        for candidate_data_index, candidate_side_index in [(0, 1), (1, 0)]:
            data_name = str(mul_op.inputs[int(candidate_data_index)])
            side_name = str(mul_op.inputs[int(candidate_side_index)])
            side_tensor = model_ir.tensors.get(side_name, None)
            if side_tensor is None or side_tensor.data is None:
                continue

            candidate_sub_idx = producers.get(data_name, None)
            if candidate_sub_idx is None:
                continue
            candidate_sub_op = model_ir.operators[int(candidate_sub_idx)]
            if (
                str(candidate_sub_op.op_type) != "SUB"
                or len(candidate_sub_op.inputs) != 2
                or len(candidate_sub_op.outputs) != 1
                or str(candidate_sub_op.outputs[0]) != data_name
                or str(data_name) in model_outputs
            ):
                continue
            if set(int(v) for v in consumers.get(data_name, [])) != {int(mul_idx)}:
                continue

            candidate_rewrites: List[Dict[str, Any]] = []
            candidate_removable: List[int] = []
            candidate_target_shape: Optional[List[int]] = None
            valid_sub_inputs = True
            for sub_input_index, sub_input_name in enumerate([str(v) for v in list(candidate_sub_op.inputs)]):
                sub_pre_idx = producers.get(str(sub_input_name), None)
                if sub_pre_idx is None:
                    valid_sub_inputs = False
                    break
                sub_pre_op = model_ir.operators[int(sub_pre_idx)]
                if (
                    str(sub_pre_op.op_type) != "TRANSPOSE"
                    or len(sub_pre_op.inputs) < 2
                    or len(sub_pre_op.outputs) != 1
                    or str(sub_pre_op.outputs[0]) != str(sub_input_name)
                    or _read_transpose_perm(model_ir, sub_pre_op) != perm_nhwc_to_nchw
                    or str(sub_input_name) in model_outputs
                ):
                    valid_sub_inputs = False
                    break
                sub_input_users = set(int(v) for v in consumers.get(str(sub_input_name), []))
                if not sub_input_users.issubset({int(candidate_sub_idx), int(consumer_idx)}):
                    valid_sub_inputs = False
                    break
                nhwc_name = str(sub_pre_op.inputs[0])
                candidate_rewrites.append(
                    {
                        "sub_input_index": int(sub_input_index),
                        "nhwc_name": nhwc_name,
                        "pre_idx": int(sub_pre_idx),
                    }
                )
                candidate_removable.append(int(sub_pre_idx))
                nhwc_tensor = model_ir.tensors.get(nhwc_name, None)
                if (
                    candidate_target_shape is None
                    and nhwc_tensor is not None
                    and _is_fully_known_positive_shape(list(nhwc_tensor.shape))
                ):
                    candidate_target_shape = [int(v) for v in list(nhwc_tensor.shape)]

            if not valid_sub_inputs or len(candidate_rewrites) != 2:
                continue

            mul_data_input_index = int(candidate_data_index)
            mul_side_input_index = int(candidate_side_index)
            mul_side_input_name = str(side_name)
            sub_idx = int(candidate_sub_idx)
            sub_out_name = str(data_name)
            sub_input_rewrites = [dict(v) for v in candidate_rewrites]
            removable_pre_indices = [int(v) for v in candidate_removable]
            target_nhwc_shape = (
                [int(v) for v in list(candidate_target_shape)]
                if candidate_target_shape is not None
                else None
            )
            break

        if (
            mul_data_input_index is None
            or mul_side_input_index is None
            or mul_side_input_name is None
            or sub_idx is None
            or sub_out_name is None
            or len(sub_input_rewrites) != 2
        ):
            return None

        side_tensor = model_ir.tensors.get(str(mul_side_input_name), None)
        if side_tensor is None or side_tensor.data is None:
            return None

        side_data = np.asarray(side_tensor.data)
        if int(side_data.size) != 1 and side_data.ndim not in {1, 3, 4}:
            return None

        nhwc_side_data: Optional[np.ndarray] = None
        side_needs_update = False
        if int(side_data.size) != 1 and side_data.ndim in {3, 4}:
            side_shape = [int(v) for v in list(side_data.shape)]
            if (
                target_nhwc_shape is not None
                and _broadcast_static_shapes(target_nhwc_shape, side_shape) is not None
            ):
                nhwc_side_data = np.asarray(side_data)
            else:
                rotated = np.asarray(side_data)
                transpose_perm = perm_nchw_to_nhwc if int(rotated.ndim) == 4 else [1, 2, 0]
                max_rotate = 1 if int(rotated.ndim) == 4 else 3
                found = False
                for _ in range(int(max_rotate)):
                    rotated = np.transpose(rotated, transpose_perm).astype(side_data.dtype, copy=False)
                    rotated_shape = [int(v) for v in list(rotated.shape)]
                    if (
                        target_nhwc_shape is None
                        or _broadcast_static_shapes(target_nhwc_shape, rotated_shape) is not None
                    ):
                        nhwc_side_data = np.asarray(rotated)
                        side_needs_update = True
                        found = True
                        break
                if not found:
                    return None

        side_shared_outside_mul = any(
            int(v) != int(mul_idx)
            for v in consumers.get(str(mul_side_input_name), [])
        )

        return {
            "sub_idx": int(sub_idx),
            "sub_input_rewrites": [dict(v) for v in sub_input_rewrites],
            "sub_out_name": str(sub_out_name),
            "mul_idx": int(mul_idx),
            "mul_data_input_index": int(mul_data_input_index),
            "mul_side_input_index": int(mul_side_input_index),
            "mul_side_input_name": str(mul_side_input_name),
            "mul_out_name": str(input_name),
            "removable_pre_indices": sorted(list({int(v) for v in removable_pre_indices})),
            "side_needs_update": bool(side_needs_update),
            "side_shared_outside_mul": bool(side_shared_outside_mul),
            "side_nhwc_data": (None if nhwc_side_data is None else np.asarray(nhwc_side_data)),
        }

    def _apply_mul_sub_const_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        sub_idx = int(plan["sub_idx"])
        sub_input_rewrites = [dict(v) for v in list(plan["sub_input_rewrites"])]
        sub_out_name = str(plan["sub_out_name"])
        mul_idx = int(plan["mul_idx"])
        mul_data_input_index = int(plan["mul_data_input_index"])
        mul_side_input_index = int(plan["mul_side_input_index"])
        mul_side_input_name = str(plan["mul_side_input_name"])
        mul_out_name = str(plan["mul_out_name"])
        removable_pre_indices = [int(v) for v in list(plan.get("removable_pre_indices", []))]

        side_input_name_for_mul = str(mul_side_input_name)
        if bool(plan.get("side_needs_update", False)):
            nhwc_data = np.asarray(plan.get("side_nhwc_data"))
            side_tensor = model_ir.tensors.get(str(mul_side_input_name), None)
            if side_tensor is None:
                return []
            if bool(plan.get("side_shared_outside_mul", False)):
                side_input_name_for_mul = _unique_tensor_name(f"{mul_side_input_name}_nhwc")
                model_ir.tensors[side_input_name_for_mul] = TensorIR(
                    name=side_input_name_for_mul,
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

        sub_op = model_ir.operators[int(sub_idx)]
        sub_inputs = [str(v) for v in list(sub_op.inputs)]
        for rewrite in sub_input_rewrites:
            input_index = int(rewrite["sub_input_index"])
            if int(input_index) < 0 or int(input_index) >= len(sub_inputs):
                return []
            sub_inputs[int(input_index)] = str(rewrite["nhwc_name"])
        _set_operator_inputs(
            model_ir=model_ir,
            op=sub_op,
            new_inputs=sub_inputs,
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(sub_out_name, None),
            perm_nchw_to_nhwc,
        )

        mul_op = model_ir.operators[int(mul_idx)]
        mul_inputs = [str(v) for v in list(mul_op.inputs)]
        if int(mul_data_input_index) < 0 or int(mul_data_input_index) >= len(mul_inputs):
            return []
        if int(mul_side_input_index) < 0 or int(mul_side_input_index) >= len(mul_inputs):
            return []
        mul_inputs[int(mul_data_input_index)] = str(sub_out_name)
        mul_inputs[int(mul_side_input_index)] = str(side_input_name_for_mul)
        _set_operator_inputs(
            model_ir=model_ir,
            op=mul_op,
            new_inputs=mul_inputs,
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(mul_out_name, None),
            perm_nchw_to_nhwc,
        )

        return [int(v) for v in list(removable_pre_indices)]

    def _analyze_const_add_input_to_nhwc(
        *,
        input_name: str,
        add_idx: int,
        consumers: Dict[str, List[int]],
    ) -> Optional[Dict[str, Any]]:
        const_tensor = model_ir.tensors.get(str(input_name), None)
        if const_tensor is None or const_tensor.data is None:
            return None

        add_op = model_ir.operators[int(add_idx)]
        if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
            return None

        const_data = np.asarray(const_tensor.data)
        const_shape = [int(v) for v in list(const_data.shape)]
        target_nhwc_shape: Optional[List[int]] = None
        add_out_tensor = model_ir.tensors.get(str(add_op.outputs[0]), None)
        if (
            add_out_tensor is not None
            and _is_fully_known_positive_shape(list(add_out_tensor.shape))
            and len(list(add_out_tensor.shape)) == 4
        ):
            target_nhwc_shape = _permute_shape(list(add_out_tensor.shape), perm_nchw_to_nhwc)

        nhwc_data: Optional[np.ndarray] = None
        const_needs_update = False
        if int(const_data.size) == 1:
            pass
        elif const_data.ndim == 4:
            is_nchw_channelwise = (
                len(const_shape) == 4
                and int(const_shape[0]) == 1
                and int(const_shape[1]) > 0
                and int(const_shape[2]) == 1
                and int(const_shape[3]) == 1
            )
            if is_nchw_channelwise:
                rotated = np.transpose(const_data, perm_nchw_to_nhwc).astype(const_data.dtype, copy=False)
                rotated_shape = [int(v) for v in list(rotated.shape)]
                if target_nhwc_shape is not None and _broadcast_static_shapes(target_nhwc_shape, rotated_shape) is None:
                    return None
                nhwc_data = np.asarray(rotated)
                const_needs_update = not np.array_equal(nhwc_data, const_data)
            elif target_nhwc_shape is not None and _broadcast_static_shapes(target_nhwc_shape, const_shape) is not None:
                pass
            else:
                rotated: Optional[np.ndarray] = None
                candidate = np.asarray(const_data)
                for _ in range(3):
                    candidate = np.transpose(candidate, perm_nchw_to_nhwc).astype(const_data.dtype, copy=False)
                    candidate_shape = [int(v) for v in list(candidate.shape)]
                    if target_nhwc_shape is None or _broadcast_static_shapes(target_nhwc_shape, candidate_shape) is not None:
                        rotated = np.asarray(candidate)
                        break
                if rotated is None:
                    return None
                nhwc_data = np.asarray(rotated)
                const_needs_update = True
        elif const_data.ndim == 3:
            if target_nhwc_shape is not None and _broadcast_static_shapes(target_nhwc_shape, const_shape) is not None:
                pass
            else:
                candidate = np.asarray(const_data)
                rotated: Optional[np.ndarray] = None
                for _ in range(3):
                    candidate = np.transpose(candidate, [1, 2, 0]).astype(const_data.dtype, copy=False)
                    candidate_shape = [int(v) for v in list(candidate.shape)]
                    if target_nhwc_shape is None or _broadcast_static_shapes(target_nhwc_shape, candidate_shape) is not None:
                        rotated = np.asarray(candidate)
                        break
                if rotated is None:
                    return None
                nhwc_data = np.asarray(rotated)
                const_needs_update = True
        else:
            if target_nhwc_shape is not None and _broadcast_static_shapes(target_nhwc_shape, const_shape) is None:
                return None

        shared_outside_add = any(
            int(v) != int(add_idx)
            for v in consumers.get(str(input_name), [])
        )
        return {
            "const_input_name": str(input_name),
            "const_needs_update": bool(const_needs_update),
            "const_shared_outside_add": bool(shared_outside_add),
            "const_nhwc_data": (None if nhwc_data is None else np.asarray(nhwc_data)),
        }

    def _apply_const_add_input_nhwc_plan(*, plan: Dict[str, Any]) -> str:
        const_input_name = str(plan["const_input_name"])
        const_input_name_for_add = str(const_input_name)
        if not bool(plan.get("const_needs_update", False)):
            return const_input_name_for_add

        nhwc_data = np.asarray(plan.get("const_nhwc_data"))
        const_tensor = model_ir.tensors.get(const_input_name, None)
        if const_tensor is None:
            return const_input_name_for_add
        if bool(plan.get("const_shared_outside_add", False)):
            const_input_name_for_add = _unique_tensor_name(f"{const_input_name}_nhwc")
            model_ir.tensors[const_input_name_for_add] = TensorIR(
                name=const_input_name_for_add,
                dtype=str(const_tensor.dtype),
                shape=[int(v) for v in list(nhwc_data.shape)],
                shape_signature=[int(v) for v in list(nhwc_data.shape)],
                data=np.asarray(nhwc_data),
                is_variable=False,
                quantization=_clone_quantization(const_tensor.quantization),
            )
        else:
            const_tensor.data = np.asarray(nhwc_data)
            const_tensor.shape = [int(v) for v in list(nhwc_data.shape)]
            const_tensor.shape_signature = [int(v) for v in list(nhwc_data.shape)]
        return str(const_input_name_for_add)

    def _build_direct_nchw_fallback_plan(
        *,
        input_name: str,
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        if str(input_name) in model_outputs:
            return None
        input_tensor = model_ir.tensors.get(str(input_name), None)
        if input_tensor is None:
            return None
        # Restrict fallback to dynamic rank-4 tensors. Constants are handled by
        # const-specific remap logic; scalar/rank!=4 paths are left unchanged.
        if input_tensor.data is not None:
            return None
        input_shape = [int(v) for v in list(input_tensor.shape)]
        if len(input_shape) != 4:
            return None

        bridge_name = _unique_tensor_name(f"{input_name}_nhwc_bridge")
        return {
            "nhwc_input_name": str(bridge_name),
            "pre_remove_indices": [],
            "direct_nchw_plan": {
                "input_name": str(input_name),
                "bridge_name": str(bridge_name),
            },
        }

    def _materialize_direct_nchw_bridge(
        *,
        plan: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        input_name = str(plan.get("input_name", ""))
        bridge_name = str(plan.get("bridge_name", ""))
        if input_name == "" or bridge_name == "":
            return None
        input_tensor = model_ir.tensors.get(input_name, None)
        if input_tensor is None:
            return None
        if len(list(input_tensor.shape)) != 4:
            return None

        perm_arr = np.asarray([0, 2, 3, 1], dtype=np.int32)
        # Keep per-bridge dedicated perm tensor to avoid cross-rewrite mutation.
        perm_const_name = _unique_tensor_name(f"{bridge_name}_perm")
        model_ir.tensors[perm_const_name] = TensorIR(
            name=perm_const_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=perm_arr,
            is_variable=False,
            quantization=None,
        )

        bridge_shape = _permute_shape(
            [int(v) for v in list(input_tensor.shape)],
            [0, 2, 3, 1],
        )
        bridge_signature_src = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else [int(v) for v in list(input_tensor.shape)]
        )
        bridge_signature = _permute_shape(bridge_signature_src, [0, 2, 3, 1])
        if bridge_shape is None or bridge_signature is None:
            return None

        if bridge_name not in model_ir.tensors:
            model_ir.tensors[bridge_name] = TensorIR(
                name=bridge_name,
                dtype=str(input_tensor.dtype),
                shape=[int(v) for v in list(bridge_shape)],
                shape_signature=[int(v) for v in list(bridge_signature)],
                data=None,
                is_variable=False,
                quantization=_clone_quantization(input_tensor.quantization),
            )

        return {
            "input_name": str(input_name),
            "bridge_name": str(bridge_name),
            "perm_name": str(perm_const_name),
        }

    def _resolve_add_input_to_nhwc(
        *,
        input_name: str,
        add_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
        allow_nested_add: bool = True,
    ) -> Optional[Dict[str, Any]]:
        input_producer_idx = producers.get(str(input_name), None)
        if input_producer_idx is not None:
            input_producer_op = model_ir.operators[int(input_producer_idx)]
            input_users = set(int(v) for v in consumers.get(str(input_name), []))

            if (
                str(input_producer_op.op_type) == "TRANSPOSE"
                and len(input_producer_op.inputs) >= 2
                and len(input_producer_op.outputs) == 1
                and str(input_producer_op.outputs[0]) == str(input_name)
                and _read_transpose_perm(model_ir, input_producer_op) == perm_nhwc_to_nchw
                and int(add_idx) in input_users
                and str(input_name) not in model_outputs
            ):
                extra_user_indices = [int(v) for v in sorted(input_users) if int(v) != int(add_idx)]
                return {
                    "nhwc_input_name": str(input_producer_op.inputs[0]),
                    "pre_remove_indices": (
                        [int(input_producer_idx)] if len(extra_user_indices) == 0 else []
                    ),
                    "swish_plan": None,
                    "unary_plan": None,
                    "mul_const_plan": None,
                    "nested_add_plan": None,
                }

        swish_plan = _analyze_swish_input_to_nhwc(
            input_name=input_name,
            consumer_idx=int(add_idx),
            producers=producers,
            consumers=consumers,
            model_outputs=model_outputs,
        )
        if swish_plan is not None:
            return {
                "nhwc_input_name": str(input_name),
                "pre_remove_indices": [],
                "swish_plan": swish_plan,
                "unary_plan": None,
                "mul_const_plan": None,
                "mul_sub_const_plan": None,
                "gather_plan": None,
                "nested_add_plan": None,
            }

        unary_plan = _analyze_unary_input_to_nhwc(
            input_name=input_name,
            consumer_idx=int(add_idx),
            producers=producers,
            consumers=consumers,
            model_outputs=model_outputs,
        )
        if unary_plan is not None:
            return {
                "nhwc_input_name": str(input_name),
                "pre_remove_indices": [],
                "swish_plan": None,
                "unary_plan": unary_plan,
                "mul_const_plan": None,
                "mul_sub_const_plan": None,
                "gather_plan": None,
                "nested_add_plan": None,
            }

        mul_const_plan = _analyze_mul_const_input_to_nhwc(
            input_name=input_name,
            consumer_idx=int(add_idx),
            producers=producers,
            consumers=consumers,
            model_outputs=model_outputs,
        )
        if mul_const_plan is not None:
            return {
                "nhwc_input_name": str(input_name),
                "pre_remove_indices": [],
                "swish_plan": None,
                "unary_plan": None,
                "mul_const_plan": mul_const_plan,
                "mul_sub_const_plan": None,
                "gather_plan": None,
                "nested_add_plan": None,
            }

        mul_sub_const_plan = _analyze_mul_sub_const_input_to_nhwc(
            input_name=input_name,
            consumer_idx=int(add_idx),
            producers=producers,
            consumers=consumers,
            model_outputs=model_outputs,
        )
        if mul_sub_const_plan is not None:
            return {
                "nhwc_input_name": str(input_name),
                "pre_remove_indices": [],
                "swish_plan": None,
                "unary_plan": None,
                "mul_const_plan": None,
                "mul_sub_const_plan": mul_sub_const_plan,
                "gather_plan": None,
                "nested_add_plan": None,
            }

        gather_plan = _analyze_gather_input_to_nhwc(
            input_name=input_name,
            consumer_idx=int(add_idx),
            producers=producers,
            consumers=consumers,
            model_outputs=model_outputs,
        )
        if gather_plan is not None:
            return {
                "nhwc_input_name": str(input_name),
                "pre_remove_indices": [],
                "swish_plan": None,
                "unary_plan": None,
                "mul_const_plan": None,
                "mul_sub_const_plan": None,
                "gather_plan": gather_plan,
                "nested_add_plan": None,
            }

        const_add_plan = _analyze_const_add_input_to_nhwc(
            input_name=input_name,
            add_idx=int(add_idx),
            consumers=consumers,
        )
        if const_add_plan is not None:
            return {
                "nhwc_input_name": str(input_name),
                "pre_remove_indices": [],
                "swish_plan": None,
                "unary_plan": None,
                "mul_const_plan": None,
                "mul_sub_const_plan": None,
                "gather_plan": None,
                "nested_add_plan": None,
                "const_add_plan": const_add_plan,
            }

        if input_producer_idx is not None and allow_nested_add:
            input_producer_op = model_ir.operators[int(input_producer_idx)]
            if (
                str(input_producer_op.op_type) == "ADD"
                and len(input_producer_op.inputs) == 2
                and len(input_producer_op.outputs) == 1
                and str(input_producer_op.outputs[0]) == str(input_name)
                and str(input_name) not in model_outputs
            ):
                input_users = set(int(v) for v in consumers.get(str(input_name), []))
                if input_users == {int(add_idx)}:
                    nested_input_plans: List[Dict[str, Any]] = []
                    nested_rewritable = True
                    for nested_input_name in [str(v) for v in list(input_producer_op.inputs)]:
                        nested_plan = _resolve_add_input_to_nhwc(
                            input_name=str(nested_input_name),
                            add_idx=int(input_producer_idx),
                            producers=producers,
                            consumers=consumers,
                            model_outputs=model_outputs,
                            allow_nested_add=False,
                        )
                        if nested_plan is None:
                            nested_rewritable = False
                            break
                        nested_input_plans.append(dict(nested_plan))
                    if nested_rewritable:
                        return {
                            "nhwc_input_name": str(input_name),
                            "pre_remove_indices": [],
                            "swish_plan": None,
                            "unary_plan": None,
                            "mul_const_plan": None,
                            "mul_sub_const_plan": None,
                            "gather_plan": None,
                            "nested_add_plan": {
                                "input_name": str(input_name),
                                "add_idx": int(input_producer_idx),
                                "input_plans": [dict(v) for v in nested_input_plans],
                            },
                        }

        if input_producer_idx is not None:
            input_producer_op = model_ir.operators[int(input_producer_idx)]
            if (
                str(input_producer_op.op_type) == "ADD"
                and len(input_producer_op.outputs) == 1
                and str(input_producer_op.outputs[0]) == str(input_name)
            ):
                input_users = [int(v) for v in consumers.get(str(input_name), []) if int(v) != int(input_producer_idx)]
                post_candidates: List[str] = []
                for user_idx in input_users:
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) == "TRANSPOSE"
                        and len(user_op.inputs) >= 2
                        and len(user_op.outputs) == 1
                        and str(user_op.inputs[0]) == str(input_name)
                        and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                        and str(user_op.outputs[0]) not in model_outputs
                    ):
                        post_candidates.append(str(user_op.outputs[0]))
                if len(post_candidates) > 0:
                    return {
                        "nhwc_input_name": str(post_candidates[0]),
                        "pre_remove_indices": [],
                        "swish_plan": None,
                        "unary_plan": None,
                        "mul_const_plan": None,
                        "mul_sub_const_plan": None,
                        "gather_plan": None,
                        "nested_add_plan": None,
                    }

        return None

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for add_idx, add_op in enumerate(model_ir.operators):
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            if bool(add_op.options.get(optimized_add_marker, False)):
                continue
            add_out_name = str(add_op.outputs[0])

            bridge_output_name = add_out_name
            bridge_output_producer_idx = int(add_idx)
            bridge_output_producer_op = add_op
            bridge_prelu_idx: Optional[int] = None

            # Also support: ADD -> unary -> post-transpose fanout.
            add_out_users = [int(v) for v in consumers.get(add_out_name, []) if int(v) != int(add_idx)]
            if len(add_out_users) == 1:
                unary_idx = int(add_out_users[0])
                unary_op = model_ir.operators[int(unary_idx)]
                if (
                    str(unary_op.op_type) in unary_passthrough_ops
                    and len(unary_op.inputs) == 1
                    and len(unary_op.outputs) == 1
                    and str(unary_op.inputs[0]) == add_out_name
                ):
                    unary_out_name = str(unary_op.outputs[0])
                    if add_out_name not in model_outputs and unary_out_name not in model_outputs:
                        bridge_output_name = unary_out_name
                        bridge_output_producer_idx = int(unary_idx)
                        bridge_output_producer_op = unary_op
                elif (
                    str(unary_op.op_type) == "PRELU"
                    and len(unary_op.inputs) == 2
                    and len(unary_op.outputs) == 1
                    and str(unary_op.inputs[0]) == add_out_name
                ):
                    prelu_out_name = str(unary_op.outputs[0])
                    if add_out_name not in model_outputs and prelu_out_name not in model_outputs:
                        bridge_output_name = prelu_out_name
                        bridge_output_producer_idx = int(unary_idx)
                        bridge_output_producer_op = unary_op
                        bridge_prelu_idx = int(unary_idx)

            out_users = [
                int(v)
                for v in consumers.get(bridge_output_name, [])
                if int(v) != int(bridge_output_producer_idx)
            ]
            if len(out_users) == 0:
                continue

            post_indices: List[int] = []
            post_output_names: List[str] = []
            legacy_users: List[int] = []
            valid_users = True
            for user_idx in out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == bridge_output_name
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                else:
                    legacy_users.append(int(user_idx))
            if not valid_users or len(post_indices) == 0:
                continue

            input_plans: List[Dict[str, Any]] = []
            rewritable = True
            for input_name in [str(v) for v in list(add_op.inputs)]:
                resolved = _resolve_add_input_to_nhwc(
                    input_name=input_name,
                    add_idx=int(add_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if resolved is None:
                    resolved = _build_direct_nchw_fallback_plan(
                        input_name=str(input_name),
                        model_outputs=model_outputs,
                    )
                    if resolved is None:
                        rewritable = False
                        break
                input_plans.append(dict(resolved))
            if not rewritable:
                continue

            planned_direct_bridge_count = int(
                sum(1 for plan in input_plans if plan.get("direct_nchw_plan", None) is not None)
            )
            if planned_direct_bridge_count > 0:
                planned_pre_remove_count = int(
                    sum(
                        len([int(v) for v in list(plan.get("pre_remove_indices", []))])
                        for plan in input_plans
                    )
                )
                planned_post_remove_count = int(len(post_indices) - 1) if len(legacy_users) > 0 else int(len(post_indices))
                # Require strict local transpose reduction before introducing
                # direct NCHW->NHWC bridge inserts.
                if int(planned_post_remove_count + planned_pre_remove_count) <= int(planned_direct_bridge_count):
                    if int(planned_post_remove_count + planned_pre_remove_count) < int(planned_direct_bridge_count):
                        continue

            def _apply_nested_add_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
                nested_input_name = str(plan["input_name"])
                nested_add_idx = int(plan["add_idx"])
                nested_input_plans = [dict(v) for v in list(plan.get("input_plans", []))]

                nested_remove_indices: List[int] = []
                nested_new_inputs: List[str] = []
                for nested_input_plan in nested_input_plans:
                    nested_nhwc_input_name = str(nested_input_plan["nhwc_input_name"])
                    nested_swish_plan = nested_input_plan.get("swish_plan", None)
                    if nested_swish_plan is not None:
                        nested_remove_indices.extend(_apply_swish_nhwc_plan(plan=dict(nested_swish_plan)))
                    nested_unary_plan = nested_input_plan.get("unary_plan", None)
                    if nested_unary_plan is not None:
                        nested_remove_indices.extend(_apply_unary_nhwc_plan(plan=dict(nested_unary_plan)))
                    nested_mul_const_plan = nested_input_plan.get("mul_const_plan", None)
                    if nested_mul_const_plan is not None:
                        nested_remove_indices.extend(_apply_mul_const_nhwc_plan(plan=dict(nested_mul_const_plan)))
                    nested_mul_sub_const_plan = nested_input_plan.get("mul_sub_const_plan", None)
                    if nested_mul_sub_const_plan is not None:
                        nested_remove_indices.extend(
                            _apply_mul_sub_const_nhwc_plan(plan=dict(nested_mul_sub_const_plan))
                        )
                    nested_gather_plan = nested_input_plan.get("gather_plan", None)
                    if nested_gather_plan is not None:
                        nested_remove_indices.extend(
                            _apply_gather_input_nhwc_plan(plan=dict(nested_gather_plan))
                        )
                    nested_nested_add_plan = nested_input_plan.get("nested_add_plan", None)
                    if nested_nested_add_plan is not None:
                        nested_remove_indices.extend(
                            _apply_nested_add_nhwc_plan(plan=dict(nested_nested_add_plan))
                        )
                    nested_const_add_plan = nested_input_plan.get("const_add_plan", None)
                    if nested_const_add_plan is not None:
                        nested_nhwc_input_name = _apply_const_add_input_nhwc_plan(
                            plan=dict(nested_const_add_plan)
                        )
                    nested_new_inputs.append(str(nested_nhwc_input_name))
                    nested_remove_indices.extend(
                        [int(v) for v in list(nested_input_plan.get("pre_remove_indices", []))]
                    )

                nested_add_op = model_ir.operators[int(nested_add_idx)]
                if (
                    str(nested_add_op.op_type) != "ADD"
                    or len(nested_add_op.inputs) != 2
                    or len(nested_add_op.outputs) != 1
                ):
                    return nested_remove_indices
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=nested_add_op,
                    new_inputs=[str(v) for v in list(nested_new_inputs)],
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(nested_input_name, None),
                    perm_nchw_to_nhwc,
                )
                return nested_remove_indices

            new_add_inputs: List[str] = []
            pre_remove_indices: List[int] = []
            direct_bridge_specs: List[Dict[str, str]] = []
            for plan in input_plans:
                nhwc_input_name = str(plan["nhwc_input_name"])
                swish_plan = plan.get("swish_plan", None)
                if swish_plan is not None:
                    pre_remove_indices.extend(_apply_swish_nhwc_plan(plan=dict(swish_plan)))
                unary_plan = plan.get("unary_plan", None)
                if unary_plan is not None:
                    pre_remove_indices.extend(_apply_unary_nhwc_plan(plan=dict(unary_plan)))
                mul_const_plan = plan.get("mul_const_plan", None)
                if mul_const_plan is not None:
                    pre_remove_indices.extend(_apply_mul_const_nhwc_plan(plan=dict(mul_const_plan)))
                mul_sub_const_plan = plan.get("mul_sub_const_plan", None)
                if mul_sub_const_plan is not None:
                    pre_remove_indices.extend(_apply_mul_sub_const_nhwc_plan(plan=dict(mul_sub_const_plan)))
                gather_plan = plan.get("gather_plan", None)
                if gather_plan is not None:
                    pre_remove_indices.extend(_apply_gather_input_nhwc_plan(plan=dict(gather_plan)))
                nested_add_plan = plan.get("nested_add_plan", None)
                if nested_add_plan is not None:
                    pre_remove_indices.extend(_apply_nested_add_nhwc_plan(plan=dict(nested_add_plan)))
                const_add_plan = plan.get("const_add_plan", None)
                if const_add_plan is not None:
                    nhwc_input_name = _apply_const_add_input_nhwc_plan(plan=dict(const_add_plan))
                direct_nchw_plan = plan.get("direct_nchw_plan", None)
                if direct_nchw_plan is not None:
                    bridge_spec = _materialize_direct_nchw_bridge(plan=dict(direct_nchw_plan))
                    if bridge_spec is None:
                        rewritable = False
                        break
                    nhwc_input_name = str(bridge_spec["bridge_name"])
                    direct_bridge_specs.append(dict(bridge_spec))
                new_add_inputs.append(str(nhwc_input_name))
                pre_remove_indices.extend([int(v) for v in list(plan.get("pre_remove_indices", []))])
            if not rewritable:
                continue

            if bridge_prelu_idx is not None:
                target_nhwc_shape: Optional[List[int]] = None
                if len(new_add_inputs) > 0:
                    target_tensor = model_ir.tensors.get(str(new_add_inputs[0]), None)
                    if target_tensor is not None and target_tensor.shape is not None:
                        target_nhwc_shape = [int(v) for v in list(target_tensor.shape)]
                bridge_prelu_op = model_ir.operators[int(bridge_prelu_idx)]
                selected_alpha_name = _rewrite_prelu_alpha_to_nhwc(
                    prelu_idx=int(bridge_prelu_idx),
                    prelu_op=bridge_prelu_op,
                    target_nhwc_shape=target_nhwc_shape,
                    consumers=consumers,
                )
                if selected_alpha_name is None:
                    continue
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=bridge_prelu_op,
                    new_inputs=[add_out_name, str(selected_alpha_name)],
                )

            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=[str(v) for v in new_add_inputs],
            )

            canonical_post_output_name = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=bridge_output_producer_op,
                new_outputs=[canonical_post_output_name],
            )
            for alias_post_output_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, alias_post_output_name, canonical_post_output_name)

            old_bridge_tensor = model_ir.tensors.get(bridge_output_name, None)
            canonical_post_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if old_bridge_tensor is not None and canonical_post_tensor is not None:
                canonical_post_tensor.dtype = str(old_bridge_tensor.dtype)
                canonical_post_tensor.quantization = _clone_quantization(old_bridge_tensor.quantization)
                canonical_post_tensor.shape = [int(v) for v in list(old_bridge_tensor.shape)]
                canonical_post_tensor.shape_signature = (
                    [int(v) for v in list(old_bridge_tensor.shape_signature)]
                    if old_bridge_tensor.shape_signature is not None
                    else [int(v) for v in list(old_bridge_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    canonical_post_tensor,
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
                    new_outputs=[bridge_output_name],
                )
                post_remove_indices = [int(v) for v in post_indices[1:]]
            else:
                post_remove_indices = [int(v) for v in post_indices]

            remove_indices = sorted(
                list({int(v) for v in pre_remove_indices + post_remove_indices}),
                reverse=True,
            )
            for remove_idx in remove_indices:
                if int(remove_idx) == int(add_idx):
                    continue
                del model_ir.operators[int(remove_idx)]

            if len(direct_bridge_specs) > 0:
                try:
                    add_current_idx = int(model_ir.operators.index(add_op))
                except ValueError:
                    add_current_idx = int(max(0, min(add_idx, len(model_ir.operators))))
                insert_offset = 0
                for bridge_spec in direct_bridge_specs:
                    model_ir.operators.insert(
                        int(add_current_idx + insert_offset),
                        OperatorIR(
                            op_type="TRANSPOSE",
                            inputs=[str(bridge_spec["input_name"]), str(bridge_spec["perm_name"])],
                            outputs=[str(bridge_spec["bridge_name"])],
                        ),
                    )
                    insert_offset += 1

            # Keep explicit unary ops after transpose-bridge rewrites for stability.
            add_opts = dict(add_op.options) if isinstance(add_op.options, dict) else {}
            add_opts[skip_add_activation_fuse_marker] = True
            add_opts[optimized_add_marker] = True
            add_op.options = add_opts

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_pre_add_nhwc_chains": int(optimized)}
