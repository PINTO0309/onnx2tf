from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _is_singleton_constant_tensor,
    _permute_shape,
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
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_onnx_shape,
)


def optimize_transpose_pre_concat_nhwc_chains_legacy(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Convert NCHW concat blocks back to NHWC when they are wrapped by transpose adapters.

    Target:
      ... -> t_i_nchw (some inputs may be pseudo-LeakyRelu outputs from transpose-wrapped NHWC)
      CONCAT(axis=1, [t_0_nchw, ...]) -> y_nchw
      y_nchw -> TRANSPOSE(0,2,3,1) -> y_nhwc

    Rewrite:
      ... -> t_i_nhwc
      CONCAT(axis=3, [t_0_nhwc, ...]) -> y_nhwc
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    # NHWC <-> NHCW (self-inverse). Used to keep SOFTMAX "last-axis" semantics
    # while lifting pre-concat chains to NHWC.
    perm_nhwc_to_nhcw = [0, 1, 3, 2]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _find_or_create_perm_tensor(
        *,
        base_name: str,
        perm: List[int],
    ) -> str:
        perm_arr = np.asarray([int(v) for v in list(perm)], dtype=np.int32)
        for tensor_name, tensor in model_ir.tensors.items():
            if tensor is None or tensor.data is None:
                continue
            try:
                data = np.asarray(tensor.data)
            except Exception:
                continue
            if data.dtype != np.int32 or int(data.size) != int(len(perm)):
                continue
            if np.array_equal(data.reshape(-1), perm_arr):
                return str(tensor_name)
        new_name = _unique_tensor_name(str(base_name))
        model_ir.tensors[new_name] = TensorIR(
            name=new_name,
            dtype="INT32",
            shape=[int(len(perm))],
            shape_signature=[int(len(perm))],
            data=np.asarray(perm_arr),
            is_variable=False,
            quantization=None,
        )
        return str(new_name)

    def _select_prelu_alpha_for_nhwc(
        *,
        alpha_data: np.ndarray,
        target_nhwc_shape: Optional[List[int]],
    ) -> Optional[np.ndarray]:
        alpha_candidates: List[np.ndarray] = []
        if int(alpha_data.ndim) == int(len(perm_nchw_to_nhwc)):
            alpha_candidates.append(
                np.transpose(alpha_data, axes=perm_nchw_to_nhwc).astype(alpha_data.dtype, copy=False)
            )
        alpha_candidates.append(np.asarray(alpha_data))
        if int(alpha_data.ndim) == 3:
            alpha_candidates.append(
                np.transpose(alpha_data, axes=[1, 2, 0]).astype(alpha_data.dtype, copy=False)
            )

        def _broadcastable(candidate: np.ndarray) -> bool:
            if target_nhwc_shape is None or not _is_fully_known_positive_shape(target_nhwc_shape):
                return True
            return (
                _broadcast_static_shapes(
                    [int(v) for v in list(target_nhwc_shape)],
                    [int(v) for v in list(candidate.shape)],
                )
                is not None
            )

        for candidate in alpha_candidates:
            if _broadcastable(candidate):
                return np.asarray(candidate)
        return None

    def _analyze_swish_input_to_nhwc(
        *,
        input_name: str,
        consumer_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
        allow_consumer_fanout_to_slice: bool = False,
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
        input_users = set(int(v) for v in consumers.get(str(input_name), []))
        if not allow_consumer_fanout_to_slice:
            if input_users != {int(consumer_idx)}:
                return None
        else:
            if int(consumer_idx) not in input_users or len(input_users) <= 0:
                return None
            for user_idx in sorted(list(input_users)):
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) != "SLICE" or len(user_op.inputs) < 3 or len(user_op.outputs) != 1:
                    return None
                if str(user_op.inputs[0]) != str(input_name):
                    return None
                begin_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(user_op.inputs[1]), None))
                size_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(user_op.inputs[2]), None))
                if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
                    return None
                if int(size_vals[1]) <= 0:
                    return None
                if int(begin_vals[2]) != 0 or int(begin_vals[3]) != 0:
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

    def _try_rewrite_swish_input_to_nhwc(
        *,
        input_name: str,
        consumer_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        return _analyze_swish_input_to_nhwc(
            input_name=input_name,
            consumer_idx=consumer_idx,
            producers=producers,
            consumers=consumers,
            model_outputs=model_outputs,
        )

    def _try_rewrite_leaky_input_to_nhwc(
        *,
        input_name: str,
        concat_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        sub_idx = producers.get(str(input_name), None)
        if sub_idx is None:
            return None
        sub_op = model_ir.operators[int(sub_idx)]
        if str(sub_op.op_type) != "SUB" or len(sub_op.inputs) != 2 or len(sub_op.outputs) != 1:
            return None
        if str(sub_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        input_users = [int(v) for v in consumers.get(str(input_name), [])]
        if set(input_users) != {int(concat_idx)}:
            return None

        relu_pos_out_name = str(sub_op.inputs[0])
        mul_out_name = str(sub_op.inputs[1])

        relu_pos_idx = producers.get(relu_pos_out_name, None)
        mul_idx = producers.get(mul_out_name, None)
        if relu_pos_idx is None or mul_idx is None:
            return None
        relu_pos_op = model_ir.operators[int(relu_pos_idx)]
        mul_op = model_ir.operators[int(mul_idx)]
        if str(relu_pos_op.op_type) != "RELU" or len(relu_pos_op.inputs) != 1 or len(relu_pos_op.outputs) != 1:
            return None
        if str(relu_pos_op.outputs[0]) != relu_pos_out_name:
            return None
        if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
            return None
        if str(mul_op.outputs[0]) != mul_out_name:
            return None
        if not _is_singleton_constant_tensor(model_ir, str(mul_op.inputs[0])) and not _is_singleton_constant_tensor(
            model_ir, str(mul_op.inputs[1])
        ):
            return None

        relu_pos_in_name = str(relu_pos_op.inputs[0])
        mul_in0 = str(mul_op.inputs[0])
        mul_in1 = str(mul_op.inputs[1])
        relu_neg_out_name = mul_in0 if mul_in0 != relu_pos_in_name else mul_in1
        if relu_neg_out_name == relu_pos_in_name:
            relu_neg_out_name = mul_in1 if mul_in0 == relu_pos_in_name else mul_in0
        relu_neg_idx = producers.get(relu_neg_out_name, None)
        if relu_neg_idx is None:
            return None
        relu_neg_op = model_ir.operators[int(relu_neg_idx)]
        if str(relu_neg_op.op_type) != "RELU" or len(relu_neg_op.inputs) != 1 or len(relu_neg_op.outputs) != 1:
            return None
        if str(relu_neg_op.outputs[0]) != relu_neg_out_name:
            return None

        neg_out_name = str(relu_neg_op.inputs[0])
        neg_idx = producers.get(neg_out_name, None)
        if neg_idx is None:
            return None
        neg_op = model_ir.operators[int(neg_idx)]
        if str(neg_op.op_type) != "NEG" or len(neg_op.inputs) != 1 or len(neg_op.outputs) != 1:
            return None
        if str(neg_op.outputs[0]) != neg_out_name:
            return None

        pre_nchw_name = str(neg_op.inputs[0])
        if relu_pos_in_name != pre_nchw_name:
            return None
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
        pre_users = [int(v) for v in consumers.get(pre_nchw_name, [])]
        if set(pre_users) != {int(neg_idx), int(relu_pos_idx)}:
            return None

        if set(int(v) for v in consumers.get(neg_out_name, [])) != {int(relu_neg_idx)}:
            return None
        if set(int(v) for v in consumers.get(relu_neg_out_name, [])) != {int(mul_idx)}:
            return None
        if set(int(v) for v in consumers.get(relu_pos_out_name, [])) != {int(sub_idx)}:
            return None
        if set(int(v) for v in consumers.get(mul_out_name, [])) != {int(sub_idx)}:
            return None

        return {
            "input_name": str(input_name),
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_op.inputs[0]),
            "neg_idx": int(neg_idx),
            "relu_pos_idx": int(relu_pos_idx),
            "tensors_to_permute": [
                str(neg_out_name),
                str(relu_pos_out_name),
                str(relu_neg_out_name),
                str(mul_out_name),
                str(input_name),
            ],
        }

    def _apply_leaky_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        pre_idx = int(plan["pre_idx"])
        pre_input_name = str(plan["pre_input_name"])
        neg_idx = int(plan["neg_idx"])
        relu_pos_idx = int(plan["relu_pos_idx"])
        tensors_to_permute = [str(v) for v in list(plan.get("tensors_to_permute", []))]

        _set_operator_inputs(
            model_ir=model_ir,
            op=model_ir.operators[int(neg_idx)],
            new_inputs=[pre_input_name],
        )
        _set_operator_inputs(
            model_ir=model_ir,
            op=model_ir.operators[int(relu_pos_idx)],
            new_inputs=[pre_input_name],
        )
        for tensor_name in tensors_to_permute:
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(tensor_name), None),
                perm_nchw_to_nhwc,
            )
        return [int(pre_idx)]

    def _try_rewrite_split_input_to_nhwc(
        *,
        input_name: str,
        consumer_idx: int,
        concat_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        split_idx = producers.get(str(input_name), None)
        if split_idx is None:
            return None
        split_op = model_ir.operators[int(split_idx)]
        if str(split_op.op_type) != "SPLIT" or len(split_op.inputs) < 2 or len(split_op.outputs) < 2:
            return None
        if str(input_name) not in set(str(v) for v in list(split_op.outputs)):
            return None
        if str(input_name) in model_outputs:
            return None

        split_axis_tensor_name = str(split_op.inputs[0])
        split_axis_vals = _read_const_ints_from_tensor(model_ir.tensors.get(split_axis_tensor_name, None))
        if split_axis_vals is None or len(split_axis_vals) != 1:
            return None

        split_input_name = str(split_op.inputs[1])
        split_input_tensor = model_ir.tensors.get(split_input_name, None)
        split_rank = int(len(list(split_input_tensor.shape))) if split_input_tensor is not None else 4
        if split_rank != 4:
            return None

        split_axis = int(split_axis_vals[0])
        if split_axis < 0:
            split_axis += int(split_rank)
        if split_axis != 1:
            return None

        source_plan: Dict[str, Any]
        swish_plan = _analyze_swish_input_to_nhwc(
            input_name=split_input_name,
            consumer_idx=int(split_idx),
            producers=producers,
            consumers=consumers,
            model_outputs=model_outputs,
        )
        if swish_plan is not None:
            source_plan = {
                "kind": "swish",
                "plan": dict(swish_plan),
            }
        else:
            pre_idx = producers.get(split_input_name, None)
            if pre_idx is None:
                return None
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != str(split_input_name)
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                or str(split_input_name) in model_outputs
            ):
                return None
            pre_users = set(int(v) for v in consumers.get(str(split_input_name), []))
            source_plan = {
                "kind": "direct",
                "pre_idx": int(pre_idx),
                "pre_input_name": str(pre_op.inputs[0]),
                "remove_pre": bool(pre_users == {int(split_idx)}),
            }

        post_transpose_indices: List[int] = []
        for split_out_name in [str(v) for v in list(split_op.outputs)]:
            if split_out_name in model_outputs:
                return None
            out_users = [int(v) for v in consumers.get(split_out_name, [])]
            for user_idx in out_users:
                if int(user_idx) in {int(concat_idx), int(consumer_idx)}:
                    continue
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)

                if user_type == "ADD":
                    if (
                        len(user_op.inputs) != 2
                        or len(user_op.outputs) != 1
                        or str(user_op.outputs[0]) in model_outputs
                    ):
                        return None
                    add_users = set(int(v) for v in consumers.get(str(user_op.outputs[0]), []))
                    if int(concat_idx) not in add_users:
                        return None
                    for add_user_idx in sorted(list(add_users)):
                        if int(add_user_idx) == int(concat_idx):
                            continue
                        add_user_op = model_ir.operators[int(add_user_idx)]
                        if (
                            str(add_user_op.op_type) == "TRANSPOSE"
                            and len(add_user_op.inputs) >= 2
                            and len(add_user_op.outputs) == 1
                            and str(add_user_op.inputs[0]) == str(user_op.outputs[0])
                            and _read_transpose_perm(model_ir, add_user_op) == perm_nchw_to_nhwc
                            and str(add_user_op.outputs[0]) not in model_outputs
                        ):
                            continue
                        if (
                            str(add_user_op.op_type) == "ADD"
                            and len(add_user_op.outputs) == 1
                            and str(add_user_op.outputs[0]) not in model_outputs
                            and int(concat_idx)
                            in set(
                                int(v)
                                for v in consumers.get(str(add_user_op.outputs[0]), [])
                            )
                        ):
                            continue
                        return None
                    continue

                if (
                    user_type == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == str(split_out_name)
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_transpose_indices.append(int(user_idx))
                    continue

                return None

        return {
            "input_name": str(input_name),
            "split_idx": int(split_idx),
            "source_plan": dict(source_plan),
            "post_transpose_indices": [int(v) for v in list(sorted(set(post_transpose_indices)))],
        }

    def _apply_split_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        split_idx = int(plan["split_idx"])
        if int(split_idx) < 0 or int(split_idx) >= len(model_ir.operators):
            return []
        split_op = model_ir.operators[int(split_idx)]
        if str(split_op.op_type) != "SPLIT" or len(split_op.inputs) < 2:
            return []

        remove_indices: List[int] = []
        source_plan = dict(plan.get("source_plan", {}))
        source_kind = str(source_plan.get("kind", ""))
        if source_kind == "swish":
            swish_plan = source_plan.get("plan", None)
            if swish_plan is not None:
                remove_indices.extend([int(v) for v in list(_apply_swish_nhwc_plan(plan=dict(swish_plan)))])
        elif source_kind == "direct":
            _replace_operator_input_at(
                model_ir=model_ir,
                op=split_op,
                input_index=1,
                new_input_name=str(source_plan["pre_input_name"]),
            )
            if bool(source_plan.get("remove_pre", False)):
                remove_indices.append(int(source_plan["pre_idx"]))
        else:
            return []

        axis_input_name = str(split_op.inputs[0])
        axis_tensor = model_ir.tensors.get(axis_input_name, None)
        axis_users = set(int(v) for v in _build_tensor_consumer_map(model_ir).get(axis_input_name, []))
        if axis_tensor is not None and axis_users == {int(split_idx)}:
            _write_const_ints_to_tensor(axis_tensor, [3])
        else:
            axis_name_new = _unique_tensor_name(f"{axis_input_name}_nhwc")
            model_ir.tensors[axis_name_new] = TensorIR(
                name=axis_name_new,
                dtype="INT32",
                shape=[1],
                shape_signature=[1],
                data=np.asarray([3], dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=split_op,
                input_index=0,
                new_input_name=str(axis_name_new),
            )

        for split_out_name in [str(v) for v in list(split_op.outputs)]:
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(split_out_name), None),
                perm_nchw_to_nhwc,
            )

        for post_idx in [int(v) for v in list(plan.get("post_transpose_indices", []))]:
            if int(post_idx) < 0 or int(post_idx) >= len(model_ir.operators):
                continue
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
                or str(post_op.outputs[0]) in set(str(v) for v in model_ir.outputs)
            ):
                continue
            _replace_tensor_inputs(
                model_ir=model_ir,
                src_name=str(post_op.outputs[0]),
                dst_name=str(post_op.inputs[0]),
            )
            remove_indices.append(int(post_idx))

        return [int(v) for v in list(remove_indices)]

    def _try_rewrite_slice_input_to_nhwc(
        *,
        input_name: str,
        concat_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        slice_idx = producers.get(str(input_name), None)
        if slice_idx is None:
            return None
        slice_op = model_ir.operators[int(slice_idx)]
        if str(slice_op.op_type) != "SLICE" or len(slice_op.inputs) < 3 or len(slice_op.outputs) != 1:
            return None
        if str(slice_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None

        begin_tensor = model_ir.tensors.get(str(slice_op.inputs[1]), None)
        size_tensor = model_ir.tensors.get(str(slice_op.inputs[2]), None)
        begin_vals = _read_const_ints_from_tensor(begin_tensor)
        size_vals = _read_const_ints_from_tensor(size_tensor)
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return None
        if int(size_vals[1]) <= 0:
            return None
        if int(begin_vals[2]) != 0 or int(begin_vals[3]) != 0:
            return None

        slice_input_name = str(slice_op.inputs[0])
        slice_input_tensor = model_ir.tensors.get(slice_input_name, None)
        if slice_input_tensor is None or len(list(slice_input_tensor.shape)) != 4:
            return None

        source_plan: Dict[str, Any]
        slice_group_indices: List[int]
        swish_plan = _analyze_swish_input_to_nhwc(
            input_name=slice_input_name,
            consumer_idx=int(slice_idx),
            producers=producers,
            consumers=consumers,
            model_outputs=model_outputs,
            allow_consumer_fanout_to_slice=True,
        )
        if swish_plan is not None:
            source_plan = {
                "kind": "swish",
                "plan": dict(swish_plan),
            }
            slice_group_indices = sorted(
                set(int(v) for v in consumers.get(str(slice_input_name), []))
            )
        else:
            pre_idx = producers.get(slice_input_name, None)
            if pre_idx is None:
                return None
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != str(slice_input_name)
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                or str(slice_input_name) in model_outputs
            ):
                return None
            pre_users = set(int(v) for v in consumers.get(str(slice_input_name), []))
            source_plan = {
                "kind": "direct",
                "pre_idx": int(pre_idx),
                "pre_input_name": str(pre_op.inputs[0]),
                "remove_pre": bool(pre_users == {int(slice_idx)}),
            }
            slice_group_indices = [int(slice_idx)]

        post_transpose_indices: set[int] = set()
        for target_slice_idx in [int(v) for v in list(slice_group_indices)]:
            if int(target_slice_idx) < 0 or int(target_slice_idx) >= len(model_ir.operators):
                return None
            target_slice_op = model_ir.operators[int(target_slice_idx)]
            if (
                str(target_slice_op.op_type) != "SLICE"
                or len(target_slice_op.inputs) < 3
                or len(target_slice_op.outputs) != 1
                or str(target_slice_op.inputs[0]) != str(slice_input_name)
            ):
                return None
            target_out_name = str(target_slice_op.outputs[0])
            if target_out_name in model_outputs:
                return None
            for user_idx in [int(v) for v in consumers.get(target_out_name, [])]:
                if int(user_idx) == int(concat_idx):
                    continue
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == target_out_name
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_transpose_indices.add(int(user_idx))
                    continue
                return None

        return {
            "input_name": str(input_name),
            "slice_idx": int(slice_idx),
            "concat_idx": int(concat_idx),
            "slice_group_key": (
                str(slice_input_name)
                if str(source_plan.get("kind", "")) == "swish" and len(slice_group_indices) > 1
                else ""
            ),
            "slice_group_indices": [int(v) for v in list(slice_group_indices)],
            "source_plan": dict(source_plan),
            "post_transpose_indices": [int(v) for v in sorted(post_transpose_indices)],
        }

    def _apply_slice_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        slice_idx = int(plan["slice_idx"])
        if int(slice_idx) < 0 or int(slice_idx) >= len(model_ir.operators):
            return []
        slice_op = model_ir.operators[int(slice_idx)]
        if str(slice_op.op_type) != "SLICE" or len(slice_op.inputs) < 3 or len(slice_op.outputs) != 1:
            return []

        remove_indices: List[int] = []
        source_plan = dict(plan.get("source_plan", {}))
        source_kind = str(source_plan.get("kind", ""))
        if source_kind == "swish":
            swish_plan = source_plan.get("plan", None)
            if swish_plan is not None:
                remove_indices.extend([int(v) for v in list(_apply_swish_nhwc_plan(plan=dict(swish_plan)))])
        elif source_kind == "direct":
            _replace_operator_input_at(
                model_ir=model_ir,
                op=slice_op,
                input_index=0,
                new_input_name=str(source_plan["pre_input_name"]),
            )
            if bool(source_plan.get("remove_pre", False)):
                remove_indices.append(int(source_plan["pre_idx"]))
        else:
            return []

        target_slice_indices = [
            int(v) for v in list(plan.get("slice_group_indices", [int(slice_idx)]))
        ]
        for target_slice_idx in target_slice_indices:
            if int(target_slice_idx) < 0 or int(target_slice_idx) >= len(model_ir.operators):
                return []
            target_slice_op = model_ir.operators[int(target_slice_idx)]
            if (
                str(target_slice_op.op_type) != "SLICE"
                or len(target_slice_op.inputs) < 3
                or len(target_slice_op.outputs) != 1
            ):
                return []
            begin_tensor = model_ir.tensors.get(str(target_slice_op.inputs[1]), None)
            size_tensor = model_ir.tensors.get(str(target_slice_op.inputs[2]), None)
            begin_vals = _read_const_ints_from_tensor(begin_tensor)
            size_vals = _read_const_ints_from_tensor(size_tensor)
            if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
                return []
            _write_const_ints_to_tensor(
                begin_tensor,
                [int(begin_vals[0]), int(begin_vals[2]), int(begin_vals[3]), int(begin_vals[1])],
            )
            _write_const_ints_to_tensor(
                size_tensor,
                [int(size_vals[0]), int(size_vals[2]), int(size_vals[3]), int(size_vals[1])],
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(target_slice_op.outputs[0]), None),
                perm_nchw_to_nhwc,
            )

        for post_idx in [int(v) for v in list(plan.get("post_transpose_indices", []))]:
            if int(post_idx) < 0 or int(post_idx) >= len(model_ir.operators):
                continue
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
                or str(post_op.outputs[0]) in set(str(v) for v in model_ir.outputs)
            ):
                continue
            _replace_tensor_inputs(
                model_ir=model_ir,
                src_name=str(post_op.outputs[0]),
                dst_name=str(post_op.inputs[0]),
            )
            remove_indices.append(int(post_idx))

        return [int(v) for v in list(remove_indices)]

    def _is_indexed_direct_slice_plan(plan: Dict[str, Any]) -> bool:
        source_plan = dict(plan.get("source_plan", {}))
        source_name = str(source_plan.get("pre_input_name", ""))
        source_tensor = model_ir.tensors.get(source_name, None)
        slice_idx = int(plan.get("slice_idx", -1))
        slice_group_indices = [
            int(value)
            for value in list(plan.get("slice_group_indices", []))
        ]
        return (
            str(source_plan.get("kind", "")) == "direct"
            and slice_group_indices == [slice_idx]
            and source_tensor is not None
            and len(list(source_tensor.shape)) == 4
        )

    def _is_indexed_direct_split_plan(
        plan: Dict[str, Any],
        *,
        concat_idx: int,
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> bool:
        source_plan = dict(plan.get("source_plan", {}))
        source_name = str(source_plan.get("pre_input_name", ""))
        source_tensor = model_ir.tensors.get(source_name, None)
        split_idx = int(plan.get("split_idx", -1))
        post_transpose_indices = {
            int(value)
            for value in list(plan.get("post_transpose_indices", []))
        }
        if (
            str(source_plan.get("kind", "")) != "direct"
            or source_tensor is None
            or len(list(source_tensor.shape)) != 4
            or split_idx < 0
            or split_idx >= len(model_ir.operators)
        ):
            return False
        split_op = model_ir.operators[split_idx]
        if (
            str(split_op.op_type) != "SPLIT"
            or len(split_op.inputs) < 2
            or len(split_op.outputs) < 2
        ):
            return False
        for output_name in [str(name) for name in split_op.outputs]:
            output_tensor = model_ir.tensors.get(output_name, None)
            if (
                output_name in model_outputs
                or output_tensor is None
                or len(list(output_tensor.shape)) != 4
                or not set(
                    int(value) for value in consumers.get(output_name, [])
                ).issubset(
                    {int(concat_idx), *post_transpose_indices}
                )
            ):
                return False
        return True

    def _is_indexed_direct_add_plan(
        plan: Dict[str, Any],
        *,
        concat_idx: int,
        consumers: Dict[str, List[int]],
    ) -> bool:
        add_idx = int(plan.get("add_idx", -1))
        actions = [dict(action) for action in plan.get("actions", [])]
        input_name = str(plan.get("input_name", ""))
        post_transpose_indices = {
            int(value)
            for value in list(
                plan.get("removable_post_transpose_indices", [])
            )
        }
        return (
            add_idx >= 0
            and add_idx < len(model_ir.operators)
            and str(model_ir.operators[add_idx].op_type) == "ADD"
            and len(model_ir.operators[add_idx].inputs) == 2
            and len(model_ir.operators[add_idx].outputs) == 1
            and len(actions) == 2
            and all(
                str(action.get("kind", "")) == "direct"
                and len(list(action.get("remove_indices", []))) >= 1
                for action in actions
            )
            and len(list(plan.get("new_add_inputs", []))) == 2
            and set(
                int(value) for value in consumers.get(input_name, [])
            )
            == {int(concat_idx), *post_transpose_indices}
        )

    def _try_rewrite_add_input_to_nhwc(
        *,
        input_name: str,
        concat_idx: int,
        root_concat_idx: Optional[int] = None,
        visited_add_outputs: Optional[set[str]] = None,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        if root_concat_idx is None:
            root_concat_idx = int(concat_idx)
        if visited_add_outputs is None:
            visited_add_outputs = set()
        if str(input_name) in visited_add_outputs:
            return None

        add_idx = producers.get(str(input_name), None)
        if add_idx is None:
            return None
        add_op = model_ir.operators[int(add_idx)]
        if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
            return None
        if str(add_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        input_users = set(int(v) for v in consumers.get(str(input_name), []))
        if int(concat_idx) not in input_users:
            return None

        removable_post_transpose_indices: List[int] = []
        for user_idx in sorted(list(input_users)):
            if int(user_idx) == int(concat_idx):
                continue
            if int(user_idx) == int(root_concat_idx):
                continue
            user_op = model_ir.operators[int(user_idx)]
            if (
                str(user_op.op_type) == "TRANSPOSE"
                and len(user_op.inputs) >= 2
                and len(user_op.outputs) == 1
                and str(user_op.inputs[0]) == str(input_name)
                and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                and str(user_op.outputs[0]) not in model_outputs
            ):
                removable_post_transpose_indices.append(int(user_idx))
                continue
            if (
                str(user_op.op_type) == "ADD"
                and len(user_op.outputs) == 1
                and str(user_op.outputs[0]) not in model_outputs
                and int(root_concat_idx) in set(int(v) for v in consumers.get(str(user_op.outputs[0]), []))
            ):
                continue
            return None

        actions: List[Dict[str, Any]] = []
        next_visited = set(str(v) for v in visited_add_outputs)
        next_visited.add(str(input_name))
        for add_input_name in [str(v) for v in list(add_op.inputs)]:
            input_producer_idx = producers.get(add_input_name, None)
            if input_producer_idx is not None:
                input_producer_op = model_ir.operators[int(input_producer_idx)]
                input_users = set(int(v) for v in consumers.get(add_input_name, []))
                if (
                    str(input_producer_op.op_type) == "TRANSPOSE"
                    and len(input_producer_op.inputs) >= 2
                    and len(input_producer_op.outputs) == 1
                    and str(input_producer_op.outputs[0]) == add_input_name
                    and _read_transpose_perm(model_ir, input_producer_op) == perm_nhwc_to_nchw
                    and int(add_idx) in input_users
                ):
                    extra_users = {int(v) for v in input_users if int(v) != int(add_idx)}
                    if extra_users.difference({int(root_concat_idx)}):
                        return None
                    actions.append(
                        {
                            "kind": "direct",
                            "new_input_name": str(input_producer_op.inputs[0]),
                            "remove_indices": (
                                [int(input_producer_idx)] if len(extra_users) == 0 else []
                            ),
                        }
                    )
                    continue

            unary_plan = _try_rewrite_unary_input_to_nhwc(
                input_name=add_input_name,
                concat_idx=int(add_idx),
                producers=producers,
                consumers=consumers,
                model_outputs=model_outputs,
            )
            if unary_plan is not None:
                actions.append(
                    {
                        "kind": "unary",
                        "new_input_name": str(add_input_name),
                        "plan": unary_plan,
                    }
                )
                continue

            swish_plan = _analyze_swish_input_to_nhwc(
                input_name=add_input_name,
                consumer_idx=int(add_idx),
                producers=producers,
                consumers=consumers,
                model_outputs=model_outputs,
            )
            if swish_plan is not None:
                actions.append(
                    {
                        "kind": "swish",
                        "new_input_name": str(add_input_name),
                        "plan": swish_plan,
                    }
                )
                continue

            split_plan = _try_rewrite_split_input_to_nhwc(
                input_name=add_input_name,
                consumer_idx=int(add_idx),
                concat_idx=int(root_concat_idx),
                producers=producers,
                consumers=consumers,
                model_outputs=model_outputs,
            )
            if split_plan is not None:
                actions.append(
                    {
                        "kind": "split",
                        "new_input_name": str(add_input_name),
                        "plan": split_plan,
                    }
                )
                continue

            nested_add_plan = _try_rewrite_add_input_to_nhwc(
                input_name=add_input_name,
                concat_idx=int(add_idx),
                root_concat_idx=int(root_concat_idx),
                visited_add_outputs=set(next_visited),
                producers=producers,
                consumers=consumers,
                model_outputs=model_outputs,
            )
            if nested_add_plan is not None:
                actions.append(
                    {
                        "kind": "add_chain",
                        "new_input_name": str(add_input_name),
                        "plan": dict(nested_add_plan),
                    }
                )
                continue

            return None

        new_add_inputs: List[str] = []
        remove_indices: List[int] = []
        for action in actions:
            action_kind = str(action.get("kind", ""))
            if action_kind == "direct":
                new_add_inputs.append(str(action["new_input_name"]))
                remove_indices.extend([int(v) for v in list(action.get("remove_indices", []))])
                continue
            if action_kind == "swish":
                new_add_inputs.append(str(action["new_input_name"]))
                remove_indices.extend([])
                continue
            if action_kind == "unary":
                new_add_inputs.append(str(action["new_input_name"]))
                remove_indices.extend([])
                continue
            if action_kind == "split":
                new_add_inputs.append(str(action["new_input_name"]))
                remove_indices.extend([])
                continue
            if action_kind == "add_chain":
                new_add_inputs.append(str(action["new_input_name"]))
                remove_indices.extend([])
                continue
            return None

        return {
            "input_name": str(input_name),
            "add_idx": int(add_idx),
            "root_concat_idx": int(root_concat_idx),
            "new_add_inputs": [str(v) for v in new_add_inputs],
            "remove_indices": [int(v) for v in remove_indices],
            "removable_post_transpose_indices": [int(v) for v in list(sorted(set(removable_post_transpose_indices)))],
            "actions": [dict(v) for v in actions],
        }

    def _apply_add_nhwc_plan(
        *,
        plan: Dict[str, Any],
        applied_split_indices: Optional[set[int]] = None,
        applied_add_indices: Optional[set[int]] = None,
    ) -> List[int]:
        input_name = str(plan["input_name"])
        add_idx = int(plan["add_idx"])
        new_add_inputs = [str(v) for v in list(plan.get("new_add_inputs", []))]
        remove_indices = [int(v) for v in list(plan.get("remove_indices", []))]
        removable_post_transpose_indices = [
            int(v) for v in list(plan.get("removable_post_transpose_indices", []))
        ]
        actions = [dict(v) for v in list(plan.get("actions", []))]
        if applied_split_indices is None:
            applied_split_indices = set()
        if applied_add_indices is None:
            applied_add_indices = set()
        if int(add_idx) in applied_add_indices:
            return []

        for action in actions:
            action_kind = str(action.get("kind", ""))
            if action_kind == "swish":
                swish_plan = action.get("plan", None)
                if swish_plan is None:
                    continue
                swish_pre_remove = _apply_swish_nhwc_plan(plan=dict(swish_plan))
                remove_indices.extend([int(v) for v in swish_pre_remove])
                continue
            if action_kind == "unary":
                unary_plan = action.get("plan", None)
                if unary_plan is None:
                    continue
                unary_pre_remove = _apply_unary_nhwc_plan(plan=dict(unary_plan))
                remove_indices.extend([int(v) for v in unary_pre_remove])
                continue
            if action_kind == "split":
                split_plan = action.get("plan", None)
                if split_plan is None:
                    continue
                split_idx = int(split_plan.get("split_idx", -1))
                if int(split_idx) in applied_split_indices:
                    continue
                split_pre_remove = _apply_split_nhwc_plan(plan=dict(split_plan))
                remove_indices.extend([int(v) for v in list(split_pre_remove)])
                applied_split_indices.add(int(split_idx))
                continue
            if action_kind == "add_chain":
                nested_plan = action.get("plan", None)
                if nested_plan is None:
                    continue
                nested_pre_remove = _apply_add_nhwc_plan(
                    plan=dict(nested_plan),
                    applied_split_indices=applied_split_indices,
                    applied_add_indices=applied_add_indices,
                )
                remove_indices.extend([int(v) for v in list(nested_pre_remove)])
                continue

        _set_operator_inputs(
            model_ir=model_ir,
            op=model_ir.operators[int(add_idx)],
            new_inputs=[str(v) for v in list(new_add_inputs)],
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(input_name, None),
            perm_nchw_to_nhwc,
        )
        for post_idx in removable_post_transpose_indices:
            if int(post_idx) < 0 or int(post_idx) >= len(model_ir.operators):
                continue
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
                or str(post_op.outputs[0]) in set(str(v) for v in model_ir.outputs)
            ):
                continue
            _replace_tensor_inputs(
                model_ir=model_ir,
                src_name=str(post_op.outputs[0]),
                dst_name=str(post_op.inputs[0]),
            )
            remove_indices.append(int(post_idx))
        applied_add_indices.add(int(add_idx))
        return [int(v) for v in remove_indices]

    def _try_rewrite_pad_input_to_nhwc(
        *,
        input_name: str,
        concat_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        pad_idx = producers.get(str(input_name), None)
        if pad_idx is None:
            return None
        pad_op = model_ir.operators[int(pad_idx)]
        if str(pad_op.op_type) != "PAD" or len(pad_op.inputs) < 2 or len(pad_op.outputs) != 1:
            return None
        if str(pad_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(concat_idx)}:
            return None

        pad_input_name = str(pad_op.inputs[0])
        pre_idx = producers.get(pad_input_name, None)
        if pre_idx is None:
            return None
        pre_op = model_ir.operators[int(pre_idx)]
        if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
            return None
        if str(pre_op.outputs[0]) != pad_input_name:
            return None
        if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
            return None
        if str(pad_input_name) in model_outputs:
            return None

        pads_tensor_name = str(pad_op.inputs[1])
        pads_tensor = model_ir.tensors.get(pads_tensor_name, None)
        if pads_tensor is None or pads_tensor.data is None:
            return None
        try:
            pads_array = np.asarray(pads_tensor.data)
            pads_pairs = pads_array.reshape(4, 2)
        except Exception:
            return None
        if int(pads_pairs.size) != 8:
            return None

        pre_users = set(int(v) for v in consumers.get(pad_input_name, []))
        remove_pre = pre_users == {int(pad_idx)}

        return {
            "input_name": str(input_name),
            "pad_idx": int(pad_idx),
            "pad_input_name": str(pad_input_name),
            "pad_input_rewritten_name": str(pre_op.inputs[0]),
            "pads_tensor_name": str(pads_tensor_name),
            "pre_idx": int(pre_idx),
            "remove_pre": bool(remove_pre),
        }

    def _apply_pad_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        input_name = str(plan["input_name"])
        pad_idx = int(plan["pad_idx"])
        pre_idx = int(plan["pre_idx"])
        remove_pre = bool(plan["remove_pre"])
        pad_rewritten_input_name = str(plan["pad_input_rewritten_name"])
        pads_tensor_name = str(plan["pads_tensor_name"])

        pad_op = model_ir.operators[int(pad_idx)]
        _set_operator_inputs(
            model_ir=model_ir,
            op=pad_op,
            new_inputs=[pad_rewritten_input_name, str(pad_op.inputs[1])],
        )

        pads_tensor = model_ir.tensors.get(pads_tensor_name, None)
        if pads_tensor is not None and pads_tensor.data is not None:
            pads_array = np.asarray(pads_tensor.data)
            pads_pairs = pads_array.reshape(4, 2)
            # NCHW [N,C,H,W] -> NHWC [N,H,W,C]
            pads_pairs = np.asarray(
                [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
                dtype=pads_array.dtype,
            )
            pads_tensor.data = pads_pairs
            pads_tensor.shape = [4, 2]
            pads_tensor.shape_signature = [4, 2]

        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(input_name, None),
            perm_nchw_to_nhwc,
        )

        return [int(pre_idx)] if remove_pre else []

    def _try_rewrite_unary_input_to_nhwc(
        *,
        input_name: str,
        concat_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        unary_idx = producers.get(str(input_name), None)
        if unary_idx is None:
            return None
        unary_op = model_ir.operators[int(unary_idx)]
        if str(unary_op.op_type) not in {"RELU", "RELU6", "LOGISTIC", "TANH", "GELU"}:
            return None
        if len(unary_op.inputs) != 1 or len(unary_op.outputs) != 1:
            return None
        if str(unary_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(concat_idx)}:
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
        if set(int(v) for v in consumers.get(pre_nchw_name, [])) != {int(unary_idx)}:
            return None

        return {
            "input_name": str(input_name),
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_op.inputs[0]),
            "unary_idx": int(unary_idx),
        }

    def _try_rewrite_softmax_input_to_nhwc(
        *,
        input_name: str,
        concat_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        softmax_idx = producers.get(str(input_name), None)
        if softmax_idx is None:
            return None
        softmax_op = model_ir.operators[int(softmax_idx)]
        if str(softmax_op.op_type) != "SOFTMAX":
            return None
        if len(softmax_op.inputs) != 1 or len(softmax_op.outputs) != 1:
            return None
        if str(softmax_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(concat_idx)}:
            return None

        pre_nchw_name = str(softmax_op.inputs[0])
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
        # Keep strict: the removed pre-transpose output must only feed SOFTMAX.
        if set(int(v) for v in consumers.get(pre_nchw_name, [])) != {int(softmax_idx)}:
            return None

        pre_input_tensor = model_ir.tensors.get(str(pre_op.inputs[0]), None)
        softmax_output_tensor = model_ir.tensors.get(str(input_name), None)
        if (
            pre_input_tensor is None
            or softmax_output_tensor is None
            or len(list(pre_input_tensor.shape)) != 4
            or len(list(softmax_output_tensor.shape)) != 4
        ):
            return None

        return {
            "input_name": str(input_name),
            "softmax_idx": int(softmax_idx),
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_op.inputs[0]),
            "softmax_input_name": str(pre_nchw_name),
        }

    def _try_rewrite_dequant_input_to_nhwc(
        *,
        input_name: str,
        concat_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        dq_idx = producers.get(str(input_name), None)
        if dq_idx is None:
            return None
        dq_op = model_ir.operators[int(dq_idx)]
        if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
            return None
        if str(dq_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(concat_idx)}:
            return None

        pre_nchw_name = str(dq_op.inputs[0])
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

        pre_users = set(int(v) for v in consumers.get(pre_nchw_name, []))
        remove_pre = pre_users == {int(dq_idx)}
        return {
            "input_name": str(input_name),
            "dq_idx": int(dq_idx),
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_op.inputs[0]),
            "remove_pre": bool(remove_pre),
        }

    def _apply_dequant_nhwc_plan(*, plan: Dict[str, Any]) -> List[int]:
        input_name = str(plan["input_name"])
        dq_idx = int(plan["dq_idx"])
        pre_idx = int(plan["pre_idx"])
        pre_input_name = str(plan["pre_input_name"])
        remove_pre = bool(plan["remove_pre"])

        _set_operator_inputs(
            model_ir=model_ir,
            op=model_ir.operators[int(dq_idx)],
            new_inputs=[pre_input_name],
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(input_name, None),
            perm_nchw_to_nhwc,
        )
        return [int(pre_idx)] if remove_pre else []

    def _try_rewrite_prelu_input_to_nhwc(
        *,
        input_name: str,
        concat_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        prelu_idx = producers.get(str(input_name), None)
        if prelu_idx is None:
            return None
        prelu_op = model_ir.operators[int(prelu_idx)]
        if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
            return None
        if str(prelu_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        if set(int(v) for v in consumers.get(str(input_name), [])) != {int(concat_idx)}:
            return None

        pre_nchw_name = str(prelu_op.inputs[0])
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
        if set(int(v) for v in consumers.get(pre_nchw_name, [])) != {int(prelu_idx)}:
            return None

        alpha_name = str(prelu_op.inputs[1])
        alpha_tensor = model_ir.tensors.get(alpha_name, None)
        if alpha_tensor is None or alpha_tensor.data is None:
            return None
        pre_input_tensor = model_ir.tensors.get(str(pre_op.inputs[0]), None)
        target_nhwc_shape = (
            [int(v) for v in list(pre_input_tensor.shape)]
            if pre_input_tensor is not None and pre_input_tensor.shape is not None
            else None
        )
        alpha_data = np.asarray(alpha_tensor.data)
        if _select_prelu_alpha_for_nhwc(
            alpha_data=alpha_data,
            target_nhwc_shape=target_nhwc_shape,
        ) is None:
            return None

        return {
            "input_name": str(input_name),
            "pre_idx": int(pre_idx),
            "pre_input_name": str(pre_op.inputs[0]),
            "prelu_idx": int(prelu_idx),
            "target_nhwc_shape": (
                [int(v) for v in list(target_nhwc_shape)]
                if target_nhwc_shape is not None
                else None
            ),
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

    def _apply_softmax_nhwc_plan(*, plan: Dict[str, Any]) -> Optional[List[int]]:
        input_name = str(plan["input_name"])
        softmax_idx = int(plan["softmax_idx"])
        pre_idx = int(plan["pre_idx"])
        pre_input_name = str(plan["pre_input_name"])
        old_softmax_input_name = str(plan["softmax_input_name"])

        if softmax_idx < 0 or softmax_idx >= len(model_ir.operators):
            return None
        softmax_op = model_ir.operators[int(softmax_idx)]
        if str(softmax_op.op_type) != "SOFTMAX" or len(softmax_op.inputs) != 1 or len(softmax_op.outputs) != 1:
            return None
        if str(softmax_op.outputs[0]) != input_name:
            return None

        pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
        old_softmax_input_tensor = model_ir.tensors.get(old_softmax_input_name, None)
        softmax_output_tensor = model_ir.tensors.get(input_name, None)
        if (
            pre_input_tensor is None
            or old_softmax_input_tensor is None
            or softmax_output_tensor is None
            or len(list(pre_input_tensor.shape)) != 4
            or len(list(old_softmax_input_tensor.shape)) != 4
            or len(list(softmax_output_tensor.shape)) != 4
        ):
            return None

        pre_input_shape = [int(v) for v in list(pre_input_tensor.shape)]
        pre_input_signature = (
            [int(v) for v in list(pre_input_tensor.shape_signature)]
            if pre_input_tensor.shape_signature is not None
            else [int(v) for v in list(pre_input_tensor.shape)]
        )
        axis_last_shape = _permute_shape(pre_input_shape, perm_nhwc_to_nhcw)
        axis_last_signature = _permute_shape(pre_input_signature, perm_nhwc_to_nhcw)
        if axis_last_shape is None or axis_last_signature is None:
            return None

        axis_last_input_name = _unique_tensor_name(f"{old_softmax_input_name}_axis_last")
        axis_last_output_name = _unique_tensor_name(f"{input_name}_axis_last")
        perm_tensor_name = _find_or_create_perm_tensor(
            base_name=f"{input_name}_nhwc_to_nhcw_perm",
            perm=perm_nhwc_to_nhcw,
        )

        model_ir.tensors[axis_last_input_name] = TensorIR(
            name=axis_last_input_name,
            dtype=str(old_softmax_input_tensor.dtype),
            shape=[int(v) for v in list(axis_last_shape)],
            shape_signature=[int(v) for v in list(axis_last_signature)],
            data=None,
            is_variable=False,
            quantization=_clone_quantization(old_softmax_input_tensor.quantization),
        )
        model_ir.tensors[axis_last_output_name] = TensorIR(
            name=axis_last_output_name,
            dtype=str(softmax_output_tensor.dtype),
            shape=[int(v) for v in list(axis_last_shape)],
            shape_signature=[int(v) for v in list(axis_last_signature)],
            data=None,
            is_variable=False,
            quantization=_clone_quantization(softmax_output_tensor.quantization),
        )

        # Keep SOFTMAX last-axis semantics by locally rotating NHWC -> NHCW -> SOFTMAX -> NHWC.
        _set_operator_inputs(
            model_ir=model_ir,
            op=softmax_op,
            new_inputs=[axis_last_input_name],
        )
        _set_operator_outputs(
            model_ir=model_ir,
            op=softmax_op,
            new_outputs=[axis_last_output_name],
        )

        model_ir.operators.insert(
            int(softmax_idx),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[pre_input_name, perm_tensor_name],
                outputs=[axis_last_input_name],
                options={},
            ),
        )
        model_ir.operators.insert(
            int(softmax_idx) + 2,
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[axis_last_output_name, perm_tensor_name],
                outputs=[input_name],
                options={},
            ),
        )

        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(input_name, None),
            perm_nchw_to_nhwc,
        )
        return [int(pre_idx)]

    def _apply_prelu_nhwc_plan(*, plan: Dict[str, Any]) -> Optional[List[int]]:
        input_name = str(plan["input_name"])
        prelu_idx = int(plan["prelu_idx"])
        pre_idx = int(plan["pre_idx"])
        pre_input_name = str(plan["pre_input_name"])
        target_nhwc_shape = (
            [int(v) for v in list(plan.get("target_nhwc_shape", []))]
            if plan.get("target_nhwc_shape", None) is not None
            else None
        )

        prelu_op = model_ir.operators[int(prelu_idx)]
        alpha_name = str(prelu_op.inputs[1])
        alpha_tensor = model_ir.tensors.get(alpha_name, None)
        if alpha_tensor is None or alpha_tensor.data is None:
            return None
        alpha_data = np.asarray(alpha_tensor.data)
        selected_alpha = _select_prelu_alpha_for_nhwc(
            alpha_data=alpha_data,
            target_nhwc_shape=target_nhwc_shape,
        )
        if selected_alpha is None:
            return None

        selected_alpha_name = str(alpha_name)
        alpha_users = [int(v) for v in _build_tensor_consumer_map(model_ir).get(alpha_name, [])]
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

        _set_operator_inputs(
            model_ir=model_ir,
            op=prelu_op,
            new_inputs=[pre_input_name, str(selected_alpha_name)],
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(input_name, None),
            perm_nchw_to_nhwc,
        )
        return [int(pre_idx)]

    def _project_shape_after_nchw_to_nhwc(tensor_name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            return None
        shape = [int(v) for v in list(tensor.shape)]
        if len(shape) != 4:
            return None
        return [int(shape[0]), int(shape[2]), int(shape[3]), int(shape[1])]

    indexed_quantized_simple_family_contracts = (
        (frozenset({"direct", "unary"}), frozenset({"unary"})),
        (frozenset({"direct", "pad"}), frozenset({"direct", "pad"})),
        (
            frozenset({"direct", "unary", "pad"}),
            frozenset({"unary", "pad"}),
        ),
        (
            frozenset({"direct", "unary", "swish"}),
            frozenset({"swish"}),
        ),
        (
            frozenset({"direct", "dequantize"}),
            frozenset({"dequantize"}),
        ),
        (frozenset({"direct", "prelu"}), frozenset({"prelu"})),
        (
            frozenset({"direct", "unary", "leaky"}),
            frozenset({"leaky"}),
        ),
    )
    indexed_float_simple_family_contracts = (
        (frozenset({"direct", "unary"}), frozenset({"unary"})),
        (frozenset({"direct", "pad"}), frozenset({"direct", "pad"})),
        (
            frozenset({"direct", "dequantize"}),
            frozenset({"dequantize"}),
        ),
        (frozenset({"direct", "prelu"}), frozenset({"prelu"})),
        (
            frozenset({"direct", "softmax"}),
            frozenset({"direct", "softmax"}),
        ),
        (
            frozenset({"direct", "unary", "swish"}),
            frozenset({"swish"}),
        ),
        (
            frozenset({"direct", "unary", "leaky"}),
            frozenset({"leaky"}),
        ),
    )

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue
            concat_out_name = str(concat_op.outputs[0])
            concat_out_tensor = model_ir.tensors.get(concat_out_name, None)
            if concat_out_tensor is None or len(list(concat_out_tensor.shape)) != 4:
                continue

            concat_axis_old = int(concat_op.options.get("axis", 1))
            if concat_axis_old < 0:
                concat_axis_old += 4
            if concat_axis_old != 1:
                continue

            concat_users = [int(v) for v in consumers.get(concat_out_name, [])]
            if len(concat_users) == 0:
                continue
            post_indices: List[int] = []
            post_output_names: List[str] = []
            post_quantize_idx: Optional[int] = None
            post_quantize_out_name: Optional[str] = None
            valid_post_only = True
            for user_idx in concat_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == concat_out_name
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_indices.append(int(user_idx))
                    post_output_names.append(str(user_op.outputs[0]))
                elif (
                    len(concat_users) == 1
                    and str(user_op.op_type) == "QUANTIZE"
                    and len(user_op.inputs) == 1
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == concat_out_name
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    q_out_name = str(user_op.outputs[0])
                    q_users = [int(v) for v in consumers.get(q_out_name, [])]
                    if len(q_users) == 0:
                        valid_post_only = False
                        break
                    valid_q_post = True
                    for q_user_idx in q_users:
                        q_user_op = model_ir.operators[int(q_user_idx)]
                        if (
                            str(q_user_op.op_type) == "TRANSPOSE"
                            and len(q_user_op.inputs) >= 2
                            and len(q_user_op.outputs) == 1
                            and str(q_user_op.inputs[0]) == q_out_name
                            and _read_transpose_perm(model_ir, q_user_op) == perm_nchw_to_nhwc
                            and str(q_user_op.outputs[0]) not in model_outputs
                        ):
                            post_indices.append(int(q_user_idx))
                            post_output_names.append(str(q_user_op.outputs[0]))
                        else:
                            valid_q_post = False
                            break
                    if not valid_q_post:
                        valid_post_only = False
                        break
                    post_quantize_idx = int(user_idx)
                    post_quantize_out_name = str(q_out_name)
                else:
                    valid_post_only = False
                    break
            if not valid_post_only or len(post_indices) == 0:
                continue
            post_op_ids: List[int] = [
                int(id(model_ir.operators[int(post_idx)]))
                for post_idx in post_indices
                if int(post_idx) >= 0 and int(post_idx) < len(model_ir.operators)
            ]

            concat_input_actions: List[Dict[str, Any]] = []
            rewritable = True
            for input_name in [str(v) for v in list(concat_op.inputs)]:
                input_producer_idx = producers.get(input_name, None)
                if input_producer_idx is not None:
                    input_producer_op = model_ir.operators[int(input_producer_idx)]
                    if (
                        str(input_producer_op.op_type) == "TRANSPOSE"
                        and len(input_producer_op.inputs) >= 2
                        and len(input_producer_op.outputs) == 1
                        and str(input_producer_op.outputs[0]) == input_name
                        and _read_transpose_perm(model_ir, input_producer_op) == perm_nhwc_to_nchw
                        and int(concat_idx) in set(int(v) for v in consumers.get(input_name, []))
                    ):
                        input_users = set(int(v) for v in consumers.get(input_name, []))
                        remove_indices = (
                            [int(input_producer_idx)]
                            if input_users == {int(concat_idx)}
                            else []
                        )
                        concat_input_actions.append(
                            {
                                "kind": "direct",
                                "input_name": str(input_name),
                                "new_input_name": str(input_producer_op.inputs[0]),
                                "remove_indices": remove_indices,
                            }
                        )
                        continue

                pad_plan = _try_rewrite_pad_input_to_nhwc(
                    input_name=input_name,
                    concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if pad_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "pad",
                            "input_name": str(input_name),
                            "plan": dict(pad_plan),
                        }
                    )
                    continue

                leaky_plan = _try_rewrite_leaky_input_to_nhwc(
                    input_name=input_name,
                    concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if leaky_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "leaky",
                            "input_name": str(input_name),
                            "plan": dict(leaky_plan),
                        }
                    )
                    continue

                dequant_plan = _try_rewrite_dequant_input_to_nhwc(
                    input_name=input_name,
                    concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if dequant_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "dequantize",
                            "input_name": str(input_name),
                            "plan": dict(dequant_plan),
                        }
                    )
                    continue

                unary_plan = _try_rewrite_unary_input_to_nhwc(
                    input_name=input_name,
                    concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if unary_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "unary",
                            "input_name": str(input_name),
                            "plan": dict(unary_plan),
                        }
                    )
                    continue

                softmax_plan = _try_rewrite_softmax_input_to_nhwc(
                    input_name=input_name,
                    concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if softmax_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "softmax",
                            "input_name": str(input_name),
                            "plan": dict(softmax_plan),
                        }
                    )
                    continue

                prelu_plan = _try_rewrite_prelu_input_to_nhwc(
                    input_name=input_name,
                    concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if prelu_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "prelu",
                            "input_name": str(input_name),
                            "plan": dict(prelu_plan),
                        }
                    )
                    continue

                swish_plan = _try_rewrite_swish_input_to_nhwc(
                    input_name=input_name,
                    consumer_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if swish_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "swish",
                            "input_name": str(input_name),
                            "plan": dict(swish_plan),
                        }
                    )
                    continue

                add_plan = _try_rewrite_add_input_to_nhwc(
                    input_name=input_name,
                    concat_idx=int(concat_idx),
                    root_concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if add_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "add",
                            "input_name": str(input_name),
                            "plan": dict(add_plan),
                        }
                    )
                    continue

                split_plan = _try_rewrite_split_input_to_nhwc(
                    input_name=input_name,
                    consumer_idx=int(concat_idx),
                    concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if split_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "split",
                            "input_name": str(input_name),
                            "plan": dict(split_plan),
                        }
                    )
                    continue

                slice_plan = _try_rewrite_slice_input_to_nhwc(
                    input_name=input_name,
                    concat_idx=int(concat_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if slice_plan is not None:
                    concat_input_actions.append(
                        {
                            "kind": "slice",
                            "input_name": str(input_name),
                            "plan": dict(slice_plan),
                        }
                    )
                    continue

                rewritable = False
                break

            if not rewritable:
                continue

            action_kind_counts: Dict[str, int] = {}
            for action in concat_input_actions:
                action_kind = str(action.get("kind", ""))
                action_kind_counts[action_kind] = (
                    action_kind_counts.get(action_kind, 0) + 1
                )
            action_kinds = set(action_kind_counts)
            softmax_action_count = int(
                action_kind_counts.get("softmax", 0)
            )
            if softmax_action_count > 0:
                # Safety + benefit gate:
                # - support only float-path concat-post-transpose rewrite
                # - require at least one non-softmax input so total transpose count decreases
                if post_quantize_idx is not None:
                    continue
                if softmax_action_count != 1:
                    continue
                if int(len(concat_input_actions) - softmax_action_count) <= 0:
                    continue

            all_direct_input_actions = (
                not action_kinds or action_kinds == {"direct"}
            )
            # Both strict direct-adapter families are owned by indexed
            # transactional passes, with or without post Quantize.
            if all_direct_input_actions:
                continue

            indexed_quantized_simple_family = (
                post_quantize_idx is not None
                and (
                    any(
                        required_kinds.issubset(action_kinds)
                        and action_kinds.issubset(allowed_kinds)
                        for allowed_kinds, required_kinds
                        in indexed_quantized_simple_family_contracts
                    )
                    or (
                        len(concat_input_actions) >= 2
                        and action_kinds == {"pad"}
                    )
                )
            )
            if indexed_quantized_simple_family:
                continue
            quantized_slice_actions = [
                action
                for action in concat_input_actions
                if str(action.get("kind", "")) == "slice"
            ]
            indexed_quantized_slice_family = (
                post_quantize_idx is not None
                and len(quantized_slice_actions) >= 1
                and all(
                    str(action.get("kind", "")) in {"direct", "slice"}
                    for action in concat_input_actions
                )
                and all(
                    _is_indexed_direct_slice_plan(
                        dict(action.get("plan", {}))
                    )
                    and not list(
                        dict(action.get("plan", {})).get(
                            "post_transpose_indices",
                            [],
                        )
                    )
                    for action in quantized_slice_actions
                )
            )
            if indexed_quantized_slice_family:
                continue
            quantized_split_actions = [
                action
                for action in concat_input_actions
                if str(action.get("kind", "")) == "split"
            ]
            indexed_quantized_split_family = (
                post_quantize_idx is not None
                and len(quantized_split_actions) >= 1
                and all(
                    str(action.get("kind", "")) in {"direct", "split"}
                    for action in concat_input_actions
                )
                and all(
                    _is_indexed_direct_split_plan(
                        dict(action.get("plan", {})),
                        concat_idx=int(concat_idx),
                        consumers=consumers,
                        model_outputs=model_outputs,
                    )
                    and not list(
                        dict(action.get("plan", {})).get(
                            "post_transpose_indices",
                            [],
                        )
                    )
                    for action in quantized_split_actions
                )
            )
            if indexed_quantized_split_family:
                continue
            quantized_add_actions = [
                action
                for action in concat_input_actions
                if str(action.get("kind", "")) == "add"
            ]
            indexed_quantized_add_family = (
                post_quantize_idx is not None
                and len(quantized_add_actions) >= 1
                and all(
                    str(action.get("kind", ""))
                    in {"direct", "unary", "add"}
                    for action in concat_input_actions
                )
                and all(
                    _is_indexed_direct_add_plan(
                        dict(action.get("plan", {})),
                        concat_idx=int(concat_idx),
                        consumers=consumers,
                    )
                    for action in quantized_add_actions
                )
            )
            if indexed_quantized_add_family:
                continue
            indexed_float_simple_family = (
                post_quantize_idx is None
                and any(
                    required_kinds.issubset(action_kinds)
                    and action_kinds.issubset(allowed_kinds)
                    for allowed_kinds, required_kinds
                    in indexed_float_simple_family_contracts
                )
            )
            if indexed_float_simple_family:
                continue
            slice_actions = [
                action
                for action in concat_input_actions
                if str(action.get("kind", "")) == "slice"
            ]
            indexed_slice_family = (
                post_quantize_idx is None
                and len(slice_actions) >= 1
                and all(
                    str(action.get("kind", "")) in {"direct", "slice"}
                    for action in concat_input_actions
                )
                and len(
                    {
                        str(action.get("input_name", ""))
                        for action in slice_actions
                    }
                )
                == len(slice_actions)
                and all(
                    _is_indexed_direct_slice_plan(
                        dict(action.get("plan", {}))
                    )
                    for action in slice_actions
                )
            )
            if indexed_slice_family:
                continue
            split_actions = [
                action
                for action in concat_input_actions
                if str(action.get("kind", "")) == "split"
            ]
            indexed_split_family = (
                post_quantize_idx is None
                and len(split_actions) >= 1
                and all(
                    str(action.get("kind", "")) in {"direct", "split"}
                    for action in concat_input_actions
                )
                and all(
                    _is_indexed_direct_split_plan(
                        dict(action.get("plan", {})),
                        concat_idx=int(concat_idx),
                        consumers=consumers,
                        model_outputs=model_outputs,
                    )
                    for action in split_actions
                )
            )
            if indexed_split_family:
                continue
            add_actions = [
                action
                for action in concat_input_actions
                if str(action.get("kind", "")) == "add"
            ]
            indexed_add_family = (
                post_quantize_idx is None
                and len(add_actions) >= 1
                and all(
                    str(action.get("kind", "")) in {"direct", "add"}
                    for action in concat_input_actions
                )
                and all(
                    _is_indexed_direct_add_plan(
                        dict(action.get("plan", {})),
                        concat_idx=int(concat_idx),
                        consumers=consumers,
                    )
                    for action in add_actions
                )
            )
            if indexed_add_family:
                continue
            nhwc_inputs_ok = True
            nhwc_ref_shape: Optional[List[int]] = None
            for action in concat_input_actions:
                action_kind = str(action.get("kind", ""))
                if action_kind == "direct":
                    projected_input_name = str(action["new_input_name"])
                    input_tensor = model_ir.tensors.get(projected_input_name, None)
                    if input_tensor is None or len(list(input_tensor.shape)) != 4:
                        nhwc_inputs_ok = False
                        break
                    shape = [int(v) for v in list(input_tensor.shape)]
                elif action_kind in {"swish", "leaky", "add", "unary", "pad", "prelu", "dequantize", "softmax", "split", "slice"}:
                    projected_input_name = str(action["input_name"])
                    projected_shape = _project_shape_after_nchw_to_nhwc(projected_input_name)
                    if projected_shape is None:
                        nhwc_inputs_ok = False
                        break
                    shape = [int(v) for v in list(projected_shape)]
                else:
                    nhwc_inputs_ok = False
                    break
                if nhwc_ref_shape is None:
                    nhwc_ref_shape = list(shape)
                else:
                    for dim_idx in [0, 1, 2]:
                        if int(shape[dim_idx]) != int(nhwc_ref_shape[dim_idx]):
                            nhwc_inputs_ok = False
                            break
                if not nhwc_inputs_ok:
                    break
            if not nhwc_inputs_ok or nhwc_ref_shape is None:
                # Some paths keep stale rank-4 metadata even though runtime concat
                # is valid (e.g. resize branches). For strict direct transpose
                # wrappers, the algebraic rewrite is still safe.
                if not all_direct_input_actions:
                    continue

            new_concat_inputs: List[str] = []
            pre_remove_indices: List[int] = []
            applied_split_indices: set[int] = set()
            applied_add_indices: set[int] = set()
            applied_slice_group_keys: set[str] = set()
            for action in concat_input_actions:
                action_kind = str(action.get("kind", ""))
                if action_kind == "direct":
                    new_concat_inputs.append(str(action["new_input_name"]))
                    pre_remove_indices.extend([int(v) for v in list(action.get("remove_indices", []))])
                    continue
                if action_kind == "pad":
                    new_concat_inputs.append(str(action["input_name"]))
                    pre_remove_indices.extend(
                        _apply_pad_nhwc_plan(plan=dict(action["plan"]))
                    )
                    continue
                if action_kind == "swish":
                    new_concat_inputs.append(str(action["input_name"]))
                    pre_remove_indices.extend(
                        _apply_swish_nhwc_plan(plan=dict(action["plan"]))
                    )
                    continue
                if action_kind == "leaky":
                    new_concat_inputs.append(str(action["input_name"]))
                    pre_remove_indices.extend(
                        _apply_leaky_nhwc_plan(plan=dict(action["plan"]))
                    )
                    continue
                if action_kind == "dequantize":
                    new_concat_inputs.append(str(action["input_name"]))
                    pre_remove_indices.extend(
                        _apply_dequant_nhwc_plan(plan=dict(action["plan"]))
                    )
                    continue
                if action_kind == "add":
                    new_concat_inputs.append(str(action["input_name"]))
                    pre_remove_indices.extend(
                        _apply_add_nhwc_plan(
                            plan=dict(action["plan"]),
                            applied_split_indices=applied_split_indices,
                            applied_add_indices=applied_add_indices,
                        )
                    )
                    continue
                if action_kind == "split":
                    new_concat_inputs.append(str(action["input_name"]))
                    split_plan = dict(action["plan"])
                    split_idx = int(split_plan.get("split_idx", -1))
                    if int(split_idx) not in applied_split_indices:
                        pre_remove_indices.extend(
                            _apply_split_nhwc_plan(plan=split_plan)
                        )
                        applied_split_indices.add(int(split_idx))
                    continue
                if action_kind == "slice":
                    new_concat_inputs.append(str(action["input_name"]))
                    slice_plan = dict(action["plan"])
                    slice_group_key = str(slice_plan.get("slice_group_key", ""))
                    if slice_group_key != "" and slice_group_key in applied_slice_group_keys:
                        continue
                    pre_remove_indices.extend(_apply_slice_nhwc_plan(plan=slice_plan))
                    if slice_group_key != "":
                        applied_slice_group_keys.add(slice_group_key)
                    continue
                if action_kind == "unary":
                    new_concat_inputs.append(str(action["input_name"]))
                    pre_remove_indices.extend(
                        _apply_unary_nhwc_plan(plan=dict(action["plan"]))
                    )
                    continue
                if action_kind == "softmax":
                    new_concat_inputs.append(str(action["input_name"]))
                    softmax_remove = _apply_softmax_nhwc_plan(plan=dict(action["plan"]))
                    if softmax_remove is None:
                        rewritable = False
                        break
                    pre_remove_indices.extend([int(v) for v in list(softmax_remove)])
                    continue
                if action_kind == "prelu":
                    new_concat_inputs.append(str(action["input_name"]))
                    prelu_remove = _apply_prelu_nhwc_plan(plan=dict(action["plan"]))
                    if prelu_remove is None:
                        rewritable = False
                        break
                    pre_remove_indices.extend([int(v) for v in list(prelu_remove)])
                    continue
                rewritable = False
                break
            if not rewritable:
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=[str(v) for v in new_concat_inputs],
            )
            concat_op.options["axis"] = 3
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(concat_out_name, None),
                perm_nchw_to_nhwc,
            )

            canonical_post_output_name = str(post_output_names[0])
            if post_quantize_idx is None:
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=concat_op,
                    new_outputs=[canonical_post_output_name],
                )
            else:
                post_quantize_op = model_ir.operators[int(post_quantize_idx)]
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=post_quantize_op,
                    new_outputs=[canonical_post_output_name],
                )
            for alias_post_output_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, alias_post_output_name, canonical_post_output_name)

            old_concat_tensor = model_ir.tensors.get(concat_out_name, None)
            canonical_post_tensor = model_ir.tensors.get(canonical_post_output_name, None)
            if canonical_post_tensor is not None:
                if post_quantize_idx is None:
                    if old_concat_tensor is not None:
                        canonical_post_tensor.dtype = str(old_concat_tensor.dtype)
                        canonical_post_tensor.quantization = _clone_quantization(old_concat_tensor.quantization)
                else:
                    old_q_out_tensor = (
                        model_ir.tensors.get(str(post_quantize_out_name), None)
                        if post_quantize_out_name is not None
                        else None
                    )
                    if old_q_out_tensor is not None:
                        canonical_post_tensor.dtype = str(old_q_out_tensor.dtype)
                        canonical_post_tensor.quantization = _clone_quantization(old_q_out_tensor.quantization)
                    if old_concat_tensor is not None:
                        canonical_post_tensor.shape = [int(v) for v in list(old_concat_tensor.shape)]
                        canonical_post_tensor.shape_signature = (
                            [int(v) for v in list(old_concat_tensor.shape_signature)]
                            if old_concat_tensor.shape_signature is not None
                            else [int(v) for v in list(old_concat_tensor.shape)]
                        )

            remove_op_ids: set[int] = set(int(v) for v in list(post_op_ids))
            for remove_idx in pre_remove_indices:
                if int(remove_idx) < 0 or int(remove_idx) >= len(model_ir.operators):
                    continue
                remove_op_ids.add(int(id(model_ir.operators[int(remove_idx)])))
            remove_indices = sorted(
                [
                    int(op_idx)
                    for op_idx, op in enumerate(model_ir.operators)
                    if int(id(op)) in remove_op_ids
                ],
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_pre_concat_nhwc_chains": int(optimized)}
