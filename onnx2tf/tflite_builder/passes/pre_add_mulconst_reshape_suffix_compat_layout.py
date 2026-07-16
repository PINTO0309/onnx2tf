from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.pre_add_mulconst_reshape_suffix_layout import (
    optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains as _optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_pass,
)


def optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_compat(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW add bridges with mul-const inputs that feed [N,C,HW] reshape + [0,2,1] transpose.

    Target:
      a_nhwc --TRANSPOSE(0,3,1,2)--> a_nchw
      b_nhwc --TRANSPOSE(0,3,1,2)--> b_nchw --MUL(const)--> b_scaled_nchw
      ADD(a_nchw, b_scaled_nchw) -> y_nchw
      y_nchw --RESHAPE([N,C,HW])--> r --TRANSPOSE([0,2,1])--> z
      (optional legacy users consume y_nchw)

    Rewrite:
      ADD(a_nhwc, b_scaled_nhwc) -> y_nhwc
      y_nhwc --RESHAPE([N,HW,C])--> z
      (if legacy users exist, keep one adapter TRANSPOSE(0,3,1,2): y_nhwc -> y_nchw)
    """
    indexed_stats = (
        _optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_pass(
            model_ir,
            graph_index=ModelIRGraphIndex(model_ir),
            layout_state=layout_state,
        )
    )
    rewritten = int(
        indexed_stats.get(
            "optimized_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains",
            0,
        )
    )
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_3d_nchw_to_nhwc = [0, 2, 1]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _maybe_swap_reshape_shape_tensor(
        *,
        reshape_op: OperatorIR,
        reshape_idx: int,
        consumers: Dict[str, List[int]],
    ) -> None:
        if len(reshape_op.inputs) < 2:
            return
        shape_name = str(reshape_op.inputs[1])
        shape_tensor = model_ir.tensors.get(shape_name, None)
        if shape_tensor is None or shape_tensor.data is None:
            return
        shape_data = np.asarray(shape_tensor.data)
        if int(shape_data.size) != 3:
            return
        flat = shape_data.reshape(-1).copy()
        flat[[1, 2]] = flat[[2, 1]]
        swapped = flat.reshape(shape_data.shape).astype(shape_data.dtype, copy=False)

        shape_consumers = [int(v) for v in consumers.get(shape_name, [])]
        if any(int(v) != int(reshape_idx) for v in shape_consumers):
            cloned_name = _unique_tensor_name(f"{shape_name}_nhwc")
            model_ir.tensors[cloned_name] = TensorIR(
                name=cloned_name,
                dtype=str(shape_tensor.dtype),
                shape=[int(v) for v in list(swapped.shape)],
                shape_signature=[int(v) for v in list(swapped.shape)],
                data=np.asarray(swapped),
                is_variable=False,
                quantization=_clone_quantization(shape_tensor.quantization),
            )
            reshape_inputs = [str(v) for v in list(reshape_op.inputs)]
            reshape_inputs[1] = str(cloned_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=reshape_op,
                new_inputs=reshape_inputs,
            )
        else:
            shape_tensor.data = np.asarray(swapped)
            shape_tensor.shape = [int(v) for v in list(swapped.shape)]
            shape_tensor.shape_signature = [int(v) for v in list(swapped.shape)]

    def _maybe_swap_reshape_option_shape(reshape_op: OperatorIR) -> None:
        if not isinstance(reshape_op.options, dict):
            return
        opts = dict(reshape_op.options)
        changed = False
        for key in ["newShape", "onnxRawNewShape"]:
            value = opts.get(key, None)
            if not isinstance(value, list) or len(value) != 3:
                continue
            swapped = [value[0], value[2], value[1]]
            if swapped != value:
                opts[key] = [int(v) for v in swapped]
                changed = True
        if changed:
            reshape_op.options = opts

    def _analyze_input_plan(
        *,
        input_name: str,
        add_idx: int,
        producers: Dict[str, int],
        consumers: Dict[str, List[int]],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        pre_idx = producers.get(str(input_name), None)
        if pre_idx is not None:
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) == "TRANSPOSE"
                and len(pre_op.inputs) >= 2
                and len(pre_op.outputs) == 1
                and str(pre_op.outputs[0]) == str(input_name)
                and _read_transpose_perm(model_ir, pre_op) == perm_nhwc_to_nchw
                and str(input_name) not in model_outputs
            ):
                users = [int(v) for v in consumers.get(str(input_name), [])]
                if int(add_idx) in users:
                    return {
                        "nhwc_input_name": str(pre_op.inputs[0]),
                        "pre_remove_indices": [int(pre_idx)]
                        if set(users) == {int(add_idx)}
                        else [],
                        "mul_plan": None,
                    }

        mul_idx = producers.get(str(input_name), None)
        if mul_idx is None:
            return None
        mul_op = model_ir.operators[int(mul_idx)]
        if (
            str(mul_op.op_type) != "MUL"
            or len(mul_op.inputs) != 2
            or len(mul_op.outputs) != 1
        ):
            return None
        if str(mul_op.outputs[0]) != str(input_name):
            return None
        if str(input_name) in model_outputs:
            return None
        input_users = [int(v) for v in consumers.get(str(input_name), [])]
        if int(add_idx) not in input_users:
            return None

        data_input_index: Optional[int] = None
        side_input_index: Optional[int] = None
        side_input_name: Optional[str] = None
        data_pre_idx: Optional[int] = None
        data_pre_input_name: Optional[str] = None

        for cand_data_idx, cand_side_idx in [(0, 1), (1, 0)]:
            data_name = str(mul_op.inputs[int(cand_data_idx)])
            side_name = str(mul_op.inputs[int(cand_side_idx)])
            side_tensor = model_ir.tensors.get(side_name, None)
            if side_tensor is None or side_tensor.data is None:
                continue
            side_data = np.asarray(side_tensor.data)
            if int(side_data.size) != 1 and side_data.ndim not in {3, 4}:
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
            if set(int(v) for v in consumers.get(data_name, [])) != {int(mul_idx)}:
                continue

            data_tensor = model_ir.tensors.get(data_name, None)
            pre_input_name = str(data_prod_op.inputs[0])
            pre_input_tensor = model_ir.tensors.get(pre_input_name, None)
            if data_tensor is None or pre_input_tensor is None:
                continue
            if (
                not _is_fully_known_positive_shape(list(data_tensor.shape))
                or not _is_fully_known_positive_shape(list(pre_input_tensor.shape))
                or len(list(data_tensor.shape)) != 4
                or len(list(pre_input_tensor.shape)) != 4
            ):
                continue

            if int(side_data.size) != 1:
                if (
                    _broadcast_static_shapes(
                        list(data_tensor.shape), [int(v) for v in list(side_data.shape)]
                    )
                    is None
                ):
                    continue

            data_input_index = int(cand_data_idx)
            side_input_index = int(cand_side_idx)
            side_input_name = str(side_name)
            data_pre_idx = int(data_prod_idx)
            data_pre_input_name = str(pre_input_name)
            break

        if (
            data_input_index is None
            or side_input_index is None
            or side_input_name is None
            or data_pre_idx is None
            or data_pre_input_name is None
        ):
            return None

        side_tensor = model_ir.tensors.get(side_input_name, None)
        if side_tensor is None or side_tensor.data is None:
            return None
        side_data = np.asarray(side_tensor.data)
        nhwc_side_data: Optional[np.ndarray] = None
        side_needs_update = False
        if int(side_data.size) != 1:
            target_shape = [
                int(v) for v in list(model_ir.tensors[str(data_pre_input_name)].shape)
            ]
            side_shape = [int(v) for v in list(side_data.shape)]
            is_nchw_channelwise = (
                int(side_data.ndim) == 4
                and len(side_shape) == 4
                and int(side_shape[0]) == 1
                and int(side_shape[1]) > 0
                and int(side_shape[2]) == 1
                and int(side_shape[3]) == 1
            )
            if is_nchw_channelwise:
                rotated = np.transpose(side_data, axes=perm_nchw_to_nhwc).astype(
                    side_data.dtype, copy=False
                )
                rotated_shape = [int(v) for v in list(rotated.shape)]
                if _broadcast_static_shapes(target_shape, rotated_shape) is None:
                    return None
                nhwc_side_data = np.asarray(rotated)
                side_needs_update = not np.array_equal(nhwc_side_data, side_data)
            elif _broadcast_static_shapes(target_shape, side_shape) is not None:
                nhwc_side_data = np.asarray(side_data)
            else:
                rotated = np.asarray(side_data)
                found = False
                transpose_perm = (
                    perm_nchw_to_nhwc if int(rotated.ndim) == 4 else [1, 2, 0]
                )
                max_rotate = 1 if int(rotated.ndim) == 4 else 3
                for _ in range(int(max_rotate)):
                    rotated = np.transpose(rotated, transpose_perm).astype(
                        side_data.dtype, copy=False
                    )
                    rotated_shape = [int(v) for v in list(rotated.shape)]
                    if (
                        _broadcast_static_shapes(target_shape, rotated_shape)
                        is not None
                    ):
                        nhwc_side_data = np.asarray(rotated)
                        side_needs_update = True
                        found = True
                        break
                if not found:
                    return None

        side_users = [int(v) for v in consumers.get(str(side_input_name), [])]
        side_shared_outside_mul = any(int(v) != int(mul_idx) for v in side_users)

        return {
            "nhwc_input_name": str(input_name),
            "pre_remove_indices": [],
            "mul_plan": {
                "mul_idx": int(mul_idx),
                "mul_data_input_index": int(data_input_index),
                "mul_side_input_index": int(side_input_index),
                "mul_side_input_name": str(side_input_name),
                "mul_out_name": str(input_name),
                "data_pre_idx": int(data_pre_idx),
                "data_pre_input_name": str(data_pre_input_name),
                "side_needs_update": bool(side_needs_update),
                "side_shared_outside_mul": bool(side_shared_outside_mul),
                "side_nhwc_data": (
                    None if nhwc_side_data is None else np.asarray(nhwc_side_data)
                ),
            },
        }

    def _apply_mul_plan(*, plan: Dict[str, Any]) -> List[int]:
        mul_idx = int(plan["mul_idx"])
        mul_data_input_index = int(plan["mul_data_input_index"])
        mul_side_input_index = int(plan["mul_side_input_index"])
        mul_side_input_name = str(plan["mul_side_input_name"])
        mul_out_name = str(plan["mul_out_name"])
        data_pre_idx = int(plan["data_pre_idx"])
        data_pre_input_name = str(plan["data_pre_input_name"])

        side_input_name_for_mul = str(mul_side_input_name)
        if bool(plan.get("side_needs_update", False)):
            nhwc_data = np.asarray(plan.get("side_nhwc_data"))
            side_tensor = model_ir.tensors.get(str(mul_side_input_name), None)
            if side_tensor is None:
                return []
            if bool(plan.get("side_shared_outside_mul", False)):
                side_input_name_for_mul = _unique_tensor_name(
                    f"{mul_side_input_name}_nhwc"
                )
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
        if int(mul_data_input_index) < 0 or int(mul_data_input_index) >= len(
            mul_inputs
        ):
            return []
        if int(mul_side_input_index) < 0 or int(mul_side_input_index) >= len(
            mul_inputs
        ):
            return []
        mul_inputs[int(mul_data_input_index)] = str(data_pre_input_name)
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
        return [int(data_pre_idx)]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for add_idx, add_op in enumerate(model_ir.operators):
            if (
                str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
            ):
                continue

            add_out_name = str(add_op.outputs[0])
            if add_out_name in model_outputs:
                continue

            add_users = [
                int(v)
                for v in consumers.get(add_out_name, [])
                if int(v) != int(add_idx)
            ]
            if len(add_users) == 0:
                continue

            suffix_reshape_idx: Optional[int] = None
            suffix_transpose_idx: Optional[int] = None
            suffix_output_name: Optional[str] = None
            legacy_users: List[int] = []
            valid_users = True
            for user_idx in add_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "RESHAPE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == add_out_name
                ):
                    reshape_out_name = str(user_op.outputs[0])
                    reshape_users = [
                        int(v)
                        for v in consumers.get(reshape_out_name, [])
                        if int(v) != int(user_idx)
                    ]
                    if len(reshape_users) != 1:
                        legacy_users.append(int(user_idx))
                        continue
                    post_idx = int(reshape_users[0])
                    post_op = model_ir.operators[int(post_idx)]
                    if (
                        str(post_op.op_type) != "TRANSPOSE"
                        or len(post_op.inputs) < 2
                        or len(post_op.outputs) != 1
                        or str(post_op.inputs[0]) != reshape_out_name
                        or _read_transpose_perm(model_ir, post_op)
                        != perm_3d_nchw_to_nhwc
                    ):
                        legacy_users.append(int(user_idx))
                        continue
                    post_out_name = str(post_op.outputs[0])
                    if post_out_name in model_outputs:
                        valid_users = False
                        break
                    if suffix_reshape_idx is not None:
                        valid_users = False
                        break
                    suffix_reshape_idx = int(user_idx)
                    suffix_transpose_idx = int(post_idx)
                    suffix_output_name = str(post_out_name)
                else:
                    legacy_users.append(int(user_idx))
            if (
                not valid_users
                or suffix_reshape_idx is None
                or suffix_transpose_idx is None
                or suffix_output_name is None
            ):
                continue

            input_plans: List[Dict[str, Any]] = []
            rewritable = True
            for input_name in [str(v) for v in list(add_op.inputs)]:
                plan = _analyze_input_plan(
                    input_name=input_name,
                    add_idx=int(add_idx),
                    producers=producers,
                    consumers=consumers,
                    model_outputs=model_outputs,
                )
                if plan is None:
                    rewritable = False
                    break
                input_plans.append(dict(plan))
            if not rewritable:
                continue

            pre_remove_indices: List[int] = []
            for plan in input_plans:
                mul_plan = plan.get("mul_plan", None)
                if mul_plan is not None:
                    pre_remove_indices.extend(_apply_mul_plan(plan=dict(mul_plan)))
                pre_remove_indices.extend(
                    [int(v) for v in list(plan.get("pre_remove_indices", []))]
                )

            add_nhwc_name = (
                _unique_tensor_name(f"{add_out_name}_nhwc")
                if len(legacy_users) > 0
                else str(add_out_name)
            )
            new_add_inputs = [str(plan["nhwc_input_name"]) for plan in input_plans]
            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=new_add_inputs,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=add_op,
                new_outputs=[add_nhwc_name],
            )

            old_add_tensor = model_ir.tensors.get(add_out_name, None)
            if add_nhwc_name != add_out_name:
                if old_add_tensor is not None and add_nhwc_name not in model_ir.tensors:
                    model_ir.tensors[add_nhwc_name] = TensorIR(
                        name=add_nhwc_name,
                        dtype=str(old_add_tensor.dtype),
                        shape=[int(v) for v in list(old_add_tensor.shape)],
                        shape_signature=(
                            [int(v) for v in list(old_add_tensor.shape_signature)]
                            if old_add_tensor.shape_signature is not None
                            else [int(v) for v in list(old_add_tensor.shape)]
                        ),
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(old_add_tensor.quantization),
                    )

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(add_nhwc_name, None),
                perm_nchw_to_nhwc,
            )

            reshape_op = model_ir.operators[int(suffix_reshape_idx)]
            reshape_inputs = [str(v) for v in list(reshape_op.inputs)]
            reshape_inputs[0] = add_nhwc_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=reshape_op,
                new_inputs=reshape_inputs,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=reshape_op,
                new_outputs=[str(suffix_output_name)],
            )
            _maybe_swap_reshape_shape_tensor(
                reshape_op=reshape_op,
                reshape_idx=int(suffix_reshape_idx),
                consumers=consumers,
            )
            _maybe_swap_reshape_option_shape(reshape_op)

            remove_indices = [int(suffix_transpose_idx)]
            remove_indices.extend([int(v) for v in pre_remove_indices])
            remove_indices = sorted(list(set(remove_indices)), reverse=True)
            removed_before_add = sum(
                1 for idx in remove_indices if int(idx) < int(add_idx)
            )
            new_add_idx = int(add_idx) - int(removed_before_add)
            for remove_idx in remove_indices:
                if int(remove_idx) == int(add_idx):
                    continue
                del model_ir.operators[int(remove_idx)]

            if len(legacy_users) > 0:
                adapter_perm_name = _unique_tensor_name(f"{add_out_name}_adapter_perm")
                model_ir.tensors[adapter_perm_name] = TensorIR(
                    name=adapter_perm_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                    is_variable=False,
                    quantization=None,
                )
                adapter_op = OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[add_nhwc_name, adapter_perm_name],
                    outputs=[add_out_name],
                    options={},
                )
                model_ir.operators.insert(int(new_add_idx) + 1, adapter_op)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "optimized_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains": int(
            rewritten
        )
    }
