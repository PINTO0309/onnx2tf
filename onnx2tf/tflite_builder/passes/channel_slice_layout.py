from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _permute_shape,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_boundary_input_transpose_channel_slice_blocks(model_ir: ModelIR) -> Dict[str, int]:
    """
    Remove shared boundary input TRANSPOSE while preserving downstream NCHW semantics.

    Target:
      input(NHWC) --TRANSPOSE(0,3,1,2)--> input_onnx_ncx_internal
        |- SLICE(channel axis=1)
        |- ...
        `- other consumers

    Rewrite:
    - For eligible channel SLICE consumers, rewrite slice source to NHWC input and
      move NHWC->NCHW transpose after each slice output.
    - Localize remaining NCHW requirements by inserting per-consumer transposes.
    - Remove shared boundary TRANSPOSE so identifier=0 is no longer TRANSPOSE.
    """
    removed_boundary = 0
    rewritten_channel_slices = 0
    inserted_local_transposes = 0
    rewritten_axis_ops = 0

    layout_passthrough_unary = {
        "CAST",
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
        "LOGISTIC",
        "TANH",
        "NEG",
        "EXP",
        "ABS",
        "SQRT",
    }
    layout_passthrough_binary = {
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "MAXIMUM",
        "MINIMUM",
    }

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _is_channel_slice_axis1(
        *,
        op: OperatorIR,
        source_shape_nchw: Optional[List[int]] = None,
    ) -> bool:
        if str(op.op_type) != "SLICE" or len(op.inputs) < 3 or len(op.outputs) != 1:
            return False
        begin_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[1]), None))
        size_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[2]), None))
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        if int(size_vals[1]) != 1:
            return False
        if int(begin_vals[2]) != 0 or int(begin_vals[3]) != 0:
            return False
        if source_shape_nchw is not None and len(source_shape_nchw) == 4:
            if int(begin_vals[1]) < 0 or int(begin_vals[1]) >= int(source_shape_nchw[1]):
                return False
            h = int(source_shape_nchw[2])
            w = int(source_shape_nchw[3])
            if int(size_vals[2]) not in {h, -1}:
                return False
            if int(size_vals[3]) not in {w, -1}:
                return False
        else:
            if int(begin_vals[1]) < 0:
                return False
        return True

    def _rewrite_channel_slice_axis1_to_axis3(
        *,
        op: OperatorIR,
    ) -> bool:
        begin_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        size_tensor = model_ir.tensors.get(str(op.inputs[2]), None)
        begin_vals = _read_const_ints_from_tensor(begin_tensor)
        size_vals = _read_const_ints_from_tensor(size_tensor)
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        new_begin = [int(begin_vals[0]), int(begin_vals[2]), int(begin_vals[3]), int(begin_vals[1])]
        new_size = [int(size_vals[0]), int(size_vals[2]), int(size_vals[3]), int(size_vals[1])]
        _write_const_ints_to_tensor(begin_tensor, new_begin)
        _write_const_ints_to_tensor(size_tensor, new_size)
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(str(v) for v in model_ir.inputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            input_name = str(pre_op.inputs[0])
            internal_name = str(pre_op.outputs[0])
            if input_name not in model_inputs:
                continue
            if not str(internal_name).endswith("_onnx_ncx_internal"):
                continue
            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre != [0, 3, 1, 2]:
                continue

            input_tensor = model_ir.tensors.get(input_name, None)
            internal_tensor = model_ir.tensors.get(internal_name, None)
            if input_tensor is None or internal_tensor is None:
                continue
            if len(list(internal_tensor.shape)) != 4:
                continue

            internal_users = [int(v) for v in consumers.get(internal_name, [])]
            if len(internal_users) == 0:
                continue

            # 1) Rewrite direct channel-slice consumers to NHWC source.
            converted_ops: set = set()
            nhwc_tensors: set = set()
            rewritten_any_slice = False
            for user_idx in sorted(internal_users):
                user_op = model_ir.operators[int(user_idx)]
                if not _is_channel_slice_axis1(
                    op=user_op,
                    source_shape_nchw=[int(v) for v in list(internal_tensor.shape)],
                ):
                    continue
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=user_op,
                    new_inputs=[input_name, str(user_op.inputs[1]), str(user_op.inputs[2])],
                )
                if _rewrite_channel_slice_axis1_to_axis3(op=user_op):
                    rewritten_channel_slices += 1
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(user_op.outputs[0]), None),
                    [0, 2, 3, 1],
                )
                converted_ops.add(int(user_idx))
                nhwc_tensors.add(str(user_op.outputs[0]))
                rewritten_any_slice = True

            if not rewritten_any_slice:
                continue

            # 2) Propagate NHWC through local axis-sensitive block.
            while True:
                propagated = False
                for op_idx, op in enumerate(model_ir.operators):
                    if int(op_idx) in converted_ops:
                        continue
                    if len(op.outputs) != 1:
                        continue
                    out_name = str(op.outputs[0])
                    op_type = str(op.op_type)

                    if op_type == "CONCATENATION":
                        axis = int(op.options.get("axis", -1))
                        if axis == 1 and len(op.inputs) > 0 and all(str(inp) in nhwc_tensors for inp in op.inputs):
                            op.options["axis"] = 3
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                [0, 2, 3, 1],
                            )
                            rewritten_axis_ops += 1
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                    if op_type == "SLICE" and len(op.inputs) >= 3:
                        in_name = str(op.inputs[0])
                        if in_name in nhwc_tensors and _is_channel_slice_axis1(op=op, source_shape_nchw=None):
                            if _rewrite_channel_slice_axis1_to_axis3(op=op):
                                rewritten_axis_ops += 1
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                [0, 2, 3, 1],
                            )
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                    if op_type in layout_passthrough_unary and len(op.inputs) >= 1:
                        if str(op.inputs[0]) in nhwc_tensors:
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                [0, 2, 3, 1],
                            )
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                    if op_type in layout_passthrough_binary and len(op.inputs) >= 2:
                        dynamic_inputs: List[str] = []
                        for inp in list(op.inputs[:2]):
                            inp_name = str(inp)
                            inp_tensor = model_ir.tensors.get(inp_name, None)
                            if inp_tensor is not None and inp_tensor.data is not None:
                                continue
                            dynamic_inputs.append(inp_name)
                        if len(dynamic_inputs) > 0 and all(inp_name in nhwc_tensors for inp_name in dynamic_inputs):
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                [0, 2, 3, 1],
                            )
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                    if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"} and len(op.inputs) >= 1:
                        # TFLite CONV_2D/DEPTHWISE_CONV_2D always produce NHWC outputs.
                        if str(op.inputs[0]) in nhwc_tensors:
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                [0, 2, 3, 1],
                            )
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                if not propagated:
                    break

            # 3) Bridge NHWC tensors back to NCHW for consumers outside converted block.
            consumers_nhwc = _build_tensor_consumer_map(model_ir)
            bridge_plans: List[Tuple[int, str, str, List[int]]] = []
            for tensor_name in sorted(list(nhwc_tensors)):
                users = [
                    int(v)
                    for v in consumers_nhwc.get(str(tensor_name), [])
                    if int(v) not in converted_ops
                ]
                if len(users) == 0:
                    continue
                tensor = model_ir.tensors.get(str(tensor_name), None)
                if tensor is None or len(list(tensor.shape)) != 4:
                    continue
                bridge_name = _unique_tensor_name(f"{tensor_name}_nchw_bridge")
                bridge_shape = _permute_shape(list(tensor.shape), [0, 3, 1, 2])
                signature_src = (
                    list(tensor.shape_signature)
                    if tensor.shape_signature is not None
                    else list(tensor.shape)
                )
                bridge_signature = _permute_shape(signature_src, [0, 3, 1, 2])
                if bridge_shape is None or bridge_signature is None:
                    continue
                model_ir.tensors[bridge_name] = TensorIR(
                    name=bridge_name,
                    dtype=str(tensor.dtype),
                    shape=[int(v) for v in list(bridge_shape)],
                    shape_signature=[int(v) for v in list(bridge_signature)],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(tensor.quantization),
                )
                for user_idx in users:
                    user_op = model_ir.operators[int(user_idx)]
                    updated_inputs = [
                        bridge_name if str(inp) == str(tensor_name) else str(inp)
                        for inp in list(user_op.inputs)
                    ]
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=user_op,
                        new_inputs=updated_inputs,
                    )
                bridge_plans.append((int(min(users)), str(tensor_name), str(bridge_name), [int(v) for v in users]))

            inserted = 0
            for insert_idx, source_name, bridge_name, _users in sorted(bridge_plans, key=lambda v: int(v[0])):
                model_ir.operators.insert(
                    int(insert_idx + inserted),
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(source_name), str(pre_op.inputs[1])],
                        outputs=[str(bridge_name)],
                    ),
                )
                inserted += 1
                inserted_local_transposes += 1

            # 4) Localize any remaining NCHW uses for this boundary tensor.
            consumers_after = _build_tensor_consumer_map(model_ir)
            remaining_users = [int(v) for v in consumers_after.get(internal_name, [])]
            inserted = 0
            for user_idx in sorted(remaining_users):
                user_op = model_ir.operators[int(user_idx + inserted)]
                local_name = _unique_tensor_name(f"{internal_name}_local")
                model_ir.tensors[local_name] = TensorIR(
                    name=local_name,
                    dtype=str(internal_tensor.dtype),
                    shape=[int(v) for v in list(internal_tensor.shape)],
                    shape_signature=(
                        [int(v) for v in list(internal_tensor.shape_signature)]
                        if internal_tensor.shape_signature is not None
                        else [int(v) for v in list(internal_tensor.shape)]
                    ),
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(internal_tensor.quantization),
                )
                model_ir.operators.insert(
                    int(user_idx + inserted),
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[input_name, str(pre_op.inputs[1])],
                        outputs=[local_name],
                    ),
                )
                inserted += 1
                updated_inputs = [
                    local_name if str(inp) == internal_name else str(inp)
                    for inp in list(user_op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=user_op,
                    new_inputs=updated_inputs,
                )
                inserted_local_transposes += 1

            # 5) Remove shared boundary transpose.
            del model_ir.operators[int(pre_idx)]
            removed_boundary += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_boundary_input_transpose": int(removed_boundary),
        "rewritten_boundary_channel_slices": int(rewritten_channel_slices),
        "rewritten_boundary_axis_ops": int(rewritten_axis_ops),
        "inserted_local_boundary_transposes": int(inserted_local_transposes),
    }


def _optimize_internal_transpose_channel_slice_nhwc_propagation_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove internal NHWC->NCHW transpose stems that only feed channel-slice branches.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
        |- SLICE(axis=1) ...
        `- SLICE(axis=1) ...

    Rewrite:
    - Rewrite slice sources/params to NHWC (axis=3 equivalent).
    - Propagate NHWC through local unary/binary/conv/pool block.
    - Bypass local TRANSPOSE wrappers fed by already-NHWC tensors.
    - Reinsert NHWC->NCHW bridge only for remaining legacy consumers.
    """
    removed_internal = 0
    rewritten_channel_slices = 0
    rewritten_axis_ops = 0
    inserted_local_transposes = 0

    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    layout_passthrough_unary = {
        "CAST",
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
        "LOGISTIC",
        "TANH",
        "LEAKY_RELU",
        "NEG",
        "EXP",
        "ABS",
        "SQRT",
    }
    layout_passthrough_binary = {
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "MAXIMUM",
        "MINIMUM",
    }

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _is_channel_slice_axis1(
        *,
        op: OperatorIR,
        source_shape_nchw: Optional[List[int]] = None,
    ) -> bool:
        if str(op.op_type) != "SLICE" or len(op.inputs) < 3 or len(op.outputs) != 1:
            return False
        begin_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[1]), None))
        size_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[2]), None))
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        if int(size_vals[1]) <= 0:
            return False
        if int(begin_vals[2]) != 0 or int(begin_vals[3]) != 0:
            return False
        if source_shape_nchw is not None and len(source_shape_nchw) == 4:
            if int(begin_vals[1]) < 0:
                return False
            c = int(source_shape_nchw[1])
            if int(begin_vals[1]) + int(size_vals[1]) > c:
                return False
            h = int(source_shape_nchw[2])
            w = int(source_shape_nchw[3])
            if int(size_vals[2]) not in {h, -1}:
                return False
            if int(size_vals[3]) not in {w, -1}:
                return False
        return True

    def _rewrite_channel_slice_axis1_to_axis3(
        *,
        op: OperatorIR,
    ) -> bool:
        begin_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        size_tensor = model_ir.tensors.get(str(op.inputs[2]), None)
        begin_vals = _read_const_ints_from_tensor(begin_tensor)
        size_vals = _read_const_ints_from_tensor(size_tensor)
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        new_begin = [int(begin_vals[0]), int(begin_vals[2]), int(begin_vals[3]), int(begin_vals[1])]
        new_size = [int(size_vals[0]), int(size_vals[2]), int(size_vals[3]), int(size_vals[1])]
        _write_const_ints_to_tensor(begin_tensor, new_begin)
        _write_const_ints_to_tensor(size_tensor, new_size)
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            input_name = str(pre_op.inputs[0])
            internal_name = str(pre_op.outputs[0])
            if input_name in model_outputs or internal_name in model_outputs:
                continue

            input_tensor = model_ir.tensors.get(input_name, None)
            internal_tensor = model_ir.tensors.get(internal_name, None)
            if input_tensor is None or internal_tensor is None:
                continue
            if len(list(input_tensor.shape)) != 4 or len(list(internal_tensor.shape)) != 4:
                continue

            internal_users = [int(v) for v in consumers.get(internal_name, [])]
            if len(internal_users) == 0:
                continue
            if not all(
                _is_channel_slice_axis1(
                    op=model_ir.operators[int(user_idx)],
                    source_shape_nchw=[int(v) for v in list(internal_tensor.shape)],
                )
                for user_idx in internal_users
            ):
                continue

            converted_ops: set = set()
            nhwc_tensors: set = set()
            removable_transposes: set = set()

            for user_idx in sorted(internal_users):
                user_op = model_ir.operators[int(user_idx)]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=user_op,
                    new_inputs=[input_name, str(user_op.inputs[1]), str(user_op.inputs[2])],
                )
                if _rewrite_channel_slice_axis1_to_axis3(op=user_op):
                    rewritten_channel_slices += 1
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(user_op.outputs[0]), None),
                    perm_nchw_to_nhwc,
                )
                converted_ops.add(int(user_idx))
                nhwc_tensors.add(str(user_op.outputs[0]))

            while True:
                propagated = False
                for op_idx, op in enumerate(model_ir.operators):
                    if int(op_idx) in converted_ops:
                        continue
                    if len(op.outputs) != 1:
                        continue
                    op_type = str(op.op_type)
                    out_name = str(op.outputs[0])
                    if out_name in model_outputs:
                        continue

                    if op_type == "CONCATENATION":
                        axis = int(op.options.get("axis", -1))
                        if axis == 1 and len(op.inputs) > 0 and all(str(inp) in nhwc_tensors for inp in op.inputs):
                            op.options["axis"] = 3
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                perm_nchw_to_nhwc,
                            )
                            rewritten_axis_ops += 1
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                    if op_type == "SLICE" and len(op.inputs) >= 3:
                        in_name = str(op.inputs[0])
                        if in_name in nhwc_tensors and _is_channel_slice_axis1(op=op, source_shape_nchw=None):
                            if _rewrite_channel_slice_axis1_to_axis3(op=op):
                                rewritten_axis_ops += 1
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                perm_nchw_to_nhwc,
                            )
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                    if op_type in layout_passthrough_unary and len(op.inputs) >= 1:
                        if str(op.inputs[0]) in nhwc_tensors:
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                perm_nchw_to_nhwc,
                            )
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                    if op_type in layout_passthrough_binary and len(op.inputs) >= 2:
                        input_names = [str(v) for v in list(op.inputs)]
                        dynamic_inputs: List[str] = []
                        for inp_name in input_names[:2]:
                            inp_tensor = model_ir.tensors.get(inp_name, None)
                            if inp_tensor is not None and inp_tensor.data is not None:
                                continue
                            dynamic_inputs.append(inp_name)

                        if len(dynamic_inputs) == 0 or not all(inp_name in nhwc_tensors for inp_name in dynamic_inputs):
                            continue

                        target_nhwc_shape: Optional[List[int]] = None
                        for dyn_name in dynamic_inputs:
                            dyn_tensor = model_ir.tensors.get(str(dyn_name), None)
                            if (
                                dyn_tensor is not None
                                and _is_fully_known_positive_shape(list(dyn_tensor.shape))
                                and len(list(dyn_tensor.shape)) == 4
                            ):
                                target_nhwc_shape = [int(v) for v in list(dyn_tensor.shape)]
                                break

                        # Rewrite NCHW channelwise constants to NHWC when needed.
                        safe_binary = True
                        new_inputs = [str(v) for v in input_names]
                        for input_index, const_name in enumerate(input_names[:2]):
                            const_tensor = model_ir.tensors.get(str(const_name), None)
                            if const_tensor is None or const_tensor.data is None:
                                continue
                            const_data = np.asarray(const_tensor.data)
                            if int(const_data.size) == 1:
                                continue

                            if target_nhwc_shape is None:
                                safe_binary = False
                                break

                            const_shape = [int(v) for v in list(const_data.shape)]
                            if _broadcast_static_shapes(target_nhwc_shape, const_shape) is not None:
                                continue

                            rotated: Optional[np.ndarray] = None
                            if int(const_data.ndim) == 4:
                                candidate = np.transpose(const_data, perm_nchw_to_nhwc).astype(const_data.dtype, copy=False)
                                if _broadcast_static_shapes(
                                    target_nhwc_shape,
                                    [int(v) for v in list(candidate.shape)],
                                ) is not None:
                                    rotated = np.asarray(candidate)
                            elif int(const_data.ndim) == 3:
                                candidate = np.transpose(const_data, [1, 2, 0]).astype(const_data.dtype, copy=False)
                                if _broadcast_static_shapes(
                                    target_nhwc_shape,
                                    [int(v) for v in list(candidate.shape)],
                                ) is not None:
                                    rotated = np.asarray(candidate)

                            if rotated is None:
                                safe_binary = False
                                break

                            const_users = [int(v) for v in consumers.get(str(const_name), [])]
                            if any(int(v) != int(op_idx) for v in const_users):
                                rotated_name = _unique_tensor_name(f"{const_name}_nhwc")
                                model_ir.tensors[rotated_name] = TensorIR(
                                    name=rotated_name,
                                    dtype=str(const_tensor.dtype),
                                    shape=[int(v) for v in list(rotated.shape)],
                                    shape_signature=[int(v) for v in list(rotated.shape)],
                                    data=np.asarray(rotated),
                                    is_variable=False,
                                    quantization=_clone_quantization(const_tensor.quantization),
                                )
                                new_inputs[int(input_index)] = str(rotated_name)
                            else:
                                const_tensor.data = np.asarray(rotated)
                                const_tensor.shape = [int(v) for v in list(rotated.shape)]
                                const_tensor.shape_signature = [int(v) for v in list(rotated.shape)]

                        if not safe_binary:
                            continue

                        if new_inputs != input_names:
                            _set_operator_inputs(
                                model_ir=model_ir,
                                op=op,
                                new_inputs=new_inputs,
                            )

                        _permute_tensor_metadata_if_rank_matches(
                            model_ir.tensors.get(out_name, None),
                            perm_nchw_to_nhwc,
                        )
                        nhwc_tensors.add(out_name)
                        converted_ops.add(int(op_idx))
                        propagated = True
                        continue

                    if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "MAX_POOL_2D"} and len(op.inputs) >= 1:
                        if str(op.inputs[0]) in nhwc_tensors:
                            _permute_tensor_metadata_if_rank_matches(
                                model_ir.tensors.get(out_name, None),
                                perm_nchw_to_nhwc,
                            )
                            nhwc_tensors.add(out_name)
                            converted_ops.add(int(op_idx))
                            propagated = True
                        continue

                if not propagated:
                    break

            consumers_nhwc = _build_tensor_consumer_map(model_ir)
            bridge_plans: List[Tuple[int, str, str, List[int]]] = []
            for tensor_name in sorted(list(nhwc_tensors)):
                users = [
                    int(v)
                    for v in consumers_nhwc.get(str(tensor_name), [])
                    if int(v) not in converted_ops
                ]
                if len(users) == 0:
                    continue
                tensor = model_ir.tensors.get(str(tensor_name), None)
                if tensor is None or len(list(tensor.shape)) != 4:
                    continue
                bridge_name = _unique_tensor_name(f"{tensor_name}_nchw_bridge")
                bridge_shape = _permute_shape(list(tensor.shape), perm_nhwc_to_nchw)
                signature_src = (
                    list(tensor.shape_signature)
                    if tensor.shape_signature is not None
                    else list(tensor.shape)
                )
                bridge_signature = _permute_shape(signature_src, perm_nhwc_to_nchw)
                if bridge_shape is None or bridge_signature is None:
                    continue
                model_ir.tensors[bridge_name] = TensorIR(
                    name=bridge_name,
                    dtype=str(tensor.dtype),
                    shape=[int(v) for v in list(bridge_shape)],
                    shape_signature=[int(v) for v in list(bridge_signature)],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(tensor.quantization),
                )
                for user_idx in users:
                    user_op = model_ir.operators[int(user_idx)]
                    updated_inputs = [
                        bridge_name if str(inp) == str(tensor_name) else str(inp)
                        for inp in list(user_op.inputs)
                    ]
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=user_op,
                        new_inputs=updated_inputs,
                    )
                bridge_plans.append((int(min(users)), str(tensor_name), str(bridge_name), [int(v) for v in users]))

            inserted = 0
            for insert_idx, source_name, bridge_name, _users in sorted(bridge_plans, key=lambda v: int(v[0])):
                model_ir.operators.insert(
                    int(insert_idx + inserted),
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(source_name), str(pre_op.inputs[1])],
                        outputs=[str(bridge_name)],
                    ),
                )
                inserted += 1
                inserted_local_transposes += 1

            remove_indices = sorted(
                list({int(pre_idx)} | {int(v) for v in removable_transposes}),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]
            removed_internal += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_internal_transpose_channel_slice_stems": int(removed_internal),
        "rewritten_internal_channel_slices": int(rewritten_channel_slices),
        "rewritten_internal_axis_ops": int(rewritten_axis_ops),
        "inserted_internal_local_transposes": int(inserted_local_transposes),
    }


def _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove NHWC->NCHW transpose stems whose channel slices immediately bridge back to NHWC.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --SLICE(axis=1)--> s_i_nchw
      s_i_nchw --TRANSPOSE(0,2,3,1)--> s_i_nhwc
      or
      s_i_nchw --MUL(const)--> m_i_nchw --TRANSPOSE(0,2,3,1)--> m_i_nhwc

    Rewrite:
      x_nhwc --SLICE(axis=3)--> s_i_nhwc
      keep downstream NHWC consumers and remove bridge transposes.
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _is_channel_slice_axis1(
        *,
        op: OperatorIR,
        source_shape_nchw: Optional[List[int]] = None,
    ) -> bool:
        if str(op.op_type) != "SLICE" or len(op.inputs) < 3 or len(op.outputs) != 1:
            return False
        begin_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[1]), None))
        size_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[2]), None))
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        if int(size_vals[1]) <= 0:
            return False
        if int(begin_vals[2]) != 0 or int(begin_vals[3]) != 0:
            return False
        if source_shape_nchw is not None and len(source_shape_nchw) == 4:
            c = int(source_shape_nchw[1])
            if int(begin_vals[1]) < 0 or int(begin_vals[1]) + int(size_vals[1]) > c:
                return False
        return True

    def _rewrite_slice_axis1_to_axis3(slice_op: OperatorIR) -> bool:
        begin_tensor = model_ir.tensors.get(str(slice_op.inputs[1]), None)
        size_tensor = model_ir.tensors.get(str(slice_op.inputs[2]), None)
        begin_vals = _read_const_ints_from_tensor(begin_tensor)
        size_vals = _read_const_ints_from_tensor(size_tensor)
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        new_begin = [int(begin_vals[0]), int(begin_vals[2]), int(begin_vals[3]), int(begin_vals[1])]
        new_size = [int(size_vals[0]), int(size_vals[2]), int(size_vals[3]), int(size_vals[1])]
        _write_const_ints_to_tensor(begin_tensor, new_begin)
        _write_const_ints_to_tensor(size_tensor, new_size)
        return True

    def _rewrite_mul_const_to_nhwc_if_needed(
        *,
        mul_idx: int,
        mul_op: OperatorIR,
        data_input_name: str,
        consumers: Dict[str, List[int]],
    ) -> bool:
        mul_inputs = [str(v) for v in list(mul_op.inputs)]
        const_index: Optional[int] = None
        for i, name in enumerate(mul_inputs):
            if str(name) == str(data_input_name):
                continue
            tensor = model_ir.tensors.get(str(name), None)
            if tensor is not None and tensor.data is not None:
                const_index = int(i)
                break
        if const_index is None:
            return False

        const_name = str(mul_inputs[int(const_index)])
        const_tensor = model_ir.tensors.get(const_name, None)
        if const_tensor is None or const_tensor.data is None:
            return False
        const_data = np.asarray(const_tensor.data)
        if int(const_data.size) == 1:
            return True

        data_tensor = model_ir.tensors.get(str(data_input_name), None)
        if data_tensor is None or not _is_fully_known_positive_shape(list(data_tensor.shape)):
            return False
        target_shape = [int(v) for v in list(data_tensor.shape)]

        const_shape = [int(v) for v in list(const_data.shape)]
        if _broadcast_static_shapes(target_shape, const_shape) is not None:
            return True

        rotated: Optional[np.ndarray] = None
        if int(const_data.ndim) == 4:
            candidate = np.transpose(const_data, perm_nchw_to_nhwc).astype(const_data.dtype, copy=False)
            if _broadcast_static_shapes(target_shape, [int(v) for v in list(candidate.shape)]) is not None:
                rotated = np.asarray(candidate)
        elif int(const_data.ndim) == 3:
            candidate = np.transpose(const_data, [1, 2, 0]).astype(const_data.dtype, copy=False)
            if _broadcast_static_shapes(target_shape, [int(v) for v in list(candidate.shape)]) is not None:
                rotated = np.asarray(candidate)
        if rotated is None:
            return False

        const_users = [int(v) for v in consumers.get(const_name, [])]
        if any(int(v) != int(mul_idx) for v in const_users):
            rotated_name = _unique_tensor_name(f"{const_name}_nhwc")
            model_ir.tensors[rotated_name] = TensorIR(
                name=rotated_name,
                dtype=str(const_tensor.dtype),
                shape=[int(v) for v in list(rotated.shape)],
                shape_signature=[int(v) for v in list(rotated.shape)],
                data=np.asarray(rotated),
                is_variable=False,
                quantization=_clone_quantization(const_tensor.quantization),
            )
            mul_inputs[int(const_index)] = str(rotated_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=mul_op,
                new_inputs=mul_inputs,
            )
        else:
            const_tensor.data = np.asarray(rotated)
            const_tensor.shape = [int(v) for v in list(rotated.shape)]
            const_tensor.shape_signature = [int(v) for v in list(rotated.shape)]
        return True

    def _is_safe_direct_post_rewire(
        *,
        post_output_name: str,
        replacement_name: str,
        consumers: Dict[str, List[int]],
    ) -> bool:
        replacement_tensor = model_ir.tensors.get(str(replacement_name), None)
        replacement_shape = (
            [int(v) for v in list(replacement_tensor.shape)]
            if replacement_tensor is not None
            else None
        )
        binary_ops = {
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "MAXIMUM",
            "MINIMUM",
            "SQUARED_DIFFERENCE",
            "POW",
        }
        for consumer_idx in [int(v) for v in consumers.get(str(post_output_name), [])]:
            consumer_op = model_ir.operators[int(consumer_idx)]
            consumer_inputs = [str(v) for v in list(consumer_op.inputs)]
            replace_positions = [i for i, n in enumerate(consumer_inputs) if str(n) == str(post_output_name)]
            if len(replace_positions) == 0:
                continue
            if len(consumer_inputs) <= 1:
                continue
            if str(consumer_op.op_type) not in binary_ops:
                return False
            if replacement_shape is None or not _is_fully_known_positive_shape(replacement_shape):
                return False
            for pos in replace_positions:
                for other_pos, other_name in enumerate(consumer_inputs):
                    if int(other_pos) == int(pos):
                        continue
                    other_tensor = model_ir.tensors.get(str(other_name), None)
                    if other_tensor is None or not _is_fully_known_positive_shape(list(other_tensor.shape)):
                        return False
                    other_shape = [int(v) for v in list(other_tensor.shape)]
                    if (
                        _broadcast_static_shapes(replacement_shape, other_shape) is None
                        and _broadcast_static_shapes(other_shape, replacement_shape) is None
                    ):
                        return False
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            if pre_in_name in model_outputs or pre_out_name in model_outputs:
                continue

            pre_out_tensor = model_ir.tensors.get(pre_out_name, None)
            if pre_out_tensor is None or len(list(pre_out_tensor.shape)) != 4:
                continue

            slice_indices = [int(v) for v in consumers.get(pre_out_name, [])]
            if len(slice_indices) == 0:
                continue
            if not all(
                _is_channel_slice_axis1(
                    op=model_ir.operators[int(slice_idx)],
                    source_shape_nchw=[int(v) for v in list(pre_out_tensor.shape)],
                )
                for slice_idx in slice_indices
            ):
                continue

            remove_indices: set[int] = {int(pre_idx)}
            rewritable = True
            for slice_idx in slice_indices:
                slice_op = model_ir.operators[int(slice_idx)]
                slice_out_name = str(slice_op.outputs[0])
                if slice_out_name in model_outputs:
                    rewritable = False
                    break

                if not _rewrite_slice_axis1_to_axis3(slice_op):
                    rewritable = False
                    break
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=slice_op,
                    input_index=0,
                    new_input_name=pre_in_name,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(slice_out_name, None),
                    perm_nchw_to_nhwc,
                )

                slice_users = [int(v) for v in consumers.get(slice_out_name, []) if int(v) != int(slice_idx)]
                if len(slice_users) == 0:
                    continue

                if len(slice_users) == 1:
                    user_idx = int(slice_users[0])
                    user_op = model_ir.operators[int(user_idx)]
                    if (
                        str(user_op.op_type) == "TRANSPOSE"
                        and len(user_op.inputs) >= 2
                        and len(user_op.outputs) == 1
                        and str(user_op.inputs[0]) == slice_out_name
                        and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                        and str(user_op.outputs[0]) not in model_outputs
                    ):
                        if not _is_safe_direct_post_rewire(
                            post_output_name=str(user_op.outputs[0]),
                            replacement_name=slice_out_name,
                            consumers=consumers,
                        ):
                            rewritable = False
                            break
                        _replace_tensor_inputs(model_ir, str(user_op.outputs[0]), slice_out_name)
                        remove_indices.add(int(user_idx))
                        continue

                    if (
                        str(user_op.op_type) == "MUL"
                        and len(user_op.inputs) == 2
                        and len(user_op.outputs) == 1
                        and slice_out_name in [str(v) for v in list(user_op.inputs)]
                        and str(user_op.outputs[0]) not in model_outputs
                    ):
                        mul_out_name = str(user_op.outputs[0])
                        mul_users = [int(v) for v in consumers.get(mul_out_name, []) if int(v) != int(user_idx)]
                        if len(mul_users) != 1:
                            rewritable = False
                            break
                        post_idx = int(mul_users[0])
                        post_op = model_ir.operators[int(post_idx)]
                        if (
                            str(post_op.op_type) != "TRANSPOSE"
                            or len(post_op.inputs) < 2
                            or len(post_op.outputs) != 1
                            or str(post_op.inputs[0]) != mul_out_name
                            or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
                            or str(post_op.outputs[0]) in model_outputs
                        ):
                            rewritable = False
                            break
                        if not _rewrite_mul_const_to_nhwc_if_needed(
                            mul_idx=int(user_idx),
                            mul_op=user_op,
                            data_input_name=slice_out_name,
                            consumers=consumers,
                        ):
                            rewritable = False
                            break
                        _permute_tensor_metadata_if_rank_matches(
                            model_ir.tensors.get(mul_out_name, None),
                            perm_nchw_to_nhwc,
                        )
                        _replace_tensor_inputs(model_ir, str(post_op.outputs[0]), mul_out_name)
                        remove_indices.add(int(post_idx))
                        continue

                rewritable = False
                break

            if not rewritable:
                continue

            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_transpose_channel_slice_muladd_nhwc_bridge_chains": int(optimized)}


def _optimize_transpose_channel_slice_dual_add_bridges_strict(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Strictly fold:
      NHWC->NCHW TRANSPOSE -> 2x channel SLICE
        branch-A: SLICE -> MUL(const) -> TRANSPOSE -> ADD
        branch-B: SLICE -> TRANSPOSE -> ADD
    back to NHWC.

    This is intentionally strict to avoid unsafe late-stage rewrites.
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _is_axis1_channel_slice(op: OperatorIR) -> bool:
        if str(op.op_type) != "SLICE" or len(op.inputs) < 3 or len(op.outputs) != 1:
            return False
        begin_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[1]), None))
        size_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[2]), None))
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        if int(size_vals[1]) <= 0:
            return False
        if int(begin_vals[2]) != 0 or int(begin_vals[3]) != 0:
            return False
        return True

    def _rewrite_slice_axis1_to_axis3(slice_op: OperatorIR) -> bool:
        begin_tensor = model_ir.tensors.get(str(slice_op.inputs[1]), None)
        size_tensor = model_ir.tensors.get(str(slice_op.inputs[2]), None)
        begin_vals = _read_const_ints_from_tensor(begin_tensor)
        size_vals = _read_const_ints_from_tensor(size_tensor)
        if begin_vals is None or size_vals is None or len(begin_vals) != 4 or len(size_vals) != 4:
            return False
        new_begin = [int(begin_vals[0]), int(begin_vals[2]), int(begin_vals[3]), int(begin_vals[1])]
        new_size = [int(size_vals[0]), int(size_vals[2]), int(size_vals[3]), int(size_vals[1])]
        _write_const_ints_to_tensor(begin_tensor, new_begin)
        _write_const_ints_to_tensor(size_tensor, new_size)
        return True

    def _rewrite_mul_const_to_nhwc_if_needed(
        *,
        mul_idx: int,
        mul_op: OperatorIR,
        data_input_name: str,
        target_shape_nhwc: Optional[List[int]],
        consumers: Dict[str, List[int]],
        chain_indices: set[int],
    ) -> bool:
        mul_inputs = [str(v) for v in list(mul_op.inputs)]
        const_index: Optional[int] = None
        for i, name in enumerate(mul_inputs):
            if str(name) == str(data_input_name):
                continue
            tensor = model_ir.tensors.get(str(name), None)
            if tensor is not None and tensor.data is not None:
                const_index = int(i)
                break
        if const_index is None:
            return False

        const_name = str(mul_inputs[int(const_index)])
        const_tensor = model_ir.tensors.get(const_name, None)
        if const_tensor is None or const_tensor.data is None:
            return False
        const_data = np.asarray(const_tensor.data)
        if int(const_data.size) == 1:
            return True

        if not _is_fully_known_positive_shape(target_shape_nhwc):
            return False
        target_shape = [int(v) for v in list(target_shape_nhwc)]

        rotated: Optional[np.ndarray] = None
        if int(const_data.ndim) == 4:
            candidate = np.transpose(const_data, perm_nchw_to_nhwc).astype(const_data.dtype, copy=False)
            if _broadcast_static_shapes(target_shape, [int(v) for v in list(candidate.shape)]) is not None:
                rotated = np.asarray(candidate)
        elif int(const_data.ndim) == 3:
            candidate = np.transpose(const_data, [1, 2, 0]).astype(const_data.dtype, copy=False)
            if _broadcast_static_shapes(target_shape, [int(v) for v in list(candidate.shape)]) is not None:
                rotated = np.asarray(candidate)
        else:
            if _broadcast_static_shapes(target_shape, [int(v) for v in list(const_data.shape)]) is not None:
                rotated = np.asarray(const_data)
        if rotated is None:
            return False

        const_users = [int(v) for v in consumers.get(const_name, [])]
        shared_outside_chain = any(int(v) not in chain_indices for v in const_users)
        if shared_outside_chain:
            rotated_name = _unique_tensor_name(f"{const_name}_nhwc")
            model_ir.tensors[rotated_name] = TensorIR(
                name=rotated_name,
                dtype=str(const_tensor.dtype),
                shape=[int(v) for v in list(rotated.shape)],
                shape_signature=[int(v) for v in list(rotated.shape)],
                data=np.asarray(rotated),
                is_variable=False,
                quantization=_clone_quantization(const_tensor.quantization),
            )
            mul_inputs[int(const_index)] = str(rotated_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=mul_op,
                new_inputs=mul_inputs,
            )
        else:
            const_tensor.data = np.asarray(rotated)
            const_tensor.shape = [int(v) for v in list(rotated.shape)]
            const_tensor.shape_signature = [int(v) for v in list(rotated.shape)]

        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            if pre_in_name in model_outputs or pre_out_name in model_outputs:
                continue

            slice_indices = [int(v) for v in consumers.get(pre_out_name, [])]
            if len(slice_indices) != 2:
                continue
            if not all(_is_axis1_channel_slice(model_ir.operators[int(slice_idx)]) for slice_idx in slice_indices):
                continue

            branch_plans: List[Dict[str, Any]] = []
            branch_kind_set: set[str] = set()
            valid = True

            for slice_idx in sorted(slice_indices):
                slice_op = model_ir.operators[int(slice_idx)]
                slice_out_name = str(slice_op.outputs[0])
                if slice_out_name in model_outputs:
                    valid = False
                    break

                slice_users = [int(v) for v in consumers.get(slice_out_name, []) if int(v) != int(slice_idx)]
                if len(slice_users) != 1:
                    valid = False
                    break
                first_idx = int(slice_users[0])
                first_op = model_ir.operators[int(first_idx)]

                if (
                    str(first_op.op_type) == "TRANSPOSE"
                    and len(first_op.inputs) >= 2
                    and len(first_op.outputs) == 1
                    and str(first_op.inputs[0]) == slice_out_name
                    and _read_transpose_perm(model_ir, first_op) == perm_nchw_to_nhwc
                    and str(first_op.outputs[0]) not in model_outputs
                ):
                    post_out_name = str(first_op.outputs[0])
                    post_users = [int(v) for v in consumers.get(post_out_name, [])]
                    if len(post_users) != 1:
                        valid = False
                        break
                    add_idx = int(post_users[0])
                    add_op = model_ir.operators[int(add_idx)]
                    if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                        valid = False
                        break
                    add_inputs = [str(v) for v in list(add_op.inputs)]
                    if str(post_out_name) not in add_inputs:
                        valid = False
                        break
                    other_name = add_inputs[1] if add_inputs[0] == str(post_out_name) else add_inputs[0]
                    other_tensor = model_ir.tensors.get(str(other_name), None)
                    slice_tensor = model_ir.tensors.get(slice_out_name, None)
                    target_shape_nhwc = (
                        _permute_shape(list(slice_tensor.shape), perm_nchw_to_nhwc)
                        if slice_tensor is not None and len(list(slice_tensor.shape)) == 4
                        else None
                    )
                    if (
                        other_tensor is None
                        or target_shape_nhwc is None
                        or not _is_fully_known_positive_shape(list(other_tensor.shape))
                        or _broadcast_static_shapes(
                            [int(v) for v in list(target_shape_nhwc)],
                            [int(v) for v in list(other_tensor.shape)],
                        ) is None
                    ):
                        valid = False
                        break
                    branch_kind_set.add("direct")
                    branch_plans.append(
                        {
                            "kind": "direct",
                            "slice_idx": int(slice_idx),
                            "slice_out_name": str(slice_out_name),
                            "post_idx": int(first_idx),
                            "post_out_name": str(post_out_name),
                        }
                    )
                    continue

                if (
                    str(first_op.op_type) == "MUL"
                    and len(first_op.inputs) == 2
                    and len(first_op.outputs) == 1
                    and slice_out_name in [str(v) for v in list(first_op.inputs)]
                    and str(first_op.outputs[0]) not in model_outputs
                ):
                    mul_idx = int(first_idx)
                    mul_op = first_op
                    mul_out_name = str(mul_op.outputs[0])
                    mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
                    if len(mul_users) != 1:
                        valid = False
                        break
                    post_idx = int(mul_users[0])
                    post_op = model_ir.operators[int(post_idx)]
                    if (
                        str(post_op.op_type) != "TRANSPOSE"
                        or len(post_op.inputs) < 2
                        or len(post_op.outputs) != 1
                        or str(post_op.inputs[0]) != mul_out_name
                        or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
                        or str(post_op.outputs[0]) in model_outputs
                    ):
                        valid = False
                        break
                    post_out_name = str(post_op.outputs[0])
                    post_users = [int(v) for v in consumers.get(post_out_name, [])]
                    if len(post_users) != 1:
                        valid = False
                        break
                    add_idx = int(post_users[0])
                    add_op = model_ir.operators[int(add_idx)]
                    if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                        valid = False
                        break
                    add_inputs = [str(v) for v in list(add_op.inputs)]
                    if str(post_out_name) not in add_inputs:
                        valid = False
                        break
                    other_name = add_inputs[1] if add_inputs[0] == str(post_out_name) else add_inputs[0]
                    other_tensor = model_ir.tensors.get(str(other_name), None)
                    slice_tensor = model_ir.tensors.get(slice_out_name, None)
                    target_shape_nhwc = (
                        _permute_shape(list(slice_tensor.shape), perm_nchw_to_nhwc)
                        if slice_tensor is not None and len(list(slice_tensor.shape)) == 4
                        else None
                    )
                    if (
                        other_tensor is None
                        or target_shape_nhwc is None
                        or not _is_fully_known_positive_shape(list(other_tensor.shape))
                        or _broadcast_static_shapes(
                            [int(v) for v in list(target_shape_nhwc)],
                            [int(v) for v in list(other_tensor.shape)],
                        ) is None
                    ):
                        valid = False
                        break
                    branch_kind_set.add("mul")
                    branch_plans.append(
                        {
                            "kind": "mul",
                            "slice_idx": int(slice_idx),
                            "slice_out_name": str(slice_out_name),
                            "mul_idx": int(mul_idx),
                            "mul_out_name": str(mul_out_name),
                            "post_idx": int(post_idx),
                            "post_out_name": str(post_out_name),
                            "target_shape_nhwc": [int(v) for v in list(target_shape_nhwc)],
                        }
                    )
                    continue

                valid = False
                break

            if not valid:
                continue
            if branch_kind_set != {"direct", "mul"}:
                continue

            remove_indices: set[int] = {int(pre_idx)}
            chain_indices: set[int] = {int(pre_idx)}
            chain_indices.update(int(v) for v in slice_indices)
            for plan in branch_plans:
                if "mul_idx" in plan:
                    chain_indices.add(int(plan["mul_idx"]))
                chain_indices.add(int(plan["post_idx"]))

            apply_ok = True
            for slice_idx in sorted(slice_indices):
                slice_op = model_ir.operators[int(slice_idx)]
                if not _rewrite_slice_axis1_to_axis3(slice_op):
                    apply_ok = False
                    break
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=slice_op,
                    input_index=0,
                    new_input_name=pre_in_name,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(slice_op.outputs[0]), None),
                    perm_nchw_to_nhwc,
                )
            if not apply_ok:
                continue

            for plan in branch_plans:
                if str(plan["kind"]) == "mul":
                    mul_idx = int(plan["mul_idx"])
                    mul_op = model_ir.operators[int(mul_idx)]
                    if not _rewrite_mul_const_to_nhwc_if_needed(
                        mul_idx=int(mul_idx),
                        mul_op=mul_op,
                        data_input_name=str(plan["slice_out_name"]),
                        target_shape_nhwc=[int(v) for v in list(plan["target_shape_nhwc"])],
                        consumers=consumers,
                        chain_indices=chain_indices,
                    ):
                        apply_ok = False
                        break
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(plan["mul_out_name"]), None),
                        perm_nchw_to_nhwc,
                    )
                    _replace_tensor_inputs(
                        model_ir,
                        str(plan["post_out_name"]),
                        str(plan["mul_out_name"]),
                    )
                    remove_indices.add(int(plan["post_idx"]))
                else:
                    _replace_tensor_inputs(
                        model_ir,
                        str(plan["post_out_name"]),
                        str(plan["slice_out_name"]),
                    )
                    remove_indices.add(int(plan["post_idx"]))

            if not apply_ok:
                continue

            for remove_idx in sorted(list(remove_indices), reverse=True):
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_transpose_channel_slice_dual_add_bridges_strict": int(optimized)}


def _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove NHWC<->NCHW boundary transpose round-trips around STRIDED_SLICE QDQ-CONCAT stems.

    Target pattern (strict):
      input(NHWC) --TRANSPOSE(0,3,1,2)--> input_onnx_ncx_internal
        -> STRIDED_SLICE* -> QUANTIZE -> DEQUANTIZE
        -> CONCAT(axis=1) -> QUANTIZE
        -> TRANSPOSE(0,2,3,1)*

    Rewrite:
    - Move STRIDED_SLICE to NHWC by permuting begin/end/stride constants.
    - Switch CONCAT axis from 1(NCHW-C) to 3(NHWC-C).
    - Remove trailing NCHW->NHWC transposes by writing QUANTIZE directly to
      their output tensor names.
    - Remove shared boundary NHWC->NCHW transpose.
    """
    removed_boundary = 0
    removed_post_transposes = 0
    rewritten_slices = 0
    rewritten_concat = 0

    perm_nchw_to_nhwc = [0, 2, 3, 1]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            input_name = str(pre_op.inputs[0])
            internal_name = str(pre_op.outputs[0])
            if input_name not in model_inputs:
                continue
            if input_name in model_outputs or internal_name in model_outputs:
                continue
            if not str(internal_name).endswith("_onnx_ncx_internal"):
                continue
            if _read_transpose_perm(model_ir, pre_op) != [0, 3, 1, 2]:
                continue

            internal_users = [int(v) for v in consumers.get(internal_name, [])]
            if len(internal_users) == 0:
                continue

            slice_indices: List[int] = []
            slice_rewrites: List[Dict[str, Any]] = []
            q_indices: List[int] = []
            dq_indices: List[int] = []
            dq_output_names: List[str] = []
            concat_idx: Optional[int] = None

            valid = True
            for user_idx in sorted(internal_users):
                op = model_ir.operators[int(user_idx)]
                if str(op.op_type) != "STRIDED_SLICE" or len(op.inputs) < 4 or len(op.outputs) != 1:
                    valid = False
                    break
                if str(op.inputs[0]) != internal_name:
                    valid = False
                    break
                # Keep the rewrite strict: axis masks must be absent or zero.
                opts = dict(op.options) if isinstance(op.options, dict) else {}
                if any(int(opts.get(k, 0)) != 0 for k in ["beginMask", "endMask", "ellipsisMask", "newAxisMask", "shrinkAxisMask"]):
                    valid = False
                    break
                begin_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                end_tensor = model_ir.tensors.get(str(op.inputs[2]), None)
                stride_tensor = model_ir.tensors.get(str(op.inputs[3]), None)
                begin_vals = _read_const_ints_from_tensor(begin_tensor)
                end_vals = _read_const_ints_from_tensor(end_tensor)
                stride_vals = _read_const_ints_from_tensor(stride_tensor)
                if (
                    begin_vals is None
                    or end_vals is None
                    or stride_vals is None
                    or len(begin_vals) != 4
                    or len(end_vals) != 4
                    or len(stride_vals) != 4
                ):
                    valid = False
                    break
                new_begin = [int(begin_vals[idx]) for idx in perm_nchw_to_nhwc]
                new_end = [int(end_vals[idx]) for idx in perm_nchw_to_nhwc]
                new_stride = [int(stride_vals[idx]) for idx in perm_nchw_to_nhwc]

                slice_out = str(op.outputs[0])
                if slice_out in model_outputs:
                    valid = False
                    break
                slice_users = [int(v) for v in consumers.get(slice_out, [])]
                if len(slice_users) != 1:
                    valid = False
                    break
                q_idx = int(slice_users[0])
                q_op = model_ir.operators[int(q_idx)]
                if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                    valid = False
                    break
                if str(q_op.inputs[0]) != slice_out:
                    valid = False
                    break

                q_out = str(q_op.outputs[0])
                q_users = [int(v) for v in consumers.get(q_out, [])]
                if len(q_users) != 1:
                    valid = False
                    break
                dq_idx = int(q_users[0])
                dq_op = model_ir.operators[int(dq_idx)]
                if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                    valid = False
                    break
                if str(dq_op.inputs[0]) != q_out:
                    valid = False
                    break

                dq_out = str(dq_op.outputs[0])
                dq_users = [int(v) for v in consumers.get(dq_out, [])]
                if len(dq_users) != 1:
                    valid = False
                    break
                cand_concat_idx = int(dq_users[0])
                cand_concat = model_ir.operators[int(cand_concat_idx)]
                if str(cand_concat.op_type) != "CONCATENATION" or len(cand_concat.outputs) != 1:
                    valid = False
                    break
                axis = int(cand_concat.options.get("axis", 1))
                if axis < 0:
                    axis += 4
                if axis != 1:
                    valid = False
                    break

                if concat_idx is None:
                    concat_idx = int(cand_concat_idx)
                elif int(concat_idx) != int(cand_concat_idx):
                    valid = False
                    break

                slice_indices.append(int(user_idx))
                slice_rewrites.append(
                    {
                        "slice_idx": int(user_idx),
                        "begin_name": str(op.inputs[1]),
                        "end_name": str(op.inputs[2]),
                        "stride_name": str(op.inputs[3]),
                        "new_begin": [int(v) for v in list(new_begin)],
                        "new_end": [int(v) for v in list(new_end)],
                        "new_stride": [int(v) for v in list(new_stride)],
                    }
                )
                q_indices.append(int(q_idx))
                dq_indices.append(int(dq_idx))
                dq_output_names.append(str(dq_out))

            if not valid or concat_idx is None or len(slice_indices) == 0:
                continue

            concat_op = model_ir.operators[int(concat_idx)]
            if set(str(v) for v in list(concat_op.inputs)) != set(dq_output_names):
                continue

            concat_out = str(concat_op.outputs[0])
            if concat_out in model_outputs:
                continue
            concat_users = [int(v) for v in consumers.get(concat_out, [])]
            if len(concat_users) != 1:
                continue
            q_concat_idx = int(concat_users[0])
            q_concat_op = model_ir.operators[int(q_concat_idx)]
            if str(q_concat_op.op_type) != "QUANTIZE" or len(q_concat_op.inputs) != 1 or len(q_concat_op.outputs) != 1:
                continue
            if str(q_concat_op.inputs[0]) != concat_out:
                continue

            q_concat_out = str(q_concat_op.outputs[0])
            if q_concat_out in model_outputs:
                continue
            post_indices = [int(v) for v in consumers.get(q_concat_out, [])]
            if len(post_indices) == 0:
                continue
            post_output_names: List[str] = []
            valid_posts = True
            for post_idx in post_indices:
                post_op = model_ir.operators[int(post_idx)]
                if (
                    str(post_op.op_type) != "TRANSPOSE"
                    or len(post_op.inputs) < 2
                    or len(post_op.outputs) != 1
                    or str(post_op.inputs[0]) != q_concat_out
                    or _read_transpose_perm(model_ir, post_op) != [0, 2, 3, 1]
                ):
                    valid_posts = False
                    break
                post_out = str(post_op.outputs[0])
                if post_out in model_outputs:
                    valid_posts = False
                    break
                post_output_names.append(post_out)
            if not valid_posts:
                continue

            # Rewrite each STRIDED_SLICE from NCHW params to NHWC params.
            for rewrite in slice_rewrites:
                slice_idx = int(rewrite["slice_idx"])
                slice_op = model_ir.operators[int(slice_idx)]
                begin_name = str(rewrite["begin_name"])
                end_name = str(rewrite["end_name"])
                stride_name = str(rewrite["stride_name"])
                new_begin = [int(v) for v in list(rewrite["new_begin"])]
                new_end = [int(v) for v in list(rewrite["new_end"])]
                new_stride = [int(v) for v in list(rewrite["new_stride"])]

                begin_tensor = model_ir.tensors.get(begin_name, None)
                end_tensor = model_ir.tensors.get(end_name, None)
                stride_tensor = model_ir.tensors.get(stride_name, None)
                begin_vals = _read_const_ints_from_tensor(begin_tensor)
                end_vals = _read_const_ints_from_tensor(end_tensor)
                stride_vals = _read_const_ints_from_tensor(stride_tensor)
                if (
                    begin_vals is None
                    or end_vals is None
                    or stride_vals is None
                    or len(begin_vals) != 4
                    or len(end_vals) != 4
                    or len(stride_vals) != 4
                ):
                    valid = False
                    break

                if begin_vals != new_begin:
                    _write_const_ints_to_tensor(begin_tensor, new_begin)
                    if _read_const_ints_from_tensor(begin_tensor) != new_begin:
                        valid = False
                        break
                if end_vals != new_end:
                    _write_const_ints_to_tensor(end_tensor, new_end)
                    if _read_const_ints_from_tensor(end_tensor) != new_end:
                        valid = False
                        break
                if stride_vals != new_stride:
                    _write_const_ints_to_tensor(stride_tensor, new_stride)
                    if _read_const_ints_from_tensor(stride_tensor) != new_stride:
                        valid = False
                        break

                _set_operator_inputs(
                    model_ir=model_ir,
                    op=slice_op,
                    new_inputs=[input_name, begin_name, end_name, stride_name],
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(slice_op.outputs[0]), None),
                    [0, 2, 3, 1],
                )
                rewritten_slices += 1

            if not valid:
                continue

            # Propagate permuted metadata through Q/DQ/CONCAT/Q.
            for q_idx in q_indices:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(model_ir.operators[int(q_idx)].outputs[0]), None),
                    [0, 2, 3, 1],
                )
            for dq_idx in dq_indices:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(model_ir.operators[int(dq_idx)].outputs[0]), None),
                    [0, 2, 3, 1],
                )
            concat_op.options["axis"] = 3
            rewritten_concat += 1
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(concat_out, None),
                [0, 2, 3, 1],
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(q_concat_out, None),
                [0, 2, 3, 1],
            )

            # Remove trailing transposes by writing quantized concat output directly.
            canonical_post_output = str(post_output_names[0])
            _set_operator_outputs(
                model_ir=model_ir,
                op=q_concat_op,
                new_outputs=[canonical_post_output],
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(model_ir, alias_name, canonical_post_output)

            canonical_tensor = model_ir.tensors.get(canonical_post_output, None)
            q_concat_tensor = model_ir.tensors.get(q_concat_out, None)
            if canonical_tensor is not None and q_concat_tensor is not None:
                canonical_tensor.dtype = str(q_concat_tensor.dtype)
                canonical_tensor.quantization = _clone_quantization(q_concat_tensor.quantization)
                canonical_tensor.shape = [int(v) for v in list(q_concat_tensor.shape)]
                canonical_tensor.shape_signature = (
                    [int(v) for v in list(q_concat_tensor.shape_signature)]
                    if q_concat_tensor.shape_signature is not None
                    else [int(v) for v in list(q_concat_tensor.shape)]
                )

            remove_indices = sorted(
                set([int(pre_idx)] + [int(v) for v in post_indices]),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]
                if int(remove_idx) in post_indices:
                    removed_post_transposes += 1
            removed_boundary += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_boundary_input_transpose_stridedslice_blocks": int(removed_boundary),
        "rewritten_boundary_stridedslices": int(rewritten_slices),
        "rewritten_boundary_qdq_concat_axis": int(rewritten_concat),
        "removed_boundary_post_transposes": int(removed_post_transposes),
    }
