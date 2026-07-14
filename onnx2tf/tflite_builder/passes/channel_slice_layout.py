from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
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
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_boundary_input_transpose_channel_slice_blocks(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
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
    propagation_op_types = (
        layout_passthrough_unary
        | layout_passthrough_binary
        | {
            "CONCATENATION",
            "SLICE",
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
        }
    )

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

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        changed = False
        model_inputs = set(str(v) for v in model_ir.inputs)

        for pre_idx in graph_index.operator_indices("TRANSPOSE"):
            pre_op = model_ir.operators[int(pre_idx)]
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
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

            internal_users = graph_index.consumer_indices(internal_name)
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
                graph_index.replace_operator_inputs(
                    int(user_idx),
                    [input_name, str(user_op.inputs[1]), str(user_op.inputs[2])],
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
                for op_idx in graph_index.operator_indices_for_types(
                    propagation_op_types
                ):
                    op = model_ir.operators[int(op_idx)]
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
            bridge_plans: List[Tuple[int, str, str, List[int]]] = []
            for tensor_name in sorted(list(nhwc_tensors)):
                users = [
                    int(v)
                    for v in graph_index.consumer_indices(str(tensor_name))
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
                    graph_index.replace_operator_inputs(
                        int(user_idx),
                        updated_inputs,
                    )
                bridge_plans.append((int(min(users)), str(tensor_name), str(bridge_name), [int(v) for v in users]))

            inserted = 0
            for insert_idx, source_name, bridge_name, _users in sorted(bridge_plans, key=lambda v: int(v[0])):
                graph_index.insert_operator(
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
            remaining_users = graph_index.consumer_indices(internal_name)
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
                graph_index.insert_operator(
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
                shifted_user_idx = graph_index.operator_index(user_op)
                if shifted_user_idx is None:
                    raise RuntimeError(
                        "channel-slice consumer disappeared during adapter insertion"
                    )
                graph_index.replace_operator_inputs(
                    int(shifted_user_idx),
                    updated_inputs,
                )
                inserted_local_transposes += 1

            # 5) Remove shared boundary transpose.
            current_pre_idx = graph_index.operator_index(pre_op)
            if current_pre_idx is None:
                raise RuntimeError(
                    "boundary transpose disappeared during channel-slice rewrite"
                )
            graph_index.remove_operator(int(current_pre_idx))
            removed_boundary += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if removed_boundary > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "removed_boundary_input_transpose": int(removed_boundary),
        "rewritten_boundary_channel_slices": int(rewritten_channel_slices),
        "rewritten_boundary_axis_ops": int(rewritten_axis_ops),
        "inserted_local_boundary_transposes": int(inserted_local_transposes),
    }


def _optimize_internal_transpose_channel_slice_nhwc_propagation_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
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
    propagation_op_types = (
        layout_passthrough_unary
        | layout_passthrough_binary
        | {
            "CONCATENATION",
            "SLICE",
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "AVERAGE_POOL_2D",
            "MAX_POOL_2D",
        }
    )

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

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        changed = False
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx in graph_index.operator_indices("TRANSPOSE"):
            pre_op = model_ir.operators[int(pre_idx)]
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
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

            internal_users = graph_index.consumer_indices(internal_name)
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
            initial_constant_users: Dict[str, List[int]] = {}

            for user_idx in sorted(internal_users):
                user_op = model_ir.operators[int(user_idx)]
                graph_index.replace_operator_inputs(
                    int(user_idx),
                    [input_name, str(user_op.inputs[1]), str(user_op.inputs[2])],
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
                for op_idx in graph_index.operator_indices_for_types(
                    propagation_op_types
                ):
                    op = model_ir.operators[int(op_idx)]
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

                            if str(const_name) not in initial_constant_users:
                                initial_constant_users[str(const_name)] = (
                                    graph_index.consumer_indices(str(const_name))
                                )
                            const_users = initial_constant_users[str(const_name)]
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
                            graph_index.replace_operator_inputs(
                                int(op_idx),
                                new_inputs,
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

            bridge_plans: List[Tuple[int, str, str, List[int]]] = []
            for tensor_name in sorted(list(nhwc_tensors)):
                users = [
                    int(v)
                    for v in graph_index.consumer_indices(str(tensor_name))
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
                    graph_index.replace_operator_inputs(
                        int(user_idx),
                        updated_inputs,
                    )
                bridge_plans.append((int(min(users)), str(tensor_name), str(bridge_name), [int(v) for v in users]))

            inserted = 0
            for insert_idx, source_name, bridge_name, _users in sorted(bridge_plans, key=lambda v: int(v[0])):
                graph_index.insert_operator(
                    int(insert_idx + inserted),
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(source_name), str(pre_op.inputs[1])],
                        outputs=[str(bridge_name)],
                    ),
                )
                inserted += 1
                inserted_local_transposes += 1

            current_pre_idx = graph_index.operator_index(pre_op)
            if current_pre_idx is None:
                raise RuntimeError(
                    "internal transpose disappeared during channel-slice rewrite"
                )
            graph_index.remove_operator(int(current_pre_idx))
            removed_internal += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if removed_internal > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "removed_internal_transpose_channel_slice_stems": int(removed_internal),
        "rewritten_internal_channel_slices": int(rewritten_channel_slices),
        "rewritten_internal_axis_ops": int(rewritten_axis_ops),
        "inserted_internal_local_transposes": int(inserted_local_transposes),
    }


def _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
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
                graph_index=graph_index,
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

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        changed = False
        consumers = {
            str(name): [int(value) for value in values]
            for name, values in graph_index.consumers.items()
        }
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx in graph_index.operator_indices("TRANSPOSE"):
            pre_op = model_ir.operators[int(pre_idx)]
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
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
                    graph_index=graph_index,
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
                        _replace_tensor_inputs(
                            model_ir,
                            str(user_op.outputs[0]),
                            slice_out_name,
                            graph_index=graph_index,
                        )
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
                        _replace_tensor_inputs(
                            model_ir,
                            str(post_op.outputs[0]),
                            mul_out_name,
                            graph_index=graph_index,
                        )
                        remove_indices.add(int(post_idx))
                        continue

                rewritable = False
                break

            if not rewritable:
                continue

            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_channel_slice_muladd_nhwc_bridge_chains": int(optimized)}


def _optimize_transpose_slice_muladd_conv_mergeadd_strict(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Strictly fold the following into a single post-merge adapter:
      T0: NHWC->NCHW
      S0/S1: channel slices from T0 output
      S0 -> MUL -> T1(NCHW->NHWC) -> ADD(const) -> CONV_2D -> T2(NHWC->NCHW)
      merge ADD(T2_out, S1_out) -> y_nchw

    Rewrite:
      S0/S1 read from T0 input in NHWC (axis=3 channel slices)
      S0 branch stays NHWC through MUL/ADD/CONV
      merge ADD runs in NHWC
      one new NHWC->NCHW transpose is inserted after merge ADD.
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _find_or_create_nhwc_to_nchw_perm_tensor() -> str:
        target = np.asarray(perm_nhwc_to_nchw, dtype=np.int32)
        for tensor_name, tensor in model_ir.tensors.items():
            if (
                tensor is None
                or tensor.data is None
                or len(list(tensor.shape)) != 1
                or int(tensor.shape[0]) != 4
            ):
                continue
            data = np.asarray(tensor.data)
            if data.shape == (4,) and data.dtype == np.int32 and np.array_equal(data, target):
                return str(tensor_name)
        perm_name = _unique_tensor_name("__nhwc_to_nchw_perm_rank4__")
        model_ir.tensors[perm_name] = TensorIR(
            name=perm_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(target, dtype=np.int32),
            is_variable=False,
        )
        return perm_name

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
                graph_index=graph_index,
            )
        else:
            const_tensor.data = np.asarray(rotated)
            const_tensor.shape = [int(v) for v in list(rotated.shape)]
            const_tensor.shape_signature = [int(v) for v in list(rotated.shape)]
        return True

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            if pre_out_name in model_outputs:
                continue

            slice_indices = [int(v) for v in consumers.get(pre_out_name, [])]
            if len(slice_indices) != 2:
                continue
            if not all(_is_axis1_channel_slice(model_ir.operators[int(v)]) for v in slice_indices):
                continue

            branch_a_plan: Optional[Dict[str, Any]] = None
            branch_b_plan: Optional[Dict[str, Any]] = None
            merge_add_idx: Optional[int] = None

            valid = True
            for slice_idx in sorted(slice_indices):
                slice_op = model_ir.operators[int(slice_idx)]
                slice_out_name = str(slice_op.outputs[0])
                if slice_out_name in model_outputs:
                    valid = False
                    break

                users1 = [int(v) for v in consumers.get(slice_out_name, []) if int(v) != int(slice_idx)]
                if len(users1) != 1:
                    valid = False
                    break
                first_idx = int(users1[0])
                first_op = model_ir.operators[int(first_idx)]

                # Branch-B: slice -> merge ADD directly
                if str(first_op.op_type) == "ADD" and len(first_op.inputs) == 2 and len(first_op.outputs) == 1:
                    if merge_add_idx is None:
                        merge_add_idx = int(first_idx)
                    elif int(merge_add_idx) != int(first_idx):
                        valid = False
                        break
                    if str(first_op.outputs[0]) in model_outputs:
                        valid = False
                        break
                    if branch_b_plan is not None:
                        valid = False
                        break
                    branch_b_plan = {
                        "slice_idx": int(slice_idx),
                        "slice_out_name": str(slice_out_name),
                        "merge_add_idx": int(first_idx),
                    }
                    continue

                # Branch-A: slice -> MUL -> T -> ADD -> CONV_2D -> T -> merge ADD
                if str(first_op.op_type) != "MUL" or len(first_op.inputs) != 2 or len(first_op.outputs) != 1:
                    valid = False
                    break
                mul_idx = int(first_idx)
                mul_out_name = str(first_op.outputs[0])
                if mul_out_name in model_outputs:
                    valid = False
                    break
                mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
                if len(mul_users) != 1:
                    valid = False
                    break
                t1_idx = int(mul_users[0])
                t1_op = model_ir.operators[int(t1_idx)]
                if (
                    str(t1_op.op_type) != "TRANSPOSE"
                    or len(t1_op.inputs) < 2
                    or len(t1_op.outputs) != 1
                    or str(t1_op.inputs[0]) != mul_out_name
                    or _read_transpose_perm(model_ir, t1_op) != perm_nchw_to_nhwc
                ):
                    valid = False
                    break
                t1_out_name = str(t1_op.outputs[0])
                if t1_out_name in model_outputs:
                    valid = False
                    break
                t1_users = [int(v) for v in consumers.get(t1_out_name, [])]
                if len(t1_users) != 1:
                    valid = False
                    break
                add0_idx = int(t1_users[0])
                add0_op = model_ir.operators[int(add0_idx)]
                if str(add0_op.op_type) != "ADD" or len(add0_op.inputs) != 2 or len(add0_op.outputs) != 1:
                    valid = False
                    break
                add0_out_name = str(add0_op.outputs[0])
                if add0_out_name in model_outputs:
                    valid = False
                    break
                add0_inputs = [str(v) for v in list(add0_op.inputs)]
                if str(t1_out_name) not in add0_inputs:
                    valid = False
                    break
                add0_side_name = add0_inputs[1] if add0_inputs[0] == str(t1_out_name) else add0_inputs[0]
                add0_side_tensor = model_ir.tensors.get(str(add0_side_name), None)
                if add0_side_tensor is None or add0_side_tensor.data is None:
                    valid = False
                    break

                conv_users = [int(v) for v in consumers.get(add0_out_name, [])]
                if len(conv_users) != 1:
                    valid = False
                    break
                conv_idx = int(conv_users[0])
                conv_op = model_ir.operators[int(conv_idx)]
                if (
                    str(conv_op.op_type) != "CONV_2D"
                    or len(conv_op.inputs) < 2
                    or len(conv_op.outputs) != 1
                    or str(conv_op.inputs[0]) != add0_out_name
                ):
                    valid = False
                    break
                conv_out_name = str(conv_op.outputs[0])
                if conv_out_name in model_outputs:
                    valid = False
                    break
                conv_out_users = [int(v) for v in consumers.get(conv_out_name, [])]
                if len(conv_out_users) != 1:
                    valid = False
                    break
                t2_idx = int(conv_out_users[0])
                t2_op = model_ir.operators[int(t2_idx)]
                if (
                    str(t2_op.op_type) != "TRANSPOSE"
                    or len(t2_op.inputs) < 2
                    or len(t2_op.outputs) != 1
                    or str(t2_op.inputs[0]) != conv_out_name
                    or _read_transpose_perm(model_ir, t2_op) != perm_nhwc_to_nchw
                ):
                    valid = False
                    break
                t2_out_name = str(t2_op.outputs[0])
                if t2_out_name in model_outputs:
                    valid = False
                    break
                merge_users = [int(v) for v in consumers.get(t2_out_name, [])]
                if len(merge_users) != 1:
                    valid = False
                    break
                cand_merge_add_idx = int(merge_users[0])
                merge_add_op = model_ir.operators[int(cand_merge_add_idx)]
                if str(merge_add_op.op_type) != "ADD" or len(merge_add_op.inputs) != 2 or len(merge_add_op.outputs) != 1:
                    valid = False
                    break
                if merge_add_idx is None:
                    merge_add_idx = int(cand_merge_add_idx)
                elif int(merge_add_idx) != int(cand_merge_add_idx):
                    valid = False
                    break
                if branch_a_plan is not None:
                    valid = False
                    break
                branch_a_plan = {
                    "slice_idx": int(slice_idx),
                    "slice_out_name": str(slice_out_name),
                    "mul_idx": int(mul_idx),
                    "mul_op": first_op,
                    "mul_out_name": str(mul_out_name),
                    "t1_idx": int(t1_idx),
                    "t1_out_name": str(t1_out_name),
                    "add0_idx": int(add0_idx),
                    "add0_side_name": str(add0_side_name),
                    "conv_idx": int(conv_idx),
                    "conv_out_name": str(conv_out_name),
                    "t2_idx": int(t2_idx),
                    "t2_out_name": str(t2_out_name),
                    "merge_add_idx": int(cand_merge_add_idx),
                }

            if not valid or branch_a_plan is None or branch_b_plan is None or merge_add_idx is None:
                continue

            merge_add_op = model_ir.operators[int(merge_add_idx)]
            merge_out_name = str(merge_add_op.outputs[0])
            if merge_out_name in model_outputs:
                continue
            if set(int(v) for v in consumers.get(pre_out_name, [])) != set(int(v) for v in slice_indices):
                continue
            if set(int(v) for v in consumers.get(str(branch_b_plan["slice_out_name"]), [])) != {int(merge_add_idx)}:
                continue
            if set(int(v) for v in consumers.get(str(branch_a_plan["t2_out_name"]), [])) != {int(merge_add_idx)}:
                continue

            a_slice_tensor = model_ir.tensors.get(str(branch_a_plan["slice_out_name"]), None)
            conv_out_tensor = model_ir.tensors.get(str(branch_a_plan["conv_out_name"]), None)
            if (
                a_slice_tensor is None
                or conv_out_tensor is None
                or len(list(a_slice_tensor.shape)) != 4
                or len(list(conv_out_tensor.shape)) != 4
            ):
                continue
            a_slice_shape_nhwc = _permute_shape(list(a_slice_tensor.shape), perm_nchw_to_nhwc)
            b_slice_tensor = model_ir.tensors.get(str(branch_b_plan["slice_out_name"]), None)
            if b_slice_tensor is None or len(list(b_slice_tensor.shape)) != 4:
                continue
            b_slice_shape_nhwc = _permute_shape(list(b_slice_tensor.shape), perm_nchw_to_nhwc)
            conv_shape_nhwc = [int(v) for v in list(conv_out_tensor.shape)]
            if _broadcast_static_shapes(conv_shape_nhwc, b_slice_shape_nhwc) is None:
                continue
            add0_side_tensor = model_ir.tensors.get(str(branch_a_plan["add0_side_name"]), None)
            if (
                add0_side_tensor is None
                or not _is_fully_known_positive_shape(list(add0_side_tensor.shape))
                or _broadcast_static_shapes(
                    a_slice_shape_nhwc,
                    [int(v) for v in list(add0_side_tensor.shape)],
                ) is None
            ):
                continue

            chain_indices = {
                int(pre_idx),
                int(branch_a_plan["slice_idx"]),
                int(branch_b_plan["slice_idx"]),
                int(branch_a_plan["mul_idx"]),
                int(branch_a_plan["t1_idx"]),
                int(branch_a_plan["add0_idx"]),
                int(branch_a_plan["conv_idx"]),
                int(branch_a_plan["t2_idx"]),
                int(merge_add_idx),
            }
            if not _rewrite_mul_const_to_nhwc_if_needed(
                mul_idx=int(branch_a_plan["mul_idx"]),
                mul_op=branch_a_plan["mul_op"],
                data_input_name=str(branch_a_plan["slice_out_name"]),
                target_shape_nhwc=[int(v) for v in list(a_slice_shape_nhwc)],
                consumers=consumers,
                chain_indices=chain_indices,
            ):
                continue

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
                    graph_index=graph_index,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(slice_op.outputs[0]), None),
                    perm_nchw_to_nhwc,
                )
            if not apply_ok:
                continue

            # Branch-A rewires: remove T1/T2 and keep path in NHWC.
            _replace_tensor_inputs(
                model_ir,
                str(branch_a_plan["t1_out_name"]),
                str(branch_a_plan["mul_out_name"]),
                graph_index=graph_index,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(branch_a_plan["mul_out_name"]), None),
                perm_nchw_to_nhwc,
            )
            _replace_tensor_inputs(
                model_ir,
                str(branch_a_plan["t2_out_name"]),
                str(branch_a_plan["conv_out_name"]),
                graph_index=graph_index,
            )

            # Merge ADD output is now NHWC, then insert one post adapter back to NCHW.
            merge_out_name = str(merge_add_op.outputs[0])
            merge_out_tensor = model_ir.tensors.get(str(merge_out_name), None)
            if merge_out_tensor is None or len(list(merge_out_tensor.shape)) != 4:
                continue
            nhwc_merge_out_name = _unique_tensor_name(f"{merge_out_name}_nhwc")
            nhwc_shape = _permute_shape(list(merge_out_tensor.shape), perm_nchw_to_nhwc)
            model_ir.tensors[nhwc_merge_out_name] = TensorIR(
                name=nhwc_merge_out_name,
                dtype=str(merge_out_tensor.dtype),
                shape=[int(v) for v in list(nhwc_shape)],
                shape_signature=[int(v) for v in list(nhwc_shape)],
                data=None,
                is_variable=False,
                quantization=_clone_quantization(merge_out_tensor.quantization),
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=merge_add_op,
                new_outputs=[nhwc_merge_out_name],
                graph_index=graph_index,
            )

            perm_tensor_name = _find_or_create_nhwc_to_nchw_perm_tensor()
            post_adapter = OperatorIR(
                op_type="TRANSPOSE",
                inputs=[nhwc_merge_out_name, perm_tensor_name],
                outputs=[merge_out_name],
                options={},
            )

            remove_indices = {
                int(pre_idx),
                int(branch_a_plan["t1_idx"]),
                int(branch_a_plan["t2_idx"]),
            }
            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))
            graph_index.append_operator(post_adapter)

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_slice_muladd_conv_mergeadd_strict": int(optimized)}


def _optimize_transpose_slice_muladd_mergeadd_posttranspose_strict(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Strictly fold:
      T0(NHWC->NCHW) -> two channel SLICE branches
        branch-A: SLICE -> MUL -> T1(NCHW->NHWC) -> ...
        branch-B: SLICE ------------------------------+
      ... -> T2(NHWC->NCHW) -------------------------+-> ADD(y_nchw)
      y_nchw -> T3(NCHW->NHWC) (plus optional legacy NCHW consumers)

    into NHWC and keep a single localized legacy adapter when required.
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _find_or_create_nhwc_to_nchw_perm_tensor() -> str:
        target = np.asarray(perm_nhwc_to_nchw, dtype=np.int32)
        for tensor_name, tensor in model_ir.tensors.items():
            if (
                tensor is None
                or tensor.data is None
                or len(list(tensor.shape)) != 1
                or int(tensor.shape[0]) != 4
            ):
                continue
            data = np.asarray(tensor.data)
            if data.shape == (4,) and data.dtype == np.int32 and np.array_equal(data, target):
                return str(tensor_name)
        perm_name = _unique_tensor_name("__nhwc_to_nchw_perm_rank4__")
        model_ir.tensors[perm_name] = TensorIR(
            name=perm_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(target, dtype=np.int32),
            is_variable=False,
        )
        return perm_name

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
                graph_index=graph_index,
            )
        else:
            const_tensor.data = np.asarray(rotated)
            const_tensor.shape = [int(v) for v in list(rotated.shape)]
            const_tensor.shape_signature = [int(v) for v in list(rotated.shape)]
        return True

    def _find_reachable_to_nchw_merge_candidate(
        *,
        start_tensor_name: str,
        expected_merge_add_idx: Optional[int],
        model_outputs: set[str],
    ) -> Optional[Dict[str, Any]]:
        visited_tensors: set[str] = set()
        stack: List[str] = [str(start_tensor_name)]
        candidate: Optional[Dict[str, Any]] = None
        steps = 0

        while len(stack) > 0 and steps < 4096:
            steps += 1
            tensor_name = str(stack.pop())
            if tensor_name in visited_tensors:
                continue
            visited_tensors.add(tensor_name)

            user_indices = [int(v) for v in consumers.get(tensor_name, [])]
            for user_idx in user_indices:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == str(tensor_name)
                    and _read_transpose_perm(model_ir, user_op) == perm_nhwc_to_nchw
                ):
                    to_nchw_out_name = str(user_op.outputs[0])
                    to_nchw_out_users = [int(v) for v in consumers.get(to_nchw_out_name, [])]
                    if len(to_nchw_out_users) != 1:
                        continue
                    cand_merge_add_idx = int(to_nchw_out_users[0])
                    cand_merge_add_op = model_ir.operators[int(cand_merge_add_idx)]
                    if (
                        str(cand_merge_add_op.op_type) != "ADD"
                        or len(cand_merge_add_op.inputs) != 2
                        or len(cand_merge_add_op.outputs) != 1
                    ):
                        continue
                    if expected_merge_add_idx is not None and int(cand_merge_add_idx) != int(expected_merge_add_idx):
                        continue
                    this_candidate = {
                        "to_nchw_idx": int(user_idx),
                        "to_nchw_out_name": str(to_nchw_out_name),
                        "path_nhwc_tensor_name": str(tensor_name),
                        "merge_add_idx": int(cand_merge_add_idx),
                    }
                    if candidate is None:
                        candidate = this_candidate
                    else:
                        if int(candidate["to_nchw_idx"]) != int(this_candidate["to_nchw_idx"]):
                            return None

                for produced_name in [str(v) for v in list(user_op.outputs)]:
                    if produced_name in visited_tensors or produced_name in model_outputs:
                        continue
                    stack.append(str(produced_name))

        return candidate

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            if pre_out_name in model_outputs:
                continue

            slice_indices = [int(v) for v in consumers.get(pre_out_name, [])]
            if len(slice_indices) != 2:
                continue
            if not all(_is_axis1_channel_slice(model_ir.operators[int(v)]) for v in slice_indices):
                continue

            branch_a: Optional[Dict[str, Any]] = None
            branch_b: Optional[Dict[str, Any]] = None
            merge_add_idx: Optional[int] = None
            valid = True

            for slice_idx in sorted(slice_indices):
                slice_op = model_ir.operators[int(slice_idx)]
                slice_out_name = str(slice_op.outputs[0])
                if slice_out_name in model_outputs:
                    valid = False
                    break
                users1 = [int(v) for v in consumers.get(slice_out_name, []) if int(v) != int(slice_idx)]
                if len(users1) != 1:
                    valid = False
                    break
                first_idx = int(users1[0])
                first_op = model_ir.operators[int(first_idx)]

                if str(first_op.op_type) == "ADD" and len(first_op.inputs) == 2 and len(first_op.outputs) == 1:
                    if merge_add_idx is None:
                        merge_add_idx = int(first_idx)
                    elif int(merge_add_idx) != int(first_idx):
                        valid = False
                        break
                    if branch_b is not None:
                        valid = False
                        break
                    branch_b = {
                        "slice_idx": int(slice_idx),
                        "slice_out_name": str(slice_out_name),
                        "merge_add_idx": int(first_idx),
                    }
                    continue

                if str(first_op.op_type) != "MUL" or len(first_op.inputs) != 2 or len(first_op.outputs) != 1:
                    valid = False
                    break
                mul_idx = int(first_idx)
                mul_out_name = str(first_op.outputs[0])
                mul_users = [int(v) for v in consumers.get(mul_out_name, [])]
                if len(mul_users) != 1:
                    valid = False
                    break
                t1_idx = int(mul_users[0])
                t1_op = model_ir.operators[int(t1_idx)]
                if (
                    str(t1_op.op_type) != "TRANSPOSE"
                    or len(t1_op.inputs) < 2
                    or len(t1_op.outputs) != 1
                    or str(t1_op.inputs[0]) != mul_out_name
                    or _read_transpose_perm(model_ir, t1_op) != perm_nchw_to_nhwc
                ):
                    valid = False
                    break
                t1_out_name = str(t1_op.outputs[0])
                t1_users = [int(v) for v in consumers.get(t1_out_name, [])]
                if len(t1_users) != 1:
                    valid = False
                    break
                reachable = _find_reachable_to_nchw_merge_candidate(
                    start_tensor_name=str(t1_out_name),
                    expected_merge_add_idx=merge_add_idx,
                    model_outputs=model_outputs,
                )
                if reachable is None:
                    valid = False
                    break
                nhwc_to_nchw_idx = int(reachable["to_nchw_idx"])
                nhwc_to_nchw_out_name = str(reachable["to_nchw_out_name"])
                cand_merge_add_idx = int(reachable["merge_add_idx"])
                if merge_add_idx is None:
                    merge_add_idx = int(cand_merge_add_idx)
                elif int(merge_add_idx) != int(cand_merge_add_idx):
                    valid = False
                    break

                if branch_a is not None:
                    valid = False
                    break
                branch_a = {
                    "slice_idx": int(slice_idx),
                    "slice_out_name": str(slice_out_name),
                    "mul_idx": int(mul_idx),
                    "mul_op": first_op,
                    "mul_out_name": str(mul_out_name),
                    "t1_idx": int(t1_idx),
                    "t1_out_name": str(t1_out_name),
                    "path_nhwc_tensor_name": str(reachable["path_nhwc_tensor_name"]),
                    "to_nchw_idx": int(nhwc_to_nchw_idx),
                    "to_nchw_out_name": str(nhwc_to_nchw_out_name),
                    "merge_add_idx": int(cand_merge_add_idx),
                }

            if not valid or branch_a is None or branch_b is None or merge_add_idx is None:
                continue
            merge_add_op = model_ir.operators[int(merge_add_idx)]
            merge_out_name = str(merge_add_op.outputs[0])
            if merge_out_name in model_outputs:
                continue

            if set(int(v) for v in consumers.get(pre_out_name, [])) != set(int(v) for v in slice_indices):
                continue
            if set(int(v) for v in consumers.get(str(branch_b["slice_out_name"]), [])) != {int(merge_add_idx)}:
                continue
            if set(int(v) for v in consumers.get(str(branch_a["to_nchw_out_name"]), [])) != {int(merge_add_idx)}:
                continue

            # Merge output must have at least one NCHW->NHWC tail to fold.
            post_transpose_indices: List[int] = []
            post_transpose_out_names: List[str] = []
            post_mul_transpose_plans: List[Dict[str, Any]] = []
            legacy_consumer_indices: List[int] = []
            for user_idx in [int(v) for v in consumers.get(merge_out_name, [])]:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == merge_out_name
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    post_transpose_indices.append(int(user_idx))
                    post_transpose_out_names.append(str(user_op.outputs[0]))
                elif (
                    str(user_op.op_type) == "MUL"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                ):
                    mul_inputs = [str(v) for v in list(user_op.inputs)]
                    if str(merge_out_name) == mul_inputs[0]:
                        mul_data_input_index = 0
                        mul_const_input_index = 1
                    elif str(merge_out_name) == mul_inputs[1]:
                        mul_data_input_index = 1
                        mul_const_input_index = 0
                    else:
                        legacy_consumer_indices.append(int(user_idx))
                        continue
                    mul_const_name = str(mul_inputs[int(mul_const_input_index)])
                    mul_const_tensor = model_ir.tensors.get(str(mul_const_name), None)
                    if mul_const_tensor is None or mul_const_tensor.data is None:
                        legacy_consumer_indices.append(int(user_idx))
                        continue
                    mul_out_name = str(user_op.outputs[0])
                    if mul_out_name in model_outputs:
                        legacy_consumer_indices.append(int(user_idx))
                        continue
                    mul_out_users = [int(v) for v in consumers.get(mul_out_name, [])]
                    if len(mul_out_users) != 1:
                        legacy_consumer_indices.append(int(user_idx))
                        continue
                    mul_post_idx = int(mul_out_users[0])
                    mul_post_op = model_ir.operators[int(mul_post_idx)]
                    if (
                        str(mul_post_op.op_type) != "TRANSPOSE"
                        or len(mul_post_op.inputs) < 2
                        or len(mul_post_op.outputs) != 1
                        or str(mul_post_op.inputs[0]) != str(mul_out_name)
                        or _read_transpose_perm(model_ir, mul_post_op) != perm_nchw_to_nhwc
                        or str(mul_post_op.outputs[0]) in model_outputs
                    ):
                        legacy_consumer_indices.append(int(user_idx))
                        continue
                    post_mul_transpose_plans.append(
                        {
                            "mul_idx": int(user_idx),
                            "mul_data_input_index": int(mul_data_input_index),
                            "mul_const_name": str(mul_const_name),
                            "mul_out_name": str(mul_out_name),
                            "post_idx": int(mul_post_idx),
                            "post_out_name": str(mul_post_op.outputs[0]),
                        }
                    )
                else:
                    legacy_consumer_indices.append(int(user_idx))
            # Accept strict legacy-only tails as well. Even when merge output has
            # no NCHW->NHWC post adapters, we can still fold pre/T1/T2 transposes
            # and preserve existing NCHW consumers through one localized adapter.
            if (
                len(post_transpose_indices) == 0
                and len(post_mul_transpose_plans) == 0
                and len(legacy_consumer_indices) == 0
            ):
                continue

            b_slice_tensor = model_ir.tensors.get(str(branch_b["slice_out_name"]), None)
            a_nhwc_tensor = model_ir.tensors.get(str(branch_a["path_nhwc_tensor_name"]), None)
            if (
                b_slice_tensor is None
                or a_nhwc_tensor is None
                or len(list(b_slice_tensor.shape)) != 4
                or len(list(a_nhwc_tensor.shape)) != 4
            ):
                continue
            b_shape_nhwc = _permute_shape(list(b_slice_tensor.shape), perm_nchw_to_nhwc)
            a_shape_nhwc = [int(v) for v in list(a_nhwc_tensor.shape)]
            if _broadcast_static_shapes(a_shape_nhwc, b_shape_nhwc) is None:
                continue

            chain_indices: set[int] = {
                int(pre_idx),
                int(branch_a["slice_idx"]),
                int(branch_b["slice_idx"]),
                int(branch_a["mul_idx"]),
                int(branch_a["t1_idx"]),
                int(branch_a["to_nchw_idx"]),
                int(merge_add_idx),
            }
            chain_indices.update(int(v) for v in post_transpose_indices)
            chain_indices.update(int(v["mul_idx"]) for v in post_mul_transpose_plans)
            chain_indices.update(int(v["post_idx"]) for v in post_mul_transpose_plans)

            a_slice_tensor = model_ir.tensors.get(str(branch_a["slice_out_name"]), None)
            a_slice_shape_nhwc = (
                _permute_shape(list(a_slice_tensor.shape), perm_nchw_to_nhwc)
                if a_slice_tensor is not None and len(list(a_slice_tensor.shape)) == 4
                else None
            )
            if not _rewrite_mul_const_to_nhwc_if_needed(
                mul_idx=int(branch_a["mul_idx"]),
                mul_op=branch_a["mul_op"],
                data_input_name=str(branch_a["slice_out_name"]),
                target_shape_nhwc=a_slice_shape_nhwc,
                consumers=consumers,
                chain_indices=chain_indices,
            ):
                continue
            post_mul_rewrite_ok = True
            for post_mul_plan in post_mul_transpose_plans:
                post_mul_op = model_ir.operators[int(post_mul_plan["mul_idx"])]
                if not _rewrite_mul_const_to_nhwc_if_needed(
                    mul_idx=int(post_mul_plan["mul_idx"]),
                    mul_op=post_mul_op,
                    data_input_name=str(merge_out_name),
                    target_shape_nhwc=[int(v) for v in list(a_shape_nhwc)],
                    consumers=consumers,
                    chain_indices=chain_indices,
                ):
                    post_mul_rewrite_ok = False
                    break
            if not post_mul_rewrite_ok:
                continue

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
                    graph_index=graph_index,
                )
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(slice_op.outputs[0]), None),
                    perm_nchw_to_nhwc,
                )
            if not apply_ok:
                continue

            _replace_tensor_inputs(
                model_ir,
                str(branch_a["t1_out_name"]),
                str(branch_a["mul_out_name"]),
                graph_index=graph_index,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(branch_a["mul_out_name"]), None),
                perm_nchw_to_nhwc,
            )

            # Merge ADD in NHWC.
            merge_out_tensor_nchw = model_ir.tensors.get(merge_out_name, None)
            merge_out_shape_nchw = (
                [int(v) for v in list(merge_out_tensor_nchw.shape)]
                if merge_out_tensor_nchw is not None and merge_out_tensor_nchw.shape is not None
                else None
            )
            merge_out_sig_nchw = (
                [int(v) for v in list(merge_out_tensor_nchw.shape_signature)]
                if merge_out_tensor_nchw is not None and merge_out_tensor_nchw.shape_signature is not None
                else None
            )
            merge_out_dtype = str(merge_out_tensor_nchw.dtype) if merge_out_tensor_nchw is not None else None
            merge_out_quant = (
                _clone_quantization(merge_out_tensor_nchw.quantization)
                if merge_out_tensor_nchw is not None
                else None
            )
            _replace_tensor_inputs(
                model_ir,
                str(branch_a["to_nchw_out_name"]),
                str(branch_a["path_nhwc_tensor_name"]),
                graph_index=graph_index,
            )

            if len(post_transpose_out_names) > 0:
                canonical_nhwc_out_name = str(post_transpose_out_names[0])
            else:
                canonical_nhwc_out_name = _unique_tensor_name(f"{merge_out_name}_nhwc")
                if canonical_nhwc_out_name not in model_ir.tensors:
                    model_ir.tensors[canonical_nhwc_out_name] = TensorIR(
                        name=canonical_nhwc_out_name,
                        dtype=str(merge_out_dtype) if merge_out_dtype is not None else "FLOAT32",
                        shape=[],
                        shape_signature=[],
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(merge_out_quant),
                    )
            _set_operator_outputs(
                model_ir=model_ir,
                op=merge_add_op,
                new_outputs=[canonical_nhwc_out_name],
                graph_index=graph_index,
            )
            for alias_name in post_transpose_out_names[1:]:
                _replace_tensor_inputs(
                    model_ir,
                    alias_name,
                    canonical_nhwc_out_name,
                    graph_index=graph_index,
                )
            for post_mul_plan in post_mul_transpose_plans:
                post_mul_idx = int(post_mul_plan["mul_idx"])
                post_mul_op = model_ir.operators[int(post_mul_idx)]
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=post_mul_op,
                    input_index=int(post_mul_plan["mul_data_input_index"]),
                    new_input_name=str(canonical_nhwc_out_name),
                    graph_index=graph_index,
                )
                old_mul_out_name = str(post_mul_plan["mul_out_name"])
                post_out_name = str(post_mul_plan["post_out_name"])
                old_mul_tensor = model_ir.tensors.get(old_mul_out_name, None)
                post_out_tensor = model_ir.tensors.get(post_out_name, None)
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=post_mul_op,
                    new_outputs=[post_out_name],
                    graph_index=graph_index,
                )
                if old_mul_tensor is not None and post_out_tensor is not None:
                    post_out_tensor.dtype = str(old_mul_tensor.dtype)
                    post_out_tensor.quantization = _clone_quantization(old_mul_tensor.quantization)
                    post_out_tensor.shape = [int(v) for v in list(old_mul_tensor.shape)]
                    post_out_tensor.shape_signature = (
                        [int(v) for v in list(old_mul_tensor.shape_signature)]
                        if old_mul_tensor.shape_signature is not None
                        else [int(v) for v in list(old_mul_tensor.shape)]
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        post_out_tensor,
                        perm_nchw_to_nhwc,
                    )

            canonical_nhwc_tensor = model_ir.tensors.get(canonical_nhwc_out_name, None)
            if canonical_nhwc_tensor is not None:
                if merge_out_dtype is not None:
                    canonical_nhwc_tensor.dtype = str(merge_out_dtype)
                if merge_out_quant is not None:
                    canonical_nhwc_tensor.quantization = _clone_quantization(merge_out_quant)
                if merge_out_shape_nchw is not None and len(merge_out_shape_nchw) == 4:
                    canonical_nhwc_tensor.shape = _permute_shape(
                        list(merge_out_shape_nchw),
                        perm_nchw_to_nhwc,
                    )
                    base_sig = (
                        list(merge_out_sig_nchw)
                        if merge_out_sig_nchw is not None and len(merge_out_sig_nchw) == 4
                        else list(merge_out_shape_nchw)
                    )
                    canonical_nhwc_tensor.shape_signature = _permute_shape(
                        list(base_sig),
                        perm_nchw_to_nhwc,
                    )

            # Keep legacy NCHW consumers via one localized adapter.
            if len(legacy_consumer_indices) > 0 or merge_out_name in model_outputs:
                perm_tensor_name = _find_or_create_nhwc_to_nchw_perm_tensor()
                adapter_op = OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[canonical_nhwc_out_name, perm_tensor_name],
                    outputs=[merge_out_name],
                    options={},
                )
                graph_index.append_operator(adapter_op)

            remove_indices = {
                int(pre_idx),
                int(branch_a["t1_idx"]),
                int(branch_a["to_nchw_idx"]),
            }
            remove_indices.update(int(v) for v in post_transpose_indices)
            remove_indices.update(int(v["post_idx"]) for v in post_mul_transpose_plans)
            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_slice_muladd_mergeadd_posttranspose_strict": int(optimized)}


def _optimize_transpose_channel_slice_dual_add_bridges_strict(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
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
                graph_index=graph_index,
            )
        else:
            const_tensor.data = np.asarray(rotated)
            const_tensor.shape = [int(v) for v in list(rotated.shape)]
            const_tensor.shape_signature = [int(v) for v in list(rotated.shape)]

        return True

    while True:
        changed = False
        consumers = graph_index.consumers
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
                    graph_index=graph_index,
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
                        graph_index=graph_index,
                    )
                    remove_indices.add(int(plan["post_idx"]))
                else:
                    _replace_tensor_inputs(
                        model_ir,
                        str(plan["post_out_name"]),
                        str(plan["slice_out_name"]),
                        graph_index=graph_index,
                    )
                    remove_indices.add(int(plan["post_idx"]))

            if not apply_ok:
                continue

            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_channel_slice_dual_add_bridges_strict": int(optimized)}


def run_channel_slice_merge_layout_cleanup(
    model_ir: ModelIR,
    *,
    include_dual_add: bool = True,
    include_conv_merge: bool = True,
    include_posttranspose_merge: bool = True,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Run the adjacent strict channel-slice merge rewrites in legacy order."""

    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "MUL", "ADD"}
        slice_count = 0
        for visited, operator in enumerate(candidate_model.operators, start=1):
            op_type = str(operator.op_type)
            required.discard(op_type)
            if op_type == "SLICE":
                slice_count += 1
            if len(required) == 0 and slice_count >= 2:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _is_axis1_slice(candidate_model: ModelIR, op: OperatorIR) -> bool:
        if str(op.op_type) != "SLICE" or len(op.inputs) < 3 or len(op.outputs) != 1:
            return False
        begin = _read_const_ints_from_tensor(
            candidate_model.tensors.get(str(op.inputs[1]))
        )
        size = _read_const_ints_from_tensor(
            candidate_model.tensors.get(str(op.inputs[2]))
        )
        return bool(
            begin is not None
            and size is not None
            and len(begin) == 4
            and len(size) == 4
            and int(size[1]) > 0
            and int(begin[2]) == 0
            and int(begin[3]) == 0
        )

    def _prefixes(
        pass_state: ModelIRPassState,
    ) -> List[Tuple[int, List[int]]]:
        prefixes: List[Tuple[int, List[int]]] = []
        for pre_idx, pre_op in enumerate(pass_state.model_ir.operators):
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or _read_transpose_perm(pass_state.model_ir, pre_op)
                != perm_nhwc_to_nchw
            ):
                continue
            slice_indices = pass_state.graph_index.consumer_indices(
                str(pre_op.outputs[0])
            )
            if len(slice_indices) != 2:
                continue
            if all(
                _is_axis1_slice(
                    pass_state.model_ir,
                    pass_state.model_ir.operators[int(slice_idx)],
                )
                for slice_idx in slice_indices
            ):
                prefixes.append(
                    (int(pre_idx), [int(value) for value in slice_indices])
                )
        return prefixes

    def _single_user(
        pass_state: ModelIRPassState,
        tensor_name: str,
    ) -> Optional[Tuple[int, OperatorIR]]:
        users = pass_state.graph_index.consumer_indices(str(tensor_name))
        if len(users) != 1:
            return None
        index = int(users[0])
        return index, pass_state.model_ir.operators[index]

    def _inverse_transpose_output(
        pass_state: ModelIRPassState,
        tensor_name: str,
    ) -> Optional[Tuple[int, str]]:
        match = _single_user(pass_state, tensor_name)
        if match is None:
            return None
        index, op = match
        if (
            str(op.op_type) != "TRANSPOSE"
            or len(op.inputs) < 2
            or len(op.outputs) != 1
            or _read_transpose_perm(pass_state.model_ir, op)
            != perm_nchw_to_nhwc
        ):
            return None
        return int(index), str(op.outputs[0])

    def _add_after(
        pass_state: ModelIRPassState,
        tensor_name: str,
    ) -> Optional[Tuple[int, OperatorIR]]:
        match = _single_user(pass_state, tensor_name)
        if match is None:
            return None
        index, op = match
        if (
            str(op.op_type) != "ADD"
            or len(op.inputs) != 2
            or len(op.outputs) != 1
            or str(tensor_name) not in [str(value) for value in op.inputs]
        ):
            return None
        return int(index), op

    def _has_dual_add_candidate(pass_state: ModelIRPassState) -> bool:
        for _, slice_indices in _prefixes(pass_state):
            kinds: set[str] = set()
            valid = True
            for slice_idx in slice_indices:
                slice_op = pass_state.model_ir.operators[int(slice_idx)]
                slice_output = str(slice_op.outputs[0])
                first = _single_user(pass_state, slice_output)
                if first is None:
                    valid = False
                    break
                _, first_op = first
                if str(first_op.op_type) == "TRANSPOSE":
                    post = _inverse_transpose_output(pass_state, slice_output)
                    if post is None or _add_after(pass_state, post[1]) is None:
                        valid = False
                        break
                    kinds.add("direct")
                    continue
                if (
                    str(first_op.op_type) != "MUL"
                    or len(first_op.inputs) != 2
                    or len(first_op.outputs) != 1
                ):
                    valid = False
                    break
                post = _inverse_transpose_output(
                    pass_state,
                    str(first_op.outputs[0]),
                )
                if post is None or _add_after(pass_state, post[1]) is None:
                    valid = False
                    break
                kinds.add("mul")
            if valid and kinds == {"direct", "mul"}:
                return True
        return False

    def _has_conv_merge_candidate(pass_state: ModelIRPassState) -> bool:
        for _, slice_indices in _prefixes(pass_state):
            for branch_idx, slice_idx in enumerate(slice_indices):
                other_slice = pass_state.model_ir.operators[
                    int(slice_indices[1 - branch_idx])
                ]
                tensor_name = str(
                    pass_state.model_ir.operators[int(slice_idx)].outputs[0]
                )
                expected = ["MUL", "TRANSPOSE", "ADD", "CONV_2D", "TRANSPOSE"]
                valid = True
                for step_index, op_type in enumerate(expected):
                    match = _single_user(pass_state, tensor_name)
                    if match is None:
                        valid = False
                        break
                    _, op = match
                    if str(op.op_type) != op_type or len(op.outputs) != 1:
                        valid = False
                        break
                    if op_type == "TRANSPOSE":
                        expected_perm = (
                            perm_nchw_to_nhwc
                            if int(step_index) == 1
                            else perm_nhwc_to_nchw
                        )
                        if _read_transpose_perm(pass_state.model_ir, op) != expected_perm:
                            valid = False
                            break
                    tensor_name = str(op.outputs[0])
                if not valid:
                    continue
                merge = _add_after(pass_state, tensor_name)
                if merge is not None and str(other_slice.outputs[0]) in [
                    str(value) for value in merge[1].inputs
                ]:
                    return True
        return False

    def _has_posttranspose_merge_candidate(pass_state: ModelIRPassState) -> bool:
        for _, slice_indices in _prefixes(pass_state):
            slice_outputs = {
                str(pass_state.model_ir.operators[int(index)].outputs[0])
                for index in slice_indices
            }
            for slice_output in slice_outputs:
                first = _single_user(pass_state, slice_output)
                if first is None or str(first[1].op_type) != "MUL":
                    continue
                if len(first[1].outputs) != 1:
                    continue
                post = _inverse_transpose_output(
                    pass_state,
                    str(first[1].outputs[0]),
                )
                if post is None:
                    continue
                pending = [str(post[1])]
                visited: set[str] = set()
                for _ in range(12):
                    if len(pending) == 0:
                        break
                    tensor_name = pending.pop()
                    if tensor_name in visited:
                        continue
                    visited.add(tensor_name)
                    match = _single_user(pass_state, tensor_name)
                    if match is None:
                        continue
                    _, op = match
                    if str(op.op_type) == "ADD" and any(
                        str(value) in slice_outputs - {slice_output}
                        for value in op.inputs
                    ):
                        return True
                    if len(op.outputs) == 1:
                        pending.append(str(op.outputs[0]))
        return False

    def _run_dual_add(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_channel_slice_dual_add_bridges_strict(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_channel_slice_dual_add_bridges_strict",
                    0,
                )
            ),
        }

    def _run_conv_merge(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_slice_muladd_conv_mergeadd_strict(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_slice_muladd_conv_mergeadd_strict",
                    0,
                )
            ),
        }

    def _run_posttranspose_merge(
        pass_state: ModelIRPassState,
    ) -> Dict[str, int | bool]:
        stats = _optimize_transpose_slice_muladd_mergeadd_posttranspose_strict(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_slice_muladd_mergeadd_posttranspose_strict",
                    0,
                )
            ),
        }

    specs: List[PassSpec[ModelIRPassState]] = []
    if include_dual_add:
        specs.append(
            PassSpec(
                pass_id="layout.channel_slice_dual_add_strict",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_dual_add,
                precondition=_has_dual_add_candidate,
                priority=10,
                transactional=True,
            )
        )
    if include_conv_merge:
        specs.append(
            PassSpec(
                pass_id="layout.slice_muladd_conv_mergeadd_strict",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_conv_merge,
                precondition=_has_conv_merge_candidate,
                priority=20,
                transactional=True,
            )
        )
    if include_posttranspose_merge:
        specs.append(
            PassSpec(
                pass_id="layout.slice_muladd_mergeadd_posttranspose_strict",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_posttranspose_merge,
                precondition=_has_posttranspose_merge_candidate,
                priority=30,
                transactional=True,
            )
        )
    default_details = {
        "optimized_transpose_channel_slice_dual_add_bridges_strict": 0,
        "optimized_transpose_slice_muladd_conv_mergeadd_strict": 0,
        "optimized_transpose_slice_muladd_mergeadd_posttranspose_strict": 0,
    }
    if len(specs) == 0:
        return default_details

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details=default_details,
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
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

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        changed = False
        model_inputs = set(str(v) for v in model_ir.inputs)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx in graph_index.operator_indices("TRANSPOSE"):
            pre_op = model_ir.operators[int(pre_idx)]
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
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

            internal_users = graph_index.consumer_indices(internal_name)
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
                slice_users = graph_index.consumer_indices(slice_out)
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
                q_users = graph_index.consumer_indices(q_out)
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
                dq_users = graph_index.consumer_indices(dq_out)
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
            concat_users = graph_index.consumer_indices(concat_out)
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
            post_indices = graph_index.consumer_indices(q_concat_out)
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
                    graph_index=graph_index,
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
                graph_index=graph_index,
            )
            for alias_name in post_output_names[1:]:
                _replace_tensor_inputs(
                    model_ir,
                    alias_name,
                    canonical_post_output,
                    graph_index=graph_index,
                )

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
                graph_index.remove_operator(int(remove_idx))
                if int(remove_idx) in post_indices:
                    removed_post_transposes += 1
            removed_boundary += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if removed_boundary > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "removed_boundary_input_transpose_stridedslice_blocks": int(removed_boundary),
        "rewritten_boundary_stridedslices": int(rewritten_slices),
        "rewritten_boundary_qdq_concat_axis": int(rewritten_concat),
        "removed_boundary_post_transposes": int(removed_post_transposes),
    }
