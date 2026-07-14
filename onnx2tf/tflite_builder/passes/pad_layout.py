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
    _append_tensor_lineage_event,
    _broadcast_static_shapes,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    candidate = str(base)
    suffix = 0
    while candidate in model_ir.tensors:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def repair_channel_last_inputs_for_channel_first_pad(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Insert an NHWC->NCHW adapter when a Pad retains its ONNX NCHW contract.

    Boundary Slice propagation can move the Pad input to NHWC after lowering,
    while the static output shape and paddings remain in ONNX NCHW order.  The
    repair is accepted only when the full input/output/padding shape equation
    proves that exact mismatch.  Names and model-specific patterns are not
    considered.
    """

    repaired = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        changed = False
        for pad_index in graph_index.operator_indices_for_types(
            {"PAD", "PADV2", "MIRROR_PAD"}
        ):
            pad_op = model_ir.operators[int(pad_index)]
            if len(pad_op.inputs) < 2 or len(pad_op.outputs) != 1:
                continue

            input_name = str(pad_op.inputs[0])
            pads_name = str(pad_op.inputs[1])
            output_name = str(pad_op.outputs[0])
            input_tensor = model_ir.tensors.get(input_name, None)
            pads_tensor = model_ir.tensors.get(pads_name, None)
            output_tensor = model_ir.tensors.get(output_name, None)
            if (
                input_tensor is None
                or pads_tensor is None
                or pads_tensor.data is None
                or output_tensor is None
                or str(input_tensor.logical_layout).upper() != "NHWC"
            ):
                continue

            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            try:
                pad_pairs = np.asarray(pads_tensor.data, dtype=np.int64).reshape(4, 2)
            except (TypeError, ValueError):
                continue
            if (
                len(input_shape) != 4
                or len(output_shape) != 4
                or any(int(v) <= 0 for v in input_shape + output_shape)
                or np.any(pad_pairs < 0)
            ):
                continue

            expected_nchw_input = [
                int(output_shape[axis])
                - int(pad_pairs[axis, 0])
                - int(pad_pairs[axis, 1])
                for axis in range(4)
            ]
            expected_nhwc_input = [
                int(expected_nchw_input[0]),
                int(expected_nchw_input[2]),
                int(expected_nchw_input[3]),
                int(expected_nchw_input[1]),
            ]
            if (
                any(int(v) <= 0 for v in expected_nchw_input)
                or input_shape != expected_nhwc_input
                or input_shape == expected_nchw_input
            ):
                continue

            adapter_name = _unique_tensor_name(
                model_ir,
                f"{output_name}_pad_input_nchw",
            )
            perm_name = _unique_tensor_name(
                model_ir,
                f"{output_name}_pad_input_nchw_perm",
            )
            input_signature = (
                [int(v) for v in list(input_tensor.shape_signature)]
                if input_tensor.shape_signature is not None
                else list(input_shape)
            )
            adapter_quantization = _clone_quantization(input_tensor.quantization)
            if isinstance(adapter_quantization, QuantParamIR):
                old_axis = int(adapter_quantization.quantized_dimension)
                if 0 <= old_axis < 4:
                    adapter_quantization.quantized_dimension = int(
                        [0, 3, 1, 2].index(old_axis)
                    )

            model_ir.tensors[adapter_name] = TensorIR(
                name=adapter_name,
                dtype=str(input_tensor.dtype),
                shape=list(expected_nchw_input),
                shape_signature=[
                    int(input_signature[0]),
                    int(input_signature[3]),
                    int(input_signature[1]),
                    int(input_signature[2]),
                ],
                quantization=adapter_quantization,
                logical_layout="NCHW",
                physical_layout="NCHW",
                onnx_tensor_name=input_tensor.onnx_tensor_name,
            )
            model_ir.tensors[perm_name] = TensorIR(
                name=perm_name,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray([0, 3, 1, 2], dtype=np.int32),
                is_variable=False,
            )
            transpose_op = OperatorIR(
                op_type="TRANSPOSE",
                inputs=[input_name, perm_name],
                outputs=[adapter_name],
            )
            updated_inputs = [str(v) for v in list(pad_op.inputs)]
            updated_inputs[0] = adapter_name
            graph_index.replace_operator_inputs(int(pad_index), updated_inputs)
            graph_index.insert_operator(int(pad_index), transpose_op)
            _append_tensor_lineage_event(
                model_ir=model_ir,
                event={
                    "kind": "replace_input",
                    "operator_type": str(pad_op.op_type),
                    "from": input_name,
                    "to": adapter_name,
                    "reason": "channel_last_input_for_channel_first_pad",
                },
            )
            repaired += 1
            changed = True
            break

        if not changed:
            break

    if repaired > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)

    return {"repaired_channel_last_inputs_for_channel_first_pad": int(repaired)}


def _optimize_transpose_pad_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Eliminate NHWC<->NCHW transpose wrappers around PAD/MIRROR_PAD when they can run in NHWC.

    Target:
      x_nhwc --T(0,3,1,2)--> x_nchw
      PAD|MIRROR_PAD(x_nchw, pads_nchw, [const]) -> p_nchw
      p_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      PAD|MIRROR_PAD(x_nhwc, pads_nhwc, [const]) -> y_nhwc

    Safety:
    - Exact NCHW/NHWC inverse transpose pair.
    - PAD output consumed only by the post-transpose.
    - Static 4x2 pads tensor available.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

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

            pad_output_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if pad_output_name in model_outputs:
                continue

            pad_idx = producers.get(str(pad_output_name), None)
            if pad_idx is None:
                continue
            pad_op = model_ir.operators[int(pad_idx)]
            if (
                str(pad_op.op_type) not in {"PAD", "MIRROR_PAD"}
                or len(pad_op.inputs) < 2
                or len(pad_op.outputs) != 1
                or str(pad_op.outputs[0]) != str(pad_output_name)
            ):
                continue

            # Keep this rewrite strict to avoid changing mixed-layout fanout behavior.
            if set(int(v) for v in consumers.get(str(pad_output_name), [])) != {int(post_idx)}:
                continue

            pre_nchw_name = str(pad_op.inputs[0])
            pre_idx = producers.get(str(pre_nchw_name), None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != str(pre_nchw_name)
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            ):
                continue
            if pre_nchw_name in model_outputs:
                continue

            pads_tensor_name = str(pad_op.inputs[1])
            pads_tensor = model_ir.tensors.get(str(pads_tensor_name), None)
            if pads_tensor is None or pads_tensor.data is None:
                continue
            try:
                pads_array = np.asarray(pads_tensor.data)
                pads_pairs = np.asarray(pads_array).reshape(4, 2)
            except Exception:
                continue
            if int(pads_pairs.size) != 8:
                continue

            # Rewire PAD input to NHWC producer.
            new_pad_inputs = [str(pre_op.inputs[0])]
            new_pad_inputs.extend(str(v) for v in list(pad_op.inputs)[1:])
            _set_operator_inputs(
                model_ir=model_ir,
                op=pad_op,
                new_inputs=new_pad_inputs,
                graph_index=graph_index,
            )

            # Convert pad spec NCHW [N,C,H,W] -> NHWC [N,H,W,C].
            pads_nhwc = np.asarray(
                [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
                dtype=pads_pairs.dtype,
            )
            pads_tensor.data = np.asarray(pads_nhwc)
            pads_tensor.shape = [4, 2]
            pads_tensor.shape_signature = [4, 2]

            # Bypass post-transpose by writing PAD directly to post output tensor.
            _set_operator_outputs(
                model_ir=model_ir,
                op=pad_op,
                new_outputs=[post_output_name],
                graph_index=graph_index,
            )

            old_pad_tensor = model_ir.tensors.get(str(pad_output_name), None)
            post_output_tensor = model_ir.tensors.get(str(post_output_name), None)
            if old_pad_tensor is not None and post_output_tensor is not None:
                post_output_tensor.dtype = str(old_pad_tensor.dtype)
                post_output_tensor.quantization = _clone_quantization(old_pad_tensor.quantization)
                post_output_tensor.shape = [int(v) for v in list(old_pad_tensor.shape)]
                post_output_tensor.shape_signature = (
                    [int(v) for v in list(old_pad_tensor.shape_signature)]
                    if old_pad_tensor.shape_signature is not None
                    else [int(v) for v in list(old_pad_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_output_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = {int(post_idx)}
            pre_remaining_users = [int(v) for v in consumers.get(str(pre_nchw_name), []) if int(v) != int(pad_idx)]
            if len(pre_remaining_users) == 0:
                remove_indices.add(int(pre_idx))
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
    return {"optimized_transpose_pad_prepost_nhwc_chains": int(rewritten)}


def _optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Eliminate NHWC<->NCHW transpose wrappers around UNARY->PAD/MIRROR_PAD tails.

    Target:
      x_nhwc --T(0,3,1,2)--> x_nchw
      x_nchw --UNARY--> u_nchw
      PAD|MIRROR_PAD(u_nchw, pads_nchw, [const]) -> p_nchw
      p_nchw --T(0,2,3,1)--> y_nhwc

    Rewrite:
      x_nhwc --UNARY--> u_nhwc
      PAD|MIRROR_PAD(u_nhwc, pads_nhwc, [const]) -> y_nhwc

    Notes:
    - Legacy side consumers of `u_nchw` are preserved through one local
      NHWC->NCHW transpose adapter.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    adapter_perm_name = "__unary_pad_tail_nhwc_to_nchw_perm_rank4__"
    unary_passthrough_ops = {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "GELU",
        "ABS",
        "NEG",
        "SQRT",
        "EXP",
        "FLOOR",
        "CEIL",
    }

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

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

            pad_output_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if pad_output_name in model_outputs:
                continue

            pad_idx = producers.get(str(pad_output_name), None)
            if pad_idx is None:
                continue
            pad_op = model_ir.operators[int(pad_idx)]
            if (
                str(pad_op.op_type) not in {"PAD", "MIRROR_PAD"}
                or len(pad_op.inputs) < 2
                or len(pad_op.outputs) != 1
                or str(pad_op.outputs[0]) != str(pad_output_name)
            ):
                continue
            if set(int(v) for v in consumers.get(str(pad_output_name), [])) != {int(post_idx)}:
                continue

            unary_output_name = str(pad_op.inputs[0])
            if unary_output_name in model_outputs:
                continue
            unary_idx = producers.get(str(unary_output_name), None)
            if unary_idx is None:
                continue
            unary_op = model_ir.operators[int(unary_idx)]
            if (
                str(unary_op.op_type) not in unary_passthrough_ops
                or len(unary_op.inputs) != 1
                or len(unary_op.outputs) != 1
                or str(unary_op.outputs[0]) != str(unary_output_name)
            ):
                continue

            pre_nchw_name = str(unary_op.inputs[0])
            pre_idx = producers.get(str(pre_nchw_name), None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != str(pre_nchw_name)
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            ):
                continue
            if set(int(v) for v in consumers.get(str(pre_nchw_name), [])) != {int(unary_idx)}:
                continue

            pre_input_name = str(pre_op.inputs[0])

            pads_tensor_name = str(pad_op.inputs[1])
            pads_tensor = model_ir.tensors.get(str(pads_tensor_name), None)
            if pads_tensor is None or pads_tensor.data is None:
                continue
            try:
                pads_pairs = np.asarray(pads_tensor.data).reshape(4, 2)
            except Exception:
                continue
            if int(pads_pairs.size) != 8:
                continue

            unary_users = [int(v) for v in consumers.get(str(unary_output_name), [])]
            legacy_consumer_slots: List[Tuple[int, int]] = []
            for user_idx in unary_users:
                if int(user_idx) == int(pad_idx):
                    continue
                user_op = model_ir.operators[int(user_idx)]
                for input_index, input_name in enumerate(list(user_op.inputs)):
                    if str(input_name) == str(unary_output_name):
                        legacy_consumer_slots.append((int(id(user_op)), int(input_index)))

            adapter_input_name = str(unary_output_name)
            old_unary_tensor = model_ir.tensors.get(str(unary_output_name), None)
            if len(legacy_consumer_slots) > 0:
                if old_unary_tensor is None:
                    continue
                unary_shape = [int(v) for v in list(old_unary_tensor.shape)] if old_unary_tensor.shape is not None else None
                unary_signature = (
                    [int(v) for v in list(old_unary_tensor.shape_signature)]
                    if old_unary_tensor.shape_signature is not None
                    else ([int(v) for v in list(old_unary_tensor.shape)] if old_unary_tensor.shape is not None else None)
                )
                unary_shape_nhwc = _permute_shape(unary_shape, perm_nchw_to_nhwc)
                unary_signature_nhwc = _permute_shape(unary_signature, perm_nchw_to_nhwc)
                if unary_shape_nhwc is None or unary_signature_nhwc is None:
                    continue
                adapter_input_name = _unique_tensor_name(f"{unary_output_name}_nhwc")
                model_ir.tensors[adapter_input_name] = TensorIR(
                    name=adapter_input_name,
                    dtype=str(old_unary_tensor.dtype),
                    shape=[int(v) for v in list(unary_shape_nhwc)],
                    shape_signature=[int(v) for v in list(unary_signature_nhwc)],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(old_unary_tensor.quantization),
                )
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=unary_op,
                    new_outputs=[str(adapter_input_name)],
                    graph_index=graph_index,
                )
            else:
                _permute_tensor_metadata_if_rank_matches(
                    old_unary_tensor,
                    perm_nchw_to_nhwc,
                )

            _set_operator_inputs(
                model_ir=model_ir,
                op=unary_op,
                new_inputs=[str(pre_input_name)],
                graph_index=graph_index,
            )

            if len(legacy_consumer_slots) > 0:
                op_index_by_id = {int(id(op)): int(op_idx) for op_idx, op in enumerate(model_ir.operators)}
                rank = (
                    int(len(list(old_unary_tensor.shape)))
                    if old_unary_tensor is not None and old_unary_tensor.shape is not None
                    else 4
                )
                rewritten_legacy_slots: set[Tuple[int, int]] = set()
                for consumer_op_id, input_index in list(legacy_consumer_slots):
                    if int(input_index) != 0:
                        continue
                    consumer_index = op_index_by_id.get(int(consumer_op_id), None)
                    if consumer_index is None:
                        continue
                    if int(consumer_index) < 0 or int(consumer_index) >= len(model_ir.operators):
                        continue
                    consumer_op = model_ir.operators[int(consumer_index)]
                    if (
                        str(consumer_op.op_type) != "MEAN"
                        or len(consumer_op.inputs) < 2
                        or len(consumer_op.outputs) != 1
                        or str(consumer_op.inputs[0]) != str(unary_output_name)
                    ):
                        continue
                    if int(rank) != 4:
                        continue
                    mean_out_name = str(consumer_op.outputs[0])
                    if mean_out_name in model_outputs:
                        continue
                    mean_out_users = [int(v) for v in consumers.get(mean_out_name, [])]
                    if len(mean_out_users) != 1:
                        continue
                    tail_op = model_ir.operators[int(mean_out_users[0])]
                    if (
                        str(tail_op.op_type) != "RESHAPE"
                        or len(tail_op.inputs) < 2
                        or len(tail_op.outputs) != 1
                        or str(tail_op.inputs[0]) != mean_out_name
                        or _read_const_ints_from_tensor(model_ir.tensors.get(str(tail_op.inputs[1]), None)) is None
                    ):
                        continue
                    axes_name = str(consumer_op.inputs[1])
                    axes_tensor = model_ir.tensors.get(axes_name, None)
                    axes_values = _read_const_ints_from_tensor(axes_tensor)
                    if axes_values is None or len(axes_values) == 0:
                        continue
                    normalized_axes: List[int] = []
                    valid_axes = True
                    for axis in axes_values:
                        axis_value = int(axis)
                        if axis_value < 0:
                            axis_value += int(rank)
                        if axis_value < 0 or axis_value >= int(rank):
                            valid_axes = False
                            break
                        normalized_axes.append(int(axis_value))
                    if not valid_axes:
                        continue
                    mapped_axes = [int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes]
                    _write_const_ints_to_tensor(axes_tensor, [int(v) for v in mapped_axes])
                    _replace_operator_input_at(
                        model_ir=model_ir,
                        op=consumer_op,
                        input_index=0,
                        new_input_name=str(adapter_input_name),
                        graph_index=graph_index,
                    )
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(mean_out_name, None),
                        perm_nchw_to_nhwc,
                    )
                    rewritten_legacy_slots.add((int(consumer_op_id), int(input_index)))
                if len(rewritten_legacy_slots) > 0:
                    legacy_consumer_slots = [
                        (int(consumer_op_id), int(input_index))
                        for consumer_op_id, input_index in legacy_consumer_slots
                        if (int(consumer_op_id), int(input_index)) not in rewritten_legacy_slots
                    ]

            pad_inputs = [str(v) for v in list(pad_op.inputs)]
            pad_inputs[0] = str(adapter_input_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=pad_op,
                new_inputs=pad_inputs,
                graph_index=graph_index,
            )

            pads_nhwc = np.asarray(
                [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
                dtype=pads_pairs.dtype,
            )
            pads_tensor.data = np.asarray(pads_nhwc)
            pads_tensor.shape = [4, 2]
            pads_tensor.shape_signature = [4, 2]

            _set_operator_outputs(
                model_ir=model_ir,
                op=pad_op,
                new_outputs=[post_output_name],
                graph_index=graph_index,
            )

            old_pad_tensor = model_ir.tensors.get(str(pad_output_name), None)
            post_output_tensor = model_ir.tensors.get(str(post_output_name), None)
            if old_pad_tensor is not None and post_output_tensor is not None:
                post_output_tensor.dtype = str(old_pad_tensor.dtype)
                post_output_tensor.quantization = _clone_quantization(old_pad_tensor.quantization)
                post_output_tensor.shape = [int(v) for v in list(old_pad_tensor.shape)]
                post_output_tensor.shape_signature = (
                    [int(v) for v in list(old_pad_tensor.shape_signature)]
                    if old_pad_tensor.shape_signature is not None
                    else [int(v) for v in list(old_pad_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_output_tensor,
                    perm_nchw_to_nhwc,
                )

            for remove_idx in sorted([int(pre_idx), int(post_idx)], reverse=True):
                graph_index.remove_operator(int(remove_idx))

            if len(legacy_consumer_slots) > 0:
                if adapter_perm_name not in model_ir.tensors:
                    model_ir.tensors[adapter_perm_name] = TensorIR(
                        name=adapter_perm_name,
                        dtype="INT32",
                        shape=[4],
                        shape_signature=[4],
                        data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                        is_variable=False,
                        quantization=None,
                    )
                op_index_by_id = {int(id(op)): int(op_idx) for op_idx, op in enumerate(model_ir.operators)}
                slot_targets: List[Tuple[int, int]] = []
                for consumer_op_id, input_index in legacy_consumer_slots:
                    consumer_index = op_index_by_id.get(int(consumer_op_id), None)
                    if consumer_index is None:
                        continue
                    if int(consumer_index) < 0 or int(consumer_index) >= len(model_ir.operators):
                        continue
                    consumer_op = model_ir.operators[int(consumer_index)]
                    if int(input_index) < 0 or int(input_index) >= len(consumer_op.inputs):
                        continue
                    if str(consumer_op.inputs[int(input_index)]) != str(unary_output_name):
                        continue
                    slot_targets.append((int(consumer_index), int(input_index)))
                if len(slot_targets) > 0:
                    graph_index.insert_operator(
                        int(min(int(v[0]) for v in slot_targets)),
                        OperatorIR(
                            op_type="TRANSPOSE",
                            inputs=[str(adapter_input_name), str(adapter_perm_name)],
                            outputs=[str(unary_output_name)],
                        ),
                    )

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains": int(rewritten)}


def _optimize_transpose_pad_mul_posttranspose_add_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Eliminate strict NHWC<->NCHW round-trips around:
      T(NHWC->NCHW) -> PAD|MIRROR_PAD -> MUL(const) -> T(NCHW->NHWC) -> ADD(const)

    Rewrite:
      PAD|MIRROR_PAD(NHWC pads) -> MUL(const_nhwc) -> ADD(const_nhwc)
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _rewrite_rank4_mul_const_to_nhwc(
        *,
        mul_op: OperatorIR,
        mul_idx: int,
        const_input_index: int,
        const_input_name: str,
        target_nhwc_shape: Optional[List[int]],
        chain_index_set: set[int],
    ) -> bool:
        const_tensor = model_ir.tensors.get(str(const_input_name), None)
        if const_tensor is None or const_tensor.data is None:
            return False
        const_data = np.asarray(const_tensor.data)
        if int(const_data.size) == 1:
            return True

        target_shape = (
            [int(v) for v in list(target_nhwc_shape)]
            if _is_fully_known_positive_shape(target_nhwc_shape)
            else None
        )

        rotated_data: Optional[np.ndarray] = None
        if int(const_data.ndim) == 4:
            as_is_shape = [int(v) for v in list(const_data.shape)]
            if target_shape is not None and _broadcast_static_shapes(target_shape, as_is_shape) is not None:
                return True
            rotated_candidate = np.transpose(const_data, perm_nchw_to_nhwc).astype(const_data.dtype, copy=False)
            rotated_shape = [int(v) for v in list(rotated_candidate.shape)]
            if target_shape is not None:
                if _broadcast_static_shapes(target_shape, rotated_shape) is None:
                    return False
            rotated_data = np.asarray(rotated_candidate)
        else:
            if target_shape is None:
                return False
            side_shape = [int(v) for v in list(const_data.shape)]
            if _broadcast_static_shapes(target_shape, side_shape) is None:
                return False
            return True

        if rotated_data is None:
            return False

        const_users = graph_index.consumer_indices(str(const_input_name))
        shared_outside_chain = any(int(v) not in chain_index_set for v in const_users)

        target_name = str(const_input_name)
        if shared_outside_chain:
            target_name = _unique_tensor_name(model_ir, f"{const_input_name}_nhwc")
            model_ir.tensors[target_name] = TensorIR(
                name=target_name,
                dtype=str(const_tensor.dtype),
                shape=[int(v) for v in list(rotated_data.shape)],
                shape_signature=[int(v) for v in list(rotated_data.shape)],
                data=np.asarray(rotated_data),
                is_variable=False,
                quantization=_clone_quantization(const_tensor.quantization),
            )
        else:
            const_tensor.data = np.asarray(rotated_data)
            const_tensor.shape = [int(v) for v in list(rotated_data.shape)]
            const_tensor.shape_signature = [int(v) for v in list(rotated_data.shape)]

        if target_name != str(const_input_name):
            mul_inputs = [str(v) for v in list(mul_op.inputs)]
            mul_inputs[int(const_input_index)] = str(target_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=mul_op,
                new_inputs=mul_inputs,
                graph_index=graph_index,
            )
        return True

    def _rewrite_pad_spec_to_nhwc(
        *,
        pad_op: OperatorIR,
        pad_idx: int,
        pads_input_name: str,
        chain_index_set: set[int],
    ) -> bool:
        pads_tensor = model_ir.tensors.get(str(pads_input_name), None)
        if pads_tensor is None or pads_tensor.data is None:
            return False
        try:
            pads_pairs = np.asarray(pads_tensor.data).reshape(4, 2)
        except Exception:
            return False
        if int(pads_pairs.size) != 8:
            return False
        pads_nhwc = np.asarray(
            [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
            dtype=pads_pairs.dtype,
        )

        pads_users = graph_index.consumer_indices(str(pads_input_name))
        shared_outside_chain = any(int(v) not in chain_index_set for v in pads_users)
        target_name = str(pads_input_name)
        if shared_outside_chain:
            target_name = _unique_tensor_name(model_ir, f"{pads_input_name}_nhwc")
            model_ir.tensors[target_name] = TensorIR(
                name=target_name,
                dtype=str(pads_tensor.dtype),
                shape=[4, 2],
                shape_signature=[4, 2],
                data=np.asarray(pads_nhwc),
                is_variable=False,
                quantization=_clone_quantization(pads_tensor.quantization),
            )
        else:
            pads_tensor.data = np.asarray(pads_nhwc)
            pads_tensor.shape = [4, 2]
            pads_tensor.shape_signature = [4, 2]

        if target_name != str(pads_input_name):
            pad_inputs = [str(v) for v in list(pad_op.inputs)]
            pad_inputs[1] = str(target_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=pad_op,
                new_inputs=pad_inputs,
                graph_index=graph_index,
            )
        return True

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

            mul_output_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if mul_output_name in model_outputs or post_output_name in model_outputs:
                continue

            mul_idx = producers.get(str(mul_output_name), None)
            if mul_idx is None:
                continue
            mul_op = model_ir.operators[int(mul_idx)]
            if (
                str(mul_op.op_type) != "MUL"
                or len(mul_op.inputs) != 2
                or len(mul_op.outputs) != 1
                or str(mul_op.outputs[0]) != str(mul_output_name)
            ):
                continue
            if set(int(v) for v in consumers.get(str(mul_output_name), [])) != {int(post_idx)}:
                continue

            mul_inputs = [str(v) for v in list(mul_op.inputs)]
            data_input_index: Optional[int] = None
            const_input_index: Optional[int] = None
            data_input_name: Optional[str] = None
            const_input_name: Optional[str] = None
            for idx, input_name in enumerate(mul_inputs):
                input_producer_idx = producers.get(str(input_name), None)
                if input_producer_idx is not None:
                    input_producer_op = model_ir.operators[int(input_producer_idx)]
                    if (
                        str(input_producer_op.op_type) in {"PAD", "MIRROR_PAD"}
                        and len(input_producer_op.outputs) == 1
                        and str(input_producer_op.outputs[0]) == str(input_name)
                    ):
                        data_input_index = int(idx)
                        data_input_name = str(input_name)
                        continue
                candidate_const = model_ir.tensors.get(str(input_name), None)
                if candidate_const is not None and candidate_const.data is not None:
                    const_input_index = int(idx)
                    const_input_name = str(input_name)

            if (
                data_input_index is None
                or data_input_name is None
                or const_input_index is None
                or const_input_name is None
            ):
                continue

            pad_idx = producers.get(str(data_input_name), None)
            if pad_idx is None:
                continue
            pad_op = model_ir.operators[int(pad_idx)]
            if (
                str(pad_op.op_type) not in {"PAD", "MIRROR_PAD"}
                or len(pad_op.inputs) < 2
                or len(pad_op.outputs) != 1
                or str(pad_op.outputs[0]) != str(data_input_name)
            ):
                continue
            if set(int(v) for v in consumers.get(str(data_input_name), [])) != {int(mul_idx)}:
                continue

            pre_nchw_name = str(pad_op.inputs[0])
            pre_idx = producers.get(str(pre_nchw_name), None)
            if pre_idx is None:
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != str(pre_nchw_name)
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
                or pre_nchw_name in model_outputs
            ):
                continue
            if set(int(v) for v in consumers.get(str(pre_nchw_name), [])) != {int(pad_idx)}:
                continue

            post_users = [int(v) for v in consumers.get(str(post_output_name), [])]
            if len(post_users) != 1:
                continue
            add_idx = int(post_users[0])
            add_op = model_ir.operators[int(add_idx)]
            if str(add_op.op_type) != "ADD" or len(add_op.inputs) != 2 or len(add_op.outputs) != 1:
                continue
            add_inputs = [str(v) for v in list(add_op.inputs)]
            add_side_name: Optional[str] = None
            if add_inputs[0] == str(post_output_name):
                add_side_name = str(add_inputs[1])
            elif add_inputs[1] == str(post_output_name):
                add_side_name = str(add_inputs[0])
            if add_side_name is None:
                continue
            add_side_tensor = model_ir.tensors.get(str(add_side_name), None)
            if add_side_tensor is None or add_side_tensor.data is None:
                continue

            target_nhwc_shape = (
                [int(v) for v in list(model_ir.tensors[str(post_output_name)].shape)]
                if str(post_output_name) in model_ir.tensors
                else None
            )
            if _is_fully_known_positive_shape(target_nhwc_shape):
                if _broadcast_static_shapes(
                    [int(v) for v in list(target_nhwc_shape)],
                    [int(v) for v in list(add_side_tensor.shape)],
                ) is None:
                    continue

            chain_index_set = {
                int(pre_idx),
                int(pad_idx),
                int(mul_idx),
                int(post_idx),
                int(add_idx),
            }

            if not _rewrite_rank4_mul_const_to_nhwc(
                mul_op=mul_op,
                mul_idx=int(mul_idx),
                const_input_index=int(const_input_index),
                const_input_name=str(const_input_name),
                target_nhwc_shape=target_nhwc_shape,
                chain_index_set=chain_index_set,
            ):
                continue

            if not _rewrite_pad_spec_to_nhwc(
                pad_op=pad_op,
                pad_idx=int(pad_idx),
                pads_input_name=str(pad_op.inputs[1]),
                chain_index_set=chain_index_set,
            ):
                continue

            pre_input_name = str(pre_op.inputs[0])
            pad_inputs = [str(v) for v in list(pad_op.inputs)]
            pad_inputs[0] = str(pre_input_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=pad_op,
                new_inputs=pad_inputs,
                graph_index=graph_index,
            )

            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(data_input_name), None),
                perm_nchw_to_nhwc,
            )

            old_mul_tensor = model_ir.tensors.get(str(mul_output_name), None)
            _set_operator_outputs(
                model_ir=model_ir,
                op=mul_op,
                new_outputs=[str(post_output_name)],
                graph_index=graph_index,
            )
            post_output_tensor = model_ir.tensors.get(str(post_output_name), None)
            if old_mul_tensor is not None and post_output_tensor is not None:
                post_output_tensor.dtype = str(old_mul_tensor.dtype)
                post_output_tensor.quantization = _clone_quantization(old_mul_tensor.quantization)
                post_output_tensor.shape = [int(v) for v in list(old_mul_tensor.shape)]
                post_output_tensor.shape_signature = (
                    [int(v) for v in list(old_mul_tensor.shape_signature)]
                    if old_mul_tensor.shape_signature is not None
                    else [int(v) for v in list(old_mul_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_output_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = {int(post_idx)}
            pre_remaining_users = [int(v) for v in consumers.get(str(pre_nchw_name), []) if int(v) != int(pad_idx)]
            if len(pre_remaining_users) == 0:
                remove_indices.add(int(pre_idx))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_pad_mul_posttranspose_add_nhwc_chains": int(rewritten)}


def run_pad_mul_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run the strict Pad/Mul/PostTranspose/Add rewrite as an ordered pass."""

    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "MUL", "ADD"}
        has_pad = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            op_type = str(operator.op_type)
            required.discard(op_type)
            has_pad = has_pad or op_type in {"PAD", "MIRROR_PAD"}
            if len(required) == 0 and has_pad:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        graph_index = pass_state.graph_index
        model_outputs = set(str(value) for value in candidate_model.outputs)

        for post_idx, post_op in enumerate(candidate_model.operators):
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or _read_transpose_perm(candidate_model, post_op)
                != perm_nchw_to_nhwc
            ):
                continue
            mul_output_name = str(post_op.inputs[0])
            post_output_name = str(post_op.outputs[0])
            if mul_output_name in model_outputs or post_output_name in model_outputs:
                continue

            mul_idx = graph_index.producers.get(mul_output_name)
            if mul_idx is None:
                continue
            mul_op = candidate_model.operators[int(mul_idx)]
            if (
                str(mul_op.op_type) != "MUL"
                or len(mul_op.inputs) != 2
                or len(mul_op.outputs) != 1
                or str(mul_op.outputs[0]) != mul_output_name
                or set(graph_index.consumer_indices(mul_output_name))
                != {int(post_idx)}
            ):
                continue

            pad_idx: Optional[int] = None
            const_input_name: Optional[str] = None
            for input_name in (str(value) for value in mul_op.inputs):
                producer_idx = graph_index.producers.get(input_name)
                if producer_idx is not None:
                    producer_op = candidate_model.operators[int(producer_idx)]
                    if (
                        str(producer_op.op_type) in {"PAD", "MIRROR_PAD"}
                        and len(producer_op.outputs) == 1
                        and str(producer_op.outputs[0]) == input_name
                    ):
                        pad_idx = int(producer_idx)
                        continue
                tensor = candidate_model.tensors.get(input_name)
                if tensor is not None and tensor.data is not None:
                    const_input_name = input_name
            if pad_idx is None or const_input_name is None:
                continue

            pad_op = candidate_model.operators[int(pad_idx)]
            pad_output_name = str(pad_op.outputs[0])
            if (
                len(pad_op.inputs) < 2
                or set(graph_index.consumer_indices(pad_output_name))
                != {int(mul_idx)}
            ):
                continue
            pre_nchw_name = str(pad_op.inputs[0])
            pre_idx = graph_index.producers.get(pre_nchw_name)
            if pre_idx is None:
                continue
            pre_op = candidate_model.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != pre_nchw_name
                or _read_transpose_perm(candidate_model, pre_op)
                != perm_nhwc_to_nchw
                or pre_nchw_name in model_outputs
                or set(graph_index.consumer_indices(pre_nchw_name))
                != {int(pad_idx)}
            ):
                continue

            post_users = graph_index.consumer_indices(post_output_name)
            if len(post_users) != 1:
                continue
            add_op = candidate_model.operators[int(post_users[0])]
            if (
                str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
            ):
                continue
            add_inputs = [str(value) for value in add_op.inputs]
            if add_inputs[0] == post_output_name:
                add_side_name = add_inputs[1]
            elif add_inputs[1] == post_output_name:
                add_side_name = add_inputs[0]
            else:
                continue
            add_side_tensor = candidate_model.tensors.get(add_side_name)
            if add_side_tensor is None or add_side_tensor.data is None:
                continue

            target_tensor = candidate_model.tensors.get(post_output_name)
            target_shape = (
                [int(value) for value in target_tensor.shape]
                if target_tensor is not None
                else None
            )
            known_target = _is_fully_known_positive_shape(target_shape)
            if known_target and _broadcast_static_shapes(
                target_shape,
                [int(value) for value in add_side_tensor.shape],
            ) is None:
                continue

            mul_const = candidate_model.tensors.get(const_input_name)
            if mul_const is None or mul_const.data is None:
                continue
            mul_data = np.asarray(mul_const.data)
            if int(mul_data.size) != 1:
                if int(mul_data.ndim) == 4:
                    as_is_shape = [int(value) for value in mul_data.shape]
                    if not (
                        known_target
                        and _broadcast_static_shapes(target_shape, as_is_shape)
                        is not None
                    ):
                        rotated_shape = [
                            int(value)
                            for value in np.transpose(
                                mul_data,
                                perm_nchw_to_nhwc,
                            ).shape
                        ]
                        if known_target and _broadcast_static_shapes(
                            target_shape,
                            rotated_shape,
                        ) is None:
                            continue
                elif not known_target or _broadcast_static_shapes(
                    target_shape,
                    [int(value) for value in mul_data.shape],
                ) is None:
                    continue

            pads_tensor = candidate_model.tensors.get(str(pad_op.inputs[1]))
            if pads_tensor is None or pads_tensor.data is None:
                continue
            try:
                pads = np.asarray(pads_tensor.data).reshape(4, 2)
            except Exception:
                continue
            if int(pads.size) != 8:
                continue
            return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pad_mul_posttranspose_add_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_pad_mul_posttranspose_add_nhwc_chains",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.pad_mul_posttranspose_add_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_transpose_pad_mul_posttranspose_add_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Eliminate NHWC<->NCHW wrappers around closed rank-4 norm subgraphs ending in PAD.

    Target:
      x_nhwc --T(0,3,1,2)--> x_nchw --(norm subgraph in NCHW)--> y_nchw
      PAD|MIRROR_PAD(y_nchw, pads_nchw, [const]) -> p_nchw
      p_nchw --T(0,2,3,1)--> z_nhwc

    Rewrite:
      x_nhwc --(same subgraph remapped to NHWC axes/broadcast)--> y_nhwc
      PAD|MIRROR_PAD(y_nhwc, pads_nhwc, [const]) -> z_nhwc

    Notes:
    - Accepts external non-rank4 inputs (e.g. rank-2 affine branches).
    - Accepts external rank-4 inputs only when they are NCHW adapters
      (TRANSPOSE NHWC->NCHW) consumed by this subgraph; those are bypassed.
    - Keeps rewrite strict with a limited op whitelist.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    allowed_op_types = {
        "MEAN",
        "SUB",
        "MUL",
        "ADD",
        "SQRT",
        "DIV",
        "RELU",
        "RELU6",
    }
    legacy_rank4_adapter_perm_name = "__norm_subgraph_shared_nhwc_to_nchw_perm_rank4__"
    external_rank4_adapter_perm_name = "__norm_subgraph_external_nchw_to_nhwc_perm_rank4__"

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _normalize_axes(axes: List[int], rank: int) -> Optional[List[int]]:
        normalized: List[int] = []
        for axis in axes:
            axis_value = int(axis)
            if axis_value < 0:
                axis_value += int(rank)
            if axis_value < 0 or axis_value >= int(rank):
                return None
            normalized.append(int(axis_value))
        return sorted(list(set(normalized)))

    def _is_channelwise_nchw_rank4_tensor(tensor: Optional[TensorIR]) -> bool:
        if tensor is None or tensor.shape is None:
            return False
        shape = [int(v) for v in list(tensor.shape)]
        return (
            len(shape) == 4
            and int(shape[0]) == 1
            and int(shape[2]) == 1
            and int(shape[3]) == 1
        )

    def _clone_const_if_shared_outside_region(
        *,
        input_name: str,
        op: OperatorIR,
        input_index: int,
        region_index_set: set[int],
        consumers: Dict[str, List[int]],
    ) -> str:
        users = [int(v) for v in consumers.get(str(input_name), [])]
        shared_outside = any(int(user_idx) not in region_index_set for user_idx in users)
        if not shared_outside:
            return str(input_name)
        src_tensor = model_ir.tensors.get(str(input_name), None)
        if src_tensor is None or src_tensor.data is None:
            return str(input_name)
        clone_name = _unique_tensor_name(f"{input_name}_nhwc")
        clone_data = np.asarray(src_tensor.data)
        model_ir.tensors[clone_name] = TensorIR(
            name=clone_name,
            dtype=str(src_tensor.dtype),
            shape=[int(v) for v in list(clone_data.shape)],
            shape_signature=[int(v) for v in list(clone_data.shape)],
            data=np.asarray(clone_data),
            is_variable=False,
            quantization=_clone_quantization(src_tensor.quantization),
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=op,
            input_index=int(input_index),
            new_input_name=clone_name,
            graph_index=graph_index,
        )
        return str(clone_name)

    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(v) for v in model_ir.outputs)
        channel_last_hints = set(
            str(v) for v in model_ir.metadata.get("assume_channel_last_layout_tensor_names", [])
        )

        for post_idx, post_op in enumerate(model_ir.operators):
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue

            pad_out_name = str(post_op.inputs[0])
            post_out_name = str(post_op.outputs[0])
            if pad_out_name in model_outputs:
                continue

            pad_idx = producers.get(pad_out_name, None)
            if pad_idx is None:
                continue
            pad_op = model_ir.operators[int(pad_idx)]
            if (
                str(pad_op.op_type) not in {"PAD", "MIRROR_PAD"}
                or len(pad_op.inputs) < 2
                or len(pad_op.outputs) != 1
                or str(pad_op.outputs[0]) != pad_out_name
            ):
                continue
            if set(int(v) for v in consumers.get(pad_out_name, [])) != {int(post_idx)}:
                continue

            end_name = str(pad_op.inputs[0])
            if end_name in model_outputs:
                continue

            def _collect_region_indices(
                *,
                stop_at_nhwc_to_nchw_transpose: bool,
            ) -> set[int]:
                region_indices_local: set[int] = set()
                stack_local: List[str] = [str(end_name)]
                while len(stack_local) > 0:
                    current_name = str(stack_local.pop())
                    producer_idx = producers.get(current_name, None)
                    if producer_idx is None:
                        continue
                    producer_idx_i = int(producer_idx)
                    if producer_idx_i in region_indices_local:
                        continue
                    producer_op = model_ir.operators[int(producer_idx_i)]
                    producer_type = str(producer_op.op_type)
                    is_nhwc_to_nchw_transpose = (
                        producer_type == "TRANSPOSE"
                        and _read_transpose_perm(model_ir, producer_op) == perm_nhwc_to_nchw
                    )
                    if producer_type not in allowed_op_types and not is_nhwc_to_nchw_transpose:
                        # Stop slice expansion at non-target producers; they are treated
                        # as external inputs to the region.
                        continue
                    region_indices_local.add(int(producer_idx_i))
                    if stop_at_nhwc_to_nchw_transpose and is_nhwc_to_nchw_transpose:
                        continue
                    for input_name in list(producer_op.inputs):
                        stack_local.append(str(input_name))
                return set(int(v) for v in region_indices_local)

            # Collect minimal producer slice from end tensor backward.
            region_indices: set[int] = _collect_region_indices(
                stop_at_nhwc_to_nchw_transpose=False,
            )

            if len(region_indices) == 0:
                continue

            # Determine root pre-transpose in the slice.
            pre_candidates = [
                int(op_idx)
                for op_idx in sorted(list(region_indices))
                if str(model_ir.operators[int(op_idx)].op_type) == "TRANSPOSE"
                and _read_transpose_perm(model_ir, model_ir.operators[int(op_idx)]) == perm_nhwc_to_nchw
            ]
            if len(pre_candidates) == 0:
                continue

            if len(pre_candidates) > 1:
                side_pre_outputs = [
                    str(model_ir.operators[int(op_idx)].outputs[0])
                    for op_idx in list(pre_candidates)
                    if len(model_ir.operators[int(op_idx)].outputs) == 1
                ]
                has_layout_fix_side = any(
                    "layout_fix" in str(output_name)
                    for output_name in list(side_pre_outputs)
                )
                if has_layout_fix_side:
                    bounded_region_indices = _collect_region_indices(
                        stop_at_nhwc_to_nchw_transpose=True,
                    )
                    bounded_pre_candidates = [
                        int(op_idx)
                        for op_idx in sorted(list(bounded_region_indices))
                        if str(model_ir.operators[int(op_idx)].op_type) == "TRANSPOSE"
                        and _read_transpose_perm(model_ir, model_ir.operators[int(op_idx)])
                        == perm_nhwc_to_nchw
                    ]
                    if len(bounded_pre_candidates) > 0:
                        region_indices = set(int(v) for v in bounded_region_indices)
                        pre_candidates = [int(v) for v in list(bounded_pre_candidates)]

            # Prefer the dominant transpose (largest number of in-region users) as the
            # root; treat the rest as bypassable side adapters.
            candidate_scores: List[Tuple[int, int]] = []
            for candidate_idx in pre_candidates:
                candidate_out = str(model_ir.operators[int(candidate_idx)].outputs[0])
                score = sum(
                    1 for user_idx in consumers.get(candidate_out, [])
                    if int(user_idx) in set(int(v) for v in region_indices)
                )
                candidate_scores.append((int(score), int(candidate_idx)))
            candidate_scores = sorted(candidate_scores, key=lambda v: (v[0], -v[1]), reverse=True)
            if len(candidate_scores) >= 2 and int(candidate_scores[0][0]) == int(candidate_scores[1][0]):
                continue
            pre_idx = int(candidate_scores[0][1])
            pre_op = model_ir.operators[int(pre_idx)]
            if len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            start_in_name = str(pre_op.inputs[0])
            start_out_name = str(pre_op.outputs[0])
            if start_out_name == end_name:
                continue
            if start_out_name in model_outputs:
                continue

            # Keep only ops reachable from the dominant pre-transpose output.
            # This avoids pulling unrelated upstream blocks into the candidate
            # region when residual branches feed the terminal ADD.
            reachable_region_indices: set[int] = set()
            reachable_stack: List[str] = [str(start_out_name)]
            while len(reachable_stack) > 0:
                current_tensor_name = str(reachable_stack.pop())
                for user_idx in list(consumers.get(current_tensor_name, [])):
                    user_idx_i = int(user_idx)
                    if user_idx_i not in region_indices:
                        continue
                    if user_idx_i in reachable_region_indices:
                        continue
                    reachable_region_indices.add(int(user_idx_i))
                    user_op = model_ir.operators[int(user_idx_i)]
                    for out_name in list(user_op.outputs):
                        reachable_stack.append(str(out_name))
            if len(reachable_region_indices) == 0:
                continue
            end_producer_idx = producers.get(str(end_name), None)
            if end_producer_idx is None or int(end_producer_idx) not in reachable_region_indices:
                continue
            region_indices = set(int(v) for v in reachable_region_indices)

            # Do not include root/side pre-transposes in the inner region.
            for transpose_idx in pre_candidates:
                if int(transpose_idx) in region_indices:
                    region_indices.remove(int(transpose_idx))
            if len(region_indices) == 0:
                continue

            # Strict op whitelist.
            if any(
                str(model_ir.operators[int(op_idx)].op_type) not in allowed_op_types
                for op_idx in region_indices
            ):
                continue

            # Pre-transpose output must be consumed only by this region.
            start_users = set(int(v) for v in consumers.get(start_out_name, []))
            if len(start_users) == 0 or not start_users.issubset(set(int(v) for v in region_indices)):
                continue

            # Region closure:
            # - Non-rank4 shared outputs are allowed as-is.
            # - Shared rank4 outputs are preserved for external users through local
            #   NHWC->NCHW transpose adapters.
            region_index_set = set(int(v) for v in region_indices)
            valid_closure = True
            shared_rank4_outputs: List[Tuple[str, int, List[int]]] = []
            for op_idx in list(region_index_set):
                op = model_ir.operators[int(op_idx)]
                for output_name in list(op.outputs):
                    output_name_s = str(output_name)
                    if output_name_s == str(end_name):
                        continue
                    if output_name_s in model_outputs:
                        valid_closure = False
                        break
                    outside_users = [
                        int(v) for v in consumers.get(output_name_s, [])
                        if int(v) not in region_index_set
                    ]
                    if len(outside_users) == 0:
                        continue
                    output_tensor = model_ir.tensors.get(output_name_s, None)
                    output_rank = (
                        int(len(list(output_tensor.shape)))
                        if output_tensor is not None and output_tensor.shape is not None
                        else 0
                    )
                    if int(output_rank) == 4:
                        if _is_channelwise_nchw_rank4_tensor(output_tensor):
                            shared_rank4_outputs.append((output_name_s, int(op_idx), outside_users))
                            continue
                        valid_closure = False
                        break
                    if int(output_rank) <= 3:
                        continue
                    valid_closure = False
                    break
                if not valid_closure:
                    break
            if not valid_closure:
                continue

            # Validate external inputs and collect bypassable NCHW adapters.
            bypass_transpose_indices: set[int] = set(
                int(v) for v in pre_candidates if int(v) != int(pre_idx)
            )
            external_channelwise_rank4_inputs: set[str] = set()
            external_rank4_ok = True
            for op_idx in list(region_index_set):
                op = model_ir.operators[int(op_idx)]
                for input_name in list(op.inputs):
                    in_name = str(input_name)
                    if in_name == start_out_name:
                        continue
                    producer_idx = producers.get(in_name, None)
                    if producer_idx is not None and int(producer_idx) in region_index_set:
                        continue

                    in_tensor = model_ir.tensors.get(in_name, None)
                    in_rank = (
                        int(len(list(in_tensor.shape)))
                        if in_tensor is not None and in_tensor.shape is not None
                        else 0
                    )

                    # Constants and rank<=3 external tensors are allowed as-is.
                    if in_tensor is not None and in_tensor.data is not None:
                        continue
                    if in_rank <= 3:
                        continue
                    if in_rank != 4:
                        external_rank4_ok = False
                        break

                    if producer_idx is None:
                        if in_name in channel_last_hints:
                            continue
                        if _is_channelwise_nchw_rank4_tensor(in_tensor):
                            external_channelwise_rank4_inputs.add(str(in_name))
                            continue
                        external_rank4_ok = False
                        break
                    producer_op = model_ir.operators[int(producer_idx)]
                    producer_type = str(producer_op.op_type)
                    if producer_type in {"CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV"}:
                        continue
                    if in_name in channel_last_hints:
                        continue
                    start_in_tensor = model_ir.tensors.get(str(start_in_name), None)
                    if (
                        start_in_tensor is not None
                        and start_in_tensor.shape is not None
                        and _is_fully_known_positive_shape(list(start_in_tensor.shape))
                        and _is_fully_known_positive_shape(list(in_tensor.shape))
                        and len(list(start_in_tensor.shape)) == 4
                    ):
                        start_shape = [int(v) for v in list(start_in_tensor.shape)]
                        in_shape = [int(v) for v in list(in_tensor.shape)]
                        if _broadcast_static_shapes(start_shape, in_shape) is not None:
                            # Accept external rank-4 NHWC-aligned residual inputs.
                            continue
                    if (
                        producer_type == "TRANSPOSE"
                        and len(producer_op.inputs) >= 2
                        and len(producer_op.outputs) == 1
                        and str(producer_op.outputs[0]) == in_name
                        and _read_transpose_perm(model_ir, producer_op) == perm_nhwc_to_nchw
                    ):
                        if not set(int(v) for v in consumers.get(in_name, [])).issubset(region_index_set):
                            external_rank4_ok = False
                            break
                        bypass_transpose_indices.add(int(producer_idx))
                        continue
                    if _is_channelwise_nchw_rank4_tensor(in_tensor):
                        external_channelwise_rank4_inputs.add(str(in_name))
                        continue
                    external_rank4_ok = False
                    break
                if not external_rank4_ok:
                    break
            if not external_rank4_ok:
                continue

            # Rewrite shared rank-4 outputs to private NHWC tensors, and keep
            # external NCHW consumers through local adapters inserted later.
            shared_output_adapter_specs: List[Tuple[str, str, List[int]]] = []
            rewrite_shared_outputs_ok = True
            for output_name, producer_idx_i, outside_users in shared_rank4_outputs:
                source_tensor = model_ir.tensors.get(str(output_name), None)
                if source_tensor is None or source_tensor.shape is None:
                    rewrite_shared_outputs_ok = False
                    break
                if len(list(source_tensor.shape)) != 4:
                    rewrite_shared_outputs_ok = False
                    break
                internal_output_name = _unique_tensor_name(f"{output_name}_nhwc")
                internal_shape = [int(v) for v in list(source_tensor.shape)]
                internal_signature = (
                    [int(v) for v in list(source_tensor.shape_signature)]
                    if source_tensor.shape_signature is not None
                    else [int(v) for v in list(source_tensor.shape)]
                )
                model_ir.tensors[internal_output_name] = TensorIR(
                    name=internal_output_name,
                    dtype=str(source_tensor.dtype),
                    shape=internal_shape,
                    shape_signature=internal_signature,
                    data=(
                        np.asarray(source_tensor.data).copy()
                        if isinstance(source_tensor.data, np.ndarray)
                        else None
                    ),
                    is_variable=bool(source_tensor.is_variable),
                    quantization=_clone_quantization(source_tensor.quantization),
                )
                producer_op = model_ir.operators[int(producer_idx_i)]
                producer_outputs = [str(v) for v in list(producer_op.outputs)]
                producer_outputs = [
                    str(internal_output_name) if str(v) == str(output_name) else str(v)
                    for v in producer_outputs
                ]
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=producer_op,
                    new_outputs=producer_outputs,
                    graph_index=graph_index,
                )
                for region_user_idx in sorted(list(region_index_set)):
                    region_user_op = model_ir.operators[int(region_user_idx)]
                    for input_index, input_name in enumerate(list(region_user_op.inputs)):
                        if str(input_name) != str(output_name):
                            continue
                        _replace_operator_input_at(
                            model_ir=model_ir,
                            op=region_user_op,
                            input_index=int(input_index),
                            new_input_name=str(internal_output_name),
                            graph_index=graph_index,
                        )
                outside_user_op_ids = [
                    int(id(model_ir.operators[int(user_idx)]))
                    for user_idx in list(outside_users)
                    if 0 <= int(user_idx) < len(model_ir.operators)
                ]
                shared_output_adapter_specs.append(
                    (str(output_name), str(internal_output_name), outside_user_op_ids)
                )
            if not rewrite_shared_outputs_ok:
                continue

            # Rewire pre-transpose root and bypassable external adapters.
            external_adapter_output_by_input: Dict[str, str] = {}
            external_adapter_consumer_op_ids: Dict[str, List[int]] = {}
            rewire_ok = True
            for op_idx in sorted(list(region_index_set)):
                op = model_ir.operators[int(op_idx)]
                new_inputs = [str(v) for v in list(op.inputs)]
                for input_index, input_name in enumerate(list(new_inputs)):
                    input_name_s = str(input_name)
                    if input_name_s == start_out_name:
                        new_inputs[int(input_index)] = str(start_in_name)
                        continue
                    producer_idx = producers.get(input_name_s, None)
                    if producer_idx is None or int(producer_idx) not in bypass_transpose_indices:
                        resolved_input_name = str(new_inputs[int(input_index)])
                    else:
                        producer_op = model_ir.operators[int(producer_idx)]
                        resolved_input_name = str(producer_op.inputs[0])
                        new_inputs[int(input_index)] = str(resolved_input_name)
                    if str(resolved_input_name) not in external_channelwise_rank4_inputs:
                        continue
                    if str(resolved_input_name) not in external_adapter_output_by_input:
                        source_tensor = model_ir.tensors.get(str(resolved_input_name), None)
                        if not _is_channelwise_nchw_rank4_tensor(source_tensor):
                            rewire_ok = False
                            break
                        source_shape = [int(v) for v in list(source_tensor.shape)]
                        source_signature = (
                            [int(v) for v in list(source_tensor.shape_signature)]
                            if source_tensor.shape_signature is not None
                            else [int(v) for v in list(source_tensor.shape)]
                        )
                        adapter_shape = _permute_shape(source_shape, perm_nchw_to_nhwc)
                        adapter_signature = _permute_shape(source_signature, perm_nchw_to_nhwc)
                        if adapter_shape is None or adapter_signature is None:
                            rewire_ok = False
                            break
                        adapter_output_name = _unique_tensor_name(f"{resolved_input_name}_nhwc")
                        model_ir.tensors[adapter_output_name] = TensorIR(
                            name=adapter_output_name,
                            dtype=str(source_tensor.dtype),
                            shape=[int(v) for v in list(adapter_shape)],
                            shape_signature=[int(v) for v in list(adapter_signature)],
                            data=None,
                            is_variable=False,
                            quantization=_clone_quantization(source_tensor.quantization),
                        )
                        external_adapter_output_by_input[str(resolved_input_name)] = str(adapter_output_name)
                        external_adapter_consumer_op_ids[str(resolved_input_name)] = []
                    adapter_output_name = str(external_adapter_output_by_input[str(resolved_input_name)])
                    new_inputs[int(input_index)] = str(adapter_output_name)
                    external_adapter_consumer_op_ids[str(resolved_input_name)].append(int(id(op)))
                if not rewire_ok:
                    break
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=op,
                    new_inputs=new_inputs,
                    graph_index=graph_index,
                )
            if not rewire_ok:
                continue

            # Remap MEAN axes and rewrite rank-4 RESHAPE target shapes.
            rewritten_axes_names: set[str] = set()
            for op_idx in sorted(list(region_index_set)):
                op = model_ir.operators[int(op_idx)]
                if str(op.op_type) == "MEAN" and len(op.inputs) >= 2 and len(op.outputs) == 1:
                    mean_in_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                    if mean_in_tensor is None or mean_in_tensor.shape is None or len(list(mean_in_tensor.shape)) != 4:
                        continue
                    axes_name = str(op.inputs[1])
                    axes_tensor = model_ir.tensors.get(axes_name, None)
                    axes_values = _read_const_ints_from_tensor(axes_tensor)
                    if axes_values is None or len(axes_values) == 0:
                        continue
                    normalized_axes = _normalize_axes([int(v) for v in list(axes_values)], 4)
                    if normalized_axes is None:
                        continue
                    mapped_axes = sorted(list(set(int(perm_nhwc_to_nchw[int(v)]) for v in normalized_axes)))
                    if set(int(v) for v in consumers.get(axes_name, [])) - region_index_set:
                        cloned_axes_name = _clone_const_if_shared_outside_region(
                            input_name=axes_name,
                            op=op,
                            input_index=1,
                            region_index_set=region_index_set,
                            consumers=consumers,
                        )
                        axes_name = str(cloned_axes_name)
                        axes_tensor = model_ir.tensors.get(axes_name, None)
                    if axes_name in rewritten_axes_names:
                        continue
                    _write_const_ints_to_tensor(axes_tensor, [int(v) for v in mapped_axes])
                    rewritten_axes_names.add(str(axes_name))
                elif str(op.op_type) == "RESHAPE" and len(op.inputs) >= 2 and len(op.outputs) == 1:
                    reshape_out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
                    if (
                        reshape_out_tensor is None
                        or reshape_out_tensor.shape is None
                        or len(list(reshape_out_tensor.shape)) != 4
                    ):
                        continue
                    target_shape = _permute_shape(
                        [int(v) for v in list(reshape_out_tensor.shape)],
                        perm_nchw_to_nhwc,
                    )
                    if target_shape is None:
                        continue
                    shape_name = str(op.inputs[1])
                    shape_tensor = model_ir.tensors.get(shape_name, None)
                    if _read_const_ints_from_tensor(shape_tensor) is None:
                        continue
                    if set(int(v) for v in consumers.get(shape_name, [])) - region_index_set:
                        cloned_shape_name = _clone_const_if_shared_outside_region(
                            input_name=shape_name,
                            op=op,
                            input_index=1,
                            region_index_set=region_index_set,
                            consumers=consumers,
                        )
                        shape_name = str(cloned_shape_name)
                        shape_tensor = model_ir.tensors.get(shape_name, None)
                    _write_const_ints_to_tensor(shape_tensor, [int(v) for v in list(target_shape)])

            # Rewrite rank-4 channelwise constants used inside region.
            for op_idx in sorted(list(region_index_set)):
                op = model_ir.operators[int(op_idx)]
                for input_index, input_name in enumerate(list(op.inputs)):
                    in_name = str(input_name)
                    producer_idx = producers.get(in_name, None)
                    if producer_idx is not None and int(producer_idx) in region_index_set:
                        continue
                    in_tensor = model_ir.tensors.get(in_name, None)
                    if in_tensor is None or in_tensor.data is None:
                        continue
                    data = np.asarray(in_tensor.data)
                    if data.ndim != 4:
                        continue
                    shape = [int(v) for v in list(data.shape)]
                    if (
                        len(shape) == 4
                        and int(shape[0]) == 1
                        and int(shape[2]) == 1
                        and int(shape[3]) == 1
                    ):
                        private_name = _clone_const_if_shared_outside_region(
                            input_name=in_name,
                            op=op,
                            input_index=int(input_index),
                            region_index_set=region_index_set,
                            consumers=consumers,
                        )
                        target_tensor = model_ir.tensors.get(str(private_name), None)
                        if target_tensor is None or target_tensor.data is None:
                            continue
                        rotated = np.transpose(np.asarray(target_tensor.data), perm_nchw_to_nhwc).astype(
                            np.asarray(target_tensor.data).dtype,
                            copy=False,
                        )
                        target_tensor.data = np.asarray(rotated)
                        target_tensor.shape = [int(v) for v in list(rotated.shape)]
                        target_tensor.shape_signature = [int(v) for v in list(rotated.shape)]

            # Convert pad spec NCHW [N,C,H,W] -> NHWC [N,H,W,C].
            pads_name = str(pad_op.inputs[1])
            pads_tensor = model_ir.tensors.get(str(pads_name), None)
            if pads_tensor is None or pads_tensor.data is None:
                continue
            try:
                pads_pairs = np.asarray(pads_tensor.data).reshape(4, 2)
            except Exception:
                continue
            if int(pads_pairs.size) != 8:
                continue
            if set(int(v) for v in consumers.get(str(pads_name), [])) - {int(pad_idx)}:
                pads_name = _clone_const_if_shared_outside_region(
                    input_name=str(pads_name),
                    op=pad_op,
                    input_index=1,
                    region_index_set={int(pad_idx)},
                    consumers=consumers,
                )
                pads_tensor = model_ir.tensors.get(str(pads_name), None)
                if pads_tensor is None or pads_tensor.data is None:
                    continue
                try:
                    pads_pairs = np.asarray(pads_tensor.data).reshape(4, 2)
                except Exception:
                    continue
            pads_nhwc = np.asarray(
                [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
                dtype=pads_pairs.dtype,
            )
            pads_tensor.data = np.asarray(pads_nhwc)
            pads_tensor.shape = [4, 2]
            pads_tensor.shape_signature = [4, 2]

            # Permute rank-4 metadata in the rewritten region.
            for op_idx in sorted(list(region_index_set)):
                op = model_ir.operators[int(op_idx)]
                for output_name in list(op.outputs):
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(str(output_name), None),
                        perm_nchw_to_nhwc,
                    )

            # Bypass post transpose by writing PAD directly to its output tensor.
            _set_operator_outputs(
                model_ir=model_ir,
                op=pad_op,
                new_outputs=[post_out_name],
                graph_index=graph_index,
            )
            old_pad_tensor = model_ir.tensors.get(str(pad_out_name), None)
            post_out_tensor = model_ir.tensors.get(str(post_out_name), None)
            if old_pad_tensor is not None and post_out_tensor is not None:
                post_out_tensor.dtype = str(old_pad_tensor.dtype)
                post_out_tensor.quantization = _clone_quantization(old_pad_tensor.quantization)
                post_out_tensor.shape = [int(v) for v in list(old_pad_tensor.shape)]
                post_out_tensor.shape_signature = (
                    [int(v) for v in list(old_pad_tensor.shape_signature)]
                    if old_pad_tensor.shape_signature is not None
                    else [int(v) for v in list(old_pad_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_out_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = {int(pre_idx), int(post_idx)}
            for bypass_idx in list(bypass_transpose_indices):
                bypass_out_name = str(model_ir.operators[int(bypass_idx)].outputs[0])
                remaining_users = graph_index.consumer_indices(bypass_out_name)
                if len(remaining_users) == 0:
                    remove_indices.add(int(bypass_idx))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            if len(shared_output_adapter_specs) > 0:
                if legacy_rank4_adapter_perm_name not in model_ir.tensors:
                    model_ir.tensors[legacy_rank4_adapter_perm_name] = TensorIR(
                        name=legacy_rank4_adapter_perm_name,
                        dtype="INT32",
                        shape=[4],
                        shape_signature=[4],
                        data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                        is_variable=False,
                        quantization=None,
                    )
                for output_name, internal_output_name, outside_user_op_ids in list(shared_output_adapter_specs):
                    op_index_by_id = {int(id(op)): int(op_idx) for op_idx, op in enumerate(model_ir.operators)}
                    insertion_candidates = [
                        int(op_index_by_id[int(op_id)])
                        for op_id in list(outside_user_op_ids)
                        if int(op_id) in op_index_by_id
                    ]
                    if len(insertion_candidates) > 0:
                        insertion_index = int(min(insertion_candidates))
                    elif str(output_name) in model_outputs:
                        insertion_index = int(len(model_ir.operators))
                    else:
                        continue
                    graph_index.insert_operator(
                        int(insertion_index),
                        OperatorIR(
                            op_type="TRANSPOSE",
                            inputs=[str(internal_output_name), str(legacy_rank4_adapter_perm_name)],
                            outputs=[str(output_name)],
                        ),
                    )

            if len(external_adapter_output_by_input) > 0:
                if external_rank4_adapter_perm_name not in model_ir.tensors:
                    model_ir.tensors[external_rank4_adapter_perm_name] = TensorIR(
                        name=external_rank4_adapter_perm_name,
                        dtype="INT32",
                        shape=[4],
                        shape_signature=[4],
                        data=np.asarray(perm_nchw_to_nhwc, dtype=np.int32),
                        is_variable=False,
                        quantization=None,
                    )
                for source_input_name, adapter_output_name in list(external_adapter_output_by_input.items()):
                    consumer_op_ids = [int(v) for v in external_adapter_consumer_op_ids.get(str(source_input_name), [])]
                    if len(consumer_op_ids) == 0:
                        continue
                    op_index_by_id = {int(id(op)): int(op_idx) for op_idx, op in enumerate(model_ir.operators)}
                    insertion_candidates = [
                        int(op_index_by_id[int(op_id)])
                        for op_id in list(consumer_op_ids)
                        if int(op_id) in op_index_by_id
                    ]
                    if len(insertion_candidates) == 0:
                        continue
                    insertion_index = int(min(insertion_candidates))
                    graph_index.insert_operator(
                        int(insertion_index),
                        OperatorIR(
                            op_type="TRANSPOSE",
                            inputs=[str(source_input_name), str(external_rank4_adapter_perm_name)],
                            outputs=[str(adapter_output_name)],
                        ),
                    )

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains": int(rewritten)}


def run_pad_layout_cleanup(
    model_ir: ModelIR,
    *,
    include_pad: bool = True,
    include_unary: bool = True,
    include_norm: bool = True,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
    state_scope: Optional[ModelIRPassStateScope] = None,
) -> Dict[str, int]:
    """Run the contiguous Pad layout propagation family in fixed order."""

    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    unary_types = {
        "RELU",
        "RELU6",
        "RELU_0_TO_1",
        "HARD_SWISH",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "GELU",
        "ABS",
        "NEG",
        "SQRT",
        "EXP",
        "FLOOR",
        "CEIL",
    }
    norm_types = {"MEAN", "SUB", "MUL", "ADD", "SQRT", "DIV", "RELU", "RELU6"}

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        transpose_count = 0
        has_pad = False
        for visited, op in enumerate(candidate_model.operators, start=1):
            op_type = str(op.op_type)
            transpose_count += int(op_type == "TRANSPOSE")
            has_pad = has_pad or op_type in {"PAD", "MIRROR_PAD"}
            if has_pad and transpose_count >= 2:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _post_pad_candidates(pass_state: ModelIRPassState) -> List[Tuple[int, OperatorIR]]:
        candidates: List[Tuple[int, OperatorIR]] = []
        for post_op in pass_state.model_ir.operators:
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or _read_transpose_perm(pass_state.model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue
            pad_idx = pass_state.graph_index.producers.get(str(post_op.inputs[0]))
            if pad_idx is None:
                continue
            pad_op = pass_state.model_ir.operators[int(pad_idx)]
            if str(pad_op.op_type) in {"PAD", "MIRROR_PAD"} and len(pad_op.inputs) >= 2:
                candidates.append((int(pad_idx), pad_op))
        return candidates

    def _has_pad_candidate(pass_state: ModelIRPassState) -> bool:
        for _, pad_op in _post_pad_candidates(pass_state):
            pre_idx = pass_state.graph_index.producers.get(str(pad_op.inputs[0]))
            if pre_idx is None:
                continue
            pre_op = pass_state.model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) == "TRANSPOSE"
                and _read_transpose_perm(pass_state.model_ir, pre_op) == perm_nhwc_to_nchw
            ):
                return True
        return False

    def _has_unary_candidate(pass_state: ModelIRPassState) -> bool:
        for _, pad_op in _post_pad_candidates(pass_state):
            unary_idx = pass_state.graph_index.producers.get(str(pad_op.inputs[0]))
            if unary_idx is None:
                continue
            unary_op = pass_state.model_ir.operators[int(unary_idx)]
            if str(unary_op.op_type) not in unary_types or len(unary_op.inputs) != 1:
                continue
            pre_idx = pass_state.graph_index.producers.get(str(unary_op.inputs[0]))
            if pre_idx is None:
                continue
            pre_op = pass_state.model_ir.operators[int(pre_idx)]
            if (
                str(pre_op.op_type) == "TRANSPOSE"
                and _read_transpose_perm(pass_state.model_ir, pre_op) == perm_nhwc_to_nchw
            ):
                return True
        return False

    def _has_norm_candidate(pass_state: ModelIRPassState) -> bool:
        for _, pad_op in _post_pad_candidates(pass_state):
            pending = [str(pad_op.inputs[0])]
            visited: set[str] = set()
            while pending:
                tensor_name = pending.pop()
                if tensor_name in visited:
                    continue
                visited.add(tensor_name)
                producer_idx = pass_state.graph_index.producers.get(tensor_name)
                if producer_idx is None:
                    continue
                producer_op = pass_state.model_ir.operators[int(producer_idx)]
                if str(producer_op.op_type) == "TRANSPOSE":
                    if _read_transpose_perm(pass_state.model_ir, producer_op) == perm_nhwc_to_nchw:
                        return True
                    continue
                if str(producer_op.op_type) not in norm_types:
                    continue
                pending.extend(str(name) for name in producer_op.inputs)
        return False

    def _run_pad(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pad_prepost_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get("optimized_transpose_pad_prepost_nhwc_chains", 0)),
        }

    def _run_unary(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains",
                    0,
                )
            ),
        }

    def _run_norm(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains", 0)
            ),
        }

    specs: List[PassSpec[ModelIRPassState]] = []
    if include_pad:
        specs.append(
            PassSpec(
                pass_id="layout.pad_prepost_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_pad,
                precondition=_has_pad_candidate,
                priority=10,
                transactional=True,
            )
        )
    if include_unary:
        specs.append(
            PassSpec(
                pass_id="layout.unary_pad_prepost_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_unary,
                precondition=_has_unary_candidate,
                priority=20,
                transactional=True,
            )
        )
    if include_norm:
        specs.append(
            PassSpec(
                pass_id="layout.norm_subgraph_pad_prepost_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_norm,
                precondition=_has_norm_candidate,
                priority=30,
                transactional=True,
            )
        )
    if len(specs) == 0:
        return {
            "optimized_transpose_pad_prepost_nhwc_chains": 0,
            "optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains": 0,
            "optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains": 0,
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details={
            "optimized_transpose_pad_prepost_nhwc_chains": 0,
            "optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains": 0,
            "optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}


def _optimize_transpose_instancenorm_pad_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Eliminate NHWC<->NCHW transpose bridges around decomposed InstanceNormalization
    when the normalized output is padded before returning to NHWC.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      x_nchw --(InstanceNormalization decomposition in NCHW)--> y_nchw
      y_nchw --PAD|MIRROR_PAD(pads_nchw)--> p_nchw
      p_nchw --TRANSPOSE(0,2,3,1)--> z_nhwc

    Rewrite:
      x_nhwc --(same decomposition rewritten to NHWC axes/broadcast)--> y_nhwc
      y_nhwc --PAD|MIRROR_PAD(pads_nhwc)--> z_nhwc

    Notes:
    - If `y_nchw` has non-pad side consumers, preserve them by adding a local
      NHWC->NCHW transpose adapter for those legacy consumers only.
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    legacy_adapter_perm_name = "__instancenorm_pad_tail_nhwc_to_nchw_perm_rank4__"

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _normalize_rank4_axes(axes: Optional[List[int]]) -> Optional[List[int]]:
        if axes is None or len(axes) == 0:
            return None
        normalized: List[int] = []
        for axis in axes:
            axis_value = int(axis)
            if axis_value < 0:
                axis_value += 4
            if axis_value < 0 or axis_value >= 4:
                return None
            normalized.append(int(axis_value))
        return sorted(list(set(normalized)))

    def _rewrite_axes_to_nhwc(
        *,
        axes_name: str,
        expected_users: set[int],
        consumers: Dict[str, List[int]],
    ) -> bool:
        axes_users = set(int(v) for v in consumers.get(str(axes_name), []))
        if axes_users != expected_users:
            return False
        axes_tensor = model_ir.tensors.get(str(axes_name), None)
        if axes_tensor is None:
            return False
        axes_values = _read_const_ints_from_tensor(axes_tensor)
        if _normalize_rank4_axes(axes_values) != [2, 3]:
            return False
        _write_const_ints_to_tensor(axes_tensor, [1, 2])
        return True

    def _rewrite_nchw_coeff_to_nhwc(
        *,
        coeff_name: str,
        expected_users: set[int],
        consumers: Dict[str, List[int]],
    ) -> bool:
        coeff_users = set(int(v) for v in consumers.get(str(coeff_name), []))
        if coeff_users != expected_users:
            return False
        coeff_tensor = model_ir.tensors.get(str(coeff_name), None)
        if coeff_tensor is None or coeff_tensor.data is None:
            return False
        coeff = np.asarray(coeff_tensor.data)
        if int(coeff.size) == 1:
            return True
        if coeff.ndim != 4:
            return False
        shape = [int(v) for v in list(coeff.shape)]
        # Already NHWC broadcast [1,1,1,C].
        if (
            len(shape) == 4
            and int(shape[0]) == 1
            and int(shape[1]) == 1
            and int(shape[2]) == 1
        ):
            return True
        # NCHW broadcast [1,C,1,1] -> NHWC [1,1,1,C].
        if (
            len(shape) == 4
            and int(shape[0]) == 1
            and int(shape[2]) == 1
            and int(shape[3]) == 1
        ):
            coeff_nhwc = np.transpose(coeff, perm_nchw_to_nhwc).astype(coeff.dtype, copy=False)
            coeff_tensor.data = np.asarray(coeff_nhwc)
            coeff_tensor.shape = [int(v) for v in list(coeff_nhwc.shape)]
            coeff_tensor.shape_signature = [int(v) for v in list(coeff_nhwc.shape)]
            return True
        return False

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

            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            if pre_in_name in model_outputs or pre_out_name in model_outputs:
                continue

            pre_users = set(int(v) for v in consumers.get(pre_out_name, []))
            if len(pre_users) != 2:
                continue

            mean1_idx: Optional[int] = None
            sub_idx: Optional[int] = None
            for user_idx in sorted(list(pre_users)):
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "MEAN"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == pre_out_name
                ):
                    mean1_idx = int(user_idx)
                elif (
                    str(user_op.op_type) == "SUB"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and pre_out_name in {str(user_op.inputs[0]), str(user_op.inputs[1])}
                ):
                    sub_idx = int(user_idx)
            if mean1_idx is None or sub_idx is None:
                continue

            mean1_op = model_ir.operators[int(mean1_idx)]
            sub_op = model_ir.operators[int(sub_idx)]
            mean1_out_name = str(mean1_op.outputs[0])
            if mean1_out_name in model_outputs:
                continue
            if mean1_out_name not in {str(sub_op.inputs[0]), str(sub_op.inputs[1])}:
                continue
            if set(int(v) for v in consumers.get(mean1_out_name, [])) != {int(sub_idx)}:
                continue

            centered_name = str(sub_op.outputs[0])
            if centered_name in model_outputs:
                continue
            centered_users = set(int(v) for v in consumers.get(centered_name, []))
            if len(centered_users) != 2:
                continue

            mul_square_idx: Optional[int] = None
            mul_norm_idx: Optional[int] = None
            for user_idx in sorted(list(centered_users)):
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) != "MUL" or len(user_op.inputs) != 2 or len(user_op.outputs) != 1:
                    continue
                in0 = str(user_op.inputs[0])
                in1 = str(user_op.inputs[1])
                if in0 == centered_name and in1 == centered_name:
                    mul_square_idx = int(user_idx)
                elif centered_name in {in0, in1}:
                    mul_norm_idx = int(user_idx)
            if mul_square_idx is None or mul_norm_idx is None:
                continue

            mul_square_op = model_ir.operators[int(mul_square_idx)]
            squared_name = str(mul_square_op.outputs[0])
            mean2_users = set(int(v) for v in consumers.get(squared_name, []))
            if len(mean2_users) != 1:
                continue
            mean2_idx = int(list(mean2_users)[0])
            mean2_op = model_ir.operators[int(mean2_idx)]
            if (
                str(mean2_op.op_type) != "MEAN"
                or len(mean2_op.inputs) != 2
                or len(mean2_op.outputs) != 1
                or str(mean2_op.inputs[0]) != squared_name
            ):
                continue
            mean2_out_name = str(mean2_op.outputs[0])

            add_eps_users = set(int(v) for v in consumers.get(mean2_out_name, []))
            if len(add_eps_users) != 1:
                continue
            add_eps_idx = int(list(add_eps_users)[0])
            add_eps_op = model_ir.operators[int(add_eps_idx)]
            if (
                str(add_eps_op.op_type) != "ADD"
                or len(add_eps_op.inputs) != 2
                or len(add_eps_op.outputs) != 1
                or mean2_out_name not in {str(add_eps_op.inputs[0]), str(add_eps_op.inputs[1])}
            ):
                continue
            add_eps_out_name = str(add_eps_op.outputs[0])

            sqrt_users = set(int(v) for v in consumers.get(add_eps_out_name, []))
            if len(sqrt_users) != 1:
                continue
            sqrt_idx = int(list(sqrt_users)[0])
            sqrt_op = model_ir.operators[int(sqrt_idx)]
            if (
                str(sqrt_op.op_type) != "SQRT"
                or len(sqrt_op.inputs) != 1
                or len(sqrt_op.outputs) != 1
                or str(sqrt_op.inputs[0]) != add_eps_out_name
            ):
                continue
            sqrt_out_name = str(sqrt_op.outputs[0])

            div_users = set(int(v) for v in consumers.get(sqrt_out_name, []))
            if len(div_users) != 1:
                continue
            div_idx = int(list(div_users)[0])
            div_op = model_ir.operators[int(div_idx)]
            if (
                str(div_op.op_type) != "DIV"
                or len(div_op.inputs) != 2
                or len(div_op.outputs) != 1
                or str(div_op.inputs[1]) != sqrt_out_name
            ):
                continue
            div_out_name = str(div_op.outputs[0])

            mul_norm_op = model_ir.operators[int(mul_norm_idx)]
            if div_out_name not in {str(mul_norm_op.inputs[0]), str(mul_norm_op.inputs[1])}:
                continue
            normalized_name = str(mul_norm_op.outputs[0])

            mul_scale_users = set(int(v) for v in consumers.get(normalized_name, []))
            if len(mul_scale_users) != 1:
                continue
            mul_scale_idx = int(list(mul_scale_users)[0])
            mul_scale_op = model_ir.operators[int(mul_scale_idx)]
            if (
                str(mul_scale_op.op_type) != "MUL"
                or len(mul_scale_op.inputs) != 2
                or len(mul_scale_op.outputs) != 1
                or normalized_name not in {str(mul_scale_op.inputs[0]), str(mul_scale_op.inputs[1])}
            ):
                continue
            scale_const_name = (
                str(mul_scale_op.inputs[0])
                if str(mul_scale_op.inputs[1]) == normalized_name
                else str(mul_scale_op.inputs[1])
            )
            scaled_name = str(mul_scale_op.outputs[0])

            add_bias_users = set(int(v) for v in consumers.get(scaled_name, []))
            if len(add_bias_users) != 1:
                continue
            add_bias_idx = int(list(add_bias_users)[0])
            add_bias_op = model_ir.operators[int(add_bias_idx)]
            if (
                str(add_bias_op.op_type) != "ADD"
                or len(add_bias_op.inputs) != 2
                or len(add_bias_op.outputs) != 1
                or scaled_name not in {str(add_bias_op.inputs[0]), str(add_bias_op.inputs[1])}
            ):
                continue
            bias_const_name = (
                str(add_bias_op.inputs[0])
                if str(add_bias_op.inputs[1]) == scaled_name
                else str(add_bias_op.inputs[1])
            )
            inst_out_name = str(add_bias_op.outputs[0])
            if inst_out_name in model_outputs:
                continue

            tail_add_idx: Optional[int] = None
            tail_add_new_inputs: Optional[List[str]] = None
            tail_add_output_name: Optional[str] = None
            tail_add_residual_pre_idx: Optional[int] = None
            legacy_adapter_source_name = str(inst_out_name)
            inst_out_users = [int(v) for v in consumers.get(inst_out_name, [])]
            if len(inst_out_users) == 0:
                continue
            pad_candidates: List[int] = []
            for user_idx in inst_out_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) in {"PAD", "MIRROR_PAD"}
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == inst_out_name
                ):
                    pad_candidates.append(int(user_idx))
            legacy_consumer_slots: List[Tuple[int, int]] = []
            if len(pad_candidates) == 1:
                pad_idx = int(pad_candidates[0])
                pad_op = model_ir.operators[int(pad_idx)]
                for user_idx in inst_out_users:
                    if int(user_idx) == int(pad_idx):
                        continue
                    user_op = model_ir.operators[int(user_idx)]
                    for input_idx, input_name in enumerate(list(user_op.inputs)):
                        if str(input_name) == inst_out_name:
                            legacy_consumer_slots.append((int(id(user_op)), int(input_idx)))
            else:
                # Residual tail variant:
                #   inst_out_nchw --ADD(skip_nchw)--> add_out_nchw --PAD--> ... --T--> nhwc
                if len(inst_out_users) != 1:
                    continue
                candidate_add_idx = int(inst_out_users[0])
                candidate_add_op = model_ir.operators[int(candidate_add_idx)]
                if (
                    str(candidate_add_op.op_type) != "ADD"
                    or len(candidate_add_op.inputs) != 2
                    or len(candidate_add_op.outputs) != 1
                    or inst_out_name not in {str(candidate_add_op.inputs[0]), str(candidate_add_op.inputs[1])}
                ):
                    continue
                candidate_add_out_name = str(candidate_add_op.outputs[0])
                if candidate_add_out_name in model_outputs:
                    continue
                candidate_add_out_users = [int(v) for v in consumers.get(candidate_add_out_name, [])]
                if len(candidate_add_out_users) == 0:
                    continue
                candidate_pad_indices: List[int] = []
                for candidate_user_idx in candidate_add_out_users:
                    candidate_user_op = model_ir.operators[int(candidate_user_idx)]
                    if (
                        str(candidate_user_op.op_type) in {"PAD", "MIRROR_PAD"}
                        and len(candidate_user_op.inputs) >= 2
                        and len(candidate_user_op.outputs) == 1
                        and str(candidate_user_op.inputs[0]) == candidate_add_out_name
                    ):
                        candidate_pad_indices.append(int(candidate_user_idx))
                if len(candidate_pad_indices) != 1:
                    continue
                candidate_pad_idx = int(candidate_pad_indices[0])
                candidate_pad_op = model_ir.operators[int(candidate_pad_idx)]

                residual_input_name = (
                    str(candidate_add_op.inputs[0])
                    if str(candidate_add_op.inputs[1]) == inst_out_name
                    else str(candidate_add_op.inputs[1])
                )
                if residual_input_name in model_outputs:
                    continue
                residual_pre_idx = producers.get(residual_input_name, None)
                if residual_pre_idx is None:
                    continue
                residual_pre_op = model_ir.operators[int(residual_pre_idx)]
                if (
                    str(residual_pre_op.op_type) != "TRANSPOSE"
                    or len(residual_pre_op.inputs) < 2
                    or len(residual_pre_op.outputs) != 1
                    or str(residual_pre_op.outputs[0]) != residual_input_name
                    or _read_transpose_perm(model_ir, residual_pre_op) != perm_nhwc_to_nchw
                ):
                    continue
                if set(int(v) for v in consumers.get(residual_input_name, [])) != {int(candidate_add_idx)}:
                    continue
                residual_src_name = str(residual_pre_op.inputs[0])
                if residual_src_name in model_outputs:
                    continue

                tail_add_idx = int(candidate_add_idx)
                tail_add_output_name = str(candidate_add_out_name)
                tail_add_residual_pre_idx = int(residual_pre_idx)
                tail_add_new_inputs = [str(v) for v in list(candidate_add_op.inputs)]
                for input_idx, input_name in enumerate(list(tail_add_new_inputs)):
                    if str(input_name) == residual_input_name:
                        tail_add_new_inputs[int(input_idx)] = str(residual_src_name)
                for user_idx in candidate_add_out_users:
                    if int(user_idx) == int(candidate_pad_idx):
                        continue
                    user_op = model_ir.operators[int(user_idx)]
                    for input_idx, input_name in enumerate(list(user_op.inputs)):
                        if str(input_name) == candidate_add_out_name:
                            legacy_consumer_slots.append((int(id(user_op)), int(input_idx)))
                legacy_adapter_source_name = str(candidate_add_out_name)

                pad_idx = int(candidate_pad_idx)
                pad_op = candidate_pad_op

            pad_out_name = str(pad_op.outputs[0])
            if pad_out_name in model_outputs:
                continue

            post_users = set(int(v) for v in consumers.get(pad_out_name, []))
            if len(post_users) != 1:
                continue
            post_idx = int(list(post_users)[0])
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != pad_out_name
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue
            post_out_name = str(post_op.outputs[0])

            mean1_axes_name = str(mean1_op.inputs[1])
            mean2_axes_name = str(mean2_op.inputs[1])
            if mean1_axes_name == mean2_axes_name:
                if not _rewrite_axes_to_nhwc(
                    axes_name=mean1_axes_name,
                    expected_users={int(mean1_idx), int(mean2_idx)},
                    consumers=consumers,
                ):
                    continue
            else:
                if not _rewrite_axes_to_nhwc(
                    axes_name=mean1_axes_name,
                    expected_users={int(mean1_idx)},
                    consumers=consumers,
                ):
                    continue
                if not _rewrite_axes_to_nhwc(
                    axes_name=mean2_axes_name,
                    expected_users={int(mean2_idx)},
                    consumers=consumers,
                ):
                    continue

            if not _rewrite_nchw_coeff_to_nhwc(
                coeff_name=scale_const_name,
                expected_users={int(mul_scale_idx)},
                consumers=consumers,
            ):
                continue
            if not _rewrite_nchw_coeff_to_nhwc(
                coeff_name=bias_const_name,
                expected_users={int(add_bias_idx)},
                consumers=consumers,
            ):
                continue

            pads_name = str(pad_op.inputs[1])
            pads_users = set(int(v) for v in consumers.get(pads_name, []))
            if pads_users != {int(pad_idx)}:
                continue
            pads_tensor = model_ir.tensors.get(pads_name, None)
            if pads_tensor is None or pads_tensor.data is None:
                continue
            try:
                pads_pairs = np.asarray(pads_tensor.data).reshape(4, 2)
            except Exception:
                continue
            if int(pads_pairs.size) != 8:
                continue

            _replace_operator_input_at(
                model_ir=model_ir,
                op=mean1_op,
                input_index=0,
                new_input_name=pre_in_name,
                graph_index=graph_index,
            )
            sub_inputs = [str(v) for v in list(sub_op.inputs)]
            if sub_inputs[0] == pre_out_name:
                sub_inputs[0] = pre_in_name
            if sub_inputs[1] == pre_out_name:
                sub_inputs[1] = pre_in_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=sub_op,
                new_inputs=sub_inputs,
                graph_index=graph_index,
            )
            if tail_add_idx is not None and tail_add_new_inputs is not None:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=model_ir.operators[int(tail_add_idx)],
                    new_inputs=[str(v) for v in list(tail_add_new_inputs)],
                    graph_index=graph_index,
                )

            pads_nhwc = np.asarray(
                [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
                dtype=pads_pairs.dtype,
            )
            pads_tensor.data = np.asarray(pads_nhwc)
            pads_tensor.shape = [4, 2]
            pads_tensor.shape_signature = [4, 2]

            _set_operator_outputs(
                model_ir=model_ir,
                op=pad_op,
                new_outputs=[post_out_name],
                graph_index=graph_index,
            )

            adapter_source_tensor = model_ir.tensors.get(str(legacy_adapter_source_name), None)
            adapter_meta: Optional[Tuple[str, Optional[Any], List[int], List[int]]] = None
            if adapter_source_tensor is not None:
                adapter_meta = (
                    str(adapter_source_tensor.dtype),
                    _clone_quantization(adapter_source_tensor.quantization),
                    [int(v) for v in list(adapter_source_tensor.shape)],
                    (
                        [int(v) for v in list(adapter_source_tensor.shape_signature)]
                        if adapter_source_tensor.shape_signature is not None
                        else [int(v) for v in list(adapter_source_tensor.shape)]
                    ),
                )

            rank4_names = [
                pre_out_name,
                mean1_out_name,
                centered_name,
                squared_name,
                mean2_out_name,
                add_eps_out_name,
                sqrt_out_name,
                div_out_name,
                normalized_name,
                scaled_name,
                inst_out_name,
                str(tail_add_output_name) if tail_add_output_name is not None else "",
                pad_out_name,
            ]
            for tensor_name in rank4_names:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            old_pad_tensor = model_ir.tensors.get(str(pad_out_name), None)
            post_out_tensor = model_ir.tensors.get(str(post_out_name), None)
            if old_pad_tensor is not None and post_out_tensor is not None:
                post_out_tensor.dtype = str(old_pad_tensor.dtype)
                post_out_tensor.quantization = _clone_quantization(old_pad_tensor.quantization)
                post_out_tensor.shape = [int(v) for v in list(old_pad_tensor.shape)]
                post_out_tensor.shape_signature = (
                    [int(v) for v in list(old_pad_tensor.shape_signature)]
                    if old_pad_tensor.shape_signature is not None
                    else [int(v) for v in list(old_pad_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_out_tensor,
                    perm_nchw_to_nhwc,
                )

            remove_indices = {int(pre_idx), int(post_idx)}
            if tail_add_residual_pre_idx is not None:
                remove_indices.add(int(tail_add_residual_pre_idx))
            for remove_idx in sorted(list(remove_indices), reverse=True):
                graph_index.remove_operator(int(remove_idx))

            if len(legacy_consumer_slots) > 0 and adapter_meta is not None:
                if legacy_adapter_perm_name not in model_ir.tensors:
                    model_ir.tensors[legacy_adapter_perm_name] = TensorIR(
                        name=legacy_adapter_perm_name,
                        dtype="INT32",
                        shape=[4],
                        shape_signature=[4],
                        data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                        is_variable=False,
                        quantization=None,
                    )
                op_index_by_id = {
                    int(id(op)): int(op_idx) for op_idx, op in enumerate(model_ir.operators)
                }
                slot_targets: List[Tuple[int, int]] = []
                for consumer_op_id, input_index in legacy_consumer_slots:
                    consumer_index = op_index_by_id.get(int(consumer_op_id), None)
                    if consumer_index is None:
                        continue
                    consumer_op = model_ir.operators[int(consumer_index)]
                    if int(input_index) < 0 or int(input_index) >= len(consumer_op.inputs):
                        continue
                    if str(consumer_op.inputs[int(input_index)]) != str(legacy_adapter_source_name):
                        continue
                    slot_targets.append((int(consumer_index), int(input_index)))

                if len(slot_targets) > 0:
                    adapter_name = _unique_tensor_name(f"{legacy_adapter_source_name}_nchw_adapter")
                    adapter_dtype, adapter_quant, adapter_shape, adapter_shape_signature = adapter_meta
                    model_ir.tensors[adapter_name] = TensorIR(
                        name=adapter_name,
                        dtype=str(adapter_dtype),
                        shape=[int(v) for v in list(adapter_shape)],
                        shape_signature=[int(v) for v in list(adapter_shape_signature)],
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(adapter_quant),
                    )
                    for consumer_index, input_index in slot_targets:
                        consumer_op = model_ir.operators[int(consumer_index)]
                        new_inputs = [str(v) for v in list(consumer_op.inputs)]
                        new_inputs[int(input_index)] = str(adapter_name)
                        _set_operator_inputs(
                            model_ir=model_ir,
                            op=consumer_op,
                            new_inputs=new_inputs,
                            graph_index=graph_index,
                        )
                    insert_index = min(int(v[0]) for v in slot_targets)
                    graph_index.insert_operator(
                        int(insert_index),
                        OperatorIR(
                            op_type="TRANSPOSE",
                            inputs=[str(legacy_adapter_source_name), str(legacy_adapter_perm_name)],
                            outputs=[str(adapter_name)],
                        ),
                    )

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_instancenorm_pad_prepost_nhwc_chains": int(rewritten)}


def _optimize_transpose_flatten_globalnorm_pad_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Remove NHWC<->NCHW transpose bridges around flattened global-normalization blocks.

    Target:
      x_nhwc --T(0,3,1,2)--> x_nchw
      x_nchw --RESHAPE([N,1,-1])--> r
      r --(global norm decomposition)--> r_out
      r_out --RESHAPE([N,C,H,W])--> y_nchw
      y_nchw --(layout-agnostic tail)*--> t_nchw
      t_nchw --PAD|MIRROR_PAD(pads_nchw)--> p_nchw
      p_nchw --T(0,2,3,1)--> z_nhwc

    Rewrite:
      x_nhwc --RESHAPE([N,1,-1])--> r
      r --(same global norm decomposition)--> r_out
      r_out --RESHAPE([N,H,W,C])--> y_nhwc
      y_nhwc --(same layout-agnostic tail)*--> t_nhwc
      t_nhwc --PAD|MIRROR_PAD(pads_nhwc)--> z_nhwc
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    unary_tail_ops = {
        "LEAKY_RELU",
        "RELU",
        "RELU6",
        "TANH",
        "LOGISTIC",
        "HARD_SWISH",
        "NEG",
        "ABS",
        "FLOOR",
    }
    binary_tail_ops = {"ADD", "SUB", "MUL", "DIV"}

    def _is_const_broadcast_layout_agnostic(name: str) -> bool:
        tensor = model_ir.tensors.get(str(name), None)
        if tensor is None or tensor.data is None:
            return False
        array = np.asarray(tensor.data)
        if int(array.size) == 1:
            return True
        shape = [int(v) for v in list(array.shape)]
        return len(shape) > 0 and all(int(dim) == 1 for dim in shape)

    def _can_rewrite_tail_const_to_nhwc(name: str) -> bool:
        tensor = model_ir.tensors.get(str(name), None)
        if tensor is None or tensor.data is None:
            return False
        array = np.asarray(tensor.data)
        if int(array.size) == 1:
            return True
        shape = [int(v) for v in list(array.shape)]
        if len(shape) > 0 and all(int(dim) == 1 for dim in shape):
            return True
        if len(shape) == 3 and int(shape[1]) == 1 and int(shape[2]) == 1:
            return True
        if (
            len(shape) == 4
            and int(shape[0]) == 1
            and int(shape[2]) == 1
            and int(shape[3]) == 1
        ):
            return True
        return False

    def _rewrite_tail_const_to_nhwc(name: str) -> bool:
        tensor = model_ir.tensors.get(str(name), None)
        if tensor is None or tensor.data is None:
            return False
        array = np.asarray(tensor.data)
        if int(array.size) == 1:
            return True
        shape = [int(v) for v in list(array.shape)]
        if len(shape) > 0 and all(int(dim) == 1 for dim in shape):
            return True
        if len(shape) == 3 and int(shape[1]) == 1 and int(shape[2]) == 1:
            nhwc = np.reshape(array, newshape=[1, 1, int(shape[0])]).astype(array.dtype, copy=False)
            tensor.data = np.asarray(nhwc)
            tensor.shape = [int(v) for v in list(nhwc.shape)]
            tensor.shape_signature = [int(v) for v in list(nhwc.shape)]
            return True
        if (
            len(shape) == 4
            and int(shape[0]) == 1
            and int(shape[2]) == 1
            and int(shape[3]) == 1
        ):
            nhwc = np.transpose(array, perm_nchw_to_nhwc).astype(array.dtype, copy=False)
            tensor.data = np.asarray(nhwc)
            tensor.shape = [int(v) for v in list(nhwc.shape)]
            tensor.shape_signature = [int(v) for v in list(nhwc.shape)]
            return True
        return False

    def _read_reshape_target(op: OperatorIR) -> Optional[List[int]]:
        target: List[int] = []
        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            shape_vals = _read_const_ints_from_tensor(shape_tensor)
            if shape_vals is not None:
                target = [int(v) for v in list(shape_vals)]
        if len(target) == 0:
            raw_target = op.options.get("newShape", [])
            try:
                target = [int(v) for v in np.asarray(raw_target).reshape(-1).tolist()]
            except Exception:
                target = []
        return [int(v) for v in list(target)] if len(target) > 0 else None

    def _set_reshape_target(op: OperatorIR, target: List[int]) -> bool:
        updated = False
        shape_values = [int(v) for v in list(target)]
        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if shape_tensor is not None:
                if not _write_const_ints_to_tensor(shape_tensor, shape_values):
                    return False
                updated = True
        if "newShape" in op.options:
            op.options["newShape"] = [int(v) for v in list(shape_values)]
            updated = True
        if "onnxRawNewShape" in op.options and isinstance(op.options["onnxRawNewShape"], list):
            raw_shape = [int(v) for v in list(op.options["onnxRawNewShape"])]
            if (
                len(raw_shape) == len(shape_values)
                and all(int(v) >= 0 for v in raw_shape)
            ):
                op.options["onnxRawNewShape"] = [int(v) for v in list(shape_values)]
                updated = True
        return updated

    while True:
        changed = False
        consumers = graph_index.consumers
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if (
                str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw
            ):
                continue
            pre_in_name = str(pre_op.inputs[0])
            pre_out_name = str(pre_op.outputs[0])
            if pre_in_name in model_outputs or pre_out_name in model_outputs:
                continue

            pre_users = [int(v) for v in consumers.get(pre_out_name, [])]
            if len(pre_users) != 1:
                continue
            reshape1_idx = int(pre_users[0])
            reshape1_op = model_ir.operators[int(reshape1_idx)]
            if (
                str(reshape1_op.op_type) != "RESHAPE"
                or len(reshape1_op.inputs) < 1
                or len(reshape1_op.outputs) != 1
                or str(reshape1_op.inputs[0]) != pre_out_name
            ):
                continue
            reshape1_out_name = str(reshape1_op.outputs[0])
            reshape1_target = _read_reshape_target(reshape1_op)
            if (
                reshape1_target is None
                or len(reshape1_target) != 3
                or int(reshape1_target[1]) != 1
                or int(reshape1_target[2]) != -1
            ):
                continue

            flat_users = set(int(v) for v in consumers.get(reshape1_out_name, []))
            if len(flat_users) != 2:
                continue
            mean1_idx: Optional[int] = None
            sub_idx: Optional[int] = None
            for user_idx in sorted(list(flat_users)):
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "MEAN"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == reshape1_out_name
                ):
                    axes_vals = _read_const_ints_from_tensor(
                        model_ir.tensors.get(str(user_op.inputs[1]), None)
                    )
                    if axes_vals is None or [int(v) for v in list(axes_vals)] != [2]:
                        break
                    mean1_idx = int(user_idx)
                elif (
                    str(user_op.op_type) == "SUB"
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and reshape1_out_name in {str(user_op.inputs[0]), str(user_op.inputs[1])}
                ):
                    sub_idx = int(user_idx)
            if mean1_idx is None or sub_idx is None:
                continue

            mean1_op = model_ir.operators[int(mean1_idx)]
            sub_op = model_ir.operators[int(sub_idx)]
            mean1_out_name = str(mean1_op.outputs[0])
            if mean1_out_name not in {str(sub_op.inputs[0]), str(sub_op.inputs[1])}:
                continue
            if set(int(v) for v in consumers.get(mean1_out_name, [])) != {int(sub_idx)}:
                continue

            centered_name = str(sub_op.outputs[0])
            centered_users = set(int(v) for v in consumers.get(centered_name, []))
            if len(centered_users) != 2:
                continue
            mul_square_idx: Optional[int] = None
            mul_norm_idx: Optional[int] = None
            for user_idx in sorted(list(centered_users)):
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) != "MUL" or len(user_op.inputs) != 2 or len(user_op.outputs) != 1:
                    continue
                in0 = str(user_op.inputs[0])
                in1 = str(user_op.inputs[1])
                if in0 == centered_name and in1 == centered_name:
                    mul_square_idx = int(user_idx)
                elif centered_name in {in0, in1}:
                    mul_norm_idx = int(user_idx)
            if mul_square_idx is None or mul_norm_idx is None:
                continue

            mul_square_op = model_ir.operators[int(mul_square_idx)]
            squared_name = str(mul_square_op.outputs[0])
            mean2_users = set(int(v) for v in consumers.get(squared_name, []))
            if len(mean2_users) != 1:
                continue
            mean2_idx = int(list(mean2_users)[0])
            mean2_op = model_ir.operators[int(mean2_idx)]
            if (
                str(mean2_op.op_type) != "MEAN"
                or len(mean2_op.inputs) != 2
                or len(mean2_op.outputs) != 1
                or str(mean2_op.inputs[0]) != squared_name
            ):
                continue
            axes2_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(mean2_op.inputs[1]), None))
            if axes2_vals is None or [int(v) for v in list(axes2_vals)] != [2]:
                continue

            mean2_out_name = str(mean2_op.outputs[0])
            add_eps_users = set(int(v) for v in consumers.get(mean2_out_name, []))
            if len(add_eps_users) != 1:
                continue
            add_eps_idx = int(list(add_eps_users)[0])
            add_eps_op = model_ir.operators[int(add_eps_idx)]
            if (
                str(add_eps_op.op_type) != "ADD"
                or len(add_eps_op.inputs) != 2
                or len(add_eps_op.outputs) != 1
                or mean2_out_name not in {str(add_eps_op.inputs[0]), str(add_eps_op.inputs[1])}
            ):
                continue
            add_eps_out_name = str(add_eps_op.outputs[0])

            sqrt_users = set(int(v) for v in consumers.get(add_eps_out_name, []))
            if len(sqrt_users) != 1:
                continue
            sqrt_idx = int(list(sqrt_users)[0])
            sqrt_op = model_ir.operators[int(sqrt_idx)]
            if (
                str(sqrt_op.op_type) != "SQRT"
                or len(sqrt_op.inputs) != 1
                or len(sqrt_op.outputs) != 1
                or str(sqrt_op.inputs[0]) != add_eps_out_name
            ):
                continue
            sqrt_out_name = str(sqrt_op.outputs[0])

            div_users = set(int(v) for v in consumers.get(sqrt_out_name, []))
            if len(div_users) != 1:
                continue
            div_idx = int(list(div_users)[0])
            div_op = model_ir.operators[int(div_idx)]
            if (
                str(div_op.op_type) != "DIV"
                or len(div_op.inputs) != 2
                or len(div_op.outputs) != 1
                or str(div_op.inputs[1]) != sqrt_out_name
            ):
                continue
            div_out_name = str(div_op.outputs[0])

            mul_norm_op = model_ir.operators[int(mul_norm_idx)]
            if div_out_name not in {str(mul_norm_op.inputs[0]), str(mul_norm_op.inputs[1])}:
                continue
            normalized_name = str(mul_norm_op.outputs[0])

            mul_scale_users = set(int(v) for v in consumers.get(normalized_name, []))
            if len(mul_scale_users) != 1:
                continue
            mul_scale_idx = int(list(mul_scale_users)[0])
            mul_scale_op = model_ir.operators[int(mul_scale_idx)]
            if (
                str(mul_scale_op.op_type) != "MUL"
                or len(mul_scale_op.inputs) != 2
                or len(mul_scale_op.outputs) != 1
                or normalized_name not in {str(mul_scale_op.inputs[0]), str(mul_scale_op.inputs[1])}
            ):
                continue
            scale_const_name = (
                str(mul_scale_op.inputs[0])
                if str(mul_scale_op.inputs[1]) == normalized_name
                else str(mul_scale_op.inputs[1])
            )
            if not _is_const_broadcast_layout_agnostic(scale_const_name):
                continue
            scaled_name = str(mul_scale_op.outputs[0])

            add_bias_users = set(int(v) for v in consumers.get(scaled_name, []))
            if len(add_bias_users) != 1:
                continue
            add_bias_idx = int(list(add_bias_users)[0])
            add_bias_op = model_ir.operators[int(add_bias_idx)]
            if (
                str(add_bias_op.op_type) != "ADD"
                or len(add_bias_op.inputs) != 2
                or len(add_bias_op.outputs) != 1
                or scaled_name not in {str(add_bias_op.inputs[0]), str(add_bias_op.inputs[1])}
            ):
                continue
            bias_const_name = (
                str(add_bias_op.inputs[0])
                if str(add_bias_op.inputs[1]) == scaled_name
                else str(add_bias_op.inputs[1])
            )
            if not _is_const_broadcast_layout_agnostic(bias_const_name):
                continue
            inst_flat_name = str(add_bias_op.outputs[0])

            reshape2_users = [int(v) for v in consumers.get(inst_flat_name, [])]
            if len(reshape2_users) != 1:
                continue
            reshape2_idx = int(reshape2_users[0])
            reshape2_op = model_ir.operators[int(reshape2_idx)]
            if (
                str(reshape2_op.op_type) != "RESHAPE"
                or len(reshape2_op.inputs) < 1
                or len(reshape2_op.outputs) != 1
                or str(reshape2_op.inputs[0]) != inst_flat_name
            ):
                continue
            reshape2_target = _read_reshape_target(reshape2_op)
            if (
                reshape2_target is None
                or len(reshape2_target) != 4
                or any(int(v) <= 0 for v in reshape2_target)
            ):
                continue

            tail_rank4_names: List[str] = [str(reshape2_op.outputs[0])]
            tail_const_names: List[str] = []
            cursor_name = str(reshape2_op.outputs[0])
            pad_idx: Optional[int] = None
            pad_op: Optional[OperatorIR] = None
            visited_names: set[str] = {cursor_name}
            while True:
                cursor_users = [int(v) for v in consumers.get(cursor_name, [])]
                if len(cursor_users) != 1:
                    break
                user_idx = int(cursor_users[0])
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if (
                    user_type in {"PAD", "MIRROR_PAD"}
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == cursor_name
                ):
                    pad_idx = int(user_idx)
                    pad_op = user_op
                    tail_rank4_names.append(str(user_op.outputs[0]))
                    break
                if (
                    user_type in unary_tail_ops
                    and len(user_op.inputs) == 1
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == cursor_name
                ):
                    cursor_name = str(user_op.outputs[0])
                    if cursor_name in visited_names:
                        break
                    visited_names.add(cursor_name)
                    tail_rank4_names.append(cursor_name)
                    continue
                if (
                    user_type in binary_tail_ops
                    and len(user_op.inputs) == 2
                    and len(user_op.outputs) == 1
                    and cursor_name in {str(user_op.inputs[0]), str(user_op.inputs[1])}
                ):
                    other_name = (
                        str(user_op.inputs[0])
                        if str(user_op.inputs[1]) == cursor_name
                        else str(user_op.inputs[1])
                    )
                    if not _can_rewrite_tail_const_to_nhwc(other_name):
                        break
                    tail_const_names.append(str(other_name))
                    cursor_name = str(user_op.outputs[0])
                    if cursor_name in visited_names:
                        break
                    visited_names.add(cursor_name)
                    tail_rank4_names.append(cursor_name)
                    continue
                break
            if pad_idx is None or pad_op is None:
                continue

            pad_out_name = str(pad_op.outputs[0])
            post_users = [int(v) for v in consumers.get(pad_out_name, [])]
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != pad_out_name
                or _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue
            post_out_name = str(post_op.outputs[0])

            pads_name = str(pad_op.inputs[1])
            pads_tensor = model_ir.tensors.get(pads_name, None)
            if pads_tensor is None or pads_tensor.data is None:
                continue
            try:
                pads_pairs = np.asarray(pads_tensor.data).reshape(4, 2)
            except Exception:
                continue
            if int(pads_pairs.size) != 8:
                continue

            _replace_operator_input_at(
                model_ir=model_ir,
                op=reshape1_op,
                input_index=0,
                new_input_name=pre_in_name,
                graph_index=graph_index,
            )

            nchw_shape = [int(v) for v in list(reshape2_target)]
            nhwc_shape = [
                int(nchw_shape[0]),
                int(nchw_shape[2]),
                int(nchw_shape[3]),
                int(nchw_shape[1]),
            ]
            if not _set_reshape_target(reshape2_op, nhwc_shape):
                continue

            unique_tail_const_names = []
            seen_const_names: set[str] = set()
            for const_name in tail_const_names:
                key = str(const_name)
                if key in seen_const_names:
                    continue
                seen_const_names.add(key)
                unique_tail_const_names.append(key)
            if any(not _rewrite_tail_const_to_nhwc(const_name) for const_name in unique_tail_const_names):
                continue

            pads_nhwc = np.asarray(
                [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
                dtype=pads_pairs.dtype,
            )
            pads_tensor.data = np.asarray(pads_nhwc)
            pads_tensor.shape = [4, 2]
            pads_tensor.shape_signature = [4, 2]

            _set_operator_outputs(
                model_ir=model_ir,
                op=pad_op,
                new_outputs=[post_out_name],
                graph_index=graph_index,
            )

            for tensor_name in tail_rank4_names:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(tensor_name), None),
                    perm_nchw_to_nhwc,
                )

            old_pad_tensor = model_ir.tensors.get(str(pad_out_name), None)
            post_out_tensor = model_ir.tensors.get(str(post_out_name), None)
            if old_pad_tensor is not None and post_out_tensor is not None:
                post_out_tensor.dtype = str(old_pad_tensor.dtype)
                post_out_tensor.quantization = _clone_quantization(old_pad_tensor.quantization)
                post_out_tensor.shape = [int(v) for v in list(old_pad_tensor.shape)]
                post_out_tensor.shape_signature = (
                    [int(v) for v in list(old_pad_tensor.shape_signature)]
                    if old_pad_tensor.shape_signature is not None
                    else [int(v) for v in list(old_pad_tensor.shape)]
                )
                _permute_tensor_metadata_if_rank_matches(
                    post_out_tensor,
                    perm_nchw_to_nhwc,
                )

            for remove_idx in sorted([int(pre_idx), int(post_idx)], reverse=True):
                graph_index.remove_operator(int(remove_idx))

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": int(rewritten)}


def run_normalization_pad_layout_cleanup(
    model_ir: ModelIR,
    *,
    include_instance: bool = True,
    include_flatten: bool = True,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run decomposed InstanceNorm and flattened global-norm Pad propagation."""

    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        transpose_count = 0
        has_pad = False
        has_mean = False
        for visited, op in enumerate(candidate_model.operators, start=1):
            op_type = str(op.op_type)
            transpose_count += int(op_type == "TRANSPOSE")
            has_pad = has_pad or op_type in {"PAD", "MIRROR_PAD"}
            has_mean = has_mean or op_type == "MEAN"
            if transpose_count >= 2 and has_pad and has_mean:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _candidate_upstream_types(pass_state: ModelIRPassState) -> List[Dict[str, int]]:
        results: List[Dict[str, int]] = []
        for post_op in pass_state.model_ir.operators:
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or _read_transpose_perm(pass_state.model_ir, post_op) != perm_nchw_to_nhwc
            ):
                continue
            pad_idx = pass_state.graph_index.producers.get(str(post_op.inputs[0]))
            if pad_idx is None:
                continue
            pad_op = pass_state.model_ir.operators[int(pad_idx)]
            if str(pad_op.op_type) not in {"PAD", "MIRROR_PAD"} or len(pad_op.inputs) < 2:
                continue
            pending = [str(pad_op.inputs[0])]
            visited_tensors: set[str] = set()
            type_counts: Dict[str, int] = {}
            found_pre = False
            while pending:
                tensor_name = pending.pop()
                if tensor_name in visited_tensors:
                    continue
                visited_tensors.add(tensor_name)
                producer_idx = pass_state.graph_index.producers.get(tensor_name)
                if producer_idx is None:
                    continue
                producer_op = pass_state.model_ir.operators[int(producer_idx)]
                op_type = str(producer_op.op_type)
                if op_type == "TRANSPOSE":
                    if _read_transpose_perm(pass_state.model_ir, producer_op) == perm_nhwc_to_nchw:
                        found_pre = True
                    continue
                type_counts[op_type] = int(type_counts.get(op_type, 0)) + 1
                pending.extend(str(name) for name in producer_op.inputs)
            if found_pre:
                results.append(type_counts)
        return results

    def _has_instance_candidate(pass_state: ModelIRPassState) -> bool:
        for counts in _candidate_upstream_types(pass_state):
            if (
                int(counts.get("MEAN", 0)) >= 2
                and int(counts.get("SUB", 0)) >= 1
                and int(counts.get("SQRT", 0)) >= 1
                and int(counts.get("DIV", 0)) >= 1
            ):
                return True
        return False

    def _has_flatten_candidate(pass_state: ModelIRPassState) -> bool:
        return any(
            int(counts.get("RESHAPE", 0)) >= 2 and int(counts.get("MEAN", 0)) >= 2
            for counts in _candidate_upstream_types(pass_state)
        )

    def _run_instance(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_instancenorm_pad_prepost_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("optimized_transpose_instancenorm_pad_prepost_nhwc_chains", 0)
            ),
        }

    def _run_flatten(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_flatten_globalnorm_pad_prepost_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains",
                    0,
                )
            ),
        }

    specs: List[PassSpec[ModelIRPassState]] = []
    if include_instance:
        specs.append(
            PassSpec(
                pass_id="layout.instancenorm_pad_prepost_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_instance,
                precondition=_has_instance_candidate,
                priority=10,
                transactional=True,
            )
        )
    if include_flatten:
        specs.append(
            PassSpec(
                pass_id="layout.flatten_globalnorm_pad_prepost_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_flatten,
                precondition=_has_flatten_candidate,
                priority=20,
                transactional=True,
            )
        )
    if len(specs) == 0:
        return {
            "optimized_transpose_instancenorm_pad_prepost_nhwc_chains": 0,
            "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 0,
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        layout_state=layout_state,
        default_details={
            "optimized_transpose_instancenorm_pad_prepost_nhwc_chains": 0,
            "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
