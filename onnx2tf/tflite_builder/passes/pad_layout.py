from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _append_tensor_lineage_event,
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
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
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    candidate = str(base)
    suffix = 0
    while candidate in model_ir.tensors:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def repair_channel_last_inputs_for_channel_first_pad(model_ir: ModelIR) -> Dict[str, int]:
    """Insert an NHWC->NCHW adapter when a Pad retains its ONNX NCHW contract.

    Boundary Slice propagation can move the Pad input to NHWC after lowering,
    while the static output shape and paddings remain in ONNX NCHW order.  The
    repair is accepted only when the full input/output/padding shape equation
    proves that exact mismatch.  Names and model-specific patterns are not
    considered.
    """

    repaired = 0
    pad_index = 0
    while pad_index < len(model_ir.operators):
        pad_op = model_ir.operators[pad_index]
        if (
            str(pad_op.op_type) not in {"PAD", "PADV2", "MIRROR_PAD"}
            or len(pad_op.inputs) < 2
            or len(pad_op.outputs) != 1
        ):
            pad_index += 1
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
            pad_index += 1
            continue

        input_shape = [int(v) for v in list(input_tensor.shape)]
        output_shape = [int(v) for v in list(output_tensor.shape)]
        try:
            pad_pairs = np.asarray(pads_tensor.data, dtype=np.int64).reshape(4, 2)
        except (TypeError, ValueError):
            pad_index += 1
            continue
        if (
            len(input_shape) != 4
            or len(output_shape) != 4
            or any(int(v) <= 0 for v in input_shape + output_shape)
            or np.any(pad_pairs < 0)
        ):
            pad_index += 1
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
            pad_index += 1
            continue

        adapter_name = _unique_tensor_name(model_ir, f"{output_name}_pad_input_nchw")
        perm_name = _unique_tensor_name(model_ir, f"{output_name}_pad_input_nchw_perm")
        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else list(input_shape)
        )
        adapter_quantization = _clone_quantization(input_tensor.quantization)
        if isinstance(adapter_quantization, QuantParamIR):
            old_axis = int(adapter_quantization.quantized_dimension)
            if 0 <= old_axis < 4:
                adapter_quantization.quantized_dimension = int([0, 3, 1, 2].index(old_axis))

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
        pad_op.inputs = updated_inputs
        model_ir.operators.insert(pad_index, transpose_op)
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
        pad_index += 2

    return {"repaired_channel_last_inputs_for_channel_first_pad": int(repaired)}


def _optimize_transpose_pad_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
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
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
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
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_pad_prepost_nhwc_chains": int(rewritten)}


def _optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains(
    model_ir: ModelIR,
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
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
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
                del model_ir.operators[int(remove_idx)]

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
                    model_ir.operators.insert(
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

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains": int(rewritten)}


def _optimize_transpose_pad_mul_posttranspose_add_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate strict NHWC<->NCHW round-trips around:
      T(NHWC->NCHW) -> PAD|MIRROR_PAD -> MUL(const) -> T(NCHW->NHWC) -> ADD(const)

    Rewrite:
      PAD|MIRROR_PAD(NHWC pads) -> MUL(const_nhwc) -> ADD(const_nhwc)
    """
    rewritten = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _rewrite_rank4_mul_const_to_nhwc(
        *,
        mul_op: OperatorIR,
        mul_idx: int,
        const_input_index: int,
        const_input_name: str,
        target_nhwc_shape: Optional[List[int]],
        chain_index_set: set[int],
        consumers: Dict[str, List[int]],
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

        const_users = [int(v) for v in consumers.get(str(const_input_name), [])]
        shared_outside_chain = any(int(v) not in chain_index_set for v in const_users)

        target_name = str(const_input_name)
        if shared_outside_chain:
            target_name = _unique_tensor_name(f"{const_input_name}_nhwc")
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
            )
        return True

    def _rewrite_pad_spec_to_nhwc(
        *,
        pad_op: OperatorIR,
        pad_idx: int,
        pads_input_name: str,
        chain_index_set: set[int],
        consumers: Dict[str, List[int]],
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

        pads_users = [int(v) for v in consumers.get(str(pads_input_name), [])]
        shared_outside_chain = any(int(v) not in chain_index_set for v in pads_users)
        target_name = str(pads_input_name)
        if shared_outside_chain:
            target_name = _unique_tensor_name(f"{pads_input_name}_nhwc")
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
            )
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
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
                consumers=consumers,
            ):
                continue

            if not _rewrite_pad_spec_to_nhwc(
                pad_op=pad_op,
                pad_idx=int(pad_idx),
                pads_input_name=str(pad_op.inputs[1]),
                chain_index_set=chain_index_set,
                consumers=consumers,
            ):
                continue

            pre_input_name = str(pre_op.inputs[0])
            pad_inputs = [str(v) for v in list(pad_op.inputs)]
            pad_inputs[0] = str(pre_input_name)
            _set_operator_inputs(
                model_ir=model_ir,
                op=pad_op,
                new_inputs=pad_inputs,
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
                del model_ir.operators[int(remove_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_transpose_pad_mul_posttranspose_add_nhwc_chains": int(rewritten)}
