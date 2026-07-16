from __future__ import annotations

from typing import Dict, List

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def optimize_convpool_output_transpose_nhwc_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Remove NHWC->NCHW TRANSPOSE directly after Conv/Pool-family outputs and
    keep downstream layout contracts via local adapters only when required.

    Pattern:
      conv_or_pool_out_nhwc --T(0,3,1,2)--> x_nchw
      x_nchw -> elementwise-only subgraph

    Rewrite:
      - bypass leading transpose (subgraph runs in NHWC)
      - keep legacy external NCHW users through local NHWC->NCHW adapters
    """
    rewritten = 0
    channel_last_hint_names = {
        str(v)
        for v in model_ir.metadata.get(
            "assume_channel_last_layout_tensor_names",
            [],
        )
        if str(v) != ""
    }
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    mean_axes_remapped_marker = "__convpool_output_nhwc_axes_remapped__"
    remapped_axes_tensor_names = {
        str(v)
        for v in model_ir.metadata.get(
            "convpool_output_nhwc_remapped_axes_tensor_names",
            [],
        )
        if str(v) != ""
    }
    convpool_ops = {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "TRANSPOSE_CONV",
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
    }
    elementwise_ops = {
        "ABS",
        "ADD",
        "DIV",
        "ERF",
        "EXP",
        "FLOOR",
        "LOG",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "NEG",
        "POW",
        "RSQRT",
        "SIGN",
        "SQRT",
        "SUB",
    }

    def _unique_tensor_name(base_name: str) -> str:
        candidate = str(base_name)
        serial = 0
        while candidate in model_ir.tensors:
            serial += 1
            candidate = f"{base_name}_{serial}"
        return candidate

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
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

            pre_input_producer_idx = producers.get(pre_input_name, None)
            if pre_input_producer_idx is None:
                continue
            if str(model_ir.operators[int(pre_input_producer_idx)].op_type) not in convpool_ops:
                continue

            # Collect forward elementwise-only region from pre_output.
            subgraph_indices: set[int] = set()
            frontier: List[str] = [pre_output_name]
            seen_tensors: set[str] = {pre_output_name}
            while len(frontier) > 0:
                cur = str(frontier.pop())
                for user_idx_raw in consumers.get(cur, []):
                    user_idx = int(user_idx_raw)
                    user_op = model_ir.operators[int(user_idx)]
                    if str(user_op.op_type) not in elementwise_ops:
                        continue
                    if int(user_idx) in subgraph_indices:
                        continue
                    subgraph_indices.add(int(user_idx))
                    for out_name_raw in list(user_op.outputs):
                        out_name = str(out_name_raw)
                        if out_name not in seen_tensors:
                            seen_tensors.add(out_name)
                            frontier.append(out_name)
            if len(subgraph_indices) == 0:
                continue

            # Validate runtime inputs and gather external runtime dependencies.
            valid = True
            external_runtime_inputs: set[str] = set()
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[int(op_idx)]
                if len(op.outputs) != 1:
                    valid = False
                    break
                for input_name_raw in list(op.inputs):
                    input_name = str(input_name_raw)
                    input_tensor = model_ir.tensors.get(input_name, None)
                    if input_tensor is not None and input_tensor.data is not None:
                        continue
                    if input_name == pre_output_name:
                        continue
                    prod_idx = producers.get(input_name, None)
                    if prod_idx is None or int(prod_idx) not in subgraph_indices:
                        external_runtime_inputs.add(str(input_name))
            if not valid:
                continue

            # pre_output must be used only by local elementwise subgraph.
            pre_users = set(int(v) for v in consumers.get(pre_output_name, []))
            if len(pre_users) == 0 or not pre_users.issubset(subgraph_indices):
                continue

            # classify boundary users outside subgraph
            boundary_legacy_users: Dict[str, List[int]] = {}
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[int(op_idx)]
                out_name = str(op.outputs[0])
                if out_name in model_outputs:
                    valid = False
                    break
                users = set(int(v) for v in consumers.get(out_name, []))
                external_users = sorted(list(users - subgraph_indices))
                if len(external_users) == 0:
                    continue
                boundary_legacy_users[out_name] = [int(v) for v in list(external_users)]
            if not valid:
                continue

            # Validate every external runtime input before the first graph
            # mutation so a rejected candidate remains an atomic no-op.
            external_runtime_input_nhwc_shapes: Dict[str, List[int]] = {}
            for external_input_name in sorted(list(external_runtime_inputs)):
                ext_tensor = model_ir.tensors.get(
                    str(external_input_name),
                    None,
                )
                if (
                    ext_tensor is None
                    or ext_tensor.shape is None
                    or len(list(ext_tensor.shape)) != 4
                ):
                    valid = False
                    break
                ext_shape = [int(v) for v in list(ext_tensor.shape)]
                ext_nhwc_shape = _permute_shape(
                    ext_shape,
                    perm_nchw_to_nhwc,
                )
                if ext_nhwc_shape is None:
                    valid = False
                    break
                external_runtime_input_nhwc_shapes[
                    str(external_input_name)
                ] = [int(v) for v in list(ext_nhwc_shape)]
            if not valid:
                continue

            channel_last_hint_names.add(str(pre_input_name))
            boundary_name_set = set(str(v) for v in boundary_legacy_users)
            for op_idx in sorted(list(subgraph_indices)):
                out_name = str(model_ir.operators[int(op_idx)].outputs[0])
                if out_name not in boundary_name_set:
                    channel_last_hint_names.add(out_name)

            # Rewire subgraph from pre_output -> pre_input.
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[int(op_idx)]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=op,
                    new_inputs=[
                        pre_input_name if str(v) == pre_output_name else str(v)
                        for v in list(op.inputs)
                    ],
                )

            # Adapt external runtime NCHW inputs used by subgraph.
            for external_input_name in sorted(list(external_runtime_inputs)):
                ext_tensor = model_ir.tensors[str(external_input_name)]
                ext_nhwc_shape = external_runtime_input_nhwc_shapes[
                    str(external_input_name)
                ]
                adapter_perm_name = _unique_tensor_name(f"{external_input_name}__to_nhwc_perm")
                adapter_output_name = _unique_tensor_name(f"{external_input_name}__to_nhwc")
                model_ir.tensors[str(adapter_perm_name)] = TensorIR(
                    name=str(adapter_perm_name),
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(perm_nchw_to_nhwc, dtype=np.int32),
                    is_variable=False,
                    quantization=None,
                )
                model_ir.tensors[str(adapter_output_name)] = TensorIR(
                    name=str(adapter_output_name),
                    dtype=str(ext_tensor.dtype),
                    shape=[int(v) for v in list(ext_nhwc_shape)],
                    shape_signature=[int(v) for v in list(ext_nhwc_shape)],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(ext_tensor.quantization),
                )
                model_ir.operators.append(
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(external_input_name), str(adapter_perm_name)],
                        outputs=[str(adapter_output_name)],
                    )
                )
                channel_last_hint_names.add(str(adapter_output_name))
                for op_idx in sorted(list(subgraph_indices)):
                    op = model_ir.operators[int(op_idx)]
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=op,
                        new_inputs=[
                            str(adapter_output_name) if str(v) == str(external_input_name) else str(v)
                            for v in list(op.inputs)
                        ],
                    )
            # Metadata update: subgraph tensors are now NHWC.
            for op_idx in sorted(list(subgraph_indices)):
                op = model_ir.operators[int(op_idx)]
                out_name = str(op.outputs[0])
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(out_name, None),
                    perm_nchw_to_nhwc,
                )

            # Keep external users with local NHWC->NCHW adapters at boundaries.
            for boundary_name, legacy_users in boundary_legacy_users.items():
                producer_idx = producers.get(str(boundary_name), None)
                if producer_idx is None or int(producer_idx) not in subgraph_indices:
                    valid = False
                    break
                boundary_nhwc_name = _unique_tensor_name(f"{boundary_name}__to_nhwc")
                channel_last_hint_names.add(str(boundary_nhwc_name))
                producer_op = model_ir.operators[int(producer_idx)]
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=producer_op,
                    new_outputs=[str(boundary_nhwc_name)],
                )
                _replace_tensor_inputs(model_ir, str(boundary_name), str(boundary_nhwc_name))

                boundary_tensor = model_ir.tensors.get(str(boundary_name), None)
                boundary_nhwc_tensor = model_ir.tensors.get(str(boundary_nhwc_name), None)
                if boundary_tensor is not None and boundary_nhwc_tensor is None:
                    boundary_nhwc_tensor = TensorIR(
                        name=str(boundary_nhwc_name),
                        dtype=str(boundary_tensor.dtype),
                        shape=[int(v) for v in list(boundary_tensor.shape)],
                        shape_signature=(
                            [int(v) for v in list(boundary_tensor.shape_signature)]
                            if boundary_tensor.shape_signature is not None
                            else [int(v) for v in list(boundary_tensor.shape)]
                        ),
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(boundary_tensor.quantization),
                    )
                    model_ir.tensors[str(boundary_nhwc_name)] = boundary_nhwc_tensor

                # First, try to absorb NCHW-only MEAN consumers by remapping axes
                # and keeping them on NHWC input directly.
                remapped_legacy_users: set[int] = set()
                for legacy_idx in list(legacy_users):
                    legacy_op = model_ir.operators[int(legacy_idx)]
                    if (
                        str(legacy_op.op_type) != "MEAN"
                        or len(legacy_op.inputs) < 2
                        or len(legacy_op.outputs) != 1
                        or str(legacy_op.inputs[0]) != str(boundary_nhwc_name)
                        or not bool(legacy_op.options.get("keepDims", False))
                    ):
                        continue
                    legacy_options = (
                        dict(legacy_op.options)
                        if isinstance(legacy_op.options, dict)
                        else {}
                    )
                    mean_was_remapped = bool(
                        legacy_options.get(mean_axes_remapped_marker, False)
                    )
                    if (
                        boundary_nhwc_tensor is None
                        or boundary_nhwc_tensor.shape is None
                        or len(list(boundary_nhwc_tensor.shape)) != 4
                    ):
                        continue

                    mean_axes_name = str(legacy_op.inputs[1])
                    mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
                    mean_axes_vals = _read_const_ints_from_tensor(mean_axes_tensor)
                    if mean_axes_vals is None or len(mean_axes_vals) == 0:
                        continue

                    normalized_axes: List[int] = []
                    valid_axes = True
                    for axis in mean_axes_vals:
                        a = int(axis)
                        if a < 0:
                            a += 4
                        if a < 0 or a >= 4:
                            valid_axes = False
                            break
                        normalized_axes.append(int(a))
                    if not valid_axes:
                        continue
                    axes_were_remapped = (
                        mean_was_remapped
                        or mean_axes_name in remapped_axes_tensor_names
                    )
                    mapped_axes = (
                        [int(v) for v in normalized_axes]
                        if axes_were_remapped
                        else [
                            int(perm_nhwc_to_nchw[int(axis)])
                            for axis in normalized_axes
                        ]
                    )

                    if not axes_were_remapped:
                        _write_const_ints_to_tensor(
                            mean_axes_tensor,
                            [int(v) for v in mapped_axes],
                        )
                        remapped_axes_tensor_names.add(mean_axes_name)
                        model_ir.metadata[
                            "convpool_output_nhwc_remapped_axes_tensor_names"
                        ] = sorted(remapped_axes_tensor_names)
                    if isinstance(legacy_op.options, dict):
                        mean_options = dict(legacy_op.options)
                        for key in ["axis", "axes", "onnxRawAxes"]:
                            value = mean_options.get(key, None)
                            if isinstance(value, list) and len(value) == len(mapped_axes):
                                mean_options[key] = [int(v) for v in mapped_axes]
                        mean_options[mean_axes_remapped_marker] = True
                        legacy_op.options = mean_options

                    mean_out_name = str(legacy_op.outputs[0])
                    _permute_tensor_metadata_if_rank_matches(
                        model_ir.tensors.get(mean_out_name, None),
                        perm_nchw_to_nhwc,
                    )

                    for mean_user_idx in consumers.get(mean_out_name, []):
                        mean_user_op = model_ir.operators[int(mean_user_idx)]
                        mean_user_options = (
                            dict(mean_user_op.options)
                            if isinstance(mean_user_op.options, dict)
                            else {}
                        )
                        if (
                            str(mean_user_op.op_type) == "RESHAPE"
                            and len(mean_user_op.inputs) >= 2
                            and len(mean_user_op.outputs) == 1
                            and str(mean_user_op.inputs[0]) == mean_out_name
                            and bool(
                                mean_user_options.get(
                                    "layoutTransposeAsReshape",
                                    False,
                                )
                            )
                            and [
                                int(v)
                                for v in mean_user_options.get(
                                    "layoutTransposePerm",
                                    [],
                                )
                            ]
                            == perm_nchw_to_nhwc
                        ):
                            mean_out_tensor = model_ir.tensors.get(
                                mean_out_name,
                                None,
                            )
                            adapter_out_tensor = model_ir.tensors.get(
                                str(mean_user_op.outputs[0]),
                                None,
                            )
                            if mean_out_tensor is not None:
                                identity_shape = [
                                    int(v) for v in mean_out_tensor.shape
                                ]
                                _write_const_ints_to_tensor(
                                    model_ir.tensors.get(
                                        str(mean_user_op.inputs[1]),
                                        None,
                                    ),
                                    identity_shape,
                                )
                                mean_user_options["newShape"] = list(
                                    identity_shape
                                )
                                mean_user_op.options = mean_user_options
                                if adapter_out_tensor is not None:
                                    adapter_out_tensor.shape = list(
                                        identity_shape
                                    )
                                    adapter_out_tensor.shape_signature = (
                                        list(mean_out_tensor.shape_signature)
                                        if mean_out_tensor.shape_signature
                                        is not None
                                        else list(identity_shape)
                                    )
                            continue
                        if (
                            str(mean_user_op.op_type) == "TRANSPOSE"
                            and len(mean_user_op.inputs) >= 2
                            and len(mean_user_op.outputs) == 1
                            and str(mean_user_op.inputs[0]) == mean_out_name
                            and _read_transpose_perm(model_ir, mean_user_op)
                            == perm_nchw_to_nhwc
                        ):
                            # MEAN now already produces NHWC. Keep the boundary
                            # tensor name but turn its former NCHW->NHWC adapter
                            # into identity so the generic transpose cleanup can
                            # remove it without a second layout permutation.
                            perm_tensor = model_ir.tensors.get(
                                str(mean_user_op.inputs[1]),
                                None,
                            )
                            _write_const_ints_to_tensor(
                                perm_tensor,
                                [0, 1, 2, 3],
                            )
                            mean_out_tensor = model_ir.tensors.get(
                                mean_out_name,
                                None,
                            )
                            adapter_out_tensor = model_ir.tensors.get(
                                str(mean_user_op.outputs[0]),
                                None,
                            )
                            if (
                                mean_out_tensor is not None
                                and adapter_out_tensor is not None
                            ):
                                adapter_out_tensor.shape = list(
                                    mean_out_tensor.shape
                                )
                                adapter_out_tensor.shape_signature = (
                                    list(mean_out_tensor.shape_signature)
                                    if mean_out_tensor.shape_signature is not None
                                    else list(mean_out_tensor.shape)
                                )
                            continue
                        if mean_was_remapped:
                            continue
                        if (
                            str(mean_user_op.op_type) != "SQUEEZE"
                            or len(mean_user_op.inputs) != 1
                            or len(mean_user_op.outputs) != 1
                            or str(mean_user_op.inputs[0]) != mean_out_name
                            or not isinstance(mean_user_op.options, dict)
                            or "squeezeDims" not in mean_user_op.options
                        ):
                            continue
                        raw_axes = list(mean_user_op.options.get("squeezeDims", []))
                        remapped_squeeze_axes: List[int] = []
                        valid_squeeze_axes = True
                        for axis in raw_axes:
                            a = int(axis)
                            if a < 0:
                                a += 4
                            if a < 0 or a >= 4:
                                valid_squeeze_axes = False
                                break
                            remapped_squeeze_axes.append(int(perm_nhwc_to_nchw[int(a)]))
                        if not valid_squeeze_axes:
                            continue
                        squeeze_options = dict(mean_user_op.options)
                        squeeze_options["squeezeDims"] = sorted([int(v) for v in remapped_squeeze_axes])
                        mean_user_op.options = squeeze_options

                    remapped_legacy_users.add(int(legacy_idx))

                pending_legacy_users = [
                    int(v) for v in list(legacy_users)
                    if int(v) not in remapped_legacy_users
                ]
                if len(pending_legacy_users) <= 0:
                    continue

                perm_name = _unique_tensor_name(f"{boundary_name}__to_nchw_perm")
                model_ir.tensors[str(perm_name)] = TensorIR(
                    name=str(perm_name),
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                    is_variable=False,
                    quantization=None,
                )
                model_ir.operators.append(
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(boundary_nhwc_name), str(perm_name)],
                        outputs=[str(boundary_name)],
                    )
                )
                for legacy_idx in list(pending_legacy_users):
                    legacy_op = model_ir.operators[int(legacy_idx)]
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=legacy_op,
                        new_inputs=[
                            str(boundary_name) if str(v) == str(boundary_nhwc_name) else str(v)
                            for v in list(legacy_op.inputs)
                        ],
                    )
            if not valid:
                continue

            # Remove leading transpose.
            del model_ir.operators[int(pre_idx)]
            model_ir.metadata["assume_channel_last_layout_tensor_names"] = sorted(
                channel_last_hint_names
            )
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_convpool_output_transpose_nhwc_passthrough_chains": int(rewritten)}
