from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_shape,
    _prune_unused_tensors,
    _quant_scale_count,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Move a private StridedSlice/Pad/Concat affine bridge to NHWC."""

    stats_key = (
        "optimized_transpose_stridedslice_pad_concat_mul_add_"
        "posttranspose_nhwc_chains"
    )
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    optimized = 0

    def _rank4_metadata(
        tensor: Optional[TensorIR],
    ) -> Optional[Tuple[List[int], List[int]]]:
        if tensor is None:
            return None
        try:
            shape = [int(value) for value in tensor.shape]
            signature = (
                [int(value) for value in tensor.shape_signature]
                if tensor.shape_signature is not None
                else list(shape)
            )
        except (TypeError, ValueError):
            return None
        if len(shape) != 4 or len(signature) != 4:
            return None
        return shape, signature

    def _planned_permuted_quantization(
        quantization: Any,
        permutation: List[int],
    ) -> Tuple[bool, Any]:
        try:
            cloned = _clone_quantization(quantization)
            scale_count = _quant_scale_count(cloned)
        except Exception:
            return False, None
        if cloned is None or int(scale_count) <= 1:
            return True, cloned
        try:
            if isinstance(cloned, dict):
                if "quantized_dimension" not in cloned:
                    return False, None
                old_dimension = int(cloned["quantized_dimension"])
                if old_dimension < 0 or old_dimension >= len(permutation):
                    return False, None
                cloned["quantized_dimension"] = int(
                    permutation.index(old_dimension)
                )
            else:
                old_dimension = int(cloned.quantized_dimension)
                if old_dimension < 0 or old_dimension >= len(permutation):
                    return False, None
                cloned.quantized_dimension = int(
                    permutation.index(old_dimension)
                )
        except (AttributeError, TypeError, ValueError):
            return False, None
        return True, cloned

    def _permute_axis_mask(
        mask: Any,
        permutation: List[int],
    ) -> Optional[int]:
        try:
            normalized_mask = int(mask)
        except (TypeError, ValueError):
            return None
        remapped = 0
        for new_axis, old_axis in enumerate(permutation):
            if ((normalized_mask >> int(old_axis)) & 1) != 0:
                remapped |= 1 << int(new_axis)
        return int(remapped)

    reserved_tensor_names = {
        str(name)
        for name in (
            list(model_ir.tensors)
            + list(model_ir.inputs)
            + list(model_ir.outputs)
            + [
                value
                for operator in model_ir.operators
                for value in list(operator.inputs) + list(operator.outputs)
            ]
        )
    }

    def _unique_tensor_name(base: str, reserved_names: set[str]) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in reserved_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        reserved_names.add(candidate)
        return candidate

    def _tensor_input_sites(
        tensor_name: str,
        graph_index: ModelIRGraphIndex,
    ) -> set[Tuple[int, int]]:
        sites: set[Tuple[int, int]] = set()
        for operator_index in set(
            graph_index.consumer_indices(tensor_name)
        ):
            operator = model_ir.operators[int(operator_index)]
            for input_index, input_name in enumerate(operator.inputs):
                if str(input_name) == str(tensor_name):
                    sites.add((int(operator_index), int(input_index)))
        return sites

    def _typed_i32_constant(
        *,
        tensor_name: str,
        expected_shape: List[int],
        graph_index: ModelIRGraphIndex,
        public_inputs: set[str],
    ) -> Optional[Tuple[TensorIR, np.ndarray]]:
        if (
            tensor_name in public_inputs
            or tensor_name in graph_index.producers
            or tensor_name in graph_index.duplicate_producers
        ):
            return None
        tensor = model_ir.tensors.get(tensor_name)
        if (
            tensor is None
            or tensor.data is None
            or bool(tensor.is_variable)
            or str(tensor.dtype) != "INT32"
            or tensor.quantization is not None
        ):
            return None
        try:
            shape = [int(value) for value in tensor.shape]
            signature = (
                [int(value) for value in tensor.shape_signature]
                if tensor.shape_signature is not None
                else list(shape)
            )
            array = np.asarray(tensor.data)
        except (TypeError, ValueError):
            return None
        if (
            shape != list(expected_shape)
            or signature != list(expected_shape)
            or array.dtype != np.dtype(np.int32)
            or list(array.shape) != list(expected_shape)
        ):
            return None
        return tensor, np.asarray(array).copy()

    def _plan_index_constant_actions(
        *,
        requirements: List[Dict[str, Any]],
        graph_index: ModelIRGraphIndex,
        public_outputs: set[str],
        reserved_names: set[str],
    ) -> Optional[List[Dict[str, Any]]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for requirement in requirements:
            tensor_name = str(requirement["tensor_name"])
            target = np.asarray(
                requirement["target"],
                dtype=np.int32,
            )
            target_shape = [
                int(value) for value in requirement["target_shape"]
            ]
            site = (
                int(requirement["operator_index"]),
                int(requirement["input_index"]),
            )
            group = grouped.get(tensor_name)
            if group is None:
                grouped[tensor_name] = {
                    "tensor": requirement["tensor"],
                    "original": np.asarray(
                        requirement["original"]
                    ).copy(),
                    "target": target,
                    "target_shape": target_shape,
                    "suffix": str(requirement["suffix"]),
                    "sites": {site},
                }
                continue
            if (
                list(group["target_shape"]) != target_shape
                or not np.array_equal(group["target"], target)
            ):
                return None
            group["sites"].add(site)

        actions: List[Dict[str, Any]] = []
        for tensor_name, group in grouped.items():
            tensor = group["tensor"]
            target = np.asarray(group["target"], dtype=np.int32)
            changed = not np.array_equal(group["original"], target)
            all_sites = _tensor_input_sites(tensor_name, graph_index)
            planned_sites = set(group["sites"])
            shared_outside_plan = any(
                site not in planned_sites for site in all_sites
            )
            mode = "none"
            new_name = tensor_name
            if changed and (
                shared_outside_plan or tensor_name in public_outputs
            ):
                mode = "clone"
                new_name = _unique_tensor_name(
                    f"{tensor_name}_{group['suffix']}",
                    reserved_names,
                )
            elif changed:
                mode = "update"
            actions.append(
                {
                    "mode": mode,
                    "tensor_name": tensor_name,
                    "new_name": new_name,
                    "tensor": tensor,
                    "target": target,
                    "target_shape": list(group["target_shape"]),
                    "sites": planned_sites,
                }
            )
        return actions

    def _plan_mul_constant(
        *,
        mul_op: OperatorIR,
        mul_idx: int,
        data_input_name: str,
        target_shape_nhwc: List[int],
        graph_index: ModelIRGraphIndex,
        public_inputs: set[str],
        public_outputs: set[str],
        reserved_names: set[str],
    ) -> Optional[Dict[str, Any]]:
        mul_inputs = [str(value) for value in mul_op.inputs]
        if len(mul_inputs) != 2 or mul_inputs.count(data_input_name) != 1:
            return None
        data_input_index = mul_inputs.index(data_input_name)
        const_input_index = 1 - int(data_input_index)
        const_name = mul_inputs[const_input_index]
        const_tensor = model_ir.tensors.get(const_name)
        if const_tensor is None or const_tensor.data is None:
            return None
        try:
            const_data = np.asarray(const_tensor.data)
        except Exception:
            return None
        if int(const_data.size) == 1:
            return {
                "mode": "none",
                "input_index": int(const_input_index),
                "const_name": const_name,
                "new_name": const_name,
            }

        target_shape = (
            [int(value) for value in target_shape_nhwc]
            if _is_fully_known_positive_shape(target_shape_nhwc)
            else None
        )
        rotated: Optional[np.ndarray] = None
        if int(const_data.ndim) == 4:
            as_is_shape = [int(value) for value in const_data.shape]
            if (
                target_shape is not None
                and _broadcast_static_shapes(target_shape, as_is_shape)
                is not None
            ):
                return {
                    "mode": "none",
                    "input_index": int(const_input_index),
                    "const_name": const_name,
                    "new_name": const_name,
                }
            rotated = np.transpose(
                const_data,
                perm_nchw_to_nhwc,
            ).astype(const_data.dtype, copy=False)
        else:
            if (
                target_shape is None
                or _broadcast_static_shapes(
                    target_shape,
                    [int(value) for value in const_data.shape],
                )
                is None
            ):
                return None
            return {
                "mode": "none",
                "input_index": int(const_input_index),
                "const_name": const_name,
                "new_name": const_name,
            }

        rotated_shape = [int(value) for value in rotated.shape]
        if (
            target_shape is not None
            and _broadcast_static_shapes(
                target_shape,
                rotated_shape,
            )
            is None
        ):
            return None
        if const_name in public_inputs or bool(const_tensor.is_variable):
            return None
        quantization_ok, target_quantization = (
            _planned_permuted_quantization(
                const_tensor.quantization,
                perm_nchw_to_nhwc,
            )
        )
        if not quantization_ok:
            return None

        planned_site = {(int(mul_idx), int(const_input_index))}
        shared_outside_plan = any(
            site not in planned_site
            for site in _tensor_input_sites(const_name, graph_index)
        )
        if shared_outside_plan or const_name in public_outputs:
            new_name = _unique_tensor_name(
                f"{const_name}_nhwc",
                reserved_names,
            )
            return {
                "mode": "clone",
                "input_index": int(const_input_index),
                "const_name": const_name,
                "new_name": new_name,
                "data": np.asarray(rotated),
                "shape": rotated_shape,
                "quantization": target_quantization,
                "dtype": str(const_tensor.dtype),
                "logical_layout": str(const_tensor.logical_layout),
                "physical_layout": str(const_tensor.physical_layout),
                "onnx_tensor_name": const_tensor.onnx_tensor_name,
            }
        return {
            "mode": "update",
            "input_index": int(const_input_index),
            "const_name": const_name,
            "new_name": const_name,
            "data": np.asarray(rotated),
            "shape": rotated_shape,
            "quantization": target_quantization,
        }

    graph_index = ModelIRGraphIndex(model_ir)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_boundaries = public_inputs | public_outputs

    while True:
        changed = False
        for post_idx in graph_index.operator_indices("TRANSPOSE"):
            post_op = model_ir.operators[int(post_idx)]
            if (
                len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or _read_transpose_perm(model_ir, post_op)
                != perm_nchw_to_nhwc
            ):
                continue
            mul_out_name = str(post_op.inputs[0])
            post_out_name = str(post_op.outputs[0])
            if (
                mul_out_name in public_boundaries
                or post_out_name in public_boundaries
                or mul_out_name in graph_index.duplicate_producers
                or post_out_name in graph_index.duplicate_producers
            ):
                continue

            mul_idx = graph_index.producers.get(mul_out_name)
            if mul_idx is None or int(mul_idx) >= int(post_idx):
                continue
            mul_op = model_ir.operators[int(mul_idx)]
            if (
                str(mul_op.op_type) != "MUL"
                or len(mul_op.inputs) != 2
                or len(mul_op.outputs) != 1
                or str(mul_op.outputs[0]) != mul_out_name
                or graph_index.consumer_indices(mul_out_name)
                != [int(post_idx)]
            ):
                continue

            raw_add_users = graph_index.consumer_indices(post_out_name)
            add_indices: List[int] = []
            for add_index in raw_add_users:
                if int(add_index) not in add_indices:
                    add_indices.append(int(add_index))
            if len(add_indices) == 0:
                continue
            add_side_arrays: List[np.ndarray] = []
            valid_add_tail = True
            for add_idx in add_indices:
                if int(add_idx) <= int(post_idx):
                    valid_add_tail = False
                    break
                add_op = model_ir.operators[int(add_idx)]
                if (
                    str(add_op.op_type) != "ADD"
                    or len(add_op.inputs) != 2
                    or len(add_op.outputs) != 1
                ):
                    valid_add_tail = False
                    break
                add_inputs = [str(value) for value in add_op.inputs]
                if add_inputs.count(post_out_name) != 1:
                    valid_add_tail = False
                    break
                add_input_index = add_inputs.index(post_out_name)
                side_name = add_inputs[1 - add_input_index]
                side_tensor = model_ir.tensors.get(side_name)
                if side_tensor is None or side_tensor.data is None:
                    valid_add_tail = False
                    break
                try:
                    side_data = np.asarray(side_tensor.data)
                except Exception:
                    valid_add_tail = False
                    break
                if int(side_data.size) != 1:
                    side_shape = [
                        int(value) for value in side_data.shape
                    ]
                    if (
                        len(side_shape) != 4
                        or not (
                            int(side_shape[0]) == 1
                            and int(side_shape[1]) == 1
                            and int(side_shape[2]) == 1
                            and int(side_shape[3]) > 0
                        )
                    ):
                        valid_add_tail = False
                        break
                add_side_arrays.append(np.asarray(side_data))
            if not valid_add_tail:
                continue

            mul_inputs = [str(value) for value in mul_op.inputs]
            concat_idx: Optional[int] = None
            concat_out_name: Optional[str] = None
            for mul_input_name in mul_inputs:
                if mul_input_name in graph_index.duplicate_producers:
                    continue
                producer_idx = graph_index.producers.get(mul_input_name)
                if (
                    producer_idx is None
                    or int(producer_idx) >= int(mul_idx)
                ):
                    continue
                producer_op = model_ir.operators[int(producer_idx)]
                if (
                    str(producer_op.op_type) != "CONCATENATION"
                    or len(producer_op.outputs) != 1
                    or str(producer_op.outputs[0]) != mul_input_name
                    or not isinstance(producer_op.options, dict)
                ):
                    continue
                try:
                    axis = int(producer_op.options.get("axis", 1))
                except (TypeError, ValueError):
                    continue
                if axis < 0:
                    axis += 4
                if axis != 1:
                    continue
                concat_idx = int(producer_idx)
                concat_out_name = mul_input_name
                break
            if (
                concat_idx is None
                or concat_out_name is None
                or concat_out_name in public_boundaries
                or set(graph_index.consumer_indices(concat_out_name))
                != {int(mul_idx)}
            ):
                continue

            concat_op = model_ir.operators[int(concat_idx)]
            concat_inputs = [str(value) for value in concat_op.inputs]
            if len(concat_inputs) < 2:
                continue

            branch_plans: List[Dict[str, Any]] = []
            index_requirements: List[Dict[str, Any]] = []
            valid_branches = True
            for concat_input_name in concat_inputs:
                if (
                    concat_input_name in public_boundaries
                    or concat_input_name in graph_index.duplicate_producers
                ):
                    valid_branches = False
                    break
                pad_idx = graph_index.producers.get(concat_input_name)
                if pad_idx is None or int(pad_idx) >= int(concat_idx):
                    valid_branches = False
                    break
                pad_op = model_ir.operators[int(pad_idx)]
                if (
                    str(pad_op.op_type) not in {"PAD", "MIRROR_PAD"}
                    or len(pad_op.inputs) < 2
                    or len(pad_op.outputs) != 1
                    or str(pad_op.outputs[0]) != concat_input_name
                    or set(
                        graph_index.consumer_indices(concat_input_name)
                    )
                    != {int(concat_idx)}
                ):
                    valid_branches = False
                    break

                slice_out_name = str(pad_op.inputs[0])
                if (
                    slice_out_name in public_boundaries
                    or slice_out_name in graph_index.duplicate_producers
                ):
                    valid_branches = False
                    break
                slice_idx = graph_index.producers.get(slice_out_name)
                if slice_idx is None or int(slice_idx) >= int(pad_idx):
                    valid_branches = False
                    break
                slice_op = model_ir.operators[int(slice_idx)]
                if (
                    str(slice_op.op_type) != "STRIDED_SLICE"
                    or len(slice_op.inputs) < 4
                    or len(slice_op.outputs) != 1
                    or str(slice_op.outputs[0]) != slice_out_name
                    or set(graph_index.consumer_indices(slice_out_name))
                    != {int(pad_idx)}
                ):
                    valid_branches = False
                    break

                slice_options = (
                    dict(slice_op.options)
                    if isinstance(slice_op.options, dict)
                    else {}
                )
                try:
                    ellipsis_mask = int(
                        slice_options.get("ellipsisMask", 0)
                    )
                    new_axis_mask = int(
                        slice_options.get("newAxisMask", 0)
                    )
                    shrink_axis_mask = int(
                        slice_options.get("shrinkAxisMask", 0)
                    )
                    begin_mask = int(
                        slice_options.get("beginMask", 0)
                    )
                    end_mask = int(slice_options.get("endMask", 0))
                except (TypeError, ValueError):
                    valid_branches = False
                    break
                if any(
                    value != 0
                    for value in (
                        ellipsis_mask,
                        new_axis_mask,
                        shrink_axis_mask,
                    )
                ):
                    valid_branches = False
                    break
                target_begin_mask = _permute_axis_mask(
                    begin_mask,
                    perm_nchw_to_nhwc,
                )
                target_end_mask = _permute_axis_mask(
                    end_mask,
                    perm_nchw_to_nhwc,
                )
                if (
                    target_begin_mask is None
                    or target_end_mask is None
                ):
                    valid_branches = False
                    break

                pre_out_name = str(slice_op.inputs[0])
                if (
                    pre_out_name in public_boundaries
                    or pre_out_name in graph_index.duplicate_producers
                ):
                    valid_branches = False
                    break
                pre_idx = graph_index.producers.get(pre_out_name)
                if pre_idx is None or int(pre_idx) >= int(slice_idx):
                    valid_branches = False
                    break
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != pre_out_name
                    or _read_transpose_perm(model_ir, pre_op)
                    != perm_nhwc_to_nchw
                    or set(graph_index.consumer_indices(pre_out_name))
                    != {int(slice_idx)}
                ):
                    valid_branches = False
                    break

                source_name = str(pre_op.inputs[0])
                source_producer_idx = graph_index.producers.get(
                    source_name
                )
                if (
                    source_name in graph_index.duplicate_producers
                    or (
                        source_producer_idx is not None
                        and int(source_producer_idx) >= int(pre_idx)
                    )
                    or _rank4_metadata(
                        model_ir.tensors.get(source_name)
                    )
                    is None
                ):
                    valid_branches = False
                    break

                slice_metadata = _rank4_metadata(
                    model_ir.tensors.get(slice_out_name)
                )
                pad_metadata = _rank4_metadata(
                    model_ir.tensors.get(concat_input_name)
                )
                if slice_metadata is None or pad_metadata is None:
                    valid_branches = False
                    break
                slice_shape, slice_signature = slice_metadata
                pad_shape, pad_signature = pad_metadata
                target_slice_shape = _permute_shape(
                    slice_shape,
                    perm_nchw_to_nhwc,
                )
                target_slice_signature = _permute_shape(
                    slice_signature,
                    perm_nchw_to_nhwc,
                )
                target_pad_shape = _permute_shape(
                    pad_shape,
                    perm_nchw_to_nhwc,
                )
                target_pad_signature = _permute_shape(
                    pad_signature,
                    perm_nchw_to_nhwc,
                )
                slice_quantization_ok, target_slice_quantization = (
                    _planned_permuted_quantization(
                        model_ir.tensors[
                            slice_out_name
                        ].quantization,
                        perm_nchw_to_nhwc,
                    )
                )
                pad_quantization_ok, target_pad_quantization = (
                    _planned_permuted_quantization(
                        model_ir.tensors[
                            concat_input_name
                        ].quantization,
                        perm_nchw_to_nhwc,
                    )
                )
                if (
                    target_slice_shape is None
                    or target_slice_signature is None
                    or target_pad_shape is None
                    or target_pad_signature is None
                    or not slice_quantization_ok
                    or not pad_quantization_ok
                ):
                    valid_branches = False
                    break

                typed_constants: List[
                    Tuple[str, TensorIR, np.ndarray, List[int], str]
                ] = []
                for input_index, suffix in (
                    (1, "nhwc_begin"),
                    (2, "nhwc_end"),
                    (3, "nhwc_stride"),
                ):
                    tensor_name = str(slice_op.inputs[input_index])
                    typed = _typed_i32_constant(
                        tensor_name=tensor_name,
                        expected_shape=[4],
                        graph_index=graph_index,
                        public_inputs=public_inputs,
                    )
                    if typed is None:
                        valid_branches = False
                        break
                    tensor, array = typed
                    typed_constants.append(
                        (
                            tensor_name,
                            tensor,
                            array,
                            [4],
                            suffix,
                        )
                    )
                if not valid_branches:
                    break
                pads_name = str(pad_op.inputs[1])
                typed_pads = _typed_i32_constant(
                    tensor_name=pads_name,
                    expected_shape=[4, 2],
                    graph_index=graph_index,
                    public_inputs=public_inputs,
                )
                if typed_pads is None:
                    valid_branches = False
                    break
                pads_tensor, pads_array = typed_pads

                target_slice_options = dict(slice_options)
                target_slice_options["beginMask"] = int(
                    target_begin_mask
                )
                target_slice_options["endMask"] = int(target_end_mask)
                target_values = [
                    np.asarray(array)[perm_nchw_to_nhwc]
                    for _, _, array, _, _ in typed_constants
                ]
                target_pads = np.asarray(pads_array)[
                    perm_nchw_to_nhwc,
                    :,
                ]
                for (
                    input_index,
                    typed_constant,
                    target_value,
                ) in zip(
                    (1, 2, 3),
                    typed_constants,
                    target_values,
                ):
                    (
                        tensor_name,
                        tensor,
                        original,
                        target_shape,
                        suffix,
                    ) = typed_constant
                    index_requirements.append(
                        {
                            "tensor_name": tensor_name,
                            "tensor": tensor,
                            "original": original,
                            "target": np.asarray(target_value),
                            "target_shape": target_shape,
                            "suffix": suffix,
                            "operator_index": int(slice_idx),
                            "input_index": int(input_index),
                        }
                    )
                index_requirements.append(
                    {
                        "tensor_name": pads_name,
                        "tensor": pads_tensor,
                        "original": pads_array,
                        "target": np.asarray(target_pads),
                        "target_shape": [4, 2],
                        "suffix": "nhwc_pads",
                        "operator_index": int(pad_idx),
                        "input_index": 1,
                    }
                )
                branch_plans.append(
                    {
                        "pre_idx": int(pre_idx),
                        "slice_idx": int(slice_idx),
                        "slice_op": slice_op,
                        "slice_out_name": slice_out_name,
                        "slice_options": target_slice_options,
                        "source_name": source_name,
                        "target_slice_shape": list(
                            target_slice_shape
                        ),
                        "target_slice_signature": list(
                            target_slice_signature
                        ),
                        "target_slice_quantization": (
                            target_slice_quantization
                        ),
                        "pad_idx": int(pad_idx),
                        "pad_op": pad_op,
                        "pad_out_name": concat_input_name,
                        "target_pad_shape": list(target_pad_shape),
                        "target_pad_signature": list(
                            target_pad_signature
                        ),
                        "target_pad_quantization": (
                            target_pad_quantization
                        ),
                    }
                )
            if not valid_branches or len(branch_plans) < 2:
                continue

            concat_metadata = _rank4_metadata(
                model_ir.tensors.get(concat_out_name)
            )
            mul_metadata = _rank4_metadata(
                model_ir.tensors.get(mul_out_name)
            )
            post_out_tensor = model_ir.tensors.get(post_out_name)
            if (
                concat_metadata is None
                or mul_metadata is None
                or post_out_tensor is None
            ):
                continue
            concat_shape, concat_signature = concat_metadata
            mul_shape, mul_signature = mul_metadata
            target_concat_shape = _permute_shape(
                concat_shape,
                perm_nchw_to_nhwc,
            )
            target_concat_signature = _permute_shape(
                concat_signature,
                perm_nchw_to_nhwc,
            )
            target_mul_shape = _permute_shape(
                mul_shape,
                perm_nchw_to_nhwc,
            )
            target_mul_signature = _permute_shape(
                mul_signature,
                perm_nchw_to_nhwc,
            )
            concat_quantization_ok, target_concat_quantization = (
                _planned_permuted_quantization(
                    model_ir.tensors[concat_out_name].quantization,
                    perm_nchw_to_nhwc,
                )
            )
            mul_quantization_ok, target_mul_quantization = (
                _planned_permuted_quantization(
                    model_ir.tensors[mul_out_name].quantization,
                    perm_nchw_to_nhwc,
                )
            )
            if (
                target_concat_shape is None
                or target_concat_signature is None
                or target_mul_shape is None
                or target_mul_signature is None
                or not concat_quantization_ok
                or not mul_quantization_ok
            ):
                continue
            add_broadcast_ok = True
            for add_side_data in add_side_arrays:
                if int(add_side_data.size) == 1:
                    continue
                if (
                    _is_fully_known_positive_shape(target_mul_shape)
                    and _broadcast_static_shapes(
                        target_mul_shape,
                        [
                            int(value)
                            for value in add_side_data.shape
                        ],
                    )
                    is None
                ):
                    add_broadcast_ok = False
                    break
            if not add_broadcast_ok:
                continue

            candidate_reserved_names = set(reserved_tensor_names)
            index_constant_actions = _plan_index_constant_actions(
                requirements=index_requirements,
                graph_index=graph_index,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if index_constant_actions is None:
                continue
            mul_constant_plan = _plan_mul_constant(
                mul_op=mul_op,
                mul_idx=int(mul_idx),
                data_input_name=concat_out_name,
                target_shape_nhwc=list(target_concat_shape),
                graph_index=graph_index,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if mul_constant_plan is None:
                continue

            input_updates: Dict[int, List[str]] = {}
            for branch_plan in branch_plans:
                slice_idx = int(branch_plan["slice_idx"])
                input_updates[slice_idx] = [
                    str(value)
                    for value in model_ir.operators[slice_idx].inputs
                ]
                input_updates[slice_idx][0] = str(
                    branch_plan["source_name"]
                )
                pad_idx = int(branch_plan["pad_idx"])
                input_updates[pad_idx] = [
                    str(value)
                    for value in model_ir.operators[pad_idx].inputs
                ]
            for action in index_constant_actions:
                replacement_name = str(action["new_name"])
                for operator_index, input_index in action["sites"]:
                    input_updates[int(operator_index)][
                        int(input_index)
                    ] = replacement_name
            input_updates[int(mul_idx)] = [
                str(value) for value in mul_op.inputs
            ]
            input_updates[int(mul_idx)][
                int(mul_constant_plan["input_index"])
            ] = str(mul_constant_plan["new_name"])

            target_concat_options = dict(concat_op.options)
            target_concat_options["axis"] = 3
            remove_indices = {
                int(post_idx),
                *(
                    int(branch_plan["pre_idx"])
                    for branch_plan in branch_plans
                ),
            }

            # Every topology, typed constant, metadata, quantization, option,
            # setter, output rename, and removal decision is complete.
            reserved_tensor_names.update(candidate_reserved_names)

            for action in index_constant_actions:
                mode = str(action["mode"])
                if mode == "clone":
                    source_tensor = action["tensor"]
                    clone_name = str(action["new_name"])
                    model_ir.tensors[clone_name] = TensorIR(
                        name=clone_name,
                        dtype="INT32",
                        shape=list(action["target_shape"]),
                        shape_signature=list(action["target_shape"]),
                        data=np.asarray(
                            action["target"],
                            dtype=np.int32,
                        ),
                        is_variable=False,
                        quantization=None,
                        logical_layout=str(
                            source_tensor.logical_layout
                        ),
                        physical_layout=str(
                            source_tensor.physical_layout
                        ),
                        onnx_tensor_name=source_tensor.onnx_tensor_name,
                    )
                elif mode == "update":
                    tensor = action["tensor"]
                    tensor.data = np.asarray(
                        action["target"],
                        dtype=np.int32,
                    )
                    tensor.shape = list(action["target_shape"])
                    tensor.shape_signature = list(
                        action["target_shape"]
                    )
                    tensor.quantization = None

            constant_mode = str(mul_constant_plan["mode"])
            if constant_mode == "clone":
                constant_name = str(mul_constant_plan["new_name"])
                model_ir.tensors[constant_name] = TensorIR(
                    name=constant_name,
                    dtype=str(mul_constant_plan["dtype"]),
                    shape=list(mul_constant_plan["shape"]),
                    shape_signature=list(mul_constant_plan["shape"]),
                    data=np.asarray(mul_constant_plan["data"]),
                    is_variable=False,
                    quantization=mul_constant_plan["quantization"],
                    logical_layout=str(
                        mul_constant_plan["logical_layout"]
                    ),
                    physical_layout=str(
                        mul_constant_plan["physical_layout"]
                    ),
                    onnx_tensor_name=mul_constant_plan[
                        "onnx_tensor_name"
                    ],
                )
            elif constant_mode == "update":
                constant_tensor = model_ir.tensors[
                    str(mul_constant_plan["const_name"])
                ]
                constant_tensor.data = np.asarray(
                    mul_constant_plan["data"]
                )
                constant_tensor.shape = list(
                    mul_constant_plan["shape"]
                )
                constant_tensor.shape_signature = list(
                    mul_constant_plan["shape"]
                )
                constant_tensor.quantization = mul_constant_plan[
                    "quantization"
                ]

            for operator_index in sorted(input_updates):
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=model_ir.operators[int(operator_index)],
                    new_inputs=list(input_updates[operator_index]),
                    graph_index=graph_index,
                )

            for branch_plan in branch_plans:
                slice_op = branch_plan["slice_op"]
                slice_op.options = dict(
                    branch_plan["slice_options"]
                )
                slice_tensor = model_ir.tensors[
                    str(branch_plan["slice_out_name"])
                ]
                slice_tensor.shape = list(
                    branch_plan["target_slice_shape"]
                )
                slice_tensor.shape_signature = list(
                    branch_plan["target_slice_signature"]
                )
                slice_tensor.quantization = branch_plan[
                    "target_slice_quantization"
                ]
                pad_tensor = model_ir.tensors[
                    str(branch_plan["pad_out_name"])
                ]
                pad_tensor.shape = list(
                    branch_plan["target_pad_shape"]
                )
                pad_tensor.shape_signature = list(
                    branch_plan["target_pad_signature"]
                )
                pad_tensor.quantization = branch_plan[
                    "target_pad_quantization"
                ]

            concat_op.options = target_concat_options
            concat_tensor = model_ir.tensors[concat_out_name]
            concat_tensor.shape = list(target_concat_shape)
            concat_tensor.shape_signature = list(
                target_concat_signature
            )
            concat_tensor.quantization = target_concat_quantization

            old_mul_tensor = model_ir.tensors[mul_out_name]
            post_out_tensor.dtype = str(old_mul_tensor.dtype)
            post_out_tensor.shape = list(target_mul_shape)
            post_out_tensor.shape_signature = list(
                target_mul_signature
            )
            post_out_tensor.quantization = target_mul_quantization
            _set_operator_outputs(
                model_ir=model_ir,
                op=mul_op,
                new_outputs=[post_out_name],
                graph_index=graph_index,
            )
            graph_index.remove_operators(remove_indices)

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {stats_key: int(optimized)}
