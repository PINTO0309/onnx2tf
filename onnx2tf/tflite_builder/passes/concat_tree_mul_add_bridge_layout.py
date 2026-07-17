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
    _replace_operator_input_at,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Move a private nested-Concat/Mul/Transpose/Add bridge to NHWC."""

    stats_key = "optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains"
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

    def _remap_nchw_axis_to_nhwc(axis: int) -> Optional[int]:
        try:
            remapped_axis = int(axis)
        except (TypeError, ValueError):
            return None
        if remapped_axis < 0:
            remapped_axis += 4
        if remapped_axis < 0 or remapped_axis >= 4:
            return None
        try:
            return int(perm_nchw_to_nhwc.index(int(remapped_axis)))
        except ValueError:
            return None

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

    def _plan_mul_constant(
        *,
        mul_op: OperatorIR,
        data_input_name: str,
        target_shape_nhwc: Optional[List[int]],
        graph_index: ModelIRGraphIndex,
        chain_indices: set[int],
        public_inputs: set[str],
        public_outputs: set[str],
        reserved_names: set[str],
    ) -> Optional[Dict[str, Any]]:
        mul_inputs = [str(value) for value in mul_op.inputs]
        if len(mul_inputs) != 2 or mul_inputs.count(data_input_name) != 1:
            return None
        data_input_index = mul_inputs.index(data_input_name)
        const_input_index = 1 - int(data_input_index)
        const_name = str(mul_inputs[const_input_index])
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
        constant_permutation: Optional[List[int]] = None
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
            constant_permutation = list(perm_nchw_to_nhwc)
            rotated = np.transpose(
                const_data,
                constant_permutation,
            ).astype(const_data.dtype, copy=False)
        else:
            if target_shape is None:
                return None
            if _broadcast_static_shapes(
                target_shape,
                [int(value) for value in const_data.shape],
            ) is None:
                return None
            return {
                "mode": "none",
                "input_index": int(const_input_index),
                "const_name": const_name,
                "new_name": const_name,
            }

        if rotated is None:
            return None
        rotated_shape = [int(value) for value in rotated.shape]
        if (
            target_shape is not None
            and _broadcast_static_shapes(target_shape, rotated_shape) is None
        ):
            return None
        if const_name in public_inputs or bool(const_tensor.is_variable):
            return None
        quantization_ok, rotated_quantization = (
            _planned_permuted_quantization(
                const_tensor.quantization,
                constant_permutation,
            )
        )
        if not quantization_ok:
            return None

        const_users = graph_index.consumer_indices(const_name)
        shared_outside_chain = any(
            int(user_index) not in chain_indices
            for user_index in const_users
        )
        if shared_outside_chain or const_name in public_outputs:
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
                "quantization": rotated_quantization,
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
            "quantization": rotated_quantization,
        }

    def _collect_concat_tree(
        *,
        root_concat_idx: int,
        mul_idx: int,
        graph_index: ModelIRGraphIndex,
        public_boundaries: set[str],
    ) -> Optional[Dict[str, Any]]:
        concat_indices: set[int] = set()
        leaf_pre_indices: set[int] = set()
        concat_input_updates: Dict[int, List[str]] = {}
        concat_axis_updates: Dict[int, int] = {}
        stack: List[Tuple[int, int]] = [(int(root_concat_idx), int(mul_idx))]

        while len(stack) > 0:
            concat_idx, expected_consumer_idx = stack.pop()
            if int(concat_idx) in concat_indices:
                continue
            if (
                int(concat_idx) < 0
                or int(concat_idx) >= len(model_ir.operators)
                or int(concat_idx) >= int(expected_consumer_idx)
            ):
                return None
            concat_op = model_ir.operators[int(concat_idx)]
            if (
                str(concat_op.op_type) != "CONCATENATION"
                or len(concat_op.outputs) != 1
                or not isinstance(concat_op.options, dict)
            ):
                return None
            remapped_axis = _remap_nchw_axis_to_nhwc(
                concat_op.options.get("axis", 1)
            )
            if remapped_axis is None:
                return None
            concat_out_name = str(concat_op.outputs[0])
            if (
                concat_out_name in public_boundaries
                or concat_out_name in graph_index.duplicate_producers
            ):
                return None
            if set(
                graph_index.consumer_indices(concat_out_name)
            ) != {int(expected_consumer_idx)}:
                return None
            concat_indices.add(int(concat_idx))
            concat_axis_updates[int(concat_idx)] = int(remapped_axis)

            concat_inputs = [str(value) for value in concat_op.inputs]
            if len(concat_inputs) < 2:
                return None
            new_concat_inputs: List[str] = []
            for concat_input_name in concat_inputs:
                if (
                    concat_input_name in public_boundaries
                    or concat_input_name in graph_index.duplicate_producers
                ):
                    return None
                producer_idx = graph_index.producers.get(concat_input_name)
                if (
                    producer_idx is None
                    or int(producer_idx) >= int(concat_idx)
                ):
                    return None
                producer_op = model_ir.operators[int(producer_idx)]
                if (
                    str(producer_op.op_type) == "CONCATENATION"
                    and len(producer_op.outputs) == 1
                    and str(producer_op.outputs[0]) == concat_input_name
                ):
                    stack.append((int(producer_idx), int(concat_idx)))
                    new_concat_inputs.append(concat_input_name)
                    continue
                if (
                    str(producer_op.op_type) == "TRANSPOSE"
                    and len(producer_op.inputs) >= 2
                    and len(producer_op.outputs) == 1
                    and str(producer_op.outputs[0]) == concat_input_name
                    and _read_transpose_perm(model_ir, producer_op)
                    == perm_nhwc_to_nchw
                    and set(
                        graph_index.consumer_indices(concat_input_name)
                    )
                    == {int(concat_idx)}
                ):
                    source_name = str(producer_op.inputs[0])
                    source_producer_idx = graph_index.producers.get(
                        source_name
                    )
                    if (
                        source_name in graph_index.duplicate_producers
                        or (
                            source_producer_idx is not None
                            and int(source_producer_idx) >= int(producer_idx)
                        )
                        or _rank4_metadata(
                            model_ir.tensors.get(source_name)
                        )
                        is None
                    ):
                        return None
                    leaf_pre_indices.add(int(producer_idx))
                    new_concat_inputs.append(source_name)
                    continue
                return None
            concat_input_updates[int(concat_idx)] = new_concat_inputs

        if len(leaf_pre_indices) < 2:
            return None
        return {
            "concat_indices": concat_indices,
            "leaf_pre_indices": leaf_pre_indices,
            "concat_input_updates": concat_input_updates,
            "concat_axis_updates": concat_axis_updates,
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

            add_users = graph_index.consumer_indices(post_out_name)
            if len(add_users) != 1:
                continue
            add_idx = int(add_users[0])
            if add_idx <= int(post_idx):
                continue
            add_op = model_ir.operators[int(add_idx)]
            if (
                str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
            ):
                continue
            add_inputs = [str(value) for value in add_op.inputs]
            if add_inputs.count(post_out_name) != 1:
                continue
            add_input_index = add_inputs.index(post_out_name)
            add_side_name = add_inputs[1 - add_input_index]
            add_side_tensor = model_ir.tensors.get(add_side_name)
            if add_side_tensor is None or add_side_tensor.data is None:
                continue
            try:
                add_side_data = np.asarray(add_side_tensor.data)
                add_side_shape = [
                    int(value) for value in add_side_data.shape
                ]
            except Exception:
                continue

            mul_inputs = [str(value) for value in mul_op.inputs]
            root_concat_idx: Optional[int] = None
            root_concat_out_name: Optional[str] = None
            for mul_input_name in mul_inputs:
                if mul_input_name in graph_index.duplicate_producers:
                    continue
                producer_idx = graph_index.producers.get(mul_input_name)
                if producer_idx is None or int(producer_idx) >= int(mul_idx):
                    continue
                producer_op = model_ir.operators[int(producer_idx)]
                if (
                    str(producer_op.op_type) != "CONCATENATION"
                    or len(producer_op.outputs) != 1
                    or str(producer_op.outputs[0]) != mul_input_name
                    or not isinstance(producer_op.options, dict)
                ):
                    continue
                remapped_root_axis = _remap_nchw_axis_to_nhwc(
                    producer_op.options.get("axis", 1)
                )
                if remapped_root_axis != 3:
                    continue
                root_concat_idx = int(producer_idx)
                root_concat_out_name = mul_input_name
                break
            if (
                root_concat_idx is None
                or root_concat_out_name is None
                or root_concat_out_name in public_boundaries
                or set(
                    graph_index.consumer_indices(root_concat_out_name)
                )
                != {int(mul_idx)}
            ):
                continue

            tree = _collect_concat_tree(
                root_concat_idx=int(root_concat_idx),
                mul_idx=int(mul_idx),
                graph_index=graph_index,
                public_boundaries=public_boundaries,
            )
            if tree is None:
                continue
            concat_indices = set(tree["concat_indices"])
            leaf_pre_indices = set(tree["leaf_pre_indices"])
            concat_input_updates = dict(tree["concat_input_updates"])
            concat_axis_updates = dict(tree["concat_axis_updates"])

            concat_metadata_plans: Dict[int, Dict[str, Any]] = {}
            metadata_ok = True
            for concat_idx in sorted(concat_indices):
                concat_op = model_ir.operators[int(concat_idx)]
                concat_out_name = str(concat_op.outputs[0])
                metadata = _rank4_metadata(
                    model_ir.tensors.get(concat_out_name)
                )
                if metadata is None:
                    metadata_ok = False
                    break
                concat_shape, concat_signature = metadata
                target_shape = _permute_shape(
                    concat_shape,
                    perm_nchw_to_nhwc,
                )
                target_signature = _permute_shape(
                    concat_signature,
                    perm_nchw_to_nhwc,
                )
                quantization_ok, target_quantization = (
                    _planned_permuted_quantization(
                        model_ir.tensors[concat_out_name].quantization,
                        perm_nchw_to_nhwc,
                    )
                )
                if (
                    target_shape is None
                    or target_signature is None
                    or not quantization_ok
                ):
                    metadata_ok = False
                    break
                target_options = dict(concat_op.options)
                target_options["axis"] = int(
                    concat_axis_updates[int(concat_idx)]
                )
                concat_metadata_plans[int(concat_idx)] = {
                    "tensor": model_ir.tensors[concat_out_name],
                    "shape": list(target_shape),
                    "signature": list(target_signature),
                    "quantization": target_quantization,
                    "options": target_options,
                }
            if not metadata_ok:
                continue

            mul_out_metadata = _rank4_metadata(
                model_ir.tensors.get(mul_out_name)
            )
            if mul_out_metadata is None:
                continue
            mul_out_shape, mul_out_signature = mul_out_metadata
            target_mul_shape = _permute_shape(
                mul_out_shape,
                perm_nchw_to_nhwc,
            )
            target_mul_signature = _permute_shape(
                mul_out_signature,
                perm_nchw_to_nhwc,
            )
            mul_quantization_ok, target_mul_quantization = (
                _planned_permuted_quantization(
                    model_ir.tensors[mul_out_name].quantization,
                    perm_nchw_to_nhwc,
                )
            )
            if (
                target_mul_shape is None
                or target_mul_signature is None
                or not mul_quantization_ok
            ):
                continue

            if int(add_side_data.size) != 1:
                if (
                    len(add_side_shape) != 4
                    or not (
                        int(add_side_shape[0]) == 1
                        and int(add_side_shape[1]) == 1
                        and int(add_side_shape[2]) == 1
                        and int(add_side_shape[3]) > 0
                    )
                    or (
                        _is_fully_known_positive_shape(target_mul_shape)
                        and _broadcast_static_shapes(
                            target_mul_shape,
                            add_side_shape,
                        )
                        is None
                    )
                ):
                    continue

            chain_indices: set[int] = {
                int(post_idx),
                int(mul_idx),
                int(add_idx),
            }
            chain_indices.update(int(v) for v in concat_indices)
            chain_indices.update(int(v) for v in leaf_pre_indices)

            candidate_reserved_names = set(reserved_tensor_names)
            root_metadata_plan = concat_metadata_plans[
                int(root_concat_idx)
            ]
            mul_constant_plan = _plan_mul_constant(
                mul_op=mul_op,
                data_input_name=root_concat_out_name,
                target_shape_nhwc=list(root_metadata_plan["shape"]),
                graph_index=graph_index,
                chain_indices=chain_indices,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if mul_constant_plan is None:
                continue

            remove_indices = {
                int(post_idx),
                *(int(value) for value in leaf_pre_indices),
            }

            # Every recursive topology, metadata, constant, quantization,
            # setter, rewire, and removal decision is complete.
            reserved_tensor_names.update(candidate_reserved_names)

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
                updated_mul_inputs = [
                    str(value) for value in mul_op.inputs
                ]
                updated_mul_inputs[
                    int(mul_constant_plan["input_index"])
                ] = constant_name
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=mul_op,
                    new_inputs=updated_mul_inputs,
                    graph_index=graph_index,
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

            for concat_idx in sorted(concat_indices):
                concat_op = model_ir.operators[int(concat_idx)]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=concat_op,
                    new_inputs=list(
                        concat_input_updates[int(concat_idx)]
                    ),
                    graph_index=graph_index,
                )
                concat_plan = concat_metadata_plans[int(concat_idx)]
                concat_op.options = dict(concat_plan["options"])
                concat_tensor = concat_plan["tensor"]
                concat_tensor.shape = list(concat_plan["shape"])
                concat_tensor.shape_signature = list(
                    concat_plan["signature"]
                )
                concat_tensor.quantization = concat_plan["quantization"]

            mul_out_tensor = model_ir.tensors[mul_out_name]
            mul_out_tensor.shape = list(target_mul_shape)
            mul_out_tensor.shape_signature = list(target_mul_signature)
            mul_out_tensor.quantization = target_mul_quantization
            _replace_operator_input_at(
                model_ir=model_ir,
                op=add_op,
                input_index=int(add_input_index),
                new_input_name=mul_out_name,
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
