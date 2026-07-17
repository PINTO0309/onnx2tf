from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _clone_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _quant_scale_count,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_concat_mul_add_transpose_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Move a private Concat/Mul/Transpose/Add bridge from NCHW to NHWC."""

    stats_key = "optimized_concat_mul_add_transpose_nhwc_bridge_chains"
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
        concat_input_name: str,
        graph_index: ModelIRGraphIndex,
        chain_indices: set[int],
        target_shape_nhwc: List[int],
        public_inputs: set[str],
        public_outputs: set[str],
        reserved_names: set[str],
    ) -> Optional[Dict[str, Any]]:
        mul_inputs = [str(value) for value in mul_op.inputs]
        const_input_index: Optional[int] = None
        for input_index, input_name in enumerate(mul_inputs):
            if input_name == concat_input_name:
                continue
            tensor = model_ir.tensors.get(input_name)
            if tensor is not None and tensor.data is not None:
                const_input_index = int(input_index)
                break
        if const_input_index is None:
            return None

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

        rotated: Optional[np.ndarray] = None
        constant_permutation: Optional[List[int]] = None
        if int(const_data.ndim) == 4:
            constant_permutation = list(perm_nchw_to_nhwc)
            rotated = np.transpose(
                const_data,
                constant_permutation,
            ).astype(const_data.dtype, copy=False)
        elif int(const_data.ndim) == 3:
            constant_permutation = [1, 2, 0]
            rotated = np.transpose(
                const_data,
                constant_permutation,
            ).astype(const_data.dtype, copy=False)
        elif _broadcast_static_shapes(
            target_shape_nhwc,
            [int(value) for value in const_data.shape],
        ) is not None:
            return {
                "mode": "none",
                "input_index": int(const_input_index),
                "const_name": const_name,
                "new_name": const_name,
            }
        if rotated is None:
            return None
        rotated_shape = [int(value) for value in rotated.shape]
        if _broadcast_static_shapes(
            target_shape_nhwc,
            rotated_shape,
        ) is None:
            return None
        if (
            const_name in public_inputs
            or bool(const_tensor.is_variable)
        ):
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
            }
        return {
            "mode": "update",
            "input_index": int(const_input_index),
            "const_name": const_name,
            "new_name": const_name,
            "data": np.asarray(rotated),
            "shape": rotated_shape,
            "quantization": rotated_quantization,
            "dtype": str(const_tensor.dtype),
        }

    def _plan_adapter_permutation(
        *,
        public_inputs: set[str],
        reserved_names: set[str],
    ) -> Dict[str, Any]:
        base_name = "__concat_mul_tail_nhwc_to_nchw_perm_rank4__"
        tensor = model_ir.tensors.get(base_name)
        can_reuse = False
        if (
            tensor is not None
            and base_name not in public_inputs
            and not bool(tensor.is_variable)
            and str(tensor.dtype) == "INT32"
            and tensor.quantization is None
            and tensor.data is not None
        ):
            try:
                array = np.asarray(tensor.data)
                tensor_shape = [int(value) for value in tensor.shape]
                tensor_signature = (
                    [int(value) for value in tensor.shape_signature]
                    if tensor.shape_signature is not None
                    else list(tensor_shape)
                )
                can_reuse = (
                    array.dtype == np.dtype(np.int32)
                    and int(array.size) == 4
                    and tensor_shape == [4]
                    and tensor_signature == [4]
                    and [
                        int(value)
                        for value in array.reshape(-1).tolist()
                    ] == perm_nhwc_to_nchw
                )
            except Exception:
                can_reuse = False
        if can_reuse:
            return {"name": base_name, "tensor": None}

        name = _unique_tensor_name(base_name, reserved_names)
        return {
            "name": name,
            "tensor": TensorIR(
                name=name,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                is_variable=False,
                quantization=None,
            ),
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
            add_op = model_ir.operators[add_idx]
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
            except Exception:
                continue

            mul_inputs = [str(value) for value in mul_op.inputs]
            concat_idx: Optional[int] = None
            concat_out_name: Optional[str] = None
            concat_input_index: Optional[int] = None
            for input_index, mul_input_name in enumerate(mul_inputs):
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
                try:
                    concat_axis = int(producer_op.options.get("axis", 1))
                except (TypeError, ValueError):
                    continue
                if concat_axis < 0:
                    concat_axis += 4
                if concat_axis != 1:
                    continue
                concat_idx = int(producer_idx)
                concat_out_name = mul_input_name
                concat_input_index = int(input_index)
                break
            if (
                concat_idx is None
                or concat_out_name is None
                or concat_input_index is None
                or concat_out_name in public_inputs
            ):
                continue

            concat_op = model_ir.operators[concat_idx]
            concat_inputs = [str(value) for value in concat_op.inputs]
            if len(concat_inputs) < 2:
                continue
            pre_indices: List[int] = []
            new_concat_inputs: List[str] = []
            valid_pre_inputs = True
            for concat_input_name in concat_inputs:
                if (
                    concat_input_name in public_boundaries
                    or concat_input_name in graph_index.duplicate_producers
                ):
                    valid_pre_inputs = False
                    break
                pre_idx = graph_index.producers.get(concat_input_name)
                if pre_idx is None or int(pre_idx) >= int(concat_idx):
                    valid_pre_inputs = False
                    break
                pre_op = model_ir.operators[int(pre_idx)]
                if (
                    str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != concat_input_name
                    or _read_transpose_perm(model_ir, pre_op)
                    != perm_nhwc_to_nchw
                    or not graph_index.consumer_indices(concat_input_name)
                    or set(graph_index.consumer_indices(concat_input_name))
                    != {int(concat_idx)}
                ):
                    valid_pre_inputs = False
                    break
                source_name = str(pre_op.inputs[0])
                if _rank4_metadata(model_ir.tensors.get(source_name)) is None:
                    valid_pre_inputs = False
                    break
                pre_indices.append(int(pre_idx))
                new_concat_inputs.append(source_name)
            if not valid_pre_inputs:
                continue

            concat_metadata = _rank4_metadata(
                model_ir.tensors.get(concat_out_name)
            )
            mul_out_metadata = _rank4_metadata(
                model_ir.tensors.get(mul_out_name)
            )
            if concat_metadata is None or mul_out_metadata is None:
                continue
            concat_shape, concat_signature = concat_metadata
            mul_out_shape, mul_out_signature = mul_out_metadata
            target_concat_shape = _permute_shape(
                concat_shape,
                perm_nchw_to_nhwc,
            )
            target_concat_signature = _permute_shape(
                concat_signature,
                perm_nchw_to_nhwc,
            )
            target_mul_shape = _permute_shape(
                mul_out_shape,
                perm_nchw_to_nhwc,
            )
            target_mul_signature = _permute_shape(
                mul_out_signature,
                perm_nchw_to_nhwc,
            )
            if (
                target_concat_shape is None
                or target_concat_signature is None
                or target_mul_shape is None
                or target_mul_signature is None
            ):
                continue

            add_side_shape = [
                int(value) for value in add_side_data.shape
            ]
            if int(add_side_data.size) != 1:
                if (
                    int(add_side_data.ndim) != 4
                    or _broadcast_static_shapes(
                        target_concat_shape,
                        add_side_shape,
                    ) is None
                    or not (
                        int(add_side_shape[0]) == 1
                        and int(add_side_shape[1]) == 1
                        and int(add_side_shape[2]) == 1
                        and int(add_side_shape[3]) > 0
                    )
                ):
                    continue

            concat_users = graph_index.consumer_indices(concat_out_name)
            legacy_concat_users = [
                int(user_index)
                for user_index in concat_users
                if int(user_index) != int(mul_idx)
            ]
            if any(
                int(user_index) <= int(concat_idx)
                for user_index in legacy_concat_users
            ):
                continue
            needs_adapter = (
                bool(legacy_concat_users)
                or concat_out_name in public_outputs
            )
            chain_indices = {
                int(concat_idx),
                int(mul_idx),
                int(post_idx),
                int(add_idx),
                *(int(value) for value in pre_indices),
            }
            candidate_reserved_names = set(reserved_tensor_names)
            mul_constant_plan = _plan_mul_constant(
                mul_op=mul_op,
                concat_input_name=concat_out_name,
                graph_index=graph_index,
                chain_indices=chain_indices,
                target_shape_nhwc=target_concat_shape,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if mul_constant_plan is None:
                continue

            concat_tensor = model_ir.tensors[concat_out_name]
            concat_quantization_ok, target_concat_quantization = (
                _planned_permuted_quantization(
                    concat_tensor.quantization,
                    perm_nchw_to_nhwc,
                )
            )
            mul_out_tensor = model_ir.tensors[mul_out_name]
            mul_quantization_ok, target_mul_quantization = (
                _planned_permuted_quantization(
                    mul_out_tensor.quantization,
                    perm_nchw_to_nhwc,
                )
            )
            if not concat_quantization_ok or not mul_quantization_ok:
                continue

            canonical_concat_name = concat_out_name
            canonical_concat_tensor: Optional[TensorIR] = None
            adapter_permutation_plan: Optional[Dict[str, Any]] = None
            if needs_adapter:
                canonical_concat_name = _unique_tensor_name(
                    f"{concat_out_name}_nhwc",
                    candidate_reserved_names,
                )
                canonical_concat_tensor = TensorIR(
                    name=canonical_concat_name,
                    dtype=str(concat_tensor.dtype),
                    shape=list(target_concat_shape),
                    shape_signature=list(target_concat_signature),
                    data=None,
                    is_variable=False,
                    quantization=target_concat_quantization,
                )
                adapter_permutation_plan = _plan_adapter_permutation(
                    public_inputs=public_inputs,
                    reserved_names=candidate_reserved_names,
                )

            target_concat_options = dict(concat_op.options)
            target_concat_options["axis"] = 3
            remove_indices = {
                int(post_idx),
                *(int(value) for value in pre_indices),
            }

            # Every topology, metadata, constant, quantization, name, adapter,
            # setter, removal, and insertion decision is complete.
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
                )
                updated_mul_inputs = [str(value) for value in mul_op.inputs]
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

            if needs_adapter:
                assert canonical_concat_tensor is not None
                model_ir.tensors[
                    canonical_concat_name
                ] = canonical_concat_tensor
                _set_operator_outputs(
                    model_ir=model_ir,
                    op=concat_op,
                    new_outputs=[canonical_concat_name],
                    graph_index=graph_index,
                )
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=mul_op,
                    input_index=int(concat_input_index),
                    new_input_name=canonical_concat_name,
                    graph_index=graph_index,
                )
            else:
                concat_tensor.shape = list(target_concat_shape)
                concat_tensor.shape_signature = list(
                    target_concat_signature
                )
                concat_tensor.quantization = target_concat_quantization

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=new_concat_inputs,
                graph_index=graph_index,
            )
            concat_op.options = target_concat_options
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

            if (
                adapter_permutation_plan is not None
                and adapter_permutation_plan["tensor"] is not None
            ):
                adapter_tensor = adapter_permutation_plan["tensor"]
                model_ir.tensors[str(adapter_tensor.name)] = adapter_tensor

            graph_index.remove_operators(remove_indices)
            if needs_adapter:
                assert adapter_permutation_plan is not None
                current_concat_idx = graph_index.operator_index(concat_op)
                assert current_concat_idx is not None
                graph_index.insert_operator(
                    int(current_concat_idx) + 1,
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[
                            canonical_concat_name,
                            str(adapter_permutation_plan["name"]),
                        ],
                        outputs=[concat_out_name],
                    ),
                )

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {stats_key: int(optimized)}

