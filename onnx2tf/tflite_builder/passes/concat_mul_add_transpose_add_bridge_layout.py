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


def _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Eliminate DenseNet-like NHWC<->NCHW bridge tails:

      leaf_i_nhwc --TRANSPOSE(0,3,1,2)--> leaf_i_nchw
      CONCAT(axis=1, leaf_i_nchw) -> cat_nchw
      MUL(cat_nchw, const_nchw) -> m_nchw
      ADD(m_nchw, const_nchw) -> a_nchw
      TRANSPOSE(a_nchw, 0,2,3,1) -> a_nhwc
      ADD(a_nhwc, const_nhwc) -> y_nhwc

    Rewrite:
      CONCAT(axis=3, leaf_i_nhwc) -> cat_nhwc
      MUL(cat_nhwc, const_nhwc) -> m_nhwc
      ADD(m_nhwc, const_nhwc) -> a_nhwc
      ADD(a_nhwc, const_nhwc) -> y_nhwc

    Notes:
    - If CONCAT output has legacy NCHW consumers, preserve them via one local
      adapter TRANSPOSE(cat_nhwc->cat_nchw).
    """
    stats_key = (
        "optimized_concat_mul_add_transpose_add_nhwc_bridge_chains"
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

    def _unique_tensor_name(
        base: str,
        reserved_names: set[str],
    ) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in reserved_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        reserved_names.add(candidate)
        return candidate

    def _plan_affine_constant(
        *,
        op: OperatorIR,
        const_input_index: int,
        const_name: str,
        target_shape_nhwc: List[int],
        graph_index: ModelIRGraphIndex,
        chain_indices: set[int],
        public_inputs: set[str],
        public_outputs: set[str],
        reserved_names: set[str],
    ) -> Optional[Dict[str, Any]]:
        if (
            int(const_input_index) < 0
            or int(const_input_index) >= len(op.inputs)
            or str(op.inputs[int(const_input_index)]) != str(const_name)
        ):
            return None
        const_tensor = model_ir.tensors.get(str(const_name))
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
                "const_name": str(const_name),
                "new_name": str(const_name),
            }

        as_is_shape = [int(value) for value in const_data.shape]
        if (
            _broadcast_static_shapes(
                target_shape_nhwc,
                as_is_shape,
            )
            is not None
        ):
            return {
                "mode": "none",
                "input_index": int(const_input_index),
                "const_name": str(const_name),
                "new_name": str(const_name),
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
        if rotated is None or constant_permutation is None:
            return None

        rotated_shape = [int(value) for value in rotated.shape]
        if (
            _broadcast_static_shapes(
                target_shape_nhwc,
                rotated_shape,
            )
            is None
        ):
            return None
        if (
            str(const_name) in public_inputs
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

        const_users = graph_index.consumer_indices(str(const_name))
        shared_outside_chain = any(
            int(user_index) not in chain_indices
            for user_index in const_users
        )
        if shared_outside_chain or str(const_name) in public_outputs:
            new_name = _unique_tensor_name(
                f"{const_name}_nhwc",
                reserved_names,
            )
            return {
                "mode": "clone",
                "input_index": int(const_input_index),
                "const_name": str(const_name),
                "new_name": new_name,
                "data": np.asarray(rotated),
                "shape": rotated_shape,
                "quantization": rotated_quantization,
                "dtype": str(const_tensor.dtype),
            }
        return {
            "mode": "update",
            "input_index": int(const_input_index),
            "const_name": str(const_name),
            "new_name": str(const_name),
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
        base_name = "__concat_affine_tail_nhwc_to_nchw_perm_rank4__"
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
                    ]
                    == perm_nhwc_to_nchw
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

    def _apply_constant_plan(
        *,
        plan: Dict[str, Any],
        op: OperatorIR,
        graph_index: ModelIRGraphIndex,
    ) -> None:
        mode = str(plan["mode"])
        if mode == "clone":
            constant_name = str(plan["new_name"])
            model_ir.tensors[constant_name] = TensorIR(
                name=constant_name,
                dtype=str(plan["dtype"]),
                shape=list(plan["shape"]),
                shape_signature=list(plan["shape"]),
                data=np.asarray(plan["data"]),
                is_variable=False,
                quantization=plan["quantization"],
            )
            updated_inputs = [str(value) for value in op.inputs]
            updated_inputs[int(plan["input_index"])] = constant_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=op,
                new_inputs=updated_inputs,
                graph_index=graph_index,
            )
        elif mode == "update":
            constant_tensor = model_ir.tensors[str(plan["const_name"])]
            constant_tensor.data = np.asarray(plan["data"])
            constant_tensor.shape = list(plan["shape"])
            constant_tensor.shape_signature = list(plan["shape"])
            constant_tensor.quantization = plan["quantization"]

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
            pre_add_out_name = str(post_op.inputs[0])
            post_out_name = str(post_op.outputs[0])
            if (
                pre_add_out_name in public_boundaries
                or post_out_name in public_boundaries
                or pre_add_out_name in graph_index.duplicate_producers
                or post_out_name in graph_index.duplicate_producers
            ):
                continue

            pre_add_idx = graph_index.producers.get(pre_add_out_name)
            if pre_add_idx is None or int(pre_add_idx) >= int(post_idx):
                continue
            pre_add_op = model_ir.operators[int(pre_add_idx)]
            if (
                str(pre_add_op.op_type) != "ADD"
                or len(pre_add_op.inputs) != 2
                or len(pre_add_op.outputs) != 1
                or str(pre_add_op.outputs[0]) != pre_add_out_name
                or graph_index.consumer_indices(pre_add_out_name)
                != [int(post_idx)]
            ):
                continue

            pre_add_inputs = [str(value) for value in pre_add_op.inputs]
            pre_add_const_input_index: Optional[int] = None
            pre_add_const_name: Optional[str] = None
            pre_mul_idx: Optional[int] = None
            pre_mul_op: Optional[OperatorIR] = None
            pre_mul_data_input_index: Optional[int] = None
            pre_mul_const_input_index: Optional[int] = None
            pre_mul_const_name: Optional[str] = None
            concat_idx: Optional[int] = None
            concat_op: Optional[OperatorIR] = None
            concat_out_name: Optional[str] = None

            for pre_add_input_index, pre_add_input_name in enumerate(
                pre_add_inputs
            ):
                if (
                    pre_add_input_name in public_boundaries
                    or pre_add_input_name in graph_index.duplicate_producers
                ):
                    continue
                candidate_mul_idx = graph_index.producers.get(
                    pre_add_input_name
                )
                if (
                    candidate_mul_idx is None
                    or int(candidate_mul_idx) >= int(pre_add_idx)
                ):
                    continue
                candidate_mul_op = model_ir.operators[
                    int(candidate_mul_idx)
                ]
                if (
                    str(candidate_mul_op.op_type) != "MUL"
                    or len(candidate_mul_op.inputs) != 2
                    or len(candidate_mul_op.outputs) != 1
                    or str(candidate_mul_op.outputs[0])
                    != pre_add_input_name
                    or graph_index.consumer_indices(pre_add_input_name)
                    != [int(pre_add_idx)]
                ):
                    continue

                side_index = 1 - int(pre_add_input_index)
                side_name = str(pre_add_inputs[side_index])
                side_tensor = model_ir.tensors.get(side_name)
                if side_tensor is None or side_tensor.data is None:
                    continue

                mul_inputs = [
                    str(value) for value in candidate_mul_op.inputs
                ]
                for mul_data_input_index, mul_data_input_name in enumerate(
                    mul_inputs
                ):
                    if (
                        mul_data_input_name in public_boundaries
                        or mul_data_input_name
                        in graph_index.duplicate_producers
                    ):
                        continue
                    candidate_concat_idx = graph_index.producers.get(
                        mul_data_input_name
                    )
                    if (
                        candidate_concat_idx is None
                        or int(candidate_concat_idx)
                        >= int(candidate_mul_idx)
                    ):
                        continue
                    candidate_concat_op = model_ir.operators[
                        int(candidate_concat_idx)
                    ]
                    if (
                        str(candidate_concat_op.op_type)
                        != "CONCATENATION"
                        or len(candidate_concat_op.outputs) != 1
                        or str(candidate_concat_op.outputs[0])
                        != mul_data_input_name
                        or not isinstance(
                            candidate_concat_op.options,
                            dict,
                        )
                    ):
                        continue
                    try:
                        concat_axis = int(
                            candidate_concat_op.options.get("axis", 1)
                        )
                    except (TypeError, ValueError):
                        continue
                    if concat_axis < 0:
                        concat_axis += 4
                    if concat_axis != 1:
                        continue

                    mul_const_input_index = (
                        1 - int(mul_data_input_index)
                    )
                    mul_const_name = str(
                        mul_inputs[mul_const_input_index]
                    )
                    mul_const_tensor = model_ir.tensors.get(
                        mul_const_name
                    )
                    if (
                        mul_const_tensor is None
                        or mul_const_tensor.data is None
                    ):
                        continue

                    pre_add_const_input_index = int(side_index)
                    pre_add_const_name = side_name
                    pre_mul_idx = int(candidate_mul_idx)
                    pre_mul_op = candidate_mul_op
                    pre_mul_data_input_index = int(
                        mul_data_input_index
                    )
                    pre_mul_const_input_index = int(
                        mul_const_input_index
                    )
                    pre_mul_const_name = mul_const_name
                    concat_idx = int(candidate_concat_idx)
                    concat_op = candidate_concat_op
                    concat_out_name = mul_data_input_name
                    break
                if concat_idx is not None:
                    break

            if (
                pre_add_const_input_index is None
                or pre_add_const_name is None
                or pre_mul_idx is None
                or pre_mul_op is None
                or pre_mul_data_input_index is None
                or pre_mul_const_input_index is None
                or pre_mul_const_name is None
                or concat_idx is None
                or concat_op is None
                or concat_out_name is None
            ):
                continue

            tail_users = graph_index.consumer_indices(post_out_name)
            if len(tail_users) != 1:
                continue
            tail_add_idx = int(tail_users[0])
            if tail_add_idx <= int(post_idx):
                continue
            tail_add_op = model_ir.operators[tail_add_idx]
            if (
                str(tail_add_op.op_type) != "ADD"
                or len(tail_add_op.inputs) != 2
                or len(tail_add_op.outputs) != 1
                or any(
                    str(output_name)
                    in graph_index.duplicate_producers
                    for output_name in tail_add_op.outputs
                )
            ):
                continue
            tail_add_inputs = [
                str(value) for value in tail_add_op.inputs
            ]
            if tail_add_inputs.count(post_out_name) != 1:
                continue
            tail_add_input_index = tail_add_inputs.index(post_out_name)
            tail_add_const_name = tail_add_inputs[
                1 - int(tail_add_input_index)
            ]
            tail_add_const_tensor = model_ir.tensors.get(
                tail_add_const_name
            )
            if (
                tail_add_const_tensor is None
                or tail_add_const_tensor.data is None
            ):
                continue
            try:
                tail_add_const_data = np.asarray(
                    tail_add_const_tensor.data
                )
            except Exception:
                continue

            concat_inputs = [str(value) for value in concat_op.inputs]
            if len(concat_inputs) < 2:
                continue
            pre_indices: List[int] = []
            new_concat_inputs: List[str] = []
            valid_pre_inputs = True
            for concat_input_name in concat_inputs:
                if (
                    concat_input_name in public_boundaries
                    or concat_input_name
                    in graph_index.duplicate_producers
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
                    or graph_index.consumer_indices(concat_input_name)
                    != [int(concat_idx)]
                ):
                    valid_pre_inputs = False
                    break
                source_name = str(pre_op.inputs[0])
                if _rank4_metadata(
                    model_ir.tensors.get(source_name)
                ) is None:
                    valid_pre_inputs = False
                    break
                pre_indices.append(int(pre_idx))
                new_concat_inputs.append(source_name)
            if not valid_pre_inputs:
                continue

            concat_metadata = _rank4_metadata(
                model_ir.tensors.get(concat_out_name)
            )
            mul_out_name = str(pre_mul_op.outputs[0])
            mul_out_metadata = _rank4_metadata(
                model_ir.tensors.get(mul_out_name)
            )
            pre_add_metadata = _rank4_metadata(
                model_ir.tensors.get(pre_add_out_name)
            )
            if (
                concat_metadata is None
                or mul_out_metadata is None
                or pre_add_metadata is None
            ):
                continue

            concat_shape, concat_signature = concat_metadata
            mul_out_shape, mul_out_signature = mul_out_metadata
            pre_add_shape, pre_add_signature = pre_add_metadata
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
            target_pre_add_shape = _permute_shape(
                pre_add_shape,
                perm_nchw_to_nhwc,
            )
            target_pre_add_signature = _permute_shape(
                pre_add_signature,
                perm_nchw_to_nhwc,
            )
            if any(
                value is None
                for value in (
                    target_concat_shape,
                    target_concat_signature,
                    target_mul_shape,
                    target_mul_signature,
                    target_pre_add_shape,
                    target_pre_add_signature,
                )
            ):
                continue
            assert target_concat_shape is not None
            assert target_concat_signature is not None
            assert target_mul_shape is not None
            assert target_mul_signature is not None
            assert target_pre_add_shape is not None
            assert target_pre_add_signature is not None

            tail_add_shape = [
                int(value) for value in tail_add_const_data.shape
            ]
            if int(tail_add_const_data.size) != 1:
                if (
                    int(tail_add_const_data.ndim) != 4
                    or _broadcast_static_shapes(
                        target_pre_add_shape,
                        tail_add_shape,
                    )
                    is None
                    or not (
                        int(tail_add_shape[0]) == 1
                        and int(tail_add_shape[1]) == 1
                        and int(tail_add_shape[2]) == 1
                        and int(tail_add_shape[3]) > 0
                    )
                ):
                    continue

            concat_users = graph_index.consumer_indices(concat_out_name)
            legacy_concat_users = [
                int(user_index)
                for user_index in concat_users
                if int(user_index) != int(pre_mul_idx)
            ]
            if any(
                int(user_index) <= int(concat_idx)
                for user_index in legacy_concat_users
            ):
                continue
            needs_adapter = bool(legacy_concat_users)
            chain_indices = {
                int(concat_idx),
                int(pre_mul_idx),
                int(pre_add_idx),
                int(post_idx),
                int(tail_add_idx),
                *(int(value) for value in pre_indices),
            }
            candidate_reserved_names = set(reserved_tensor_names)
            mul_constant_plan = _plan_affine_constant(
                op=pre_mul_op,
                const_input_index=int(pre_mul_const_input_index),
                const_name=pre_mul_const_name,
                target_shape_nhwc=target_concat_shape,
                graph_index=graph_index,
                chain_indices=chain_indices,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if mul_constant_plan is None:
                continue
            add_constant_plan = _plan_affine_constant(
                op=pre_add_op,
                const_input_index=int(pre_add_const_input_index),
                const_name=pre_add_const_name,
                target_shape_nhwc=target_mul_shape,
                graph_index=graph_index,
                chain_indices=chain_indices,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if add_constant_plan is None:
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
            pre_add_tensor = model_ir.tensors[pre_add_out_name]
            pre_add_quantization_ok, target_pre_add_quantization = (
                _planned_permuted_quantization(
                    pre_add_tensor.quantization,
                    perm_nchw_to_nhwc,
                )
            )
            if not (
                concat_quantization_ok
                and mul_quantization_ok
                and pre_add_quantization_ok
            ):
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

            # Topology, metadata, constants, quantization, names, adapters,
            # setters, removals, and insertion are fully planned.
            reserved_tensor_names.update(candidate_reserved_names)

            _apply_constant_plan(
                plan=mul_constant_plan,
                op=pre_mul_op,
                graph_index=graph_index,
            )
            _apply_constant_plan(
                plan=add_constant_plan,
                op=pre_add_op,
                graph_index=graph_index,
            )

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
                    op=pre_mul_op,
                    input_index=int(pre_mul_data_input_index),
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
            pre_add_tensor.shape = list(target_pre_add_shape)
            pre_add_tensor.shape_signature = list(
                target_pre_add_signature
            )
            pre_add_tensor.quantization = target_pre_add_quantization

            _replace_operator_input_at(
                model_ir=model_ir,
                op=tail_add_op,
                input_index=int(tail_add_input_index),
                new_input_name=pre_add_out_name,
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

