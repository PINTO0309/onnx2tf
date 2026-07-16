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
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Eliminate DenseNet-like terminal NCHW bridge tails:

      leaf_i_nhwc --TRANSPOSE(0,3,1,2)--> leaf_i_nchw
      CONCAT(axis=1, leaf_i_nchw) -> c_nchw
      MUL(c_nchw, const) -> m_nchw
      ADD(m_nchw, const) -> a0_nchw
      ADD(a0_nchw, const) -> a1_nchw
      MEAN(a1_nchw, axes_nchw, keepDims=True) -> mean_nchw
      RESHAPE(mean_nchw, shape4) -> out_nhwc

    Rewrite:
      CONCAT(axis=3, leaf_i_nhwc) -> c_nhwc
      MUL(c_nhwc, const_nhwc) -> m_nhwc
      ADD(m_nhwc, const_nhwc) -> a0_nhwc
      ADD(a0_nhwc, const_nhwc) -> a1_nhwc
      MEAN(a1_nhwc, axes_nhwc, keepDims=True) -> mean_nhwc
      RESHAPE(mean_nhwc, shape4_nhwc) -> out_nhwc
    """
    stats_key = (
        "optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains"
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

    def _plan_int32_input_rewrite(
        *,
        op: OperatorIR,
        input_index: int,
        input_name: str,
        new_values: List[int],
        graph_index: ModelIRGraphIndex,
        chain_indices: set[int],
        public_inputs: set[str],
        public_outputs: set[str],
        reserved_names: set[str],
        clone_suffix: str,
    ) -> Optional[Dict[str, Any]]:
        if (
            int(input_index) < 0
            or int(input_index) >= len(op.inputs)
            or str(op.inputs[int(input_index)]) != str(input_name)
        ):
            return None
        tensor = model_ir.tensors.get(str(input_name))
        if tensor is None or tensor.data is None:
            return None
        try:
            array = np.asarray(tensor.data)
            shape = [int(value) for value in tensor.shape]
            signature = (
                [int(value) for value in tensor.shape_signature]
                if tensor.shape_signature is not None
                else list(shape)
            )
            current_values = [
                int(value) for value in array.reshape(-1).tolist()
            ]
        except (TypeError, ValueError):
            return None
        normalized_values = [int(value) for value in new_values]
        if (
            str(tensor.dtype) != "INT32"
            or array.dtype != np.dtype(np.int32)
            or tensor.quantization is not None
            or shape != [len(current_values)]
            or signature != [len(current_values)]
            or len(current_values) != len(normalized_values)
        ):
            return None
        if current_values == normalized_values:
            return {
                "mode": "none",
                "input_index": int(input_index),
                "const_name": str(input_name),
                "new_name": str(input_name),
            }
        if (
            str(input_name) in public_inputs
            or bool(tensor.is_variable)
        ):
            return None

        users = graph_index.consumer_indices(str(input_name))
        shared_outside_chain = any(
            int(user_index) not in chain_indices
            for user_index in users
        )
        if shared_outside_chain or str(input_name) in public_outputs:
            new_name = _unique_tensor_name(
                f"{input_name}_{clone_suffix}",
                reserved_names,
            )
            return {
                "mode": "clone",
                "input_index": int(input_index),
                "const_name": str(input_name),
                "new_name": new_name,
                "data": np.asarray(normalized_values, dtype=np.int32),
                "shape": [len(normalized_values)],
                "dtype": "INT32",
                "quantization": None,
            }
        return {
            "mode": "update",
            "input_index": int(input_index),
            "const_name": str(input_name),
            "new_name": str(input_name),
            "data": np.asarray(normalized_values, dtype=np.int32),
            "shape": [len(normalized_values)],
            "dtype": "INT32",
            "quantization": None,
        }

    def _plan_mean_axes(
        *,
        mean_op: OperatorIR,
        graph_index: ModelIRGraphIndex,
        chain_indices: set[int],
        public_inputs: set[str],
        public_outputs: set[str],
        reserved_names: set[str],
    ) -> Optional[Dict[str, Any]]:
        if len(mean_op.inputs) < 2:
            return None
        axes_name = str(mean_op.inputs[1])
        axes_tensor = model_ir.tensors.get(axes_name)
        axes_values = _read_const_ints_from_tensor(axes_tensor)
        if axes_tensor is None or axes_values is None:
            return None
        remapped_axes: List[int] = []
        for raw_axis in axes_values:
            axis = int(raw_axis)
            if axis < 0:
                axis += 4
            if axis < 0 or axis >= 4:
                return None
            remapped_axes.append(int(perm_nhwc_to_nchw[axis]))
        return _plan_int32_input_rewrite(
            op=mean_op,
            input_index=1,
            input_name=axes_name,
            new_values=remapped_axes,
            graph_index=graph_index,
            chain_indices=chain_indices,
            public_inputs=public_inputs,
            public_outputs=public_outputs,
            reserved_names=reserved_names,
            clone_suffix="nhwc",
        )

    def _plan_reshape_shape(
        *,
        reshape_op: OperatorIR,
        old_mean_shape: List[int],
        new_mean_shape: List[int],
        graph_index: ModelIRGraphIndex,
        chain_indices: set[int],
        public_inputs: set[str],
        public_outputs: set[str],
        reserved_names: set[str],
    ) -> Optional[Dict[str, Any]]:
        if len(reshape_op.inputs) < 2:
            return None
        shape_name = str(reshape_op.inputs[1])
        shape_tensor = model_ir.tensors.get(shape_name)
        shape_values = _read_const_ints_from_tensor(shape_tensor)
        if (
            shape_tensor is None
            or shape_values is None
            or len(shape_values) != 4
            or [int(value) for value in shape_values]
            != [int(value) for value in old_mean_shape]
        ):
            return {
                "mode": "none",
                "input_index": 1,
                "const_name": shape_name,
                "new_name": shape_name,
            }
        return _plan_int32_input_rewrite(
            op=reshape_op,
            input_index=1,
            input_name=shape_name,
            new_values=[int(value) for value in new_mean_shape],
            graph_index=graph_index,
            chain_indices=chain_indices,
            public_inputs=public_inputs,
            public_outputs=public_outputs,
            reserved_names=reserved_names,
            clone_suffix="nhwc",
        )

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
            tensor = model_ir.tensors[str(plan["const_name"])]
            tensor.data = np.asarray(plan["data"])
            tensor.shape = list(plan["shape"])
            tensor.shape_signature = list(plan["shape"])
            tensor.quantization = plan["quantization"]

    graph_index = ModelIRGraphIndex(model_ir)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_boundaries = public_inputs | public_outputs

    while True:
        changed = False
        for reshape_idx in graph_index.operator_indices("RESHAPE"):
            reshape_op = model_ir.operators[int(reshape_idx)]
            if (
                len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or any(
                    str(output_name) in graph_index.duplicate_producers
                    for output_name in reshape_op.outputs
                )
            ):
                continue
            mean_out_name = str(reshape_op.inputs[0])
            if (
                mean_out_name in public_boundaries
                or mean_out_name in graph_index.duplicate_producers
            ):
                continue

            mean_idx = graph_index.producers.get(mean_out_name)
            if mean_idx is None or int(mean_idx) >= int(reshape_idx):
                continue
            mean_op = model_ir.operators[int(mean_idx)]
            if (
                str(mean_op.op_type) != "MEAN"
                or len(mean_op.inputs) < 2
                or len(mean_op.outputs) != 1
                or str(mean_op.outputs[0]) != mean_out_name
                or not isinstance(mean_op.options, dict)
                or not bool(mean_op.options.get("keepDims", False))
                or graph_index.consumer_indices(mean_out_name)
                != [int(reshape_idx)]
            ):
                continue

            add1_out_name = str(mean_op.inputs[0])
            if (
                add1_out_name in public_boundaries
                or add1_out_name in graph_index.duplicate_producers
            ):
                continue
            add1_idx = graph_index.producers.get(add1_out_name)
            if add1_idx is None or int(add1_idx) >= int(mean_idx):
                continue
            add1_op = model_ir.operators[int(add1_idx)]
            if (
                str(add1_op.op_type) != "ADD"
                or len(add1_op.inputs) != 2
                or len(add1_op.outputs) != 1
                or str(add1_op.outputs[0]) != add1_out_name
                or graph_index.consumer_indices(add1_out_name)
                != [int(mean_idx)]
            ):
                continue

            add1_inputs = [str(value) for value in add1_op.inputs]
            add0_idx: Optional[int] = None
            add1_const_index: Optional[int] = None
            add1_const_name: Optional[str] = None
            add0_out_name: Optional[str] = None
            for input_index, candidate_name in enumerate(add1_inputs):
                if (
                    candidate_name in public_boundaries
                    or candidate_name in graph_index.duplicate_producers
                ):
                    continue
                candidate_idx = graph_index.producers.get(candidate_name)
                if candidate_idx is None or int(candidate_idx) >= int(add1_idx):
                    continue
                candidate_op = model_ir.operators[int(candidate_idx)]
                if (
                    str(candidate_op.op_type) != "ADD"
                    or len(candidate_op.inputs) != 2
                    or len(candidate_op.outputs) != 1
                    or str(candidate_op.outputs[0]) != candidate_name
                    or graph_index.consumer_indices(candidate_name)
                    != [int(add1_idx)]
                ):
                    continue
                side_index = 1 - int(input_index)
                side_name = str(add1_inputs[side_index])
                side_tensor = model_ir.tensors.get(side_name)
                if side_tensor is None or side_tensor.data is None:
                    continue
                add0_idx = int(candidate_idx)
                add1_const_index = int(side_index)
                add1_const_name = side_name
                add0_out_name = candidate_name
                break
            if (
                add0_idx is None
                or add1_const_index is None
                or add1_const_name is None
                or add0_out_name is None
            ):
                continue

            add0_op = model_ir.operators[int(add0_idx)]
            add0_inputs = [str(value) for value in add0_op.inputs]
            mul_idx: Optional[int] = None
            add0_const_index: Optional[int] = None
            add0_const_name: Optional[str] = None
            mul_out_name: Optional[str] = None
            for input_index, candidate_name in enumerate(add0_inputs):
                if (
                    candidate_name in public_boundaries
                    or candidate_name in graph_index.duplicate_producers
                ):
                    continue
                candidate_idx = graph_index.producers.get(candidate_name)
                if candidate_idx is None or int(candidate_idx) >= int(add0_idx):
                    continue
                candidate_op = model_ir.operators[int(candidate_idx)]
                if (
                    str(candidate_op.op_type) != "MUL"
                    or len(candidate_op.inputs) != 2
                    or len(candidate_op.outputs) != 1
                    or str(candidate_op.outputs[0]) != candidate_name
                    or graph_index.consumer_indices(candidate_name)
                    != [int(add0_idx)]
                ):
                    continue
                side_index = 1 - int(input_index)
                side_name = str(add0_inputs[side_index])
                side_tensor = model_ir.tensors.get(side_name)
                if side_tensor is None or side_tensor.data is None:
                    continue
                mul_idx = int(candidate_idx)
                add0_const_index = int(side_index)
                add0_const_name = side_name
                mul_out_name = candidate_name
                break
            if (
                mul_idx is None
                or add0_const_index is None
                or add0_const_name is None
                or mul_out_name is None
            ):
                continue

            mul_op = model_ir.operators[int(mul_idx)]
            mul_inputs = [str(value) for value in mul_op.inputs]
            concat_idx: Optional[int] = None
            concat_data_index: Optional[int] = None
            concat_out_name: Optional[str] = None
            mul_const_index: Optional[int] = None
            mul_const_name: Optional[str] = None
            for input_index, candidate_name in enumerate(mul_inputs):
                if (
                    candidate_name in public_boundaries
                    or candidate_name in graph_index.duplicate_producers
                ):
                    continue
                candidate_idx = graph_index.producers.get(candidate_name)
                if candidate_idx is None or int(candidate_idx) >= int(mul_idx):
                    continue
                candidate_op = model_ir.operators[int(candidate_idx)]
                if (
                    str(candidate_op.op_type) != "CONCATENATION"
                    or len(candidate_op.outputs) != 1
                    or str(candidate_op.outputs[0]) != candidate_name
                    or not isinstance(candidate_op.options, dict)
                    or graph_index.consumer_indices(candidate_name)
                    != [int(mul_idx)]
                ):
                    continue
                try:
                    concat_axis = int(
                        candidate_op.options.get("axis", 1)
                    )
                except (TypeError, ValueError):
                    continue
                if concat_axis < 0:
                    concat_axis += 4
                if concat_axis != 1:
                    continue
                side_index = 1 - int(input_index)
                side_name = str(mul_inputs[side_index])
                side_tensor = model_ir.tensors.get(side_name)
                if side_tensor is None or side_tensor.data is None:
                    continue
                concat_idx = int(candidate_idx)
                concat_data_index = int(input_index)
                concat_out_name = candidate_name
                mul_const_index = int(side_index)
                mul_const_name = side_name
                break
            if (
                concat_idx is None
                or concat_data_index is None
                or concat_out_name is None
                or mul_const_index is None
                or mul_const_name is None
            ):
                continue

            concat_op = model_ir.operators[int(concat_idx)]
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

            required_names = (
                concat_out_name,
                mul_out_name,
                add0_out_name,
                add1_out_name,
                mean_out_name,
            )
            metadata = [
                _rank4_metadata(model_ir.tensors.get(name))
                for name in required_names
            ]
            if any(value is None for value in metadata):
                continue
            assert all(value is not None for value in metadata)
            typed_metadata = [
                value for value in metadata if value is not None
            ]
            target_shapes: List[List[int]] = []
            target_signatures: List[List[int]] = []
            valid_targets = True
            for shape, signature in typed_metadata:
                target_shape = _permute_shape(
                    shape,
                    perm_nchw_to_nhwc,
                )
                target_signature = _permute_shape(
                    signature,
                    perm_nchw_to_nhwc,
                )
                if target_shape is None or target_signature is None:
                    valid_targets = False
                    break
                target_shapes.append(list(target_shape))
                target_signatures.append(list(target_signature))
            if not valid_targets or len(target_shapes) != 5:
                continue

            chain_indices = {
                int(concat_idx),
                int(mul_idx),
                int(add0_idx),
                int(add1_idx),
                int(mean_idx),
                int(reshape_idx),
                *(int(value) for value in pre_indices),
            }
            candidate_reserved_names = set(reserved_tensor_names)
            mul_constant_plan = _plan_affine_constant(
                op=mul_op,
                const_input_index=int(mul_const_index),
                const_name=mul_const_name,
                target_shape_nhwc=target_shapes[0],
                graph_index=graph_index,
                chain_indices=chain_indices,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if mul_constant_plan is None:
                continue
            add0_constant_plan = _plan_affine_constant(
                op=add0_op,
                const_input_index=int(add0_const_index),
                const_name=add0_const_name,
                target_shape_nhwc=target_shapes[1],
                graph_index=graph_index,
                chain_indices=chain_indices,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if add0_constant_plan is None:
                continue
            add1_constant_plan = _plan_affine_constant(
                op=add1_op,
                const_input_index=int(add1_const_index),
                const_name=add1_const_name,
                target_shape_nhwc=target_shapes[2],
                graph_index=graph_index,
                chain_indices=chain_indices,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if add1_constant_plan is None:
                continue

            axes_plan = _plan_mean_axes(
                mean_op=mean_op,
                graph_index=graph_index,
                chain_indices=chain_indices,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if axes_plan is None:
                continue
            reshape_shape_plan = _plan_reshape_shape(
                reshape_op=reshape_op,
                old_mean_shape=typed_metadata[4][0],
                new_mean_shape=target_shapes[4],
                graph_index=graph_index,
                chain_indices=chain_indices,
                public_inputs=public_inputs,
                public_outputs=public_outputs,
                reserved_names=candidate_reserved_names,
            )
            if reshape_shape_plan is None:
                continue

            target_quantizations: List[Any] = []
            quantization_ok = True
            for name in required_names:
                tensor = model_ir.tensors[name]
                ok, target_quantization = (
                    _planned_permuted_quantization(
                        tensor.quantization,
                        perm_nchw_to_nhwc,
                    )
                )
                if not ok:
                    quantization_ok = False
                    break
                target_quantizations.append(target_quantization)
            if not quantization_ok:
                continue

            target_concat_options = dict(concat_op.options)
            target_concat_options["axis"] = 3
            remove_indices = {
                int(value) for value in pre_indices
            }

            # Topology, metadata, constants, quantization, axes, shape,
            # names, setters, and removals are fully planned.
            reserved_tensor_names.update(candidate_reserved_names)

            _apply_constant_plan(
                plan=mul_constant_plan,
                op=mul_op,
                graph_index=graph_index,
            )
            _apply_constant_plan(
                plan=add0_constant_plan,
                op=add0_op,
                graph_index=graph_index,
            )
            _apply_constant_plan(
                plan=add1_constant_plan,
                op=add1_op,
                graph_index=graph_index,
            )
            _apply_constant_plan(
                plan=axes_plan,
                op=mean_op,
                graph_index=graph_index,
            )
            _apply_constant_plan(
                plan=reshape_shape_plan,
                op=reshape_op,
                graph_index=graph_index,
            )

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=new_concat_inputs,
                graph_index=graph_index,
            )
            concat_op.options = target_concat_options

            for (
                name,
                target_shape,
                target_signature,
                target_quantization,
            ) in zip(
                required_names,
                target_shapes,
                target_signatures,
                target_quantizations,
            ):
                tensor = model_ir.tensors[name]
                tensor.shape = list(target_shape)
                tensor.shape_signature = list(target_signature)
                tensor.quantization = target_quantization

            graph_index.remove_operators(remove_indices)

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {stats_key: int(optimized)}

