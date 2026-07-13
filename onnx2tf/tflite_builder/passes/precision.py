from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
)

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_logical_layout,
    normalize_onnx_shape,
)


def _rewrite_constant_divisors_to_multiplicative_reciprocals(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    integer_output_dtypes = {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    }
    changed_count = 0
    existing_tensor_names = set(str(name) for name in model_ir.tensors.keys())
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base_name: str) -> str:
        candidate = str(base_name)
        suffix = 0
        while candidate in existing_tensor_names:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        existing_tensor_names.add(candidate)
        return candidate

    div_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices("DIV")
    ]
    for op in div_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue
        if bool(op.options.get("preserveDivisionForOnnxRequantization", False)):
            continue

        lhs_name = str(op.inputs[0])
        rhs_name = str(op.inputs[1])
        output_name = str(op.outputs[0])
        rhs_tensor = model_ir.tensors.get(rhs_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        lhs_tensor = model_ir.tensors.get(lhs_name, None)
        if (
            rhs_tensor is None
            or not isinstance(rhs_tensor.data, np.ndarray)
            or output_tensor is None
        ):
            continue

        output_dtype = str(output_tensor.dtype).upper()
        if output_dtype in integer_output_dtypes:
            continue
        direct_consumers = graph_index.consumers_of(output_name)
        if any(
            str(consumer.op_type) == "CAST"
            and str(consumer.options.get("outDataType", "")).upper() in integer_output_dtypes
            for consumer in direct_consumers
        ):
            continue

        calc_dtype = "FLOAT16" if output_dtype == "FLOAT16" else "FLOAT32"
        np_calc_dtype = np.float16 if calc_dtype == "FLOAT16" else np.float32
        reciprocal_name = _unique_tensor_name(f"{output_name}_div_reciprocal")
        model_ir.tensors[reciprocal_name] = TensorIR(
            name=reciprocal_name,
            dtype=calc_dtype,
            shape=[int(v) for v in list(np.asarray(rhs_tensor.data, dtype=np_calc_dtype).shape)],
            shape_signature=[int(v) for v in list(np.asarray(rhs_tensor.data, dtype=np_calc_dtype).shape)],
            data=np.asarray(
                np.reciprocal(np.asarray(rhs_tensor.data, dtype=np_calc_dtype)),
                dtype=np_calc_dtype,
            ),
            logical_layout=normalize_logical_layout(rhs_tensor.logical_layout),
        )

        mul_lhs_name = lhs_name
        replacement_ops: List[OperatorIR] = []
        lhs_dtype = str(lhs_tensor.dtype).upper() if lhs_tensor is not None else "FLOAT32"
        if lhs_dtype != calc_dtype:
            lhs_shape = (
                [int(v) for v in list(lhs_tensor.shape)]
                if lhs_tensor is not None
                else [int(v) for v in list(output_tensor.shape)]
            )
            lhs_shape_signature = (
                [int(v) for v in list(lhs_tensor.shape_signature)]
                if lhs_tensor is not None and lhs_tensor.shape_signature is not None
                else list(lhs_shape)
            )
            lhs_cast_name = _unique_tensor_name(f"{output_name}_div_lhs_cast")
            model_ir.tensors[lhs_cast_name] = TensorIR(
                name=lhs_cast_name,
                dtype=calc_dtype,
                shape=lhs_shape,
                shape_signature=lhs_shape_signature,
                quantization=copy.deepcopy(lhs_tensor.quantization) if lhs_tensor is not None else None,
                logical_layout=normalize_logical_layout(
                    lhs_tensor.logical_layout if lhs_tensor is not None else output_tensor.logical_layout
                ),
            )
            replacement_ops.append(
                OperatorIR(
                    op_type="CAST",
                    inputs=[lhs_name],
                    outputs=[lhs_cast_name],
                    options={
                        "inDataType": lhs_dtype,
                        "outDataType": calc_dtype,
                    },
                )
            )
            mul_lhs_name = lhs_cast_name

        mul_out_name = output_name
        if output_dtype != calc_dtype:
            mul_out_name = _unique_tensor_name(f"{output_name}_div_mul_out")
            model_ir.tensors[mul_out_name] = TensorIR(
                name=mul_out_name,
                dtype=calc_dtype,
                shape=[int(v) for v in list(output_tensor.shape)],
                shape_signature=(
                    [int(v) for v in list(output_tensor.shape_signature)]
                    if output_tensor.shape_signature is not None
                    else [int(v) for v in list(output_tensor.shape)]
                ),
                quantization=None,
                logical_layout=normalize_logical_layout(output_tensor.logical_layout),
            )

        replacement_ops.append(
            OperatorIR(
                op_type="MUL",
                inputs=[mul_lhs_name, reciprocal_name],
                outputs=[mul_out_name],
                options=dict(op.options),
            )
        )

        if mul_out_name != output_name:
            replacement_ops.append(
                OperatorIR(
                    op_type="CAST",
                    inputs=[mul_out_name],
                    outputs=[output_name],
                    options={
                        "inDataType": calc_dtype,
                        "outDataType": output_dtype,
                    },
                )
            )
        current_index = graph_index.operator_index(op)
        if current_index is None:
            continue
        graph_index.remove_operator(int(current_index))
        for offset, replacement_op in enumerate(replacement_ops):
            graph_index.insert_operator(
                int(current_index) + int(offset),
                replacement_op,
            )
        changed_count += 1

    if changed_count > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "rewritten_constant_div_to_mul": int(changed_count),
    }


def _restore_precision_sensitive_reciprocal_divisions(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    integer_output_dtypes = {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    }
    shape_only_passthrough_ops = {
        "RESHAPE",
        "TRANSPOSE",
        "SQUEEZE",
        "EXPAND_DIMS",
        "SLICE",
    }
    affine_ops = {
        "ADD",
        "SUB",
        "MUL",
        "DIV",
    }
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _is_constant_float_tensor(tensor_name: str) -> bool:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None or not isinstance(tensor.data, np.ndarray):
            return False
        return bool(np.issubdtype(np.asarray(tensor.data).dtype, np.floating))

    def _feeds_integer_cast_through_affine_chain(start_tensor_name: str) -> bool:
        pending: List[Tuple[str, int]] = [(str(start_tensor_name), 0)]
        visited: set[Tuple[str, int]] = set()
        max_depth = 8
        while pending:
            current_tensor_name, depth = pending.pop(0)
            state = (str(current_tensor_name), int(depth))
            if state in visited:
                continue
            visited.add(state)
            if int(depth) >= int(max_depth):
                continue
            for consumer_idx in graph_index.consumer_indices(
                str(current_tensor_name)
            ):
                consumer_op = model_ir.operators[int(consumer_idx)]
                consumer_type = str(consumer_op.op_type)
                if consumer_type == "CAST":
                    out_dtype = str(consumer_op.options.get("outDataType", "")).upper()
                    if out_dtype in integer_output_dtypes:
                        return True
                    if len(consumer_op.outputs) == 1:
                        pending.append((str(consumer_op.outputs[0]), int(depth) + 1))
                    continue
                if len(consumer_op.outputs) != 1:
                    continue
                if consumer_type in shape_only_passthrough_ops:
                    pending.append((str(consumer_op.outputs[0]), int(depth) + 1))
                    continue
                if (
                    consumer_type in affine_ops
                    and str(current_tensor_name) in {str(v) for v in list(consumer_op.inputs)}
                ):
                    other_inputs = [
                        str(v)
                        for v in list(consumer_op.inputs)
                        if str(v) != str(current_tensor_name)
                    ]
                    if all(_is_constant_float_tensor(name) for name in other_inputs):
                        pending.append((str(consumer_op.outputs[0]), int(depth) + 1))
        return False

    restored = 0
    mul_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices("MUL")
    ]
    for op in mul_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue
        output_name = str(op.outputs[0])
        if output_name in {str(v) for v in list(model_ir.outputs)}:
            continue
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None:
            continue
        output_dtype = str(output_tensor.dtype).upper()
        if output_dtype in integer_output_dtypes:
            continue
        reciprocal_input_index: Optional[int] = None
        for input_index, input_name in enumerate(op.inputs):
            input_name_str = str(input_name)
            if "div_reciprocal" not in input_name_str:
                continue
            reciprocal_tensor = model_ir.tensors.get(input_name_str, None)
            if reciprocal_tensor is None or not isinstance(reciprocal_tensor.data, np.ndarray):
                continue
            reciprocal_values = np.asarray(reciprocal_tensor.data)
            if (
                not np.issubdtype(reciprocal_values.dtype, np.floating)
                or not np.all(np.isfinite(reciprocal_values))
                or np.any(np.equal(reciprocal_values, 0.0))
            ):
                continue
            reciprocal_input_index = int(input_index)
            break
        if reciprocal_input_index is None:
            continue
        if not _feeds_integer_cast_through_affine_chain(output_name):
            continue
        reciprocal_name = str(op.inputs[int(reciprocal_input_index)])
        reciprocal_tensor = model_ir.tensors[reciprocal_name]
        reciprocal_values = np.asarray(reciprocal_tensor.data)
        divisor_values = np.asarray(
            np.reciprocal(reciprocal_values.astype(np.float32, copy=False)),
            dtype=np.float32,
        )
        divisor_name = reciprocal_name.replace("div_reciprocal", "div_rhs_cast", 1)
        suffix = 0
        while divisor_name in model_ir.tensors:
            suffix += 1
            divisor_name = f"{reciprocal_name}_div_rhs_cast_{suffix}"
        divisor_shape, divisor_signature = normalize_onnx_shape(list(divisor_values.shape))
        model_ir.tensors[divisor_name] = TensorIR(
            name=divisor_name,
            dtype="FLOAT32",
            shape=[int(v) for v in divisor_shape],
            shape_signature=[int(v) for v in divisor_signature],
            data=divisor_values,
            logical_layout=normalize_logical_layout(reciprocal_tensor.logical_layout),
        )
        data_input_index = 1 - int(reciprocal_input_index)
        graph_index.replace_operator_inputs(
            int(op_index),
            [str(op.inputs[int(data_input_index)]), divisor_name],
        )
        graph_index.replace_operator_type(int(op_index), "DIV")
        op.options = {"fusedActivationFunction": "NONE"}
        restored += 1

    if restored > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {
        "restored_precision_sensitive_reciprocal_divisions": int(restored),
    }
