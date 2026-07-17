from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def sanitize_probable_nhwc_axis_sensitive_ops(model_ir: ModelIR) -> Dict[str, int]:
    """
    Repair late axis-sensitive metadata when tensors are clearly NHWC but still
    carry NCHW-era axis constants or derived shapes.
    """
    rewritten = 0
    inserted_terminal_transposes = 0
    public_layout_map = model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _is_probable_nhwc_shape(shape: Optional[List[int]]) -> bool:
        if shape is None or len(shape) != 4:
            return False
        values = [int(v) for v in list(shape)]
        return (
            all(int(v) > 0 for v in values)
            and int(values[1]) > 4
            and int(values[2]) > 4
            and 0 < int(values[3]) <= 8
        )

    def _set_axis_input(op: OperatorIR, op_idx: int, new_axis: int, consumers: Dict[str, List[int]]) -> None:
        axis_name = str(op.inputs[0])
        axis_tensor = model_ir.tensors.get(axis_name, None)
        if axis_tensor is None:
            return
        axis_users = [int(v) for v in consumers.get(axis_name, [])]
        if set(axis_users) == {int(op_idx)}:
            _write_const_ints_to_tensor(axis_tensor, [int(new_axis)])
            return
        clone_name = _unique_tensor_name(f"{axis_name}_nhwc")
        model_ir.tensors[clone_name] = TensorIR(
            name=clone_name,
            dtype=str(axis_tensor.dtype),
            shape=[1],
            shape_signature=[1],
            data=np.asarray([int(new_axis)], dtype=np.int32),
            is_variable=False,
            quantization=_clone_quantization(axis_tensor.quantization),
        )
        _set_operator_inputs(
            model_ir=model_ir,
            op=op,
            new_inputs=[clone_name] + [str(v) for v in list(op.inputs[1:])],
        )

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for op_idx, op in enumerate(model_ir.operators):
            op_type = str(op.op_type)

            if op_type == "SPLIT" and len(op.inputs) >= 2:
                input_name = str(op.inputs[1])
                input_tensor = model_ir.tensors.get(input_name, None)
                axis_vals = _read_const_ints_from_tensor(model_ir.tensors.get(str(op.inputs[0]), None))
                if (
                    input_tensor is not None
                    and _is_probable_nhwc_shape(input_tensor.shape)
                    and axis_vals is not None
                    and len(axis_vals) == 1
                    and int(axis_vals[0]) == 1
                ):
                    input_shape = [int(v) for v in list(input_tensor.shape)]
                    num_splits = int(op.options.get("numSplits", len(list(op.outputs))))
                    if num_splits > 0 and int(input_shape[3]) % int(num_splits) == 0:
                        _set_axis_input(op, int(op_idx), 3, consumers)
                        c_each = int(input_shape[3]) // int(num_splits)
                        for output_name in list(op.outputs):
                            output_tensor = model_ir.tensors.get(str(output_name), None)
                            if output_tensor is not None:
                                output_tensor.shape = [input_shape[0], input_shape[1], input_shape[2], c_each]
                                output_tensor.shape_signature = [input_shape[0], input_shape[1], input_shape[2], c_each]
                        rewritten += 1
                        changed = True
                continue

            if op_type == "CONCATENATION" and len(op.inputs) > 0:
                axis = int(op.options.get("axis", -1))
                input_tensors = [model_ir.tensors.get(str(name), None) for name in list(op.inputs)]
                output_name = str(op.outputs[0]) if len(op.outputs) == 1 else ""
                has_explicit_nchw_input = False
                for input_name in op.inputs:
                    producer_idx = producers.get(str(input_name), None)
                    if producer_idx is None:
                        continue
                    producer_op = model_ir.operators[int(producer_idx)]
                    if (
                        str(producer_op.op_type) == "TRANSPOSE"
                        and _read_transpose_perm(model_ir, producer_op) == [0, 3, 1, 2]
                    ):
                        has_explicit_nchw_input = True
                        break
                if (
                    axis == 1
                    and str(public_layout_map.get(output_name, "")).upper() != "NCHW"
                    and not has_explicit_nchw_input
                    and all(t is not None and _is_probable_nhwc_shape(t.shape) for t in input_tensors)
                ):
                    base = [int(v) for v in list(input_tensors[0].shape)]
                    if all(
                        [int(v) for v in list(t.shape)][:3] == base[:3]
                        for t in input_tensors[1:]
                    ):
                        total_c = sum(int(list(t.shape)[3]) for t in input_tensors if t is not None)
                        op.options["axis"] = 3
                        out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
                        if out_tensor is not None:
                            out_tensor.shape = [base[0], base[1], base[2], total_c]
                            out_tensor.shape_signature = [base[0], base[1], base[2], total_c]
                        rewritten += 1
                        changed = True
                continue

            if op_type == "SLICE" and len(op.inputs) >= 3:
                input_name = str(op.inputs[0])
                input_tensor = model_ir.tensors.get(input_name, None)
                begin_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                size_tensor = model_ir.tensors.get(str(op.inputs[2]), None)
                begin_vals = _read_const_ints_from_tensor(begin_tensor)
                size_vals = _read_const_ints_from_tensor(size_tensor)
                if (
                    input_tensor is not None
                    and _is_probable_nhwc_shape(input_tensor.shape)
                    and begin_vals is not None
                    and size_vals is not None
                    and len(begin_vals) == 4
                    and len(size_vals) == 4
                    and int(size_vals[1]) == 1
                    and int(begin_vals[2]) == 0
                    and int(begin_vals[3]) == 0
                ):
                    new_begin = [int(begin_vals[0]), int(begin_vals[2]), int(begin_vals[3]), int(begin_vals[1])]
                    new_size = [int(size_vals[0]), int(size_vals[2]), int(size_vals[3]), int(size_vals[1])]
                    _write_const_ints_to_tensor(begin_tensor, new_begin)
                    _write_const_ints_to_tensor(size_tensor, new_size)
                    input_shape = [int(v) for v in list(input_tensor.shape)]
                    out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
                    if out_tensor is not None:
                        out_tensor.shape = [input_shape[0], input_shape[1], input_shape[2], int(new_size[3])]
                        out_tensor.shape_signature = [input_shape[0], input_shape[1], input_shape[2], int(new_size[3])]
                    rewritten += 1
                    changed = True
                continue

            if op_type in {"RELU", "RELU6", "RELU_0_TO_1", "LOGISTIC", "HARD_SWISH", "LEAKY_RELU", "TANH", "NEG", "EXP"} and len(op.inputs) == 1:
                input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                out_tensor = model_ir.tensors.get(str(op.outputs[0]), None) if len(op.outputs) == 1 else None
                if input_tensor is not None and out_tensor is not None and _is_probable_nhwc_shape(input_tensor.shape):
                    out_tensor.shape = [int(v) for v in list(input_tensor.shape)]
                    out_tensor.shape_signature = [int(v) for v in list(input_tensor.shape_signature)] if input_tensor.shape_signature is not None else [int(v) for v in list(input_tensor.shape)]
                continue

            if op_type in {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM"} and len(op.inputs) == 2 and len(op.outputs) == 1:
                a_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                b_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
                if out_tensor is None:
                    continue
                shapes = []
                for tensor in [a_tensor, b_tensor]:
                    if tensor is not None and tensor.data is None and tensor.shape is not None:
                        shapes.append([int(v) for v in list(tensor.shape)])
                if len(shapes) == 0:
                    continue
                if all(_is_probable_nhwc_shape(shape) or shape in ([1], []) for shape in shapes):
                    inferred = shapes[0]
                    for shape in shapes[1:]:
                        candidate = _broadcast_static_shapes(inferred, shape)
                        if candidate is not None:
                            inferred = [int(v) for v in list(candidate)]
                    if len(inferred) == 4:
                        out_tensor.shape = list(inferred)
                        out_tensor.shape_signature = list(inferred)
                continue

        if not changed:
            break

    if rewritten > 0:
        producers = _build_tensor_producer_map(model_ir)
        for output_name in list(model_ir.outputs):
            output_name = str(output_name)
            output_tensor = model_ir.tensors.get(output_name, None)
            if output_tensor is None or not _is_probable_nhwc_shape(output_tensor.shape):
                continue
            prod_idx = producers.get(output_name, None)
            if prod_idx is None:
                continue
            output_nhwc_name = _unique_tensor_name(f"{output_name}_nhwc")
            nhwc_shape = [int(v) for v in list(output_tensor.shape)]
            nhwc_sig = [int(v) for v in list(output_tensor.shape_signature)] if output_tensor.shape_signature is not None else list(nhwc_shape)
            nchw_shape = [nhwc_shape[0], nhwc_shape[3], nhwc_shape[1], nhwc_shape[2]]
            nchw_sig = [nhwc_sig[0], nhwc_sig[3], nhwc_sig[1], nhwc_sig[2]]
            model_ir.tensors[output_nhwc_name] = TensorIR(
                name=output_nhwc_name,
                dtype=str(output_tensor.dtype),
                shape=list(nhwc_shape),
                shape_signature=list(nhwc_sig),
                data=None,
                is_variable=False,
                quantization=_clone_quantization(output_tensor.quantization),
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=model_ir.operators[int(prod_idx)],
                new_outputs=[output_nhwc_name],
            )
            perm_name = _unique_tensor_name(f"{output_name}_perm")
            model_ir.tensors[perm_name] = TensorIR(
                name=perm_name,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray([0, 3, 1, 2], dtype=np.int32),
                is_variable=False,
            )
            output_tensor.shape = list(nchw_shape)
            output_tensor.shape_signature = list(nchw_sig)
            model_ir.operators.insert(
                int(prod_idx) + 1,
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[output_nhwc_name, perm_name],
                    outputs=[output_name],
                    options={},
                ),
            )
            inserted_terminal_transposes += 1

    return {
        "sanitized_probable_nhwc_axis_sensitive_ops": int(rewritten),
        "inserted_probable_nhwc_terminal_transposes": int(inserted_terminal_transposes),
    }
