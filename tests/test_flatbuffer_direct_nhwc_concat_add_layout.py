from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_concat_nhwc_chains,
)


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
    )


def _int_tensor(name: str, values: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT32",
        shape=[len(values)],
        shape_signature=[len(values)],
        data=np.asarray(values, dtype=np.int32),
    )


def _add_model(
    *,
    all_add: bool = False,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("nhwc_add_pre_concat")
    model_ir.inputs = ["x_nhwc", "a_nhwc", "b_nhwc"]
    model_ir.outputs = ["y"]
    a_source_shape = (
        [1, 5, 3]
        if boundary == "invalid_source_rank"
        else [1, 5, 7, 3]
    )
    a_adapter_shape = (
        [1, 3, 5]
        if boundary == "invalid_adapter_rank"
        else [1, 3, 5, 7]
    )
    add_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_add_output_rank":
        add_shape = [1, 3, 5]
    output_channels = 3 if all_add else 5
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 5, 7, 2]),
        "a_nhwc": _tensor("a_nhwc", a_source_shape),
        "b_nhwc": _tensor("b_nhwc", [1, 5, 7, 3]),
        "pre_perm": _int_tensor("pre_perm", [0, 3, 1, 2]),
        "post_perm": _int_tensor(
            "post_perm",
            [0, 3, 1, 2]
            if boundary == "wrong_post_permutation"
            else [0, 2, 3, 1],
        ),
        "x_nchw": _tensor("x_nchw", [1, 2, 5, 7]),
        "a_nchw": _tensor("a_nchw", a_adapter_shape),
        "b_nchw": _tensor("b_nchw", [1, 3, 5, 7]),
        "sum_nchw": _tensor("sum_nchw", add_shape),
        "concat_nchw": _tensor(
            "concat_nchw",
            [1, output_channels, 5, 7],
        ),
        "concat_nhwc": _tensor(
            "concat_nhwc",
            [1, 5, 7, output_channels],
        ),
        "y": _tensor("y", [1, 5, 7, output_channels]),
    }
    if boundary == "wrong_add_pre_permutation":
        model_ir.tensors["bad_pre_perm"] = _int_tensor(
            "bad_pre_perm",
            [0, 2, 3, 1],
        )
    model_ir.tensors["sum_nchw"].quantization = QuantParamIR(
        scale=[0.25] * 3,
        zero_point=[0] * 3,
        quantized_dimension=1,
    )
    add_inputs = ["a_nchw", "b_nchw"]
    concat_inputs = ["sum_nchw"] if all_add else ["x_nchw", "sum_nchw"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "pre_perm"], ["x_nchw"]),
        OperatorIR(
            "TRANSPOSE",
            [
                "a_nhwc",
                "bad_pre_perm" if boundary == "wrong_add_pre_permutation" else "pre_perm",
            ],
            ["a_nchw"],
        ),
        OperatorIR("TRANSPOSE", ["b_nhwc", "pre_perm"], ["b_nchw"]),
        OperatorIR(
            "SUB" if boundary == "unsupported_add" else "ADD",
            add_inputs,
            ["sum_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "CONCATENATION",
            concat_inputs,
            ["concat_nchw"],
            options={"axis": 1},
        ),
        OperatorIR(
            "TRANSPOSE",
            ["concat_nchw", "post_perm"],
            ["concat_nhwc"],
        ),
        OperatorIR("RELU", ["concat_nhwc"], ["y"]),
    ]
    if all_add:
        model_ir.inputs = ["a_nhwc", "b_nhwc"]
        model_ir.operators = [
            op for op in model_ir.operators if op.outputs != ["x_nchw"]
        ]
        del model_ir.tensors["x_nhwc"]
        del model_ir.tensors["x_nchw"]

    if boundary == "add_output_fanout":
        model_ir.tensors["sum_side"] = _tensor(
            "sum_side",
            list(add_shape),
        )
        model_ir.outputs.append("sum_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["sum_nchw"], ["sum_side"])
        )
    if boundary == "add_output_post_transpose":
        model_ir.tensors["sum_nhwc"] = _tensor(
            "sum_nhwc",
            [1, 5, 7, 3],
        )
        model_ir.tensors["sum_side"] = _tensor(
            "sum_side",
            [1, 5, 7, 3],
        )
        model_ir.outputs.append("sum_side")
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["sum_nchw", "post_perm"],
                    ["sum_nhwc"],
                ),
                OperatorIR("IDENTITY", ["sum_nhwc"], ["sum_side"]),
            ]
        )
    if boundary == "public_add_post":
        model_ir.tensors["sum_nhwc"] = _tensor(
            "sum_nhwc",
            [1, 5, 7, 3],
        )
        model_ir.outputs.append("sum_nhwc")
        model_ir.operators.append(
            OperatorIR(
                "TRANSPOSE",
                ["sum_nchw", "post_perm"],
                ["sum_nhwc"],
            )
        )
    if boundary == "public_add_output":
        model_ir.outputs.append("sum_nchw")
    if boundary in {"add_adapter_fanout", "add_second_adapter_fanout"}:
        adapter_name = (
            "b_nchw"
            if boundary == "add_second_adapter_fanout"
            else "a_nchw"
        )
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            list(model_ir.tensors[adapter_name].shape),
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", [adapter_name], ["adapter_side"])
        )
    if boundary == "public_add_adapter":
        model_ir.outputs.append("a_nchw")
    if boundary == "adapter_shared_with_concat":
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["x_nchw", "a_nchw", "sum_nchw"]
        model_ir.tensors["concat_nchw"].shape = [1, 8, 5, 7]
        model_ir.tensors["concat_nchw"].shape_signature = [1, 8, 5, 7]
        model_ir.tensors["concat_nhwc"].shape = [1, 5, 7, 8]
        model_ir.tensors["concat_nhwc"].shape_signature = [1, 5, 7, 8]
        model_ir.tensors["y"].shape = [1, 5, 7, 8]
        model_ir.tensors["y"].shape_signature = [1, 5, 7, 8]
    if boundary == "unary_operand":
        model_ir.tensors["a_relu"] = _tensor(
            "a_relu",
            [1, 3, 5, 7],
        )
        add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
        add_index = model_ir.operators.index(add_op)
        model_ir.operators.insert(
            add_index,
            OperatorIR("RELU", ["a_nchw"], ["a_relu"]),
        )
        add_op.inputs[0] = "a_relu"
    if boundary == "swish_operand":
        model_ir.tensors["a_logistic"] = _tensor(
            "a_logistic",
            [1, 3, 5, 7],
        )
        model_ir.tensors["a_swish"] = _tensor(
            "a_swish",
            [1, 3, 5, 7],
        )
        add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
        add_index = model_ir.operators.index(add_op)
        model_ir.operators[add_index:add_index] = [
            OperatorIR("LOGISTIC", ["a_nchw"], ["a_logistic"]),
            OperatorIR(
                "MUL",
                ["a_logistic", "a_nchw"],
                ["a_swish"],
            ),
        ]
        add_op.inputs[0] = "a_swish"
    if boundary == "split_operand":
        model_ir.tensors["a_nhwc"].shape = [1, 5, 7, 6]
        model_ir.tensors["a_nhwc"].shape_signature = [1, 5, 7, 6]
        model_ir.tensors["a_nchw"].shape = [1, 6, 5, 7]
        model_ir.tensors["a_nchw"].shape_signature = [1, 6, 5, 7]
        model_ir.tensors["split_axis"] = _int_tensor("split_axis", [1])
        model_ir.tensors["a_split0"] = _tensor(
            "a_split0",
            [1, 3, 5, 7],
        )
        model_ir.tensors["a_split1"] = _tensor(
            "a_split1",
            [1, 3, 5, 7],
        )
        add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
        add_index = model_ir.operators.index(add_op)
        model_ir.operators.insert(
            add_index,
            OperatorIR(
                "SPLIT",
                ["split_axis", "a_nchw"],
                ["a_split0", "a_split1"],
            ),
        )
        add_op.inputs[0] = "a_split0"
    if boundary in {
        "pad_operand",
        "pad_operand_fanout",
        "pad_companion",
    }:
        pads_data = np.asarray(
            [[0, 0], [0, 0], [0, 1], [0, 0]],
            dtype=np.int32,
        )
        model_ir.tensors["pads_nchw"] = TensorIR(
            name="pads_nchw",
            dtype="INT32",
            shape=[4, 2],
            shape_signature=[4, 2],
            data=pads_data,
        )
        model_ir.tensors["pad_value"] = TensorIR(
            name="pad_value",
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
            data=np.asarray([0.0], dtype=np.float32),
        )
        if boundary == "pad_companion":
            model_ir.tensors["x_nhwc"].shape = [1, 4, 7, 2]
            model_ir.tensors["x_nhwc"].shape_signature = [1, 4, 7, 2]
            model_ir.tensors["x_nchw"].shape = [1, 2, 4, 7]
            model_ir.tensors["x_nchw"].shape_signature = [1, 2, 4, 7]
            model_ir.tensors["x_pad"] = _tensor(
                "x_pad",
                [1, 2, 5, 7],
            )
            concat_op = next(
                op
                for op in model_ir.operators
                if op.op_type == "CONCATENATION"
            )
            concat_index = model_ir.operators.index(concat_op)
            model_ir.operators.insert(
                concat_index,
                OperatorIR(
                    "PAD",
                    ["x_nchw", "pads_nchw", "pad_value"],
                    ["x_pad"],
                ),
            )
            concat_op.inputs[0] = "x_pad"
        else:
            model_ir.tensors["a_nhwc"].shape = [1, 4, 7, 3]
            model_ir.tensors["a_nhwc"].shape_signature = [1, 4, 7, 3]
            model_ir.tensors["a_nchw"].shape = [1, 3, 4, 7]
            model_ir.tensors["a_nchw"].shape_signature = [1, 3, 4, 7]
            model_ir.tensors["a_pad"] = _tensor(
                "a_pad",
                [1, 3, 5, 7],
            )
            model_ir.tensors["a_pad"].quantization = QuantParamIR(
                scale=[0.7] * 3,
                zero_point=[0] * 3,
                quantized_dimension=1,
            )
            add_op = next(
                op for op in model_ir.operators if op.op_type == "ADD"
            )
            add_index = model_ir.operators.index(add_op)
            model_ir.operators.insert(
                add_index,
                OperatorIR(
                    "PAD",
                    ["a_nchw", "pads_nchw", "pad_value"],
                    ["a_pad"],
                ),
            )
            add_op.inputs[0] = "a_pad"
            if boundary == "pad_operand_fanout":
                model_ir.tensors["pad_side"] = _tensor(
                    "pad_side",
                    [1, 3, 5, 7],
                )
                model_ir.outputs.append("pad_side")
                model_ir.operators.append(
                    OperatorIR("IDENTITY", ["a_pad"], ["pad_side"])
                )
    if boundary in {
        "slice_operand",
        "slice_operand_fanout",
        "slice_companion",
    }:
        model_ir.tensors["slice_begin"] = _int_tensor(
            "slice_begin",
            [0, 0, 0, 0],
        )
        if boundary == "slice_companion":
            model_ir.tensors["x_nhwc"].shape = [1, 5, 7, 4]
            model_ir.tensors["x_nhwc"].shape_signature = [1, 5, 7, 4]
            model_ir.tensors["x_nchw"].shape = [1, 4, 5, 7]
            model_ir.tensors["x_nchw"].shape_signature = [1, 4, 5, 7]
            model_ir.tensors["slice_size"] = _int_tensor(
                "slice_size",
                [1, 2, 5, 7],
            )
            model_ir.tensors["x_slice"] = _tensor(
                "x_slice",
                [1, 2, 5, 7],
            )
            concat_op = next(
                op
                for op in model_ir.operators
                if op.op_type == "CONCATENATION"
            )
            concat_index = model_ir.operators.index(concat_op)
            model_ir.operators.insert(
                concat_index,
                OperatorIR(
                    "SLICE",
                    ["x_nchw", "slice_begin", "slice_size"],
                    ["x_slice"],
                ),
            )
            concat_op.inputs[0] = "x_slice"
        else:
            model_ir.tensors["a_nhwc"].shape = [1, 5, 7, 6]
            model_ir.tensors["a_nhwc"].shape_signature = [1, 5, 7, 6]
            model_ir.tensors["a_nchw"].shape = [1, 6, 5, 7]
            model_ir.tensors["a_nchw"].shape_signature = [1, 6, 5, 7]
            model_ir.tensors["slice_size"] = _int_tensor(
                "slice_size",
                [1, 3, 5, 7],
            )
            model_ir.tensors["a_slice"] = _tensor(
                "a_slice",
                [1, 3, 5, 7],
            )
            model_ir.tensors["a_slice"].quantization = QuantParamIR(
                scale=[0.8] * 3,
                zero_point=[0] * 3,
                quantized_dimension=1,
            )
            add_op = next(
                op for op in model_ir.operators if op.op_type == "ADD"
            )
            add_index = model_ir.operators.index(add_op)
            model_ir.operators.insert(
                add_index,
                OperatorIR(
                    "SLICE",
                    ["a_nchw", "slice_begin", "slice_size"],
                    ["a_slice"],
                ),
            )
            add_op.inputs[0] = "a_slice"
            if boundary == "slice_operand_fanout":
                model_ir.tensors["slice_side"] = _tensor(
                    "slice_side",
                    [1, 3, 5, 7],
                )
                model_ir.outputs.append("slice_side")
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["a_slice"],
                        ["slice_side"],
                    )
                )
    if boundary in {
        "dequantize_operand",
        "dequantize_operand_fanout",
        "dequantize_companion",
    }:
        companion = boundary == "dequantize_companion"
        source_name = "x_nhwc" if companion else "a_nhwc"
        adapter_name = "x_nchw" if companion else "a_nchw"
        output_name = "x_dequantized" if companion else "a_dequantized"
        channels = 2 if companion else 3
        model_ir.tensors[source_name].dtype = "INT8"
        model_ir.tensors[source_name].quantization = QuantParamIR(
            scale=[0.2] * channels,
            zero_point=[0] * channels,
            quantized_dimension=3,
        )
        model_ir.tensors[adapter_name].dtype = "INT8"
        model_ir.tensors[adapter_name].quantization = QuantParamIR(
            scale=[0.2] * channels,
            zero_point=[0] * channels,
            quantized_dimension=1,
        )
        model_ir.tensors[output_name] = _tensor(
            output_name,
            [1, channels, 5, 7],
        )
        model_ir.tensors[output_name].quantization = {
            "scale": [0.4] * channels,
            "zero_point": [0] * channels,
            "quantized_dimension": 1,
        }
        if companion:
            concat_op = next(
                op
                for op in model_ir.operators
                if op.op_type == "CONCATENATION"
            )
            concat_index = model_ir.operators.index(concat_op)
            model_ir.operators.insert(
                concat_index,
                OperatorIR("DEQUANTIZE", [adapter_name], [output_name]),
            )
            concat_op.inputs[0] = output_name
        else:
            add_op = next(
                op for op in model_ir.operators if op.op_type == "ADD"
            )
            add_index = model_ir.operators.index(add_op)
            model_ir.operators.insert(
                add_index,
                OperatorIR("DEQUANTIZE", [adapter_name], [output_name]),
            )
            add_op.inputs[0] = output_name
            if boundary == "dequantize_operand_fanout":
                model_ir.tensors["dequantize_side"] = _tensor(
                    "dequantize_side",
                    [1, channels, 5, 7],
                )
                model_ir.outputs.append("dequantize_side")
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        [output_name],
                        ["dequantize_side"],
                    )
                )
    if boundary in {
        "prelu_operand",
        "prelu_operand_fanout",
        "prelu_companion",
        "prelu_shared_alpha_operands",
    }:
        companion = boundary == "prelu_companion"
        shared_operands = boundary == "prelu_shared_alpha_operands"
        channels = 2 if companion else 3
        alpha_data = np.arange(
            1,
            channels + 1,
            dtype=np.float32,
        ).reshape(1, channels, 1, 1)
        model_ir.tensors["prelu_alpha"] = TensorIR(
            name="prelu_alpha",
            dtype="FLOAT32",
            shape=[1, channels, 1, 1],
            shape_signature=[1, channels, 1, 1],
            data=alpha_data,
            is_variable=True,
            quantization=QuantParamIR(
                scale=[0.1] * channels,
                zero_point=[0] * channels,
                quantized_dimension=1,
            ),
            logical_layout="NCHW",
            physical_layout="NCHW",
            onnx_tensor_name="onnx_prelu_alpha",
        )
        if companion:
            model_ir.tensors["x_prelu"] = _tensor(
                "x_prelu",
                [1, 2, 5, 7],
            )
            concat_op = next(
                op
                for op in model_ir.operators
                if op.op_type == "CONCATENATION"
            )
            concat_index = model_ir.operators.index(concat_op)
            model_ir.operators.insert(
                concat_index,
                OperatorIR(
                    "PRELU",
                    ["x_nchw", "prelu_alpha"],
                    ["x_prelu"],
                ),
            )
            concat_op.inputs[0] = "x_prelu"
        else:
            model_ir.tensors["a_prelu"] = _tensor(
                "a_prelu",
                [1, 3, 5, 7],
            )
            model_ir.tensors["a_prelu"].quantization = QuantParamIR(
                scale=[0.9] * 3,
                zero_point=[0] * 3,
                quantized_dimension=1,
            )
            add_op = next(
                op for op in model_ir.operators if op.op_type == "ADD"
            )
            add_index = model_ir.operators.index(add_op)
            prelu_ops = [
                OperatorIR(
                    "PRELU",
                    ["a_nchw", "prelu_alpha"],
                    ["a_prelu"],
                )
            ]
            add_op.inputs[0] = "a_prelu"
            if shared_operands:
                model_ir.tensors["b_prelu"] = _tensor(
                    "b_prelu",
                    [1, 3, 5, 7],
                )
                prelu_ops.append(
                    OperatorIR(
                        "PRELU",
                        ["b_nchw", "prelu_alpha"],
                        ["b_prelu"],
                    )
                )
                add_op.inputs[1] = "b_prelu"
            model_ir.operators[add_index:add_index] = prelu_ops
            if boundary == "prelu_operand_fanout":
                model_ir.tensors["prelu_side"] = _tensor(
                    "prelu_side",
                    [1, 3, 5, 7],
                )
                model_ir.outputs.append("prelu_side")
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["a_prelu"],
                        ["prelu_side"],
                    )
                )
    if boundary in {
        "softmax_operand",
        "softmax_operand_fanout",
        "softmax_companion",
    }:
        companion = boundary == "softmax_companion"
        source_name = "x_nhwc" if companion else "a_nhwc"
        adapter_name = "x_nchw" if companion else "a_nchw"
        output_name = "x_softmax" if companion else "a_softmax"
        channels = 2 if companion else 3
        model_ir.tensors[source_name].quantization = QuantParamIR(
            scale=[0.3] * channels,
            zero_point=[0] * channels,
            quantized_dimension=3,
        )
        model_ir.tensors[adapter_name].quantization = QuantParamIR(
            scale=[0.3] * channels,
            zero_point=[0] * channels,
            quantized_dimension=1,
        )
        model_ir.tensors[output_name] = _tensor(
            output_name,
            [1, channels, 5, 7],
        )
        model_ir.tensors[output_name].quantization = {
            "scale": [0.6] * channels,
            "zero_point": [0] * channels,
            "quantized_dimension": 1,
        }
        softmax_op = OperatorIR(
            "SOFTMAX",
            [adapter_name],
            [output_name],
            options={"beta": 0.75},
        )
        if companion:
            concat_op = next(
                op
                for op in model_ir.operators
                if op.op_type == "CONCATENATION"
            )
            concat_index = model_ir.operators.index(concat_op)
            model_ir.operators.insert(concat_index, softmax_op)
            concat_op.inputs[0] = output_name
        else:
            add_op = next(
                op for op in model_ir.operators if op.op_type == "ADD"
            )
            add_index = model_ir.operators.index(add_op)
            model_ir.operators.insert(add_index, softmax_op)
            add_op.inputs[0] = output_name
            if boundary == "softmax_operand_fanout":
                model_ir.tensors["softmax_side"] = _tensor(
                    "softmax_side",
                    [1, channels, 5, 7],
                )
                model_ir.outputs.append("softmax_side")
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        [output_name],
                        ["softmax_side"],
                    )
                )
    if boundary in {
        "leaky_operand",
        "leaky_operand_fanout",
        "leaky_companion",
    }:
        companion = boundary == "leaky_companion"
        prefix = "x" if companion else "a"
        adapter_name = f"{prefix}_nchw"
        source_name = f"{prefix}_nhwc"
        channels = 2 if companion else 3
        internal_names = [
            f"{prefix}_neg_out",
            f"{prefix}_neg_relu",
            f"{prefix}_pos_relu",
            f"{prefix}_neg_scaled",
            f"{prefix}_leaky",
        ]
        for tensor_name in internal_names:
            model_ir.tensors[tensor_name] = _tensor(
                tensor_name,
                [1, channels, 5, 7],
            )
            model_ir.tensors[tensor_name].quantization = QuantParamIR(
                scale=[0.4] * channels,
                zero_point=[0] * channels,
                quantized_dimension=1,
            )
        alpha_name = f"{prefix}_leaky_alpha"
        model_ir.tensors[alpha_name] = TensorIR(
            name=alpha_name,
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
            data=np.asarray([0.1], dtype=np.float32),
        )
        leaky_ops = [
            OperatorIR("NEG", [adapter_name], [f"{prefix}_neg_out"]),
            OperatorIR(
                "RELU",
                [f"{prefix}_neg_out"],
                [f"{prefix}_neg_relu"],
            ),
            OperatorIR(
                "RELU",
                [adapter_name],
                [f"{prefix}_pos_relu"],
            ),
            OperatorIR(
                "MUL",
                [f"{prefix}_neg_relu", alpha_name],
                [f"{prefix}_neg_scaled"],
            ),
            OperatorIR(
                "SUB",
                [f"{prefix}_pos_relu", f"{prefix}_neg_scaled"],
                [f"{prefix}_leaky"],
            ),
        ]
        if companion:
            concat_op = next(
                op
                for op in model_ir.operators
                if op.op_type == "CONCATENATION"
            )
            concat_index = model_ir.operators.index(concat_op)
            model_ir.operators[concat_index:concat_index] = leaky_ops
            concat_op.inputs[0] = f"{prefix}_leaky"
        else:
            add_op = next(
                op for op in model_ir.operators if op.op_type == "ADD"
            )
            add_index = model_ir.operators.index(add_op)
            model_ir.operators[add_index:add_index] = leaky_ops
            add_op.inputs[0] = f"{prefix}_leaky"
            if boundary == "leaky_operand_fanout":
                model_ir.tensors["leaky_side"] = _tensor(
                    "leaky_side",
                    [1, channels, 5, 7],
                )
                model_ir.outputs.append("leaky_side")
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        [f"{prefix}_leaky"],
                        ["leaky_side"],
                    )
                )
    if boundary in {
        "shared_split_add_tree",
        "shared_split_external_consumer",
    }:
        model_ir.tensors["a_nhwc"].shape = [1, 5, 7, 6]
        model_ir.tensors["a_nhwc"].shape_signature = [1, 5, 7, 6]
        model_ir.tensors["a_nchw"].shape = [1, 6, 5, 7]
        model_ir.tensors["a_nchw"].shape_signature = [1, 6, 5, 7]
        model_ir.tensors["split_axis"] = _int_tensor("split_axis", [1])
        for split_output_name in ["a_split0", "a_split1"]:
            model_ir.tensors[split_output_name] = _tensor(
                split_output_name,
                [1, 3, 5, 7],
            )
            model_ir.tensors[split_output_name].quantization = QuantParamIR(
                scale=[0.5] * 3,
                zero_point=[0] * 3,
                quantized_dimension=1,
            )
        model_ir.tensors["inner_sum_nchw"] = _tensor(
            "inner_sum_nchw",
            [1, 3, 5, 7],
        )
        add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
        add_index = model_ir.operators.index(add_op)
        model_ir.operators[add_index:add_index] = [
            OperatorIR(
                "SPLIT",
                ["split_axis", "a_nchw"],
                ["a_split0", "a_split1"],
            ),
            OperatorIR(
                "ADD",
                ["a_split0", "b_nchw"],
                ["inner_sum_nchw"],
                options={"fusedActivationFunction": "NONE"},
            ),
        ]
        add_op.inputs = ["inner_sum_nchw", "a_split1"]
        if boundary == "shared_split_external_consumer":
            model_ir.tensors["split_side"] = _tensor(
                "split_side",
                [1, 3, 5, 7],
            )
            model_ir.outputs.append("split_side")
            model_ir.operators.append(
                OperatorIR("IDENTITY", ["a_split1"], ["split_side"])
            )
    if boundary in {
        "split_shared_root_concat",
        "split_shared_root_external_consumer",
    }:
        model_ir.tensors["a_nhwc"].shape = [1, 5, 7, 6]
        model_ir.tensors["a_nhwc"].shape_signature = [1, 5, 7, 6]
        model_ir.tensors["a_nchw"].shape = [1, 6, 5, 7]
        model_ir.tensors["a_nchw"].shape_signature = [1, 6, 5, 7]
        model_ir.tensors["split_axis"] = _int_tensor("split_axis", [1])
        for split_output_name in ["a_split0", "a_split1"]:
            model_ir.tensors[split_output_name] = _tensor(
                split_output_name,
                [1, 3, 5, 7],
            )
        add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
        add_index = model_ir.operators.index(add_op)
        model_ir.operators.insert(
            add_index,
            OperatorIR(
                "SPLIT",
                ["split_axis", "a_nchw"],
                ["a_split0", "a_split1"],
            ),
        )
        add_op.inputs[0] = "a_split0"
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["x_nchw", "sum_nchw", "a_split1"]
        for tensor_name, shape in {
            "concat_nchw": [1, 8, 5, 7],
            "concat_nhwc": [1, 5, 7, 8],
            "y": [1, 5, 7, 8],
        }.items():
            model_ir.tensors[tensor_name].shape = list(shape)
            model_ir.tensors[tensor_name].shape_signature = list(shape)
        if boundary == "split_shared_root_external_consumer":
            model_ir.tensors["split_root_side"] = _tensor(
                "split_root_side",
                [1, 3, 5, 7],
            )
            model_ir.outputs.append("split_root_side")
            model_ir.operators.append(
                OperatorIR(
                    "IDENTITY",
                    ["a_split1"],
                    ["split_root_side"],
                )
            )
    if boundary in {
        "shared_add_fanout_branches",
        "shared_add_fanout_root",
        "shared_add_fanout_external_consumer",
    }:
        model_ir.inputs.append("c_nhwc")
        model_ir.tensors["c_nhwc"] = _tensor(
            "c_nhwc",
            [1, 5, 7, 3],
        )
        model_ir.tensors["c_nchw"] = _tensor(
            "c_nchw",
            [1, 3, 5, 7],
        )
        model_ir.tensors["inner_sum_nchw"] = _tensor(
            "inner_sum_nchw",
            [1, 3, 5, 7],
        )
        model_ir.tensors["inner_sum_nchw"].quantization = QuantParamIR(
            scale=[0.6] * 3,
            zero_point=[0] * 3,
            quantized_dimension=1,
        )
        outer_add = next(
            op for op in model_ir.operators if op.op_type == "ADD"
        )
        outer_add_index = model_ir.operators.index(outer_add)
        prefix_ops = [
            OperatorIR(
                "TRANSPOSE",
                ["c_nhwc", "pre_perm"],
                ["c_nchw"],
            ),
            OperatorIR(
                "ADD",
                ["a_nchw", "b_nchw"],
                ["inner_sum_nchw"],
                options={"fusedActivationFunction": "NONE"},
            ),
        ]
        outer_add.inputs = ["inner_sum_nchw", "c_nchw"]
        branch_fanout = boundary != "shared_add_fanout_root"
        if branch_fanout:
            model_ir.inputs.append("d_nhwc")
            model_ir.tensors["d_nhwc"] = _tensor(
                "d_nhwc",
                [1, 5, 7, 3],
            )
            model_ir.tensors["d_nchw"] = _tensor(
                "d_nchw",
                [1, 3, 5, 7],
            )
            model_ir.tensors["right_sum_nchw"] = _tensor(
                "right_sum_nchw",
                [1, 3, 5, 7],
            )
            prefix_ops.insert(
                1,
                OperatorIR(
                    "TRANSPOSE",
                    ["d_nhwc", "pre_perm"],
                    ["d_nchw"],
                ),
            )
        model_ir.operators[outer_add_index:outer_add_index] = prefix_ops
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        if branch_fanout:
            concat_index = model_ir.operators.index(concat_op)
            model_ir.operators.insert(
                concat_index,
                OperatorIR(
                    "ADD",
                    ["inner_sum_nchw", "d_nchw"],
                    ["right_sum_nchw"],
                    options={"fusedActivationFunction": "NONE"},
                ),
            )
            concat_op.inputs = ["x_nchw", "sum_nchw", "right_sum_nchw"]
        else:
            concat_op.inputs = ["x_nchw", "inner_sum_nchw", "sum_nchw"]
        for tensor_name, shape in {
            "concat_nchw": [1, 8, 5, 7],
            "concat_nhwc": [1, 5, 7, 8],
            "y": [1, 5, 7, 8],
        }.items():
            model_ir.tensors[tensor_name].shape = list(shape)
            model_ir.tensors[tensor_name].shape_signature = list(shape)
        if boundary == "shared_add_fanout_external_consumer":
            model_ir.tensors["inner_sum_side"] = _tensor(
                "inner_sum_side",
                [1, 3, 5, 7],
            )
            model_ir.outputs.append("inner_sum_side")
            model_ir.operators.append(
                OperatorIR(
                    "IDENTITY",
                    ["inner_sum_nchw"],
                    ["inner_sum_side"],
                )
            )
    if boundary in {"recursive_operand", "recursive_operand_post"}:
        model_ir.inputs.append("c_nhwc")
        model_ir.tensors["c_nhwc"] = _tensor(
            "c_nhwc",
            [1, 5, 7, 3],
        )
        model_ir.tensors["c_nchw"] = _tensor(
            "c_nchw",
            [1, 3, 5, 7],
        )
        model_ir.tensors["inner_sum_nchw"] = _tensor(
            "inner_sum_nchw",
            [1, 3, 5, 7],
        )
        model_ir.tensors["inner_sum_nchw"].quantization = QuantParamIR(
            scale=[0.5] * 3,
            zero_point=[0] * 3,
            quantized_dimension=1,
        )
        add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
        add_index = model_ir.operators.index(add_op)
        model_ir.operators[add_index:add_index] = [
            OperatorIR(
                "TRANSPOSE",
                ["c_nhwc", "pre_perm"],
                ["c_nchw"],
            ),
            OperatorIR(
                "ADD",
                ["a_nchw", "b_nchw"],
                ["inner_sum_nchw"],
                options={"fusedActivationFunction": "NONE"},
            ),
        ]
        add_op.inputs = ["inner_sum_nchw", "c_nchw"]
        if boundary == "recursive_operand_post":
            model_ir.tensors["inner_sum_nhwc"] = _tensor(
                "inner_sum_nhwc",
                [1, 5, 7, 3],
            )
            model_ir.tensors["inner_observed"] = _tensor(
                "inner_observed",
                [1, 5, 7, 3],
            )
            model_ir.operators.extend(
                [
                    OperatorIR(
                        "TRANSPOSE",
                        ["inner_sum_nchw", "post_perm"],
                        ["inner_sum_nhwc"],
                    ),
                    OperatorIR(
                        "RELU",
                        ["inner_sum_nhwc"],
                        ["inner_observed"],
                    ),
                ]
            )
            model_ir.outputs.append("inner_observed")
    if boundary == "recursive_cycle":
        model_ir.tensors["inner_sum_nchw"] = _tensor(
            "inner_sum_nchw",
            [1, 3, 5, 7],
        )
        add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
        add_index = model_ir.operators.index(add_op)
        model_ir.operators.insert(
            add_index,
            OperatorIR(
                "ADD",
                ["sum_nchw", "a_nchw"],
                ["inner_sum_nchw"],
            ),
        )
        add_op.inputs = ["inner_sum_nchw", "b_nchw"]
    if boundary == "public_concat":
        model_ir.outputs.append("concat_nchw")
    if boundary == "public_post":
        model_ir.outputs.append("concat_nhwc")
    if boundary == "raw_residual_input":
        model_ir.inputs.append("residual_nchw")
        model_ir.tensors["residual_nchw"] = _tensor(
            "residual_nchw",
            [1, 2, 5, 7],
        )
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["residual_nchw", "sum_nchw"]
    return model_ir


def _assert_model_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert actual.metadata == expected.metadata
    assert [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in actual.operators
    ] == [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in expected.operators
    ]
    assert actual.tensors.keys() == expected.tensors.keys()
    for name, tensor in actual.tensors.items():
        expected_tensor = expected.tensors[name]
        assert tensor.dtype == expected_tensor.dtype
        assert tensor.shape == expected_tensor.shape
        assert tensor.shape_signature == expected_tensor.shape_signature
        assert tensor.quantization == expected_tensor.quantization
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


def _assert_add_rewritten(
    model_ir: ModelIR,
    *,
    expected_inputs: list[str] | None = None,
) -> None:
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == (expected_inputs or ["a_nhwc", "b_nhwc"])
    assert add_op.options == {"fusedActivationFunction": "NONE"}
    assert model_ir.tensors["sum_nchw"].shape == [1, 5, 7, 3]
    quantization = model_ir.tensors["sum_nchw"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs[-1] == "sum_nchw"
    assert concat_op.outputs == ["concat_nhwc"]
    assert concat_op.options["axis"] == 3


@pytest.mark.parametrize("all_add", [False, True])
def test_nhwc_direct_only_add_is_indexed(all_add: bool) -> None:
    model_ir = _add_model(all_add=all_add)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_add",
        "add_output_fanout",
        "public_add_output",
        "public_add_post",
        "invalid_source_rank",
        "invalid_adapter_rank",
        "invalid_add_output_rank",
        "wrong_add_pre_permutation",
        "spatial_shape_mismatch",
        "raw_residual_input",
        "wrong_post_permutation",
        "public_concat",
        "public_post",
        "recursive_cycle",
        "shared_split_external_consumer",
        "split_shared_root_external_consumer",
        "shared_add_fanout_external_consumer",
        "pad_operand_fanout",
        "slice_operand_fanout",
        "dequantize_operand_fanout",
        "prelu_operand_fanout",
        "softmax_operand_fanout",
        "leaky_operand_fanout",
    ],
)
def test_nhwc_add_rejects_unsafe_or_partial_match(boundary: str) -> None:
    model_ir = _add_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


@pytest.mark.parametrize(
    "boundary",
    [
        "add_adapter_fanout",
        "add_second_adapter_fanout",
        "public_add_adapter",
    ],
)
def test_nhwc_add_retains_shared_or_public_source_adapter(
    boundary: str,
) -> None:
    model_ir = _add_model(boundary=boundary)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(model_ir)
    remaining_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    retained_adapter = (
        "b_nchw"
        if boundary == "add_second_adapter_fanout"
        else "a_nchw"
    )
    assert [op.outputs for op in remaining_transposes] == [[retained_adapter]]
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


def test_nhwc_add_adapter_shared_with_root_concat_is_indexed() -> None:
    model_ir = _add_model(boundary="adapter_shared_with_concat")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(model_ir)
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x_nhwc", "a_nhwc", "sum_nchw"]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


def test_nhwc_add_unary_operand_is_indexed() -> None:
    model_ir = _add_model(boundary="unary_operand")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(
        model_ir,
        expected_inputs=["a_relu", "b_nhwc"],
    )
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == ["a_relu", "b_nhwc"]
    unary_op = next(op for op in model_ir.operators if op.op_type == "RELU")
    assert unary_op.inputs == ["a_nhwc"]
    assert model_ir.tensors["a_relu"].shape == [1, 5, 7, 3]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


def test_nhwc_add_swish_operand_is_indexed() -> None:
    model_ir = _add_model(boundary="swish_operand")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(
        model_ir,
        expected_inputs=["a_swish", "b_nhwc"],
    )
    logistic_op = next(
        op for op in model_ir.operators if op.op_type == "LOGISTIC"
    )
    mul_op = next(op for op in model_ir.operators if op.op_type == "MUL")
    assert logistic_op.inputs == ["a_nhwc"]
    assert mul_op.inputs == ["a_logistic", "a_nhwc"]
    assert model_ir.tensors["a_logistic"].shape == [1, 5, 7, 3]
    assert model_ir.tensors["a_swish"].shape == [1, 5, 7, 3]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


def test_nhwc_add_split_operand_is_indexed() -> None:
    model_ir = _add_model(boundary="split_operand")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(
        model_ir,
        expected_inputs=["a_split0", "b_nhwc"],
    )
    split_op = next(op for op in model_ir.operators if op.op_type == "SPLIT")
    assert split_op.inputs[1] == "a_nhwc"
    np.testing.assert_array_equal(
        model_ir.tensors[split_op.inputs[0]].data,
        np.asarray([3], dtype=np.int32),
    )
    for output_name in ("a_split0", "a_split1"):
        assert model_ir.tensors[output_name].shape == [1, 5, 7, 3]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


@pytest.mark.parametrize("pad_location", ["operand", "companion"])
def test_nhwc_add_reuses_bounded_pad_plan(pad_location: str) -> None:
    model_ir = _add_model(boundary=f"pad_{pad_location}")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    pad_output = "a_pad" if pad_location == "operand" else "x_pad"
    pad_op = next(op for op in model_ir.operators if op.outputs == [pad_output])
    expected_source = "a_nhwc" if pad_location == "operand" else "x_nhwc"
    assert pad_op.inputs == [expected_source, "pads_nchw", "pad_value"]
    np.testing.assert_array_equal(
        model_ir.tensors["pads_nchw"].data,
        np.asarray(
            [[0, 0], [0, 1], [0, 0], [0, 0]],
            dtype=np.int32,
        ),
    )
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    expected_add_inputs = (
        ["a_pad", "b_nhwc"]
        if pad_location == "operand"
        else ["a_nhwc", "b_nhwc"]
    )
    assert add_op.inputs == expected_add_inputs
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    if pad_location == "companion":
        assert concat_op.inputs == ["x_pad", "sum_nchw"]
    expected_pad_shape = (
        [1, 5, 7, 3]
        if pad_location == "operand"
        else [1, 5, 7, 2]
    )
    assert model_ir.tensors[pad_output].shape == expected_pad_shape
    if pad_location == "operand":
        quantization = model_ir.tensors[pad_output].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("slice_location", ["operand", "companion"])
def test_nhwc_add_reuses_bounded_slice_plan(slice_location: str) -> None:
    model_ir = _add_model(boundary=f"slice_{slice_location}")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    slice_output = (
        "a_slice" if slice_location == "operand" else "x_slice"
    )
    slice_op = next(
        op for op in model_ir.operators if op.outputs == [slice_output]
    )
    expected_source = (
        "a_nhwc" if slice_location == "operand" else "x_nhwc"
    )
    assert slice_op.inputs == [
        expected_source,
        "slice_begin",
        "slice_size",
    ]
    expected_channels = 3 if slice_location == "operand" else 2
    np.testing.assert_array_equal(
        model_ir.tensors["slice_begin"].data,
        np.asarray([0, 0, 0, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["slice_size"].data,
        np.asarray([1, 5, 7, expected_channels], dtype=np.int32),
    )
    assert model_ir.tensors[slice_output].shape == [
        1,
        5,
        7,
        expected_channels,
    ]
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    expected_add_inputs = (
        ["a_slice", "b_nhwc"]
        if slice_location == "operand"
        else ["a_nhwc", "b_nhwc"]
    )
    assert add_op.inputs == expected_add_inputs
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    if slice_location == "companion":
        assert concat_op.inputs == ["x_slice", "sum_nchw"]
    if slice_location == "operand":
        quantization = model_ir.tensors[slice_output].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("dequantize_location", ["operand", "companion"])
def test_nhwc_add_reuses_dequantize_plan(
    dequantize_location: str,
) -> None:
    model_ir = _add_model(
        boundary=f"dequantize_{dequantize_location}"
    )

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    companion = dequantize_location == "companion"
    source_name = "x_nhwc" if companion else "a_nhwc"
    output_name = "x_dequantized" if companion else "a_dequantized"
    channels = 2 if companion else 3
    dequantize_op = next(
        op for op in model_ir.operators if op.outputs == [output_name]
    )
    assert dequantize_op.inputs == [source_name]
    assert model_ir.tensors[output_name].shape == [1, 5, 7, channels]
    output_quantization = model_ir.tensors[output_name].quantization
    assert isinstance(output_quantization, dict)
    assert output_quantization["quantized_dimension"] == 3
    source_quantization = model_ir.tensors[source_name].quantization
    assert isinstance(source_quantization, QuantParamIR)
    assert source_quantization.quantized_dimension == 3
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == (
        ["a_nhwc", "b_nhwc"]
        if companion
        else ["a_dequantized", "b_nhwc"]
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    if companion:
        assert concat_op.inputs == ["x_dequantized", "sum_nchw"]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("prelu_location", ["operand", "companion"])
def test_nhwc_add_reuses_prelu_plan(prelu_location: str) -> None:
    model_ir = _add_model(boundary=f"prelu_{prelu_location}")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    companion = prelu_location == "companion"
    source_name = "x_nhwc" if companion else "a_nhwc"
    output_name = "x_prelu" if companion else "a_prelu"
    channels = 2 if companion else 3
    prelu_op = next(
        op for op in model_ir.operators if op.outputs == [output_name]
    )
    assert prelu_op.inputs == [source_name, "prelu_alpha"]
    assert model_ir.tensors[output_name].shape == [1, 5, 7, channels]
    alpha = model_ir.tensors["prelu_alpha"]
    assert alpha.shape == [1, 1, 1, channels]
    assert isinstance(alpha.quantization, QuantParamIR)
    assert alpha.quantization.quantized_dimension == 3
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == (
        ["a_nhwc", "b_nhwc"]
        if companion
        else ["a_prelu", "b_nhwc"]
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    if companion:
        assert concat_op.inputs == ["x_prelu", "sum_nchw"]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


def test_nhwc_add_reuses_one_shared_prelu_alpha_clone() -> None:
    model_ir = _add_model(boundary="prelu_shared_alpha_operands")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    prelu_ops = [op for op in model_ir.operators if op.op_type == "PRELU"]
    assert len(prelu_ops) == 2
    alpha_names = {str(op.inputs[1]) for op in prelu_ops}
    assert len(alpha_names) == 1
    alpha_name = alpha_names.pop()
    alpha = model_ir.tensors[alpha_name]
    assert alpha.shape == [1, 1, 1, 3]
    assert alpha.is_variable
    assert alpha.logical_layout == "NCHW"
    assert alpha.physical_layout == "NCHW"
    assert alpha.onnx_tensor_name == "onnx_prelu_alpha"
    assert isinstance(alpha.quantization, QuantParamIR)
    assert alpha.quantization.quantized_dimension == 3
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == ["a_prelu", "b_prelu"]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("softmax_location", ["operand", "companion"])
def test_nhwc_add_reuses_semantics_preserving_softmax_plan(
    softmax_location: str,
) -> None:
    model_ir = _add_model(boundary=f"softmax_{softmax_location}")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    companion = softmax_location == "companion"
    adapter_name = "x_nchw" if companion else "a_nchw"
    output_name = "x_softmax" if companion else "a_softmax"
    channels = 2 if companion else 3
    softmax_op = next(
        op for op in model_ir.operators if op.op_type == "SOFTMAX"
    )
    assert softmax_op.inputs == [f"{adapter_name}_axis_last"]
    assert softmax_op.outputs == [f"{output_name}_axis_last"]
    assert softmax_op.options == {"beta": 0.75}
    assert model_ir.tensors[f"{adapter_name}_axis_last"].shape == [
        1,
        5,
        channels,
        7,
    ]
    assert model_ir.tensors[output_name].shape == [1, 5, 7, channels]
    output_quantization = model_ir.tensors[output_name].quantization
    assert isinstance(output_quantization, dict)
    assert output_quantization["quantized_dimension"] == 3
    local_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    assert len(local_transposes) == 2
    for transpose_op in local_transposes:
        np.testing.assert_array_equal(
            model_ir.tensors[transpose_op.inputs[1]].data,
            np.asarray([0, 1, 3, 2], dtype=np.int32),
        )
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == (
        ["a_nhwc", "b_nhwc"]
        if companion
        else ["a_softmax", "b_nhwc"]
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    if companion:
        assert concat_op.inputs == ["x_softmax", "sum_nchw"]


@pytest.mark.parametrize("leaky_location", ["operand", "companion"])
def test_nhwc_add_reuses_exact_pseudo_leaky_plan(
    leaky_location: str,
) -> None:
    model_ir = _add_model(boundary=f"leaky_{leaky_location}")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    companion = leaky_location == "companion"
    prefix = "x" if companion else "a"
    channels = 2 if companion else 3
    neg_op = next(
        op
        for op in model_ir.operators
        if op.outputs == [f"{prefix}_neg_out"]
    )
    pos_relu_op = next(
        op
        for op in model_ir.operators
        if op.outputs == [f"{prefix}_pos_relu"]
    )
    assert neg_op.inputs == [f"{prefix}_nhwc"]
    assert pos_relu_op.inputs == [f"{prefix}_nhwc"]
    for suffix in [
        "neg_out",
        "neg_relu",
        "pos_relu",
        "neg_scaled",
        "leaky",
    ]:
        tensor = model_ir.tensors[f"{prefix}_{suffix}"]
        assert tensor.shape == [1, 5, 7, channels]
        assert isinstance(tensor.quantization, QuantParamIR)
        assert tensor.quantization.quantized_dimension == 3
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == (
        ["a_nhwc", "b_nhwc"]
        if companion
        else ["a_leaky", "b_nhwc"]
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    if companion:
        assert concat_op.inputs == ["x_leaky", "sum_nchw"]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


def test_nhwc_recursive_add_operand_is_indexed_once() -> None:
    model_ir = _add_model(boundary="recursive_operand")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    inner_add = next(
        op for op in model_ir.operators if op.outputs == ["inner_sum_nchw"]
    )
    outer_add = next(
        op for op in model_ir.operators if op.outputs == ["sum_nchw"]
    )
    assert inner_add.inputs == ["a_nhwc", "b_nhwc"]
    assert outer_add.inputs == ["inner_sum_nchw", "c_nhwc"]
    assert model_ir.tensors["inner_sum_nchw"].shape == [1, 5, 7, 3]
    inner_quantization = model_ir.tensors["inner_sum_nchw"].quantization
    assert isinstance(inner_quantization, QuantParamIR)
    assert inner_quantization.quantized_dimension == 3
    assert model_ir.tensors["sum_nchw"].shape == [1, 5, 7, 3]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


def test_nhwc_recursive_add_keeps_nested_post_bound_to_inner_output() -> None:
    model_ir = _add_model(boundary="recursive_operand_post")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    observer = next(
        op for op in model_ir.operators if op.outputs == ["inner_observed"]
    )
    assert observer.inputs == ["inner_sum_nchw"]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


def test_nhwc_shared_split_outputs_feed_one_recursive_add_tree() -> None:
    model_ir = _add_model(boundary="shared_split_add_tree")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    split_op = next(op for op in model_ir.operators if op.op_type == "SPLIT")
    assert split_op.inputs == ["split_axis", "a_nhwc"]
    np.testing.assert_array_equal(
        model_ir.tensors["split_axis"].data,
        np.asarray([3], dtype=np.int32),
    )
    inner_add = next(
        op for op in model_ir.operators if op.outputs == ["inner_sum_nchw"]
    )
    outer_add = next(
        op for op in model_ir.operators if op.outputs == ["sum_nchw"]
    )
    assert inner_add.inputs == ["a_split0", "b_nhwc"]
    assert outer_add.inputs == ["inner_sum_nchw", "a_split1"]
    for tensor_name in ["a_split0", "a_split1"]:
        tensor = model_ir.tensors[tensor_name]
        assert tensor.shape == [1, 5, 7, 3]
        assert isinstance(tensor.quantization, QuantParamIR)
        assert tensor.quantization.quantized_dimension == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("split_input_first", [False, True])
def test_nhwc_split_outputs_feed_add_and_same_root_concat(
    split_input_first: bool,
) -> None:
    model_ir = _add_model(boundary="split_shared_root_concat")
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    if split_input_first:
        concat_op.inputs = ["a_split1", "x_nchw", "sum_nchw"]

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    expected_inputs = (
        ["a_split1", "x_nhwc", "sum_nchw"]
        if split_input_first
        else ["x_nhwc", "sum_nchw", "a_split1"]
    )
    assert concat_op.inputs == expected_inputs
    split_op = next(op for op in model_ir.operators if op.op_type == "SPLIT")
    assert split_op.inputs == ["split_axis", "a_nhwc"]
    np.testing.assert_array_equal(
        model_ir.tensors["split_axis"].data,
        np.asarray([3], dtype=np.int32),
    )
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == ["a_split0", "b_nhwc"]
    for tensor_name in ["a_split0", "a_split1"]:
        assert model_ir.tensors[tensor_name].shape == [1, 5, 7, 3]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("fanout_target", ["branches", "root"])
def test_nhwc_shared_add_output_stays_inside_selected_candidate(
    fanout_target: str,
) -> None:
    model_ir = _add_model(boundary=f"shared_add_fanout_{fanout_target}")

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    inner_add = next(
        op for op in model_ir.operators if op.outputs == ["inner_sum_nchw"]
    )
    outer_add = next(
        op for op in model_ir.operators if op.outputs == ["sum_nchw"]
    )
    assert inner_add.inputs == ["a_nhwc", "b_nhwc"]
    assert outer_add.inputs == ["inner_sum_nchw", "c_nhwc"]
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    if fanout_target == "branches":
        right_add = next(
            op
            for op in model_ir.operators
            if op.outputs == ["right_sum_nchw"]
        )
        assert right_add.inputs == ["inner_sum_nchw", "d_nhwc"]
        assert concat_op.inputs == [
            "x_nhwc",
            "sum_nchw",
            "right_sum_nchw",
        ]
    else:
        assert concat_op.inputs == ["x_nhwc", "inner_sum_nchw", "sum_nchw"]
    inner_tensor = model_ir.tensors["inner_sum_nchw"]
    assert inner_tensor.shape == [1, 5, 7, 3]
    assert isinstance(inner_tensor.quantization, QuantParamIR)
    assert inner_tensor.quantization.quantized_dimension == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


def test_nhwc_add_output_post_adapter_is_indexed() -> None:
    model_ir = _add_model(boundary="add_output_post_transpose")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    side_identity = next(
        op for op in model_ir.operators if op.outputs == ["sum_side"]
    )
    assert side_identity.inputs == ["sum_nchw"]
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"
