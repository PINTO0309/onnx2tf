from __future__ import annotations

from copy import deepcopy
import os
from typing import Any, Dict, List, Optional, Tuple

import flatbuffers

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.opcodes import build_operator_codes, operator_code_key
from onnx2tf.tflite_builder.signature_builder import build_signature_defs
from onnx2tf.tflite_builder.tensor_buffer_builder import build_tensors_and_buffers


def _enum(schema_tflite: Dict[str, Any], enum_name: str, item_name: str) -> int:
    return int(getattr(schema_tflite[enum_name], item_name))


def _require_tensor_indices(
    *,
    tensor_index_map: Dict[str, int],
    tensor_names: List[str],
    op_type: str,
    tensor_role: str,
) -> List[int]:
    indices: List[int] = []
    for name in tensor_names:
        if name not in tensor_index_map:
            raise KeyError(
                f"Tensor index is missing for {tensor_role}: name={name}, op={op_type}"
            )
        indices.append(tensor_index_map[name])
    return indices


def _prune_unused_tensors_in_place(model_ir: ModelIR) -> None:
    used_tensor_names = set(model_ir.inputs + model_ir.outputs)
    for op in model_ir.operators:
        used_tensor_names.update(op.inputs)
        used_tensor_names.update(op.outputs)
    unused_tensor_names = [
        name for name in model_ir.tensors.keys() if name not in used_tensor_names
    ]
    for tensor_name in unused_tensor_names:
        del model_ir.tensors[tensor_name]


def _prune_dead_operators_in_place(model_ir: ModelIR) -> None:
    if len(model_ir.operators) == 0:
        return

    live_tensors = set(model_ir.outputs)
    keep_flags = [False for _ in model_ir.operators]

    for op_idx in range(len(model_ir.operators) - 1, -1, -1):
        op = model_ir.operators[op_idx]
        if len(op.outputs) == 0:
            continue
        if any(output_name in live_tensors for output_name in op.outputs):
            keep_flags[op_idx] = True
            for input_name in op.inputs:
                live_tensors.add(input_name)

    if all(keep_flags):
        return

    model_ir.operators = [
        op for idx, op in enumerate(model_ir.operators) if keep_flags[idx]
    ]


def _sanitize_model_ir_for_serialization(model_ir: ModelIR) -> ModelIR:
    # Keep serialization side-effect free because one ModelIR instance can be
    # reused across multiple output variants.
    sanitized_model_ir = deepcopy(model_ir)
    _prune_dead_operators_in_place(sanitized_model_ir)
    _prune_unused_tensors_in_place(sanitized_model_ir)
    return sanitized_model_ir


def _build_binary_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options_name = {
        "ADD": "AddOptions",
        "SUB": "SubOptions",
        "MUL": "MulOptions",
        "DIV": "DivOptions",
    }[op.op_type]
    options_cls = schema_tflite[f"{options_name}T"]
    options = options_cls()
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    return _enum(schema_tflite, "BuiltinOptions", options_name), options


def _build_reshape_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ReshapeOptionsT"]()
    options.newShape = [int(v) for v in op.options.get("newShape", [])]
    return _enum(schema_tflite, "BuiltinOptions", "ReshapeOptions"), options


def _build_concat_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ConcatenationOptionsT"]()
    options.axis = int(op.options["axis"])
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    return _enum(schema_tflite, "BuiltinOptions", "ConcatenationOptions"), options


def _build_softmax_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["SoftmaxOptionsT"]()
    options.beta = float(op.options.get("beta", 1.0))
    return _enum(schema_tflite, "BuiltinOptions", "SoftmaxOptions"), options


def _build_transpose_options(schema_tflite: Dict[str, Any], _op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["TransposeOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "TransposeOptions"), options


def _build_reducer_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ReducerOptionsT"]()
    options.keepDims = bool(op.options.get("keepDims", True))
    return _enum(schema_tflite, "BuiltinOptions", "ReducerOptions"), options


def _build_squeeze_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["SqueezeOptionsT"]()
    options.squeezeDims = [int(v) for v in op.options.get("squeezeDims", [])]
    return _enum(schema_tflite, "BuiltinOptions", "SqueezeOptions"), options


def _build_gather_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["GatherOptionsT"]()
    options.axis = int(op.options.get("axis", 0))
    options.batchDims = int(op.options.get("batchDims", 0))
    return _enum(schema_tflite, "BuiltinOptions", "GatherOptions"), options


def _build_l2_norm_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["L2NormOptionsT"]()
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    return _enum(schema_tflite, "BuiltinOptions", "L2NormOptions"), options


def _build_space_to_depth_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["SpaceToDepthOptionsT"]()
    options.blockSize = int(op.options.get("blockSize", 1))
    return _enum(schema_tflite, "BuiltinOptions", "SpaceToDepthOptions"), options


def _build_conv_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["Conv2DOptionsT"]()
    options.padding = _enum(schema_tflite, "Padding", str(op.options["padding"]))
    options.strideH = int(op.options["strideH"])
    options.strideW = int(op.options["strideW"])
    options.dilationHFactor = int(op.options.get("dilationHFactor", 1))
    options.dilationWFactor = int(op.options.get("dilationWFactor", 1))
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    return _enum(schema_tflite, "BuiltinOptions", "Conv2DOptions"), options


def _build_depthwise_conv_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["DepthwiseConv2DOptionsT"]()
    options.padding = _enum(schema_tflite, "Padding", str(op.options["padding"]))
    options.strideH = int(op.options["strideH"])
    options.strideW = int(op.options["strideW"])
    options.dilationHFactor = int(op.options.get("dilationHFactor", 1))
    options.dilationWFactor = int(op.options.get("dilationWFactor", 1))
    options.depthMultiplier = int(op.options["depthMultiplier"])
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    return _enum(schema_tflite, "BuiltinOptions", "DepthwiseConv2DOptions"), options


def _build_pool2d_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["Pool2DOptionsT"]()
    options.padding = _enum(schema_tflite, "Padding", str(op.options["padding"]))
    options.strideH = int(op.options["strideH"])
    options.strideW = int(op.options["strideW"])
    options.filterHeight = int(op.options["filterHeight"])
    options.filterWidth = int(op.options["filterWidth"])
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    return _enum(schema_tflite, "BuiltinOptions", "Pool2DOptions"), options


def _build_resize_nearest_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ResizeNearestNeighborOptionsT"]()
    options.alignCorners = bool(op.options.get("alignCorners", False))
    options.halfPixelCenters = bool(op.options.get("halfPixelCenters", False))
    return _enum(schema_tflite, "BuiltinOptions", "ResizeNearestNeighborOptions"), options


def _build_resize_bilinear_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ResizeBilinearOptionsT"]()
    options.alignCorners = bool(op.options.get("alignCorners", False))
    options.halfPixelCenters = bool(op.options.get("halfPixelCenters", False))
    return _enum(schema_tflite, "BuiltinOptions", "ResizeBilinearOptions"), options


def _build_fully_connected_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["FullyConnectedOptionsT"]()
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    weights_format = str(op.options.get("weightsFormat", "DEFAULT"))
    options.weightsFormat = _enum(
        schema_tflite,
        "FullyConnectedOptionsWeightsFormat",
        weights_format,
    )
    options.keepNumDims = bool(op.options.get("keepNumDims", False))
    options.asymmetricQuantizeInputs = bool(
        op.options.get("asymmetricQuantizeInputs", False)
    )
    return _enum(schema_tflite, "BuiltinOptions", "FullyConnectedOptions"), options


def _build_builtin_options(
    schema_tflite: Dict[str, Any],
    op: OperatorIR,
) -> Tuple[int, Optional[object]]:
    if op.op_type == "CUSTOM":
        return _enum(schema_tflite, "BuiltinOptions", "NONE"), None
    if op.op_type in ["ADD", "SUB", "MUL", "DIV"]:
        return _build_binary_options(schema_tflite, op)
    if op.op_type == "RESHAPE":
        return _build_reshape_options(schema_tflite, op)
    if op.op_type == "CONCATENATION":
        return _build_concat_options(schema_tflite, op)
    if op.op_type == "SOFTMAX":
        return _build_softmax_options(schema_tflite, op)
    if op.op_type == "TRANSPOSE":
        return _build_transpose_options(schema_tflite, op)
    if op.op_type in ["MEAN", "SUM"]:
        return _build_reducer_options(schema_tflite, op)
    if op.op_type == "SQUEEZE":
        return _build_squeeze_options(schema_tflite, op)
    if op.op_type == "GATHER":
        return _build_gather_options(schema_tflite, op)
    if op.op_type == "L2_NORMALIZATION":
        return _build_l2_norm_options(schema_tflite, op)
    if op.op_type == "SPACE_TO_DEPTH":
        return _build_space_to_depth_options(schema_tflite, op)
    if op.op_type == "CONV_2D":
        return _build_conv_options(schema_tflite, op)
    if op.op_type == "DEPTHWISE_CONV_2D":
        return _build_depthwise_conv_options(schema_tflite, op)
    if op.op_type in ["AVERAGE_POOL_2D", "MAX_POOL_2D"]:
        return _build_pool2d_options(schema_tflite, op)
    if op.op_type == "RESIZE_NEAREST_NEIGHBOR":
        return _build_resize_nearest_options(schema_tflite, op)
    if op.op_type == "RESIZE_BILINEAR":
        return _build_resize_bilinear_options(schema_tflite, op)
    if op.op_type == "FULLY_CONNECTED":
        return _build_fully_connected_options(schema_tflite, op)
    if op.op_type in [
        "LOGISTIC",
        "RELU",
        "RELU6",
        "TANH",
        "EXP",
        "SQRT",
        "NEG",
        "PRELU",
        "DEQUANTIZE",
        "QUANTIZE",
        "PAD",
        "PADV2",
        "SLICE",
    ]:
        return _enum(schema_tflite, "BuiltinOptions", "NONE"), None
    raise NotImplementedError(
        f"BuiltinOptions mapping is not implemented for op_type={op.op_type}"
    )


def _build_operator_table(
    *,
    schema_tflite: Dict[str, Any],
    operators: List[OperatorIR],
    op_index_map: Dict[Tuple[str, int, str], int],
    tensor_index_map: Dict[str, int],
) -> List[object]:
    table: List[object] = []
    for op in operators:
        key = operator_code_key(op)
        if key not in op_index_map:
            raise KeyError(f"OperatorCode not found for op={op.op_type} version={op.version}")

        op_obj = schema_tflite["OperatorT"]()
        op_obj.opcodeIndex = int(op_index_map[key])
        op_obj.inputs = _require_tensor_indices(
            tensor_index_map=tensor_index_map,
            tensor_names=op.inputs,
            op_type=op.op_type,
            tensor_role="input",
        )
        op_obj.outputs = _require_tensor_indices(
            tensor_index_map=tensor_index_map,
            tensor_names=op.outputs,
            op_type=op.op_type,
            tensor_role="output",
        )

        builtin_options_type, builtin_options = _build_builtin_options(schema_tflite, op)
        op_obj.builtinOptionsType = int(builtin_options_type)
        op_obj.builtinOptions = builtin_options
        if op.op_type == "CUSTOM":
            custom_options = op.options.get("customOptions", b"")
            if isinstance(custom_options, str):
                custom_options = custom_options.encode("utf-8")
            op_obj.customOptions = bytes(custom_options)
            op_obj.customOptionsFormat = _enum(
                schema_tflite, "CustomOptionsFormat", "FLEXBUFFERS"
            )

        table.append(op_obj)
    return table


def build_model_object(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    with_signature_defs: bool = True,
) -> object:
    operator_codes, op_index_map = build_operator_codes(schema_tflite, model_ir.operators)
    tensors, buffers, tensor_index_map = build_tensors_and_buffers(schema_tflite, model_ir.tensors)
    operators = _build_operator_table(
        schema_tflite=schema_tflite,
        operators=model_ir.operators,
        op_index_map=op_index_map,
        tensor_index_map=tensor_index_map,
    )

    subgraph = schema_tflite["SubGraphT"]()
    subgraph.name = model_ir.name
    subgraph.tensors = tensors
    subgraph.operators = operators
    subgraph.inputs = _require_tensor_indices(
        tensor_index_map=tensor_index_map,
        tensor_names=model_ir.inputs,
        op_type="MODEL",
        tensor_role="graph input",
    )
    subgraph.outputs = _require_tensor_indices(
        tensor_index_map=tensor_index_map,
        tensor_names=model_ir.outputs,
        op_type="MODEL",
        tensor_role="graph output",
    )

    model = schema_tflite["ModelT"]()
    model.version = 3
    model.description = model_ir.description
    model.operatorCodes = operator_codes
    model.subgraphs = [subgraph]
    model.buffers = buffers
    if with_signature_defs:
        model.signatureDefs = build_signature_defs(
            schema_tflite=schema_tflite,
            tensor_index_map=tensor_index_map,
            input_names=model_ir.inputs,
            output_names=model_ir.outputs,
        )
    return model


def serialize_model(*, schema_tflite: Dict[str, Any], model_ir: ModelIR) -> bytes:
    sanitized_model_ir = _sanitize_model_ir_for_serialization(model_ir)
    model = build_model_object(
        schema_tflite=schema_tflite,
        model_ir=sanitized_model_ir,
        with_signature_defs=True,
    )
    builder = flatbuffers.Builder(0)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    return bytes(builder.Output())


def write_model_file(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    output_tflite_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_tflite_path) or ".", exist_ok=True)
    model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
    with open(output_tflite_path, "wb") as f:
        f.write(model_bytes)
    return output_tflite_path
