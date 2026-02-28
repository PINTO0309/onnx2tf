from __future__ import annotations

import os
import time
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
        if name is None:
            indices.append(-1)
            continue
        normalized_name = str(name)
        if normalized_name == "":
            indices.append(-1)
            continue
        if normalized_name not in tensor_index_map:
            raise KeyError(
                f"Tensor index is missing for {tensor_role}: name={normalized_name}, op={op_type}"
            )
        indices.append(tensor_index_map[normalized_name])
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
    #
    # NOTE:
    # We only mutate graph containers (operator/tensor membership) during
    # sanitization. Tensor payload arrays are read-only in serialization paths,
    # so avoid deep-copying full weight buffers here.
    sanitized_model_ir = ModelIR(
        name=model_ir.name,
        description=model_ir.description,
        tensors=dict(model_ir.tensors),
        operators=list(model_ir.operators),
        inputs=list(model_ir.inputs),
        outputs=list(model_ir.outputs),
        subgraphs=list(model_ir.subgraphs),
        metadata=dict(model_ir.metadata),
    )
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


def _build_maximum_minimum_options(
    schema_tflite: Dict[str, Any],
) -> Tuple[int, object]:
    options = schema_tflite["MaximumMinimumOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "MaximumMinimumOptions"), options


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


def _build_argmax_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ArgMaxOptionsT"]()
    output_type = str(op.options.get("outputType", "INT64")).upper()
    options.outputType = _enum(schema_tflite, "TensorType", output_type)
    return _enum(schema_tflite, "BuiltinOptions", "ArgMaxOptions"), options


def _build_argmin_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ArgMinOptionsT"]()
    output_type = str(op.options.get("outputType", "INT64")).upper()
    options.outputType = _enum(schema_tflite, "TensorType", output_type)
    return _enum(schema_tflite, "BuiltinOptions", "ArgMinOptions"), options


def _build_one_hot_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["OneHotOptionsT"]()
    if hasattr(options, "axis"):
        options.axis = int(op.options.get("axis", -1))
    return _enum(schema_tflite, "BuiltinOptions", "OneHotOptions"), options


def _build_cast_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["CastOptionsT"]()
    in_dtype = str(op.options.get("inDataType", "FLOAT32")).upper()
    out_dtype = str(op.options.get("outDataType", "FLOAT32")).upper()
    options.inDataType = _enum(schema_tflite, "TensorType", in_dtype)
    options.outDataType = _enum(schema_tflite, "TensorType", out_dtype)
    return _enum(schema_tflite, "BuiltinOptions", "CastOptions"), options


def _build_gather_nd_options(schema_tflite: Dict[str, Any], _op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["GatherNdOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "GatherNdOptions"), options


def _build_scatter_nd_options(schema_tflite: Dict[str, Any], _op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ScatterNdOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "ScatterNdOptions"), options


def _build_non_max_suppression_v4_options(
    schema_tflite: Dict[str, Any],
    _op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["NonMaxSuppressionV4OptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "NonMaxSuppressionV4Options"), options


def _build_non_max_suppression_v5_options(
    schema_tflite: Dict[str, Any],
    _op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["NonMaxSuppressionV5OptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "NonMaxSuppressionV5Options"), options


def _build_broadcast_to_options(schema_tflite: Dict[str, Any], _op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["BroadcastToOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "BroadcastToOptions"), options


def _build_floor_mod_options(schema_tflite: Dict[str, Any], _op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["FloorModOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "FloorModOptions"), options


def _build_tile_options(schema_tflite: Dict[str, Any], _op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["TileOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "TileOptions"), options


def _build_l2_norm_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["L2NormOptionsT"]()
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    return _enum(schema_tflite, "BuiltinOptions", "L2NormOptions"), options


def _build_lrn_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["LocalResponseNormalizationOptionsT"]()
    options.radius = int(op.options.get("radius", 0))
    options.bias = float(op.options.get("bias", 1.0))
    options.alpha = float(op.options.get("alpha", 0.0))
    options.beta = float(op.options.get("beta", 0.0))
    return _enum(schema_tflite, "BuiltinOptions", "LocalResponseNormalizationOptions"), options


def _build_space_to_depth_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["SpaceToDepthOptionsT"]()
    options.blockSize = int(op.options.get("blockSize", 1))
    return _enum(schema_tflite, "BuiltinOptions", "SpaceToDepthOptions"), options


def _build_depth_to_space_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["DepthToSpaceOptionsT"]()
    options.blockSize = int(op.options.get("blockSize", 1))
    return _enum(schema_tflite, "BuiltinOptions", "DepthToSpaceOptions"), options


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


def _build_transpose_conv_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["TransposeConvOptionsT"]()
    options.padding = _enum(schema_tflite, "Padding", str(op.options["padding"]))
    options.strideH = int(op.options["strideH"])
    options.strideW = int(op.options["strideW"])
    return _enum(schema_tflite, "BuiltinOptions", "TransposeConvOptions"), options


def _build_conv3d_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["Conv3DOptionsT"]()
    options.padding = _enum(schema_tflite, "Padding", str(op.options["padding"]))
    options.strideD = int(op.options["strideD"])
    options.strideH = int(op.options["strideH"])
    options.strideW = int(op.options["strideW"])
    options.dilationDFactor = int(op.options.get("dilationDFactor", 1))
    options.dilationHFactor = int(op.options.get("dilationHFactor", 1))
    options.dilationWFactor = int(op.options.get("dilationWFactor", 1))
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    if hasattr(options, "fusedActivationFunction"):
        options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    return _enum(schema_tflite, "BuiltinOptions", "Conv3DOptions"), options


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


def _build_leaky_relu_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["LeakyReluOptionsT"]()
    options.alpha = float(op.options.get("alpha", 0.01))
    return _enum(schema_tflite, "BuiltinOptions", "LeakyReluOptions"), options


def _build_hard_swish_options(schema_tflite: Dict[str, Any], _op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["HardSwishOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "HardSwishOptions"), options


def _build_shape_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ShapeOptionsT"]()
    out_type = str(op.options.get("outType", "INT32")).upper()
    if hasattr(options, "outType"):
        options.outType = _enum(schema_tflite, "TensorType", out_type)
    return _enum(schema_tflite, "BuiltinOptions", "ShapeOptions"), options


def _build_split_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["SplitOptionsT"]()
    options.numSplits = int(op.options.get("numSplits", len(op.outputs)))
    return _enum(schema_tflite, "BuiltinOptions", "SplitOptions"), options


def _build_expand_dims_options(schema_tflite: Dict[str, Any], _op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["ExpandDimsOptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "ExpandDimsOptions"), options


def _build_mirror_pad_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["MirrorPadOptionsT"]()
    mode = str(op.options.get("mode", "REFLECT")).upper()
    options.mode = _enum(schema_tflite, "MirrorPadMode", mode)
    return _enum(schema_tflite, "BuiltinOptions", "MirrorPadOptions"), options


def _build_cumsum_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["CumsumOptionsT"]()
    options.exclusive = bool(op.options.get("exclusive", False))
    options.reverse = bool(op.options.get("reverse", False))
    return _enum(schema_tflite, "BuiltinOptions", "CumsumOptions"), options


def _build_reverse_v2_options(
    schema_tflite: Dict[str, Any],
    _op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["ReverseV2OptionsT"]()
    return _enum(schema_tflite, "BuiltinOptions", "ReverseV2Options"), options


def _build_random_options(
    schema_tflite: Dict[str, Any],
    op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["RandomOptionsT"]()
    if hasattr(options, "seed"):
        options.seed = int(op.options.get("seed", 0))
    if hasattr(options, "seed2"):
        options.seed2 = int(op.options.get("seed2", 0))
    return _enum(schema_tflite, "BuiltinOptions", "RandomOptions"), options


def _build_bidirectional_sequence_lstm_options(
    schema_tflite: Dict[str, Any],
    op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["BidirectionalSequenceLSTMOptionsT"]()
    fused = str(op.options.get("fusedActivationFunction", "TANH"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    options.cellClip = float(op.options.get("cellClip", 0.0))
    options.projClip = float(op.options.get("projClip", 0.0))
    options.mergeOutputs = bool(op.options.get("mergeOutputs", True))
    options.timeMajor = bool(op.options.get("timeMajor", True))
    options.asymmetricQuantizeInputs = bool(
        op.options.get("asymmetricQuantizeInputs", False)
    )
    return _enum(schema_tflite, "BuiltinOptions", "BidirectionalSequenceLSTMOptions"), options


def _build_unidirectional_sequence_lstm_options(
    schema_tflite: Dict[str, Any],
    op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["UnidirectionalSequenceLSTMOptionsT"]()
    fused = str(op.options.get("fusedActivationFunction", "TANH"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    options.cellClip = float(op.options.get("cellClip", 0.0))
    options.projClip = float(op.options.get("projClip", 0.0))
    options.timeMajor = bool(op.options.get("timeMajor", True))
    options.asymmetricQuantizeInputs = bool(
        op.options.get("asymmetricQuantizeInputs", False)
    )
    if hasattr(options, "diagonalRecurrentTensors"):
        options.diagonalRecurrentTensors = bool(
            op.options.get("diagonalRecurrentTensors", False)
        )
    return _enum(schema_tflite, "BuiltinOptions", "UnidirectionalSequenceLSTMOptions"), options


def _build_sequence_rnn_options(
    schema_tflite: Dict[str, Any],
    op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["SequenceRNNOptionsT"]()
    options.timeMajor = bool(op.options.get("timeMajor", True))
    fused = str(op.options.get("fusedActivationFunction", "TANH"))
    options.fusedActivationFunction = _enum(schema_tflite, "ActivationFunctionType", fused)
    options.asymmetricQuantizeInputs = bool(
        op.options.get("asymmetricQuantizeInputs", False)
    )
    return _enum(schema_tflite, "BuiltinOptions", "SequenceRNNOptions"), options


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


def _build_batch_matmul_options(schema_tflite: Dict[str, Any], op: OperatorIR) -> Tuple[int, object]:
    options = schema_tflite["BatchMatMulOptionsT"]()
    options.adjX = bool(op.options.get("adjX", False))
    options.adjY = bool(op.options.get("adjY", False))
    options.asymmetricQuantizeInputs = bool(
        op.options.get("asymmetricQuantizeInputs", False)
    )
    return _enum(schema_tflite, "BuiltinOptions", "BatchMatMulOptions"), options


def _build_strided_slice_options(
    schema_tflite: Dict[str, Any],
    op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["StridedSliceOptionsT"]()
    options.beginMask = int(op.options.get("beginMask", 0))
    options.endMask = int(op.options.get("endMask", 0))
    options.ellipsisMask = int(op.options.get("ellipsisMask", 0))
    options.newAxisMask = int(op.options.get("newAxisMask", 0))
    options.shrinkAxisMask = int(op.options.get("shrinkAxisMask", 0))
    if hasattr(options, "offset"):
        options.offset = bool(op.options.get("offset", False))
    return _enum(schema_tflite, "BuiltinOptions", "StridedSliceOptions"), options


def _build_while_options(
    schema_tflite: Dict[str, Any],
    op: OperatorIR,
) -> Tuple[int, object]:
    options = schema_tflite["WhileOptionsT"]()
    options.condSubgraphIndex = int(op.options.get("condSubgraphIndex", 0))
    options.bodySubgraphIndex = int(op.options.get("bodySubgraphIndex", 0))
    return _enum(schema_tflite, "BuiltinOptions", "WhileOptions"), options


def _build_builtin_options(
    schema_tflite: Dict[str, Any],
    op: OperatorIR,
) -> Tuple[int, Optional[object]]:
    if op.op_type == "CUSTOM":
        return _enum(schema_tflite, "BuiltinOptions", "NONE"), None
    if op.op_type in ["ADD", "SUB", "MUL", "DIV"]:
        return _build_binary_options(schema_tflite, op)
    if op.op_type in ["MAXIMUM", "MINIMUM"]:
        return _build_maximum_minimum_options(schema_tflite)
    if op.op_type == "RESHAPE":
        return _build_reshape_options(schema_tflite, op)
    if op.op_type == "CONCATENATION":
        return _build_concat_options(schema_tflite, op)
    if op.op_type == "SOFTMAX":
        return _build_softmax_options(schema_tflite, op)
    if op.op_type == "TRANSPOSE":
        return _build_transpose_options(schema_tflite, op)
    if op.op_type in ["MEAN", "SUM", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD"]:
        return _build_reducer_options(schema_tflite, op)
    if op.op_type == "SQUEEZE":
        return _build_squeeze_options(schema_tflite, op)
    if op.op_type == "GATHER":
        return _build_gather_options(schema_tflite, op)
    if op.op_type == "ARG_MAX":
        return _build_argmax_options(schema_tflite, op)
    if op.op_type == "ARG_MIN":
        return _build_argmin_options(schema_tflite, op)
    if op.op_type == "ONE_HOT":
        return _build_one_hot_options(schema_tflite, op)
    if op.op_type == "CAST":
        return _build_cast_options(schema_tflite, op)
    if op.op_type == "GATHER_ND":
        return _build_gather_nd_options(schema_tflite, op)
    if op.op_type == "SCATTER_ND":
        return _build_scatter_nd_options(schema_tflite, op)
    if op.op_type == "NON_MAX_SUPPRESSION_V4":
        return _build_non_max_suppression_v4_options(schema_tflite, op)
    if op.op_type == "NON_MAX_SUPPRESSION_V5":
        return _build_non_max_suppression_v5_options(schema_tflite, op)
    if op.op_type == "BROADCAST_TO":
        return _build_broadcast_to_options(schema_tflite, op)
    if op.op_type == "FLOOR_MOD":
        return _build_floor_mod_options(schema_tflite, op)
    if op.op_type == "TILE":
        return _build_tile_options(schema_tflite, op)
    if op.op_type == "L2_NORMALIZATION":
        return _build_l2_norm_options(schema_tflite, op)
    if op.op_type == "LOCAL_RESPONSE_NORMALIZATION":
        return _build_lrn_options(schema_tflite, op)
    if op.op_type == "SPACE_TO_DEPTH":
        return _build_space_to_depth_options(schema_tflite, op)
    if op.op_type == "DEPTH_TO_SPACE":
        return _build_depth_to_space_options(schema_tflite, op)
    if op.op_type == "CONV_2D":
        return _build_conv_options(schema_tflite, op)
    if op.op_type == "DEPTHWISE_CONV_2D":
        return _build_depthwise_conv_options(schema_tflite, op)
    if op.op_type == "TRANSPOSE_CONV":
        return _build_transpose_conv_options(schema_tflite, op)
    if op.op_type in ["CONV_3D", "CONV_3D_TRANSPOSE"]:
        return _build_conv3d_options(schema_tflite, op)
    if op.op_type in ["AVERAGE_POOL_2D", "MAX_POOL_2D"]:
        return _build_pool2d_options(schema_tflite, op)
    if op.op_type == "RESIZE_NEAREST_NEIGHBOR":
        return _build_resize_nearest_options(schema_tflite, op)
    if op.op_type == "RESIZE_BILINEAR":
        return _build_resize_bilinear_options(schema_tflite, op)
    if op.op_type == "LEAKY_RELU":
        return _build_leaky_relu_options(schema_tflite, op)
    if op.op_type == "HARD_SWISH":
        return _build_hard_swish_options(schema_tflite, op)
    if op.op_type == "SHAPE":
        return _build_shape_options(schema_tflite, op)
    if op.op_type == "SPLIT":
        return _build_split_options(schema_tflite, op)
    if op.op_type == "EXPAND_DIMS":
        return _build_expand_dims_options(schema_tflite, op)
    if op.op_type == "MIRROR_PAD":
        return _build_mirror_pad_options(schema_tflite, op)
    if op.op_type == "CUMSUM":
        return _build_cumsum_options(schema_tflite, op)
    if op.op_type == "REVERSE_V2":
        return _build_reverse_v2_options(schema_tflite, op)
    if op.op_type == "RANDOM_STANDARD_NORMAL":
        return _build_random_options(schema_tflite, op)
    if op.op_type == "FULLY_CONNECTED":
        return _build_fully_connected_options(schema_tflite, op)
    if op.op_type == "BATCH_MATMUL":
        return _build_batch_matmul_options(schema_tflite, op)
    if op.op_type == "BIDIRECTIONAL_SEQUENCE_LSTM":
        return _build_bidirectional_sequence_lstm_options(schema_tflite, op)
    if op.op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
        return _build_unidirectional_sequence_lstm_options(schema_tflite, op)
    if op.op_type == "UNIDIRECTIONAL_SEQUENCE_RNN":
        return _build_sequence_rnn_options(schema_tflite, op)
    if op.op_type == "STRIDED_SLICE":
        return _build_strided_slice_options(schema_tflite, op)
    if op.op_type == "WHILE":
        return _build_while_options(schema_tflite, op)
    if op.op_type in [
        "LOGISTIC",
        "LOGICAL_AND",
        "LOGICAL_OR",
        "LOGICAL_NOT",
        "EQUAL",
        "NOT_EQUAL",
        "GREATER",
        "GREATER_EQUAL",
        "LESS",
        "LESS_EQUAL",
        "RELU",
        "RELU6",
        "RELU_N1_TO_1",
        "RELU_0_TO_1",
        "TANH",
        "ATAN2",
        "LOG",
        "EXP",
        "COS",
        "SIN",
        "SQRT",
        "ABS",
        "CEIL",
        "FLOOR",
        "ROUND",
        "SIGN",
        "ELU",
        "GELU",
        "NEG",
        "POW",
        "SELECT",
        "WHERE",
        "RANGE",
        "RIGHT_SHIFT",
        "BITWISE_XOR",
        "PRELU",
        "DEQUANTIZE",
        "QUANTIZE",
        "PAD",
        "PADV2",
        "TOPK_V2",
        "SLICE",
        "FILL",
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


def _build_subgraph_tensors_and_append_buffers(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    global_buffer_table: List[object],
) -> Tuple[List[object], Dict[str, int]]:
    tensors, local_buffers, tensor_index_map = build_tensors_and_buffers(
        schema_tflite=schema_tflite,
        tensors=model_ir.tensors,
    )
    if len(global_buffer_table) == 0:
        raise ValueError("Global buffer table must contain Buffer[0].")
    buffer_offset = int(len(global_buffer_table) - 1)
    for tensor in tensors:
        current_buffer = int(getattr(tensor, "buffer", 0))
        if current_buffer > 0:
            tensor.buffer = int(current_buffer + buffer_offset)
    global_buffer_table.extend(local_buffers[1:])
    return tensors, tensor_index_map


def _build_serialization_tables(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    with_signature_defs: bool = True,
) -> Dict[str, Any]:
    all_subgraphs: List[ModelIR] = [model_ir] + list(model_ir.subgraphs)
    all_operators: List[OperatorIR] = []
    for subgraph_ir in all_subgraphs:
        all_operators.extend(list(subgraph_ir.operators))
    operator_codes, op_index_map = build_operator_codes(schema_tflite, all_operators)

    buffers: List[object] = []
    empty_buffer = schema_tflite["BufferT"]()
    empty_buffer.data = bytes()
    buffers.append(empty_buffer)

    serialized_subgraphs: List[object] = []
    main_tensor_index_map: Optional[Dict[str, int]] = None
    for subgraph_index, subgraph_ir in enumerate(all_subgraphs):
        tensors, tensor_index_map = _build_subgraph_tensors_and_append_buffers(
            schema_tflite=schema_tflite,
            model_ir=subgraph_ir,
            global_buffer_table=buffers,
        )
        operators = _build_operator_table(
            schema_tflite=schema_tflite,
            operators=subgraph_ir.operators,
            op_index_map=op_index_map,
            tensor_index_map=tensor_index_map,
        )
        subgraph = schema_tflite["SubGraphT"]()
        subgraph.name = subgraph_ir.name
        subgraph.tensors = tensors
        subgraph.operators = operators
        subgraph.inputs = _require_tensor_indices(
            tensor_index_map=tensor_index_map,
            tensor_names=subgraph_ir.inputs,
            op_type="MODEL",
            tensor_role="graph input",
        )
        subgraph.outputs = _require_tensor_indices(
            tensor_index_map=tensor_index_map,
            tensor_names=subgraph_ir.outputs,
            op_type="MODEL",
            tensor_role="graph output",
        )
        serialized_subgraphs.append(subgraph)
        if int(subgraph_index) == 0:
            main_tensor_index_map = dict(tensor_index_map)

    if main_tensor_index_map is None:
        raise ValueError("Main subgraph tensor index map is missing.")

    signature_defs: List[object] = []
    if with_signature_defs:
        signature_defs = build_signature_defs(
            schema_tflite=schema_tflite,
            tensor_index_map=main_tensor_index_map,
            input_names=model_ir.inputs,
            output_names=model_ir.outputs,
        )

    return {
        "operator_codes": operator_codes,
        "subgraphs": serialized_subgraphs,
        "buffers": buffers,
        "signature_defs": signature_defs,
    }


def build_model_object(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    with_signature_defs: bool = True,
) -> object:
    tables = _build_serialization_tables(
        schema_tflite=schema_tflite,
        model_ir=model_ir,
        with_signature_defs=with_signature_defs,
    )

    model = schema_tflite["ModelT"]()
    model.version = 3
    model.description = model_ir.description
    model.operatorCodes = tables["operator_codes"]
    model.subgraphs = tables["subgraphs"]
    model.buffers = tables["buffers"]
    if with_signature_defs:
        model.signatureDefs = tables["signature_defs"]
    return model


def _pack_string(builder: flatbuffers.Builder, value: Optional[str]) -> int:
    if value is None:
        return 0
    return int(builder.CreateString(str(value)))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    return list(value)


def _pack_int32_vector(
    *,
    builder: flatbuffers.Builder,
    start_vector_fn: Any,
    values: List[int],
) -> int:
    if len(values) == 0:
        return 0
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependInt32(int(value))
    return int(builder.EndVector())


def _pack_int64_vector(
    *,
    builder: flatbuffers.Builder,
    start_vector_fn: Any,
    values: List[int],
) -> int:
    if len(values) == 0:
        return 0
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependInt64(int(value))
    return int(builder.EndVector())


def _pack_float32_vector(
    *,
    builder: flatbuffers.Builder,
    start_vector_fn: Any,
    values: List[float],
) -> int:
    if len(values) == 0:
        return 0
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependFloat32(float(value))
    return int(builder.EndVector())


def _pack_bool_vector(
    *,
    builder: flatbuffers.Builder,
    start_vector_fn: Any,
    values: List[bool],
) -> int:
    if len(values) == 0:
        return 0
    start_vector_fn(builder, len(values))
    for value in reversed(values):
        builder.PrependBool(bool(value))
    return int(builder.EndVector())


def _pack_bytes_vector(
    *,
    builder: flatbuffers.Builder,
    start_vector_fn: Any,
    values: bytes,
) -> int:
    if len(values) == 0:
        return 0
    # FlatBuffers Builder has a dedicated fast path for raw byte vectors.
    # Using it avoids Python-side per-byte loops for large constant buffers.
    return int(builder.CreateByteVector(bytes(values)))


def _pack_uoffset_vector(
    *,
    builder: flatbuffers.Builder,
    start_vector_fn: Any,
    offsets: List[int],
) -> int:
    if len(offsets) == 0:
        return 0
    start_vector_fn(builder, len(offsets))
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(int(offset))
    return int(builder.EndVector())


def _pack_quantization_parameters(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    quantization: Optional[object],
) -> int:
    if quantization is None:
        return 0

    min_offset = _pack_float32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["QuantizationParametersStartMinVector"],
        values=[float(v) for v in _as_list(getattr(quantization, "min", None))],
    )
    max_offset = _pack_float32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["QuantizationParametersStartMaxVector"],
        values=[float(v) for v in _as_list(getattr(quantization, "max", None))],
    )
    scale_offset = _pack_float32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["QuantizationParametersStartScaleVector"],
        values=[float(v) for v in _as_list(getattr(quantization, "scale", None))],
    )
    zero_point_offset = _pack_int64_vector(
        builder=builder,
        start_vector_fn=schema_tflite["QuantizationParametersStartZeroPointVector"],
        values=[int(v) for v in _as_list(getattr(quantization, "zeroPoint", None))],
    )

    schema_tflite["QuantizationParametersStart"](builder)
    if min_offset > 0:
        schema_tflite["QuantizationParametersAddMin"](builder, min_offset)
    if max_offset > 0:
        schema_tflite["QuantizationParametersAddMax"](builder, max_offset)
    if scale_offset > 0:
        schema_tflite["QuantizationParametersAddScale"](builder, scale_offset)
    if zero_point_offset > 0:
        schema_tflite["QuantizationParametersAddZeroPoint"](builder, zero_point_offset)
    schema_tflite["QuantizationParametersAddQuantizedDimension"](
        builder,
        int(getattr(quantization, "quantizedDimension", 0)),
    )
    return int(schema_tflite["QuantizationParametersEnd"](builder))


def _pack_tensor(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    tensor: object,
) -> int:
    shape_offset = _pack_int32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["TensorStartShapeVector"],
        values=[int(v) for v in _as_list(getattr(tensor, "shape", None))],
    )
    shape_signature_offset = _pack_int32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["TensorStartShapeSignatureVector"],
        values=[int(v) for v in _as_list(getattr(tensor, "shapeSignature", None))],
    )
    name_offset = _pack_string(builder, getattr(tensor, "name", None))
    quantization_offset = _pack_quantization_parameters(
        schema_tflite=schema_tflite,
        builder=builder,
        quantization=getattr(tensor, "quantization", None),
    )

    schema_tflite["TensorStart"](builder)
    if shape_offset > 0:
        schema_tflite["TensorAddShape"](builder, shape_offset)
    schema_tflite["TensorAddType"](builder, int(getattr(tensor, "type", 0)))
    schema_tflite["TensorAddBuffer"](builder, int(getattr(tensor, "buffer", 0)))
    if name_offset > 0:
        schema_tflite["TensorAddName"](builder, name_offset)
    if quantization_offset > 0:
        schema_tflite["TensorAddQuantization"](builder, quantization_offset)
    schema_tflite["TensorAddIsVariable"](
        builder,
        bool(getattr(tensor, "isVariable", False)),
    )
    if shape_signature_offset > 0:
        schema_tflite["TensorAddShapeSignature"](builder, shape_signature_offset)
    return int(schema_tflite["TensorEnd"](builder))


def _pack_operator(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    operator: object,
) -> int:
    inputs_offset = _pack_int32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["OperatorStartInputsVector"],
        values=[int(v) for v in _as_list(getattr(operator, "inputs", None))],
    )
    outputs_offset = _pack_int32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["OperatorStartOutputsVector"],
        values=[int(v) for v in _as_list(getattr(operator, "outputs", None))],
    )
    intermediates_offset = _pack_int32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["OperatorStartIntermediatesVector"],
        values=[int(v) for v in _as_list(getattr(operator, "intermediates", None))],
    )
    mutating_inputs_offset = _pack_bool_vector(
        builder=builder,
        start_vector_fn=schema_tflite["OperatorStartMutatingVariableInputsVector"],
        values=[
            bool(v)
            for v in _as_list(getattr(operator, "mutatingVariableInputs", None))
        ],
    )

    custom_options = getattr(operator, "customOptions", b"")
    if custom_options is None:
        custom_options = b""
    if isinstance(custom_options, str):
        custom_options = custom_options.encode("utf-8")
    custom_options_offset = _pack_bytes_vector(
        builder=builder,
        start_vector_fn=schema_tflite["OperatorStartCustomOptionsVector"],
        values=bytes(custom_options),
    )

    builtin_options_offset = 0
    builtin_options = getattr(operator, "builtinOptions", None)
    if builtin_options is not None and hasattr(builtin_options, "Pack"):
        builtin_options_offset = int(builtin_options.Pack(builder))

    builtin_options2_offset = 0
    builtin_options2 = getattr(operator, "builtinOptions2", None)
    if builtin_options2 is not None and hasattr(builtin_options2, "Pack"):
        builtin_options2_offset = int(builtin_options2.Pack(builder))

    schema_tflite["OperatorStart"](builder)
    schema_tflite["OperatorAddOpcodeIndex"](builder, int(getattr(operator, "opcodeIndex", 0)))
    if inputs_offset > 0:
        schema_tflite["OperatorAddInputs"](builder, inputs_offset)
    if outputs_offset > 0:
        schema_tflite["OperatorAddOutputs"](builder, outputs_offset)
    schema_tflite["OperatorAddBuiltinOptionsType"](
        builder,
        int(getattr(operator, "builtinOptionsType", 0)),
    )
    if builtin_options_offset > 0:
        schema_tflite["OperatorAddBuiltinOptions"](builder, builtin_options_offset)
    if custom_options_offset > 0:
        schema_tflite["OperatorAddCustomOptions"](builder, custom_options_offset)
    schema_tflite["OperatorAddCustomOptionsFormat"](
        builder,
        int(getattr(operator, "customOptionsFormat", 0)),
    )
    if mutating_inputs_offset > 0:
        schema_tflite["OperatorAddMutatingVariableInputs"](
            builder,
            mutating_inputs_offset,
        )
    if intermediates_offset > 0:
        schema_tflite["OperatorAddIntermediates"](builder, intermediates_offset)
    large_custom_offset = int(getattr(operator, "largeCustomOptionsOffset", 0))
    large_custom_size = int(getattr(operator, "largeCustomOptionsSize", 0))
    if large_custom_offset > 0:
        schema_tflite["OperatorAddLargeCustomOptionsOffset"](builder, large_custom_offset)
    if large_custom_size > 0:
        schema_tflite["OperatorAddLargeCustomOptionsSize"](builder, large_custom_size)
    schema_tflite["OperatorAddBuiltinOptions2Type"](
        builder,
        int(getattr(operator, "builtinOptions2Type", 0)),
    )
    if builtin_options2_offset > 0:
        schema_tflite["OperatorAddBuiltinOptions2"](builder, builtin_options2_offset)
    debug_metadata_index = int(getattr(operator, "debugMetadataIndex", -1))
    if debug_metadata_index >= 0:
        schema_tflite["OperatorAddDebugMetadataIndex"](builder, debug_metadata_index)
    return int(schema_tflite["OperatorEnd"](builder))


def _pack_subgraph(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    subgraph: object,
) -> int:
    tensor_offsets = [
        _pack_tensor(schema_tflite=schema_tflite, builder=builder, tensor=tensor)
        for tensor in _as_list(getattr(subgraph, "tensors", None))
    ]
    operator_offsets = [
        _pack_operator(schema_tflite=schema_tflite, builder=builder, operator=operator)
        for operator in _as_list(getattr(subgraph, "operators", None))
    ]

    tensors_offset = _pack_uoffset_vector(
        builder=builder,
        start_vector_fn=schema_tflite["SubGraphStartTensorsVector"],
        offsets=tensor_offsets,
    )
    inputs_offset = _pack_int32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["SubGraphStartInputsVector"],
        values=[int(v) for v in _as_list(getattr(subgraph, "inputs", None))],
    )
    outputs_offset = _pack_int32_vector(
        builder=builder,
        start_vector_fn=schema_tflite["SubGraphStartOutputsVector"],
        values=[int(v) for v in _as_list(getattr(subgraph, "outputs", None))],
    )
    operators_offset = _pack_uoffset_vector(
        builder=builder,
        start_vector_fn=schema_tflite["SubGraphStartOperatorsVector"],
        offsets=operator_offsets,
    )
    name_offset = _pack_string(builder, getattr(subgraph, "name", None))

    schema_tflite["SubGraphStart"](builder)
    if tensors_offset > 0:
        schema_tflite["SubGraphAddTensors"](builder, tensors_offset)
    if inputs_offset > 0:
        schema_tflite["SubGraphAddInputs"](builder, inputs_offset)
    if outputs_offset > 0:
        schema_tflite["SubGraphAddOutputs"](builder, outputs_offset)
    if operators_offset > 0:
        schema_tflite["SubGraphAddOperators"](builder, operators_offset)
    if name_offset > 0:
        schema_tflite["SubGraphAddName"](builder, name_offset)
    debug_metadata_index = int(getattr(subgraph, "debugMetadataIndex", -1))
    if debug_metadata_index >= 0:
        schema_tflite["SubGraphAddDebugMetadataIndex"](builder, debug_metadata_index)
    return int(schema_tflite["SubGraphEnd"](builder))


def _pack_operator_code(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    operator_code: object,
) -> int:
    custom_code_value = getattr(operator_code, "customCode", None)
    custom_code_offset = 0
    if custom_code_value is not None and str(custom_code_value) != "":
        custom_code_offset = _pack_string(builder, str(custom_code_value))

    schema_tflite["OperatorCodeStart"](builder)
    schema_tflite["OperatorCodeAddDeprecatedBuiltinCode"](
        builder,
        int(getattr(operator_code, "deprecatedBuiltinCode", 0)),
    )
    if custom_code_offset > 0:
        schema_tflite["OperatorCodeAddCustomCode"](builder, custom_code_offset)
    schema_tflite["OperatorCodeAddVersion"](
        builder,
        int(getattr(operator_code, "version", 1)),
    )
    schema_tflite["OperatorCodeAddBuiltinCode"](
        builder,
        int(getattr(operator_code, "builtinCode", 0)),
    )
    return int(schema_tflite["OperatorCodeEnd"](builder))


def _pack_buffer(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    buffer_obj: object,
) -> int:
    data = getattr(buffer_obj, "data", b"")
    if isinstance(data, str):
        data = data.encode("utf-8")
    data_offset = _pack_bytes_vector(
        builder=builder,
        start_vector_fn=schema_tflite["BufferStartDataVector"],
        values=bytes(data),
    )

    schema_tflite["BufferStart"](builder)
    if data_offset > 0:
        schema_tflite["BufferAddData"](builder, data_offset)
    size_value = int(getattr(buffer_obj, "size", 0))
    offset_value = int(getattr(buffer_obj, "offset", 0))
    if offset_value > 0:
        schema_tflite["BufferAddOffset"](builder, offset_value)
    if size_value > 0:
        schema_tflite["BufferAddSize"](builder, size_value)
    return int(schema_tflite["BufferEnd"](builder))


def _pack_tensor_map(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    tensor_map: object,
) -> int:
    name_offset = _pack_string(builder, getattr(tensor_map, "name", None))
    schema_tflite["TensorMapStart"](builder)
    if name_offset > 0:
        schema_tflite["TensorMapAddName"](builder, name_offset)
    schema_tflite["TensorMapAddTensorIndex"](
        builder,
        int(getattr(tensor_map, "tensorIndex", 0)),
    )
    return int(schema_tflite["TensorMapEnd"](builder))


def _pack_signature_def(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    signature_def: object,
) -> int:
    input_offsets = [
        _pack_tensor_map(schema_tflite=schema_tflite, builder=builder, tensor_map=tm)
        for tm in _as_list(getattr(signature_def, "inputs", None))
    ]
    output_offsets = [
        _pack_tensor_map(schema_tflite=schema_tflite, builder=builder, tensor_map=tm)
        for tm in _as_list(getattr(signature_def, "outputs", None))
    ]
    inputs_offset = _pack_uoffset_vector(
        builder=builder,
        start_vector_fn=schema_tflite["SignatureDefStartInputsVector"],
        offsets=input_offsets,
    )
    outputs_offset = _pack_uoffset_vector(
        builder=builder,
        start_vector_fn=schema_tflite["SignatureDefStartOutputsVector"],
        offsets=output_offsets,
    )
    signature_key_offset = _pack_string(builder, getattr(signature_def, "signatureKey", None))

    schema_tflite["SignatureDefStart"](builder)
    if inputs_offset > 0:
        schema_tflite["SignatureDefAddInputs"](builder, inputs_offset)
    if outputs_offset > 0:
        schema_tflite["SignatureDefAddOutputs"](builder, outputs_offset)
    if signature_key_offset > 0:
        schema_tflite["SignatureDefAddSignatureKey"](builder, signature_key_offset)
    schema_tflite["SignatureDefAddSubgraphIndex"](
        builder,
        int(getattr(signature_def, "subgraphIndex", 0)),
    )
    return int(schema_tflite["SignatureDefEnd"](builder))


def _pack_model(
    *,
    schema_tflite: Dict[str, Any],
    builder: flatbuffers.Builder,
    model_description: str,
    operator_codes: List[object],
    subgraphs: List[object],
    buffers: List[object],
    signature_defs: List[object],
) -> int:
    operator_code_offsets = [
        _pack_operator_code(
            schema_tflite=schema_tflite,
            builder=builder,
            operator_code=operator_code,
        )
        for operator_code in operator_codes
    ]
    subgraph_offsets = [
        _pack_subgraph(
            schema_tflite=schema_tflite,
            builder=builder,
            subgraph=subgraph,
        )
        for subgraph in subgraphs
    ]
    buffer_offsets = [
        _pack_buffer(
            schema_tflite=schema_tflite,
            builder=builder,
            buffer_obj=buffer_obj,
        )
        for buffer_obj in buffers
    ]
    signature_def_offsets = [
        _pack_signature_def(
            schema_tflite=schema_tflite,
            builder=builder,
            signature_def=signature_def,
        )
        for signature_def in signature_defs
    ]

    operator_codes_offset = _pack_uoffset_vector(
        builder=builder,
        start_vector_fn=schema_tflite["ModelStartOperatorCodesVector"],
        offsets=operator_code_offsets,
    )
    subgraphs_offset = _pack_uoffset_vector(
        builder=builder,
        start_vector_fn=schema_tflite["ModelStartSubgraphsVector"],
        offsets=subgraph_offsets,
    )
    description_offset = _pack_string(builder, model_description)
    buffers_offset = _pack_uoffset_vector(
        builder=builder,
        start_vector_fn=schema_tflite["ModelStartBuffersVector"],
        offsets=buffer_offsets,
    )
    signature_defs_offset = _pack_uoffset_vector(
        builder=builder,
        start_vector_fn=schema_tflite["ModelStartSignatureDefsVector"],
        offsets=signature_def_offsets,
    )

    schema_tflite["ModelStart"](builder)
    schema_tflite["ModelAddVersion"](builder, 3)
    if operator_codes_offset > 0:
        schema_tflite["ModelAddOperatorCodes"](builder, operator_codes_offset)
    if subgraphs_offset > 0:
        schema_tflite["ModelAddSubgraphs"](builder, subgraphs_offset)
    if description_offset > 0:
        schema_tflite["ModelAddDescription"](builder, description_offset)
    if buffers_offset > 0:
        schema_tflite["ModelAddBuffers"](builder, buffers_offset)
    if signature_defs_offset > 0:
        schema_tflite["ModelAddSignatureDefs"](builder, signature_defs_offset)
    return int(schema_tflite["ModelEnd"](builder))


def _serialize_model_with_object_pack(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    timing: Optional[Dict[str, Any]] = None,
) -> bytearray:
    t0 = time.perf_counter()
    sanitized_model_ir = _sanitize_model_ir_for_serialization(model_ir)
    t1 = time.perf_counter()
    model = build_model_object(
        schema_tflite=schema_tflite,
        model_ir=sanitized_model_ir,
        with_signature_defs=True,
    )
    t2 = time.perf_counter()
    builder = flatbuffers.Builder(0)
    model_offset = model.Pack(builder)
    builder.Finish(model_offset, file_identifier=b"TFL3")
    t3 = time.perf_counter()
    model_bytes = builder.Output()
    t4 = time.perf_counter()
    if timing is not None:
        timing["serializer_mode"] = "object_pack"
        timing["sanitize_model_ir_sec"] = float(t1 - t0)
        timing["build_model_object_sec"] = float(t2 - t1)
        timing["pack_builder_sec"] = float(t3 - t2)
        timing["output_buffer_sec"] = float(t4 - t3)
        timing["serialize_total_sec"] = float(t4 - t0)
        timing["model_bytes"] = float(len(model_bytes))
    return model_bytes


def _serialize_model_with_direct_builder(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    timing: Optional[Dict[str, Any]] = None,
) -> bytearray:
    t0 = time.perf_counter()
    sanitized_model_ir = _sanitize_model_ir_for_serialization(model_ir)
    t1 = time.perf_counter()
    tables = _build_serialization_tables(
        schema_tflite=schema_tflite,
        model_ir=sanitized_model_ir,
        with_signature_defs=True,
    )
    t2 = time.perf_counter()
    builder = flatbuffers.Builder(0)
    model_offset = _pack_model(
        schema_tflite=schema_tflite,
        builder=builder,
        model_description=sanitized_model_ir.description,
        operator_codes=tables["operator_codes"],
        subgraphs=tables["subgraphs"],
        buffers=tables["buffers"],
        signature_defs=tables["signature_defs"],
    )
    builder.Finish(model_offset, file_identifier=b"TFL3")
    t3 = time.perf_counter()
    model_bytes = builder.Output()
    t4 = time.perf_counter()
    if timing is not None:
        timing["serializer_mode"] = "builder_direct"
        timing["sanitize_model_ir_sec"] = float(t1 - t0)
        timing["build_model_object_sec"] = float(t2 - t1)
        timing["pack_builder_sec"] = float(t3 - t2)
        timing["output_buffer_sec"] = float(t4 - t3)
        timing["serialize_total_sec"] = float(t4 - t0)
        timing["model_bytes"] = float(len(model_bytes))
    return model_bytes


def _is_enabled_env(var_name: str, default: bool) -> bool:
    value = os.environ.get(var_name, "")
    if str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() not in {"0", "false", "off", "no"}


def serialize_model(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    timing: Optional[Dict[str, Any]] = None,
) -> bytearray:
    serializer_mode = str(
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_SERIALIZER", "builder_direct")
    ).strip().lower()
    prefer_direct_builder = serializer_mode in {
        "",
        "builder_direct",
        "builder",
        "direct",
    }
    if serializer_mode in {"object_pack", "pack"}:
        prefer_direct_builder = False
    if serializer_mode not in {
        "",
        "builder_direct",
        "builder",
        "direct",
        "object_pack",
        "pack",
    }:
        raise ValueError(
            "Unsupported ONNX2TF_FLATBUFFER_DIRECT_SERIALIZER value. "
            f"got: {serializer_mode}"
        )

    if not prefer_direct_builder:
        return _serialize_model_with_object_pack(
            schema_tflite=schema_tflite,
            model_ir=model_ir,
            timing=timing,
        )

    try:
        return _serialize_model_with_direct_builder(
            schema_tflite=schema_tflite,
            model_ir=model_ir,
            timing=timing,
        )
    except Exception as ex:
        allow_fallback = _is_enabled_env(
            "ONNX2TF_FLATBUFFER_DIRECT_SERIALIZER_FALLBACK_TO_OBJECT_PACK",
            True,
        )
        if not allow_fallback:
            raise
        fallback_error = f"{type(ex).__name__}: {ex}"
        model_bytes = _serialize_model_with_object_pack(
            schema_tflite=schema_tflite,
            model_ir=model_ir,
            timing=timing,
        )
        if timing is not None:
            timing["serializer_mode"] = "object_pack_fallback_from_builder_direct"
            timing["builder_direct_error"] = fallback_error
        return model_bytes


def write_model_file(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    output_tflite_path: str,
    timing: Optional[Dict[str, Any]] = None,
) -> str:
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(output_tflite_path) or ".", exist_ok=True)
    t1 = time.perf_counter()
    model_bytes = serialize_model(
        schema_tflite=schema_tflite,
        model_ir=model_ir,
        timing=timing,
    )
    t2 = time.perf_counter()
    with open(output_tflite_path, "wb") as f:
        f.write(model_bytes)
    t3 = time.perf_counter()
    if timing is not None:
        timing["mkdir_sec"] = float(t1 - t0)
        timing["serialize_call_sec"] = float(t2 - t1)
        timing["file_write_sec"] = float(t3 - t2)
        timing["write_model_file_total_sec"] = float(t3 - t0)
        timing["output_tflite_path"] = str(output_tflite_path)
    return output_tflite_path
