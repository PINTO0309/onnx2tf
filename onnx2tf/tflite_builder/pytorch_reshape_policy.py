from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _shape_lists_equal_relaxed,
)
from onnx2tf.tflite_builder.pytorch_graph_policy import (
    _resolve_channel_first_named_tensor_shape_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_indexing_codegen import (
    _reshape_is_plain_singleton_axis_drop,
)
from onnx2tf.tflite_builder.pytorch_shape_policy import (
    _reshape_special_layout_plan,
)


def _reshape_codegen_is_plain_data_only_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    infer_effective_rank4_runtime_layout_fn: Callable[[str], Optional[str]],
    reshape_preserves_channel_last_sequence_fn: Callable[
        [Sequence[int], Sequence[int], str], Optional[List[int]]
    ],
    reshape_prefers_feature_last_for_adjx_batch_matmul_fn: Callable[
        [str, str], Optional[Tuple[List[int], List[int]]]
    ],
) -> bool:
    if (
        str(op.op_type) != "RESHAPE"
        or len(op.inputs) == 0
        or len(op.outputs) == 0
    ):
        return False
    input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    if input_tensor is None or output_tensor is None:
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    input_layout = str(input_tensor.logical_layout)
    output_layout = str(output_tensor.logical_layout)
    reshape_is_lowered_onnx_flatten = "onnxFlattenAxis" in op.options
    if (
        not reshape_is_lowered_onnx_flatten
        and len(input_shape) == 4
        and len(output_shape) in {2, 3}
    ):
        effective_layout = infer_effective_rank4_runtime_layout_fn(
            str(op.inputs[0])
        )
        if effective_layout is not None:
            input_layout = effective_layout
    reshape_plain_singleton_axis_drop = _reshape_is_plain_singleton_axis_drop(
        input_shape,
        output_shape,
    )
    if reshape_is_lowered_onnx_flatten or reshape_plain_singleton_axis_drop:
        return True
    reshape_special_plan = _reshape_special_layout_plan(
        input_shape=input_shape,
        output_shape=output_shape,
        input_layout=input_layout,
        output_layout=output_layout,
    )
    reshape_pre_perm = reshape_preserves_channel_last_sequence_fn(
        input_shape,
        output_shape,
        input_layout,
    )
    if (
        reshape_special_plan is not None
        and reshape_special_plan.get("pre_perm", None) is not None
    ):
        reshape_pre_perm = list(reshape_special_plan["pre_perm"])
    reshape_feature_last_target = (
        reshape_prefers_feature_last_for_adjx_batch_matmul_fn(
            str(op.inputs[0]),
            str(op.outputs[0]),
        )
    )
    if reshape_feature_last_target is not None:
        reshape_pre_perm = list(reshape_feature_last_target[0])
    reshape_channel_first_alias_shape = (
        len(input_shape) == 2
        and len(output_shape) == 4
        and is_channel_last_logical_layout(
            normalize_logical_layout(output_tensor.logical_layout)
        )
        and all(int(dim) == 1 for dim in list(output_shape[1:-1]))
        and _shape_lists_equal_relaxed(
            input_shape,
            [int(output_shape[0]), int(output_shape[-1])],
        )
    )
    return not bool(
        reshape_special_plan is not None
        or reshape_pre_perm is not None
        or reshape_feature_last_target is not None
        or reshape_channel_first_alias_shape
    )


def _tensor_exact_static_shape_list_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    if (
        tensor.shape_signature is not None
        and len(list(tensor.shape_signature)) == len(list(tensor.shape))
    ):
        signature = [int(v) for v in list(tensor.shape_signature)]
        if all(int(v) > 0 for v in signature):
            return _resolve_channel_first_named_tensor_shape_for_codegen(
                model_ir=model_ir,
                tensor_name=str(tensor_name),
                preferred=signature,
                logical_layout=str(tensor.logical_layout),
            )
    return None


def _static_sequence_length_for_model_ir(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[int]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    if len(list(tensor.shape)) >= 1 and int(tensor.shape[0]) > 0:
        return int(tensor.shape[0])
    if (
        tensor.shape_signature is not None
        and len(list(tensor.shape_signature)) >= 1
        and int(tensor.shape_signature[0]) > 0
    ):
        return int(tensor.shape_signature[0])
    return None


def _reshape_prefers_feature_last_for_adjx_batch_matmul_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    input_tensor_name: str,
    output_name: str,
) -> Optional[Tuple[List[int], List[int]]]:
    input_tensor = model_ir.tensors.get(str(input_tensor_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if input_tensor is None or output_tensor is None:
        return None
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    preferred_pre_perm: Optional[List[int]] = None
    preferred_shape: Optional[List[int]] = None
    if (
        len(input_shape) == 3
        and len(output_shape) == 3
        and input_layout == "NCW"
        and output_layout == "NCW"
        and input_shape[0] == 1
        and input_shape[1] > 1
        and input_shape[2] > 1
        and output_shape[0] == input_shape[2]
        and output_shape[1] == 1
        and output_shape[2] == input_shape[1]
    ):
        preferred_pre_perm = [0, 1, 2]
        preferred_shape = [
            int(output_shape[0]),
            int(output_shape[2]),
            int(output_shape[1]),
        ]
    elif (
        len(input_shape) == 4
        and len(output_shape) == 3
        and input_layout == "NCHW"
        and output_layout == "NCW"
        and output_shape[0] == input_shape[0]
        and int(np.prod(input_shape, dtype=np.int64))
        == int(np.prod(output_shape, dtype=np.int64))
    ):
        preferred_pre_perm = [0, 3, 1, 2]
        preferred_shape = [
            int(input_shape[0]),
            int(input_shape[3]),
            int(input_shape[1]) * int(input_shape[2]),
        ]
    if preferred_pre_perm is not None and preferred_shape is not None:
        pending_outputs: List[str] = [str(output_name)]
        visited_outputs: Set[str] = set()
        while pending_outputs:
            current_name = pending_outputs.pop()
            if current_name in visited_outputs:
                continue
            visited_outputs.add(current_name)
            for consumer_idx in consumer_index.get(current_name, []):
                consumer_op = model_ir.operators[int(consumer_idx)]
                consumer_type = str(consumer_op.op_type)
                if consumer_type == "BATCH_MATMUL":
                    if (
                        len(consumer_op.inputs) < 2
                        or str(consumer_op.inputs[0]) != current_name
                    ):
                        continue
                    if not bool(consumer_op.options.get("adjX", False)):
                        continue
                    rhs_tensor = model_ir.tensors.get(
                        str(consumer_op.inputs[1]),
                        None,
                    )
                    if rhs_tensor is None or len(list(rhs_tensor.shape)) < 2:
                        continue
                    rhs_contract = int(list(rhs_tensor.shape)[-2])
                    expected_contract = (
                        int(input_shape[1])
                        if len(input_shape) == 3
                        else int(preferred_shape[1])
                    )
                    if rhs_contract != expected_contract:
                        continue
                    return (list(preferred_pre_perm), list(preferred_shape))
                if consumer_type in {
                    "ABS",
                    "ATAN",
                    "CAST",
                    "ELU",
                    "ERF",
                    "EXP",
                    "GELU",
                    "IDENTITY",
                    "LEAKY_RELU",
                    "LOG",
                    "LOGISTIC",
                    "NEG",
                    "RELU",
                    "RELU6",
                    "RELU_0_TO_1",
                    "RELU_N1_TO_1",
                    "SIGN",
                    "SIN",
                    "SQRT",
                    "SQUARE",
                    "TANH",
                }:
                    if (
                        len(consumer_op.outputs) == 1
                        and len(consumer_op.inputs) >= 1
                        and str(consumer_op.inputs[0]) == current_name
                    ):
                        pending_outputs.append(str(consumer_op.outputs[0]))
        return None
    return None
