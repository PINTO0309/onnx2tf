from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    preflight_required_op_types,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR
from onnx2tf.tflite_builder.passes.nhwc_concat_layout import (
    _NhwcConcatInputPlan,
    _apply_dequantize_input_plan,
    _apply_leaky_input_plan,
    _apply_prelu_input_plan,
    _apply_softmax_input_plan,
    _apply_slice_input_plan,
    _apply_swish_input_plan,
    _resolve_dequantize_input_plan,
    _resolve_leaky_input_plan,
    _resolve_prelu_input_plan,
    _resolve_softmax_input_plan,
    _resolve_slice_input_plan,
    _resolve_swish_input_plan,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_pad import (
    NhwcConcatPadPlan,
    apply_nhwc_concat_pad_plan,
    resolve_nhwc_concat_pad_plan,
)


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
_DIRECT_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_direct_chains"
)
_UNARY_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_unary_chains"
)
_PAD_STATS_KEY = "optimized_transpose_pre_concat_nhwc_quantized_pad_chains"
_UNARY_PAD_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_unary_pad_chains"
)
_ALL_PAD_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_all_pad_chains"
)
_SWISH_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_swish_chains"
)
_DEQUANTIZE_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_dequantize_chains"
)
_PRELU_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_prelu_chains"
)
_SOFTMAX_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_softmax_chains"
)
_LEAKY_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_leaky_chains"
)
_SLICE_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_quantized_slice_chains"
)
_UNARY_OPS = {"RELU", "RELU6", "LOGISTIC", "TANH", "GELU"}


@dataclass(frozen=True)
class _QuantizedInputPlan:
    kind: str
    adapter_op: OperatorIR
    concat_input: str
    source_name: str
    remove_adapter: bool
    unary_op: Optional[OperatorIR] = None
    pad_plan: Optional[NhwcConcatPadPlan] = None
    nhwc_plan: Optional[_NhwcConcatInputPlan] = None


@dataclass(frozen=True)
class _QuantizedConcatCandidate:
    input_plans: Tuple[_QuantizedInputPlan, ...]
    concat_op: OperatorIR
    concat_output: str
    quantize_op: OperatorIR
    quantized_output: str
    post_ops: Tuple[OperatorIR, ...]
    post_outputs: Tuple[str, ...]


def _clone_permuted_quantization(
    quantization: Any,
    permutation: List[int],
) -> Any:
    cloned = _clone_quantization(quantization)
    if isinstance(cloned, QuantParamIR):
        old_dimension = int(cloned.quantized_dimension)
        if 0 <= old_dimension < len(permutation):
            cloned.quantized_dimension = int(permutation.index(old_dimension))
    elif isinstance(cloned, dict) and "quantized_dimension" in cloned:
        old_dimension = int(cloned["quantized_dimension"])
        if 0 <= old_dimension < len(permutation):
            cloned["quantized_dimension"] = int(
                permutation.index(old_dimension)
            )
    return cloned


def _resolve_direct_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    adapter_op = graph_index.producer(concat_input)
    if (
        adapter_op is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != concat_input
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
    ):
        return None
    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    input_consumers = set(graph_index.consumer_indices(concat_input))
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or int(concat_index) not in input_consumers
    ):
        return None
    return _QuantizedInputPlan(
        kind="direct",
        adapter_op=adapter_op,
        concat_input=concat_input,
        source_name=source_name,
        remove_adapter=(
            input_consumers == {int(concat_index)}
            and concat_input not in model_outputs
        ),
    )


def _resolve_unary_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    unary_op = graph_index.producer(concat_input)
    unary_index = (
        None if unary_op is None else graph_index.operator_index(unary_op)
    )
    if (
        unary_op is None
        or unary_index is None
        or str(unary_op.op_type) not in _UNARY_OPS
        or len(unary_op.inputs) != 1
        or len(unary_op.outputs) != 1
        or str(unary_op.outputs[0]) != concat_input
        or concat_input in model_outputs
        or set(graph_index.consumer_indices(concat_input))
        != {int(concat_index)}
    ):
        return None
    adapter_output = str(unary_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output in model_outputs
        or set(graph_index.consumer_indices(adapter_output))
        != {int(unary_index)}
    ):
        return None
    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(concat_input)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
    ):
        return None
    return _QuantizedInputPlan(
        kind="unary",
        adapter_op=adapter_op,
        unary_op=unary_op,
        concat_input=concat_input,
        source_name=source_name,
        remove_adapter=True,
    )


def _resolve_pad_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_QuantizedInputPlan]:
    pad_plan = resolve_nhwc_concat_pad_plan(
        model_ir,
        graph_index,
        output_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
        public_names=public_names,
    )
    if pad_plan is None:
        return None
    return _QuantizedInputPlan(
        kind="pad",
        adapter_op=pad_plan.adapter_op,
        pad_plan=pad_plan,
        concat_input=concat_input,
        source_name=pad_plan.source_name,
        remove_adapter=pad_plan.remove_adapter,
    )


def _resolve_swish_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    swish_plan = _resolve_swish_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
    )
    if swish_plan is None:
        return None
    return _QuantizedInputPlan(
        kind="swish",
        adapter_op=swish_plan.adapter_op,
        nhwc_plan=swish_plan,
        concat_input=concat_input,
        source_name=swish_plan.source_name,
        remove_adapter=swish_plan.remove_adapter,
    )


def _resolve_dequantize_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    dequantize_plan = _resolve_dequantize_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
    )
    if dequantize_plan is None:
        return None
    source_tensor = model_ir.tensors.get(dequantize_plan.source_name)
    if source_tensor is None or len(list(source_tensor.shape)) != 4:
        return None
    return _QuantizedInputPlan(
        kind="dequantize",
        adapter_op=dequantize_plan.adapter_op,
        nhwc_plan=dequantize_plan,
        concat_input=concat_input,
        source_name=dequantize_plan.source_name,
        remove_adapter=dequantize_plan.remove_adapter,
    )


def _resolve_prelu_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_QuantizedInputPlan]:
    prelu_plan = _resolve_prelu_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
        public_names=public_names,
    )
    if prelu_plan is None:
        return None
    return _QuantizedInputPlan(
        kind="prelu",
        adapter_op=prelu_plan.adapter_op,
        nhwc_plan=prelu_plan,
        concat_input=concat_input,
        source_name=prelu_plan.source_name,
        remove_adapter=prelu_plan.remove_adapter,
    )


def _resolve_softmax_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    softmax_plan = _resolve_softmax_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
    )
    if softmax_plan is None:
        return None
    return _QuantizedInputPlan(
        kind="softmax",
        adapter_op=softmax_plan.adapter_op,
        nhwc_plan=softmax_plan,
        concat_input=concat_input,
        source_name=softmax_plan.source_name,
        remove_adapter=softmax_plan.remove_adapter,
    )


def _resolve_leaky_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_QuantizedInputPlan]:
    leaky_plan = _resolve_leaky_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
    )
    if leaky_plan is None:
        return None
    return _QuantizedInputPlan(
        kind="leaky",
        adapter_op=leaky_plan.adapter_op,
        nhwc_plan=leaky_plan,
        concat_input=concat_input,
        source_name=leaky_plan.source_name,
        remove_adapter=leaky_plan.remove_adapter,
    )


def _resolve_slice_quantized_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    concat_input: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_QuantizedInputPlan]:
    slice_plan = _resolve_slice_input_plan(
        model_ir,
        graph_index,
        input_name=concat_input,
        concat_index=concat_index,
        model_outputs=model_outputs,
        public_names=public_names,
    )
    if slice_plan is None or slice_plan.output_post_adapter_ops:
        return None
    return _QuantizedInputPlan(
        kind="slice",
        adapter_op=slice_plan.adapter_op,
        nhwc_plan=slice_plan,
        concat_input=concat_input,
        source_name=slice_plan.source_name,
        remove_adapter=slice_plan.remove_adapter,
    )


def _resolve_quantized_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    family: str,
) -> Optional[_QuantizedConcatCandidate]:
    model_outputs = {str(name) for name in model_ir.outputs}
    public_names = {
        *[str(name) for name in model_ir.inputs],
        *model_outputs,
    }
    for concat_op in model_ir.operators:
        concat_index = graph_index.operator_index(concat_op)
        if (
            concat_index is None
            or str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.inputs) == 0
            or len(concat_op.outputs) != 1
        ):
            continue
        concat_output = str(concat_op.outputs[0])
        concat_tensor = model_ir.tensors.get(concat_output)
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 4
        if (
            concat_axis != 1
            or concat_output in model_outputs
            or concat_tensor is None
            or len(list(concat_tensor.shape)) != 4
        ):
            continue

        concat_users = graph_index.consumer_indices(concat_output)
        if len(concat_users) != 1:
            continue
        quantize_op = model_ir.operators[int(concat_users[0])]
        quantize_index = graph_index.operator_index(quantize_op)
        if (
            quantize_index is None
            or str(quantize_op.op_type) != "QUANTIZE"
            or len(quantize_op.inputs) != 1
            or len(quantize_op.outputs) != 1
            or str(quantize_op.inputs[0]) != concat_output
        ):
            continue
        quantized_output = str(quantize_op.outputs[0])
        quantized_tensor = model_ir.tensors.get(quantized_output)
        if (
            quantized_output in model_outputs
            or quantized_tensor is None
            or len(list(quantized_tensor.shape)) != 4
        ):
            continue

        post_ops: List[OperatorIR] = []
        post_outputs: List[str] = []
        quantized_users = graph_index.consumer_indices(quantized_output)
        if not quantized_users:
            continue
        posts_valid = True
        for post_index in quantized_users:
            post_op = model_ir.operators[int(post_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != quantized_output
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
                or str(post_op.outputs[0]) in model_outputs
            ):
                posts_valid = False
                break
            post_output = str(post_op.outputs[0])
            post_tensor = model_ir.tensors.get(post_output)
            if post_tensor is None or len(list(post_tensor.shape)) != 4:
                posts_valid = False
                break
            post_ops.append(post_op)
            post_outputs.append(post_output)
        if not posts_valid or not post_ops:
            continue

        input_plans: List[_QuantizedInputPlan] = []
        inputs_valid = True
        for concat_input in [str(name) for name in concat_op.inputs]:
            input_plan = _resolve_direct_input_plan(
                model_ir,
                graph_index,
                concat_input=concat_input,
                concat_index=int(concat_index),
                model_outputs=model_outputs,
            )
            if input_plan is None and family in {"unary", "unary_pad"}:
                input_plan = _resolve_unary_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "swish":
                input_plan = _resolve_swish_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "dequantize":
                input_plan = _resolve_dequantize_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "prelu":
                input_plan = _resolve_prelu_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                    public_names=public_names,
                )
            if input_plan is None and family == "softmax":
                input_plan = _resolve_softmax_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "leaky":
                input_plan = _resolve_leaky_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                )
            if input_plan is None and family == "slice":
                input_plan = _resolve_slice_quantized_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                    public_names=public_names,
                )
            if input_plan is None and family in {
                "pad",
                "unary_pad",
                "all_pad",
            }:
                input_plan = _resolve_pad_input_plan(
                    model_ir,
                    graph_index,
                    concat_input=concat_input,
                    concat_index=int(concat_index),
                    model_outputs=model_outputs,
                    public_names=public_names,
                )
            if input_plan is None:
                inputs_valid = False
                break
            input_plans.append(input_plan)
        if not inputs_valid or not input_plans:
            continue
        unary_count = sum(plan.kind == "unary" for plan in input_plans)
        if family == "direct" and unary_count != 0:
            continue
        if family == "unary" and unary_count < 1:
            continue
        pad_count = sum(plan.kind == "pad" for plan in input_plans)
        if family == "pad" and (
            pad_count < 1 or len(input_plans) <= pad_count
        ):
            continue
        if family == "unary_pad" and (unary_count < 1 or pad_count < 1):
            continue
        if family == "all_pad" and (
            len(input_plans) < 2 or pad_count != len(input_plans)
        ):
            continue
        swish_count = sum(plan.kind == "swish" for plan in input_plans)
        if family == "swish" and swish_count < 1:
            continue
        dequantize_count = sum(
            plan.kind == "dequantize" for plan in input_plans
        )
        if family == "dequantize" and dequantize_count < 1:
            continue
        prelu_count = sum(plan.kind == "prelu" for plan in input_plans)
        if family == "prelu" and prelu_count < 1:
            continue
        softmax_count = sum(plan.kind == "softmax" for plan in input_plans)
        if family == "softmax" and (
            softmax_count != 1
            or len(input_plans) - softmax_count < 1
        ):
            continue
        leaky_count = sum(plan.kind == "leaky" for plan in input_plans)
        if family == "leaky" and leaky_count < 1:
            continue
        slice_count = sum(plan.kind == "slice" for plan in input_plans)
        if family == "slice" and slice_count < 1:
            continue
        if family in {
            "unary",
            "pad",
            "unary_pad",
            "all_pad",
            "swish",
            "dequantize",
            "prelu",
            "softmax",
            "leaky",
            "slice",
        }:
            reference_shape: Optional[List[int]] = None
            shapes_compatible = True
            for input_plan in input_plans:
                tensor = model_ir.tensors.get(
                    input_plan.source_name
                    if input_plan.kind == "direct"
                    else input_plan.concat_input
                )
                if tensor is None or len(list(tensor.shape)) != 4:
                    shapes_compatible = False
                    break
                shape = [int(value) for value in tensor.shape]
                if input_plan.kind in {
                    "unary",
                    "pad",
                    "swish",
                    "dequantize",
                    "prelu",
                    "softmax",
                    "leaky",
                    "slice",
                }:
                    shape = [shape[index] for index in _PERM_NCHW_TO_NHWC]
                if reference_shape is None:
                    reference_shape = shape
                elif any(
                    int(shape[index]) != int(reference_shape[index])
                    for index in (0, 1, 2)
                ):
                    shapes_compatible = False
                    break
            if not shapes_compatible:
                continue
        return _QuantizedConcatCandidate(
            input_plans=tuple(input_plans),
            concat_op=concat_op,
            concat_output=concat_output,
            quantize_op=quantize_op,
            quantized_output=quantized_output,
            post_ops=tuple(post_ops),
            post_outputs=tuple(post_outputs),
        )
    return None


def _has_quantized_direct_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="direct",
        )
        is not None
    )


def _has_quantized_unary_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="unary",
        )
        is not None
    )


def _has_quantized_pad_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="pad",
        )
        is not None
    )


def _has_quantized_unary_pad_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="unary_pad",
        )
        is not None
    )


def _has_quantized_all_pad_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="all_pad",
        )
        is not None
    )


def _has_quantized_swish_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="swish",
        )
        is not None
    )


def _has_quantized_dequantize_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="dequantize",
        )
        is not None
    )


def _has_quantized_prelu_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="prelu",
        )
        is not None
    )


def _has_quantized_softmax_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="softmax",
        )
        is not None
    )


def _has_quantized_leaky_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="leaky",
        )
        is not None
    )


def _has_quantized_slice_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="slice",
        )
        is not None
    )


def _optimize_quantized_concat_chains(
    model_ir: ModelIR,
    *,
    family: str,
    stats_key: str,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    optimized = 0
    while True:
        candidate = _resolve_quantized_concat_candidate(
            model_ir,
            graph_index,
            family=family,
        )
        if candidate is None:
            break

        new_concat_inputs: List[str] = []
        materialized_pads: Dict[str, str] = {}
        materialized_alphas: Dict[
            Tuple[str, Optional[Tuple[int, ...]], Tuple[int, ...]],
            str,
        ] = {}
        materialized_int_parameters: Dict[
            Tuple[str, Tuple[int, ...]],
            str,
        ] = {}
        for input_plan in candidate.input_plans:
            if input_plan.nhwc_plan is not None:
                if input_plan.kind == "dequantize":
                    _apply_dequantize_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                    )
                elif input_plan.kind == "swish":
                    _apply_swish_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                    )
                elif input_plan.kind == "prelu":
                    _apply_prelu_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                        materialized_alphas=materialized_alphas,
                    )
                elif input_plan.kind == "softmax":
                    _apply_softmax_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                    )
                elif input_plan.kind == "leaky":
                    _apply_leaky_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                    )
                elif input_plan.kind == "slice":
                    _apply_slice_input_plan(
                        model_ir,
                        graph_index,
                        input_plan.nhwc_plan,
                        materialized=materialized_int_parameters,
                    )
                else:
                    raise RuntimeError(
                        "unsupported shared quantized Concat input plan: "
                        f"{input_plan.kind}"
                    )
                new_concat_inputs.append(input_plan.concat_input)
            elif input_plan.unary_op is not None:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=input_plan.unary_op,
                    new_inputs=[input_plan.source_name],
                    graph_index=graph_index,
                )
                unary_tensor = model_ir.tensors.get(input_plan.concat_input)
                _permute_tensor_metadata_if_rank_matches(
                    unary_tensor,
                    _PERM_NCHW_TO_NHWC,
                )
                if unary_tensor is not None:
                    unary_tensor.quantization = _clone_permuted_quantization(
                        unary_tensor.quantization,
                        _PERM_NCHW_TO_NHWC,
                    )
                new_concat_inputs.append(input_plan.concat_input)
            elif input_plan.pad_plan is not None:
                apply_nhwc_concat_pad_plan(
                    model_ir,
                    graph_index,
                    input_plan.pad_plan,
                    materialized_pads=materialized_pads,
                )
                new_concat_inputs.append(input_plan.concat_input)
            else:
                new_concat_inputs.append(input_plan.source_name)
        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=new_concat_inputs,
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 3
        concat_tensor = model_ir.tensors.get(candidate.concat_output)
        _permute_tensor_metadata_if_rank_matches(
            concat_tensor,
            _PERM_NCHW_TO_NHWC,
        )
        if concat_tensor is not None:
            concat_tensor.quantization = _clone_permuted_quantization(
                concat_tensor.quantization,
                _PERM_NCHW_TO_NHWC,
            )

        canonical_output = candidate.post_outputs[0]
        _set_operator_outputs(
            model_ir=model_ir,
            op=candidate.quantize_op,
            new_outputs=[canonical_output],
            graph_index=graph_index,
        )
        for alias_output in candidate.post_outputs[1:]:
            _replace_tensor_inputs(
                model_ir=model_ir,
                src_name=alias_output,
                dst_name=canonical_output,
                graph_index=graph_index,
            )

        quantized_tensor = model_ir.tensors.get(candidate.quantized_output)
        canonical_tensor = model_ir.tensors.get(canonical_output)
        if canonical_tensor is not None and quantized_tensor is not None:
            canonical_tensor.dtype = str(quantized_tensor.dtype)
            canonical_tensor.quantization = _clone_permuted_quantization(
                quantized_tensor.quantization,
                _PERM_NCHW_TO_NHWC,
            )
        if canonical_tensor is not None and concat_tensor is not None:
            canonical_tensor.shape = [int(value) for value in concat_tensor.shape]
            canonical_tensor.shape_signature = (
                [int(value) for value in concat_tensor.shape_signature]
                if concat_tensor.shape_signature is not None
                else [int(value) for value in concat_tensor.shape]
            )

        remove_ops = [
            *candidate.post_ops,
            *[
                plan.adapter_op
                for plan in candidate.input_plans
                if plan.remove_adapter
            ],
        ]
        remove_indices = sorted(
            {
                int(index)
                for op in remove_ops
                if (index := graph_index.operator_index(op)) is not None
            },
            reverse=True,
        )
        for remove_index in remove_indices:
            graph_index.remove_operator(remove_index)
        optimized += 1

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {stats_key: int(optimized)}


def run_nhwc_concat_quantized_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Lift bounded Concat→Quantize post-adapter families into NHWC."""

    def _run_direct(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="direct",
            stats_key=_DIRECT_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_DIRECT_STATS_KEY, 0)),
        }

    def _run_unary(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="unary",
            stats_key=_UNARY_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_UNARY_STATS_KEY, 0)),
        }

    def _run_pad(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="pad",
            stats_key=_PAD_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_PAD_STATS_KEY, 0)),
        }

    def _run_unary_pad(
        pass_state: ModelIRPassState,
    ) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="unary_pad",
            stats_key=_UNARY_PAD_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_UNARY_PAD_STATS_KEY, 0)),
        }

    def _run_all_pad(
        pass_state: ModelIRPassState,
    ) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="all_pad",
            stats_key=_ALL_PAD_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_ALL_PAD_STATS_KEY, 0)),
        }

    def _run_swish(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="swish",
            stats_key=_SWISH_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_SWISH_STATS_KEY, 0)),
        }

    def _run_dequantize(
        pass_state: ModelIRPassState,
    ) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="dequantize",
            stats_key=_DEQUANTIZE_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_DEQUANTIZE_STATS_KEY, 0)),
        }

    def _run_prelu(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="prelu",
            stats_key=_PRELU_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_PRELU_STATS_KEY, 0)),
        }

    def _run_softmax(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="softmax",
            stats_key=_SOFTMAX_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_SOFTMAX_STATS_KEY, 0)),
        }

    def _run_leaky(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="leaky",
            stats_key=_LEAKY_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_LEAKY_STATS_KEY, 0)),
        }

    def _run_slice(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_concat_chains(
            pass_state.model_ir,
            family="slice",
            stats_key=_SLICE_STATS_KEY,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_SLICE_STATS_KEY, 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_direct",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_direct,
                precondition=_has_quantized_direct_concat_candidate,
                priority=10,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_unary",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_unary,
                precondition=_has_quantized_unary_concat_candidate,
                priority=20,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_pad",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_pad,
                precondition=_has_quantized_pad_concat_candidate,
                priority=30,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_unary_pad",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_unary_pad,
                precondition=_has_quantized_unary_pad_concat_candidate,
                priority=40,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_all_pad",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_all_pad,
                precondition=_has_quantized_all_pad_concat_candidate,
                priority=50,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_swish",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_swish,
                precondition=_has_quantized_swish_concat_candidate,
                priority=60,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_dequantize",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_dequantize,
                precondition=_has_quantized_dequantize_concat_candidate,
                priority=70,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_prelu",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_prelu,
                precondition=_has_quantized_prelu_concat_candidate,
                priority=80,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_softmax",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_softmax,
                precondition=_has_quantized_softmax_concat_candidate,
                priority=90,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_leaky",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_leaky,
                precondition=_has_quantized_leaky_concat_candidate,
                priority=100,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_slice",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_slice,
                precondition=_has_quantized_slice_concat_candidate,
                priority=110,
                transactional=True,
            ),
        ],
        layout_state=layout_state,
        default_details={
            _DIRECT_STATS_KEY: 0,
            _UNARY_STATS_KEY: 0,
            _PAD_STATS_KEY: 0,
            _UNARY_PAD_STATS_KEY: 0,
            _ALL_PAD_STATS_KEY: 0,
            _SWISH_STATS_KEY: 0,
            _DEQUANTIZE_STATS_KEY: 0,
            _PRELU_STATS_KEY: 0,
            _SOFTMAX_STATS_KEY: 0,
            _LEAKY_STATS_KEY: 0,
            _SLICE_STATS_KEY: 0,
        },
        diagnostics=diagnostics,
        preflight=lambda candidate_model: preflight_required_op_types(
            candidate_model,
            {"TRANSPOSE", "CONCATENATION", "QUANTIZE"},
        ),
    )
    return {str(key): int(value) for key, value in details.items()}
