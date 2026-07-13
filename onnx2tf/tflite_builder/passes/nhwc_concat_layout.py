from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
_DIRECT_STATS_KEY = "optimized_transpose_pre_concat_nhwc_direct_chains"
_UNARY_STATS_KEY = "optimized_transpose_pre_concat_nhwc_unary_chains"
_PAD_STATS_KEY = "optimized_transpose_pre_concat_nhwc_pad_chains"
_DEQUANTIZE_STATS_KEY = (
    "optimized_transpose_pre_concat_nhwc_dequantize_chains"
)
_UNARY_OPS = {"RELU", "RELU6", "LOGISTIC", "TANH", "GELU"}


@dataclass(frozen=True)
class _NhwcConcatInputPlan:
    kind: str
    adapter_op: OperatorIR
    source_name: str
    output_name: str
    remove_adapter: bool
    unary_op: Optional[OperatorIR] = None
    pad_op: Optional[OperatorIR] = None
    dequantize_op: Optional[OperatorIR] = None
    pads_tensor_name: Optional[str] = None
    pads_nhwc: Optional[np.ndarray] = None
    clone_pads: bool = False


@dataclass(frozen=True)
class _NhwcConcatCandidate:
    input_plans: Tuple[_NhwcConcatInputPlan, ...]
    concat_op: OperatorIR
    concat_output_name: str
    post_ops: Tuple[OperatorIR, ...]
    post_output_names: Tuple[str, ...]


def _clone_nhwc_quantization(quantization: Any) -> Any:
    cloned = _clone_quantization(quantization)
    if isinstance(cloned, QuantParamIR):
        old_dimension = int(cloned.quantized_dimension)
        if 0 <= old_dimension < len(_PERM_NCHW_TO_NHWC):
            cloned.quantized_dimension = int(
                _PERM_NCHW_TO_NHWC.index(old_dimension)
            )
    elif isinstance(cloned, dict) and "quantized_dimension" in cloned:
        old_dimension = int(cloned["quantized_dimension"])
        if 0 <= old_dimension < len(_PERM_NCHW_TO_NHWC):
            cloned["quantized_dimension"] = int(
                _PERM_NCHW_TO_NHWC.index(old_dimension)
            )
    return cloned


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    name = str(base)
    suffix = 1
    while name in model_ir.tensors:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _resolve_direct_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    adapter_op = graph_index.producer(input_name)
    if (
        adapter_op is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != input_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
    ):
        return None
    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    consumers = set(graph_index.consumer_indices(input_name))
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or int(concat_index) not in consumers
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="direct",
        adapter_op=adapter_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=(
            consumers == {int(concat_index)}
            and input_name not in model_outputs
        ),
    )


def _resolve_unary_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    unary_op = graph_index.producer(input_name)
    unary_index = (
        None if unary_op is None else graph_index.operator_index(unary_op)
    )
    if (
        unary_op is None
        or unary_index is None
        or str(unary_op.op_type) not in _UNARY_OPS
        or len(unary_op.inputs) != 1
        or len(unary_op.outputs) != 1
        or str(unary_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    adapter_output_name = str(unary_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
        or set(graph_index.consumer_indices(adapter_output_name))
        != {int(unary_index)}
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(input_name)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="unary",
        adapter_op=adapter_op,
        unary_op=unary_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=True,
    )


def _resolve_pad_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    pad_op = graph_index.producer(input_name)
    pad_index = (
        None if pad_op is None else graph_index.operator_index(pad_op)
    )
    if (
        pad_op is None
        or pad_index is None
        or str(pad_op.op_type) != "PAD"
        or len(pad_op.inputs) < 2
        or len(pad_op.outputs) != 1
        or str(pad_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    adapter_output_name = str(pad_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(input_name)
    pads_tensor_name = str(pad_op.inputs[1])
    pads_tensor = model_ir.tensors.get(pads_tensor_name)
    if (
        source_tensor is None
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
        or pads_tensor is None
        or pads_tensor.data is None
    ):
        return None
    try:
        pads_pairs = np.asarray(pads_tensor.data).reshape(4, 2)
    except Exception:
        return None
    if int(pads_pairs.size) != 8:
        return None
    pads_nhwc = np.asarray(
        [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
        dtype=pads_pairs.dtype,
    )
    return _NhwcConcatInputPlan(
        kind="pad",
        adapter_op=adapter_op,
        pad_op=pad_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=(
            set(graph_index.consumer_indices(adapter_output_name))
            == {int(pad_index)}
        ),
        pads_tensor_name=pads_tensor_name,
        pads_nhwc=np.asarray(pads_nhwc),
        clone_pads=(
            pads_tensor_name in public_names
            or set(graph_index.consumer_indices(pads_tensor_name))
            != {int(pad_index)}
        ),
    )


def _resolve_dequantize_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    dequantize_op = graph_index.producer(input_name)
    dequantize_index = (
        None
        if dequantize_op is None
        else graph_index.operator_index(dequantize_op)
    )
    if (
        dequantize_op is None
        or dequantize_index is None
        or str(dequantize_op.op_type) != "DEQUANTIZE"
        or len(dequantize_op.inputs) != 1
        or len(dequantize_op.outputs) != 1
        or str(dequantize_op.outputs[0]) != input_name
        or input_name in model_outputs
        or set(graph_index.consumer_indices(input_name))
        != {int(concat_index)}
    ):
        return None

    adapter_output_name = str(dequantize_op.inputs[0])
    adapter_op = graph_index.producer(adapter_output_name)
    adapter_index = (
        None if adapter_op is None else graph_index.operator_index(adapter_op)
    )
    if (
        adapter_op is None
        or adapter_index is None
        or str(adapter_op.op_type) != "TRANSPOSE"
        or len(adapter_op.inputs) < 2
        or len(adapter_op.outputs) != 1
        or str(adapter_op.outputs[0]) != adapter_output_name
        or _read_transpose_perm(model_ir, adapter_op)
        != _PERM_NHWC_TO_NCHW
        or adapter_output_name in model_outputs
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(input_name)
    if (
        source_tensor is None
        or output_tensor is None
        or len(list(output_tensor.shape)) != 4
    ):
        return None
    return _NhwcConcatInputPlan(
        kind="dequantize",
        adapter_op=adapter_op,
        dequantize_op=dequantize_op,
        source_name=source_name,
        output_name=input_name,
        remove_adapter=(
            set(graph_index.consumer_indices(adapter_output_name))
            == {int(dequantize_index)}
        ),
    )


def _resolve_family_input_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    family: str,
    input_name: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[_NhwcConcatInputPlan]:
    direct_plan = _resolve_direct_input_plan(
        model_ir,
        graph_index,
        input_name=input_name,
        concat_index=concat_index,
        model_outputs=model_outputs,
    )
    if direct_plan is not None:
        return direct_plan
    if family == "unary":
        return _resolve_unary_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
    if family == "pad":
        return _resolve_pad_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
            public_names=public_names,
        )
    if family == "dequantize":
        return _resolve_dequantize_input_plan(
            model_ir,
            graph_index,
            input_name=input_name,
            concat_index=concat_index,
            model_outputs=model_outputs,
        )
    return None


def _resolve_nhwc_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    family: str,
) -> _NhwcConcatCandidate | None:
    """Find one strict float-path direct or unary Concat island."""

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

        concat_output_name = str(concat_op.outputs[0])
        concat_output_tensor = model_ir.tensors.get(concat_output_name)
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 4
        if (
            concat_axis != 1
            or concat_output_name in model_outputs
            or concat_output_tensor is None
            or len(list(concat_output_tensor.shape)) != 4
        ):
            continue

        input_plans: List[_NhwcConcatInputPlan] = []
        inputs_valid = True
        for input_name in [str(name) for name in concat_op.inputs]:
            input_plan = _resolve_family_input_plan(
                model_ir,
                graph_index,
                family=family,
                input_name=input_name,
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
        dequantize_count = sum(
            plan.kind == "dequantize" for plan in input_plans
        )
        if family == "dequantize" and dequantize_count < 1:
            continue

        if family in {"unary", "pad", "dequantize"}:
            reference_shape: Optional[List[int]] = None
            shapes_compatible = True
            for input_plan in input_plans:
                tensor = model_ir.tensors.get(
                    input_plan.source_name
                    if input_plan.kind == "direct"
                    else input_plan.output_name
                )
                if tensor is None or len(list(tensor.shape)) != 4:
                    shapes_compatible = False
                    break
                shape = [int(value) for value in tensor.shape]
                if input_plan.kind in {"unary", "pad", "dequantize"}:
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

        concat_user_indices = graph_index.consumer_indices(concat_output_name)
        if not concat_user_indices:
            continue
        post_ops: List[OperatorIR] = []
        post_output_names: List[str] = []
        posts_valid = True
        for user_index in concat_user_indices:
            post_op = model_ir.operators[int(user_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != concat_output_name
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
                or str(post_op.outputs[0]) in model_outputs
            ):
                posts_valid = False
                break
            post_ops.append(post_op)
            post_output_names.append(str(post_op.outputs[0]))
        if not posts_valid or not post_ops:
            continue

        return _NhwcConcatCandidate(
            input_plans=tuple(input_plans),
            concat_op=concat_op,
            concat_output_name=concat_output_name,
            post_ops=tuple(post_ops),
            post_output_names=tuple(post_output_names),
        )
    return None


def _has_nhwc_direct_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="direct",
        )
        is not None
    )


def _has_nhwc_unary_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="unary",
        )
        is not None
    )


def _has_nhwc_pad_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="pad",
        )
        is not None
    )


def _has_nhwc_dequantize_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_nhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
            family="dequantize",
        )
        is not None
    )


def _optimize_transpose_pre_concat_nhwc_direct_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="direct",
        stats_key=_DIRECT_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_unary_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="unary",
        stats_key=_UNARY_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_pad_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="pad",
        stats_key=_PAD_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_dequantize_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_family(
        model_ir,
        family="dequantize",
        stats_key=_DEQUANTIZE_STATS_KEY,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_concat_nhwc_family(
    model_ir: ModelIR,
    *,
    family: str,
    stats_key: str,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Lift one strict direct/unary NCHW Concat family into NHWC."""

    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        candidate = _resolve_nhwc_concat_candidate(
            model_ir,
            graph_index,
            family=family,
        )
        if candidate is None:
            break

        new_concat_inputs: List[str] = []
        materialized_pads: Dict[str, str] = {}
        for input_plan in candidate.input_plans:
            if input_plan.unary_op is not None:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=input_plan.unary_op,
                    new_inputs=[input_plan.source_name],
                    graph_index=graph_index,
                )
                unary_output_tensor = model_ir.tensors.get(input_plan.output_name)
                _permute_tensor_metadata_if_rank_matches(
                    unary_output_tensor,
                    _PERM_NCHW_TO_NHWC,
                )
                if unary_output_tensor is not None:
                    unary_output_tensor.quantization = _clone_nhwc_quantization(
                        unary_output_tensor.quantization
                    )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.dequantize_op is not None:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=input_plan.dequantize_op,
                    new_inputs=[input_plan.source_name],
                    graph_index=graph_index,
                )
                dequantize_output_tensor = model_ir.tensors.get(
                    input_plan.output_name
                )
                _permute_tensor_metadata_if_rank_matches(
                    dequantize_output_tensor,
                    _PERM_NCHW_TO_NHWC,
                )
                if dequantize_output_tensor is not None:
                    dequantize_output_tensor.quantization = (
                        _clone_nhwc_quantization(
                            dequantize_output_tensor.quantization
                        )
                    )
                new_concat_inputs.append(input_plan.output_name)
            elif input_plan.pad_op is not None:
                assert input_plan.pads_tensor_name is not None
                assert input_plan.pads_nhwc is not None
                pads_tensor_name = input_plan.pads_tensor_name
                rewritten_pads_name = materialized_pads.get(pads_tensor_name)
                if rewritten_pads_name is None:
                    pads_tensor = model_ir.tensors[pads_tensor_name]
                    rewritten_pads_name = pads_tensor_name
                    if input_plan.clone_pads:
                        rewritten_pads_name = _unique_tensor_name(
                            model_ir,
                            f"{pads_tensor_name}_nhwc",
                        )
                        model_ir.tensors[rewritten_pads_name] = TensorIR(
                            name=rewritten_pads_name,
                            dtype=str(pads_tensor.dtype),
                            shape=[4, 2],
                            shape_signature=[4, 2],
                            data=np.array(input_plan.pads_nhwc, copy=True),
                            is_variable=bool(pads_tensor.is_variable),
                            quantization=_clone_quantization(
                                pads_tensor.quantization
                            ),
                            logical_layout=str(pads_tensor.logical_layout),
                            physical_layout=str(pads_tensor.physical_layout),
                            onnx_tensor_name=pads_tensor.onnx_tensor_name,
                        )
                    else:
                        pads_tensor.data = np.array(
                            input_plan.pads_nhwc,
                            copy=True,
                        )
                        pads_tensor.shape = [4, 2]
                        pads_tensor.shape_signature = [4, 2]
                    materialized_pads[pads_tensor_name] = rewritten_pads_name

                new_pad_inputs = [str(name) for name in input_plan.pad_op.inputs]
                new_pad_inputs[0] = input_plan.source_name
                new_pad_inputs[1] = rewritten_pads_name
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=input_plan.pad_op,
                    new_inputs=new_pad_inputs,
                    graph_index=graph_index,
                )
                pad_output_tensor = model_ir.tensors.get(input_plan.output_name)
                _permute_tensor_metadata_if_rank_matches(
                    pad_output_tensor,
                    _PERM_NCHW_TO_NHWC,
                )
                if pad_output_tensor is not None:
                    pad_output_tensor.quantization = _clone_nhwc_quantization(
                        pad_output_tensor.quantization
                    )
                new_concat_inputs.append(input_plan.output_name)
            else:
                new_concat_inputs.append(input_plan.source_name)
        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=new_concat_inputs,
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 3

        canonical_output_name = candidate.post_output_names[0]
        _set_operator_outputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_outputs=[canonical_output_name],
            graph_index=graph_index,
        )
        for alias_output_name in candidate.post_output_names[1:]:
            _replace_tensor_inputs(
                model_ir,
                alias_output_name,
                canonical_output_name,
                graph_index=graph_index,
            )

        old_concat_tensor = model_ir.tensors.get(candidate.concat_output_name)
        canonical_output_tensor = model_ir.tensors.get(canonical_output_name)
        if old_concat_tensor is not None and canonical_output_tensor is not None:
            canonical_output_tensor.dtype = str(old_concat_tensor.dtype)
            canonical_output_tensor.shape = [
                int(value) for value in old_concat_tensor.shape
            ]
            canonical_output_tensor.shape_signature = (
                [int(value) for value in old_concat_tensor.shape_signature]
                if old_concat_tensor.shape_signature is not None
                else [int(value) for value in old_concat_tensor.shape]
            )
            _permute_tensor_metadata_if_rank_matches(
                canonical_output_tensor,
                _PERM_NCHW_TO_NHWC,
            )
            canonical_output_tensor.quantization = _clone_nhwc_quantization(
                old_concat_tensor.quantization
            )

        remove_ops = [
            *[
                plan.adapter_op
                for plan in candidate.input_plans
                if plan.remove_adapter
            ],
            *candidate.post_ops,
        ]
        remove_indices = sorted(
            {
                int(operator_index)
                for operator in remove_ops
                if (
                    operator_index := graph_index.operator_index(operator)
                )
                is not None
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


def run_nhwc_concat_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run the transactional rank-four Concat layout family group."""

    def _run_direct(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_direct_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_DIRECT_STATS_KEY, 0))}

    def _run_unary(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_unary_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_UNARY_STATS_KEY, 0))}

    def _run_pad(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_pad_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_PAD_STATS_KEY, 0))}

    def _run_dequantize(
        pass_state: ModelIRPassState,
    ) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_dequantize_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(stats.get(_DEQUANTIZE_STATS_KEY, 0)),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.nhwc_pre_concat_direct",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_direct,
                precondition=_has_nhwc_direct_concat_candidate,
                priority=10,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_unary",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_unary,
                precondition=_has_nhwc_unary_concat_candidate,
                priority=20,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_pad",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_pad,
                precondition=_has_nhwc_pad_concat_candidate,
                priority=30,
                transactional=True,
            ),
            PassSpec(
                pass_id="layout.nhwc_pre_concat_dequantize",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run_dequantize,
                precondition=_has_nhwc_dequantize_concat_candidate,
                priority=40,
                transactional=True,
            ),
        ],
        layout_state=layout_state,
        default_details={
            _DIRECT_STATS_KEY: 0,
            _UNARY_STATS_KEY: 0,
            _PAD_STATS_KEY: 0,
            _DEQUANTIZE_STATS_KEY: 0,
        },
        diagnostics=diagnostics,
        preflight=lambda candidate_model: preflight_required_op_types(
            candidate_model,
            {"TRANSPOSE", "CONCATENATION"},
        ),
    )
    return {str(key): int(value) for key, value in details.items()}
