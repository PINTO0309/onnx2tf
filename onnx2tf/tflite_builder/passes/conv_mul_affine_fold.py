from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _set_operator_outputs
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    _View,
    _consumer_slots,
    _layout_of,
    _op_type,
    _operator_contract,
    _operator_index,
    _resolved_source,
    _tensor_contract,
    _view,
)


_TOTAL_KEY = "folded_conv_mul_add_affine_chains"
_ADD_ONLY_KEY = "folded_conv_add_only_affine_chains"
_MUL_ONLY_KEY = "folded_conv_mul_only_affine_chains"
_MUL_ADD_KEY = "folded_conv_mul_add_only_affine_chains"


@dataclass(frozen=True)
class _Plan:
    conv: OperatorIR
    mul: OperatorIR
    source_name: str
    filter_name: str
    bias_name: str
    side_name: str
    conv_output: str
    mul_output: str
    mul_data_input_index: int
    mul_side_input_index: int
    out_channels: int
    source_view: _View
    output_view: _View
    tensor_contracts: Tuple[Tuple[Any, ...], ...]
    operator_contracts: Tuple[Tuple[Any, ...], ...]
    graph_inputs: Tuple[str, ...]
    graph_outputs: Tuple[str, ...]


def _stats(rewritten: int) -> Dict[str, int]:
    value = int(rewritten)
    return {
        _TOTAL_KEY: value,
        _ADD_ONLY_KEY: 0,
        _MUL_ONLY_KEY: value,
        _MUL_ADD_KEY: 0,
    }


def _plan_signature(plan: _Plan) -> Tuple[Any, ...]:
    return (
        id(plan.conv),
        id(plan.mul),
        plan.source_name,
        plan.filter_name,
        plan.bias_name,
        plan.side_name,
        plan.conv_output,
        plan.mul_output,
        plan.mul_data_input_index,
        plan.mul_side_input_index,
        plan.out_channels,
        plan.source_view,
        plan.output_view,
        plan.tensor_contracts,
        plan.operator_contracts,
        plan.graph_inputs,
        plan.graph_outputs,
    )


def _static_rank4(view: _View) -> bool:
    return bool(
        len(view.shape) == 4
        and len(view.signature) == 4
        and all(int(value) > 0 for value in view.shape)
        and all(int(value) > 0 for value in view.signature)
    )


def _layout_in(
    name: str,
    tensor: TensorIR,
    layout_state: Optional[LayoutState],
    allowed: set[str],
) -> bool:
    return str(_layout_of(str(name), tensor, layout_state)).upper() in allowed


def _exclusive_slot_consumers(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    name: str,
    expected: Tuple[Tuple[OperatorIR, int], ...],
) -> bool:
    actual = _consumer_slots(model_ir, graph_index, str(name))
    return sorted((id(operator), int(slot)) for operator, slot in actual) == sorted(
        (id(operator), int(slot)) for operator, slot in expected
    )


def _strict_float_constant(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    name: str,
    shape: Tuple[int, ...],
    consumer: OperatorIR,
    slot: int,
) -> Optional[Tuple[TensorIR, np.ndarray]]:
    tensor = model_ir.tensors.get(str(name))
    public_names = {str(value) for value in model_ir.inputs} | {
        str(value) for value in model_ir.outputs
    }
    if (
        tensor is None
        or tensor.data is None
        or str(tensor.dtype).upper() != "FLOAT32"
        or bool(tensor.is_variable)
        or tensor.quantization is not None
        or str(name) in public_names
        or str(name) in graph_index.producers
        or str(name) in graph_index.duplicate_producers
        or tuple(int(value) for value in tensor.shape) != shape
        or tuple(
            int(value)
            for value in (
                tensor.shape
                if tensor.shape_signature is None
                else tensor.shape_signature
            )
        )
        != shape
        or not _exclusive_slot_consumers(
            model_ir,
            graph_index,
            str(name),
            ((consumer, int(slot)),),
        )
    ):
        return None
    data = np.asarray(tensor.data)
    if (
        data.dtype != np.dtype(np.float32)
        or tuple(int(value) for value in data.shape) != shape
        or not bool(np.isfinite(data).all())
    ):
        return None
    return tensor, data


def _has_constant_add_suffix(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    mul: OperatorIR,
) -> bool:
    mul_output = str(mul.outputs[0])
    users = graph_index.consumers_of(mul_output)
    if len(users) != 1:
        return False
    add = users[0]
    if _op_type(add) != "ADD" or len(add.inputs) != 2 or len(add.outputs) != 1:
        return False
    side_names = [str(value) for value in add.inputs if str(value) != mul_output]
    if len(side_names) != 1:
        return False
    side = model_ir.tensors.get(side_names[0])
    return bool(side is not None and side.data is not None)


def _supported_conv_options(options: Dict[str, Any]) -> bool:
    try:
        return bool(
            str(options.get("fusedActivationFunction", "NONE")).upper()
            == "NONE"
            and str(options.get("padding", "")).upper() == "SAME"
            and int(options.get("strideH", 0)) == 1
            and int(options.get("strideW", 0)) == 1
            and int(options.get("dilationHFactor", 0)) == 1
            and int(options.get("dilationWFactor", 0)) == 1
        )
    except (TypeError, ValueError):
        return False


def _resolve_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    conv: OperatorIR,
    *,
    layout_state: Optional[LayoutState],
) -> Optional[_Plan]:
    conv_index = _operator_index(graph_index, conv)
    if (
        conv_index is None
        or _op_type(conv) != "CONV_2D"
        or len(conv.inputs) != 3
        or len(conv.outputs) != 1
        or not isinstance(conv.options, dict)
    ):
        return None
    options = dict(conv.options)
    if not _supported_conv_options(options):
        return None

    source_name = str(conv.inputs[0])
    filter_name = str(conv.inputs[1])
    bias_name = str(conv.inputs[2])
    conv_output = str(conv.outputs[0])
    public_names = {str(value) for value in model_ir.inputs} | {
        str(value) for value in model_ir.outputs
    }
    if (
        conv_output in public_names
        or conv_output in graph_index.duplicate_producers
        or graph_index.producer(conv_output) is not conv
        or not _resolved_source(
            model_ir,
            graph_index,
            name=source_name,
            before_index=int(conv_index),
        )
    ):
        return None
    users = graph_index.consumers_of(conv_output)
    if len(users) != 1:
        return None
    mul = users[0]
    mul_index = _operator_index(graph_index, mul)
    if (
        mul_index is None
        or int(mul_index) <= int(conv_index)
        or _op_type(mul) != "MUL"
        or len(mul.inputs) != 2
        or len(mul.outputs) != 1
        or not isinstance(mul.options, dict)
        or str(mul.options.get("fusedActivationFunction", "NONE")).upper()
        != "NONE"
    ):
        return None
    mul_output = str(mul.outputs[0])
    if (
        mul_output in public_names
        or mul_output in graph_index.duplicate_producers
        or graph_index.producer(mul_output) is not mul
        or _has_constant_add_suffix(model_ir, graph_index, mul)
    ):
        return None

    data_slots = [
        index
        for index, name in enumerate(mul.inputs)
        if str(name) == conv_output
    ]
    if len(data_slots) != 1:
        return None
    data_slot = int(data_slots[0])
    side_slot = 1 - data_slot
    side_name = str(mul.inputs[side_slot])
    if not _exclusive_slot_consumers(
        model_ir,
        graph_index,
        conv_output,
        ((mul, data_slot),),
    ):
        return None

    source_tensor = model_ir.tensors.get(source_name)
    conv_tensor = model_ir.tensors.get(conv_output)
    mul_tensor = model_ir.tensors.get(mul_output)
    if source_tensor is None or conv_tensor is None or mul_tensor is None:
        return None
    source_view = _view(source_tensor)
    output_view = _view(conv_tensor)
    if (
        not _static_rank4(source_view)
        or not _static_rank4(output_view)
        or _view(mul_tensor) != output_view
        or source_view.dtype.upper() != "FLOAT32"
        or output_view.dtype.upper() != "FLOAT32"
        or any(
            tensor.quantization is not None
            for tensor in (source_tensor, conv_tensor, mul_tensor)
        )
        or not _layout_in(
            source_name,
            source_tensor,
            layout_state,
            {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN},
        )
    ):
        return None
    conv_layout = str(_layout_of(conv_output, conv_tensor, layout_state)).upper()
    mul_layout = str(_layout_of(mul_output, mul_tensor, layout_state)).upper()
    if (
        conv_layout not in {LOGICAL_LAYOUT_NHWC, LOGICAL_LAYOUT_UNKNOWN}
        or mul_layout != conv_layout
    ):
        return None

    out_channels = int(output_view.shape[-1])
    input_channels = int(source_view.shape[-1])
    if out_channels <= 0 or input_channels <= 0:
        return None
    filter_resolved = _strict_float_constant(
        model_ir,
        graph_index,
        name=filter_name,
        shape=(out_channels, 1, 1, input_channels),
        consumer=conv,
        slot=1,
    )
    bias_resolved = _strict_float_constant(
        model_ir,
        graph_index,
        name=bias_name,
        shape=(out_channels,),
        consumer=conv,
        slot=2,
    )
    side_resolved = _strict_float_constant(
        model_ir,
        graph_index,
        name=side_name,
        shape=(1, 1, 1, out_channels),
        consumer=mul,
        slot=side_slot,
    )
    if (
        filter_resolved is None
        or bias_resolved is None
        or side_resolved is None
    ):
        return None

    involved_names = {
        source_name,
        filter_name,
        bias_name,
        side_name,
        conv_output,
        mul_output,
    }
    return _Plan(
        conv=conv,
        mul=mul,
        source_name=source_name,
        filter_name=filter_name,
        bias_name=bias_name,
        side_name=side_name,
        conv_output=conv_output,
        mul_output=mul_output,
        mul_data_input_index=data_slot,
        mul_side_input_index=side_slot,
        out_channels=out_channels,
        source_view=source_view,
        output_view=output_view,
        tensor_contracts=tuple(
            _tensor_contract(name, model_ir.tensors[name])
            for name in sorted(involved_names)
        ),
        operator_contracts=(
            _operator_contract(conv),
            _operator_contract(mul),
        ),
        graph_inputs=tuple(str(value) for value in model_ir.inputs),
        graph_outputs=tuple(str(value) for value in model_ir.outputs),
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _Plan,
    *,
    layout_state: Optional[LayoutState],
) -> bool:
    current = _resolve_candidate(
        model_ir,
        graph_index,
        plan.conv,
        layout_state=layout_state,
    )
    if current is None or _plan_signature(current) != _plan_signature(plan):
        return False
    if (
        tuple(str(value) for value in model_ir.inputs) != plan.graph_inputs
        or tuple(str(value) for value in model_ir.outputs) != plan.graph_outputs
    ):
        return False
    mul_index = _operator_index(graph_index, plan.mul)
    if mul_index is None:
        return False

    filter_tensor = model_ir.tensors[plan.filter_name]
    bias_tensor = model_ir.tensors[plan.bias_name]
    side_tensor = model_ir.tensors[plan.side_name]
    coefficient = np.asarray(side_tensor.data, dtype=np.float32).reshape(
        int(plan.out_channels)
    )
    filter_dtype = np.asarray(filter_tensor.data).dtype
    bias_dtype = np.asarray(bias_tensor.data).dtype
    folded_filter = np.asarray(filter_tensor.data, dtype=np.float32) * coefficient.reshape(
        int(plan.out_channels), 1, 1, 1
    )
    # Preserve the compatibility path's float32 operation order exactly.
    # Even in the Mul-only case it adds a positive-zero vector, which
    # canonicalizes negative zero in the product to positive zero.  Omitting
    # that numerically redundant add changes the serialized bias buffers.
    folded_bias = (
        np.asarray(bias_tensor.data, dtype=np.float32).reshape(
            int(plan.out_channels)
        )
        * coefficient
        + np.zeros((int(plan.out_channels),), dtype=np.float32)
    )
    if not bool(np.isfinite(folded_filter).all()) or not bool(
        np.isfinite(folded_bias).all()
    ):
        return False

    filter_tensor.data = np.asarray(folded_filter, dtype=filter_dtype)
    bias_tensor.data = np.asarray(folded_bias, dtype=bias_dtype)
    bias_tensor.shape = [int(plan.out_channels)]
    bias_tensor.shape_signature = [int(plan.out_channels)]
    _set_operator_outputs(
        model_ir=model_ir,
        op=plan.conv,
        new_outputs=[plan.mul_output],
        graph_index=graph_index,
    )
    graph_index.remove_operator(int(mul_index))
    return True


def optimize_conv_mul_affine_mul_only_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Fold the strict static production Conv/Mul affine family."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    candidates = (
        [candidate]
        if candidate is not None
        else [
            model_ir.operators[index]
            for index in active_index.operator_indices_for_normalized_types(
                {"CONV_2D"}
            )
        ]
    )
    rewrite_limit = (
        len(candidates) if max_rewrites is None else max(0, int(max_rewrites))
    )
    rewritten = 0
    pending = tuple(operator for operator in candidates if operator is not None)
    for conv in pending:
        if rewritten >= rewrite_limit:
            break
        if _operator_index(active_index, conv) is None:
            continue
        plan = _resolve_candidate(
            model_ir,
            active_index,
            conv,
            layout_state=layout_state,
        )
        if plan is None:
            continue
        if _apply_plan(
            model_ir,
            active_index,
            plan,
            layout_state=layout_state,
        ):
            rewritten += 1

    return _stats(rewritten)
