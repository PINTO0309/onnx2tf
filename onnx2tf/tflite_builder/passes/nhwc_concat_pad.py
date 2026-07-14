from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _read_transpose_perm,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]


@dataclass(frozen=True)
class NhwcConcatPadPlan:
    adapter_op: OperatorIR
    pad_op: OperatorIR
    output_name: str
    source_name: str
    remove_adapter: bool
    pads_tensor_name: str
    pads_nhwc: np.ndarray
    clone_pads: bool


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


def resolve_nhwc_concat_pad_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    output_name: str,
    concat_index: int,
    model_outputs: set[str],
    public_names: set[str],
) -> Optional[NhwcConcatPadPlan]:
    pad_op = graph_index.producer(output_name)
    pad_index = (
        None if pad_op is None else graph_index.operator_index(pad_op)
    )
    if (
        pad_op is None
        or pad_index is None
        or str(pad_op.op_type) != "PAD"
        or len(pad_op.inputs) < 2
        or len(pad_op.outputs) != 1
        or str(pad_op.outputs[0]) != output_name
        or output_name in model_outputs
        or set(graph_index.consumer_indices(output_name))
        != {int(concat_index)}
    ):
        return None

    adapter_output = str(pad_op.inputs[0])
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
    ):
        return None

    source_name = str(adapter_op.inputs[0])
    source_tensor = model_ir.tensors.get(source_name)
    output_tensor = model_ir.tensors.get(output_name)
    pads_tensor_name = str(pad_op.inputs[1])
    pads_tensor = model_ir.tensors.get(pads_tensor_name)
    if (
        source_tensor is None
        or len(list(source_tensor.shape)) != 4
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
    pads_nhwc = np.asarray(
        [pads_pairs[0], pads_pairs[2], pads_pairs[3], pads_pairs[1]],
        dtype=pads_pairs.dtype,
    )
    return NhwcConcatPadPlan(
        adapter_op=adapter_op,
        pad_op=pad_op,
        output_name=output_name,
        source_name=source_name,
        remove_adapter=(
            set(graph_index.consumer_indices(adapter_output))
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


def apply_nhwc_concat_pad_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: NhwcConcatPadPlan,
    *,
    materialized_pads: Dict[str, str],
) -> None:
    rewritten_pads_name = materialized_pads.get(plan.pads_tensor_name)
    if rewritten_pads_name is None:
        pads_tensor = model_ir.tensors[plan.pads_tensor_name]
        rewritten_pads_name = plan.pads_tensor_name
        if plan.clone_pads:
            rewritten_pads_name = _unique_tensor_name(
                model_ir,
                f"{plan.pads_tensor_name}_nhwc",
            )
            model_ir.tensors[rewritten_pads_name] = TensorIR(
                name=rewritten_pads_name,
                dtype=str(pads_tensor.dtype),
                shape=[4, 2],
                shape_signature=[4, 2],
                data=np.array(plan.pads_nhwc, copy=True),
                is_variable=bool(pads_tensor.is_variable),
                quantization=_clone_quantization(pads_tensor.quantization),
                logical_layout=str(pads_tensor.logical_layout),
                physical_layout=str(pads_tensor.physical_layout),
                onnx_tensor_name=pads_tensor.onnx_tensor_name,
            )
        else:
            pads_tensor.data = np.array(plan.pads_nhwc, copy=True)
            pads_tensor.shape = [4, 2]
            pads_tensor.shape_signature = [4, 2]
        materialized_pads[plan.pads_tensor_name] = rewritten_pads_name

    new_pad_inputs = [str(name) for name in plan.pad_op.inputs]
    new_pad_inputs[0] = plan.source_name
    new_pad_inputs[1] = rewritten_pads_name
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.pad_op,
        new_inputs=new_pad_inputs,
        graph_index=graph_index,
    )
    output_tensor = model_ir.tensors.get(plan.output_name)
    _permute_tensor_metadata_if_rank_matches(
        output_tensor,
        _PERM_NCHW_TO_NHWC,
    )
    if output_tensor is not None:
        output_tensor.quantization = _clone_nhwc_quantization(
            output_tensor.quantization
        )
