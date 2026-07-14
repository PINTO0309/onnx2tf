from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    logical_layout_permutation,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _compose_axis_permutations,
    _read_transpose_perm,
)


def _fold_single_consumer_public_input_bridge_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    consumer_index: Dict[str, List[int]],
    public_layout_bridge_tensor_names: Set[str],
    public_input_names: Set[str],
    tensor_name: str,
    downstream_permute: Optional[Sequence[int]],
) -> Tuple[str, Optional[List[int]], Optional[int]]:
    producer_idx = producer_index.get(str(tensor_name), None)
    resolved_downstream_permute = (
        [int(v) for v in list(downstream_permute)]
        if downstream_permute is not None
        else None
    )
    if producer_idx is None:
        return str(tensor_name), resolved_downstream_permute, None
    producer_op = model_ir.operators[int(producer_idx)]
    if (
        str(producer_op.op_type) != "TRANSPOSE"
        or len(producer_op.outputs) != 1
        or len(producer_op.inputs) < 1
    ):
        return str(tensor_name), resolved_downstream_permute, None
    bridge_output_name = str(producer_op.outputs[0])
    bridge_input_name = str(producer_op.inputs[0])
    if (
        bridge_output_name not in public_layout_bridge_tensor_names
        and not bridge_output_name.endswith("_public_layout_bridge")
    ):
        return str(tensor_name), resolved_downstream_permute, None
    if bridge_input_name not in public_input_names:
        return str(tensor_name), resolved_downstream_permute, None
    if len(consumer_index.get(bridge_output_name, [])) != 1:
        return str(tensor_name), resolved_downstream_permute, None
    bridge_perm = _read_transpose_perm(model_ir, producer_op)
    if bridge_perm is None:
        return str(tensor_name), resolved_downstream_permute, None
    composed_perm = _compose_axis_permutations(
        bridge_perm,
        downstream_permute,
    )
    return bridge_input_name, composed_perm, int(producer_idx)


def _match_single_consumer_layout_bridge_transpose_for_codegen(
    *,
    model_ir: ModelIR,
    consumer_index: Dict[str, List[int]],
    tensor_name: str,
    required_output_layout: Optional[str] = None,
) -> Optional[Tuple[str, int]]:
    consumer_indices = consumer_index.get(str(tensor_name), [])
    if len(consumer_indices) != 1:
        return None
    bridge_op_idx = int(consumer_indices[0])
    bridge_op = model_ir.operators[bridge_op_idx]
    if str(bridge_op.op_type) != "TRANSPOSE" or len(bridge_op.outputs) != 1:
        return None
    input_tensor = model_ir.tensors.get(str(tensor_name), None)
    output_name = str(bridge_op.outputs[0])
    output_tensor = model_ir.tensors.get(output_name, None)
    if input_tensor is None or output_tensor is None:
        return None
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if (
        input_layout == LOGICAL_LAYOUT_UNKNOWN
        or output_layout == LOGICAL_LAYOUT_UNKNOWN
        or input_layout == output_layout
    ):
        return None
    if required_output_layout is not None and output_layout != normalize_logical_layout(
        required_output_layout
    ):
        return None
    expected_perm = logical_layout_permutation(
        source_layout=input_layout,
        target_layout=output_layout,
    )
    actual_perm = _read_transpose_perm(model_ir, bridge_op)
    if expected_perm is None or actual_perm is None:
        return None
    if [int(v) for v in list(expected_perm)] != [
        int(v) for v in list(actual_perm)
    ]:
        return None
    return output_name, bridge_op_idx
