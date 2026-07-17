from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _quant_scale_count,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir: ModelIR) -> Dict[str, int]:
    """Restore local NHWC split branches at an otherwise NCHW Concat.

    A channel Split may be propagated to NHWC for Conv consumers while another
    split output still feeds a legacy channel-axis Concat. Establish the
    Concat's NCHW spatial contract from at least two agreeing inputs, then add
    a local NHWC->NCHW adapter only for inputs whose leading spatial dimensions
    match that contract. This preserves both branches without guessing layout
    from tensor names.
    """

    repaired = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]

    def _unique_tensor_name(base: str, existing: set[str]) -> str:
        if str(base) not in existing:
            return str(base)
        suffix = 1
        while f"{base}_{suffix}" in existing:
            suffix += 1
        return f"{base}_{suffix}"

    while True:
        changed = False
        for concat_idx, concat_op in enumerate(model_ir.operators):
            if (
                str(concat_op.op_type) != "CONCATENATION"
                or len(concat_op.inputs) < 2
                or len(concat_op.outputs) != 1
            ):
                continue
            concat_axis = int(concat_op.options.get("axis", 0))
            if concat_axis < 0:
                concat_axis += 4
            if concat_axis != 1:
                continue

            input_tensors = [
                model_ir.tensors.get(str(name), None)
                for name in list(concat_op.inputs)
            ]
            if any(tensor is None for tensor in input_tensors):
                continue
            output_tensor = model_ir.tensors.get(
                str(concat_op.outputs[0]),
                None,
            )
            if output_tensor is None:
                continue
            input_shapes = [
                [int(v) for v in list(tensor.shape)]
                for tensor in input_tensors
            ]
            if any(
                len(shape) != 4 or any(int(v) <= 0 for v in shape)
                for shape in input_shapes
            ):
                continue

            spatial_counts: Dict[Tuple[int, int], int] = {}
            for shape in input_shapes:
                spatial = (int(shape[2]), int(shape[3]))
                spatial_counts[spatial] = int(spatial_counts.get(spatial, 0)) + 1
            canonical_spatial, canonical_count = max(
                spatial_counts.items(),
                key=lambda item: int(item[1]),
            )
            if int(canonical_count) < 2:
                output_shape = [int(v) for v in list(output_tensor.shape)]
                if (
                    len(output_shape) != 4
                    or any(int(v) <= 0 for v in output_shape)
                ):
                    continue
                output_spatial = (
                    int(output_shape[2]),
                    int(output_shape[3]),
                )
                if output_spatial not in spatial_counts:
                    continue
                canonical_spatial = output_spatial
            height, width = [int(v) for v in canonical_spatial]

            nhwc_input_indices: List[int] = []
            valid = True
            for input_idx, shape in enumerate(input_shapes):
                if [int(shape[2]), int(shape[3])] == [height, width]:
                    continue
                if [int(shape[1]), int(shape[2])] == [height, width]:
                    nhwc_input_indices.append(int(input_idx))
                    continue
                valid = False
                break
            if not valid or len(nhwc_input_indices) == 0:
                continue

            new_inputs = [str(v) for v in list(concat_op.inputs)]
            reserved_tensor_names = set(model_ir.tensors.keys())
            adapter_plans: List[Dict[str, Any]] = []
            planned_input_shapes = [list(shape) for shape in input_shapes]
            for input_idx in nhwc_input_indices:
                source_name = str(new_inputs[int(input_idx)])
                source_tensor = model_ir.tensors[source_name]
                source_shape = [int(v) for v in list(source_tensor.shape)]
                source_signature = (
                    [int(v) for v in list(source_tensor.shape_signature)]
                    if source_tensor.shape_signature is not None
                    else list(source_shape)
                )
                if len(source_signature) != 4:
                    valid = False
                    break
                adapter_name = _unique_tensor_name(
                    f"{source_name}_nchw_concat_adapter",
                    reserved_tensor_names,
                )
                reserved_tensor_names.add(adapter_name)
                perm_name = _unique_tensor_name(
                    f"{adapter_name}_perm",
                    reserved_tensor_names,
                )
                reserved_tensor_names.add(perm_name)
                adapter_shape = [
                    int(source_shape[0]),
                    int(source_shape[3]),
                    int(source_shape[1]),
                    int(source_shape[2]),
                ]
                adapter_signature = [
                    int(source_signature[0]),
                    int(source_signature[3]),
                    int(source_signature[1]),
                    int(source_signature[2]),
                ]
                adapter_quantization = _clone_quantization(
                    source_tensor.quantization
                )
                if _quant_scale_count(adapter_quantization) > 1:
                    if isinstance(adapter_quantization, dict):
                        old_qdim = int(
                            adapter_quantization.get("quantized_dimension", 0)
                        )
                    else:
                        old_qdim = int(adapter_quantization.quantized_dimension)
                    if old_qdim not in perm_nhwc_to_nchw:
                        valid = False
                        break
                    new_qdim = int(perm_nhwc_to_nchw.index(old_qdim))
                    if isinstance(adapter_quantization, dict):
                        adapter_quantization["quantized_dimension"] = new_qdim
                    else:
                        adapter_quantization.quantized_dimension = new_qdim
                adapter_plans.append(
                    {
                        "input_idx": int(input_idx),
                        "source_name": source_name,
                        "source_tensor": source_tensor,
                        "adapter_name": adapter_name,
                        "perm_name": perm_name,
                        "adapter_shape": adapter_shape,
                        "adapter_signature": adapter_signature,
                        "adapter_quantization": adapter_quantization,
                    }
                )
                new_inputs[int(input_idx)] = adapter_name
                planned_input_shapes[int(input_idx)] = list(adapter_shape)
            if not valid:
                continue

            output_shape = [
                int(input_shapes[0][0]),
                int(sum(int(shape[1]) for shape in planned_input_shapes)),
                height,
                width,
            ]

            insert_pos = int(concat_idx)
            for adapter_plan in adapter_plans:
                source_name = str(adapter_plan["source_name"])
                source_tensor = adapter_plan["source_tensor"]
                adapter_name = str(adapter_plan["adapter_name"])
                perm_name = str(adapter_plan["perm_name"])
                model_ir.tensors[perm_name] = TensorIR(
                    name=perm_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                    is_variable=False,
                )
                model_ir.tensors[adapter_name] = TensorIR(
                    name=adapter_name,
                    dtype=str(source_tensor.dtype),
                    shape=list(adapter_plan["adapter_shape"]),
                    shape_signature=list(adapter_plan["adapter_signature"]),
                    data=None,
                    is_variable=False,
                    quantization=adapter_plan["adapter_quantization"],
                )
                model_ir.operators.insert(
                    insert_pos,
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[source_name, perm_name],
                        outputs=[adapter_name],
                    ),
                )
                insert_pos += 1

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=new_inputs,
            )
            output_tensor.shape = list(output_shape)
            output_tensor.shape_signature = list(output_shape)
            repaired += 1
            changed = True
            break
        if not changed:
            break

    return {"repaired_mixed_nhwc_inputs_for_nchw_concat": int(repaired)}
