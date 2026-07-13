from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Collapse NCHW adapters around axis=3 CONCAT with const suffixes to a single NHWC->NCHW bridge.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      const_i_nchw --------------------^
      CONCAT(axis=3, [const..., x_nchw, ...]) -> y_nchw
      y_nchw --TRANSPOSE(0,2,3,1)--> y_nhwc   (at least one branch)
      y_nchw may also have legacy NCHW consumers.

    Rewrite:
      const_i_nchw -> const_i_nhwc (constant data permuted)
      CONCAT(axis=2, [const..., x_nhwc, ...]) -> y_nhwc
      remove post TRANSPOSE branches to y_nhwc
      if legacy NCHW consumers exist:
        y_nhwc --TRANSPOSE(0,3,1,2)--> y_nchw_bridge
        legacy consumers read y_nchw_bridge
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]

    def _dims_compatible(a: int, b: int) -> bool:
        if int(a) <= 0 or int(b) <= 0:
            return True
        return int(a) == int(b)

    def _shape_compatible_except_axis2(shape_a: List[int], shape_b: List[int]) -> bool:
        if len(shape_a) != 4 or len(shape_b) != 4:
            return False
        for dim_idx in [0, 1, 3]:
            if not _dims_compatible(int(shape_a[dim_idx]), int(shape_b[dim_idx])):
                return False
        return True

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        suffix = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _op_index(op_ref: OperatorIR) -> Optional[int]:
        for idx, op in enumerate(model_ir.operators):
            if int(id(op)) == int(id(op_ref)):
                return int(idx)
        return None

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.outputs) != 1:
                continue
            concat_out_name = str(concat_op.outputs[0])
            if concat_out_name in model_outputs:
                continue

            concat_out_tensor = model_ir.tensors.get(concat_out_name, None)
            if (
                concat_out_tensor is None
                or concat_out_tensor.shape is None
                or len(list(concat_out_tensor.shape)) != 4
            ):
                continue

            concat_axis = int(concat_op.options.get("axis", 3))
            if concat_axis < 0:
                concat_axis += 4
            if int(concat_axis) != 3:
                continue

            concat_inputs = [str(v) for v in list(concat_op.inputs)]
            if len(concat_inputs) < 2:
                continue

            adapter_input_index: Optional[int] = None
            adapter_pre_idx: Optional[int] = None
            adapter_pre_op: Optional[OperatorIR] = None
            adapter_source_name: Optional[str] = None
            adapter_output_name: Optional[str] = None
            adapter_perm_name: Optional[str] = None

            for input_index, input_name in enumerate(concat_inputs):
                producer_idx = producers.get(str(input_name), None)
                if producer_idx is None:
                    continue
                producer_op = model_ir.operators[int(producer_idx)]
                if (
                    str(producer_op.op_type) == "TRANSPOSE"
                    and len(producer_op.inputs) >= 2
                    and len(producer_op.outputs) == 1
                    and str(producer_op.outputs[0]) == str(input_name)
                    and _read_transpose_perm(model_ir, producer_op) == perm_nhwc_to_nchw
                ):
                    if adapter_pre_idx is not None:
                        adapter_pre_idx = None
                        break
                    adapter_input_index = int(input_index)
                    adapter_pre_idx = int(producer_idx)
                    adapter_pre_op = producer_op
                    adapter_source_name = str(producer_op.inputs[0])
                    adapter_output_name = str(producer_op.outputs[0])
                    adapter_perm_name = str(producer_op.inputs[1])

            if (
                adapter_input_index is None
                or adapter_pre_idx is None
                or adapter_pre_op is None
                or adapter_source_name is None
                or adapter_output_name is None
                or adapter_perm_name is None
            ):
                continue

            adapter_source_tensor = model_ir.tensors.get(str(adapter_source_name), None)
            if (
                adapter_source_tensor is None
                or adapter_source_tensor.shape is None
                or len(list(adapter_source_tensor.shape)) != 4
            ):
                continue
            adapter_shape_nhwc = [int(v) for v in list(adapter_source_tensor.shape)]

            const_rewrites: Dict[str, np.ndarray] = {}
            new_concat_inputs: List[str] = list(concat_inputs)
            new_concat_inputs[int(adapter_input_index)] = str(adapter_source_name)
            valid_inputs = True

            for input_index, input_name in enumerate(concat_inputs):
                if int(input_index) == int(adapter_input_index):
                    continue
                tensor = model_ir.tensors.get(str(input_name), None)
                if (
                    tensor is None
                    or tensor.data is None
                    or tensor.shape is None
                    or len(list(tensor.shape)) != 4
                ):
                    valid_inputs = False
                    break
                # Avoid mutating shared constants used by non-concat consumers.
                if set(int(v) for v in consumers.get(str(input_name), [])) != {int(concat_idx)}:
                    valid_inputs = False
                    break
                data = np.asarray(tensor.data)
                if int(data.ndim) != 4:
                    valid_inputs = False
                    break
                converted = np.transpose(data, axes=perm_nchw_to_nhwc).astype(data.dtype, copy=False)
                if not _shape_compatible_except_axis2(
                    adapter_shape_nhwc,
                    [int(v) for v in list(converted.shape)],
                ):
                    valid_inputs = False
                    break
                const_rewrites[str(input_name)] = np.asarray(converted)
            if not valid_inputs:
                continue

            concat_users = [int(v) for v in consumers.get(concat_out_name, [])]
            if len(concat_users) == 0:
                continue
            post_ops: List[OperatorIR] = []
            post_output_names: List[str] = []
            legacy_ops: List[OperatorIR] = []
            valid_consumers = True
            for user_idx in concat_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == concat_out_name
                    and _read_transpose_perm(model_ir, user_op) == perm_nchw_to_nhwc
                ):
                    post_output_name = str(user_op.outputs[0])
                    if post_output_name in model_outputs:
                        valid_consumers = False
                        break
                    post_ops.append(user_op)
                    post_output_names.append(post_output_name)
                else:
                    legacy_ops.append(user_op)
            if not valid_consumers or len(post_ops) == 0:
                continue

            # Apply constants rewrite and concat axis/layout conversion.
            for input_name, converted in const_rewrites.items():
                tensor = model_ir.tensors.get(str(input_name), None)
                if tensor is None:
                    valid_inputs = False
                    break
                tensor.data = np.asarray(converted)
                tensor.shape = [int(v) for v in list(converted.shape)]
                tensor.shape_signature = [int(v) for v in list(converted.shape)]
            if not valid_inputs:
                continue

            _set_operator_inputs(
                model_ir=model_ir,
                op=concat_op,
                new_inputs=[str(v) for v in new_concat_inputs],
            )
            concat_op.options["axis"] = 2
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(concat_out_name, None),
                perm_nchw_to_nhwc,
            )

            # Bypass all post inverse transposes.
            for post_output_name in post_output_names:
                _replace_tensor_inputs(model_ir, str(post_output_name), concat_out_name)

            # Bridge legacy consumers (if any) back to NCHW once.
            bridge_name: Optional[str] = None
            bridge_insert_before: Optional[int] = None
            if len(legacy_ops) > 0:
                concat_out_tensor_after = model_ir.tensors.get(concat_out_name, None)
                if (
                    concat_out_tensor_after is None
                    or concat_out_tensor_after.shape is None
                    or len(list(concat_out_tensor_after.shape)) != 4
                ):
                    continue
                bridge_shape = _permute_shape(
                    [int(v) for v in list(concat_out_tensor_after.shape)],
                    perm_nhwc_to_nchw,
                )
                bridge_sig_src = (
                    list(concat_out_tensor_after.shape_signature)
                    if concat_out_tensor_after.shape_signature is not None
                    else list(concat_out_tensor_after.shape)
                )
                bridge_signature = _permute_shape(
                    [int(v) for v in list(bridge_sig_src)],
                    perm_nhwc_to_nchw,
                )
                if bridge_shape is None or bridge_signature is None:
                    continue
                bridge_name = _unique_tensor_name(f"{concat_out_name}_nchw_bridge")
                model_ir.tensors[str(bridge_name)] = TensorIR(
                    name=str(bridge_name),
                    dtype=str(concat_out_tensor_after.dtype),
                    shape=[int(v) for v in list(bridge_shape)],
                    shape_signature=[int(v) for v in list(bridge_signature)],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(concat_out_tensor_after.quantization),
                )
                for legacy_op in legacy_ops:
                    legacy_inputs = [
                        str(bridge_name) if str(inp) == str(concat_out_name) else str(inp)
                        for inp in list(legacy_op.inputs)
                    ]
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=legacy_op,
                        new_inputs=legacy_inputs,
                    )

            remove_ops: List[OperatorIR] = list(post_ops)
            adapter_users = [int(v) for v in consumers.get(str(adapter_output_name), [])]
            if set(int(v) for v in adapter_users) == {int(concat_idx)}:
                remove_ops.append(adapter_pre_op)

            remove_indices: List[int] = []
            for remove_op in remove_ops:
                remove_index = _op_index(remove_op)
                if remove_index is not None:
                    remove_indices.append(int(remove_index))
            for remove_index in sorted(set(remove_indices), reverse=True):
                del model_ir.operators[int(remove_index)]

            if bridge_name is not None:
                legacy_indices_after = [
                    _op_index(legacy_op)
                    for legacy_op in legacy_ops
                ]
                legacy_indices_after = [
                    int(v)
                    for v in legacy_indices_after
                    if v is not None
                ]
                if len(legacy_indices_after) == 0:
                    continue
                bridge_insert_before = int(min(legacy_indices_after))
                model_ir.operators.insert(
                    int(bridge_insert_before),
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(concat_out_name), str(adapter_perm_name)],
                        outputs=[str(bridge_name)],
                    ),
                )

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"optimized_transpose_axis3_const_concat_bridge_nhwc_chains": int(optimized)}
