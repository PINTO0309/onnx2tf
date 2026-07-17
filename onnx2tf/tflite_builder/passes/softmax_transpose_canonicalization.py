from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _permute_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes.terminal_softmax_layout import (
    _SOFTMAX_NHWC_PROPAGATED_MARKER,
)


def _canonicalize_softmax_transpose_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Canonicalize ONNX-driven transpose chains around SOFTMAX to expose removable pairs.

    Target template:
      T0(0,3,1,2) -> T1(0,3,2,1) -> SOFTMAX -> T2(0,3,2,1)

    Rewritten to:
      T0(0,3,1,2) -> T1(0,2,3,1) -> SOFTMAX -> T2(0,3,1,2)

    Then T0/T1 become inverse pairs and are removed by `_optimize_layout_transpose_chains`.
    """
    rewritten = 0
    softmax_nhwc_propagated_marker = _SOFTMAX_NHWC_PROPAGATED_MARKER
    rank4_perm_nhwc_to_nchw = [0, 3, 1, 2]
    rank4_perm_nchw_to_nhwc = [0, 2, 3, 1]
    rank4_perm_nchw_to_nwhc = [0, 3, 2, 1]

    def _rank4_metadata(
        tensor: Optional[TensorIR],
    ) -> Optional[Tuple[List[int], List[int]]]:
        if tensor is None:
            return None
        try:
            shape = [int(value) for value in tensor.shape]
            signature = (
                [int(value) for value in tensor.shape_signature]
                if tensor.shape_signature is not None
                else list(shape)
            )
        except (TypeError, ValueError):
            return None
        if len(shape) != 4 or len(signature) != 4:
            return None
        return shape, signature

    def _has_last_axis_softmax(options: Any) -> bool:
        if not isinstance(options, dict) or "axis" not in options:
            return True
        raw_axis = options.get("axis")
        if isinstance(raw_axis, bool) or not isinstance(
            raw_axis,
            (int, np.integer),
        ):
            return False
        axis = int(raw_axis)
        if axis < 0:
            axis += 4
        return axis == 3

    reserved_tensor_names = {
        str(name)
        for name in (
            list(model_ir.tensors)
            + list(model_ir.inputs)
            + list(model_ir.outputs)
            + [
                value
                for operator in model_ir.operators
                for value in list(operator.inputs) + list(operator.outputs)
            ]
        )
    }

    def _unique_tensor_name(base: str, reserved_names: set[str]) -> str:
        name = str(base)
        suffix = 1
        while name in reserved_names:
            name = f"{base}_{suffix}"
            suffix += 1
        reserved_names.add(name)
        return name

    def _plan_transpose_perm_for_op(
        op_idx: int,
        new_perm: List[int],
        graph_index: ModelIRGraphIndex,
        public_inputs: set[str],
        public_outputs: set[str],
        reserved_names: set[str],
    ) -> Optional[Dict[str, Any]]:
        op = model_ir.operators[int(op_idx)]
        if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 2:
            return None
        perm_name = str(op.inputs[1])
        perm_tensor = model_ir.tensors.get(perm_name, None)
        if (
            perm_tensor is None
            or perm_name in public_inputs
            or bool(perm_tensor.is_variable)
            or str(perm_tensor.dtype) != "INT32"
            or perm_tensor.quantization is not None
            or perm_tensor.data is None
        ):
            return None
        try:
            perm_array = np.asarray(perm_tensor.data)
        except Exception:
            return None
        if perm_array.dtype != np.dtype(np.int32):
            return None
        perm_users = [int(value) for value in graph_index.consumer_indices(perm_name)]
        perm_data = np.asarray(new_perm, dtype=np.int32)

        # Shared and public-output constants need private clones. Public inputs
        # remain ineligible because their runtime value can differ from data.
        if perm_users != [int(op_idx)] or perm_name in public_outputs:
            new_perm_name = _unique_tensor_name(
                f"{perm_name}_canon",
                reserved_names,
            )
            return {
                "mode": "clone",
                "op_idx": int(op_idx),
                "perm_name": perm_name,
                "new_perm_name": new_perm_name,
                "perm_data": perm_data,
            }
        return {
            "mode": "update",
            "op_idx": int(op_idx),
            "perm_name": perm_name,
            "new_perm_name": None,
            "perm_data": perm_data,
        }

    def _commit_transpose_perm_plan(
        plan: Dict[str, Any],
        graph_index: ModelIRGraphIndex,
    ) -> None:
        perm_data = np.asarray(plan["perm_data"], dtype=np.int32)
        if str(plan["mode"]) == "clone":
            new_perm_name = str(plan["new_perm_name"])
            model_ir.tensors[new_perm_name] = TensorIR(
                name=new_perm_name,
                dtype="INT32",
                shape=[int(perm_data.size)],
                shape_signature=[int(perm_data.size)],
                data=perm_data,
                is_variable=False,
                quantization=None,
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=model_ir.operators[int(plan["op_idx"])],
                input_index=1,
                new_input_name=new_perm_name,
                graph_index=graph_index,
            )
            return
        perm_tensor = model_ir.tensors[str(plan["perm_name"])]
        perm_tensor.data = perm_data
        perm_tensor.dtype = "INT32"
        perm_tensor.shape = [int(perm_data.size)]
        perm_tensor.shape_signature = [int(perm_data.size)]

    graph_index = ModelIRGraphIndex(model_ir)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_boundaries = public_inputs | public_outputs
    while True:
        changed = False

        for softmax_idx in graph_index.operator_indices("SOFTMAX"):
            softmax_op = model_ir.operators[int(softmax_idx)]
            if str(softmax_op.op_type) != "SOFTMAX" or len(softmax_op.inputs) != 1 or len(softmax_op.outputs) != 1:
                continue
            if not _has_last_axis_softmax(softmax_op.options):
                continue
            softmax_in = str(softmax_op.inputs[0])
            softmax_out = str(softmax_op.outputs[0])
            if softmax_in in public_boundaries or softmax_out in public_boundaries:
                continue
            if (
                softmax_in in graph_index.duplicate_producers
                or softmax_out in graph_index.duplicate_producers
            ):
                continue

            pre_idx = graph_index.producers.get(softmax_in, None)
            if pre_idx is None or int(pre_idx) >= int(softmax_idx):
                continue
            pre_op = model_ir.operators[int(pre_idx)]
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if str(pre_op.outputs[0]) != softmax_in:
                continue
            if _read_transpose_perm(model_ir, pre_op) != rank4_perm_nchw_to_nwhc:
                continue
            pre_users = graph_index.consumer_indices(softmax_in)
            if pre_users != [int(softmax_idx)]:
                continue

            pre_input = str(pre_op.inputs[0])
            if (
                pre_input in public_inputs
                or pre_input in graph_index.duplicate_producers
            ):
                continue
            pre_prev_idx = graph_index.producers.get(pre_input, None)
            if pre_prev_idx is None or int(pre_prev_idx) >= int(pre_idx):
                continue
            pre_prev_op = model_ir.operators[int(pre_prev_idx)]
            if (
                str(pre_prev_op.op_type) != "TRANSPOSE"
                or len(pre_prev_op.inputs) < 2
                or len(pre_prev_op.outputs) != 1
                or str(pre_prev_op.outputs[0]) != pre_input
            ):
                continue
            if _read_transpose_perm(model_ir, pre_prev_op) != rank4_perm_nhwc_to_nchw:
                continue
            pre_prev_users = graph_index.consumer_indices(pre_input)
            if pre_prev_users != [int(pre_idx)]:
                continue

            post_users = graph_index.consumer_indices(softmax_out)
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            if int(post_idx) <= int(softmax_idx):
                continue
            post_op = model_ir.operators[int(post_idx)]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != softmax_out:
                continue
            if _read_transpose_perm(model_ir, post_op) != rank4_perm_nchw_to_nwhc:
                continue
            post_out_name = str(post_op.outputs[0])
            if (
                post_out_name in public_inputs
                or post_out_name in graph_index.duplicate_producers
            ):
                continue
            # Terminal graph outputs are safe as long as they are not internally consumed.
            # This enables removing redundant pre-softmax transpose pairs while preserving
            # the final output layout adapter.
            if post_out_name in public_outputs and graph_index.consumer_indices(post_out_name):
                continue

            # Avoid qdim remapping complexity for per-axis activation quantization.
            pre_in_tensor = model_ir.tensors.get(pre_input, None)
            pre_out_tensor = model_ir.tensors.get(softmax_in, None)
            softmax_out_tensor = model_ir.tensors.get(softmax_out, None)
            post_out_tensor = model_ir.tensors.get(post_out_name, None)
            metadata = [
                _rank4_metadata(tensor)
                for tensor in (
                    pre_in_tensor,
                    pre_out_tensor,
                    softmax_out_tensor,
                    post_out_tensor,
                )
            ]
            if any(value is None for value in metadata):
                continue
            if not _all_per_tensor_quantized(
                [
                    pre_in_tensor,
                    pre_out_tensor,
                    softmax_out_tensor,
                    post_out_tensor,
                ]
            ):
                continue
            pre_input_metadata = metadata[0]
            if pre_input_metadata is None:
                continue
            pre_input_shape, pre_input_signature = pre_input_metadata
            new_softmax_shape = _permute_shape(
                pre_input_shape,
                rank4_perm_nchw_to_nhwc,
            )
            new_softmax_signature = _permute_shape(
                pre_input_signature,
                rank4_perm_nchw_to_nhwc,
            )
            if new_softmax_shape is None or new_softmax_signature is None:
                continue
            new_post_shape = _permute_shape(
                new_softmax_shape,
                rank4_perm_nhwc_to_nchw,
            )
            new_post_signature = _permute_shape(
                new_softmax_signature,
                rank4_perm_nhwc_to_nchw,
            )
            if new_post_shape is None or new_post_signature is None:
                continue

            candidate_reserved_names = set(reserved_tensor_names)
            pre_perm_plan = _plan_transpose_perm_for_op(
                int(pre_idx),
                rank4_perm_nchw_to_nhwc,
                graph_index,
                public_inputs,
                public_outputs,
                candidate_reserved_names,
            )
            post_perm_plan = _plan_transpose_perm_for_op(
                int(post_idx),
                rank4_perm_nhwc_to_nchw,
                graph_index,
                public_inputs,
                public_outputs,
                candidate_reserved_names,
            )
            if pre_perm_plan is None or post_perm_plan is None:
                continue
            softmax_opts = (
                dict(softmax_op.options)
                if isinstance(softmax_op.options, dict)
                else {}
            )
            softmax_opts[softmax_nhwc_propagated_marker] = True

            # Every topology, ownership, axis, quantization, metadata, name,
            # and option guard is complete before the first mutation.
            reserved_tensor_names.update(candidate_reserved_names)
            _commit_transpose_perm_plan(pre_perm_plan, graph_index)
            _commit_transpose_perm_plan(post_perm_plan, graph_index)

            assert pre_out_tensor is not None
            assert softmax_out_tensor is not None
            assert post_out_tensor is not None
            pre_out_tensor.shape = list(new_softmax_shape)
            pre_out_tensor.shape_signature = list(new_softmax_signature)
            softmax_out_tensor.shape = list(new_softmax_shape)
            softmax_out_tensor.shape_signature = list(new_softmax_signature)
            post_out_tensor.shape = list(new_post_shape)
            post_out_tensor.shape_signature = list(new_post_signature)
            softmax_op.options = softmax_opts

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"canonicalized_softmax_transpose_chains": int(rewritten)}
