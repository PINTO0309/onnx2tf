from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def replace_expand_dims_and_squeeze_with_reshape(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Replace EXPAND_DIMS/SQUEEZE with RESHAPE for LiteRT.js WebGPU compatibility.

    The replacement uses output tensor metadata as target shape. When possible,
    a single unknown dimension from shape_signature is preserved as -1.
    """
    rewritten = 0
    shape_tensors_created = 0
    pre_ops_by_index: Dict[int, List[OperatorIR]] = {}

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 0
        while name in model_ir.tensors:
            suffix += 1
            name = f"{base}_{suffix}"
        return name

    def _pick_dynamic_axis_for_reshape_target(
        *,
        op: OperatorIR,
        input_name: str,
        output_shape: List[int],
    ) -> Optional[int]:
        if len(output_shape) == 0:
            return None
        input_tensor = model_ir.tensors.get(input_name, None)
        if input_tensor is None:
            return None
        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else [int(v) for v in list(input_tensor.shape)]
        )
        dynamic_input_axes = [
            int(i) for i, dim in enumerate(input_signature) if int(dim) < 0
        ]
        if len(dynamic_input_axes) == 0:
            return None

        op_type = str(op.op_type)
        if op_type == "EXPAND_DIMS" and len(op.inputs) >= 2:
            axis_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            axis_values = _read_const_ints_from_tensor(axis_tensor)
            if axis_values is not None and len(axis_values) > 0:
                out_rank = len(output_shape)
                axis = int(axis_values[0])
                if axis < 0:
                    axis += int(out_rank)
                if 0 <= axis < out_rank:
                    mapped_axes: List[int] = []
                    for in_axis in dynamic_input_axes:
                        out_axis = (
                            int(in_axis)
                            if int(in_axis) < axis
                            else int(in_axis) + 1
                        )
                        if 0 <= out_axis < out_rank:
                            mapped_axes.append(int(out_axis))
                    if len(mapped_axes) > 0:
                        for out_axis in mapped_axes:
                            if int(output_shape[out_axis]) > 1:
                                return int(out_axis)
                        return int(mapped_axes[0])

        for axis, dim in enumerate(output_shape):
            if int(dim) > 1:
                return int(axis)
        return 0

    for op_index, op in enumerate(model_ir.operators):
        op_type = str(op.op_type)
        if op_type not in {"EXPAND_DIMS", "SQUEEZE"}:
            continue
        if len(op.inputs) < 1 or len(op.outputs) != 1:
            continue

        input_name = str(op.inputs[0])
        output_name = str(op.outputs[0])
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        if (
            input_tensor is None
            or output_tensor is None
            or output_tensor.shape is None
        ):
            continue

        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else [int(v) for v in list(input_tensor.shape)]
        )
        output_shape = [int(v) for v in list(output_tensor.shape)]
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_shape)]
        )
        squeeze_dims_for_reshape: Optional[List[int]] = None
        expand_axis_for_reshape: Optional[int] = None
        if op_type == "EXPAND_DIMS" and len(op.inputs) >= 2:
            axis_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            axis_values = _read_const_ints_from_tensor(axis_tensor)
            if axis_values is not None and len(axis_values) > 0:
                expand_axis_for_reshape = int(axis_values[0])
        if op_type == "SQUEEZE":
            squeeze_dims_raw = op.options.get("squeezeDims", [])
            try:
                squeeze_dims_for_reshape = [
                    int(v)
                    for v in np.asarray(squeeze_dims_raw).reshape(-1).tolist()
                ]
            except Exception:
                squeeze_dims_for_reshape = []

        statically_non_singleton_squeeze = False
        if op_type == "SQUEEZE" and squeeze_dims_for_reshape:
            input_shape = [int(v) for v in list(input_tensor.shape)]
            normalized_squeeze_dims: List[int] = []
            for squeeze_axis in squeeze_dims_for_reshape:
                normalized_axis = int(squeeze_axis)
                if normalized_axis < 0:
                    normalized_axis += len(input_shape)
                if 0 <= normalized_axis < len(input_shape):
                    normalized_squeeze_dims.append(int(normalized_axis))
            statically_non_singleton_squeeze = any(
                int(input_shape[axis]) > 1
                for axis in normalized_squeeze_dims
            )

        if statically_non_singleton_squeeze and len(output_shape) > 0:
            # A flattened If executes both branches. A Squeeze that is valid in
            # its selected branch can therefore receive a non-singleton axis
            # while its enclosing branch is inactive. Keep the valid-path shape,
            # but let one retained dimension absorb the inactive-path extent so
            # LiteRT can safely execute the speculative branch.
            dynamic_axis = next(
                (
                    int(axis)
                    for axis, dim in enumerate(output_shape)
                    if int(dim) == 1
                ),
                0,
            )
            reshape_target = [int(v) for v in list(output_shape)]
            reshape_target[int(dynamic_axis)] = -1
            shape_name = _unique_tensor_name(f"{output_name}_reshape_shape")
            model_ir.tensors[shape_name] = TensorIR(
                name=shape_name,
                dtype="INT32",
                shape=[int(len(reshape_target))],
                shape_signature=[int(len(reshape_target))],
                data=np.asarray(reshape_target, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
            output_tensor.shape_signature = [int(v) for v in reshape_target]
            op.op_type = "RESHAPE"
            op.inputs = [input_name, shape_name]
            op.options = {
                "newShape": [int(v) for v in reshape_target],
                "onnxSqueezeDims": [
                    int(v) for v in list(squeeze_dims_for_reshape)
                ],
                "preserveSemanticRank": True,
                "preserveDynamicShape": True,
                "speculativeBranchSafe": True,
            }
            rewritten += 1
            shape_tensors_created += 1
            continue

        if (
            op_type == "SQUEEZE"
            and len(output_shape) > 0
            and all(int(v) == 1 for v in output_shape)
        ):
            # Conservative guard:
            # all-ones squeeze outputs are frequently metadata-ambiguous in
            # dynamic coordinate paths (e.g. GridSample auxiliaries). Rewriting
            # to RESHAPE can hard-code "1" and break runtime element counts.
            continue
        if op_type == "EXPAND_DIMS" and (
            any(int(v) < 0 for v in input_signature)
            or any(int(v) < 0 for v in output_signature)
        ):
            # EXPAND_DIMS can preserve multiple dynamic axes, while static
            # RESHAPE targets can represent at most one unknown axis reliably.
            # Keep EXPAND_DIMS to avoid hard-coding dynamic extents.
            continue
        if op_type == "SQUEEZE" and (
            any(int(v) < 0 for v in input_signature)
            or any(int(v) < 0 for v in output_signature)
        ):
            # Dynamic SQUEEZE can still be rewritten with runtime shape plumbing:
            #   SHAPE(x) -> GATHER(kept_axes) -> RESHAPE(x, gathered_shape)
            squeeze_dims_raw = op.options.get("squeezeDims", [])
            squeeze_dims = (
                [
                    int(v)
                    for v in np.asarray(squeeze_dims_raw).reshape(-1).tolist()
                ]
                if squeeze_dims_raw is not None
                else []
            )
            input_rank = len(list(input_signature))
            if input_rank <= 0 or len(squeeze_dims) == 0:
                # squeezeDims=[] semantics are data-dependent and cannot be
                # represented as a rank-stable RESHAPE.
                continue
            normalized_axes: List[int] = []
            valid_axes = True
            for axis in squeeze_dims:
                norm_axis = int(axis)
                if norm_axis < 0:
                    norm_axis += int(input_rank)
                if norm_axis < 0 or norm_axis >= int(input_rank):
                    valid_axes = False
                    break
                if norm_axis not in normalized_axes:
                    normalized_axes.append(int(norm_axis))
            if not valid_axes:
                continue

            kept_axes = [
                int(axis)
                for axis in range(input_rank)
                if int(axis) not in set(int(v) for v in normalized_axes)
            ]
            runtime_shape_name = _unique_tensor_name(
                f"{output_name}_runtime_shape"
            )
            runtime_shape_filtered_name = _unique_tensor_name(
                f"{output_name}_runtime_shape_filtered"
            )
            kept_axes_name = _unique_tensor_name(
                f"{output_name}_squeeze_kept_axes"
            )
            model_ir.tensors[runtime_shape_name] = TensorIR(
                name=runtime_shape_name,
                dtype="INT32",
                shape=[int(input_rank)],
                shape_signature=[int(input_rank)],
                data=None,
                is_variable=False,
                quantization=None,
            )
            model_ir.tensors[kept_axes_name] = TensorIR(
                name=kept_axes_name,
                dtype="INT32",
                shape=[int(len(kept_axes))],
                shape_signature=[int(len(kept_axes))],
                data=np.asarray(
                    [int(v) for v in list(kept_axes)], dtype=np.int32
                ),
                is_variable=False,
                quantization=None,
            )
            model_ir.tensors[runtime_shape_filtered_name] = TensorIR(
                name=runtime_shape_filtered_name,
                dtype="INT32",
                shape=[int(len(kept_axes))],
                shape_signature=[int(len(kept_axes))],
                data=None,
                is_variable=False,
                quantization=None,
            )
            pre_ops = pre_ops_by_index.setdefault(int(op_index), [])
            pre_ops.append(
                OperatorIR(
                    op_type="SHAPE",
                    inputs=[input_name],
                    outputs=[runtime_shape_name],
                    options={"outType": "INT32"},
                )
            )
            pre_ops.append(
                OperatorIR(
                    op_type="GATHER",
                    inputs=[runtime_shape_name, kept_axes_name],
                    outputs=[runtime_shape_filtered_name],
                    options={"axis": 0, "batchDims": 0},
                )
            )
            op.op_type = "RESHAPE"
            op.inputs = [input_name, runtime_shape_filtered_name]
            op.options = {
                "newShape": [],
                "onnxSqueezeDims": [int(v) for v in normalized_axes],
                "preserveSemanticRank": True,
            }
            rewritten += 1
            shape_tensors_created += 1
            continue

        if len(output_signature) == 0:
            reshape_target = []
        else:
            reshape_target = [int(v) for v in list(output_shape)]
            if len(output_signature) == len(output_shape):
                if all(int(v) > 0 for v in output_signature):
                    reshape_target = [int(v) for v in list(output_signature)]
                else:
                    negative_count = sum(
                        1 for dim in output_signature if int(dim) < 0
                    )
                    if negative_count == 1:
                        reshape_target = [
                            int(dim) if int(dim) > 0 else -1
                            for dim in output_signature
                        ]

        preserve_dynamic_shape = False
        if len(reshape_target) > 0 and all(
            int(v) > 0 for v in reshape_target
        ):
            dynamic_axis = _pick_dynamic_axis_for_reshape_target(
                op=op,
                input_name=input_name,
                output_shape=reshape_target,
            )
            if (
                dynamic_axis is not None
                and 0 <= int(dynamic_axis) < len(reshape_target)
            ):
                reshape_target[int(dynamic_axis)] = -1
                preserve_dynamic_shape = True
                if output_tensor is not None and len(output_tensor.shape) == len(
                    reshape_target
                ):
                    output_tensor.shape_signature = [
                        int(v) for v in list(reshape_target)
                    ]

        shape_name = _unique_tensor_name(f"{output_name}_reshape_shape")
        model_ir.tensors[shape_name] = TensorIR(
            name=shape_name,
            dtype="INT32",
            shape=[int(len(reshape_target))],
            shape_signature=[int(len(reshape_target))],
            data=np.asarray(reshape_target, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )
        op.op_type = "RESHAPE"
        op.inputs = [input_name, shape_name]
        op.options = {"newShape": [int(v) for v in list(reshape_target)]}
        if expand_axis_for_reshape is not None:
            op.options["onnxExpandDimsAxis"] = int(expand_axis_for_reshape)
        if squeeze_dims_for_reshape is not None:
            op.options["onnxSqueezeDims"] = [
                int(v) for v in list(squeeze_dims_for_reshape)
            ]
            op.options["preserveSemanticRank"] = True
        if preserve_dynamic_shape:
            op.options["preserveDynamicShape"] = True
        rewritten += 1
        shape_tensors_created += 1

    if len(pre_ops_by_index) > 0:
        graph_index = ModelIRGraphIndex(model_ir)
        for op_index in sorted(pre_ops_by_index, reverse=True):
            for pre_op in reversed(pre_ops_by_index[op_index]):
                graph_index.insert_operator(int(op_index), pre_op)

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {
        "replaced_expand_dims_and_squeeze_with_reshape": int(rewritten),
        "expand_dims_squeeze_rewrite_shape_tensors": int(
            shape_tensors_created
        ),
    }
