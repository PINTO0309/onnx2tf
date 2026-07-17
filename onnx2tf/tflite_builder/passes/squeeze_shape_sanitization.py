from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _is_fully_known_positive_shape,
)
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    _infer_squeeze_output_shape_and_signature,
    _parse_axes_option,
)


def sanitize_squeeze_axes_with_static_input_shapes(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Keep SQUEEZE axes consistent with concrete input metadata.

    Late rewrite passes can leave a SQUEEZE axis targeting a non-singleton
    dimension, which crashes LiteRT during ``allocate_tensors()``. Invalid or
    duplicated axes are removed, non-singleton axes on non-constant inputs are
    repaired to singleton, and constant payloads remain authoritative when
    they prove that an axis cannot be squeezed.
    """

    sanitized_ops = 0
    repaired_input_dims = 0
    updated_output_shapes = 0

    for op in model_ir.operators:
        if str(op.op_type) != "SQUEEZE":
            continue
        if len(op.inputs) < 1 or len(op.outputs) != 1:
            continue

        input_name = str(op.inputs[0])
        output_name = str(op.outputs[0])
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        if input_tensor is None or output_tensor is None:
            continue
        if not _is_fully_known_positive_shape(input_tensor.shape):
            continue

        input_shape = [int(v) for v in list(input_tensor.shape)]
        rank = len(input_shape)
        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            and len(list(input_tensor.shape_signature)) == rank
            else [int(v) for v in list(input_shape)]
        )

        options = dict(op.options) if isinstance(op.options, dict) else {}
        raw_axes = _parse_axes_option(options.get("squeezeDims", []))

        normalized_axes: List[int] = []
        axes_modified = False
        for axis in raw_axes:
            normalized_axis = int(axis)
            if normalized_axis < 0:
                normalized_axis += int(rank)
            if normalized_axis < 0 or normalized_axis >= int(rank):
                axes_modified = True
                continue
            if normalized_axis in normalized_axes:
                axes_modified = True
                continue
            normalized_axes.append(normalized_axis)

        data_shape: Optional[List[int]] = None
        if input_tensor.data is not None:
            try:
                data_arr = np.asarray(input_tensor.data)
                if int(data_arr.ndim) == int(rank):
                    data_shape = [int(v) for v in list(data_arr.shape)]
            except Exception:
                data_shape = None

        sanitized_axes: List[int] = []
        repaired_shape = [int(v) for v in list(input_shape)]
        repaired_signature = [int(v) for v in list(input_signature)]
        for axis in normalized_axes:
            current_dim = int(repaired_shape[axis])
            if current_dim == 1:
                sanitized_axes.append(int(axis))
                if int(repaired_signature[axis]) != 1:
                    repaired_signature[axis] = 1
                continue

            can_force_singleton = (
                data_shape is None
                or int(axis) >= len(data_shape)
                or int(data_shape[axis]) == 1
            )
            if can_force_singleton:
                repaired_shape[axis] = 1
                repaired_signature[axis] = 1
                sanitized_axes.append(int(axis))
                repaired_input_dims += 1
                continue

            axes_modified = True

        input_changed = (
            repaired_shape != [int(v) for v in list(input_tensor.shape)]
            or input_tensor.shape_signature is None
            or [int(v) for v in list(input_tensor.shape_signature)]
            != repaired_signature
        )
        if input_changed:
            input_tensor.shape = [int(v) for v in list(repaired_shape)]
            input_tensor.shape_signature = [
                int(v) for v in list(repaired_signature)
            ]

        if axes_modified or sanitized_axes != raw_axes:
            options["squeezeDims"] = [int(v) for v in sanitized_axes]
            op.options = options
            sanitized_ops += 1

        out_shape, out_signature = _infer_squeeze_output_shape_and_signature(
            input_shape=[int(v) for v in list(input_tensor.shape)],
            input_signature=(
                [int(v) for v in list(input_tensor.shape_signature)]
                if input_tensor.shape_signature is not None
                else [int(v) for v in list(input_tensor.shape)]
            ),
            squeeze_axes=[int(v) for v in list(sanitized_axes)],
        )
        if out_shape is None or out_signature is None:
            continue
        if not _is_fully_known_positive_shape(out_shape):
            continue

        if (
            [int(v) for v in list(output_tensor.shape)]
            != [int(v) for v in list(out_shape)]
            or output_tensor.shape_signature is None
            or [int(v) for v in list(output_tensor.shape_signature)]
            != [int(v) for v in list(out_signature)]
        ):
            output_tensor.shape = [int(v) for v in list(out_shape)]
            output_tensor.shape_signature = [
                int(v) for v in list(out_signature)
            ]
            updated_output_shapes += 1

    return {
        "sanitized_squeeze_axes_with_static_input_shapes": int(sanitized_ops),
        "repaired_squeeze_input_singleton_dims": int(repaired_input_dims),
        "updated_squeeze_output_shapes": int(updated_output_shapes),
    }
