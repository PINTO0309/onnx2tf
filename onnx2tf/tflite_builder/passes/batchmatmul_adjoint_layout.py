from __future__ import annotations

from typing import Dict, List

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _is_fully_known_positive_shape,
    _permute_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def optimize_batchmatmul_transpose_input_to_adj_flags(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Eliminate rank-3 transpose adapters directly feeding BATCH_MATMUL inputs.

    Pattern A:
      x --TRANSPOSE([0,2,1])--> x_t --BATCH_MATMUL(...)
      -> remove transpose and toggle adjX/adjY.

    Pattern B (singleton-preserving pre-reshape):
      x[s,1,c] --TRANSPOSE([1,2,0])--> x_t[1,c,s] --BATCH_MATMUL(...)
      -> rewrite TRANSPOSE to RESHAPE([1,s,c]) and toggle adjX/adjY.
    """
    rewritten = 0

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _is_memory_order_equivalent_by_singletons(
        signature: List[int],
        perm: List[int],
    ) -> bool:
        if len(signature) != 3 or len(perm) != 3:
            return False
        if any(int(v) <= 0 for v in signature):
            return False
        non_singleton_input_axes = [
            int(i) for i, dim in enumerate(signature) if int(dim) != 1
        ]
        non_singleton_permuted_axes = [
            int(i) for i in perm if int(signature[int(i)]) != 1
        ]
        return non_singleton_input_axes == non_singleton_permuted_axes

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for bmm_idx, bmm_op in enumerate(model_ir.operators):
            if str(bmm_op.op_type) != "BATCH_MATMUL" or len(bmm_op.inputs) != 2:
                continue

            bmm_options = (
                dict(bmm_op.options) if isinstance(bmm_op.options, dict) else {}
            )

            rewritten_this_bmm = False
            for input_index in [0, 1]:
                bmm_input_name = str(bmm_op.inputs[int(input_index)])
                transpose_idx = producers.get(str(bmm_input_name), None)
                if transpose_idx is None:
                    continue
                transpose_op = model_ir.operators[int(transpose_idx)]
                if (
                    str(transpose_op.op_type) != "TRANSPOSE"
                    or len(transpose_op.inputs) < 2
                    or len(transpose_op.outputs) != 1
                    or str(transpose_op.outputs[0]) != str(bmm_input_name)
                    or str(bmm_input_name) in model_outputs
                ):
                    continue
                if set(
                    int(v) for v in consumers.get(str(bmm_input_name), [])
                ) != {int(bmm_idx)}:
                    continue

                perm = _read_transpose_perm(model_ir, transpose_op)
                if perm is None or len(list(perm)) != 3:
                    continue

                transpose_input_name = str(transpose_op.inputs[0])
                transpose_input_tensor = model_ir.tensors.get(
                    str(transpose_input_name),
                    None,
                )
                transpose_output_tensor = model_ir.tensors.get(
                    str(bmm_input_name),
                    None,
                )
                if (
                    transpose_input_tensor is None
                    or transpose_output_tensor is None
                    or not _is_fully_known_positive_shape(
                        list(transpose_input_tensor.shape)
                    )
                    or not _is_fully_known_positive_shape(
                        list(transpose_output_tensor.shape)
                    )
                ):
                    continue
                input_shape = [int(v) for v in list(transpose_input_tensor.shape)]
                output_shape = [
                    int(v) for v in list(transpose_output_tensor.shape)
                ]
                if len(input_shape) != 3 or len(output_shape) != 3:
                    continue
                expected_output_shape = _permute_shape(
                    input_shape,
                    [int(v) for v in list(perm)],
                )
                if expected_output_shape is None or [
                    int(v) for v in list(expected_output_shape)
                ] != output_shape:
                    continue

                removed_transpose = False
                if [int(v) for v in list(perm)] == [0, 2, 1]:
                    _replace_operator_input_at(
                        model_ir=model_ir,
                        op=bmm_op,
                        input_index=int(input_index),
                        new_input_name=str(transpose_input_name),
                    )
                    del model_ir.operators[int(transpose_idx)]
                    removed_transpose = True  # noqa: F841
                else:
                    reshape_axes = [int(perm[0]), int(perm[2]), int(perm[1])]
                    if sorted(reshape_axes) != [0, 1, 2]:
                        continue
                    input_signature = (
                        [
                            int(v)
                            for v in list(transpose_input_tensor.shape_signature)
                        ]
                        if transpose_input_tensor.shape_signature is not None
                        else [int(v) for v in list(input_shape)]
                    )
                    if not _is_memory_order_equivalent_by_singletons(
                        input_signature,
                        reshape_axes,
                    ):
                        continue
                    reshape_shape = [
                        int(input_shape[int(axis)]) for axis in reshape_axes
                    ]
                    if [
                        int(reshape_shape[0]),
                        int(reshape_shape[2]),
                        int(reshape_shape[1]),
                    ] != output_shape:
                        continue

                    reshape_shape_name = _unique_tensor_name(
                        f"{bmm_input_name}_reshape_shape"
                    )
                    model_ir.tensors[reshape_shape_name] = TensorIR(
                        name=reshape_shape_name,
                        dtype="INT32",
                        shape=[3],
                        shape_signature=[3],
                        data=np.asarray(
                            [int(v) for v in list(reshape_shape)],
                            dtype=np.int32,
                        ),
                        is_variable=False,
                    )
                    transpose_op.op_type = "RESHAPE"
                    _set_operator_inputs(
                        model_ir=model_ir,
                        op=transpose_op,
                        new_inputs=[str(transpose_input_name), reshape_shape_name],
                    )
                    transpose_op.options = {
                        "newShape": [int(v) for v in list(reshape_shape)]
                    }
                    transpose_output_tensor.shape = [
                        int(v) for v in list(reshape_shape)
                    ]
                    transpose_output_tensor.shape_signature = [
                        int(v) for v in list(reshape_shape)
                    ]

                flag_name = "adjX" if int(input_index) == 0 else "adjY"
                bmm_options[flag_name] = bool(
                    not bool(bmm_options.get(flag_name, False))
                )
                bmm_op.options = bmm_options

                rewritten += 1
                changed = True
                rewritten_this_bmm = True
                break

            if rewritten_this_bmm:
                break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_batchmatmul_transpose_input_to_adj_flags": int(rewritten)}
