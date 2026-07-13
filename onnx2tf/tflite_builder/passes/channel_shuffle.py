from __future__ import annotations

from typing import Dict

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """
    Collapse NCHW channel-shuffle blocks into a single GATHER(axis=1).

    Target:
      x_nchw
        -> RESHAPE([N,g,cpg,H,W])
        -> TRANSPOSE([0,2,1,3,4])
        -> RESHAPE([N,C,H,W]) -> y_nchw

    Rewrite:
      x_nchw -> GATHER(axis=1, shuffle_indices) -> y_nchw

    where C=g*cpg and shuffle_indices[k] = (k % g) * cpg + (k // g).
    """
    optimized = 0
    perm_shuffle_swap = [0, 2, 1, 3, 4]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for r1_idx, r1_op in enumerate(model_ir.operators):
            if str(r1_op.op_type) != "RESHAPE" or len(r1_op.inputs) < 1 or len(r1_op.outputs) != 1:
                continue
            x_nchw_name = str(r1_op.inputs[0])
            r1_out_name = str(r1_op.outputs[0])

            r1_users = [int(v) for v in consumers.get(r1_out_name, [])]
            if len(r1_users) != 1:
                continue
            t1_idx = int(r1_users[0])
            t1_op = model_ir.operators[int(t1_idx)]
            if (
                str(t1_op.op_type) != "TRANSPOSE"
                or len(t1_op.inputs) < 2
                or len(t1_op.outputs) != 1
                or str(t1_op.inputs[0]) != r1_out_name
                or _read_transpose_perm(model_ir, t1_op) != perm_shuffle_swap
            ):
                continue
            t1_out_name = str(t1_op.outputs[0])

            t1_users = [int(v) for v in consumers.get(t1_out_name, [])]
            if len(t1_users) != 1:
                continue
            r2_idx = int(t1_users[0])
            r2_op = model_ir.operators[int(r2_idx)]
            if (
                str(r2_op.op_type) != "RESHAPE"
                or len(r2_op.inputs) < 1
                or len(r2_op.outputs) != 1
                or str(r2_op.inputs[0]) != t1_out_name
            ):
                continue
            y_nchw_name = str(r2_op.outputs[0])

            x_tensor = model_ir.tensors.get(x_nchw_name, None)
            r1_tensor = model_ir.tensors.get(r1_out_name, None)
            t1_tensor = model_ir.tensors.get(t1_out_name, None)
            y_tensor = model_ir.tensors.get(y_nchw_name, None)
            if x_tensor is None or r1_tensor is None or t1_tensor is None or y_tensor is None:
                continue

            x_shape = [int(v) for v in list(x_tensor.shape)]
            r1_shape = [int(v) for v in list(r1_tensor.shape)]
            t1_shape = [int(v) for v in list(t1_tensor.shape)]
            y_shape = [int(v) for v in list(y_tensor.shape)]
            if (
                not _is_fully_known_positive_shape(x_shape)
                or not _is_fully_known_positive_shape(r1_shape)
                or not _is_fully_known_positive_shape(t1_shape)
                or not _is_fully_known_positive_shape(y_shape)
            ):
                continue
            if len(x_shape) != 4 or len(r1_shape) != 5 or len(t1_shape) != 5 or len(y_shape) != 4:
                continue

            n, c, h, w = [int(v) for v in x_shape]
            groups = int(r1_shape[1])
            cpg = int(r1_shape[2])
            if (
                int(groups) <= 1
                or int(cpg) <= 0
                or int(groups * cpg) != int(c)
                or int(r1_shape[0]) != int(n)
                or int(r1_shape[3]) != int(h)
                or int(r1_shape[4]) != int(w)
                or int(t1_shape[0]) != int(n)
                or int(t1_shape[1]) != int(cpg)
                or int(t1_shape[2]) != int(groups)
                or int(t1_shape[3]) != int(h)
                or int(t1_shape[4]) != int(w)
                or [int(v) for v in list(y_shape)] != [int(n), int(c), int(h), int(w)]
            ):
                continue

            shuffle_indices = np.asarray(
                [int((k % groups) * cpg + (k // groups)) for k in range(int(c))],
                dtype=np.int32,
            )
            if np.array_equal(shuffle_indices, np.arange(int(c), dtype=np.int32)):
                continue

            gather_idx_name = _unique_tensor_name(f"{x_nchw_name}_shuffle_indices_nchw")
            model_ir.tensors[gather_idx_name] = TensorIR(
                name=gather_idx_name,
                dtype="INT32",
                shape=[int(c)],
                shape_signature=[int(c)],
                data=np.asarray(shuffle_indices, dtype=np.int32),
                is_variable=False,
            )

            r2_op.op_type = "GATHER"
            r2_op.version = 1
            _set_operator_inputs(
                model_ir=model_ir,
                op=r2_op,
                new_inputs=[x_nchw_name, gather_idx_name],
            )
            r2_op.options = {"axis": 1, "batchDims": 0}

            for remove_idx in sorted([int(r1_idx), int(t1_idx)], reverse=True):
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {"optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather": int(optimized)}


