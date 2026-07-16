from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.attention_qkv_reshape_layout import (
    optimize_attention_qkv_had_reshape_transpose_chains as _optimize_attention_qkv_had_reshape_transpose_chains_pass,
)


def optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Reduce QKV-style `RESHAPE -> TRANSPOSE -> RESHAPE` rank adapters.

    Target:
      x[A,1,C] --RESHAPE([A,H,D])--> r
               --TRANSPOSE([1,0,2])--> t
               --RESHAPE([1,H,A,D])--> y
    or
      x[A,1,C] --RESHAPE([A,H,D])--> r
               --TRANSPOSE([1,2,0])--> t
               --RESHAPE([1,H,D,A])--> y

    Rewrite:
      x[A,1,C] --RESHAPE([1,A,H,D])--> r'
               --TRANSPOSE([0,2,1,3] or [0,2,3,1])--> y
    """
    indexed_stats = _optimize_attention_qkv_had_reshape_transpose_chains_pass(
        model_ir,
        graph_index=ModelIRGraphIndex(model_ir),
        layout_state=layout_state,
    )
    rewritten = int(
        indexed_stats.get(
            "optimized_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains",
            0,
        )
    )
    tensors_before_fallback_prune = (
        set(str(name) for name in model_ir.tensors)
        if layout_state is not None
        else set()
    )
    rank3_perm_had = [1, 0, 2]
    rank3_perm_hda = [1, 2, 0]

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        serial = 1
        while name in model_ir.tensors:
            name = f"{base}_{serial}"
            serial += 1
        return name

    def _set_const_input(
        *,
        op: OperatorIR,
        op_idx: int,
        input_index: int,
        consumers: Dict[str, List[int]],
        values: List[int],
        base_suffix: str,
    ) -> bool:
        if int(input_index) >= len(op.inputs):
            return False
        input_name = str(op.inputs[int(input_index)])
        input_tensor = model_ir.tensors.get(input_name, None)
        if input_tensor is None:
            return False
        target_vals = [int(v) for v in list(values)]
        input_users = set(int(v) for v in consumers.get(input_name, []))
        if input_users == {int(op_idx)}:
            return _write_const_ints_to_tensor(input_tensor, target_vals)

        np_dtype = np.int32
        if input_tensor.data is not None:
            try:
                np_dtype = np.asarray(input_tensor.data).dtype
            except Exception:
                np_dtype = np.int32
        cloned_name = _unique_tensor_name(f"{input_name}_{base_suffix}")
        model_ir.tensors[cloned_name] = TensorIR(
            name=cloned_name,
            dtype=str(input_tensor.dtype),
            shape=[int(len(target_vals))],
            shape_signature=[int(len(target_vals))],
            data=np.asarray(target_vals, dtype=np_dtype),
            is_variable=False,
            quantization=_clone_quantization(input_tensor.quantization),
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=op,
            input_index=int(input_index),
            new_input_name=cloned_name,
        )
        return True

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for reshape1_idx, reshape1_op in enumerate(model_ir.operators):
            if str(reshape1_op.op_type) != "RESHAPE" or len(reshape1_op.inputs) < 2 or len(reshape1_op.outputs) != 1:
                continue

            reshape1_in_name = str(reshape1_op.inputs[0])
            reshape1_out_name = str(reshape1_op.outputs[0])
            if reshape1_out_name in model_outputs:
                continue

            reshape1_users = [int(v) for v in consumers.get(reshape1_out_name, [])]
            if len(reshape1_users) != 1:
                continue
            transpose_idx = int(reshape1_users[0])
            transpose_op = model_ir.operators[int(transpose_idx)]
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.inputs) < 2
                or len(transpose_op.outputs) != 1
                or str(transpose_op.inputs[0]) != str(reshape1_out_name)
            ):
                continue

            transpose_perm = _read_transpose_perm(model_ir, transpose_op)
            if transpose_perm not in [rank3_perm_had, rank3_perm_hda]:
                continue

            transpose_out_name = str(transpose_op.outputs[0])
            if transpose_out_name in model_outputs:
                continue
            transpose_users = [int(v) for v in consumers.get(transpose_out_name, [])]
            if len(transpose_users) != 1:
                continue
            reshape2_idx = int(transpose_users[0])
            reshape2_op = model_ir.operators[int(reshape2_idx)]
            if (
                str(reshape2_op.op_type) != "RESHAPE"
                or len(reshape2_op.inputs) < 2
                or len(reshape2_op.outputs) != 1
                or str(reshape2_op.inputs[0]) != str(transpose_out_name)
            ):
                continue

            reshape2_out_name = str(reshape2_op.outputs[0])

            reshape1_in_tensor = model_ir.tensors.get(reshape1_in_name, None)
            reshape1_out_tensor = model_ir.tensors.get(reshape1_out_name, None)
            transpose_out_tensor = model_ir.tensors.get(transpose_out_name, None)
            reshape2_out_tensor = model_ir.tensors.get(reshape2_out_name, None)
            if (
                reshape1_in_tensor is None
                or reshape1_out_tensor is None
                or transpose_out_tensor is None
                or reshape2_out_tensor is None
            ):
                continue
            if (
                not _is_fully_known_positive_shape(reshape1_in_tensor.shape)
                or not _is_fully_known_positive_shape(reshape1_out_tensor.shape)
                or not _is_fully_known_positive_shape(transpose_out_tensor.shape)
                or not _is_fully_known_positive_shape(reshape2_out_tensor.shape)
            ):
                continue

            in_shape = [int(v) for v in list(reshape1_in_tensor.shape)]
            reshape1_shape = [int(v) for v in list(reshape1_out_tensor.shape)]
            transpose_shape = [int(v) for v in list(transpose_out_tensor.shape)]
            reshape2_shape = [int(v) for v in list(reshape2_out_tensor.shape)]
            if len(in_shape) != 3 or len(reshape1_shape) != 3 or len(transpose_shape) != 3 or len(reshape2_shape) != 4:
                continue

            a, b, c = [int(v) for v in list(in_shape)]
            if int(b) != 1:
                continue
            if int(reshape1_shape[0]) != int(a):
                continue
            h = int(reshape1_shape[1])
            d = int(reshape1_shape[2])
            if int(h) <= 0 or int(d) <= 0:
                continue
            if int(h) * int(d) != int(c):
                continue

            if transpose_perm == rank3_perm_had:
                if transpose_shape != [int(h), int(a), int(d)]:
                    continue
                if reshape2_shape != [1, int(h), int(a), int(d)]:
                    continue
                new_transpose_perm = [0, 2, 1, 3]
            else:
                if transpose_shape != [int(h), int(d), int(a)]:
                    continue
                if reshape2_shape != [1, int(h), int(d), int(a)]:
                    continue
                new_transpose_perm = [0, 2, 3, 1]

            new_reshape_shape = [1, int(a), int(h), int(d)]
            if not _set_const_input(
                op=reshape1_op,
                op_idx=int(reshape1_idx),
                input_index=1,
                consumers=consumers,
                values=new_reshape_shape,
                base_suffix="qkv_shape",
            ):
                continue
            if not _set_const_input(
                op=transpose_op,
                op_idx=int(transpose_idx),
                input_index=1,
                consumers=consumers,
                values=new_transpose_perm,
                base_suffix="qkv_perm",
            ):
                continue

            if isinstance(reshape1_op.options, dict):
                reshape_opts = dict(reshape1_op.options)
                for key in ["newShape", "onnxRawNewShape"]:
                    if isinstance(reshape_opts.get(key, None), list):
                        reshape_opts[key] = [int(v) for v in list(new_reshape_shape)]
                reshape1_op.options = reshape_opts

            _set_operator_outputs(
                model_ir=model_ir,
                op=transpose_op,
                new_outputs=[str(reshape2_out_name)],
            )

            reshape1_out_tensor.shape = [int(v) for v in list(new_reshape_shape)]
            reshape1_out_tensor.shape_signature = [int(v) for v in list(new_reshape_shape)]
            reshape2_out_tensor.shape = [int(v) for v in list(reshape2_shape)]
            reshape2_out_tensor.shape_signature = [int(v) for v in list(reshape2_shape)]

            del model_ir.operators[int(reshape2_idx)]

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    if layout_state is not None:
        layout_state.remove(
            tensors_before_fallback_prune - set(str(name) for name in model_ir.tensors)
        )
    return {"optimized_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains": int(rewritten)}
