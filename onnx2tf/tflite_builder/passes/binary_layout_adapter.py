from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _replace_operator_input_at,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def repair_rank4_binary_layout_mismatch_with_transpose_adapter(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Insert an input-1 adapter for exact rank-four NHWC/NCHW mismatch."""

    inserted = 0
    binary_ops = {"ADD", "MUL", "SUB", "DIV", "MAXIMUM", "MINIMUM"}
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_nhwc_to_nchw = [0, 3, 1, 2]

    def _make_unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        for op_idx, op in enumerate(model_ir.operators):
            op_type = str(op.op_type)
            if (
                op_type not in binary_ops
                or len(op.inputs) != 2
                or len(op.outputs) != 1
            ):
                continue

            in0_name = str(op.inputs[0])
            in1_name = str(op.inputs[1])
            t0 = model_ir.tensors.get(in0_name, None)
            t1 = model_ir.tensors.get(in1_name, None)
            if t0 is None or t1 is None:
                continue
            s0 = [int(v) for v in list(t0.shape)]
            s1 = [int(v) for v in list(t1.shape)]
            if len(s0) != 4 or len(s1) != 4:
                continue
            if any(int(v) <= 0 for v in s0 + s1):
                continue
            if s0 == s1:
                continue

            perm_to_use: Optional[List[int]] = None
            if s0 == [int(s1[idx]) for idx in perm_nchw_to_nhwc]:
                perm_to_use = [int(v) for v in list(perm_nchw_to_nhwc)]
            elif s0 == [int(s1[idx]) for idx in perm_nhwc_to_nchw]:
                perm_to_use = [int(v) for v in list(perm_nhwc_to_nchw)]
            else:
                continue

            perm_name = _make_unique_tensor_name(
                f"{in1_name}_layout_fix_perm"
            )
            adapted_name = _make_unique_tensor_name(f"{in1_name}_layout_fix")
            model_ir.tensors[perm_name] = TensorIR(
                name=perm_name,
                dtype="INT32",
                shape=[4],
                shape_signature=[4],
                data=np.asarray(perm_to_use, dtype=np.int32),
                is_variable=False,
            )
            model_ir.tensors[adapted_name] = TensorIR(
                name=adapted_name,
                dtype=str(t1.dtype),
                shape=[int(v) for v in list(s0)],
                shape_signature=[int(v) for v in list(s0)],
                data=None,
                is_variable=False,
                quantization=_clone_quantization(t1.quantization),
            )
            model_ir.operators.insert(
                int(op_idx),
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[in1_name, perm_name],
                    outputs=[adapted_name],
                ),
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=op,
                input_index=1,
                new_input_name=adapted_name,
            )
            inserted += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"inserted_rank4_binary_layout_fix_transpose": int(inserted)}


def repair_rank4_binary_singleton_broadcast_layout_mismatch(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Repair a singleton-channel NCHW/NHWC rank-four binary mismatch."""

    repaired = 0
    binary_ops = {"ADD", "MUL", "SUB", "DIV", "MAXIMUM", "MINIMUM"}
    perm_nhwc_to_nchw = [0, 3, 1, 2]

    def _make_unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        changed = False
        for op_idx, op in enumerate(model_ir.operators):
            if (
                str(op.op_type) not in binary_ops
                or len(op.inputs) != 2
                or len(op.outputs) != 1
            ):
                continue
            out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
            t0 = model_ir.tensors.get(str(op.inputs[0]), None)
            t1 = model_ir.tensors.get(str(op.inputs[1]), None)
            if out_tensor is None or t0 is None or t1 is None:
                continue
            s0 = [int(v) for v in list(t0.shape)]
            s1 = [int(v) for v in list(t1.shape)]
            sout = [int(v) for v in list(out_tensor.shape)]
            if any(
                len(shape) != 4 or any(int(dim) <= 0 for dim in shape)
                for shape in [s0, s1, sout]
            ):
                continue

            def _singleton_nchw(shape: List[int]) -> bool:
                return int(shape[1]) == 1

            def _already_broadcasts_to_output(
                lhs_shape: List[int],
                rhs_shape: List[int],
                output_shape: List[int],
            ) -> bool:
                try:
                    return list(
                        np.broadcast_shapes(tuple(lhs_shape), tuple(rhs_shape))
                    ) == [int(v) for v in list(output_shape)]
                except Exception:
                    return False

            def _insert_transpose_adapter(
                input_index: int,
                source_name: str,
                source_tensor: TensorIR,
                target_shape: List[int],
            ) -> None:
                perm_name = _make_unique_tensor_name(
                    f"{source_name}_singleton_layout_fix_perm"
                )
                adapted_name = _make_unique_tensor_name(
                    f"{source_name}_singleton_layout_fix"
                )
                model_ir.tensors[perm_name] = TensorIR(
                    name=perm_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                    is_variable=False,
                )
                model_ir.tensors[adapted_name] = TensorIR(
                    name=adapted_name,
                    dtype=str(source_tensor.dtype),
                    shape=[int(v) for v in list(target_shape)],
                    shape_signature=[int(v) for v in list(target_shape)],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(source_tensor.quantization),
                )
                model_ir.operators.insert(
                    int(op_idx),
                    OperatorIR(
                        op_type="TRANSPOSE",
                        inputs=[str(source_name), str(perm_name)],
                        outputs=[str(adapted_name)],
                    ),
                )
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=op,
                    input_index=int(input_index),
                    new_input_name=str(adapted_name),
                )

            def _insert_reshape_adapter(
                input_index: int,
                source_name: str,
                source_tensor: TensorIR,
                target_shape: List[int],
            ) -> None:
                shape_name = _make_unique_tensor_name(
                    f"{source_name}_singleton_layout_fix_shape"
                )
                adapted_name = _make_unique_tensor_name(
                    f"{source_name}_singleton_layout_fix"
                )
                model_ir.tensors[shape_name] = TensorIR(
                    name=str(shape_name),
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(target_shape, dtype=np.int32),
                    is_variable=False,
                )
                model_ir.tensors[adapted_name] = TensorIR(
                    name=str(adapted_name),
                    dtype=str(source_tensor.dtype),
                    shape=[int(v) for v in list(target_shape)],
                    shape_signature=[int(v) for v in list(target_shape)],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(source_tensor.quantization),
                )
                model_ir.operators.insert(
                    int(op_idx),
                    OperatorIR(
                        op_type="RESHAPE",
                        inputs=[str(source_name), str(shape_name)],
                        outputs=[str(adapted_name)],
                        options={
                            "newShape": [int(v) for v in list(target_shape)]
                        },
                    ),
                )
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=op,
                    input_index=int(input_index),
                    new_input_name=str(adapted_name),
                )

            if (
                _singleton_nchw(s0)
                and s1[0] == s0[0]
                and s1[1] == s0[2]
                and s1[2] == s0[3]
                and sout
                == [int(s0[0]), int(s1[3]), int(s0[2]), int(s0[3])]
            ):
                if _already_broadcasts_to_output(s0, s1, sout):
                    continue
                _insert_transpose_adapter(1, str(op.inputs[1]), t1, sout)
                repaired += 1
                changed = True
                break
            if (
                _singleton_nchw(s1)
                and s0[0] == s1[0]
                and s0[1] == s1[2]
                and s0[2] == s1[3]
                and sout
                == [int(s1[0]), int(s0[3]), int(s1[2]), int(s1[3])]
            ):
                if _already_broadcasts_to_output(s0, s1, sout):
                    continue
                _insert_transpose_adapter(0, str(op.inputs[0]), t0, sout)
                repaired += 1
                changed = True
                break
            if (
                _singleton_nchw(s0)
                and s1[0] == s0[0]
                and s1[1] == s0[2]
                and s1[2] == s0[3]
                and sout == s1
            ):
                if _already_broadcasts_to_output(s0, s1, sout):
                    continue
                _insert_reshape_adapter(
                    0,
                    str(op.inputs[0]),
                    t0,
                    [int(s0[0]), int(s0[2]), int(s0[3]), 1],
                )
                repaired += 1
                changed = True
                break
            if (
                _singleton_nchw(s1)
                and s0[0] == s1[0]
                and s0[1] == s1[2]
                and s0[2] == s1[3]
                and sout == s0
            ):
                if _already_broadcasts_to_output(s0, s1, sout):
                    continue
                _insert_reshape_adapter(
                    1,
                    str(op.inputs[1]),
                    t1,
                    [int(s1[0]), int(s1[2]), int(s1[3]), 1],
                )
                repaired += 1
                changed = True
                break

        if not changed:
            break

    if repaired > 0:
        _prune_unused_tensors(model_ir)
    return {
        "repaired_rank4_binary_singleton_broadcast_layout_mismatch": int(
            repaired
        )
    }
