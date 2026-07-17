from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _clone_quantization,
    _prune_unused_tensors,
    _replace_operator_input_at,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
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


def repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """
    Repair channelwise constants that became layout-misaligned after
    transpose/layout rewrites.

    If a binary op consumes rank-4 runtime data tensor `X` and a rank-3/4
    channelwise constant `C` where:
      - as-is broadcast does not preserve `X` shape
      - NHWC-rotated broadcast preserves `X` shape
    then rotate `C` to NHWC layout. Shared constants are cloned.
    """
    repaired = 0
    binary_ops = {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM", "POW"}
    graph_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    consumer_indices_snapshot = {
        str(tensor_name): tuple(int(index) for index in indices)
        for tensor_name, indices in graph_index.consumers.items()
    }

    def _unique_tensor_name(base_name: str) -> str:
        candidate = str(base_name)
        serial = 0
        while candidate in model_ir.tensors:
            serial += 1
            candidate = f"{base_name}_{serial}"
        return candidate

    def _prefer_runtime_layout_for_rank4_data_tensor(data_tensor_name: str) -> Optional[str]:
        name = str(data_tensor_name)
        data_tensor = model_ir.tensors.get(name, None)
        if data_tensor is not None:
            logical_layout = str(getattr(data_tensor, "logical_layout", "") or "").upper()
            if logical_layout in {"NHWC", "NCHW"}:
                return logical_layout
        lower_name = name.lower()
        if lower_name.endswith("_nhwc"):
            return "NHWC"
        if lower_name.endswith("_nchw"):
            return "NCHW"
        producer_op = graph_index.producer(name)
        if producer_op is None:
            return None
        producer_op_type = str(producer_op.op_type)
        if producer_op_type in {
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "TRANSPOSE_CONV",
            "AVERAGE_POOL_2D",
            "MAX_POOL_2D",
            "RESIZE_BILINEAR",
            "RESIZE_NEAREST_NEIGHBOR",
        }:
            return "NHWC"
        return None

    for operator_index in graph_index.operator_indices_for_types(binary_ops):
        op = model_ir.operators[int(operator_index)]
        if len(op.inputs) != 2:
            continue
        input_names = [str(v) for v in list(op.inputs)]

        for const_input_name in list(input_names):
            const_tensor = model_ir.tensors.get(str(const_input_name), None)
            if const_tensor is None or const_tensor.data is None:
                continue
            const_data = np.asarray(const_tensor.data)
            if int(const_data.ndim) not in {3, 4}:
                continue

            data_input_name = next((name for name in input_names if str(name) != str(const_input_name)), None)
            if data_input_name is None:
                continue
            data_tensor = model_ir.tensors.get(str(data_input_name), None)
            if data_tensor is None or data_tensor.shape is None or len(list(data_tensor.shape)) != 4:
                continue
            data_shape = [int(v) for v in list(data_tensor.shape)]
            if any(int(v) <= 0 for v in data_shape):
                continue

            const_shape = [int(v) for v in list(const_data.shape)]
            as_is_broadcast = _broadcast_static_shapes(data_shape, const_shape)

            rotated_data: Optional[np.ndarray] = None
            if int(const_data.ndim) == 4:
                rotated_data = np.transpose(const_data, axes=[0, 2, 3, 1])
            elif (
                int(const_data.ndim) == 3
                and int(const_shape[0]) > 0
                and int(const_shape[1]) == 1
                and int(const_shape[2]) == 1
            ):
                rotated_data = np.transpose(const_data, axes=[1, 2, 0])
            else:
                continue

            rotated_shape = [int(v) for v in list(rotated_data.shape)]
            rotated_broadcast = _broadcast_static_shapes(data_shape, rotated_shape)
            if (
                (rotated_broadcast is None or rotated_broadcast != data_shape)
                and int(const_data.ndim) == 4
            ):
                # A previous NHWC rewrite can become stale after later shape
                # reconciliation.  Try the exact inverse standard layout
                # permutation and accept it only when it restores the runtime
                # broadcast equation.
                inverse_rotated_data = np.transpose(
                    const_data,
                    axes=[0, 3, 1, 2],
                )
                inverse_rotated_shape = [
                    int(v) for v in list(inverse_rotated_data.shape)
                ]
                inverse_broadcast = _broadcast_static_shapes(
                    data_shape,
                    inverse_rotated_shape,
                )
                if inverse_broadcast == data_shape:
                    rotated_data = inverse_rotated_data
                    rotated_shape = inverse_rotated_shape
                    rotated_broadcast = inverse_broadcast
            if rotated_broadcast is None or rotated_broadcast != data_shape:
                continue
            force_rotate_even_if_ambiguous = False
            if as_is_broadcast == data_shape and int(const_data.ndim) in {3, 4}:
                # Ambiguous case:
                # [1,C,1,1] can broadcast to NHWC [N,H,W,C] when H==C, but this
                # semantically applies scale over height, not channel.
                # If runtime data layout is NHWC, prefer [1,1,1,C].
                preferred_layout = _prefer_runtime_layout_for_rank4_data_tensor(
                    data_input_name
                )
                if (
                    preferred_layout == "NHWC"
                    and int(const_data.ndim) == 4
                    and int(const_shape[0]) == 1
                    and int(const_shape[1]) > 1
                    and int(const_shape[2]) == 1
                    and int(const_shape[3]) == 1
                    and int(rotated_shape[0]) == 1
                    and int(rotated_shape[1]) == 1
                    and int(rotated_shape[2]) == 1
                    and int(rotated_shape[3]) > 1
                ):
                    force_rotate_even_if_ambiguous = True
                elif (
                    preferred_layout == "NHWC"
                    and int(const_data.ndim) == 3
                    and int(const_shape[0]) > 1
                    and int(const_shape[1]) == 1
                    and int(const_shape[2]) == 1
                    and int(rotated_shape[0]) == 1
                    and int(rotated_shape[1]) == 1
                    and int(rotated_shape[2]) == int(const_shape[0])
                    and int(data_shape[3]) == int(const_shape[0])
                ):
                    force_rotate_even_if_ambiguous = True
            if as_is_broadcast == data_shape and not force_rotate_even_if_ambiguous:
                continue

            const_users = set(
                consumer_indices_snapshot.get(str(const_input_name), ())
            )
            if len(const_users) <= 1:
                const_tensor.data = rotated_data.astype(const_data.dtype, copy=False)
                const_tensor.shape = [int(v) for v in list(rotated_shape)]
                const_tensor.shape_signature = [int(v) for v in list(rotated_shape)]
            else:
                clone_name = _unique_tensor_name(f"{const_input_name}__nhwc")
                model_ir.tensors[str(clone_name)] = TensorIR(
                    name=str(clone_name),
                    dtype=str(const_tensor.dtype),
                    shape=[int(v) for v in list(rotated_shape)],
                    shape_signature=[int(v) for v in list(rotated_shape)],
                    data=rotated_data.astype(const_data.dtype, copy=False),
                    is_variable=False,
                    quantization=_clone_quantization(const_tensor.quantization),
                )
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=op,
                    new_inputs=[
                        str(clone_name) if str(v) == str(const_input_name) else str(v)
                        for v in list(op.inputs)
                    ],
                    graph_index=graph_index,
                )

            repaired += 1
            break

    return {"repaired_rank4_channelwise_broadcast_constants": int(repaired)}
