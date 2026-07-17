from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def optimize_transpose_mean_hardsigmoid_muladd_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Eliminate redundant transpose bridges around mixed MEAN and HardSigmoid branches.

    Target pattern (rank-4, quantized):
      q0_raw --TRANSPOSE(0,3,1,2)--> q0_nchw
             --DEQUANTIZE--> ...MEAN(axes=[2,3])... --QUANTIZE--> qm_nchw --TRANSPOSE(0,2,3,1)--> qm_nhwc -> CONV
      q1_raw --TRANSPOSE(0,3,1,2)--> q1_nchw
             --DEQUANTIZE--> MUL(c)->ADD(c)->MAXIMUM(c)->MINIMUM(c) --QUANTIZE--> qsig_nchw
      MUL(q0_nchw, qsig_nchw) -> m
      ADD(q0_nchw, m) -> y

    Rewrite:
    - Bypass both pre-transposes on the two dequantize branches.
    - Convert MEAN axes to NHWC-side and remove post-quantize transpose.
    - Rewire MUL/ADD main inputs to q0_raw (NHWC).
    - Remove the three bridge transposes.
    - Insert NHWC->NCHW adapters only for legacy consumers of `y`.
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_nhwc_to_nchw_const_name = "__nhwc_to_nchw_perm_rank4__"

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _tensor_signature(tensor: TensorIR) -> List[int]:
        if tensor.shape_signature is not None:
            return [int(v) for v in list(tensor.shape_signature)]
        return [int(v) for v in list(tensor.shape)]

    if perm_nhwc_to_nchw_const_name not in model_ir.tensors:
        model_ir.tensors[perm_nhwc_to_nchw_const_name] = TensorIR(
            name=perm_nhwc_to_nchw_const_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)

        for pre0_idx, pre0_op in enumerate(model_ir.operators):
            if str(pre0_op.op_type) != "TRANSPOSE" or len(pre0_op.inputs) < 2 or len(pre0_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre0_op) != perm_nhwc_to_nchw:
                continue

            q0_raw_name = str(pre0_op.inputs[0])
            q0_nchw_name = str(pre0_op.outputs[0])
            if q0_raw_name in model_ir.outputs or q0_nchw_name in model_ir.outputs:
                continue

            q0_users = [int(v) for v in consumers.get(q0_nchw_name, [])]
            if len(q0_users) != 3:
                continue

            dq0_idx: Optional[int] = None
            mul0_idx: Optional[int] = None
            add0_idx: Optional[int] = None
            for u_idx in q0_users:
                u_op = model_ir.operators[int(u_idx)]
                u_type = str(u_op.op_type)
                if u_type == "DEQUANTIZE":
                    if dq0_idx is not None:
                        dq0_idx = None
                        break
                    dq0_idx = int(u_idx)
                elif u_type == "MUL":
                    if mul0_idx is not None:
                        mul0_idx = None
                        break
                    mul0_idx = int(u_idx)
                elif u_type == "ADD":
                    if add0_idx is not None:
                        add0_idx = None
                        break
                    add0_idx = int(u_idx)
            if dq0_idx is None or mul0_idx is None or add0_idx is None:
                continue

            dq0_op = model_ir.operators[int(dq0_idx)]
            if len(dq0_op.inputs) != 1 or len(dq0_op.outputs) != 1 or str(dq0_op.inputs[0]) != q0_nchw_name:
                continue
            dq0_out_name = str(dq0_op.outputs[0])
            if dq0_out_name in model_ir.outputs:
                continue

            mean_users = [int(v) for v in consumers.get(dq0_out_name, [])]
            if len(mean_users) != 1:
                continue
            mean_idx = int(mean_users[0])
            mean_op = model_ir.operators[int(mean_idx)]
            if str(mean_op.op_type) != "MEAN" or len(mean_op.inputs) < 2 or len(mean_op.outputs) != 1:
                continue
            if str(mean_op.inputs[0]) != dq0_out_name or not bool(mean_op.options.get("keepDims", False)):
                continue
            mean_out_name = str(mean_op.outputs[0])
            if mean_out_name in model_ir.outputs:
                continue

            mean_axes_name = str(mean_op.inputs[1])
            mean_axes_tensor = model_ir.tensors.get(mean_axes_name, None)
            mean_axes = _read_const_ints_from_tensor(mean_axes_tensor)
            if mean_axes is None or len(mean_axes) == 0:
                continue

            qmean_users = [int(v) for v in consumers.get(mean_out_name, [])]
            if len(qmean_users) != 1:
                continue
            qmean_idx = int(qmean_users[0])
            qmean_op = model_ir.operators[int(qmean_idx)]
            if str(qmean_op.op_type) != "QUANTIZE" or len(qmean_op.inputs) != 1 or len(qmean_op.outputs) != 1:
                continue
            if str(qmean_op.inputs[0]) != mean_out_name:
                continue
            qmean_out_name = str(qmean_op.outputs[0])
            if qmean_out_name in model_ir.outputs:
                continue

            post_users = [int(v) for v in consumers.get(qmean_out_name, [])]
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[int(post_idx)]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != qmean_out_name:
                continue
            if _read_transpose_perm(model_ir, post_op) != perm_nchw_to_nhwc:
                continue
            post_out_name = str(post_op.outputs[0])
            if post_out_name in model_ir.outputs:
                continue

            # Ensure post-transpose is a pure adapter to downstream ops.
            post_out_users = [int(v) for v in consumers.get(post_out_name, [])]
            if len(post_out_users) == 0:
                continue

            mul0_op = model_ir.operators[int(mul0_idx)]
            add0_op = model_ir.operators[int(add0_idx)]
            if len(mul0_op.inputs) != 2 or len(mul0_op.outputs) != 1:
                continue
            if len(add0_op.inputs) != 2 or len(add0_op.outputs) != 1:
                continue

            mul0_inputs = [str(v) for v in list(mul0_op.inputs)]
            if q0_nchw_name == mul0_inputs[0]:
                mul0_q0_input_idx = 0
                sig_q_name = mul0_inputs[1]
            elif q0_nchw_name == mul0_inputs[1]:
                mul0_q0_input_idx = 1
                sig_q_name = mul0_inputs[0]
            else:
                continue
            if sig_q_name in model_ir.outputs:
                continue

            add0_inputs = [str(v) for v in list(add0_op.inputs)]
            mul0_out_name = str(mul0_op.outputs[0])
            if q0_nchw_name == add0_inputs[0] and mul0_out_name == add0_inputs[1]:
                add0_q0_input_idx = 0
            elif q0_nchw_name == add0_inputs[1] and mul0_out_name == add0_inputs[0]:
                add0_q0_input_idx = 1
            else:
                continue

            sig_q_users = [int(v) for v in consumers.get(sig_q_name, [])]
            if len(sig_q_users) != 1 or int(sig_q_users[0]) != int(mul0_idx):
                continue
            mul0_users = [int(v) for v in consumers.get(mul0_out_name, [])]
            if len(mul0_users) != 1 or int(mul0_users[0]) != int(add0_idx):
                continue

            qsig_prod_idx = producers.get(sig_q_name, None)
            if qsig_prod_idx is None:
                continue
            qsig_op = model_ir.operators[int(qsig_prod_idx)]
            if str(qsig_op.op_type) != "QUANTIZE" or len(qsig_op.inputs) != 1 or len(qsig_op.outputs) != 1:
                continue
            hsig_min_out_name = str(qsig_op.inputs[0])

            hsig_min_idx = producers.get(hsig_min_out_name, None)
            if hsig_min_idx is None:
                continue
            hsig_min_op = model_ir.operators[int(hsig_min_idx)]
            if str(hsig_min_op.op_type) != "MINIMUM" or len(hsig_min_op.inputs) != 2 or len(hsig_min_op.outputs) != 1:
                continue
            hsig_max_out_name = (
                str(hsig_min_op.inputs[0])
                if str(hsig_min_op.inputs[0]) != str(hsig_min_op.inputs[1])
                else str(hsig_min_op.inputs[0])
            )
            # Identify main path input of MINIMUM.
            if producers.get(str(hsig_min_op.inputs[0]), None) is not None:
                hsig_max_out_name = str(hsig_min_op.inputs[0])
            elif producers.get(str(hsig_min_op.inputs[1]), None) is not None:
                hsig_max_out_name = str(hsig_min_op.inputs[1])
            else:
                continue

            hsig_max_idx = producers.get(hsig_max_out_name, None)
            if hsig_max_idx is None:
                continue
            hsig_max_op = model_ir.operators[int(hsig_max_idx)]
            if str(hsig_max_op.op_type) != "MAXIMUM" or len(hsig_max_op.inputs) != 2 or len(hsig_max_op.outputs) != 1:
                continue

            hsig_add_out_name = (
                str(hsig_max_op.inputs[0])
                if producers.get(str(hsig_max_op.inputs[0]), None) is not None
                else str(hsig_max_op.inputs[1])
            )
            hsig_add_idx = producers.get(hsig_add_out_name, None)
            if hsig_add_idx is None:
                continue
            hsig_add_op = model_ir.operators[int(hsig_add_idx)]
            if str(hsig_add_op.op_type) != "ADD" or len(hsig_add_op.inputs) != 2 or len(hsig_add_op.outputs) != 1:
                continue

            hsig_mul_out_name = (
                str(hsig_add_op.inputs[0])
                if producers.get(str(hsig_add_op.inputs[0]), None) is not None
                else str(hsig_add_op.inputs[1])
            )
            hsig_mul_idx = producers.get(hsig_mul_out_name, None)
            if hsig_mul_idx is None:
                continue
            hsig_mul_op = model_ir.operators[int(hsig_mul_idx)]
            if str(hsig_mul_op.op_type) != "MUL" or len(hsig_mul_op.inputs) != 2 or len(hsig_mul_op.outputs) != 1:
                continue

            dq1_out_name = (
                str(hsig_mul_op.inputs[0])
                if producers.get(str(hsig_mul_op.inputs[0]), None) is not None
                else str(hsig_mul_op.inputs[1])
            )
            dq1_idx = producers.get(dq1_out_name, None)
            if dq1_idx is None:
                continue
            dq1_op = model_ir.operators[int(dq1_idx)]
            if str(dq1_op.op_type) != "DEQUANTIZE" or len(dq1_op.inputs) != 1 or len(dq1_op.outputs) != 1:
                continue
            if str(dq1_op.outputs[0]) != dq1_out_name:
                continue
            q1_nchw_name = str(dq1_op.inputs[0])
            if q1_nchw_name in model_ir.outputs:
                continue

            pre1_idx = producers.get(q1_nchw_name, None)
            if pre1_idx is None:
                continue
            pre1_op = model_ir.operators[int(pre1_idx)]
            if str(pre1_op.op_type) != "TRANSPOSE" or len(pre1_op.inputs) < 2 or len(pre1_op.outputs) != 1:
                continue
            if str(pre1_op.outputs[0]) != q1_nchw_name:
                continue
            if _read_transpose_perm(model_ir, pre1_op) != perm_nhwc_to_nchw:
                continue
            q1_raw_name = str(pre1_op.inputs[0])
            if q1_raw_name in model_ir.outputs:
                continue
            q1_users = [int(v) for v in consumers.get(q1_nchw_name, [])]
            if len(q1_users) != 1 or int(q1_users[0]) != int(dq1_idx):
                continue

            q0_raw_tensor = model_ir.tensors.get(q0_raw_name, None)
            q0_nchw_tensor = model_ir.tensors.get(q0_nchw_name, None)
            q1_raw_tensor = model_ir.tensors.get(q1_raw_name, None)
            q1_nchw_tensor = model_ir.tensors.get(q1_nchw_name, None)
            dq0_out_tensor = model_ir.tensors.get(dq0_out_name, None)
            mean_out_tensor = model_ir.tensors.get(mean_out_name, None)
            qmean_out_tensor = model_ir.tensors.get(qmean_out_name, None)
            dq1_out_tensor = model_ir.tensors.get(dq1_out_name, None)
            hsig_mul_out_tensor = model_ir.tensors.get(hsig_mul_out_name, None)
            hsig_add_out_tensor = model_ir.tensors.get(hsig_add_out_name, None)
            hsig_max_out_tensor = model_ir.tensors.get(hsig_max_out_name, None)
            hsig_min_out_tensor = model_ir.tensors.get(hsig_min_out_name, None)
            sig_q_tensor = model_ir.tensors.get(sig_q_name, None)
            mul0_out_tensor = model_ir.tensors.get(mul0_out_name, None)
            add0_out_name = str(add0_op.outputs[0])
            if add0_out_name in model_ir.outputs:
                continue
            add0_out_tensor = model_ir.tensors.get(add0_out_name, None)
            if any(
                t is None
                for t in [
                    q0_raw_tensor,
                    q0_nchw_tensor,
                    q1_raw_tensor,
                    q1_nchw_tensor,
                    dq0_out_tensor,
                    mean_out_tensor,
                    qmean_out_tensor,
                    dq1_out_tensor,
                    hsig_mul_out_tensor,
                    hsig_add_out_tensor,
                    hsig_max_out_tensor,
                    hsig_min_out_tensor,
                    sig_q_tensor,
                    mul0_out_tensor,
                    add0_out_tensor,
                ]
            ):
                continue

            # Avoid per-axis qdim remap complexity in this rewrite.
            if not _all_per_tensor_quantized(
                [
                    q0_raw_tensor,
                    q0_nchw_tensor,
                    q1_raw_tensor,
                    q1_nchw_tensor,
                    qmean_out_tensor,
                    sig_q_tensor,
                    mul0_out_tensor,
                    add0_out_tensor,
                ]
            ):
                continue

            # Validate and commit the axes constant before graph rewiring or
            # dependent metadata mutation so a rejected update remains a no-op.
            rank = len(list(q0_raw_tensor.shape))
            normalized_axes: List[int] = []
            valid_axes = True
            for axis in mean_axes:
                a = int(axis)
                if a < 0:
                    a += rank
                if a < 0 or a >= rank:
                    valid_axes = False
                    break
                normalized_axes.append(int(a))
            if not valid_axes:
                continue
            new_axes = [
                int(perm_nhwc_to_nchw[int(axis)]) for axis in normalized_axes
            ]
            if not _write_const_ints_to_tensor(
                mean_axes_tensor,
                [int(v) for v in new_axes],
            ):
                continue

            # 1) Bypass pre0 transpose on mean branch.
            _set_operator_inputs(
                model_ir=model_ir,
                op=dq0_op,
                new_inputs=[q0_raw_name],
            )
            dq0_out_tensor.shape = [int(v) for v in list(q0_raw_tensor.shape)]
            dq0_out_tensor.shape_signature = _tensor_signature(q0_raw_tensor)

            # 2) Remap mean axes from NCHW to NHWC side.
            mean_shape = [int(v) for v in list(q0_raw_tensor.shape)]
            mean_sig = _tensor_signature(q0_raw_tensor)
            for axis in new_axes:
                if 0 <= int(axis) < len(mean_shape):
                    mean_shape[int(axis)] = 1
                if 0 <= int(axis) < len(mean_sig):
                    mean_sig[int(axis)] = 1
            mean_out_tensor.shape = [int(v) for v in list(mean_shape)]
            mean_out_tensor.shape_signature = [int(v) for v in list(mean_sig)]

            # qmean now follows MEAN (NHWC-side), so remove post transpose.
            qmean_out_tensor.shape = [int(v) for v in list(mean_shape)]
            qmean_out_tensor.shape_signature = [int(v) for v in list(mean_sig)]
            for post_user_idx in post_out_users:
                post_user_op = model_ir.operators[int(post_user_idx)]
                updated_inputs = [
                    str(qmean_out_name) if str(inp) == str(post_out_name) else str(inp)
                    for inp in list(post_user_op.inputs)
                ]
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=post_user_op,
                    new_inputs=updated_inputs,
                )

            # 3) Bypass pre1 transpose on HardSigmoid branch.
            _set_operator_inputs(
                model_ir=model_ir,
                op=dq1_op,
                new_inputs=[q1_raw_name],
            )
            dq1_out_tensor.shape = [int(v) for v in list(q1_raw_tensor.shape)]
            dq1_out_tensor.shape_signature = _tensor_signature(q1_raw_tensor)

            # 4) Rewire MUL/ADD main path to q0_raw.
            _replace_operator_input_at(
                model_ir=model_ir,
                op=mul0_op,
                input_index=int(mul0_q0_input_idx),
                new_input_name=q0_raw_name,
            )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=add0_op,
                input_index=int(add0_q0_input_idx),
                new_input_name=q0_raw_name,
            )

            # 5) Metadata now follows NHWC on rewritten branches.
            for tensor in [
                dq1_out_tensor,
                hsig_mul_out_tensor,
                hsig_add_out_tensor,
                hsig_max_out_tensor,
                hsig_min_out_tensor,
                sig_q_tensor,
                mul0_out_tensor,
                add0_out_tensor,
            ]:
                _permute_tensor_metadata_if_rank_matches(tensor, perm_nchw_to_nhwc)

            # Preserve legacy NCHW semantics for add0 downstream via adapters.
            legacy_add0_shape = [int(v) for v in list(add0_out_tensor.shape)]
            legacy_add0_sig = _tensor_signature(add0_out_tensor)
            adapter_insertions: List[Tuple[int, OperatorIR]] = []
            consumers_after = _build_tensor_consumer_map(model_ir)
            add0_consumers = [int(v) for v in consumers_after.get(add0_out_name, [])]
            for legacy_idx in add0_consumers:
                if int(legacy_idx) == int(add0_idx):
                    continue
                legacy_op = model_ir.operators[int(legacy_idx)]
                slots = [
                    int(i)
                    for i, inp in enumerate(list(legacy_op.inputs))
                    if str(inp) == str(add0_out_name)
                ]
                if len(slots) == 0:
                    continue
                adapter_name = _unique_tensor_name(f"{add0_out_name}_nchw_adapter")
                model_ir.tensors[adapter_name] = TensorIR(
                    name=adapter_name,
                    dtype=str(add0_out_tensor.dtype),
                    shape=[int(v) for v in legacy_add0_shape],
                    shape_signature=[int(v) for v in legacy_add0_sig],
                    data=None,
                    is_variable=False,
                    quantization=_clone_quantization(add0_out_tensor.quantization),
                )
                adapter_op = OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[add0_out_name, perm_nhwc_to_nchw_const_name],
                    outputs=[adapter_name],
                )
                adapter_insertions.append((int(legacy_idx), adapter_op))
                updated_inputs = [str(v) for v in list(legacy_op.inputs)]
                for slot in slots:
                    if 0 <= int(slot) < len(updated_inputs):
                        updated_inputs[int(slot)] = str(adapter_name)
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=legacy_op,
                    new_inputs=updated_inputs,
                )

            if len(adapter_insertions) > 0:
                inserted = 0
                for insert_idx, adapter_op in sorted(adapter_insertions, key=lambda v: int(v[0])):
                    model_ir.operators.insert(int(insert_idx + inserted), adapter_op)
                    inserted += 1

            # 6) Remove bridge transposes.
            remove_indices = sorted(
                set([int(pre0_idx), int(post_idx), int(pre1_idx)]),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "optimized_transpose_mean_hardsigmoid_muladd_chains": int(optimized),
    }
