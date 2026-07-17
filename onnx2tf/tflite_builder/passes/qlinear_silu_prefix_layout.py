from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _build_tensor_consumer_map,
    _clone_quantization,
    _is_singleton_constant_tensor,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_nhwc_prefix_qlinear_silu_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Propagate NHWC through early QLinear SiLU chains and remove unnecessary
    NCHW round-trips around Conv outputs.

    Target pattern (rank-4):
      q_raw_nhwc --TRANSPOSE(0,3,1,2)--> q_nchw
      q_nchw --DEQUANTIZE--> dq --LOGISTIC--> sig_f --QUANTIZE--> sig_q
      MUL(q_nchw, sig_q) --> y_q
    or
      q_raw_nhwc --TRANSPOSE(0,3,1,2)--> q_nchw
      q_nchw --DEQUANTIZE--> dq --MUL(c1)--> --ADD(c2)--> --MAXIMUM(c3)--> --MINIMUM(c4)--> --QUANTIZE--> sig_q
      MUL(q_nchw, sig_q) --> y_q

    Rewrite:
      q_raw_nhwc --DEQUANTIZE--> dq --LOGISTIC--> sig_f --QUANTIZE--> sig_q
      MUL(q_raw_nhwc, sig_q) --> y_q

    Plus optional removal of:
      y_q --TRANSPOSE(0,2,3,1)--> y_nhwc

    Safety:
    - Strict SiLU topology
      : single DEQUANTIZE->LOGISTIC->QUANTIZE branch
      : or DEQUANTIZE->(MUL/ADD/MAXIMUM/MINIMUM)->QUANTIZE with singleton constants.
    - Per-tensor quantization only on quantized tensors touched by the rewrite.
    - For non-transpose consumers, preserve legacy NCHW semantics by inserting
      NHWC->NCHW adapters at the consumer input slots when the op is considered
      layout-safe (axis count preserving / non-layout-sensitive).
    """
    optimized = 0
    perm_nhwc_to_nchw = [0, 3, 1, 2]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    perm_nhwc_to_nchw_const_name = "__nhwc_to_nchw_perm_rank4__"
    # Ops excluded from generic legacy-adapter rewiring because they are
    # layout-sensitive or modify axis semantics directly.
    legacy_adapter_blocklist = {
        "CONCATENATION",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "MAX_POOL_2D",
        "AVERAGE_POOL_2D",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "RESHAPE",
        "TRANSPOSE",
        "STRIDED_SLICE",
        "SLICE",
        "SOFTMAX",
        "FULLY_CONNECTED",
        "MEAN",
    }

    def _unique_tensor_name(
        base: str,
        *,
        reserved_names: Optional[set[str]] = None,
    ) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors or (
            reserved_names is not None and name in reserved_names
        ):
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _is_exact_nhwc_to_nchw_perm_tensor(tensor_name: str) -> bool:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None or tensor.data is None:
            return False
        try:
            data = np.asarray(tensor.data)
        except Exception:
            return False
        return (
            str(tensor.dtype).upper() == "INT32"
            and list(tensor.shape) == [4]
            and list(tensor.shape_signature or []) == [4]
            and data.dtype == np.dtype(np.int32)
            and list(data.shape) == [4]
            and np.array_equal(
                data,
                np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
            )
            and not bool(tensor.is_variable)
            and tensor.quantization is None
        )

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue
            if _read_transpose_perm(model_ir, pre_op) != perm_nhwc_to_nchw:
                continue

            q_raw_name = str(pre_op.inputs[0])
            q_nchw_name = str(pre_op.outputs[0])
            if q_raw_name in model_outputs or q_nchw_name in model_outputs:
                continue

            q_raw_tensor = model_ir.tensors.get(q_raw_name, None)
            q_nchw_tensor = model_ir.tensors.get(q_nchw_name, None)
            if q_raw_tensor is None or q_nchw_tensor is None:
                continue
            if len(list(q_raw_tensor.shape)) != 4 or len(list(q_nchw_tensor.shape)) != 4:
                continue
            if not _all_per_tensor_quantized([q_raw_tensor, q_nchw_tensor]):
                continue

            q_nchw_users = [int(v) for v in consumers.get(q_nchw_name, [])]
            if len(q_nchw_users) != 2:
                continue

            dq_idx: Optional[int] = None
            mul_idx: Optional[int] = None
            for user_idx in q_nchw_users:
                user_op = model_ir.operators[int(user_idx)]
                if str(user_op.op_type) == "DEQUANTIZE":
                    if dq_idx is not None:
                        dq_idx = None
                        break
                    dq_idx = int(user_idx)
                elif str(user_op.op_type) == "MUL":
                    if mul_idx is not None:
                        mul_idx = None
                        break
                    mul_idx = int(user_idx)
                else:
                    dq_idx = None
                    mul_idx = None
                    break
            if dq_idx is None or mul_idx is None:
                continue

            dq_op = model_ir.operators[int(dq_idx)]
            mul_op = model_ir.operators[int(mul_idx)]
            if len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if str(dq_op.inputs[0]) != q_nchw_name:
                continue
            dq_out_name = str(dq_op.outputs[0])
            if dq_out_name in model_outputs:
                continue

            dq_users = [int(v) for v in consumers.get(dq_out_name, [])]
            if len(dq_users) != 1:
                continue

            sig_head_idx = int(dq_users[0])
            sig_head_op = model_ir.operators[int(sig_head_idx)]
            sig_head_type = str(sig_head_op.op_type)
            sig_intermediate_names: List[str] = []
            sig_q_name: Optional[str] = None

            # Pattern A:
            #   DEQUANTIZE -> LOGISTIC -> QUANTIZE
            if sig_head_type == "LOGISTIC":
                if len(sig_head_op.inputs) != 1 or len(sig_head_op.outputs) != 1:
                    continue
                if str(sig_head_op.inputs[0]) != dq_out_name:
                    continue
                logistic_out_name = str(sig_head_op.outputs[0])
                if logistic_out_name in model_outputs:
                    continue
                logistic_users = [int(v) for v in consumers.get(logistic_out_name, [])]
                if len(logistic_users) != 1:
                    continue
                qsig_idx = int(logistic_users[0])
                qsig_op = model_ir.operators[int(qsig_idx)]
                if str(qsig_op.op_type) != "QUANTIZE" or len(qsig_op.inputs) != 1 or len(qsig_op.outputs) != 1:
                    continue
                if str(qsig_op.inputs[0]) != logistic_out_name:
                    continue
                sig_intermediate_names = [logistic_out_name]
                sig_q_name = str(qsig_op.outputs[0])

            # Pattern B:
            #   DEQUANTIZE -> MUL(const) -> ADD(const) -> MAXIMUM(const) -> MINIMUM(const) -> QUANTIZE
            else:
                if sig_head_type != "MUL" or len(sig_head_op.inputs) != 2 or len(sig_head_op.outputs) != 1:
                    continue
                sig_head_inputs = [str(v) for v in list(sig_head_op.inputs)]
                if dq_out_name == sig_head_inputs[0]:
                    hs_mul_side = sig_head_inputs[1]
                elif dq_out_name == sig_head_inputs[1]:
                    hs_mul_side = sig_head_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_mul_side):
                    continue
                hs_mul_out_name = str(sig_head_op.outputs[0])
                if hs_mul_out_name in model_outputs:
                    continue

                hs_mul_users = [int(v) for v in consumers.get(hs_mul_out_name, [])]
                if len(hs_mul_users) != 1:
                    continue
                hs_add_idx = int(hs_mul_users[0])
                hs_add_op = model_ir.operators[int(hs_add_idx)]
                if str(hs_add_op.op_type) != "ADD" or len(hs_add_op.inputs) != 2 or len(hs_add_op.outputs) != 1:
                    continue
                hs_add_inputs = [str(v) for v in list(hs_add_op.inputs)]
                if hs_mul_out_name == hs_add_inputs[0]:
                    hs_add_side = hs_add_inputs[1]
                elif hs_mul_out_name == hs_add_inputs[1]:
                    hs_add_side = hs_add_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_add_side):
                    continue
                hs_add_out_name = str(hs_add_op.outputs[0])
                if hs_add_out_name in model_outputs:
                    continue

                hs_add_users = [int(v) for v in consumers.get(hs_add_out_name, [])]
                if len(hs_add_users) != 1:
                    continue
                hs_max_idx = int(hs_add_users[0])
                hs_max_op = model_ir.operators[int(hs_max_idx)]
                if str(hs_max_op.op_type) != "MAXIMUM" or len(hs_max_op.inputs) != 2 or len(hs_max_op.outputs) != 1:
                    continue
                hs_max_inputs = [str(v) for v in list(hs_max_op.inputs)]
                if hs_add_out_name == hs_max_inputs[0]:
                    hs_max_side = hs_max_inputs[1]
                elif hs_add_out_name == hs_max_inputs[1]:
                    hs_max_side = hs_max_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_max_side):
                    continue
                hs_max_out_name = str(hs_max_op.outputs[0])
                if hs_max_out_name in model_outputs:
                    continue

                hs_max_users = [int(v) for v in consumers.get(hs_max_out_name, [])]
                if len(hs_max_users) != 1:
                    continue
                hs_min_idx = int(hs_max_users[0])
                hs_min_op = model_ir.operators[int(hs_min_idx)]
                if str(hs_min_op.op_type) != "MINIMUM" or len(hs_min_op.inputs) != 2 or len(hs_min_op.outputs) != 1:
                    continue
                hs_min_inputs = [str(v) for v in list(hs_min_op.inputs)]
                if hs_max_out_name == hs_min_inputs[0]:
                    hs_min_side = hs_min_inputs[1]
                elif hs_max_out_name == hs_min_inputs[1]:
                    hs_min_side = hs_min_inputs[0]
                else:
                    continue
                if not _is_singleton_constant_tensor(model_ir, hs_min_side):
                    continue
                hs_min_out_name = str(hs_min_op.outputs[0])
                if hs_min_out_name in model_outputs:
                    continue

                hs_min_users = [int(v) for v in consumers.get(hs_min_out_name, [])]
                if len(hs_min_users) != 1:
                    continue
                qsig_idx = int(hs_min_users[0])
                qsig_op = model_ir.operators[int(qsig_idx)]
                if str(qsig_op.op_type) != "QUANTIZE" or len(qsig_op.inputs) != 1 or len(qsig_op.outputs) != 1:
                    continue
                if str(qsig_op.inputs[0]) != hs_min_out_name:
                    continue
                sig_intermediate_names = [
                    hs_mul_out_name,
                    hs_add_out_name,
                    hs_max_out_name,
                    hs_min_out_name,
                ]
                sig_q_name = str(qsig_op.outputs[0])

            if sig_q_name is None or sig_q_name in model_outputs:
                continue

            if len(mul_op.inputs) != 2 or len(mul_op.outputs) != 1:
                continue
            mul_inputs = [str(v) for v in list(mul_op.inputs)]
            if q_nchw_name not in set(mul_inputs):
                continue
            if sig_q_name not in set(mul_inputs):
                continue
            sig_q_users = [int(v) for v in consumers.get(sig_q_name, [])]
            if sig_q_users != [int(mul_idx)] and set(sig_q_users) != {int(mul_idx)}:
                continue

            sig_q_tensor = model_ir.tensors.get(sig_q_name, None)
            mul_out_name = str(mul_op.outputs[0])
            mul_out_tensor = model_ir.tensors.get(mul_out_name, None)
            if mul_out_name in model_outputs or sig_q_tensor is None or mul_out_tensor is None:
                continue
            if not _all_per_tensor_quantized([sig_q_tensor, mul_out_tensor]):
                continue

            mul_users = list(
                dict.fromkeys(
                    int(v) for v in consumers.get(mul_out_name, [])
                )
            )
            removable_post_ops: List[OperatorIR] = []
            legacy_user_input_slots: List[Tuple[int, int]] = []
            users_supported = True
            for user_idx in mul_users:
                user_op = model_ir.operators[int(user_idx)]
                user_type = str(user_op.op_type)
                if user_type == "TRANSPOSE":
                    if len(user_op.inputs) < 2 or len(user_op.outputs) != 1:
                        users_supported = False
                        break
                    if str(user_op.inputs[0]) != mul_out_name:
                        users_supported = False
                        break
                    if _read_transpose_perm(model_ir, user_op) != perm_nchw_to_nhwc:
                        users_supported = False
                        break
                    if str(user_op.outputs[0]) in model_outputs:
                        users_supported = False
                        break
                    removable_post_ops.append(user_op)
                else:
                    if user_type in legacy_adapter_blocklist:
                        users_supported = False
                        break
                    input_indices = [
                        int(input_idx)
                        for input_idx, input_name in enumerate(user_op.inputs)
                        if str(input_name) == str(mul_out_name)
                    ]
                    if len(input_indices) == 0:
                        users_supported = False
                        break
                    if len(user_op.outputs) == 0:
                        users_supported = False
                        break
                    if any(str(output_name) in model_outputs for output_name in list(user_op.outputs)):
                        users_supported = False
                        break
                    for input_idx in input_indices:
                        legacy_user_input_slots.append((int(user_idx), int(input_idx)))
            if not users_supported:
                continue

            metadata_target_names = (
                [dq_out_name]
                + list(sig_intermediate_names)
                + [sig_q_name, mul_out_name]
            )
            metadata_targets: List[TensorIR] = []
            metadata_is_rank4 = True
            for tensor_name in metadata_target_names:
                tensor = model_ir.tensors.get(str(tensor_name), None)
                if tensor is None:
                    metadata_is_rank4 = False
                    break
                tensor_shape = [int(v) for v in list(tensor.shape)]
                tensor_signature = (
                    [int(v) for v in list(tensor.shape_signature)]
                    if tensor.shape_signature is not None
                    else list(tensor_shape)
                )
                if len(tensor_shape) != 4 or len(tensor_signature) != 4:
                    metadata_is_rank4 = False
                    break
                metadata_targets.append(tensor)
            if not metadata_is_rank4:
                continue

            legacy_mul_shape = [int(v) for v in list(mul_out_tensor.shape)]
            legacy_mul_signature = (
                [int(v) for v in list(mul_out_tensor.shape_signature)]
                if mul_out_tensor.shape_signature is not None
                else [int(v) for v in list(mul_out_tensor.shape)]
            )
            if len(legacy_mul_shape) != 4 or len(legacy_mul_signature) != 4:
                continue

            reserved_tensor_names = set(str(v) for v in model_ir.tensors)
            legacy_perm_name: Optional[str] = None
            legacy_perm_tensor: Optional[TensorIR] = None
            if len(legacy_user_input_slots) > 0:
                if _is_exact_nhwc_to_nchw_perm_tensor(
                    perm_nhwc_to_nchw_const_name
                ):
                    legacy_perm_name = str(perm_nhwc_to_nchw_const_name)
                else:
                    legacy_perm_name = _unique_tensor_name(
                        perm_nhwc_to_nchw_const_name,
                        reserved_names=reserved_tensor_names,
                    )
                    reserved_tensor_names.add(str(legacy_perm_name))
                    legacy_perm_tensor = TensorIR(
                        name=str(legacy_perm_name),
                        dtype="INT32",
                        shape=[4],
                        shape_signature=[4],
                        data=np.asarray(perm_nhwc_to_nchw, dtype=np.int32),
                        is_variable=False,
                        quantization=None,
                    )
            if (
                len(legacy_user_input_slots) > 0
                and legacy_perm_name is None
            ):
                continue

            adapter_tensors: List[TensorIR] = []
            adapter_insertions: List[Tuple[int, OperatorIR]] = []
            legacy_user_inputs: Dict[int, List[str]] = {}
            for legacy_user_idx, legacy_input_index in legacy_user_input_slots:
                legacy_user_op = model_ir.operators[int(legacy_user_idx)]
                adapter_name = _unique_tensor_name(
                    f"{mul_out_name}_nchw_adapter",
                    reserved_names=reserved_tensor_names,
                )
                reserved_tensor_names.add(str(adapter_name))
                adapter_tensors.append(
                    TensorIR(
                        name=adapter_name,
                        dtype=str(mul_out_tensor.dtype),
                        shape=list(legacy_mul_shape),
                        shape_signature=list(legacy_mul_signature),
                        data=None,
                        is_variable=False,
                        quantization=_clone_quantization(
                            mul_out_tensor.quantization
                        ),
                    )
                )
                adapter_op = OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[mul_out_name, str(legacy_perm_name)],
                    outputs=[adapter_name],
                )
                adapter_insertions.append((int(legacy_user_idx), adapter_op))
                updated_inputs = legacy_user_inputs.setdefault(
                    int(legacy_user_idx),
                    [str(v) for v in list(legacy_user_op.inputs)],
                )
                updated_inputs[int(legacy_input_index)] = str(adapter_name)

            # Commit only after every tensor, signature, and adapter is valid.
            if legacy_perm_tensor is not None:
                model_ir.tensors[str(legacy_perm_tensor.name)] = legacy_perm_tensor
            for adapter_tensor in adapter_tensors:
                model_ir.tensors[str(adapter_tensor.name)] = adapter_tensor

            _set_operator_inputs(
                model_ir=model_ir,
                op=dq_op,
                new_inputs=[q_raw_name],
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=mul_op,
                new_inputs=[
                    q_raw_name if str(input_name) == q_nchw_name else str(input_name)
                    for input_name in mul_inputs
                ],
            )
            for legacy_user_idx, updated_inputs in legacy_user_inputs.items():
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=model_ir.operators[int(legacy_user_idx)],
                    new_inputs=list(updated_inputs),
                )

            if len(adapter_insertions) > 0:
                inserted = 0
                for insert_idx, adapter_op in sorted(adapter_insertions, key=lambda v: int(v[0])):
                    model_ir.operators.insert(int(insert_idx + inserted), adapter_op)
                    inserted += 1

            # Metadata now follows NHWC for the rewritten chain.
            for tensor in metadata_targets:
                _permute_tensor_metadata_if_rank_matches(
                    tensor,
                    perm_nchw_to_nhwc,
                )

            # Remove redundant output transpose adapters.
            for post_op in removable_post_ops:
                post_out_name = str(post_op.outputs[0])
                _replace_tensor_inputs(model_ir, post_out_name, mul_out_name)

            removable_post_op_ids = {id(op) for op in removable_post_ops}
            removable_post_indices = [
                int(op_idx)
                for op_idx, op in enumerate(model_ir.operators)
                if id(op) in removable_post_op_ids
            ]
            remove_indices = sorted(
                set([int(pre_idx)] + [int(v) for v in removable_post_indices]),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir)
    return {
        "optimized_nhwc_prefix_qlinear_silu_chains": int(optimized),
    }
