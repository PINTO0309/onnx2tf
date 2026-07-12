from __future__ import annotations

from typing import Dict

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _all_per_tensor_quantized,
    _build_tensor_consumer_map,
    _clone_quantization,
    _get_per_tensor_scale_zero_point,
    _prune_unused_tensors,
    _quantize_prelu_alpha,
    _quantize_tensor_per_tensor,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    QuantParamIR,
    TensorIR,
    _is_inverse_perm,
)

def _optimize_transpose_dequant_prelu_quantize_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose wrappers around DEQUANTIZE->PRELU->QUANTIZE chains.

    Target pattern:
      Xq --Transpose(P)--> Aq --DEQUANTIZE--> A --PRELU(alpha)--> B --QUANTIZE--> Bq --Transpose(inv(P))--> Yq

    Rewritten:
      Xq --DEQUANTIZE--> A --PRELU(alpha')--> B --QUANTIZE--> Yq

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - Pre/Post transpose permutations are exact inverses
    - Quantized tensors use per-tensor quantization only
    - PRELU alpha tensor is constant if rank remap is required
    """
    removed_prelu_bridges = 0

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue

            bridge_q_in = str(pre_op.outputs[0])
            dq_users = consumers.get(bridge_q_in, [])
            if len(dq_users) != 1:
                continue
            dq_idx = int(dq_users[0])
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if str(dq_op.inputs[0]) != bridge_q_in:
                continue

            bridge_f_in = str(dq_op.outputs[0])
            prelu_users = consumers.get(bridge_f_in, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != bridge_f_in:
                continue

            bridge_f_out = str(prelu_op.outputs[0])
            q_users = consumers.get(bridge_f_out, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != bridge_f_out:
                continue

            bridge_q_out = str(q_op.outputs[0])
            post_users = consumers.get(bridge_q_out, [])
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != bridge_q_out:
                continue

            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                continue

            # Keep user-visible output names stable and avoid breaking observable intermediates.
            if (
                bridge_q_in in model_ir.outputs
                or bridge_f_in in model_ir.outputs
                or bridge_f_out in model_ir.outputs
                or bridge_q_out in model_ir.outputs
            ):
                continue

            q_src_name = str(pre_op.inputs[0])
            q_dst_name = str(post_op.outputs[0])
            if q_src_name in model_ir.outputs:
                continue

            q_src_tensor = model_ir.tensors.get(q_src_name, None)
            q_mid_in_tensor = model_ir.tensors.get(bridge_q_in, None)
            q_mid_out_tensor = model_ir.tensors.get(bridge_q_out, None)
            q_dst_tensor = model_ir.tensors.get(q_dst_name, None)
            if not _all_per_tensor_quantized([q_src_tensor, q_mid_in_tensor, q_mid_out_tensor, q_dst_tensor]):
                continue

            # PRELU alpha layout follows data layout. When alpha rank matches the transposed rank,
            # remap alpha using post permutation so PRELU can run directly on non-transposed data.
            alpha_name = str(prelu_op.inputs[1])
            alpha_tensor = model_ir.tensors.get(alpha_name, None)
            if alpha_tensor is not None and isinstance(alpha_tensor.data, np.ndarray):
                alpha_data = np.asarray(alpha_tensor.data)
                if alpha_data.ndim == len(perm_post):
                    alpha_data = np.transpose(alpha_data, axes=perm_post)
                    alpha_tensor.data = alpha_data
                    alpha_tensor.shape = [int(v) for v in alpha_data.shape]
                    alpha_tensor.shape_signature = [int(v) for v in alpha_data.shape]

            _set_operator_inputs(
                model_ir=model_ir,
                op=dq_op,
                new_inputs=[q_src_name],
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=q_op,
                new_outputs=[q_dst_name],
            )

            # Update bridge tensor metadata to the non-transposed layout.
            q_src_shape = list(q_src_tensor.shape) if q_src_tensor is not None else None
            q_src_signature = (
                list(q_src_tensor.shape_signature)
                if q_src_tensor is not None and q_src_tensor.shape_signature is not None
                else q_src_shape
            )
            if q_src_shape is not None:
                dq_out_tensor = model_ir.tensors.get(bridge_f_in, None)
                prelu_out_tensor = model_ir.tensors.get(bridge_f_out, None)
                if dq_out_tensor is not None:
                    dq_out_tensor.shape = [int(v) for v in q_src_shape]
                    dq_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )
                if prelu_out_tensor is not None:
                    prelu_out_tensor.shape = [int(v) for v in q_src_shape]
                    prelu_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )

            if q_dst_tensor is not None and q_mid_out_tensor is not None:
                q_dst_tensor.dtype = str(q_mid_out_tensor.dtype)
                q_dst_tensor.quantization = _clone_quantization(q_mid_out_tensor.quantization)

            for remove_idx in sorted([pre_idx, post_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_prelu_bridges += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_dequant_prelu_quantize_bridges": int(removed_prelu_bridges),
    }


def _optimize_transpose_dequant_prelu_transpose_bridges(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold transpose wrappers around DEQUANTIZE->PRELU chains.

    Target pattern:
      Xq --Transpose(P)--> Aq --DEQUANTIZE--> A --PRELU(alpha)--> B --Transpose(inv(P))--> Y

    Rewritten:
      Xq --DEQUANTIZE--> A' --PRELU(alpha')--> Y

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - Pre/Post transpose permutations are exact inverses
    - Quantized tensors on DEQUANTIZE input path are per-tensor
    """
    removed_prelu_transpose_bridges = 0

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

        for pre_idx, pre_op in enumerate(model_ir.operators):
            if str(pre_op.op_type) != "TRANSPOSE" or len(pre_op.inputs) < 2 or len(pre_op.outputs) != 1:
                continue

            perm_pre = _read_transpose_perm(model_ir, pre_op)
            if perm_pre is None:
                continue

            bridge_q_in = str(pre_op.outputs[0])
            dq_users = consumers.get(bridge_q_in, [])
            if len(dq_users) != 1:
                continue
            dq_idx = int(dq_users[0])
            dq_op = model_ir.operators[dq_idx]
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            if str(dq_op.inputs[0]) != bridge_q_in:
                continue

            bridge_f_in = str(dq_op.outputs[0])
            prelu_users = consumers.get(bridge_f_in, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != bridge_f_in:
                continue

            bridge_f_out = str(prelu_op.outputs[0])
            post_users = consumers.get(bridge_f_out, [])
            if len(post_users) != 1:
                continue
            post_idx = int(post_users[0])
            post_op = model_ir.operators[post_idx]
            if str(post_op.op_type) != "TRANSPOSE" or len(post_op.inputs) < 2 or len(post_op.outputs) != 1:
                continue
            if str(post_op.inputs[0]) != bridge_f_out:
                continue

            perm_post = _read_transpose_perm(model_ir, post_op)
            if perm_post is None or not _is_inverse_perm(perm_pre, perm_post):
                continue

            if (
                bridge_q_in in model_ir.outputs
                or bridge_f_in in model_ir.outputs
                or bridge_f_out in model_ir.outputs
                or post_op.outputs[0] in model_ir.outputs
            ):
                continue

            q_src_name = str(pre_op.inputs[0])
            y_name = str(post_op.outputs[0])
            q_src_tensor = model_ir.tensors.get(q_src_name, None)
            q_mid_tensor = model_ir.tensors.get(bridge_q_in, None)
            if not _all_per_tensor_quantized([q_src_tensor, q_mid_tensor]):
                continue

            alpha_name = str(prelu_op.inputs[1])
            alpha_tensor = model_ir.tensors.get(alpha_name, None)
            if alpha_tensor is not None:
                alpha_data = alpha_tensor.data
                if isinstance(alpha_data, np.ndarray) and alpha_data.ndim == len(perm_post):
                    transposed_alpha = np.transpose(alpha_data, axes=perm_post)
                    alpha_users = consumers.get(alpha_name, [])
                    if len(alpha_users) == 1 and int(alpha_users[0]) == int(prelu_idx):
                        alpha_tensor.data = transposed_alpha
                        alpha_tensor.shape = [int(v) for v in transposed_alpha.shape]
                        alpha_tensor.shape_signature = [int(v) for v in transposed_alpha.shape]
                    else:
                        new_alpha_name = _unique_tensor_name(f"{alpha_name}_nhwc")
                        model_ir.tensors[new_alpha_name] = TensorIR(
                            name=new_alpha_name,
                            dtype=str(alpha_tensor.dtype),
                            shape=[int(v) for v in transposed_alpha.shape],
                            shape_signature=[int(v) for v in transposed_alpha.shape],
                            data=np.asarray(transposed_alpha),
                            is_variable=False,
                            quantization=_clone_quantization(alpha_tensor.quantization),
                        )
                        _replace_operator_input_at(
                            model_ir=model_ir,
                            op=prelu_op,
                            input_index=1,
                            new_input_name=new_alpha_name,
                        )

            _set_operator_inputs(
                model_ir=model_ir,
                op=dq_op,
                new_inputs=[q_src_name],
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=prelu_op,
                new_outputs=[y_name],
            )

            q_src_shape = list(q_src_tensor.shape) if q_src_tensor is not None else None
            q_src_signature = (
                list(q_src_tensor.shape_signature)
                if q_src_tensor is not None and q_src_tensor.shape_signature is not None
                else q_src_shape
            )
            if q_src_shape is not None:
                dq_out_tensor = model_ir.tensors.get(bridge_f_in, None)
                y_tensor = model_ir.tensors.get(y_name, None)
                if dq_out_tensor is not None:
                    dq_out_tensor.shape = [int(v) for v in q_src_shape]
                    dq_out_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )
                if y_tensor is not None:
                    y_tensor.shape = [int(v) for v in q_src_shape]
                    y_tensor.shape_signature = (
                        [int(v) for v in q_src_signature]
                        if q_src_signature is not None
                        else [int(v) for v in q_src_shape]
                    )

            for remove_idx in sorted([pre_idx, post_idx], reverse=True):
                del model_ir.operators[remove_idx]
            removed_prelu_transpose_bridges += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "removed_transpose_dequant_prelu_transpose_bridges": int(removed_prelu_transpose_bridges),
    }


def _optimize_dequant_prelu_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold DEQUANTIZE->PRELU->QUANTIZE into quantized PRELU.

    Target pattern:
      Xq --DEQUANTIZE--> Xf --PRELU(alpha_f)--> Yf --QUANTIZE--> Yq

    Rewritten:
      Xq --PRELU(alpha_q)--> Yq
    """
    folded = 0

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

        for dq_idx, dq_op in enumerate(model_ir.operators):
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            q_in_name = str(dq_op.inputs[0])
            f_in_name = str(dq_op.outputs[0])

            prelu_users = consumers.get(f_in_name, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != f_in_name:
                continue

            f_out_name = str(prelu_op.outputs[0])
            q_users = consumers.get(f_out_name, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != f_out_name:
                continue
            q_out_name = str(q_op.outputs[0])

            if f_in_name in model_ir.outputs or f_out_name in model_ir.outputs:
                continue

            q_in_tensor = model_ir.tensors.get(q_in_name, None)
            q_out_tensor = model_ir.tensors.get(q_out_name, None)
            alpha_name = str(prelu_op.inputs[1])
            alpha_tensor = model_ir.tensors.get(alpha_name, None)
            if q_in_tensor is None or q_out_tensor is None or alpha_tensor is None:
                continue
            if not _all_per_tensor_quantized([q_in_tensor, q_out_tensor]):
                continue

            target_dtype = str(q_in_tensor.dtype)
            if target_dtype not in {"INT8", "UINT8"}:
                continue
            if str(q_out_tensor.dtype) != target_dtype:
                continue
            if not isinstance(alpha_tensor.data, np.ndarray):
                continue

            try:
                alpha_q, alpha_qparams = _quantize_prelu_alpha(alpha_tensor.data, target_dtype)
            except Exception:
                continue

            alpha_users = consumers.get(alpha_name, [])
            if len(alpha_users) == 1 and int(alpha_users[0]) == int(prelu_idx):
                alpha_q_name = alpha_name
                alpha_tensor.data = np.asarray(alpha_q)
                alpha_tensor.dtype = target_dtype
                alpha_tensor.shape = [int(v) for v in alpha_q.shape]
                alpha_tensor.shape_signature = [int(v) for v in alpha_q.shape]
                alpha_tensor.quantization = alpha_qparams
            else:
                alpha_q_name = _unique_tensor_name(f"{alpha_name}_q")
                model_ir.tensors[alpha_q_name] = TensorIR(
                    name=alpha_q_name,
                    dtype=target_dtype,
                    shape=[int(v) for v in alpha_q.shape],
                    shape_signature=[int(v) for v in alpha_q.shape],
                    data=np.asarray(alpha_q),
                    is_variable=False,
                    quantization=alpha_qparams,
                )

            _set_operator_inputs(
                model_ir=model_ir,
                op=prelu_op,
                new_inputs=[q_in_name, alpha_q_name],
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=prelu_op,
                new_outputs=[q_out_name],
            )

            for remove_idx in sorted([dq_idx, q_idx], reverse=True):
                del model_ir.operators[remove_idx]
            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"folded_dequant_prelu_quantize_chains": int(folded)}


def _optimize_dequant_prelu_depthwise_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    """
    Fold DEQUANTIZE->PRELU->DEPTHWISE_CONV_2D->QUANTIZE into quantized PRELU+DEPTHWISE_CONV_2D.

    Target pattern:
      Xq --DEQUANTIZE--> Xf --PRELU(alpha_f)--> Pf --DEPTHWISE_CONV_2D(w_f,b_f)--> Yf --QUANTIZE--> Yq

    Rewritten:
      Xq --PRELU(alpha_q)--> Pq --DEPTHWISE_CONV_2D(w_q,b_q)--> Yq
    """
    folded = 0

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

        for dq_idx, dq_op in enumerate(model_ir.operators):
            if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                continue
            q_in_name = str(dq_op.inputs[0])
            f_in_name = str(dq_op.outputs[0])

            prelu_users = consumers.get(f_in_name, [])
            if len(prelu_users) != 1:
                continue
            prelu_idx = int(prelu_users[0])
            prelu_op = model_ir.operators[prelu_idx]
            if str(prelu_op.op_type) != "PRELU" or len(prelu_op.inputs) != 2 or len(prelu_op.outputs) != 1:
                continue
            if str(prelu_op.inputs[0]) != f_in_name:
                continue
            p_f_name = str(prelu_op.outputs[0])

            dw_users = consumers.get(p_f_name, [])
            if len(dw_users) != 1:
                continue
            dw_idx = int(dw_users[0])
            dw_op = model_ir.operators[dw_idx]
            if str(dw_op.op_type) != "DEPTHWISE_CONV_2D" or len(dw_op.inputs) != 3 or len(dw_op.outputs) != 1:
                continue
            if str(dw_op.inputs[0]) != p_f_name:
                continue
            y_f_name = str(dw_op.outputs[0])

            q_users = consumers.get(y_f_name, [])
            if len(q_users) != 1:
                continue
            q_idx = int(q_users[0])
            q_op = model_ir.operators[q_idx]
            if str(q_op.op_type) != "QUANTIZE" or len(q_op.inputs) != 1 or len(q_op.outputs) != 1:
                continue
            if str(q_op.inputs[0]) != y_f_name:
                continue
            y_q_name = str(q_op.outputs[0])

            if f_in_name in model_ir.outputs or p_f_name in model_ir.outputs or y_f_name in model_ir.outputs:
                continue

            q_in_tensor = model_ir.tensors.get(q_in_name, None)
            y_q_tensor = model_ir.tensors.get(y_q_name, None)
            alpha_tensor = model_ir.tensors.get(str(prelu_op.inputs[1]), None)
            w_f_tensor = model_ir.tensors.get(str(dw_op.inputs[1]), None)
            b_f_tensor = model_ir.tensors.get(str(dw_op.inputs[2]), None)
            if (
                q_in_tensor is None
                or y_q_tensor is None
                or alpha_tensor is None
                or w_f_tensor is None
                or b_f_tensor is None
            ):
                continue
            if not _all_per_tensor_quantized([q_in_tensor, y_q_tensor]):
                continue

            target_dtype = str(q_in_tensor.dtype)
            if target_dtype not in {"INT8", "UINT8"}:
                continue
            if str(y_q_tensor.dtype) != target_dtype:
                continue
            if not isinstance(alpha_tensor.data, np.ndarray):
                continue
            if not isinstance(w_f_tensor.data, np.ndarray):
                continue
            if not isinstance(b_f_tensor.data, np.ndarray):
                continue

            x_qparams = _get_per_tensor_scale_zero_point(q_in_tensor.quantization)
            if x_qparams is None:
                continue
            x_scale, _x_zero_point = x_qparams

            weights_f = np.asarray(w_f_tensor.data, dtype=np.float32)
            if weights_f.ndim != 4:
                continue
            bias_f = np.asarray(b_f_tensor.data, dtype=np.float32).reshape(-1)
            if bias_f.size != int(weights_f.shape[-1]):
                continue

            try:
                alpha_q, alpha_qparams = _quantize_prelu_alpha(alpha_tensor.data, target_dtype)
                w_q, w_qparams = _quantize_tensor_per_tensor(weights_f, target_dtype)
            except Exception:
                continue

            w_scale = float(w_qparams.scale[0])
            bias_scale = max(float(x_scale * w_scale), 1e-12)
            bias_q = np.clip(
                np.round(bias_f / bias_scale),
                np.iinfo(np.int32).min,
                np.iinfo(np.int32).max,
            ).astype(np.int32)
            b_qparams = QuantParamIR(
                scale=[float(bias_scale)],
                zero_point=[0],
                quantized_dimension=0,
            )

            alpha_q_name = _unique_tensor_name(f"{prelu_op.inputs[1]}_q")
            model_ir.tensors[alpha_q_name] = TensorIR(
                name=alpha_q_name,
                dtype=target_dtype,
                shape=[int(v) for v in alpha_q.shape],
                shape_signature=[int(v) for v in alpha_q.shape],
                data=np.asarray(alpha_q),
                is_variable=False,
                quantization=alpha_qparams,
            )

            w_q_name = _unique_tensor_name(f"{dw_op.inputs[1]}_q")
            model_ir.tensors[w_q_name] = TensorIR(
                name=w_q_name,
                dtype=target_dtype,
                shape=[int(v) for v in w_q.shape],
                shape_signature=[int(v) for v in w_q.shape],
                data=np.asarray(w_q),
                is_variable=False,
                quantization=w_qparams,
            )

            b_q_name = _unique_tensor_name(f"{dw_op.inputs[2]}_q")
            model_ir.tensors[b_q_name] = TensorIR(
                name=b_q_name,
                dtype="INT32",
                shape=[int(v) for v in bias_q.shape],
                shape_signature=[int(v) for v in bias_q.shape],
                data=np.asarray(bias_q),
                is_variable=False,
                quantization=b_qparams,
            )

            p_q_tensor = model_ir.tensors.get(p_f_name, None)
            if p_q_tensor is not None:
                p_q_tensor.dtype = target_dtype
                p_q_tensor.quantization = _clone_quantization(q_in_tensor.quantization)

            _set_operator_inputs(
                model_ir=model_ir,
                op=prelu_op,
                new_inputs=[q_in_name, alpha_q_name],
            )
            _set_operator_inputs(
                model_ir=model_ir,
                op=dw_op,
                new_inputs=[p_f_name, w_q_name, b_q_name],
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=dw_op,
                new_outputs=[y_q_name],
            )

            for remove_idx in sorted([dq_idx, q_idx], reverse=True):
                del model_ir.operators[remove_idx]
            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {"folded_dequant_prelu_depthwise_quantize_chains": int(folded)}

