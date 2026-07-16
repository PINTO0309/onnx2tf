from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_consumer_map,
    _build_tensor_producer_map,
    _invert_perm,
    _permute_shape,
    _prune_unused_tensors,
    _quant_scale_count,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def optimize_nhwc_propagation_qlinear_concat_conv(model_ir: ModelIR) -> Dict[str, int]:
    """
    Propagate NHWC layout through a QLinearConcat -> QLinearConv bridge.

    Target IR shape:
      (NCHW quantized inputs)
        -> DEQUANTIZE[*]
        -> CONCATENATION(axis=1)
        -> QUANTIZE(q_out_nchw)
        -> TRANSPOSE([0,2,3,1])  # q_out_nhwc
        -> (CONV_2D|DEPTHWISE_CONV_2D)

    Rewrites:
    - Fold per-input NCHW adapters (TRANSPOSE [0,3,1,2]) into DQ inputs when safe.
    - Convert CONCAT axis from NCHW to NHWC.
    - Remove post-QUANTIZE TRANSPOSE adapters to CONV inputs.
    - Allow additional CONCAT consumers that are NCHW->NHWC TRANSPOSE adapters
      and remove them after CONCAT becomes NHWC.
    """
    propagated = 0
    rank4_perm_nchw_to_nhwc = [0, 2, 3, 1]
    rank4_perm_nhwc_to_nchw = [0, 3, 1, 2]
    inv_nchw_to_nhwc = _invert_perm(rank4_perm_nchw_to_nhwc)
    if inv_nchw_to_nhwc is None:
        return {"propagated_qlinear_concat_conv_nhwc_chains": 0}

    def _remap_qdim_for_permute(tensor: Optional[TensorIR], perm: List[int]) -> None:
        if tensor is None or tensor.quantization is None:
            return
        if _quant_scale_count(tensor.quantization) <= 1:
            return
        inv = _invert_perm(perm)
        if inv is None:
            return
        old_qdim = int(tensor.quantization.quantized_dimension)
        if 0 <= old_qdim < len(inv):
            tensor.quantization.quantized_dimension = int(inv[old_qdim])

    def _is_nchw_nhwc_reinterpret_safe(tensor: Optional[TensorIR]) -> bool:
        if tensor is None:
            return False
        shape = list(tensor.shape)
        if len(shape) != 4:
            return False
        signature = (
            list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else list(shape)
        )
        if len(signature) != 4:
            return False
        # Reinterpretation is safe when spatial dimensions are statically singleton.
        return int(shape[2]) == 1 and int(shape[3]) == 1 and int(signature[2]) == 1 and int(signature[3]) == 1

    def _is_nhwc_singleton_spatial(tensor: Optional[TensorIR]) -> bool:
        if tensor is None:
            return False
        shape = [int(v) for v in list(tensor.shape)]
        signature = (
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else list(shape)
        )
        return (
            len(shape) == 4
            and len(signature) == 4
            and int(shape[1]) == 1
            and int(shape[2]) == 1
            and int(signature[1]) == 1
            and int(signature[2]) == 1
        )

    def _is_singleton_nchw_to_nhwc_reshape(op: OperatorIR) -> bool:
        if str(op.op_type) != "RESHAPE" or len(op.inputs) < 1 or len(op.outputs) != 1:
            return False
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if not _is_nchw_nhwc_reinterpret_safe(input_tensor) or not _is_nhwc_singleton_spatial(output_tensor):
            return False
        expected = _permuted_shape_signature(input_tensor, rank4_perm_nchw_to_nhwc)
        if expected is None or output_tensor is None:
            return False
        output_shape = [int(v) for v in list(output_tensor.shape)]
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else list(output_shape)
        )
        return output_shape == list(expected[0]) and output_signature == list(expected[1])

    def _permute_tensor_shape_signature(tensor: Optional[TensorIR], perm: List[int]) -> bool:
        if tensor is None:
            return False
        new_shape = _permute_shape(list(tensor.shape), perm)
        signature_source = (
            list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else list(tensor.shape)
        )
        new_signature = _permute_shape(signature_source, perm)
        if new_shape is None or new_signature is None:
            return False
        tensor.shape = [int(v) for v in new_shape]
        tensor.shape_signature = [int(v) for v in new_signature]
        _remap_qdim_for_permute(tensor, perm)
        return True

    def _permuted_shape_signature(
        tensor: Optional[TensorIR],
        perm: List[int],
    ) -> Optional[Tuple[List[int], List[int]]]:
        if tensor is None:
            return None
        new_shape = _permute_shape(list(tensor.shape), perm)
        signature_source = (
            list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else list(tensor.shape)
        )
        new_signature = _permute_shape(signature_source, perm)
        if new_shape is None or new_signature is None:
            return None
        return [int(v) for v in new_shape], [int(v) for v in new_signature]

    while True:
        changed = False
        consumers = _build_tensor_consumer_map(model_ir)
        producers = _build_tensor_producer_map(model_ir)
        model_outputs = set(str(v) for v in model_ir.outputs)

        for concat_idx, concat_op in enumerate(model_ir.operators):
            if str(concat_op.op_type) != "CONCATENATION" or len(concat_op.inputs) == 0 or len(concat_op.outputs) != 1:
                continue
            concat_out_name = str(concat_op.outputs[0])
            if concat_out_name in model_outputs:
                continue

            q_users = [int(v) for v in consumers.get(concat_out_name, [])]
            if len(q_users) == 0:
                continue
            q_idx_candidates: List[int] = []
            removable_concat_post_indices: List[int] = []
            valid_concat_users = True
            for user_idx in q_users:
                user_op = model_ir.operators[int(user_idx)]
                if (
                    str(user_op.op_type) == "QUANTIZE"
                    and len(user_op.inputs) == 1
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == concat_out_name
                ):
                    q_idx_candidates.append(int(user_idx))
                    continue
                if (
                    str(user_op.op_type) == "TRANSPOSE"
                    and len(user_op.inputs) >= 2
                    and len(user_op.outputs) == 1
                    and str(user_op.inputs[0]) == concat_out_name
                    and _read_transpose_perm(model_ir, user_op) == rank4_perm_nchw_to_nhwc
                    and str(user_op.outputs[0]) not in model_outputs
                ):
                    removable_concat_post_indices.append(int(user_idx))
                    continue
                valid_concat_users = False
                break
            if not valid_concat_users:
                continue
            if len(q_idx_candidates) != 1:
                continue
            q_idx = int(q_idx_candidates[0])
            q_op = model_ir.operators[q_idx]
            q_out_name = str(q_op.outputs[0])
            if q_out_name in model_outputs:
                continue

            post_users = [int(v) for v in consumers.get(q_out_name, [])]
            if len(post_users) == 0:
                continue

            removable_post_indices: List[int] = []
            post_output_names: List[str] = []
            valid_posts = True
            for post_idx in post_users:
                post_op = model_ir.operators[post_idx]
                if len(post_op.inputs) < 1 or len(post_op.outputs) != 1:
                    valid_posts = False
                    break
                if str(post_op.inputs[0]) != q_out_name:
                    valid_posts = False
                    break
                is_layout_transpose = (
                    str(post_op.op_type) == "TRANSPOSE"
                    and len(post_op.inputs) >= 2
                    and _read_transpose_perm(model_ir, post_op) == rank4_perm_nchw_to_nhwc
                )
                if not is_layout_transpose and not _is_singleton_nchw_to_nhwc_reshape(post_op):
                    valid_posts = False
                    break
                post_output_name = str(post_op.outputs[0])
                if post_output_name in model_outputs:
                    valid_posts = False
                    break
                removable_post_indices.append(int(post_idx))
                post_output_names.append(post_output_name)
            if not valid_posts:
                continue

            # Collect and rewrite concat input DQ adapters.
            removable_pre_indices: List[int] = []
            convertible = True
            pending_dq_input_rewrites: Dict[int, str] = {}
            pending_quant_input_rewrites: Dict[int, str] = {}
            pending_tensor_shape_updates: Dict[str, Tuple[List[int], List[int]]] = {}
            pending_qdim_remaps: set[str] = set()
            for dq_input_name in list(concat_op.inputs):
                dq_idx = producers.get(str(dq_input_name), None)
                if dq_idx is None:
                    convertible = False
                    break
                dq_op = model_ir.operators[int(dq_idx)]
                if str(dq_op.op_type) != "DEQUANTIZE" or len(dq_op.inputs) != 1 or len(dq_op.outputs) != 1:
                    convertible = False
                    break
                if str(dq_op.outputs[0]) != str(dq_input_name):
                    convertible = False
                    break

                q_in_name = str(dq_op.inputs[0])
                if q_in_name in model_ir.outputs:
                    convertible = False
                    break

                q_in_tensor = model_ir.tensors.get(q_in_name, None)
                if q_in_tensor is None or len(list(q_in_tensor.shape)) != 4:
                    convertible = False
                    break

                q_in_producer_idx = producers.get(q_in_name, None)
                if q_in_producer_idx is None:
                    convertible = False
                    break
                q_in_producer = model_ir.operators[int(q_in_producer_idx)]

                # The input may already be a physical NHWC singleton-spatial
                # tensor (for example, a quantized global-average-pool result).
                # It needs no adapter or metadata rewrite when DEQUANTIZE is its
                # only consumer.
                q_in_users = [int(v) for v in consumers.get(q_in_name, [])]
                dq_out_tensor = model_ir.tensors.get(str(dq_op.outputs[0]), None)
                if (
                    len(q_in_users) == 1
                    and int(q_in_users[0]) == int(dq_idx)
                    and _is_nhwc_singleton_spatial(q_in_tensor)
                    and _is_nhwc_singleton_spatial(dq_out_tensor)
                ):
                    continue

                # Pattern 1:
                #   q_raw --TRANSPOSE(0,3,1,2)--> q_nchw --DEQUANTIZE--> dq
                if (
                    str(q_in_producer.op_type) == "TRANSPOSE"
                    and len(q_in_producer.inputs) >= 2
                    and len(q_in_producer.outputs) == 1
                    and _read_transpose_perm(model_ir, q_in_producer) == rank4_perm_nhwc_to_nchw
                ):
                    q_raw_name = str(q_in_producer.inputs[0])
                    q_in_users = [int(v) for v in consumers.get(q_in_name, [])]
                    if len(q_in_users) != 1 or int(q_in_users[0]) != int(dq_idx):
                        convertible = False
                        break
                    if q_in_name in model_ir.outputs:
                        convertible = False
                        break

                    q_raw_tensor = model_ir.tensors.get(q_raw_name, None)
                    dq_out_tensor = model_ir.tensors.get(str(dq_op.outputs[0]), None)
                    if q_raw_tensor is None or dq_out_tensor is None:
                        convertible = False
                        break
                    pending_dq_input_rewrites[int(dq_idx)] = str(q_raw_name)
                    pending_tensor_shape_updates[str(dq_op.outputs[0])] = (
                        [int(v) for v in list(q_raw_tensor.shape)],
                        [int(v) for v in list(q_raw_tensor.shape_signature)]
                        if q_raw_tensor.shape_signature is not None
                        else [int(v) for v in list(q_raw_tensor.shape)]
                    )
                    removable_pre_indices.append(int(q_in_producer_idx))
                    continue

                # Pattern 2:
                #   x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw --QUANTIZE--> q_nchw --DEQUANTIZE--> dq
                if (
                    str(q_in_producer.op_type) == "QUANTIZE"
                    and len(q_in_producer.inputs) == 1
                    and len(q_in_producer.outputs) == 1
                    and str(q_in_producer.outputs[0]) == q_in_name
                ):
                    q_float_name = str(q_in_producer.inputs[0])
                    q_float_producer_idx = producers.get(q_float_name, None)
                    if q_float_producer_idx is not None:
                        q_float_producer = model_ir.operators[int(q_float_producer_idx)]
                        if (
                            str(q_float_producer.op_type) == "TRANSPOSE"
                            and len(q_float_producer.inputs) >= 2
                            and len(q_float_producer.outputs) == 1
                            and _read_transpose_perm(model_ir, q_float_producer) == rank4_perm_nhwc_to_nchw
                        ):
                            q_float_users = [int(v) for v in consumers.get(q_float_name, [])]
                            if len(q_float_users) != 1 or int(q_float_users[0]) != int(q_in_producer_idx):
                                convertible = False
                                break
                            if q_float_name in model_ir.outputs:
                                convertible = False
                                break

                            permuted_q_in = _permuted_shape_signature(q_in_tensor, rank4_perm_nchw_to_nhwc)
                            if permuted_q_in is None:
                                convertible = False
                                break
                            pending_quant_input_rewrites[int(q_in_producer_idx)] = str(q_float_producer.inputs[0])
                            pending_tensor_shape_updates[q_in_name] = permuted_q_in
                            pending_qdim_remaps.add(q_in_name)

                            dq_out_tensor = model_ir.tensors.get(str(dq_op.outputs[0]), None)
                            if dq_out_tensor is None:
                                convertible = False
                                break
                            pending_tensor_shape_updates[str(dq_op.outputs[0])] = (
                                list(permuted_q_in[0]),
                                list(permuted_q_in[1]),
                            )
                            removable_pre_indices.append(int(q_float_producer_idx))
                            continue

                        # Pattern 3:
                        #   x_nhwc --RESHAPE--> x_nchw(N,C,1,1)
                        #          --QUANTIZE--> q_nchw --DEQUANTIZE--> dq
                        #
                        # TFLite commonly represents a singleton-spatial NCHW
                        # layout restore as RESHAPE instead of TRANSPOSE.  The
                        # byte order is identical for N,C,1,1 and N,1,1,C, so
                        # bypass the reshape when the source already has the
                        # prospective NHWC shape.  Updating both the QUANTIZE
                        # input and its output metadata keeps later shape
                        # reconciliation from recreating the NCHW shape.
                        if (
                            str(q_float_producer.op_type) == "RESHAPE"
                            and len(q_float_producer.inputs) >= 1
                            and len(q_float_producer.outputs) == 1
                            and str(q_float_producer.outputs[0]) == q_float_name
                            and _is_nchw_nhwc_reinterpret_safe(q_in_tensor)
                        ):
                            q_float_users = [int(v) for v in consumers.get(q_float_name, [])]
                            q_in_users = [int(v) for v in consumers.get(q_in_name, [])]
                            if (
                                len(q_float_users) == 1
                                and int(q_float_users[0]) == int(q_in_producer_idx)
                                and len(q_in_users) == 1
                                and int(q_in_users[0]) == int(dq_idx)
                                and q_float_name not in model_ir.outputs
                            ):
                                q_raw_name = str(q_float_producer.inputs[0])
                                q_raw_tensor = model_ir.tensors.get(q_raw_name, None)
                                permuted_q_in = _permuted_shape_signature(
                                    q_in_tensor,
                                    rank4_perm_nchw_to_nhwc,
                                )
                                if q_raw_tensor is not None and permuted_q_in is not None:
                                    q_raw_shape = [int(v) for v in list(q_raw_tensor.shape)]
                                    q_raw_signature = (
                                        [int(v) for v in list(q_raw_tensor.shape_signature)]
                                        if q_raw_tensor.shape_signature is not None
                                        else list(q_raw_shape)
                                    )
                                    if (
                                        q_raw_shape == list(permuted_q_in[0])
                                        and q_raw_signature == list(permuted_q_in[1])
                                    ):
                                        pending_quant_input_rewrites[int(q_in_producer_idx)] = q_raw_name
                                        pending_tensor_shape_updates[q_in_name] = permuted_q_in
                                        pending_qdim_remaps.add(q_in_name)
                                        dq_out_tensor = model_ir.tensors.get(str(dq_op.outputs[0]), None)
                                        if dq_out_tensor is None:
                                            convertible = False
                                            break
                                        pending_tensor_shape_updates[str(dq_op.outputs[0])] = (
                                            list(permuted_q_in[0]),
                                            list(permuted_q_in[1]),
                                        )
                                        removable_pre_indices.append(int(q_float_producer_idx))
                                        continue

                # Pattern 4:
                #   q_nchw --DEQUANTIZE--> dq, where q_nchw is effectively layout-invariant
                #   (e.g., NCHW N,C,1,1), so we can reinterpret metadata without data movement.
                q_in_users = [int(v) for v in consumers.get(q_in_name, [])]
                if len(q_in_users) == 1 and int(q_in_users[0]) == int(dq_idx):
                    # q_in produced directly by QUANTIZE is shape-reconciled from its float input
                    # in later passes. Metadata-only reinterpretation here is unstable and can
                    # recreate NCHW/NHWC mismatches at CONCATENATION prepare time.
                    if str(q_in_producer.op_type) == "QUANTIZE":
                        convertible = False
                        break
                    if _is_nchw_nhwc_reinterpret_safe(q_in_tensor):
                        permuted_q_in = _permuted_shape_signature(q_in_tensor, rank4_perm_nchw_to_nhwc)
                        if permuted_q_in is None:
                            convertible = False
                            break
                        pending_tensor_shape_updates[q_in_name] = permuted_q_in
                        pending_qdim_remaps.add(q_in_name)
                        dq_out_tensor = model_ir.tensors.get(str(dq_op.outputs[0]), None)
                        if dq_out_tensor is None:
                            convertible = False
                            break
                        pending_tensor_shape_updates[str(dq_op.outputs[0])] = (
                            list(permuted_q_in[0]),
                            list(permuted_q_in[1]),
                        )
                        continue

                convertible = False
                break

            if not convertible:
                continue
            if any(
                str(tensor_name) in model_outputs
                for tensor_name in pending_tensor_shape_updates
            ):
                continue

            concat_axis_old = int(concat_op.options.get("axis", 1))
            if concat_axis_old < 0 or concat_axis_old >= len(inv_nchw_to_nhwc):
                continue
            concat_axis_new = int(inv_nchw_to_nhwc[concat_axis_old])

            # Validate prospective concat input layout compatibility before committing rewrites.
            prospective_concat_shapes: List[List[int]] = []
            prospective_concat_signatures: List[List[int]] = []
            prospective_valid = True
            for concat_input_name in list(concat_op.inputs):
                tensor = model_ir.tensors.get(str(concat_input_name), None)
                if tensor is None:
                    prospective_valid = False
                    break
                pending_shape_sig = pending_tensor_shape_updates.get(str(concat_input_name), None)
                if pending_shape_sig is not None:
                    shape_i, sig_i = pending_shape_sig
                else:
                    shape_i = [int(v) for v in list(tensor.shape)]
                    sig_i = (
                        [int(v) for v in list(tensor.shape_signature)]
                        if tensor.shape_signature is not None
                        else [int(v) for v in list(tensor.shape)]
                    )
                if len(list(shape_i)) != 4 or len(list(sig_i)) != 4:
                    prospective_valid = False
                    break
                prospective_concat_shapes.append([int(v) for v in list(shape_i)])
                prospective_concat_signatures.append([int(v) for v in list(sig_i)])
            if not prospective_valid or len(prospective_concat_shapes) == 0:
                continue
            ref_shape = list(prospective_concat_shapes[0])
            for shape_i in prospective_concat_shapes[1:]:
                for dim_idx in range(4):
                    if int(dim_idx) == int(concat_axis_new):
                        continue
                    if (
                        int(ref_shape[dim_idx]) >= 0
                        and int(shape_i[dim_idx]) >= 0
                        and int(ref_shape[dim_idx]) != int(shape_i[dim_idx])
                    ):
                        prospective_valid = False
                        break
                if not prospective_valid:
                    break
            if not prospective_valid:
                continue

            concat_out_tensor = model_ir.tensors.get(concat_out_name, None)
            q_out_tensor = model_ir.tensors.get(q_out_name, None)
            if concat_out_tensor is None or q_out_tensor is None:
                continue

            for op_idx, new_input_name in pending_dq_input_rewrites.items():
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=model_ir.operators[int(op_idx)],
                    new_inputs=[str(new_input_name)],
                )
            for op_idx, new_input_name in pending_quant_input_rewrites.items():
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=model_ir.operators[int(op_idx)],
                    new_inputs=[str(new_input_name)],
                )
            for tensor_name, (new_shape, new_signature) in pending_tensor_shape_updates.items():
                tensor = model_ir.tensors.get(str(tensor_name), None)
                if tensor is None:
                    continue
                tensor.shape = [int(v) for v in list(new_shape)]
                tensor.shape_signature = [int(v) for v in list(new_signature)]
            for tensor_name in pending_qdim_remaps:
                tensor = model_ir.tensors.get(str(tensor_name), None)
                _remap_qdim_for_permute(tensor, rank4_perm_nchw_to_nhwc)

            concat_op.options["axis"] = int(concat_axis_new)

            concat_input_tensors: List[Optional[TensorIR]] = [
                model_ir.tensors.get(str(input_name), None)
                for input_name in concat_op.inputs
            ]
            if any(t is None for t in concat_input_tensors):
                continue
            first_tensor = concat_input_tensors[0]
            if first_tensor is None:
                continue
            rank = len(list(first_tensor.shape))
            if rank != 4:
                continue

            concat_shape = [int(v) for v in list(first_tensor.shape)]
            concat_signature = (
                [int(v) for v in list(first_tensor.shape_signature)]
                if first_tensor.shape_signature is not None
                else [int(v) for v in list(first_tensor.shape)]
            )
            dynamic_concat_axis = False
            for tensor in concat_input_tensors[1:]:
                if tensor is None:
                    continue
                if len(list(tensor.shape)) != rank:
                    convertible = False
                    break
                for dim_idx in range(rank):
                    if int(dim_idx) == int(concat_axis_new):
                        continue
                    if (
                        int(concat_shape[dim_idx]) >= 0
                        and int(tensor.shape[dim_idx]) >= 0
                        and int(concat_shape[dim_idx]) != int(tensor.shape[dim_idx])
                    ):
                        convertible = False
                        break
                if not convertible:
                    break
                concat_shape[concat_axis_new] += int(tensor.shape[concat_axis_new])
            if not convertible:
                continue
            for tensor in concat_input_tensors:
                if tensor is None:
                    continue
                sig = (
                    [int(v) for v in list(tensor.shape_signature)]
                    if tensor.shape_signature is not None
                    else [int(v) for v in list(tensor.shape)]
                )
                if int(sig[concat_axis_new]) < 0:
                    dynamic_concat_axis = True
                    break
            if dynamic_concat_axis:
                concat_signature[concat_axis_new] = -1
            else:
                concat_signature[concat_axis_new] = int(
                    sum(
                        int(
                            (
                                t.shape_signature[concat_axis_new]
                                if t is not None and t.shape_signature is not None
                                else t.shape[concat_axis_new]
                            )
                        )
                        for t in concat_input_tensors
                        if t is not None
                    )
                )

            concat_out_tensor.shape = [int(v) for v in concat_shape]
            concat_out_tensor.shape_signature = [int(v) for v in concat_signature]

            q_out_tensor.shape = [int(v) for v in concat_shape]
            q_out_tensor.shape_signature = [int(v) for v in concat_signature]
            _remap_qdim_for_permute(q_out_tensor, rank4_perm_nchw_to_nhwc)

            # Remove output transpose adapters and reconnect their consumers to q_out directly.
            for post_idx in removable_post_indices:
                post_op = model_ir.operators[int(post_idx)]
                post_out_name = str(post_op.outputs[0])
                _replace_tensor_inputs(model_ir, post_out_name, q_out_name)
            for post_idx in removable_concat_post_indices:
                post_op = model_ir.operators[int(post_idx)]
                post_out_name = str(post_op.outputs[0])
                _replace_tensor_inputs(model_ir, post_out_name, concat_out_name)

            remove_indices = sorted(
                set(int(v) for v in (removable_pre_indices + removable_post_indices + removable_concat_post_indices)),
                reverse=True,
            )
            for remove_idx in remove_indices:
                del model_ir.operators[int(remove_idx)]

            propagated += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir)
    return {
        "propagated_qlinear_concat_conv_nhwc_chains": int(propagated),
    }
