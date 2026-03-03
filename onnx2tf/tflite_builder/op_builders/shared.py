from __future__ import annotations

import copy
from typing import Any, List, Optional

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR, QuantParamIR


def _read_transpose_perm(ctx: Any, op: OperatorIR) -> Optional[List[int]]:
    if str(op.op_type) != "TRANSPOSE":
        return None
    if len(op.inputs) < 2:
        return None
    perm_tensor = ctx.model_ir.tensors.get(op.inputs[1], None)
    if perm_tensor is None or perm_tensor.data is None:
        return None
    perm = np.asarray(perm_tensor.data).reshape(-1)
    if perm.size == 0:
        return None
    return [int(v) for v in perm.tolist()]


def _is_inverse_perm(perm_a: List[int], perm_b: List[int]) -> bool:
    if len(perm_a) != len(perm_b):
        return False
    rank = len(perm_a)
    if sorted(perm_a) != [int(i) for i in range(rank)]:
        return False
    if sorted(perm_b) != [int(i) for i in range(rank)]:
        return False
    for idx, value in enumerate(perm_a):
        if perm_b[value] != idx:
            return False
    return True


def _clone_quantization(quantization: Any) -> Any:
    if quantization is None:
        return None
    if isinstance(quantization, QuantParamIR):
        return QuantParamIR(
            scale=list(quantization.scale),
            zero_point=list(quantization.zero_point),
            quantized_dimension=int(quantization.quantized_dimension),
            min=list(quantization.min) if quantization.min is not None else None,
            max=list(quantization.max) if quantization.max is not None else None,
        )
    return copy.deepcopy(quantization)


def _quant_param_scale_count(quantization: Any) -> int:
    if quantization is None:
        return 0
    if isinstance(quantization, QuantParamIR):
        return int(len(list(quantization.scale)))
    if isinstance(quantization, dict):
        scale = quantization.get("scale", [])
        if isinstance(scale, np.ndarray):
            return int(scale.size)
        if isinstance(scale, (list, tuple)):
            return int(len(scale))
    return 0


def _get_quantized_dimension(quantization: Any) -> Optional[int]:
    if quantization is None:
        return None
    if isinstance(quantization, QuantParamIR):
        return int(quantization.quantized_dimension)
    if isinstance(quantization, dict) and "quantized_dimension" in quantization:
        try:
            return int(quantization.get("quantized_dimension", 0))
        except Exception:
            return None
    return None


def _set_quantized_dimension(quantization: Any, qdim: int) -> None:
    if quantization is None:
        return
    if isinstance(quantization, QuantParamIR):
        quantization.quantized_dimension = int(qdim)
        return
    if isinstance(quantization, dict):
        quantization["quantized_dimension"] = int(qdim)


def _invert_perm(perm: List[int]) -> Optional[List[int]]:
    rank = len(list(perm))
    if sorted(int(v) for v in perm) != [int(i) for i in range(rank)]:
        return None
    inv = [0] * rank
    for out_axis, in_axis in enumerate(perm):
        inv[int(in_axis)] = int(out_axis)
    return inv


def _remap_quantized_dimension_for_transpose(quantization: Any, perm_values: List[int]) -> Any:
    if quantization is None:
        return None
    scale_count = _quant_param_scale_count(quantization)
    if scale_count <= 1:
        return quantization
    qdim = _get_quantized_dimension(quantization)
    if qdim is None:
        return quantization
    inv = _invert_perm([int(v) for v in list(perm_values)])
    if inv is None:
        return quantization
    if 0 <= int(qdim) < len(inv):
        _set_quantized_dimension(quantization, int(inv[int(qdim)]))
    return quantization


def resolve_padding(node: Any) -> str:
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET"))
    if auto_pad in ["SAME_UPPER", "SAME_LOWER"]:
        return "SAME"
    pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
    if len(pads) == 4 and sum([abs(int(v)) for v in pads]) == 0:
        return "VALID"
    if len(pads) == 4:
        top, left, bottom, right = [int(v) for v in pads]
        if top == bottom and left == right and top >= 0 and left >= 0:
            return "SAME"
        # ONNX exports from TF frequently encode SAME padding as asymmetric pads
        # (e.g. [0, 0, 1, 1]). TFLite SAME supports this distribution internally.
        if (
            top >= 0
            and left >= 0
            and bottom >= 0
            and right >= 0
            and abs(top - bottom) <= 1
            and abs(left - right) <= 1
        ):
            return "SAME"
    raise NotImplementedError(
        "Only zero pads, symmetric pads, or SAME auto_pad are supported in flatbuffer_direct. "
        f"op={node.name} pads={pads} auto_pad={auto_pad}"
    )


def _is_valid_perm(perm_values: List[int], rank: int) -> bool:
    perm = [int(v) for v in list(perm_values)]
    if len(perm) != int(rank):
        return False
    return sorted(perm) == [int(i) for i in range(int(rank))]


def _decompose_transpose_over_rank6(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    perm_values: List[int],
) -> Optional[str]:
    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    input_rank = int(len(input_shape))
    target_rank = int(
        max(
            2,
            min(
                6,
                int(getattr(ctx, "number_of_dimensions_after_flextranspose_compression", 6)),
            ),
        )
    )
    if input_rank <= int(target_rank):
        return None
    if not _is_valid_perm(perm_values, input_rank):
        return None
    if any(int(v) <= 0 for v in input_shape):
        return None

    num_dims_to_compress = int(input_rank - int(target_rank))
    sorted_minimum_idxs = np.argsort(np.asarray(input_shape, dtype=np.int64))[
        :num_dims_to_compress
    ].tolist()
    if len(sorted_minimum_idxs) != num_dims_to_compress:
        return None

    target_minimum_dims = [int(input_shape[int(idx)]) for idx in sorted_minimum_idxs]
    if any(int(v) <= 0 for v in target_minimum_dims):
        return None

    removed_split_perm = [int(dim) for dim in perm_values if int(dim) not in sorted_minimum_idxs]
    sorted_removed_split_perm = sorted(removed_split_perm)
    removed_splited_transpose_perm = [
        int(sorted_removed_split_perm.index(int(idx)))
        for idx in removed_split_perm
    ]
    target_sorted_minimum_idxs = [
        int([int(v) for v in perm_values].index(int(idx)))
        for idx in sorted_minimum_idxs
    ]

    rank_tag = f"rank{int(target_rank)}"
    split_tensors = [str(input_name)]
    split_axes = [int(v) for v in list(sorted_minimum_idxs)]
    split_step = 0
    while split_step < len(split_axes):
        axis = int(split_axes[split_step])
        next_split_tensors: List[str] = []
        for tensor_idx, split_tensor_name in enumerate(split_tensors):
            split_tensor_shape = [int(v) for v in list(ctx.get_tensor_shape(split_tensor_name))]
            axis_dim = int(split_tensor_shape[axis])
            if axis_dim <= 0:
                return None
            split_tensor_ir = ctx.model_ir.tensors.get(split_tensor_name, None)
            split_tensor_signature = (
                list(split_tensor_ir.shape_signature)
                if split_tensor_ir is not None and split_tensor_ir.shape_signature is not None
                else list(split_tensor_shape)
            )
            for gather_index in range(axis_dim):
                gather_index_name = ctx.add_const_tensor(
                    f"{output_name}_{rank_tag}_split_axis{axis}_idx{gather_index}",
                    np.asarray(int(gather_index), dtype=np.int32),
                )
                gather_index_ir = ctx.model_ir.tensors.get(gather_index_name, None)
                if gather_index_ir is not None:
                    gather_index_ir.shape = []
                    gather_index_ir.shape_signature = []
                gathered_shape = [int(v) for i, v in enumerate(split_tensor_shape) if int(i) != int(axis)]
                gathered_signature = [
                    int(v) for i, v in enumerate(split_tensor_signature) if int(i) != int(axis)
                ]
                gathered_name = ctx.add_intermediate_tensor(
                    f"{output_name}_{rank_tag}_split_{split_step}_{tensor_idx}_{gather_index}",
                    dtype=ctx.get_tensor_dtype(split_tensor_name),
                    shape=list(gathered_shape),
                )
                gathered_ir = ctx.model_ir.tensors.get(gathered_name, None)
                if gathered_ir is not None:
                    gathered_ir.shape_signature = [int(v) for v in list(gathered_signature)]
                    if split_tensor_ir is not None:
                        gathered_ir.quantization = _clone_quantization(split_tensor_ir.quantization)
                ctx.add_operator(
                    OperatorIR(
                        op_type="GATHER",
                        inputs=[split_tensor_name, gather_index_name],
                        outputs=[gathered_name],
                        options={"axis": int(axis), "batchDims": 0},
                    )
                )
                next_split_tensors.append(str(gathered_name))
        split_tensors = next_split_tensors
        split_step += 1
        if split_step >= len(split_axes):
            break
        current_axis = int(axis)
        split_axes = [
            int(v) if int(v) <= int(current_axis) else int(v) - 1
            for v in split_axes
        ]

    transposed_tensors: List[str] = []
    for idx, split_tensor_name in enumerate(split_tensors):
        split_tensor_shape = [int(v) for v in list(ctx.get_tensor_shape(split_tensor_name))]
        transposed_shape = [
            int(split_tensor_shape[int(axis)])
            for axis in removed_splited_transpose_perm
        ]
        transposed_name = ctx.add_intermediate_tensor(
            f"{output_name}_{rank_tag}_transposed_{idx}",
            dtype=ctx.get_tensor_dtype(split_tensor_name),
            shape=list(transposed_shape),
        )
        transposed_name = make_transpose(
            ctx=ctx,
            input_name=split_tensor_name,
            output_name=transposed_name,
            perm_values=[int(v) for v in list(removed_splited_transpose_perm)],
            allow_elide_inverse_chain=False,
        )
        transposed_tensors.append(str(transposed_name))

    expanded_tensors = list(transposed_tensors)
    for expand_axis in sorted([int(v) for v in list(target_sorted_minimum_idxs)]):
        axis_name = ctx.add_const_tensor(
            f"{output_name}_{rank_tag}_expand_axis_{expand_axis}",
            np.asarray([int(expand_axis)], dtype=np.int32),
        )
        next_expanded: List[str] = []
        for idx, tensor_name in enumerate(expanded_tensors):
            tensor_shape = [int(v) for v in list(ctx.get_tensor_shape(tensor_name))]
            expanded_shape = (
                [int(v) for v in tensor_shape[: int(expand_axis)]]
                + [1]
                + [int(v) for v in tensor_shape[int(expand_axis):]]
            )
            expanded_name = ctx.add_intermediate_tensor(
                f"{output_name}_{rank_tag}_expanded_{expand_axis}_{idx}",
                dtype=ctx.get_tensor_dtype(tensor_name),
                shape=list(expanded_shape),
            )
            tensor_ir = ctx.model_ir.tensors.get(tensor_name, None)
            expanded_ir = ctx.model_ir.tensors.get(expanded_name, None)
            if tensor_ir is not None and expanded_ir is not None:
                tensor_signature = (
                    list(tensor_ir.shape_signature)
                    if tensor_ir.shape_signature is not None
                    else list(tensor_shape)
                )
                expanded_ir.shape_signature = (
                    [int(v) for v in tensor_signature[: int(expand_axis)]]
                    + [1]
                    + [int(v) for v in tensor_signature[int(expand_axis):]]
                )
                expanded_ir.quantization = _clone_quantization(tensor_ir.quantization)
            ctx.add_operator(
                OperatorIR(
                    op_type="EXPAND_DIMS",
                    inputs=[tensor_name, axis_name],
                    outputs=[expanded_name],
                )
            )
            next_expanded.append(str(expanded_name))
        expanded_tensors = next_expanded

    grouped_tensors = list(expanded_tensors)
    concat_axes = list(reversed([int(v) for v in list(target_sorted_minimum_idxs)]))
    grouping_dims = list(reversed([int(v) for v in list(target_minimum_dims)]))
    for stage_idx, (concat_axis, target_concat_dim) in enumerate(zip(concat_axes, grouping_dims)):
        if int(target_concat_dim) <= 0:
            return None
        next_grouped: List[str] = []
        for group_idx in range(0, len(grouped_tensors), int(target_concat_dim)):
            chunk = grouped_tensors[group_idx: group_idx + int(target_concat_dim)]
            if len(chunk) == 0:
                continue
            if len(chunk) == 1:
                next_grouped.append(str(chunk[0]))
                continue
            concat_out = f"{output_name}_{rank_tag}_concat_{stage_idx}_{group_idx // int(target_concat_dim)}"
            if stage_idx == int(len(concat_axes) - 1) and len(grouped_tensors) == len(chunk):
                concat_out = output_name
            concat_shape = [int(v) for v in list(ctx.get_tensor_shape(chunk[0]))]
            concat_shape[int(concat_axis)] = int(
                sum(int(ctx.get_tensor_shape(name)[int(concat_axis)]) for name in chunk)
            )
            if concat_out != output_name:
                ctx.add_intermediate_tensor(
                    concat_out,
                    dtype=ctx.get_tensor_dtype(chunk[0]),
                    shape=list(concat_shape),
                )
                concat_out_ir = ctx.model_ir.tensors.get(concat_out, None)
                chunk_ir = ctx.model_ir.tensors.get(chunk[0], None)
                if concat_out_ir is not None and chunk_ir is not None:
                    chunk_signature = (
                        list(chunk_ir.shape_signature)
                        if chunk_ir.shape_signature is not None
                        else [int(v) for v in list(ctx.get_tensor_shape(chunk[0]))]
                    )
                    concat_signature = [int(v) for v in list(chunk_signature)]
                    if 0 <= int(concat_axis) < len(concat_signature):
                        concat_signature[int(concat_axis)] = int(
                            sum(int(ctx.get_tensor_shape(name)[int(concat_axis)]) for name in chunk)
                        )
                    concat_out_ir.shape_signature = [int(v) for v in concat_signature]
                    concat_out_ir.quantization = _clone_quantization(chunk_ir.quantization)
            ctx.add_operator(
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=[str(v) for v in chunk],
                    outputs=[concat_out],
                    options={"axis": int(concat_axis), "fusedActivationFunction": "NONE"},
                )
            )
            next_grouped.append(str(concat_out))
        grouped_tensors = next_grouped

    if len(grouped_tensors) != 1:
        return None
    final_name = str(grouped_tensors[0])
    if final_name != output_name:
        out_shape = [int(v) for v in list(ctx.get_tensor_shape(final_name))]
        out_shape_name = ctx.add_const_tensor(
            f"{output_name}_{rank_tag}_identity_shape",
            np.asarray(out_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[final_name, out_shape_name],
                outputs=[output_name],
                options={"newShape": [int(v) for v in out_shape]},
            )
        )
    return output_name


def make_transpose(
    ctx: Any,
    input_name: str,
    output_name: str,
    perm_values: List[int],
    allow_elide_inverse_chain: bool = False,
) -> str:
    perm = [int(v) for v in list(perm_values)]
    if allow_elide_inverse_chain:
        producer_idx = None
        producer_op = None
        for op_idx in range(len(ctx.model_ir.operators) - 1, -1, -1):
            op = ctx.model_ir.operators[op_idx]
            if output_name in op.outputs:
                # output_name is already produced by some op, so do not alias.
                producer_op = None
                producer_idx = None
                break
            if input_name in op.outputs:
                producer_op = op
                producer_idx = int(op_idx)
                break
        if producer_op is not None and str(producer_op.op_type) == "TRANSPOSE":
            prev_perm = _read_transpose_perm(ctx, producer_op)
            if prev_perm is not None and _is_inverse_perm(prev_perm, [int(v) for v in perm_values]):
                prev_input_name = producer_op.inputs[0]
                prev_input_shape = list(ctx.get_tensor_shape(prev_input_name))
                output_shape = list(ctx.get_tensor_shape(output_name))
                if prev_input_shape == output_shape:
                    consumer_count = int(ctx.tensor_consumer_count.get(str(input_name), 0))
                    if (
                        consumer_count <= 1
                        and str(input_name) not in set(ctx.graph_output_names)
                        and producer_idx is not None
                    ):
                        # input_name is an ONNX edge consumed only by this transpose.
                        # Remove the previous transpose as well to avoid leaving dead ops.
                        del ctx.model_ir.operators[int(producer_idx)]
                    return prev_input_name

    input_shape = [int(v) for v in list(ctx.get_tensor_shape(input_name))]
    target_rank = int(
        max(
            2,
            min(
                6,
                int(getattr(ctx, "number_of_dimensions_after_flextranspose_compression", 6)),
            ),
        )
    )
    if (
        len(input_shape) > int(target_rank)
        and _is_valid_perm(perm, len(input_shape))
        and perm != [int(v) for v in range(len(input_shape))]
    ):
        decomposed_output = _decompose_transpose_over_rank6(
            ctx=ctx,
            input_name=input_name,
            output_name=output_name,
            perm_values=perm,
        )
        if decomposed_output is not None:
            input_tensor = ctx.model_ir.tensors.get(input_name, None)
            output_tensor = ctx.model_ir.tensors.get(output_name, None)
            if input_tensor is not None and output_tensor is not None:
                if len(input_tensor.shape) == len(perm):
                    output_tensor.shape = [int(input_tensor.shape[int(axis)]) for axis in perm]
                input_signature = (
                    list(input_tensor.shape_signature)
                    if input_tensor.shape_signature is not None
                    else list(input_tensor.shape)
                )
                if len(input_signature) == len(perm):
                    output_tensor.shape_signature = [int(input_signature[int(axis)]) for axis in perm]
                output_tensor.quantization = _clone_quantization(input_tensor.quantization)
                output_tensor.quantization = _remap_quantized_dimension_for_transpose(
                    output_tensor.quantization,
                    perm,
                )
            return str(decomposed_output)

    perm_name = ctx.add_const_tensor(
        f"{output_name}_perm",
        np.asarray(perm, dtype=np.int32),
    )
    ctx.add_operator(
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[input_name, perm_name],
            outputs=[output_name],
        )
    )
    input_tensor = ctx.model_ir.tensors.get(input_name, None)
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if input_tensor is not None and output_tensor is not None:
        if len(input_tensor.shape) == len(perm):
            output_tensor.shape = [int(input_tensor.shape[int(axis)]) for axis in perm]
        input_signature = (
            list(input_tensor.shape_signature)
            if input_tensor.shape_signature is not None
            else list(input_tensor.shape)
        )
        if len(input_signature) == len(perm):
            output_tensor.shape_signature = [int(input_signature[int(axis)]) for axis in perm]
        output_tensor.quantization = _clone_quantization(input_tensor.quantization)
        output_tensor.quantization = _remap_quantized_dimension_for_transpose(
            output_tensor.quantization,
            perm,
        )
    return output_name
