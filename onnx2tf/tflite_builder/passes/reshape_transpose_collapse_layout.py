from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Collapse a private rank-three layout shim into one NHWC Reshape."""

    stats_key = (
        "optimized_reshape_transpose_reshape_transpose_to_nhwc_"
        "reshape_chains"
    )
    perm_nsc1_to_n1cs = [0, 3, 2, 1]
    perm_nchw_to_nhwc = [0, 2, 3, 1]
    rewritten = 0

    def _metadata(
        tensor: Optional[TensorIR],
        rank: int,
    ) -> Optional[Tuple[List[int], List[int]]]:
        if tensor is None:
            return None
        try:
            shape = [int(value) for value in tensor.shape]
            signature = (
                [int(value) for value in tensor.shape_signature]
                if tensor.shape_signature is not None
                else list(shape)
            )
        except (TypeError, ValueError):
            return None
        if (
            len(shape) != int(rank)
            or len(signature) != int(rank)
            or any(int(value) <= 0 for value in shape)
        ):
            return None
        return shape, signature

    reserved_tensor_names = {
        str(name)
        for name in (
            list(model_ir.tensors)
            + list(model_ir.inputs)
            + list(model_ir.outputs)
            + [
                value
                for operator in model_ir.operators
                for value in list(operator.inputs) + list(operator.outputs)
            ]
        )
    }

    def _unique_tensor_name(base: str, reserved_names: set[str]) -> str:
        candidate = str(base)
        serial = 1
        while candidate in reserved_names:
            candidate = f"{base}_{serial}"
            serial += 1
        reserved_names.add(candidate)
        return candidate

    def _tensor_input_sites(
        tensor_name: str,
        graph_index: ModelIRGraphIndex,
    ) -> set[Tuple[int, int]]:
        sites: set[Tuple[int, int]] = set()
        for operator_index in set(
            graph_index.consumer_indices(tensor_name)
        ):
            operator = model_ir.operators[int(operator_index)]
            for input_index, input_name in enumerate(operator.inputs):
                if str(input_name) == str(tensor_name):
                    sites.add((int(operator_index), int(input_index)))
        return sites

    graph_index = ModelIRGraphIndex(model_ir)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_boundaries = public_inputs | public_outputs

    while True:
        changed = False
        for reshape1_idx in graph_index.operator_indices("RESHAPE"):
            reshape1_op = model_ir.operators[int(reshape1_idx)]
            if (
                len(reshape1_op.inputs) < 2
                or len(reshape1_op.outputs) != 1
            ):
                continue
            input_name = str(reshape1_op.inputs[0])
            reshape1_out_name = str(reshape1_op.outputs[0])
            if (
                reshape1_out_name in public_boundaries
                or reshape1_out_name in graph_index.duplicate_producers
            ):
                continue
            source_producer_idx = graph_index.producers.get(input_name)
            if (
                input_name in graph_index.duplicate_producers
                or (
                    source_producer_idx is not None
                    and int(source_producer_idx) >= int(reshape1_idx)
                )
            ):
                continue

            reshape1_users = graph_index.consumer_indices(
                reshape1_out_name
            )
            if len(reshape1_users) != 1:
                continue
            transpose1_idx = int(reshape1_users[0])
            if transpose1_idx <= int(reshape1_idx):
                continue
            transpose1_op = model_ir.operators[transpose1_idx]
            if (
                str(transpose1_op.op_type) != "TRANSPOSE"
                or len(transpose1_op.inputs) < 2
                or len(transpose1_op.outputs) != 1
                or str(transpose1_op.inputs[0]) != reshape1_out_name
                or _read_transpose_perm(model_ir, transpose1_op)
                != perm_nsc1_to_n1cs
            ):
                continue

            transpose1_out_name = str(transpose1_op.outputs[0])
            if (
                transpose1_out_name in public_boundaries
                or transpose1_out_name in graph_index.duplicate_producers
            ):
                continue
            transpose1_users = graph_index.consumer_indices(
                transpose1_out_name
            )
            if len(transpose1_users) != 1:
                continue
            reshape2_idx = int(transpose1_users[0])
            if reshape2_idx <= int(transpose1_idx):
                continue
            reshape2_op = model_ir.operators[reshape2_idx]
            if (
                str(reshape2_op.op_type) != "RESHAPE"
                or len(reshape2_op.inputs) < 2
                or len(reshape2_op.outputs) != 1
                or str(reshape2_op.inputs[0]) != transpose1_out_name
            ):
                continue

            reshape2_out_name = str(reshape2_op.outputs[0])
            if (
                reshape2_out_name in public_boundaries
                or reshape2_out_name in graph_index.duplicate_producers
            ):
                continue
            reshape2_users = graph_index.consumer_indices(
                reshape2_out_name
            )
            if len(reshape2_users) != 1:
                continue
            transpose2_idx = int(reshape2_users[0])
            if transpose2_idx <= int(reshape2_idx):
                continue
            transpose2_op = model_ir.operators[transpose2_idx]
            if (
                str(transpose2_op.op_type) != "TRANSPOSE"
                or len(transpose2_op.inputs) < 2
                or len(transpose2_op.outputs) != 1
                or str(transpose2_op.inputs[0]) != reshape2_out_name
                or _read_transpose_perm(model_ir, transpose2_op)
                != perm_nchw_to_nhwc
            ):
                continue

            output_name = str(transpose2_op.outputs[0])
            if (
                output_name in public_inputs
                or output_name in graph_index.duplicate_producers
            ):
                continue
            input_metadata = _metadata(
                model_ir.tensors.get(input_name),
                3,
            )
            reshape1_metadata = _metadata(
                model_ir.tensors.get(reshape1_out_name),
                4,
            )
            transpose1_metadata = _metadata(
                model_ir.tensors.get(transpose1_out_name),
                4,
            )
            reshape2_metadata = _metadata(
                model_ir.tensors.get(reshape2_out_name),
                4,
            )
            output_metadata = _metadata(
                model_ir.tensors.get(output_name),
                4,
            )
            if any(
                metadata is None
                for metadata in (
                    input_metadata,
                    reshape1_metadata,
                    transpose1_metadata,
                    reshape2_metadata,
                    output_metadata,
                )
            ):
                continue
            assert input_metadata is not None
            assert reshape1_metadata is not None
            assert transpose1_metadata is not None
            assert reshape2_metadata is not None
            assert output_metadata is not None
            in_shape, in_signature = input_metadata
            reshape1_shape, reshape1_signature = reshape1_metadata
            transpose1_shape, transpose1_signature = (
                transpose1_metadata
            )
            reshape2_shape, reshape2_signature = reshape2_metadata
            output_shape, output_signature = output_metadata

            n, s, c = in_shape
            if reshape1_shape != [n, s, c, 1]:
                continue
            if transpose1_shape != [n, 1, c, s]:
                continue
            n2, c2, h, w = reshape2_shape
            if (
                n2 != n
                or c2 != c
                or int(h) <= 0
                or int(w) <= 0
                or int(s) != int(h) * int(w)
                or output_shape != [n, h, w, c]
            ):
                continue

            batch_signatures = [
                int(in_signature[0]),
                int(reshape1_signature[0]),
                int(transpose1_signature[0]),
                int(reshape2_signature[0]),
                int(output_signature[0]),
            ]
            if any(
                value not in {-1, int(n)}
                for value in batch_signatures
            ):
                continue
            expected_nonbatch_signatures = (
                (in_signature[1:], [s, c]),
                (reshape1_signature[1:], [s, c, 1]),
                (transpose1_signature[1:], [1, c, s]),
                (reshape2_signature[1:], [c, h, w]),
                (output_signature[1:], [h, w, c]),
            )
            if any(
                [int(value) for value in signature] != expected
                for signature, expected in expected_nonbatch_signatures
            ):
                continue
            target_batch = (
                -1 if -1 in batch_signatures else int(n)
            )
            target_shape = [target_batch, int(h), int(w), int(c)]

            shape_name = str(reshape1_op.inputs[1])
            shape_tensor = model_ir.tensors.get(shape_name)
            if (
                shape_name in public_inputs
                or shape_name in graph_index.producers
                or shape_name in graph_index.duplicate_producers
                or shape_tensor is None
                or shape_tensor.data is None
                or bool(shape_tensor.is_variable)
                or str(shape_tensor.dtype) != "INT32"
                or shape_tensor.quantization is not None
            ):
                continue
            try:
                shape_tensor_shape = [
                    int(value) for value in shape_tensor.shape
                ]
                shape_tensor_signature = (
                    [
                        int(value)
                        for value in shape_tensor.shape_signature
                    ]
                    if shape_tensor.shape_signature is not None
                    else list(shape_tensor_shape)
                )
                shape_array = np.asarray(shape_tensor.data)
            except (TypeError, ValueError):
                continue
            if (
                shape_tensor_shape != [4]
                or shape_tensor_signature != [4]
                or shape_array.dtype != np.dtype(np.int32)
                or list(shape_array.shape) != [4]
                or [
                    int(value)
                    for value in shape_array.reshape(-1).tolist()
                ]
                != reshape1_shape
            ):
                continue

            candidate_reserved_names = set(reserved_tensor_names)
            planned_site = {(int(reshape1_idx), 1)}
            shared_outside_plan = any(
                site not in planned_site
                for site in _tensor_input_sites(
                    shape_name,
                    graph_index,
                )
            )
            shape_mode = "update"
            target_shape_name = shape_name
            if (
                shared_outside_plan
                or shape_name in public_outputs
            ):
                shape_mode = "clone"
                target_shape_name = _unique_tensor_name(
                    f"{shape_name}_nhwc",
                    candidate_reserved_names,
                )

            reshape1_inputs = [
                str(value) for value in reshape1_op.inputs
            ]
            reshape1_inputs[1] = target_shape_name
            reshape1_options = (
                dict(reshape1_op.options)
                if isinstance(reshape1_op.options, dict)
                else reshape1_op.options
            )
            if isinstance(reshape1_options, dict):
                for key in ("newShape", "onnxRawNewShape"):
                    if isinstance(reshape1_options.get(key), list):
                        reshape1_options[key] = list(target_shape)
            remove_indices = {
                int(transpose1_idx),
                int(reshape2_idx),
                int(transpose2_idx),
            }

            # Topology, metadata, dynamic shape, ownership, options, output,
            # removal, and pruning decisions are complete.
            reserved_tensor_names.update(candidate_reserved_names)

            if shape_mode == "clone":
                model_ir.tensors[target_shape_name] = TensorIR(
                    name=target_shape_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(target_shape, dtype=np.int32),
                    is_variable=False,
                    quantization=None,
                    logical_layout=str(shape_tensor.logical_layout),
                    physical_layout=str(shape_tensor.physical_layout),
                    onnx_tensor_name=shape_tensor.onnx_tensor_name,
                )
            else:
                shape_tensor.data = np.asarray(
                    target_shape,
                    dtype=np.int32,
                )
                shape_tensor.shape = [4]
                shape_tensor.shape_signature = [4]
                shape_tensor.quantization = None

            _set_operator_inputs(
                model_ir=model_ir,
                op=reshape1_op,
                new_inputs=reshape1_inputs,
                graph_index=graph_index,
            )
            if isinstance(reshape1_options, dict):
                reshape1_op.options = reshape1_options
            _set_operator_outputs(
                model_ir=model_ir,
                op=reshape1_op,
                new_outputs=[output_name],
                graph_index=graph_index,
            )
            graph_index.remove_operators(remove_indices)

            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {stats_key: int(rewritten)}
