from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
    _read_transpose_perm,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
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
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
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
        consumers = graph_index.consumers

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
                graph_index=graph_index,
            )
            r2_op.options = {"axis": 1, "batchDims": 0}

            for remove_idx in sorted([int(r1_idx), int(t1_idx)], reverse=True):
                graph_index.remove_operator(int(remove_idx))

            optimized += 1
            changed = True
            break

        if not changed:
            break

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather": int(optimized)}


def _repair_nchw_channel_shuffle_concat_gathers(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Restore NCHW concat axis before an NCHW channel-shuffle GATHER.

    A later layout pass can remap CONCATENATION to axis=3 after the original
    reshape/transpose channel-shuffle has already been collapsed to
    GATHER(axis=1). The gather index count is the exact expected channel count,
    so it safely distinguishes this stale mixed-layout state.
    """

    repaired = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    producers = graph_index.producers
    for gather_op in model_ir.operators:
        if (
            str(gather_op.op_type) != "GATHER"
            or len(gather_op.inputs) < 2
            or len(gather_op.outputs) != 1
            or int(gather_op.options.get("axis", -1)) != 1
            or int(gather_op.options.get("batchDims", 0)) != 0
        ):
            continue
        data_name = str(gather_op.inputs[0])
        indices_tensor = model_ir.tensors.get(str(gather_op.inputs[1]), None)
        concat_idx = producers.get(data_name, None)
        if concat_idx is None or indices_tensor is None or indices_tensor.data is None:
            continue
        concat_op = model_ir.operators[int(concat_idx)]
        if (
            str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.inputs) < 2
            or len(concat_op.outputs) != 1
            or int(concat_op.options.get("axis", 1)) == 1
        ):
            continue
        input_tensors = [model_ir.tensors.get(str(name), None) for name in concat_op.inputs]
        if any(tensor is None for tensor in input_tensors):
            continue
        input_shapes = [[int(v) for v in list(tensor.shape)] for tensor in input_tensors if tensor is not None]
        if not input_shapes or any(len(shape) != 4 for shape in input_shapes):
            continue
        reference = input_shapes[0]
        if any(
            any(int(shape[axis]) != int(reference[axis]) for axis in [0, 2, 3])
            for shape in input_shapes[1:]
        ):
            continue
        expected_channels = int(sum(int(shape[1]) for shape in input_shapes))
        gather_index_count = int(np.asarray(indices_tensor.data).size)
        if expected_channels <= 0 or gather_index_count != expected_channels:
            continue

        concat_op.options["axis"] = 1
        repaired_shape = [int(v) for v in reference]
        repaired_shape[1] = int(expected_channels)
        concat_tensor = model_ir.tensors.get(data_name, None)
        gather_tensor = model_ir.tensors.get(str(gather_op.outputs[0]), None)
        for tensor in [concat_tensor, gather_tensor]:
            if tensor is None:
                continue
            tensor.shape = [int(v) for v in repaired_shape]
            tensor.shape_signature = [int(v) for v in repaired_shape]
        repaired += 1
    if repaired > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"repaired_nchw_channel_shuffle_concat_gathers": int(repaired)}


def run_stale_nchw_channel_shuffle_repair(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Repair stale NHWC Concat metadata before NCHW shuffle Gather."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        found_concat = False
        found_gather = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            found_concat = found_concat or operator_type == "CONCATENATION"
            found_gather = found_gather or operator_type == "GATHER"
            if found_concat and found_gather:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        for gather_op in candidate_model.operators:
            if (
                str(gather_op.op_type) != "GATHER"
                or len(gather_op.inputs) < 2
                or len(gather_op.outputs) != 1
                or int(gather_op.options.get("axis", -1)) != 1
                or int(gather_op.options.get("batchDims", 0)) != 0
            ):
                continue
            concat_op = pass_state.graph_index.producer(str(gather_op.inputs[0]))
            indices_tensor = candidate_model.tensors.get(str(gather_op.inputs[1]))
            if (
                concat_op is None
                or indices_tensor is None
                or indices_tensor.data is None
                or str(concat_op.op_type) != "CONCATENATION"
                or len(concat_op.inputs) < 2
                or len(concat_op.outputs) != 1
                or int(concat_op.options.get("axis", 1)) == 1
            ):
                continue
            input_tensors = [
                candidate_model.tensors.get(str(name))
                for name in concat_op.inputs
            ]
            if any(tensor is None for tensor in input_tensors):
                continue
            input_shapes = [
                [int(value) for value in tensor.shape]
                for tensor in input_tensors
                if tensor is not None
            ]
            if not input_shapes or any(len(shape) != 4 for shape in input_shapes):
                continue
            reference = input_shapes[0]
            if any(
                any(shape[axis] != reference[axis] for axis in (0, 2, 3))
                for shape in input_shapes[1:]
            ):
                continue
            expected_channels = sum(shape[1] for shape in input_shapes)
            if (
                expected_channels > 0
                and int(np.asarray(indices_tensor.data).size) == expected_channels
            ):
                return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _repair_nchw_channel_shuffle_concat_gathers(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("repaired_nchw_channel_shuffle_concat_gathers", 0)
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.repair_nchw_channel_shuffle_concat",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={"repaired_nchw_channel_shuffle_concat_gathers": 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}



def run_nchw_channel_shuffle_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Canonicalize strict static NCHW channel shuffle to Gather."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        found_reshape = False
        found_transpose = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            operator_type = str(operator.op_type)
            found_reshape = found_reshape or operator_type == "RESHAPE"
            found_transpose = found_transpose or operator_type == "TRANSPOSE"
            if found_reshape and found_transpose:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _has_candidate(pass_state: ModelIRPassState) -> bool:
        candidate_model = pass_state.model_ir
        for r1_op in candidate_model.operators:
            if (
                str(r1_op.op_type) != "RESHAPE"
                or len(r1_op.inputs) < 1
                or len(r1_op.outputs) != 1
            ):
                continue
            x_name = str(r1_op.inputs[0])
            r1_output_name = str(r1_op.outputs[0])
            r1_users = pass_state.graph_index.consumer_indices(r1_output_name)
            if len(r1_users) != 1:
                continue
            transpose_op = candidate_model.operators[int(r1_users[0])]
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.inputs) < 2
                or len(transpose_op.outputs) != 1
                or str(transpose_op.inputs[0]) != r1_output_name
                or _read_transpose_perm(candidate_model, transpose_op)
                != [0, 2, 1, 3, 4]
            ):
                continue
            transpose_output_name = str(transpose_op.outputs[0])
            transpose_users = pass_state.graph_index.consumer_indices(
                transpose_output_name
            )
            if len(transpose_users) != 1:
                continue
            r2_op = candidate_model.operators[int(transpose_users[0])]
            if (
                str(r2_op.op_type) != "RESHAPE"
                or len(r2_op.inputs) < 1
                or len(r2_op.outputs) != 1
                or str(r2_op.inputs[0]) != transpose_output_name
            ):
                continue
            tensors = [
                candidate_model.tensors.get(name)
                for name in (
                    x_name,
                    r1_output_name,
                    transpose_output_name,
                    str(r2_op.outputs[0]),
                )
            ]
            if any(tensor is None for tensor in tensors):
                continue
            shapes = [
                [int(value) for value in tensor.shape]
                for tensor in tensors
                if tensor is not None
            ]
            if not all(_is_fully_known_positive_shape(shape) for shape in shapes):
                continue
            x_shape, r1_shape, transpose_shape, y_shape = shapes
            if [len(shape) for shape in shapes] != [4, 5, 5, 4]:
                continue
            n, channels, height, width = x_shape
            groups = int(r1_shape[1])
            channels_per_group = int(r1_shape[2])
            if (
                groups > 1
                and channels_per_group > 0
                and groups * channels_per_group == channels
                and r1_shape == [n, groups, channels_per_group, height, width]
                and transpose_shape
                == [n, channels_per_group, groups, height, width]
                and y_shape == [n, channels, height, width]
            ):
                shuffle_indices = np.asarray(
                    [
                        (index % groups) * channels_per_group
                        + (index // groups)
                        for index in range(channels)
                    ],
                    dtype=np.int32,
                )
                if not np.array_equal(
                    shuffle_indices,
                    np.arange(channels, dtype=np.int32),
                ):
                    return True
        return False

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather",
                    0,
                )
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.nchw_channel_shuffle_gather",
                phase=PassPhase.CANONICALIZE,
                callback=_run,
                precondition=_has_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={
            "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
