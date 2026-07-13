from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    ModelIRPreflightResult,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
_UNARY_OPS = {
    "RELU",
    "RELU6",
    "LEAKY_RELU",
    "LOGISTIC",
    "TANH",
    "HARD_SWISH",
}
_CONV_LIKE_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D"}


@dataclass(frozen=True)
class _ConcatUnaryConvCandidate:
    pre_ops: Tuple[OperatorIR, ...]
    concat_sources: Tuple[str, ...]
    concat_op: OperatorIR
    concat_output: str
    unary_ops: Tuple[OperatorIR, ...]
    unary_outputs: Tuple[str, ...]
    tail_name: str
    post_ops: Tuple[OperatorIR, ...]
    post_outputs: Tuple[str, ...]


def _rank_four_tensor(model_ir: ModelIR, tensor_name: str) -> bool:
    tensor = model_ir.tensors.get(str(tensor_name))
    return bool(
        tensor is not None
        and tensor.shape is not None
        and len(tensor.shape) == 4
    )


def _resolve_concat_unary_conv_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_ConcatUnaryConvCandidate]:
    model_outputs = {str(name) for name in model_ir.outputs}
    for concat_op in model_ir.operators:
        concat_index = graph_index.operator_index(concat_op)
        if (
            concat_index is None
            or str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.outputs) != 1
        ):
            continue
        axis = int(concat_op.options.get("axis", 1))
        if axis < 0:
            axis += 4
        concat_output = str(concat_op.outputs[0])
        if (
            axis != 1
            or concat_output in model_outputs
            or not _rank_four_tensor(model_ir, concat_output)
        ):
            continue

        pre_ops: List[OperatorIR] = []
        concat_sources: List[str] = []
        valid_inputs = True
        for concat_input in [str(name) for name in concat_op.inputs]:
            pre_op = graph_index.producer(concat_input)
            pre_index = (
                None if pre_op is None else graph_index.operator_index(pre_op)
            )
            if (
                pre_op is None
                or pre_index is None
                or str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != concat_input
                or _read_transpose_perm(model_ir, pre_op)
                != _PERM_NHWC_TO_NCHW
                or concat_input in model_outputs
                or set(graph_index.consumer_indices(concat_input))
                != {int(concat_index)}
            ):
                valid_inputs = False
                break
            source_name = str(pre_op.inputs[0])
            if not _rank_four_tensor(model_ir, source_name):
                valid_inputs = False
                break
            pre_ops.append(pre_op)
            concat_sources.append(source_name)
        if not valid_inputs or not pre_ops:
            continue

        unary_ops: List[OperatorIR] = []
        unary_outputs: List[str] = []
        tail_name = concat_output
        while True:
            tail_users = graph_index.consumer_indices(tail_name)
            if len(tail_users) != 1:
                break
            unary_op = model_ir.operators[int(tail_users[0])]
            if (
                str(unary_op.op_type) not in _UNARY_OPS
                or len(unary_op.inputs) != 1
                or len(unary_op.outputs) != 1
                or str(unary_op.inputs[0]) != tail_name
            ):
                break
            unary_output = str(unary_op.outputs[0])
            if (
                unary_output in model_outputs
                or not _rank_four_tensor(model_ir, unary_output)
            ):
                break
            unary_ops.append(unary_op)
            unary_outputs.append(unary_output)
            tail_name = unary_output

        post_ops: List[OperatorIR] = []
        post_outputs: List[str] = []
        valid_tail = True
        tail_users = graph_index.consumer_indices(tail_name)
        if not tail_users:
            continue
        for user_index in tail_users:
            post_op = model_ir.operators[int(user_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != tail_name
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
            ):
                valid_tail = False
                break
            post_output = str(post_op.outputs[0])
            if (
                post_output in model_outputs
                or not _rank_four_tensor(model_ir, post_output)
            ):
                valid_tail = False
                break
            post_users = graph_index.consumer_indices(post_output)
            if not post_users or any(
                str(model_ir.operators[int(index)].op_type)
                not in _CONV_LIKE_OPS
                for index in post_users
            ):
                valid_tail = False
                break
            post_ops.append(post_op)
            post_outputs.append(post_output)
        if not valid_tail or not post_ops:
            continue

        return _ConcatUnaryConvCandidate(
            pre_ops=tuple(pre_ops),
            concat_sources=tuple(concat_sources),
            concat_op=concat_op,
            concat_output=concat_output,
            unary_ops=tuple(unary_ops),
            unary_outputs=tuple(unary_outputs),
            tail_name=tail_name,
            post_ops=tuple(post_ops),
            post_outputs=tuple(post_outputs),
        )
    return None


def _has_concat_unary_conv_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_concat_unary_conv_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_transpose_concat_unary_fanout_conv_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Collapse NCHW concat blocks that are bracketed by transpose adapters.

    Target:
      x_i_nhwc --TRANSPOSE(0,3,1,2)--> x_i_nchw
      CONCAT(axis=1, [x_i_nchw...]) -> y_nchw
      (optional unary chain) -> z_nchw
      z_nchw --TRANSPOSE(0,2,3,1)--> z_i_nhwc -> CONV_2D/DEPTHWISE_CONV_2D

    Rewrite:
      CONCAT(axis=3, [x_i_nhwc...]) -> y_nhwc
      (same unary chain in NHWC semantics) -> z_nhwc
      z_nhwc -> CONV_2D/DEPTHWISE_CONV_2D
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        candidate = _resolve_concat_unary_conv_candidate(
            model_ir,
            graph_index,
        )
        if candidate is None:
            break

        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=list(candidate.concat_sources),
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 3
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(candidate.concat_output),
            _PERM_NCHW_TO_NHWC,
        )
        for unary_output in candidate.unary_outputs:
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(unary_output),
                _PERM_NCHW_TO_NHWC,
            )
        for post_output in candidate.post_outputs:
            _replace_tensor_inputs(
                model_ir,
                post_output,
                candidate.tail_name,
                graph_index=graph_index,
            )

        remove_ops = list(candidate.pre_ops) + list(candidate.post_ops)
        remove_indices = sorted(
            (
                int(index)
                for op in remove_ops
                if (index := graph_index.operator_index(op)) is not None
            ),
            reverse=True,
        )
        for remove_index in remove_indices:
            graph_index.remove_operator(remove_index)

        optimized += 1

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {
        "optimized_transpose_concat_unary_fanout_conv_nhwc_chains": int(
            optimized
        )
    }


def run_concat_unary_conv_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Propagate a validated Concat/unary/Conv fan-out island to NHWC."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        seen_transpose = False
        seen_concat = False
        seen_conv = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            op_type = str(operator.op_type)
            seen_transpose = seen_transpose or op_type == "TRANSPOSE"
            seen_concat = seen_concat or op_type == "CONCATENATION"
            seen_conv = seen_conv or op_type in _CONV_LIKE_OPS
            if seen_transpose and seen_concat and seen_conv:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    stats_key = "optimized_transpose_concat_unary_fanout_conv_nhwc_chains"

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_concat_unary_fanout_conv_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(stats_key, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.concat_unary_conv_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_concat_unary_conv_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={stats_key: 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
