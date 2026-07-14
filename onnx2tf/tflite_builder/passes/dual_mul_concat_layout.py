from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    ModelIRPreflightResult,
    ModelIRPassStateScope,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _clone_quantization,
    _is_fully_known_positive_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]


@dataclass(frozen=True)
class _MulConstantPlan:
    mul_op: OperatorIR
    data_input_index: int
    side_input_index: int
    side_input_name: str
    nhwc_data: Optional[np.ndarray]
    clone_required: bool


@dataclass(frozen=True)
class _DualMulConcatCandidate:
    pre_op: OperatorIR
    mul_plans: Tuple[_MulConstantPlan, ...]
    concat_op: OperatorIR
    post_op: OperatorIR
    post_output: str


def _plan_constant_data(
    side_tensor: TensorIR,
    target_shape_nhwc: Optional[List[int]],
) -> Optional[np.ndarray | bool]:
    if side_tensor.data is None:
        return None
    side_data = np.asarray(side_tensor.data)
    if int(side_data.size) == 1:
        return False
    if int(side_data.ndim) != 4:
        return None
    target_shape = (
        [int(value) for value in target_shape_nhwc]
        if _is_fully_known_positive_shape(target_shape_nhwc)
        else None
    )
    if target_shape is not None:
        side_shape = [int(value) for value in side_data.shape]
        if _broadcast_static_shapes(target_shape, side_shape) is not None:
            return np.asarray(side_data)
        rotated = np.transpose(side_data, _PERM_NCHW_TO_NHWC).astype(
            side_data.dtype,
            copy=False,
        )
        if _broadcast_static_shapes(
            target_shape,
            [int(value) for value in rotated.shape],
        ) is None:
            return None
        return np.asarray(rotated)
    side_shape = [int(value) for value in side_data.shape]
    if (
        int(side_shape[0]) == 1
        and int(side_shape[1]) > 0
        and int(side_shape[2]) == 1
        and int(side_shape[3]) == 1
    ):
        return np.transpose(side_data, _PERM_NCHW_TO_NHWC).astype(
            side_data.dtype,
            copy=False,
        )
    return np.asarray(side_data)


def _resolve_dual_mul_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_DualMulConcatCandidate]:
    model_outputs = {str(name) for name in model_ir.outputs}
    for post_op in model_ir.operators:
        post_index = graph_index.operator_index(post_op)
        if (
            post_index is None
            or str(post_op.op_type) != "TRANSPOSE"
            or len(post_op.inputs) < 2
            or len(post_op.outputs) != 1
            or _read_transpose_perm(model_ir, post_op) != _PERM_NCHW_TO_NHWC
        ):
            continue
        concat_output = str(post_op.inputs[0])
        post_output = str(post_op.outputs[0])
        if concat_output in model_outputs or post_output in model_outputs:
            continue
        concat_op = graph_index.producer(concat_output)
        concat_index = (
            None if concat_op is None else graph_index.operator_index(concat_op)
        )
        if (
            concat_op is None
            or concat_index is None
            or str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.inputs) != 2
            or len(concat_op.outputs) != 1
            or set(graph_index.consumer_indices(concat_output))
            != {int(post_index)}
        ):
            continue
        axis = int(concat_op.options.get("axis", 1))
        if axis < 0:
            axis += 4
        if axis != 1:
            continue

        raw_plans: List[Tuple[OperatorIR, int, int, str]] = []
        shared_data_name: str | None = None
        rewritable = True
        for concat_input in [str(value) for value in concat_op.inputs]:
            mul_op = graph_index.producer(concat_input)
            mul_index = (
                None if mul_op is None else graph_index.operator_index(mul_op)
            )
            if (
                mul_op is None
                or mul_index is None
                or str(mul_op.op_type) != "MUL"
                or len(mul_op.inputs) != 2
                or len(mul_op.outputs) != 1
                or str(mul_op.outputs[0]) != concat_input
                or set(graph_index.consumer_indices(concat_input))
                != {int(concat_index)}
            ):
                rewritable = False
                break
            inputs = [str(value) for value in mul_op.inputs]
            data_index: int | None = None
            side_index: int | None = None
            if shared_data_name is None:
                for candidate_data_index in [0, 1]:
                    candidate_name = inputs[candidate_data_index]
                    producer = graph_index.producer(candidate_name)
                    if (
                        producer is not None
                        and str(producer.op_type) == "TRANSPOSE"
                        and len(producer.inputs) >= 2
                        and len(producer.outputs) == 1
                        and str(producer.outputs[0]) == candidate_name
                        and _read_transpose_perm(model_ir, producer)
                        == _PERM_NHWC_TO_NCHW
                    ):
                        shared_data_name = candidate_name
                        data_index = int(candidate_data_index)
                        side_index = 1 - int(candidate_data_index)
                        break
            elif inputs[0] == shared_data_name:
                data_index, side_index = 0, 1
            elif inputs[1] == shared_data_name:
                data_index, side_index = 1, 0
            if data_index is None or side_index is None:
                rewritable = False
                break
            side_name = inputs[int(side_index)]
            side_tensor = model_ir.tensors.get(side_name, None)
            if side_tensor is None or side_tensor.data is None:
                rewritable = False
                break
            raw_plans.append((mul_op, data_index, side_index, side_name))
        if not rewritable or shared_data_name is None:
            continue

        pre_op = graph_index.producer(shared_data_name)
        if (
            pre_op is None
            or str(pre_op.op_type) != "TRANSPOSE"
            or len(pre_op.inputs) < 2
            or len(pre_op.outputs) != 1
            or str(pre_op.outputs[0]) != shared_data_name
            or _read_transpose_perm(model_ir, pre_op) != _PERM_NHWC_TO_NCHW
            or shared_data_name in model_outputs
            or str(pre_op.inputs[0]) in model_outputs
        ):
            continue
        mul_indices = {
            int(index)
            for mul_op, _, _, _ in raw_plans
            if (index := graph_index.operator_index(mul_op)) is not None
        }
        if set(graph_index.consumer_indices(shared_data_name)) != mul_indices:
            continue
        pre_index = graph_index.operator_index(pre_op)
        if pre_index is None:
            continue
        chain_indices = {
            int(pre_index),
            int(concat_index),
            int(post_index),
            *mul_indices,
        }
        source_tensor = model_ir.tensors.get(str(pre_op.inputs[0]), None)
        target_shape = (
            None
            if source_tensor is None
            else [int(value) for value in source_tensor.shape]
        )
        plans: List[_MulConstantPlan] = []
        for mul_op, data_index, side_index, side_name in raw_plans:
            side_tensor = model_ir.tensors[side_name]
            planned_data = _plan_constant_data(side_tensor, target_shape)
            if planned_data is None:
                rewritable = False
                break
            side_users = graph_index.consumer_indices(side_name)
            plans.append(
                _MulConstantPlan(
                    mul_op=mul_op,
                    data_input_index=int(data_index),
                    side_input_index=int(side_index),
                    side_input_name=side_name,
                    nhwc_data=(
                        None
                        if planned_data is False
                        else np.asarray(planned_data)
                    ),
                    clone_required=any(
                        int(user) not in chain_indices for user in side_users
                    ),
                )
            )
        if not rewritable:
            continue
        return _DualMulConcatCandidate(
            pre_op=pre_op,
            mul_plans=tuple(plans),
            concat_op=concat_op,
            post_op=post_op,
            post_output=post_output,
        )
    return None


def _has_dual_mul_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_dual_mul_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_transpose_dual_mul_concat_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around dual-MUL concat blocks.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      MUL(x_nchw, c0_nchw) -> m0_nchw
      MUL(x_nchw, c1_nchw) -> m1_nchw
      CONCAT(axis=1, [m0_nchw, m1_nchw]) -> c_nchw
      c_nchw --TRANSPOSE(0,2,3,1)--> y_nhwc

    Rewrite:
      MUL(x_nhwc, c0_nhwc) -> m0_nhwc
      MUL(x_nhwc, c1_nhwc) -> m1_nhwc
      CONCAT(axis=3, [m0_nhwc, m1_nhwc]) -> y_nhwc
    """
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    while True:
        candidate = _resolve_dual_mul_concat_candidate(model_ir, graph_index)
        if candidate is None:
            break
        source_name = str(candidate.pre_op.inputs[0])
        for plan in candidate.mul_plans:
            if plan.nhwc_data is not None:
                side_tensor = model_ir.tensors[plan.side_input_name]
                rewritten_name = plan.side_input_name
                if plan.clone_required:
                    rewritten_name = _unique_tensor_name(
                        f"{plan.side_input_name}_nhwc"
                    )
                    model_ir.tensors[rewritten_name] = TensorIR(
                        name=rewritten_name,
                        dtype=str(side_tensor.dtype),
                        shape=[int(value) for value in plan.nhwc_data.shape],
                        shape_signature=[
                            int(value) for value in plan.nhwc_data.shape
                        ],
                        data=np.asarray(plan.nhwc_data),
                        is_variable=False,
                        quantization=_clone_quantization(
                            side_tensor.quantization
                        ),
                    )
                else:
                    side_tensor.data = np.asarray(plan.nhwc_data)
                    side_tensor.shape = [
                        int(value) for value in plan.nhwc_data.shape
                    ]
                    side_tensor.shape_signature = [
                        int(value) for value in plan.nhwc_data.shape
                    ]
                _replace_operator_input_at(
                    model_ir=model_ir,
                    op=plan.mul_op,
                    input_index=plan.side_input_index,
                    new_input_name=rewritten_name,
                    graph_index=graph_index,
                )

            new_inputs = [str(value) for value in plan.mul_op.inputs]
            new_inputs[int(plan.data_input_index)] = source_name
            _set_operator_inputs(
                model_ir=model_ir,
                op=plan.mul_op,
                new_inputs=new_inputs,
                graph_index=graph_index,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(str(plan.mul_op.outputs[0]), None),
                _PERM_NCHW_TO_NHWC,
            )

        concat_op = candidate.concat_op
        concat_output = str(concat_op.outputs[0])
        concat_op.options["axis"] = 3
        _set_operator_outputs(
            model_ir=model_ir,
            op=concat_op,
            new_outputs=[candidate.post_output],
            graph_index=graph_index,
        )
        old_concat_tensor = model_ir.tensors.get(concat_output, None)
        post_tensor = model_ir.tensors.get(candidate.post_output, None)
        if old_concat_tensor is not None:
            _permute_tensor_metadata_if_rank_matches(
                old_concat_tensor,
                _PERM_NCHW_TO_NHWC,
            )
        if old_concat_tensor is not None and post_tensor is not None:
            post_tensor.dtype = str(old_concat_tensor.dtype)
            post_tensor.quantization = _clone_quantization(
                old_concat_tensor.quantization
            )
            post_tensor.shape = [int(value) for value in old_concat_tensor.shape]
            post_tensor.shape_signature = (
                [int(value) for value in old_concat_tensor.shape_signature]
                if old_concat_tensor.shape_signature is not None
                else [int(value) for value in old_concat_tensor.shape]
            )

        remove_indices = sorted(
            (
                int(index)
                for op in [candidate.pre_op, candidate.post_op]
                if (index := graph_index.operator_index(op)) is not None
            ),
            reverse=True,
        )
        for remove_index in remove_indices:
            graph_index.remove_operator(remove_index)
        rewritten += 1

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"optimized_transpose_dual_mul_concat_prepost_nhwc_chains": int(rewritten)}


def run_dual_mul_concat_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
    state_scope: ModelIRPassStateScope | None = None,
) -> Dict[str, int]:
    """Propagate a validated dual-Mul/Concat island to NHWC."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "MUL", "CONCATENATION"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    stats_key = "optimized_transpose_dual_mul_concat_prepost_nhwc_chains"

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_dual_mul_concat_prepost_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(stats_key, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.dual_mul_concat_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_dual_mul_concat_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={stats_key: 0},
        diagnostics=diagnostics,
        state_scope=state_scope,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
