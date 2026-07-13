from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    preflight_required_op_types,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
_STATS_KEY = "optimized_transpose_pre_concat_nhwc_direct_chains"


@dataclass(frozen=True)
class _NhwcDirectInputPlan:
    adapter_op: OperatorIR
    source_name: str
    adapter_output_name: str
    remove_adapter: bool


@dataclass(frozen=True)
class _NhwcDirectConcatCandidate:
    input_plans: Tuple[_NhwcDirectInputPlan, ...]
    concat_op: OperatorIR
    concat_output_name: str
    post_ops: Tuple[OperatorIR, ...]
    post_output_names: Tuple[str, ...]


def _clone_nhwc_quantization(quantization: Any) -> Any:
    cloned = _clone_quantization(quantization)
    if isinstance(cloned, QuantParamIR):
        old_dimension = int(cloned.quantized_dimension)
        if 0 <= old_dimension < len(_PERM_NCHW_TO_NHWC):
            cloned.quantized_dimension = int(
                _PERM_NCHW_TO_NHWC.index(old_dimension)
            )
    elif isinstance(cloned, dict) and "quantized_dimension" in cloned:
        old_dimension = int(cloned["quantized_dimension"])
        if 0 <= old_dimension < len(_PERM_NCHW_TO_NHWC):
            cloned["quantized_dimension"] = int(
                _PERM_NCHW_TO_NHWC.index(old_dimension)
            )
    return cloned


def _resolve_nhwc_direct_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> _NhwcDirectConcatCandidate | None:
    """Find one strict, float-path, direct-adapter Concat island."""

    model_outputs = {str(name) for name in model_ir.outputs}
    for concat_op in model_ir.operators:
        concat_index = graph_index.operator_index(concat_op)
        if (
            concat_index is None
            or str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.inputs) == 0
            or len(concat_op.outputs) != 1
        ):
            continue

        concat_output_name = str(concat_op.outputs[0])
        concat_output_tensor = model_ir.tensors.get(concat_output_name)
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 4
        if (
            concat_axis != 1
            or concat_output_name in model_outputs
            or concat_output_tensor is None
            or len(list(concat_output_tensor.shape)) != 4
        ):
            continue

        input_plans: List[_NhwcDirectInputPlan] = []
        inputs_valid = True
        for adapter_output_name in [str(name) for name in concat_op.inputs]:
            adapter_op = graph_index.producer(adapter_output_name)
            if (
                adapter_op is None
                or str(adapter_op.op_type) != "TRANSPOSE"
                or len(adapter_op.inputs) < 2
                or len(adapter_op.outputs) != 1
                or str(adapter_op.outputs[0]) != adapter_output_name
                or _read_transpose_perm(model_ir, adapter_op)
                != _PERM_NHWC_TO_NCHW
            ):
                inputs_valid = False
                break
            source_name = str(adapter_op.inputs[0])
            source_tensor = model_ir.tensors.get(source_name)
            if source_tensor is None or len(list(source_tensor.shape)) != 4:
                inputs_valid = False
                break
            adapter_consumers = set(
                graph_index.consumer_indices(adapter_output_name)
            )
            if int(concat_index) not in adapter_consumers:
                inputs_valid = False
                break
            input_plans.append(
                _NhwcDirectInputPlan(
                    adapter_op=adapter_op,
                    source_name=source_name,
                    adapter_output_name=adapter_output_name,
                    remove_adapter=(
                        adapter_consumers == {int(concat_index)}
                        and adapter_output_name not in model_outputs
                    ),
                )
            )
        if not inputs_valid or not input_plans:
            continue

        concat_user_indices = graph_index.consumer_indices(concat_output_name)
        if not concat_user_indices:
            continue
        post_ops: List[OperatorIR] = []
        post_output_names: List[str] = []
        posts_valid = True
        for user_index in concat_user_indices:
            post_op = model_ir.operators[int(user_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != concat_output_name
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
                or str(post_op.outputs[0]) in model_outputs
            ):
                posts_valid = False
                break
            post_ops.append(post_op)
            post_output_names.append(str(post_op.outputs[0]))
        if not posts_valid or not post_ops:
            continue

        return _NhwcDirectConcatCandidate(
            input_plans=tuple(input_plans),
            concat_op=concat_op,
            concat_output_name=concat_output_name,
            post_ops=tuple(post_ops),
            post_output_names=tuple(post_output_names),
        )
    return None


def _has_nhwc_direct_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_nhwc_direct_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_transpose_pre_concat_nhwc_direct_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Lift strict direct-adapter NCHW Concat islands into NHWC."""

    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        candidate = _resolve_nhwc_direct_concat_candidate(model_ir, graph_index)
        if candidate is None:
            break

        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=[plan.source_name for plan in candidate.input_plans],
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 3

        canonical_output_name = candidate.post_output_names[0]
        _set_operator_outputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_outputs=[canonical_output_name],
            graph_index=graph_index,
        )
        for alias_output_name in candidate.post_output_names[1:]:
            _replace_tensor_inputs(
                model_ir,
                alias_output_name,
                canonical_output_name,
                graph_index=graph_index,
            )

        old_concat_tensor = model_ir.tensors.get(candidate.concat_output_name)
        canonical_output_tensor = model_ir.tensors.get(canonical_output_name)
        if old_concat_tensor is not None and canonical_output_tensor is not None:
            canonical_output_tensor.dtype = str(old_concat_tensor.dtype)
            canonical_output_tensor.shape = [
                int(value) for value in old_concat_tensor.shape
            ]
            canonical_output_tensor.shape_signature = (
                [int(value) for value in old_concat_tensor.shape_signature]
                if old_concat_tensor.shape_signature is not None
                else [int(value) for value in old_concat_tensor.shape]
            )
            _permute_tensor_metadata_if_rank_matches(
                canonical_output_tensor,
                _PERM_NCHW_TO_NHWC,
            )
            canonical_output_tensor.quantization = _clone_nhwc_quantization(
                old_concat_tensor.quantization
            )

        remove_ops = [
            *[
                plan.adapter_op
                for plan in candidate.input_plans
                if plan.remove_adapter
            ],
            *candidate.post_ops,
        ]
        remove_indices = sorted(
            {
                int(operator_index)
                for operator in remove_ops
                if (
                    operator_index := graph_index.operator_index(operator)
                )
                is not None
            },
            reverse=True,
        )
        for remove_index in remove_indices:
            graph_index.remove_operator(remove_index)
        optimized += 1

    if optimized > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(optimized)}


def run_nhwc_concat_direct_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run the transactional direct-adapter rank-four Concat pass."""

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_nhwc_direct_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_STATS_KEY, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.nhwc_pre_concat_direct",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_nhwc_direct_concat_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={_STATS_KEY: 0},
        diagnostics=diagnostics,
        preflight=lambda candidate_model: preflight_required_op_types(
            candidate_model,
            {"TRANSPOSE", "CONCATENATION"},
        ),
    )
    return {str(key): int(value) for key, value in details.items()}
