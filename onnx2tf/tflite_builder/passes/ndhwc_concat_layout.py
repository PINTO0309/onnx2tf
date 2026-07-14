from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPassState,
    preflight_required_op_types,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR


_PERM_NDHWC_TO_NCDHW = [0, 4, 1, 2, 3]
_PERM_NCDHW_TO_NDHWC = [0, 2, 3, 4, 1]
_UNARY_OPS = {
    "LOGISTIC",
    "RELU",
    "RELU6",
    "RELU_0_TO_1",
    "ELU",
    "LEAKY_RELU",
    "TANH",
    "GELU",
    "HARD_SWISH",
    "ABS",
    "EXP",
    "NEG",
    "SQRT",
}
_STATS_KEY = "optimized_transpose_pre_concat_ndhwc_chains"


@dataclass(frozen=True)
class _NdhwcConcatInputPlan:
    kind: str
    adapter_op: OperatorIR
    output_name: str
    new_input_name: str
    unary_op: Optional[OperatorIR] = None


@dataclass(frozen=True)
class _NdhwcConcatCandidate:
    input_plans: Tuple[_NdhwcConcatInputPlan, ...]
    concat_op: OperatorIR
    concat_output_name: str
    post_ops: Tuple[OperatorIR, ...]
    post_output_names: Tuple[str, ...]


def _project_ncdhw_shape_to_ndhwc(
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name))
    if tensor is None or len(list(tensor.shape)) != 5:
        return None
    return _permute_shape(
        [int(value) for value in tensor.shape],
        _PERM_NCDHW_TO_NDHWC,
    )


def _clone_ndhwc_quantization(quantization: Any) -> Any:
    cloned = _clone_quantization(quantization)
    if isinstance(cloned, QuantParamIR):
        old_dimension = int(cloned.quantized_dimension)
        if 0 <= old_dimension < len(_PERM_NCDHW_TO_NDHWC):
            cloned.quantized_dimension = int(_PERM_NCDHW_TO_NDHWC.index(old_dimension))
    elif isinstance(cloned, dict) and "quantized_dimension" in cloned:
        old_dimension = int(cloned["quantized_dimension"])
        if 0 <= old_dimension < len(_PERM_NCDHW_TO_NDHWC):
            cloned["quantized_dimension"] = int(
                _PERM_NCDHW_TO_NDHWC.index(old_dimension)
            )
    return cloned


def _resolve_ndhwc_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_NdhwcConcatCandidate]:
    model_outputs = {str(name) for name in model_ir.outputs}
    for concat_index in graph_index.operator_indices("CONCATENATION"):
        concat_op = model_ir.operators[int(concat_index)]
        if (
            len(concat_op.outputs) != 1
        ):
            continue
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 5
        concat_output_name = str(concat_op.outputs[0])
        if concat_axis != 1 or concat_output_name in model_outputs:
            continue

        input_plans: List[_NdhwcConcatInputPlan] = []
        inputs_valid = True
        for input_name in [str(name) for name in concat_op.inputs]:
            producer_op = graph_index.producer(input_name)
            producer_index = (
                None if producer_op is None else graph_index.operator_index(producer_op)
            )
            if producer_op is None or producer_index is None:
                inputs_valid = False
                break

            if (
                str(producer_op.op_type) == "TRANSPOSE"
                and len(producer_op.inputs) >= 2
                and len(producer_op.outputs) == 1
                and str(producer_op.outputs[0]) == input_name
                and _read_transpose_perm(model_ir, producer_op) == _PERM_NDHWC_TO_NCDHW
                and set(graph_index.consumer_indices(input_name)) == {int(concat_index)}
                and input_name not in model_outputs
            ):
                source_name = str(producer_op.inputs[0])
                source_tensor = model_ir.tensors.get(source_name)
                if source_tensor is None or len(list(source_tensor.shape)) != 5:
                    inputs_valid = False
                    break
                input_plans.append(
                    _NdhwcConcatInputPlan(
                        kind="direct",
                        adapter_op=producer_op,
                        output_name=input_name,
                        new_input_name=source_name,
                    )
                )
                continue

            if (
                str(producer_op.op_type) in _UNARY_OPS
                and len(producer_op.inputs) == 1
                and len(producer_op.outputs) == 1
                and str(producer_op.outputs[0]) == input_name
                and set(graph_index.consumer_indices(input_name)) == {int(concat_index)}
                and input_name not in model_outputs
            ):
                unary_input_name = str(producer_op.inputs[0])
                adapter_op = graph_index.producer(unary_input_name)
                adapter_index = (
                    None
                    if adapter_op is None
                    else graph_index.operator_index(adapter_op)
                )
                if (
                    adapter_op is None
                    or adapter_index is None
                    or str(adapter_op.op_type) != "TRANSPOSE"
                    or len(adapter_op.inputs) < 2
                    or len(adapter_op.outputs) != 1
                    or str(adapter_op.outputs[0]) != unary_input_name
                    or _read_transpose_perm(model_ir, adapter_op)
                    != _PERM_NDHWC_TO_NCDHW
                    or unary_input_name in model_outputs
                    or set(graph_index.consumer_indices(unary_input_name))
                    != {int(producer_index)}
                ):
                    inputs_valid = False
                    break
                projected_shape = _project_ncdhw_shape_to_ndhwc(
                    model_ir,
                    input_name,
                )
                if projected_shape is None:
                    inputs_valid = False
                    break
                input_plans.append(
                    _NdhwcConcatInputPlan(
                        kind="unary",
                        adapter_op=adapter_op,
                        unary_op=producer_op,
                        output_name=input_name,
                        new_input_name=str(adapter_op.inputs[0]),
                    )
                )
                continue

            inputs_valid = False
            break
        if not inputs_valid or not input_plans:
            continue

        reference_shape: Optional[List[int]] = None
        shapes_compatible = True
        for input_plan in input_plans:
            if input_plan.kind == "direct":
                source_tensor = model_ir.tensors.get(input_plan.new_input_name)
                if source_tensor is None or len(list(source_tensor.shape)) != 5:
                    shapes_compatible = False
                    break
                projected_shape = [int(value) for value in source_tensor.shape]
            else:
                projected_shape = _project_ncdhw_shape_to_ndhwc(
                    model_ir,
                    input_plan.output_name,
                )
                if projected_shape is None:
                    shapes_compatible = False
                    break
            if reference_shape is None:
                reference_shape = list(projected_shape)
                continue
            if any(
                int(projected_shape[index]) != int(reference_shape[index])
                for index in (0, 1, 2, 3)
            ):
                shapes_compatible = False
                break
        if not shapes_compatible:
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
                or _read_transpose_perm(model_ir, post_op) != _PERM_NCDHW_TO_NDHWC
                or str(post_op.outputs[0]) in model_outputs
            ):
                posts_valid = False
                break
            post_ops.append(post_op)
            post_output_names.append(str(post_op.outputs[0]))
        if not posts_valid or not post_ops:
            continue

        return _NdhwcConcatCandidate(
            input_plans=tuple(input_plans),
            concat_op=concat_op,
            concat_output_name=concat_output_name,
            post_ops=tuple(post_ops),
            post_output_names=tuple(post_output_names),
        )
    return None


def _has_ndhwc_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_ndhwc_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_transpose_pre_concat_ndhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Convert safely adapter-wrapped NCDHW Concat islands to NDHWC."""

    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        candidate = _resolve_ndhwc_concat_candidate(model_ir, graph_index)
        if candidate is None:
            break

        new_concat_inputs: List[str] = []
        for input_plan in candidate.input_plans:
            if input_plan.unary_op is not None:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=input_plan.unary_op,
                    new_inputs=[input_plan.new_input_name],
                    graph_index=graph_index,
                )
                unary_output_tensor = model_ir.tensors.get(input_plan.output_name)
                _permute_tensor_metadata_if_rank_matches(
                    unary_output_tensor,
                    _PERM_NCDHW_TO_NDHWC,
                )
                if unary_output_tensor is not None:
                    unary_output_tensor.quantization = _clone_ndhwc_quantization(
                        unary_output_tensor.quantization
                    )
            new_concat_inputs.append(
                input_plan.output_name
                if input_plan.unary_op
                else input_plan.new_input_name
            )

        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=new_concat_inputs,
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 4

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
            canonical_output_tensor.quantization = _clone_ndhwc_quantization(
                old_concat_tensor.quantization
            )
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
                _PERM_NCDHW_TO_NDHWC,
            )

        remove_ops = [
            *[input_plan.adapter_op for input_plan in candidate.input_plans],
            *candidate.post_ops,
        ]
        remove_indices = sorted(
            {
                int(operator_index)
                for operator in remove_ops
                if (operator_index := graph_index.operator_index(operator)) is not None
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


def run_ndhwc_concat_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run the transactional rank-five NDHWC pre-Concat layout pass."""

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_pre_concat_ndhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_STATS_KEY, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.ndhwc_pre_concat",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_ndhwc_concat_candidate,
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
