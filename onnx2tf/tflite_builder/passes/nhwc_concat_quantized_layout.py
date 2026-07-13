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
_STATS_KEY = "optimized_transpose_pre_concat_nhwc_quantized_direct_chains"


@dataclass(frozen=True)
class _DirectInputPlan:
    adapter_op: OperatorIR
    concat_input: str
    source_name: str
    remove_adapter: bool


@dataclass(frozen=True)
class _QuantizedDirectConcatCandidate:
    input_plans: Tuple[_DirectInputPlan, ...]
    concat_op: OperatorIR
    concat_output: str
    quantize_op: OperatorIR
    quantized_output: str
    post_ops: Tuple[OperatorIR, ...]
    post_outputs: Tuple[str, ...]


def _clone_permuted_quantization(
    quantization: Any,
    permutation: List[int],
) -> Any:
    cloned = _clone_quantization(quantization)
    if isinstance(cloned, QuantParamIR):
        old_dimension = int(cloned.quantized_dimension)
        if 0 <= old_dimension < len(permutation):
            cloned.quantized_dimension = int(permutation.index(old_dimension))
    elif isinstance(cloned, dict) and "quantized_dimension" in cloned:
        old_dimension = int(cloned["quantized_dimension"])
        if 0 <= old_dimension < len(permutation):
            cloned["quantized_dimension"] = int(
                permutation.index(old_dimension)
            )
    return cloned


def _resolve_quantized_direct_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_QuantizedDirectConcatCandidate]:
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
        concat_output = str(concat_op.outputs[0])
        concat_tensor = model_ir.tensors.get(concat_output)
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 4
        if (
            concat_axis != 1
            or concat_output in model_outputs
            or concat_tensor is None
            or len(list(concat_tensor.shape)) != 4
        ):
            continue

        concat_users = graph_index.consumer_indices(concat_output)
        if len(concat_users) != 1:
            continue
        quantize_op = model_ir.operators[int(concat_users[0])]
        quantize_index = graph_index.operator_index(quantize_op)
        if (
            quantize_index is None
            or str(quantize_op.op_type) != "QUANTIZE"
            or len(quantize_op.inputs) != 1
            or len(quantize_op.outputs) != 1
            or str(quantize_op.inputs[0]) != concat_output
        ):
            continue
        quantized_output = str(quantize_op.outputs[0])
        quantized_tensor = model_ir.tensors.get(quantized_output)
        if (
            quantized_output in model_outputs
            or quantized_tensor is None
            or len(list(quantized_tensor.shape)) != 4
        ):
            continue

        post_ops: List[OperatorIR] = []
        post_outputs: List[str] = []
        quantized_users = graph_index.consumer_indices(quantized_output)
        if not quantized_users:
            continue
        posts_valid = True
        for post_index in quantized_users:
            post_op = model_ir.operators[int(post_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != quantized_output
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
                or str(post_op.outputs[0]) in model_outputs
            ):
                posts_valid = False
                break
            post_output = str(post_op.outputs[0])
            post_tensor = model_ir.tensors.get(post_output)
            if post_tensor is None or len(list(post_tensor.shape)) != 4:
                posts_valid = False
                break
            post_ops.append(post_op)
            post_outputs.append(post_output)
        if not posts_valid or not post_ops:
            continue

        input_plans: List[_DirectInputPlan] = []
        inputs_valid = True
        for concat_input in [str(name) for name in concat_op.inputs]:
            adapter_op = graph_index.producer(concat_input)
            if (
                adapter_op is None
                or str(adapter_op.op_type) != "TRANSPOSE"
                or len(adapter_op.inputs) < 2
                or len(adapter_op.outputs) != 1
                or str(adapter_op.outputs[0]) != concat_input
                or _read_transpose_perm(model_ir, adapter_op)
                != _PERM_NHWC_TO_NCHW
            ):
                inputs_valid = False
                break
            source_name = str(adapter_op.inputs[0])
            source_tensor = model_ir.tensors.get(source_name)
            input_consumers = set(
                graph_index.consumer_indices(concat_input)
            )
            if (
                source_tensor is None
                or len(list(source_tensor.shape)) != 4
                or int(concat_index) not in input_consumers
            ):
                inputs_valid = False
                break
            input_plans.append(
                _DirectInputPlan(
                    adapter_op=adapter_op,
                    concat_input=concat_input,
                    source_name=source_name,
                    remove_adapter=(
                        input_consumers == {int(concat_index)}
                        and concat_input not in model_outputs
                    ),
                )
            )
        if not inputs_valid or not input_plans:
            continue
        return _QuantizedDirectConcatCandidate(
            input_plans=tuple(input_plans),
            concat_op=concat_op,
            concat_output=concat_output,
            quantize_op=quantize_op,
            quantized_output=quantized_output,
            post_ops=tuple(post_ops),
            post_outputs=tuple(post_outputs),
        )
    return None


def _has_quantized_direct_concat_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_quantized_direct_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_quantized_direct_concat_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    optimized = 0
    while True:
        candidate = _resolve_quantized_direct_concat_candidate(
            model_ir,
            graph_index,
        )
        if candidate is None:
            break

        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=[plan.source_name for plan in candidate.input_plans],
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 3
        concat_tensor = model_ir.tensors.get(candidate.concat_output)
        _permute_tensor_metadata_if_rank_matches(
            concat_tensor,
            _PERM_NCHW_TO_NHWC,
        )
        if concat_tensor is not None:
            concat_tensor.quantization = _clone_permuted_quantization(
                concat_tensor.quantization,
                _PERM_NCHW_TO_NHWC,
            )

        canonical_output = candidate.post_outputs[0]
        _set_operator_outputs(
            model_ir=model_ir,
            op=candidate.quantize_op,
            new_outputs=[canonical_output],
            graph_index=graph_index,
        )
        for alias_output in candidate.post_outputs[1:]:
            _replace_tensor_inputs(
                model_ir=model_ir,
                src_name=alias_output,
                dst_name=canonical_output,
                graph_index=graph_index,
            )

        quantized_tensor = model_ir.tensors.get(candidate.quantized_output)
        canonical_tensor = model_ir.tensors.get(canonical_output)
        if canonical_tensor is not None and quantized_tensor is not None:
            canonical_tensor.dtype = str(quantized_tensor.dtype)
            canonical_tensor.quantization = _clone_permuted_quantization(
                quantized_tensor.quantization,
                _PERM_NCHW_TO_NHWC,
            )
        if canonical_tensor is not None and concat_tensor is not None:
            canonical_tensor.shape = [int(value) for value in concat_tensor.shape]
            canonical_tensor.shape_signature = (
                [int(value) for value in concat_tensor.shape_signature]
                if concat_tensor.shape_signature is not None
                else [int(value) for value in concat_tensor.shape]
            )

        remove_ops = [
            *candidate.post_ops,
            *[
                plan.adapter_op
                for plan in candidate.input_plans
                if plan.remove_adapter
            ],
        ]
        remove_indices = sorted(
            {
                int(index)
                for op in remove_ops
                if (index := graph_index.operator_index(op)) is not None
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


def run_nhwc_concat_quantized_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Lift the strict all-direct Concat→Quantize post-adapter family."""

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_quantized_direct_concat_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_STATS_KEY, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.nhwc_pre_concat_quantized_direct",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_quantized_direct_concat_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={_STATS_KEY: 0},
        diagnostics=diagnostics,
        preflight=lambda candidate_model: preflight_required_op_types(
            candidate_model,
            {"TRANSPOSE", "CONCATENATION", "QUANTIZE"},
        ),
    )
    return {str(key): int(value) for key, value in details.items()}
