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
    _clone_quantization,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]


@dataclass(frozen=True)
class _DequantizeBranchPlan:
    pre_op: OperatorIR
    dq_op: OperatorIR
    concat_input: str
    new_dq_input: str
    remove_pre_adapter: bool


@dataclass(frozen=True)
class _DequantConcatQuantizeCandidate:
    branches: Tuple[_DequantizeBranchPlan, ...]
    concat_op: OperatorIR
    concat_output: str
    quantize_op: OperatorIR
    quantized_output: str
    post_ops: Tuple[OperatorIR, ...]
    post_outputs: Tuple[str, ...]
    canonical_output: str


def _resolve_dequant_concat_quantize_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_DequantConcatQuantizeCandidate]:
    model_outputs = {str(name) for name in model_ir.outputs}
    for concat_op in model_ir.operators:
        concat_index = graph_index.operator_index(concat_op)
        if (
            concat_index is None
            or str(concat_op.op_type) != "CONCATENATION"
            or len(concat_op.outputs) != 1
        ):
            continue
        concat_output = str(concat_op.outputs[0])
        concat_tensor = model_ir.tensors.get(concat_output)
        if (
            concat_output in model_outputs
            or concat_tensor is None
            or concat_tensor.shape is None
            or len(concat_tensor.shape) != 4
        ):
            continue
        axis = int(concat_op.options.get("axis", 1))
        if axis < 0:
            axis += 4
        if axis != 1:
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
            or quantized_tensor.shape is None
            or len(quantized_tensor.shape) != 4
            or quantized_tensor.quantization is None
        ):
            continue

        post_ops: List[OperatorIR] = []
        post_outputs: List[str] = []
        valid_posts = True
        quantized_users = graph_index.consumer_indices(quantized_output)
        if not quantized_users:
            continue
        for user_index in quantized_users:
            post_op = model_ir.operators[int(user_index)]
            if (
                str(post_op.op_type) != "TRANSPOSE"
                or len(post_op.inputs) < 2
                or len(post_op.outputs) != 1
                or str(post_op.inputs[0]) != quantized_output
                or _read_transpose_perm(model_ir, post_op)
                != _PERM_NCHW_TO_NHWC
            ):
                valid_posts = False
                break
            post_output = str(post_op.outputs[0])
            post_tensor = model_ir.tensors.get(post_output)
            if (
                post_output in model_outputs
                or post_tensor is None
                or post_tensor.shape is None
                or len(post_tensor.shape) != 4
            ):
                valid_posts = False
                break
            post_ops.append(post_op)
            post_outputs.append(post_output)
        if not valid_posts or not post_ops:
            continue

        branches: List[_DequantizeBranchPlan] = []
        valid_branches = True
        for concat_input in [str(name) for name in concat_op.inputs]:
            dq_op = graph_index.producer(concat_input)
            dq_index = (
                None if dq_op is None else graph_index.operator_index(dq_op)
            )
            dq_tensor = model_ir.tensors.get(concat_input)
            if (
                dq_op is None
                or dq_index is None
                or str(dq_op.op_type) != "DEQUANTIZE"
                or len(dq_op.inputs) != 1
                or len(dq_op.outputs) != 1
                or str(dq_op.outputs[0]) != concat_input
                or concat_input in model_outputs
                or dq_tensor is None
                or dq_tensor.shape is None
                or len(dq_tensor.shape) != 4
                or set(graph_index.consumer_indices(concat_input))
                != {int(concat_index)}
            ):
                valid_branches = False
                break
            pre_output = str(dq_op.inputs[0])
            pre_op = graph_index.producer(pre_output)
            pre_index = (
                None if pre_op is None else graph_index.operator_index(pre_op)
            )
            if (
                pre_op is None
                or pre_index is None
                or str(pre_op.op_type) != "TRANSPOSE"
                or len(pre_op.inputs) < 2
                or len(pre_op.outputs) != 1
                or str(pre_op.outputs[0]) != pre_output
                or _read_transpose_perm(model_ir, pre_op)
                != _PERM_NHWC_TO_NCHW
                or pre_output in model_outputs
            ):
                valid_branches = False
                break
            new_dq_input = str(pre_op.inputs[0])
            source_tensor = model_ir.tensors.get(new_dq_input)
            if (
                source_tensor is None
                or source_tensor.shape is None
                or len(source_tensor.shape) != 4
                or source_tensor.quantization is None
            ):
                valid_branches = False
                break
            branches.append(
                _DequantizeBranchPlan(
                    pre_op=pre_op,
                    dq_op=dq_op,
                    concat_input=concat_input,
                    new_dq_input=new_dq_input,
                    remove_pre_adapter=(
                        set(graph_index.consumer_indices(pre_output))
                        == {int(dq_index)}
                    ),
                )
            )
        if not valid_branches or not branches:
            continue

        return _DequantConcatQuantizeCandidate(
            branches=tuple(branches),
            concat_op=concat_op,
            concat_output=concat_output,
            quantize_op=quantize_op,
            quantized_output=quantized_output,
            post_ops=tuple(post_ops),
            post_outputs=tuple(post_outputs),
            canonical_output=str(post_outputs[0]),
        )
    return None


def _has_dequant_concat_quantize_candidate(
    pass_state: ModelIRPassState,
) -> bool:
    return (
        _resolve_dequant_concat_quantize_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Convert strict NCHW concat quantize bridges back to NHWC and remove adapter transposes.

    Target:
      x_i_nhwc --TRANSPOSE(0,3,1,2)--> x_i_nchw --DEQUANTIZE--> f_i_nchw
      CONCAT(axis=1, [f_0_nchw, ...]) -> f_cat_nchw
      f_cat_nchw --QUANTIZE--> q_cat_nchw --TRANSPOSE(0,2,3,1)--> q_cat_nhwc

    Rewrite:
      x_i_nhwc --DEQUANTIZE--> f_i_nhwc
      CONCAT(axis=3, [f_0_nhwc, ...]) -> f_cat_nhwc
      f_cat_nhwc --QUANTIZE--> q_cat_nhwc
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        candidate = _resolve_dequant_concat_quantize_candidate(
            model_ir,
            graph_index,
        )
        if candidate is None:
            break

        for branch in candidate.branches:
            _set_operator_inputs(
                model_ir=model_ir,
                op=branch.dq_op,
                new_inputs=[branch.new_dq_input],
                graph_index=graph_index,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(branch.concat_input),
                _PERM_NCHW_TO_NHWC,
            )

        candidate.concat_op.options["axis"] = 3
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(candidate.concat_output),
            _PERM_NCHW_TO_NHWC,
        )

        _set_operator_outputs(
            model_ir=model_ir,
            op=candidate.quantize_op,
            new_outputs=[candidate.canonical_output],
            graph_index=graph_index,
        )
        for alias_output in candidate.post_outputs[1:]:
            _replace_tensor_inputs(
                model_ir,
                alias_output,
                candidate.canonical_output,
                graph_index=graph_index,
            )

        old_quantized_tensor = model_ir.tensors[candidate.quantized_output]
        concat_tensor = model_ir.tensors[candidate.concat_output]
        canonical_tensor = model_ir.tensors[candidate.canonical_output]
        canonical_tensor.dtype = str(old_quantized_tensor.dtype)
        canonical_tensor.quantization = _clone_quantization(
            old_quantized_tensor.quantization
        )
        canonical_tensor.shape = [int(value) for value in concat_tensor.shape]
        canonical_tensor.shape_signature = (
            [int(value) for value in concat_tensor.shape_signature]
            if concat_tensor.shape_signature is not None
            else [int(value) for value in concat_tensor.shape]
        )

        remove_ops = list(candidate.post_ops) + [
            branch.pre_op
            for branch in candidate.branches
            if branch.remove_pre_adapter
        ]
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

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if optimized > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains": int(
            optimized
        )
    }


def run_dequant_concat_quantize_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Propagate a validated Dequantize/Concat/Quantize island to NHWC."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "DEQUANTIZE", "CONCATENATION", "QUANTIZE"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    stats_key = (
        "optimized_transpose_pre_dequant_concat_quantize_post_nhwc_chains"
    )

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = (
            _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains(
                pass_state.model_ir,
                graph_index=pass_state.graph_index,
                layout_state=pass_state.layout_state,
            )
        )
        return {**stats, "changed": bool(stats.get(stats_key, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.dequant_concat_quantize_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_dequant_concat_quantize_candidate,
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
