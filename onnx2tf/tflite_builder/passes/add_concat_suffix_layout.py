from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]


@dataclass(frozen=True)
class _AddConcatSuffixCandidate:
    concat_op: OperatorIR
    add_rewrites: Tuple[Tuple[OperatorIR, Tuple[str, ...]], ...]
    branch_pre_ops: Tuple[OperatorIR, ...]
    base_pre_op: OperatorIR
    remove_base_pre: bool
    mul_op: OperatorIR
    mul_side_index: int
    add2_op: OperatorIR
    add2_side_index: int
    post_op: OperatorIR
    post_output: str


def _resolve_add_concat_suffix_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_AddConcatSuffixCandidate]:
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
        if concat_output in model_outputs:
            continue
        concat_axis = int(concat_op.options.get("axis", 1))
        if concat_axis < 0:
            concat_axis += 4
        concat_inputs = [str(value) for value in concat_op.inputs]
        if concat_axis != 1 or len(concat_inputs) < 2:
            continue

        add_rewrites: List[Tuple[OperatorIR, Tuple[str, ...]]] = []
        branch_pre_ops: List[OperatorIR] = []
        base_pre_op: OperatorIR | None = None
        rewritable = True
        for add_output in concat_inputs:
            add_op = graph_index.producer(add_output)
            add_index = (
                None if add_op is None else graph_index.operator_index(add_op)
            )
            if (
                add_op is None
                or add_index is None
                or str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
                or str(add_op.outputs[0]) != add_output
                or set(graph_index.consumer_indices(add_output))
                != {int(concat_index)}
            ):
                rewritable = False
                break
            add_inputs = [str(value) for value in add_op.inputs]
            transpose_inputs: List[Tuple[int, OperatorIR, str, bool]] = []
            for input_index, input_name in enumerate(add_inputs):
                pre_op = graph_index.producer(input_name)
                if (
                    pre_op is None
                    or str(pre_op.op_type) != "TRANSPOSE"
                    or len(pre_op.inputs) < 2
                    or len(pre_op.outputs) != 1
                    or str(pre_op.outputs[0]) != input_name
                    or _read_transpose_perm(model_ir, pre_op)
                    != _PERM_NHWC_TO_NCHW
                    or input_name in model_outputs
                ):
                    continue
                direct_only = set(graph_index.consumer_indices(input_name)) == {
                    int(add_index)
                }
                transpose_inputs.append(
                    (
                        int(input_index),
                        pre_op,
                        str(pre_op.inputs[0]),
                        bool(direct_only),
                    )
                )
            branch_plan = next(
                (plan for plan in transpose_inputs if bool(plan[3])),
                None,
            )
            base_plan = (
                None
                if branch_plan is None
                else next(
                    (
                        plan
                        for plan in transpose_inputs
                        if plan[1] is not branch_plan[1]
                    ),
                    None,
                )
            )
            if branch_plan is None or base_plan is None:
                rewritable = False
                break
            if base_pre_op is None:
                base_pre_op = base_plan[1]
            elif base_pre_op is not base_plan[1]:
                rewritable = False
                break
            new_inputs = list(add_inputs)
            new_inputs[int(branch_plan[0])] = str(branch_plan[2])
            new_inputs[int(base_plan[0])] = str(base_plan[2])
            add_rewrites.append((add_op, tuple(new_inputs)))
            branch_pre_ops.append(branch_plan[1])
        if not rewritable or base_pre_op is None:
            continue

        concat_users = graph_index.consumer_indices(concat_output)
        if len(concat_users) != 1:
            continue
        mul_op = model_ir.operators[int(concat_users[0])]
        if (
            str(mul_op.op_type) != "MUL"
            or len(mul_op.inputs) != 2
            or len(mul_op.outputs) != 1
        ):
            continue
        if str(mul_op.inputs[0]) == concat_output:
            mul_side_index = 1
        elif str(mul_op.inputs[1]) == concat_output:
            mul_side_index = 0
        else:
            continue
        mul_side_tensor = model_ir.tensors.get(
            str(mul_op.inputs[int(mul_side_index)]),
            None,
        )
        mul_output = str(mul_op.outputs[0])
        if (
            mul_side_tensor is None
            or mul_side_tensor.data is None
            or mul_output in model_outputs
        ):
            continue

        mul_users = graph_index.consumer_indices(mul_output)
        if len(mul_users) != 1:
            continue
        add2_op = model_ir.operators[int(mul_users[0])]
        if (
            str(add2_op.op_type) != "ADD"
            or len(add2_op.inputs) != 2
            or len(add2_op.outputs) != 1
        ):
            continue
        if str(add2_op.inputs[0]) == mul_output:
            add2_side_index = 1
        elif str(add2_op.inputs[1]) == mul_output:
            add2_side_index = 0
        else:
            continue
        add2_side_tensor = model_ir.tensors.get(
            str(add2_op.inputs[int(add2_side_index)]),
            None,
        )
        add2_output = str(add2_op.outputs[0])
        if (
            add2_side_tensor is None
            or add2_side_tensor.data is None
            or add2_output in model_outputs
        ):
            continue

        add2_users = graph_index.consumer_indices(add2_output)
        if len(add2_users) != 1:
            continue
        post_op = model_ir.operators[int(add2_users[0])]
        if (
            str(post_op.op_type) != "TRANSPOSE"
            or len(post_op.inputs) < 2
            or len(post_op.outputs) != 1
            or str(post_op.inputs[0]) != add2_output
            or _read_transpose_perm(model_ir, post_op)
            != _PERM_NCHW_TO_NHWC
        ):
            continue
        post_output = str(post_op.outputs[0])
        if post_output in model_outputs:
            continue

        add_ops = {id(op) for op, _ in add_rewrites}
        base_output = str(base_pre_op.outputs[0])
        remove_base_pre = all(
            id(model_ir.operators[index]) in add_ops
            for index in graph_index.consumer_indices(base_output)
        )
        return _AddConcatSuffixCandidate(
            concat_op=concat_op,
            add_rewrites=tuple(add_rewrites),
            branch_pre_ops=tuple(branch_pre_ops),
            base_pre_op=base_pre_op,
            remove_base_pre=bool(remove_base_pre),
            mul_op=mul_op,
            mul_side_index=int(mul_side_index),
            add2_op=add2_op,
            add2_side_index=int(add2_side_index),
            post_op=post_op,
            post_output=post_output,
        )
    return None


def _has_add_concat_suffix_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_add_concat_suffix_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_transpose_add_concat_const_suffix_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Eliminate NCHW round-trips around ADD fan-in concat blocks with const suffixes.

    Target:
      (x_i_nhwc --T(0,3,1,2)--> x_i_nchw)
      (b_nhwc  --T(0,3,1,2)--> b_nchw; shared)
      ADD(x_i_nchw, b_nchw) -> a_i_nchw
      CONCAT(axis=1, [a_i_nchw...]) -> c_nchw
      c_nchw --(MUL|ADD with const)*--> z_nchw
      z_nchw --T(0,2,3,1)--> z_nhwc

    Rewrite:
      ADD(x_i_nhwc, b_nhwc) -> a_i_nhwc
      CONCAT(axis=3, [a_i_nhwc...]) -> c_nhwc
      c_nhwc --(MUL|ADD with const_nhwc)*--> z_nhwc
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base: str) -> str:
        name = str(base)
        suffix = 1
        while name in model_ir.tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        return name

    def _ensure_exclusive_const_input(
        op: OperatorIR,
        input_index: int,
    ) -> Optional[TensorIR]:
        op_index = graph_index.operator_index(op)
        if (
            op_index is None
            or int(input_index) < 0
            or int(input_index) >= len(op.inputs)
        ):
            return None
        tensor_name = str(op.inputs[int(input_index)])
        tensor = model_ir.tensors.get(tensor_name, None)
        if tensor is None or tensor.data is None:
            return None
        if set(graph_index.consumer_indices(tensor_name)) == {int(op_index)}:
            return tensor
        new_name = _unique_tensor_name(f"{tensor_name}_layout")
        shape_signature = (
            [int(value) for value in tensor.shape_signature]
            if tensor.shape_signature is not None
            else [int(value) for value in tensor.shape]
        )
        model_ir.tensors[new_name] = TensorIR(
            name=new_name,
            dtype=str(tensor.dtype),
            shape=[int(value) for value in tensor.shape],
            shape_signature=shape_signature,
            data=np.asarray(tensor.data).copy(),
            is_variable=bool(tensor.is_variable),
            quantization=_clone_quantization(tensor.quantization),
        )
        _replace_operator_input_at(
            model_ir=model_ir,
            op=op,
            input_index=int(input_index),
            new_input_name=new_name,
            graph_index=graph_index,
        )
        return model_ir.tensors[new_name]

    while True:
        candidate = _resolve_add_concat_suffix_candidate(model_ir, graph_index)
        if candidate is None:
            break
        concat_op = candidate.concat_op
        concat_output = str(concat_op.outputs[0])
        mul_op = candidate.mul_op
        mul_output = str(mul_op.outputs[0])
        add2_op = candidate.add2_op
        add2_output = str(add2_op.outputs[0])
        post_output = str(candidate.post_output)

        mul_side_tensor = _ensure_exclusive_const_input(
            mul_op,
            candidate.mul_side_index,
        )
        add2_side_tensor = _ensure_exclusive_const_input(
            add2_op,
            candidate.add2_side_index,
        )
        if mul_side_tensor is None or add2_side_tensor is None:
            break

        for add_op, new_inputs in candidate.add_rewrites:
            _set_operator_inputs(
                model_ir=model_ir,
                op=add_op,
                new_inputs=[str(value) for value in new_inputs],
                graph_index=graph_index,
            )
            if len(add_op.outputs) == 1:
                _permute_tensor_metadata_if_rank_matches(
                    model_ir.tensors.get(str(add_op.outputs[0]), None),
                    _PERM_NCHW_TO_NHWC,
                )

        for suffix_tensor in [mul_side_tensor, add2_side_tensor]:
            suffix_data = np.asarray(suffix_tensor.data)
            if int(suffix_data.ndim) != 4:
                continue
            nhwc_data = np.transpose(
                suffix_data,
                _PERM_NCHW_TO_NHWC,
            ).astype(suffix_data.dtype, copy=False)
            suffix_tensor.data = np.asarray(nhwc_data)
            suffix_tensor.shape = [int(value) for value in nhwc_data.shape]
            suffix_tensor.shape_signature = [
                int(value) for value in nhwc_data.shape
            ]

        concat_op.options["axis"] = 3
        for output_name in [concat_output, mul_output, add2_output]:
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(output_name, None),
                _PERM_NCHW_TO_NHWC,
            )

        _set_operator_outputs(
            model_ir=model_ir,
            op=add2_op,
            new_outputs=[post_output],
            graph_index=graph_index,
        )
        old_add2_tensor = model_ir.tensors.get(add2_output, None)
        post_tensor = model_ir.tensors.get(post_output, None)
        if old_add2_tensor is not None and post_tensor is not None:
            post_tensor.dtype = str(old_add2_tensor.dtype)
            post_tensor.quantization = _clone_quantization(
                old_add2_tensor.quantization
            )
            post_tensor.shape = [int(value) for value in old_add2_tensor.shape]
            post_tensor.shape_signature = (
                [int(value) for value in old_add2_tensor.shape_signature]
                if old_add2_tensor.shape_signature is not None
                else [int(value) for value in old_add2_tensor.shape]
            )

        remove_ops = {id(op): op for op in candidate.branch_pre_ops}
        remove_ops[id(candidate.post_op)] = candidate.post_op
        if candidate.remove_base_pre:
            remove_ops[id(candidate.base_pre_op)] = candidate.base_pre_op
        remove_indices = sorted(
            (
                int(index)
                for op in remove_ops.values()
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
    return {"optimized_transpose_add_concat_const_suffix_nhwc_chains": int(optimized)}


def run_add_concat_suffix_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Propagate a validated Add/Concat constant-suffix island to NHWC."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "ADD", "MUL", "CONCATENATION"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    stats_key = "optimized_transpose_add_concat_const_suffix_nhwc_chains"

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_add_concat_const_suffix_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(stats_key, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.add_concat_const_suffix_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_add_concat_suffix_candidate,
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
