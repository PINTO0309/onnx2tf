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
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]
_CONV_LIKE_OPS = {"CONV_2D", "DEPTHWISE_CONV_2D"}
_STATS_KEY = "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"


@dataclass(frozen=True)
class _SppConstantPlan:
    mul_op: OperatorIR
    input_index: int
    tensor_name: str
    nhwc_data: np.ndarray
    clone_required: bool


@dataclass(frozen=True)
class _SppBranchPlan:
    add_op: OperatorIR
    new_inputs: Tuple[str, str]
    output_name: str
    resize_post_op: OperatorIR


@dataclass(frozen=True)
class _SppLayoutCandidate:
    base_pre_op: OperatorIR
    base_nhwc_name: str
    branches: Tuple[_SppBranchPlan, ...]
    concat0_op: OperatorIR
    concat0_out_name: str
    mul0_out_name: str
    mul0_constant: _SppConstantPlan
    post0_op: OperatorIR
    post0_out_name: str
    add0_op: OperatorIR
    conv0_out_name: str
    conv0_post_op: OperatorIR
    concat1_op: OperatorIR
    concat1_out_name: str
    mul1_out_name: str
    mul1_constant: _SppConstantPlan
    post1_op: OperatorIR
    post1_out_name: str
    add1_op: OperatorIR


def _rank_four_tensor(model_ir: ModelIR, tensor_name: str) -> bool:
    tensor = model_ir.tensors.get(str(tensor_name))
    return bool(
        tensor is not None
        and tensor.shape is not None
        and len(tensor.shape) == 4
    )


def _normalized_axis(op: OperatorIR) -> int:
    axis = int(op.options.get("axis", 1))
    return axis + 4 if axis < 0 else axis


def _resolve_constant_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    mul_op: OperatorIR,
    data_name: str,
    allowed_consumer_indices: set[int],
) -> Optional[_SppConstantPlan]:
    inputs = [str(name) for name in mul_op.inputs]
    if len(inputs) != 2 or inputs.count(str(data_name)) != 1:
        return None
    input_index = 1 - inputs.index(str(data_name))
    tensor_name = inputs[input_index]
    tensor = model_ir.tensors.get(tensor_name)
    if tensor is None or tensor.data is None:
        return None
    data = np.asarray(tensor.data)
    if data.ndim != 4:
        return None
    nhwc_data = np.transpose(data, axes=_PERM_NCHW_TO_NHWC).astype(
        data.dtype,
        copy=False,
    )
    public_names = {
        *[str(name) for name in model_ir.inputs],
        *[str(name) for name in model_ir.outputs],
    }
    clone_required = tensor_name in public_names or any(
        int(index) not in allowed_consumer_indices
        for index in graph_index.consumer_indices(tensor_name)
    )
    return _SppConstantPlan(
        mul_op=mul_op,
        input_index=int(input_index),
        tensor_name=tensor_name,
        nhwc_data=np.asarray(nhwc_data),
        clone_required=bool(clone_required),
    )


def _resolve_spp_layout_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_SppLayoutCandidate]:
    model_outputs = {str(name) for name in model_ir.outputs}
    for concat0_op in model_ir.operators:
        concat0_idx = graph_index.operator_index(concat0_op)
        if (
            concat0_idx is None
            or str(concat0_op.op_type) != "CONCATENATION"
            or len(concat0_op.inputs) != 4
            or len(concat0_op.outputs) != 1
            or _normalized_axis(concat0_op) != 1
        ):
            continue
        concat0_out_name = str(concat0_op.outputs[0])
        if (
            concat0_out_name in model_outputs
            or not _rank_four_tensor(model_ir, concat0_out_name)
        ):
            continue

        raw_branches: List[
            Tuple[
                OperatorIR,
                int,
                str,
                Tuple[Tuple[OperatorIR, str, str], ...],
            ]
        ] = []
        valid = True
        for add_out_name in [str(name) for name in concat0_op.inputs]:
            add_op = graph_index.producer(add_out_name)
            add_idx = (
                None if add_op is None else graph_index.operator_index(add_op)
            )
            if (
                add_op is None
                or add_idx is None
                or str(add_op.op_type) != "ADD"
                or len(add_op.inputs) != 2
                or len(add_op.outputs) != 1
                or str(add_op.outputs[0]) != add_out_name
                or add_out_name in model_outputs
                or not _rank_four_tensor(model_ir, add_out_name)
                or set(graph_index.consumer_indices(add_out_name))
                != {int(concat0_idx)}
            ):
                valid = False
                break
            adapters: List[Tuple[OperatorIR, str, str]] = []
            for input_name in [str(name) for name in add_op.inputs]:
                adapter = graph_index.producer(input_name)
                if (
                    adapter is not None
                    and graph_index.operator_index(adapter) is not None
                    and str(adapter.op_type) == "TRANSPOSE"
                    and len(adapter.inputs) >= 2
                    and len(adapter.outputs) == 1
                    and str(adapter.outputs[0]) == input_name
                    and _read_transpose_perm(model_ir, adapter)
                    == _PERM_NHWC_TO_NCHW
                    and input_name not in model_outputs
                    and _rank_four_tensor(model_ir, str(adapter.inputs[0]))
                    and _rank_four_tensor(model_ir, input_name)
                ):
                    adapters.append(
                        (adapter, str(adapter.inputs[0]), input_name)
                    )
            if len(adapters) != 2:
                valid = False
                break
            raw_branches.append(
                (add_op, int(add_idx), add_out_name, tuple(adapters))
            )
        if not valid or len(raw_branches) != 4:
            continue

        common_adapter_outputs = set(
            record[3][0][2] for record in raw_branches
        ) | set(record[3][1][2] for record in raw_branches)
        for _, _, _, adapters in raw_branches:
            common_adapter_outputs &= {entry[2] for entry in adapters}
        if len(common_adapter_outputs) != 1:
            continue
        base_nchw_name = next(iter(common_adapter_outputs))
        base_pre_op = graph_index.producer(base_nchw_name)
        base_pre_idx = (
            None
            if base_pre_op is None
            else graph_index.operator_index(base_pre_op)
        )
        if (
            base_pre_op is None
            or base_pre_idx is None
            or str(base_pre_op.op_type) != "TRANSPOSE"
            or len(base_pre_op.inputs) < 2
            or len(base_pre_op.outputs) != 1
            or str(base_pre_op.outputs[0]) != base_nchw_name
            or _read_transpose_perm(model_ir, base_pre_op)
            != _PERM_NHWC_TO_NCHW
            or base_nchw_name in model_outputs
        ):
            continue
        base_nhwc_name = str(base_pre_op.inputs[0])
        if not _rank_four_tensor(model_ir, base_nhwc_name):
            continue

        branches: List[_SppBranchPlan] = []
        branch_add_indices: set[int] = set()
        for add_op, add_idx, add_out_name, adapters in raw_branches:
            resize_entries = [entry for entry in adapters if entry[2] != base_nchw_name]
            if len(resize_entries) != 1:
                valid = False
                break
            resize_post_op, resize_nhwc_name, resize_nchw_name = resize_entries[0]
            resize_post_idx = graph_index.operator_index(resize_post_op)
            resize_op = graph_index.producer(resize_nhwc_name)
            if (
                resize_post_idx is None
                or resize_op is None
                or str(resize_op.op_type) != "RESIZE_BILINEAR"
                or resize_nchw_name in model_outputs
                or set(graph_index.consumer_indices(resize_nchw_name))
                != {int(add_idx)}
            ):
                valid = False
                break
            new_inputs = tuple(
                base_nhwc_name
                if str(name) == base_nchw_name
                else resize_nhwc_name
                for name in add_op.inputs
            )
            if len(new_inputs) != 2:
                valid = False
                break
            branches.append(
                _SppBranchPlan(
                    add_op=add_op,
                    new_inputs=(str(new_inputs[0]), str(new_inputs[1])),
                    output_name=add_out_name,
                    resize_post_op=resize_post_op,
                )
            )
            branch_add_indices.add(int(add_idx))
        if not valid:
            continue

        concat0_users = graph_index.consumer_indices(concat0_out_name)
        if len(concat0_users) != 1:
            continue
        mul0_idx = int(concat0_users[0])
        mul0_op = model_ir.operators[mul0_idx]
        if (
            str(mul0_op.op_type) != "MUL"
            or len(mul0_op.inputs) != 2
            or len(mul0_op.outputs) != 1
        ):
            continue
        mul0_out_name = str(mul0_op.outputs[0])
        if mul0_out_name in model_outputs or not _rank_four_tensor(
            model_ir, mul0_out_name
        ):
            continue
        mul0_users = graph_index.consumer_indices(mul0_out_name)
        if len(mul0_users) != 1:
            continue
        post0_idx = int(mul0_users[0])
        post0_op = model_ir.operators[post0_idx]
        if (
            str(post0_op.op_type) != "TRANSPOSE"
            or len(post0_op.inputs) < 2
            or len(post0_op.outputs) != 1
            or str(post0_op.inputs[0]) != mul0_out_name
            or _read_transpose_perm(model_ir, post0_op)
            != _PERM_NCHW_TO_NHWC
        ):
            continue
        post0_out_name = str(post0_op.outputs[0])
        if post0_out_name in model_outputs or not _rank_four_tensor(
            model_ir, post0_out_name
        ):
            continue
        post0_users = graph_index.consumer_indices(post0_out_name)
        if len(post0_users) != 1:
            continue
        add0_idx = int(post0_users[0])
        add0_op = model_ir.operators[add0_idx]
        if (
            str(add0_op.op_type) != "ADD"
            or len(add0_op.inputs) != 2
            or len(add0_op.outputs) != 1
            or [str(name) for name in add0_op.inputs].count(post0_out_name) != 1
        ):
            continue
        add0_out_name = str(add0_op.outputs[0])
        add0_users = graph_index.consumer_indices(add0_out_name)
        if len(add0_users) != 1:
            continue
        conv0_idx = int(add0_users[0])
        conv0_op = model_ir.operators[conv0_idx]
        if (
            str(conv0_op.op_type) not in _CONV_LIKE_OPS
            or len(conv0_op.outputs) != 1
        ):
            continue
        conv0_out_name = str(conv0_op.outputs[0])
        if not _rank_four_tensor(model_ir, conv0_out_name):
            continue
        conv0_users = graph_index.consumer_indices(conv0_out_name)
        if len(conv0_users) != 1:
            continue
        conv0_post_idx = int(conv0_users[0])
        conv0_post_op = model_ir.operators[conv0_post_idx]
        if (
            str(conv0_post_op.op_type) != "TRANSPOSE"
            or len(conv0_post_op.inputs) < 2
            or len(conv0_post_op.outputs) != 1
            or str(conv0_post_op.inputs[0]) != conv0_out_name
            or _read_transpose_perm(model_ir, conv0_post_op)
            != _PERM_NHWC_TO_NCHW
        ):
            continue
        conv0_nchw_name = str(conv0_post_op.outputs[0])
        if conv0_nchw_name in model_outputs or not _rank_four_tensor(
            model_ir, conv0_nchw_name
        ):
            continue
        conv0_nchw_users = graph_index.consumer_indices(conv0_nchw_name)
        if len(conv0_nchw_users) != 1:
            continue
        concat1_idx = int(conv0_nchw_users[0])
        concat1_op = model_ir.operators[concat1_idx]
        if (
            str(concat1_op.op_type) != "CONCATENATION"
            or len(concat1_op.inputs) != 2
            or len(concat1_op.outputs) != 1
            or _normalized_axis(concat1_op) != 1
            or set(str(name) for name in concat1_op.inputs)
            != {base_nchw_name, conv0_nchw_name}
            or set(graph_index.consumer_indices(base_nchw_name))
            != branch_add_indices | {int(concat1_idx)}
        ):
            continue
        concat1_out_name = str(concat1_op.outputs[0])
        if concat1_out_name in model_outputs or not _rank_four_tensor(
            model_ir, concat1_out_name
        ):
            continue
        concat1_users = graph_index.consumer_indices(concat1_out_name)
        if len(concat1_users) != 1:
            continue
        mul1_idx = int(concat1_users[0])
        mul1_op = model_ir.operators[mul1_idx]
        if (
            str(mul1_op.op_type) != "MUL"
            or len(mul1_op.inputs) != 2
            or len(mul1_op.outputs) != 1
        ):
            continue
        mul1_out_name = str(mul1_op.outputs[0])
        if mul1_out_name in model_outputs or not _rank_four_tensor(
            model_ir, mul1_out_name
        ):
            continue
        mul1_users = graph_index.consumer_indices(mul1_out_name)
        if len(mul1_users) != 1:
            continue
        post1_idx = int(mul1_users[0])
        post1_op = model_ir.operators[post1_idx]
        if (
            str(post1_op.op_type) != "TRANSPOSE"
            or len(post1_op.inputs) < 2
            or len(post1_op.outputs) != 1
            or str(post1_op.inputs[0]) != mul1_out_name
            or _read_transpose_perm(model_ir, post1_op)
            != _PERM_NCHW_TO_NHWC
        ):
            continue
        post1_out_name = str(post1_op.outputs[0])
        if post1_out_name in model_outputs or not _rank_four_tensor(
            model_ir, post1_out_name
        ):
            continue
        post1_users = graph_index.consumer_indices(post1_out_name)
        if len(post1_users) != 1:
            continue
        add1_idx = int(post1_users[0])
        add1_op = model_ir.operators[add1_idx]
        if (
            str(add1_op.op_type) != "ADD"
            or len(add1_op.inputs) != 2
            or len(add1_op.outputs) != 1
            or [str(name) for name in add1_op.inputs].count(post1_out_name) != 1
        ):
            continue
        add1_users = graph_index.consumer_indices(str(add1_op.outputs[0]))
        if len(add1_users) != 1 or str(
            model_ir.operators[int(add1_users[0])].op_type
        ) not in _CONV_LIKE_OPS:
            continue

        allowed_constant_users = {int(mul0_idx), int(mul1_idx)}
        mul0_constant = _resolve_constant_plan(
            model_ir,
            graph_index,
            mul_op=mul0_op,
            data_name=concat0_out_name,
            allowed_consumer_indices=allowed_constant_users,
        )
        mul1_constant = _resolve_constant_plan(
            model_ir,
            graph_index,
            mul_op=mul1_op,
            data_name=concat1_out_name,
            allowed_consumer_indices=allowed_constant_users,
        )
        if mul0_constant is None or mul1_constant is None:
            continue

        return _SppLayoutCandidate(
            base_pre_op=base_pre_op,
            base_nhwc_name=base_nhwc_name,
            branches=tuple(branches),
            concat0_op=concat0_op,
            concat0_out_name=concat0_out_name,
            mul0_out_name=mul0_out_name,
            mul0_constant=mul0_constant,
            post0_op=post0_op,
            post0_out_name=post0_out_name,
            add0_op=add0_op,
            conv0_out_name=conv0_out_name,
            conv0_post_op=conv0_post_op,
            concat1_op=concat1_op,
            concat1_out_name=concat1_out_name,
            mul1_out_name=mul1_out_name,
            mul1_constant=mul1_constant,
            post1_op=post1_op,
            post1_out_name=post1_out_name,
            add1_op=add1_op,
        )
    return None


def _has_spp_layout_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_spp_layout_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in model_ir.tensors:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _clone_permuted_quantization(quantization: Any) -> Any:
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


def _apply_constant_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _SppConstantPlan,
    materialized_names: Dict[str, str],
) -> None:
    rewritten_name = materialized_names.get(plan.tensor_name)
    if rewritten_name is None:
        tensor = model_ir.tensors[plan.tensor_name]
        rewritten_name = plan.tensor_name
        if plan.clone_required:
            rewritten_name = _unique_tensor_name(
                model_ir,
                f"{plan.tensor_name}_nhwc",
            )
            model_ir.tensors[rewritten_name] = TensorIR(
                name=rewritten_name,
                dtype=str(tensor.dtype),
                shape=[int(value) for value in plan.nhwc_data.shape],
                shape_signature=[int(value) for value in plan.nhwc_data.shape],
                data=np.asarray(plan.nhwc_data),
                is_variable=bool(tensor.is_variable),
                quantization=_clone_permuted_quantization(
                    tensor.quantization
                ),
                logical_layout=str(tensor.logical_layout),
                physical_layout=str(tensor.physical_layout),
                onnx_tensor_name=tensor.onnx_tensor_name,
            )
        else:
            tensor.data = np.asarray(plan.nhwc_data)
            tensor.shape = [int(value) for value in plan.nhwc_data.shape]
            tensor.shape_signature = [
                int(value) for value in plan.nhwc_data.shape
            ]
            tensor.quantization = _clone_permuted_quantization(
                tensor.quantization
            )
        materialized_names[plan.tensor_name] = rewritten_name
    if str(plan.mul_op.inputs[plan.input_index]) != rewritten_name:
        _replace_operator_input_at(
            model_ir=model_ir,
            op=plan.mul_op,
            input_index=plan.input_index,
            new_input_name=rewritten_name,
            graph_index=graph_index,
        )


def _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """Collapse a fully validated two-island SPP bridge to NHWC."""

    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        candidate = _resolve_spp_layout_candidate(model_ir, graph_index)
        if candidate is None:
            break

        for branch in candidate.branches:
            _set_operator_inputs(
                model_ir=model_ir,
                op=branch.add_op,
                new_inputs=list(branch.new_inputs),
                graph_index=graph_index,
            )
            _permute_tensor_metadata_if_rank_matches(
                model_ir.tensors.get(branch.output_name),
                _PERM_NCHW_TO_NHWC,
            )

        candidate.concat0_op.options["axis"] = 3
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(candidate.concat0_out_name),
            _PERM_NCHW_TO_NHWC,
        )
        materialized_names: Dict[str, str] = {}
        _apply_constant_plan(
            model_ir,
            graph_index,
            candidate.mul0_constant,
            materialized_names,
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(candidate.mul0_out_name),
            _PERM_NCHW_TO_NHWC,
        )
        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.add0_op,
            new_inputs=[
                candidate.mul0_out_name
                if str(name) == candidate.post0_out_name
                else str(name)
                for name in candidate.add0_op.inputs
            ],
            graph_index=graph_index,
        )

        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat1_op,
            new_inputs=[
                candidate.base_nhwc_name
                if str(name) == str(candidate.base_pre_op.outputs[0])
                else candidate.conv0_out_name
                if str(name) == str(candidate.conv0_post_op.outputs[0])
                else str(name)
                for name in candidate.concat1_op.inputs
            ],
            graph_index=graph_index,
        )
        candidate.concat1_op.options["axis"] = 3
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(candidate.concat1_out_name),
            _PERM_NCHW_TO_NHWC,
        )
        _apply_constant_plan(
            model_ir,
            graph_index,
            candidate.mul1_constant,
            materialized_names,
        )
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(candidate.mul1_out_name),
            _PERM_NCHW_TO_NHWC,
        )
        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.add1_op,
            new_inputs=[
                candidate.mul1_out_name
                if str(name) == candidate.post1_out_name
                else str(name)
                for name in candidate.add1_op.inputs
            ],
            graph_index=graph_index,
        )

        remove_ops = [
            candidate.base_pre_op,
            candidate.post0_op,
            candidate.conv0_post_op,
            candidate.post1_op,
            *[branch.resize_post_op for branch in candidate.branches],
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
        rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(rewritten)}


def run_spp_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Run the two-island SPP NHWC propagation pass transactionally."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {
            "TRANSPOSE",
            "RESIZE_BILINEAR",
            "ADD",
            "CONCATENATION",
            "MUL",
        }
        seen_conv = False
        for visited, operator in enumerate(candidate_model.operators, start=1):
            op_type = str(operator.op_type)
            required.discard(op_type)
            seen_conv = seen_conv or op_type in _CONV_LIKE_OPS
            if not required and seen_conv:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(_STATS_KEY, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.generic_spp_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_spp_layout_candidate,
                priority=10,
                transactional=True,
            )
        ],
        layout_state=layout_state,
        default_details={_STATS_KEY: 0},
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
