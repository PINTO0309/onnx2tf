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
    _permute_shape,
    _permute_tensor_metadata_if_rank_matches,
    _prune_unused_tensors,
    _read_transpose_perm,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_PERM_NHWC_TO_NCHW = [0, 3, 1, 2]
_PERM_NCHW_TO_NHWC = [0, 2, 3, 1]


@dataclass(frozen=True)
class _ConstantRewrite:
    tensor_name: str
    data: np.ndarray


@dataclass(frozen=True)
class _Axis3ConstConcatCandidate:
    concat_op: OperatorIR
    concat_output: str
    adapter_pre_op: OperatorIR
    adapter_source: str
    adapter_output: str
    adapter_perm: str
    new_concat_inputs: Tuple[str, ...]
    constant_rewrites: Tuple[_ConstantRewrite, ...]
    post_ops: Tuple[OperatorIR, ...]
    post_outputs: Tuple[str, ...]
    legacy_ops: Tuple[OperatorIR, ...]
    remove_pre_adapter: bool
    bridge_name: Optional[str]
    bridge_shape: Optional[Tuple[int, ...]]
    bridge_signature: Optional[Tuple[int, ...]]


def _dims_compatible(a: int, b: int) -> bool:
    if int(a) <= 0 or int(b) <= 0:
        return True
    return int(a) == int(b)


def _shape_compatible_except_axis2(
    shape_a: List[int],
    shape_b: List[int],
) -> bool:
    if len(shape_a) != 4 or len(shape_b) != 4:
        return False
    return all(
        _dims_compatible(int(shape_a[index]), int(shape_b[index]))
        for index in [0, 1, 3]
    )


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in model_ir.tensors:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _resolve_axis3_const_concat_candidate(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> Optional[_Axis3ConstConcatCandidate]:
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
        axis = int(concat_op.options.get("axis", 3))
        if axis < 0:
            axis += 4
        concat_inputs = [str(name) for name in concat_op.inputs]
        if axis != 3 or len(concat_inputs) < 2:
            continue

        adapter_plan: Optional[Tuple[int, OperatorIR, str, str, str]] = None
        multiple_adapters = False
        for input_index, input_name in enumerate(concat_inputs):
            producer_op = graph_index.producer(input_name)
            if (
                producer_op is None
                or str(producer_op.op_type) != "TRANSPOSE"
                or len(producer_op.inputs) < 2
                or len(producer_op.outputs) != 1
                or str(producer_op.outputs[0]) != input_name
                or _read_transpose_perm(model_ir, producer_op)
                != _PERM_NHWC_TO_NCHW
            ):
                continue
            if adapter_plan is not None:
                multiple_adapters = True
                break
            adapter_plan = (
                int(input_index),
                producer_op,
                str(producer_op.inputs[0]),
                str(producer_op.outputs[0]),
                str(producer_op.inputs[1]),
            )
        if adapter_plan is None or multiple_adapters:
            continue
        (
            adapter_input_index,
            adapter_pre_op,
            adapter_source,
            adapter_output,
            adapter_perm,
        ) = adapter_plan
        source_tensor = model_ir.tensors.get(adapter_source)
        if (
            adapter_output in model_outputs
            or source_tensor is None
            or source_tensor.shape is None
            or len(source_tensor.shape) != 4
        ):
            continue
        adapter_shape = [int(value) for value in source_tensor.shape]

        new_concat_inputs = list(concat_inputs)
        new_concat_inputs[int(adapter_input_index)] = adapter_source
        constant_rewrites: List[_ConstantRewrite] = []
        valid_constants = True
        for input_index, input_name in enumerate(concat_inputs):
            if int(input_index) == int(adapter_input_index):
                continue
            tensor = model_ir.tensors.get(input_name)
            if (
                input_name in model_outputs
                or tensor is None
                or tensor.data is None
                or tensor.shape is None
                or len(tensor.shape) != 4
                or set(graph_index.consumer_indices(input_name))
                != {int(concat_index)}
            ):
                valid_constants = False
                break
            data = np.asarray(tensor.data)
            if data.ndim != 4:
                valid_constants = False
                break
            converted = np.transpose(data, axes=_PERM_NCHW_TO_NHWC).astype(
                data.dtype,
                copy=False,
            )
            if not _shape_compatible_except_axis2(
                adapter_shape,
                [int(value) for value in converted.shape],
            ):
                valid_constants = False
                break
            constant_rewrites.append(
                _ConstantRewrite(input_name, np.asarray(converted))
            )
        if not valid_constants:
            continue

        concat_user_indices = graph_index.consumer_indices(concat_output)
        if not concat_user_indices:
            continue
        post_ops: List[OperatorIR] = []
        post_outputs: List[str] = []
        legacy_ops: List[OperatorIR] = []
        valid_consumers = True
        for user_index in concat_user_indices:
            user_op = model_ir.operators[int(user_index)]
            if (
                str(user_op.op_type) == "TRANSPOSE"
                and len(user_op.inputs) >= 2
                and len(user_op.outputs) == 1
                and str(user_op.inputs[0]) == concat_output
                and _read_transpose_perm(model_ir, user_op)
                == _PERM_NCHW_TO_NHWC
            ):
                post_output = str(user_op.outputs[0])
                if post_output in model_outputs:
                    valid_consumers = False
                    break
                post_ops.append(user_op)
                post_outputs.append(post_output)
            else:
                legacy_ops.append(user_op)
        if not valid_consumers or not post_ops:
            continue

        bridge_name: Optional[str] = None
        bridge_shape: Optional[Tuple[int, ...]] = None
        bridge_signature: Optional[Tuple[int, ...]] = None
        if legacy_ops:
            nhwc_shape = _permute_shape(
                [int(value) for value in concat_tensor.shape],
                _PERM_NCHW_TO_NHWC,
            )
            signature_source = (
                [int(value) for value in concat_tensor.shape_signature]
                if concat_tensor.shape_signature is not None
                else [int(value) for value in concat_tensor.shape]
            )
            nhwc_signature = _permute_shape(
                signature_source,
                _PERM_NCHW_TO_NHWC,
            )
            if nhwc_shape is None or nhwc_signature is None:
                continue
            planned_shape = _permute_shape(nhwc_shape, _PERM_NHWC_TO_NCHW)
            planned_signature = _permute_shape(
                nhwc_signature,
                _PERM_NHWC_TO_NCHW,
            )
            if planned_shape is None or planned_signature is None:
                continue
            bridge_name = _unique_tensor_name(
                model_ir,
                f"{concat_output}_nchw_bridge",
            )
            bridge_shape = tuple(int(value) for value in planned_shape)
            bridge_signature = tuple(
                int(value) for value in planned_signature
            )

        return _Axis3ConstConcatCandidate(
            concat_op=concat_op,
            concat_output=concat_output,
            adapter_pre_op=adapter_pre_op,
            adapter_source=adapter_source,
            adapter_output=adapter_output,
            adapter_perm=adapter_perm,
            new_concat_inputs=tuple(new_concat_inputs),
            constant_rewrites=tuple(constant_rewrites),
            post_ops=tuple(post_ops),
            post_outputs=tuple(post_outputs),
            legacy_ops=tuple(legacy_ops),
            remove_pre_adapter=(
                set(graph_index.consumer_indices(adapter_output))
                == {int(concat_index)}
            ),
            bridge_name=bridge_name,
            bridge_shape=bridge_shape,
            bridge_signature=bridge_signature,
        )
    return None


def _has_axis3_const_concat_candidate(pass_state: ModelIRPassState) -> bool:
    return (
        _resolve_axis3_const_concat_candidate(
            pass_state.model_ir,
            pass_state.graph_index,
        )
        is not None
    )


def _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex | None = None,
    layout_state: LayoutState | None = None,
) -> Dict[str, int]:
    """
    Collapse NCHW adapters around axis=3 CONCAT with const suffixes to a single NHWC->NCHW bridge.

    Target:
      x_nhwc --TRANSPOSE(0,3,1,2)--> x_nchw
      const_i_nchw --------------------^
      CONCAT(axis=3, [const..., x_nchw, ...]) -> y_nchw
      y_nchw --TRANSPOSE(0,2,3,1)--> y_nhwc   (at least one branch)
      y_nchw may also have legacy NCHW consumers.

    Rewrite:
      const_i_nchw -> const_i_nhwc (constant data permuted)
      CONCAT(axis=2, [const..., x_nhwc, ...]) -> y_nhwc
      remove post TRANSPOSE branches to y_nhwc
      if legacy NCHW consumers exist:
        y_nhwc --TRANSPOSE(0,3,1,2)--> y_nchw_bridge
        legacy consumers read y_nchw_bridge
    """
    optimized = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    while True:
        candidate = _resolve_axis3_const_concat_candidate(model_ir, graph_index)
        if candidate is None:
            break

        for rewrite in candidate.constant_rewrites:
            tensor = model_ir.tensors[rewrite.tensor_name]
            tensor.data = np.asarray(rewrite.data)
            tensor.shape = [int(value) for value in rewrite.data.shape]
            tensor.shape_signature = [
                int(value) for value in rewrite.data.shape
            ]

        _set_operator_inputs(
            model_ir=model_ir,
            op=candidate.concat_op,
            new_inputs=list(candidate.new_concat_inputs),
            graph_index=graph_index,
        )
        candidate.concat_op.options["axis"] = 2
        _permute_tensor_metadata_if_rank_matches(
            model_ir.tensors.get(candidate.concat_output),
            _PERM_NCHW_TO_NHWC,
        )

        for post_output in candidate.post_outputs:
            _replace_tensor_inputs(
                model_ir,
                post_output,
                candidate.concat_output,
                graph_index=graph_index,
            )

        if candidate.bridge_name is not None:
            concat_tensor = model_ir.tensors[candidate.concat_output]
            assert candidate.bridge_shape is not None
            assert candidate.bridge_signature is not None
            model_ir.tensors[candidate.bridge_name] = TensorIR(
                name=candidate.bridge_name,
                dtype=str(concat_tensor.dtype),
                shape=list(candidate.bridge_shape),
                shape_signature=list(candidate.bridge_signature),
                data=None,
                is_variable=False,
                quantization=_clone_quantization(concat_tensor.quantization),
            )
            for legacy_op in candidate.legacy_ops:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=legacy_op,
                    new_inputs=[
                        candidate.bridge_name
                        if str(input_name) == candidate.concat_output
                        else str(input_name)
                        for input_name in legacy_op.inputs
                    ],
                    graph_index=graph_index,
                )

        remove_ops = list(candidate.post_ops)
        if candidate.remove_pre_adapter:
            remove_ops.append(candidate.adapter_pre_op)
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

        if candidate.bridge_name is not None:
            legacy_indices = [
                int(index)
                for op in candidate.legacy_ops
                if (index := graph_index.operator_index(op)) is not None
            ]
            if not legacy_indices:
                raise RuntimeError(
                    "axis3 constant-Concat legacy bridge lost all consumers"
                )
            graph_index.insert_operator(
                min(legacy_indices),
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[candidate.concat_output, candidate.adapter_perm],
                    outputs=[candidate.bridge_name],
                ),
            )

        optimized += 1

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if optimized > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "optimized_transpose_axis3_const_concat_bridge_nhwc_chains": int(
            optimized
        )
    }


def run_axis3_const_concat_layout_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: LayoutState | None = None,
    diagnostics: List[Dict[str, Any]] | None = None,
) -> Dict[str, int]:
    """Propagate a validated axis-3 constant-Concat island to NHWC."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        required = {"TRANSPOSE", "CONCATENATION"}
        for visited, operator in enumerate(candidate_model.operators, start=1):
            required.discard(str(operator.op_type))
            if not required:
                return ModelIRPreflightResult(True, visited)
        return ModelIRPreflightResult(False, len(candidate_model.operators))

    stats_key = (
        "optimized_transpose_axis3_const_concat_bridge_nhwc_chains"
    )

    def _run(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {**stats, "changed": bool(stats.get(stats_key, 0))}

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="layout.axis3_const_concat_bridge_nhwc",
                phase=PassPhase.LAYOUT_PLAN,
                callback=_run,
                precondition=_has_axis3_const_concat_candidate,
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
