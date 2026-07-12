from __future__ import annotations

from typing import Any, Dict, List, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    preflight_any_operator,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _replace_tensor_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR


def _optimize_redundant_int64_to_int32_cast_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Collapse immediate signed/unsigned 64-to-32 Cast chains."""

    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(name) for name in model_ir.outputs)

        for cast2_idx, cast2_op in enumerate(model_ir.operators):
            if (
                str(cast2_op.op_type) != "CAST"
                or len(cast2_op.inputs) != 1
                or len(cast2_op.outputs) != 1
            ):
                continue

            mid_name = str(cast2_op.inputs[0])
            cast2_out_name = str(cast2_op.outputs[0])
            if mid_name in model_outputs or cast2_out_name in model_outputs:
                continue

            cast2_in_dtype = str(cast2_op.options.get("inDataType", "")).upper()
            cast2_out_dtype = str(cast2_op.options.get("outDataType", "")).upper()
            target_dtype = ""
            if cast2_in_dtype == "INT64" and cast2_out_dtype == "INT32":
                target_dtype = "INT32"
            elif cast2_in_dtype == "UINT64" and cast2_out_dtype == "UINT32":
                target_dtype = "UINT32"
            if target_dtype == "":
                continue

            cast1_idx = producers.get(mid_name, None)
            if cast1_idx is None or int(cast1_idx) == int(cast2_idx):
                continue
            cast1_op = model_ir.operators[int(cast1_idx)]
            if (
                str(cast1_op.op_type) != "CAST"
                or len(cast1_op.inputs) != 1
                or len(cast1_op.outputs) != 1
                or str(cast1_op.outputs[0]) != mid_name
            ):
                continue

            mid_users = [int(index) for index in consumers.get(mid_name, [])]
            if len(mid_users) != 1 or int(mid_users[0]) != int(cast2_idx):
                continue
            if str(cast1_op.options.get("outDataType", "")).upper() != cast2_in_dtype:
                continue

            cast1_op.options["outDataType"] = target_dtype
            mid_tensor = model_ir.tensors.get(mid_name, None)
            if mid_tensor is not None:
                mid_tensor.dtype = target_dtype
                src_tensor = model_ir.tensors.get(str(cast1_op.inputs[0]), None)
                if src_tensor is not None:
                    mid_tensor.quantization = _clone_quantization(
                        src_tensor.quantization
                    )

            _replace_tensor_inputs(
                model_ir=model_ir,
                src_name=cast2_out_name,
                dst_name=mid_name,
                graph_index=graph_index,
            )
            graph_index.remove_operator(int(cast2_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"optimized_redundant_int64_to_int32_cast_chains": int(rewritten)}


def _optimize_redundant_int32_to_int64_passthrough_cast_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Remove exclusive signed/unsigned 32-to-64 alias Casts."""

    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    while True:
        changed = False
        consumers = graph_index.consumers
        producers = graph_index.producers
        model_outputs = set(str(name) for name in model_ir.outputs)

        for widen_idx, widen_op in enumerate(model_ir.operators):
            if (
                str(widen_op.op_type) != "CAST"
                or len(widen_op.inputs) != 1
                or len(widen_op.outputs) != 1
            ):
                continue

            work_name = str(widen_op.inputs[0])
            alias_name = str(widen_op.outputs[0])
            if alias_name in model_outputs:
                continue

            widen_in_dtype = str(widen_op.options.get("inDataType", "")).upper()
            widen_out_dtype = str(widen_op.options.get("outDataType", "")).upper()
            if (widen_in_dtype, widen_out_dtype) not in {
                ("INT32", "INT64"),
                ("UINT32", "UINT64"),
            }:
                continue

            work_users = [int(index) for index in consumers.get(work_name, [])]
            if len(work_users) != 1 or int(work_users[0]) != int(widen_idx):
                continue
            alias_users = [int(index) for index in consumers.get(alias_name, [])]
            if len(alias_users) <= 0:
                continue
            if not all(
                str(model_ir.operators[int(user_idx)].op_type) == "CAST"
                for user_idx in alias_users
            ):
                continue

            producer_idx = producers.get(work_name, None)
            if producer_idx is None or int(producer_idx) == int(widen_idx):
                continue
            producer_op = model_ir.operators[int(producer_idx)]
            if (
                len(producer_op.outputs) != 1
                or str(producer_op.outputs[0]) != work_name
            ):
                continue

            work_tensor = model_ir.tensors.get(work_name, None)
            alias_tensor = model_ir.tensors.get(alias_name, None)
            if alias_tensor is None:
                continue

            _set_operator_outputs(
                model_ir=model_ir,
                op=producer_op,
                new_outputs=[alias_name],
                graph_index=graph_index,
            )
            alias_tensor.dtype = widen_in_dtype
            if work_tensor is not None:
                alias_tensor.shape = [int(dim) for dim in work_tensor.shape]
                alias_tensor.shape_signature = (
                    [int(dim) for dim in work_tensor.shape_signature]
                    if work_tensor.shape_signature is not None
                    else [int(dim) for dim in work_tensor.shape]
                )
                alias_tensor.quantization = _clone_quantization(
                    work_tensor.quantization
                )

            for alias_user_idx in alias_users:
                alias_user = model_ir.operators[int(alias_user_idx)]
                alias_user.options["inDataType"] = widen_in_dtype

            graph_index.remove_operator(int(widen_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {
        "optimized_redundant_int32_to_int64_passthrough_cast_chains": int(
            rewritten
        )
    }


def run_redundant_cast_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run widening-alias then narrowing-chain Cast cleanup in fixed order."""

    def _preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        relevant_pairs = {
            ("INT32", "INT64"),
            ("UINT32", "UINT64"),
            ("INT64", "INT32"),
            ("UINT64", "UINT32"),
        }
        return preflight_any_operator(
            candidate_model,
            lambda op: (
                str(op.op_type) == "CAST"
                and (
                    str(op.options.get("inDataType", "")).upper(),
                    str(op.options.get("outDataType", "")).upper(),
                )
                in relevant_pairs
            ),
        )

    def _has_widening_alias_candidate(pass_state: ModelIRPassState) -> bool:
        return any(
            str(op.op_type) == "CAST"
            and (
                str(op.options.get("inDataType", "")).upper(),
                str(op.options.get("outDataType", "")).upper(),
            )
            in {("INT32", "INT64"), ("UINT32", "UINT64")}
            for op in pass_state.model_ir.operators
        )

    def _has_narrowing_candidate(pass_state: ModelIRPassState) -> bool:
        return any(
            str(op.op_type) == "CAST"
            and (
                str(op.options.get("inDataType", "")).upper(),
                str(op.options.get("outDataType", "")).upper(),
            )
            in {("INT64", "INT32"), ("UINT64", "UINT32")}
            for op in pass_state.model_ir.operators
        )

    def _run_widening_alias(
        pass_state: ModelIRPassState,
    ) -> Dict[str, int | bool]:
        stats = _optimize_redundant_int32_to_int64_passthrough_cast_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get(
                    "optimized_redundant_int32_to_int64_passthrough_cast_chains",
                    0,
                )
            ),
        }

    def _run_narrowing(pass_state: ModelIRPassState) -> Dict[str, int | bool]:
        stats = _optimize_redundant_int64_to_int32_cast_chains(
            pass_state.model_ir,
            graph_index=pass_state.graph_index,
            layout_state=pass_state.layout_state,
        )
        return {
            **stats,
            "changed": bool(
                stats.get("optimized_redundant_int64_to_int32_cast_chains", 0)
            ),
        }

    details, _ = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="cleanup.cast_widening_alias",
                phase=PassPhase.POST_LOWERING_CLEANUP,
                priority=10,
                callback=_run_widening_alias,
                precondition=_has_widening_alias_candidate,
                transactional=True,
            ),
            PassSpec(
                pass_id="cleanup.cast_narrowing_chain",
                phase=PassPhase.POST_LOWERING_CLEANUP,
                priority=20,
                callback=_run_narrowing,
                precondition=_has_narrowing_candidate,
                transactional=True,
            ),
        ],
        layout_state=layout_state,
        default_details={
            "optimized_redundant_int32_to_int64_passthrough_cast_chains": 0,
            "optimized_redundant_int64_to_int32_cast_chains": 0,
        },
        diagnostics=diagnostics,
        preflight=_preflight,
    )
    return {str(key): int(value) for key, value in details.items()}
