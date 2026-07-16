from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _append_tensor_lineage_event,
    _prune_unused_tensors,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR


def _protected_boundary_tensor_names(model_ir: ModelIR) -> set[str]:
    raw_names = model_ir.metadata.get("protected_boundary_tensor_names", [])
    if not isinstance(raw_names, list):
        return set()
    return {str(name) for name in raw_names if str(name).strip() != ""}


def optimize_fuse_activation_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Fuse supported producer/activation chains without rebuilding maps."""

    fused = 0
    fused_conv = 0
    fused_add = 0
    fused_sub = 0
    fused_mul = 0
    fused_div = 0
    skip_add_activation_fuse_marker = "__skip_add_activation_fuse__"
    binary_activation_map = {
        "RELU": "RELU",
        "RELU6": "RELU6",
    }
    activation_map_by_producer = {
        "CONV_2D": {
            "RELU": "RELU",
            "RELU6": "RELU6",
            "RELU_N1_TO_1": "RELU_N1_TO_1",
        },
        "DEPTHWISE_CONV_2D": {
            "RELU": "RELU",
            "RELU6": "RELU6",
            "RELU_N1_TO_1": "RELU_N1_TO_1",
        },
        "ADD": dict(binary_activation_map),
        "SUB": dict(binary_activation_map),
        "MUL": dict(binary_activation_map),
        "DIV": dict(binary_activation_map),
    }
    protected_boundary_tensor_names = _protected_boundary_tensor_names(model_ir)
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )

    while True:
        changed = False

        for producer_idx in active_index.operator_indices_for_normalized_types(
            activation_map_by_producer
        ):
            producer_op = model_ir.operators[int(producer_idx)]
            producer_type = str(producer_op.op_type).upper()
            activation_map = activation_map_by_producer.get(producer_type, None)
            if activation_map is None:
                continue
            if len(producer_op.outputs) != 1:
                continue

            producer_opts = (
                dict(producer_op.options)
                if isinstance(producer_op.options, dict)
                else {}
            )
            fused_act = str(
                producer_opts.get("fusedActivationFunction", "NONE")
            ).upper()
            if fused_act != "NONE":
                continue
            producer_out_name = str(producer_op.outputs[0])
            producer_users = active_index.consumer_indices(producer_out_name)
            if len(producer_users) != 1:
                continue

            act_idx = int(producer_users[0])
            if act_idx < 0 or act_idx >= len(model_ir.operators):
                continue
            if act_idx == int(producer_idx):
                continue
            act_op = model_ir.operators[act_idx]
            act_type = str(act_op.op_type).upper()
            fused_target = activation_map.get(act_type, None)
            if fused_target is None:
                continue
            if len(act_op.inputs) != 1 or len(act_op.outputs) != 1:
                continue
            if str(act_op.inputs[0]) != producer_out_name:
                continue

            act_out_name = str(act_op.outputs[0])
            producer_out_tensor = model_ir.tensors.get(producer_out_name, None)
            act_out_tensor = model_ir.tensors.get(act_out_name, None)
            if producer_out_tensor is not None and act_out_tensor is not None:
                if str(producer_out_tensor.dtype).upper() != str(
                    act_out_tensor.dtype
                ).upper():
                    continue
            if producer_out_name in protected_boundary_tensor_names:
                continue
            if act_out_name in protected_boundary_tensor_names:
                continue
            # Keep explicit activation when its output is both a graph output
            # and an internal bridge. Fusing could relabel an NHWC bridge to an
            # ONNX/NCHW name and trigger a wrong later layout adapter.
            if (
                act_out_name in model_ir.outputs
                and len(active_index.consumer_indices(act_out_name)) > 0
            ):
                continue

            if producer_type == "ADD":
                # Transpose bridge rewrites may leave this marker behind. For
                # strict single-consumer Add->Activation chains, fusion is safe.
                producer_opts.pop(skip_add_activation_fuse_marker, None)
            producer_opts["fusedActivationFunction"] = str(fused_target)
            producer_op.options = producer_opts
            _set_operator_outputs(
                model_ir=model_ir,
                op=producer_op,
                new_outputs=[act_out_name],
                graph_index=active_index,
            )
            if producer_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                _append_tensor_lineage_event(
                    model_ir=model_ir,
                    event={
                        "kind": "fuse_conv_activation",
                        "conv_op_type": str(producer_op.op_type),
                        "activation_op_type": str(act_op.op_type),
                        "fused_activation": str(fused_target),
                        "conv_output": str(producer_out_name),
                        "fused_output": str(act_out_name),
                    },
                )
            elif producer_type == "ADD":
                _append_tensor_lineage_event(
                    model_ir=model_ir,
                    event={
                        "kind": "fuse_add_activation",
                        "add_op_type": str(producer_op.op_type),
                        "activation_op_type": str(act_op.op_type),
                        "fused_activation": str(fused_target),
                        "add_output": str(producer_out_name),
                        "fused_output": str(act_out_name),
                    },
                )
            elif producer_type in {"SUB", "MUL", "DIV"}:
                _append_tensor_lineage_event(
                    model_ir=model_ir,
                    event={
                        "kind": "fuse_binary_activation",
                        "binary_op_type": str(producer_op.op_type),
                        "activation_op_type": str(act_op.op_type),
                        "fused_activation": str(fused_target),
                        "binary_output": str(producer_out_name),
                        "fused_output": str(act_out_name),
                    },
                )

            active_index.remove_operator(int(act_idx))
            fused += 1
            if producer_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                fused_conv += 1
            elif producer_type == "ADD":
                fused_add += 1
            elif producer_type == "SUB":
                fused_sub += 1
            elif producer_type == "MUL":
                fused_mul += 1
            elif producer_type == "DIV":
                fused_div += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {
        "fused_conv_activation_chains": int(fused_conv),
        "fused_add_activation_chains": int(fused_add),
        "fused_sub_activation_chains": int(fused_sub),
        "fused_mul_activation_chains": int(fused_mul),
        "fused_div_activation_chains": int(fused_div),
        "fused_binary_activation_chains": int(
            fused_add + fused_sub + fused_mul + fused_div
        ),
        "fused_activation_chains_total": int(fused),
    }
