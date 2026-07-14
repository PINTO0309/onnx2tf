from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    channel_last_logical_layout,
    channel_first_logical_layout,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    logical_layout_permutation,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _constant_int_list,
    _shape_lists_equal,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _is_inconsistent_same_layout_transpose,
    _is_inconsistent_standard_layout_transpose,
    _perm_cf_to_cl,
    _read_transpose_perm,
    _tensor_name_suggests_channel_last_layout_for_codegen,
)
from onnx2tf.tflite_builder.passes.pytorch_compat import (
    _is_reshape_only_residual_layout_bridge_transpose,
)
from onnx2tf.tflite_builder.passes.pytorch_recurrent import (
    _sequence_lstm_index_spec,
    _sequence_lstm_input_name,
)


_DIRECT_CODEGEN_UNARY_EXPRESSIONS: Dict[str, str] = {
    "ABS": "torch.abs({x})",
    "ACOS": "torch.acos({x})",
    "ASIN": "torch.asin({x})",
    "ATAN": "torch.atan({x})",
    "CEIL": "torch.ceil({x})",
    "COS": "torch.cos({x})",
    "ELU": "F.elu({x})",
    "EXP": "torch.exp({x})",
    "FLOOR": "torch.floor({x})",
    "GELU": "F.gelu({x})",
    "HARD_SWISH": "F.hardswish({x})",
    "IDENTITY": "{x}",
    "LEAKY_RELU": "F.leaky_relu({x}, negative_slope={alpha})",
    "LOG": "torch.log({x})",
    "LOGICAL_NOT": "torch.logical_not({x})",
    "LOGISTIC": "torch.sigmoid({x})",
    "NEG": "torch.neg({x})",
    "RELU": "torch.relu({x})",
    "RELU_0_TO_1": "torch.clamp({x}, min=0.0, max=1.0)",
    "RELU_N1_TO_1": "torch.clamp({x}, min=-1.0, max=1.0)",
    "RELU6": "torch.clamp({x}, min=0.0, max=6.0)",
    "ROUND": "torch.round({x})",
    "RSQRT": "torch.rsqrt({x})",
    "SIGMOID": "torch.sigmoid({x})",
    "SIGN": "torch.sign({x})",
    "SIN": "torch.sin({x})",
    "SQRT": "torch.sqrt({x})",
    "SQUARE": "torch.square({x})",
    "TAN": "torch.tan({x})",
    "TANH": "torch.tanh({x})",
}

_DIRECT_CODEGEN_BINARY_FUNCTIONS: Dict[str, str] = {
    "ADD": "torch.add",
    "ATAN2": "torch.atan2",
    "DIV": "torch.div",
    "EQUAL": "torch.eq",
    "FLOOR_MOD": "torch.remainder",
    "GREATER": "torch.gt",
    "GREATER_EQUAL": "torch.ge",
    "LESS": "torch.lt",
    "LOGICAL_AND": "torch.logical_and",
    "LOGICAL_OR": "torch.logical_or",
    "MAXIMUM": "torch.maximum",
    "MINIMUM": "torch.minimum",
    "MUL": "torch.mul",
    "NOT_EQUAL": "torch.ne",
    "POW": "torch.pow",
    "SUB": "torch.sub",
}

_DIRECT_CODEGEN_MODULE_OP_TYPES: Set[str] = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "TRANSPOSE_CONV",
    "CONV_3D",
    "CONV_3D_TRANSPOSE",
    "FULLY_CONNECTED",
    "PRELU",
    "UNIDIRECTIONAL_SEQUENCE_RNN",
    "UNIDIRECTIONAL_SEQUENCE_LSTM",
    "BIDIRECTIONAL_SEQUENCE_LSTM",
}


def _activation_lines_for_codegen(var_name: str, fused: str) -> List[str]:
    key = str(fused).upper()
    if key in {"", "NONE"}:
        return []
    if key == "RELU":
        return [f"{var_name} = torch.relu({var_name})"]
    if key == "RELU6":
        return [f"{var_name} = torch.clamp({var_name}, min=0.0, max=6.0)"]
    if key == "RELU_N1_TO_1":
        return [f"{var_name} = torch.clamp({var_name}, min=-1.0, max=1.0)"]
    if key == "RELU_0_TO_1":
        return [f"{var_name} = torch.clamp({var_name}, min=0.0, max=1.0)"]
    if key == "SILU":
        return [f"{var_name} = torch.mul({var_name}, torch.sigmoid({var_name}))"]
    if key == "TANH":
        return [f"{var_name} = torch.tanh({var_name})"]
    return [f"{var_name} = _apply_fused_activation({var_name}, {fused!r})"]


def _emit_maybe_aligned_expr_for_codegen(
    *,
    runtime_imports: Set[str],
    output_name: str,
    expr: str,
    inferred_shape: Optional[Sequence[int]],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    target_shape_literal_fn: Callable[[str], str],
) -> str:
    output_shape = tensor_shape_list_fn(output_name)
    if _shape_lists_equal(inferred_shape, output_shape):
        return expr
    runtime_imports.add("_align_tensor_to_target_shape")
    return (
        f"_align_tensor_to_target_shape("
        f"{expr}, {target_shape_literal_fn(output_name)})"
    )


def _emit_module_output_expr_for_codegen(
    *,
    model_ir: ModelIR,
    runtime_imports: Set[str],
    output_name: str,
    expr: str,
    raw_output_layout: str,
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    target_shape_literal_fn: Callable[[str], str],
) -> str:
    output_tensor = model_ir.tensors.get(str(output_name), None)
    output_layout = (
        normalize_logical_layout(output_tensor.logical_layout)
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    normalized_raw_layout = normalize_logical_layout(raw_output_layout)
    if (
        output_layout != LOGICAL_LAYOUT_UNKNOWN
        and normalized_raw_layout != LOGICAL_LAYOUT_UNKNOWN
        and output_layout != normalized_raw_layout
    ):
        output_shape = tensor_shape_list_fn(output_name)
        rank = (
            len(list(output_shape))
            if output_shape is not None
            else len(list(output_tensor.shape))
            if output_tensor is not None
            else 0
        )
        if rank in {3, 4, 5}:
            perm = logical_layout_permutation(
                source_layout=normalized_raw_layout,
                target_layout=output_layout,
            )
            if perm is not None:
                expr = (
                    f"{expr}.permute("
                    f"{', '.join(str(int(v)) for v in perm)}).contiguous()"
                )
    runtime_imports.add("_align_tensor_to_target_shape")
    return (
        f"_align_tensor_to_target_shape("
        f"{expr}, {target_shape_literal_fn(output_name)})"
    )


def _emit_native_recurrent_module_op_for_codegen(
    *,
    op: OperatorIR,
    op_type: str,
    attr_name: str,
    output_vars: Sequence[str],
    output_target_shape: str,
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
) -> bool:
    if op_type == "UNIDIRECTIONAL_SEQUENCE_RNN":
        x_expr = tensor_expr_fn(str(op.inputs[0]))
        h0_name = _sequence_lstm_input_name(op, 4)
        state_arg = tensor_expr_fn(h0_name) if h0_name != "" else "None"
        forward_lines.append(
            f"{output_vars[0]} = _align_tensor_to_target_shape("
            f"self.{attr_name}({x_expr}, {state_arg}), "
            f"{output_target_shape})"
        )
        runtime_imports.add("_align_tensor_to_target_shape")
        return True
    if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
        x_expr = tensor_expr_fn(str(op.inputs[0]))
        index_spec = _sequence_lstm_index_spec(op)
        if index_spec is None:
            raise ModelIRPyTorchExportError(
                "Native PyTorch-like model.py codegen could not resolve "
                "UNIDIRECTIONAL_SEQUENCE_LSTM state layout."
            )
        state_indices = list(index_spec["state_indices"])
        h0_name = _sequence_lstm_input_name(op, state_indices[0])
        c0_name = _sequence_lstm_input_name(op, state_indices[1])
        state_args = [
            tensor_expr_fn(h0_name) if h0_name != "" else "None",
            tensor_expr_fn(c0_name) if c0_name != "" else "None",
        ]
        forward_lines.append(
            f"{output_vars[0]} = _align_tensor_to_target_shape("
            f"self.{attr_name}({x_expr}, {', '.join(state_args)}), "
            f"{output_target_shape})"
        )
        runtime_imports.add("_align_tensor_to_target_shape")
        return True
    if op_type == "BIDIRECTIONAL_SEQUENCE_LSTM":
        x_expr = tensor_expr_fn(str(op.inputs[0]))
        index_spec = _sequence_lstm_index_spec(op)
        if index_spec is None:
            raise ModelIRPyTorchExportError(
                "Native PyTorch-like model.py codegen could not resolve "
                "BIDIRECTIONAL_SEQUENCE_LSTM state layout."
            )
        state_indices = list(index_spec["state_indices"])
        fw_h0_name = _sequence_lstm_input_name(op, state_indices[0])
        fw_c0_name = _sequence_lstm_input_name(op, state_indices[1])
        bw_h0_name = _sequence_lstm_input_name(op, state_indices[2])
        bw_c0_name = _sequence_lstm_input_name(op, state_indices[3])
        state_args = [
            tensor_expr_fn(fw_h0_name) if fw_h0_name != "" else "None",
            tensor_expr_fn(fw_c0_name) if fw_c0_name != "" else "None",
            tensor_expr_fn(bw_h0_name) if bw_h0_name != "" else "None",
            tensor_expr_fn(bw_c0_name) if bw_c0_name != "" else "None",
        ]
        forward_lines.append(
            f"{output_vars[0]} = _align_tensor_to_target_shape("
            f"self.{attr_name}({x_expr}, {', '.join(state_args)}), "
            f"{output_target_shape})"
        )
        runtime_imports.add("_align_tensor_to_target_shape")
        return True
    return False


def _emit_native_fully_connected_module_op_for_codegen(
    *,
    op: OperatorIR,
    op_type: str,
    attr_name: str,
    output_vars: Sequence[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    activation_lines_fn: Callable[[str, str], List[str]],
) -> bool:
    if op_type != "FULLY_CONNECTED":
        return False
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    forward_lines.append(
        f"{output_vars[0]} = self.{attr_name}("
        f"{tensor_expr_fn(str(op.inputs[0]))})"
    )
    forward_lines.extend(activation_lines_fn(output_vars[0], fused))
    return True


def _emit_native_prelu_module_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_type: str,
    attr_name: str,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    emit_maybe_aligned_expr_fn: Callable[..., str],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    should_skip_align_for_shape_preserving_unary_fn: Callable[
        [str, str], bool
    ],
) -> bool:
    if op_type != "PRELU":
        return False
    prelu_input_name = str(op.inputs[0])
    prelu_input_expr = tensor_expr_fn(prelu_input_name)
    prelu_input_tensor = model_ir.tensors.get(prelu_input_name, None)
    prelu_output_tensor = model_ir.tensors.get(outputs[0], None)
    prelu_input_layout = (
        normalize_logical_layout(prelu_input_tensor.logical_layout)
        if prelu_input_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    prelu_output_layout = (
        normalize_logical_layout(prelu_output_tensor.logical_layout)
        if prelu_output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    prelu_rank = (
        len(list(prelu_input_tensor.shape))
        if prelu_input_tensor is not None
        else 0
    )
    prelu_num_parameters = 1
    prelu_weight_tensor = (
        model_ir.tensors.get(str(op.inputs[1]), None)
        if len(op.inputs) >= 2
        else None
    )
    if prelu_weight_tensor is not None:
        if isinstance(prelu_weight_tensor.data, np.ndarray):
            prelu_num_parameters = max(
                1, int(np.asarray(prelu_weight_tensor.data).size)
            )
        elif len(list(prelu_weight_tensor.shape)) > 0:
            shape_values = [
                int(value)
                for value in list(prelu_weight_tensor.shape)
                if int(value) > 0
            ]
            if len(shape_values) > 0:
                prelu_num_parameters = max(
                    1, int(np.prod(shape_values, dtype=np.int64))
                )
    expr = f"self.{attr_name}({prelu_input_expr})"
    if (
        prelu_num_parameters > 1
        and prelu_rank in {3, 4, 5}
        and is_channel_last_logical_layout(prelu_input_layout)
    ):
        pre_perm = logical_layout_permutation(
            source_layout=prelu_input_layout,
            target_layout=channel_first_logical_layout(prelu_rank),
        )
        post_perm = logical_layout_permutation(
            source_layout=channel_first_logical_layout(prelu_rank),
            target_layout=(
                prelu_output_layout
                if is_channel_last_logical_layout(prelu_output_layout)
                else prelu_input_layout
            ),
        )
        if pre_perm is not None and post_perm is not None:
            permuted_input_expr = (
                f"{prelu_input_expr}.permute("
                f"{', '.join(str(int(value)) for value in pre_perm)}"
                ").contiguous()"
            )
            expr = (
                f"self.{attr_name}({permuted_input_expr}).permute("
                f"{', '.join(str(int(value)) for value in post_perm)}"
                ").contiguous()"
            )
    inferred_shape = tensor_shape_list_fn(str(op.inputs[0]))
    if should_skip_align_for_shape_preserving_unary_fn(
        str(op.inputs[0]), str(outputs[0])
    ):
        forward_lines.append(f"{output_vars[0]} = {expr}")
    else:
        aligned_expr = emit_maybe_aligned_expr_fn(
            output_name=str(outputs[0]),
            expr=expr,
            inferred_shape=inferred_shape,
        )
        forward_lines.append(f"{output_vars[0]} = {aligned_expr}")
    return True


def _emit_native_transpose_conv2d_module_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_type: str,
    attr_name: str,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    activation_lines_fn: Callable[[str, str], List[str]],
) -> bool:
    if op_type != "TRANSPOSE_CONV":
        return False
    runtime_imports.add("_apply_module_transpose_conv2d")
    output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    output_tensor = model_ir.tensors.get(str(outputs[0]), None)
    fallback_shape = (
        [
            int(value)
            for value in np.asarray(output_shape_tensor.data)
            .reshape(-1)
            .tolist()
        ]
        if output_shape_tensor is not None
        and isinstance(output_shape_tensor.data, np.ndarray)
        else [int(value) for value in list(model_ir.tensors[outputs[0]].shape)]
    )
    transpose_conv_target_shape = output_target_shape
    transpose_conv_target_layout = normalize_logical_layout(
        model_ir.tensors[outputs[0]].logical_layout
    )
    if (
        output_tensor is not None
        and len(list(output_tensor.shape)) == 4
        and is_channel_first_logical_layout(transpose_conv_target_layout)
        and _tensor_name_suggests_channel_last_layout_for_codegen(
            str(outputs[0])
        )
    ):
        transpose_conv_target_shape = repr(
            [int(value) for value in list(output_tensor.shape)]
        )
        transpose_conv_target_layout = "NHWC"
    forward_lines.append(
        f"{output_vars[0]} = _apply_module_transpose_conv2d("
        f"{tensor_expr_fn(str(op.inputs[2]))}, "
        f"self.{attr_name}.weight, self.{attr_name}.bias, "
        f"list(self.{attr_name}.stride), list(self.{attr_name}.padding), "
        f"list(self.{attr_name}.dilation), "
        f"list(self.{attr_name}.output_padding), self.{attr_name}.groups, "
        f"target_shape={transpose_conv_target_shape}, "
        f"fallback_shape={repr(fallback_shape)}, "
        f"target_logical_layout={repr(transpose_conv_target_layout)}, "
        "fused='NONE')"
    )
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    forward_lines.extend(activation_lines_fn(output_vars[0], fused))
    return True


def _emit_native_transpose_conv3d_module_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_type: str,
    attr_name: str,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    activation_lines_fn: Callable[[str, str], List[str]],
) -> bool:
    if op_type != "CONV_3D_TRANSPOSE":
        return False
    runtime_imports.add("_apply_module_transpose_conv3d")
    output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    fallback_shape = (
        [
            int(value)
            for value in np.asarray(output_shape_tensor.data)
            .reshape(-1)
            .tolist()
        ]
        if output_shape_tensor is not None
        and isinstance(output_shape_tensor.data, np.ndarray)
        else [int(value) for value in list(model_ir.tensors[outputs[0]].shape)]
    )
    target_layout = normalize_logical_layout(
        model_ir.tensors[outputs[0]].logical_layout
    )
    forward_lines.append(
        f"{output_vars[0]} = _apply_module_transpose_conv3d("
        f"{tensor_expr_fn(str(op.inputs[2]))}, "
        f"self.{attr_name}.weight, self.{attr_name}.bias, "
        f"list(self.{attr_name}.stride), list(self.{attr_name}.padding), "
        f"list(self.{attr_name}.dilation), "
        f"list(self.{attr_name}.output_padding), self.{attr_name}.groups, "
        f"target_shape={output_target_shape}, "
        f"fallback_shape={repr(fallback_shape)}, "
        f"target_logical_layout={repr(target_layout)}, fused='NONE')"
    )
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    forward_lines.extend(activation_lines_fn(output_vars[0], fused))
    return True


def _emit_native_conv3d_module_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_type: str,
    attr_name: str,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    derived_local_var_name_fn: Callable[[str, str], str],
    emit_module_output_expr_fn: Callable[..., str],
    can_emit_direct_module_call_fn: Callable[[OperatorIR], bool],
    activation_lines_fn: Callable[[str, str], List[str]],
) -> bool:
    if op_type != "CONV_3D":
        return False
    output_name = str(outputs[0])
    output_tensor = model_ir.tensors.get(output_name, None)
    output_layout = normalize_logical_layout(
        output_tensor.logical_layout
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    raw_output_layout = (
        channel_first_logical_layout(len(list(output_tensor.shape)))
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    needs_materialized_output_bridge = (
        output_tensor is not None
        and len(list(output_tensor.shape)) in {3, 4, 5}
        and output_layout != LOGICAL_LAYOUT_UNKNOWN
        and output_layout != raw_output_layout
    )
    raw_output_var = (
        derived_local_var_name_fn(f"{output_vars[0]}_cf", "t")
        if needs_materialized_output_bridge
        else output_vars[0]
    )
    used_direct_module_call = can_emit_direct_module_call_fn(op)
    if used_direct_module_call:
        if needs_materialized_output_bridge:
            channel_first_tensor_expr_aliases[output_name] = raw_output_var
        else:
            channel_first_tensor_expr_aliases.pop(output_name, None)
        forward_lines.append(
            f"{raw_output_var} = self.{attr_name}("
            f"{tensor_expr_fn(str(op.inputs[0]))})"
        )
    else:
        runtime_imports.add("_apply_module_conv3d")
        target_layout = normalize_logical_layout(
            model_ir.tensors[outputs[0]].logical_layout
        )
        forward_lines.append(
            f"{output_vars[0]} = _apply_module_conv3d("
            f"self.{attr_name}, {tensor_expr_fn(str(op.inputs[0]))}, "
            f"target_shape={output_target_shape}, "
            f"target_logical_layout={repr(target_layout)}, fused='NONE')"
        )
        channel_first_tensor_expr_aliases.pop(output_name, None)
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    if used_direct_module_call and raw_output_var != output_vars[0]:
        if fused != "NONE":
            forward_lines[-1] = forward_lines[-1].replace(
                f"{output_vars[0]} =", f"{raw_output_var} =", 1
            )
        output_expr = emit_module_output_expr_fn(
            output_name=output_name,
            expr=raw_output_var,
            raw_output_layout=raw_output_layout,
        )
        forward_lines.append(f"{output_vars[0]} = {output_expr}")
    forward_lines.extend(activation_lines_fn(output_vars[0], fused))
    return True


def _emit_native_conv2d_module_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_type: str,
    op_index: int,
    attr_name: str,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    conv_module_pad_specs: Dict[int, Optional[List[int]]],
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    tensor_expr_for_channel_first_bridge_fn: Callable[
        [str, Sequence[int]], Optional[str]
    ],
    derived_local_var_name_fn: Callable[[str, str], str],
    emit_module_output_expr_fn: Callable[..., str],
    target_shape_literal_fn: Callable[[str], str],
    conv2d_input_pre_permute_fn: Callable[..., Optional[List[int]]],
    can_emit_direct_module_call_fn: Callable[[OperatorIR], bool],
    activation_lines_fn: Callable[[str, str], List[str]],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
) -> bool:
    if op_type not in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
        return False
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    conv_input_name = str(op.inputs[0])
    conv_input_expr = tensor_expr_fn(conv_input_name)
    output_name = str(outputs[0])
    output_tensor = model_ir.tensors.get(output_name, None)
    raw_output_layout = (
        channel_first_logical_layout(len(list(output_tensor.shape)))
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    output_layout = normalize_logical_layout(
        output_tensor.logical_layout
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    needs_materialized_output_bridge = (
        output_tensor is not None
        and len(list(output_tensor.shape)) in {3, 4, 5}
        and output_layout != LOGICAL_LAYOUT_UNKNOWN
        and output_layout != raw_output_layout
    )
    use_channel_first_alias = (
        output_tensor is not None
        and len(list(output_tensor.shape)) in {3, 4, 5}
        and is_channel_first_logical_layout(output_layout)
    )
    raw_output_var = (
        derived_local_var_name_fn(f"{output_vars[0]}_cf", "t")
        if use_channel_first_alias or needs_materialized_output_bridge
        else output_vars[0]
    )
    conv_input_tensor = model_ir.tensors.get(conv_input_name, None)
    conv_weight_tensor = (
        model_ir.tensors.get(str(op.inputs[1]), None)
        if len(op.inputs) >= 2
        else None
    )
    conv_input_layout = normalize_logical_layout(
        conv_input_tensor.logical_layout
        if conv_input_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    existing_channel_first_input_alias = channel_first_tensor_expr_aliases.get(
        conv_input_name, None
    )
    if (
        conv_input_tensor is not None
        and conv_weight_tensor is not None
        and len(list(conv_input_tensor.shape)) == 4
        and len(list(conv_weight_tensor.shape)) == 4
        and int(conv_input_tensor.shape[1]) == int(conv_weight_tensor.shape[1])
    ):
        input_pre_permute = None
    elif (
        conv_input_tensor is not None
        and len(list(conv_input_tensor.shape)) == 4
        and is_channel_first_logical_layout(conv_input_layout)
    ):
        input_pre_permute = None
    elif (
        existing_channel_first_input_alias is not None
        and conv_input_tensor is not None
        and conv_weight_tensor is not None
        and len(list(conv_input_tensor.shape)) == 4
        and len(list(conv_weight_tensor.shape)) == 4
        and int(conv_input_tensor.shape[1]) == 1
        and int(conv_input_tensor.shape[2]) == 1
        and int(conv_input_tensor.shape[3]) == int(conv_weight_tensor.shape[3])
    ):
        conv_input_expr = str(existing_channel_first_input_alias)
        input_pre_permute = None
    elif (
        conv_input_tensor is not None
        and conv_weight_tensor is not None
        and len(list(conv_input_tensor.shape)) == 4
        and len(list(conv_weight_tensor.shape)) == 4
        and int(conv_input_tensor.shape[1]) == 1
        and int(conv_input_tensor.shape[2]) == 1
        and int(conv_input_tensor.shape[3]) == int(conv_weight_tensor.shape[3])
    ):
        input_pre_permute = None
    else:
        input_pre_permute = conv2d_input_pre_permute_fn(
            tensor_shape_list_fn(str(op.inputs[0])),
            tensor_shape_list_fn(str(outputs[0])),
            tensor_shape_list_fn(str(op.inputs[1])),
            op.options,
            input_logical_layout=conv_input_layout,
            output_logical_layout=normalize_logical_layout(
                model_ir.tensors[outputs[0]].logical_layout
            ),
            depthwise=(op_type == "DEPTHWISE_CONV_2D"),
        )
    if input_pre_permute is not None:
        folded_channel_first_expr = tensor_expr_for_channel_first_bridge_fn(
            str(op.inputs[0]),
            input_pre_permute,
        )
        if folded_channel_first_expr is not None:
            conv_input_expr = str(folded_channel_first_expr)
        else:
            conv_input_expr = (
                f"{conv_input_expr}.permute("
                f"{', '.join(str(int(value)) for value in input_pre_permute)}"
                ").contiguous()"
            )
    conv_pad_arg = conv_module_pad_specs.get(int(op_index), None)
    if conv_pad_arg is not None:
        conv_input_expr = (
            f"F.pad({conv_input_expr}, {repr(conv_pad_arg)}, "
            "mode='constant', value=0.0)"
        )
    used_direct_module_call = can_emit_direct_module_call_fn(op)
    if used_direct_module_call:
        if use_channel_first_alias or needs_materialized_output_bridge:
            channel_first_tensor_expr_aliases[output_name] = raw_output_var
        else:
            channel_first_tensor_expr_aliases.pop(output_name, None)
        forward_lines.append(
            f"{raw_output_var} = self.{attr_name}({conv_input_expr})"
        )
    else:
        runtime_imports.add("_apply_module_conv2d")
        target_layout = normalize_logical_layout(
            model_ir.tensors[outputs[0]].logical_layout
        )
        forward_lines.append(
            f"{output_vars[0]} = _apply_module_conv2d("
            f"self.{attr_name}, {conv_input_expr}, "
            f"target_shape={output_target_shape}, "
            f"target_logical_layout={repr(target_layout)}, fused='NONE')"
        )
        channel_first_tensor_expr_aliases.pop(output_name, None)
    forward_lines.extend(activation_lines_fn(output_vars[0], fused))
    if use_channel_first_alias:
        if fused != "NONE":
            forward_lines[-1] = forward_lines[-1].replace(
                f"{output_vars[0]} =", f"{raw_output_var} =", 1
            )
        if output_name in model_ir.outputs:
            public_output_expr = emit_module_output_expr_fn(
                output_name=output_name,
                expr=raw_output_var,
                raw_output_layout=raw_output_layout,
            )
            weight_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if (
                output_tensor is not None
                and weight_tensor is not None
                and len(list(output_tensor.shape)) == 4
                and len(list(weight_tensor.shape)) >= 1
                and int(output_tensor.shape[-1]) == int(weight_tensor.shape[0])
                and int(output_tensor.shape[1]) != int(weight_tensor.shape[0])
            ):
                runtime_imports.add("_align_tensor_to_target_shape")
                public_output_expr = (
                    f"_align_tensor_to_target_shape("
                    f"{raw_output_var}.permute(0, 2, 3, 1).contiguous(), "
                    f"{target_shape_literal_fn(output_name)})"
                )
            forward_lines.append(f"{output_vars[0]} = {public_output_expr}")
    elif used_direct_module_call and raw_output_var != output_vars[0]:
        if fused != "NONE":
            forward_lines[-1] = forward_lines[-1].replace(
                f"{output_vars[0]} =", f"{raw_output_var} =", 1
            )
        output_expr = emit_module_output_expr_fn(
            output_name=output_name,
            expr=raw_output_var,
            raw_output_layout=raw_output_layout,
        )
        forward_lines.append(f"{output_vars[0]} = {output_expr}")
    return True


def _emit_native_fused_module_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_type: str,
    attr_name: str,
    fused_module_spec: Optional[Dict[str, Any]],
    tensor_var_names: Dict[str, str],
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    tensor_expr_for_channel_first_bridge_fn: Callable[
        [str, Sequence[int]], Optional[str]
    ],
    all_consumers_are_channel_first_binary_ops_fn: Callable[[str], bool],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    derived_local_var_name_fn: Callable[[str, str], str],
    emit_module_output_expr_fn: Callable[..., str],
    target_shape_literal_fn: Callable[[str], str],
) -> bool:
    if fused_module_spec is None:
        return False
    output_name = str(fused_module_spec["output_name"])
    output_var = tensor_var_names[output_name]
    fused_input_name = str(fused_module_spec["input_name"])
    fused_input_expr = tensor_expr_fn(fused_input_name)
    input_pre_permute = fused_module_spec.get("input_pre_permute", None)
    fused_input_tensor = model_ir.tensors.get(fused_input_name, None)
    output_tensor = model_ir.tensors.get(output_name, None)
    fallback_channel_last_conv_bridge = False
    if isinstance(input_pre_permute, list) and len(input_pre_permute) == 4:
        folded_channel_first_expr = tensor_expr_for_channel_first_bridge_fn(
            fused_input_name,
            input_pre_permute,
        )
        if folded_channel_first_expr is not None:
            fused_input_expr = str(folded_channel_first_expr)
        else:
            fused_input_expr = (
                f"{fused_input_expr}.permute("
                f"{', '.join(str(int(value)) for value in input_pre_permute)}"
                ").contiguous()"
            )
    elif (
        op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
        and fused_input_tensor is not None
        and output_tensor is not None
        and is_channel_last_logical_layout(
            normalize_logical_layout(fused_input_tensor.logical_layout)
        )
        and is_channel_last_logical_layout(
            normalize_logical_layout(output_tensor.logical_layout)
        )
    ):
        fused_input_expr = (
            f"{fused_input_expr}.permute(0, 3, 1, 2).contiguous()"
        )
        fallback_channel_last_conv_bridge = True
    raw_output_layout = LOGICAL_LAYOUT_UNKNOWN
    if output_tensor is not None:
        output_rank = len(list(output_tensor.shape))
        if op_type in {
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "TRANSPOSE_CONV",
            "CONV_3D",
        }:
            raw_output_layout = channel_first_logical_layout(output_rank)
    fused_module_expr = f"self.{attr_name}({fused_input_expr})"
    if fallback_channel_last_conv_bridge:
        fused_module_expr = (
            f"{fused_module_expr}.permute(0, 2, 3, 1).contiguous()"
        )
        raw_output_layout = normalize_logical_layout(
            output_tensor.logical_layout
            if output_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
    output_layout = normalize_logical_layout(
        output_tensor.logical_layout
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    normalized_raw_output_layout = normalize_logical_layout(raw_output_layout)
    if output_tensor is not None and is_channel_first_logical_layout(
        normalized_raw_output_layout
    ):
        output_rank = len(list(output_tensor.shape))
        if output_rank in {3, 4, 5}:
            raw_output_var = derived_local_var_name_fn(
                f"{output_var}_cf", "t"
            )
            channel_first_tensor_expr_aliases[output_name] = raw_output_var
            forward_lines.append(f"{raw_output_var} = {fused_module_expr}")
            if output_layout == normalized_raw_output_layout:
                if output_name in model_ir.outputs:
                    weight_tensor = (
                        model_ir.tensors.get(str(op.inputs[1]), None)
                        if len(op.inputs) >= 2
                        else None
                    )
                    if (
                        output_rank == 4
                        and weight_tensor is not None
                        and len(list(weight_tensor.shape)) >= 1
                        and int(output_tensor.shape[-1])
                        == int(weight_tensor.shape[0])
                        and int(output_tensor.shape[1])
                        != int(weight_tensor.shape[0])
                    ):
                        runtime_imports.add("_align_tensor_to_target_shape")
                        forward_lines.append(
                            f"{output_var} = _align_tensor_to_target_shape("
                            f"{raw_output_var}.permute(0, 2, 3, 1)"
                            f".contiguous(), "
                            f"{target_shape_literal_fn(output_name)})"
                        )
                    else:
                        forward_lines.append(f"{output_var} = {raw_output_var}")
                return True
            if is_channel_last_logical_layout(output_layout):
                if all_consumers_are_channel_first_binary_ops_fn(
                    output_name
                ) or can_omit_materialized_channel_last_alias_fn(output_name):
                    return True
                perm_to_output = logical_layout_permutation(
                    source_layout=normalized_raw_output_layout,
                    target_layout=output_layout,
                )
                if perm_to_output is None:
                    raise ModelIRPyTorchExportError(
                        "Native PyTorch-like model.py codegen could not derive "
                        "a layout bridge for fused module output. "
                        f"output={output_name} "
                        f"raw_layout={normalized_raw_output_layout} "
                        f"target_layout={output_layout}"
                    )
                runtime_imports.add("_align_tensor_to_target_shape")
                forward_lines.append(
                    f"{output_var} = _align_tensor_to_target_shape("
                    f"{raw_output_var}.permute("
                    f"{', '.join(str(int(value)) for value in perm_to_output)}"
                    f").contiguous(), {target_shape_literal_fn(output_name)})"
                )
                return True
            output_expr = emit_module_output_expr_fn(
                output_name=output_name,
                expr=raw_output_var,
                raw_output_layout=raw_output_layout,
            )
            forward_lines.append(f"{output_var} = {output_expr}")
            return True
    channel_first_tensor_expr_aliases.pop(output_name, None)
    output_expr = emit_module_output_expr_fn(
        output_name=output_name,
        expr=fused_module_expr,
        raw_output_layout=raw_output_layout,
    )
    forward_lines.append(f"{output_var} = {output_expr}")
    return True


def _emit_native_direct_module_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_index: int,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    op_module_attr_names: Dict[int, str],
    fused_module_specs: Dict[int, Dict[str, Any]],
    conv_module_pad_specs: Dict[int, Optional[List[int]]],
    tensor_var_names: Dict[str, str],
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    tensor_expr_for_channel_first_bridge_fn: Callable[
        [str, Sequence[int]], Optional[str]
    ],
    all_consumers_are_channel_first_binary_ops_fn: Callable[[str], bool],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    derived_local_var_name_fn: Callable[[str, str], str],
    emit_module_output_expr_fn: Callable[..., str],
    target_shape_literal_fn: Callable[[str], str],
    conv2d_input_pre_permute_fn: Callable[..., Optional[List[int]]],
    can_emit_direct_module_call_fn: Callable[[OperatorIR], bool],
    activation_lines_fn: Callable[[str, str], List[str]],
    emit_maybe_aligned_expr_fn: Callable[..., str],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    should_skip_align_for_shape_preserving_unary_fn: Callable[
        [str, str], bool
    ],
) -> bool:
    op_type = str(op.op_type)
    if op_type not in _DIRECT_CODEGEN_MODULE_OP_TYPES:
        return False
    attr_name = op_module_attr_names[int(op_index)]
    fused_module_spec = fused_module_specs.get(int(op_index), None)
    if _emit_native_recurrent_module_op_for_codegen(
        op=op,
        op_type=op_type,
        attr_name=attr_name,
        output_vars=output_vars,
        output_target_shape=output_target_shape,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=tensor_expr_fn,
    ):
        return True
    if _emit_native_fused_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=op_type,
        attr_name=attr_name,
        fused_module_spec=fused_module_spec,
        tensor_var_names=tensor_var_names,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=tensor_expr_fn,
        tensor_expr_for_channel_first_bridge_fn=(
            tensor_expr_for_channel_first_bridge_fn
        ),
        all_consumers_are_channel_first_binary_ops_fn=(
            all_consumers_are_channel_first_binary_ops_fn
        ),
        can_omit_materialized_channel_last_alias_fn=(
            can_omit_materialized_channel_last_alias_fn
        ),
        derived_local_var_name_fn=derived_local_var_name_fn,
        emit_module_output_expr_fn=emit_module_output_expr_fn,
        target_shape_literal_fn=target_shape_literal_fn,
    ):
        return True
    if _emit_native_conv2d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=op_type,
        op_index=op_index,
        attr_name=attr_name,
        outputs=outputs,
        output_vars=output_vars,
        output_target_shape=output_target_shape,
        conv_module_pad_specs=conv_module_pad_specs,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=tensor_expr_fn,
        tensor_expr_for_channel_first_bridge_fn=(
            tensor_expr_for_channel_first_bridge_fn
        ),
        derived_local_var_name_fn=derived_local_var_name_fn,
        emit_module_output_expr_fn=emit_module_output_expr_fn,
        target_shape_literal_fn=target_shape_literal_fn,
        conv2d_input_pre_permute_fn=conv2d_input_pre_permute_fn,
        can_emit_direct_module_call_fn=can_emit_direct_module_call_fn,
        activation_lines_fn=activation_lines_fn,
        tensor_shape_list_fn=tensor_shape_list_fn,
    ):
        return True
    if _emit_native_transpose_conv2d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=op_type,
        attr_name=attr_name,
        outputs=outputs,
        output_vars=output_vars,
        output_target_shape=output_target_shape,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=tensor_expr_fn,
        activation_lines_fn=activation_lines_fn,
    ):
        return True
    if _emit_native_conv3d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=op_type,
        attr_name=attr_name,
        outputs=outputs,
        output_vars=output_vars,
        output_target_shape=output_target_shape,
        channel_first_tensor_expr_aliases=channel_first_tensor_expr_aliases,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=tensor_expr_fn,
        derived_local_var_name_fn=derived_local_var_name_fn,
        emit_module_output_expr_fn=emit_module_output_expr_fn,
        can_emit_direct_module_call_fn=can_emit_direct_module_call_fn,
        activation_lines_fn=activation_lines_fn,
    ):
        return True
    if _emit_native_transpose_conv3d_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=op_type,
        attr_name=attr_name,
        outputs=outputs,
        output_vars=output_vars,
        output_target_shape=output_target_shape,
        runtime_imports=runtime_imports,
        forward_lines=forward_lines,
        tensor_expr_fn=tensor_expr_fn,
        activation_lines_fn=activation_lines_fn,
    ):
        return True
    if _emit_native_fully_connected_module_op_for_codegen(
        op=op,
        op_type=op_type,
        attr_name=attr_name,
        output_vars=output_vars,
        forward_lines=forward_lines,
        tensor_expr_fn=tensor_expr_fn,
        activation_lines_fn=activation_lines_fn,
    ):
        return True
    if _emit_native_prelu_module_op_for_codegen(
        model_ir=model_ir,
        op=op,
        op_type=op_type,
        attr_name=attr_name,
        outputs=outputs,
        output_vars=output_vars,
        forward_lines=forward_lines,
        tensor_expr_fn=tensor_expr_fn,
        emit_maybe_aligned_expr_fn=emit_maybe_aligned_expr_fn,
        tensor_shape_list_fn=tensor_shape_list_fn,
        should_skip_align_for_shape_preserving_unary_fn=(
            should_skip_align_for_shape_preserving_unary_fn
        ),
    ):
        return True
    return False


def _emit_native_shape_transform_misc_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_type: str,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    axis_expr_from_input_fn: Callable[..., str],
) -> bool:
    if op_type == "REVERSE_V2":
        data_expr = tensor_expr_fn(str(op.inputs[0]))
        axes_expr = tensor_expr_fn(str(op.inputs[1]))
        axes_values = _constant_int_list(
            model_ir.tensors.get(str(op.inputs[1]), None)
        )
        data_rank = len(list(model_ir.tensors[str(op.inputs[0])].shape))
        if axes_values is not None:
            normalized_dims = [
                int(axis) if int(axis) >= 0 else int(axis) + int(data_rank)
                for axis in list(axes_values)
            ]
            dims_expr = repr(normalized_dims)
        else:
            dims_expr = (
                f"[int(v) if int(v) >= 0 else int(v) + {data_expr}.ndim "
                f"for v in {axes_expr}.to(dtype=torch.int64).reshape(-1)]"
            )
        forward_lines.append(
            f"{output_vars[0]} = torch.flip("
            f"{data_expr}, "
            f"dims={dims_expr}"
            f")"
        )
        return True
    if op_type == "EXPAND_DIMS":
        axis_expr = (
            axis_expr_from_input_fn(
                str(op.inputs[1]),
                device_expr=tensor_expr_fn(str(op.inputs[0])),
            )
            if len(op.inputs) >= 2
            else repr(int(op.options.get("axis", 0)))
        )
        forward_lines.append(
            f"{output_vars[0]} = torch.unsqueeze("
            f"{tensor_expr_fn(str(op.inputs[0]))}, dim={axis_expr})"
        )
        return True
    if op_type == "SQUEEZE":
        squeeze_dims = [
            int(value) for value in list(op.options.get("squeezeDims", []))
        ]
        if len(squeeze_dims) == 0:
            forward_lines.append(
                f"{output_vars[0]} = torch.squeeze("
                f"{tensor_expr_fn(str(op.inputs[0]))})"
            )
        else:
            runtime_imports.add("_normalize_dim")
            forward_lines.append(
                f"{output_vars[0]} = {tensor_expr_fn(str(op.inputs[0]))}"
            )
            for axis in sorted(squeeze_dims, reverse=True):
                forward_lines.append(
                    f"{output_vars[0]} = torch.squeeze({output_vars[0]}, "
                    f"dim=_normalize_dim({int(axis)}, {output_vars[0]}.ndim))"
                )
        return True
    if op_type == "PACK":
        axis = int(op.options.get("axis", 0))
        inputs_expr = ", ".join(
            tensor_expr_fn(str(name)) for name in op.inputs
        )
        forward_lines.append(
            f"{output_vars[0]} = torch.stack([{inputs_expr}], dim={axis})"
        )
        return True
    if op_type == "UNPACK":
        runtime_imports.add("_normalize_dim")
        axis = int(op.options.get("axis", 0))
        input_expr = tensor_expr_fn(str(op.inputs[0]))
        forward_lines.append(
            f"{', '.join(output_vars)} = list(torch.unbind({input_expr}, "
            f"dim=_normalize_dim({axis}, {input_expr}.ndim)))"
        )
        return True
    if op_type == "SPLIT":
        runtime_imports.add("_normalize_dim")
        data_expr = tensor_expr_fn(str(op.inputs[-1]))
        if len(op.inputs) >= 2:
            axis_expr = axis_expr_from_input_fn(
                str(op.inputs[0]), device_expr=data_expr
            )
        else:
            axis_expr = repr(int(op.options.get("axis", 0)))
        sections = int(op.options.get("numSplits", len(outputs)))
        forward_lines.append(
            f"{', '.join(output_vars)} = list(torch.tensor_split({data_expr}, "
            f"{sections}, dim=_normalize_dim({axis_expr}, {data_expr}.ndim)))"
        )
        return True
    return False


def _emit_native_unary_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    channel_first_passthrough_input_expr_fn: Callable[[str], Optional[str]],
    can_emit_channel_first_shape_preserving_unary_op_fn: Callable[
        [OperatorIR], bool
    ],
    derived_local_var_name_fn: Callable[[str, str], str],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    target_shape_literal_fn: Callable[[str], str],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    should_skip_align_for_shape_preserving_unary_fn: Callable[[str, str], bool],
    emit_maybe_aligned_expr_fn: Callable[..., str],
) -> bool:
    op_type = str(op.op_type)
    if op_type not in _DIRECT_CODEGEN_UNARY_EXPRESSIONS:
        return False
    template = _DIRECT_CODEGEN_UNARY_EXPRESSIONS[op_type]
    input_name = str(op.inputs[0])
    output_name = str(outputs[0])
    channel_first_input_expr = channel_first_passthrough_input_expr_fn(
        input_name
    )
    output_tensor = model_ir.tensors.get(output_name, None)
    output_layout = (
        normalize_logical_layout(output_tensor.logical_layout)
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    if (
        can_emit_channel_first_shape_preserving_unary_op_fn(op)
        and channel_first_input_expr is not None
        and output_tensor is not None
    ):
        if op_type == "LEAKY_RELU":
            channel_first_expr = template.format(
                x=channel_first_input_expr,
                alpha=float(op.options.get("alpha", 0.2)),
            )
        else:
            channel_first_expr = template.format(x=channel_first_input_expr)
        output_rank = len(list(output_tensor.shape))
        if is_channel_first_logical_layout(output_layout):
            channel_first_tensor_expr_aliases.pop(output_name, None)
            forward_lines.append(f"{output_vars[0]} = {channel_first_expr}")
            return True
        raw_output_var = (
            output_vars[0]
            if output_layout == LOGICAL_LAYOUT_UNKNOWN
            else derived_local_var_name_fn(f"{output_vars[0]}_cf", "t")
        )
        channel_first_tensor_expr_aliases[output_name] = raw_output_var
        forward_lines.append(f"{raw_output_var} = {channel_first_expr}")
        if raw_output_var != output_vars[0]:
            if can_omit_materialized_channel_last_alias_fn(output_name):
                return True
            perm_to_output = logical_layout_permutation(
                source_layout=channel_first_logical_layout(output_rank),
                target_layout=output_layout,
            )
            if perm_to_output is None:
                raise ModelIRPyTorchExportError(
                    "Native PyTorch-like model.py codegen could not derive a "
                    f"unary layout bridge. output={output_name} "
                    f"output_layout={output_layout} rank={output_rank}"
                )
            runtime_imports.add("_align_tensor_to_target_shape")
            forward_lines.append(
                f"{output_vars[0]} = _align_tensor_to_target_shape("
                f"{raw_output_var}.permute("
                f"{', '.join(str(int(value)) for value in perm_to_output)}"
                f").contiguous(), {target_shape_literal_fn(output_name)})"
            )
        return True
    if op_type == "LEAKY_RELU":
        expr = template.format(
            x=tensor_expr_fn(str(op.inputs[0])),
            alpha=float(op.options.get("alpha", 0.2)),
        )
    else:
        expr = template.format(x=tensor_expr_fn(str(op.inputs[0])))
    channel_first_tensor_expr_aliases.pop(output_name, None)
    inferred_shape = tensor_shape_list_fn(str(op.inputs[0]))
    if should_skip_align_for_shape_preserving_unary_fn(
        str(op.inputs[0]),
        str(outputs[0]),
    ):
        forward_lines.append(f"{output_vars[0]} = {expr}")
    else:
        aligned_expr = emit_maybe_aligned_expr_fn(
            output_name=outputs[0],
            expr=expr,
            inferred_shape=inferred_shape,
        )
        forward_lines.append(f"{output_vars[0]} = {aligned_expr}")
    return True


def _emit_native_binary_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_index: int,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    runtime_shape_uncertain_tensors: Set[str],
    tensor_dtype_name_fn: Callable[[str], Optional[str]],
    binary_operand_expr_fn: Callable[[str, str], str],
    scalar_literal_expr_fn: Callable[[str], Optional[str]],
    can_emit_channel_first_binary_op_fn: Callable[[OperatorIR], bool],
    channel_first_binary_input_expr_fn: Callable[[str, str], Optional[str]],
    derived_local_var_name_fn: Callable[[str, str], str],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    target_shape_literal_fn: Callable[[str], str],
    emit_maybe_aligned_expr_fn: Callable[..., str],
    binary_runtime_shape_passthrough_operand_fn: Callable[
        [str, str], Optional[str]
    ],
    binary_requires_runtime_alignment_fn: Callable[[str, str, str], bool],
    preferred_binary_alignment_anchor_fn: Callable[
        [str, str, str], Optional[str]
    ],
    activation_lines_fn: Callable[[str, str], List[str]],
    binary_output_target_shape_literal_fn: Callable[..., str],
) -> bool:
    op_type = str(op.op_type)
    if op_type not in _DIRECT_CODEGEN_BINARY_FUNCTIONS:
        return False
    fn_name = _DIRECT_CODEGEN_BINARY_FUNCTIONS[op_type]
    fused = str(op.options.get("fusedActivationFunction", "NONE"))
    lhs_name = str(op.inputs[0])
    rhs_name = str(op.inputs[1])
    output_name = str(outputs[0])
    resolved_output_target_shape = binary_output_target_shape_literal_fn(
        lhs_name=lhs_name,
        rhs_name=rhs_name,
        output_name=output_name,
        fallback_literal=output_target_shape,
    )
    lhs_dtype_name = tensor_dtype_name_fn(lhs_name)
    rhs_dtype_name = tensor_dtype_name_fn(rhs_name)
    lhs_expr = binary_operand_expr_fn(lhs_name, rhs_name)
    rhs_scalar_literal = scalar_literal_expr_fn(rhs_name)
    if op_type in {"MAXIMUM", "MINIMUM"}:
        rhs_scalar_literal = None
    rhs_expr = rhs_scalar_literal or binary_operand_expr_fn(rhs_name, lhs_name)
    if (
        rhs_scalar_literal in {"True", "False"}
        and op_type in {"EQUAL", "NOT_EQUAL"}
    ):
        rhs_expr = (
            f"torch.as_tensor({rhs_scalar_literal}, "
            f"dtype={lhs_expr}.dtype, device={lhs_expr}.device)"
        )
    is_integer_div = (
        op_type == "DIV"
        and lhs_dtype_name in {"INT8", "INT16", "INT32", "INT64", "UINT8"}
        and rhs_dtype_name in {"INT8", "INT16", "INT32", "INT64", "UINT8"}
    )
    runtime_shape_passthrough_operand = (
        binary_runtime_shape_passthrough_operand_fn(lhs_name, rhs_name)
    )
    requires_runtime_alignment = binary_requires_runtime_alignment_fn(
        lhs_name, rhs_name, str(outputs[0])
    )

    def _binary_expr(left_expr: str, right_expr: str) -> str:
        if is_integer_div:
            return (
                f"torch.div({left_expr}, {right_expr}, "
                "rounding_mode='trunc')"
            )
        return f"{fn_name}({left_expr}, {right_expr})"

    if (
        can_emit_channel_first_binary_op_fn(op)
        and runtime_shape_passthrough_operand is None
    ):
        output_tensor = model_ir.tensors.get(output_name, None)
        output_var = output_vars[0]
        output_rank = (
            len(list(output_tensor.shape)) if output_tensor is not None else 0
        )
        output_layout = normalize_logical_layout(
            output_tensor.logical_layout
            if output_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
        raw_output_var = (
            output_var
            if output_layout == LOGICAL_LAYOUT_UNKNOWN
            or output_name in set(model_ir.outputs)
            else derived_local_var_name_fn(f"{output_var}_cf", "t")
        )
        lhs_cf_expr = channel_first_binary_input_expr_fn(lhs_name, rhs_name)
        rhs_cf_expr = channel_first_binary_input_expr_fn(rhs_name, lhs_name)
        if lhs_cf_expr is None or rhs_cf_expr is None:
            raise ModelIRPyTorchExportError(
                "Native PyTorch-like model.py codegen expected "
                "channel-first-capable binary inputs. "
                f"op={op_type} lhs={lhs_name} rhs={rhs_name}"
            )
        channel_first_tensor_expr_aliases[output_name] = raw_output_var
        forward_lines.append(
            f"{raw_output_var} = {_binary_expr(lhs_cf_expr, rhs_cf_expr)}"
        )
        forward_lines.extend(activation_lines_fn(raw_output_var, fused))
        if (
            output_tensor is not None
            and is_channel_last_logical_layout(output_layout)
            and output_rank in {3, 4, 5}
        ):
            if can_omit_materialized_channel_last_alias_fn(output_name):
                return True
            perm_to_output = logical_layout_permutation(
                source_layout=channel_first_logical_layout(output_rank),
                target_layout=output_layout,
            )
            if perm_to_output is not None:
                runtime_imports.add("_align_tensor_to_target_shape")
                forward_lines.append(
                    f"{output_var} = _align_tensor_to_target_shape("
                    f"{raw_output_var}.permute("
                    f"{', '.join(str(int(value)) for value in perm_to_output)}"
                    f").contiguous(), {target_shape_literal_fn(output_name)})"
                )
                return True
        return True
    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
    if rhs_scalar_literal is not None:
        aligned_expr = emit_maybe_aligned_expr_fn(
            output_name=outputs[0],
            expr=_binary_expr(lhs_expr, rhs_expr),
            inferred_shape=None,
        )
        forward_lines.append(f"{output_vars[0]} = {aligned_expr}")
    elif runtime_shape_passthrough_operand is not None:
        forward_lines.append(
            f"{output_vars[0]} = {_binary_expr(lhs_expr, rhs_expr)}"
        )
    elif requires_runtime_alignment:
        lhs_uncertain = lhs_name in runtime_shape_uncertain_tensors
        rhs_uncertain = rhs_name in runtime_shape_uncertain_tensors
        preferred_anchor = preferred_binary_alignment_anchor_fn(
            lhs_name, rhs_name, str(outputs[0])
        )
        lhs_var = f"_binary_lhs_{op_index}"
        rhs_var = f"_binary_rhs_{op_index}"
        if lhs_uncertain ^ rhs_uncertain or preferred_anchor is not None:
            runtime_imports.add("_align_binary_inputs_to_anchor")
            if lhs_uncertain or preferred_anchor == "lhs":
                forward_lines.append(
                    f"{lhs_var}, {rhs_var} = _align_binary_inputs_to_anchor("
                    f"{lhs_expr}, {binary_operand_expr_fn(rhs_name, lhs_name)}, "
                    f"{resolved_output_target_shape})"
                )
            else:
                forward_lines.append(
                    f"{rhs_var}, {lhs_var} = _align_binary_inputs_to_anchor("
                    f"{binary_operand_expr_fn(rhs_name, lhs_name)}, {lhs_expr}, "
                    f"{resolved_output_target_shape})"
                )
        else:
            runtime_imports.add("_align_binary_inputs")
            forward_lines.append(
                f"{lhs_var}, {rhs_var} = _align_binary_inputs("
                f"{lhs_expr}, {binary_operand_expr_fn(rhs_name, lhs_name)}, "
                f"{resolved_output_target_shape})"
            )
        aligned_expr = emit_maybe_aligned_expr_fn(
            output_name=outputs[0],
            expr=_binary_expr(lhs_var, rhs_var),
            inferred_shape=None,
        )
        forward_lines.append(f"{output_vars[0]} = {aligned_expr}")
    else:
        aligned_expr = emit_maybe_aligned_expr_fn(
            output_name=outputs[0],
            expr=_binary_expr(lhs_expr, rhs_expr),
            inferred_shape=None,
        )
        forward_lines.append(f"{output_vars[0]} = {aligned_expr}")
    forward_lines.extend(activation_lines_fn(output_vars[0], fused))
    return True


def _emit_native_transpose_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    preserve_channel_last_tensor_names: Set[str],
    consumer_index: Dict[str, List[int]],
    producer_index: Dict[str, int],
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    tensor_expr_for_channel_first_bridge_fn: Callable[
        [str, Sequence[int]], Optional[str]
    ],
    can_fold_channel_last_alias_slice_consumer_fn: Callable[..., bool],
    all_consumers_are_channel_first_binary_ops_fn: Callable[[str], bool],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    has_channel_last_consumer_hint_for_same_shape_transpose_fn: Callable[
        [OperatorIR], bool
    ],
    is_batchless_rank3_public_output_transpose_fn: Callable[
        [OperatorIR], bool
    ],
    target_shape_literal_fn: Callable[[str], str],
) -> bool:
    if str(op.op_type) != "TRANSPOSE":
        return False
    transpose_related_names = {
        str(value) for value in list(op.inputs) + list(op.outputs)
    }
    stale_channel_last_transpose = (
        has_channel_last_consumer_hint_for_same_shape_transpose_fn(op)
    )
    batchless_public_output_transpose = (
        is_batchless_rank3_public_output_transpose_fn(op)
    )
    transpose_perm = _read_transpose_perm(model_ir, op)
    transpose_input_name = str(op.inputs[0]) if len(op.inputs) >= 1 else ""
    transpose_output_name = str(outputs[0]) if len(outputs) == 1 else ""
    transpose_input_tensor = model_ir.tensors.get(transpose_input_name, None)
    transpose_output_tensor = model_ir.tensors.get(transpose_output_name, None)
    transpose_consumer_indices = [
        int(value) for value in consumer_index.get(transpose_output_name, [])
    ]
    reshape_only_consumers = len(transpose_consumer_indices) > 0 and all(
        str(model_ir.operators[int(consumer_idx)].op_type) == "RESHAPE"
        for consumer_idx in transpose_consumer_indices
    )
    allow_transpose_elision = stale_channel_last_transpose or not any(
        name in preserve_channel_last_tensor_names
        for name in transpose_related_names
    )
    if batchless_public_output_transpose or allow_transpose_elision and (
        stale_channel_last_transpose
        or _is_reshape_only_residual_layout_bridge_transpose(
            model_ir=model_ir,
            op=op,
            consumers=consumer_index,
        )
        or _is_inconsistent_standard_layout_transpose(
            input_tensor=(
                model_ir.tensors.get(str(op.inputs[0]), None)
                if len(op.inputs) >= 1
                else None
            ),
            output_tensor=(
                model_ir.tensors.get(str(outputs[0]), None)
                if len(outputs) == 1
                else None
            ),
            perm=transpose_perm,
        )
        or _is_inconsistent_same_layout_transpose(
            input_tensor=(
                model_ir.tensors.get(str(op.inputs[0]), None)
                if len(op.inputs) >= 1
                else None
            ),
            output_tensor=(
                model_ir.tensors.get(str(outputs[0]), None)
                if len(outputs) == 1
                else None
            ),
            perm=transpose_perm,
        )
        and not reshape_only_consumers
    ):
        forward_lines.append(
            f"{output_vars[0]} = {tensor_expr_fn(str(op.inputs[0]))}"
        )
        return True
    folded_channel_first_expr = (
        None
        if transpose_perm is None
        else tensor_expr_for_channel_first_bridge_fn(
            transpose_input_name,
            transpose_perm,
        )
    )
    if folded_channel_first_expr is not None:
        output_layout = normalize_logical_layout(
            transpose_output_tensor.logical_layout
            if transpose_output_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
        if (
            transpose_output_tensor is not None
            and output_layout == LOGICAL_LAYOUT_UNKNOWN
        ):
            channel_first_tensor_expr_aliases[transpose_output_name] = (
                output_vars[0]
            )
        else:
            channel_first_tensor_expr_aliases.pop(transpose_output_name, None)
        forward_lines.append(f"{output_vars[0]} = {folded_channel_first_expr}")
        return True
    if transpose_input_tensor is not None and transpose_output_tensor is not None:
        rank = len(list(transpose_input_tensor.shape))
        expected_cf_to_cl_perm = _perm_cf_to_cl(rank)
        input_cf_expr = channel_first_tensor_expr_aliases.get(
            transpose_input_name, None
        )
        if input_cf_expr is None and is_channel_first_logical_layout(
            normalize_logical_layout(transpose_input_tensor.logical_layout)
        ):
            input_cf_expr = tensor_expr_fn(transpose_input_name)
        if (
            input_cf_expr is not None
            and expected_cf_to_cl_perm is not None
            and list(transpose_perm or []) == list(expected_cf_to_cl_perm)
            and len(consumer_index.get(transpose_output_name, [])) > 0
            and all(
                can_fold_channel_last_alias_slice_consumer_fn(
                    model_ir.operators[int(consumer_idx)],
                    expected_input_name=transpose_output_name,
                )
                for consumer_idx in consumer_index.get(
                    transpose_output_name, []
                )
            )
        ):
            channel_first_tensor_expr_aliases[transpose_output_name] = str(
                input_cf_expr
            )
            return True
        if (
            input_cf_expr is not None
            and expected_cf_to_cl_perm is not None
            and list(transpose_perm or []) == list(expected_cf_to_cl_perm)
            and all_consumers_are_channel_first_binary_ops_fn(
                transpose_output_name
            )
        ):
            channel_first_tensor_expr_aliases[transpose_output_name] = str(
                input_cf_expr
            )
            return True
        if (
            input_cf_expr is not None
            and expected_cf_to_cl_perm is not None
            and list(transpose_perm or []) == list(expected_cf_to_cl_perm)
            and can_omit_materialized_channel_last_alias_fn(
                transpose_output_name
            )
        ):
            channel_first_tensor_expr_aliases[transpose_output_name] = str(
                input_cf_expr
            )
            return True
    runtime_imports.add("_shape_list")
    runtime_imports.add("_torch_permute")
    if len(op.inputs) >= 2:
        const_perm_values = _constant_int_list(
            model_ir.tensors.get(str(op.inputs[1]), None)
        )
        if const_perm_values is not None:
            perm_values = [int(value) for value in list(const_perm_values)]
            input_layout = normalize_logical_layout(
                transpose_input_tensor.logical_layout
                if transpose_input_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            output_layout = normalize_logical_layout(
                transpose_output_tensor.logical_layout
                if transpose_output_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            if (
                reshape_only_consumers
                and input_layout != LOGICAL_LAYOUT_UNKNOWN
                and input_layout == output_layout
            ):
                forward_lines.append(
                    f"{output_vars[0]} = {tensor_expr_fn(str(op.inputs[0]))}"
                    f".permute("
                    f"{', '.join(str(int(value)) for value in perm_values)}"
                    ").contiguous()"
                )
                return True
            perm_expr = repr(perm_values)
        else:
            perm_expr = (
                f"_shape_list({tensor_expr_fn(str(op.inputs[1]))})"
            )
    else:
        perm_expr = repr(
            [int(value) for value in list(op.options.get("perm", []))]
        )
    forward_lines.append(
        f"{output_vars[0]} = _torch_permute("
        f"{tensor_expr_fn(str(op.inputs[0]))}, {perm_expr})"
    )
    return True


def _concat_channel_first_codegen_breaks_channel_last_consumers_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
) -> bool:
    if str(op.op_type) != "CONCATENATION" or len(op.outputs) != 1:
        return False
    output_name = str(op.outputs[0])
    output_tensor = model_ir.tensors.get(output_name, None)
    if output_tensor is None:
        return False
    output_shape = [int(value) for value in list(output_tensor.shape)]
    output_rank = len(output_shape)
    if output_rank not in {3, 4, 5}:
        return False
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    output_looks_channel_last = is_channel_last_logical_layout(output_layout) or (
        output_layout == LOGICAL_LAYOUT_UNKNOWN
        and _tensor_name_suggests_channel_last_layout_for_codegen(output_name)
    )
    if not output_looks_channel_last:
        return False
    channel_axis = output_rank - 1
    for consumer_op in model_ir.operators:
        consumer_type = str(consumer_op.op_type)
        if consumer_type == "GATHER":
            if (
                len(consumer_op.inputs) < 2
                or str(consumer_op.inputs[0]) != output_name
            ):
                continue
            axis = int(consumer_op.options.get("axis", 0))
            if axis < 0:
                axis += output_rank
            if axis == channel_axis:
                return True
            continue
        if consumer_type == "SPLIT":
            if (
                len(consumer_op.inputs) == 0
                or str(consumer_op.inputs[-1]) != output_name
            ):
                continue
            axis = int(consumer_op.options.get("axis", 0))
            if len(consumer_op.inputs) >= 2:
                axis_values = _constant_int_list(
                    model_ir.tensors.get(str(consumer_op.inputs[0]), None)
                )
                if axis_values is not None and len(axis_values) == 1:
                    axis = int(axis_values[0])
            if axis < 0:
                axis += output_rank
            if axis == channel_axis:
                return True
            continue
        if consumer_type == "UNPACK":
            if (
                len(consumer_op.inputs) == 0
                or str(consumer_op.inputs[0]) != output_name
            ):
                continue
            axis = int(consumer_op.options.get("axis", 0))
            if axis < 0:
                axis += output_rank
            if axis == channel_axis:
                return True
    return False


def _emit_native_concat_op_for_codegen(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    op_index: int,
    outputs: Sequence[str],
    output_vars: Sequence[str],
    output_target_shape: str,
    channel_first_tensor_expr_aliases: Dict[str, str],
    runtime_imports: Set[str],
    forward_lines: List[str],
    tensor_expr_fn: Callable[[str], str],
    derived_local_var_name_fn: Callable[[str, str], str],
    activation_lines_fn: Callable[[str, str], List[str]],
    resolve_concat_axis_for_channel_first_fn: Callable[
        [OperatorIR], Optional[Tuple[int, List[int], List[int]]]
    ],
    channel_first_concat_input_expr_fn: Callable[[str], Optional[str]],
    tensor_shape_list_fn: Callable[[str], Optional[List[int]]],
    can_omit_materialized_channel_last_alias_fn: Callable[[str], bool],
    target_shape_literal_fn: Callable[[str], str],
    tensor_exact_static_shape_list_fn: Callable[[str], Optional[List[int]]],
    target_shape_values_fn: Callable[[str], Optional[List[int]]],
) -> bool:
    if str(op.op_type) != "CONCATENATION":
        return False
    axis = int(op.options.get("axis", 0))
    gather_elements_axis_coord_input_index = next(
        (
            index
            for index, name in enumerate(op.inputs)
            if str(name).endswith("_gather_elements_axis_coord")
        ),
        None,
    )
    gather_elements_coords_concat = bool(
        str(outputs[0]).endswith("_gather_elements_coords")
        and gather_elements_axis_coord_input_index is not None
        and axis == len(op.inputs)
    )
    if gather_elements_coords_concat:
        assert gather_elements_axis_coord_input_index is not None
        axis_coord_expr = tensor_expr_fn(
            str(op.inputs[gather_elements_axis_coord_input_index])
        )
        coord_shape_var = f"_gather_elements_coords_shape_{op_index}"
        coord_vars: List[str] = []
        forward_lines.append(
            f"{coord_shape_var} = "
            f"[int(v) for v in list({axis_coord_expr}.shape[:-1])]"
        )
        for dim_index in range(len(op.inputs)):
            if dim_index == gather_elements_axis_coord_input_index:
                coord_vars.append(axis_coord_expr)
                continue
            coord_var = f"_gather_elements_coord_{op_index}_{dim_index}"
            view_shape_var = (
                f"_gather_elements_coord_view_shape_{op_index}_{dim_index}"
            )
            forward_lines.append(
                f"{view_shape_var} = [1] * {len(op.inputs)}"
            )
            forward_lines.append(
                f"{view_shape_var}[{dim_index}] = "
                f"{coord_shape_var}[{dim_index}]"
            )
            forward_lines.append(
                f"{coord_var} = torch.arange("
                f"{coord_shape_var}[{dim_index}], "
                f"dtype={axis_coord_expr}.dtype, "
                f"device={axis_coord_expr}.device)"
                f".reshape(*{view_shape_var}, 1)"
                f".expand(*{coord_shape_var}, 1)"
            )
            coord_vars.append(coord_var)
        forward_lines.append(
            f"{output_vars[0]} = torch.cat([{', '.join(coord_vars)}], "
            f"dim={len(op.inputs)})"
        )
        return True
    concat_cf_spec = resolve_concat_axis_for_channel_first_fn(op)
    concat_cf_inputs = [
        channel_first_concat_input_expr_fn(str(name)) for name in op.inputs
    ]
    output_tensor = model_ir.tensors.get(outputs[0], None)
    output_layout = (
        normalize_logical_layout(output_tensor.logical_layout)
        if output_tensor is not None
        else LOGICAL_LAYOUT_UNKNOWN
    )
    output_rank = (
        len(list(output_tensor.shape)) if output_tensor is not None else 0
    )
    has_channel_last_axis_sensitive_consumer = (
        _concat_channel_first_codegen_breaks_channel_last_consumers_for_codegen(
            model_ir=model_ir,
            op=op,
        )
    )
    if (
        concat_cf_spec is not None
        and output_rank in {4, 5}
        and output_layout
        in {
            LOGICAL_LAYOUT_UNKNOWN,
            channel_first_logical_layout(output_rank),
            channel_last_logical_layout(output_rank),
        }
        and not has_channel_last_axis_sensitive_consumer
        and all(input_expr is not None for input_expr in concat_cf_inputs)
    ):
        concat_cf_axis, concat_cf_output_shape, concat_perm_from_cf = (
            concat_cf_spec
        )
        fused = str(op.options.get("fusedActivationFunction", "NONE"))
        stored_output_shape = tensor_shape_list_fn(outputs[0]) or []
        raw_matches_stored_shape = [
            int(value) for value in list(concat_cf_output_shape)
        ] == [int(value) for value in list(stored_output_shape)]
        needs_materialized_output_bridge = (
            len(concat_perm_from_cf) == output_rank
            and [int(value) for value in list(concat_perm_from_cf)]
            != [int(value) for value in list(range(output_rank))]
            and not raw_matches_stored_shape
        )
        if (
            is_channel_first_logical_layout(output_layout)
            and raw_matches_stored_shape
        ):
            raw_output_var = output_vars[0]
            channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
        elif (
            output_layout == LOGICAL_LAYOUT_UNKNOWN
            and not needs_materialized_output_bridge
        ):
            raw_output_var = output_vars[0]
            channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
        else:
            raw_output_var = derived_local_var_name_fn(
                f"{output_vars[0]}_cf", "t"
            )
            channel_first_tensor_expr_aliases[str(outputs[0])] = raw_output_var
        forward_lines.append(
            f"{raw_output_var} = torch.cat("
            f"[{', '.join(str(value) for value in concat_cf_inputs)}], "
            f"dim={int(concat_cf_axis)})"
        )
        forward_lines.extend(activation_lines_fn(raw_output_var, fused))
        if raw_output_var != output_vars[0]:
            if can_omit_materialized_channel_last_alias_fn(outputs[0]):
                return True
            if needs_materialized_output_bridge:
                runtime_imports.add("_align_tensor_to_target_shape")
                forward_lines.append(
                    f"{output_vars[0]} = _align_tensor_to_target_shape("
                    f"{raw_output_var}.permute("
                    f"{', '.join(str(int(value)) for value in concat_perm_from_cf)}"
                    f").contiguous(), {target_shape_literal_fn(outputs[0])})"
                )
            else:
                perm_to_output = logical_layout_permutation(
                    source_layout=channel_first_logical_layout(output_rank),
                    target_layout=output_layout,
                )
                if perm_to_output is None:
                    raise ModelIRPyTorchExportError(
                        "Native PyTorch-like model.py codegen could not derive "
                        "a channel-first concat bridge. "
                        f"output={outputs[0]} output_layout={output_layout} "
                        f"rank={output_rank}"
                    )
                runtime_imports.add("_align_tensor_to_target_shape")
                forward_lines.append(
                    f"{output_vars[0]} = _align_tensor_to_target_shape("
                    f"{raw_output_var}.permute("
                    f"{', '.join(str(int(value)) for value in perm_to_output)}"
                    f").contiguous(), {target_shape_literal_fn(outputs[0])})"
                )
        return True
    def _is_inlined_scalar_constant(tensor_name: str) -> bool:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        return bool(
            tensor is not None
            and tensor.data is not None
            and int(np.asarray(tensor.data).ndim) == 0
        )

    concat_anchor_name = next(
        (
            str(name)
            for name in op.inputs
            if not _is_inlined_scalar_constant(str(name))
        ),
        None,
    )
    concat_anchor_expr = (
        tensor_expr_fn(concat_anchor_name)
        if concat_anchor_name is not None
        else None
    )
    input_exprs: List[str] = []
    for name in op.inputs:
        input_name = str(name)
        input_expr = tensor_expr_fn(input_name)
        if _is_inlined_scalar_constant(input_name):
            if concat_anchor_expr is None:
                input_expr = f"torch.as_tensor({input_expr})"
            else:
                input_expr = (
                    f"torch.as_tensor({input_expr}, "
                    f"dtype={concat_anchor_expr}.dtype, "
                    f"device={concat_anchor_expr}.device)"
                )
        input_exprs.append(input_expr)
    inputs_expr = ", ".join(input_exprs)
    runtime_imports.add("_apply_concat")
    concat_expr = (
        f"_apply_concat([{inputs_expr}], axis={axis}, "
        f"target_shape={output_target_shape}, "
        f"fused={str(op.options.get('fusedActivationFunction', 'NONE'))!r})"
    )
    exact_output_shape = tensor_exact_static_shape_list_fn(outputs[0])
    target_output_shape = target_shape_values_fn(outputs[0])
    if exact_output_shape is not None and (
        target_output_shape is None
        or [int(value) for value in list(exact_output_shape)]
        != [int(value) for value in list(target_output_shape)]
    ):
        concat_expr = f"torch.reshape({concat_expr}, {repr(exact_output_shape)})"
    channel_first_tensor_expr_aliases.pop(str(outputs[0]), None)
    forward_lines.append(f"{output_vars[0]} = {concat_expr}")
    return True
