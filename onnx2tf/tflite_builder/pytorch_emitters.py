from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    channel_first_logical_layout,
    is_channel_first_logical_layout,
    logical_layout_permutation,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import _constant_int_list


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
