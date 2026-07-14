from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    is_channel_first_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_values import (
    _python_literal_for_constant_tensor,
    _torch_dtype_literal,
)
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
)
from onnx2tf.tflite_builder.pytorch_graph_policy import (
    _base_target_shape_values_for_model_ir,
    _channel_first_shape_values_for_model_ir,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import _perm_cl_to_cf
from onnx2tf.tflite_builder.pytorch_naming import (
    _make_unique_identifier,
    _shorten_generated_python_identifier,
)


def _tensor_dtype_name_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[str]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    return str(tensor.dtype).upper()


def _tensor_expr_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    tensor_expr_aliases: Dict[str, str],
    channel_first_tensor_expr_aliases: Dict[str, str],
    buffer_attr_names: Dict[str, str],
    runtime_imports: Set[str],
    tensor_var_names: Dict[str, str],
    tensor_name: str,
) -> str:
    if str(tensor_name) in tensor_expr_aliases:
        return str(tensor_expr_aliases[str(tensor_name)])
    tensor = model_ir.tensors.get(str(tensor_name), None)
    channel_first_alias_expr = channel_first_tensor_expr_aliases.get(
        str(tensor_name), None
    )
    if channel_first_alias_expr is not None and tensor is not None:
        tensor_layout = normalize_logical_layout(tensor.logical_layout)
        if is_channel_first_logical_layout(tensor_layout):
            return str(channel_first_alias_expr)
        base_target_shape = _base_target_shape_values_for_model_ir(
            model_ir=model_ir,
            tensor_name=str(tensor_name),
        )
        channel_first_shape = _channel_first_shape_values_for_model_ir(
            model_ir=model_ir,
            tensor_name=str(tensor_name),
        )
        if (
            base_target_shape is not None
            and channel_first_shape is not None
            and [int(v) for v in list(base_target_shape)]
            == [int(v) for v in list(channel_first_shape)]
        ):
            return str(channel_first_alias_expr)
    if (
        channel_first_alias_expr is not None
        and tensor is not None
        and is_channel_first_logical_layout(
            normalize_logical_layout(tensor.logical_layout)
        )
    ):
        return str(channel_first_alias_expr)
    if (
        tensor is not None
        and str(tensor_name) not in model_ir.inputs
        and str(tensor_name) not in producer_index
    ):
        if str(tensor_name) in buffer_attr_names:
            return f"self.{buffer_attr_names[str(tensor_name)]}"
        literal = _python_literal_for_constant_tensor(tensor)
        if literal is not None:
            runtime_imports.add("_module_device")
            return (
                f"torch.as_tensor({literal}, dtype={_torch_dtype_literal(str(tensor.dtype).upper())}, "
                "device=_module_device(self))"
            )
    if str(tensor_name) in tensor_var_names:
        return str(tensor_var_names[str(tensor_name)])
    if str(tensor_name) in buffer_attr_names:
        return f"self.{buffer_attr_names[str(tensor_name)]}"
    literal = (
        _python_literal_for_constant_tensor(tensor) if tensor is not None else None
    )
    if tensor is not None and literal is not None:
        runtime_imports.add("_module_device")
        return (
            f"torch.as_tensor({literal}, dtype={_torch_dtype_literal(str(tensor.dtype).upper())}, "
            "device=_module_device(self))"
        )
    raise ModelIRPyTorchExportError(
        "Native PyTorch-like model.py codegen could not resolve a tensor expression. "
        f"tensor={tensor_name}"
    )


def _derived_local_var_name_for_codegen(
    *,
    synthetic_local_var_names: Dict[str, str],
    used_local_var_names: Set[str],
    base_name: str,
    prefix: str = "t",
) -> str:
    cached_name = synthetic_local_var_names.get(str(base_name), None)
    if cached_name is not None:
        return str(cached_name)
    shortened_name = _shorten_generated_python_identifier(
        str(base_name),
        prefix=prefix,
    )
    unique_name = _make_unique_identifier(shortened_name, used_local_var_names)
    synthetic_local_var_names[str(base_name)] = str(unique_name)
    return str(unique_name)


def _channel_first_constant_expr_for_buffer_attr_for_codegen(
    *,
    buffer_attr_name_to_tensor_name: Dict[str, str],
    channel_first_constant_buffer_alias_exprs: Dict[str, str],
    channel_first_rank4_constant_buffer_alias_shape_fn: Callable[
        [str], Optional[List[int]]
    ],
    buffer_expr: str,
    target_shape: Sequence[int],
) -> Optional[str]:
    if not str(buffer_expr).startswith("self."):
        return None
    attr_name = str(buffer_expr)[5:]
    tensor_name = buffer_attr_name_to_tensor_name.get(str(attr_name), None)
    if tensor_name is None:
        return None
    alias_expr = channel_first_constant_buffer_alias_exprs.get(
        str(tensor_name), None
    )
    alias_shape = channel_first_rank4_constant_buffer_alias_shape_fn(
        str(tensor_name)
    )
    if alias_expr is None or alias_shape is None:
        return None
    if [int(v) for v in alias_shape] != [int(v) for v in list(target_shape)]:
        return None
    return str(alias_expr)


def _permuted_constant_expr_for_tensor_name_for_codegen(
    *,
    permuted_constant_buffer_alias_exprs: Dict[
        Tuple[str, Tuple[int, ...]], str
    ],
    tensor_name: str,
    perm: Sequence[int],
) -> Optional[str]:
    return permuted_constant_buffer_alias_exprs.get(
        (str(tensor_name), tuple(int(v) for v in list(perm))),
        None,
    )


def _transposed_constant_expr_for_tensor_name_for_codegen(
    *,
    transposed_constant_buffer_alias_exprs: Dict[str, str],
    tensor_name: str,
) -> Optional[str]:
    return transposed_constant_buffer_alias_exprs.get(str(tensor_name), None)


def _tensor_expr_for_channel_first_bridge_for_codegen(
    *,
    model_ir: ModelIR,
    channel_first_tensor_expr_aliases: Dict[str, str],
    tensor_name: str,
    perm: Optional[Sequence[int]],
) -> Optional[str]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None:
        return None
    rank = len(list(tensor.shape))
    if rank == 3:
        return None
    expected_perm = _perm_cl_to_cf(rank)
    if expected_perm is None or list(perm or []) != list(expected_perm):
        return None
    return channel_first_tensor_expr_aliases.get(str(tensor_name), None)
