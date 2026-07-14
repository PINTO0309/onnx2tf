from __future__ import annotations

import ast
import re
from typing import Dict, List, Optional, Sequence, Tuple

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    channel_last_logical_layout,
    is_channel_first_logical_layout,
    logical_layout_permutation,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import _shape_lists_equal
from onnx2tf.tflite_builder.pytorch_layout_utils import _permute_shape
from onnx2tf.tflite_builder.pytorch_source_parser import (
    _normalize_permute_dims_expr,
    _parse_simple_assignment_line,
    _split_top_level_csv_exprs,
)


def _infer_gather_nd_shape_for_codegen(
    *,
    model_ir: ModelIR,
    params_shape: Optional[Sequence[int]],
    indices_tensor_name: str,
) -> Optional[List[int]]:
    if params_shape is None:
        return None
    indices_tensor = model_ir.tensors.get(str(indices_tensor_name), None)
    if indices_tensor is None:
        return None
    indices_shape = [int(v) for v in list(indices_tensor.shape)]
    if len(indices_shape) == 0:
        return None
    index_depth = int(indices_shape[-1])
    params_items = [int(v) for v in list(params_shape)]
    if index_depth > len(params_items):
        return None
    return indices_shape[:-1] + params_items[index_depth:]


def _bridge_boundary_metadata_gather_nd_inputs(
    lines: Sequence[str],
    *,
    model_ir: ModelIR,
    tensor_var_names: Dict[str, str],
) -> List[str]:
    def _parse_torch_permute_expr(
        expr: str,
    ) -> Tuple[str, List[int]] | None:
        stripped = expr.strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args: str) -> Tuple[str, List[int]] | None:
            parts = _split_top_level_csv_exprs(str(args))
            input_expr: str | None = None
            perm_expr: str | None = None
            if len(parts) == 2 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
            ):
                input_expr, perm_expr = parts[0].strip(), parts[1].strip()
            else:
                kwargs: Dict[str, str] = {}
                for part in parts:
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                        continue
                    key, value = part.split("=", 1)
                    kwargs[key.strip()] = value.strip()
                input_expr = kwargs.get("input", kwargs.get("x"))
                perm_expr = kwargs.get("perm", kwargs.get("dims"))
            if (
                input_expr is None
                or perm_expr is None
                or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None
            ):
                return None
            try:
                perm_value = ast.literal_eval(perm_expr)
            except Exception:
                return None
            if not isinstance(perm_value, (list, tuple)):
                return None
            try:
                perm = [int(v) for v in list(perm_value)]
            except Exception:
                return None
            return input_expr, perm

        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                return _parse_permute_like_args(stripped[len(prefix) : -1])
        method_match = re.fullmatch(
            r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped
        )
        if method_match is None:
            return None
        try:
            perm = [
                int(v)
                for v in _normalize_permute_dims_expr(
                    str(method_match.group("dims"))
                ).split(",")
            ]
        except Exception:
            return None
        return str(method_match.group("input")), perm

    def _parse_gather_nd_assign(
        line: str,
    ) -> Tuple[str, str, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        stripped = rhs.strip()
        prefix = "_apply_gather_nd("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        if len(parts) != 3:
            return None
        input_expr = parts[0].strip()
        indices_expr = parts[1].strip()
        target_part = parts[2].strip()
        if not target_part.startswith("target_shape="):
            return None
        return (
            lhs,
            input_expr,
            indices_expr,
            target_part[len("target_shape=") :].strip(),
        )

    def _parse_redundant_double_permute_expr(
        expr: str,
    ) -> Tuple[str, List[int], List[int]] | None:
        match = re.fullmatch(
            r"(?P<base>.+)\.permute\((?P<perm_args>[^\)]+)\)\.contiguous\(\)",
            expr.strip(),
        )
        if match is None:
            return None
        base = str(match.group("base"))
        permute_match = _parse_torch_permute_expr(base)
        if permute_match is None:
            return None
        try:
            perm_args = [
                int(v.strip()) for v in str(match.group("perm_args")).split(",")
            ]
        except Exception:
            return None
        return permute_match[0], permute_match[1], perm_args

    gather_nd_input_perms: Dict[str, List[int]] = {}
    for op in model_ir.operators:
        if str(op.op_type) != "GATHER_ND" or len(op.inputs) < 2 or len(op.outputs) != 1:
            continue
        params_name = str(op.inputs[0])
        if params_name not in model_ir.inputs:
            continue
        params_tensor = model_ir.tensors.get(params_name, None)
        output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if params_tensor is None or output_tensor is None:
            continue
        params_shape = [int(v) for v in list(params_tensor.shape)]
        output_shape = [int(v) for v in list(output_tensor.shape)]
        params_rank = len(params_shape)
        params_layout = normalize_logical_layout(params_tensor.logical_layout)
        if params_rank not in {3, 4, 5} or not is_channel_first_logical_layout(
            params_layout
        ):
            continue
        cf_to_cl_perm = logical_layout_permutation(
            source_layout=params_layout,
            target_layout=channel_last_logical_layout(params_rank),
        )
        if cf_to_cl_perm is None:
            continue
        actual_gather_nd_shape = _infer_gather_nd_shape_for_codegen(
            model_ir=model_ir,
            params_shape=params_shape,
            indices_tensor_name=str(op.inputs[1]),
        )
        permuted_shape = _permute_shape(params_shape, cf_to_cl_perm)
        permuted_gather_nd_shape = _infer_gather_nd_shape_for_codegen(
            model_ir=model_ir,
            params_shape=permuted_shape,
            indices_tensor_name=str(op.inputs[1]),
        )
        if not _shape_lists_equal(
            actual_gather_nd_shape, output_shape
        ) and _shape_lists_equal(permuted_gather_nd_shape, output_shape):
            gather_nd_input_perms[str(tensor_var_names[params_name])] = [
                int(v) for v in list(cf_to_cl_perm)
            ]

    if not gather_nd_input_perms:
        return [str(line) for line in lines]

    rewritten = [str(line) for line in lines]
    for index, line in enumerate(rewritten):
        gather_assign = _parse_gather_nd_assign(line)
        if gather_assign is None:
            continue
        lhs, input_expr, indices_expr, target_expr = gather_assign
        double_perm_match = _parse_redundant_double_permute_expr(input_expr)
        if double_perm_match is not None:
            input_var, perm_values, perm_args = double_perm_match
            if perm_values == perm_args:
                rewritten[index] = (
                    f"{lhs} = _apply_gather_nd(_torch_permute({input_var}, {repr(perm_values)}), "
                    f"{indices_expr}, target_shape={target_expr})"
                )
                continue
        if input_expr not in gather_nd_input_perms:
            continue
        perm = gather_nd_input_perms[input_expr]
        rewritten[index] = (
            f"{lhs} = _apply_gather_nd(_torch_permute({input_expr}, {repr(perm)}), "
            f"{indices_expr}, target_shape={target_expr})"
        )
    return rewritten
