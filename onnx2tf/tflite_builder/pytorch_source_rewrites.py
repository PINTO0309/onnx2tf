from __future__ import annotations

import ast
import re
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _extract_statement_assignments,
    _extract_statement_loads,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import _permute_shape
from onnx2tf.tflite_builder.pytorch_shape_policy import _normalize_nhwc_rank4_shape
from onnx2tf.tflite_builder.pytorch_source_parser import (
    _normalize_permute_dims_expr,
    _parse_align_binary_inputs_to_anchor_assign_with_shape,
    _parse_align_tensor_target_shape_expr,
    _parse_binary_add_args,
    _parse_binary_mul_args,
    _parse_rank4_shape_literal,
    _parse_simple_assignment_line,
    _parse_static_binary_add_align_assign,
    _resolve_nhwc_to_nchw_bridge_source,
    _split_top_level_csv_exprs,
)


def _fold_channel_first_gap_conv_bridges(
    lines: Sequence[str],
) -> List[str]:
    def _parse_conv_input_bridge_assign(line: str) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        match = re.fullmatch(
            r"self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\.permute\(0,\s*3,\s*1,\s*2\)\.contiguous\(\)\)",
            rhs.strip(),
        )
        if match is not None:
            return lhs, str(match.group("module")), str(match.group("input"))
        stripped = rhs.strip()
        prefix = "self."
        open_paren = stripped.find("(")
        if (
            not stripped.startswith(prefix)
            or open_paren <= len(prefix)
            or not stripped.endswith(")")
        ):
            return None
        module_name = stripped[len(prefix) : open_paren]
        args_expr = stripped[open_paren + 1 : -1]
        input_name = _resolve_nhwc_to_nchw_bridge_source(args_expr)
        if input_name is None or re.fullmatch(r"[A-Za-z0-9_]+", input_name) is None:
            return None
        return lhs, module_name, input_name

    def _parse_cf_gap_mean_assign(line: str) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        match = re.fullmatch(r"torch\.mean\((?P<args>.+)\)", rhs.strip())
        if match is None:
            return None
        parts = _split_top_level_csv_exprs(str(match.group("args")))
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    dim_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            positional_index += 1
        if input_expr is None or dim_expr is None:
            return None
        try:
            dim_value = ast.literal_eval(dim_expr)
        except Exception:
            return None
        if not isinstance(dim_value, (list, tuple)):
            return None
        if [int(v) for v in list(dim_value)] != [2, 3]:
            return None
        if keepdim_expr is not None and keepdim_expr != "True":
            return None
        if not re.fullmatch(r"[A-Za-z0-9_]+", input_expr):
            return None
        return lhs, input_expr

    rewritten = [str(line) for line in lines]
    channel_first_gap_vars: Set[str] = set()
    for index, line in enumerate(rewritten):
        parsed_mean = _parse_cf_gap_mean_assign(line)
        if parsed_mean is not None:
            channel_first_gap_vars.add(str(parsed_mean[0]))
            continue
        conv_match = _parse_conv_input_bridge_assign(line)
        if conv_match is None:
            continue
        input_var = str(conv_match[2])
        if input_var not in channel_first_gap_vars:
            continue
        rewritten[index] = f"{conv_match[0]} = self.{conv_match[1]}({input_var})"
    return rewritten


def _fold_channel_last_affine_conv_bridges(
    lines: Sequence[str],
    *,
    derive_local_var_name: Callable[[str], str],
    channel_first_constant_expr_for_buffer_attr: Callable[
        [str, Sequence[int]], Optional[str]
    ],
) -> List[str]:
    if len(lines) < 6:
        return [str(line) for line in lines]

    relu_re = re.compile(
        r"^(?P<out>[A-Za-z0-9_]+)\s*=\s*torch\.relu\((?P<input>[A-Za-z0-9_]+)\)$"
    )

    def _parse_nchw_to_nhwc_bridge_source(expr: str) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args_expr: str) -> str | None:
            parts = _split_top_level_csv_exprs(args_expr)
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
            if _normalize_permute_dims_expr(perm_expr) != "0,2,3,1":
                return None
            return input_expr

        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                return _parse_permute_like_args(stripped[len(prefix) : -1])
        method_match = re.fullmatch(
            r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped
        )
        if method_match is None:
            return None
        if _normalize_permute_dims_expr(str(method_match.group("dims"))) != "0,2,3,1":
            return None
        return str(method_match.group("input"))

    def _parse_materialize_assign(line: str) -> Tuple[str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        shape = _parse_rank4_shape_literal(shape_expr)
        if shape is None:
            return None
        source_name = _parse_nchw_to_nhwc_bridge_source(input_expr)
        if source_name is None or re.fullmatch(r"[A-Za-z0-9_]+", source_name) is None:
            return None
        return lhs, source_name, list(shape)

    def _parse_conv_input_bridge_assign(line: str) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        match = re.fullmatch(
            r"self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\.permute\(0,\s*3,\s*1,\s*2\)\.contiguous\(\)\)",
            rhs.strip(),
        )
        if match is not None:
            return lhs, str(match.group("module")), str(match.group("input"))
        stripped = rhs.strip()
        prefix = "self."
        open_paren = stripped.find("(")
        if (
            not stripped.startswith(prefix)
            or open_paren <= len(prefix)
            or not stripped.endswith(")")
        ):
            return None
        module_name = stripped[len(prefix) : open_paren]
        args_expr = stripped[open_paren + 1 : -1]
        input_name = _resolve_nhwc_to_nchw_bridge_source(args_expr)
        if input_name is None or re.fullmatch(r"[A-Za-z0-9_]+", input_name) is None:
            return None
        return lhs, module_name, input_name

    def _parse_align_binary_inputs_assign(
        line: str,
    ) -> Tuple[str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^\(*\s*(?P<lhs>[A-Za-z0-9_]+)\s*,\s*(?P<rhs>[A-Za-z0-9_]+)\s*\)*\s*=\s*(?P<expr>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        expr = str(assign_match.group("expr")).strip()
        prefix = "_align_binary_inputs("
        if not expr.startswith(prefix) or not expr.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(expr[len(prefix) : -1])
        input_expr: str | None = None
        const_expr: str | None = None
        target_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_expr, const_expr, target_expr = (part.strip() for part in parts)
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_expr = kwargs.get("input")
            const_expr = kwargs.get("other", kwargs.get("rhs"))
            target_expr = kwargs.get("target_shape", kwargs.get("shape"))
        shape = (
            _parse_rank4_shape_literal(target_expr) if target_expr is not None else None
        )
        if input_expr is None or const_expr is None or shape is None:
            return None
        return (
            str(assign_match.group("lhs")),
            str(assign_match.group("rhs")),
            input_expr,
            const_expr,
            list(shape),
        )

    def _parse_binary_aligned_assign(
        line: str, op_name: str
    ) -> Tuple[str, str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        shape = _parse_rank4_shape_literal(shape_expr)
        if shape is None:
            return None
        binary_match = re.fullmatch(
            rf"torch\.{op_name}\((?P<args>.+)\)", input_expr.strip()
        )
        if binary_match is None:
            return None
        binary_args = (
            _parse_binary_mul_args(str(binary_match.group("args")))
            if op_name == "mul"
            else _parse_binary_add_args(str(binary_match.group("args")))
        )
        if binary_args is None:
            return None
        return (
            lhs,
            str(binary_args[0]).strip(),
            str(binary_args[1]).strip(),
            input_expr,
            list(shape),
        )

    rewritten: List[str] = []
    index = 0
    while index < len(lines):
        if index + 5 >= len(lines):
            rewritten.extend(str(line) for line in lines[index:])
            break
        materialize_match = _parse_materialize_assign(str(lines[index]))
        if materialize_match is None:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        align1_match = _parse_align_binary_inputs_assign(str(lines[index + 1]))
        mul_match = _parse_binary_aligned_assign(str(lines[index + 2]), "mul")
        align2_match = _parse_align_binary_inputs_assign(str(lines[index + 3]))
        add_match = _parse_binary_aligned_assign(str(lines[index + 4]), "add")
        if (
            align1_match is None
            or mul_match is None
            or align2_match is None
            or add_match is None
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        has_relu = False
        conv_line_index = index + 5
        relu_match = relu_re.match(str(lines[index + 5]))
        if relu_match is not None:
            has_relu = True
            conv_line_index = index + 6
            if conv_line_index >= len(lines):
                rewritten.append(str(lines[index]))
                index += 1
                continue
        conv_match = _parse_conv_input_bridge_assign(str(lines[conv_line_index]))
        if conv_match is None:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if (
            align1_match[2] != materialize_match[0]
            or align1_match[4] != materialize_match[2]
            or mul_match[1] != align1_match[0]
            or mul_match[2] != align1_match[1]
            or mul_match[4] != materialize_match[2]
            or align2_match[2] != mul_match[0]
            or align2_match[4] != materialize_match[2]
            or add_match[1] != align2_match[0]
            or add_match[2] != align2_match[1]
            or add_match[4] != materialize_match[2]
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if has_relu:
            resolved_relu_match = relu_match
            if resolved_relu_match is None:
                rewritten.append(str(lines[index]))
                index += 1
                continue
            if resolved_relu_match.group("input") != add_match[0] or conv_match[
                2
            ] != resolved_relu_match.group("out"):
                rewritten.append(str(lines[index]))
                index += 1
                continue
        elif conv_match[2] != add_match[0]:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if len(materialize_match[2]) != 4 or int(materialize_match[2][3]) <= 0:
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if not all(
            token.strip().startswith("self.")
            for token in [align1_match[3], align2_match[3]]
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        channel_count = int(materialize_match[2][3])
        channel_first_shape = [1, channel_count, 1, 1]
        mul_cf_var = derive_local_var_name(f"{mul_match[0]}_cf")
        add_cf_var = derive_local_var_name(f"{add_match[0]}_cf")
        final_cf_var = add_cf_var
        mul_rhs_expr = channel_first_constant_expr_for_buffer_attr(
            align1_match[3],
            channel_first_shape,
        )
        if mul_rhs_expr is None:
            mul_rhs_expr = (
                f"torch.reshape({align1_match[3]}, {repr(channel_first_shape)})"
            )
        add_rhs_expr = channel_first_constant_expr_for_buffer_attr(
            align2_match[3],
            channel_first_shape,
        )
        if add_rhs_expr is None:
            add_rhs_expr = (
                f"torch.reshape({align2_match[3]}, {repr(channel_first_shape)})"
            )
        rewritten.append(
            f"{mul_cf_var} = torch.mul({materialize_match[1]}, {mul_rhs_expr})"
        )
        rewritten.append(f"{add_cf_var} = torch.add({mul_cf_var}, {add_rhs_expr})")
        if has_relu:
            resolved_relu_match = relu_match
            if resolved_relu_match is None:
                rewritten.append(str(lines[index]))
                index += 1
                continue
            final_cf_var = derive_local_var_name(
                f"{resolved_relu_match.group('out')}_cf"
            )
            rewritten.append(f"{final_cf_var} = torch.relu({add_cf_var})")
        rewritten.append(f"{conv_match[0]} = self.{conv_match[1]}({final_cf_var})")
        index = conv_line_index + 1
    return rewritten


def _rewrite_channel_last_gap_means_to_reduce_mean(
    lines: Sequence[str],
) -> List[str]:
    def _parse_rank3_nchw_to_nhwc_source(expr: str) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args_expr: str) -> str | None:
            parts = _split_top_level_csv_exprs(args_expr)
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
            if _normalize_permute_dims_expr(perm_expr) != "0,2,1":
                return None
            return input_expr

        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                return _parse_permute_like_args(stripped[len(prefix) : -1])
        method_match = re.fullmatch(
            r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped
        )
        if method_match is None:
            return None
        if _normalize_permute_dims_expr(str(method_match.group("dims"))) != "0,2,1":
            return None
        return str(method_match.group("input"))

    def _parse_rank4_nchw_to_nhwc_expr(expr: str) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args_expr: str) -> str | None:
            parts = _split_top_level_csv_exprs(args_expr)
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
            if input_expr is None or perm_expr is None:
                return None
            if _normalize_permute_dims_expr(perm_expr) != "0,2,3,1":
                return None
            return f"{input_expr}.permute(0, 2, 3, 1).contiguous()"

        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                return _parse_permute_like_args(stripped[len(prefix) : -1])
        method_match = re.fullmatch(r"(?P<input>.+)\.permute\((?P<dims>.+)\)", stripped)
        if method_match is None:
            return None
        if _normalize_permute_dims_expr(str(method_match.group("dims"))) != "0,2,3,1":
            return None
        return f"{str(method_match.group('input'))}.permute(0, 2, 3, 1).contiguous()"

    def _parse_rank3_cf_materialize_assign(
        line: str,
    ) -> Tuple[str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        try:
            shape_value = ast.literal_eval(shape_expr)
        except Exception:
            return None
        if not isinstance(shape_value, (list, tuple)) or len(shape_value) != 3:
            return None
        source_expr = _parse_rank3_nchw_to_nhwc_source(input_expr)
        if source_expr is None:
            return None
        try:
            shape = [int(v) for v in shape_value]
        except Exception:
            return None
        return lhs, source_expr, shape

    def _parse_rank3_mean_assign(line: str) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        match = re.fullmatch(r"torch\.mean\((?P<args>.+)\)", rhs.strip())
        if match is None:
            return None
        parts = _split_top_level_csv_exprs(str(match.group("args")))
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    dim_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                dim_expr = part.strip()
            positional_index += 1
        if input_expr is None or dim_expr is None:
            return None
        try:
            dim_value = ast.literal_eval(dim_expr)
        except Exception:
            try:
                dim_value = int(dim_expr)
            except Exception:
                return None
        if isinstance(dim_value, (list, tuple)):
            if list(dim_value) != [2]:
                return None
        elif int(dim_value) != 2:
            return None
        if keepdim_expr is not None and keepdim_expr != "True":
            return None
        if not re.fullmatch(r"[A-Za-z0-9_]+", input_expr):
            return None
        return lhs, input_expr

    def _rewrite_line(line: str) -> str:
        search_from = 0
        rewritten_line = str(line)
        while True:
            start = rewritten_line.find("torch.mean(", search_from)
            if start < 0:
                return rewritten_line
            cursor = start + len("torch.mean(")
            depth = 1
            while cursor < len(rewritten_line) and depth > 0:
                char = rewritten_line[cursor]
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                cursor += 1
            if depth != 0:
                return rewritten_line
            end = cursor
            args_expr = rewritten_line[start + len("torch.mean(") : end - 1]
            parts = _split_top_level_csv_exprs(args_expr)
            input_expr: str | None = None
            dim_expr: str | None = None
            keepdim_expr: str | None = None
            positional_index = 0
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "input":
                        input_expr = value
                    elif key == "dim":
                        dim_expr = value
                    elif key == "keepdim":
                        keepdim_expr = value
                    continue
                if positional_index == 0:
                    input_expr = part.strip()
                elif positional_index == 1:
                    dim_expr = part.strip()
                positional_index += 1
            if input_expr is None or dim_expr is None:
                search_from = end
                continue
            try:
                dim_value = ast.literal_eval(dim_expr)
            except Exception:
                search_from = end
                continue
            if not isinstance(dim_value, (list, tuple)) or list(dim_value) != [1, 2]:
                search_from = end
                continue
            if keepdim_expr is not None and keepdim_expr != "True":
                search_from = end
                continue
            expr = _parse_rank4_nchw_to_nhwc_expr(input_expr)
            if expr is None:
                search_from = end
                continue
            replacement = f"_reduce_mean({expr}, _normalize_axes([1, 2], {expr}.ndim), keepdims=True)"
            rewritten_line = rewritten_line[:start] + replacement + rewritten_line[end:]
            search_from = start + len(replacement)

    rewritten = [_rewrite_line(str(line)) for line in lines]
    for index in range(1, len(rewritten)):
        materialize_match = _parse_rank3_cf_materialize_assign(
            str(rewritten[index - 1])
        )
        mean_match = _parse_rank3_mean_assign(str(rewritten[index]))
        if materialize_match is None or mean_match is None:
            continue
        if mean_match[1] != materialize_match[0]:
            continue
        if len(materialize_match[2]) != 3 or not all(
            isinstance(value, int) for value in list(materialize_match[2])
        ):
            continue
        rewritten[index] = (
            f"{mean_match[0]} = torch.mean({materialize_match[1]}, dim=1, keepdim=True)"
        )
    return rewritten


def _rewrite_channel_first_gap_outputs_to_explicit_channel_last(
    lines: Sequence[str],
) -> List[str]:
    def _parse_nchw_to_nhwc_permute_assign(line: str) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        stripped = rhs.strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args_expr: str) -> Tuple[str, str] | None:
            parts = _split_top_level_csv_exprs(args_expr)
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
            if _normalize_permute_dims_expr(perm_expr) != "0,2,3,1":
                return None
            return lhs, input_expr

        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                return _parse_permute_like_args(stripped[len(prefix) : -1])
        method_match = re.fullmatch(
            r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped
        )
        if method_match is None:
            return None
        if _normalize_permute_dims_expr(str(method_match.group("dims"))) != "0,2,3,1":
            return None
        return lhs, str(method_match.group("input"))

    def _parse_cf_gap_mean_assign(line: str) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        match = re.fullmatch(r"torch\.mean\((?P<args>.+)\)", rhs.strip())
        if match is None:
            return None
        parts = _split_top_level_csv_exprs(str(match.group("args")))
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    dim_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            positional_index += 1
        if input_expr is None or dim_expr is None:
            return None
        try:
            dim_value = ast.literal_eval(dim_expr)
        except Exception:
            return None
        if not isinstance(dim_value, (list, tuple)):
            return None
        if [int(v) for v in list(dim_value)] != [2, 3]:
            return None
        if keepdim_expr is not None and keepdim_expr != "True":
            return None
        if not re.fullmatch(r"[A-Za-z0-9_]+", input_expr):
            return None
        return lhs, input_expr

    rewritten = [str(line) for line in lines]
    for index in range(len(rewritten) - 1):
        parsed_mean = _parse_cf_gap_mean_assign(rewritten[index])
        if parsed_mean is None:
            continue
        permute_match = _parse_nchw_to_nhwc_permute_assign(rewritten[index + 1])
        if permute_match is None:
            continue
        if str(permute_match[1]) != str(parsed_mean[0]):
            continue
        rewritten[index] = (
            f"{permute_match[0]} = torch.mean("
            f"{parsed_mean[1]}.permute(0, 2, 3, 1).contiguous(), "
            f"dim=[1, 2], keepdim=True)"
        )
        rewritten[index + 1] = ""
    return [line for line in rewritten if line != ""]


def _rewrite_channel_first_se_scale_binary_bridges(
    lines: Sequence[str],
) -> List[str]:
    rewritten = [str(line) for line in lines]

    def _parse_nchw_to_nhwc_bridge_source(expr: str) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args: str) -> str | None:
            parts = _split_top_level_csv_exprs(str(args))
            if len(parts) == 2 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
            ):
                source_expr = parts[0].strip()
                dims_expr = _normalize_permute_dims_expr(parts[1])
                if (
                    re.fullmatch(r"[A-Za-z0-9_]+", source_expr) is not None
                    and dims_expr == "0,2,3,1"
                ):
                    return source_expr
                return None
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            source_expr = kwargs.get("input", kwargs.get("x"))
            dims_expr = kwargs.get("dims", kwargs.get("perm"))
            if (
                source_expr is None
                or dims_expr is None
                or re.fullmatch(r"[A-Za-z0-9_]+", source_expr) is None
                or _normalize_permute_dims_expr(dims_expr) != "0,2,3,1"
            ):
                return None
            return source_expr

        functional_match = re.match(r"^torch\.permute\((?P<args>.+)\)$", stripped)
        if functional_match is not None:
            return _parse_permute_like_args(str(functional_match.group("args")))
        helper_match = re.match(r"^_torch_permute\((?P<args>.+)\)$", stripped)
        if helper_match is not None:
            return _parse_permute_like_args(str(helper_match.group("args")))
        method_match = re.match(
            r"^(?P<src>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)$", stripped
        )
        if (
            method_match is not None
            and _normalize_permute_dims_expr(str(method_match.group("dims")))
            == "0,2,3,1"
        ):
            return str(method_match.group("src"))
        return None

    def _parse_nchw_to_nhwc_alias(line: str) -> Tuple[str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, target_expr = align_parts
        target_shape = _parse_rank4_shape_literal(target_expr)
        if target_shape is None:
            return None
        source_expr = _parse_nchw_to_nhwc_bridge_source(input_expr)
        if source_expr is None:
            return None
        return lhs, source_expr, list(target_shape)

    def _parse_binary_align_assign(
        line: str,
    ) -> Tuple[str, str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*(?P<expr>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        expr = str(assign_match.group("expr")).strip()
        prefix = "_align_binary_inputs("
        if not expr.startswith(prefix) or not expr.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(expr[len(prefix) : -1])
        input0: str | None = None
        input1: str | None = None
        target_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input0, input1, target_expr = (part.strip() for part in parts)
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input0 = kwargs.get("input", kwargs.get("lhs"))
            input1 = kwargs.get("other", kwargs.get("rhs"))
            target_expr = kwargs.get("target_shape", kwargs.get("shape"))
        target_shape = (
            _parse_rank4_shape_literal(target_expr) if target_expr is not None else None
        )
        if (
            input0 is None
            or input1 is None
            or target_shape is None
            or re.fullmatch(r"[A-Za-z0-9_]+", input0) is None
            or re.fullmatch(r"[A-Za-z0-9_]+", input1) is None
        ):
            return None
        return (
            str(assign_match.group("indent")),
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            input0,
            input1,
            list(target_shape),
        )

    dynamic_cf_mul_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\("
        r"torch\.mul\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\), "
        r"\[int\((?P<ref>[A-Za-z0-9_]+)\.shape\[0\]\), (?P<c>\d+), int\((?P=ref)\.shape\[2\]\), int\((?P=ref)\.shape\[3\]\)\]\)$"
    )
    nhwc_alias_sources: Dict[str, Tuple[str, List[int]]] = {}
    for line in rewritten:
        alias_match = _parse_nchw_to_nhwc_alias(line)
        if alias_match is None:
            continue
        nhwc_alias_sources[str(alias_match[0])] = (
            str(alias_match[1]),
            list(alias_match[2]),
        )

    for index in range(len(rewritten) - 1):
        align_match = _parse_binary_align_assign(rewritten[index])
        mul_match = dynamic_cf_mul_re.match(rewritten[index + 1])
        if align_match is None or mul_match is None:
            continue
        indent, lhs0, lhs1, input0, input1, target_shape = align_match
        feature_source: str | None = None
        scale_source: str | None = None
        feature_target: List[int] | None = None
        if input1 in nhwc_alias_sources:
            feature_source, feature_target = nhwc_alias_sources[input1]
            scale_source = input0
        elif input0 in nhwc_alias_sources:
            feature_source, feature_target = nhwc_alias_sources[input0]
            scale_source = input1
        if feature_source is None or feature_target is None or scale_source is None:
            continue
        if list(target_shape) != list(feature_target):
            continue
        mul_inputs = {str(mul_match.group("a")), str(mul_match.group("b"))}
        if mul_inputs != {lhs0, lhs1}:
            continue
        ref_var = str(mul_match.group("ref"))
        cf_channel_dim = int(mul_match.group("c"))
        cf_target = [
            int(feature_target[0]),
            cf_channel_dim,
            int(feature_target[1]),
            int(feature_target[2]),
        ]
        rewritten[index] = (
            f"{indent}{lhs0}, {lhs1} = _align_binary_inputs_to_anchor("
            f"{feature_source}, {scale_source}, [{cf_target[0]}, {cf_target[1]}, {cf_target[2]}, {cf_target[3]}])"
        )
        if ref_var != lhs0:
            rewritten[index + 1] = (
                f"{mul_match.group('indent')}{mul_match.group('lhs')} = "
                f"_align_tensor_to_target_shape(torch.mul({mul_match.group('a')}, {mul_match.group('b')}), "
                f"[int({lhs0}.shape[0]), {cf_channel_dim}, int({lhs0}.shape[2]), int({lhs0}.shape[3])])"
            )
    return rewritten


def _fold_boundary_transpose_pad_conv_bridges(
    lines: Sequence[str],
) -> List[str]:
    def _parse_nchw_to_nhwc_bridge_source(expr: str) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args: str) -> str | None:
            parts = _split_top_level_csv_exprs(str(args))
            if len(parts) == 2 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
            ):
                source_expr = parts[0].strip()
                dims_expr = _normalize_permute_dims_expr(parts[1])
                if (
                    re.fullmatch(r"[A-Za-z0-9_]+", source_expr) is not None
                    and dims_expr == "0,2,3,1"
                ):
                    return source_expr
                return None
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            source_expr = kwargs.get("input", kwargs.get("x"))
            dims_expr = kwargs.get("dims", kwargs.get("perm"))
            if (
                source_expr is None
                or dims_expr is None
                or re.fullmatch(r"[A-Za-z0-9_]+", source_expr) is None
                or _normalize_permute_dims_expr(dims_expr) != "0,2,3,1"
            ):
                return None
            return source_expr

        functional_match = re.match(r"^torch\.permute\((?P<args>.+)\)$", stripped)
        if functional_match is not None:
            return _parse_permute_like_args(str(functional_match.group("args")))

        helper_match = re.match(r"^_torch_permute\((?P<args>.+)\)$", stripped)
        if helper_match is not None:
            return _parse_permute_like_args(str(helper_match.group("args")))

        method_match = re.match(
            r"^(?P<src>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)$", stripped
        )
        if (
            method_match is not None
            and _normalize_permute_dims_expr(str(method_match.group("dims")))
            == "0,2,3,1"
        ):
            return str(method_match.group("src"))
        return None

    def _parse_boundary_conv_assign(line: str) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        match = re.fullmatch(
            r"self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\.permute\(0,\s*2,\s*3,\s*1\)\.contiguous\(\)\)",
            rhs.strip(),
        )
        if match is not None:
            return lhs, str(match.group("module")), str(match.group("input"))
        stripped = rhs.strip()
        prefix = "self."
        open_paren = stripped.find("(")
        if (
            not stripped.startswith(prefix)
            or open_paren <= len(prefix)
            or not stripped.endswith(")")
        ):
            return None
        module_name = stripped[len(prefix) : open_paren]
        args_expr = stripped[open_paren + 1 : -1]
        input_expr = _parse_nchw_to_nhwc_bridge_source(args_expr)
        if input_expr is None or not re.fullmatch(r"[A-Za-z0-9_]+", input_expr):
            return None
        return lhs, module_name, input_expr

    def _parse_output_bridge_assign(line: str) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        stripped = rhs.strip()
        source_expr = _parse_nchw_to_nhwc_bridge_source(stripped)
        if source_expr is None:
            return None
        return lhs, source_expr

    rewritten = [str(line) for line in lines]
    for index in range(len(rewritten) - 1):
        conv_match = _parse_boundary_conv_assign(rewritten[index])
        if conv_match is None:
            continue
        output_bridge_match = _parse_output_bridge_assign(rewritten[index + 1])
        if output_bridge_match is None:
            continue
        if str(output_bridge_match[1]) != str(conv_match[0]):
            continue
        rewritten[index] = f"{conv_match[0]} = self.{conv_match[1]}({conv_match[2]})"
    return rewritten


def _collapse_redundant_torch_permute_chains(
    lines: Sequence[str],
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

    def _rewrite_line(line: str) -> str:
        match = re.search(r"_torch_permute\(|torch\.permute\(", line)
        if match is None:
            return line
        start = int(match.start())
        cursor = start
        depth = 0
        end: int | None = None
        while cursor < len(line):
            char = line[cursor]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = cursor + 1
                    break
            cursor += 1
        if end is None:
            return line
        suffix = line[end:]
        tail_match = re.match(
            r"(?:\.contiguous\(\))?\.permute\((?P<perm_args>[^\)]+)\)\.contiguous\(\)",
            suffix,
        )
        if tail_match is None:
            return line
        permute_match = _parse_torch_permute_expr(line[start:end])
        if permute_match is None:
            return line
        try:
            perm_args = [
                int(v.strip()) for v in str(tail_match.group("perm_args")).split(",")
            ]
        except Exception:
            return line
        input_expr, perm_values = permute_match
        if perm_values != perm_args:
            return line
        return (
            line[:start]
            + f"_torch_permute({input_expr}, {repr(perm_values)})"
            + line[end + tail_match.end() :]
        )

    return [_rewrite_line(str(line)) for line in lines]


def _inline_trivial_public_layout_bridge_aliases(
    lines: Sequence[str],
) -> List[str]:
    assign_re = re.compile(
        r"^(?P<alias>[A-Za-z0-9_]+_public_layout_bridge)\s*=\s*(?P<source>[A-Za-z0-9_]+)$"
    )
    rewritten = [str(line) for line in lines]
    alias_map: Dict[str, str] = {}
    kept_lines: List[str] = []
    for line in rewritten:
        match = assign_re.match(line)
        if match is not None:
            alias_map[str(match.group("alias"))] = str(match.group("source"))
            continue
        rewritten_line = str(line)
        for alias, source in alias_map.items():
            rewritten_line = re.sub(rf"\b{re.escape(alias)}\b", source, rewritten_line)
        kept_lines.append(rewritten_line)
    return kept_lines


def _fold_channel_last_prelu_bridges(
    lines: Sequence[str],
) -> List[str]:
    prelu_re = re.compile(
        r"^(?P<var>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>prelu_[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )

    def _parse_permute_assign(
        line: str, expected_perm: Sequence[int]
    ) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        stripped = rhs.strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args_expr: str) -> Tuple[str, str] | None:
            parts = _split_top_level_csv_exprs(args_expr)
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
                or not re.fullmatch(r"[A-Za-z0-9_]+", input_expr)
            ):
                return None
            try:
                perm_value = ast.literal_eval(perm_expr)
            except Exception:
                return None
            if list(perm_value) != list(expected_perm):
                return None
            return lhs, input_expr

        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                parsed = _parse_permute_like_args(stripped[len(prefix) : -1])
                if parsed is not None:
                    return parsed
        method_match = re.fullmatch(
            r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<perm>.+)\)", stripped
        )
        if method_match is None:
            return None
        if _normalize_permute_dims_expr(str(method_match.group("perm"))) != ",".join(
            [str(int(v)) for v in list(expected_perm)]
        ):
            return None
        return lhs, str(method_match.group("input"))

    rewritten = [str(line) for line in lines]
    index = 0
    while index <= len(rewritten) - 3:
        in_match = _parse_permute_assign(rewritten[index], [0, 3, 1, 2])
        if in_match is None:
            index += 1
            continue
        prelu_match = prelu_re.match(rewritten[index + 1])
        out_match = _parse_permute_assign(rewritten[index + 2], [0, 2, 3, 1])
        if prelu_match is None or out_match is None:
            index += 1
            continue
        bridge_var = str(in_match[0])
        if str(prelu_match.group("input")) != bridge_var:
            index += 1
            continue
        prelu_out_var = str(prelu_match.group("var"))
        if str(out_match[1]) != prelu_out_var:
            index += 1
            continue
        rewritten[index] = (
            f"{out_match[0]} = self.{prelu_match.group('module')}("
            f"{in_match[1]}.permute(0, 3, 1, 2).contiguous()"
            f").permute(0, 2, 3, 1).contiguous()"
        )
        rewritten[index + 1] = ""
        rewritten[index + 2] = ""
        index += 3
    return [line for line in rewritten if line != ""]


def _fold_rank4_reshape_permute_conv_bridges(
    lines: Sequence[str],
) -> List[str]:
    if len(lines) < 3:
        return [str(line) for line in lines]

    rewritten: List[str] = []

    def _parse_torch_reshape_assign(line: str) -> Tuple[str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        rhs = rhs.strip()
        prefix = "torch.reshape("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        input_expr: str | None = None
        shape_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "shape":
                    shape_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                shape_expr = part.strip()
            positional_index += 1
        rank4_shape = (
            _parse_rank4_shape_literal(shape_expr) if shape_expr is not None else None
        )
        if input_expr is None or rank4_shape is None:
            return None
        return lhs, input_expr, list(rank4_shape)

    def _parse_permute_source(expr: str, expected_perm: Sequence[int]) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args: str) -> str | None:
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
            if _normalize_permute_dims_expr(perm_expr) != ",".join(
                str(v) for v in list(expected_perm)
            ):
                return None
            return input_expr

        functional_match = re.fullmatch(r"torch\.permute\((?P<args>.+)\)", stripped)
        if functional_match is not None:
            return _parse_permute_like_args(str(functional_match.group("args")))
        helper_match = re.fullmatch(r"_torch_permute\((?P<args>.+)\)", stripped)
        if helper_match is not None:
            return _parse_permute_like_args(str(helper_match.group("args")))
        method_match = re.fullmatch(
            r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped
        )
        if method_match is not None and _normalize_permute_dims_expr(
            str(method_match.group("dims"))
        ) == ",".join(str(v) for v in list(expected_perm)):
            return str(method_match.group("input"))
        return None

    def _parse_nhwc_bridge_assign(line: str) -> Tuple[str, str, List[int]] | None:
        parsed = _parse_torch_reshape_assign(line)
        if parsed is None:
            return None
        lhs, input_expr, rank4_shape = parsed
        source_expr = _parse_permute_source(input_expr, [0, 2, 3, 1])
        if source_expr is None:
            return None
        return lhs, source_expr, rank4_shape

    def _parse_conv_bridge_assign(line: str) -> Tuple[str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, target_expr = align_parts
        target_shape = _parse_rank4_shape_literal(target_expr)
        if target_shape is None:
            return None
        stripped = input_expr.strip()
        prefix = "self."
        open_paren = stripped.find("(")
        if (
            not stripped.startswith(prefix)
            or open_paren <= len(prefix)
            or not stripped.endswith(")")
        ):
            return None
        module_name = stripped[len(prefix) : open_paren]
        args_expr = stripped[open_paren + 1 : -1]
        source_expr = _parse_permute_source(args_expr, [0, 3, 1, 2])
        if source_expr is None:
            return None
        return lhs, module_name, source_expr, list(target_shape)

    index = 0
    while index < len(lines):
        if index + 2 >= len(lines):
            rewritten.extend(str(line) for line in lines[index:])
            break
        reshape_match = _parse_torch_reshape_assign(str(lines[index]))
        bridge_match = _parse_nhwc_bridge_assign(str(lines[index + 1]))
        conv_match = _parse_conv_bridge_assign(str(lines[index + 2]))
        if (
            reshape_match is None
            or bridge_match is None
            or conv_match is None
            or bridge_match[1] != reshape_match[0]
            or conv_match[2] != bridge_match[0]
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        if (
            len(reshape_match[2]) != 4
            or len(bridge_match[2]) != 4
            or _permute_shape(reshape_match[2], [0, 2, 3, 1])
            != [int(v) for v in list(bridge_match[2])]
        ):
            rewritten.append(str(lines[index]))
            index += 1
            continue
        rewritten.append(str(lines[index]))
        rewritten.append(
            f"{conv_match[0]} = _align_tensor_to_target_shape(self.{conv_match[1]}({reshape_match[0]}), {repr(conv_match[3])})"
        )
        index += 3
    return rewritten


def _fold_channel_first_hardsigmoid_gate_conv_bridges(
    lines: Sequence[str],
) -> List[str]:
    clamp_re = re.compile(
        r"^(?P<indent>\s*)(?P<out>[A-Za-z0-9_]+)\s*=\s*torch\.clamp\((?P<input>[A-Za-z0-9_]+), min=0\.0, max=1\.0\)$"
    )
    anchor_cf_re = re.compile(
        r"^\s*[A-Za-z0-9_]+,\s*[A-Za-z0-9_]+ = _align_binary_inputs_to_anchor\((?P<input0>[A-Za-z0-9_]+), (?P<input1>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<c>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    binary_cf_re = re.compile(
        r"^\s*[A-Za-z0-9_]+,\s*[A-Za-z0-9_]+ = _align_binary_inputs\((?P<input0>[A-Za-z0-9_]+), (?P<input1>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<c>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    rewritten = [str(line) for line in lines]
    alpha_ref = float(1.0 / 6.0)
    beta_ref = 0.5
    eps = 1e-6

    def _function_end_index(line_index: int) -> int:
        for candidate in range(line_index + 1, len(rewritten)):
            if rewritten[candidate].startswith("    def "):
                return candidate
        return len(rewritten)

    def _line_mentions_name(line: str, name: str) -> bool:
        return re.search(rf"\b{re.escape(name)}\b", line) is not None

    def _parse_gap_mean_assign(line: str) -> Tuple[str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        stripped = rhs.strip()
        prefix = "torch.mean("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        if len(parts) >= 1 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts[:1]
        ):
            input_expr = parts[0].strip()
            for part in parts[1:]:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                if key.strip() == "dim":
                    dim_expr = value.strip()
                elif key.strip() == "keepdim":
                    keepdim_expr = value.strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_expr = kwargs.get("input")
            dim_expr = kwargs.get("dim")
            keepdim_expr = kwargs.get("keepdim")
        if (
            input_expr is None
            or dim_expr is None
            or not re.fullmatch(r"[A-Za-z0-9_]+", input_expr)
        ):
            return None
        try:
            dim_value = ast.literal_eval(dim_expr)
        except Exception:
            return None
        if isinstance(dim_value, tuple):
            dim_list = [int(v) for v in list(dim_value)]
        elif isinstance(dim_value, list):
            dim_list = [int(v) for v in dim_value]
        else:
            return None
        if dim_list not in ([1, 2], [2, 3]):
            return None
        if keepdim_expr is not None and keepdim_expr != "True":
            return None
        return indent, lhs, input_expr, list(dim_list)

    def _parse_cf_conv_assign(line: str) -> Tuple[str, str, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        match = re.fullmatch(
            r"self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\.permute\(0,\s*3,\s*1,\s*2\)\.contiguous\(\)\)",
            rhs.strip(),
        )
        if match is not None:
            return indent, lhs, str(match.group("module")), str(match.group("input"))
        direct_match = re.fullmatch(
            r"self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\)",
            rhs.strip(),
        )
        if direct_match is not None:
            return (
                indent,
                lhs,
                str(direct_match.group("module")),
                str(direct_match.group("input")),
            )
        stripped = rhs.strip()
        prefix = "self."
        open_paren = stripped.find("(")
        if (
            not stripped.startswith(prefix)
            or open_paren <= len(prefix)
            or not stripped.endswith(")")
        ):
            return None
        module_name = stripped[len(prefix) : open_paren]
        args_expr = stripped[open_paren + 1 : -1]
        input_expr = _resolve_nhwc_to_nchw_bridge_source(args_expr)
        if input_expr is None or not re.fullmatch(r"[A-Za-z0-9_]+", input_expr):
            return None
        return indent, lhs, module_name, input_expr

    def _parse_scalar_mul_assign(line: str) -> Tuple[str, str, str, float] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        binary_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", rhs.strip())
        if binary_match is None:
            return None
        binary_args = _parse_binary_mul_args(str(binary_match.group("args")))
        if binary_args is None:
            return None
        input_name: str | None = None
        scalar_expr: str | None = None
        for candidate in binary_args:
            candidate = str(candidate).strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", candidate):
                input_name = candidate
            elif re.fullmatch(r"[-+0-9.eE]+", candidate):
                scalar_expr = candidate
        if input_name is None or scalar_expr is None:
            return None
        try:
            scalar_value = float(scalar_expr)
        except Exception:
            return None
        return indent, lhs, input_name, scalar_value

    def _parse_scalar_add_align_assign(
        line: str,
    ) -> Tuple[str, str, str, float, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, target_expr = align_parts
        target_shape = _parse_rank4_shape_literal(target_expr)
        if target_shape is None:
            return None
        binary_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", input_expr.strip())
        if binary_match is None:
            return None
        binary_args = _parse_binary_add_args(str(binary_match.group("args")))
        if binary_args is None:
            return None
        input_name: str | None = None
        scalar_expr: str | None = None
        for candidate in binary_args:
            candidate = str(candidate).strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", candidate):
                input_name = candidate
            elif re.fullmatch(r"[-+0-9.eE]+", candidate):
                scalar_expr = candidate
        if input_name is None or scalar_expr is None:
            return None
        try:
            scalar_value = float(scalar_expr)
        except Exception:
            return None
        return indent, lhs, input_name, scalar_value, list(target_shape)

    def _parse_anchor_assign(
        line: str,
    ) -> Tuple[str, str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*(?P<expr>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        expr = str(assign_match.group("expr")).strip()
        prefix = "_align_binary_inputs_to_anchor("
        if not expr.startswith(prefix) or not expr.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(expr[len(prefix) : -1])
        input0: str | None = None
        input1: str | None = None
        target_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input0, input1, target_expr = (part.strip() for part in parts)
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input0 = kwargs.get("a", kwargs.get("input", kwargs.get("lhs")))
            input1 = kwargs.get("b", kwargs.get("other", kwargs.get("rhs")))
            target_expr = kwargs.get("target_shape", kwargs.get("shape"))
        target_shape = (
            _parse_rank4_shape_literal(target_expr) if target_expr is not None else None
        )
        if (
            input0 is None
            or input1 is None
            or target_shape is None
            or re.fullmatch(r"[A-Za-z0-9_]+", input0) is None
            or re.fullmatch(r"[A-Za-z0-9_]+", input1) is None
        ):
            return None
        return (
            str(assign_match.group("indent")),
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            input0,
            input1,
            list(target_shape),
        )

    def _parse_mul_out_assign(line: str) -> Tuple[str, str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, target_expr = align_parts
        target_shape = _parse_rank4_shape_literal(target_expr)
        if target_shape is None:
            return None
        binary_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", input_expr.strip())
        if binary_match is None:
            return None
        binary_args = _parse_binary_mul_args(str(binary_match.group("args")))
        if (
            binary_args is None
            or re.fullmatch(r"[A-Za-z0-9_]+", str(binary_args[0]).strip()) is None
            or re.fullmatch(r"[A-Za-z0-9_]+", str(binary_args[1]).strip()) is None
        ):
            return None
        return (
            indent,
            lhs,
            str(binary_args[0]).strip(),
            str(binary_args[1]).strip(),
            list(target_shape),
        )

    def _parse_cf_dynamic_mul_out_assign(
        line: str,
    ) -> Tuple[str, str, str, str, str, int] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, target_expr = align_parts
        binary_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", input_expr.strip())
        if binary_match is None:
            return None
        binary_args = _parse_binary_mul_args(str(binary_match.group("args")))
        if (
            binary_args is None
            or re.fullmatch(r"[A-Za-z0-9_]+", str(binary_args[0]).strip()) is None
            or re.fullmatch(r"[A-Za-z0-9_]+", str(binary_args[1]).strip()) is None
        ):
            return None
        target_match = re.fullmatch(
            r"\[int\((?P<ref>[A-Za-z0-9_]+)\.shape\[0\]\), (?P<c>\d+), "
            r"int\((?P=ref)\.shape\[2\]\), int\((?P=ref)\.shape\[3\]\)\]",
            target_expr.strip(),
        )
        if target_match is None:
            return None
        return (
            indent,
            lhs,
            str(binary_args[0]).strip(),
            str(binary_args[1]).strip(),
            str(target_match.group("ref")),
            int(target_match.group("c")),
        )

    def _parse_return_identifier(line: str) -> str | None:
        stripped = str(line).strip()
        direct_match = re.fullmatch(r"return\s+(?P<value>[A-Za-z0-9_]+)", stripped)
        if direct_match is not None:
            return str(direct_match.group("value"))
        parenthesized_match = re.fullmatch(
            r"return\s+\(\s*(?P<value>[A-Za-z0-9_]+)\s*\)", stripped
        )
        if parenthesized_match is not None:
            return str(parenthesized_match.group("value"))
        return None

    index = 0
    while index <= len(rewritten) - 5:
        mul_match = _parse_scalar_mul_assign(rewritten[index])
        add_match = _parse_scalar_add_align_assign(rewritten[index + 1])
        clamp_match = clamp_re.match(rewritten[index + 2])
        anchor_match = _parse_anchor_assign(rewritten[index + 3])
        mul_out_match = _parse_mul_out_assign(rewritten[index + 4])
        mul_out_cf_match = _parse_cf_dynamic_mul_out_assign(rewritten[index + 4])
        if (
            mul_match is None
            or add_match is None
            or clamp_match is None
            or anchor_match is None
            or (mul_out_match is None and mul_out_cf_match is None)
        ):
            index += 1
            continue
        try:
            alpha = float(mul_match[3])
            beta = float(add_match[3])
        except Exception:
            index += 1
            continue
        if abs(alpha - alpha_ref) > eps or abs(beta - beta_ref) > eps:
            index += 1
            continue
        mul_out = str(mul_match[1])
        add_out = str(add_match[1])
        clamp_out = str(clamp_match.group("out"))
        source_input = str(mul_match[2])
        if (
            str(add_match[2]) != mul_out
            or str(clamp_match.group("input")) != add_out
            or str(anchor_match[3]) != clamp_out
            or sorted(int(v) for v in list(anchor_match[5]))
            != sorted(int(v) for v in list(add_match[4]))
        ):
            index += 1
            continue
        anchor_source = str(anchor_match[4])
        if anchor_source != source_input:
            index += 1
            continue
        try:
            nhwc_target = [int(v) for v in list(add_match[4])]
        except Exception:
            index += 1
            continue
        if len(nhwc_target) != 4:
            index += 1
            continue
        cf_target = [nhwc_target[0], nhwc_target[3], nhwc_target[1], nhwc_target[2]]
        anchor_lhs = {str(anchor_match[1]), str(anchor_match[2])}
        if mul_out_match is not None:
            if list(mul_out_match[4]) != list(add_match[4]):
                index += 1
                continue
            mul_inputs = {str(mul_out_match[2]), str(mul_out_match[3])}
            gated_output = str(mul_out_match[1])
            gated_indent = str(mul_out_match[0])
        else:
            assert mul_out_cf_match is not None
            cf_ref = str(mul_out_cf_match[4])
            cf_channel = int(mul_out_cf_match[5])
            if cf_ref not in anchor_lhs or cf_channel != int(cf_target[1]):
                index += 1
                continue
            mul_inputs = {str(mul_out_cf_match[2]), str(mul_out_cf_match[3])}
            gated_output = str(mul_out_cf_match[1])
            gated_indent = str(mul_out_cf_match[0])
        if anchor_lhs != mul_inputs:
            index += 1
            continue
        function_end = _function_end_index(index)
        supported_gap_vars: Set[str] = set()
        safe_to_rewrite = True
        for lookahead_index in range(index + 5, function_end):
            line = rewritten[lookahead_index]
            if not _line_mentions_name(line, gated_output):
                continue
            conv_match = _parse_cf_conv_assign(line)
            if conv_match is not None and str(conv_match[3]) == gated_output:
                continue
            mean_match = _parse_gap_mean_assign(line)
            if mean_match is not None and str(mean_match[2]) == gated_output:
                supported_gap_vars.add(str(mean_match[1]))
                continue
            return_value = _parse_return_identifier(line)
            if return_value == gated_output:
                continue
            anchor_cf_match = anchor_cf_re.match(line) or binary_cf_re.match(line)
            if anchor_cf_match is not None and gated_output in {
                str(anchor_cf_match.group("input0")),
                str(anchor_cf_match.group("input1")),
            }:
                continue
            safe_to_rewrite = False
            break
        if not safe_to_rewrite:
            index += 1
            continue
        for gap_var in supported_gap_vars:
            for lookahead_index in range(index + 5, function_end):
                line = rewritten[lookahead_index]
                if not _line_mentions_name(line, gap_var):
                    continue
                conv_match = _parse_cf_conv_assign(line)
                if conv_match is not None and str(conv_match[3]) == gap_var:
                    continue
                mean_match = _parse_gap_mean_assign(line)
                if mean_match is not None and str(mean_match[1]) == gap_var:
                    continue
                return_value = _parse_return_identifier(line)
                if return_value == gap_var:
                    continue
                safe_to_rewrite = False
                break
            if not safe_to_rewrite:
                break
        if not safe_to_rewrite:
            index += 1
            continue
        indent = str(mul_match[0])
        rewritten[index] = (
            f"{indent}{clamp_out} = torch.nn.functional.hardsigmoid({source_input})"
        )
        rewritten[index + 1] = ""
        rewritten[index + 2] = ""
        rewritten[index + 3] = ""
        rewritten[index + 4] = (
            f"{gated_indent}{gated_output} = torch.mul({source_input}, {clamp_out})"
        )
        for lookahead_index in range(index + 5, function_end):
            conv_match = _parse_cf_conv_assign(rewritten[lookahead_index])
            if conv_match is not None and str(conv_match[3]) == gated_output:
                rewritten[lookahead_index] = (
                    f"{conv_match[0]}{conv_match[1]} = "
                    f"self.{conv_match[2]}({gated_output})"
                )
                continue
            mean_match = _parse_gap_mean_assign(rewritten[lookahead_index])
            if mean_match is not None and str(mean_match[2]) == gated_output:
                gap_var = str(mean_match[1])
                if list(mean_match[3]) == [1, 2]:
                    rewritten[lookahead_index] = (
                        f"{mean_match[0]}{gap_var} = "
                        f"torch.mean({gated_output}, dim=[2, 3], keepdim=True)"
                    )
                continue
            conv_match = _parse_cf_conv_assign(rewritten[lookahead_index])
            if conv_match is not None and str(conv_match[3]) in supported_gap_vars:
                rewritten[lookahead_index] = (
                    f"{conv_match[0]}{conv_match[1]} = "
                    f"self.{conv_match[2]}({conv_match[3]})"
                )
        index += 5
    return [line for line in rewritten if line != ""]


def _rewrite_channel_last_binary_bridge_chains(
    lines: Sequence[str],
    *,
    derive_local_var_name: Callable[[str], str],
    channel_first_constant_expr_for_buffer_attr: Callable[
        [str, Sequence[int]], Optional[str]
    ],
) -> List[str]:
    def _parse_torch_permute_assign(
        line: str,
        expected_perm: Sequence[int],
    ) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        stripped = rhs.strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args: str) -> str | None:
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
            if perm != list(expected_perm):
                return None
            return input_expr

        functional_match = re.fullmatch(r"torch\.permute\((?P<args>.+)\)", stripped)
        if functional_match is not None:
            source_expr = _parse_permute_like_args(str(functional_match.group("args")))
            if source_expr is not None:
                return lhs, source_expr
        helper_match = re.fullmatch(r"_torch_permute\((?P<args>.+)\)", stripped)
        if helper_match is not None:
            source_expr = _parse_permute_like_args(str(helper_match.group("args")))
            if source_expr is not None:
                return lhs, source_expr
        method_match = re.fullmatch(
            r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped
        )
        if method_match is not None and _normalize_permute_dims_expr(
            str(method_match.group("dims"))
        ) == ",".join(str(v) for v in list(expected_perm)):
            return lhs, str(method_match.group("input"))
        return None

    def _parse_nchw_to_nhwc_bridge_source(expr: str) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args: str) -> str | None:
            parts = _split_top_level_csv_exprs(str(args))
            if len(parts) == 2 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
            ):
                source_expr = parts[0].strip()
                dims_expr = _normalize_permute_dims_expr(parts[1])
                if (
                    re.fullmatch(r"[A-Za-z0-9_]+", source_expr) is not None
                    and dims_expr == "0,2,3,1"
                ):
                    return source_expr
                return None
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            source_expr = kwargs.get("input", kwargs.get("x"))
            dims_expr = kwargs.get("dims", kwargs.get("perm"))
            if (
                source_expr is None
                or dims_expr is None
                or re.fullmatch(r"[A-Za-z0-9_]+", source_expr) is None
                or _normalize_permute_dims_expr(dims_expr) != "0,2,3,1"
            ):
                return None
            return source_expr

        functional_match = re.match(r"^torch\.permute\((?P<args>.+)\)$", stripped)
        if functional_match is not None:
            return _parse_permute_like_args(str(functional_match.group("args")))
        helper_match = re.match(r"^_torch_permute\((?P<args>.+)\)$", stripped)
        if helper_match is not None:
            return _parse_permute_like_args(str(helper_match.group("args")))
        method_match = re.match(
            r"^(?P<src>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)$", stripped
        )
        if (
            method_match is not None
            and _normalize_permute_dims_expr(str(method_match.group("dims")))
            == "0,2,3,1"
        ):
            return str(method_match.group("src"))
        return None

    def _parse_conv_input_bridge_assign(line: str) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        match = re.fullmatch(
            r"self\.(?P<module>\w+)\((?P<input>\w+)\.permute\(0,\s*3,\s*1,\s*2\)\.contiguous\(\)\)",
            rhs.strip(),
        )
        if match is not None:
            return lhs, str(match.group("module")), str(match.group("input"))
        stripped = rhs.strip()
        prefix = "self."
        open_paren = stripped.find("(")
        if (
            not stripped.startswith(prefix)
            or open_paren <= len(prefix)
            or not stripped.endswith(")")
        ):
            return None
        module_name = stripped[len(prefix) : open_paren]
        args_expr = stripped[open_paren + 1 : -1]
        input_expr = _resolve_nhwc_to_nchw_bridge_source(args_expr)
        if input_expr is None or not re.fullmatch(r"[A-Za-z0-9_]+", input_expr):
            return None
        return lhs, module_name, input_expr

    def _parse_output_bridge_assign(line: str) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        bridge_source = _parse_nchw_to_nhwc_bridge_source(input_expr)
        if bridge_source is None:
            return None
        return lhs, bridge_source, shape_expr

    def _parse_align_binary_inputs_assign(
        line: str,
    ) -> Tuple[str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^\(*\s*(?P<lhs>[A-Za-z0-9_]+)\s*,\s*(?P<rhs>[A-Za-z0-9_]+)\s*\)*\s*=\s*(?P<expr>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        expr = str(assign_match.group("expr")).strip()
        prefix = "_align_binary_inputs("
        if not expr.startswith(prefix) or not expr.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(expr[len(prefix) : -1])
        lhs_expr: str | None = None
        rhs_expr: str | None = None
        target_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            lhs_expr, rhs_expr, target_expr = (part.strip() for part in parts)
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            lhs_expr = kwargs.get("input", kwargs.get("lhs"))
            rhs_expr = kwargs.get("other", kwargs.get("rhs"))
            target_expr = kwargs.get("target_shape", kwargs.get("shape"))
        target_shape = (
            _parse_rank4_shape_literal(target_expr) if target_expr is not None else None
        )
        if lhs_expr is None or rhs_expr is None or target_shape is None:
            return None
        return (
            str(assign_match.group("lhs")),
            str(assign_match.group("rhs")),
            lhs_expr,
            rhs_expr,
            list(target_shape),
        )

    def _parse_binary_args(expr: str) -> Tuple[str, str] | None:
        parts = _split_top_level_csv_exprs(str(expr))
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            return parts[0].strip(), parts[1].strip()
        kwargs: Dict[str, str] = {}
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            kwargs[key.strip()] = value.strip()
        lhs_expr = kwargs.get("input", kwargs.get("lhs"))
        rhs_expr = kwargs.get("other", kwargs.get("rhs"))
        if lhs_expr is None or rhs_expr is None:
            return None
        return lhs_expr, rhs_expr

    def _parse_binary_align_assign(
        line: str,
    ) -> Tuple[str, str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        target_shape = _parse_rank4_shape_literal(shape_expr)
        if target_shape is None:
            return None
        binary_match = re.fullmatch(
            r"torch\.(?P<fn>add|sub|mul|div|maximum|minimum)\((?P<args>.+)\)",
            input_expr.strip(),
        )
        if binary_match is None:
            return None
        binary_args = _parse_binary_args(str(binary_match.group("args")))
        if binary_args is None:
            return None
        return (
            lhs,
            str(binary_match.group("fn")),
            str(binary_args[0]).strip(),
            str(binary_args[1]).strip(),
            list(target_shape),
        )

    rewritten: List[str] = []
    line_count = len(lines)
    index = 0

    def _resolve_channel_first_constant_expr(
        buffer_expr: str,
        target_shape: Sequence[int],
    ) -> Optional[str]:
        if len(target_shape) != 4:
            return None
        batch_dim = int(target_shape[0])
        candidate_shapes: List[List[int]] = []
        channel_last_candidate = [batch_dim, int(target_shape[3]), 1, 1]
        channel_first_candidate = [batch_dim, int(target_shape[1]), 1, 1]
        for candidate in (channel_last_candidate, channel_first_candidate):
            if candidate not in candidate_shapes:
                candidate_shapes.append(candidate)
        for candidate_shape in candidate_shapes:
            buffer_expr_resolved = channel_first_constant_expr_for_buffer_attr(
                buffer_expr,
                candidate_shape,
            )
            if buffer_expr_resolved is not None:
                return buffer_expr_resolved
        return None

    def _name_use_count(
        candidate_lines: Sequence[str], name: str, *, start: int
    ) -> int:
        pattern = re.compile(rf"\b{re.escape(str(name))}\b")
        return sum(1 for line in candidate_lines[start:] if pattern.search(str(line)))

    while index < line_count:
        if index + 3 < line_count:
            bridge_match = _parse_torch_permute_assign(str(lines[index]), [0, 2, 3, 1])
            aligned_binary_match = _parse_align_binary_inputs_assign(
                str(lines[index + 1])
            )
            binary_match = _parse_binary_align_assign(str(lines[index + 2]))
            conv_match = _parse_conv_input_bridge_assign(str(lines[index + 3]))
            if (
                bridge_match is not None
                and aligned_binary_match is not None
                and binary_match is not None
                and conv_match is not None
                and aligned_binary_match[2].strip() == bridge_match[0]
                and binary_match[0] == conv_match[2]
                and binary_match[2].strip() == aligned_binary_match[0]
                and binary_match[3].strip() == aligned_binary_match[1]
                and aligned_binary_match[4] == binary_match[4]
                and _name_use_count(lines, bridge_match[0], start=index + 1) == 1
                and _name_use_count(lines, aligned_binary_match[0], start=index + 2)
                == 1
                and _name_use_count(lines, aligned_binary_match[1], start=index + 2)
                == 1
                and _name_use_count(lines, binary_match[0], start=index + 3) == 1
            ):
                target_shape = binary_match[4]
                if len(target_shape) == 4 and all(
                    isinstance(v, int) for v in list(target_shape)
                ):
                    source_var = bridge_match[1]
                    buffer_expr = aligned_binary_match[3].strip()
                    cf_constant_expr = _resolve_channel_first_constant_expr(
                        buffer_expr,
                        target_shape,
                    )
                    if cf_constant_expr is not None:
                        binary_cf_var = derive_local_var_name(f"{binary_match[0]}_cf")
                        rewritten.append(
                            f"{binary_cf_var} = torch.{binary_match[1]}({source_var}, {cf_constant_expr})"
                        )
                        rewritten.append(
                            f"{conv_match[0]} = self.{conv_match[1]}({binary_cf_var})"
                        )
                        index += 4
                        continue
        if index + 2 < line_count:
            bridge_match = _parse_torch_permute_assign(str(lines[index]), [0, 2, 3, 1])
            binary_match = _parse_binary_align_assign(str(lines[index + 1]))
            conv_match = _parse_conv_input_bridge_assign(str(lines[index + 2]))
            if (
                bridge_match is not None
                and binary_match is not None
                and conv_match is not None
                and binary_match[0] == conv_match[2]
                and _name_use_count(lines, bridge_match[0], start=index + 1) == 1
                and _name_use_count(lines, binary_match[0], start=index + 2) == 1
            ):
                bridge_var = bridge_match[0]
                source_var = bridge_match[1]
                lhs_expr = binary_match[2].strip()
                rhs_expr = binary_match[3].strip()
                target_shape = binary_match[4]
                if (
                    isinstance(target_shape, list)
                    and len(target_shape) == 4
                    and all(isinstance(v, int) for v in list(target_shape))
                ):
                    buffer_expr = None
                    if lhs_expr == bridge_var and rhs_expr.startswith("self."):
                        buffer_expr = rhs_expr
                    elif rhs_expr == bridge_var and lhs_expr.startswith("self."):
                        buffer_expr = lhs_expr
                    if buffer_expr is not None:
                        cf_constant_expr = _resolve_channel_first_constant_expr(
                            buffer_expr,
                            target_shape,
                        )
                        if cf_constant_expr is not None:
                            binary_cf_var = derive_local_var_name(
                                f"{binary_match[0]}_cf"
                            )
                            cf_lhs = (
                                source_var
                                if lhs_expr == bridge_var
                                else cf_constant_expr
                            )
                            cf_rhs = (
                                source_var
                                if rhs_expr == bridge_var
                                else cf_constant_expr
                            )
                            rewritten.append(
                                f"{binary_cf_var} = torch.{binary_match[1]}({cf_lhs}, {cf_rhs})"
                            )
                            rewritten.append(
                                f"{conv_match[0]} = self.{conv_match[1]}({binary_cf_var})"
                            )
                            index += 3
                            continue
        if index + 3 < line_count:
            output_bridge_match = _parse_output_bridge_assign(str(lines[index + 1]))
            binary_match = _parse_binary_align_assign(str(lines[index + 2]))
            transpose_back_match = _parse_torch_permute_assign(
                str(lines[index + 3]), [0, 3, 1, 2]
            )
            current_line = str(lines[index])
            if (
                output_bridge_match is not None
                and binary_match is not None
                and transpose_back_match is not None
                and output_bridge_match[0]
                in {
                    binary_match[2].strip(),
                    binary_match[3].strip(),
                }
                and binary_match[0] == transpose_back_match[1]
                and _name_use_count(lines, output_bridge_match[0], start=index + 2) == 1
                and _name_use_count(lines, binary_match[0], start=index + 3) == 1
            ):
                target_shape = _parse_rank4_shape_literal(output_bridge_match[2])
                if (
                    target_shape is not None
                    and len(target_shape) == 4
                    and all(isinstance(v, int) for v in list(target_shape))
                ):
                    bridge_var = output_bridge_match[0]
                    cf_source_var = output_bridge_match[1]
                    lhs_expr = binary_match[2].strip()
                    rhs_expr = binary_match[3].strip()
                    buffer_expr = None
                    if lhs_expr == bridge_var and rhs_expr.startswith("self."):
                        buffer_expr = rhs_expr
                    elif rhs_expr == bridge_var and lhs_expr.startswith("self."):
                        buffer_expr = lhs_expr
                    if buffer_expr is not None:
                        cf_constant_expr = _resolve_channel_first_constant_expr(
                            buffer_expr,
                            target_shape,
                        )
                        if cf_constant_expr is not None:
                            rewritten.append(current_line)
                            cf_lhs = (
                                cf_source_var
                                if lhs_expr == bridge_var
                                else cf_constant_expr
                            )
                            cf_rhs = (
                                cf_source_var
                                if rhs_expr == bridge_var
                                else cf_constant_expr
                            )
                            rewritten.append(
                                f"{transpose_back_match[0]} = torch.{binary_match[1]}({cf_lhs}, {cf_rhs})"
                            )
                            index += 4
                            continue
        rewritten.append(str(lines[index]))
        index += 1
    return rewritten


def _repair_channel_last_gap_conv_inputs(
    lines: Sequence[str],
) -> List[str]:
    conv_re = re.compile(
        r"^(?P<indent>\s*)(?P<out>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )

    def _parse_channel_last_reduce_mean_assign(line: str) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        stripped = rhs.strip()
        prefix = "_reduce_mean("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        axes_expr: str | None = None
        keepdims_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "axes":
                    axes_expr = value
                elif key == "keepdims":
                    keepdims_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                axes_expr = part.strip()
            elif positional_index == 2:
                keepdims_expr = part.strip()
            positional_index += 1
        if input_expr is None or axes_expr is None:
            return None
        axes_match = re.fullmatch(
            r"_normalize_axes\(\[(?P<a0>-?\d+),\s*(?P<a1>-?\d+)\],\s*.+\.ndim\)",
            axes_expr,
        )
        if axes_match is None:
            return None
        if [int(axes_match.group("a0")), int(axes_match.group("a1"))] != [1, 2]:
            return None
        if keepdims_expr is not None and keepdims_expr != "True":
            return None
        if re.fullmatch(r"[A-Za-z0-9_]+", input_expr or "") is None:
            return None
        return lhs, str(input_expr)

    def _parse_channel_last_gap_mean_assign(line: str) -> Tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        match = re.fullmatch(r"torch\.mean\((?P<args>.+)\)", rhs.strip())
        if match is None:
            return None
        parts = _split_top_level_csv_exprs(str(match.group("args")))
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    dim_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            positional_index += 1
        if input_expr is None or dim_expr is None:
            return None
        try:
            dim_value = ast.literal_eval(dim_expr)
        except Exception:
            return None
        if not isinstance(dim_value, (list, tuple)):
            return None
        if [int(v) for v in list(dim_value)] != [1, 2]:
            return None
        if keepdim_expr is not None and keepdim_expr != "True":
            return None
        if not re.fullmatch(r"[A-Za-z0-9_]+", input_expr):
            return None
        return lhs, input_expr

    def _parse_dynamic_rank4_binary_add_align_assign_with_ref(
        line: str,
    ) -> Tuple[str, str, str, str, str, int] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        match = re.fullmatch(
            r"_align_tensor_to_target_shape\("
            r"torch\.add\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\), "
            r"\[int\((?P<ref>[A-Za-z0-9_]+)\.shape\[0\]\), (?P<c>\d+), "
            r"int\((?P=ref)\.shape\[2\]\), int\((?P=ref)\.shape\[3\]\)\]\)",
            rhs.strip(),
        )
        if match is None:
            return None
        return (
            str(indent),
            str(lhs),
            str(match.group("a")),
            str(match.group("b")),
            str(match.group("ref")),
            int(match.group("c")),
        )

    def _parse_dynamic_align_binary_inputs_to_anchor_assign_with_ref(
        line: str,
    ) -> Tuple[str, str, str, str, str, str, int] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        match = re.fullmatch(
            r"_align_binary_inputs_to_anchor\("
            r"(?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+), "
            r"\[int\((?P<ref>[A-Za-z0-9_]+)\.shape\[0\]\), (?P<c>\d+), "
            r"int\((?P=ref)\.shape\[2\]\), int\((?P=ref)\.shape\[3\]\)\]\)",
            rhs,
        )
        if match is None:
            return None
        return (
            str(assign_match.group("indent")),
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            str(match.group("a")),
            str(match.group("b")),
            str(match.group("ref")),
            int(match.group("c")),
        )

    rewritten = [str(line) for line in lines]
    channel_last_gap_vars: Set[str] = set()
    channel_last_gap_input_names: Set[str] = set()
    for index, line in enumerate(rewritten):
        parsed_mean = _parse_channel_last_gap_mean_assign(line)
        if parsed_mean is not None:
            channel_last_gap_vars.add(str(parsed_mean[0]))
            channel_last_gap_input_names.add(str(parsed_mean[1]))
            continue
        reduce_mean_var = _parse_channel_last_reduce_mean_assign(line)
        if reduce_mean_var is not None:
            channel_last_gap_vars.add(str(reduce_mean_var[0]))
            channel_last_gap_input_names.add(str(reduce_mean_var[1]))
            continue
        conv_match = conv_re.match(line)
        if conv_match is None:
            continue
        input_var = str(conv_match.group("input"))
        if input_var not in channel_last_gap_vars:
            continue
        rewritten[index] = (
            f"{conv_match.group('indent')}{conv_match.group('out')} = self.{conv_match.group('module')}("
            f"{input_var}.permute(0, 3, 1, 2).contiguous())"
        )

    for gap_input_name in sorted(channel_last_gap_input_names):
        for index, line in enumerate(rewritten):
            dynamic_add_match = _parse_dynamic_rank4_binary_add_align_assign_with_ref(
                line
            )
            if (
                dynamic_add_match is not None
                and str(dynamic_add_match[1]) == gap_input_name
            ):
                dynamic_shape_expr = (
                    f"[int({dynamic_add_match[4]}.shape[0]), "
                    f"int({dynamic_add_match[4]}.shape[2]), "
                    f"int({dynamic_add_match[4]}.shape[3]), {int(dynamic_add_match[5])}]"
                )
                rewritten[index] = (
                    f"{dynamic_add_match[0]}{dynamic_add_match[1]} = _align_tensor_to_target_shape("
                    f"torch.add({dynamic_add_match[2]}, {dynamic_add_match[3]}), {dynamic_shape_expr})"
                )
                for anchor_back in range(max(0, index - 4), index):
                    dynamic_anchor_match = (
                        _parse_dynamic_align_binary_inputs_to_anchor_assign_with_ref(
                            rewritten[anchor_back]
                        )
                    )
                    if dynamic_anchor_match is not None and {
                        str(dynamic_anchor_match[1]),
                        str(dynamic_anchor_match[2]),
                    } == {str(dynamic_add_match[2]), str(dynamic_add_match[3])}:
                        rewritten[anchor_back] = (
                            f"{dynamic_anchor_match[0]}{dynamic_anchor_match[1]}, {dynamic_anchor_match[2]} = "
                            f"_align_binary_inputs_to_anchor({dynamic_anchor_match[3]}, {dynamic_anchor_match[4]}, "
                            f"{dynamic_shape_expr})"
                        )
                        break
                break
            static_add_match = _parse_static_binary_add_align_assign(line)
            if static_add_match is None or str(static_add_match[1]) != gap_input_name:
                continue
            current_shape = [int(v) for v in list(static_add_match[4])]
            if len(current_shape) != 4:
                continue
            normalized_shape = _normalize_nhwc_rank4_shape(current_shape)
            rewritten[index] = (
                f"{static_add_match[0]}{static_add_match[1]} = _align_tensor_to_target_shape("
                f"torch.add({static_add_match[2]}, {static_add_match[3]}), {repr(normalized_shape)})"
            )
            for anchor_back in range(max(0, index - 4), index):
                static_anchor_match = (
                    _parse_align_binary_inputs_to_anchor_assign_with_shape(
                        rewritten[anchor_back]
                    )
                )
                if static_anchor_match is not None and {
                    str(static_anchor_match[1]),
                    str(static_anchor_match[2]),
                } == {str(static_add_match[2]), str(static_add_match[3])}:
                    rewritten[anchor_back] = (
                        f"{static_anchor_match[0]}{static_anchor_match[1]}, {static_anchor_match[2]} = "
                        f"_align_binary_inputs_to_anchor({static_anchor_match[3]}, {static_anchor_match[4]}, "
                        f"{repr(normalized_shape)})"
                    )
                    break
            break
    return rewritten


def _repair_exported_program_direct_conv_cf_add_targets(
    lines: List[str],
) -> List[str]:
    rewritten = list(lines)
    conv_block_decl_re = re.compile(
        r"^\s*self\.(?P<module>[A-Za-z0-9_]+) = _Conv2dBlock\($"
    )
    in_channels_re = re.compile(r"^\s*in_channels=(?P<channels>\d+),$")
    aligned_add_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\(torch\.add\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\), \[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    relu_same_lhs_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.relu\((?P=lhs)\)$"
    )
    module_call_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )

    conv_block_in_channels: Dict[str, int] = {}
    for index, line in enumerate(rewritten):
        conv_block_decl_match = conv_block_decl_re.match(line)
        if conv_block_decl_match is None:
            continue
        module_name = str(conv_block_decl_match.group("module"))
        for lookahead in range(index + 1, min(len(rewritten), index + 12)):
            in_channels_match = in_channels_re.match(rewritten[lookahead])
            if in_channels_match is not None:
                conv_block_in_channels[module_name] = int(
                    in_channels_match.group("channels")
                )
                break

    for index, line in enumerate(rewritten):
        aligned_add_match = aligned_add_re.match(line)
        if aligned_add_match is None:
            continue
        lhs = str(aligned_add_match.group("lhs"))
        relu_index = index + 1
        if relu_index >= len(rewritten):
            continue
        relu_match = relu_same_lhs_re.match(rewritten[relu_index])
        if relu_match is None or str(relu_match.group("lhs")) != lhs:
            continue
        consumer_match = None
        for lookahead in range(relu_index + 1, min(len(rewritten), relu_index + 4)):
            candidate_match = module_call_re.match(rewritten[lookahead])
            if candidate_match is not None and str(
                candidate_match.group("input")
            ) == lhs:
                consumer_match = candidate_match
                break
        if consumer_match is None:
            continue
        in_channels = conv_block_in_channels.get(
            str(consumer_match.group("module")), None
        )
        if in_channels is None:
            continue
        n = int(aligned_add_match.group("n"))
        d1 = int(aligned_add_match.group("d1"))
        d2 = int(aligned_add_match.group("d2"))
        d3 = int(aligned_add_match.group("d3"))
        if d2 != int(in_channels) or d1 == int(in_channels):
            continue
        rewritten[index] = (
            f"{aligned_add_match.group('indent')}{lhs} = _align_tensor_to_target_shape("
            f"torch.add({aligned_add_match.group('a')}, {aligned_add_match.group('b')}), "
            f"[{n}, {d2}, {d3}, {d1}])"
        )
    return rewritten


def _prune_dead_forward_lines(
    lines: Sequence[str],
    *,
    input_var_names: Sequence[str],
    output_var_names: Sequence[str],
) -> List[str]:
    if len(lines) == 0:
        return []

    parsed_statements: List[ast.stmt] = []
    top_level_assigned_names: List[List[str]] = []
    raw_used_names: List[List[str]] = []
    for line in lines:
        statement = ast.parse(str(line)).body[0]
        parsed_statements.append(statement)
        top_level_assigned_names.append(_extract_statement_assignments(statement))
        raw_used_names.append(_extract_statement_loads(statement))

    local_name_candidates: Set[str] = {str(name) for name in list(input_var_names)}
    local_name_candidates.update(str(name) for name in list(output_var_names))
    for assigned_names in top_level_assigned_names:
        local_name_candidates.update(str(name) for name in assigned_names)

    assigned_names_by_line: List[List[str]] = []
    used_names_by_line: List[List[str]] = []
    for assigned_names, used_names in zip(top_level_assigned_names, raw_used_names):
        assigned_filtered = [
            str(name) for name in assigned_names if str(name) in local_name_candidates
        ]
        used_filtered = [
            str(name) for name in used_names if str(name) in local_name_candidates
        ]
        assigned_names_by_line.append(assigned_filtered)
        used_names_by_line.append(used_filtered)

    live_names: Set[str] = {str(name) for name in list(output_var_names)}
    kept_lines_reversed: List[str] = []
    for line_index in range(len(lines) - 1, -1, -1):
        assigned_names = assigned_names_by_line[line_index]
        used_names = used_names_by_line[line_index]
        if len(assigned_names) > 0 and all(
            str(name) not in live_names for name in assigned_names
        ):
            continue
        kept_lines_reversed.append(str(lines[line_index]))
        for name in assigned_names:
            live_names.discard(str(name))
        live_names.update(str(name) for name in used_names)
    kept_lines_reversed.reverse()
    return kept_lines_reversed
