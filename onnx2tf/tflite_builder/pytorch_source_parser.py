from __future__ import annotations

import functools
import re
from typing import Dict, List, Tuple


def _parse_int_list_literal(text: str) -> List[int]:
    return [int(value.strip()) for value in str(text).split(",") if value.strip()]


def _strip_outer_parentheses(expr: str) -> str:
    text = str(expr).strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        balanced = True
        for index, char in enumerate(text):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
                if depth == 0 and index != len(text) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        text = text[1:-1].strip()
    return text


def _split_top_level_csv_exprs(expr: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in str(expr):
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_binary_add_args(expr: str) -> Tuple[str, str] | None:
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
    input_expr = kwargs.get("input", None)
    other_expr = kwargs.get("other", None)
    if input_expr is None or other_expr is None:
        return None
    return input_expr, other_expr


def _parse_binary_mul_args(expr: str) -> Tuple[str, str] | None:
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
    input_expr = kwargs.get("input", None)
    other_expr = kwargs.get("other", None)
    if input_expr is None or other_expr is None:
        return None
    return input_expr, other_expr


def _parse_align_tensor_target_shape_expr(expr: str) -> Tuple[str, str] | None:
    stripped = str(expr).strip()
    prefix = "_align_tensor_to_target_shape("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
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
    input_expr = kwargs.get("input", None)
    target_shape_expr = kwargs.get("target_shape", None)
    if input_expr is None or target_shape_expr is None:
        return None
    return input_expr, target_shape_expr


@functools.lru_cache(maxsize=131072)
def _parse_simple_assignment_line_cached(
    line: str,
) -> Tuple[str, str, str] | None:
    current_line = str(line)
    line_length = len(current_line)
    indent_end = 0
    while indent_end < line_length and current_line[indent_end].isspace():
        indent_end += 1
    if indent_end >= line_length:
        return None
    lhs_start = indent_end
    first_char = current_line[lhs_start]
    if not (first_char.isalnum() or first_char == "_"):
        return None
    lhs_end = lhs_start + 1
    while lhs_end < line_length:
        current_char = current_line[lhs_end]
        if not (current_char.isalnum() or current_char == "_"):
            break
        lhs_end += 1
    lhs = current_line[lhs_start:lhs_end]
    cursor = lhs_end
    while cursor < line_length and current_line[cursor].isspace():
        cursor += 1
    if cursor < line_length and current_line[cursor] == ":":
        cursor += 1
        equal_index = current_line.find("=", cursor)
        if equal_index < 0:
            return None
    else:
        equal_index = cursor
        while equal_index < line_length and current_line[equal_index].isspace():
            equal_index += 1
        if equal_index >= line_length or current_line[equal_index] != "=":
            return None
    rhs = current_line[equal_index + 1 :].strip()
    if not rhs:
        return None
    return current_line[:indent_end], lhs, rhs


def _parse_simple_assignment_line(line: str) -> Tuple[str, str, str] | None:
    return _parse_simple_assignment_line_cached(str(line))


def _parse_rank4_shape_literal(shape_expr: str) -> Tuple[int, int, int, int] | None:
    shape_match = re.fullmatch(
        r"[\[\(]\s*(?P<n>\d+)\s*,\s*(?P<d1>\d+)\s*,\s*(?P<d2>\d+)\s*,\s*(?P<d3>\d+)\s*[\]\)]",
        str(shape_expr).strip(),
    )
    if shape_match is None:
        return None
    return (
        int(shape_match.group("n")),
        int(shape_match.group("d1")),
        int(shape_match.group("d2")),
        int(shape_match.group("d3")),
    )


def _parse_apply_concat_inputs_axis_and_shape(
    expr: str,
) -> Tuple[List[str], int, List[int] | None] | None:
    stripped = str(expr).strip()
    prefix = "_apply_concat("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    inputs_expr: str | None = None
    axis_expr: str | None = None
    shape_expr: str | None = None
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        inputs_expr = parts[0].strip()
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "inputs":
            inputs_expr = value
        elif key == "axis":
            axis_expr = value
        elif key == "target_shape":
            shape_expr = value
    if inputs_expr is None or axis_expr is None:
        return None
    inputs_match = re.fullmatch(r"[\[\(](?P<inputs>.+)[\]\)]", inputs_expr)
    if inputs_match is None:
        return None
    try:
        axis_value = int(axis_expr)
    except ValueError:
        return None
    shape_values = None
    if shape_expr is not None:
        shape_match = re.fullmatch(r"[\[\(](?P<shape>[0-9,\s]+)[\]\)]", shape_expr)
        if shape_match is not None:
            shape_values = _parse_int_list_literal(str(shape_match.group("shape")))
    return (
        _split_top_level_csv_exprs(str(inputs_match.group("inputs"))),
        axis_value,
        shape_values,
    )


def _parse_torch_cat_inputs_and_dim(expr: str) -> Tuple[List[str], int] | None:
    stripped = str(expr).strip()
    prefix = "torch.cat("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    list_expr: str | None = None
    dim_expr: str | None = None
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        list_expr = parts[0].strip()
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "tensors":
            list_expr = value
        elif key == "dim":
            dim_expr = value
    if list_expr is None or dim_expr is None:
        return None
    list_match = re.fullmatch(r"[\[\(](?P<inputs>.+)[\]\)]", list_expr)
    if list_match is None:
        return None
    try:
        dim_value = int(dim_expr)
    except ValueError:
        return None
    return _split_top_level_csv_exprs(str(list_match.group("inputs"))), dim_value


def _normalize_permute_dims_expr(expr: str) -> str:
    stripped = str(expr).strip()
    if (stripped.startswith("(") and stripped.endswith(")")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        stripped = stripped[1:-1].strip()
    return re.sub(r"\s+", "", stripped)
