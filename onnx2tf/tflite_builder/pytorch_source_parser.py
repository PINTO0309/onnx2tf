from __future__ import annotations

import ast
import functools
import re
from typing import Dict, List, Tuple


_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN = r"[^,]+"


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


def _parse_channel_last_gather_slice_assign(line: str) -> Tuple[str, str, str] | None:
    assign = _parse_simple_assignment_line(line)
    if assign is None:
        return None
    _, lhs, rhs = assign
    slice_match = re.fullmatch(
        r"(?P<input>[A-Za-z0-9_]+)\[:,\s*:\s*,\s*:\s*,\s*\[(?P<indices>[0-9,\s-]+)\]\]",
        str(rhs).strip(),
    )
    if slice_match is None:
        return None
    return (
        lhs,
        str(slice_match.group("input")),
        str(slice_match.group("indices")).strip(),
    )


def _parse_rank4_shape_expr(
    shape_expr: str,
) -> Tuple[str, int, int, int] | None:
    shape_match = re.fullmatch(
        rf"[\[\(]\s*(?P<n>{_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN}|\d+)\s*,\s*(?P<d1>\d+)\s*,\s*(?P<d2>\d+)\s*,\s*(?P<d3>\d+)\s*[\]\)]",
        str(shape_expr).strip(),
    )
    if shape_match is None:
        return None
    return (
        str(shape_match.group("n")).strip(),
        int(shape_match.group("d1")),
        int(shape_match.group("d2")),
        int(shape_match.group("d3")),
    )


def _parse_apply_resize_input_size_shape_and_channel_last(
    expr: str,
) -> Tuple[str, Tuple[int, int] | None, Tuple[str, int, int, int] | None, bool] | None:
    stripped = str(expr).strip()
    prefix = "_apply_resize("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    input_expr: str | None = None
    size_expr: str | None = None
    shape_expr: str | None = None
    channel_last_expr: str | None = None
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        input_expr = parts[0].strip()
    if len(parts) >= 2 and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[1]) is None:
        size_expr = parts[1].strip()
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "input":
            input_expr = value
        elif key == "size":
            size_expr = value
        elif key == "target_shape":
            shape_expr = value
        elif key == "channel_last":
            channel_last_expr = value
    if input_expr is None or channel_last_expr not in {"True", "False"}:
        return None
    size_value = None
    if size_expr is not None:
        size_match = re.fullmatch(
            r"[\[\(]\s*(?P<h>\d+)\s*,\s*(?P<w>\d+)\s*[\]\)]", size_expr
        )
        if size_match is not None:
            size_value = (int(size_match.group("h")), int(size_match.group("w")))
    shape_value = (
        _parse_rank4_shape_expr(shape_expr) if shape_expr is not None else None
    )
    return input_expr, size_value, shape_value, channel_last_expr == "True"


def _parse_apply_pool2d_input_channel_last_and_is_max(
    expr: str,
) -> Tuple[str, bool, bool | None] | None:
    stripped = str(expr).strip()
    prefix = "_apply_pool2d("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    input_expr: str | None = None
    channel_last_expr: str | None = None
    is_max_expr: str | None = None
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        input_expr = parts[0].strip()
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "input":
            input_expr = value
        elif key == "channel_last":
            channel_last_expr = value
        elif key == "is_max_pool":
            is_max_expr = value
    if input_expr is None or channel_last_expr not in {"True", "False"}:
        return None
    is_max_value = None
    if is_max_expr in {"True", "False"}:
        is_max_value = is_max_expr == "True"
    return input_expr, channel_last_expr == "True", is_max_value


def _parse_apply_pool2d_assign_with_shape(
    line: str,
) -> Tuple[str, str, str, str, List[int], bool, bool] | None:
    assign = _parse_simple_assignment_line(line)
    if assign is None:
        return None
    indent, lhs, rhs = assign
    stripped = rhs.strip()
    prefix = "_apply_pool2d("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parsed = _parse_apply_pool2d_input_channel_last_and_is_max(rhs)
    if parsed is None or parsed[2] is None:
        return None
    input_name, channel_last, is_max_pool = parsed
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    target_shape_expr: str | None = None
    rest_parts: List[str] = []
    positional_index = 0
    for part in parts:
        keyword_match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part)
        if keyword_match is not None:
            key, value = part.split("=", 1)
            if key.strip() == "target_shape":
                target_shape_expr = value.strip()
                continue
            if key.strip() in {"input", "is_max_pool", "channel_last"}:
                continue
            rest_parts.append(part.strip())
            continue
        if positional_index == 0:
            positional_index += 1
            continue
        rest_parts.append(part.strip())
        positional_index += 1
    shape_value = (
        _parse_rank4_shape_expr(target_shape_expr)
        if target_shape_expr is not None
        else None
    )
    if shape_value is None:
        return None
    return (
        indent,
        lhs,
        input_name,
        ", ".join(rest_parts),
        [int(v) for v in list(shape_value)],
        bool(is_max_pool),
        channel_last,
    )


def _parse_tensor_split_assign(
    line: str,
) -> Tuple[str, List[str], str, int, int] | None:
    assign_match = re.match(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_, ]+)\s*=\s*(?P<rhs>.+)$",
        str(line),
    )
    if assign_match is None:
        return None
    indent = str(assign_match.group("indent"))
    lhs = str(assign_match.group("lhs"))
    rhs = str(assign_match.group("rhs"))
    list_match = re.fullmatch(
        r"list\(torch\.tensor_split\((?P<args>.+)\)\)", rhs.strip()
    )
    if list_match is None:
        return None
    parts = _split_top_level_csv_exprs(str(list_match.group("args")))
    input_expr: str | None = None
    sections_expr: str | None = None
    dim_expr: str | None = None
    positional_index = 0
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"input", "tensor"}:
                input_expr = value
            elif key in {"indices_or_sections", "sections"}:
                sections_expr = value
            elif key == "dim":
                dim_expr = value
            continue
        if positional_index == 0:
            input_expr = part.strip()
        elif positional_index == 1:
            sections_expr = part.strip()
        elif positional_index == 2:
            dim_expr = part.strip()
        positional_index += 1
    if input_expr is None or sections_expr is None or dim_expr is None:
        return None
    if re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
        return None
    sections_match = re.fullmatch(r"\d+", sections_expr)
    dim_match = re.fullmatch(
        rf"_normalize_dim\(\s*(?P<axis>-?\d+)\s*,\s*{re.escape(input_expr)}\.ndim\s*\)",
        dim_expr,
    )
    if sections_match is None or dim_match is None:
        return None
    outputs = [token.strip() for token in lhs.split(",") if token.strip()]
    if len(outputs) == 0:
        return None
    return (
        indent,
        outputs,
        input_expr,
        int(sections_match.group(0)),
        int(dim_match.group("axis")),
    )


def _parse_apply_softmax_input_axis_and_shape(
    expr: str,
) -> Tuple[str, int, Tuple[str, int, int, int] | None] | None:
    stripped = str(expr).strip()
    prefix = "_apply_softmax("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    input_expr: str | None = None
    axis_expr: str | None = None
    shape_expr: str | None = None
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        input_expr = parts[0].strip()
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "input":
            input_expr = value
        elif key == "axis":
            axis_expr = value
        elif key == "target_shape":
            shape_expr = value
    if input_expr is None or axis_expr is None:
        return None
    try:
        axis_value = int(axis_expr)
    except ValueError:
        return None
    rank4_shape = (
        _parse_rank4_shape_expr(shape_expr) if shape_expr is not None else None
    )
    return input_expr, axis_value, rank4_shape


def _resolve_nhwc_to_nchw_bridge_source(expr: str) -> str | None:
    stripped = str(expr).strip()
    if stripped.endswith(".contiguous()"):
        stripped = stripped[: -len(".contiguous()")].strip()

    def _parse_functional_permute_args(args: str) -> str | None:
        parts = _split_top_level_csv_exprs(str(args))
        if len(parts) == 2 and "=" not in parts[0] and "=" not in parts[1]:
            source_expr = parts[0].strip()
            dims_expr = _normalize_permute_dims_expr(parts[1])
            if (
                re.match(r"^[A-Za-z0-9_]+$", source_expr) is not None
                and dims_expr == "0,3,1,2"
            ):
                return source_expr
            return None

        kwargs: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            kwargs[key.strip()] = value.strip()
        source_expr = kwargs.get("input", None)
        if source_expr is None:
            source_expr = kwargs.get("x", None)
        dims_expr = kwargs.get("dims", None)
        if dims_expr is None:
            dims_expr = kwargs.get("perm", None)
        if source_expr is None or dims_expr is None:
            return None
        if re.match(r"^[A-Za-z0-9_]+$", source_expr) is None:
            return None
        if _normalize_permute_dims_expr(dims_expr) != "0,3,1,2":
            return None
        return source_expr

    functional_match = re.match(
        r"^torch\.permute\((?P<args>.+)\)$",
        stripped,
    )
    if functional_match is not None:
        return _parse_functional_permute_args(str(functional_match.group("args")))

    helper_match = re.match(
        r"^_torch_permute\((?P<args>.+)\)$",
        stripped,
    )
    if helper_match is not None:
        return _parse_functional_permute_args(str(helper_match.group("args")))

    method_match = re.match(
        r"^(?P<src>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)$",
        stripped,
    )
    if method_match is not None:
        dims_expr = _normalize_permute_dims_expr(str(method_match.group("dims")))
        if dims_expr == "0,3,1,2":
            return str(method_match.group("src"))
        return None

    return None


def _parse_copy_call_expr(
    line: str,
) -> Tuple[str, str, str, str, str] | None:
    current_line = str(line)
    indent = current_line[: len(current_line) - len(current_line.lstrip())]
    stripped = current_line.strip()
    copy_token = ".copy_("
    copy_index = stripped.find(copy_token)
    if copy_index <= 0 or not stripped.endswith(")"):
        return None
    target_expr = stripped[:copy_index].strip()
    args_expr = stripped[copy_index + len(copy_token) : -1].strip()
    normalized_target = _strip_outer_parentheses(target_expr)
    buffer_name = (
        normalized_target[5:]
        if normalized_target.startswith("self.")
        else normalized_target
    )
    if re.fullmatch(r"[A-Za-z0-9_]+", buffer_name) is None:
        return None
    parts = _split_top_level_csv_exprs(args_expr)
    if not parts:
        return None
    src_expr = parts[0].strip()
    copy_kwargs_parts: List[str] = []
    for part in parts[1:]:
        stripped_part = part.strip()
        if not stripped_part:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped_part) is None:
            return None
        copy_kwargs_parts.append(stripped_part)
    copy_kwargs = "".join(f", {part}" for part in copy_kwargs_parts)
    return indent, target_expr, buffer_name, src_expr, copy_kwargs


def _parse_align_tensor_target_shape_assign(line: str) -> Tuple[str, str] | None:
    current_line = str(line)
    if "=" not in current_line:
        return None
    lhs, rhs = current_line.split("=", 1)
    if re.fullmatch(r"\s*[A-Za-z0-9_]+\s*", lhs) is None:
        return None
    return _parse_align_tensor_target_shape_expr(rhs.strip())


@functools.lru_cache(maxsize=131072)
def _parse_torch_permute_assign(
    line: str,
) -> Tuple[str, str, str, List[int]] | None:
    assign = _parse_simple_assignment_line(line)
    if assign is None:
        return None
    indent, lhs, rhs = assign
    stripped = rhs.strip()
    if stripped.endswith(".contiguous()"):
        stripped = stripped[: -len(".contiguous()")].strip()

    def _parse_permute_like_args(args_expr: str) -> Tuple[str, List[int]] | None:
        parts = _split_top_level_csv_exprs(args_expr)
        input_expr: str | None = None
        perm_expr: str | None = None
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_expr = parts[0].strip()
            perm_expr = parts[1].strip()
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
        if re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
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
            parsed = _parse_permute_like_args(stripped[len(prefix) : -1])
            if parsed is not None:
                return indent, lhs, parsed[0], parsed[1]
    method_match = re.fullmatch(
        r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<perm>.+)\)",
        stripped,
    )
    if method_match is None:
        return None
    if re.fullmatch(r"[A-Za-z0-9_]+", str(method_match.group("input"))) is None:
        return None
    try:
        perm = [
            int(v)
            for v in _normalize_permute_dims_expr(
                str(method_match.group("perm"))
            ).split(",")
        ]
    except Exception:
        return None
    return indent, lhs, str(method_match.group("input")), perm


def _parse_local_response_norm_input_expr(expr: str) -> str | None:
    stripped = str(expr).strip()
    prefix = "F.local_response_norm("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    input_expr: str | None = None
    positional_index = 0
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
            key, value = part.split("=", 1)
            if key.strip() == "input":
                input_expr = value.strip()
            continue
        if positional_index == 0:
            input_expr = part.strip()
        positional_index += 1
    if input_expr is None or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
        return None
    return input_expr


def _parse_apply_pool2d_input_expr(expr: str) -> str | None:
    stripped = str(expr).strip()
    prefix = "_apply_pool2d("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        return parts[0].strip()

    kwargs: Dict[str, str] = {}
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        kwargs[key.strip()] = value.strip()
    return kwargs.get("input")


def _parse_apply_resize_input_and_channel_last(expr: str) -> Tuple[str, bool] | None:
    stripped = str(expr).strip()
    prefix = "_apply_resize("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    input_expr: str | None = None
    channel_last_expr: str | None = None
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        input_expr = parts[0].strip()
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "input":
            input_expr = value
        elif key == "channel_last":
            channel_last_expr = value
    if input_expr is None or channel_last_expr not in {"True", "False"}:
        return None
    return input_expr, channel_last_expr == "True"


def _parse_apply_pool2d_input_and_channel_last(expr: str) -> Tuple[str, bool] | None:
    stripped = str(expr).strip()
    prefix = "_apply_pool2d("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    input_expr: str | None = None
    channel_last_expr: str | None = None
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        input_expr = parts[0].strip()
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "input":
            input_expr = value
        elif key == "channel_last":
            channel_last_expr = value
    if input_expr is None or channel_last_expr not in {"True", "False"}:
        return None
    return input_expr, channel_last_expr == "True"


def _parse_apply_softmax_input_and_axis(expr: str) -> Tuple[str, int] | None:
    stripped = str(expr).strip()
    prefix = "_apply_softmax("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    input_expr: str | None = None
    axis_expr: str | None = None
    if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
        input_expr = parts[0].strip()
    for part in parts:
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "input":
            input_expr = value
        elif key == "axis":
            axis_expr = value
    if input_expr is None or axis_expr is None:
        return None
    try:
        axis_value = int(axis_expr)
    except ValueError:
        return None
    return input_expr, axis_value


def _parse_constant_pad_assign(
    line: str,
) -> Tuple[str, str, str, List[int], str] | None:
    assign = _parse_simple_assignment_line(line)
    if assign is None:
        return None
    indent, lhs, rhs = assign
    stripped = rhs.strip()
    prefix = "F.pad("
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
    input_expr: str | None = None
    pad_expr: str | None = None
    mode_expr: str | None = None
    value_expr: str | None = None
    positional_index = 0
    for part in parts:
        keyword_match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part)
        if keyword_match is not None:
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "input":
                input_expr = value
            elif key == "pad":
                pad_expr = value
            elif key == "mode":
                mode_expr = value
            elif key == "value":
                value_expr = value
            continue
        if positional_index == 0:
            input_expr = part.strip()
        elif positional_index == 1:
            pad_expr = part.strip()
        elif positional_index == 2:
            mode_expr = part.strip()
        elif positional_index == 3:
            value_expr = part.strip()
        positional_index += 1
    if (
        input_expr is None
        or pad_expr is None
        or mode_expr != "'constant'"
        or value_expr is None
    ):
        return None
    pad_match = re.fullmatch(r"[\[\(](?P<values>[0-9,\s-]+)[\]\)]", pad_expr.strip())
    if pad_match is None:
        return None
    return (
        indent,
        lhs,
        input_expr,
        _parse_int_list_literal(str(pad_match.group("values"))),
        value_expr,
    )


def _parse_dynamic_binary_add_align_assign(
    line: str,
) -> Tuple[str, str, str, str, int] | None:
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
        indent,
        lhs,
        str(match.group("a")),
        str(match.group("b")),
        int(match.group("c")),
    )


def _parse_static_binary_add_align_assign(
    line: str,
) -> Tuple[str, str, str, str, List[int]] | None:
    assign = _parse_simple_assignment_line(line)
    if assign is None:
        return None
    indent, lhs, rhs = assign
    align_parts = _parse_align_tensor_target_shape_expr(rhs)
    if align_parts is None:
        return None
    input_expr, target_shape_expr = align_parts
    add_match = re.fullmatch(
        r"torch\.add\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\)", input_expr.strip()
    )
    target_shape = _parse_rank4_shape_literal(target_shape_expr)
    if add_match is None or target_shape is None:
        return None
    return (
        indent,
        lhs,
        str(add_match.group("a")),
        str(add_match.group("b")),
        [int(v) for v in list(target_shape)],
    )


def _parse_align_binary_inputs_to_anchor_assign_with_shape(
    line: str,
) -> Tuple[str, str, str, str, str, List[int]] | None:
    assign_match = re.match(
        r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*\)*\s*=\s*(?P<rhs>.+)$",
        str(line),
    )
    if assign_match is None:
        return None
    rhs = str(assign_match.group("rhs")).strip()
    prefix = "_align_binary_inputs_to_anchor("
    if not rhs.startswith(prefix) or not rhs.endswith(")"):
        return None
    parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
    if len(parts) != 3:
        return None
    target_shape = _parse_rank4_shape_literal(parts[2].strip())
    if (
        target_shape is None
        or re.fullmatch(r"[A-Za-z0-9_]+", parts[0].strip()) is None
        or re.fullmatch(r"[A-Za-z0-9_]+", parts[1].strip()) is None
    ):
        return None
    return (
        str(assign_match.group("indent")),
        str(assign_match.group("lhs0")),
        str(assign_match.group("lhs1")),
        parts[0].strip(),
        parts[1].strip(),
        [int(v) for v in list(target_shape)],
    )
