from __future__ import annotations

import ast
import re
from typing import Dict, List, Sequence, Set, Tuple

from onnx2tf.tflite_builder.pytorch_source_parser import (
    _normalize_permute_dims_expr,
    _parse_align_tensor_target_shape_expr,
    _parse_rank4_shape_literal,
    _parse_simple_assignment_line,
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
