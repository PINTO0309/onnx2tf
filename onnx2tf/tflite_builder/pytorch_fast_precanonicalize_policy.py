from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

from onnx2tf.tflite_builder.pytorch_source_parser import (
    _parse_apply_concat_inputs_axis_and_shape,
    _parse_apply_pool2d_assign_with_shape,
    _parse_apply_pool2d_input_and_channel_last,
    _parse_apply_resize_input_and_channel_last,
    _parse_apply_resize_input_size_shape_and_channel_last,
    _parse_apply_softmax_input_and_axis,
    _parse_align_binary_inputs_to_anchor_assign_with_shape,
    _parse_align_tensor_target_shape_expr,
    _parse_binary_add_args,
    _parse_binary_mul_args,
    _parse_constant_pad_assign,
    _parse_int_list_literal,
    _parse_local_response_norm_input_expr,
    _parse_rank4_shape_literal,
    _parse_simple_assignment_line,
    _parse_tensor_split_assign,
    _parse_torch_cat_inputs_and_dim,
    _parse_torch_permute_assign,
    _split_top_level_csv_exprs,
)
from onnx2tf.tflite_builder.pytorch_shape_policy import (
    _fast_precanonicalize_rank4_layout_hint,
    _normalize_cf_rank4_shape,
    _normalize_nhwc_rank4_shape,
)


def _convert_nhwc_pad_to_nchw_pad_values(pad_values: Sequence[int]) -> List[int] | None:
    values = [int(value) for value in list(pad_values)]
    if len(values) % 2 != 0 or len(values) > 8:
        return None
    nhwc_inner_to_outer = ["C", "W", "H", "N"]
    nchw_inner_to_outer = ["W", "H", "C", "N"]
    semantic_pairs = {name: [0, 0] for name in nhwc_inner_to_outer}
    pair_count = len(values) // 2
    for idx in range(pair_count):
        semantic_pairs[nhwc_inner_to_outer[idx]] = [
            int(values[idx * 2]),
            int(values[idx * 2 + 1]),
        ]
    output_pairs = [semantic_pairs[name] for name in nchw_inner_to_outer]
    last_nonzero = -1
    for idx, pair in enumerate(output_pairs):
        if pair != [0, 0]:
            last_nonzero = idx
    if last_nonzero < 0:
        return []
    flattened: List[int] = []
    for pair in output_pairs[: last_nonzero + 1]:
        flattened.extend([int(pair[0]), int(pair[1])])
    return flattened

def _convert_nchw_pad_to_nhwc_pad_values(pad_values: Sequence[int]) -> List[int] | None:
    values = [int(value) for value in list(pad_values)]
    if len(values) % 2 != 0 or len(values) > 8:
        return None
    nchw_inner_to_outer = ["W", "H", "C", "N"]
    nhwc_inner_to_outer = ["C", "W", "H", "N"]
    semantic_pairs = {name: [0, 0] for name in nchw_inner_to_outer}
    pair_count = len(values) // 2
    for idx in range(pair_count):
        semantic_pairs[nchw_inner_to_outer[idx]] = [
            int(values[idx * 2]),
            int(values[idx * 2 + 1]),
        ]
    output_pairs = [semantic_pairs[name] for name in nhwc_inner_to_outer]
    last_nonzero = -1
    for idx, pair in enumerate(output_pairs):
        if pair != [0, 0]:
            last_nonzero = idx
    if last_nonzero < 0:
        return []
    flattened: List[int] = []
    for pair in output_pairs[: last_nonzero + 1]:
        flattened.extend([int(pair[0]), int(pair[1])])
    return flattened

def _infer_unique_channel_count_from_rank4_shape(shape: Sequence[int]) -> int | None:
    shape_values = [int(value) for value in list(shape)]
    if len(shape_values) != 4:
        return None
    dims = [int(value) for value in shape_values[1:]]
    candidates = [
        int(value)
        for value in dims
        if int(value) > 1 and dims.count(int(value)) == 1
    ]
    if len(candidates) == 1:
        return int(candidates[0])
    if len(candidates) > 1:
        return int(max(candidates))
    return None

def _fast_precanonicalize_expr_identifiers(expr: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(expr))
        if token not in {"torch", "self", "True", "False"}
    }


def _has_immediate_rank4_permute_source(
    lines: Sequence[str],
    index: int,
    tensor_name: str,
    expected_perm: Sequence[int],
) -> bool:
    lookback = index - 1
    while lookback >= 0:
        candidate_line = str(lines[lookback]).strip()
        if candidate_line == "":
            lookback -= 1
            continue
        permute_assign = _parse_torch_permute_assign(lines[lookback])
        return (
            permute_assign is not None
            and permute_assign[1] == str(tensor_name)
            and list(permute_assign[3]) == [int(v) for v in list(expected_perm)]
        )
    return False


@dataclasses.dataclass(slots=True)
class _FastPrecanonicalizeRepairContext:
    aliases: Dict[str, str]
    consumers: Dict[str, List[int]]
    static_shapes: Dict[str, List[int]]
    cf_like_names: Set[str]
    nhwc_like_names: Set[str]
    const_channel_counts: Dict[str, int]
    conv_block_in_channels: Dict[str, int]
    conv_block_out_channels: Dict[str, int]
    module_output_producers: Dict[str, str]
    module_input_consumers: Dict[str, List[str]]


def _build_fast_precanonicalize_repair_context(
    lines: Sequence[str],
) -> _FastPrecanonicalizeRepairContext:
    register_buffer_re = re.compile(
        r"^\s*self\.register_buffer\('(?P<name>[A-Za-z0-9_]+)', torch\.zeros\(\[(?P<shape>[0-9, ]+)\], dtype=torch\.[A-Za-z0-9_]+\), persistent=(?:True|False)\)$"
    )
    conv_block_decl_re = re.compile(r"^\s*self\.(?P<module>conv_block_[0-9]+) = _Conv2dBlock\($")
    in_channels_re = re.compile(r"^\s*in_channels=(?P<channels>\d+),$")
    out_channels_re = re.compile(r"^\s*out_channels=(?P<channels>\d+),$")
    module_output_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\)"
    )
    generic_expr_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+)$"
    )
    simple_alias_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>[A-Za-z0-9_]+)$"
    )
    aligned_rank4_any_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\((?P<expr>.+), \[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    apply_pool2d_re = re.compile(  # noqa: F841 - retained for AST-compatible move
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_pool2d\((?:input=)?(?P<input>[A-Za-z0-9_]+), (?P<rest>.+), target_shape=[\[\(](?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)[\]\)], is_max_pool=(?P<is_max>True|False), channel_last=(?P<channel_last>True|False)\)$"
    )
    apply_softmax_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_softmax\((?:input=)?(?P<input>[A-Za-z0-9_]+), axis=(?P<axis>-?\d+), beta=(?P<beta>[-0-9.eE]+), target_shape=[\[\(](?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)[\]\)]\)$"
    )
    const_pad_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*F\.pad\((?P<input>[A-Za-z0-9_]+), \[(?P<pads>[0-9,\s]+)\], mode='constant', value=(?P<value>[-+0-9.eE]+)\)$"
    )
    aliases: Dict[str, str] = {}
    consumers: Dict[str, List[int]] = {}
    static_shapes: Dict[str, List[int]] = {}
    cf_like_names: Set[str] = set()
    nhwc_like_names: Set[str] = set()
    const_channel_counts: Dict[str, int] = {}
    conv_block_in_channels: Dict[str, int] = {}
    conv_block_out_channels: Dict[str, int] = {}
    module_output_producers: Dict[str, str] = {}
    module_input_consumers: Dict[str, List[str]] = {}

    def _parse_apply_resize_assign(
        src_line: str,
    ) -> Tuple[str, str, str, int, int, str, List[int], bool, bool, bool] | None:
        assign = _parse_simple_assignment_line(src_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        parsed = _parse_apply_resize_input_size_shape_and_channel_last(rhs)
        if parsed is None:
            return None
        input_name, size_value, shape_value, channel_last = parsed
        if size_value is None or shape_value is None:
            return None
        stripped = rhs.strip()
        if not stripped.startswith("_apply_resize(") or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len("_apply_resize(") : -1])
        method_expr: str | None = None
        align_expr: str | None = None
        hpc_expr: str | None = None
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "method":
                method_expr = value
            elif key == "align_corners":
                align_expr = value
            elif key == "half_pixel_centers":
                hpc_expr = value
        if (
            method_expr is None
            or align_expr not in {"True", "False"}
            or hpc_expr not in {"True", "False"}
            or not (method_expr.startswith("'") and method_expr.endswith("'"))
        ):
            return None
        return (
            indent,
            lhs,
            input_name,
            int(size_value[0]),
            int(size_value[1]),
            method_expr[1:-1],
            [int(v) for v in list(shape_value)],
            align_expr == "True",
            hpc_expr == "True",
            channel_last,
        )

    pending_conv_block_name: str | None = None
    for index, line in enumerate(lines):
        conv_block_decl_match = conv_block_decl_re.match(line)
        if conv_block_decl_match is not None:
            pending_conv_block_name = str(conv_block_decl_match.group("module"))
            continue
        in_channels_match = in_channels_re.match(line)
        if in_channels_match is not None:
            if pending_conv_block_name is not None:
                conv_block_in_channels[pending_conv_block_name] = int(in_channels_match.group("channels"))
            continue
        out_channels_match = out_channels_re.match(line)
        if out_channels_match is not None:
            if pending_conv_block_name is not None:
                conv_block_out_channels[pending_conv_block_name] = int(out_channels_match.group("channels"))
            continue
        if (
            pending_conv_block_name is not None
            and (
                line.strip() == ")"
                or re.match(r"^\s*self\.[A-Za-z0-9_]+ =", line) is not None
            )
        ):
            pending_conv_block_name = None
        register_buffer_match = register_buffer_re.match(line)
        if register_buffer_match is not None:
            shape_values = _parse_int_list_literal(str(register_buffer_match.group("shape")))
            non_singleton_dims = [value for value in shape_values if value != 1]
            if len(non_singleton_dims) == 1:
                const_channel_counts[str(register_buffer_match.group("name"))] = int(non_singleton_dims[0])
        module_output_assign_match = module_output_assign_re.match(line)
        if module_output_assign_match is not None:
            lhs_name = str(module_output_assign_match.group("lhs"))
            module_name = str(module_output_assign_match.group("module"))
            module_output_producers[lhs_name] = module_name
            module_input_name = str(module_output_assign_match.group("input"))
            module_input_consumers.setdefault(module_input_name, []).append(module_name)
            cf_like_names.add(lhs_name)
        generic_expr_assign_match = generic_expr_assign_re.match(line)
        if generic_expr_assign_match is not None:
            lhs_name = str(generic_expr_assign_match.group("lhs"))
            rhs_expr = str(generic_expr_assign_match.group("rhs"))
            for token in _fast_precanonicalize_expr_identifiers(rhs_expr):
                consumers.setdefault(token, []).append(index)
            simple_alias_match = simple_alias_re.match(line)
            if simple_alias_match is not None:
                alias_lhs = str(simple_alias_match.group("lhs"))
                alias_rhs = str(simple_alias_match.group("rhs"))
                aliases[alias_lhs] = alias_rhs
                if (
                    alias_rhs in cf_like_names
                    or alias_rhs.endswith("_cf")
                    or alias_rhs.endswith("_out_cf")
                ):
                    cf_like_names.add(alias_lhs)
                if alias_rhs in nhwc_like_names or "_nhwc" in alias_rhs:
                    nhwc_like_names.add(alias_lhs)
            if lhs_name.endswith("_cf") or lhs_name.endswith("_out_cf"):
                cf_like_names.add(lhs_name)
            if "_nhwc" in lhs_name:
                nhwc_like_names.add(lhs_name)
            permute_assign = _parse_torch_permute_assign(line)
            if permute_assign is not None and str(permute_assign[1]) == lhs_name:
                perm_values = [int(v) for v in list(permute_assign[3])]
                if perm_values == [0, 2, 3, 1]:
                    nhwc_like_names.add(lhs_name)
                elif perm_values == [0, 3, 1, 2]:
                    cf_like_names.add(lhs_name)
            local_response_norm_input = _parse_local_response_norm_input_expr(rhs_expr)
            if local_response_norm_input is not None:
                if (
                    local_response_norm_input in cf_like_names
                    or local_response_norm_input.endswith("_cf")
                    or local_response_norm_input.endswith("_out_cf")
                ):
                    cf_like_names.add(lhs_name)
                if (
                    local_response_norm_input in nhwc_like_names
                    or "_nhwc" in local_response_norm_input
                ):
                    nhwc_like_names.add(lhs_name)
            apply_concat_args = _parse_apply_concat_inputs_axis_and_shape(rhs_expr)
            if apply_concat_args is not None:
                _, axis, shape_values = apply_concat_args
                if shape_values is not None:
                    static_shapes[lhs_name] = shape_values
                if axis == 1:
                    cf_like_names.add(lhs_name)
                elif axis == 3:
                    nhwc_like_names.add(lhs_name)
            torch_cat_args = _parse_torch_cat_inputs_and_dim(rhs_expr)
            if torch_cat_args is not None:
                if torch_cat_args[1] == 1:
                    cf_like_names.add(lhs_name)
                elif torch_cat_args[1] == 3:
                    nhwc_like_names.add(lhs_name)
        aligned_rank4_any_match = aligned_rank4_any_re.match(line)
        if aligned_rank4_any_match is not None:
            static_shapes[str(aligned_rank4_any_match.group("lhs"))] = [
                int(aligned_rank4_any_match.group("n")),
                int(aligned_rank4_any_match.group("d1")),
                int(aligned_rank4_any_match.group("d2")),
                int(aligned_rank4_any_match.group("d3")),
            ]
        apply_resize_match = _parse_apply_resize_assign(line)
        if apply_resize_match is not None:
            lhs_name = str(apply_resize_match[1])
            static_shapes[lhs_name] = [int(v) for v in list(apply_resize_match[6])]
            if bool(apply_resize_match[9]) is False:
                cf_like_names.add(lhs_name)
            else:
                nhwc_like_names.add(lhs_name)
        apply_pool2d_assign = _parse_apply_pool2d_assign_with_shape(line)
        if apply_pool2d_assign is not None:
            lhs_name = str(apply_pool2d_assign[1])
            static_shapes[lhs_name] = [int(v) for v in list(apply_pool2d_assign[4])]
            if bool(apply_pool2d_assign[6]) is False:
                cf_like_names.add(lhs_name)
            else:
                nhwc_like_names.add(lhs_name)
        apply_softmax_match = apply_softmax_re.match(line)
        if apply_softmax_match is not None:
            lhs_name = str(apply_softmax_match.group("lhs"))
            static_shapes[lhs_name] = [
                int(apply_softmax_match.group("n")),
                int(apply_softmax_match.group("h")),
                int(apply_softmax_match.group("w")),
                int(apply_softmax_match.group("c")),
            ]
            if int(apply_softmax_match.group("axis")) == 1:
                cf_like_names.add(lhs_name)
            elif int(apply_softmax_match.group("axis")) == 3:
                nhwc_like_names.add(lhs_name)
        const_pad_assign_match = const_pad_assign_re.match(line)
        if const_pad_assign_match is not None:
            input_name = str(const_pad_assign_match.group("input"))
            if (
                input_name in cf_like_names
                or input_name.endswith("_cf")
                or input_name.endswith("_out_cf")
            ):
                cf_like_names.add(str(const_pad_assign_match.group("lhs")))

    changed = True
    while changed:
        changed = False
        for lhs_name, rhs_name in aliases.items():
            if rhs_name in cf_like_names and lhs_name not in cf_like_names:
                cf_like_names.add(lhs_name)
                changed = True
            if rhs_name in nhwc_like_names and lhs_name not in nhwc_like_names:
                nhwc_like_names.add(lhs_name)
                changed = True

    return _FastPrecanonicalizeRepairContext(
        aliases=aliases,
        consumers=consumers,
        static_shapes=static_shapes,
        cf_like_names=cf_like_names,
        nhwc_like_names=nhwc_like_names,
        const_channel_counts=const_channel_counts,
        conv_block_in_channels=conv_block_in_channels,
        conv_block_out_channels=conv_block_out_channels,
        module_output_producers=module_output_producers,
        module_input_consumers=module_input_consumers,
    )


def _fast_precanonicalize_resolve_alias(
    name: str,
    aliases: Dict[str, str],
) -> str:
    resolved = str(name)
    seen: Set[str] = set()
    while resolved not in seen and resolved in aliases:
        seen.add(resolved)
        resolved = str(aliases[resolved])
    return resolved


def _fast_precanonicalize_is_cf_like(
    name: str,
    dynamic_cf_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    resolved = _fast_precanonicalize_resolve_alias(str(name), context.aliases)
    return (
        resolved in dynamic_cf_like_names
        or resolved in context.cf_like_names
        or resolved.endswith("_cf")
        or resolved.endswith("_out_cf")
    )


def _fast_precanonicalize_is_nhwc_like(
    name: str,
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    resolved = _fast_precanonicalize_resolve_alias(str(name), context.aliases)
    return (
        resolved in dynamic_nhwc_like_names
        or resolved in context.nhwc_like_names
        or "_nhwc" in resolved
    )


def _fast_precanonicalize_preferred_channel_count(
    name: str,
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
    shape_hint: Sequence[int] | None = None,
) -> int | None:
    resolved = _fast_precanonicalize_resolve_alias(str(name), context.aliases)
    resolved_is_cf_like = _fast_precanonicalize_is_cf_like(
        resolved,
        dynamic_cf_like_names,
        context,
    )
    resolved_is_nhwc_like = _fast_precanonicalize_is_nhwc_like(
        resolved,
        dynamic_nhwc_like_names,
        context,
    )
    shape_hint_values = (
        [int(value) for value in list(shape_hint)]
        if shape_hint is not None and len(list(shape_hint)) == 4
        else None
    )

    def _pick_shape_hint_candidate(candidates: Sequence[int]) -> int | None:
        if shape_hint_values is None:
            return None
        hinted_candidates: list[tuple[int, str]] = []
        for candidate in candidates:
            layout_hint = _fast_precanonicalize_rank4_layout_hint(
                shape_hint_values,
                preferred_channel_count=int(candidate),
            )
            if layout_hint is not None:
                hinted_candidates.append((int(candidate), layout_hint))
        if len(hinted_candidates) == 0:
            return None
        if resolved_is_cf_like:
            for candidate, layout_hint in hinted_candidates:
                if layout_hint == "cf":
                    return int(candidate)
        if resolved_is_nhwc_like:
            for candidate, layout_hint in hinted_candidates:
                if layout_hint == "nhwc":
                    return int(candidate)
        unique_candidates: list[int] = []
        for candidate, _ in hinted_candidates:
            if int(candidate) not in unique_candidates:
                unique_candidates.append(int(candidate))
        if len(unique_candidates) == 1:
            return int(unique_candidates[0])
        return None

    consumer_modules = context.module_input_consumers.get(resolved, [])
    consumer_channel_candidates: list[int] = []
    for consumer_module in consumer_modules:
        consumer_channels = context.conv_block_in_channels.get(consumer_module, None)
        if consumer_channels is None:
            continue
        consumer_channel_value = int(consumer_channels)
        if consumer_channel_value not in consumer_channel_candidates:
            consumer_channel_candidates.append(consumer_channel_value)
    hinted_consumer_channel = _pick_shape_hint_candidate(consumer_channel_candidates)
    if hinted_consumer_channel is not None:
        return int(hinted_consumer_channel)
    if len(consumer_channel_candidates) > 0:
        return int(consumer_channel_candidates[0])
    producer_name = context.module_output_producers.get(resolved, None)
    if producer_name is not None:
        producer_channels = context.conv_block_out_channels.get(producer_name, None)
        if producer_channels is not None:
            hinted_producer_channel = _pick_shape_hint_candidate([int(producer_channels)])
            if hinted_producer_channel is not None:
                return int(hinted_producer_channel)
            return int(producer_channels)
    static_shape = context.static_shapes.get(resolved, None)
    if static_shape is not None and len(static_shape) == 4:
        static_layout_hint = _fast_precanonicalize_rank4_layout_hint(static_shape)
        if (
            resolved_is_cf_like
            and static_layout_hint == "cf"
        ):
            return int(static_shape[1])
        if (
            resolved_is_nhwc_like
            and static_layout_hint == "nhwc"
        ):
            return int(static_shape[3])
    return None


def _fast_precanonicalize_infer_consumer_layout(
    name: str,
    start_index: int,
    lines: Sequence[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> str | None:
    consumer_indexes = context.consumers.get(
        _fast_precanonicalize_resolve_alias(str(name), context.aliases),
        [],
    )
    cf_score = 0
    nhwc_score = 0
    for consumer_index in consumer_indexes:
        if consumer_index <= start_index:
            continue
        consumer_line = str(lines[consumer_index])
        if re.search(rf"self\.conv_block_[0-9]+\({re.escape(name)}\)", consumer_line):
            cf_score += 3
        if f"{name}.permute(0, 3, 1, 2).contiguous()" in consumer_line:
            nhwc_score += 2
        consumer_assign = _parse_simple_assignment_line(consumer_line)
        consumer_expr = consumer_assign[2] if consumer_assign is not None else consumer_line.strip()
        resize_args = _parse_apply_resize_input_and_channel_last(consumer_expr)
        if resize_args is not None and resize_args[0] == name:
            if resize_args[1]:
                nhwc_score += 2
            else:
                cf_score += 2
        pool_args = _parse_apply_pool2d_input_and_channel_last(consumer_expr)
        if pool_args is not None and pool_args[0] == name:
            if pool_args[1]:
                nhwc_score += 2
            else:
                cf_score += 2
        softmax_args = _parse_apply_softmax_input_and_axis(consumer_expr)
        if softmax_args is not None and softmax_args[0] == name:
            if softmax_args[1] == 1:
                cf_score += 2
            elif softmax_args[1] == 3:
                nhwc_score += 2
        concat_args = _parse_apply_concat_inputs_axis_and_shape(consumer_expr)
        if concat_args is not None and name in concat_args[0]:
            if concat_args[1] == 1:
                cf_score += 2
            elif concat_args[1] == 3:
                nhwc_score += 2
        torch_cat_args = _parse_torch_cat_inputs_and_dim(consumer_expr)
        if torch_cat_args is not None and name in torch_cat_args[0]:
            if torch_cat_args[1] == 1:
                cf_score += 2
            elif torch_cat_args[1] == 3:
                nhwc_score += 2
        if f"{name}[:, :, :, [" in consumer_line:
            nhwc_score += 1
        if f"{name}[:, [" in consumer_line:
            cf_score += 1
        align_shape_args = _parse_align_tensor_target_shape_expr(consumer_expr)
        align_rank4_shape = (
            _parse_rank4_shape_literal(align_shape_args[1])
            if align_shape_args is not None
            else None
        )
        if align_rank4_shape is not None:
            preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
                name,
                dynamic_cf_like_names,
                dynamic_nhwc_like_names,
                context,
                shape_hint=list(align_rank4_shape),
            )
            layout_hint = _fast_precanonicalize_rank4_layout_hint(
                list(align_rank4_shape),
                preferred_channel_count=preferred_channel_count,
            )
            if layout_hint == "cf":
                cf_score += 1
            elif layout_hint == "nhwc":
                nhwc_score += 1
    if cf_score > nhwc_score and cf_score > 0:
        return "cf"
    if nhwc_score > cf_score and nhwc_score > 0:
        return "nhwc"
    if _fast_precanonicalize_is_cf_like(name, dynamic_cf_like_names, context):
        return "cf"
    if _fast_precanonicalize_is_nhwc_like(name, dynamic_nhwc_like_names, context):
        return "nhwc"
    return None


def _fast_precanonicalize_has_channel_last_spatial_consumer(
    name: str,
    start_index: int,
    lines: Sequence[str],
    context: _FastPrecanonicalizeRepairContext,
    visited_names: Set[str] | None = None,
) -> bool:
    if visited_names is None:
        visited_names = set()
    if name in visited_names:
        return False
    visited_names.add(name)
    resolved_name = _fast_precanonicalize_resolve_alias(str(name), context.aliases)
    consumer_indexes = context.consumers.get(resolved_name, [])
    direct_slice_re = re.compile(
        rf"^\s*[A-Za-z0-9_]+ = {re.escape(name)}\[[^,\]]+, [^,\]]+, [^,\]]+, [^,\]]+\]$"
    )
    direct_alias_re = re.compile(
        rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = {re.escape(name)}$"
    )
    for consumer_index in consumer_indexes:
        if consumer_index <= start_index:
            continue
        consumer_line = str(lines[consumer_index]).strip()
        if consumer_line == "":
            continue
        if direct_slice_re.match(consumer_line) is not None:
            return True
        direct_alias_match = direct_alias_re.match(consumer_line)
        if direct_alias_match is not None:
            alias_name = str(direct_alias_match.group("lhs"))
            if alias_name.startswith("_space_to_depth_x_") or alias_name.startswith("_depth_to_space_x_"):
                return True
        consumer_assign = _parse_simple_assignment_line(consumer_line)
        pool_args = (
            _parse_apply_pool2d_input_and_channel_last(consumer_assign[2])
            if consumer_assign is not None
            else None
        )
        direct_pool_match = (
            (consumer_assign[1], pool_args[0], pool_args[1])
            if consumer_assign is not None and pool_args is not None
            else None
        )
        if (
            direct_pool_match is not None
            and str(direct_pool_match[1]) == name
            and direct_pool_match[2]
            and _fast_precanonicalize_has_channel_last_spatial_consumer(
                str(direct_pool_match[0]),
                consumer_index,
                lines,
                context,
                visited_names,
            )
        ):
            return True
    return False


def _repair_split_axis_from_consumers(
    line: str,
    index: int,
    lines: Sequence[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> Tuple[str | None, Set[str]]:
    split_assign = _parse_tensor_split_assign(line)
    if split_assign is None:
        return None, set()
    outputs = [token.strip() for token in list(split_assign[1]) if token.strip()]
    current_axis = int(split_assign[4])
    cf_votes = 0
    nhwc_votes = 0
    for output_name in outputs:
        for consumer_index in context.consumers.get(output_name, []):
            if consumer_index <= index:
                continue
            consumer_line = str(lines[consumer_index])
            consumer_assign = _parse_simple_assignment_line(consumer_line)
            consumer_expr = consumer_assign[2] if consumer_assign is not None else consumer_line.strip()
            if re.search(rf"self\.conv_block_[0-9]+\({re.escape(output_name)}\)", consumer_line):
                cf_votes += 2
                continue
            resize_args = _parse_apply_resize_input_and_channel_last(consumer_expr)
            if resize_args is not None and resize_args[0] == output_name and not resize_args[1]:
                cf_votes += 2
                continue
            pool_args = _parse_apply_pool2d_input_and_channel_last(consumer_expr)
            if pool_args is not None and pool_args[0] == output_name and not pool_args[1]:
                cf_votes += 2
                continue
            concat_args = _parse_apply_concat_inputs_axis_and_shape(consumer_expr)
            if concat_args is not None and output_name in concat_args[0]:
                if concat_args[1] == 1:
                    cf_votes += 2
                elif concat_args[1] == 3:
                    nhwc_votes += 2
                continue
            torch_cat_args = _parse_torch_cat_inputs_and_dim(consumer_expr)
            if torch_cat_args is not None and output_name in torch_cat_args[0]:
                if torch_cat_args[1] == 1:
                    cf_votes += 2
                elif torch_cat_args[1] == 3:
                    nhwc_votes += 2
                continue
            softmax_args = _parse_apply_softmax_input_and_axis(consumer_expr)
            if softmax_args is not None and softmax_args[0] == output_name:
                if softmax_args[1] == 1:
                    cf_votes += 2
                elif softmax_args[1] == 3:
                    nhwc_votes += 2
    preferred_axis = current_axis
    if cf_votes > nhwc_votes and cf_votes > 0:
        preferred_axis = 1
    elif nhwc_votes > cf_votes and nhwc_votes > 0:
        preferred_axis = 3
    elif _fast_precanonicalize_is_cf_like(
        str(split_assign[2]),
        dynamic_cf_like_names,
        context,
    ):
        preferred_axis = 1
    if preferred_axis == current_axis:
        return None, set()
    rewritten = (
        f"{split_assign[0]}"
        f"{', '.join(split_assign[1])} = list(torch.tensor_split("
        f"{split_assign[2]}, "
        f"{split_assign[3]}, "
        f"dim=_normalize_dim({preferred_axis}, {split_assign[2]}.ndim)))"
    )
    return rewritten, set(outputs if preferred_axis == 1 else [])


def _repair_cf_resize_target_shape(
    line: str,
    index: int,
    lines: Sequence[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> Tuple[str | None, str | None]:
    aligned_bn_const_re = re.compile(
        r"^\s*[A-Za-z0-9_]+ = _align_tensor_to_target_shape\(torch\.(?:mul|add)\((?P<input>[A-Za-z0-9_]+), (?:self|torch\.reshape\(self)\.(?P<const_attr>[A-Za-z0-9_]+).*$"
    )

    def _parse_apply_resize_assign(
        src_line: str,
    ) -> Tuple[str, str, str, int, int, str, List[int], bool, bool, bool] | None:
        assign = _parse_simple_assignment_line(src_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        parsed = _parse_apply_resize_input_size_shape_and_channel_last(rhs)
        if parsed is None:
            return None
        input_name, size_value, shape_value, channel_last = parsed
        if size_value is None or shape_value is None:
            return None
        stripped = rhs.strip()
        parts = _split_top_level_csv_exprs(stripped[len("_apply_resize(") : -1])
        method_expr: str | None = None
        align_expr: str | None = None
        hpc_expr: str | None = None
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "method":
                method_expr = value
            elif key == "align_corners":
                align_expr = value
            elif key == "half_pixel_centers":
                hpc_expr = value
        if (
            method_expr is None
            or align_expr not in {"True", "False"}
            or hpc_expr not in {"True", "False"}
            or not (method_expr.startswith("'") and method_expr.endswith("'"))
        ):
            return None
        return (
            indent,
            lhs,
            input_name,
            int(size_value[0]),
            int(size_value[1]),
            method_expr[1:-1],
            [int(v) for v in list(shape_value)],
            align_expr == "True",
            hpc_expr == "True",
            channel_last,
        )

    apply_resize_match = _parse_apply_resize_assign(line)
    if apply_resize_match is None:
        return None, None
    indent, lhs_name, input_name, out_h, out_w, method, current_shape, align_corners, half_pixel_centers, channel_last = apply_resize_match
    consumer_layout = _fast_precanonicalize_infer_consumer_layout(
        lhs_name,
        index,
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    )
    should_repair = (
        not channel_last
        or _fast_precanonicalize_is_cf_like(input_name, dynamic_cf_like_names, context)
        or consumer_layout == "cf"
    )
    if not should_repair:
        return None, None
    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
        lhs_name,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
        shape_hint=current_shape,
    )
    if preferred_channel_count is None:
        preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
            input_name,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
            shape_hint=current_shape,
        )
    if _fast_precanonicalize_rank4_layout_hint(
        current_shape,
        preferred_channel_count=preferred_channel_count,
    ) == "cf":
        return None, lhs_name
    next_line = str(lines[index + 1]) if index + 1 < len(lines) else ""
    next_bn_match = aligned_bn_const_re.match(next_line)
    if next_bn_match is not None and str(next_bn_match.group("input")) == lhs_name:
        const_attr = next_bn_match.groupdict().get("const_attr", None)
        if const_attr is not None:
            preferred_channel_count = context.const_channel_counts.get(str(const_attr), preferred_channel_count)
    normalized_shape = _normalize_cf_rank4_shape(
        current_shape,
        preferred_channel_count=preferred_channel_count,
        out_hw=(out_h, out_w),
    )
    rewritten = (
        f"{indent}{lhs_name} = _apply_resize("
        f"{input_name}, [{out_h}, {out_w}], "
        f"method='{method}', target_shape={repr(normalized_shape)}, "
        f"align_corners={align_corners}, "
        f"half_pixel_centers={half_pixel_centers}, channel_last=False)"
    )
    if rewritten == line:
        return None, lhs_name
    return rewritten, lhs_name


def _repair_cf_pool_target_shape(
    line: str,
    index: int,
    lines: Sequence[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> Tuple[str | None, str | None]:
    apply_pool2d_match = _parse_apply_pool2d_assign_with_shape(line)
    if apply_pool2d_match is None:
        return None, None
    indent, lhs_name, input_name, rest, current_shape, is_max_pool, channel_last = apply_pool2d_match
    input_is_immediate_nhwc_bridge = _has_immediate_rank4_permute_source(
        lines,
        index,
        input_name,
        [0, 2, 3, 1],
    )
    consumer_layout = _fast_precanonicalize_infer_consumer_layout(
        lhs_name,
        index,
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    )
    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
        lhs_name,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
        shape_hint=current_shape,
    )
    if preferred_channel_count is None:
        preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
            input_name,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
            shape_hint=current_shape,
        )
    current_layout_hint = _fast_precanonicalize_rank4_layout_hint(
        current_shape,
        preferred_channel_count=preferred_channel_count,
    )
    if (
        channel_last
        and _fast_precanonicalize_is_nhwc_like(
            input_name,
            dynamic_nhwc_like_names,
            context,
        )
        and not _fast_precanonicalize_is_cf_like(
            input_name,
            dynamic_cf_like_names,
            context,
        )
    ):
        return None, lhs_name
    if (
        input_is_immediate_nhwc_bridge
        and channel_last
        and current_layout_hint == "nhwc"
    ):
        return None, lhs_name
    should_repair = (
        not channel_last
        or (
            not input_is_immediate_nhwc_bridge
            and _fast_precanonicalize_is_cf_like(input_name, dynamic_cf_like_names, context)
        )
        or (not input_is_immediate_nhwc_bridge and consumer_layout == "cf")
    )
    if not should_repair:
        return None, None
    if current_layout_hint == "cf":
        if not channel_last:
            return None, lhs_name
        normalized_shape = list(current_shape)
    else:
        normalized_shape = _normalize_cf_rank4_shape(
            current_shape,
            preferred_channel_count=preferred_channel_count,
        )
    rewritten = (
        f"{indent}{lhs_name} = _apply_pool2d("
        f"{input_name}, {rest}, target_shape={repr(normalized_shape)}, "
        f"is_max_pool={is_max_pool}, channel_last=False)"
    )
    if rewritten == line:
        return None, lhs_name
    return rewritten, lhs_name


def _repair_nhwc_average_pool_binary_bridge(
    index: int,
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> Tuple[bool, Set[str]]:
    def _parse_binary_anchor_assign(
        current_line: str,
    ) -> Tuple[str, str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(current_line),
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
        input_a = parts[0].strip()
        input_b = parts[1].strip()
        anchor_shape = _parse_rank4_shape_literal(parts[2].strip())
        if anchor_shape is None:
            return None
        return (
            str(assign_match.group("indent")),
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            input_a,
            input_b,
            [int(v) for v in list(anchor_shape)],
        )

    if index + 2 >= len(lines):
        return False, set()
    apply_pool2d_match = _parse_apply_pool2d_assign_with_shape(lines[index])
    if apply_pool2d_match is None:
        return False, set()
    (
        pool_indent,
        pool_lhs_name,
        pool_input_name,
        pool_rest,
        pool_shape,
        pool_is_max,
        _pool_channel_last,
    ) = apply_pool2d_match
    if pool_is_max:
        return False, set()
    if not (
        _fast_precanonicalize_is_nhwc_like(
            pool_input_name,
            dynamic_nhwc_like_names,
            context,
        )
        or "_nhwc" in str(pool_lhs_name)
    ):
        return False, set()
    binary_anchor_assign = _parse_binary_anchor_assign(lines[index + 1])
    if binary_anchor_assign is None:
        return False, set()
    binary_indent, lhs0, lhs1, input_a, input_b, anchor_shape = binary_anchor_assign
    if pool_lhs_name not in {str(input_a).strip(), str(input_b).strip()}:
        return False, set()
    mul_assign = _parse_simple_assignment_line(lines[index + 2])
    if mul_assign is None:
        return False, set()
    mul_indent, mul_lhs, mul_rhs = mul_assign
    align_args = _parse_align_tensor_target_shape_expr(mul_rhs)
    if align_args is None:
        return False, set()
    mul_expr, mul_target_shape_expr = align_args
    mul_target_shape = _parse_rank4_shape_literal(mul_target_shape_expr)
    mul_match = re.fullmatch(r"torch\.mul\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\)", mul_expr.strip())
    if (
        mul_target_shape is None
        or mul_match is None
        or {str(mul_match.group("a")), str(mul_match.group("b"))} != {str(lhs0), str(lhs1)}
    ):
        return False, set()
    has_direct_nhwc_concat_consumer = False
    for lookahead in range(index + 3, len(lines)):
        future_assign = _parse_simple_assignment_line(lines[lookahead])
        if future_assign is None:
            continue
        future_concat_args = _parse_apply_concat_inputs_axis_and_shape(future_assign[2])
        if future_concat_args is not None:
            future_inputs = [name.strip() for name in future_concat_args[0] if name.strip()]
            if mul_lhs in future_inputs and int(future_concat_args[1]) == 3:
                has_direct_nhwc_concat_consumer = True
                break
        future_cat_args = _parse_torch_cat_inputs_and_dim(future_assign[2])
        if future_cat_args is not None:
            future_inputs = [name.strip() for name in future_cat_args[0] if name.strip()]
            if mul_lhs in future_inputs and int(future_cat_args[1]) == 3:
                has_direct_nhwc_concat_consumer = True
                break
    mul_consumer_layout = _fast_precanonicalize_infer_consumer_layout(
        mul_lhs,
        index + 2,
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    )
    if not has_direct_nhwc_concat_consumer and mul_consumer_layout != "nhwc":
        return False, set()
    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
        mul_lhs,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
        shape_hint=pool_shape,
    )
    if preferred_channel_count is None:
        preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
            pool_input_name,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
            shape_hint=pool_shape,
        )
    if preferred_channel_count is None:
        pool_dims = [int(v) for v in list(pool_shape[1:])]
        for candidate in pool_dims:
            if pool_dims.count(int(candidate)) == 1:
                preferred_channel_count = int(candidate)
                break
    normalized_nhwc_shape = _normalize_nhwc_rank4_shape(
        pool_shape,
        preferred_channel_count=preferred_channel_count,
    )
    changed = False
    if index > 0:
        previous_pad_assign = _parse_constant_pad_assign(lines[index - 1])
        if (
            previous_pad_assign is not None
            and previous_pad_assign[1] == pool_input_name
            and re.fullmatch(r"[A-Za-z0-9_]+", previous_pad_assign[2]) is not None
        ):
            nhwc_pad_values = _convert_nchw_pad_to_nhwc_pad_values(previous_pad_assign[3])
            if nhwc_pad_values is not None:
                rewritten_pad_line = (
                    f"{previous_pad_assign[0]}{previous_pad_assign[1]} = "
                    f"F.pad({previous_pad_assign[2]}, {repr(nhwc_pad_values)}, mode='constant', value={previous_pad_assign[4]})"
                )
                if rewritten_pad_line != lines[index - 1]:
                    lines[index - 1] = rewritten_pad_line
                    changed = True

    def _rewrite_binary_other_expr(expr: str) -> str:
        stripped = str(expr).strip()
        reshape_match = re.fullmatch(r"torch\.reshape\((?P<args>.+)\)", stripped)
        if reshape_match is None:
            return stripped
        parts = _split_top_level_csv_exprs(str(reshape_match.group("args")))
        reshape_input: str | None = None
        reshape_shape_expr: str | None = None
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            reshape_input = parts[0].strip()
            reshape_shape_expr = parts[1].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            reshape_input = kwargs.get("input")
            reshape_shape_expr = kwargs.get("shape")
        reshape_shape = (
            _parse_rank4_shape_literal(reshape_shape_expr)
            if reshape_shape_expr is not None
            else None
        )
        if (
            reshape_input is None
            or reshape_shape is None
        ):
            return stripped
        if list(reshape_shape) == list(normalized_nhwc_shape):
            return stripped
        return f"{reshape_input}.permute(0, 2, 3, 1).contiguous()"

    rewritten_other_expr = _rewrite_binary_other_expr(
        input_b if str(input_a).strip() == pool_lhs_name else input_a
    )
    rewritten_input_a = pool_lhs_name if str(input_a).strip() == pool_lhs_name else rewritten_other_expr
    rewritten_input_b = rewritten_other_expr if str(input_a).strip() == pool_lhs_name else pool_lhs_name
    rewritten_pool_line = (
        f"{pool_indent}{pool_lhs_name} = _apply_pool2d("
        f"{pool_input_name}, {pool_rest}, "
        f"target_shape={repr(normalized_nhwc_shape)}, "
        f"is_max_pool={pool_is_max}, channel_last=True)"
    )
    if rewritten_pool_line != lines[index]:
        lines[index] = rewritten_pool_line
        changed = True
    rewritten_binary_line = (
        f"{binary_indent}{lhs0}, {lhs1} = _align_binary_inputs_to_anchor("
        f"{rewritten_input_a}, {rewritten_input_b}, {repr(normalized_nhwc_shape)})"
    )
    if rewritten_binary_line != lines[index + 1]:
        lines[index + 1] = rewritten_binary_line
        changed = True
    rewritten_mul_line = (
        f"{mul_indent}{mul_lhs} = _align_tensor_to_target_shape("
        f"torch.mul({mul_match.group('a')}, {mul_match.group('b')}), {repr(normalized_nhwc_shape)})"
    )
    if rewritten_mul_line != lines[index + 2]:
        lines[index + 2] = rewritten_mul_line
        changed = True
    updated_names: Set[str] = set()
    if changed:
        updated_names.update({str(pool_lhs_name), str(lhs0), str(lhs1), str(mul_lhs)})
    return changed, updated_names


def _restore_channel_last_spatial_pool_chains(model_path: Path) -> None:
    if not model_path.exists():
        return
    lines = model_path.read_text(encoding="utf-8").splitlines()
    context = _build_fast_precanonicalize_repair_context(lines)
    changed = False
    for index, line in enumerate(lines):
        apply_pool2d_match = _parse_apply_pool2d_assign_with_shape(str(line))
        if (
            apply_pool2d_match is None
            or apply_pool2d_match[6]
            or not apply_pool2d_match[5]
            or not _fast_precanonicalize_has_channel_last_spatial_consumer(
                apply_pool2d_match[1],
                index,
                lines,
                context,
            )
        ):
            continue
        pool_indent, pool_lhs_name, pool_input_name, pool_rest, pool_shape, pool_is_max, _ = apply_pool2d_match
        lines[index] = (
            f"{pool_indent}{pool_lhs_name} = _apply_pool2d("
            f"{pool_input_name}, {pool_rest}, "
            f"target_shape={repr(pool_shape)}, "
            f"is_max_pool={pool_is_max}, channel_last=True)"
        )
        changed = True
    if changed:
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _repair_binary_alignment_layout(
    line: str,
    index: int,
    lines: Sequence[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> Tuple[str | None, str | None]:
    assign = _parse_simple_assignment_line(line)
    if assign is None:
        return None, None
    indent, lhs_name, rhs_expr = assign
    align_parts = _parse_align_tensor_target_shape_expr(rhs_expr)
    if align_parts is None:
        return None, None
    input_expr, target_shape_expr = align_parts
    target_shape = _parse_rank4_shape_literal(target_shape_expr)
    if target_shape is None:
        return None, None
    binary_match = re.fullmatch(
        r"torch\.(?P<op>mul|add|sub|div|minimum|maximum)\((?P<args>.+)\)",
        input_expr.strip(),
    )
    if binary_match is None:
        return None, None
    op_name = str(binary_match.group("op"))
    if op_name == "mul":
        binary_args = _parse_binary_mul_args(str(binary_match.group("args")))
    else:
        binary_args = _parse_binary_add_args(str(binary_match.group("args")))
    if binary_args is None:
        return None, None
    arg_a = str(binary_args[0]).strip()
    arg_b = str(binary_args[1]).strip()
    if (
        re.fullmatch(r"[A-Za-z0-9_]+", arg_a) is None
        or re.fullmatch(r"[A-Za-z0-9_]+", arg_b) is None
    ):
        return None, None
    for lookback in range(max(0, index - 4), index):
        binary_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[lookback])
        if (
            binary_anchor_assign is not None
            and {str(binary_anchor_assign[1]), str(binary_anchor_assign[2])} == {arg_a, arg_b}
        ):
            return None, None
    arg_a_is_cf = _fast_precanonicalize_is_cf_like(arg_a, dynamic_cf_like_names, context)
    arg_b_is_cf = _fast_precanonicalize_is_cf_like(arg_b, dynamic_cf_like_names, context)
    arg_a_is_nhwc = _fast_precanonicalize_is_nhwc_like(arg_a, dynamic_nhwc_like_names, context)
    arg_b_is_nhwc = _fast_precanonicalize_is_nhwc_like(arg_b, dynamic_nhwc_like_names, context)
    if arg_a_is_nhwc or arg_b_is_nhwc:
        return None, None
    operands_are_cf_like = arg_a_is_cf and arg_b_is_cf
    consumer_layout = _fast_precanonicalize_infer_consumer_layout(
        lhs_name,
        index,
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    )
    if not operands_are_cf_like and consumer_layout != "cf":
        return None, None
    current_shape = [int(v) for v in list(target_shape)]
    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
        lhs_name,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
        shape_hint=current_shape,
    )
    if preferred_channel_count is None:
        preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
            arg_a,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
            shape_hint=current_shape,
        )
    if preferred_channel_count is None:
        preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
            arg_b,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
            shape_hint=current_shape,
        )
    if _fast_precanonicalize_rank4_layout_hint(
        current_shape,
        preferred_channel_count=preferred_channel_count,
    ) == "cf":
        return None, lhs_name
    normalized_shape = _normalize_cf_rank4_shape(
        current_shape,
        preferred_channel_count=preferred_channel_count,
    )
    rewritten = (
        f"{indent}{lhs_name} = _align_tensor_to_target_shape("
        f"torch.{op_name}({arg_a}, {arg_b}), {repr(normalized_shape)})"
    )
    if rewritten == line:
        return None, lhs_name
    return rewritten, lhs_name
