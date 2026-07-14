from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Sequence, Set, Tuple

from onnx2tf.tflite_builder.pytorch_source_parser import (
    _parse_apply_concat_inputs_axis_and_shape,
    _parse_apply_pool2d_assign_with_shape,
    _parse_apply_pool2d_input_and_channel_last,
    _parse_apply_resize_input_and_channel_last,
    _parse_apply_resize_input_size_shape_and_channel_last,
    _parse_apply_softmax_input_and_axis,
    _parse_align_tensor_target_shape_expr,
    _parse_int_list_literal,
    _parse_local_response_norm_input_expr,
    _parse_rank4_shape_literal,
    _parse_simple_assignment_line,
    _parse_torch_cat_inputs_and_dim,
    _parse_torch_permute_assign,
    _split_top_level_csv_exprs,
)
from onnx2tf.tflite_builder.pytorch_shape_policy import (
    _fast_precanonicalize_rank4_layout_hint,
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
