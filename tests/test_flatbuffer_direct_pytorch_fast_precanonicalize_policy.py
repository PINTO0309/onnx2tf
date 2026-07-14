from __future__ import annotations

from onnx2tf.tflite_builder.pytorch_fast_precanonicalize_policy import (
    _build_fast_precanonicalize_repair_context,
    _convert_nchw_pad_to_nhwc_pad_values,
    _convert_nhwc_pad_to_nchw_pad_values,
    _fast_precanonicalize_expr_identifiers,
    _fast_precanonicalize_has_channel_last_spatial_consumer,
    _fast_precanonicalize_infer_consumer_layout,
    _fast_precanonicalize_is_cf_like,
    _fast_precanonicalize_is_nhwc_like,
    _fast_precanonicalize_preferred_channel_count,
    _fast_precanonicalize_resolve_alias,
    _has_immediate_rank4_permute_source,
    _infer_unique_channel_count_from_rank4_shape,
    _repair_binary_alignment_layout,
    _repair_binary_alignment_from_downstream_evidence,
    _repair_aligned_scalar_binary_shape_at,
    _repair_cf_pool_target_shape,
    _repair_cf_pool_neighbor_layout_at,
    _repair_cf_gather_slice_at,
    _repair_cf_reduce_max_axis,
    _repair_cf_resize_from_input_and_bn_evidence,
    _repair_cf_resize_target_shape,
    _repair_cf_softmax_axis,
    _repair_concat_axis_from_input_layouts,
    _repair_depth_to_space_gather_at,
    _repair_dynamic_cf_binary_anchor_shapes,
    _repair_dynamic_pool_layout_at,
    _repair_nhwc_average_pool_binary_bridge,
    _repair_nhwc_buffer_binary_alignment_at,
    _repair_nhwc_pool_layout,
    _repair_simple_alias_layout_at,
    _repair_split_axis_from_consumers,
    _repair_singleton_reshape_cf_binary_at,
    _repair_terminal_classifier_tail_layout,
    _propagate_cf_prelu_output,
)
from onnx2tf.tflite_builder.pytorch_source_parser import (
    _parse_apply_pool2d_assign_with_shape,
    _parse_dynamic_apply_pool2d_assign,
)


def test_rank4_pad_axis_conversion_round_trips() -> None:
    nhwc_values = [1, 2, 3, 4, 5, 6, 7, 8]
    nchw_values = [3, 4, 5, 6, 1, 2, 7, 8]

    assert _convert_nhwc_pad_to_nchw_pad_values(nhwc_values) == nchw_values
    assert _convert_nchw_pad_to_nhwc_pad_values(nchw_values) == nhwc_values
    assert _convert_nhwc_pad_to_nchw_pad_values([1]) is None


def test_unique_rank4_channel_count_requires_rank4_evidence() -> None:
    assert _infer_unique_channel_count_from_rank4_shape([1, 3, 8, 8]) == 3
    assert _infer_unique_channel_count_from_rank4_shape([1, 8, 8]) is None


def test_fast_precanonicalize_identifier_filter_excludes_runtime_names() -> None:
    assert _fast_precanonicalize_expr_identifiers(
        "torch.add(self.alpha, beta) if True else False"
    ) == {"add", "alpha", "beta", "if", "else"}


def test_fast_precanonicalize_context_collects_module_and_alias_evidence() -> None:
    lines = [
        "        self.conv_block_0 = _Conv2dBlock(",
        "            in_channels=3,",
        "            out_channels=8,",
        "        )",
        "        self.register_buffer('scale', torch.zeros([1, 3, 1, 1], dtype=torch.float32), persistent=True)",
        "        out_cf = self.conv_block_0(input_nhwc)",
        "        alias = out_cf",
        "        softmax_cf = _apply_softmax(input=out_cf, axis=1, beta=1.0, target_shape=[1, 8, 4, 4])",
        "        padded_cf = F.pad(out_cf, [1, 1, 1, 1], mode='constant', value=0.0)",
        "        aligned = _align_tensor_to_target_shape(input=out_cf, target_shape=[1, 8, 4, 4])",
    ]
    context = _build_fast_precanonicalize_repair_context(lines)

    assert context.const_channel_counts == {"scale": 3}
    assert context.registered_buffer_shapes == {"scale": [1, 3, 1, 1]}
    assert context.conv_block_in_channels == {"conv_block_0": 3}
    assert context.conv_block_out_channels == {"conv_block_0": 8}
    assert context.module_output_producers == {"out_cf": "conv_block_0"}
    assert context.module_input_consumers == {"input_nhwc": ["conv_block_0"]}
    assert context.static_shapes["softmax_cf"] == [1, 8, 4, 4]
    assert context.static_shapes["aligned"] == [1, 8, 4, 4]
    assert {"softmax_cf", "padded_cf"} <= context.cf_like_names
    assert _fast_precanonicalize_resolve_alias("alias", context.aliases) == "out_cf"
    assert _fast_precanonicalize_is_cf_like("alias", set(), context)
    assert _fast_precanonicalize_is_nhwc_like(
        "input_nhwc", {"input_nhwc"}, context
    )
    assert (
        _fast_precanonicalize_preferred_channel_count(
            "input_nhwc", set(), {"input_nhwc"}, context
        )
        == 3
    )
    assert (
        _fast_precanonicalize_infer_consumer_layout(
            "alias", -1, lines, {"out_cf"}, set(), context
        )
        == "cf"
    )


def test_channel_last_spatial_consumer_detects_direct_rank4_slice() -> None:
    lines = ["        output = input[:, :, :, :]"]
    context = _build_fast_precanonicalize_repair_context(lines)

    assert _fast_precanonicalize_has_channel_last_spatial_consumer(
        "input", -1, lines, context
    )


def test_split_axis_repair_uses_channel_first_consumer_votes() -> None:
    lines = [
        "        split0, split1 = list(torch.tensor_split(input, 2, dim=_normalize_dim(3, input.ndim)))",
        "        output = self.conv_block_0(split0)",
    ]
    context = _build_fast_precanonicalize_repair_context(lines)

    rewritten, cf_outputs = _repair_split_axis_from_consumers(
        lines[0], 0, lines, set(), set(), context
    )

    assert rewritten is not None
    assert "dim=_normalize_dim(1, input.ndim)" in rewritten
    assert cf_outputs == {"split0", "split1"}


def test_resize_and_pool_repairs_normalize_channel_first_targets() -> None:
    resize_line = (
        "        resized = _apply_resize(input_cf, [20, 30], method='bilinear', "
        "target_shape=[1, 20, 30, 3], align_corners=False, "
        "half_pixel_centers=True, channel_last=False)"
    )
    resize_lines = [resize_line]
    resize_context = _build_fast_precanonicalize_repair_context(resize_lines)
    resized, resized_name = _repair_cf_resize_target_shape(
        resize_line,
        0,
        resize_lines,
        {"input_cf"},
        set(),
        resize_context,
    )
    assert resized_name == "resized"
    assert resized is not None and "target_shape=[1, 3, 20, 30]" in resized

    pool_line = (
        "        pooled = _apply_pool2d(input_cf, filter_height=2, filter_width=2, "
        "stride_h=2, stride_w=2, padding='SAME', target_shape=[1, 20, 30, 3], "
        "is_max_pool=False, channel_last=False)"
    )
    pool_lines = [pool_line]
    pool_context = _build_fast_precanonicalize_repair_context(pool_lines)
    pooled, pooled_name = _repair_cf_pool_target_shape(
        pool_line,
        0,
        pool_lines,
        {"input_cf"},
        set(),
        pool_context,
    )
    assert pooled_name == "pooled"
    assert pooled is not None and "target_shape=[1, 3, 20, 30]" in pooled


def test_resize_input_evidence_preserves_bn_and_noop_decisions() -> None:
    positive_cases = [
        (
            [
                "        self.register_buffer('scale', torch.zeros("
                "[1, 8, 1, 1], dtype=torch.float32), persistent=True)",
                "        resized = _apply_resize("
                "source_cf, [5, 7], method='nearest', "
                "target_shape=[1, 5, 7, 8], align_corners=False, "
                "half_pixel_centers=False, channel_last=True)",
                "        normalized = _align_tensor_to_target_shape("
                "torch.mul(resized, self.scale), [1, 5, 7, 8])",
            ],
            1,
            "target_shape=[1, 8, 5, 7]",
        ),
        (
            [
                "        self.register_buffer('scale', torch.zeros("
                "[8], dtype=torch.float32), persistent=True)",
                "        resized = _apply_resize("
                "source_cf, [5, 7], method='nearest', "
                "target_shape=[1, 5, 7, 8], align_corners=False, "
                "half_pixel_centers=False, channel_last=True)",
                "        normalized = _align_tensor_to_target_shape("
                "torch.add(resized, torch.reshape("
                "self.scale, [1, 8, 1, 1])), [1, 8, 5, 7])",
            ],
            1,
            "target_shape=[1, 8, 5, 7]",
        ),
        (
            [
                "        resized = _apply_resize("
                "source_cf, [5, 7], method='nearest', "
                "target_shape=[1, 5, 7, 16], align_corners=False, "
                "half_pixel_centers=False, channel_last=True)",
                "        return resized",
            ],
            0,
            "target_shape=[1, 16, 5, 7]",
        ),
    ]
    for lines, resize_index, expected_shape in positive_cases:
        rewritten, lhs_name = _repair_cf_resize_from_input_and_bn_evidence(
            lines[resize_index],
            resize_index,
            lines,
            {"source_cf"},
            set(),
            _build_fast_precanonicalize_repair_context(lines),
        )
        assert rewritten is not None and expected_shape in rewritten
        assert "channel_last=False" in rewritten
        assert lhs_name == "resized"

    negative_cases = [
        (
            [
                "        resized = _apply_resize("
                "source_nhwc, [5, 7], method='nearest', "
                "target_shape=[1, 5, 7, 16], align_corners=False, "
                "half_pixel_centers=False, channel_last=True)",
                "        return resized",
            ],
            set(),
            {"source_nhwc"},
        ),
        (
            [
                "        resized = _apply_resize("
                "source_cf, [5, 7], method='nearest', "
                "target_shape=[1, 1, 5, 7], align_corners=False, "
                "half_pixel_centers=False, channel_last=False)",
                "        return resized",
            ],
            {"source_cf"},
            set(),
        ),
    ]
    for lines, cf_names, nhwc_names in negative_cases:
        assert _repair_cf_resize_from_input_and_bn_evidence(
            lines[0],
            0,
            lines,
            cf_names,
            nhwc_names,
            _build_fast_precanonicalize_repair_context(lines),
        ) == (None, None)


def test_nhwc_pool_layout_uses_consumer_and_immediate_bridge_evidence() -> None:
    consumer_lines = [
        "        pooled = _apply_pool2d(input_cf, filter_height=2, "
        "filter_width=2, stride_h=2, stride_w=2, padding='VALID', "
        "target_shape=[1, 4, 4, 3], is_max_pool=True, channel_last=False)",
        "        output = pooled[:, :, :, :]",
    ]
    consumer_context = _build_fast_precanonicalize_repair_context(consumer_lines)
    consumer_rewrite, consumer_lhs = _repair_nhwc_pool_layout(
        consumer_lines[0],
        0,
        consumer_lines,
        {"input_cf"},
        set(),
        consumer_context,
    )
    assert consumer_lhs == "pooled"
    assert consumer_rewrite is not None
    assert "target_shape=[1, 4, 4, 3]" in consumer_rewrite
    assert "channel_last=True" in consumer_rewrite

    bridge_lines = [
        "        bridge = source.permute(0, 2, 3, 1).contiguous()",
        "        pooled = _apply_pool2d(bridge, filter_height=2, "
        "filter_width=2, stride_h=2, stride_w=2, padding='VALID', "
        "target_shape=[1, 3, 4, 4], is_max_pool=False, channel_last=False)",
    ]
    bridge_context = _build_fast_precanonicalize_repair_context(bridge_lines)
    bridge_context.module_input_consumers["bridge"] = ["conv_block_0"]
    bridge_context.conv_block_in_channels["conv_block_0"] = 3
    bridge_rewrite, bridge_lhs = _repair_nhwc_pool_layout(
        bridge_lines[1],
        1,
        bridge_lines,
        set(),
        set(),
        bridge_context,
    )
    assert bridge_lhs == "pooled"
    assert bridge_rewrite is not None
    assert "target_shape=[1, 4, 4, 3]" in bridge_rewrite
    assert "channel_last=True" in bridge_rewrite


def test_cf_pool_neighbor_layout_repairs_pad_conv_and_lrn_paths() -> None:
    pad_conv_lines = [
        "        padded = F.pad(input_cf, [1, 1, 1, 1], "
        "mode='constant', value=0.0)",
        "        pooled = _apply_pool2d(padded, filter_height=3, "
        "filter_width=3, stride_h=1, stride_w=1, padding='VALID', "
        "target_shape=[1, 4, 4, 3], is_max_pool=True, channel_last=True)",
        "        output = self.conv_block_0("
        "pooled.permute(0, 3, 1, 2).contiguous())",
    ]
    pad_conv_context = _build_fast_precanonicalize_repair_context(pad_conv_lines)
    pad_conv_assign = _parse_apply_pool2d_assign_with_shape(pad_conv_lines[1])
    assert pad_conv_assign is not None
    pad_conv_cf_names = {"input_cf"}
    changed, short_circuit = _repair_cf_pool_neighbor_layout_at(
        1,
        pad_conv_lines,
        pad_conv_assign,
        pad_conv_cf_names,
        set(),
        pad_conv_context,
    )
    assert changed and short_circuit
    assert "target_shape=[1, 3, 4, 4]" in pad_conv_lines[1]
    assert "channel_last=False" in pad_conv_lines[1]
    assert "pooled" in pad_conv_cf_names

    lrn_lines = [
        "        pooled = _apply_pool2d(input_cf, filter_height=2, "
        "filter_width=2, stride_h=2, stride_w=2, padding='VALID', "
        "target_shape=[1, 4, 3, 8], is_max_pool=False, channel_last=False)",
        "        normalized = F.local_response_norm("
        "pooled, size=5, alpha=0.0001, beta=0.75, k=1.0)",
    ]
    lrn_context = _build_fast_precanonicalize_repair_context(lrn_lines)
    lrn_assign = _parse_apply_pool2d_assign_with_shape(lrn_lines[0])
    assert lrn_assign is not None
    lrn_cf_names = {"input_cf"}
    changed, short_circuit = _repair_cf_pool_neighbor_layout_at(
        0,
        lrn_lines,
        lrn_assign,
        lrn_cf_names,
        set(),
        lrn_context,
    )
    assert changed and not short_circuit
    assert "target_shape=[1, 8, 4, 3]" in lrn_lines[0]
    assert "pooled" in lrn_cf_names


def test_dynamic_pool_layout_repairs_nhwc_and_cf_paths() -> None:
    nhwc_lines = [
        "        bridge = source.permute(0, 2, 3, 1).contiguous()",
        "        pooled = _apply_pool2d(bridge, filter_height=2, "
        "filter_width=2, stride_h=2, stride_w=2, padding='VALID', "
        "target_shape=_tensor_shape_list(bridge), is_max_pool=True, "
        "channel_last=False)",
    ]
    nhwc_context = _build_fast_precanonicalize_repair_context(nhwc_lines)
    nhwc_assign = _parse_dynamic_apply_pool2d_assign(nhwc_lines[1])
    assert nhwc_assign is not None
    nhwc_names: set[str] = set()
    changed, short_circuit = _repair_dynamic_pool_layout_at(
        1,
        nhwc_lines,
        nhwc_assign,
        set(),
        nhwc_names,
        nhwc_context,
    )
    assert changed and short_circuit
    assert "channel_last=True" in nhwc_lines[1]
    assert "pooled" in nhwc_names

    cf_lines = [
        "        pooled = _apply_pool2d(input_cf, filter_height=2, "
        "filter_width=2, stride_h=2, stride_w=2, padding='VALID', "
        "target_shape=_tensor_shape_list(input_cf), is_max_pool=False, "
        "channel_last=False)",
        "        combined = torch.add(pooled, peer)",
        "        output = _align_tensor_to_target_shape("
        "torch.relu(combined), [1, 3, 4, 5])",
    ]
    cf_context = _build_fast_precanonicalize_repair_context(cf_lines)
    cf_assign = _parse_dynamic_apply_pool2d_assign(cf_lines[0])
    assert cf_assign is not None
    cf_names = {"input_cf"}
    changed, short_circuit = _repair_dynamic_pool_layout_at(
        0,
        cf_lines,
        cf_assign,
        cf_names,
        set(),
        cf_context,
    )
    assert changed and not short_circuit
    assert "target_shape=[1, 3, 4, 5]" in cf_lines[0]
    assert "pooled" in cf_names


def test_simple_alias_layout_repairs_consumers_and_propagates_evidence() -> None:
    reshape_lines = [
        "        alias_nhwc = source_cf",
        "        reshaped = torch.reshape(alias_nhwc, [1, 8, 3])",
    ]
    reshape_context = _build_fast_precanonicalize_repair_context(reshape_lines)
    reshape_nhwc_names: set[str] = set()
    changed, rewritten = _repair_simple_alias_layout_at(
        0,
        reshape_lines,
        {"source_cf"},
        reshape_nhwc_names,
        reshape_context,
    )
    assert changed
    assert rewritten == (
        "        alias_nhwc = source_cf.permute(0, 2, 3, 1).contiguous()"
    )
    assert "alias_nhwc" in reshape_nhwc_names

    conv_lines = [
        "        feature_nhwc = feature_cf",
        "        output = self.conv_block_0("
        "feature_nhwc.permute(0, 3, 1, 2).contiguous())",
    ]
    conv_context = _build_fast_precanonicalize_repair_context(conv_lines)
    conv_context.static_shapes["feature_nhwc"] = [1, 4, 5, 3]
    changed, rewritten = _repair_simple_alias_layout_at(
        0,
        conv_lines,
        {"feature_cf"},
        set(),
        conv_context,
    )
    assert changed
    assert rewritten == (
        "        feature_nhwc = _align_tensor_to_target_shape("
        "feature_cf.permute(0, 2, 3, 1).contiguous(), [1, 4, 5, 3])"
    )

    alias_lines = ["        alias = source_cf", "        output = alias"]
    alias_context = _build_fast_precanonicalize_repair_context(alias_lines)
    alias_cf_names = {"source_cf"}
    changed, rewritten = _repair_simple_alias_layout_at(
        0,
        alias_lines,
        alias_cf_names,
        set(),
        alias_context,
    )
    assert not changed
    assert rewritten == alias_lines[0]
    assert "alias" in alias_cf_names


def test_aligned_scalar_binary_shape_uses_neighbor_consensus() -> None:
    lines = [
        "        base = _align_tensor_to_target_shape(source, [1, 2, 3, 4])",
        "        scaled = _align_tensor_to_target_shape("
        "torch.mul(base, 2.0), [1, 3, 2, 4])",
        "        output = _align_tensor_to_target_shape("
        "torch.relu(scaled), [1, 2, 3, 4])",
    ]

    assert _repair_aligned_scalar_binary_shape_at(1, lines)
    assert lines[1] == (
        "        scaled = _align_tensor_to_target_shape("
        "torch.mul(base, 2.0), [1, 2, 3, 4])"
    )


def test_binary_alignment_uses_only_exact_downstream_cf_evidence() -> None:
    positive_cases = [
        (
            [
                "        fused = _align_tensor_to_target_shape("
                "torch.add(left_in, right_in), [1, 5, 7, 16])",
                "        normalized = _align_tensor_to_target_shape("
                "torch.mul(fused, self.scale), [1, 5, 7, 16])",
                "        return normalized",
            ],
            "        fused = _align_tensor_to_target_shape("
            "torch.add(left_in, right_in), [1, 16, 5, 7])",
        ),
        (
            [
                "        fused = _align_tensor_to_target_shape("
                "torch.maximum(left_in, right_in), [1, 5, 7, 16])",
                "        return fused",
            ],
            "        fused = _align_tensor_to_target_shape("
            "torch.maximum(left_in, right_in), [1, 16, 5, 7])",
        ),
        (
            [
                "        fused = _align_tensor_to_target_shape("
                "torch.sub(left_in, right_in), [1, 5, 7, 16])",
                "        resized = _apply_resize("
                "fused, [10, 14], method='nearest', "
                "target_shape=[1, 5, 7, 16], align_corners=False, "
                "half_pixel_centers=False, channel_last=False)",
                "        return resized",
            ],
            "        fused = _align_tensor_to_target_shape("
            "torch.sub(left_in, right_in), [1, 16, 5, 7])",
        ),
    ]
    for lines, expected in positive_cases:
        context = _build_fast_precanonicalize_repair_context(lines)
        rewritten, lhs_name = _repair_binary_alignment_from_downstream_evidence(
            lines[0],
            0,
            lines,
            set(),
            set(),
            context,
        )
        assert rewritten == expected
        assert lhs_name == "fused"

    negative_cases = [
        [
            "        fused = _align_tensor_to_target_shape("
            "torch.add(left_in, right_in), [1, 5, 7, 16])",
            "        normalized = _align_tensor_to_target_shape("
            "torch.mul(fused, self.scale), [1, 5, 7, 8])",
            "        return normalized",
        ],
        [
            "        fused = _align_tensor_to_target_shape("
            "torch.sub(left_in, right_in), [1, 5, 7, 16])",
            "        resized = _apply_resize("
            "fused, [10, 14], method='nearest', "
            "target_shape=[1, 5, 7, 16], align_corners=False, "
            "half_pixel_centers=False, channel_last=True)",
            "        return resized",
        ],
        [
            "        fused = _align_tensor_to_target_shape("
            "torch.add(left_in, right_nhwc), [1, 5, 7, 16])",
            "        return fused",
        ],
        [
            "        fused = _align_tensor_to_target_shape("
            "torch.add(left_in, right_in), [1, 1, 5, 7])",
            "        return fused",
        ],
    ]
    for lines in negative_cases:
        context = _build_fast_precanonicalize_repair_context(lines)
        assert _repair_binary_alignment_from_downstream_evidence(
            lines[0],
            0,
            lines,
            set(),
            set(),
            context,
        ) == (None, None)


def test_immediate_rank4_permute_source_requires_exact_permutation() -> None:
    lines = [
        "        bridge = source.permute(0, 2, 3, 1).contiguous()",
        "        output = bridge",
    ]

    assert _has_immediate_rank4_permute_source(
        lines, 1, "bridge", [0, 2, 3, 1]
    )
    assert not _has_immediate_rank4_permute_source(
        lines, 1, "bridge", [0, 3, 1, 2]
    )


def test_nhwc_average_pool_binary_bridge_normalizes_the_whole_chain() -> None:
    lines = [
        "        pool_out_nhwc = _apply_pool2d(input_nhwc, filter_height=2, filter_width=2, stride_h=2, stride_w=2, padding='VALID', target_shape=[1, 8, 4, 1], is_max_pool=False, channel_last=False)",
        "        lhs, rhs = _align_binary_inputs_to_anchor(pool_out_nhwc, torch.reshape(scale, [1, 8, 1, 1]), [1, 8, 4, 1])",
        "        mul_out = _align_tensor_to_target_shape(torch.mul(lhs, rhs), [1, 8, 4, 1])",
        "        concat_out = _apply_concat([mul_out, peer], axis=3, target_shape=[1, 4, 1, 16], fused='NONE')",
    ]
    context = _build_fast_precanonicalize_repair_context(lines)

    changed, updated_names = _repair_nhwc_average_pool_binary_bridge(
        0,
        lines,
        set(),
        {"input_nhwc"},
        context,
    )

    assert changed
    assert updated_names == {"pool_out_nhwc", "lhs", "rhs", "mul_out"}
    assert "target_shape=[1, 4, 1, 8]" in lines[0]
    assert "channel_last=True" in lines[0]
    assert "scale.permute(0, 2, 3, 1).contiguous()" in lines[1]
    assert "target_shape(torch.mul(lhs, rhs), [1, 4, 1, 8])" in lines[2]


def test_binary_alignment_repair_normalizes_channel_first_target() -> None:
    line = (
        "        output = _align_tensor_to_target_shape("
        "torch.add(left_cf, right_cf), [1, 20, 30, 3])"
    )
    lines = [line, "        used = self.conv_block_0(output)"]
    context = _build_fast_precanonicalize_repair_context(lines)

    rewritten, output_name = _repair_binary_alignment_layout(
        line,
        0,
        lines,
        {"left_cf", "right_cf"},
        set(),
        context,
    )

    assert output_name == "output"
    assert rewritten is not None
    assert "torch.add(left_cf, right_cf), [1, 3, 20, 30]" in rewritten


def test_concat_and_terminal_tail_repairs_follow_channel_first_layout() -> None:
    concat_line = (
        "        concat = _apply_concat([left_cf, right_cf], axis=3, "
        "target_shape=[1, 20, 30, 6], fused='NONE')"
    )
    concat_context = _build_fast_precanonicalize_repair_context([concat_line])
    rewritten_concat, concat_name = _repair_concat_axis_from_input_layouts(
        concat_line,
        {"left_cf", "right_cf"},
        concat_context,
    )

    assert concat_name == "concat"
    assert rewritten_concat == "        concat = torch.cat([left_cf, right_cf], dim=1)"

    tail_line = (
        "        score = _align_tensor_to_target_shape("
        "torch.sub(torch.as_tensor(1.0, dtype=torch.float32, "
        "device=_module_device(self)), input_cf), [1, 20, 30])"
    )
    tail_context = _build_fast_precanonicalize_repair_context([tail_line])
    rewritten_tail, tail_name = _repair_terminal_classifier_tail_layout(
        tail_line,
        {"input_cf"},
        tail_context,
    )

    assert tail_name == "score"
    assert rewritten_tail is not None
    assert "input_cf), [1, 1, 20, 30])" in rewritten_tail


def test_dynamic_cf_binary_anchor_phase_normalizes_shared_shape() -> None:
    lines = [
        "        left, right = _align_binary_inputs_to_anchor(input_cf, scale, [1, 20, 30, 3])",
        "        output = _align_tensor_to_target_shape(torch.mul(left, right), [int(ref.shape[0]), 3, int(ref.shape[2]), int(ref.shape[3])])",
    ]
    context = _build_fast_precanonicalize_repair_context(lines)
    cf_like_names = {"input_cf"}

    changed = _repair_dynamic_cf_binary_anchor_shapes(
        lines,
        cf_like_names,
        context,
    )

    assert changed
    assert lines[0].endswith("(input_cf, scale, [1, 3, 20, 30])")
    assert {"left", "right"} <= cf_like_names
    assert context.static_shapes["left"] == [1, 3, 20, 30]
    assert context.static_shapes["right"] == [1, 3, 20, 30]


def test_singleton_reshape_cf_binary_repair_moves_feature_axis() -> None:
    lines = [
        "        reshaped = torch.reshape(source, [1, 1, 1, 16])",
        "        output = torch.add(reshaped, branch_cf)",
    ]
    cf_like_names: set[str] = set()

    changed = _repair_singleton_reshape_cf_binary_at(
        0,
        lines,
        cf_like_names,
    )

    assert changed
    assert lines[0] == "        reshaped = torch.reshape(source, [1, 16, 1, 1])"
    assert cf_like_names == {"reshaped"}


def test_softmax_and_reduce_max_repairs_use_channel_first_axis() -> None:
    softmax_line = (
        "        score = _apply_softmax(input=input_cf, axis=3, beta=1.0, "
        "target_shape=[1, 20, 30, 3])"
    )
    rewritten_softmax, softmax_lhs = _repair_cf_softmax_axis(
        softmax_line,
        {"input_cf"},
    )

    assert softmax_lhs == "score"
    assert rewritten_softmax is not None
    assert "axis=1" in rewritten_softmax
    assert "target_shape=[1, 3, 20, 30]" in rewritten_softmax

    reduce_line = (
        "        maximum = _reduce_max(input_cf, "
        "_normalize_axes([3], input_cf.ndim), True)"
    )
    rewritten_reduce, reduce_lhs = _repair_cf_reduce_max_axis(
        reduce_line,
        {"input_cf"},
    )

    assert reduce_lhs == "maximum"
    assert rewritten_reduce is not None
    assert "_normalize_axes([1], input_cf.ndim)" in rewritten_reduce


def test_prelu_and_gather_slice_propagate_channel_first_layout() -> None:
    cf_like_names = {"input_cf"}

    assert _propagate_cf_prelu_output(
        "        prelu_out = self.prelu_0(input_cf)",
        cf_like_names,
    )
    lines = ["        gathered = prelu_out[:, :, :, [0, 2]]"]
    assert _repair_cf_gather_slice_at(0, lines, cf_like_names)

    assert {"prelu_out", "gathered"} <= cf_like_names
    assert lines == ["        gathered = prelu_out[:, [0, 2], :, :]"]


def test_depth_to_space_gather_uses_indexed_layout_evidence() -> None:
    cf_lines = [
        "        gathered = input_cf[:, [0, 2], :, :]",
        "        output = self.conv_block_0(gathered.permute(0, 3, 1, 2).contiguous())",
    ]
    cf_context = _build_fast_precanonicalize_repair_context(cf_lines)
    cf_context.static_shapes["input_cf"] = [1, 4, 5, 6]
    cf_like_names = {"input_cf"}
    nhwc_like_names: set[str] = set()

    assert _repair_depth_to_space_gather_at(
        0,
        cf_lines,
        cf_like_names,
        nhwc_like_names,
        cf_context,
        candidate_line=cf_lines[0],
    )
    assert cf_lines[1] == "        output = self.conv_block_0(gathered)"
    assert {"gathered", "output"} <= cf_like_names
    assert cf_context.static_shapes["gathered"] == [1, 2, 5, 6]

    nhwc_lines = [
        "        depth_to_space_gather = input_nhwc[:, [0, 2], :, :]",
    ]
    nhwc_context = _build_fast_precanonicalize_repair_context(nhwc_lines)
    assert _repair_depth_to_space_gather_at(
        0,
        nhwc_lines,
        set(),
        {"input_nhwc"},
        nhwc_context,
    )
    assert nhwc_lines == [
        "        depth_to_space_gather = input_nhwc[:, :, :, [0, 2]]"
    ]


def test_nhwc_buffer_binary_alignment_permutates_matching_constant() -> None:
    lines = [
        "        self.register_buffer('scale', torch.zeros([1, 4, 5, 2], dtype=torch.float32), persistent=True)",
        "        left, right = _align_binary_inputs(input_cf, self.scale, [1, 2, 4, 5])",
    ]
    context = _build_fast_precanonicalize_repair_context(lines)

    assert _repair_nhwc_buffer_binary_alignment_at(1, lines, context)
    assert lines[1] == (
        "        left, right = _align_binary_inputs(input_cf, "
        "self.scale.permute(0, 3, 1, 2).contiguous(), [1, 2, 4, 5])"
    )
