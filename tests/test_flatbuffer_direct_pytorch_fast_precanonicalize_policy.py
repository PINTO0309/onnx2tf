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
    _repair_cf_pool_target_shape,
    _repair_cf_reduce_max_axis,
    _repair_cf_resize_target_shape,
    _repair_cf_softmax_axis,
    _repair_concat_axis_from_input_layouts,
    _repair_dynamic_cf_binary_anchor_shapes,
    _repair_nhwc_average_pool_binary_bridge,
    _repair_split_axis_from_consumers,
    _repair_singleton_reshape_cf_binary_at,
    _repair_terminal_classifier_tail_layout,
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
