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
    _repair_cf_pool_target_shape,
    _repair_cf_resize_target_shape,
    _repair_split_axis_from_consumers,
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
    ]
    context = _build_fast_precanonicalize_repair_context(lines)

    assert context.const_channel_counts == {"scale": 3}
    assert context.conv_block_in_channels == {"conv_block_0": 3}
    assert context.conv_block_out_channels == {"conv_block_0": 8}
    assert context.module_output_producers == {"out_cf": "conv_block_0"}
    assert context.module_input_consumers == {"input_nhwc": ["conv_block_0"]}
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
