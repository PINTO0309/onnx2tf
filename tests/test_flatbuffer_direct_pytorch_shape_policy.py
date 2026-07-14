from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.pytorch_shape_policy import (
    _conv2d_output_spatial_shape_for_codegen,
    _conv2d_same_pad_padding_arg_for_codegen,
    _conv3d_output_spatial_shape_for_codegen,
    _conv3d_transpose_output_spatial_shape_for_codegen,
    _fast_precanonicalize_rank4_layout_hint,
    _infer_batch_matmul_shape_for_codegen,
    _infer_conv3d_ctor_params_for_codegen,
    _infer_conv3d_transpose_ctor_params_for_codegen,
    _infer_reduction_shape_for_codegen,
    _matmul_broadcast_shape_for_codegen,
    _normalize_cf_rank4_shape,
    _normalize_nhwc_rank4_shape,
    _reshape_special_layout_plan,
    _reshape_preserves_channel_last_sequence_for_codegen,
)


@pytest.mark.parametrize(
    ("padding_mode", "expected"),
    [
        ("SAME", [4, 4]),
        ("VALID", [3, 3]),
        ("unsupported", None),
    ],
)
def test_conv2d_output_spatial_shape_policy(
    padding_mode: str,
    expected: list[int] | None,
) -> None:
    assert _conv2d_output_spatial_shape_for_codegen(
        input_hw=[7, 8],
        kernel_hw=[3, 3],
        stride_hw=[2, 2],
        dilation_hw=[1, 1],
        padding_mode=padding_mode,
    ) == expected


@pytest.mark.parametrize(
    ("function", "padding_mode", "expected"),
    [
        (_conv3d_output_spatial_shape_for_codegen, "SAME", [2, 2, 3]),
        (_conv3d_output_spatial_shape_for_codegen, "VALID", [1, 1, 2]),
        (
            _conv3d_transpose_output_spatial_shape_for_codegen,
            "SAME",
            [6, 8, 10],
        ),
        (
            _conv3d_transpose_output_spatial_shape_for_codegen,
            "VALID",
            [7, 9, 11],
        ),
    ],
)
def test_conv3d_output_spatial_shape_policy(
    function,
    padding_mode: str,
    expected: list[int],
) -> None:
    assert function(
        input_dhw=[3, 4, 5],
        kernel_dhw=[3, 3, 3],
        stride_dhw=[2, 2, 2],
        dilation_dhw=[1, 1, 1],
        padding_mode=padding_mode,
    ) == expected


def test_conv_output_spatial_shape_policy_rejects_invalid_rank_and_extent() -> None:
    assert _conv2d_output_spatial_shape_for_codegen(
        input_hw=[7],
        kernel_hw=[3, 3],
        stride_hw=[1, 1],
        dilation_hw=[1, 1],
        padding_mode="SAME",
    ) is None
    assert _conv3d_transpose_output_spatial_shape_for_codegen(
        input_dhw=[0, 4, 5],
        kernel_dhw=[3, 3, 3],
        stride_dhw=[1, 1, 1],
        dilation_dhw=[1, 1, 1],
        padding_mode="VALID",
    ) is None


def test_conv3d_constructor_shape_policy_handles_channel_first_weights() -> None:
    assert _infer_conv3d_ctor_params_for_codegen(
        input_shape=[1, 3, 5, 6, 7],
        output_shape=[1, 4, 5, 6, 7],
        weight_shape=[4, 3, 1, 1, 1],
        options={
            "padding": "SAME",
            "strideD": 1,
            "strideH": 1,
            "strideW": 1,
            "dilationDFactor": 1,
            "dilationHFactor": 1,
            "dilationWFactor": 1,
        },
        input_logical_layout="NCDHW",
        output_logical_layout="NCDHW",
    ) == (3, 1, 4, [1, 1, 1])


def test_transpose_conv3d_constructor_shape_policy_handles_stride() -> None:
    assert _infer_conv3d_transpose_ctor_params_for_codegen(
        input_shape=[1, 3, 4, 5, 6],
        output_shape=[1, 4, 8, 10, 12],
        weight_shape=[3, 4, 2, 2, 2],
        options={
            "padding": "SAME",
            "strideD": 2,
            "strideH": 2,
            "strideW": 2,
            "dilationDFactor": 1,
            "dilationHFactor": 1,
            "dilationWFactor": 1,
        },
        input_logical_layout="NCDHW",
        output_logical_layout="NCDHW",
    ) == (3, 4, [2, 2, 2], 1)


def test_conv3d_constructor_shape_policies_preserve_invalid_fallbacks() -> None:
    assert _infer_conv3d_ctor_params_for_codegen(
        input_shape=None,
        output_shape=[1, 4, 5, 6, 7],
        weight_shape=[4, 3, 1, 1, 1],
        options=None,
    ) == (1, 1, 1, [1, 1, 1])
    assert _infer_conv3d_transpose_ctor_params_for_codegen(
        input_shape=[1, 3, 4, 5, 6],
        output_shape=[1, 4, 8, 10],
        weight_shape=[3, 4, 2, 2, 2],
        options=None,
    ) == (1, 4, [1, 1, 1], 1)


def test_conv2d_same_padding_policy_handles_channel_last_stored_shapes() -> None:
    assert _conv2d_same_pad_padding_arg_for_codegen(
        input_shape=[1, 1, 64, 1],
        output_shape=[1, 1, 64, 64],
        weight_shape=[64, 1, 1, 9],
        options={
            "padding": "SAME",
            "strideH": 1,
            "strideW": 1,
            "dilationHFactor": 1,
            "dilationWFactor": 1,
        },
        input_logical_layout="NCHW",
        output_logical_layout="NCHW",
    ) == [4, 4, 0, 0]


def test_conv2d_same_padding_policy_rejects_non_same_and_invalid_permutation() -> None:
    common = {
        "input_shape": [1, 3, 8, 8],
        "output_shape": [1, 4, 8, 8],
        "weight_shape": [4, 3, 3, 3],
    }
    assert _conv2d_same_pad_padding_arg_for_codegen(
        **common,
        options={"padding": "VALID"},
    ) is None
    assert _conv2d_same_pad_padding_arg_for_codegen(
        **common,
        options={"padding": "SAME"},
        input_pre_permute=[0, 2, 1],
    ) is None


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "layout", "expected"),
    [
        ([1, 6, 2, 3], [1, 6, 6], "NCHW", [0, 2, 3, 1]),
        ([1, 4, 2, 3, 5], [1, 30, 4], "NCDHW", [0, 2, 3, 4, 1]),
        ([1, 4, 8], [1, 8, 4], "NCW", [0, 2, 1]),
        ([1, 4, 8], [1, 7, 4], "NCW", None),
    ],
)
def test_reshape_preserved_sequence_shape_policy(
    input_shape: list[int],
    output_shape: list[int],
    layout: str,
    expected: list[int] | None,
) -> None:
    assert _reshape_preserves_channel_last_sequence_for_codegen(
        input_shape=input_shape,
        output_shape=output_shape,
        input_layout=layout,
    ) == expected


def test_matmul_shape_policy_handles_batch_broadcast_and_vectors() -> None:
    assert _matmul_broadcast_shape_for_codegen(
        lhs_batch=[2, 1, 4],
        rhs_batch=[1, 3, 4],
    ) == [2, 3, 4]
    assert _matmul_broadcast_shape_for_codegen(
        lhs_batch=[2],
        rhs_batch=[3],
    ) is None
    assert _infer_batch_matmul_shape_for_codegen(
        lhs_shape=[2, 3, 4],
        rhs_shape=[1, 4, 5],
        adj_x=False,
        adj_y=False,
    ) == [2, 3, 5]
    assert _infer_batch_matmul_shape_for_codegen(
        lhs_shape=[4],
        rhs_shape=[4],
        adj_x=False,
        adj_y=False,
    ) == [1, 1]


@pytest.mark.parametrize(
    ("axes", "keepdims", "expected"),
    [
        ([1], True, [2, 1, 4]),
        ([1], False, [2, 4]),
        (None, True, [1, 1, 1]),
        (None, False, []),
    ],
)
def test_reduction_shape_policy(
    axes: list[int] | None,
    keepdims: bool,
    expected: list[int],
) -> None:
    assert _infer_reduction_shape_for_codegen(
        input_shape=[2, 3, 4],
        axes=axes,
        keepdims=keepdims,
    ) == expected


@pytest.mark.parametrize(
    ("shape", "preferred_channel_count", "expected"),
    [
        ([1, 32, 8], None, None),
        ([1, 32, 8, 16], 32, "cf"),
        ([1, 8, 16, 32], 32, "nhwc"),
        ([1, 1, 8, 16], None, "cf"),
        ([1, 8, 16, 1], None, "nhwc"),
        ([1, 8, 16, 32], None, None),
    ],
)
def test_rank4_layout_hint(
    shape: list[int],
    preferred_channel_count: int | None,
    expected: str | None,
) -> None:
    assert (
        _fast_precanonicalize_rank4_layout_hint(
            shape,
            preferred_channel_count=preferred_channel_count,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("shape", "preferred_channel_count", "out_hw", "expected"),
    [
        ([1, 8, 16], None, None, [1, 8, 16]),
        ([1, 8, 16, 32], None, None, [1, 32, 8, 16]),
        ([1, 32, 8, 16], 32, None, [1, 32, 8, 16]),
        ([1, 8, 32, 16], 32, (8, 16), [1, 32, 8, 16]),
    ],
)
def test_normalize_cf_rank4_shape(
    shape: list[int],
    preferred_channel_count: int | None,
    out_hw: tuple[int, int] | None,
    expected: list[int],
) -> None:
    assert (
        _normalize_cf_rank4_shape(
            shape,
            preferred_channel_count=preferred_channel_count,
            out_hw=out_hw,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("shape", "preferred_channel_count", "expected"),
    [
        ([1, 8, 16], None, [1, 8, 16]),
        ([1, 8, 16, 32], None, [1, 8, 16, 32]),
        ([1, 32, 8, 16], 32, [1, 8, 16, 32]),
        ([1, 1, 8, 16], None, [1, 8, 16, 1]),
    ],
)
def test_normalize_nhwc_rank4_shape(
    shape: list[int],
    preferred_channel_count: int | None,
    expected: list[int],
) -> None:
    assert (
        _normalize_nhwc_rank4_shape(
            shape,
            preferred_channel_count=preferred_channel_count,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "input_layout", "output_layout", "expected"),
    [
        (
            [1, 6, 2, 3],
            [2, 3, 6],
            "NCHW",
            None,
            {"pre_perm": [0, 2, 3, 1], "reshape_shape": [2, 3, 6], "post_perm": None},
        ),
        (
            [2, 3, 6],
            [1, 6, 2, 3],
            None,
            "NCHW",
            {
                "pre_perm": None,
                "reshape_shape": [1, 2, 3, 6],
                "post_perm": [0, 3, 1, 2],
            },
        ),
        (
            [1, 1, 2, 3],
            [1, 3, 1, 2],
            None,
            None,
            {
                "pre_perm": [0, 3, 1, 2],
                "reshape_shape": [1, 3, 1, 2],
                "post_perm": None,
            },
        ),
        (
            [1, 1, 4, 6],
            [1, 6, 1, 1, 4],
            "NCHW",
            "NCDHW",
            {
                "pre_perm": [0, 3, 1, 2],
                "reshape_shape": [1, 6, 1, 1, 4],
                "post_perm": None,
            },
        ),
        (
            [1, 6, 2, 3],
            [1, 2, 3, 2, 3],
            "NCHW",
            None,
            {
                "pre_perm": [0, 2, 3, 1],
                "reshape_shape": [1, 2, 3, 2, 3],
                "post_perm": None,
            },
        ),
        ([1, 6, 2, 3], [1, 12, 3], "NHWC", "NCHW", None),
        (None, [1, 2], "NCHW", None, None),
    ],
)
def test_reshape_special_layout_plan(
    input_shape: list[int] | None,
    output_shape: list[int],
    input_layout: str | None,
    output_layout: str | None,
    expected: dict[str, object] | None,
) -> None:
    assert (
        _reshape_special_layout_plan(
            input_shape=input_shape,
            output_shape=output_shape,
            input_layout=input_layout,
            output_layout=output_layout,
        )
        == expected
    )
