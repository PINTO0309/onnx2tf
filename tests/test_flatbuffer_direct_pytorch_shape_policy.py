from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.pytorch_shape_policy import (
    _conv2d_output_spatial_shape_for_codegen,
    _conv3d_output_spatial_shape_for_codegen,
    _conv3d_transpose_output_spatial_shape_for_codegen,
    _fast_precanonicalize_rank4_layout_hint,
    _normalize_cf_rank4_shape,
    _normalize_nhwc_rank4_shape,
    _reshape_special_layout_plan,
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
