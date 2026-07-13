from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.pytorch_shape_policy import (
    _fast_precanonicalize_rank4_layout_hint,
    _normalize_cf_rank4_shape,
    _normalize_nhwc_rank4_shape,
)


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
