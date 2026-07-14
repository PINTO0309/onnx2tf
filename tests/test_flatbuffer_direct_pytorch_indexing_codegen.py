from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.pytorch_indexing_codegen import (
    _direct_dynamic_gather_expr,
    _direct_gather_expr,
    _direct_gather_reshape_expr,
    _direct_slice_expr,
    _direct_strided_slice_expr,
    _direct_symbolic_strided_slice_expr,
    _is_suffix_flatten_gather_reshape,
    _reshape_is_plain_singleton_axis_drop,
    _should_elide_crd_to_dcr_gather_for_depth_to_space,
)


def test_plain_singleton_axis_drop_requires_one_removed_singleton() -> None:
    assert _reshape_is_plain_singleton_axis_drop([2, 1, 3], [2, 3])
    assert not _reshape_is_plain_singleton_axis_drop([1, 2, 1], [2, 1])
    assert not _reshape_is_plain_singleton_axis_drop(None, [2, 3])


def test_direct_slice_expr_handles_rank1_unbounded_and_clamped_ranges() -> None:
    assert (
        _direct_slice_expr(
            x_expr="x",
            begin_values=[1],
            size_values=[-1],
            input_rank=1,
        )
        == "x.reshape(-1)[1:]"
    )
    assert (
        _direct_slice_expr(
            x_expr="x",
            begin_values=[0, 2],
            size_values=[-1, 9],
            input_rank=2,
            input_shape=[4, 5],
        )
        == "x[:, 2:5]"
    )
    assert (
        _direct_slice_expr(
            x_expr="x", begin_values=[0], size_values=[1, 2], input_rank=1
        )
        is None
    )


def test_direct_strided_slice_expr_applies_masks_and_rejects_zero_stride() -> None:
    assert (
        _direct_strided_slice_expr(
            x_expr="x",
            begin_values=[7, 0],
            end_values=[9, 2147483647],
            stride_values=[1, 2],
            begin_mask=1,
            end_mask=0,
            input_rank=2,
        )
        == "x[:9, 0::2]"
    )
    assert (
        _direct_strided_slice_expr(
            x_expr="x",
            begin_values=[0],
            end_values=[1],
            stride_values=[0],
            begin_mask=0,
            end_mask=0,
            input_rank=1,
        )
        is None
    )


def test_symbolic_strided_slice_expr_uses_scalar_and_list_end_expressions() -> None:
    assert (
        _direct_symbolic_strided_slice_expr(
            x_expr="x",
            begin_values=[0],
            stride_values=[1],
            begin_mask=0,
            end_mask=0,
            input_rank=1,
            end_scalar_expr="limit",
        )
        == "x.reshape(-1)[:limit]"
    )
    assert (
        _direct_symbolic_strided_slice_expr(
            x_expr="x",
            begin_values=[0, 1],
            stride_values=[1, 2],
            begin_mask=0,
            end_mask=1,
            input_rank=2,
            end_list_expr="ends",
        )
        == "x[:, 1:(ends)[1]:2]"
    )


def test_direct_static_gather_expr_selects_scalar_flat_and_shaped_forms() -> None:
    assert _direct_gather_expr(
        params_expr="x",
        indices_values=[2],
        indices_shape=[],
        axis=-1,
        batch_dims=0,
        input_rank=2,
    ) == (
        "torch.index_select(x, 1, torch.as_tensor([2], dtype=torch.int64, "
        "device=x.device)).squeeze(1)"
    )
    assert (
        _direct_gather_expr(
            params_expr="x",
            indices_values=[2, 0],
            indices_shape=[2],
            axis=0,
            batch_dims=0,
            input_rank=1,
        )
        == "x.reshape(-1)[[2, 0]]"
    )
    assert _direct_gather_expr(
        params_expr="x",
        indices_values=[0, 1, 2, 3],
        indices_shape=[2, 2],
        axis=1,
        batch_dims=0,
        input_rank=3,
    ) == (
        "_reshape_gather_output(torch.index_select(x, 1, torch.as_tensor([0, 1, 2, 3], "
        "dtype=torch.int64, device=x.device)), x, [2, 2], axis=1)"
    )


def test_direct_dynamic_gather_uses_static_or_runtime_indices_shape() -> None:
    static_expr = _direct_dynamic_gather_expr(
        params_expr="x",
        indices_expr="idx",
        axis=1,
        batch_dims=0,
        input_rank=3,
        indices_name="idx",
        indices_shape=[2, 2],
        indices_shape_signature=[2, 2],
    )
    assert static_expr is not None and ", x, [2, 2], axis=1)" in static_expr
    dynamic_expr = _direct_dynamic_gather_expr(
        params_expr="x",
        indices_expr="idx",
        axis=1,
        batch_dims=0,
        input_rank=3,
        indices_name="idx",
        indices_shape=[1, 2],
        indices_shape_signature=[-1, 2],
    )
    assert dynamic_expr is not None and "_shape_tensor(idx" in dynamic_expr
    assert (
        _direct_dynamic_gather_expr(
            params_expr="x",
            indices_expr="idx",
            axis=1,
            batch_dims=0,
            input_rank=3,
            indices_name="shuffle_crd_to_dcr_indices",
        )
        is None
    )


def test_suffix_flatten_gather_reshape_requires_exact_static_suffix_product() -> None:
    assert _is_suffix_flatten_gather_reshape([1, 4, 2, 3], [1, 4, 6])
    assert _is_suffix_flatten_gather_reshape([-1, 4, 2, 3], [1, 4, 6])
    assert not _is_suffix_flatten_gather_reshape([1, 4, 2, -1], [1, 4, 6])
    assert not _is_suffix_flatten_gather_reshape([1, 4, 2, 3], [1, 4, 5])


def test_direct_gather_reshape_expr_supports_static_and_dynamic_indices() -> None:
    static_expr = _direct_gather_reshape_expr(
        params_expr="x",
        indices_expr="idx",
        indices_values=[0, 2],
        indices_shape=[2],
        indices_shape_signature=[2],
        axis=-1,
        batch_dims=0,
        input_rank=3,
        indices_name="idx",
        final_shape_values=[-1, 6],
    )
    assert static_expr == (
        "torch.reshape(torch.index_select(x, 2, torch.as_tensor([0, 2], "
        "dtype=torch.int64, device=x.device)), [int(x.shape[0]), 6])"
    )
    dynamic_expr = _direct_gather_reshape_expr(
        params_expr="x",
        indices_expr="idx",
        indices_values=None,
        indices_shape=[2, 2],
        indices_shape_signature=[2, 2],
        axis=1,
        batch_dims=0,
        input_rank=3,
        indices_name="idx",
        final_shape_values=[1, 4, 8],
    )
    assert (
        dynamic_expr is not None
        and "idx.to(dtype=torch.int64).reshape(-1)" in dynamic_expr
    )


def test_crd_to_dcr_gather_elision_requires_channel_first_depth_to_space_consumers() -> (
    None
):
    model_ir = ModelIR(name="depth_to_space_gather")
    model_ir.tensors["x"] = TensorIR(
        name="x", dtype="FLOAT32", shape=[1, 8, 2, 2], logical_layout="NCHW"
    )
    model_ir.tensors["gathered"] = TensorIR(
        name="gathered",
        dtype="FLOAT32",
        shape=[1, 8, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="DEPTH_TO_SPACE", inputs=["gathered"], outputs=["y"])
    )

    assert _should_elide_crd_to_dcr_gather_for_depth_to_space(
        model_ir=model_ir,
        params_name="x",
        indices_name="shuffle_crd_to_dcr_indices",
        output_name="gathered",
        axis=1,
        batch_dims=0,
    )
    model_ir.operators.append(
        OperatorIR(op_type="ADD", inputs=["gathered", "bias"], outputs=["z"])
    )
    assert not _should_elide_crd_to_dcr_gather_for_depth_to_space(
        model_ir=model_ir,
        params_name="x",
        indices_name="shuffle_crd_to_dcr_indices",
        output_name="gathered",
        axis=1,
        batch_dims=0,
    )
