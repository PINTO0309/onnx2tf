from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.boundary_input_layout import (
    run_boundary_input_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.channel_slice_layout import (
    _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks,
    _optimize_boundary_input_transpose_channel_slice_blocks,
    _optimize_internal_transpose_channel_slice_nhwc_propagation_chains,
    _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    logical_layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        logical_layout=logical_layout,
    )


def test_boundary_input_ordered_cleanup_shares_index_and_layout_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR("boundary_input_ordered_cleanup")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 3, 4, 4], logical_layout="NHWC"),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_onnx_ncx_internal": _tensor(
            "x_onnx_ncx_internal",
            [1, 3, 4, 4],
            logical_layout="NCHW",
        ),
        "y": _tensor("y", [1, 3, 4, 4]),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x", "perm"],
            outputs=["x_onnx_ncx_internal"],
        ),
        OperatorIR(
            op_type="IDENTITY",
            inputs=["x_onnx_ncx_internal"],
            outputs=["y"],
        ),
    ]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_boundary_input_layout_cleanup(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"removed_boundary_input_layout_transpose": 1}
    assert refresh_count == 1
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].inputs == ["x"]
    assert "x_onnx_ncx_internal" not in model_ir.tensors
    assert "x_onnx_ncx_internal" not in layout_state.logical
    assert layout_state.logical_of("x") == "NHWC"
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_boundary_input_channel_slice_result_schema_is_stable() -> None:
    model_ir = ModelIR("boundary_input_channel_slice_result_schema")

    assert _optimize_boundary_input_transpose_channel_slice_blocks(model_ir) == {
        "removed_boundary_input_transpose": 0,
        "rewritten_boundary_channel_slices": 0,
        "rewritten_boundary_axis_ops": 0,
        "inserted_local_boundary_transposes": 0,
    }


def test_internal_channel_slice_result_schema_is_stable() -> None:
    model_ir = ModelIR("internal_channel_slice_result_schema")

    assert _optimize_internal_transpose_channel_slice_nhwc_propagation_chains(
        model_ir
    ) == {
        "removed_internal_transpose_channel_slice_stems": 0,
        "rewritten_internal_channel_slices": 0,
        "rewritten_internal_axis_ops": 0,
        "inserted_internal_local_transposes": 0,
    }


def test_channel_slice_muladd_bridge_result_schema_is_stable() -> None:
    model_ir = ModelIR("channel_slice_muladd_bridge_result_schema")

    assert _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains(
        model_ir
    ) == {"optimized_transpose_channel_slice_muladd_nhwc_bridge_chains": 0}


def test_boundary_stridedslice_qdq_concat_result_schema_is_stable() -> None:
    model_ir = ModelIR("boundary_stridedslice_qdq_concat_result_schema")

    assert _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks(
        model_ir
    ) == {
        "removed_boundary_input_transpose_stridedslice_blocks": 0,
        "rewritten_boundary_stridedslices": 0,
        "rewritten_boundary_qdq_concat_axis": 0,
        "removed_boundary_post_transposes": 0,
    }
