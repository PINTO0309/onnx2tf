from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_add_concat_const_suffix_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.add_concat_suffix_layout import (
    run_add_concat_suffix_layout_cleanup,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32" if data is None else str(data.dtype).upper(),
        shape=list(shape),
        shape_signature=list(shape),
        data=None if data is None else np.asarray(data),
        is_variable=False,
    )


def _model(*, boundary: str | None = None) -> ModelIR:
    model_ir = ModelIR("add_concat_const_suffix_nhwc")
    model_ir.inputs = ["x0", "x1", "base"]
    model_ir.outputs = ["out"]
    perm_pre = [0, 1, 2, 3] if boundary == "permutation" else [0, 3, 1, 2]
    model_ir.tensors = {
        "x0": _tensor("x0", [1, 2, 3, 4]),
        "x1": _tensor("x1", [1, 2, 3, 4]),
        "base": _tensor("base", [1, 2, 3, 4]),
        "x0_nchw": _tensor("x0_nchw", [1, 4, 2, 3]),
        "x1_nchw": _tensor("x1_nchw", [1, 4, 2, 3]),
        "base_nchw": _tensor("base_nchw", [1, 4, 2, 3]),
        "add0_nchw": _tensor("add0_nchw", [1, 4, 2, 3]),
        "add1_nchw": _tensor("add1_nchw", [1, 4, 2, 3]),
        "concat_nchw": _tensor("concat_nchw", [1, 8, 2, 3]),
        "mul_nchw": _tensor("mul_nchw", [1, 8, 2, 3]),
        "suffix_nchw": _tensor("suffix_nchw", [1, 8, 2, 3]),
        "suffix_nhwc": _tensor("suffix_nhwc", [1, 2, 3, 8]),
        "out": _tensor("out", [1, 2, 3, 2]),
        "mul_const": _tensor(
            "mul_const",
            [1, 8, 1, 1],
            data=np.arange(8, dtype=np.float32).reshape(1, 8, 1, 1),
        ),
        "add_const": _tensor(
            "add_const",
            [1, 8, 1, 1],
            data=(
                None
                if boundary == "missing_constant"
                else np.arange(8, dtype=np.float32).reshape(1, 8, 1, 1)
            ),
        ),
        "conv_w": _tensor(
            "conv_w",
            [2, 1, 1, 8],
            data=np.ones([2, 1, 1, 8], dtype=np.float32),
        ),
        "conv_b": _tensor(
            "conv_b",
            [2],
            data=np.zeros([2], dtype=np.float32),
        ),
        "perm_pre": TensorIR(
            name="perm_pre",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(perm_pre, dtype=np.int32),
            is_variable=False,
        ),
        "perm_post": TensorIR(
            name="perm_post",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            is_variable=False,
        ),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0", "perm_pre"], ["x0_nchw"]),
        OperatorIR("TRANSPOSE", ["x1", "perm_pre"], ["x1_nchw"]),
        OperatorIR("TRANSPOSE", ["base", "perm_pre"], ["base_nchw"]),
        OperatorIR("ADD", ["x0_nchw", "base_nchw"], ["add0_nchw"]),
        OperatorIR("ADD", ["x1_nchw", "base_nchw"], ["add1_nchw"]),
        OperatorIR(
            "CONCATENATION",
            ["add0_nchw", "add1_nchw"],
            ["concat_nchw"],
            options={"axis": 2 if boundary == "concat_axis" else 1},
        ),
        OperatorIR("MUL", ["concat_nchw", "mul_const"], ["mul_nchw"]),
        OperatorIR("ADD", ["mul_nchw", "add_const"], ["suffix_nchw"]),
        OperatorIR(
            "TRANSPOSE",
            ["suffix_nchw", "perm_post"],
            ["suffix_nhwc"],
        ),
        OperatorIR(
            "CONV_2D",
            ["suffix_nhwc", "conv_w", "conv_b"],
            ["out"],
            options={"padding": "SAME"},
        ),
    ]
    side_sources = {
        "branch_adapter_fanout": "x0_nchw",
        "add_output_fanout": "add0_nchw",
        "concat_fanout": "concat_nchw",
        "mul_output_fanout": "mul_nchw",
        "mul_constant_fanout": "mul_const",
        "add_constant_fanout": "add_const",
    }
    if boundary in side_sources:
        source = side_sources[boundary]
        model_ir.tensors["side"] = _tensor(
            "side",
            list(model_ir.tensors[source].shape),
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    if boundary == "public_intermediate":
        model_ir.outputs.append("suffix_nchw")
    if boundary == "public_post_output":
        model_ir.outputs.append("suffix_nhwc")
    return model_ir


def _assert_model_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in actual.operators
    ] == [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in expected.operators
    ]
    assert actual.tensors.keys() == expected.tensors.keys()
    for name, tensor in actual.tensors.items():
        expected_tensor = expected.tensors[name]
        assert tensor.dtype == expected_tensor.dtype
        assert tensor.shape == expected_tensor.shape
        assert tensor.shape_signature == expected_tensor.shape_signature
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


def test_add_concat_suffix_layout_characterization() -> None:
    model_ir = _model()

    stats = _optimize_transpose_add_concat_const_suffix_nhwc_chains(model_ir)

    assert stats["optimized_transpose_add_concat_const_suffix_nhwc_chains"] == 1
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    add_ops = [
        op
        for op in model_ir.operators
        if op.op_type == "ADD" and op.outputs != ["suffix_nhwc"]
    ]
    assert {tuple(op.inputs) for op in add_ops} == {
        ("x0", "base"),
        ("x1", "base"),
    }
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.options["axis"] == 3
    np.testing.assert_array_equal(
        model_ir.tensors["mul_const"].data,
        np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["add_const"].data,
        np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8),
    )
    suffix_add = next(op for op in model_ir.operators if op.outputs == ["suffix_nhwc"])
    assert suffix_add.op_type == "ADD"
    conv_op = next(op for op in model_ir.operators if op.op_type == "CONV_2D")
    assert conv_op.inputs[0] == "suffix_nhwc"
    assert model_ir.tensors["concat_nchw"].shape == [1, 2, 3, 8]
    assert model_ir.tensors["suffix_nhwc"].shape == [1, 2, 3, 8]


@pytest.mark.parametrize(
    ("boundary", "constant_name", "suffix_op_type"),
    [
        ("mul_constant_fanout", "mul_const", "MUL"),
        ("add_constant_fanout", "add_const", "ADD"),
    ],
)
def test_add_concat_suffix_layout_clones_shared_suffix_constant(
    boundary: str,
    constant_name: str,
    suffix_op_type: str,
) -> None:
    model_ir = _model(boundary=boundary)
    original_data = np.asarray(model_ir.tensors[constant_name].data).copy()

    stats = _optimize_transpose_add_concat_const_suffix_nhwc_chains(model_ir)

    assert stats["optimized_transpose_add_concat_const_suffix_nhwc_chains"] == 1
    np.testing.assert_array_equal(
        model_ir.tensors[constant_name].data,
        original_data,
    )
    assert model_ir.tensors[constant_name].shape == [1, 8, 1, 1]
    side_op = next(op for op in model_ir.operators if op.outputs == ["side"])
    assert side_op.inputs == [constant_name]
    suffix_op = next(
        op
        for op in model_ir.operators
        if op.op_type == suffix_op_type
        and any(
            name.startswith(f"{constant_name}_layout") for name in op.inputs
        )
    )
    clone_name = next(
        name for name in suffix_op.inputs if name.startswith(f"{constant_name}_layout")
    )
    np.testing.assert_array_equal(
        model_ir.tensors[clone_name].data,
        np.transpose(original_data, [0, 2, 3, 1]),
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "branch_adapter_fanout",
        "add_output_fanout",
        "concat_fanout",
        "mul_output_fanout",
        "public_intermediate",
        "public_post_output",
        "permutation",
        "concat_axis",
        "missing_constant",
    ],
)
def test_add_concat_suffix_layout_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_add_concat_const_suffix_nhwc_chains(model_ir)

    assert stats["optimized_transpose_add_concat_const_suffix_nhwc_chains"] == 0
    _assert_model_equal(model_ir, original)


def test_add_concat_suffix_layout_runner_reuses_one_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = _model()
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_add_concat_suffix_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    stats_key = "optimized_transpose_add_concat_const_suffix_nhwc_chains"
    assert stats[stats_key] == 1
    assert refresh_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.add_concat_const_suffix_nhwc"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "branch_adapter_fanout",
        "add_output_fanout",
        "concat_fanout",
        "mul_output_fanout",
        "public_intermediate",
        "public_post_output",
        "permutation",
        "concat_axis",
        "missing_constant",
    ],
)
def test_add_concat_suffix_layout_runner_rejects_before_snapshot(
    boundary: str,
) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)
    diagnostics: list[dict[str, object]] = []

    stats = run_add_concat_suffix_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    stats_key = "optimized_transpose_add_concat_const_suffix_nhwc_chains"
    assert stats[stats_key] == 0
    assert len(diagnostics) == 1
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
    _assert_model_equal(model_ir, original)
