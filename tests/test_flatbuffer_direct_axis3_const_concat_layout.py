from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_axis3_const_concat_bridge_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.axis3_const_concat_layout import (
    run_axis3_const_concat_layout_cleanup,
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


def _model(
    *,
    legacy_consumer: bool = True,
    post_count: int = 1,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("axis3_const_concat_bridge_nhwc")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = [f"y{index}" for index in range(post_count)]
    if legacy_consumer:
        model_ir.outputs.append("legacy_out")

    const_shape = [1, 4, 2, 1]
    if boundary == "constant_rank":
        const_shape = [1, 4, 2]
    elif boundary == "constant_shape":
        const_shape = [1, 5, 2, 1]
    const_data = np.arange(np.prod(const_shape), dtype=np.float32).reshape(
        const_shape
    )
    if boundary == "missing_constant":
        const_data = None

    pre_perm = [0, 1, 2, 3] if boundary == "pre_permutation" else [0, 3, 1, 2]
    post_perm = [0, 1, 2, 3] if boundary == "post_permutation" else [0, 2, 3, 1]
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 2, 3, 4]),
        "x_nchw": _tensor("x_nchw", [1, 4, 2, 3]),
        "const_nchw": _tensor("const_nchw", const_shape, data=const_data),
        "cat_nchw": _tensor("cat_nchw", [1, 4, 2, 4]),
        "pre_perm": TensorIR(
            name="pre_perm",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(pre_perm, dtype=np.int32),
            is_variable=False,
        ),
        "post_perm": TensorIR(
            name="post_perm",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(post_perm, dtype=np.int32),
            is_variable=False,
        ),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "pre_perm"], ["x_nchw"]),
        OperatorIR(
            "CONCATENATION",
            ["const_nchw", "x_nchw"],
            ["cat_nchw"],
            options={"axis": 2 if boundary == "concat_axis" else 3},
        ),
    ]
    for index in range(post_count):
        post_name = f"post{index}_nhwc"
        output_name = f"y{index}"
        model_ir.tensors[post_name] = _tensor(post_name, [1, 2, 4, 4])
        model_ir.tensors[output_name] = _tensor(output_name, [1, 2, 4, 4])
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["cat_nchw", "post_perm"],
                    [post_name],
                ),
                OperatorIR("RELU", [post_name], [output_name]),
            ]
        )
    if legacy_consumer:
        model_ir.tensors["legacy_out"] = _tensor(
            "legacy_out",
            [1, 4, 2, 4],
        )
        model_ir.operators.append(
            OperatorIR("RELU", ["cat_nchw"], ["legacy_out"])
        )

    if boundary == "adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            [1, 4, 2, 3],
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x_nchw"], ["adapter_side"])
        )
    elif boundary == "shared_constant":
        model_ir.tensors["constant_side"] = _tensor(
            "constant_side",
            const_shape,
        )
        model_ir.outputs.append("constant_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["const_nchw"], ["constant_side"])
        )
    elif boundary == "public_concat":
        model_ir.outputs.append("cat_nchw")
    elif boundary == "public_post":
        model_ir.outputs.append("post0_nhwc")
    elif boundary == "public_adapter":
        model_ir.outputs.append("x_nchw")
    elif boundary == "public_constant":
        model_ir.outputs.append("const_nchw")
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


def _assert_common_nhwc_rewrite(model_ir: ModelIR) -> None:
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["const_nchw", "x_nhwc"]
    assert concat_op.options["axis"] == 2
    assert model_ir.tensors["cat_nchw"].shape == [1, 2, 4, 4]
    np.testing.assert_array_equal(
        model_ir.tensors["const_nchw"].data,
        np.arange(8, dtype=np.float32).reshape(1, 4, 2, 1).transpose(0, 2, 3, 1),
    )


def test_axis3_const_concat_layout_bridges_legacy_nchw_consumers() -> None:
    model_ir = _model()

    stats = _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(model_ir)

    assert stats["optimized_transpose_axis3_const_concat_bridge_nhwc_chains"] == 1
    _assert_common_nhwc_rewrite(model_ir)
    transposes = [op for op in model_ir.operators if op.op_type == "TRANSPOSE"]
    assert len(transposes) == 1
    bridge = transposes[0]
    assert bridge.inputs == ["cat_nchw", "pre_perm"]
    assert model_ir.tensors[bridge.outputs[0]].shape == [1, 4, 2, 4]
    legacy = next(op for op in model_ir.operators if op.outputs == ["legacy_out"])
    assert legacy.inputs == bridge.outputs
    post_relu = next(op for op in model_ir.operators if op.outputs == ["y0"])
    assert post_relu.inputs == ["cat_nchw"]


def test_axis3_const_concat_layout_bypasses_all_post_branches() -> None:
    model_ir = _model(legacy_consumer=False, post_count=2)

    stats = _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(model_ir)

    assert stats["optimized_transpose_axis3_const_concat_bridge_nhwc_chains"] == 1
    _assert_common_nhwc_rewrite(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    for output_name in ["y0", "y1"]:
        post_relu = next(
            op for op in model_ir.operators if op.outputs == [output_name]
        )
        assert post_relu.inputs == ["cat_nchw"]


def test_axis3_const_concat_layout_retains_shared_input_adapter() -> None:
    model_ir = _model(
        legacy_consumer=False,
        boundary="adapter_fanout",
    )

    stats = _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(model_ir)

    assert stats["optimized_transpose_axis3_const_concat_bridge_nhwc_chains"] == 1
    _assert_common_nhwc_rewrite(model_ir)
    transposes = [op for op in model_ir.operators if op.op_type == "TRANSPOSE"]
    assert len(transposes) == 1
    assert transposes[0].outputs == ["x_nchw"]
    side = next(op for op in model_ir.operators if op.outputs == ["adapter_side"])
    assert side.inputs == ["x_nchw"]


@pytest.mark.parametrize(
    "boundary",
    [
        "public_concat",
        "public_post",
        "public_adapter",
        "public_constant",
        "pre_permutation",
        "post_permutation",
        "concat_axis",
        "constant_rank",
        "constant_shape",
        "missing_constant",
        "shared_constant",
    ],
)
def test_axis3_const_concat_layout_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(model_ir)

    assert stats["optimized_transpose_axis3_const_concat_bridge_nhwc_chains"] == 0
    _assert_model_equal(model_ir, original)


def test_axis3_const_concat_layout_runner_reuses_one_index(
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

    stats = run_axis3_const_concat_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    stats_key = (
        "optimized_transpose_axis3_const_concat_bridge_nhwc_chains"
    )
    assert stats[stats_key] == 1
    assert refresh_count == 1
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.axis3_const_concat_bridge_nhwc"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "public_concat",
        "public_post",
        "public_adapter",
        "public_constant",
        "pre_permutation",
        "post_permutation",
        "concat_axis",
        "constant_rank",
        "constant_shape",
        "missing_constant",
        "shared_constant",
    ],
)
def test_axis3_const_concat_layout_runner_rejects_before_snapshot(
    boundary: str,
) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)
    diagnostics: list[dict[str, object]] = []

    stats = run_axis3_const_concat_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    stats_key = (
        "optimized_transpose_axis3_const_concat_bridge_nhwc_chains"
    )
    assert stats[stats_key] == 0
    assert len(diagnostics) == 1
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
    _assert_model_equal(model_ir, original)
