from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.spp_layout import run_spp_layout_cleanup


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
    model_ir = ModelIR("generic_two_island_spp_nhwc")
    model_ir.inputs = ["base_nhwc"] + [f"resize_source{index}" for index in range(4)]
    model_ir.outputs = ["y"]
    pre_perm = [0, 1, 2, 3] if boundary == "pre_permutation" else [0, 3, 1, 2]
    model_ir.tensors = {
        "base_nhwc": _tensor("base_nhwc", [1, 2, 3, 2]),
        "base_nchw": _tensor("base_nchw", [1, 2, 2, 3]),
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
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            is_variable=False,
        ),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["base_nhwc", "pre_perm"], ["base_nchw"])
    ]
    branch_outputs: list[str] = []
    for index in range(4):
        source = f"resize_source{index}"
        resized_nhwc = f"resized{index}_nhwc"
        resized_nchw = f"resized{index}_nchw"
        add_output = f"branch{index}_nchw"
        model_ir.tensors[source] = _tensor(source, [1, 1, 1, 2])
        model_ir.tensors[resized_nhwc] = _tensor(resized_nhwc, [1, 2, 3, 2])
        model_ir.tensors[resized_nchw] = _tensor(resized_nchw, [1, 2, 2, 3])
        model_ir.tensors[add_output] = _tensor(add_output, [1, 2, 2, 3])
        model_ir.operators.extend(
            [
                OperatorIR(
                    "IDENTITY" if boundary == "resize_producer" and index == 0 else "RESIZE_BILINEAR",
                    [source],
                    [resized_nhwc],
                ),
                OperatorIR(
                    "TRANSPOSE",
                    [resized_nhwc, "pre_perm"],
                    [resized_nchw],
                ),
                OperatorIR(
                    "ADD",
                    ["base_nchw", resized_nchw],
                    [add_output],
                ),
            ]
        )
        branch_outputs.append(add_output)

    tensors = {
        "cat0_nchw": _tensor("cat0_nchw", [1, 8, 2, 3]),
        "mul0_const": _tensor(
            "mul0_const",
            [1, 8, 1, 1],
            data=(
                None
                if boundary == "missing_mul0_constant"
                else np.arange(8, dtype=np.float32).reshape(1, 8, 1, 1)
            ),
        ),
        "mul0_nchw": _tensor("mul0_nchw", [1, 8, 2, 3]),
        "mul0_nhwc": _tensor("mul0_nhwc", [1, 2, 3, 8]),
        "bias0": _tensor(
            "bias0",
            [1, 1, 1, 8],
            data=np.zeros([1, 1, 1, 8], dtype=np.float32),
        ),
        "affine0": _tensor("affine0", [1, 2, 3, 8]),
        "weight0": _tensor(
            "weight0",
            [2, 1, 1, 8],
            data=np.ones([2, 1, 1, 8], dtype=np.float32),
        ),
        "conv_bias0": _tensor(
            "conv_bias0",
            [2],
            data=np.zeros([2], dtype=np.float32),
        ),
        "conv0_nhwc": _tensor("conv0_nhwc", [1, 2, 3, 2]),
        "conv0_nchw": _tensor("conv0_nchw", [1, 2, 2, 3]),
        "cat1_nchw": _tensor("cat1_nchw", [1, 4, 2, 3]),
        "mul1_const": _tensor(
            "mul1_const",
            [1, 4, 1, 1],
            data=(
                None
                if boundary == "missing_mul1_constant"
                else np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1)
            ),
        ),
        "mul1_nchw": _tensor("mul1_nchw", [1, 4, 2, 3]),
        "mul1_nhwc": _tensor("mul1_nhwc", [1, 2, 3, 4]),
        "bias1": _tensor(
            "bias1",
            [1, 1, 1, 4],
            data=np.zeros([1, 1, 1, 4], dtype=np.float32),
        ),
        "affine1": _tensor("affine1", [1, 2, 3, 4]),
        "weight1": _tensor(
            "weight1",
            [3, 1, 1, 4],
            data=np.ones([3, 1, 1, 4], dtype=np.float32),
        ),
        "conv_bias1": _tensor(
            "conv_bias1",
            [3],
            data=np.zeros([3], dtype=np.float32),
        ),
        "y": _tensor("y", [1, 2, 3, 3]),
    }
    model_ir.tensors.update(tensors)
    model_ir.operators.extend(
        [
            OperatorIR(
                "CONCATENATION",
                branch_outputs,
                ["cat0_nchw"],
                options={"axis": 2 if boundary == "concat0_axis" else 1},
            ),
            OperatorIR("MUL", ["cat0_nchw", "mul0_const"], ["mul0_nchw"]),
            OperatorIR("TRANSPOSE", ["mul0_nchw", "post_perm"], ["mul0_nhwc"]),
            OperatorIR("ADD", ["mul0_nhwc", "bias0"], ["affine0"]),
            OperatorIR(
                "CONV_2D",
                ["affine0", "weight0", "conv_bias0"],
                ["conv0_nhwc"],
            ),
            OperatorIR("TRANSPOSE", ["conv0_nhwc", "pre_perm"], ["conv0_nchw"]),
            OperatorIR(
                "CONCATENATION",
                ["base_nchw", "conv0_nchw"],
                ["cat1_nchw"],
                options={"axis": 2 if boundary == "concat1_axis" else 1},
            ),
            OperatorIR("MUL", ["cat1_nchw", "mul1_const"], ["mul1_nchw"]),
            OperatorIR("TRANSPOSE", ["mul1_nchw", "post_perm"], ["mul1_nhwc"]),
            OperatorIR("ADD", ["mul1_nhwc", "bias1"], ["affine1"]),
            OperatorIR(
                "CONV_2D",
                ["affine1", "weight1", "conv_bias1"],
                ["y"],
            ),
        ]
    )

    side_sources = {
        "branch_fanout": "branch0_nchw",
        "concat0_fanout": "cat0_nchw",
        "mul0_fanout": "mul0_nchw",
        "post0_fanout": "mul0_nhwc",
        "conv0_fanout": "conv0_nhwc",
        "concat1_fanout": "cat1_nchw",
        "mul1_fanout": "mul1_nchw",
        "post1_fanout": "mul1_nhwc",
    }
    if boundary in side_sources:
        source = side_sources[boundary]
        model_ir.tensors["side"] = _tensor(
            "side",
            list(model_ir.tensors[source].shape),
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    if boundary in {"shared_mul0_constant", "shared_mul1_constant"}:
        source = (
            "mul0_const"
            if boundary == "shared_mul0_constant"
            else "mul1_const"
        )
        model_ir.tensors["constant_side"] = _tensor(
            "constant_side",
            list(model_ir.tensors[source].shape),
        )
        model_ir.outputs.append("constant_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", [source], ["constant_side"])
        )
    if boundary == "public_pre":
        model_ir.outputs.append("base_nchw")
    elif boundary == "public_concat0":
        model_ir.outputs.append("cat0_nchw")
    elif boundary == "invalid_rank":
        model_ir.tensors["cat0_nchw"].shape = [1, 8, 6]
        model_ir.tensors["cat0_nchw"].shape_signature = [1, 8, 6]
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


def test_spp_layout_characterization() -> None:
    model_ir = _model()

    stats = _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"
    ] == 1
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    concat_ops = [op for op in model_ir.operators if op.op_type == "CONCATENATION"]
    assert len(concat_ops) == 2
    assert concat_ops[0].options["axis"] == 3
    assert concat_ops[1].options["axis"] == 3
    assert concat_ops[1].inputs == ["base_nhwc", "conv0_nhwc"]
    for index in range(4):
        branch = next(
            op for op in model_ir.operators if op.outputs == [f"branch{index}_nchw"]
        )
        assert branch.inputs == ["base_nhwc", f"resized{index}_nhwc"]
        assert model_ir.tensors[f"branch{index}_nchw"].shape == [1, 2, 3, 2]
    np.testing.assert_array_equal(
        model_ir.tensors["mul0_const"].data,
        np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["mul1_const"].data,
        np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),
    )
    assert model_ir.tensors["cat0_nchw"].shape == [1, 2, 3, 8]
    assert model_ir.tensors["cat1_nchw"].shape == [1, 2, 3, 4]
    affine0 = next(op for op in model_ir.operators if op.outputs == ["affine0"])
    affine1 = next(op for op in model_ir.operators if op.outputs == ["affine1"])
    assert affine0.inputs == ["mul0_nchw", "bias0"]
    assert affine1.inputs == ["mul1_nchw", "bias1"]


@pytest.mark.parametrize(
    ("boundary", "constant_name", "expected_data"),
    [
        (
            "shared_mul0_constant",
            "mul0_const",
            np.arange(8, dtype=np.float32).reshape(1, 8, 1, 1),
        ),
        (
            "shared_mul1_constant",
            "mul1_const",
            np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1),
        ),
    ],
)
def test_spp_layout_clones_shared_constant(
    boundary: str,
    constant_name: str,
    expected_data: np.ndarray,
) -> None:
    model_ir = _model(boundary=boundary)

    stats = _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"
    ] == 1
    np.testing.assert_array_equal(
        model_ir.tensors[constant_name].data,
        expected_data,
    )
    side_op = next(op for op in model_ir.operators if op.outputs == ["constant_side"])
    assert side_op.inputs == [constant_name]
    mul_output = "mul0_nchw" if constant_name == "mul0_const" else "mul1_nchw"
    mul_op = next(op for op in model_ir.operators if op.outputs == [mul_output])
    cloned_name = next(
        name for name in mul_op.inputs if name.startswith(f"{constant_name}_nhwc")
    )
    np.testing.assert_array_equal(
        model_ir.tensors[cloned_name].data,
        np.transpose(expected_data, [0, 2, 3, 1]),
    )


def test_spp_layout_remaps_constant_quantized_dimension() -> None:
    model_ir = _model()
    model_ir.tensors["mul0_const"].quantization = QuantParamIR(
        scale=[0.25] * 8,
        zero_point=[0] * 8,
        quantized_dimension=1,
    )

    stats = _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"
    ] == 1
    quantization = model_ir.tensors["mul0_const"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 3


@pytest.mark.parametrize(
    "boundary",
    [
        "branch_fanout",
        "concat0_fanout",
        "mul0_fanout",
        "post0_fanout",
        "conv0_fanout",
        "concat1_fanout",
        "mul1_fanout",
        "post1_fanout",
        "public_pre",
        "public_concat0",
        "pre_permutation",
        "concat0_axis",
        "concat1_axis",
        "resize_producer",
        "missing_mul0_constant",
        "missing_mul1_constant",
        "invalid_rank",
    ],
)
def test_spp_layout_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"
    ] == 0
    _assert_model_equal(model_ir, original)


def test_spp_layout_runner_reuses_one_index_and_syncs_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = _model()
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict[str, object]] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_spp_layout_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )

    assert stats[
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"
    ] == 1
    assert refresh_count == 1
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "layout.generic_spp_nhwc"
    assert diagnostics[0]["changed"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "branch_fanout",
        "concat0_fanout",
        "mul0_fanout",
        "post0_fanout",
        "conv0_fanout",
        "concat1_fanout",
        "mul1_fanout",
        "post1_fanout",
        "public_pre",
        "public_concat0",
        "pre_permutation",
        "concat0_axis",
        "concat1_axis",
        "resize_producer",
        "missing_mul0_constant",
        "missing_mul1_constant",
        "invalid_rank",
    ],
)
def test_spp_layout_runner_rejects_before_snapshot(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)
    diagnostics: list[dict[str, object]] = []

    stats = run_spp_layout_cleanup(model_ir, diagnostics=diagnostics)

    assert stats[
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains"
    ] == 0
    assert len(diagnostics) == 1
    assert diagnostics[0]["changed"] is False
    assert diagnostics[0]["skipped_by_precondition"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
    _assert_model_equal(model_ir, original)
