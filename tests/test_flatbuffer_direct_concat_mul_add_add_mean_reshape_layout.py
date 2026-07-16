from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.validation import (
    validate_model_ir_invariants,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_STATS = {
    "optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": 1,
}
_ZERO_STATS = {
    "optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": 0,
}


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if is_dataclass(value):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
    is_variable: bool = False,
    quantization: QuantParamIR | None = None,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=(
            list(signature) if signature is not None else list(shape)
        ),
        data=data,
        is_variable=bool(is_variable),
        quantization=quantization,
    )


def _add_chain(
    model_ir: ModelIR,
    *,
    prefix: str,
    dynamic_batch: bool = False,
) -> None:
    source0_signature = (
        [-1, 2, 2, 3] if dynamic_batch else [1, 2, 2, 3]
    )
    source1_signature = (
        [-1, 2, 2, 1] if dynamic_batch else [1, 2, 2, 1]
    )
    pre0_signature = (
        [-1, 3, 2, 2] if dynamic_batch else [1, 3, 2, 2]
    )
    pre1_signature = (
        [-1, 1, 2, 2] if dynamic_batch else [1, 1, 2, 2]
    )
    nchw_signature = (
        [-1, 4, 2, 2] if dynamic_batch else [1, 4, 2, 2]
    )
    mean_signature = (
        [-1, 4, 1, 1] if dynamic_batch else [1, 4, 1, 1]
    )
    output_signature = (
        [-1, 1, 1, 4] if dynamic_batch else [1, 1, 1, 4]
    )

    names = {
        "source0": f"{prefix}x0_nhwc",
        "source1": f"{prefix}x1_nhwc",
        "pre0": f"{prefix}x0_nchw",
        "pre1": f"{prefix}x1_nchw",
        "concat": f"{prefix}cat_nchw",
        "mul_const": f"{prefix}mul_const",
        "mul_out": f"{prefix}mul_out",
        "add0_const": f"{prefix}add0_const",
        "add0_out": f"{prefix}add0_out",
        "add1_const": f"{prefix}add1_const",
        "add1_out": f"{prefix}add1_out",
        "mean_axes": f"{prefix}mean_axes",
        "mean_out": f"{prefix}mean_out",
        "reshape_shape": f"{prefix}reshape_shape",
        "output": f"{prefix}y",
        "perm": f"{prefix}to_nchw_perm",
    }
    model_ir.inputs.extend([names["source0"], names["source1"]])
    model_ir.outputs.append(names["output"])

    _tensor(
        model_ir,
        names["source0"],
        [1, 2, 2, 3],
        signature=source0_signature,
    )
    _tensor(
        model_ir,
        names["source1"],
        [1, 2, 2, 1],
        signature=source1_signature,
    )
    _tensor(
        model_ir,
        names["pre0"],
        [1, 3, 2, 2],
        signature=pre0_signature,
    )
    _tensor(
        model_ir,
        names["pre1"],
        [1, 1, 2, 2],
        signature=pre1_signature,
    )
    for name in (
        names["concat"],
        names["mul_out"],
        names["add0_out"],
        names["add1_out"],
    ):
        _tensor(
            model_ir,
            name,
            [1, 4, 2, 2],
            signature=nchw_signature,
        )
    _tensor(
        model_ir,
        names["mul_const"],
        [1, 4, 1, 1],
        data=np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1, 1),
    )
    _tensor(
        model_ir,
        names["add0_const"],
        [1, 4, 1, 1],
        data=np.arange(5, 9, dtype=np.float32).reshape(1, 4, 1, 1),
    )
    _tensor(
        model_ir,
        names["add1_const"],
        [4, 1, 1],
        data=np.arange(9, 13, dtype=np.float32).reshape(4, 1, 1),
    )
    _tensor(
        model_ir,
        names["mean_axes"],
        [2],
        dtype="INT32",
        data=np.asarray([2, 3], dtype=np.int32),
    )
    _tensor(
        model_ir,
        names["mean_out"],
        [1, 4, 1, 1],
        signature=mean_signature,
    )
    _tensor(
        model_ir,
        names["reshape_shape"],
        [4],
        dtype="INT32",
        data=np.asarray([1, 1, 1, 4], dtype=np.int32),
    )
    _tensor(
        model_ir,
        names["output"],
        [1, 1, 1, 4],
        signature=output_signature,
    )
    _tensor(
        model_ir,
        names["perm"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )

    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [names["source0"], names["perm"]],
                [names["pre0"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["source1"], names["perm"]],
                [names["pre1"]],
            ),
            OperatorIR(
                "CONCATENATION",
                [names["pre0"], names["pre1"]],
                [names["concat"]],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
                axis_semantics={"axis": "physical"},
                version=2,
                onnx_node_name=f"{prefix}concat",
                onnx_op_type="Concat",
            ),
            OperatorIR(
                "MUL",
                [names["concat"], names["mul_const"]],
                [names["mul_out"]],
            ),
            OperatorIR(
                "ADD",
                [names["mul_out"], names["add0_const"]],
                [names["add0_out"]],
            ),
            OperatorIR(
                "ADD",
                [names["add0_out"], names["add1_const"]],
                [names["add1_out"]],
            ),
            OperatorIR(
                "MEAN",
                [names["add1_out"], names["mean_axes"]],
                [names["mean_out"]],
                options={"keepDims": True},
                version=2,
                onnx_node_name=f"{prefix}mean",
                onnx_op_type="ReduceMean",
            ),
            OperatorIR(
                "RESHAPE",
                [names["mean_out"], names["reshape_shape"]],
                [names["output"]],
            ),
        ]
    )


def _model(
    *,
    branches: int = 1,
    dynamic_batch: bool = False,
) -> ModelIR:
    model_ir = ModelIR("concat_mul_add_add_mean_reshape_characterization")
    for branch_index in range(int(branches)):
        _add_chain(
            model_ir,
            prefix=f"branch{branch_index}_",
            dynamic_batch=dynamic_batch,
        )
    return model_ir


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _ZERO_STATS
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_concat_mean_reshape_rewrites_ordinary_chain(
    dynamic_batch: bool,
) -> None:
    model_ir = _model(dynamic_batch=dynamic_batch)

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    assert [operator.op_type for operator in model_ir.operators] == [
        "CONCATENATION",
        "MUL",
        "ADD",
        "ADD",
        "MEAN",
        "RESHAPE",
    ]
    concat = model_ir.operators[0]
    assert concat.inputs == ["branch0_x0_nhwc", "branch0_x1_nhwc"]
    assert concat.options == {
        "axis": 3,
        "fusedActivationFunction": "NONE",
    }
    assert concat.axis_semantics == {"axis": "physical"}
    assert concat.version == 2
    assert concat.onnx_node_name == "branch0_concat"
    assert concat.onnx_op_type == "Concat"
    mean = model_ir.operators[4]
    assert mean.version == 2
    assert mean.onnx_node_name == "branch0_mean"
    assert mean.onnx_op_type == "ReduceMean"
    assert np.asarray(model_ir.tensors["branch0_mean_axes"].data).tolist() == [
        1,
        2,
    ]
    assert model_ir.tensors["branch0_mul_const"].shape == [1, 1, 1, 4]
    assert model_ir.tensors["branch0_add0_const"].shape == [1, 1, 1, 4]
    assert model_ir.tensors["branch0_add1_const"].shape == [1, 1, 4]
    for name in (
        "branch0_cat_nchw",
        "branch0_mul_out",
        "branch0_add0_out",
        "branch0_add1_out",
    ):
        assert model_ir.tensors[name].shape == [1, 2, 2, 4]
    assert model_ir.tensors["branch0_mean_out"].shape == [1, 1, 1, 4]
    assert validate_model_ir_invariants(model_ir) == []


def test_concat_mean_reshape_rewrites_multiple_and_fixed_point() -> None:
    model_ir = _model(branches=2)

    first = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )
    after_first = _normalize(copy.deepcopy(model_ir))
    second = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert first == {
        "optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": 2,
    }
    assert second == _ZERO_STATS
    assert _normalize(model_ir) == after_first


def test_concat_mean_reshape_reuses_one_graph_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = _model(branches=2)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == {
        "optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": 2,
    }
    assert refresh_count == 1


def test_concat_mean_reshape_accepts_scalar_affine_constants() -> None:
    model_ir = _model()
    for name, value in (
        ("branch0_mul_const", 2.0),
        ("branch0_add0_const", 3.0),
        ("branch0_add1_const", 4.0),
    ):
        tensor = model_ir.tensors[name]
        tensor.shape = [1]
        tensor.shape_signature = [1]
        tensor.data = np.asarray([value], dtype=np.float32)

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    for name, value in (
        ("branch0_mul_const", 2.0),
        ("branch0_add0_const", 3.0),
        ("branch0_add1_const", 4.0),
    ):
        assert model_ir.tensors[name].shape == [1]
        assert np.asarray(model_ir.tensors[name].data).tolist() == [value]


@pytest.mark.parametrize(
    "constant_name,op_output,data_input,expected_shape",
    [
        (
            "branch0_mul_const",
            "branch0_mul_out",
            "branch0_cat_nchw",
            [1, 1, 1, 4],
        ),
        (
            "branch0_add0_const",
            "branch0_add0_out",
            "branch0_mul_out",
            [1, 1, 1, 4],
        ),
        (
            "branch0_add1_const",
            "branch0_add1_out",
            "branch0_add0_out",
            [1, 1, 4],
        ),
    ],
)
def test_concat_mean_reshape_clones_shared_affine_constant(
    constant_name: str,
    op_output: str,
    data_input: str,
    expected_shape: list[int],
) -> None:
    model_ir = _model()
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    shared_output = f"{constant_name}_copy"
    _tensor(model_ir, shared_output, list(original.shape))
    model_ir.outputs.append(shared_output)
    model_ir.operators.append(
        OperatorIR("IDENTITY", [constant_name], [shared_output])
    )
    collision_name = f"{constant_name}_nhwc"
    _tensor(
        model_ir,
        collision_name,
        [1],
        data=np.asarray([99.0], dtype=np.float32),
    )
    model_ir.outputs.append(collision_name)

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    operator = next(
        candidate
        for candidate in model_ir.operators
        if candidate.outputs == [op_output]
    )
    clone_name = next(name for name in operator.inputs if name != data_input)
    assert clone_name == f"{constant_name}_nhwc_1"
    assert np.array_equal(model_ir.tensors[constant_name].data, original)
    assert model_ir.tensors[clone_name].shape == expected_shape


def test_concat_mean_reshape_clones_shared_mean_axes() -> None:
    model_ir = _model()
    _tensor(model_ir, "axes_copy", [2], dtype="INT32")
    model_ir.outputs.append("axes_copy")
    model_ir.operators.append(
        OperatorIR("IDENTITY", ["branch0_mean_axes"], ["axes_copy"])
    )

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    mean = next(op for op in model_ir.operators if op.op_type == "MEAN")
    assert mean.inputs[1] == "branch0_mean_axes_nhwc"
    assert np.asarray(model_ir.tensors["branch0_mean_axes"].data).tolist() == [
        2,
        3,
    ]
    assert np.asarray(
        model_ir.tensors["branch0_mean_axes_nhwc"].data
    ).tolist() == [1, 2]


def test_concat_mean_reshape_updates_exact_old_mean_shape() -> None:
    model_ir = _model()
    model_ir.tensors["branch0_reshape_shape"].data = np.asarray(
        [1, 4, 1, 1],
        dtype=np.int32,
    )

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    assert np.asarray(
        model_ir.tensors["branch0_reshape_shape"].data
    ).tolist() == [1, 1, 1, 4]


def test_concat_mean_reshape_does_not_prune_unmatched_graph() -> None:
    model_ir = _model()
    _tensor(model_ir, "unused", [1])
    model_ir.tensors["branch0_to_nchw_perm"].data = np.asarray(
        [0, 2, 3, 1],
        dtype=np.int32,
    )

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _ZERO_STATS
    assert "unused" in model_ir.tensors
    assert model_ir.metadata == {}


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.operators[2].options.__setitem__(
                "axis",
                3,
            ),
            id="wrong-concat-axis",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[2].inputs.pop(),
            id="single-concat-input",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_cat_nchw"),
            id="public-concat-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_add1_out"),
            id="public-add1-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_mean_out"),
            id="public-mean-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[6].options.__setitem__(
                "keepDims",
                False,
            ),
            id="mean-without-keepdims",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_mul_const"
            ].__setattr__("data", None),
            id="dynamic-first-constant",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "mean_copy", [1, 4, 1, 1]),
                model_ir.outputs.append("mean_copy"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_mean_out"],
                        ["mean_copy"],
                    )
                ),
            ),
            id="mean-fanout",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "concat_copy", [1, 4, 2, 2]),
                model_ir.outputs.append("concat_copy"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_cat_nchw"],
                        ["concat_copy"],
                    )
                ),
            ),
            id="concat-fanout",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "pre_copy", [1, 3, 2, 2]),
                model_ir.outputs.append("pre_copy"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_x0_nchw"],
                        ["pre_copy"],
                    )
                ),
            ),
            id="pre-adapter-fanout",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[7].inputs.pop(),
            id="short-reshape-inputs",
        ),
    ],
)
def test_concat_mean_reshape_preserves_existing_rejections(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    [
        "missing-concat",
        "missing-mul-output",
        "missing-add0-output",
        "missing-add1-output",
        "missing-mean-output",
        "rank-three-source",
        "short-concat-signature",
        "short-mul-signature",
        "short-add0-signature",
        "short-add1-signature",
        "short-mean-signature",
    ],
)
def test_concat_mean_reshape_rejects_incomplete_metadata(case: str) -> None:
    model_ir = _model()
    tensor_names = {
        "missing-concat": "branch0_cat_nchw",
        "missing-mul-output": "branch0_mul_out",
        "missing-add0-output": "branch0_add0_out",
        "missing-add1-output": "branch0_add1_out",
        "missing-mean-output": "branch0_mean_out",
    }
    if case in tensor_names:
        del model_ir.tensors[tensor_names[case]]
    elif case == "rank-three-source":
        model_ir.tensors["branch0_x0_nhwc"].shape = [1, 2, 3]
        model_ir.tensors["branch0_x0_nhwc"].shape_signature = [1, 2, 3]
    else:
        signature_names = {
            "short-concat-signature": "branch0_cat_nchw",
            "short-mul-signature": "branch0_mul_out",
            "short-add0-signature": "branch0_add0_out",
            "short-add1-signature": "branch0_add1_out",
            "short-mean-signature": "branch0_mean_out",
        }
        model_ir.tensors[signature_names[case]].shape_signature = [1, 4, 2]

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "constant_name,ownership",
    [
        ("branch0_mul_const", "public-input"),
        ("branch0_mul_const", "variable"),
        ("branch0_add0_const", "public-input"),
        ("branch0_add0_const", "variable"),
        ("branch0_add1_const", "public-input"),
        ("branch0_add1_const", "variable"),
    ],
)
def test_concat_mean_reshape_preserves_affine_constant_ownership(
    constant_name: str,
    ownership: str,
) -> None:
    model_ir = _model()
    if ownership == "public-input":
        model_ir.inputs.append(constant_name)
    else:
        model_ir.tensors[constant_name].is_variable = True

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "constant_name,op_output,data_input",
    [
        ("branch0_mul_const", "branch0_mul_out", "branch0_cat_nchw"),
        ("branch0_add0_const", "branch0_add0_out", "branch0_mul_out"),
        ("branch0_add1_const", "branch0_add1_out", "branch0_add0_out"),
    ],
)
def test_concat_mean_reshape_clones_public_affine_constant_output(
    constant_name: str,
    op_output: str,
    data_input: str,
) -> None:
    model_ir = _model()
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    model_ir.outputs.append(constant_name)

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    operator = next(op for op in model_ir.operators if op.outputs == [op_output])
    clone_name = next(name for name in operator.inputs if name != data_input)
    assert clone_name != constant_name
    assert np.array_equal(model_ir.tensors[constant_name].data, original)


def test_concat_mean_reshape_remaps_per_axis_quantization() -> None:
    model_ir = _model()
    rank4_names = (
        "branch0_cat_nchw",
        "branch0_mul_const",
        "branch0_mul_out",
        "branch0_add0_const",
        "branch0_add0_out",
        "branch0_add1_out",
        "branch0_mean_out",
    )
    for name in rank4_names:
        model_ir.tensors[name].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3, 0.4],
            zero_point=[0, 0, 0, 0],
            quantized_dimension=1,
        )
    model_ir.tensors["branch0_add1_const"].quantization = QuantParamIR(
        scale=[0.1, 0.2, 0.3, 0.4],
        zero_point=[0, 0, 0, 0],
        quantized_dimension=0,
    )

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    for name in rank4_names:
        quantization = model_ir.tensors[name].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3
    add1_quantization = model_ir.tensors[
        "branch0_add1_const"
    ].quantization
    assert isinstance(add1_quantization, QuantParamIR)
    assert add1_quantization.quantized_dimension == 2


@pytest.mark.parametrize(
    "case",
    ["public-input", "variable", "wrong-dtype", "wrong-buffer", "quantized"],
)
def test_concat_mean_reshape_rejects_unsafe_mean_axes(case: str) -> None:
    model_ir = _model()
    axes = model_ir.tensors["branch0_mean_axes"]
    if case == "public-input":
        model_ir.inputs.append(axes.name)
    elif case == "variable":
        axes.is_variable = True
    elif case == "wrong-dtype":
        axes.dtype = "FLOAT32"
    elif case == "wrong-buffer":
        axes.data = np.asarray([2.0, 3.0], dtype=np.float32)
    elif case == "quantized":
        axes.quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )

    _assert_transactional_rejection(model_ir)


def test_concat_mean_reshape_clones_public_mean_axes_output() -> None:
    model_ir = _model()
    original = np.asarray(model_ir.tensors["branch0_mean_axes"].data).copy()
    model_ir.outputs.append("branch0_mean_axes")

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    mean = next(op for op in model_ir.operators if op.op_type == "MEAN")
    assert mean.inputs[1] != "branch0_mean_axes"
    assert np.array_equal(model_ir.tensors["branch0_mean_axes"].data, original)


@pytest.mark.parametrize(
    "case",
    ["public-input", "variable", "wrong-dtype", "quantized"],
)
def test_concat_mean_reshape_rejects_unsafe_reshape_shape(case: str) -> None:
    model_ir = _model()
    shape_tensor = model_ir.tensors["branch0_reshape_shape"]
    shape_tensor.data = np.asarray([1, 4, 1, 1], dtype=np.int32)
    if case == "public-input":
        model_ir.inputs.append(shape_tensor.name)
    elif case == "variable":
        shape_tensor.is_variable = True
    elif case == "wrong-dtype":
        shape_tensor.dtype = "FLOAT32"
    elif case == "quantized":
        shape_tensor.quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize("ownership", ["public-output", "shared"])
def test_concat_mean_reshape_clones_owned_reshape_shape(
    ownership: str,
) -> None:
    model_ir = _model()
    shape_name = "branch0_reshape_shape"
    shape_tensor = model_ir.tensors[shape_name]
    shape_tensor.data = np.asarray([1, 4, 1, 1], dtype=np.int32)
    original = np.asarray(shape_tensor.data).copy()
    if ownership == "public-output":
        model_ir.outputs.append(shape_name)
    else:
        _tensor(model_ir, "shape_copy", [4], dtype="INT32")
        model_ir.outputs.append("shape_copy")
        model_ir.operators.append(
            OperatorIR("IDENTITY", [shape_name], ["shape_copy"])
        )

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    reshape = next(op for op in model_ir.operators if op.op_type == "RESHAPE")
    assert reshape.inputs[1] != shape_name
    assert np.array_equal(model_ir.tensors[shape_name].data, original)


@pytest.mark.parametrize("case", ["add0", "add1", "axes"])
def test_concat_mean_reshape_rejects_late_constant_error_atomically(
    case: str,
) -> None:
    model_ir = _model()
    if case == "add0":
        tensor = model_ir.tensors["branch0_add0_const"]
        tensor.shape = [2, 2]
        tensor.shape_signature = [2, 2]
        tensor.data = np.zeros((2, 2), dtype=np.float32)
    elif case == "add1":
        tensor = model_ir.tensors["branch0_add1_const"]
        tensor.shape = [2, 2]
        tensor.shape_signature = [2, 2]
        tensor.data = np.zeros((2, 2), dtype=np.float32)
    elif case == "axes":
        model_ir.tensors["branch0_mean_axes"].data = np.asarray(
            [5],
            dtype=np.int32,
        )

    _assert_transactional_rejection(model_ir)


def test_concat_mean_reshape_rejects_late_mean_metadata_atomically() -> None:
    model_ir = _model()
    model_ir.tensors["branch0_reshape_shape"].data = np.asarray(
        [1, 4, 1, 1],
        dtype=np.int32,
    )
    model_ir.tensors["branch0_mean_out"].shape = [1, None, 1, 1]

    _assert_transactional_rejection(model_ir)


def test_concat_mean_reshape_rejects_malformed_concat_axis() -> None:
    model_ir = _model()
    model_ir.operators[2].options["axis"] = None

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    ["duplicate-mean-output", "reverse-mean-reshape", "public-pre-output"],
)
def test_concat_mean_reshape_rejects_invalid_topology(case: str) -> None:
    model_ir = _model()
    if case == "duplicate-mean-output":
        model_ir.operators.insert(
            6,
            OperatorIR(
                "IDENTITY",
                ["branch0_x0_nhwc"],
                ["branch0_mean_out"],
            ),
        )
    elif case == "reverse-mean-reshape":
        model_ir.operators[6], model_ir.operators[7] = (
            model_ir.operators[7],
            model_ir.operators[6],
        )
    elif case == "public-pre-output":
        model_ir.inputs.append("branch0_x0_nchw")

    _assert_transactional_rejection(model_ir)


def test_concat_mean_reshape_accepts_identity_axis_mapping() -> None:
    model_ir = _model()
    model_ir.tensors["branch0_mean_axes"].shape = [1]
    model_ir.tensors["branch0_mean_axes"].shape_signature = [1]
    model_ir.tensors["branch0_mean_axes"].data = np.asarray(
        [0],
        dtype=np.int32,
    )

    stats = _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    assert np.asarray(model_ir.tensors["branch0_mean_axes"].data).tolist() == [
        0,
    ]


def test_concat_mean_reshape_keeps_raw_owner_and_ordered_boundaries() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains"
    )
    assert owner.end_lineno - owner.lineno + 1 == 869
    assert sum(isinstance(node, ast.While) for node in ast.walk(owner)) == 2

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    expected = {
        "_run_terminal_slice_concat_layout_recovery_sequence": (
            "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
            "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains",
        ),
        "_run_terminal_affine_concat_split_recovery_sequence": (
            "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
            "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains",
        ),
    }
    observed: dict[str, tuple[str, str]] = {}
    for statement in lowerer.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        if statement.name not in expected:
            continue
        calls = [
            candidate.value
            for candidate in statement.body
            if isinstance(candidate, ast.Expr)
            and isinstance(candidate.value, ast.Call)
            and isinstance(candidate.value.func, ast.Name)
        ]
        call_names = [call.func.id for call in calls]
        index = call_names.index(
            "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains"
        )
        observed[statement.name] = (
            call_names[index - 1],
            call_names[index + 1],
        )
        assert len(calls[index].args) == 1
        assert isinstance(calls[index].args[0], ast.Name)
        assert calls[index].args[0].id == "model_ir"
        assert calls[index].keywords == []
    assert observed == expected
