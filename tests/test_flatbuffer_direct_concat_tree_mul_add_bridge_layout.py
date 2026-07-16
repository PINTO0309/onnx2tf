from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

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
    _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_STATS = {
    "optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains": 1,
}
_ZERO_STATS = {
    "optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains": 0,
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
    batch = -1 if dynamic_batch else 1
    names = {
        "x0": f"{prefix}x0_nhwc",
        "x1": f"{prefix}x1_nhwc",
        "x2": f"{prefix}x2_nhwc",
        "p0": f"{prefix}x0_nchw",
        "p1": f"{prefix}x1_nchw",
        "p2": f"{prefix}x2_nchw",
        "inner": f"{prefix}cat_h_nchw",
        "outer": f"{prefix}cat_c_nchw",
        "mul_const": f"{prefix}mul_const",
        "mul_out": f"{prefix}mul_out",
        "post": f"{prefix}mul_out_nhwc",
        "bias": f"{prefix}add_bias",
        "output": f"{prefix}y",
        "to_nchw": f"{prefix}to_nchw_perm",
        "to_nhwc": f"{prefix}to_nhwc_perm",
    }
    model_ir.inputs.extend([names["x0"], names["x1"], names["x2"]])
    model_ir.outputs.append(names["output"])

    for name, shape, signature in (
        (names["x0"], [1, 3, 2, 1], [batch, 3, 2, 1]),
        (names["x1"], [1, 4, 2, 1], [batch, 4, 2, 1]),
        (names["x2"], [1, 4, 2, 1], [batch, 4, 2, 1]),
        (names["p0"], [1, 1, 3, 2], [batch, 1, 3, 2]),
        (names["p1"], [1, 1, 4, 2], [batch, 1, 4, 2]),
        (names["p2"], [1, 1, 4, 2], [batch, 1, 4, 2]),
        (names["inner"], [1, 1, 7, 2], [batch, 1, 7, 2]),
        (names["outer"], [1, 2, 7, 2], [batch, 2, 7, 2]),
        (names["mul_out"], [1, 2, 7, 2], [batch, 2, 7, 2]),
        (names["post"], [1, 7, 2, 2], [batch, 7, 2, 2]),
        (names["output"], [1, 7, 2, 2], [batch, 7, 2, 2]),
    ):
        _tensor(model_ir, name, shape, signature=signature)
    _tensor(
        model_ir,
        names["mul_const"],
        [1, 2, 1, 1],
        data=np.arange(1, 3, dtype=np.float32).reshape(1, 2, 1, 1),
    )
    _tensor(
        model_ir,
        names["bias"],
        [1, 1, 1, 2],
        data=np.arange(2, dtype=np.float32).reshape(1, 1, 1, 2),
    )
    _tensor(
        model_ir,
        names["to_nchw"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _tensor(
        model_ir,
        names["to_nhwc"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )

    model_ir.operators.extend(
        [
            OperatorIR("TRANSPOSE", [names["x0"], names["to_nchw"]], [names["p0"]]),
            OperatorIR("TRANSPOSE", [names["x1"], names["to_nchw"]], [names["p1"]]),
            OperatorIR("TRANSPOSE", [names["x2"], names["to_nchw"]], [names["p2"]]),
            OperatorIR(
                "CONCATENATION",
                [names["p0"], names["p1"]],
                [names["inner"]],
                options={"axis": 2, "fusedActivationFunction": "NONE"},
                axis_semantics={"axis": "physical"},
                version=2,
                onnx_node_name=f"{prefix}inner_concat",
                onnx_op_type="Concat",
            ),
            OperatorIR(
                "CONCATENATION",
                [names["inner"], names["p2"]],
                [names["outer"]],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
                axis_semantics={"axis": "physical"},
                version=2,
                onnx_node_name=f"{prefix}outer_concat",
                onnx_op_type="Concat",
            ),
            OperatorIR(
                "MUL",
                [names["outer"], names["mul_const"]],
                [names["mul_out"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["mul_out"], names["to_nhwc"]],
                [names["post"]],
            ),
            OperatorIR("ADD", [names["post"], names["bias"]], [names["output"]]),
        ]
    )


def _model(
    *,
    branches: int = 1,
    dynamic_batch: bool = False,
) -> ModelIR:
    model_ir = ModelIR("concat_tree_mul_add_characterization")
    for branch_index in range(int(branches)):
        _add_chain(
            model_ir,
            prefix=f"branch{branch_index}_",
            dynamic_batch=dynamic_batch,
        )
    return model_ir


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _ZERO_STATS
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_concat_tree_rewrites_mixed_axis_tree(dynamic_batch: bool) -> None:
    model_ir = _model(dynamic_batch=dynamic_batch)

    stats = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    assert [operator.op_type for operator in model_ir.operators] == [
        "CONCATENATION",
        "CONCATENATION",
        "MUL",
        "ADD",
    ]
    inner, outer, mul, add = model_ir.operators
    assert inner.inputs == ["branch0_x0_nhwc", "branch0_x1_nhwc"]
    assert inner.outputs == ["branch0_cat_h_nchw"]
    assert inner.options == {
        "axis": 1,
        "fusedActivationFunction": "NONE",
    }
    assert outer.inputs == ["branch0_cat_h_nchw", "branch0_x2_nhwc"]
    assert outer.options == {
        "axis": 3,
        "fusedActivationFunction": "NONE",
    }
    for concat, node_name in (
        (inner, "branch0_inner_concat"),
        (outer, "branch0_outer_concat"),
    ):
        assert concat.axis_semantics == {"axis": "physical"}
        assert concat.version == 2
        assert concat.onnx_node_name == node_name
        assert concat.onnx_op_type == "Concat"
    assert mul.inputs == ["branch0_cat_c_nchw", "branch0_mul_const"]
    assert add.inputs == ["branch0_mul_out", "branch0_add_bias"]
    assert model_ir.tensors["branch0_cat_h_nchw"].shape == [1, 7, 2, 1]
    assert model_ir.tensors["branch0_cat_c_nchw"].shape == [1, 7, 2, 2]
    assert model_ir.tensors["branch0_mul_const"].shape == [1, 1, 1, 2]
    assert model_ir.tensors["branch0_mul_out"].shape == [1, 7, 2, 2]
    expected_signature = (
        [-1, 7, 2, 2] if dynamic_batch else [1, 7, 2, 2]
    )
    assert model_ir.tensors["branch0_cat_c_nchw"].shape_signature == (
        expected_signature
    )
    assert model_ir.tensors["branch0_mul_out"].shape_signature == (
        expected_signature
    )
    assert validate_model_ir_invariants(model_ir) == []


def test_concat_tree_rewrites_multiple_and_fixed_point() -> None:
    model_ir = _model(branches=2)

    first = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )
    after_first = _normalize(copy.deepcopy(model_ir))
    second = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert first == {
        "optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains": 2,
    }
    assert second == _ZERO_STATS
    assert _normalize(model_ir) == after_first


def test_concat_tree_accepts_scalar_mul_constant() -> None:
    model_ir = _model()
    constant = model_ir.tensors["branch0_mul_const"]
    constant.shape = [1]
    constant.shape_signature = [1]
    constant.data = np.asarray([2.0], dtype=np.float32)

    stats = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    assert model_ir.tensors["branch0_mul_const"].shape == [1]
    assert np.asarray(model_ir.tensors["branch0_mul_const"].data).tolist() == [
        2.0
    ]


def test_concat_tree_clones_shared_mul_constant_collision_safely() -> None:
    model_ir = _model()
    constant_name = "branch0_mul_const"
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    _tensor(model_ir, "constant_copy", [1, 2, 1, 1])
    model_ir.outputs.append("constant_copy")
    model_ir.operators.append(
        OperatorIR("IDENTITY", [constant_name], ["constant_copy"])
    )
    _tensor(
        model_ir,
        "branch0_mul_const_nhwc",
        [1],
        data=np.asarray([99.0], dtype=np.float32),
    )
    model_ir.outputs.append("branch0_mul_const_nhwc")

    stats = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    mul = next(op for op in model_ir.operators if op.outputs == ["branch0_mul_out"])
    assert mul.inputs[1] == "branch0_mul_const_nhwc_1"
    assert np.array_equal(model_ir.tensors[constant_name].data, original)
    assert np.asarray(
        model_ir.tensors["branch0_mul_const_nhwc"].data
    ).tolist() == [99.0]


def test_concat_tree_normalizes_negative_axes() -> None:
    model_ir = _model()
    model_ir.operators[3].options["axis"] = -2
    model_ir.operators[4].options["axis"] = -3

    stats = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    assert model_ir.operators[0].options["axis"] == 1
    assert model_ir.operators[1].options["axis"] == 3


def test_concat_tree_does_not_prune_unmatched_graph() -> None:
    model_ir = _model()
    _tensor(model_ir, "unused", [1])
    model_ir.tensors["branch0_to_nhwc_perm"].data = np.asarray(
        [0, 3, 1, 2],
        dtype=np.int32,
    )

    stats = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _ZERO_STATS
    assert "unused" in model_ir.tensors
    assert model_ir.metadata == {}


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_to_nhwc_perm"
            ].__setattr__(
                "data",
                np.asarray([0, 3, 1, 2], dtype=np.int32),
            ),
            id="wrong-post-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_mul_out"),
            id="public-mul-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append(
                "branch0_mul_out_nhwc"
            ),
            id="public-post-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[4].options.__setitem__(
                "axis",
                3,
            ),
            id="wrong-root-axis",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_cat_c_nchw"),
            id="public-root-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[3].inputs.pop(),
            id="single-inner-input",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_mul_const"
            ].__setattr__("data", None),
            id="dynamic-mul-constant",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_add_bias"
            ].__setattr__("data", None),
            id="dynamic-add-bias",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_add_bias"
            ].__setattr__(
                "data",
                np.zeros((1, 2, 1, 1), dtype=np.float32),
            ),
            id="nchw-add-bias",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "root_copy", [1, 2, 7, 2]),
                model_ir.outputs.append("root_copy"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_cat_c_nchw"],
                        ["root_copy"],
                    )
                ),
            ),
            id="root-fanout",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "inner_copy", [1, 1, 7, 2]),
                model_ir.outputs.append("inner_copy"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_cat_h_nchw"],
                        ["inner_copy"],
                    )
                ),
            ),
            id="inner-fanout",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "pre_copy", [1, 1, 3, 2]),
                model_ir.outputs.append("pre_copy"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_x0_nchw"],
                        ["pre_copy"],
                    )
                ),
            ),
            id="pre-fanout",
        ),
    ],
)
def test_concat_tree_preserves_existing_rejections(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="required rank-four metadata is not prevalidated",
)
@pytest.mark.parametrize(
    "case",
    [
        "missing-source",
        "missing-inner",
        "missing-outer",
        "missing-mul-output",
        "rank-three-source",
        "short-inner-signature",
        "short-outer-signature",
        "short-mul-signature",
    ],
)
def test_concat_tree_rejects_incomplete_metadata(case: str) -> None:
    model_ir = _model()
    missing_names = {
        "missing-source": "branch0_x0_nhwc",
        "missing-inner": "branch0_cat_h_nchw",
        "missing-outer": "branch0_cat_c_nchw",
        "missing-mul-output": "branch0_mul_out",
    }
    if case in missing_names:
        del model_ir.tensors[missing_names[case]]
    elif case == "rank-three-source":
        model_ir.tensors["branch0_x0_nhwc"].shape = [1, 3, 1]
        model_ir.tensors["branch0_x0_nhwc"].shape_signature = [1, 3, 1]
    else:
        signature_names = {
            "short-inner-signature": "branch0_cat_h_nchw",
            "short-outer-signature": "branch0_cat_c_nchw",
            "short-mul-signature": "branch0_mul_out",
        }
        model_ir.tensors[signature_names[case]].shape_signature = [1, 2, 7]

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="mutable/public Mul constants are rotated without ownership plan",
)
@pytest.mark.parametrize("ownership", ["public-input", "variable"])
def test_concat_tree_preserves_mul_constant_ownership(ownership: str) -> None:
    model_ir = _model()
    constant = model_ir.tensors["branch0_mul_const"]
    if ownership == "public-input":
        model_ir.inputs.append(constant.name)
    else:
        constant.is_variable = True

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="public Mul constant output is changed instead of cloned",
)
def test_concat_tree_clones_public_mul_constant_output() -> None:
    model_ir = _model()
    constant_name = "branch0_mul_const"
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    model_ir.outputs.append(constant_name)

    stats = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    mul = next(op for op in model_ir.operators if op.outputs == ["branch0_mul_out"])
    assert mul.inputs[1] != constant_name
    assert np.array_equal(model_ir.tensors[constant_name].data, original)


@pytest.mark.xfail(
    strict=True,
    reason="per-axis quantized dimensions are not remapped",
)
def test_concat_tree_remaps_per_axis_quantization() -> None:
    model_ir = _model()
    for name in (
        "branch0_cat_h_nchw",
        "branch0_cat_c_nchw",
        "branch0_mul_const",
        "branch0_mul_out",
    ):
        model_ir.tensors[name].quantization = QuantParamIR(
            scale=[0.1, 0.2],
            zero_point=[0, 0],
            quantized_dimension=1,
        )

    stats = _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    for name in (
        "branch0_cat_h_nchw",
        "branch0_cat_c_nchw",
        "branch0_mul_const",
        "branch0_mul_out",
    ):
        quantization = model_ir.tensors[name].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3


@pytest.mark.xfail(
    strict=True,
    reason="malformed Concat axes are not transactional no-ops",
)
@pytest.mark.parametrize("concat_index", [3, 4])
def test_concat_tree_rejects_malformed_axis(concat_index: int) -> None:
    model_ir = _model()
    model_ir.operators[concat_index].options["axis"] = None

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="metadata failure after constant rotation leaves partial mutation",
)
def test_concat_tree_rejects_late_inner_metadata_atomically() -> None:
    model_ir = _model()
    model_ir.tensors["branch0_cat_h_nchw"].shape_signature = [1, None, 7, 2]

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="duplicate producers, reverse order, and public aliases are accepted",
)
@pytest.mark.parametrize(
    "case",
    [
        "duplicate-post-output",
        "reverse-post-add",
        "reverse-inner-outer",
        "public-pre-output",
    ],
)
def test_concat_tree_rejects_invalid_topology(case: str) -> None:
    model_ir = _model()
    if case == "duplicate-post-output":
        model_ir.operators.insert(
            6,
            OperatorIR(
                "IDENTITY",
                ["branch0_x0_nhwc"],
                ["branch0_mul_out_nhwc"],
            ),
        )
    elif case == "reverse-post-add":
        model_ir.operators[6], model_ir.operators[7] = (
            model_ir.operators[7],
            model_ir.operators[6],
        )
    elif case == "reverse-inner-outer":
        model_ir.operators[3], model_ir.operators[4] = (
            model_ir.operators[4],
            model_ir.operators[3],
        )
    elif case == "public-pre-output":
        model_ir.inputs.append("branch0_x0_nchw")

    _assert_transactional_rejection(model_ir)


def test_concat_tree_keeps_raw_owner_and_ordered_boundaries() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains"
    )
    assert owner.end_lineno - owner.lineno + 1 == 356
    assert sum(isinstance(node, ast.While) for node in ast.walk(owner)) == 3

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    expected = {
        "_run_terminal_slice_concat_layout_recovery_sequence": (
            "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
            "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks",
        ),
        "_run_terminal_affine_concat_split_recovery_sequence": (
            "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
            "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks",
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
            "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains"
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
