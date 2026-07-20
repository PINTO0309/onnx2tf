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
    _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains,
)
from onnx2tf.tflite_builder.passes.reshape_transpose_collapse_layout import (
    _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains as _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains_owner,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    ATTENTION_RECOVERY_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.late_reshape_layout_orchestration import (
    LATE_RESHAPE_LAYOUT_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_STATS = {
    "optimized_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains": 1,
}
_ZERO_STATS = {
    "optimized_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains": 0,
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
        "input": f"{prefix}x",
        "shape1": f"{prefix}shape1",
        "r1": f"{prefix}r1",
        "perm1": f"{prefix}perm1",
        "t1": f"{prefix}t1",
        "shape2": f"{prefix}shape2",
        "r2": f"{prefix}r2",
        "perm2": f"{prefix}perm2",
        "output": f"{prefix}y",
    }
    model_ir.inputs.append(names["input"])
    model_ir.outputs.append(names["output"])

    for name, shape, signature in (
        (names["input"], [1, 6, 4], [batch, 6, 4]),
        (names["r1"], [1, 6, 4, 1], [batch, 6, 4, 1]),
        (names["t1"], [1, 1, 4, 6], [batch, 1, 4, 6]),
        (names["r2"], [1, 4, 2, 3], [batch, 4, 2, 3]),
        (names["output"], [1, 2, 3, 4], [batch, 2, 3, 4]),
    ):
        _tensor(model_ir, name, shape, signature=signature)
    for name, values in (
        (names["shape1"], [1, 6, 4, 1]),
        (names["perm1"], [0, 3, 2, 1]),
        (names["shape2"], [1, 4, 2, 3]),
        (names["perm2"], [0, 2, 3, 1]),
    ):
        _tensor(
            model_ir,
            name,
            [4],
            dtype="INT32",
            data=np.asarray(values, dtype=np.int32),
        )

    model_ir.operators.extend(
        [
            OperatorIR(
                "RESHAPE",
                [names["input"], names["shape1"]],
                [names["r1"]],
                options={
                    "newShape": [1, 6, 4, 1],
                    "onnxRawNewShape": [1, 6, 4, 1],
                    "fusedActivationFunction": "NONE",
                },
                version=2,
                onnx_node_name=f"{prefix}reshape1",
                onnx_op_type="Reshape",
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["r1"], names["perm1"]],
                [names["t1"]],
            ),
            OperatorIR(
                "RESHAPE",
                [names["t1"], names["shape2"]],
                [names["r2"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["r2"], names["perm2"]],
                [names["output"]],
            ),
        ]
    )


def _model(
    *,
    branches: int = 1,
    dynamic_batch: bool = False,
) -> ModelIR:
    model_ir = ModelIR("reshape_transpose_collapse_characterization")
    for branch_index in range(int(branches)):
        _add_chain(
            model_ir,
            prefix=f"branch{branch_index}_",
            dynamic_batch=dynamic_batch,
        )
    return model_ir


def _owner_wrapper_case(case: str) -> ModelIR:
    if case == "dynamic":
        return _model(dynamic_batch=True)
    if case == "multiple":
        return _model(branches=2)
    model_ir = _model()

    if case == "shared-shape":
        _tensor(model_ir, "shape_copy", [4], dtype="INT32")
        model_ir.outputs.append("shape_copy")
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_shape1"],
                ["shape_copy"],
            )
        )
        _tensor(
            model_ir,
            "branch0_shape1_nhwc",
            [1],
            dtype="INT32",
            data=np.asarray([99], dtype=np.int32),
        )
        model_ir.outputs.append("branch0_shape1_nhwc")
    elif case == "public-shape":
        model_ir.outputs.append("branch0_shape1")
    elif case == "public-input-shape":
        model_ir.inputs.append("branch0_shape1")
    elif case == "variable-shape":
        model_ir.tensors["branch0_shape1"].is_variable = True
    elif case == "wrong-shape-dtype":
        tensor = model_ir.tensors["branch0_shape1"]
        tensor.dtype = "FLOAT32"
        tensor.data = np.asarray(tensor.data, dtype=np.float32)
    elif case == "quantized-shape":
        model_ir.tensors["branch0_shape1"].quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )
    elif case == "missing-shape-data":
        model_ir.tensors["branch0_shape1"].data = None
    elif case == "unmatched":
        _tensor(model_ir, "unused", [1])
        model_ir.tensors["branch0_perm2"].data = np.asarray(
            [0, 3, 1, 2],
            dtype=np.int32,
        )
    elif case == "short-signature":
        model_ir.tensors["branch0_t1"].shape_signature = [1, 2]
    elif case == "reverse-topology":
        model_ir.operators[1], model_ir.operators[2] = (
            model_ir.operators[2],
            model_ir.operators[1],
        )
    elif case == "public-internal":
        model_ir.inputs.append("branch0_r1")
    elif case == "duplicate-source":
        model_ir.operators.extend(
            [
                OperatorIR(
                    "IDENTITY",
                    ["branch0_y"],
                    ["branch0_x"],
                ),
                OperatorIR(
                    "IDENTITY",
                    ["branch0_y"],
                    ["branch0_x"],
                ),
            ]
        )
    elif case == "missing-output":
        del model_ir.tensors["branch0_y"]
    return model_ir


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "dynamic",
        "multiple",
        "shared-shape",
        "public-shape",
        "public-input-shape",
        "variable-shape",
        "wrong-shape-dtype",
        "quantized-shape",
        "missing-shape-data",
        "unmatched",
        "short-signature",
        "reverse-topology",
        "public-internal",
        "duplicate-source",
        "missing-output",
    ],
)
def test_reshape_transpose_collapse_owner_and_wrapper_are_identical(
    case: str,
) -> None:
    owner_model = _owner_wrapper_case(case)
    wrapper_model = copy.deepcopy(owner_model)

    owner_stats = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains_owner(
            owner_model
        )
    )
    wrapper_stats = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            wrapper_model
        )
    )

    assert wrapper_stats == owner_stats
    assert _normalize(wrapper_model) == _normalize(owner_model)


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            model_ir
        )
    )

    assert stats == _ZERO_STATS
    assert _normalize(model_ir) == before


def test_reshape_transpose_collapse_rewrites_static_chain() -> None:
    model_ir = _model()

    stats = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            model_ir
        )
    )

    assert stats == _STATS
    assert len(model_ir.operators) == 1
    reshape = model_ir.operators[0]
    assert reshape.op_type == "RESHAPE"
    assert reshape.inputs == ["branch0_x", "branch0_shape1"]
    assert reshape.outputs == ["branch0_y"]
    assert reshape.options == {
        "newShape": [1, 2, 3, 4],
        "onnxRawNewShape": [1, 2, 3, 4],
        "fusedActivationFunction": "NONE",
    }
    assert reshape.version == 2
    assert reshape.onnx_node_name == "branch0_reshape1"
    assert reshape.onnx_op_type == "Reshape"
    assert np.asarray(model_ir.tensors["branch0_shape1"].data).tolist() == [
        1,
        2,
        3,
        4,
    ]
    assert set(model_ir.tensors) == {
        "branch0_x",
        "branch0_shape1",
        "branch0_y",
    }
    assert validate_model_ir_invariants(model_ir) == []


def test_reshape_transpose_collapse_rewrites_multiple_and_fixed_point() -> None:
    model_ir = _model(branches=2)

    first = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            model_ir
        )
    )
    after_first = _normalize(copy.deepcopy(model_ir))
    second = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            model_ir
        )
    )

    assert first == {
        "optimized_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains": 2,
    }
    assert second == _ZERO_STATS
    assert _normalize(model_ir) == after_first


def test_reshape_transpose_collapse_reuses_one_graph_index(
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

    stats = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            model_ir
        )
    )

    assert stats == {
        "optimized_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains": 2,
    }
    assert refresh_count == 1


def test_reshape_transpose_collapse_clones_shared_shape_collision_safely() -> None:
    model_ir = _model()
    shape_name = "branch0_shape1"
    original = np.asarray(model_ir.tensors[shape_name].data).copy()
    _tensor(model_ir, "shape_copy", [4], dtype="INT32")
    model_ir.outputs.append("shape_copy")
    model_ir.operators.append(
        OperatorIR("IDENTITY", [shape_name], ["shape_copy"])
    )
    _tensor(
        model_ir,
        "branch0_shape1_nhwc",
        [1],
        dtype="INT32",
        data=np.asarray([99], dtype=np.int32),
    )
    model_ir.outputs.append("branch0_shape1_nhwc")

    stats = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            model_ir
        )
    )

    assert stats == _STATS
    reshape = next(op for op in model_ir.operators if op.op_type == "RESHAPE")
    assert reshape.inputs[1] == "branch0_shape1_nhwc_1"
    assert np.array_equal(model_ir.tensors[shape_name].data, original)
    assert np.asarray(
        model_ir.tensors["branch0_shape1_nhwc"].data
    ).tolist() == [99]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_perm1"
            ].__setattr__(
                "data",
                np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            id="wrong-first-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_perm2"
            ].__setattr__(
                "data",
                np.asarray([0, 3, 1, 2], dtype=np.int32),
            ),
            id="wrong-second-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_r1"),
            id="public-reshape1-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_t1"),
            id="public-transpose1-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_r2"),
            id="public-reshape2-output",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir,
                "branch0_r1",
                "r1_copy",
            ),
            id="reshape1-fanout",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir,
                "branch0_t1",
                "t1_copy",
            ),
            id="transpose1-fanout",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir,
                "branch0_r2",
                "r2_copy",
            ),
            id="reshape2-fanout",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_r2"
            ].__setattr__(
                "shape",
                [1, 4, 4, 2],
            ),
            id="incompatible-spatial-product",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors.pop("branch0_y"),
            id="missing-output-tensor",
        ),
    ],
)
def test_reshape_transpose_collapse_preserves_existing_rejections(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


def _append_identity_consumer(
    model_ir: ModelIR,
    input_name: str,
    output_name: str,
) -> None:
    source = model_ir.tensors[input_name]
    _tensor(model_ir, output_name, list(source.shape))
    model_ir.outputs.append(output_name)
    model_ir.operators.append(
        OperatorIR("IDENTITY", [input_name], [output_name])
    )


def test_reshape_transpose_collapse_preserves_dynamic_batch() -> None:
    model_ir = _model(dynamic_batch=True)

    stats = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            model_ir
        )
    )

    assert stats == _STATS
    assert np.asarray(model_ir.tensors["branch0_shape1"].data).tolist() == [
        -1,
        2,
        3,
        4,
    ]
    reshape = model_ir.operators[0]
    assert reshape.options["newShape"] == [-1, 2, 3, 4]
    assert reshape.options["onnxRawNewShape"] == [-1, 2, 3, 4]


def test_reshape_transpose_collapse_does_not_prune_unmatched_graph() -> None:
    model_ir = _model()
    _tensor(model_ir, "unused", [1])
    model_ir.tensors["branch0_perm2"].data = np.asarray(
        [0, 3, 1, 2],
        dtype=np.int32,
    )

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    [
        "public-input",
        "variable",
        "wrong-dtype",
        "wrong-buffer",
        "quantized",
        "dynamic-data",
    ],
)
def test_reshape_transpose_collapse_rejects_unsafe_shape_constant(
    case: str,
) -> None:
    model_ir = _model()
    tensor = model_ir.tensors["branch0_shape1"]
    if case == "public-input":
        model_ir.inputs.append(tensor.name)
    elif case == "variable":
        tensor.is_variable = True
    elif case == "wrong-dtype":
        tensor.dtype = "FLOAT32"
        tensor.data = np.asarray(tensor.data, dtype=np.float32)
    elif case == "wrong-buffer":
        tensor.data = np.asarray(tensor.data, dtype=np.int64)
    elif case == "quantized":
        tensor.quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )
    elif case == "dynamic-data":
        tensor.data = None

    _assert_transactional_rejection(model_ir)


def test_reshape_transpose_collapse_clones_public_shape_output() -> None:
    model_ir = _model()
    shape_name = "branch0_shape1"
    original = np.asarray(model_ir.tensors[shape_name].data).copy()
    model_ir.outputs.append(shape_name)

    stats = (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
            model_ir
        )
    )

    assert stats == _STATS
    reshape = model_ir.operators[0]
    assert reshape.inputs[1] != shape_name
    assert np.array_equal(model_ir.tensors[shape_name].data, original)


@pytest.mark.parametrize(
    "name",
    [
        "branch0_x",
        "branch0_r1",
        "branch0_t1",
        "branch0_r2",
        "branch0_y",
    ],
)
def test_reshape_transpose_collapse_rejects_short_signature(
    name: str,
) -> None:
    model_ir = _model()
    model_ir.tensors[name].shape_signature = [1, 2]

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate-output-producer",
        "reverse-transpose-reshape",
        "public-internal-input",
        "reverse-source-producer",
        "duplicate-source-producer",
    ],
)
def test_reshape_transpose_collapse_rejects_invalid_topology(
    case: str,
) -> None:
    model_ir = _model()
    if case == "duplicate-output-producer":
        model_ir.operators.insert(
            3,
            OperatorIR(
                "IDENTITY",
                ["branch0_x"],
                ["branch0_y"],
            ),
        )
    elif case == "reverse-transpose-reshape":
        model_ir.operators[1], model_ir.operators[2] = (
            model_ir.operators[2],
            model_ir.operators[1],
        )
    elif case == "public-internal-input":
        model_ir.inputs.append("branch0_r1")
    elif case == "reverse-source-producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_y"],
                ["branch0_x"],
            )
        )
    elif case == "duplicate-source-producer":
        model_ir.operators.extend(
            [
                OperatorIR(
                    "IDENTITY",
                    ["branch0_y"],
                    ["branch0_x"],
                ),
                OperatorIR(
                    "IDENTITY",
                    ["branch0_y"],
                    ["branch0_x"],
                ),
            ]
        )

    _assert_transactional_rejection(model_ir)


def test_reshape_transpose_collapse_keeps_owner_wrapper_and_calls() -> None:
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "reshape_transpose_collapse_layout.py"
    )
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in pass_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains"
    )
    assert owner.end_lineno - owner.lineno + 1 == 399
    assert sum(isinstance(node, ast.While) for node in ast.walk(owner)) == 2
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(
                alias.name == "onnx2tf.tflite_builder.lower_from_onnx2tf"
                for alias in node.names
            )
            if isinstance(node, ast.Import)
            else node.module == "onnx2tf.tflite_builder.lower_from_onnx2tf"
        )
        for node in pass_tree.body
    )

    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains"
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    wrapper_call = wrapper.body[0].value
    assert isinstance(wrapper_call, ast.Call)
    assert isinstance(wrapper_call.func, ast.Name)
    assert (
        wrapper_call.func.id
        == "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains_pass"
    )
    assert len(wrapper_call.args) == 1
    assert isinstance(wrapper_call.args[0], ast.Name)
    assert wrapper_call.args[0].id == "model_ir"
    assert wrapper_call.keywords == []

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains"
    ]
    assert (
        len(calls)
        + ATTENTION_RECOVERY_PASS_IDS.count(
            "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains"
        )
        + LATE_RESHAPE_LAYOUT_PASS_IDS.count(
            "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains"
        )
        == 2
    )
    for call in calls:
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "model_ir"
        assert call.keywords == []
