from __future__ import annotations

import ast
import copy
from copy import deepcopy
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
    _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.elementwise_roundtrip_nchw_nhwc_layout import (
    _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains as _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains_owner,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LAYOUT_RECOVERY_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_STATS = {
    "optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": 1,
}
_ZERO_STATS = {
    "optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": 0,
}


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
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


def _fingerprint(model_ir: ModelIR) -> Any:
    return _normalize(model_ir)


def _make_model_ir(
    *,
    leak_pre_transpose: bool = False,
    expose_post_output: bool = False,
) -> ModelIR:
    model_ir = ModelIR("elementwise_roundtrip_nchw_nhwc_test")
    model_ir.inputs = ["x_nchw", "y_nchw"]
    model_ir.outputs = ["root_nchw" if expose_post_output else "final"]
    if leak_pre_transpose:
        model_ir.outputs.append("leaked")

    def _add_tensor(
        name: str,
        shape: list[int],
        *,
        dtype: str = "FLOAT32",
        data: np.ndarray | None = None,
    ) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(value) for value in shape],
            shape_signature=[int(value) for value in shape],
            data=data,
            is_variable=data is None,
            onnx_tensor_name=f"onnx::{name}",
        )

    _add_tensor("x_nchw", [1, 3, 8, 8])
    _add_tensor("y_nchw", [1, 3, 8, 8])
    _add_tensor("x_nhwc", [1, 8, 8, 3])
    _add_tensor("y_nhwc", [1, 8, 8, 3])
    _add_tensor("sum_nhwc", [1, 8, 8, 3])
    _add_tensor("root_nhwc", [1, 8, 8, 3])
    _add_tensor("root_nchw", [1, 3, 8, 8])
    _add_tensor("final", [1, 3, 8, 8])
    _add_tensor(
        "perm_to_nhwc",
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    _add_tensor(
        "perm_to_nchw",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _add_tensor(
        "bias",
        [1],
        data=np.asarray([0.5], dtype=np.float32),
    )
    if leak_pre_transpose:
        _add_tensor("leaked", [1, 8, 8, 3])

    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["x_nchw", "perm_to_nhwc"],
            outputs=["x_nhwc"],
            options={"fixture": "pre_x"},
            version=2,
            onnx_node_name="pre_x",
            onnx_op_type="Transpose",
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["y_nchw", "perm_to_nhwc"],
            outputs=["y_nhwc"],
            options={"fixture": "pre_y"},
            version=2,
            onnx_node_name="pre_y",
            onnx_op_type="Transpose",
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["x_nhwc", "bias"],
            outputs=["sum_nhwc"],
            options={"fusedActivationFunction": "NONE"},
            version=2,
            onnx_node_name="sum",
            onnx_op_type="Add",
        ),
        OperatorIR(
            op_type="MUL",
            inputs=["sum_nhwc", "y_nhwc"],
            outputs=["root_nhwc"],
            options={"fusedActivationFunction": "NONE"},
            version=2,
            onnx_node_name="root",
            onnx_op_type="Mul",
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["root_nhwc", "perm_to_nchw"],
            outputs=["root_nchw"],
            options={"fixture": "post"},
            version=2,
            onnx_node_name="post",
            onnx_op_type="Transpose",
        ),
        OperatorIR(
            op_type="RELU",
            inputs=["root_nchw"],
            outputs=["final"],
            options={"fusedActivationFunction": "NONE"},
            version=2,
            onnx_node_name="final",
            onnx_op_type="Relu",
        ),
    ]
    if leak_pre_transpose:
        model_ir.operators.append(
            OperatorIR(
                op_type="RELU",
                inputs=["x_nhwc"],
                outputs=["leaked"],
                options={},
            )
        )
    return model_ir


def _evaluate(
    model_ir: ModelIR,
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    values = {
        name: np.asarray(tensor.data)
        for name, tensor in model_ir.tensors.items()
        if tensor.data is not None
    }
    values.update({name: np.asarray(value) for name, value in feeds.items()})
    unary_functions: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "ABS": np.abs,
        "EXP": np.exp,
        "FLOOR": np.floor,
        "LOG": np.log,
        "NEG": np.negative,
        "RSQRT": lambda value: np.reciprocal(np.sqrt(value)),
        "SIGN": np.sign,
        "SQRT": np.sqrt,
    }
    binary_functions: dict[
        str,
        Callable[[np.ndarray, np.ndarray], np.ndarray],
    ] = {
        "ADD": np.add,
        "DIV": np.divide,
        "MAXIMUM": np.maximum,
        "MINIMUM": np.minimum,
        "MUL": np.multiply,
        "POW": np.power,
        "SUB": np.subtract,
    }
    for op in model_ir.operators:
        if op.op_type == "TRANSPOSE":
            result = np.transpose(
                values[op.inputs[0]],
                axes=np.asarray(values[op.inputs[1]]).reshape(-1).tolist(),
            )
        elif op.op_type == "RELU":
            result = np.maximum(values[op.inputs[0]], 0)
        elif op.op_type in unary_functions:
            result = unary_functions[op.op_type](values[op.inputs[0]])
        elif op.op_type in binary_functions:
            result = binary_functions[op.op_type](
                values[op.inputs[0]],
                values[op.inputs[1]],
            )
        else:
            raise AssertionError(f"unsupported test operator: {op.op_type}")
        values[op.outputs[0]] = np.asarray(result)
    return {name: values[name] for name in model_ir.outputs}


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _fingerprint(copy.deepcopy(model_ir))

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == _ZERO_STATS
    assert _fingerprint(model_ir) == before


def _append_runtime_consumer(
    model_ir: ModelIR,
    input_name: str,
    output_name: str,
) -> None:
    source = model_ir.tensors[input_name]
    model_ir.tensors[output_name] = TensorIR(
        name=output_name,
        dtype=source.dtype,
        shape=list(source.shape),
        shape_signature=(
            list(source.shape_signature)
            if source.shape_signature is not None
            else list(source.shape)
        ),
        onnx_tensor_name=f"onnx::{output_name}",
    )
    model_ir.outputs.append(output_name)
    model_ir.operators.append(
        OperatorIR("RELU", [input_name], [output_name], onnx_node_name=output_name)
    )


def _prefix_model_ir(model_ir: ModelIR, prefix: str) -> ModelIR:
    prefixed = copy.deepcopy(model_ir)
    name_map = {name: f"{prefix}{name}" for name in prefixed.tensors}
    prefixed.tensors = {
        name_map[name]: tensor for name, tensor in prefixed.tensors.items()
    }
    for old_name, tensor in list(prefixed.tensors.items()):
        tensor.name = old_name
        if tensor.onnx_tensor_name is not None:
            tensor.onnx_tensor_name = f"onnx::{old_name}"
    prefixed.inputs = [name_map[name] for name in prefixed.inputs]
    prefixed.outputs = [name_map[name] for name in prefixed.outputs]
    for op in prefixed.operators:
        op.inputs = [name_map[name] for name in op.inputs]
        op.outputs = [name_map[name] for name in op.outputs]
        if op.onnx_node_name is not None:
            op.onnx_node_name = f"{prefix}{op.onnx_node_name}"
    return prefixed


def _two_chain_model_ir() -> ModelIR:
    first = _make_model_ir()
    second = _prefix_model_ir(_make_model_ir(), "second_")
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    return first


def _owner_wrapper_case(case: str) -> ModelIR:
    if case == "multiple":
        return _two_chain_model_ir()

    model_ir = _make_model_ir()
    if case == "dynamic":
        for name in ("x_nchw", "y_nchw", "root_nchw", "final"):
            model_ir.tensors[name].shape_signature = [1, 3, -1, -1]
        for name in ("x_nhwc", "y_nhwc", "sum_nhwc", "root_nhwc"):
            model_ir.tensors[name].shape_signature = [1, -1, -1, 3]
    elif case in {"local-channel", "constant-qdim", "variable-feature"}:
        bias = model_ir.tensors["bias"]
        bias.data = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
        bias.shape = [3]
        bias.shape_signature = [3]
        if case == "constant-qdim":
            bias.quantization = QuantParamIR(
                scale=[0.1, 0.2, 0.3],
                zero_point=[0, 0, 0],
                quantized_dimension=0,
            )
        elif case == "variable-feature":
            bias.is_variable = True
    elif case == "shared-constant":
        bias = model_ir.tensors["bias"]
        bias.data = np.asarray([0.5, 1.0, 1.5], dtype=np.float32).reshape(1, 1, 1, 3)
        bias.shape = [1, 1, 1, 3]
        bias.shape_signature = [1, 1, 1, 3]
        model_ir.inputs.append("other_nhwc")
        model_ir.outputs.append("shared_out")
        model_ir.tensors["other_nhwc"] = TensorIR(
            name="other_nhwc",
            dtype="FLOAT32",
            shape=[1, 8, 8, 3],
            shape_signature=[1, 8, 8, 3],
        )
        model_ir.tensors["shared_out"] = TensorIR(
            name="shared_out",
            dtype="FLOAT32",
            shape=[1, 8, 8, 3],
            shape_signature=[1, 8, 8, 3],
        )
        model_ir.operators.append(
            OperatorIR("ADD", ["other_nhwc", "bias"], ["shared_out"])
        )
    elif case == "per-axis":
        for name in ("sum_nhwc", "root_nhwc"):
            model_ir.tensors[name].quantization = QuantParamIR(
                scale=[0.1, 0.2, 0.3],
                zero_point=[0, 0, 0],
                quantized_dimension=3,
            )
    elif case == "variable-permutation":
        model_ir.tensors["perm_to_nhwc"].is_variable = True
    elif case == "public-permutation":
        model_ir.inputs.append("perm_to_nchw")
    elif case == "unmatched":
        model_ir.tensors["unrelated"] = TensorIR(
            name="unrelated",
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
        )
        model_ir.tensors["perm_to_nchw"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)
    elif case == "missing-output":
        model_ir.tensors.pop("root_nchw")
    elif case == "public-internal":
        model_ir.inputs.append("sum_nhwc")
    elif case == "reverse-topology":
        model_ir.operators[3], model_ir.operators[4] = (
            model_ir.operators[4],
            model_ir.operators[3],
        )
    elif case == "duplicate-root":
        model_ir.operators.insert(
            4,
            OperatorIR("MUL", ["sum_nhwc", "y_nhwc"], ["root_nhwc"]),
        )
    elif case == "duplicate-pre":
        model_ir.operators.insert(
            1,
            OperatorIR(
                "TRANSPOSE",
                ["x_nchw", "perm_to_nhwc"],
                ["x_nhwc"],
            ),
        )
    return model_ir


def test_elementwise_roundtrip_nchw_nhwc_rewrites_closed_subgraph() -> None:
    model_ir = _make_model_ir()

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == _STATS
    assert [operator.op_type for operator in model_ir.operators] == [
        "ADD",
        "MUL",
        "RELU",
    ]
    assert model_ir.operators[0].inputs == ["x_nchw", "bias"]
    assert model_ir.operators[1].inputs == ["sum_nhwc", "y_nchw"]
    assert model_ir.operators[1].outputs == ["root_nchw"]
    assert model_ir.operators[2].inputs == ["root_nchw"]
    assert model_ir.operators[0].options == {"fusedActivationFunction": "NONE"}
    assert model_ir.operators[0].version == 2
    assert model_ir.operators[0].onnx_node_name == "sum"
    assert model_ir.operators[0].onnx_op_type == "Add"
    assert model_ir.operators[1].onnx_node_name == "root"
    assert model_ir.tensors["sum_nhwc"].shape == [1, 3, 8, 8]
    assert model_ir.tensors["root_nchw"].shape == [1, 3, 8, 8]
    assert "x_nhwc" not in model_ir.tensors
    assert "y_nhwc" not in model_ir.tensors
    assert "root_nhwc" not in model_ir.tensors
    assert "perm_to_nhwc" not in model_ir.tensors
    assert "perm_to_nchw" not in model_ir.tensors
    assert validate_model_ir_invariants(model_ir) == []

    before_noop = _fingerprint(model_ir)
    assert (
        _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)
        == _ZERO_STATS
    )
    assert _fingerprint(model_ir) == before_noop


def test_elementwise_roundtrip_nchw_nhwc_rejects_pre_transpose_fanout() -> None:
    model_ir = _make_model_ir(leak_pre_transpose=True)
    before = _fingerprint(deepcopy(model_ir))

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == _ZERO_STATS
    assert _fingerprint(model_ir) == before


def test_elementwise_roundtrip_nchw_nhwc_rejects_public_post_output() -> None:
    model_ir = _make_model_ir(expose_post_output=True)
    before = _fingerprint(deepcopy(model_ir))

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == _ZERO_STATS
    assert _fingerprint(model_ir) == before


def test_elementwise_roundtrip_nchw_nhwc_is_numerically_exact() -> None:
    model_ir = _make_model_ir()
    before = copy.deepcopy(model_ir)
    feeds = {
        "x_nchw": np.arange(192, dtype=np.float32).reshape(1, 3, 8, 8) / 64.0 + 0.25,
        "y_nchw": np.arange(192, dtype=np.float32).reshape(1, 3, 8, 8) / 96.0 + 0.5,
    }
    expected = _evaluate(before, feeds)

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)
    actual = _evaluate(model_ir, feeds)

    assert stats == _STATS
    assert np.array_equal(actual["final"], expected["final"])


@pytest.mark.parametrize(
    "op_type",
    [
        "ABS",
        "ADD",
        "DIV",
        "EXP",
        "FLOOR",
        "LOG",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "NEG",
        "POW",
        "RSQRT",
        "SIGN",
        "SQRT",
        "SUB",
    ],
)
def test_elementwise_roundtrip_nchw_nhwc_preserves_allowed_op_family(
    op_type: str,
) -> None:
    model_ir = _make_model_ir()
    unary_ops = {"ABS", "EXP", "FLOOR", "LOG", "NEG", "RSQRT", "SIGN", "SQRT"}
    elementwise = model_ir.operators[2]
    elementwise.op_type = op_type
    elementwise.onnx_op_type = op_type.title()
    if op_type in unary_ops:
        elementwise.inputs = ["x_nhwc"]
    before = copy.deepcopy(model_ir)
    feeds = {
        "x_nchw": np.arange(192, dtype=np.float32).reshape(1, 3, 8, 8) / 128.0 + 1.0,
        "y_nchw": np.arange(192, dtype=np.float32).reshape(1, 3, 8, 8) / 192.0 + 0.75,
    }
    expected = _evaluate(before, feeds)

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)
    actual = _evaluate(model_ir, feeds)

    assert stats == _STATS
    assert np.array_equal(actual["final"], expected["final"])


def test_elementwise_roundtrip_nchw_nhwc_rewrites_multiple_and_fixed_point() -> None:
    model_ir = _two_chain_model_ir()

    first = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)
    after_first = _fingerprint(copy.deepcopy(model_ir))
    second = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert first == {
        "optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": 2,
    }
    assert second == _ZERO_STATS
    assert _fingerprint(model_ir) == after_first
    assert [op.op_type for op in model_ir.operators] == [
        "ADD",
        "MUL",
        "RELU",
        "ADD",
        "MUL",
        "RELU",
    ]
    assert validate_model_ir_invariants(model_ir) == []


def test_elementwise_roundtrip_nchw_nhwc_preserves_dynamic_signatures() -> None:
    model_ir = _make_model_ir()
    dynamic_signatures = {
        "x_nchw": [1, 3, -1, -1],
        "y_nchw": [1, 3, -1, -1],
        "x_nhwc": [1, -1, -1, 3],
        "y_nhwc": [1, -1, -1, 3],
        "sum_nhwc": [1, -1, -1, 3],
        "root_nhwc": [1, -1, -1, 3],
        "root_nchw": [1, 3, -1, -1],
        "final": [1, 3, -1, -1],
    }
    for name, signature in dynamic_signatures.items():
        model_ir.tensors[name].shape_signature = list(signature)

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == _STATS
    assert model_ir.tensors["sum_nhwc"].shape_signature == [1, 3, -1, -1]
    assert model_ir.tensors["root_nchw"].shape_signature == [1, 3, -1, -1]
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.tensors["perm_to_nhwc"].__setattr__(
                "data", np.asarray([0, 3, 1, 2], dtype=np.int32)
            ),
            id="wrong-pre-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["perm_to_nchw"].__setattr__(
                "data", np.asarray([0, 2, 3, 1], dtype=np.int32)
            ),
            id="wrong-post-permutation",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[2].__setattr__("op_type", "RELU"),
            id="unsupported-interior-op",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("root_nhwc"),
            id="public-root",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("x_nhwc"),
            id="public-pre-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("x_nchw"),
            id="public-pre-source",
        ),
        pytest.param(
            lambda model_ir: _append_runtime_consumer(model_ir, "sum_nhwc", "sum_copy"),
            id="interior-fanout",
        ),
        pytest.param(
            lambda model_ir: _append_runtime_consumer(
                model_ir, "root_nhwc", "root_copy"
            ),
            id="root-fanout",
        ),
        pytest.param(
            lambda model_ir: (
                model_ir.inputs.append("bias"),
                model_ir.tensors["bias"].__setattr__("data", None),
                model_ir.tensors["bias"].__setattr__("is_variable", True),
            ),
            id="external-runtime-input",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators.insert(
                4,
                OperatorIR(
                    "MUL",
                    ["sum_nhwc", "y_nhwc"],
                    ["root_nhwc"],
                ),
            ),
            id="duplicate-root-producer",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators.insert(
                1,
                OperatorIR(
                    "TRANSPOSE",
                    ["x_nchw", "perm_to_nhwc"],
                    ["x_nhwc"],
                ),
            ),
            id="duplicate-pre-producer",
        ),
    ],
)
def test_elementwise_roundtrip_nchw_nhwc_preserves_existing_rejections(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _make_model_ir()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


def test_elementwise_roundtrip_nchw_nhwc_keeps_owner_wrapper_and_call() -> None:
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "elementwise_roundtrip_nchw_nhwc_layout.py"
    )
    pass_source = pass_path.read_text(encoding="utf-8")
    pass_tree = ast.parse(pass_source)
    owner = next(
        node
        for node in pass_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains"
    )
    assert owner.end_lineno is not None
    assert owner.end_lineno - owner.lineno + 1 == 705
    assert sum(isinstance(node, ast.While) for node in ast.walk(owner)) == 3
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ModelIRGraphIndex"
            for node in ast.walk(owner)
        )
        == 1
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_build_tensor_consumer_map", "_build_tensor_producer_map"}
        for node in ast.walk(owner)
    )
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "remove_operators"
            for node in ast.walk(owner)
        )
        == 1
    )
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

    lowerer_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    lowerer_tree = ast.parse(lowerer_path.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains"
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    wrapper_call = wrapper.body[0].value
    assert isinstance(wrapper_call, ast.Call)
    assert isinstance(wrapper_call.func, ast.Name)
    assert (
        wrapper_call.func.id
        == "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains_pass"
    )
    assert len(wrapper_call.args) == 1
    assert isinstance(wrapper_call.args[0], ast.Name)
    assert wrapper_call.args[0].id == "model_ir"
    assert wrapper_call.keywords == []

    lowerer = next(
        node
        for node in lowerer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains"
    ]
    assert (
        len(calls)
        + LAYOUT_RECOVERY_PASS_IDS.count(
            "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains"
        )
        == 1
    )
    for call in calls:
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "model_ir"
        assert call.keywords == []


def test_elementwise_roundtrip_nchw_nhwc_does_not_prune_unmatched_graph() -> None:
    model_ir = _make_model_ir()
    model_ir.tensors["unrelated"] = TensorIR(
        name="unrelated",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.tensors["perm_to_nchw"].data = np.asarray(
        [0, 2, 3, 1],
        dtype=np.int32,
    )

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize("role", ["pre", "post"])
@pytest.mark.parametrize(
    "condition",
    [
        "public-input",
        "variable",
        "wrong-dtype",
        "wrong-buffer",
        "quantized",
        "runtime-producer",
    ],
)
def test_elementwise_roundtrip_nchw_nhwc_rejects_unsafe_permutation_constant(
    role: str,
    condition: str,
) -> None:
    model_ir = _make_model_ir()
    name = "perm_to_nhwc" if role == "pre" else "perm_to_nchw"
    tensor = model_ir.tensors[name]
    if condition == "public-input":
        model_ir.inputs.append(name)
    elif condition == "variable":
        tensor.is_variable = True
    elif condition == "wrong-dtype":
        tensor.dtype = "FLOAT32"
    elif condition == "wrong-buffer":
        tensor.data = np.asarray(tensor.data, dtype=np.float32)
    elif condition == "quantized":
        tensor.quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )
    elif condition == "runtime-producer":
        seed_name = f"{name}_seed"
        model_ir.tensors[seed_name] = TensorIR(
            name=seed_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(tensor.data, dtype=np.int32),
        )
        model_ir.operators.insert(
            0,
            OperatorIR("IDENTITY", [seed_name], [name]),
        )

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "bias_data,expected_shape",
    [
        pytest.param(
            np.arange(3, dtype=np.float32) + 0.5,
            [1, 3, 1, 1],
            id="channel-vector",
        ),
        pytest.param(
            np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3) + 0.5,
            [1, 3, 1, 1],
            id="rank-four-channel",
        ),
        pytest.param(
            np.arange(192, dtype=np.float32).reshape(1, 8, 8, 3) / 128.0 + 0.5,
            [1, 3, 8, 8],
            id="full-rank-four",
        ),
    ],
)
def test_elementwise_roundtrip_nchw_nhwc_remaps_local_broadcast_constant(
    bias_data: np.ndarray,
    expected_shape: list[int],
) -> None:
    model_ir = _make_model_ir()
    bias = model_ir.tensors["bias"]
    bias.data = np.asarray(bias_data)
    bias.shape = list(bias_data.shape)
    bias.shape_signature = list(bias_data.shape)
    before = copy.deepcopy(model_ir)
    feeds = {
        "x_nchw": np.arange(192, dtype=np.float32).reshape(1, 3, 8, 8) / 64.0 + 1.0,
        "y_nchw": np.ones((1, 3, 8, 8), dtype=np.float32),
    }
    expected = _evaluate(before, feeds)

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)
    actual = _evaluate(model_ir, feeds)

    assert stats == _STATS
    assert np.array_equal(actual["final"], expected["final"])
    assert model_ir.tensors["bias"].shape == expected_shape
    assert list(np.asarray(model_ir.tensors["bias"].data).shape) == expected_shape


def test_elementwise_roundtrip_nchw_nhwc_clones_shared_broadcast_constant() -> None:
    model_ir = _make_model_ir()
    bias_data = np.arange(3, dtype=np.float32).reshape(1, 1, 1, 3) + 0.5
    bias = model_ir.tensors["bias"]
    bias.data = bias_data
    bias.shape = list(bias_data.shape)
    bias.shape_signature = list(bias_data.shape)
    model_ir.inputs.append("other_nhwc")
    model_ir.outputs.append("shared_out")
    model_ir.tensors["other_nhwc"] = TensorIR(
        name="other_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 3],
        shape_signature=[1, 8, 8, 3],
    )
    model_ir.tensors["shared_out"] = TensorIR(
        name="shared_out",
        dtype="FLOAT32",
        shape=[1, 8, 8, 3],
        shape_signature=[1, 8, 8, 3],
    )
    model_ir.operators.append(OperatorIR("ADD", ["other_nhwc", "bias"], ["shared_out"]))
    before = copy.deepcopy(model_ir)
    feeds = {
        "x_nchw": np.ones((1, 3, 8, 8), dtype=np.float32),
        "y_nchw": np.ones((1, 3, 8, 8), dtype=np.float32),
        "other_nhwc": np.ones((1, 8, 8, 3), dtype=np.float32),
    }
    expected = _evaluate(before, feeds)

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)
    actual = _evaluate(model_ir, feeds)

    assert stats == _STATS
    assert np.array_equal(actual["final"], expected["final"])
    assert np.array_equal(actual["shared_out"], expected["shared_out"])
    assert np.array_equal(model_ir.tensors["bias"].data, bias_data)
    rewritten_add = next(
        op for op in model_ir.operators if list(op.outputs) == ["sum_nhwc"]
    )
    assert rewritten_add.inputs == ["x_nchw", "bias__nchw"]
    assert model_ir.tensors["bias__nchw"].shape == [1, 3, 1, 1]
    assert model_ir.tensors["bias__nchw"].onnx_tensor_name == "onnx::bias"


def test_elementwise_roundtrip_nchw_nhwc_remaps_layout_metadata() -> None:
    model_ir = _make_model_ir()
    for name in ("x_nchw", "y_nchw", "root_nchw", "final"):
        model_ir.tensors[name].logical_layout = "NCHW"
        model_ir.tensors[name].physical_layout = "NCHW"
    for name in ("x_nhwc", "y_nhwc", "sum_nhwc", "root_nhwc"):
        model_ir.tensors[name].logical_layout = "NHWC"
        model_ir.tensors[name].physical_layout = "NHWC"

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == _STATS
    for name in ("sum_nhwc", "root_nchw"):
        assert model_ir.tensors[name].logical_layout == "NCHW"
        assert model_ir.tensors[name].physical_layout == "NCHW"


def test_elementwise_roundtrip_nchw_nhwc_remaps_per_axis_quantization() -> None:
    model_ir = _make_model_ir()
    for name in ("sum_nhwc", "root_nhwc"):
        model_ir.tensors[name].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3],
            zero_point=[0, 0, 0],
            quantized_dimension=3,
        )

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == _STATS
    assert model_ir.tensors["sum_nhwc"].quantization.quantized_dimension == 1
    assert model_ir.tensors["root_nchw"].quantization.quantized_dimension == 1


def test_elementwise_roundtrip_nchw_nhwc_remaps_constant_per_axis_quantization() -> (
    None
):
    model_ir = _make_model_ir()
    bias = model_ir.tensors["bias"]
    bias.data = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
    bias.shape = [3]
    bias.shape_signature = [3]
    bias.quantization = QuantParamIR(
        scale=[0.1, 0.2, 0.3],
        zero_point=[0, 0, 0],
        quantized_dimension=0,
    )

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == _STATS
    assert bias.shape == [1, 3, 1, 1]
    assert bias.quantization.quantized_dimension == 1


@pytest.mark.parametrize("condition", ["variable", "metadata-mismatch"])
def test_elementwise_roundtrip_nchw_nhwc_rejects_unsafe_feature_constant(
    condition: str,
) -> None:
    model_ir = _make_model_ir()
    bias = model_ir.tensors["bias"]
    bias.data = np.asarray([0.5, 1.0, 1.5], dtype=np.float32)
    bias.shape = [3]
    bias.shape_signature = [3]
    if condition == "variable":
        bias.is_variable = True
    else:
        bias.shape = [1, 1, 1, 3]
        bias.shape_signature = [1, 1, 1, 3]

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.tensors.pop("x_nchw"),
            id="missing-pre-source-tensor",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors.pop("sum_nhwc"),
            id="missing-interior-output-tensor",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors.pop("root_nchw"),
            id="missing-post-output-tensor",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["x_nhwc"].__setattr__("dtype", "FLOAT16"),
            id="dtype-mismatch",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["x_nchw"].__setattr__(
                "shape", [1, 4, 8, 8]
            ),
            id="pre-shape-mismatch",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["sum_nhwc"].__setattr__(
                "shape_signature", [1, -1, 3]
            ),
            id="signature-rank-mismatch",
        ),
        pytest.param(
            lambda model_ir: model_ir.inputs.append("sum_nhwc"),
            id="public-interior-alias",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators.__setitem__(
                slice(3, 5),
                [model_ir.operators[4], model_ir.operators[3]],
            ),
            id="reverse-root-post-topology",
        ),
        pytest.param(
            lambda model_ir: (
                model_ir.tensors.__setitem__(
                    "aux_nhwc",
                    TensorIR(
                        name="aux_nhwc",
                        dtype="FLOAT32",
                        shape=[1, 8, 8, 3],
                        shape_signature=[1, 8, 8, 3],
                    ),
                ),
                model_ir.operators[3].outputs.append("aux_nhwc"),
            ),
            id="multiple-root-outputs",
        ),
    ],
)
def test_elementwise_roundtrip_nchw_nhwc_rejects_incomplete_candidate(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _make_model_ir()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


def test_elementwise_roundtrip_nchw_nhwc_reuses_one_graph_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = _two_chain_model_ir()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(model_ir)

    assert stats == {
        "optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains": 2,
    }
    assert refresh_count == 1


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "multiple",
        "dynamic",
        "local-channel",
        "shared-constant",
        "per-axis",
        "constant-qdim",
        "variable-permutation",
        "public-permutation",
        "unmatched",
        "missing-output",
        "public-internal",
        "reverse-topology",
        "duplicate-root",
        "duplicate-pre",
        "variable-feature",
    ],
)
def test_elementwise_roundtrip_nchw_nhwc_owner_and_wrapper_are_identical(
    case: str,
) -> None:
    owner_model = _owner_wrapper_case(case)
    wrapper_model = copy.deepcopy(owner_model)

    owner_stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains_owner(
        owner_model
    )
    wrapper_stats = _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(
        wrapper_model
    )

    assert wrapper_stats == owner_stats
    assert _normalize(wrapper_model) == _normalize(owner_model)
