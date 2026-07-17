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
    _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_STATS = {
    "optimized_attention_preproj_reshape_to_batchmatmul_ranklift_chains": 1,
}
_ZERO_STATS = {
    "optimized_attention_preproj_reshape_to_batchmatmul_ranklift_chains": 0,
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
        shape_signature=(list(signature) if signature is not None else list(shape)),
        data=data,
        is_variable=bool(is_variable),
        quantization=quantization,
        onnx_tensor_name=f"onnx::{name}",
    )


def _const_ints(model_ir: ModelIR, name: str, values: list[int]) -> None:
    _tensor(
        model_ir,
        name,
        [len(values)],
        dtype="INT32",
        data=np.asarray(values, dtype=np.int32),
    )


def _model(
    *,
    branches: int = 1,
    binary_ops: list[str] | None = None,
    reverse_binary_inputs: set[int] | None = None,
) -> ModelIR:
    model_ir = ModelIR("attention_preproj_ranklift_characterization")
    model_ir.inputs = ["x"]
    _tensor(model_ir, "x", [1, 1, 3, 4])
    _const_ints(model_ir, "lead_shape", [3, 1, 4])
    _tensor(model_ir, "lead", [3, 1, 4])
    model_ir.operators.append(
        OperatorIR(
            "RESHAPE",
            ["x", "lead_shape"],
            ["lead"],
            options={"newShape": [3, 1, 4]},
            version=2,
            onnx_node_name="lead_reshape",
            onnx_op_type="Reshape",
        )
    )

    selected_binary_ops = binary_ops or ["ADD"] * int(branches)
    reversed_inputs = reverse_binary_inputs or set()
    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}_"
        k_dim = 8 + 4 * branch_index
        binary_op_type = selected_binary_ops[branch_index]
        weight_name = f"{prefix}weight"
        bmm_name = f"{prefix}bmm"
        bias_name = f"{prefix}bias"
        binary_name = f"{prefix}binary"
        tail_shape_name = f"{prefix}tail_shape"
        output_name = f"{prefix}output"

        weight = np.arange(4 * k_dim, dtype=np.float32).reshape(4, k_dim) / 32.0 + 0.5
        bias = np.arange(k_dim, dtype=np.float32) / 16.0 + 0.25
        _tensor(model_ir, weight_name, [4, k_dim], data=weight)
        _tensor(model_ir, bmm_name, [3, 1, k_dim])
        _tensor(model_ir, bias_name, [k_dim], data=bias)
        _tensor(model_ir, binary_name, [3, 1, k_dim])
        _const_ints(model_ir, tail_shape_name, [1, 3, 2, k_dim // 2])
        _tensor(model_ir, output_name, [1, 3, 2, k_dim // 2])
        model_ir.outputs.append(output_name)

        binary_inputs = [bmm_name, bias_name]
        if branch_index in reversed_inputs:
            binary_inputs.reverse()
        model_ir.operators.extend(
            [
                OperatorIR(
                    "BATCH_MATMUL",
                    ["lead", weight_name],
                    [bmm_name],
                    options={"adjX": False, "adjY": False},
                    version=2,
                    onnx_node_name=f"{prefix}matmul",
                    onnx_op_type="MatMul",
                ),
                OperatorIR(
                    binary_op_type,
                    binary_inputs,
                    [binary_name],
                    options={"fusedActivationFunction": "NONE"},
                    version=2,
                    onnx_node_name=f"{prefix}{binary_op_type.lower()}",
                    onnx_op_type=binary_op_type.title(),
                ),
                OperatorIR(
                    "RESHAPE",
                    [binary_name, tail_shape_name],
                    [output_name],
                    options={"newShape": [1, 3, 2, k_dim // 2]},
                    version=2,
                    onnx_node_name=f"{prefix}tail_reshape",
                    onnx_op_type="Reshape",
                ),
            ]
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
    for op in model_ir.operators:
        if op.op_type == "RESHAPE":
            shape = np.asarray(values[op.inputs[1]]).reshape(-1).tolist()
            result = np.reshape(values[op.inputs[0]], shape)
        elif op.op_type == "BATCH_MATMUL":
            lhs = values[op.inputs[0]]
            rhs = values[op.inputs[1]]
            if bool(op.options.get("adjX", False)):
                lhs = np.swapaxes(lhs, -1, -2)
            if bool(op.options.get("adjY", False)):
                rhs = np.swapaxes(rhs, -1, -2)
            result = np.matmul(lhs, rhs)
        elif op.op_type in {"ADD", "SUB", "MUL", "DIV"}:
            lhs = values[op.inputs[0]]
            rhs = values[op.inputs[1]]
            result = {
                "ADD": np.add,
                "SUB": np.subtract,
                "MUL": np.multiply,
                "DIV": np.divide,
            }[op.op_type](lhs, rhs)
        else:
            raise AssertionError(f"unsupported test operator: {op.op_type}")
        values[op.outputs[0]] = np.asarray(result)
    return {name: values[name] for name in model_ir.outputs}


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(model_ir)

    assert stats == _ZERO_STATS
    assert _normalize(model_ir) == before


def _append_identity_consumer(
    model_ir: ModelIR,
    input_name: str,
    output_name: str,
) -> None:
    source = model_ir.tensors[input_name]
    _tensor(model_ir, output_name, list(source.shape))
    model_ir.outputs.append(output_name)
    model_ir.operators.append(OperatorIR("IDENTITY", [input_name], [output_name]))


def test_attention_preproj_ranklift_rewrites_one_branch() -> None:
    model_ir = _model()

    stats = _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(model_ir)

    assert stats == _STATS
    assert [op.op_type for op in model_ir.operators] == [
        "BATCH_MATMUL",
        "ADD",
        "RESHAPE",
    ]
    bmm, binary, tail = model_ir.operators
    assert bmm.inputs == ["x", "branch0_weight"]
    assert bmm.options == {"adjX": False, "adjY": False}
    assert bmm.version == 2
    assert bmm.onnx_node_name == "branch0_matmul"
    assert binary.inputs == ["branch0_bmm", "branch0_bias"]
    assert tail.inputs == ["branch0_binary", "branch0_tail_shape"]
    assert model_ir.tensors["branch0_bmm"].shape == [1, 1, 3, 8]
    assert model_ir.tensors["branch0_binary"].shape == [1, 1, 3, 8]
    assert "lead" not in model_ir.tensors
    assert "lead_shape" not in model_ir.tensors
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "binary_op,reverse_inputs",
    [
        ("ADD", False),
        ("SUB", False),
        ("SUB", True),
        ("MUL", False),
        ("DIV", False),
        ("DIV", True),
    ],
)
def test_attention_preproj_ranklift_is_numerically_exact(
    binary_op: str,
    reverse_inputs: bool,
) -> None:
    model_ir = _model(
        binary_ops=[binary_op],
        reverse_binary_inputs={0} if reverse_inputs else set(),
    )
    before = copy.deepcopy(model_ir)
    feed = np.arange(12, dtype=np.float32).reshape(1, 1, 3, 4) + 1.0
    expected = _evaluate(before, {"x": feed})

    _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(model_ir)
    actual = _evaluate(model_ir, {"x": feed})

    for name in expected:
        assert np.allclose(actual[name], expected[name], rtol=0.0, atol=0.0)


def test_attention_preproj_ranklift_rewrites_multiple_and_fixed_point() -> None:
    model_ir = _model(
        branches=2,
        binary_ops=["ADD", "MUL"],
    )

    first = _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(model_ir)
    after_first = _normalize(copy.deepcopy(model_ir))
    second = _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(
        model_ir
    )

    assert first == _STATS
    assert second == _ZERO_STATS
    assert _normalize(model_ir) == after_first
    assert [model_ir.operators[index].inputs[0] for index in (0, 3)] == [
        "x",
        "x",
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda model_ir: model_ir.outputs.append("lead"),
            id="public-leading-reshape",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["x"].__setattr__("shape", [1, 2, 3, 4]),
            id="nonsingleton-second-dimension",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["lead"].__setattr__("shape", [3, 2, 4]),
            id="wrong-leading-output-shape",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["lead_shape"].__setattr__(
                "data", np.asarray([3, 2, 4], dtype=np.int32)
            ),
            id="wrong-leading-target",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(model_ir, "lead", "lead_copy"),
            id="non-bmm-leading-consumer",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_bmm"),
            id="public-bmm-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["branch0_weight"].__setattr__(
                "shape", [5, 8]
            ),
            id="wrong-weight-input-dimension",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir, "branch0_bmm", "bmm_copy"
            ),
            id="bmm-fanout",
        ),
        pytest.param(
            lambda model_ir: model_ir.operators[2].__setattr__("op_type", "MAXIMUM"),
            id="unsupported-binary",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_binary"),
            id="public-binary-output",
        ),
        pytest.param(
            lambda model_ir: _append_identity_consumer(
                model_ir, "branch0_binary", "binary_copy"
            ),
            id="binary-fanout",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors["branch0_tail_shape"].__setattr__(
                "data",
                np.asarray([1, 2, 2, 2], dtype=np.int32),
            ),
            id="wrong-tail-target",
        ),
    ],
)
def test_attention_preproj_ranklift_preserves_existing_rejections(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="no-match execution prunes tensors instead of remaining a no-op",
)
def test_attention_preproj_ranklift_does_not_prune_unmatched_graph() -> None:
    model_ir = _model()
    _tensor(model_ir, "unused", [1])
    model_ir.tensors["lead_shape"].data = np.asarray([3, 2, 4], dtype=np.int32)

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="matched shape constants lack immutable ownership and typing",
)
@pytest.mark.parametrize("role", ["lead", "tail"])
@pytest.mark.parametrize(
    "condition",
    ["public-input", "variable", "wrong-dtype", "wrong-buffer", "quantized"],
)
def test_attention_preproj_ranklift_rejects_unsafe_shape_constant(
    role: str,
    condition: str,
) -> None:
    model_ir = _model()
    name = "lead_shape" if role == "lead" else "branch0_tail_shape"
    tensor = model_ir.tensors[name]
    if condition == "public-input":
        model_ir.inputs.append(name)
    elif condition == "variable":
        tensor.is_variable = True
    elif condition == "wrong-dtype":
        tensor.dtype = "FLOAT32"
        tensor.data = np.asarray(tensor.data, dtype=np.float32)
    elif condition == "wrong-buffer":
        tensor.data = np.asarray(tensor.data, dtype=np.int64)
    elif condition == "quantized":
        tensor.quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="rank-lifted dynamic signatures are replaced with static values",
)
def test_attention_preproj_ranklift_preserves_dynamic_signature() -> None:
    model_ir = _model()
    model_ir.tensors["x"].shape_signature = [1, 1, -1, 4]
    model_ir.tensors["lead"].shape_signature = [-1, 1, 4]
    model_ir.tensors["branch0_bmm"].shape_signature = [-1, 1, 8]
    model_ir.tensors["branch0_binary"].shape_signature = [-1, 1, 8]

    stats = _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(model_ir)

    assert stats == _STATS
    assert model_ir.tensors["branch0_bmm"].shape_signature == [1, 1, -1, 8]
    assert model_ir.tensors["branch0_binary"].shape_signature == [1, 1, -1, 8]


@pytest.mark.xfail(
    strict=True,
    reason="rank-lifted per-axis quantized dimensions are not remapped",
)
def test_attention_preproj_ranklift_remaps_per_axis_quantization() -> None:
    model_ir = _model()
    for name in ("branch0_bmm", "branch0_binary"):
        model_ir.tensors[name].quantization = QuantParamIR(
            scale=[0.1] * 8,
            zero_point=[0] * 8,
            quantized_dimension=2,
        )

    stats = _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(model_ir)

    assert stats == _STATS
    for name in ("branch0_bmm", "branch0_binary"):
        quantization = model_ir.tensors[name].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3


@pytest.mark.xfail(
    strict=True,
    reason="binary broadcast compatibility is not revalidated after rank lift",
)
def test_attention_preproj_ranklift_rejects_rank_sensitive_bias() -> None:
    model_ir = _model()
    bias = model_ir.tensors["branch0_bias"]
    bias.shape = [3, 1, 8]
    bias.shape_signature = [3, 1, 8]
    bias.data = np.ones([3, 1, 8], dtype=np.float32)

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="BatchMatMul transpose flags are ignored by the matcher",
)
@pytest.mark.parametrize("flag", ["adjX", "adjY"])
def test_attention_preproj_ranklift_rejects_transposed_bmm(flag: str) -> None:
    model_ir = _model()
    model_ir.operators[1].options[flag] = True

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="nonpositive tail dimensions are accepted by product alone",
)
def test_attention_preproj_ranklift_rejects_nonpositive_tail_shape() -> None:
    model_ir = _model()
    model_ir.tensors["branch0_tail_shape"].data = np.asarray(
        [1, 3, -2, -4], dtype=np.int32
    )

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="complete tensor and binary-input metadata is not prevalidated",
)
@pytest.mark.parametrize(
    "case",
    [
        "short-input-signature",
        "short-bmm-signature",
        "missing-tail-output",
        "missing-bias",
        "dtype-mismatch",
    ],
)
def test_attention_preproj_ranklift_rejects_incomplete_metadata(case: str) -> None:
    model_ir = _model()
    if case == "short-input-signature":
        model_ir.tensors["x"].shape_signature = [1, 2]
    elif case == "short-bmm-signature":
        model_ir.tensors["branch0_bmm"].shape_signature = [1, 2]
    elif case == "missing-tail-output":
        del model_ir.tensors["branch0_output"]
    elif case == "missing-bias":
        del model_ir.tensors["branch0_bias"]
    elif case == "dtype-mismatch":
        model_ir.tensors["branch0_bmm"].dtype = "INT8"

    _assert_transactional_rejection(model_ir)


@pytest.mark.xfail(
    strict=True,
    reason="duplicate producers, reverse order, and public aliases are accepted",
)
@pytest.mark.parametrize(
    "case",
    [
        "reverse-leading-bmm",
        "duplicate-bmm-output",
        "public-internal-input",
        "reverse-source-producer",
        "duplicate-source-producer",
    ],
)
def test_attention_preproj_ranklift_rejects_invalid_topology(case: str) -> None:
    model_ir = _model()
    if case == "reverse-leading-bmm":
        model_ir.operators[0], model_ir.operators[1] = (
            model_ir.operators[1],
            model_ir.operators[0],
        )
    elif case == "duplicate-bmm-output":
        model_ir.operators.insert(
            2,
            OperatorIR("IDENTITY", ["x"], ["branch0_bmm"]),
        )
    elif case == "public-internal-input":
        model_ir.inputs.append("lead")
    elif case == "reverse-source-producer":
        model_ir.operators.append(OperatorIR("IDENTITY", ["branch0_output"], ["x"]))
    elif case == "duplicate-source-producer":
        model_ir.operators.extend(
            [
                OperatorIR("IDENTITY", ["branch0_output"], ["x"]),
                OperatorIR("IDENTITY", ["branch0_output"], ["x"]),
            ]
        )

    _assert_transactional_rejection(model_ir)


def test_attention_preproj_ranklift_keeps_raw_owner_and_calls() -> None:
    lowering_path = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    lowering_source = lowering_path.read_text(encoding="utf-8")
    lowering_tree = ast.parse(lowering_source)
    owner = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains"
    )
    assert owner.end_lineno - owner.lineno + 1 == 190
    assert sum(isinstance(node, ast.While) for node in ast.walk(owner)) == 1
    owner_source = ast.get_source_segment(lowering_source, owner)
    assert owner_source is not None
    assert "_build_tensor_consumer_map(model_ir)" in owner_source
    assert "_prune_unused_tensors(model_ir)" in owner_source

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
        == "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains"
    ]
    assert len(calls) == 2
    for call in calls:
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "model_ir"
        assert call.keywords == []
