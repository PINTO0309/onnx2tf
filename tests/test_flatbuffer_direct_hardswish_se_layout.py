from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.hardswish_se_layout import (
    optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TERMINAL_HARDSWISH_SE_OWNER = (
    "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _add_scalar(model_ir: ModelIR, name: str, value: float) -> None:
    model_ir.tensors[name] = _tensor(
        name,
        [1],
        data=np.asarray([value], dtype=np.float32),
    )


def _normalize(model_ir: ModelIR) -> dict[str, object]:
    return {
        "inputs": list(model_ir.inputs),
        "outputs": list(model_ir.outputs),
        "tensors": {
            name: {
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "shape_signature": (
                    list(tensor.shape_signature)
                    if tensor.shape_signature is not None
                    else None
                ),
                "data": (
                    tensor.data.tolist()
                    if isinstance(tensor.data, np.ndarray)
                    else tensor.data
                ),
                "is_variable": tensor.is_variable,
                "quantization": copy.deepcopy(tensor.quantization),
                "logical_layout": tensor.logical_layout,
                "physical_layout": tensor.physical_layout,
            }
            for name, tensor in model_ir.tensors.items()
        },
        "operators": [
            {
                "op_type": operator.op_type,
                "inputs": list(operator.inputs),
                "outputs": list(operator.outputs),
                "options": copy.deepcopy(operator.options),
                "axis_semantics": copy.deepcopy(operator.axis_semantics),
                "version": operator.version,
            }
            for operator in model_ir.operators
        ],
    }


def _make_hardswish_se_model_ir(
    *,
    root_kind: str,
    gate_kind: str,
) -> ModelIR:
    model_ir = ModelIR(f"hardswish_se_{root_kind}_{gate_kind}")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]

    model_ir.tensors["x"] = _tensor("x", [1, 8, 8, 4])
    model_ir.tensors["pre_perm"] = _tensor(
        "pre_perm",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["post_perm"] = _tensor(
        "post_perm",
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["pre"] = _tensor("pre", [1, 4, 8, 8])
    model_ir.operators.append(
        OperatorIR("TRANSPOSE", ["x", "pre_perm"], ["pre"])
    )

    if root_kind == "direct":
        model_ir.tensors["hsw"] = _tensor("hsw", [1, 4, 8, 8])
        model_ir.operators.append(OperatorIR("HARD_SWISH", ["pre"], ["hsw"]))
    elif root_kind == "decomposed":
        _add_scalar(model_ir, "root_add_const", 3.0)
        _add_scalar(model_ir, "root_scale_const", 1.0 / 6.0)
        model_ir.tensors["root_add"] = _tensor("root_add", [1, 4, 8, 8])
        model_ir.tensors["root_scaled"] = _tensor(
            "root_scaled", [1, 4, 8, 8]
        )
        model_ir.tensors["hsw"] = _tensor("hsw", [1, 4, 8, 8])
        model_ir.operators.extend(
            [
                OperatorIR(
                    "ADD",
                    ["pre", "root_add_const"],
                    ["root_add"],
                    options={"fusedActivationFunction": "RELU6"},
                ),
                OperatorIR(
                    "MUL",
                    ["root_add", "root_scale_const"],
                    ["root_scaled"],
                ),
                OperatorIR("MUL", ["pre", "root_scaled"], ["hsw"]),
            ]
        )
    else:
        raise AssertionError(f"unsupported root_kind: {root_kind}")

    model_ir.tensors["mean_axes"] = _tensor(
        "mean_axes",
        [2],
        dtype="INT32",
        data=np.asarray([2, 3], dtype=np.int32),
    )
    model_ir.tensors["mean"] = _tensor("mean", [1, 4, 1, 1])
    model_ir.tensors["pooled"] = _tensor("pooled", [1, 1, 1, 4])
    model_ir.tensors["conv1"] = _tensor("conv1", [1, 1, 1, 2])
    model_ir.tensors["conv2"] = _tensor("conv2", [1, 1, 1, 4])
    model_ir.tensors["gate_nchw"] = _tensor("gate_nchw", [1, 4, 1, 1])
    model_ir.operators.extend(
        [
            OperatorIR(
                "MEAN",
                ["hsw", "mean_axes"],
                ["mean"],
                options={"keepDims": True},
            ),
            OperatorIR("TRANSPOSE", ["mean", "post_perm"], ["pooled"]),
            OperatorIR("CONV_2D", ["pooled"], ["conv1"]),
            OperatorIR("CONV_2D", ["conv1"], ["conv2"]),
            OperatorIR("TRANSPOSE", ["conv2", "pre_perm"], ["gate_nchw"]),
        ]
    )

    _add_scalar(model_ir, "gate_alpha", 1.0 / 6.0)
    _add_scalar(model_ir, "gate_beta", 0.5)
    if gate_kind == "expanded":
        model_ir.tensors["gate_mul"] = _tensor("gate_mul", [1, 4, 1, 1])
        model_ir.tensors["gate_add"] = _tensor("gate_add", [1, 4, 1, 1])
        model_ir.tensors["gate"] = _tensor("gate", [1, 4, 1, 1])
        model_ir.operators.extend(
            [
                OperatorIR(
                    "MUL", ["gate_nchw", "gate_alpha"], ["gate_mul"]
                ),
                OperatorIR("ADD", ["gate_mul", "gate_beta"], ["gate_add"]),
                OperatorIR("RELU_0_TO_1", ["gate_add"], ["gate"]),
            ]
        )
    elif gate_kind == "fused":
        model_ir.tensors["gate_add"] = _tensor("gate_add", [1, 4, 1, 1])
        model_ir.tensors["gate"] = _tensor("gate", [1, 4, 1, 1])
        model_ir.operators.extend(
            [
                OperatorIR(
                    "ADD",
                    ["gate_nchw", "gate_beta"],
                    ["gate_add"],
                    options={"fusedActivationFunction": "RELU6"},
                ),
                OperatorIR("MUL", ["gate_add", "gate_alpha"], ["gate"]),
            ]
        )
    else:
        raise AssertionError(f"unsupported gate_kind: {gate_kind}")

    model_ir.tensors["residual"] = _tensor("residual", [1, 4, 8, 8])
    model_ir.tensors["y"] = _tensor("y", [1, 8, 8, 4])
    model_ir.operators.extend(
        [
            OperatorIR("MUL", ["hsw", "gate"], ["residual"]),
            OperatorIR("TRANSPOSE", ["residual", "post_perm"], ["y"]),
        ]
    )
    return model_ir


@pytest.mark.parametrize("root_kind", ["direct", "decomposed"])
@pytest.mark.parametrize("gate_kind", ["expanded", "fused"])
def test_hardswish_se_layout_rewrites_all_root_and_gate_families(
    root_kind: str,
    gate_kind: str,
) -> None:
    model_ir = _make_hardswish_se_model_ir(
        root_kind=root_kind,
        gate_kind=gate_kind,
    )

    stats = (
        _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            model_ir
        )
    )

    assert stats == {
        "optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains": 1,
    }
    assert not any(
        str(operator.op_type) == "TRANSPOSE" for operator in model_ir.operators
    )
    activation_inputs = [
        list(operator.inputs)
        for operator in model_ir.operators
        if str(operator.op_type) == "HARD_SWISH"
    ]
    if root_kind == "direct":
        assert activation_inputs == [["x"]]
    else:
        root_add = next(
            operator
            for operator in model_ir.operators
            if str(operator.outputs[0]) == "root_add"
        )
        root_multiply = next(
            operator
            for operator in model_ir.operators
            if str(operator.outputs[0]) == "hsw"
        )
        assert list(root_add.inputs) == ["x", "root_add_const"]
        assert list(root_multiply.inputs) == ["x", "root_scaled"]

    mean = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "MEAN"
    )
    assert list(mean.inputs) == ["hsw", "mean_axes"]
    assert list(mean.outputs) == ["pooled"]
    assert model_ir.tensors["mean_axes"].data.tolist() == [1, 2]
    gate_entry = next(
        operator
        for operator in model_ir.operators
        if str(operator.outputs[0]) in {"gate_mul", "gate_add"}
        and "conv2" in operator.inputs
    )
    assert "conv2" in gate_entry.inputs
    residual = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "MUL" and list(operator.outputs) == ["y"]
    )
    assert list(residual.inputs) == ["hsw", "gate"]
    assert model_ir.tensors["hsw"].shape == [1, 8, 8, 4]
    assert model_ir.tensors["gate"].shape == [1, 1, 1, 4]
    assert model_ir.tensors["y"].shape == [1, 8, 8, 4]

    assert _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
        model_ir
    ) == {
        "optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains": 0,
    }


@pytest.mark.parametrize("guard", ["public_pre", "invalid_axes", "hsw_fanout"])
def test_hardswish_se_layout_rejects_unsafe_boundaries(guard: str) -> None:
    model_ir = _make_hardswish_se_model_ir(
        root_kind="direct",
        gate_kind="expanded",
    )
    if guard == "public_pre":
        model_ir.outputs.append("pre")
    elif guard == "invalid_axes":
        model_ir.tensors["mean_axes"].data = np.asarray([4], dtype=np.int32)
    elif guard == "hsw_fanout":
        model_ir.tensors["tap"] = _tensor("tap", [1, 4, 8, 8])
        model_ir.operators.append(OperatorIR("RELU", ["hsw"], ["tap"]))
    else:
        raise AssertionError(f"unsupported guard: {guard}")
    before = copy.deepcopy(model_ir)

    stats = (
        _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            model_ir
        )
    )

    assert stats == {
        "optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains": 0,
    }
    assert _normalize(model_ir) == _normalize(before)


def test_hardswish_se_layout_owner_matches_lowerer_compatibility_wrapper() -> None:
    model_ir = _make_hardswish_se_model_ir(
        root_kind="decomposed",
        gate_kind="fused",
    )
    direct_model_ir = copy.deepcopy(model_ir)
    wrapper_model_ir = copy.deepcopy(model_ir)

    direct_stats = (
        optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            direct_model_ir
        )
    )
    wrapper_stats = (
        _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            wrapper_model_ir
        )
    )

    assert direct_stats == wrapper_stats
    assert _normalize(direct_model_ir) == _normalize(wrapper_model_ir)


def test_hardswish_se_layout_prunes_unused_tensors_on_zero_rewrite() -> None:
    model_ir = ModelIR("hardswish_se_zero_rewrite_prune")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["x"]
    model_ir.tensors["x"] = _tensor("x", [1, 2, 3, 4])
    model_ir.tensors["unused"] = _tensor("unused", [1])

    stats = (
        _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            model_ir
        )
    )

    assert stats == {
        "optimized_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains": 0,
    }
    assert "unused" not in model_ir.tensors


def test_terminal_hardswish_se_call_captures_complete_mutation_evidence() -> None:
    tree = ast.parse(
        (
            REPO_ROOT
            / "onnx2tf"
            / "tflite_builder"
            / "lower_from_onnx2tf.py"
        ).read_text(encoding="utf-8")
    )
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    first_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_terminal_hardswish_se_stats"
    )
    statement = lowerer.body[first_index]
    assert isinstance(statement, ast.Assign)
    summary = statement.value
    assert isinstance(summary, ast.Call)
    assert isinstance(summary.func, ast.Name)
    assert summary.func.id == "run_hardswish_se_layout_summary"
    assert [ast.unparse(argument) for argument in summary.args] == ["model_ir"]
    assert summary.keywords == []

    previous = lowerer.body[first_index - 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_terminal_split_conv_concat_bridge_stats"
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )
    following = lowerer.body[first_index + 1]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "late_hard_activation_tensor_count"
