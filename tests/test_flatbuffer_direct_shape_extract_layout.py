from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_shape_extract_nhwc_to_nchw_chains,
)
from onnx2tf.tflite_builder.passes.shape_extract_layout import (
    optimize_transpose_shape_extract_nhwc_to_nchw_chains,
)
from onnx2tf.tflite_builder.passes.terminal_concat_bridge_layout_orchestration import (
    TERMINAL_CONCAT_BRIDGE_LAYOUT_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SHAPE_EXTRACT_OWNER = "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
TERMINAL_QKV_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_shape_attention_orchestration.py"
)
TERMINAL_QKV_OWNER = "run_terminal_qkv_shape_attention_cleanup"
TERMINAL_QKV_RESULT = "_terminal_qkv_shape_attention_results"
TERMINAL_LAYOUT_SHAPE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_layout_shape_orchestration.py"
)
TERMINAL_LAYOUT_SHAPE_OWNER = "run_terminal_layout_shape_cleanup"


def _terminal_qkv_shape_calls() -> list[ast.Call]:
    tree = ast.parse(TERMINAL_QKV_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_QKV_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SHAPE_EXTRACT_OWNER.removeprefix("_")
    ]


def _terminal_layout_shape_calls() -> list[ast.Call]:
    tree = ast.parse(
        TERMINAL_LAYOUT_SHAPE_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_LAYOUT_SHAPE_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SHAPE_EXTRACT_OWNER.removeprefix("_")
    ]


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


def _int_tensor(name: str, values: list[int]) -> TensorIR:
    return _tensor(
        name,
        [len(values)],
        dtype="INT64",
        data=np.asarray(values, dtype=np.int64),
    )


def _snapshot(model_ir: ModelIR) -> dict[str, object]:
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
                "quantization": copy.deepcopy(tensor.quantization),
            }
            for name, tensor in model_ir.tensors.items()
        },
        "operators": [
            {
                "op_type": operator.op_type,
                "inputs": list(operator.inputs),
                "outputs": list(operator.outputs),
                "options": copy.deepcopy(operator.options),
                "version": operator.version,
            }
            for operator in model_ir.operators
        ],
    }


def _base_model_ir() -> ModelIR:
    model_ir = ModelIR("shape_extract_nhwc")
    model_ir.inputs = ["x"]
    model_ir.tensors["x"] = _tensor("x", [1, 5, 7, 3])
    model_ir.tensors["perm"] = _tensor(
        "perm",
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["x_nchw"] = _tensor("x_nchw", [1, 3, 5, 7])
    model_ir.tensors["shape_nchw"] = _tensor(
        "shape_nchw", [4], dtype="INT64"
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["x_nchw"]),
        OperatorIR("SHAPE", ["x_nchw"], ["shape_nchw"]),
    ]
    return model_ir


@pytest.mark.parametrize("shared_indices", [False, True])
def test_shape_extract_layout_remaps_gather_indices(
    shared_indices: bool,
) -> None:
    model_ir = _base_model_ir()
    model_ir.outputs = ["selected"]
    model_ir.tensors["indices"] = _int_tensor("indices", [1, -1])
    model_ir.tensors["selected"] = _tensor(
        "selected", [2], dtype="INT64"
    )
    model_ir.operators.append(
        OperatorIR(
            "GATHER",
            ["shape_nchw", "indices"],
            ["selected"],
            options={"axis": 0, "batchDims": 0},
        )
    )
    if shared_indices:
        model_ir.inputs.append("other_shape")
        model_ir.tensors["other_shape"] = _tensor(
            "other_shape", [4], dtype="INT64"
        )
        model_ir.tensors["other_selected"] = _tensor(
            "other_selected", [2], dtype="INT64"
        )
        model_ir.operators.append(
            OperatorIR(
                "GATHER",
                ["other_shape", "indices"],
                ["other_selected"],
                options={"axis": 0, "batchDims": 0},
            )
        )

    stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)

    assert stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 1,
    }
    shape_op = next(op for op in model_ir.operators if op.op_type == "SHAPE")
    assert list(shape_op.inputs) == ["x"]
    target_gather = next(
        op for op in model_ir.operators if op.outputs == ["selected"]
    )
    target_indices = model_ir.tensors[target_gather.inputs[1]].data
    np.testing.assert_array_equal(
        target_indices,
        np.asarray([3, 2], dtype=np.int64),
    )
    if shared_indices:
        np.testing.assert_array_equal(
            model_ir.tensors["indices"].data,
            np.asarray([1, -1], dtype=np.int64),
        )
        assert target_gather.inputs[1] != "indices"
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    assert _optimize_transpose_shape_extract_nhwc_to_nchw_chains(
        model_ir
    ) == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 0,
    }


def _slice_model_ir(*, begin: int, size: int) -> ModelIR:
    model_ir = _base_model_ir()
    model_ir.outputs = ["selected"]
    model_ir.tensors["begin"] = _int_tensor("begin", [begin])
    model_ir.tensors["size"] = _int_tensor("size", [size])
    model_ir.tensors["selected"] = _tensor(
        "selected", [2], dtype="INT64"
    )
    model_ir.operators.append(
        OperatorIR(
            "SLICE",
            ["shape_nchw", "begin", "size"],
            ["selected"],
        )
    )
    return model_ir


def test_shape_extract_layout_remaps_contiguous_slice() -> None:
    model_ir = _slice_model_ir(begin=2, size=2)

    stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)

    assert stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 1,
    }
    slice_op = next(op for op in model_ir.operators if op.op_type == "SLICE")
    assert list(slice_op.inputs) == ["shape_nchw", "begin", "size"]
    np.testing.assert_array_equal(
        model_ir.tensors["begin"].data,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["size"].data,
        np.asarray([2], dtype=np.int64),
    )


def test_shape_extract_layout_converts_noncontiguous_slice_to_gather() -> None:
    model_ir = _slice_model_ir(begin=1, size=2)

    stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)

    assert stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 1,
    }
    gather_op = next(op for op in model_ir.operators if op.outputs == ["selected"])
    assert gather_op.op_type == "GATHER"
    assert gather_op.version == 1
    assert gather_op.options == {"axis": 0, "batchDims": 0}
    assert list(gather_op.inputs[:1]) == ["shape_nchw"]
    np.testing.assert_array_equal(
        model_ir.tensors[gather_op.inputs[1]].data,
        np.asarray([3, 1], dtype=np.int64),
    )
    assert model_ir.tensors["selected"].shape == [2]
    assert model_ir.tensors["selected"].shape_signature == [2]


@pytest.mark.parametrize(
    "guard",
    [
        "public_transpose",
        "public_shape",
        "transpose_fanout",
        "unsupported_shape_user",
        "gather_axis",
        "invalid_gather_index",
        "nonconstant_indices",
        "empty_slice",
    ],
)
def test_shape_extract_layout_rejects_unsafe_or_unsupported_graphs(
    guard: str,
) -> None:
    model_ir = _base_model_ir()
    model_ir.outputs = ["selected"]
    model_ir.tensors["indices"] = _int_tensor("indices", [1])
    model_ir.tensors["selected"] = _tensor(
        "selected", [1], dtype="INT64"
    )
    model_ir.operators.append(
        OperatorIR(
            "GATHER",
            ["shape_nchw", "indices"],
            ["selected"],
            options={"axis": 0, "batchDims": 0},
        )
    )
    if guard == "public_transpose":
        model_ir.outputs.append("x_nchw")
    elif guard == "public_shape":
        model_ir.outputs.append("shape_nchw")
    elif guard == "transpose_fanout":
        model_ir.tensors["tap"] = _tensor("tap", [1, 3, 5, 7])
        model_ir.operators.append(OperatorIR("RELU", ["x_nchw"], ["tap"]))
    elif guard == "unsupported_shape_user":
        model_ir.tensors["shape_tap"] = _tensor(
            "shape_tap", [4], dtype="INT64"
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["shape_nchw"], ["shape_tap"])
        )
    elif guard == "gather_axis":
        model_ir.operators[-1].options["axis"] = 1
    elif guard == "invalid_gather_index":
        model_ir.tensors["indices"].data = np.asarray([4], dtype=np.int64)
    elif guard == "nonconstant_indices":
        model_ir.tensors["indices"].data = None
    elif guard == "empty_slice":
        model_ir.operators[-1] = OperatorIR(
            "SLICE",
            ["shape_nchw", "indices", "indices"],
            ["selected"],
        )
        model_ir.tensors["indices"].data = np.asarray([4], dtype=np.int64)
    else:
        raise AssertionError(f"unsupported guard: {guard}")
    before = _snapshot(model_ir)

    stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)

    assert stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 0,
    }
    assert _snapshot(model_ir) == before


def test_shape_extract_owner_matches_lowerer_compatibility_wrapper() -> None:
    direct_model_ir = _slice_model_ir(begin=1, size=2)
    wrapper_model_ir = copy.deepcopy(direct_model_ir)

    direct_stats = optimize_transpose_shape_extract_nhwc_to_nchw_chains(
        direct_model_ir
    )
    wrapper_stats = _optimize_transpose_shape_extract_nhwc_to_nchw_chains(
        wrapper_model_ir
    )

    assert direct_stats == {
        "optimized_transpose_shape_extract_nhwc_to_nchw_chains": 1,
    }
    assert wrapper_stats == direct_stats
    assert _snapshot(wrapper_model_ir) == _snapshot(direct_model_ir)


def test_pre_qkv_terminal_shape_extract_captures_complete_mutation_evidence() -> None:
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
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == TERMINAL_QKV_RESULT
    )
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Assign)
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == TERMINAL_QKV_OWNER
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "shared_model_ir_pass_context"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in invocation.value.keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}

    previous = lowerer.body[invocation_index - 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == (
        "_pre_terminal_affine_slice_spp_results"
    )
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "run_pre_terminal_affine_slice_spp_cleanup"
    )
    assert [ast.unparse(argument) for argument in previous.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert previous.value.keywords == []
    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_terminal_activation_bridge_results"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == "run_terminal_activation_bridge_cleanup"

    all_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SHAPE_EXTRACT_OWNER
    ]
    assert (
        len(all_calls)
        + len(_terminal_qkv_shape_calls())
        + len(_terminal_layout_shape_calls())
        + TERMINAL_CONCAT_BRIDGE_LAYOUT_PASS_IDS.count(SHAPE_EXTRACT_OWNER)
        == 3
    )
    assert len(_terminal_qkv_shape_calls()) == 1
    assert [
        ast.unparse(argument)
        for argument in _terminal_qkv_shape_calls()[0].args
    ] == ["context.model_ir"]
    assert len(_terminal_layout_shape_calls()) == 1
    assert [
        ast.unparse(argument)
        for argument in _terminal_layout_shape_calls()[0].args
    ] == ["context.model_ir"]
