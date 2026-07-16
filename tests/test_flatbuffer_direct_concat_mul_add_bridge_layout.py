from __future__ import annotations

import ast
import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import onnx2tf.tflite_builder.lower_from_onnx2tf as lowerer_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_concat_mul_add_transpose_nhwc_bridge_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_PERMUTATION = "__concat_mul_tail_nhwc_to_nchw_perm_rank4__"


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
    legacy_consumer: bool = False,
    concat_public_output: bool = False,
) -> None:
    source_signature = [-1, 3, 2, 2] if dynamic_batch else [1, 3, 2, 2]
    pre_signature = [-1, 2, 3, 2] if dynamic_batch else [1, 2, 3, 2]
    concat_nchw_signature = (
        [-1, 4, 3, 2] if dynamic_batch else [1, 4, 3, 2]
    )
    nhwc_signature = [-1, 3, 2, 4] if dynamic_batch else [1, 3, 2, 4]

    source0 = f"{prefix}x0_nhwc"
    source1 = f"{prefix}x1_nhwc"
    pre0 = f"{prefix}x0_nchw"
    pre1 = f"{prefix}x1_nchw"
    concat_output = f"{prefix}cat_nchw"
    mul_constant = f"{prefix}mul_const"
    mul_output = f"{prefix}mul_out"
    post_output = f"{prefix}mul_out_nhwc"
    add_bias = f"{prefix}add_bias"
    output = f"{prefix}y"
    to_nchw = f"{prefix}to_nchw_perm"
    to_nhwc = f"{prefix}to_nhwc_perm"

    model_ir.inputs.extend([source0, source1])
    model_ir.outputs.append(output)
    if concat_public_output:
        model_ir.outputs.append(concat_output)

    for source in (source0, source1):
        _tensor(
            model_ir,
            source,
            [1, 3, 2, 2],
            signature=source_signature,
        )
    for pre_output in (pre0, pre1):
        _tensor(
            model_ir,
            pre_output,
            [1, 2, 3, 2],
            signature=pre_signature,
        )
    _tensor(
        model_ir,
        concat_output,
        [1, 4, 3, 2],
        signature=concat_nchw_signature,
    )
    _tensor(
        model_ir,
        mul_constant,
        [1, 4, 1, 1],
        data=np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1),
    )
    _tensor(
        model_ir,
        mul_output,
        [1, 4, 3, 2],
        signature=concat_nchw_signature,
    )
    _tensor(
        model_ir,
        post_output,
        [1, 3, 2, 4],
        signature=nhwc_signature,
    )
    _tensor(
        model_ir,
        add_bias,
        [1, 1, 1, 4],
        data=np.arange(4, dtype=np.float32).reshape(1, 1, 1, 4),
    )
    _tensor(
        model_ir,
        output,
        [1, 3, 2, 4],
        signature=nhwc_signature,
    )
    _tensor(
        model_ir,
        to_nchw,
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    _tensor(
        model_ir,
        to_nhwc,
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )

    model_ir.operators.extend(
        [
            OperatorIR("TRANSPOSE", [source0, to_nchw], [pre0]),
            OperatorIR("TRANSPOSE", [source1, to_nchw], [pre1]),
            OperatorIR(
                "CONCATENATION",
                [pre0, pre1],
                [concat_output],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
                axis_semantics={"axis": "physical"},
                version=2,
                onnx_node_name=f"{prefix}concat",
                onnx_op_type="Concat",
            ),
            OperatorIR("MUL", [concat_output, mul_constant], [mul_output]),
            OperatorIR("TRANSPOSE", [mul_output, to_nhwc], [post_output]),
            OperatorIR("ADD", [post_output, add_bias], [output]),
        ]
    )

    if legacy_consumer:
        legacy_constant = f"{prefix}legacy_const"
        legacy_output = f"{prefix}legacy_out"
        _tensor(
            model_ir,
            legacy_constant,
            [1, 4, 1, 1],
            data=np.ones((1, 4, 1, 1), dtype=np.float32),
        )
        _tensor(
            model_ir,
            legacy_output,
            [1, 4, 3, 2],
            signature=concat_nchw_signature,
        )
        model_ir.outputs.append(legacy_output)
        model_ir.operators.append(
            OperatorIR(
                "MUL",
                [concat_output, legacy_constant],
                [legacy_output],
            )
        )


def _model(
    *,
    branches: int = 1,
    dynamic_batch: bool = False,
    legacy_consumer: bool = False,
    concat_public_output: bool = False,
) -> ModelIR:
    model_ir = ModelIR("concat_mul_add_bridge_characterization")
    for branch_index in range(int(branches)):
        _add_chain(
            model_ir,
            prefix=f"branch{branch_index}_",
            dynamic_batch=dynamic_batch,
            legacy_consumer=legacy_consumer,
            concat_public_output=concat_public_output,
        )
    return model_ir


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 0,
    }
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_concat_mul_add_bridge_rewrites_ordinary_chain(
    dynamic_batch: bool,
) -> None:
    model_ir = _model(dynamic_batch=dynamic_batch)

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 1,
    }
    assert [operator.op_type for operator in model_ir.operators] == [
        "CONCATENATION",
        "MUL",
        "ADD",
    ]
    concat, mul, add = model_ir.operators
    assert concat.inputs == ["branch0_x0_nhwc", "branch0_x1_nhwc"]
    assert concat.outputs == ["branch0_cat_nchw"]
    assert concat.options == {
        "axis": 3,
        "fusedActivationFunction": "NONE",
    }
    assert concat.axis_semantics == {"axis": "physical"}
    assert concat.version == 2
    assert concat.onnx_node_name == "branch0_concat"
    assert concat.onnx_op_type == "Concat"
    assert mul.inputs == ["branch0_cat_nchw", "branch0_mul_const"]
    assert add.inputs == ["branch0_mul_out", "branch0_add_bias"]
    assert model_ir.tensors["branch0_cat_nchw"].shape == [1, 3, 2, 4]
    assert model_ir.tensors["branch0_mul_const"].shape == [1, 1, 1, 4]
    assert np.asarray(
        model_ir.tensors["branch0_mul_const"].data
    ).reshape(-1).tolist() == [0.0, 1.0, 2.0, 3.0]
    assert model_ir.tensors["branch0_mul_out"].shape == [1, 3, 2, 4]
    expected_signature = (
        [-1, 3, 2, 4] if dynamic_batch else [1, 3, 2, 4]
    )
    assert model_ir.tensors["branch0_cat_nchw"].shape_signature == (
        expected_signature
    )
    assert model_ir.tensors["branch0_mul_out"].shape_signature == (
        expected_signature
    )
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "legacy_consumer,concat_public_output",
    [(True, False), (False, True)],
)
def test_concat_mul_add_bridge_preserves_nchw_concat_boundary_with_adapter(
    legacy_consumer: bool,
    concat_public_output: bool,
) -> None:
    model_ir = _model(
        legacy_consumer=legacy_consumer,
        concat_public_output=concat_public_output,
    )

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 1,
    }
    concat = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    )
    assert concat.inputs == ["branch0_x0_nhwc", "branch0_x1_nhwc"]
    assert concat.outputs == ["branch0_cat_nchw_nhwc"]
    main_mul = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["branch0_mul_out"]
    )
    assert main_mul.inputs[0] == "branch0_cat_nchw_nhwc"
    adapter = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE"
        and operator.outputs == ["branch0_cat_nchw"]
    )
    assert adapter.inputs == ["branch0_cat_nchw_nhwc", _ADAPTER_PERMUTATION]
    assert np.asarray(model_ir.tensors[_ADAPTER_PERMUTATION].data).tolist() == [
        0,
        3,
        1,
        2,
    ]
    assert model_ir.tensors["branch0_cat_nchw"].shape == [1, 4, 3, 2]
    assert model_ir.tensors["branch0_cat_nchw_nhwc"].shape == [1, 3, 2, 4]
    assert validate_model_ir_invariants(model_ir) == []


def test_concat_mul_add_bridge_rewrites_multiple_chains_and_reaches_fixed_point() -> None:
    model_ir = _model(branches=2)

    first_stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)
    after_first = _normalize(copy.deepcopy(model_ir))
    second_stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert first_stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 2,
    }
    assert second_stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 0,
    }
    assert _normalize(model_ir) == after_first
    assert [
        operator.options["axis"]
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    ] == [3, 3]


def test_concat_mul_add_bridge_reuses_one_graph_index(
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

    stats = lowerer_module._optimize_concat_mul_add_transpose_nhwc_bridge_chains(
        model_ir
    )

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 2,
    }
    assert refresh_count == 1


def test_concat_mul_add_bridge_clones_shared_mul_constant_collision_safely() -> None:
    model_ir = _model()
    original = np.asarray(model_ir.tensors["branch0_mul_const"].data).copy()
    _tensor(model_ir, "constant_copy", [1, 4, 1, 1])
    model_ir.outputs.append("constant_copy")
    model_ir.operators.append(
        OperatorIR("IDENTITY", ["branch0_mul_const"], ["constant_copy"])
    )
    _tensor(
        model_ir,
        "branch0_mul_const_nhwc",
        [1],
        data=np.asarray([99.0], dtype=np.float32),
    )
    model_ir.outputs.append("branch0_mul_const_nhwc")

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 1,
    }
    main_mul = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["branch0_mul_out"]
    )
    assert main_mul.inputs == [
        "branch0_cat_nchw",
        "branch0_mul_const_nhwc_1",
    ]
    assert np.array_equal(model_ir.tensors["branch0_mul_const"].data, original)
    assert np.asarray(
        model_ir.tensors["branch0_mul_const_nhwc"].data
    ).tolist() == [99.0]
    assert model_ir.tensors["branch0_mul_const_nhwc_1"].shape == [1, 1, 1, 4]


def test_concat_mul_add_bridge_accepts_scalar_mul_constant() -> None:
    model_ir = _model()
    tensor = model_ir.tensors["branch0_mul_const"]
    tensor.shape = [1]
    tensor.shape_signature = [1]
    tensor.data = np.asarray([2.0], dtype=np.float32)

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 1,
    }
    assert model_ir.tensors["branch0_mul_const"].shape == [1]
    assert np.asarray(model_ir.tensors["branch0_mul_const"].data).tolist() == [
        2.0
    ]


def test_concat_mul_add_bridge_does_not_prune_unmatched_graph() -> None:
    model_ir = _model()
    _tensor(model_ir, "unused", [1])
    model_ir.tensors["branch0_to_nhwc_perm"].data = np.asarray(
        [0, 3, 1, 2],
        dtype=np.int32,
    )

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 0,
    }
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
            lambda model_ir: model_ir.outputs.append("branch0_mul_out_nhwc"),
            id="public-post-output",
        ),
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
            lambda model_ir: model_ir.tensors[
                "branch0_add_bias"
            ].__setattr__("data", None),
            id="dynamic-add-side",
        ),
        pytest.param(
            lambda model_ir: (
                model_ir.tensors["branch0_add_bias"].__setattr__(
                    "shape",
                    [1, 4, 1, 1],
                ),
                model_ir.tensors["branch0_add_bias"].__setattr__(
                    "shape_signature",
                    [1, 4, 1, 1],
                ),
                model_ir.tensors["branch0_add_bias"].__setattr__(
                    "data",
                    np.zeros((1, 4, 1, 1), dtype=np.float32),
                ),
            ),
            id="nchw-add-side",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_mul_const"
            ].__setattr__("data", None),
            id="dynamic-mul-constant",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "side", [1, 2, 3, 2]),
                model_ir.outputs.append("side"),
                model_ir.operators.append(
                    OperatorIR("IDENTITY", ["branch0_x0_nchw"], ["side"])
                ),
            ),
            id="pre-adapter-fanout",
        ),
    ],
)
def test_concat_mul_add_bridge_rejects_existing_unsafe_contracts(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


def test_concat_mul_add_bridge_keeps_legacy_adapter_topological() -> None:
    model_ir = _model(legacy_consumer=True)

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 1,
    }
    assert validate_model_ir_invariants(model_ir) == []
    adapter_index = next(
        index
        for index, operator in enumerate(model_ir.operators)
        if operator.op_type == "TRANSPOSE"
        and operator.outputs == ["branch0_cat_nchw"]
    )
    legacy_index = next(
        index
        for index, operator in enumerate(model_ir.operators)
        if operator.outputs == ["branch0_legacy_out"]
    )
    assert adapter_index < legacy_index


@pytest.mark.parametrize(
    "case",
    [
        "missing-concat-tensor",
        "missing-mul-output-tensor",
        "rank-three-source",
        "short-concat-signature",
        "short-mul-output-signature",
    ],
)
def test_concat_mul_add_bridge_rejects_incomplete_metadata(case: str) -> None:
    model_ir = _model()
    if case == "missing-concat-tensor":
        del model_ir.tensors["branch0_cat_nchw"]
    elif case == "missing-mul-output-tensor":
        del model_ir.tensors["branch0_mul_out"]
    elif case == "rank-three-source":
        model_ir.tensors["branch0_x0_nhwc"].shape = [1, 3, 2]
        model_ir.tensors["branch0_x0_nhwc"].shape_signature = [1, 3, 2]
    elif case == "short-concat-signature":
        model_ir.tensors["branch0_cat_nchw"].shape_signature = [1, 4, 3]
    elif case == "short-mul-output-signature":
        model_ir.tensors["branch0_mul_out"].shape_signature = [1, 4, 3]

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize("case", ["public-input", "variable"])
def test_concat_mul_add_bridge_rejects_mutable_mul_constant(case: str) -> None:
    model_ir = _model()
    constant = model_ir.tensors["branch0_mul_const"]
    if case == "public-input":
        model_ir.inputs.append(constant.name)
    elif case == "variable":
        constant.is_variable = True

    _assert_transactional_rejection(model_ir)


def test_concat_mul_add_bridge_clones_public_mul_constant_output() -> None:
    model_ir = _model()
    constant_name = "branch0_mul_const"
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    model_ir.outputs.append(constant_name)

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 1,
    }
    main_mul = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["branch0_mul_out"]
    )
    assert main_mul.inputs[1] != constant_name
    assert np.array_equal(model_ir.tensors[constant_name].data, original)


@pytest.mark.parametrize("legacy_consumer", [False, True])
def test_concat_mul_add_bridge_remaps_per_axis_quantization(
    legacy_consumer: bool,
) -> None:
    model_ir = _model(legacy_consumer=legacy_consumer)
    for name in (
        "branch0_cat_nchw",
        "branch0_mul_const",
        "branch0_mul_out",
    ):
        model_ir.tensors[name].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3, 0.4],
            zero_point=[0, 0, 0, 0],
            quantized_dimension=1,
        )

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 1,
    }
    concat_name = (
        "branch0_cat_nchw_nhwc"
        if legacy_consumer
        else "branch0_cat_nchw"
    )
    main_mul = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["branch0_mul_out"]
    )
    for name in (
        concat_name,
        main_mul.inputs[1],
        "branch0_mul_out",
    ):
        quantization = model_ir.tensors[name].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3
    assert validate_model_ir_invariants(model_ir) == []


def test_concat_mul_add_bridge_uses_private_collision_safe_adapter_constant() -> None:
    model_ir = _model(legacy_consumer=True)
    _tensor(
        model_ir,
        _ADAPTER_PERMUTATION,
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.inputs.append(_ADAPTER_PERMUTATION)
    original = np.asarray(model_ir.tensors[_ADAPTER_PERMUTATION].data).copy()

    stats = _optimize_concat_mul_add_transpose_nhwc_bridge_chains(model_ir)

    assert stats == {
        "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 1,
    }
    adapter = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE"
        and operator.outputs == ["branch0_cat_nchw"]
    )
    assert adapter.inputs[1] != _ADAPTER_PERMUTATION
    assert np.array_equal(
        model_ir.tensors[_ADAPTER_PERMUTATION].data,
        original,
    )


def test_concat_mul_add_bridge_rejects_late_metadata_error_atomically() -> None:
    model_ir = _model(legacy_consumer=True)
    model_ir.tensors["branch0_cat_nchw"].shape_signature = [1, None, 3, 2]

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    ["duplicate-post-output", "reverse-post-add", "public-pre-output-input"],
)
def test_concat_mul_add_bridge_rejects_invalid_topology(case: str) -> None:
    model_ir = _model()
    if case == "duplicate-post-output":
        model_ir.operators.insert(
            5,
            OperatorIR(
                "IDENTITY",
                ["branch0_x0_nhwc"],
                ["branch0_mul_out_nhwc"],
            ),
        )
    elif case == "reverse-post-add":
        model_ir.operators[4], model_ir.operators[5] = (
            model_ir.operators[5],
            model_ir.operators[4],
        )
    elif case == "public-pre-output-input":
        model_ir.inputs.append("branch0_x0_nchw")

    _assert_transactional_rejection(model_ir)


def test_concat_mul_add_bridge_keeps_raw_owner_and_ordered_boundaries() -> None:
    lowering_path = (
        REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
    )
    lowering_tree = ast.parse(lowering_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_optimize_concat_mul_add_transpose_nhwc_bridge_chains"
    )
    assert owner.end_lineno - owner.lineno + 1 == 652
    assert any(isinstance(node, ast.While) for node in ast.walk(owner))

    lowerer = next(
        node
        for node in lowering_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    expected = {
        "_run_terminal_slice_concat_layout_recovery_sequence": (
            "_optimize_transpose_mul_posttranspose_add_nhwc_chains",
            "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
        ),
        "_run_terminal_affine_concat_split_recovery_sequence": (
            "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
            "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
        ),
    }
    observed: dict[str, tuple[str, str]] = {}
    for statement in lowerer.body:
        if not isinstance(statement, ast.FunctionDef) or statement.name not in expected:
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
            "_optimize_concat_mul_add_transpose_nhwc_bridge_chains"
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
