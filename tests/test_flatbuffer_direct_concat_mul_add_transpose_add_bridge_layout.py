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
    _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains,
)
from onnx2tf.tflite_builder.passes.concat_mul_add_transpose_add_bridge_layout import (
    _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains as _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains_owner,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_PERMUTATION = (
    "__concat_affine_tail_nhwc_to_nchw_perm_rank4__"
)
_STATS = {
    "optimized_concat_mul_add_transpose_add_nhwc_bridge_chains": 1,
}
_ZERO_STATS = {
    "optimized_concat_mul_add_transpose_add_nhwc_bridge_chains": 0,
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
    legacy_consumer: bool = False,
) -> None:
    source_signature = (
        [-1, 3, 2, 2] if dynamic_batch else [1, 3, 2, 2]
    )
    pre_signature = (
        [-1, 2, 3, 2] if dynamic_batch else [1, 2, 3, 2]
    )
    nchw_signature = (
        [-1, 4, 3, 2] if dynamic_batch else [1, 4, 3, 2]
    )
    nhwc_signature = (
        [-1, 3, 2, 4] if dynamic_batch else [1, 3, 2, 4]
    )

    source0 = f"{prefix}x0_nhwc"
    source1 = f"{prefix}x1_nhwc"
    pre0 = f"{prefix}x0_nchw"
    pre1 = f"{prefix}x1_nchw"
    concat_output = f"{prefix}cat_nchw"
    mul_constant = f"{prefix}mul_const"
    mul_output = f"{prefix}mul_out"
    pre_add_constant = f"{prefix}pre_add_const"
    pre_add_output = f"{prefix}pre_add_out"
    post_output = f"{prefix}pre_add_out_nhwc"
    tail_constant = f"{prefix}tail_const"
    output = f"{prefix}y"
    to_nchw = f"{prefix}to_nchw_perm"
    to_nhwc = f"{prefix}to_nhwc_perm"

    model_ir.inputs.extend([source0, source1])
    model_ir.outputs.append(output)

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
        signature=nchw_signature,
    )
    _tensor(
        model_ir,
        mul_constant,
        [1, 4, 1, 1],
        data=np.arange(1, 5, dtype=np.float32).reshape(1, 4, 1, 1),
    )
    _tensor(
        model_ir,
        mul_output,
        [1, 4, 3, 2],
        signature=nchw_signature,
    )
    _tensor(
        model_ir,
        pre_add_constant,
        [1, 4, 1, 1],
        data=np.arange(5, 9, dtype=np.float32).reshape(1, 4, 1, 1),
    )
    _tensor(
        model_ir,
        pre_add_output,
        [1, 4, 3, 2],
        signature=nchw_signature,
    )
    _tensor(
        model_ir,
        post_output,
        [1, 3, 2, 4],
        signature=nhwc_signature,
    )
    _tensor(
        model_ir,
        tail_constant,
        [1, 1, 1, 4],
        data=np.arange(9, 13, dtype=np.float32).reshape(1, 1, 1, 4),
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
            OperatorIR(
                "ADD",
                [mul_output, pre_add_constant],
                [pre_add_output],
            ),
            OperatorIR(
                "TRANSPOSE",
                [pre_add_output, to_nhwc],
                [post_output],
            ),
            OperatorIR("ADD", [post_output, tail_constant], [output]),
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
            signature=nchw_signature,
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
) -> ModelIR:
    model_ir = ModelIR("concat_mul_add_transpose_add_characterization")
    for branch_index in range(int(branches)):
        _add_chain(
            model_ir,
            prefix=f"branch{branch_index}_",
            dynamic_batch=dynamic_batch,
            legacy_consumer=legacy_consumer,
        )
    return model_ir


def _owner_wrapper_case(case: str) -> ModelIR:
    if case == "dynamic":
        return _model(dynamic_batch=True)
    if case == "multiple":
        return _model(branches=2)
    if case in {"legacy", "per-axis-legacy", "late-metadata"}:
        model_ir = _model(legacy_consumer=True)
    elif case == "adapter-collision":
        model_ir = _model(legacy_consumer=True)
    else:
        model_ir = _model()

    if case == "scalar":
        for name, value in (
            ("branch0_mul_const", 2.0),
            ("branch0_pre_add_const", 3.0),
        ):
            tensor = model_ir.tensors[name]
            tensor.shape = [1]
            tensor.shape_signature = [1]
            tensor.data = np.asarray([value], dtype=np.float32)
    elif case in {"shared-mul", "shared-add"}:
        constant_name = (
            "branch0_mul_const"
            if case == "shared-mul"
            else "branch0_pre_add_const"
        )
        shared_output = f"{constant_name}_copy"
        _tensor(model_ir, shared_output, [1, 4, 1, 1])
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
    elif case in {"public-mul", "public-add"}:
        model_ir.outputs.append(
            "branch0_mul_const"
            if case == "public-mul"
            else "branch0_pre_add_const"
        )
    elif case in {"per-axis", "per-axis-legacy"}:
        for name in (
            "branch0_cat_nchw",
            "branch0_mul_const",
            "branch0_mul_out",
            "branch0_pre_add_const",
            "branch0_pre_add_out",
        ):
            model_ir.tensors[name].quantization = QuantParamIR(
                scale=[0.1, 0.2, 0.3, 0.4],
                zero_point=[0, 0, 0, 0],
                quantized_dimension=1,
            )
    elif case == "adapter-collision":
        _tensor(
            model_ir,
            _ADAPTER_PERMUTATION,
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        )
        model_ir.inputs.append(_ADAPTER_PERMUTATION)
    elif case == "unmatched":
        _tensor(model_ir, "unused", [1])
        model_ir.tensors["branch0_to_nhwc_perm"].data = np.asarray(
            [0, 3, 1, 2],
            dtype=np.int32,
        )
    elif case == "missing-metadata":
        del model_ir.tensors["branch0_pre_add_out"]
    elif case == "late-constant":
        tensor = model_ir.tensors["branch0_pre_add_const"]
        tensor.shape = [2, 2]
        tensor.shape_signature = [2, 2]
        tensor.data = np.zeros((2, 2), dtype=np.float32)
    elif case == "late-metadata":
        model_ir.tensors["branch0_cat_nchw"].shape_signature = [
            1,
            None,
            3,
            2,
        ]
    elif case == "malformed-axis":
        model_ir.operators[2].options["axis"] = None
    elif case == "reverse-topology":
        model_ir.operators[5], model_ir.operators[6] = (
            model_ir.operators[6],
            model_ir.operators[5],
        )
    elif case == "public-internal":
        model_ir.inputs.append("branch0_x0_nchw")
    return model_ir


@pytest.mark.parametrize(
    "case",
    [
        "ordinary",
        "dynamic",
        "multiple",
        "scalar",
        "shared-mul",
        "shared-add",
        "public-mul",
        "public-add",
        "legacy",
        "per-axis",
        "per-axis-legacy",
        "adapter-collision",
        "unmatched",
        "missing-metadata",
        "late-constant",
        "late-metadata",
        "malformed-axis",
        "reverse-topology",
        "public-internal",
    ],
)
def test_concat_mul_add_transpose_add_owner_and_wrapper_are_identical(
    case: str,
) -> None:
    owner_model = _owner_wrapper_case(case)
    wrapper_model = copy.deepcopy(owner_model)

    owner_stats = (
        _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains_owner(
            owner_model
        )
    )
    wrapper_stats = (
        _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
            wrapper_model
        )
    )

    assert wrapper_stats == owner_stats
    assert _normalize(wrapper_model) == _normalize(owner_model)


def _assert_transactional_rejection(model_ir: ModelIR) -> None:
    before = _normalize(copy.deepcopy(model_ir))

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _ZERO_STATS
    assert _normalize(model_ir) == before


@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_concat_mul_add_transpose_add_rewrites_ordinary_chain(
    dynamic_batch: bool,
) -> None:
    model_ir = _model(dynamic_batch=dynamic_batch)

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    assert [operator.op_type for operator in model_ir.operators] == [
        "CONCATENATION",
        "MUL",
        "ADD",
        "ADD",
    ]
    concat, mul, pre_add, tail_add = model_ir.operators
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
    assert pre_add.inputs == ["branch0_mul_out", "branch0_pre_add_const"]
    assert tail_add.inputs == ["branch0_pre_add_out", "branch0_tail_const"]
    for name in (
        "branch0_cat_nchw",
        "branch0_mul_out",
        "branch0_pre_add_out",
    ):
        assert model_ir.tensors[name].shape == [1, 3, 2, 4]
    for name in ("branch0_mul_const", "branch0_pre_add_const"):
        assert model_ir.tensors[name].shape == [1, 1, 1, 4]
    expected_signature = (
        [-1, 3, 2, 4] if dynamic_batch else [1, 3, 2, 4]
    )
    for name in (
        "branch0_cat_nchw",
        "branch0_mul_out",
        "branch0_pre_add_out",
    ):
        assert model_ir.tensors[name].shape_signature == expected_signature
    assert validate_model_ir_invariants(model_ir) == []


def test_concat_mul_add_transpose_add_preserves_legacy_concat_adapter() -> None:
    model_ir = _model(legacy_consumer=True)

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
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
    assert adapter.inputs == [
        "branch0_cat_nchw_nhwc",
        _ADAPTER_PERMUTATION,
    ]
    assert np.asarray(
        model_ir.tensors[_ADAPTER_PERMUTATION].data
    ).tolist() == [0, 3, 1, 2]


def test_concat_mul_add_transpose_add_rewrites_multiple_and_fixed_point() -> None:
    model_ir = _model(branches=2)

    first = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )
    after_first = _normalize(copy.deepcopy(model_ir))
    second = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert first == {
        "optimized_concat_mul_add_transpose_add_nhwc_bridge_chains": 2,
    }
    assert second == _ZERO_STATS
    assert _normalize(model_ir) == after_first
    assert [
        operator.options["axis"]
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    ] == [3, 3]


def test_concat_mul_add_transpose_add_reuses_one_graph_index(
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

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == {
        "optimized_concat_mul_add_transpose_add_nhwc_bridge_chains": 2,
    }
    assert refresh_count == 1


def test_concat_mul_add_transpose_add_accepts_scalar_affine_constants() -> None:
    model_ir = _model()
    for name, value in (
        ("branch0_mul_const", 2.0),
        ("branch0_pre_add_const", 3.0),
    ):
        tensor = model_ir.tensors[name]
        tensor.shape = [1]
        tensor.shape_signature = [1]
        tensor.data = np.asarray([value], dtype=np.float32)

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    for name, value in (
        ("branch0_mul_const", 2.0),
        ("branch0_pre_add_const", 3.0),
    ):
        assert model_ir.tensors[name].shape == [1]
        assert np.asarray(model_ir.tensors[name].data).tolist() == [value]


@pytest.mark.parametrize(
    "constant_name,op_output,data_input",
    [
        (
            "branch0_mul_const",
            "branch0_mul_out",
            "branch0_cat_nchw",
        ),
        (
            "branch0_pre_add_const",
            "branch0_pre_add_out",
            "branch0_mul_out",
        ),
    ],
)
def test_concat_mul_add_transpose_add_clones_shared_constant_safely(
    constant_name: str,
    op_output: str,
    data_input: str,
) -> None:
    model_ir = _model()
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    shared_output = f"{constant_name}_copy"
    _tensor(model_ir, shared_output, [1, 4, 1, 1])
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

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
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
    assert np.asarray(model_ir.tensors[collision_name].data).tolist() == [99.0]
    assert model_ir.tensors[clone_name].shape == [1, 1, 1, 4]


def test_concat_mul_add_transpose_add_does_not_prune_unmatched_graph() -> None:
    model_ir = _model()
    _tensor(model_ir, "unused", [1])
    model_ir.tensors["branch0_to_nhwc_perm"].data = np.asarray(
        [0, 3, 1, 2],
        dtype=np.int32,
    )

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
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
            lambda model_ir: model_ir.outputs.append(
                "branch0_pre_add_out"
            ),
            id="public-pre-add-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append(
                "branch0_pre_add_out_nhwc"
            ),
            id="public-post-output",
        ),
        pytest.param(
            lambda model_ir: model_ir.outputs.append("branch0_cat_nchw"),
            id="public-concat-output",
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
                "branch0_tail_const"
            ].__setattr__("data", None),
            id="dynamic-tail-constant",
        ),
        pytest.param(
            lambda model_ir: model_ir.tensors[
                "branch0_tail_const"
            ].__setattr__(
                "data",
                np.zeros((1, 4, 1, 1), dtype=np.float32),
            ),
            id="nchw-tail-constant",
        ),
        pytest.param(
            lambda model_ir: (
                _tensor(model_ir, "side", [1, 2, 3, 2]),
                model_ir.outputs.append("side"),
                model_ir.operators.append(
                    OperatorIR(
                        "IDENTITY",
                        ["branch0_x0_nchw"],
                        ["side"],
                    )
                ),
            ),
            id="pre-adapter-fanout",
        ),
    ],
)
def test_concat_mul_add_transpose_add_preserves_existing_rejections(
    mutate: Callable[[ModelIR], object],
) -> None:
    model_ir = _model()
    mutate(model_ir)

    _assert_transactional_rejection(model_ir)


def test_concat_mul_add_transpose_add_keeps_legacy_adapter_topological() -> None:
    model_ir = _model(legacy_consumer=True)

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
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
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "case",
    [
        "missing-concat-tensor",
        "missing-mul-output-tensor",
        "missing-pre-add-output-tensor",
        "rank-three-source",
        "short-concat-signature",
        "short-mul-output-signature",
        "short-pre-add-output-signature",
    ],
)
def test_concat_mul_add_transpose_add_rejects_incomplete_metadata(
    case: str,
) -> None:
    model_ir = _model()
    if case == "missing-concat-tensor":
        del model_ir.tensors["branch0_cat_nchw"]
    elif case == "missing-mul-output-tensor":
        del model_ir.tensors["branch0_mul_out"]
    elif case == "missing-pre-add-output-tensor":
        del model_ir.tensors["branch0_pre_add_out"]
    elif case == "rank-three-source":
        model_ir.tensors["branch0_x0_nhwc"].shape = [1, 3, 2]
        model_ir.tensors["branch0_x0_nhwc"].shape_signature = [1, 3, 2]
    elif case == "short-concat-signature":
        model_ir.tensors["branch0_cat_nchw"].shape_signature = [1, 4, 3]
    elif case == "short-mul-output-signature":
        model_ir.tensors["branch0_mul_out"].shape_signature = [1, 4, 3]
    elif case == "short-pre-add-output-signature":
        model_ir.tensors["branch0_pre_add_out"].shape_signature = [1, 4, 3]

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "constant_name,ownership",
    [
        ("branch0_mul_const", "public-input"),
        ("branch0_mul_const", "variable"),
        ("branch0_pre_add_const", "public-input"),
        ("branch0_pre_add_const", "variable"),
    ],
)
def test_concat_mul_add_transpose_add_preserves_constant_ownership(
    constant_name: str,
    ownership: str,
) -> None:
    model_ir = _model()
    if ownership == "public-input":
        model_ir.inputs.append(constant_name)
    elif ownership == "variable":
        model_ir.tensors[constant_name].is_variable = True
    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "constant_name,op_output,data_input",
    [
        (
            "branch0_mul_const",
            "branch0_mul_out",
            "branch0_cat_nchw",
        ),
        (
            "branch0_pre_add_const",
            "branch0_pre_add_out",
            "branch0_mul_out",
        ),
    ],
)
def test_concat_mul_add_transpose_add_clones_public_constant_output(
    constant_name: str,
    op_output: str,
    data_input: str,
) -> None:
    model_ir = _model()
    original = np.asarray(model_ir.tensors[constant_name].data).copy()
    model_ir.outputs.append(constant_name)

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    operator = next(
        candidate
        for candidate in model_ir.operators
        if candidate.outputs == [op_output]
    )
    clone_name = next(name for name in operator.inputs if name != data_input)
    assert clone_name != constant_name
    assert np.array_equal(model_ir.tensors[constant_name].data, original)
    assert model_ir.tensors[clone_name].shape == [1, 1, 1, 4]


@pytest.mark.parametrize("legacy_consumer", [False, True])
def test_concat_mul_add_transpose_add_remaps_per_axis_quantization(
    legacy_consumer: bool,
) -> None:
    model_ir = _model(legacy_consumer=legacy_consumer)
    for name in (
        "branch0_cat_nchw",
        "branch0_mul_const",
        "branch0_mul_out",
        "branch0_pre_add_const",
        "branch0_pre_add_out",
    ):
        model_ir.tensors[name].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3, 0.4],
            zero_point=[0, 0, 0, 0],
            quantized_dimension=1,
        )

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    concat_name = (
        "branch0_cat_nchw_nhwc"
        if legacy_consumer
        else "branch0_cat_nchw"
    )
    for name in (
        concat_name,
        "branch0_mul_const",
        "branch0_mul_out",
        "branch0_pre_add_const",
        "branch0_pre_add_out",
    ):
        quantization = model_ir.tensors[name].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3


@pytest.mark.parametrize(
    "case",
    [
        "public-input",
        "variable",
        "wrong-dtype",
        "quantized",
        "wrong-value",
    ],
)
def test_concat_mul_add_transpose_add_uses_private_adapter_permutation(
    case: str,
) -> None:
    model_ir = _model(legacy_consumer=True)
    _tensor(
        model_ir,
        _ADAPTER_PERMUTATION,
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    if case == "public-input":
        model_ir.inputs.append(_ADAPTER_PERMUTATION)
    elif case == "variable":
        model_ir.tensors[_ADAPTER_PERMUTATION].is_variable = True
    elif case == "wrong-dtype":
        model_ir.tensors[_ADAPTER_PERMUTATION].dtype = "FLOAT32"
    elif case == "quantized":
        model_ir.tensors[_ADAPTER_PERMUTATION].quantization = QuantParamIR(
            scale=[0.1],
            zero_point=[0],
            quantized_dimension=0,
        )
    elif case == "wrong-value":
        model_ir.tensors[_ADAPTER_PERMUTATION].data = np.asarray(
            [0, 2, 3, 1],
            dtype=np.int32,
        )
    if case != "public-input":
        model_ir.outputs.append(_ADAPTER_PERMUTATION)
    before = _normalize(copy.deepcopy(model_ir.tensors[_ADAPTER_PERMUTATION]))

    stats = _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
        model_ir
    )

    assert stats == _STATS
    adapter = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "TRANSPOSE"
        and operator.outputs == ["branch0_cat_nchw"]
    )
    assert adapter.inputs[1] != _ADAPTER_PERMUTATION
    assert _normalize(model_ir.tensors[_ADAPTER_PERMUTATION]) == before


def test_concat_mul_add_transpose_add_rejects_late_constant_error_atomically() -> None:
    model_ir = _model()
    tensor = model_ir.tensors["branch0_pre_add_const"]
    tensor.shape = [2, 2]
    tensor.shape_signature = [2, 2]
    tensor.data = np.zeros((2, 2), dtype=np.float32)

    _assert_transactional_rejection(model_ir)


def test_concat_mul_add_transpose_add_rejects_late_metadata_error_atomically() -> None:
    model_ir = _model(legacy_consumer=True)
    model_ir.tensors["branch0_cat_nchw"].shape_signature = [1, None, 3, 2]

    _assert_transactional_rejection(model_ir)


def test_concat_mul_add_transpose_add_rejects_malformed_axis() -> None:
    model_ir = _model()
    model_ir.operators[2].options["axis"] = None

    _assert_transactional_rejection(model_ir)


@pytest.mark.parametrize(
    "case",
    ["duplicate-post-output", "reverse-post-add", "public-pre-output-input"],
)
def test_concat_mul_add_transpose_add_rejects_invalid_topology(
    case: str,
) -> None:
    model_ir = _model()
    if case == "duplicate-post-output":
        model_ir.operators.insert(
            5,
            OperatorIR(
                "IDENTITY",
                ["branch0_x0_nhwc"],
                ["branch0_pre_add_out_nhwc"],
            ),
        )
    elif case == "reverse-post-add":
        model_ir.operators[5], model_ir.operators[6] = (
            model_ir.operators[6],
            model_ir.operators[5],
        )
    elif case == "public-pre-output-input":
        model_ir.inputs.append("branch0_x0_nchw")

    _assert_transactional_rejection(model_ir)


def test_concat_mul_add_transpose_add_keeps_owner_wrapper_and_boundaries() -> None:
    pass_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "concat_mul_add_transpose_add_bridge_layout.py"
    )
    pass_tree = ast.parse(pass_path.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in pass_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        == "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains"
    )
    assert owner.end_lineno - owner.lineno + 1 == 866
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
        == "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains"
    )
    assert len(wrapper.body) == 1
    assert isinstance(wrapper.body[0], ast.Return)
    wrapper_call = wrapper.body[0].value
    assert isinstance(wrapper_call, ast.Call)
    assert isinstance(wrapper_call.func, ast.Name)
    assert (
        wrapper_call.func.id
        == "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains_pass"
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
    target_name = "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains"
    terminal_index = TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS.index(target_name)
    assert (
        TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[terminal_index - 1],
        TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[terminal_index + 1],
    ) == (
        "_optimize_concat_mul_add_transpose_nhwc_bridge_chains",
        "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
    )
    expected = {
        "_run_terminal_affine_concat_split_recovery_sequence": (
            "_optimize_concat_mul_add_transpose_nhwc_bridge_chains",
            "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
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
        index = call_names.index(target_name)
        observed[statement.name] = (
            call_names[index - 1],
            call_names[index + 1],
        )
        assert len(calls[index].args) == 1
        assert isinstance(calls[index].args[0], ast.Name)
        assert calls[index].args[0].id == "model_ir"
        assert calls[index].keywords == []
    assert observed == expected
