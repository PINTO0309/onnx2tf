from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.split_channelwise_layout import (
    optimize_transpose_unary_split_concat_single_post_nchw,
)


_H = 3
_W = 4
_SOURCE_NHWC = [1, _H, _W, 2]
_SOURCE_NCHW = [1, 2, _H, _W]
_BRANCH_NHWC = [1, _H, _W, 1]
_BRANCH_NCHW = [1, 1, _H, _W]
_CONCAT_NHWC = [1, _H, _W, 3]
_CONCAT_NCHW = [1, 3, _H, _W]


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: Any = None,
    signature: list[int] | None = None,
    logical_layout: str = "UNKNOWN",
    physical_layout: str = "UNKNOWN",
    quantization: QuantParamIR | None = None,
    is_variable: bool = False,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=is_variable,
        quantization=quantization,
        logical_layout=logical_layout,
        physical_layout=physical_layout,
    )


def _signature(shape: list[int], dynamic: bool) -> list[int]:
    if not dynamic or len(shape) != 4:
        return list(shape)
    return [shape[0], -1, -1, shape[3]]


def _nchw_signature(shape: list[int], dynamic: bool) -> list[int]:
    if not dynamic:
        return list(shape)
    return [shape[0], shape[1], -1, -1]


def _make_model(
    *,
    axis_dtype: str = "INT32",
    dynamic: bool = False,
    public_concat: bool = True,
    shared_axis: bool = False,
    shared_external_reshape: bool = False,
    unrelated_source_consumer: bool = False,
    direct_external: bool = False,
) -> ModelIR:
    numpy_axis_dtype = np.int64 if axis_dtype == "INT64" else np.int32
    model_ir = ModelIR("indexed_unary_split_concat")
    model_ir.inputs = ["x_nhwc", "ext_nhwc"]
    model_ir.outputs = ["cat_nchw" if public_concat else "y_nchw"]
    model_ir.tensors = {
        "x_nhwc": _tensor(
            "x_nhwc",
            _SOURCE_NHWC,
            signature=_signature(_SOURCE_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT64",
            data=np.asarray([0, 3, 1, 2], dtype=np.int64),
        ),
        "x_nchw": _tensor(
            "x_nchw",
            _SOURCE_NCHW,
            signature=_nchw_signature(_SOURCE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "logits_nchw": _tensor(
            "logits_nchw",
            _SOURCE_NCHW,
            signature=_nchw_signature(_SOURCE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "axis": _tensor(
            "axis",
            [1],
            dtype=axis_dtype,
            data=np.asarray([1], dtype=numpy_axis_dtype),
        ),
        "s0": _tensor(
            "s0",
            _BRANCH_NCHW,
            signature=_nchw_signature(_BRANCH_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "s1": _tensor(
            "s1",
            _BRANCH_NCHW,
            signature=_nchw_signature(_BRANCH_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "u0": _tensor(
            "u0",
            _BRANCH_NCHW,
            signature=_nchw_signature(_BRANCH_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "u1": _tensor(
            "u1",
            _BRANCH_NCHW,
            signature=_nchw_signature(_BRANCH_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "ext_nhwc": _tensor(
            "ext_nhwc",
            _BRANCH_NHWC,
            signature=_signature(_BRANCH_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "cat_nchw": _tensor(
            "cat_nchw",
            _CONCAT_NCHW,
            signature=_nchw_signature(_CONCAT_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
    }
    operators = []
    external_name = "ext_nhwc"
    if not direct_external:
        model_ir.tensors["ext_shape"] = _tensor(
            "ext_shape",
            [4],
            dtype="INT32",
            data=np.asarray(_BRANCH_NCHW, dtype=np.int32),
        )
        model_ir.tensors["ext_nchw"] = _tensor(
            "ext_nchw",
            _BRANCH_NCHW,
            signature=_nchw_signature(_BRANCH_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
        operators.append(
            OperatorIR(
                "RESHAPE",
                ["ext_nhwc", "ext_shape"],
                ["ext_nchw"],
                options={"newShape": list(_BRANCH_NCHW)},
            )
        )
        external_name = "ext_nchw"
    operators.extend(
        [
            OperatorIR("TRANSPOSE", ["x_nhwc", "perm"], ["x_nchw"]),
            OperatorIR("LOGISTIC", ["x_nchw"], ["logits_nchw"]),
            OperatorIR(
                "SPLIT",
                ["axis", "logits_nchw"],
                ["s0", "s1"],
                options={"numSplits": 2},
            ),
            OperatorIR("RELU_0_TO_1", ["s0"], ["u0"]),
            OperatorIR("TANH", ["s1"], ["u1"]),
            OperatorIR(
                "CONCATENATION",
                [external_name, "u0", "u1"],
                ["cat_nchw"],
                options={"axis": 1},
            ),
        ]
    )
    if not public_concat:
        model_ir.tensors["y_nchw"] = _tensor(
            "y_nchw",
            _CONCAT_NCHW,
            signature=_nchw_signature(_CONCAT_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
        operators.append(OperatorIR("RELU", ["cat_nchw"], ["y_nchw"]))
    if shared_axis:
        model_ir.inputs.append("other_nchw")
        model_ir.tensors["other_nchw"] = _tensor(
            "other_nchw",
            _SOURCE_NCHW,
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
        model_ir.tensors["other0"] = _tensor("other0", _BRANCH_NCHW)
        model_ir.tensors["other1"] = _tensor("other1", _BRANCH_NCHW)
        operators.append(
            OperatorIR(
                "SPLIT",
                ["axis", "other_nchw"],
                ["other0", "other1"],
                options={"numSplits": 2},
            )
        )
    if shared_external_reshape:
        model_ir.tensors["ext_side"] = _tensor("ext_side", _BRANCH_NCHW)
        operators.append(OperatorIR("RELU", ["ext_nchw"], ["ext_side"]))
        model_ir.outputs.append("ext_side")
    if unrelated_source_consumer:
        model_ir.tensors["source_side"] = _tensor("source_side", _SOURCE_NHWC)
        operators.append(OperatorIR("RELU", ["x_nhwc"], ["source_side"]))
        model_ir.outputs.append("source_side")
    model_ir.operators = operators
    return model_ir


def _feeds(model_ir: ModelIR) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    feeds = {
        "x_nhwc": rng.normal(size=_SOURCE_NHWC).astype(np.float32),
        "ext_nhwc": rng.normal(size=_BRANCH_NHWC).astype(np.float32),
    }
    if "other_nchw" in model_ir.inputs:
        feeds["other_nchw"] = rng.normal(size=_SOURCE_NCHW).astype(np.float32)
    return feeds


def _evaluate(model_ir: ModelIR, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = {name: np.asarray(value) for name, value in feeds.items()}
    values.update(
        {
            name: np.asarray(tensor.data)
            for name, tensor in model_ir.tensors.items()
            if tensor.data is not None
        }
    )
    for operator in model_ir.operators:
        op_type = str(operator.op_type).upper()
        inputs = [values[name] for name in operator.inputs]
        if op_type == "RESHAPE":
            result = np.reshape(inputs[0], model_ir.tensors[operator.outputs[0]].shape)
        elif op_type == "TRANSPOSE":
            result = np.transpose(inputs[0], tuple(int(value) for value in inputs[1]))
        elif op_type == "LOGISTIC":
            result = 1.0 / (1.0 + np.exp(-inputs[0]))
        elif op_type == "RELU_0_TO_1":
            result = np.clip(inputs[0], 0.0, 1.0)
        elif op_type == "RELU":
            result = np.maximum(inputs[0], 0.0)
        elif op_type == "TANH":
            result = np.tanh(inputs[0])
        elif op_type == "SPLIT":
            axis = int(np.asarray(inputs[0]).reshape(-1)[0])
            outputs = np.split(inputs[1], len(operator.outputs), axis=axis)
            for name, value in zip(operator.outputs, outputs):
                values[name] = value
            continue
        elif op_type == "CONCATENATION":
            result = np.concatenate(inputs, axis=int(operator.options["axis"]))
        else:
            raise AssertionError(op_type)
        values[operator.outputs[0]] = result
    return values


def _fingerprint(model_ir: ModelIR) -> tuple[Any, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                repr(operator.options),
            )
            for operator in model_ir.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or []),
                tensor.logical_layout,
                tensor.physical_layout,
                repr(tensor.quantization),
                None
                if tensor.data is None
                else (
                    str(np.asarray(tensor.data).dtype),
                    tuple(np.asarray(tensor.data).shape),
                    tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                ),
            )
            for name, tensor in model_ir.tensors.items()
        ),
    )


def _assert_index_current(
    graph_index: ModelIRGraphIndex,
    model_ir: ModelIR,
) -> None:
    rebuilt = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == rebuilt.producers
    assert graph_index.consumers == rebuilt.consumers
    assert graph_index.duplicate_producers == rebuilt.duplicate_producers


@pytest.mark.parametrize("axis_dtype", ["INT32", "INT64"])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("public_concat", [False, True])
def test_indexed_unary_split_concat_preserves_semantics(
    axis_dtype: str,
    dynamic: bool,
    public_concat: bool,
) -> None:
    model_ir = _make_model(
        axis_dtype=axis_dtype,
        dynamic=dynamic,
        public_concat=public_concat,
    )
    feeds = _feeds(model_ir)
    before = _evaluate(model_ir, feeds)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_unary_split_concat_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    after = _evaluate(model_ir, feeds)
    output_name = "cat_nchw" if public_concat else "y_nchw"
    np.testing.assert_allclose(
        after[output_name], before[output_name], rtol=0.0, atol=0.0
    )
    assert stats == {"optimized_transpose_unary_split_concat_single_post_nchw": 1}
    split = next(
        operator for operator in model_ir.operators if operator.op_type == "SPLIT"
    )
    concat = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    )
    assert np.asarray(model_ir.tensors[split.inputs[0]].data).reshape(-1).tolist() == [
        3
    ]
    assert concat.options["axis"] == 3
    assert concat.inputs[0] == "ext_nhwc"
    assert model_ir.tensors["cat_nchw"].shape == _CONCAT_NCHW
    assert model_ir.tensors["cat_nchw"].physical_layout == LOGICAL_LAYOUT_NCHW
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_unary_split_concat_supports_proven_direct_nhwc_external() -> None:
    model_ir = _make_model(direct_external=True)
    feeds = _feeds(model_ir)

    stats = optimize_transpose_unary_split_concat_single_post_nchw(model_ir)

    values = _evaluate(model_ir, feeds)
    expected = np.transpose(
        np.concatenate(
            [
                feeds["ext_nhwc"],
                np.clip(1.0 / (1.0 + np.exp(-feeds["x_nhwc"][:, :, :, :1])), 0.0, 1.0),
                np.tanh(1.0 / (1.0 + np.exp(-feeds["x_nhwc"][:, :, :, 1:]))),
            ],
            axis=3,
        ),
        (0, 3, 1, 2),
    )
    assert stats["optimized_transpose_unary_split_concat_single_post_nchw"] == 1
    np.testing.assert_allclose(values["cat_nchw"], expected, rtol=0.0, atol=0.0)


def test_indexed_unary_split_concat_clones_shared_axis_and_preserves_side_uses() -> (
    None
):
    model_ir = _make_model(
        axis_dtype="INT64",
        shared_axis=True,
        shared_external_reshape=True,
        unrelated_source_consumer=True,
    )
    side_reshape = model_ir.operators[0]
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_unary_split_concat_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats["optimized_transpose_unary_split_concat_single_post_nchw"] == 1
    main_split = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "SPLIT" and operator.inputs[1] == "logits_nchw"
    )
    other_split = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "SPLIT" and operator.inputs[1] == "other_nchw"
    )
    assert main_split.inputs[0] != "axis"
    assert other_split.inputs[0] == "axis"
    assert np.asarray(model_ir.tensors["axis"].data).tolist() == [1]
    assert np.asarray(model_ir.tensors[main_split.inputs[0]].data).dtype == np.int64
    assert side_reshape in model_ir.operators
    assert any(
        operator.inputs == ["ext_nchw"] and operator.outputs == ["ext_side"]
        for operator in model_ir.operators
    )
    assert any(
        operator.inputs == ["x_nhwc"] and operator.outputs == ["source_side"]
        for operator in model_ir.operators
    )
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_unary_split_concat_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    original = _fingerprint(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    concat = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    )

    assert optimize_transpose_unary_split_concat_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=0,
    ) == {"optimized_transpose_unary_split_concat_single_post_nchw": 0}
    assert _fingerprint(model_ir) == original
    assert optimize_transpose_unary_split_concat_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        candidate=model_ir.operators[1],
    ) == {"optimized_transpose_unary_split_concat_single_post_nchw": 0}
    assert _fingerprint(model_ir) == original
    assert optimize_transpose_unary_split_concat_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        candidate=concat,
    ) == {"optimized_transpose_unary_split_concat_single_post_nchw": 1}
    after = _fingerprint(model_ir)
    assert optimize_transpose_unary_split_concat_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {"optimized_transpose_unary_split_concat_single_post_nchw": 0}
    assert _fingerprint(model_ir) == after


def _mutate_unsafe(model_ir: ModelIR, case: str) -> None:
    if case == "wrong_permutation":
        model_ir.tensors["perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int64)
    elif case == "wrong_axis":
        model_ir.tensors["axis"].data = np.asarray([2], dtype=np.int32)
    elif case == "variable_axis":
        model_ir.tensors["axis"].is_variable = True
    elif case == "per_axis_quantization":
        model_ir.tensors["s0"].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3, 0.4],
            zero_point=[0, 0, 0, 0],
            quantized_dimension=3,
        )
    elif case == "duplicate_branch":
        concat = next(
            operator
            for operator in model_ir.operators
            if operator.op_type == "CONCATENATION"
        )
        concat.inputs = [concat.inputs[0], "u0", "u0"]
    elif case == "second_external":
        model_ir.inputs.append("ext2_nhwc")
        model_ir.tensors["ext2_nhwc"] = _tensor(
            "ext2_nhwc",
            _BRANCH_NHWC,
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        )
        concat = next(
            operator
            for operator in model_ir.operators
            if operator.op_type == "CONCATENATION"
        )
        concat.inputs.append("ext2_nhwc")
    elif case == "shared_pre_output":
        model_ir.tensors["pre_side"] = _tensor("pre_side", _SOURCE_NCHW)
        model_ir.operators.append(OperatorIR("RELU", ["x_nchw"], ["pre_side"]))
    elif case == "shared_pre_unary":
        model_ir.tensors["logit_side"] = _tensor("logit_side", _SOURCE_NCHW)
        model_ir.operators.append(OperatorIR("RELU", ["logits_nchw"], ["logit_side"]))
    elif case == "shared_split_branch":
        model_ir.tensors["split_side"] = _tensor("split_side", _BRANCH_NCHW)
        model_ir.operators.append(OperatorIR("RELU", ["s0"], ["split_side"]))
    elif case == "shared_branch_unary":
        model_ir.tensors["unary_side"] = _tensor("unary_side", _BRANCH_NCHW)
        model_ir.operators.append(OperatorIR("RELU", ["u0"], ["unary_side"]))
    elif case == "non_singleton_reshape":
        model_ir.tensors["ext_nhwc"].shape = [1, _H, _W, 2]
        model_ir.tensors["ext_nhwc"].shape_signature = [1, _H, _W, 2]
        model_ir.tensors["ext_nchw"].shape = [1, 2, _H, _W]
        model_ir.tensors["ext_nchw"].shape_signature = [1, 2, _H, _W]
    elif case == "explicit_nchw_source":
        model_ir.tensors["ext_nhwc"].logical_layout = LOGICAL_LAYOUT_NCHW
        model_ir.tensors["ext_nhwc"].physical_layout = LOGICAL_LAYOUT_NCHW
    elif case == "malformed_branch_unary":
        model_ir.tensors["extra"] = _tensor(
            "extra", [1], data=np.asarray([0.0], dtype=np.float32)
        )
        unary = next(
            operator for operator in model_ir.operators if operator.outputs == ["u0"]
        )
        unary.inputs.append("extra")
    elif case == "public_intermediate":
        model_ir.outputs = ["s0"]
    elif case == "consumer_before_producer":
        concat_index = next(
            index
            for index, operator in enumerate(model_ir.operators)
            if operator.op_type == "CONCATENATION"
        )
        unary_index = next(
            index
            for index, operator in enumerate(model_ir.operators)
            if operator.outputs == ["u1"]
        )
        model_ir.operators[concat_index], model_ir.operators[unary_index] = (
            model_ir.operators[unary_index],
            model_ir.operators[concat_index],
        )
    elif case == "duplicate_producer":
        model_ir.operators.append(OperatorIR("RELU", ["s1"], ["s0"]))
    elif case == "missing_concat_tensor":
        del model_ir.tensors["cat_nchw"]
    elif case == "concat_shape_mismatch":
        model_ir.tensors["cat_nchw"].shape = [1, 4, _H, _W]
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_permutation",
        "wrong_axis",
        "variable_axis",
        "per_axis_quantization",
        "duplicate_branch",
        "second_external",
        "shared_pre_output",
        "shared_pre_unary",
        "shared_split_branch",
        "shared_branch_unary",
        "non_singleton_reshape",
        "explicit_nchw_source",
        "malformed_branch_unary",
        "public_intermediate",
        "consumer_before_producer",
        "duplicate_producer",
        "missing_concat_tensor",
        "concat_shape_mismatch",
    ],
)
def test_indexed_unary_split_concat_rejects_unsafe_candidate_transactionally(
    case: str,
) -> None:
    model_ir = _make_model()
    _mutate_unsafe(model_ir, case)
    before = copy.deepcopy(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_unary_split_concat_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_transpose_unary_split_concat_single_post_nchw": 0}
    assert _fingerprint(model_ir) == _fingerprint(before)
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)
