from __future__ import annotations

import numpy as np
import pytest
import onnx2tf.tflite_builder.passes.quantization_cleanup as cleanup_module

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.quantization_cleanup import (
    _optimize_concat_pre_quantize_dequantize,
    _optimize_transpose_dequantize_mean_quantize_bridges,
    _quantized_tensors_share_exact_grid,
    _sanitize_terminal_transpose_before_dequantize,
    run_terminal_quantize_dequantize_cleanup,
)


def _tensor(
    name: str,
    dtype: str,
    *,
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=[1, 3],
        shape_signature=[1, 3],
        quantization=quantization,
    )


def _grid(
    *,
    scale: float = 0.125,
    zero_point: int = -3,
    dimension: int = 0,
) -> QuantParamIR:
    return QuantParamIR(
        scale=[scale],
        zero_point=[zero_point],
        quantized_dimension=dimension,
    )


def _terminal_qdq_model() -> ModelIR:
    model_ir = ModelIR("terminal_quantize_dequantize")
    model_ir.inputs = ["q_source"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "q_source": _tensor("q_source", "INT8", quantization=_grid()),
        "float_input": _tensor("float_input", "FLOAT32"),
        "terminal_q": _tensor("terminal_q", "INT8", quantization=_grid()),
        "y": _tensor("y", "FLOAT32"),
    }
    model_ir.operators = [
        OperatorIR("DEQUANTIZE", ["q_source"], ["float_input"]),
        OperatorIR("QUANTIZE", ["float_input"], ["terminal_q"]),
        OperatorIR("DEQUANTIZE", ["terminal_q"], ["y"]),
    ]
    return model_ir


def _concat_qdq_model(*, branches: int = 2) -> ModelIR:
    model_ir = ModelIR("concat_pre_quantize_dequantize")
    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        source = f"{prefix}_source"
        float_name = f"{prefix}_float"
        quantized = f"{prefix}_quantized"
        dequantized = f"{prefix}_dequantized"
        other = f"{prefix}_other"
        output = f"{prefix}_output"
        model_ir.inputs.extend([source, other])
        model_ir.outputs.append(output)
        model_ir.tensors.update(
            {
                source: _tensor(source, "INT8", quantization=_grid()),
                float_name: _tensor(float_name, "FLOAT32"),
                quantized: _tensor(
                    quantized,
                    "INT8",
                    quantization=_grid(),
                ),
                dequantized: _tensor(dequantized, "FLOAT32"),
                other: _tensor(other, "FLOAT32"),
                output: _tensor(output, "FLOAT32"),
            }
        )
        model_ir.operators.extend(
            [
                OperatorIR("DEQUANTIZE", [source], [float_name]),
                OperatorIR("QUANTIZE", [float_name], [quantized]),
                OperatorIR("DEQUANTIZE", [quantized], [dequantized]),
                OperatorIR(
                    "CONCATENATION",
                    [dequantized, other],
                    [output],
                    options={"axis": 1},
                ),
            ]
        )
    return model_ir


def _terminal_transpose_dequantize_model(
    *,
    pattern: str = "transpose_dequantize",
    branches: int = 1,
) -> ModelIR:
    model_ir = ModelIR("terminal_transpose_before_dequantize")
    for branch_index in range(int(branches)):
        prefix = f"terminal{branch_index}"
        quantized = f"{prefix}_quantized"
        permutation = f"{prefix}_permutation"
        intermediate = f"{prefix}_intermediate"
        output = f"{prefix}_output"
        model_ir.inputs.append(quantized)
        model_ir.outputs.append(output)
        model_ir.tensors[quantized] = TensorIR(
            name=quantized,
            dtype="INT8",
            shape=[1, 3],
            shape_signature=[-1, 3],
            quantization=_grid(),
        )
        model_ir.tensors[permutation] = TensorIR(
            name=permutation,
            dtype="INT32",
            shape=[2],
            shape_signature=[2],
            data=np.asarray([1, 0], dtype=np.int32),
        )
        if pattern == "transpose_dequantize":
            model_ir.tensors[intermediate] = TensorIR(
                name=intermediate,
                dtype="INT8",
                shape=[3, 1],
                shape_signature=[3, -1],
                quantization=_grid(),
            )
            model_ir.tensors[output] = TensorIR(
                name=output,
                dtype="FLOAT32",
                shape=[3, 1],
                shape_signature=[3, -1],
            )
            model_ir.operators.extend(
                [
                    OperatorIR(
                        "TRANSPOSE",
                        [quantized, permutation],
                        [intermediate],
                    ),
                    OperatorIR(
                        "DEQUANTIZE",
                        [intermediate],
                        [output],
                    ),
                ]
            )
        else:
            model_ir.tensors[intermediate] = TensorIR(
                name=intermediate,
                dtype="FLOAT32",
                shape=[1, 3],
                shape_signature=[-1, 3],
            )
            model_ir.tensors[output] = TensorIR(
                name=output,
                dtype="FLOAT32",
                shape=[3, 1],
                shape_signature=[3, -1],
            )
            model_ir.operators.extend(
                [
                    OperatorIR(
                        "DEQUANTIZE",
                        [quantized],
                        [intermediate],
                    ),
                    OperatorIR(
                        "TRANSPOSE",
                        [intermediate, permutation],
                        [output],
                    ),
                ]
            )
    return model_ir


def _transpose_dequantize_mean_quantize_model(
    *,
    branches: int = 1,
) -> ModelIR:
    model_ir = ModelIR("transpose_dequantize_mean_quantize")
    for branch_index in range(int(branches)):
        prefix = f"mean_bridge{branch_index}"
        source = f"{prefix}_source"
        permutation = f"{prefix}_permutation"
        transposed = f"{prefix}_transposed"
        dequantized = f"{prefix}_dequantized"
        axes = f"{prefix}_axes"
        reduced = f"{prefix}_reduced"
        output = f"{prefix}_output"
        model_ir.inputs.append(source)
        model_ir.outputs.append(output)
        model_ir.tensors.update(
            {
                source: TensorIR(
                    name=source,
                    dtype="INT8",
                    shape=[1, 2, 3, 4],
                    shape_signature=[-1, 2, 3, 4],
                    quantization=_grid(),
                ),
                permutation: TensorIR(
                    name=permutation,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray([0, 2, 3, 1], dtype=np.int32),
                ),
                transposed: TensorIR(
                    name=transposed,
                    dtype="INT8",
                    shape=[1, 3, 4, 2],
                    shape_signature=[-1, 3, 4, 2],
                    quantization=_grid(),
                ),
                dequantized: TensorIR(
                    name=dequantized,
                    dtype="FLOAT32",
                    shape=[1, 3, 4, 2],
                    shape_signature=[-1, 3, 4, 2],
                ),
                axes: TensorIR(
                    name=axes,
                    dtype="INT64",
                    shape=[2],
                    shape_signature=[2],
                    data=np.asarray([-3, -2], dtype=np.int64),
                ),
                reduced: TensorIR(
                    name=reduced,
                    dtype="FLOAT32",
                    shape=[1, 1, 1, 2],
                    shape_signature=[-1, 1, 1, 2],
                ),
                output: TensorIR(
                    name=output,
                    dtype="INT8",
                    shape=[1, 1, 1, 2],
                    shape_signature=[-1, 1, 1, 2],
                    quantization=_grid(),
                ),
            }
        )
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    [source, permutation],
                    [transposed],
                ),
                OperatorIR("DEQUANTIZE", [transposed], [dequantized]),
                OperatorIR(
                    "MEAN",
                    [dequantized, axes],
                    [reduced],
                    options={"keepDims": True},
                ),
                OperatorIR("QUANTIZE", [reduced], [output]),
            ]
        )
    return model_ir


def test_exact_quantization_grid_compares_all_semantic_fields() -> None:
    model_ir = _terminal_qdq_model()

    assert _quantized_tensors_share_exact_grid(
        model_ir,
        "q_source",
        "terminal_q",
    )
    model_ir.tensors["terminal_q"].quantization = _grid(scale=0.25)
    assert not _quantized_tensors_share_exact_grid(
        model_ir,
        "q_source",
        "terminal_q",
    )
    model_ir.tensors["terminal_q"].quantization = _grid(zero_point=4)
    assert not _quantized_tensors_share_exact_grid(
        model_ir,
        "q_source",
        "terminal_q",
    )
    model_ir.tensors["terminal_q"].quantization = _grid(dimension=1)
    assert not _quantized_tensors_share_exact_grid(
        model_ir,
        "q_source",
        "terminal_q",
    )
    model_ir.tensors["terminal_q"].quantization = _grid()
    model_ir.tensors["terminal_q"].dtype = "UINT8"
    assert not _quantized_tensors_share_exact_grid(
        model_ir,
        "q_source",
        "terminal_q",
    )


def test_concat_qdq_cleanup_rewrites_multiple_matches_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _concat_qdq_model()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_concat_pre_quantize_dequantize(model_ir)

    assert stats == {"bypassed_concat_pre_quantize_dequantize": 2}
    assert refresh_count == 1
    concat_inputs = [
        [str(name) for name in operator.inputs]
        for operator in model_ir.operators
        if str(operator.op_type) == "CONCATENATION"
    ]
    assert concat_inputs == [
        ["branch0_float", "branch0_other"],
        ["branch1_float", "branch1_other"],
    ]


def test_concat_qdq_cleanup_keeps_supplied_index_current() -> None:
    model_ir = _concat_qdq_model()
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _optimize_concat_pre_quantize_dequantize(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {"bypassed_concat_pre_quantize_dequantize": 2}
    fresh_index = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh_index.producers
    assert graph_index.consumers == fresh_index.consumers
    assert graph_index.duplicate_producers == fresh_index.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh_index._operator_indices_by_id
    assert (
        graph_index._operator_indices_by_type == fresh_index._operator_indices_by_type
    )


@pytest.mark.parametrize(
    "case",
    [
        "different_grid",
        "different_dtype",
        "quantized_fanout",
        "public_quantized",
        "public_dequantized",
        "shape_mismatch",
        "non_dequantize_source",
    ],
)
def test_concat_qdq_cleanup_preserves_rounding_and_boundary_guards(
    case: str,
) -> None:
    model_ir = _concat_qdq_model(branches=1)
    if case == "different_grid":
        model_ir.tensors["branch0_quantized"].quantization = _grid(scale=0.25)
    elif case == "different_dtype":
        model_ir.tensors["branch0_quantized"].dtype = "UINT8"
    elif case == "quantized_fanout":
        model_ir.tensors["branch0_side"] = _tensor(
            "branch0_side",
            "FLOAT32",
        )
        model_ir.outputs.append("branch0_side")
        model_ir.operators.append(
            OperatorIR(
                "DEQUANTIZE",
                ["branch0_quantized"],
                ["branch0_side"],
            )
        )
    elif case == "public_quantized":
        model_ir.outputs.append("branch0_quantized")
    elif case == "public_dequantized":
        model_ir.outputs.append("branch0_dequantized")
    elif case == "shape_mismatch":
        model_ir.tensors["branch0_dequantized"].shape = [1, 4]
    elif case == "non_dequantize_source":
        model_ir.operators[0] = OperatorIR(
            "IDENTITY",
            ["branch0_source"],
            ["branch0_float"],
        )

    stats = _optimize_concat_pre_quantize_dequantize(model_ir)

    assert stats == {"bypassed_concat_pre_quantize_dequantize": 0}
    concat = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "CONCATENATION"
    )
    assert [str(name) for name in concat.inputs] == [
        "branch0_dequantized",
        "branch0_other",
    ]


def test_concat_qdq_cleanup_skips_index_without_concat_and_still_prunes(
    monkeypatch,
) -> None:
    model_ir = ModelIR("concat_qdq_without_concat")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", "FLOAT32"),
        "y": _tensor("y", "FLOAT32"),
        "unused": _tensor("unused", "FLOAT32"),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(cleanup_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_concat_pre_quantize_dequantize(model_ir)

    assert stats == {"bypassed_concat_pre_quantize_dequantize": 0}
    assert set(model_ir.tensors) == {"x", "y"}


def test_terminal_transpose_dequantize_sanitizer_handles_multiple_matches_once(
    monkeypatch,
) -> None:
    model_ir = _terminal_transpose_dequantize_model(branches=2)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _sanitize_terminal_transpose_before_dequantize(model_ir)

    assert stats == {
        "sanitized_terminal_transpose_before_dequantize": 2,
        "removed_terminal_dequantize_transpose": 2,
    }
    assert refresh_count == 1
    assert [str(operator.op_type) for operator in model_ir.operators] == [
        "DEQUANTIZE",
        "DEQUANTIZE",
    ]
    assert [
        (
            [str(name) for name in operator.inputs],
            [str(name) for name in operator.outputs],
        )
        for operator in model_ir.operators
    ] == [
        (["terminal0_quantized"], ["terminal0_output"]),
        (["terminal1_quantized"], ["terminal1_output"]),
    ]


def test_terminal_dequantize_transpose_removal_keeps_index_and_output_name() -> None:
    model_ir = _terminal_transpose_dequantize_model(pattern="dequantize_transpose")
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _sanitize_terminal_transpose_before_dequantize(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {
        "sanitized_terminal_transpose_before_dequantize": 0,
        "removed_terminal_dequantize_transpose": 1,
    }
    assert len(model_ir.operators) == 1
    assert [str(name) for name in model_ir.operators[0].inputs] == [
        "terminal0_quantized"
    ]
    assert [str(name) for name in model_ir.operators[0].outputs] == ["terminal0_output"]
    assert model_ir.tensors["terminal0_output"].shape == [1, 3]
    fresh_index = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh_index.producers
    assert graph_index.consumers == fresh_index.consumers
    assert graph_index.duplicate_producers == fresh_index.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh_index._operator_indices_by_id
    assert (
        graph_index._operator_indices_by_type == fresh_index._operator_indices_by_type
    )


@pytest.mark.parametrize(
    "case",
    [
        "nonterminal_output",
        "consumed_output",
        "public_quantized_input",
        "shared_transposed_input",
        "per_channel_quantization",
        "invalid_permutation",
        "missing_output_tensor",
    ],
)
def test_terminal_transpose_dequantize_sanitizer_preserves_guards(
    case: str,
) -> None:
    model_ir = _terminal_transpose_dequantize_model()
    if case == "nonterminal_output":
        model_ir.outputs = []
    elif case == "consumed_output":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["terminal0_output"], ["side"])
        )
    elif case == "public_quantized_input":
        model_ir.outputs.append("terminal0_quantized")
    elif case == "shared_transposed_input":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("DEQUANTIZE", ["terminal0_intermediate"], ["side"])
        )
    elif case == "per_channel_quantization":
        per_channel = QuantParamIR(
            scale=[0.1, 0.2, 0.3],
            zero_point=[0, 0, 0],
            quantized_dimension=1,
        )
        model_ir.tensors["terminal0_quantized"].quantization = per_channel
        model_ir.tensors["terminal0_intermediate"].quantization = per_channel
    elif case == "invalid_permutation":
        model_ir.tensors["terminal0_permutation"].data = np.asarray(
            [0, 0],
            dtype=np.int32,
        )
    elif case == "missing_output_tensor":
        del model_ir.tensors["terminal0_output"]

    stats = _sanitize_terminal_transpose_before_dequantize(model_ir)

    assert stats == {
        "sanitized_terminal_transpose_before_dequantize": 0,
        "removed_terminal_dequantize_transpose": 0,
    }
    assert [str(operator.op_type) for operator in model_ir.operators[:2]] == [
        "TRANSPOSE",
        "DEQUANTIZE",
    ]
    assert [str(name) for name in model_ir.operators[0].inputs] == [
        "terminal0_quantized",
        "terminal0_permutation",
    ]
    assert [str(name) for name in model_ir.operators[1].inputs] == [
        "terminal0_intermediate"
    ]


def test_terminal_transpose_dequantize_sanitizer_skips_index_without_pair(
    monkeypatch,
) -> None:
    model_ir = ModelIR("terminal_without_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", "INT8", quantization=_grid()),
        "y": _tensor("y", "FLOAT32"),
        "unused": _tensor("unused", "FLOAT32"),
    }
    model_ir.operators = [OperatorIR("DEQUANTIZE", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(cleanup_module, "ModelIRGraphIndex", unexpected_index)

    stats = _sanitize_terminal_transpose_before_dequantize(model_ir)

    assert stats == {
        "sanitized_terminal_transpose_before_dequantize": 0,
        "removed_terminal_dequantize_transpose": 0,
    }
    assert set(model_ir.tensors) == {"x", "y"}


def test_transpose_dq_mean_q_bridge_moves_multiple_matches_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _transpose_dequantize_mean_quantize_model(branches=2)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_dequantize_mean_quantize_bridges(model_ir)

    assert stats == {
        "moved_transpose_dequantize_mean_quantize_bridges": 2,
    }
    assert refresh_count == 1
    assert [str(operator.op_type) for operator in model_ir.operators] == [
        "DEQUANTIZE",
        "MEAN",
        "TRANSPOSE",
        "QUANTIZE",
        "DEQUANTIZE",
        "MEAN",
        "TRANSPOSE",
        "QUANTIZE",
    ]
    for branch_index in range(2):
        prefix = f"mean_bridge{branch_index}"
        dequantize = next(
            operator
            for operator in model_ir.operators
            if str(operator.outputs[0]) == f"{prefix}_dequantized"
        )
        mean = next(
            operator
            for operator in model_ir.operators
            if str(operator.outputs[0]) == f"{prefix}_reduced"
        )
        quantize = next(
            operator
            for operator in model_ir.operators
            if str(operator.outputs[0]) == f"{prefix}_output"
        )
        assert [str(name) for name in dequantize.inputs] == [f"{prefix}_source"]
        assert model_ir.tensors[f"{prefix}_axes"].dtype == "INT32"
        assert np.asarray(model_ir.tensors[f"{prefix}_axes"].data).tolist() == [2, 3]
        assert model_ir.tensors[f"{prefix}_dequantized"].shape_signature == [
            -1,
            2,
            3,
            4,
        ]
        assert model_ir.tensors[f"{prefix}_reduced"].shape == [1, 2, 1, 1]
        bridge_name = str(quantize.inputs[0])
        bridge = model_ir.tensors[bridge_name]
        assert bridge.shape == [1, 1, 1, 2]
        bridge_transpose = next(
            operator
            for operator in model_ir.operators
            if str(operator.op_type) == "TRANSPOSE"
            and str(operator.outputs[0]) == bridge_name
        )
        assert [str(name) for name in bridge_transpose.inputs[:1]] == [
            f"{prefix}_reduced"
        ]
        assert mean.options["keepDims"] is True


def test_transpose_dq_mean_q_bridge_keeps_supplied_index_current() -> None:
    model_ir = _transpose_dequantize_mean_quantize_model()
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _optimize_transpose_dequantize_mean_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {
        "moved_transpose_dequantize_mean_quantize_bridges": 1,
    }
    fresh_index = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh_index.producers
    assert graph_index.consumers == fresh_index.consumers
    assert graph_index.duplicate_producers == fresh_index.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh_index._operator_indices_by_id
    assert (
        graph_index._operator_indices_by_type == fresh_index._operator_indices_by_type
    )


@pytest.mark.parametrize(
    "case",
    [
        "public_pre_output",
        "pre_fanout",
        "public_dequantized",
        "dequantized_fanout",
        "no_keepdims",
        "shared_axes",
        "public_mean_output",
        "mean_fanout",
        "invalid_axis",
        "missing_tensor",
        "invalid_permutation",
    ],
)
def test_transpose_dq_mean_q_bridge_preserves_guards_transactionally(
    case: str,
) -> None:
    model_ir = _transpose_dequantize_mean_quantize_model()
    if case == "public_pre_output":
        model_ir.outputs.append("mean_bridge0_transposed")
    elif case == "pre_fanout":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("DEQUANTIZE", ["mean_bridge0_transposed"], ["side"])
        )
    elif case == "public_dequantized":
        model_ir.outputs.append("mean_bridge0_dequantized")
    elif case == "dequantized_fanout":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["mean_bridge0_dequantized"], ["side"])
        )
    elif case == "no_keepdims":
        model_ir.operators[2].options["keepDims"] = False
    elif case == "shared_axes":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR(
                "MEAN",
                ["mean_bridge0_dequantized", "mean_bridge0_axes"],
                ["side"],
                options={"keepDims": True},
            )
        )
    elif case == "public_mean_output":
        model_ir.outputs.append("mean_bridge0_reduced")
    elif case == "mean_fanout":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["mean_bridge0_reduced"], ["side"])
        )
    elif case == "invalid_axis":
        model_ir.tensors["mean_bridge0_axes"].data = np.asarray(
            [4],
            dtype=np.int64,
        )
    elif case == "missing_tensor":
        del model_ir.tensors["mean_bridge0_reduced"]
    elif case == "invalid_permutation":
        model_ir.tensors["mean_bridge0_permutation"].data = np.asarray(
            [0, 2, 2, 1],
            dtype=np.int32,
        )

    before_operators = [
        (
            str(operator.op_type),
            [str(name) for name in operator.inputs],
            [str(name) for name in operator.outputs],
        )
        for operator in model_ir.operators
    ]
    before_tensor_names = set(model_ir.tensors)
    before_axes = np.asarray(model_ir.tensors["mean_bridge0_axes"].data).tolist()
    before_dequantized_shape = list(model_ir.tensors["mean_bridge0_dequantized"].shape)
    before_mean_shape = (
        list(model_ir.tensors["mean_bridge0_reduced"].shape)
        if "mean_bridge0_reduced" in model_ir.tensors
        else None
    )

    stats = _optimize_transpose_dequantize_mean_quantize_bridges(model_ir)

    assert stats == {
        "moved_transpose_dequantize_mean_quantize_bridges": 0,
    }
    assert [
        (
            str(operator.op_type),
            [str(name) for name in operator.inputs],
            [str(name) for name in operator.outputs],
        )
        for operator in model_ir.operators
    ] == before_operators
    assert set(model_ir.tensors) == before_tensor_names
    assert np.asarray(model_ir.tensors["mean_bridge0_axes"].data).tolist() == (
        before_axes
    )
    assert model_ir.tensors["mean_bridge0_dequantized"].shape == (
        before_dequantized_shape
    )
    if before_mean_shape is not None:
        assert model_ir.tensors["mean_bridge0_reduced"].shape == before_mean_shape


def test_transpose_dq_mean_q_bridge_skips_index_without_required_types(
    monkeypatch,
) -> None:
    model_ir = ModelIR("mean_bridge_without_quantize")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", "INT8", quantization=_grid()),
        "y": _tensor("y", "FLOAT32"),
        "unused": _tensor("unused", "FLOAT32"),
    }
    model_ir.operators = [OperatorIR("DEQUANTIZE", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(cleanup_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_transpose_dequantize_mean_quantize_bridges(model_ir)

    assert stats == {
        "moved_transpose_dequantize_mean_quantize_bridges": 0,
    }
    assert set(model_ir.tensors) == {"x", "y"}


def test_terminal_qdq_cleanup_renames_exact_grid_output_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _terminal_qdq_model()
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = run_terminal_quantize_dequantize_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )

    assert stats == {"removed_terminal_quantize_dequantize_pairs": 1}
    assert refresh_count == 1
    assert [op.op_type for op in model_ir.operators] == ["DEQUANTIZE"]
    assert model_ir.operators[0].inputs == ["q_source"]
    assert model_ir.operators[0].outputs == ["y"]
    assert model_ir.outputs == ["y"]
    assert set(model_ir.tensors) == {"q_source", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert diagnostics[0]["code"] == "cleanup.terminal_quantize_dequantize"
    assert diagnostics[0]["status"] == "changed"


@pytest.mark.parametrize(
    "case",
    [
        "scale",
        "zero_point",
        "dimension",
        "dtype",
        "shared_quantized",
        "nonterminal_output",
        "consumed_output",
        "shared_float_input",
        "non_dequantize_producer",
        "public_float_input",
    ],
)
def test_terminal_qdq_cleanup_preserves_rounding_and_boundary_guards(
    case: str,
) -> None:
    model_ir = _terminal_qdq_model()
    if case == "scale":
        model_ir.tensors["terminal_q"].quantization = _grid(scale=0.25)
    elif case == "zero_point":
        model_ir.tensors["terminal_q"].quantization = _grid(zero_point=2)
    elif case == "dimension":
        model_ir.tensors["terminal_q"].quantization = _grid(dimension=1)
    elif case == "dtype":
        model_ir.tensors["terminal_q"].dtype = "UINT8"
    elif case == "shared_quantized":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("DEQUANTIZE", ["terminal_q"], ["side"]))
    elif case == "nonterminal_output":
        model_ir.tensors["out"] = _tensor("out", "FLOAT32")
        model_ir.outputs = ["out"]
        model_ir.operators.append(OperatorIR("IDENTITY", ["y"], ["out"]))
    elif case == "consumed_output":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", ["y"], ["side"]))
    elif case == "shared_float_input":
        model_ir.tensors["side"] = _tensor("side", "FLOAT32")
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", ["float_input"], ["side"]))
    elif case == "non_dequantize_producer":
        model_ir.operators[0] = OperatorIR(
            "IDENTITY",
            ["q_source"],
            ["float_input"],
        )
    elif case == "public_float_input":
        model_ir.inputs.append("float_input")

    stats = run_terminal_quantize_dequantize_cleanup(model_ir)

    assert stats == {"removed_terminal_quantize_dequantize_pairs": 0}
    assert [op.op_type for op in model_ir.operators[:3]] == [
        "DEQUANTIZE" if case != "non_dequantize_producer" else "IDENTITY",
        "QUANTIZE",
        "DEQUANTIZE",
    ]


def test_terminal_qdq_cleanup_skips_snapshot_without_quantize(monkeypatch) -> None:
    model_ir = ModelIR("no_quantize")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", "FLOAT32"),
        "y": _tensor("y", "FLOAT32"),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]
    snapshot_count = 0
    original_snapshot = ModelIRPassState.snapshot

    def counted_snapshot(state: ModelIRPassState) -> ModelIR:
        nonlocal snapshot_count
        snapshot_count += 1
        return original_snapshot(state)

    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)

    stats = run_terminal_quantize_dequantize_cleanup(model_ir)

    assert stats == {"removed_terminal_quantize_dequantize_pairs": 0}
    assert snapshot_count == 0
