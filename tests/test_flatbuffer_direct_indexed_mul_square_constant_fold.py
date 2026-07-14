from __future__ import annotations

import numpy as np
import pytest
import onnx2tf.tflite_builder.passes.constant_fold as fold_module

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.constant_fold import (
    _optimize_mul_square_anchor_constant_chains,
)


def _tensor(
    name: str,
    *,
    dtype: str = "FLOAT32",
    shape: list[int] | None = None,
    data: np.ndarray | None = None,
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    tensor_shape = list(shape if shape is not None else [1, 2])
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=tensor_shape,
        shape_signature=list(tensor_shape),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _mul_square_constant_model(*, branches: int = 2) -> ModelIR:
    model_ir = ModelIR("indexed_mul_square_constant_fold")
    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        source = f"{prefix}_source"
        pre_constant = f"{prefix}_pre_constant"
        scaled = f"{prefix}_scaled"
        square = f"{prefix}_square"
        anchor_constant = f"{prefix}_anchor_constant"
        anchored = f"{prefix}_anchored"
        scale_constant = f"{prefix}_scale_constant"
        output = f"{prefix}_output"
        model_ir.inputs.append(source)
        model_ir.outputs.append(output)
        quantization = QuantParamIR(
            scale=[0.25],
            zero_point=[0],
            quantized_dimension=0,
        )
        model_ir.tensors.update(
            {
                source: _tensor(source),
                pre_constant: _tensor(
                    pre_constant,
                    shape=[],
                    data=np.asarray(2.0 + branch_index, dtype=np.float32),
                ),
                scaled: _tensor(scaled),
                square: _tensor(square),
                anchor_constant: _tensor(
                    anchor_constant,
                    dtype="FLOAT16",
                    data=np.asarray(
                        [3.0 + branch_index, 4.0 + branch_index],
                        dtype=np.float16,
                    ),
                    quantization=quantization,
                ),
                anchored: _tensor(anchored),
                scale_constant: _tensor(
                    scale_constant,
                    shape=[],
                    data=np.asarray(0.5 - branch_index * 0.25, dtype=np.float32),
                ),
                output: _tensor(output),
            }
        )
        pre_inputs = (
            [pre_constant, source] if branch_index % 2 == 0 else [source, pre_constant]
        )
        anchor_inputs = (
            [anchor_constant, square]
            if branch_index % 2 == 0
            else [square, anchor_constant]
        )
        scale_inputs = (
            [scale_constant, anchored]
            if branch_index % 2 == 0
            else [anchored, scale_constant]
        )
        model_ir.operators.extend(
            [
                OperatorIR("MUL", pre_inputs, [scaled]),
                OperatorIR("MUL", [scaled, scaled], [square]),
                OperatorIR("MUL", anchor_inputs, [anchored]),
                OperatorIR("MUL", scale_inputs, [output]),
            ]
        )
    return model_ir


def _assert_index_current(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_mul_square_constant_fold_rewrites_both_input_orders_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _mul_square_constant_model()
    original_quantization = model_ir.tensors["branch0_anchor_constant"].quantization
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_mul_square_anchor_constant_chains(model_ir)

    assert stats == {"optimized_yolo_decode_mul_square_anchor_chains": 2}
    assert refresh_count == 1
    assert [str(operator.op_type) for operator in model_ir.operators] == [
        "MUL",
        "MUL",
        "MUL",
        "MUL",
    ]
    for branch_index, expected_data in (
        (0, np.asarray([6.0, 8.0], dtype=np.float16)),
        (1, np.asarray([9.0, 11.25], dtype=np.float16)),
    ):
        prefix = f"branch{branch_index}"
        square = next(
            operator
            for operator in model_ir.operators
            if str(operator.outputs[0]) == f"{prefix}_square"
        )
        anchor = next(
            operator
            for operator in model_ir.operators
            if str(operator.outputs[0]) == f"{prefix}_output"
        )
        assert [str(name) for name in square.inputs] == [
            f"{prefix}_source",
            f"{prefix}_source",
        ]
        fused_name = next(
            str(name)
            for name in anchor.inputs
            if model_ir.tensors[str(name)].data is not None
        )
        fused = model_ir.tensors[fused_name]
        assert fused_name.startswith(f"{prefix}_anchor_constant_mulsq_fused")
        assert fused.dtype == "FLOAT16"
        assert fused.shape == [2]
        np.testing.assert_array_equal(np.asarray(fused.data), expected_data)
        assert fused.quantization is not None
        assert fused.quantization.scale == [0.25]
    fused0 = next(
        tensor
        for name, tensor in model_ir.tensors.items()
        if name.startswith("branch0_anchor_constant_mulsq_fused")
    )
    assert fused0.quantization is not original_quantization
    assert set(model_ir.tensors) == {
        "branch0_source",
        "branch0_square",
        "branch0_output",
        next(
            name
            for name in model_ir.tensors
            if name.startswith("branch0_anchor_constant_mulsq_fused")
        ),
        "branch1_source",
        "branch1_square",
        "branch1_output",
        next(
            name
            for name in model_ir.tensors
            if name.startswith("branch1_anchor_constant_mulsq_fused")
        ),
    }


def test_mul_square_constant_fold_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _mul_square_constant_model(branches=1)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_mul_square_anchor_constant_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_yolo_decode_mul_square_anchor_chains": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize(
    "case",
    [
        "public_scaled",
        "public_square",
        "public_anchored",
        "scaled_fanout",
        "square_fanout",
        "anchored_fanout",
        "non_singleton_pre",
        "nonfinite_pre",
        "integer_anchor",
        "integer_scale",
        "nonfinite_result",
        "not_square",
        "nonconstant_anchor",
        "nonconstant_scale",
    ],
)
def test_mul_square_constant_fold_preserves_numeric_and_topology_guards(
    case: str,
) -> None:
    model_ir = _mul_square_constant_model(branches=1)
    public_names = {
        "public_scaled": "branch0_scaled",
        "public_square": "branch0_square",
        "public_anchored": "branch0_anchored",
    }
    fanout_names = {
        "scaled_fanout": "branch0_scaled",
        "square_fanout": "branch0_square",
        "anchored_fanout": "branch0_anchored",
    }
    if case in public_names:
        model_ir.outputs.append(public_names[case])
    elif case in fanout_names:
        model_ir.tensors["side"] = _tensor("side")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", [fanout_names[case]], ["side"])
        )
    elif case == "non_singleton_pre":
        model_ir.tensors["branch0_pre_constant"].data = np.asarray(
            [2.0, 3.0],
            dtype=np.float32,
        )
    elif case == "nonfinite_pre":
        model_ir.tensors["branch0_pre_constant"].data = np.asarray(
            np.inf,
            dtype=np.float32,
        )
    elif case == "integer_anchor":
        model_ir.tensors["branch0_anchor_constant"].data = np.asarray(
            [3, 4],
            dtype=np.int32,
        )
    elif case == "integer_scale":
        model_ir.tensors["branch0_scale_constant"].data = np.asarray(
            2,
            dtype=np.int32,
        )
    elif case == "nonfinite_result":
        model_ir.tensors["branch0_anchor_constant"].data = np.asarray(
            [np.inf, 4.0],
            dtype=np.float16,
        )
    elif case == "not_square":
        model_ir.tensors["other"] = _tensor("other")
        model_ir.inputs.append("other")
        model_ir.operators[1].inputs[1] = "other"
    elif case == "nonconstant_anchor":
        model_ir.tensors["branch0_anchor_constant"].data = None
    elif case == "nonconstant_scale":
        model_ir.tensors["branch0_scale_constant"].data = None

    before_operators = [
        (
            str(operator.op_type),
            [str(name) for name in operator.inputs],
            [str(name) for name in operator.outputs],
        )
        for operator in model_ir.operators
    ]
    before_tensors = set(model_ir.tensors)

    stats = _optimize_mul_square_anchor_constant_chains(model_ir)

    assert stats == {"optimized_yolo_decode_mul_square_anchor_chains": 0}
    assert [
        (
            str(operator.op_type),
            [str(name) for name in operator.inputs],
            [str(name) for name in operator.outputs],
        )
        for operator in model_ir.operators
    ] == before_operators
    assert set(model_ir.tensors) == before_tensors


def test_mul_square_constant_fold_skips_index_and_prune_without_mul(
    monkeypatch,
) -> None:
    model_ir = ModelIR("constant_fold_without_mul")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "y": _tensor("y"),
        "unused": _tensor("unused"),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(fold_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_mul_square_anchor_constant_chains(model_ir)

    assert stats == {"optimized_yolo_decode_mul_square_anchor_chains": 0}
    assert set(model_ir.tensors) == {"x", "y", "unused"}
