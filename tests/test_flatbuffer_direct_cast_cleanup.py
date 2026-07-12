from __future__ import annotations

import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.cast_cleanup import run_redundant_cast_cleanup


def _tensor(name: str, dtype: str, shape: list[int] | None = None) -> TensorIR:
    normalized_shape = list(shape or [2])
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=normalized_shape,
        shape_signature=normalized_shape,
    )


def _narrowing_model(*, unsigned: bool = False) -> ModelIR:
    wide = "UINT64" if unsigned else "INT64"
    narrow = "UINT32" if unsigned else "INT32"
    model_ir = ModelIR("redundant_narrowing_cast")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["out"]
    model_ir.tensors = {
        "x": _tensor("x", "FLOAT32"),
        "mid": _tensor("mid", wide),
        "narrowed": _tensor("narrowed", narrow),
        "out": _tensor("out", narrow),
    }
    model_ir.tensors["x"].quantization = {
        "scale": [0.5],
        "zero_point": [0],
    }
    model_ir.operators = [
        OperatorIR(
            "CAST",
            ["x"],
            ["mid"],
            options={"inDataType": "FLOAT32", "outDataType": wide},
        ),
        OperatorIR(
            "CAST",
            ["mid"],
            ["narrowed"],
            options={"inDataType": wide, "outDataType": narrow},
        ),
        OperatorIR("IDENTITY", ["narrowed"], ["out"]),
    ]
    return model_ir


def _widening_alias_model(*, unsigned: bool = False) -> ModelIR:
    work_dtype = "UINT32" if unsigned else "INT32"
    alias_dtype = "UINT64" if unsigned else "INT64"
    model_ir = ModelIR("redundant_widening_alias")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["float_out", "bool_out"]
    model_ir.tensors = {
        "x": _tensor("x", "FLOAT32", [1, 2]),
        "work": _tensor("work", work_dtype),
        "alias": _tensor("alias", alias_dtype, [1]),
        "float_out": _tensor("float_out", "FLOAT32"),
        "bool_out": _tensor("bool_out", "BOOL"),
    }
    model_ir.tensors["work"].quantization = {
        "scale": [1.0],
        "zero_point": [0],
    }
    model_ir.operators = [
        OperatorIR("SHAPE", ["x"], ["work"]),
        OperatorIR(
            "CAST",
            ["work"],
            ["alias"],
            options={"inDataType": work_dtype, "outDataType": alias_dtype},
        ),
        OperatorIR(
            "CAST",
            ["alias"],
            ["float_out"],
            options={"inDataType": alias_dtype, "outDataType": "FLOAT32"},
        ),
        OperatorIR(
            "CAST",
            ["alias"],
            ["bool_out"],
            options={"inDataType": alias_dtype, "outDataType": "BOOL"},
        ),
    ]
    return model_ir


@pytest.mark.parametrize("unsigned", [False, True])
def test_redundant_cast_cleanup_collapses_narrowing_chain(
    unsigned: bool,
) -> None:
    model_ir = _narrowing_model(unsigned=unsigned)
    diagnostics: list[dict] = []

    stats = run_redundant_cast_cleanup(model_ir, diagnostics=diagnostics)

    narrow = "UINT32" if unsigned else "INT32"
    assert stats == {
        "optimized_redundant_int32_to_int64_passthrough_cast_chains": 0,
        "optimized_redundant_int64_to_int32_cast_chains": 1,
    }
    assert [op.op_type for op in model_ir.operators] == ["CAST", "IDENTITY"]
    assert model_ir.operators[0].options["outDataType"] == narrow
    assert model_ir.operators[1].inputs == ["mid"]
    assert model_ir.tensors["mid"].dtype == narrow
    assert model_ir.tensors["mid"].quantization == {
        "scale": [0.5],
        "zero_point": [0],
    }
    assert "narrowed" not in model_ir.tensors
    assert [event["code"] for event in diagnostics] == [
        "cleanup.cast_widening_alias",
        "cleanup.cast_narrowing_chain",
    ]
    assert [event["status"] for event in diagnostics] == ["skipped", "changed"]


@pytest.mark.parametrize("unsigned", [False, True])
def test_redundant_cast_cleanup_removes_widening_alias(unsigned: bool) -> None:
    model_ir = _widening_alias_model(unsigned=unsigned)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = run_redundant_cast_cleanup(
        model_ir,
        layout_state=layout_state,
    )

    work_dtype = "UINT32" if unsigned else "INT32"
    assert stats == {
        "optimized_redundant_int32_to_int64_passthrough_cast_chains": 1,
        "optimized_redundant_int64_to_int32_cast_chains": 0,
    }
    assert [op.op_type for op in model_ir.operators] == ["SHAPE", "CAST", "CAST"]
    assert model_ir.operators[0].outputs == ["alias"]
    assert all(op.inputs == ["alias"] for op in model_ir.operators[1:])
    assert all(
        op.options["inDataType"] == work_dtype for op in model_ir.operators[1:]
    )
    assert model_ir.tensors["alias"].dtype == work_dtype
    assert model_ir.tensors["alias"].shape == [2]
    assert model_ir.tensors["alias"].quantization == {
        "scale": [1.0],
        "zero_point": [0],
    }
    assert "work" not in model_ir.tensors
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_redundant_cast_cleanup_uses_one_shared_index(monkeypatch) -> None:
    model_ir = _narrowing_model()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    run_redundant_cast_cleanup(model_ir)

    assert refresh_count == 1


def test_redundant_cast_cleanup_preserves_narrowing_fanout() -> None:
    model_ir = _narrowing_model()
    model_ir.tensors["side"] = _tensor("side", "INT64")
    model_ir.outputs.append("side")
    model_ir.operators.append(OperatorIR("IDENTITY", ["mid"], ["side"]))

    stats = run_redundant_cast_cleanup(model_ir)

    assert stats["optimized_redundant_int64_to_int32_cast_chains"] == 0
    assert [op.op_type for op in model_ir.operators] == [
        "CAST",
        "CAST",
        "IDENTITY",
        "IDENTITY",
    ]


@pytest.mark.parametrize("public_tensor", ["mid", "narrowed"])
def test_redundant_cast_cleanup_preserves_public_narrowing_tensors(
    public_tensor: str,
) -> None:
    model_ir = _narrowing_model()
    model_ir.outputs.append(public_tensor)

    stats = run_redundant_cast_cleanup(model_ir)

    assert stats["optimized_redundant_int64_to_int32_cast_chains"] == 0


def test_redundant_cast_cleanup_preserves_mixed_alias_consumers() -> None:
    model_ir = _widening_alias_model()
    model_ir.tensors["side"] = _tensor("side", "INT64")
    model_ir.outputs.append("side")
    model_ir.operators.append(OperatorIR("IDENTITY", ["alias"], ["side"]))

    stats = run_redundant_cast_cleanup(model_ir)

    assert stats["optimized_redundant_int32_to_int64_passthrough_cast_chains"] == 0
    assert model_ir.operators[0].outputs == ["work"]


def test_redundant_cast_cleanup_preserves_public_alias() -> None:
    model_ir = _widening_alias_model()
    model_ir.outputs.append("alias")

    stats = run_redundant_cast_cleanup(model_ir)

    assert stats["optimized_redundant_int32_to_int64_passthrough_cast_chains"] == 0
