from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.mixed_singleton_concat_layout as mixed_concat_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.mixed_singleton_concat_layout import (
    _repair_mixed_singleton_nchw_inputs_for_nhwc_concat,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data=None,
    quantization=None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _add_branch(
    model_ir: ModelIR,
    prefix: str,
    *,
    produced_source: bool = False,
    constant_source: bool = False,
) -> dict[str, str]:
    names = {
        key: f"{prefix}_{key}"
        for key in ("upstream", "nchw", "nhwc0", "nhwc1", "output")
    }
    quantization = QuantParamIR(scale=[0.25], zero_point=[3], quantized_dimension=1)
    source_data = (
        np.arange(20, dtype=np.int8).reshape(1, 1, 4, 5) if constant_source else None
    )
    dtype = "INT8" if constant_source else "FLOAT32"
    model_ir.tensors.update(
        {
            names["nchw"]: _tensor(
                names["nchw"],
                [1, 1, 4, 5],
                dtype=dtype,
                data=source_data,
                quantization=quantization,
            ),
            names["nhwc0"]: _tensor(names["nhwc0"], [1, 4, 5, 1], dtype=dtype),
            names["nhwc1"]: _tensor(names["nhwc1"], [1, 4, 5, 1], dtype=dtype),
            names["output"]: _tensor(names["output"], [1, 4, 5, 3], dtype=dtype),
        }
    )
    if produced_source:
        model_ir.tensors[names["upstream"]] = _tensor(names["upstream"], [1, 1, 4, 5])
        model_ir.inputs.append(names["upstream"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["upstream"]], [names["nchw"]])
        )
    elif not constant_source:
        model_ir.inputs.append(names["nchw"])
    model_ir.inputs.extend([names["nhwc0"], names["nhwc1"]])
    model_ir.outputs.append(names["output"])
    model_ir.operators.append(
        OperatorIR(
            "CONCATENATION",
            [names["nchw"], names["nhwc0"], names["nhwc1"]],
            [names["output"]],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        )
    )
    return names


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_mixed_singleton_concat_repairs_multiple_branches_with_one_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("indexed_mixed_singleton_concat")
    branches = [
        _add_branch(model_ir, "input"),
        _add_branch(model_ir, "produced", produced_source=True),
        _add_branch(model_ir, "constant", constant_source=True),
    ]
    refreshes = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refreshes
        refreshes += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir)

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 3}
    assert refreshes == 1
    assert [op.op_type for op in model_ir.operators].count("RESHAPE") == 3
    for names in branches:
        concat = next(
            op for op in model_ir.operators if op.outputs == [names["output"]]
        )
        adapter_name = str(concat.inputs[0])
        adapter = model_ir.tensors[adapter_name]
        reshape = next(op for op in model_ir.operators if op.outputs == [adapter_name])
        assert adapter.shape == [1, 4, 5, 1]
        assert reshape.inputs == [
            names["nchw"],
            f"{adapter_name}_reshape_shape",
        ]
        assert concat.inputs[1:] == [names["nhwc0"], names["nhwc1"]]
        assert adapter.quantization == model_ir.tensors[names["nchw"]].quantization
        assert adapter.quantization is not model_ir.tensors[names["nchw"]].quantization


def test_mixed_singleton_concat_keeps_supplied_index_and_layout_current() -> None:
    model_ir = ModelIR("maintained_mixed_singleton_concat")
    _add_branch(model_ir, "branch")
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_mixed_singleton_concat_reuses_adapter_for_repeated_source() -> None:
    model_ir = ModelIR("repeated_mixed_singleton_concat")
    names = _add_branch(model_ir, "branch")
    concat = model_ir.operators[-1]
    concat.inputs = [names["nchw"], names["nchw"], names["nhwc0"]]

    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir)

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 1}
    assert [op.op_type for op in model_ir.operators] == [
        "RESHAPE",
        "CONCATENATION",
    ]
    assert concat.inputs[0] == concat.inputs[1]
    assert concat.inputs[0] != names["nchw"]
    assert ModelIRGraphIndex(model_ir).duplicate_producers == {}


def test_mixed_singleton_concat_allocates_names_across_all_candidates() -> None:
    model_ir = ModelIR("named_mixed_singleton_concat")
    first = _add_branch(model_ir, "branch")
    second = _add_branch(model_ir, "branch2")
    collision = f"{first['nchw']}_nhwc_concat_adapter"
    model_ir.tensors[collision] = _tensor(collision, [1])

    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir)

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 2}
    first_concat = next(
        op for op in model_ir.operators if op.outputs == [first["output"]]
    )
    second_concat = next(
        op for op in model_ir.operators if op.outputs == [second["output"]]
    )
    assert first_concat.inputs[0] == f"{collision}_1"
    assert second_concat.inputs[0] == f"{second['nchw']}_nhwc_concat_adapter"


@pytest.mark.parametrize(
    ("output_signature", "nchw_signature", "nhwc_signature"),
    [
        ([-1, 4, 5, 3], [-1, 1, 4, 5], [-1, 4, 5, 1]),
        ([1, -1, 5, 3], [1, 1, -1, 5], [1, -1, 5, 1]),
        ([1, 4, -1, 3], [1, 1, 4, -1], [1, 4, -1, 1]),
    ],
)
def test_mixed_singleton_concat_preserves_one_dynamic_dimension(
    output_signature: list[int],
    nchw_signature: list[int],
    nhwc_signature: list[int],
) -> None:
    from onnx2tf.tflite_builder.lower_from_onnx2tf import (
        _reconcile_static_tensor_shapes,
    )

    model_ir = ModelIR("dynamic_mixed_singleton_concat")
    names = _add_branch(model_ir, "branch")
    model_ir.tensors[names["output"]].shape_signature = output_signature
    model_ir.tensors[names["nchw"]].shape_signature = nchw_signature
    model_ir.tensors[names["nhwc0"]].shape_signature = nhwc_signature
    model_ir.tensors[names["nhwc1"]].shape_signature = nhwc_signature

    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir)
    _reconcile_static_tensor_shapes(model_ir)

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 1}
    reshape = model_ir.operators[0]
    adapter = model_ir.tensors[str(reshape.outputs[0])]
    shape_tensor = model_ir.tensors[str(reshape.inputs[1])]
    assert adapter.shape_signature == nhwc_signature
    assert reshape.options["newShape"] == nhwc_signature
    assert np.asarray(shape_tensor.data).tolist() == nhwc_signature


def test_mixed_singleton_concat_skips_multiple_dynamic_reshape_dimensions() -> None:
    model_ir = ModelIR("multi_dynamic_mixed_singleton_concat")
    names = _add_branch(model_ir, "branch")
    model_ir.tensors[names["output"]].shape_signature = [-1, -1, 5, 3]
    model_ir.tensors[names["nchw"]].shape_signature = [-1, 1, -1, 5]
    model_ir.tensors[names["nhwc0"]].shape_signature = [-1, -1, 5, 1]
    model_ir.tensors[names["nhwc1"]].shape_signature = [-1, -1, 5, 1]
    before = repr(model_ir)

    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir)

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize(
    "case",
    [
        "concat_arity",
        "concat_output_arity",
        "options_type",
        "axis",
        "axis_invalid",
        "output_missing",
        "output_dynamic",
        "output_channel_equation",
        "duplicate_output_producer",
        "output_is_input",
        "input_missing",
        "input_dynamic",
        "input_rank",
        "input_nonpositive",
        "input_spatial",
        "input_dtype",
        "duplicate_input_producer",
        "input_boundary_and_producer",
        "input_producer_after_concat",
        "unbound_input",
    ],
)
def test_mixed_singleton_concat_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = ModelIR("rejected_mixed_singleton_concat")
    names = _add_branch(model_ir, "branch")
    concat = model_ir.operators[-1]
    if case == "concat_arity":
        concat.inputs = [names["nchw"]]
        model_ir.tensors[names["output"]].shape[3] = 1
        model_ir.tensors[names["output"]].shape_signature[3] = 1
    elif case == "concat_output_arity":
        concat.outputs.append("extra")
    elif case == "options_type":
        concat.options = None
    elif case == "axis":
        concat.options["axis"] = 1
    elif case == "axis_invalid":
        concat.options["axis"] = "channel"
    elif case == "output_missing":
        del model_ir.tensors[names["output"]]
    elif case == "output_dynamic":
        model_ir.tensors[names["output"]].shape_signature[0] = -1
    elif case == "output_channel_equation":
        model_ir.tensors[names["output"]].shape[3] = 4
        model_ir.tensors[names["output"]].shape_signature[3] = 4
    elif case == "duplicate_output_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["nhwc0"]], [names["output"]])
        )
    elif case == "output_is_input":
        model_ir.inputs.append(names["output"])
    elif case == "input_missing":
        del model_ir.tensors[names["nchw"]]
    elif case == "input_dynamic":
        model_ir.tensors[names["nchw"]].shape_signature[0] = -1
    elif case == "input_rank":
        model_ir.tensors[names["nchw"]].shape = [1, 1, 20]
    elif case == "input_nonpositive":
        model_ir.tensors[names["nchw"]].shape[2] = 0
    elif case == "input_spatial":
        model_ir.tensors[names["nchw"]].shape[3] = 6
        model_ir.tensors[names["nchw"]].shape_signature[3] = 6
    elif case == "input_dtype":
        model_ir.tensors[names["nchw"]].dtype = "INT8"
    elif case == "duplicate_input_producer":
        model_ir.operators.extend(
            [
                OperatorIR("IDENTITY", [names["nhwc0"]], [names["nchw"]]),
                OperatorIR("IDENTITY", [names["nhwc1"]], [names["nchw"]]),
            ]
        )
    elif case == "input_boundary_and_producer":
        model_ir.operators.insert(
            0,
            OperatorIR("IDENTITY", [names["nhwc0"]], [names["nchw"]]),
        )
    elif case == "input_producer_after_concat":
        model_ir.inputs.remove(names["nchw"])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["nhwc0"]], [names["nchw"]])
        )
    elif case == "unbound_input":
        model_ir.inputs.remove(names["nchw"])

    before = repr(model_ir)
    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir)

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 0}
    assert repr(model_ir) == before


def test_mixed_singleton_concat_quantization_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir = ModelIR("unclonable_mixed_singleton_concat")
    names = _add_branch(model_ir, "branch")
    late_name = "branch_late_nchw"
    model_ir.tensors[late_name] = _tensor(late_name, [1, 1, 4, 5])
    model_ir.tensors[late_name].quantization = Unclonable()
    model_ir.inputs.append(late_name)
    model_ir.operators[-1].inputs = [
        names["nchw"],
        late_name,
        names["nhwc0"],
    ]
    before = copy.copy(repr(model_ir))

    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir)

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 0}
    assert repr(model_ir) == before


def test_mixed_singleton_concat_complete_nhwc_inputs_are_noop() -> None:
    model_ir = ModelIR("already_nhwc_concat")
    names = _add_branch(model_ir, "branch")
    model_ir.tensors[names["nchw"]].shape = [1, 4, 5, 1]
    model_ir.tensors[names["nchw"]].shape_signature = [1, 4, 5, 1]
    before = repr(model_ir)

    stats = _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir)

    assert stats == {"repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 0}
    assert repr(model_ir) == before


def test_mixed_singleton_concat_skips_index_without_concat(monkeypatch) -> None:
    model_ir = ModelIR("no_mixed_singleton_concat")
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(
        mixed_concat_module,
        "ModelIRGraphIndex",
        unexpected_index,
    )

    assert _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(model_ir) == {
        "repaired_mixed_singleton_nchw_inputs_for_nhwc_concat": 0
    }
