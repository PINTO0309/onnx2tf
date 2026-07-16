from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.terminal_squeeze_mean_layout as terminal_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.terminal_squeeze_mean_layout import (
    _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains,
)


_STATS = "optimized_transpose_squeeze_mean_squeeze_terminal_nhwc_chains"


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data=None,
    signature: list[int] | None = None,
    quantization=None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _build_model(
    *,
    prefix: str = "",
    produced_source: bool = False,
    negative_squeeze1: bool = False,
    negative_mean: bool = False,
    negative_squeeze2: bool = False,
) -> tuple[ModelIR, dict[str, str]]:
    model_ir = ModelIR(f"{prefix}terminal_squeeze_mean")
    names = {
        key: f"{prefix}{key}"
        for key in (
            "upstream",
            "source",
            "perm",
            "transposed",
            "squeezed1",
            "mean_axis",
            "mean",
            "output",
        )
    }
    shapes = {
        names["source"]: [2, 1, 4, 3],
        names["transposed"]: [2, 3, 1, 4],
        names["squeezed1"]: [2, 3, 4],
        names["mean"]: [2, 1, 4],
        names["output"]: [2, 4],
    }
    for name, shape in shapes.items():
        model_ir.tensors[name] = _tensor(name, shape)
    model_ir.tensors[names["perm"]] = _tensor(
        names["perm"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors[names["mean_axis"]] = _tensor(
        names["mean_axis"],
        [1],
        dtype="INT64",
        data=np.asarray([-2 if negative_mean else 1], dtype=np.int64),
    )
    if produced_source:
        model_ir.inputs = [names["upstream"]]
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"],
            shapes[names["source"]],
        )
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                [names["upstream"]],
                [names["source"]],
            )
        )
    else:
        model_ir.inputs = [names["source"]]
    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [names["source"], names["perm"]],
                [names["transposed"]],
                {"marker": "transpose"},
                axis_semantics={"marker": "preserved"},
                version=2,
                onnx_node_name="terminal_transpose",
                onnx_op_type="Transpose",
            ),
            OperatorIR(
                "SQUEEZE",
                [names["transposed"]],
                [names["squeezed1"]],
                {
                    "squeezeDims": [-2 if negative_squeeze1 else 2],
                    "marker": "squeeze1",
                },
            ),
            OperatorIR(
                "MEAN",
                [names["squeezed1"], names["mean_axis"]],
                [names["mean"]],
                {"keepDims": True, "marker": "mean"},
            ),
            OperatorIR(
                "SQUEEZE",
                [names["mean"]],
                [names["output"]],
                {
                    "squeezeDims": [-2 if negative_squeeze2 else 1],
                    "marker": "squeeze2",
                },
            ),
        ]
    )
    model_ir.outputs = [names["output"]]
    return model_ir, names


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
    for operator in model_ir.operators:
        inputs = [values[name] for name in operator.inputs]
        if operator.op_type == "IDENTITY":
            output = inputs[0]
        elif operator.op_type == "TRANSPOSE":
            output = np.transpose(
                inputs[0],
                tuple(int(value) for value in inputs[1].reshape(-1)),
            )
        elif operator.op_type == "SQUEEZE":
            axes = tuple(
                int(value)
                for value in np.asarray(
                    operator.options["squeezeDims"]
                ).reshape(-1)
            )
            output = np.squeeze(inputs[0], axis=axes)
        elif operator.op_type == "MEAN":
            axes = tuple(int(value) for value in inputs[1].reshape(-1))
            output = np.mean(
                inputs[0],
                axis=axes,
                keepdims=bool(operator.options["keepDims"]),
            )
        else:
            raise AssertionError(f"unsupported evaluator op: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {name: values[name] for name in model_ir.outputs}


def _operator(model_ir: ModelIR, output_name: str) -> OperatorIR:
    return next(
        operator
        for operator in model_ir.operators
        if output_name in operator.outputs
    )


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


@pytest.mark.parametrize("produced_source", (False, True))
@pytest.mark.parametrize("negative_squeeze1", (False, True))
@pytest.mark.parametrize("negative_mean", (False, True))
@pytest.mark.parametrize("negative_squeeze2", (False, True))
def test_terminal_squeeze_mean_rewrite_is_indexed_and_numerically_equivalent(
    produced_source: bool,
    negative_squeeze1: bool,
    negative_mean: bool,
    negative_squeeze2: bool,
) -> None:
    model_ir, names = _build_model(
        produced_source=produced_source,
        negative_squeeze1=negative_squeeze1,
        negative_mean=negative_mean,
        negative_squeeze2=negative_squeeze2,
    )
    feed_name = names["upstream"] if produced_source else names["source"]
    feeds = {
        feed_name: np.random.default_rng(53)
        .normal(size=model_ir.tensors[feed_name].shape)
        .astype(np.float32)
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    original_transpose = copy.deepcopy(
        _operator(model_ir, names["transposed"])
    )
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    actual = _evaluate(model_ir, feeds)
    np.testing.assert_allclose(
        actual[names["output"]],
        expected[names["output"]],
        rtol=1e-6,
        atol=1e-6,
    )
    assert names["transposed"] not in model_ir.tensors
    squeeze1 = _operator(model_ir, names["squeezed1"])
    mean = _operator(model_ir, names["mean"])
    squeeze2 = _operator(model_ir, names["output"])
    assert squeeze1.inputs == [names["source"]]
    assert squeeze1.options == {"squeezeDims": [1], "marker": "squeeze1"}
    assert mean.options == {"keepDims": True, "marker": "mean"}
    assert squeeze2.options == {"squeezeDims": [2], "marker": "squeeze2"}
    assert model_ir.tensors[names["squeezed1"]].shape == [2, 4, 3]
    assert model_ir.tensors[names["mean"]].shape == [2, 4, 1]
    assert model_ir.tensors[names["output"]].shape == [2, 4]
    assert model_ir.outputs == [names["output"]]
    assert np.asarray(model_ir.tensors[mean.inputs[1]].data).tolist() == [2]
    assert original_transpose.options == {"marker": "transpose"}
    assert original_transpose.axis_semantics == {"marker": "preserved"}
    assert original_transpose.version == 2
    assert original_transpose.onnx_node_name == "terminal_transpose"


def test_terminal_squeeze_mean_clones_shared_mean_axis() -> None:
    model_ir, names = _build_model()
    original_axis = np.asarray(model_ir.tensors[names["mean_axis"]].data).copy()
    model_ir.tensors["preserved_axis"] = _tensor(
        "preserved_axis",
        [1],
        dtype="INT64",
    )
    model_ir.operators.append(
        OperatorIR("IDENTITY", [names["mean_axis"]], ["preserved_axis"])
    )
    model_ir.outputs.append("preserved_axis")

    stats = _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    np.testing.assert_array_equal(
        model_ir.tensors[names["mean_axis"]].data,
        original_axis,
    )
    mean = _operator(model_ir, names["mean"])
    assert mean.inputs[1] != names["mean_axis"]
    assert np.asarray(model_ir.tensors[mean.inputs[1]].data).tolist() == [2]
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("dynamic_axis", ("batch", "width", "channel"))
def test_terminal_squeeze_mean_preserves_dynamic_signature(
    dynamic_axis: str,
) -> None:
    model_ir, names = _build_model()
    signature_indices = {
        "batch": {
            names["source"]: 0,
            names["transposed"]: 0,
            names["squeezed1"]: 0,
            names["mean"]: 0,
            names["output"]: 0,
        },
        "width": {
            names["source"]: 2,
            names["transposed"]: 3,
            names["squeezed1"]: 2,
            names["mean"]: 2,
            names["output"]: 1,
        },
        "channel": {
            names["source"]: 3,
            names["transposed"]: 1,
            names["squeezed1"]: 1,
        },
    }[dynamic_axis]
    for name, index in signature_indices.items():
        model_ir.tensors[name].shape_signature[index] = -1

    stats = _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    expected = {
        "batch": ([-1, 4, 3], [-1, 4, 1], [-1, 4]),
        "width": ([2, -1, 3], [2, -1, 1], [2, -1]),
        "channel": ([2, 4, -1], [2, 4, 1], [2, 4]),
    }[dynamic_axis]
    assert model_ir.tensors[names["squeezed1"]].shape_signature == expected[0]
    assert model_ir.tensors[names["mean"]].shape_signature == expected[1]
    assert model_ir.tensors[names["output"]].shape_signature == expected[2]


def test_terminal_squeeze_mean_rewrites_multiple_chains() -> None:
    first, _ = _build_model(prefix="a_")
    second, _ = _build_model(prefix="b_", produced_source=True)
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)

    stats = _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(first)

    assert stats == {_STATS: 2}
    assert sum(op.op_type == "TRANSPOSE" for op in first.operators) == 0
    assert validate_model_ir_invariants(first) == []


@pytest.mark.parametrize(
    "case",
    (
        "floating_perm",
        "produced_perm",
        "transpose_public",
        "transpose_fanout",
        "source_unbound",
        "backward_source",
        "squeeze1_axis",
        "squeeze1_dynamic_singleton",
        "squeeze1_shape",
        "squeeze1_public",
        "squeeze1_fanout",
        "mean_keepdims",
        "mean_axis",
        "floating_mean_axis",
        "produced_mean_axis",
        "mean_shape",
        "mean_public",
        "mean_fanout",
        "squeeze2_axis",
        "output_shape",
        "output_public_input",
        "duplicate_output",
        "dtype_mismatch",
        "per_axis_quantization",
        "quantization_grid",
        "backward_mean",
        "backward_output_consumer",
    ),
)
def test_terminal_squeeze_mean_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_model(produced_source=case == "backward_source")
    transpose = _operator(model_ir, names["transposed"])
    squeeze1 = _operator(model_ir, names["squeezed1"])
    mean = _operator(model_ir, names["mean"])
    squeeze2 = _operator(model_ir, names["output"])
    if case == "floating_perm":
        model_ir.tensors[names["perm"]].dtype = "FLOAT32"
        model_ir.tensors[names["perm"]].data = np.asarray(
            [0, 3, 1, 2], dtype=np.float32
        )
    elif case == "produced_perm":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["mean_axis"]], [names["perm"]])
        )
    elif case == "transpose_public":
        model_ir.outputs.append(names["transposed"])
    elif case == "transpose_fanout":
        model_ir.tensors["side"] = _tensor("side", [2, 3, 1, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["transposed"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "source_unbound":
        model_ir.inputs.remove(names["source"])
    elif case == "backward_source":
        producer = model_ir.operators.pop(0)
        model_ir.operators.insert(model_ir.operators.index(transpose) + 1, producer)
    elif case == "squeeze1_axis":
        squeeze1.options["squeezeDims"] = [1]
    elif case == "squeeze1_dynamic_singleton":
        model_ir.tensors[names["source"]].shape_signature[1] = -1
        model_ir.tensors[names["transposed"]].shape_signature[2] = -1
        model_ir.tensors[names["squeezed1"]].shape_signature = [-1, 3, 4]
    elif case == "squeeze1_shape":
        model_ir.tensors[names["squeezed1"]].shape[1] = 4
        model_ir.tensors[names["squeezed1"]].shape_signature[1] = 4
    elif case == "squeeze1_public":
        model_ir.outputs.append(names["squeezed1"])
    elif case == "squeeze1_fanout":
        model_ir.tensors["side"] = _tensor("side", [2, 3, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["squeezed1"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "mean_keepdims":
        mean.options["keepDims"] = False
    elif case == "mean_axis":
        model_ir.tensors[names["mean_axis"]].data[0] = 0
    elif case == "floating_mean_axis":
        model_ir.tensors[names["mean_axis"]].dtype = "FLOAT32"
        model_ir.tensors[names["mean_axis"]].data = np.asarray(
            [1], dtype=np.float32
        )
    elif case == "produced_mean_axis":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                [names["perm"]],
                [names["mean_axis"]],
            )
        )
    elif case == "mean_shape":
        model_ir.tensors[names["mean"]].shape[2] = 5
        model_ir.tensors[names["mean"]].shape_signature[2] = 5
    elif case == "mean_public":
        model_ir.outputs.append(names["mean"])
    elif case == "mean_fanout":
        model_ir.tensors["side"] = _tensor("side", [2, 1, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["mean"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "squeeze2_axis":
        squeeze2.options["squeezeDims"] = [2]
    elif case == "output_shape":
        model_ir.tensors[names["output"]].shape[1] = 5
        model_ir.tensors[names["output"]].shape_signature[1] = 5
    elif case == "output_public_input":
        model_ir.inputs.append(names["output"])
    elif case == "duplicate_output":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["output"]])
        )
    elif case == "dtype_mismatch":
        model_ir.tensors[names["mean"]].dtype = "FLOAT16"
    elif case == "per_axis_quantization":
        model_ir.tensors[names["squeezed1"]].quantization = {
            "scale": [0.25, 0.5, 0.75],
            "zero_point": [0, 0, 0],
            "quantized_dimension": 1,
        }
    elif case == "quantization_grid":
        model_ir.tensors[names["source"]].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
        model_ir.tensors[names["transposed"]].quantization = {
            "scale": [0.5],
            "zero_point": [0],
        }
        model_ir.tensors[names["squeezed1"]].quantization = {
            "scale": [0.5],
            "zero_point": [0],
        }
    elif case == "backward_mean":
        model_ir.operators.remove(mean)
        model_ir.operators.insert(model_ir.operators.index(squeeze1), mean)
    elif case == "backward_output_consumer":
        model_ir.tensors["side"] = _tensor("side", [2, 4])
        model_ir.operators.insert(
            model_ir.operators.index(squeeze2),
            OperatorIR("IDENTITY", [names["output"]], ["side"]),
        )
        model_ir.outputs.append("side")
    before = repr(model_ir)

    stats = _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_terminal_squeeze_mean_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir, names = _build_model()
    model_ir.tensors[names["mean_axis"]].quantization = {
        "fault": Unclonable()
    }
    model_ir.tensors["preserved_axis"] = _tensor(
        "preserved_axis",
        [1],
        dtype="INT64",
    )
    model_ir.operators.append(
        OperatorIR("IDENTITY", [names["mean_axis"]], ["preserved_axis"])
    )
    model_ir.outputs.append("preserved_axis")
    before = repr(model_ir)

    stats = _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_terminal_squeeze_mean_apply_preflight_is_transactional(
    monkeypatch,
) -> None:
    model_ir, names = _build_model()
    model_ir.tensors["preserved_axis"] = _tensor(
        "preserved_axis",
        [1],
        dtype="INT64",
    )
    model_ir.operators.append(
        OperatorIR("IDENTITY", [names["mean_axis"]], ["preserved_axis"])
    )
    model_ir.outputs.append("preserved_axis")
    original_resolve = terminal_module._resolve_candidate

    def resolve_with_collision(*args, **kwargs):
        plan = original_resolve(*args, **kwargs)
        assert plan is not None
        clone_name = plan.mean_axis_update.clone_name
        assert clone_name is not None
        model_ir.tensors[clone_name] = _tensor(
            clone_name,
            [1],
            dtype="INT64",
            data=np.asarray([9], dtype=np.int64),
        )
        return plan

    monkeypatch.setattr(
        terminal_module,
        "_resolve_candidate",
        resolve_with_collision,
    )
    before_operators = repr(model_ir.operators)
    before_axis = np.asarray(model_ir.tensors[names["mean_axis"]].data).copy()

    stats = _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir.operators) == before_operators
    np.testing.assert_array_equal(
        model_ir.tensors[names["mean_axis"]].data,
        before_axis,
    )


def test_terminal_squeeze_mean_preflight_does_not_allocate_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_terminal_mean")
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(terminal_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
        model_ir
    ) == {_STATS: 0}
