from __future__ import annotations

import copy
import pickle
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_NWC,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.center_size_offset_layout import (
    optimize_center_size_offset_terminal_transpose_chains,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    layout: str = "UNKNOWN",
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=False,
        quantization=quantization,
        logical_layout=layout,
        physical_layout=layout,
    )


def _make_model(
    *,
    dynamic: bool = False,
    integer_dtype: str = "INT32",
    reverse_clip_inputs: bool = False,
    shared_shape: bool = False,
    unrelated_shape_consumer: bool = False,
    per_tensor_quantized: bool = False,
) -> ModelIR:
    batch, height, width, channels = 1, 2, 3, 2
    spatial = height * width
    integer_numpy_dtype = np.int64 if integer_dtype == "INT64" else np.int32
    source_signature_1 = [1, -1, -1, 1] if dynamic else [1, height, width, 1]
    source_signature_c = (
        [1, -1, -1, channels]
        if dynamic
        else [1, height, width, channels]
    )
    pre_signature_1 = [1, 1, -1, -1] if dynamic else [1, 1, height, width]
    pre_signature_c = (
        [1, channels, -1, -1]
        if dynamic
        else [1, channels, height, width]
    )
    flat_signature_1 = [1, -1] if dynamic else [1, spatial]
    flat_signature_c = [1, channels, -1] if dynamic else [1, channels, spatial]
    quantization = (
        QuantParamIR(scale=[0.125], zero_point=[0])
        if per_tensor_quantized
        else None
    )

    model_ir = ModelIR("indexed_center_size_offset")
    model_ir.inputs = ["center_nhwc", "size_nhwc", "offset_nhwc"]
    model_ir.outputs = ["center_flat", "gather_size", "gather_offset"]
    for name, channel, source_signature, pre_signature in (
        ("center", 1, source_signature_1, pre_signature_1),
        ("size", channels, source_signature_c, pre_signature_c),
        ("offset", channels, source_signature_c, pre_signature_c),
    ):
        model_ir.tensors[f"{name}_nhwc"] = _tensor(
            f"{name}_nhwc",
            [batch, height, width, channel],
            signature=source_signature,
            layout=LOGICAL_LAYOUT_NHWC,
            quantization=copy.deepcopy(quantization),
        )
        model_ir.tensors[f"{name}_nchw"] = _tensor(
            f"{name}_nchw",
            [batch, channel, height, width],
            signature=pre_signature,
            layout=LOGICAL_LAYOUT_NCHW,
            quantization=copy.deepcopy(quantization),
        )
    model_ir.tensors["perm"] = _tensor(
        "perm",
        [4],
        dtype=integer_dtype,
        data=np.asarray([0, 3, 1, 2], dtype=integer_numpy_dtype),
    )
    for name, value in (("clip_min", 0.0), ("clip_max", 1.0)):
        model_ir.tensors[name] = _tensor(
            name,
            [1],
            data=np.asarray([value], dtype=np.float32),
            quantization=copy.deepcopy(quantization),
        )
    for prefix, channel, signature in (
        ("center", 1, pre_signature_1),
        ("size", channels, pre_signature_c),
    ):
        for suffix in ("sig", "max", "min"):
            model_ir.tensors[f"{prefix}_{suffix}"] = _tensor(
                f"{prefix}_{suffix}",
                [batch, channel, height, width],
                signature=signature,
                layout=LOGICAL_LAYOUT_NCHW,
                quantization=copy.deepcopy(quantization),
            )

    model_ir.tensors["center_shape"] = _tensor(
        "center_shape",
        [2],
        dtype=integer_dtype,
        data=np.asarray([batch, spatial], dtype=integer_numpy_dtype),
    )
    model_ir.tensors["center_flat"] = _tensor(
        "center_flat",
        [batch, spatial],
        signature=flat_signature_1,
        quantization=copy.deepcopy(quantization),
    )
    size_shape_name = "shared_shape" if shared_shape else "size_shape"
    offset_shape_name = "shared_shape" if shared_shape else "offset_shape"
    for name in sorted({size_shape_name, offset_shape_name}):
        model_ir.tensors[name] = _tensor(
            name,
            [3],
            dtype=integer_dtype,
            data=np.asarray(
                [batch, channels, spatial],
                dtype=integer_numpy_dtype,
            ),
        )
    for name in ("size_reshaped", "offset_reshaped"):
        model_ir.tensors[name] = _tensor(
            name,
            [batch, channels, spatial],
            signature=flat_signature_c,
            quantization=copy.deepcopy(quantization),
        )

    model_ir.tensors["axis_indices"] = _tensor(
        "axis_indices",
        [1, 2, 1],
        dtype="INT32",
        data=np.asarray([[[0], [5]]], dtype=np.int32),
    )
    model_ir.tensors["axis_shape"] = _tensor(
        "axis_shape",
        [4],
        dtype=integer_dtype,
        data=np.asarray([1, 2, 1, 1], dtype=integer_numpy_dtype),
    )
    model_ir.tensors["axis_coord"] = _tensor(
        "axis_coord",
        [1, 2, 1, 1],
        dtype="INT32",
    )
    model_ir.tensors["batch_coord"] = _tensor(
        "batch_coord",
        [1, 2, 1, 1],
        dtype="INT32",
        data=np.zeros([1, 2, 1, 1], dtype=np.int32),
    )
    model_ir.tensors["channel_coord"] = _tensor(
        "channel_coord",
        [1, 2, 1, 1],
        dtype="INT32",
        data=np.asarray([[[[0]], [[1]]]], dtype=np.int32),
    )
    for name in ("coords_size", "coords_offset"):
        model_ir.tensors[name] = _tensor(
            name,
            [1, 2, 1, 3],
            dtype="INT32",
        )
    for name in ("gather_size", "gather_offset"):
        model_ir.tensors[name] = _tensor(
            name,
            [1, 2, 1],
            quantization=copy.deepcopy(quantization),
        )

    def _clip_inputs(value: str, constant: str) -> list[str]:
        return [constant, value] if reverse_clip_inputs else [value, constant]

    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["center_nhwc", "perm"], ["center_nchw"]),
        OperatorIR("TRANSPOSE", ["size_nhwc", "perm"], ["size_nchw"]),
        OperatorIR("TRANSPOSE", ["offset_nhwc", "perm"], ["offset_nchw"]),
        OperatorIR("LOGISTIC", ["center_nchw"], ["center_sig"]),
        OperatorIR("LOGISTIC", ["size_nchw"], ["size_sig"]),
        OperatorIR("MAXIMUM", _clip_inputs("center_sig", "clip_min"), ["center_max"]),
        OperatorIR("MINIMUM", _clip_inputs("center_max", "clip_max"), ["center_min"]),
        OperatorIR("MAXIMUM", _clip_inputs("size_sig", "clip_min"), ["size_max"]),
        OperatorIR("MINIMUM", _clip_inputs("size_max", "clip_max"), ["size_min"]),
        OperatorIR(
            "RESHAPE",
            ["center_min", "center_shape"],
            ["center_flat"],
            options={"newShape": [batch, spatial]},
        ),
        OperatorIR(
            "RESHAPE",
            ["size_min", size_shape_name],
            ["size_reshaped"],
            options={
                "newShape": [batch, channels, spatial],
                "onnxRawNewShape": [batch, channels, spatial],
            },
        ),
        OperatorIR(
            "RESHAPE",
            ["offset_nchw", offset_shape_name],
            ["offset_reshaped"],
            options={
                "newShape": [batch, channels, spatial],
                "onnxRawNewShape": [batch, channels, spatial],
            },
        ),
        OperatorIR(
            "RESHAPE",
            ["axis_indices", "axis_shape"],
            ["axis_coord"],
            options={"newShape": [1, 2, 1, 1]},
        ),
        OperatorIR(
            "CONCATENATION",
            ["batch_coord", "channel_coord", "axis_coord"],
            ["coords_size"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "GATHER_ND",
            ["size_reshaped", "coords_size"],
            ["gather_size"],
        ),
        OperatorIR(
            "CONCATENATION",
            ["batch_coord", "channel_coord", "axis_coord"],
            ["coords_offset"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "GATHER_ND",
            ["offset_reshaped", "coords_offset"],
            ["gather_offset"],
        ),
    ]
    if unrelated_shape_consumer:
        model_ir.tensors["unrelated"] = _tensor(
            "unrelated",
            [batch, channels, spatial],
        )
        model_ir.inputs.append("unrelated")
        model_ir.tensors["unrelated_out"] = _tensor(
            "unrelated_out",
            [batch, channels, spatial],
        )
        model_ir.outputs.append("unrelated_out")
        model_ir.operators.append(
            OperatorIR(
                "RESHAPE",
                ["unrelated", size_shape_name],
                ["unrelated_out"],
                options={"newShape": [batch, channels, spatial]},
            )
        )
    return model_ir


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("integer_dtype", ["INT32", "INT64"])
@pytest.mark.parametrize("reverse_clip_inputs", [False, True], ids=["data_first", "const_first"])
def test_indexed_center_size_offset_preserves_terminal_contract(
    dynamic: bool,
    integer_dtype: str,
    reverse_clip_inputs: bool,
) -> None:
    model_ir = _make_model(
        dynamic=dynamic,
        integer_dtype=integer_dtype,
        reverse_clip_inputs=reverse_clip_inputs,
        per_tensor_quantized=True,
    )
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_center_size_offset_terminal_transpose_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_center_size_offset_terminal_transpose_chains": 1}
    assert all(str(operator.op_type) != "TRANSPOSE" for operator in model_ir.operators)
    assert next(op for op in model_ir.operators if op.outputs == ["center_sig"]).inputs == ["center_nhwc"]
    assert next(op for op in model_ir.operators if op.outputs == ["size_sig"]).inputs == ["size_nhwc"]
    assert next(op for op in model_ir.operators if op.outputs == ["offset_reshaped"]).inputs[0] == "offset_nhwc"
    assert np.asarray(model_ir.tensors["size_shape"].data).tolist() == [1, 6, 2]
    assert np.asarray(model_ir.tensors["offset_shape"].data).tolist() == [1, 6, 2]
    assert model_ir.tensors["size_reshaped"].shape == [1, 6, 2]
    assert model_ir.tensors["offset_reshaped"].shape == [1, 6, 2]
    expected_signature = [1, -1, 2] if dynamic else [1, 6, 2]
    assert model_ir.tensors["size_reshaped"].shape_signature == expected_signature
    assert model_ir.tensors["size_reshaped"].physical_layout == LOGICAL_LAYOUT_NWC
    for name in ("coords_size", "coords_offset"):
        concat = next(op for op in model_ir.operators if op.outputs == [name])
        assert concat.inputs == ["batch_coord", "axis_coord", "channel_coord"]
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_shared_shape_constant_uses_one_copy_on_write_clone() -> None:
    model_ir = _make_model(
        shared_shape=True,
        unrelated_shape_consumer=True,
    )

    stats = optimize_center_size_offset_terminal_transpose_chains(model_ir)

    assert stats["optimized_center_size_offset_terminal_transpose_chains"] == 1
    assert np.asarray(model_ir.tensors["shared_shape"].data).tolist() == [1, 2, 6]
    assert np.asarray(model_ir.tensors["shared_shape_nhwc"].data).tolist() == [1, 6, 2]
    size = next(op for op in model_ir.operators if op.outputs == ["size_reshaped"])
    offset = next(op for op in model_ir.operators if op.outputs == ["offset_reshaped"])
    unrelated = next(op for op in model_ir.operators if op.outputs == ["unrelated_out"])
    assert size.inputs[1] == "shared_shape_nhwc"
    assert offset.inputs[1] == "shared_shape_nhwc"
    assert unrelated.inputs[1] == "shared_shape"
    assert validate_model_ir_invariants(model_ir) == []


def test_inferred_reshape_dimensions_rotate_without_becoming_concrete() -> None:
    model_ir = _make_model(dynamic=True)
    for name in ("size_shape", "offset_shape"):
        model_ir.tensors[name].data = np.asarray([1, 2, -1], dtype=np.int32)
    for output in ("size_reshaped", "offset_reshaped"):
        reshape = next(op for op in model_ir.operators if op.outputs == [output])
        reshape.options = {
            "newShape": [1, 2, -1],
            "onnxRawNewShape": [1, 2, -1],
        }

    stats = optimize_center_size_offset_terminal_transpose_chains(model_ir)

    assert stats["optimized_center_size_offset_terminal_transpose_chains"] == 1
    for name in ("size_shape", "offset_shape"):
        assert np.asarray(model_ir.tensors[name].data).tolist() == [1, -1, 2]
    for output in ("size_reshaped", "offset_reshaped"):
        reshape = next(op for op in model_ir.operators if op.outputs == [output])
        assert reshape.options["newShape"] == [1, -1, 2]
        assert reshape.options["onnxRawNewShape"] == [1, -1, 2]


def test_shared_coordinate_concat_is_reordered_once_for_both_gathers() -> None:
    model_ir = _make_model()
    offset_gather = next(
        op for op in model_ir.operators if op.outputs == ["gather_offset"]
    )
    offset_gather.inputs[1] = "coords_size"
    offset_concat = next(
        op for op in model_ir.operators if op.outputs == ["coords_offset"]
    )
    model_ir.operators.remove(offset_concat)
    model_ir.tensors.pop("coords_offset")

    stats = optimize_center_size_offset_terminal_transpose_chains(model_ir)

    assert stats["optimized_center_size_offset_terminal_transpose_chains"] == 1
    concat = next(op for op in model_ir.operators if op.outputs == ["coords_size"])
    assert concat.inputs == ["batch_coord", "axis_coord", "channel_coord"]
    assert validate_model_ir_invariants(model_ir) == []


def test_coordinate_semantics_support_int64_and_arbitrary_concat_order() -> None:
    model_ir = _make_model()
    for name in (
        "axis_indices",
        "axis_coord",
        "batch_coord",
        "channel_coord",
        "coords_size",
        "coords_offset",
    ):
        tensor = model_ir.tensors[name]
        tensor.dtype = "INT64"
        if tensor.data is not None:
            tensor.data = np.asarray(tensor.data, dtype=np.int64)
    for output in ("coords_size", "coords_offset"):
        concat = next(op for op in model_ir.operators if op.outputs == [output])
        concat.inputs = ["axis_coord", "channel_coord", "batch_coord"]

    stats = optimize_center_size_offset_terminal_transpose_chains(model_ir)

    assert stats["optimized_center_size_offset_terminal_transpose_chains"] == 1
    for output in ("coords_size", "coords_offset"):
        concat = next(op for op in model_ir.operators if op.outputs == [output])
        assert concat.inputs == ["batch_coord", "axis_coord", "channel_coord"]


def test_equal_batch_and_channel_coordinates_are_not_ambiguously_rejected() -> None:
    model_ir = _make_model()
    model_ir.tensors["channel_coord"].data = np.zeros(
        [1, 2, 1, 1],
        dtype=np.int32,
    )

    stats = optimize_center_size_offset_terminal_transpose_chains(model_ir)

    assert stats["optimized_center_size_offset_terminal_transpose_chains"] == 1


def test_candidate_limit_and_maintained_graph_index_are_deterministic() -> None:
    model_ir = _make_model()
    graph_index = ModelIRGraphIndex(model_ir)
    center = model_ir.operators[0]
    size = model_ir.operators[1]

    assert optimize_center_size_offset_terminal_transpose_chains(
        model_ir,
        graph_index=graph_index,
        candidate=size,
    ) == {"optimized_center_size_offset_terminal_transpose_chains": 0}
    assert optimize_center_size_offset_terminal_transpose_chains(
        model_ir,
        graph_index=graph_index,
        candidate=center,
        max_rewrites=0,
    ) == {"optimized_center_size_offset_terminal_transpose_chains": 0}
    assert optimize_center_size_offset_terminal_transpose_chains(
        model_ir,
        graph_index=graph_index,
        candidate=center,
        max_rewrites=1,
    ) == {"optimized_center_size_offset_terminal_transpose_chains": 1}
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert optimize_center_size_offset_terminal_transpose_chains(
        model_ir,
        graph_index=graph_index,
    ) == {"optimized_center_size_offset_terminal_transpose_chains": 0}


def test_layout_rewrite_is_numerically_equivalent() -> None:
    rng = np.random.default_rng(20260715)
    center = rng.normal(size=[1, 2, 3, 1]).astype(np.float32)
    size = rng.normal(size=[1, 2, 3, 2]).astype(np.float32)
    offset = rng.normal(size=[1, 2, 3, 2]).astype(np.float32)
    axis = np.asarray([0, 5], dtype=np.int64)
    channel = np.asarray([0, 1], dtype=np.int64)

    def sigmoid(value: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-value))

    center_old = np.clip(
        sigmoid(np.transpose(center, [0, 3, 1, 2])),
        0.0,
        1.0,
    ).reshape(1, 6)
    center_new = np.clip(sigmoid(center), 0.0, 1.0).reshape(1, 6)
    size_old_data = np.clip(
        sigmoid(np.transpose(size, [0, 3, 1, 2])),
        0.0,
        1.0,
    ).reshape(1, 2, 6)
    size_new_data = np.clip(sigmoid(size), 0.0, 1.0).reshape(1, 6, 2)
    offset_old_data = np.transpose(offset, [0, 3, 1, 2]).reshape(1, 2, 6)
    offset_new_data = offset.reshape(1, 6, 2)
    size_old = np.asarray(
        [size_old_data[0, channel[index], axis[index]] for index in range(2)]
    )
    size_new = np.asarray(
        [size_new_data[0, axis[index], channel[index]] for index in range(2)]
    )
    offset_old = np.asarray(
        [offset_old_data[0, channel[index], axis[index]] for index in range(2)]
    )
    offset_new = np.asarray(
        [offset_new_data[0, axis[index], channel[index]] for index in range(2)]
    )

    np.testing.assert_allclose(center_new, center_old, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(size_new, size_old, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(offset_new, offset_old, rtol=0.0, atol=0.0)


Mutation = Callable[[ModelIR], None]


def _add_consumer(model_ir: ModelIR, source: str, suffix: str) -> None:
    source_tensor = model_ir.tensors[source]
    output = f"{source}_{suffix}"
    model_ir.tensors[output] = copy.deepcopy(source_tensor)
    model_ir.tensors[output].name = output
    model_ir.operators.append(OperatorIR("RELU", [source], [output]))


def _wrong_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _center_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "center_nchw", "fanout")


def _public_center_intermediate(model_ir: ModelIR) -> None:
    model_ir.outputs.append("center_sig")


def _center_channel_not_singleton(model_ir: ModelIR) -> None:
    for name in ("center_nhwc", "center_nchw", "center_sig", "center_max", "center_min"):
        tensor = model_ir.tensors[name]
        if name == "center_nhwc":
            tensor.shape[-1] = 2
            tensor.shape_signature[-1] = 2
        else:
            tensor.shape[1] = 2
            tensor.shape_signature[1] = 2


def _dynamic_signature_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["size_nchw"].shape_signature = [1, 2, 3, -1]


def _duplicate_pre_output(model_ir: ModelIR) -> None:
    model_ir.operators.insert(3, OperatorIR("RELU", ["center_nhwc"], ["center_nchw"]))


def _wrong_center_activation(model_ir: ModelIR) -> None:
    next(op for op in model_ir.operators if op.outputs == ["center_sig"]).op_type = "TANH"


def _mutable_clip_constant(model_ir: ModelIR) -> None:
    model_ir.tensors["clip_min"].is_variable = True


def _per_axis_clip_constant(model_ir: ModelIR) -> None:
    model_ir.tensors["clip_min"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=0,
    )


def _size_chain_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "size_max", "fanout")


def _wrong_size_shape(model_ir: ModelIR) -> None:
    model_ir.tensors["size_shape"].data = np.asarray([1, 3, 4], dtype=np.int32)


def _different_offset_shape(model_ir: ModelIR) -> None:
    model_ir.tensors["offset_shape"].data = np.asarray([1, 1, 12], dtype=np.int32)


def _public_size_reshape(model_ir: ModelIR) -> None:
    model_ir.outputs.append("size_reshaped")


def _size_reshape_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "size_reshaped", "fanout")


def _concat_axis_mismatch(model_ir: ModelIR) -> None:
    next(op for op in model_ir.operators if op.outputs == ["coords_size"]).options["axis"] = 2


def _coords_public(model_ir: ModelIR) -> None:
    model_ir.outputs.append("coords_size")


def _coords_fanout(model_ir: ModelIR) -> None:
    _add_consumer(model_ir, "coords_size", "fanout")


def _ambiguous_coordinate_constants(model_ir: ModelIR) -> None:
    model_ir.tensors["batch_coord"].data = np.asarray(
        [[[[0]], [[1]]]],
        dtype=np.int32,
    )


def _axis_producer_mismatch(model_ir: ModelIR) -> None:
    next(op for op in model_ir.operators if op.outputs == ["axis_coord"]).op_type = "EXPAND_DIMS"


def _gather_input_mismatch(model_ir: ModelIR) -> None:
    gather = next(op for op in model_ir.operators if op.outputs == ["gather_size"])
    gather.inputs[0] = "offset_reshaped"


def _gather_output_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["gather_size"].shape = [1, 1, 2]


def _stale_graph_order(model_ir: ModelIR) -> None:
    logistic = next(op for op in model_ir.operators if op.outputs == ["center_sig"])
    model_ir.operators.remove(logistic)
    model_ir.operators.insert(0, logistic)


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_permutation,
        _center_fanout,
        _public_center_intermediate,
        _center_channel_not_singleton,
        _dynamic_signature_mismatch,
        _duplicate_pre_output,
        _wrong_center_activation,
        _mutable_clip_constant,
        _per_axis_clip_constant,
        _size_chain_fanout,
        _wrong_size_shape,
        _different_offset_shape,
        _public_size_reshape,
        _size_reshape_fanout,
        _concat_axis_mismatch,
        _coords_public,
        _coords_fanout,
        _ambiguous_coordinate_constants,
        _axis_producer_mismatch,
        _gather_input_mismatch,
        _gather_output_shape_mismatch,
        _stale_graph_order,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_unsafe_candidate_is_transactional_noop(mutation: Mutation) -> None:
    model_ir = _make_model()
    mutation(model_ir)
    before = pickle.dumps(model_ir)

    stats = optimize_center_size_offset_terminal_transpose_chains(model_ir)

    assert stats == {"optimized_center_size_offset_terminal_transpose_chains": 0}
    assert pickle.dumps(model_ir) == before
