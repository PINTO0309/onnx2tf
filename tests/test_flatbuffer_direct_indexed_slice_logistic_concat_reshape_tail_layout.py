from __future__ import annotations

import copy
from typing import Callable, Iterable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains as compatibility_dispatcher,
)
from onnx2tf.tflite_builder.passes.slice_logistic_concat_reshape_tail_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains,
)


_STATS_KEY = "optimized_transpose_slice_logistic_concat_reshape_tail_nhwc_chains"


def _add_tensor(
    model_ir: ModelIR,
    name: str,
    dtype: str,
    shape: Iterable[int],
    *,
    signature: Iterable[int] | None = None,
    data: np.ndarray | None = None,
    logical_layout: str = "UNKNOWN",
    physical_layout: str = "UNKNOWN",
) -> None:
    normalized_shape = [int(value) for value in shape]
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=normalized_shape,
        shape_signature=(
            list(normalized_shape)
            if signature is None
            else [int(value) for value in signature]
        ),
        data=None if data is None else np.asarray(data),
        is_variable=False,
        logical_layout=logical_layout,
        physical_layout=physical_layout,
    )


def _make_model(
    *,
    group_count: int = 1,
    branch_shapes: tuple[tuple[int, int], ...] = ((4, 5), (2, 3)),
    unary_slots: frozenset[int] = frozenset({0}),
    constant_dtype: np.dtype = np.dtype(np.int32),
    dynamic: bool = False,
    full_extent: bool = False,
    existing_post_permutation: bool = True,
) -> ModelIR:
    model_ir = ModelIR("indexed_slice_logistic_concat_reshape_tail")
    int_dtype = "INT64" if constant_dtype == np.dtype(np.int64) else "INT32"
    _add_tensor(
        model_ir,
        "perm4",
        int_dtype,
        [4],
        data=np.asarray([0, 3, 1, 2], dtype=constant_dtype),
    )
    if existing_post_permutation:
        _add_tensor(
            model_ir,
            "perm3",
            "INT32",
            [3],
            data=np.asarray([0, 2, 1], dtype=np.int32),
        )

    operators: list[OperatorIR] = []
    public_outputs = []
    for group_index in range(group_count):
        tail_inputs = []
        spatial_total = 0
        for branch_index, (height, width) in enumerate(branch_shapes):
            prefix = f"g{group_index}_b{branch_index}"
            spatial = height * width
            spatial_total += spatial
            source = f"{prefix}_nhwc"
            pre_output = f"{prefix}_nchw"
            source_signature = [1, -1, -1, 5] if dynamic else [1, height, width, 5]
            pre_signature = [1, 5, -1, -1] if dynamic else [1, 5, height, width]
            _add_tensor(
                model_ir,
                source,
                "FLOAT32",
                [1, height, width, 5],
                signature=source_signature,
                logical_layout=LOGICAL_LAYOUT_NHWC,
                physical_layout=LOGICAL_LAYOUT_NHWC,
            )
            _add_tensor(
                model_ir,
                pre_output,
                "FLOAT32",
                [1, 5, height, width],
                signature=pre_signature,
                logical_layout=LOGICAL_LAYOUT_NCHW,
                physical_layout=LOGICAL_LAYOUT_NCHW,
            )
            model_ir.inputs.append(source)
            operators.append(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[source, "perm4"],
                    outputs=[pre_output],
                )
            )

            concat_inputs = []
            for slice_index, (channel_begin, channels) in enumerate(((0, 2), (2, 3))):
                slice_output = f"{prefix}_slice{slice_index}"
                begin_name = f"{prefix}_begin{slice_index}"
                size_name = f"{prefix}_size{slice_index}"
                output_signature = (
                    [1, channels, -1, -1] if dynamic else [1, channels, height, width]
                )
                _add_tensor(
                    model_ir,
                    slice_output,
                    "FLOAT32",
                    [1, channels, height, width],
                    signature=output_signature,
                    logical_layout=LOGICAL_LAYOUT_NCHW,
                    physical_layout=LOGICAL_LAYOUT_NCHW,
                )
                _add_tensor(
                    model_ir,
                    begin_name,
                    int_dtype,
                    [4],
                    data=np.asarray(
                        [0, channel_begin, 0, 0],
                        dtype=constant_dtype,
                    ),
                )
                size_values = (
                    [-1, channels, -1, -1]
                    if full_extent
                    else [1, channels, height, width]
                )
                _add_tensor(
                    model_ir,
                    size_name,
                    int_dtype,
                    [4],
                    data=np.asarray(size_values, dtype=constant_dtype),
                )
                operators.append(
                    OperatorIR(
                        op_type="SLICE",
                        inputs=[pre_output, begin_name, size_name],
                        outputs=[slice_output],
                    )
                )
                concat_input = slice_output
                if slice_index in unary_slots:
                    concat_input = f"{prefix}_sig{slice_index}"
                    _add_tensor(
                        model_ir,
                        concat_input,
                        "FLOAT32",
                        [1, channels, height, width],
                        signature=output_signature,
                        logical_layout=LOGICAL_LAYOUT_NCHW,
                        physical_layout=LOGICAL_LAYOUT_NCHW,
                    )
                    operators.append(
                        OperatorIR(
                            op_type="LOGISTIC",
                            inputs=[slice_output],
                            outputs=[concat_input],
                        )
                    )
                concat_inputs.append(concat_input)

            concat_output = f"{prefix}_concat"
            reshape_output = f"{prefix}_reshape"
            shape_name = f"{prefix}_reshape_shape"
            _add_tensor(
                model_ir,
                concat_output,
                "FLOAT32",
                [1, 5, height, width],
                signature=pre_signature,
                logical_layout=LOGICAL_LAYOUT_NCHW,
                physical_layout=LOGICAL_LAYOUT_NCHW,
            )
            reshape_signature = [1, 5, -1] if dynamic else [1, 5, spatial]
            _add_tensor(
                model_ir,
                reshape_output,
                "FLOAT32",
                [1, 5, spatial],
                signature=reshape_signature,
                logical_layout=LOGICAL_LAYOUT_NCHW,
                physical_layout=LOGICAL_LAYOUT_NCHW,
            )
            _add_tensor(
                model_ir,
                shape_name,
                int_dtype,
                [3],
                data=np.asarray([1, 5, spatial], dtype=constant_dtype),
            )
            operators.extend(
                [
                    OperatorIR(
                        op_type="CONCATENATION",
                        inputs=concat_inputs,
                        outputs=[concat_output],
                        options={
                            "axis": 1,
                            "fusedActivationFunction": "NONE",
                        },
                    ),
                    OperatorIR(
                        op_type="RESHAPE",
                        inputs=[concat_output, shape_name],
                        outputs=[reshape_output],
                        options={
                            "newShape": [1, 5, spatial],
                            "onnxRawNewShape": [1, 5, -1],
                            "allowZero": False,
                        },
                    ),
                ]
            )
            tail_inputs.append(reshape_output)

        tail_output = f"g{group_index}_tail"
        canonical_public = f"g{group_index}_public"
        tail_signature = [1, 5, -1] if dynamic else [1, 5, spatial_total]
        public_signature = [1, -1, 5] if dynamic else [1, spatial_total, 5]
        _add_tensor(
            model_ir,
            tail_output,
            "FLOAT32",
            [1, 5, spatial_total],
            signature=tail_signature,
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
        _add_tensor(
            model_ir,
            canonical_public,
            "FLOAT32",
            [1, spatial_total, 5],
            signature=public_signature,
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        )
        operators.append(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=tail_inputs,
                outputs=[tail_output],
                options={"axis": 2, "fusedActivationFunction": "NONE"},
            )
        )
        post_perm = "perm3"
        if not existing_post_permutation:
            post_perm = f"g{group_index}_existing_post_perm"
            _add_tensor(
                model_ir,
                post_perm,
                "INT64",
                [3],
                data=np.asarray([0, 2, 1], dtype=np.int64),
            )
        operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[tail_output, post_perm],
                outputs=[canonical_public],
            )
        )
        public_outputs.append(canonical_public)

    model_ir.operators = operators
    model_ir.outputs = public_outputs
    return model_ir


def _snapshot(model_ir: ModelIR) -> tuple:
    tensors = []
    for name, tensor in model_ir.tensors.items():
        data = None
        if tensor.data is not None:
            array = np.asarray(tensor.data)
            data = (str(array.dtype), tuple(array.shape), array.tobytes())
        quantization = tensor.quantization
        if isinstance(quantization, QuantParamIR):
            quantization = (
                tuple(quantization.scale),
                tuple(quantization.zero_point),
                quantization.quantized_dimension,
            )
        tensors.append(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                None
                if tensor.shape_signature is None
                else tuple(tensor.shape_signature),
                data,
                bool(tensor.is_variable),
                repr(quantization),
                tensor.logical_layout,
                tensor.physical_layout,
                tensor.onnx_tensor_name,
            )
        )
    operators = tuple(
        (
            operator.op_type,
            tuple(operator.inputs),
            tuple(operator.outputs),
            repr(operator.options),
            repr(operator.axis_semantics),
            operator.version,
            operator.onnx_node_name,
            operator.onnx_op_type,
        )
        for operator in model_ir.operators
    )
    return (
        tuple(tensors),
        operators,
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        repr(model_ir.metadata),
    )


def _evaluate(model_ir: ModelIR, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    values = {name: np.asarray(value) for name, value in feeds.items()}
    for name, tensor in model_ir.tensors.items():
        if tensor.data is not None:
            values[name] = np.asarray(tensor.data)
    for operator in model_ir.operators:
        inputs = [values[name] for name in operator.inputs]
        if operator.op_type == "TRANSPOSE":
            outputs = [np.transpose(inputs[0], axes=inputs[1].reshape(-1))]
        elif operator.op_type == "SLICE":
            begin = inputs[1].reshape(-1)
            size = inputs[2].reshape(-1)
            slices = tuple(
                slice(
                    int(start),
                    None if int(extent) == -1 else int(start) + int(extent),
                )
                for start, extent in zip(begin, size)
            )
            outputs = [inputs[0][slices]]
        elif operator.op_type == "LOGISTIC":
            outputs = [1.0 / (1.0 + np.exp(-inputs[0]))]
        elif operator.op_type == "CONCATENATION":
            outputs = [np.concatenate(inputs, axis=int(operator.options["axis"]))]
        elif operator.op_type == "RESHAPE":
            shape = tuple(int(value) for value in inputs[1].reshape(-1))
            outputs = [np.reshape(inputs[0], shape)]
        elif operator.op_type == "RELU":
            outputs = [np.maximum(inputs[0], 0)]
        elif operator.op_type == "IDENTITY":
            outputs = [inputs[0]]
        else:
            raise AssertionError(f"unsupported test evaluator op: {operator.op_type}")
        for name, value in zip(operator.outputs, outputs):
            values[name] = value
    return [values[name] for name in model_ir.outputs]


def _feeds(model_ir: ModelIR) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(7)
    return {
        name: generator.normal(size=model_ir.tensors[name].shape).astype(np.float32)
        for name in model_ir.inputs
    }


def _tail(model_ir: ModelIR, group_index: int = 0) -> OperatorIR:
    return next(
        operator
        for operator in model_ir.operators
        if operator.outputs == [f"g{group_index}_tail"]
    )


def test_indexed_active_group_is_numerically_equivalent_and_updates_index_layout() -> (
    None
):
    model_ir = _make_model()
    feeds = _feeds(model_ir)
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats[_STATS_KEY] == 1
    actual = _evaluate(model_ir, feeds)
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-6, atol=1e-6)
    assert graph_index.model_ir is model_ir
    assert graph_index.operator_indices("TRANSPOSE") == [
        index
        for index, operator in enumerate(model_ir.operators)
        if operator.op_type == "TRANSPOSE"
    ]
    assert all(
        operator.inputs[0].endswith("_nhwc")
        for operator in model_ir.operators
        if operator.op_type == "SLICE"
    )
    assert all(
        int(operator.options["axis"]) == 3
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
        and operator.outputs[0].endswith("_concat")
    )
    rewritten_tail = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION" and operator.outputs[0] == "g0_tail_nhwc"
    )
    assert rewritten_tail.options["axis"] == 1
    assert layout_state.physical_of("g0_tail_nhwc") == LOGICAL_LAYOUT_NHWC
    assert layout_state.physical_of("g0_tail") == LOGICAL_LAYOUT_NCHW


@pytest.mark.parametrize(
    "unary_slots",
    [frozenset(), frozenset({0}), frozenset({1}), frozenset({0, 1})],
)
def test_all_historical_optional_logistic_forms_are_supported(
    unary_slots: frozenset[int],
) -> None:
    model_ir = _make_model(unary_slots=unary_slots)
    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)
    assert stats[_STATS_KEY] == 1
    assert sum(operator.op_type == "LOGISTIC" for operator in model_ir.operators) == (
        len(unary_slots) * 2
    )


@pytest.mark.parametrize("constant_dtype", [np.dtype(np.int32), np.dtype(np.int64)])
def test_dynamic_full_extent_and_typed_constants_are_preserved(
    constant_dtype: np.dtype,
) -> None:
    model_ir = _make_model(
        constant_dtype=constant_dtype,
        dynamic=True,
        full_extent=True,
    )
    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)
    assert stats[_STATS_KEY] == 1
    assert model_ir.tensors["g0_b0_slice0"].shape_signature == [1, -1, -1, 2]
    assert model_ir.tensors["g0_b0_reshape"].shape_signature == [1, -1, 5]
    assert model_ir.tensors["g0_tail_nhwc"].shape_signature == [1, -1, 5]
    assert np.asarray(model_ir.tensors["g0_b0_size0"].data).dtype == constant_dtype
    assert np.asarray(model_ir.tensors["g0_b0_size0"].data).tolist() == [
        -1,
        -1,
        -1,
        2,
    ]


def test_shared_reshape_shape_constant_is_cloned_once() -> None:
    model_ir = _make_model()
    shape_name = "g0_b0_reshape_shape"
    _add_tensor(model_ir, "legacy_data", "FLOAT32", [1, 5, 20])
    _add_tensor(model_ir, "legacy_output", "FLOAT32", [1, 5, 20])
    model_ir.inputs.append("legacy_data")
    model_ir.outputs.append("legacy_output")
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["legacy_data", shape_name],
            outputs=["legacy_output"],
            options={"newShape": [1, 5, 20]},
        )
    )
    original = np.asarray(model_ir.tensors[shape_name].data).copy()

    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)

    assert stats[_STATS_KEY] == 1
    np.testing.assert_array_equal(model_ir.tensors[shape_name].data, original)
    reshape = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["g0_b0_reshape"]
    )
    assert reshape.inputs[1] == f"{shape_name}_nhwc"
    assert np.asarray(model_ir.tensors[reshape.inputs[1]].data).tolist() == [1, 20, 5]


def test_existing_typed_int32_post_permutation_is_reused() -> None:
    model_ir = _make_model(existing_post_permutation=True)
    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)
    assert stats[_STATS_KEY] == 1
    terminal = next(
        operator for operator in model_ir.operators if operator.outputs == ["g0_tail"]
    )
    assert terminal.inputs[1] == "perm3"
    assert "transpose_tail_3d_nhwc_to_nchw_perm" not in model_ir.tensors


def test_independent_pre_adapters_may_share_one_nhwc_source() -> None:
    model_ir = _make_model(branch_shapes=((4, 5), (4, 5)))
    second_pre = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["g0_b1_nchw"]
    )
    second_pre.inputs[0] = "g0_b0_nhwc"
    model_ir.inputs.remove("g0_b1_nhwc")
    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)
    assert stats[_STATS_KEY] == 1
    assert all(
        operator.inputs[0] == "g0_b0_nhwc"
        for operator in model_ir.operators
        if operator.op_type == "SLICE"
    )


def test_missing_safe_post_permutation_creates_one_int32_buffer() -> None:
    model_ir = _make_model(existing_post_permutation=False)
    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)
    assert stats[_STATS_KEY] == 1
    name = "transpose_tail_3d_nhwc_to_nchw_perm"
    assert model_ir.tensors[name].dtype == "INT32"
    assert np.asarray(model_ir.tensors[name].data).dtype == np.dtype(np.int32)
    assert np.asarray(model_ir.tensors[name].data).tolist() == [0, 2, 1]


def test_compatibility_dispatcher_uses_caller_index_and_layout_state() -> None:
    model_ir = _make_model()
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    stats = compatibility_dispatcher(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=1,
        candidate=_tail(model_ir),
    )
    assert stats[_STATS_KEY] == 1
    assert graph_index.producer("g0_tail") is not None
    assert layout_state.physical_of("g0_tail_nhwc") == LOGICAL_LAYOUT_NHWC


def test_candidate_and_max_rewrites_bound_multiple_groups() -> None:
    model_ir = _make_model(group_count=2)
    second = _tail(model_ir, 1)
    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(
        model_ir,
        candidate=second,
        max_rewrites=1,
    )
    assert stats[_STATS_KEY] == 1
    assert _tail(model_ir, 0).outputs == ["g0_tail"]
    assert any(operator.outputs == ["g1_tail_nhwc"] for operator in model_ir.operators)

    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(
        model_ir,
        max_rewrites=1,
    )
    assert stats[_STATS_KEY] == 1
    assert any(operator.outputs == ["g0_tail_nhwc"] for operator in model_ir.operators)


def test_stale_plan_is_revalidated_before_any_mutation() -> None:
    model_ir = _make_model()
    graph_index = ModelIRGraphIndex(model_ir)
    plan = _resolve_candidate(
        model_ir,
        graph_index,
        _tail(model_ir),
        layout_state=None,
    )
    assert plan is not None
    model_ir.tensors["g0_b1_size1"].data = np.asarray(
        [1, 2, 3, 2],
        dtype=np.int32,
    )
    before = _snapshot(model_ir)
    assert not _apply_plan(model_ir, graph_index, plan, layout_state=None)
    assert _snapshot(model_ir) == before


def test_zero_match_still_prunes_unused_tensors() -> None:
    model_ir = ModelIR("zero_match")
    _add_tensor(model_ir, "input", "FLOAT32", [1])
    _add_tensor(model_ir, "unused", "FLOAT32", [1])
    model_ir.inputs = ["input"]
    model_ir.outputs = ["input"]
    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)
    assert stats[_STATS_KEY] == 0
    assert "unused" not in model_ir.tensors


def test_copy_isolation_and_repeated_sweeps_are_deterministic() -> None:
    first = _make_model()
    second = copy.deepcopy(first)
    assert (
        optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(first)[
            _STATS_KEY
        ]
        == 1
    )
    assert (
        optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(second)[
            _STATS_KEY
        ]
        == 1
    )
    assert _snapshot(first) == _snapshot(second)
    stable = _snapshot(first)
    assert (
        optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(first)[
            _STATS_KEY
        ]
        == 0
    )
    assert _snapshot(first) == stable


def _tail_wrong_axis(model_ir: ModelIR) -> None:
    _tail(model_ir).options["axis"] = 1


def _tail_repeated_input(model_ir: ModelIR) -> None:
    tail = _tail(model_ir)
    tail.inputs[1] = tail.inputs[0]


def _branch_wrong_axis(model_ir: ModelIR) -> None:
    next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["g0_b0_concat"]
    ).options["axis"] = 3


def _branch_repeated_input(model_ir: ModelIR) -> None:
    concat = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["g0_b0_concat"]
    )
    concat.inputs[1] = concat.inputs[0]


def _wrong_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["perm4"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _explicit_wrong_source_layout(model_ir: ModelIR) -> None:
    model_ir.tensors["g0_b0_nhwc"].physical_layout = LOGICAL_LAYOUT_NCHW


def _per_axis_quantization(model_ir: ModelIR) -> None:
    model_ir.tensors["g0_b0_nhwc"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=3,
    )


def _float_slice_constant(model_ir: ModelIR) -> None:
    tensor = model_ir.tensors["g0_b0_begin0"]
    tensor.dtype = "FLOAT32"
    tensor.data = np.asarray(tensor.data, dtype=np.float32)


def _produced_slice_constant(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        0,
        OperatorIR(
            op_type="IDENTITY",
            inputs=["g0_b0_begin1"],
            outputs=["g0_b0_begin0"],
        ),
    )


def _shared_slice_constant(model_ir: ModelIR) -> None:
    _add_tensor(model_ir, "constant_alias", "INT32", [4])
    model_ir.operators.append(
        OperatorIR(
            op_type="IDENTITY",
            inputs=["g0_b0_begin0"],
            outputs=["constant_alias"],
        )
    )
    model_ir.outputs.append("constant_alias")


def _same_begin_and_size(model_ir: ModelIR) -> None:
    slice_op = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["g0_b0_slice0"]
    )
    slice_op.inputs[2] = slice_op.inputs[1]
    model_ir.outputs.append("g0_b0_size0")


def _invalid_spatial_begin(model_ir: ModelIR) -> None:
    model_ir.tensors["g0_b0_begin0"].data = np.asarray(
        [0, 0, 1, 0],
        dtype=np.int32,
    )


def _invalid_channel_size(model_ir: ModelIR) -> None:
    model_ir.tensors["g0_b0_size0"].data = np.asarray(
        [1, 0, 4, 5],
        dtype=np.int32,
    )


def _slice_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["g0_b0_slice0"].shape[1] = 1


def _reshape_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["g0_b0_reshape_shape"].data = np.asarray(
        [1, 4, 25],
        dtype=np.int32,
    )


def _reshape_option_mismatch(model_ir: ModelIR) -> None:
    reshape = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["g0_b0_reshape"]
    )
    reshape.options["newShape"] = [1, 4, 25]


def _reshape_zero_dimension(model_ir: ModelIR) -> None:
    reshape = next(
        operator
        for operator in model_ir.operators
        if operator.outputs == ["g0_b0_reshape"]
    )
    reshape.options["newShape"] = [0, 5, 20]


def _tail_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["g0_tail"].shape[2] -= 1


def _pre_fanout(model_ir: ModelIR) -> None:
    _add_tensor(model_ir, "pre_fanout", "FLOAT32", [1, 5, 4, 5])
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["g0_b0_nchw"],
            outputs=["pre_fanout"],
        )
    )
    model_ir.outputs.append("pre_fanout")


def _slice_fanout(model_ir: ModelIR) -> None:
    _add_tensor(model_ir, "slice_fanout", "FLOAT32", [1, 2, 4, 5])
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["g0_b0_slice0"],
            outputs=["slice_fanout"],
        )
    )
    model_ir.outputs.append("slice_fanout")


def _reshape_fanout(model_ir: ModelIR) -> None:
    _add_tensor(model_ir, "reshape_fanout", "FLOAT32", [1, 5, 20])
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["g0_b0_reshape"],
            outputs=["reshape_fanout"],
        )
    )
    model_ir.outputs.append("reshape_fanout")


def _public_branch_intermediate(model_ir: ModelIR) -> None:
    model_ir.outputs.append("g0_b0_concat")


def _backward_tail_consumer(model_ir: ModelIR) -> None:
    _add_tensor(model_ir, "backward_tail", "FLOAT32", [1, 5, 26])
    model_ir.operators.insert(
        0,
        OperatorIR(
            op_type="RELU",
            inputs=["g0_tail"],
            outputs=["backward_tail"],
        ),
    )
    model_ir.outputs.append("backward_tail")


def _duplicate_slice_producer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        0,
        OperatorIR(
            op_type="IDENTITY",
            inputs=["g0_b0_nhwc"],
            outputs=["g0_b0_slice0"],
        ),
    )


_REJECTIONS: tuple[Callable[[ModelIR], None], ...] = (
    _tail_wrong_axis,
    _tail_repeated_input,
    _branch_wrong_axis,
    _branch_repeated_input,
    _wrong_pre_permutation,
    _explicit_wrong_source_layout,
    _per_axis_quantization,
    _float_slice_constant,
    _produced_slice_constant,
    _shared_slice_constant,
    _same_begin_and_size,
    _invalid_spatial_begin,
    _invalid_channel_size,
    _slice_shape_mismatch,
    _reshape_shape_mismatch,
    _reshape_option_mismatch,
    _reshape_zero_dimension,
    _tail_shape_mismatch,
    _pre_fanout,
    _slice_fanout,
    _reshape_fanout,
    _public_branch_intermediate,
    _backward_tail_consumer,
    _duplicate_slice_producer,
)


@pytest.mark.parametrize("mutate", _REJECTIONS, ids=lambda value: value.__name__)
def test_unsafe_candidates_are_rejected_atomically(
    mutate: Callable[[ModelIR], None],
) -> None:
    model_ir = _make_model()
    mutate(model_ir)
    before = _snapshot(model_ir)
    stats = optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(model_ir)
    assert stats[_STATS_KEY] == 0
    assert _snapshot(model_ir) == before
