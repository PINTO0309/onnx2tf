from __future__ import annotations

import ast
import copy
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.split_conv_concat_bridge_layout import (
    optimize_split_conv_concat_transpose_bridge_to_single_post_nchw,
)


_N = 1
_H = 3
_W = 5
_SPLIT_CHANNELS = 4
REPO_ROOT = Path(__file__).resolve().parents[1]
TERMINAL_SPLIT_CONV_CONCAT_BRIDGE_OWNER = (
    "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
)
TERMINAL_ACTIVATION_BRIDGE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_activation_bridge_orchestration.py"
)
TERMINAL_ACTIVATION_BRIDGE_OWNER = (
    "run_terminal_activation_bridge_cleanup"
)
OUTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_activation_bridge_orchestration.py"
)
OUTER_OWNER = "run_terminal_qkv_activation_bridge_cleanup"
OUTER_RESULT = "_terminal_qkv_activation_bridge_results"
TOP_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_qkv_activation_layout_shape_orchestration.py"
)
TOP_OWNER = "run_terminal_qkv_activation_layout_shape_cleanup"
TOP_RESULT = "_terminal_qkv_activation_layout_shape_results"
LOWERER_OWNER = "run_terminal_affine_qkv_layout_shape_cleanup"
LOWERER_RESULT = "_terminal_affine_qkv_layout_shape_results"
TERMINAL_SINGLETON_CLAMP_SINET_OWNER = (
    "run_terminal_singleton_clamp_sinet_cleanup"
)
TERMINAL_SINGLETON_CLAMP_SINET_RESULT = (
    "_terminal_singleton_clamp_sinet_results"
)
PUBLIC_SPLIT_CONV_CONCAT_BRIDGE_OWNER = (
    "optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
    dtype: str = "FLOAT32",
    is_variable: bool = False,
    quantization: QuantParamIR | None = None,
    layout: str = LOGICAL_LAYOUT_UNKNOWN,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=is_variable,
        quantization=quantization,
        logical_layout=layout,
        physical_layout=layout,
    )


def _make_model(
    *,
    split_count: int = 3,
    branch_output_index: int = 1,
    post_count: int = 2,
    reverse_concat: bool = False,
    dynamic: bool = False,
    integer_dtype: str = "INT32",
    negative_axis: bool = False,
    shared_axis: bool = False,
    branch_side_consumer: bool = False,
) -> ModelIR:
    source_channels = int(split_count) * _SPLIT_CHANNELS
    conv_channels = [3, 2][: int(post_count)]
    output_channels = (int(split_count) - 1) * _SPLIT_CHANNELS + sum(
        conv_channels
    )
    numpy_integer_dtype = np.int64 if integer_dtype == "INT64" else np.int32
    source_nhwc = [_N, _H, _W, source_channels]
    source_nchw = [_N, source_channels, _H, _W]
    split_nhwc = [_N, _H, _W, _SPLIT_CHANNELS]
    split_nchw = [_N, _SPLIT_CHANNELS, _H, _W]
    output_nchw = [_N, output_channels, _H, _W]

    def _nhwc_signature(channels: int) -> list[int]:
        return [_N, -1, -1, channels] if dynamic else [_N, _H, _W, channels]

    def _nchw_signature(channels: int) -> list[int]:
        return [_N, channels, -1, -1] if dynamic else [_N, channels, _H, _W]

    model_ir = ModelIR("indexed_split_conv_concat_bridge")
    model_ir.inputs = ["src_nhwc"]
    model_ir.outputs = ["sink"]
    model_ir.tensors = {
        "src_nhwc": _tensor(
            "src_nhwc",
            source_nhwc,
            signature=_nhwc_signature(source_channels),
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=numpy_integer_dtype),
            dtype=integer_dtype,
        ),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=numpy_integer_dtype),
            dtype=integer_dtype,
        ),
        "axis": _tensor(
            "axis",
            [1],
            data=np.asarray(
                [-3 if negative_axis else 1],
                dtype=numpy_integer_dtype,
            ),
            dtype=integer_dtype,
        ),
        "pre_nchw": _tensor(
            "pre_nchw",
            source_nchw,
            signature=_nchw_signature(source_channels),
            layout=LOGICAL_LAYOUT_NCHW,
        ),
    }
    split_outputs = [f"split_{index}_nchw" for index in range(split_count)]
    for name in split_outputs:
        model_ir.tensors[name] = _tensor(
            name,
            split_nchw,
            signature=_nchw_signature(_SPLIT_CHANNELS),
            layout=LOGICAL_LAYOUT_NCHW,
        )
    branch_name = split_outputs[int(branch_output_index)]
    model_ir.tensors["branch_nhwc"] = _tensor(
        "branch_nhwc",
        split_nhwc,
        signature=_nhwc_signature(_SPLIT_CHANNELS),
        layout=LOGICAL_LAYOUT_NHWC,
    )
    operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["src_nhwc", "to_nchw"],
            outputs=["pre_nchw"],
        ),
        OperatorIR(
            op_type="SPLIT",
            inputs=["axis", "pre_nchw"],
            outputs=split_outputs,
            options={"numSplits": int(split_count)},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[branch_name, "to_nhwc"],
            outputs=["branch_nhwc"],
        ),
    ]
    post_outputs = []
    for index, channels in enumerate(conv_channels):
        filter_data = (
            np.arange(channels * _SPLIT_CHANNELS, dtype=np.float32)
            .reshape(channels, 1, 1, _SPLIT_CHANNELS)
            / float(13 + index)
        )
        bias_data = np.linspace(-0.25, 0.25, channels, dtype=np.float32)
        filter_name = f"filter_{index}"
        bias_name = f"bias_{index}"
        conv_name = f"conv_{index}_nhwc"
        post_name = f"conv_{index}_nchw"
        model_ir.tensors[filter_name] = _tensor(
            filter_name,
            list(filter_data.shape),
            data=filter_data,
        )
        model_ir.tensors[bias_name] = _tensor(
            bias_name,
            [channels],
            data=bias_data,
        )
        model_ir.tensors[conv_name] = _tensor(
            conv_name,
            [_N, _H, _W, channels],
            signature=_nhwc_signature(channels),
            layout=LOGICAL_LAYOUT_NHWC,
        )
        model_ir.tensors[post_name] = _tensor(
            post_name,
            [_N, channels, _H, _W],
            signature=_nchw_signature(channels),
            layout=LOGICAL_LAYOUT_NCHW,
        )
        operators.extend(
            [
                OperatorIR(
                    op_type="CONV_2D",
                    inputs=["branch_nhwc", filter_name, bias_name],
                    outputs=[conv_name],
                    options={
                        "padding": "VALID",
                        "strideH": 1,
                        "strideW": 1,
                        "fusedActivationFunction": "NONE",
                    },
                ),
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[conv_name, "to_nchw"],
                    outputs=[post_name],
                ),
            ]
        )
        post_outputs.append(post_name)
    direct_outputs = [name for name in split_outputs if name != branch_name]
    concat_inputs = [*direct_outputs, *post_outputs]
    if reverse_concat:
        concat_inputs.reverse()
    model_ir.tensors["concat_nchw"] = _tensor(
        "concat_nchw",
        output_nchw,
        signature=_nchw_signature(output_channels),
        layout=LOGICAL_LAYOUT_NCHW,
    )
    model_ir.tensors["sink"] = _tensor(
        "sink",
        output_nchw,
        signature=_nchw_signature(output_channels),
        layout=LOGICAL_LAYOUT_NCHW,
    )
    operators.extend(
        [
            OperatorIR(
                op_type="CONCATENATION",
                inputs=concat_inputs,
                outputs=["concat_nchw"],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(op_type="RELU", inputs=["concat_nchw"], outputs=["sink"]),
        ]
    )
    if branch_side_consumer:
        model_ir.outputs.append("branch_side")
        model_ir.tensors["branch_side"] = _tensor(
            "branch_side",
            split_nhwc,
            signature=_nhwc_signature(_SPLIT_CHANNELS),
            layout=LOGICAL_LAYOUT_NHWC,
        )
        operators.append(
            OperatorIR(
                op_type="RELU6",
                inputs=["branch_nhwc"],
                outputs=["branch_side"],
            )
        )
    if shared_axis:
        model_ir.inputs.append("legacy_value")
        model_ir.outputs.append("legacy_keep")
        model_ir.tensors["legacy_value"] = _tensor("legacy_value", [1])
        model_ir.tensors["legacy_keep"] = _tensor("legacy_keep", [1])
        operators.append(
            OperatorIR(
                op_type="RESHAPE",
                inputs=["legacy_value", "axis"],
                outputs=["legacy_keep"],
            )
        )
    model_ir.operators = operators
    return model_ir


def _snapshot(model_ir: ModelIR) -> bytes:
    return pickle.dumps(model_ir, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("split_count", [2, 3])
@pytest.mark.parametrize("reverse_concat", [False, True])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("integer_dtype", "negative_axis"),
    [("INT32", False), ("INT64", True)],
    ids=["int32_positive", "int64_negative"],
)
def test_indexed_split_conv_concat_bridge_preserves_local_nchw_boundary(
    split_count: int,
    reverse_concat: bool,
    dynamic: bool,
    integer_dtype: str,
    negative_axis: bool,
) -> None:
    model_ir = _make_model(
        split_count=split_count,
        branch_output_index=split_count - 1,
        post_count=1 if split_count == 2 else 2,
        reverse_concat=reverse_concat,
        dynamic=dynamic,
        integer_dtype=integer_dtype,
        negative_axis=negative_axis,
    )
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_split_conv_concat_transpose_bridge_to_single_post_nchw": 1
    }
    transposes = [
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "TRANSPOSE"
    ]
    assert len(transposes) == 1
    assert list(transposes[0].outputs) == ["concat_nchw"]
    assert list(transposes[0].inputs) == ["concat_nchw__nhwc", "to_nchw"]
    split = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "SPLIT"
    )
    concat = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "CONCATENATION"
    )
    assert list(split.inputs) == ["axis", "src_nhwc"]
    assert int(concat.options["axis"]) == 3
    assert list(concat.outputs) == ["concat_nchw__nhwc"]
    assert list(model_ir.operators[-1].inputs) == ["concat_nchw"]
    for name in split.outputs:
        tensor = model_ir.tensors[str(name)]
        assert tensor.shape == [_N, _H, _W, _SPLIT_CHANNELS]
        assert tensor.logical_layout == LOGICAL_LAYOUT_NHWC
    for operator in model_ir.operators:
        if str(operator.op_type) == "CONV_2D":
            assert str(operator.inputs[0]) == str(split.outputs[-1])
    np.testing.assert_array_equal(
        model_ir.tensors["axis"].data,
        np.asarray([3], dtype=np.int64 if integer_dtype == "INT64" else np.int32),
    )
    assert model_ir.tensors["concat_nchw"].shape[1] > 0
    assert model_ir.tensors["concat_nchw"].physical_layout == LOGICAL_LAYOUT_NCHW
    assert model_ir.tensors["concat_nchw__nhwc"].physical_layout == LOGICAL_LAYOUT_NHWC
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("branch_output_index", [0, 2])
def test_indexed_split_conv_concat_bridge_accepts_either_split_branch(
    branch_output_index: int,
) -> None:
    model_ir = _make_model(
        branch_output_index=branch_output_index,
        reverse_concat=branch_output_index == 0,
    )

    stats = optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir
    )

    assert stats[
        "optimized_split_conv_concat_transpose_bridge_to_single_post_nchw"
    ] == 1
    split = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "SPLIT"
    )
    concat = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "CONCATENATION"
    )
    for operator in model_ir.operators:
        if str(operator.op_type) == "CONV_2D":
            assert str(operator.inputs[0]) == str(split.outputs[branch_output_index])
    expected_inputs = [
        name
        for index, name in enumerate(split.outputs)
        if index != branch_output_index
    ] + ["conv_0_nhwc", "conv_1_nhwc"]
    if branch_output_index == 0:
        expected_inputs.reverse()
    assert list(concat.inputs) == expected_inputs


def test_indexed_split_conv_concat_bridge_preserves_numerical_semantics() -> None:
    model_ir = _make_model(
        split_count=3,
        branch_output_index=1,
        post_count=2,
        reverse_concat=True,
    )
    rng = np.random.default_rng(83)
    source = rng.normal(size=(_N, _H, _W, 12)).astype(np.float32)
    old_split = np.split(np.transpose(source, (0, 3, 1, 2)), 3, axis=1)
    branch_nhwc = np.transpose(old_split[1], (0, 2, 3, 1))
    old_values = []
    new_values = []
    for index in range(2):
        weights = np.asarray(model_ir.tensors[f"filter_{index}"].data)[:, 0, 0, :]
        bias = np.asarray(model_ir.tensors[f"bias_{index}"].data)
        conv = np.einsum("nhwi,oi->nhwo", branch_nhwc, weights) + bias
        old_values.append(np.transpose(conv, (0, 3, 1, 2)))
        new_values.append(conv)
    old_concat_inputs = [old_split[0], old_split[2], *old_values]
    old_concat_inputs.reverse()
    expected = np.concatenate(old_concat_inputs, axis=1)

    stats = optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir
    )

    assert stats[
        "optimized_split_conv_concat_transpose_bridge_to_single_post_nchw"
    ] == 1
    new_split = np.split(source, 3, axis=3)
    new_concat_inputs = [new_split[0], new_split[2], *new_values]
    new_concat_inputs.reverse()
    actual = np.transpose(np.concatenate(new_concat_inputs, axis=3), (0, 3, 1, 2))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_indexed_split_conv_concat_bridge_preserves_nhwc_side_consumer() -> None:
    model_ir = _make_model(branch_side_consumer=True)

    stats = optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir
    )

    assert stats[
        "optimized_split_conv_concat_transpose_bridge_to_single_post_nchw"
    ] == 1
    side = next(
        operator
        for operator in model_ir.operators
        if list(operator.outputs) == ["branch_side"]
    )
    split = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "SPLIT"
    )
    assert list(side.inputs) == [str(split.outputs[1])]


def test_indexed_split_conv_concat_bridge_clones_shared_int64_axis() -> None:
    model_ir = _make_model(integer_dtype="INT64", shared_axis=True)
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats[
        "optimized_split_conv_concat_transpose_bridge_to_single_post_nchw"
    ] == 1
    np.testing.assert_array_equal(
        model_ir.tensors["axis"].data,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["axis_nhwc"].data,
        np.asarray([3], dtype=np.int64),
    )
    split = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "SPLIT"
    )
    reshape = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "RESHAPE"
    )
    assert list(split.inputs) == ["axis_nhwc", "src_nhwc"]
    assert list(reshape.inputs) == ["legacy_value", "axis"]
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_split_conv_concat_bridge_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    split = model_ir.operators[1]
    index = ModelIRGraphIndex(model_ir)

    assert optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir,
        graph_index=index,
        candidate=split,
        max_rewrites=0,
    ) == {"optimized_split_conv_concat_transpose_bridge_to_single_post_nchw": 0}
    assert optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir,
        graph_index=index,
        candidate=split,
    ) == {"optimized_split_conv_concat_transpose_bridge_to_single_post_nchw": 1}
    assert optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir,
        graph_index=index,
    ) == {"optimized_split_conv_concat_transpose_bridge_to_single_post_nchw": 0}


UnsafeMutation = Callable[[ModelIR], None]


def _wrong_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_nchw"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _wrong_axis(model_ir: ModelIR) -> None:
    model_ir.tensors["axis"].data = np.asarray([2], dtype=np.int32)


def _variable_axis(model_ir: ModelIR) -> None:
    model_ir.tensors["axis"].is_variable = True


def _unresolved_source(model_ir: ModelIR) -> None:
    model_ir.inputs.remove("src_nhwc")


def _pre_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["pre_fanout"] = _tensor("pre_fanout", [1, 12, _H, _W])
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["pre_nchw"], outputs=["pre_fanout"])
    )


def _unequal_split(model_ir: ModelIR) -> None:
    model_ir.tensors["split_0_nchw"].shape[1] -= 1


def _wrong_split_count(model_ir: ModelIR) -> None:
    model_ir.operators[1].options["numSplits"] = 4


def _branch_split_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["branch_fanout"] = _tensor(
        "branch_fanout",
        [_N, _SPLIT_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["split_1_nchw"],
            outputs=["branch_fanout"],
        )
    )


def _wrong_branch_permutation(model_ir: ModelIR) -> None:
    model_ir.operators[2].inputs[1] = "to_nchw"


def _branch_output_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["branch_nhwc"].shape[-1] += 1


def _public_branch_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("branch_nhwc")


def _direct_split_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["direct_fanout"] = _tensor(
        "direct_fanout",
        [_N, _SPLIT_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["split_0_nchw"],
            outputs=["direct_fanout"],
        )
    )


def _extra_unclassified_concat_input(model_ir: ModelIR) -> None:
    model_ir.inputs.append("legacy_nchw")
    model_ir.tensors["legacy_nchw"] = _tensor(
        "legacy_nchw",
        [_N, 1, _H, _W],
        layout=LOGICAL_LAYOUT_NCHW,
    )
    concat = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "CONCATENATION"
    )
    concat.inputs.append("legacy_nchw")
    model_ir.tensors["concat_nchw"].shape[1] += 1
    model_ir.tensors["sink"].shape[1] += 1


def _wrong_concat_axis(model_ir: ModelIR) -> None:
    concat = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "CONCATENATION"
    )
    concat.options["axis"] = 2


def _public_concat_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("concat_nchw")


def _wrong_post_permutation(model_ir: ModelIR) -> None:
    post = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "TRANSPOSE"
        and str(operator.outputs[0]).startswith("conv_")
        and str(operator.outputs[0]).endswith("_nchw")
    )
    post.inputs[1] = "to_nhwc"


def _post_output_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["post_fanout"] = _tensor("post_fanout", [1, 3, _H, _W])
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["conv_0_nchw"],
            outputs=["post_fanout"],
        )
    )


def _post_shape_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["conv_0_nchw"].shape[1] += 1


def _post_quantization_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["conv_0_nchw"].quantization = QuantParamIR(
        scale=[0.25],
        zero_point=[0],
        quantized_dimension=0,
    )


def _post_layout_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["conv_0_nhwc"].physical_layout = LOGICAL_LAYOUT_NCHW


def _unreachable_branch(model_ir: ModelIR) -> None:
    for index, channels in enumerate([3, 2]):
        input_name = f"unrelated_{index}_nhwc"
        model_ir.inputs.append(input_name)
        model_ir.tensors[input_name] = _tensor(
            input_name,
            [_N, _H, _W, channels],
            layout=LOGICAL_LAYOUT_NHWC,
        )
        post = next(
            operator
            for operator in model_ir.operators
            if list(operator.outputs) == [f"conv_{index}_nchw"]
        )
        post.inputs[0] = input_name


def _missing_tensor(model_ir: ModelIR) -> None:
    del model_ir.tensors["conv_0_nchw"]


def _duplicate_producer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["src_nhwc"], outputs=["pre_nchw"]),
    )


def _per_axis_split_tensor(model_ir: ModelIR) -> None:
    model_ir.tensors["split_0_nchw"].quantization = QuantParamIR(
        scale=[0.25, 0.5],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_pre_permutation,
        _wrong_axis,
        _variable_axis,
        _unresolved_source,
        _pre_fanout,
        _unequal_split,
        _wrong_split_count,
        _branch_split_fanout,
        _wrong_branch_permutation,
        _branch_output_shape_mismatch,
        _public_branch_output,
        _direct_split_fanout,
        _extra_unclassified_concat_input,
        _wrong_concat_axis,
        _public_concat_output,
        _wrong_post_permutation,
        _post_output_fanout,
        _post_shape_mismatch,
        _post_quantization_mismatch,
        _post_layout_mismatch,
        _unreachable_branch,
        _missing_tensor,
        _duplicate_producer,
        _per_axis_split_tensor,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_indexed_split_conv_concat_bridge_rejects_unsafe_candidate_transactionally(
    mutation: UnsafeMutation,
) -> None:
    model_ir = _make_model()
    mutation(model_ir)
    before = _snapshot(copy.deepcopy(model_ir))
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_split_conv_concat_transpose_bridge_to_single_post_nchw": 0
    }
    assert _snapshot(model_ir) == before
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_terminal_split_conv_concat_bridge_captures_complete_mutation_evidence() -> None:
    owner_tree = ast.parse(
        TERMINAL_ACTIVATION_BRIDGE_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_ACTIVATION_BRIDGE_OWNER
    )
    invocation = next(
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PUBLIC_SPLIT_CONV_CONCAT_BRIDGE_OWNER
    )
    assert [ast.unparse(argument) for argument in invocation.args] == [
        "context.model_ir"
    ]
    assert [keyword.arg for keyword in invocation.keywords] == [
        "layout_state"
    ]
    assert ast.unparse(invocation.keywords[0].value) == "context.layout_state"

    tree = ast.parse(
        (
            REPO_ROOT
            / "onnx2tf"
            / "tflite_builder"
            / "lower_from_onnx2tf.py"
        ).read_text(encoding="utf-8")
    )
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LOWERER_RESULT
    )
    composite = lowerer.body[invocation_index]
    assert isinstance(composite, ast.Assign)
    assert isinstance(composite.value, ast.Call)
    assert isinstance(composite.value.func, ast.Name)
    assert composite.value.func.id == LOWERER_OWNER
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in composite.value.keywords
    } == {
        "include_layout_transpose": "optimize_layout_transpose_chains",
    }
    previous = lowerer.body[invocation_index - 1]
    assert isinstance(previous, ast.If)
    assert ast.unparse(previous.test) == (
        "_late_binary_layout_recovery_requires_reconciliation"
    )
    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.Expr)
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Attribute)
    assert ast.unparse(following.value.func) == "session.record_phase_result"
    assert ast.literal_eval(following.value.args[0]) == (
        "shape_reconciliation.terminal.expand_squeeze"
    )
    top_tree = ast.parse(TOP_OWNER_PATH.read_text(encoding="utf-8"))
    top_owner = next(
        node
        for node in top_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == TOP_OWNER
    )
    top_calls = [
        node
        for node in ast.walk(top_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OUTER_OWNER
    ]
    assert len(top_calls) == 1
    outer_tree = ast.parse(OUTER_OWNER_PATH.read_text(encoding="utf-8"))
    outer_owner = next(
        node
        for node in outer_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == OUTER_OWNER
    )
    outer_calls = [
        node
        for node in ast.walk(outer_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_ACTIVATION_BRIDGE_OWNER
    ]
    assert len(outer_calls) == 1


def test_split_conv_concat_bridge_retains_both_earlier_results() -> None:
    tree = ast.parse(
        (
            REPO_ROOT
            / "onnx2tf"
            / "tflite_builder"
            / "lower_from_onnx2tf.py"
        ).read_text(encoding="utf-8")
    )
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )

    def _statement_call(statement: ast.stmt) -> ast.Call | None:
        if not isinstance(statement, (ast.Assign, ast.Expr)):
            return None
        if not isinstance(statement.value, ast.Call):
            return None
        call = statement.value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "session"
            and call.func.attr == "record_phase_result"
            and len(call.args) == 2
            and isinstance(call.args[1], ast.Call)
        ):
            return call.args[1]
        return call

    def _call_name(statement: ast.stmt) -> str | None:
        call = _statement_call(statement)
        if call is None or not isinstance(call.func, ast.Name):
            return None
        return call.func.id

    lowerer_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_SPLIT_CONV_CONCAT_BRIDGE_OWNER
    ]
    assert len(lowerer_calls) == 2
    owner_tree = ast.parse(
        TERMINAL_ACTIVATION_BRIDGE_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in owner_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_ACTIVATION_BRIDGE_OWNER
    )
    owner_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PUBLIC_SPLIT_CONV_CONCAT_BRIDGE_OWNER
    ]
    assert len(owner_calls) == 1

    terminal_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Name)
            and node.id == "_terminal_qkv_attention_results"
            for node in ast.walk(statement)
        )
    )
    terminal_index = next(
        index
        for index, statement in enumerate(terminal_guard.body)
        if _call_name(statement) == TERMINAL_SPLIT_CONV_CONCAT_BRIDGE_OWNER
    )
    terminal = terminal_guard.body[terminal_index]
    assert isinstance(terminal, ast.Expr)
    record = terminal.value
    assert isinstance(record, ast.Call)
    assert isinstance(record.func, ast.Attribute)
    assert isinstance(record.func.value, ast.Name)
    assert record.func.value.id == "session"
    assert record.func.attr == "record_phase_result"
    assert ast.literal_eval(record.args[0]) == (
        "cleanup.terminal.qkv_split_conv_concat_bridge"
    )
    predecessor = terminal_guard.body[terminal_index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert isinstance(predecessor.targets[0], ast.Name)
    assert predecessor.targets[0].id == "_terminal_qkv_attention_results"
    assert terminal_index == len(terminal_guard.body) - 1
    terminal_guard_index = lowerer.body.index(terminal_guard)
    terminal_successor = lowerer.body[terminal_guard_index + 1]
    assert isinstance(terminal_successor, ast.Assign)
    assert isinstance(terminal_successor.targets[0], ast.Name)
    assert terminal_successor.targets[0].id == (
        TERMINAL_SINGLETON_CLAMP_SINET_RESULT
    )
    assert _call_name(terminal_successor) == TERMINAL_SINGLETON_CLAMP_SINET_OWNER

    post_sinet_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == TERMINAL_SPLIT_CONV_CONCAT_BRIDGE_OWNER
    )
    post_sinet = lowerer.body[post_sinet_index]
    assert isinstance(post_sinet, ast.Expr)
    post_sinet_record = post_sinet.value
    assert isinstance(post_sinet_record, ast.Call)
    assert isinstance(post_sinet_record.func, ast.Attribute)
    assert isinstance(post_sinet_record.func.value, ast.Name)
    assert post_sinet_record.func.value.id == "session"
    assert post_sinet_record.func.attr == "record_phase_result"
    assert ast.literal_eval(post_sinet_record.args[0]) == (
        "cleanup.post_sinet.split_conv_concat_bridge"
    )
    assert _call_name(lowerer.body[post_sinet_index - 1]) == (
        "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
    )
    assert _call_name(lowerer.body[post_sinet_index + 1]) == (
        "_optimize_sinet_mix_attention_double_logistic_nhwc_chains"
    )

    for statement in (terminal, post_sinet):
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == [
            "model_ir"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}

    late = owner_calls[0]
    assert [ast.unparse(argument) for argument in late.args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in late.keywords
    } == {"layout_state": "context.layout_state"}
