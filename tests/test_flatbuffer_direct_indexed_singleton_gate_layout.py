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
from onnx2tf.tflite_builder.passes.singleton_gate_layout import (
    optimize_singleton_gate_conv_concat_nhwc_bridge_blocks,
)


_H = 3
_W = 4
_ONE_NHWC = [1, _H, _W, 1]
_ONE_NCHW = [1, 1, _H, _W]
_RGB_NHWC = [1, _H, _W, 3]
_RGB_NCHW = [1, 3, _H, _W]
_OUT_NHWC = [1, _H, _W, 4]


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: Any = None,
    signature: list[int] | None = None,
    logical_layout: str = "UNKNOWN",
    physical_layout: str = "UNKNOWN",
    quantization: QuantParamIR | None = None,
    is_variable: bool = False,
    dtype: str = "FLOAT32",
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


def _sig(shape: list[int], dynamic: bool) -> list[int]:
    if not dynamic or len(shape) != 4:
        return list(shape)
    if shape[1] == 1 and shape[-1] != 1:
        return [shape[0], shape[1], -1, -1]
    return [shape[0], -1, -1, shape[-1]]


def _shape_tensor(name: str, shape: list[int]) -> TensorIR:
    return _tensor(
        name,
        [4],
        dtype="INT64",
        data=np.asarray(shape, dtype=np.int64),
    )


def _make_model(
    *,
    logistic_aux: bool = True,
    dynamic: bool = False,
    rgb_bridge: bool = True,
    direct_rgb: bool = False,
    clip3_side_consumer: bool = False,
) -> ModelIR:
    rng = np.random.default_rng(23)
    model_ir = ModelIR("indexed_singleton_gate")
    model_ir.inputs = [
        "clip_nhwc",
        "aux_nhwc",
        "keep1_nhwc",
        "keep2_nhwc",
    ]
    if rgb_bridge and not direct_rgb:
        model_ir.inputs.append("rgb_nhwc")
    model_ir.outputs = ["y_nhwc"]
    model_ir.tensors = {
        "split_nchw": _tensor(
            "split_nchw",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            data=rng.normal(size=_ONE_NCHW).astype(np.float32),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "split_shape": _shape_tensor("split_shape", _ONE_NHWC),
        "split_nhwc": _tensor(
            "split_nhwc",
            _ONE_NHWC,
            signature=_sig(_ONE_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "clip_nhwc": _tensor(
            "clip_nhwc",
            _ONE_NHWC,
            signature=_sig(_ONE_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "clip_shape": _shape_tensor("clip_shape", _ONE_NCHW),
        "clip_nchw": _tensor(
            "clip_nchw",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "one": _tensor(
            "one",
            [1],
            data=np.asarray([1.0], dtype=np.float32),
        ),
        "gate": _tensor(
            "gate",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "sub": _tensor(
            "sub",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "aux_nhwc": _tensor(
            "aux_nhwc",
            _ONE_NHWC,
            signature=_sig(_ONE_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "aux_shape": _shape_tensor("aux_shape", _ONE_NCHW),
        "aux_nchw": _tensor(
            "aux_nchw",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "signal": _tensor(
            "signal",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "signal_mul": _tensor(
            "signal_mul",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "fused": _tensor(
            "fused",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "clip3_nchw": _tensor(
            "clip3_nchw",
            _ONE_NCHW,
            signature=_sig(_ONE_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        ),
        "clip3_shape": _shape_tensor("clip3_shape", _ONE_NHWC),
        "clip3_nhwc": _tensor(
            "clip3_nhwc",
            _ONE_NHWC,
            signature=_sig(_ONE_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "keep1_nhwc": _tensor(
            "keep1_nhwc",
            _ONE_NHWC,
            signature=_sig(_ONE_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "keep2_nhwc": _tensor(
            "keep2_nhwc",
            _ONE_NHWC,
            signature=_sig(_ONE_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "div_out": _tensor(
            "div_out",
            _ONE_NHWC,
            signature=_sig(_ONE_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
        "y_nhwc": _tensor(
            "y_nhwc",
            _OUT_NHWC,
            signature=_sig(_OUT_NHWC, dynamic),
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        ),
    }
    operators = [
        OperatorIR(
            "RESHAPE",
            ["split_nchw", "split_shape"],
            ["split_nhwc"],
            options={"newShape": list(_ONE_NHWC)},
        ),
        OperatorIR(
            "RESHAPE",
            ["clip_nhwc", "clip_shape"],
            ["clip_nchw"],
            options={"newShape": list(_ONE_NCHW)},
        ),
        OperatorIR("MUL", ["clip_nchw", "split_nchw"], ["gate"]),
        OperatorIR("SUB", ["one", "clip_nchw"], ["sub"]),
    ]
    if rgb_bridge:
        model_ir.tensors["rgb_mul"] = _tensor(
            "rgb_mul",
            _RGB_NCHW,
            signature=_sig(_RGB_NCHW, dynamic),
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
        if direct_rgb:
            model_ir.tensors["rgb_direct_nchw"] = _tensor(
                "rgb_direct_nchw",
                _RGB_NCHW,
                signature=_sig(_RGB_NCHW, dynamic),
                data=rng.normal(size=_RGB_NCHW).astype(np.float32),
                logical_layout=LOGICAL_LAYOUT_NCHW,
                physical_layout=LOGICAL_LAYOUT_NHWC,
            )
            operators.append(
                OperatorIR("MUL", ["clip_nchw", "rgb_direct_nchw"], ["rgb_mul"])
            )
        else:
            model_ir.tensors["rgb_nhwc"] = _tensor(
                "rgb_nhwc",
                _RGB_NHWC,
                signature=_sig(_RGB_NHWC, dynamic),
                logical_layout=LOGICAL_LAYOUT_NHWC,
                physical_layout=LOGICAL_LAYOUT_NHWC,
            )
            model_ir.tensors["perm"] = _shape_tensor("perm", [0, 3, 1, 2])
            model_ir.tensors["rgb_nchw"] = _tensor(
                "rgb_nchw",
                _RGB_NCHW,
                signature=_sig(_RGB_NCHW, dynamic),
                logical_layout=LOGICAL_LAYOUT_NCHW,
                physical_layout=LOGICAL_LAYOUT_NCHW,
            )
            operators.extend(
                [
                    OperatorIR("TRANSPOSE", ["rgb_nhwc", "perm"], ["rgb_nchw"]),
                    OperatorIR("MUL", ["clip_nchw", "rgb_nchw"], ["rgb_mul"]),
                ]
            )
    operators.append(
        OperatorIR(
            "RESHAPE",
            ["aux_nhwc", "aux_shape"],
            ["aux_nchw"],
            options={"newShape": list(_ONE_NCHW)},
        )
    )
    signal_name = "aux_nchw"
    if logistic_aux:
        operators.append(OperatorIR("LOGISTIC", ["aux_nchw"], ["signal"]))
        signal_name = "signal"
    operators.extend(
        [
            OperatorIR("MUL", ["sub", signal_name], ["signal_mul"]),
            OperatorIR("ADD", ["gate", "signal_mul"], ["fused"]),
            OperatorIR("RELU_0_TO_1", ["fused"], ["clip3_nchw"]),
            OperatorIR(
                "RESHAPE",
                ["clip3_nchw", "clip3_shape"],
                ["clip3_nhwc"],
                options={"newShape": list(_ONE_NHWC)},
            ),
            OperatorIR("DIV", ["split_nhwc", "keep1_nhwc"], ["div_out"]),
            OperatorIR(
                "CONCATENATION",
                ["split_nhwc", "keep1_nhwc", "keep2_nhwc", "clip3_nhwc"],
                ["y_nhwc"],
                options={"axis": 3},
            ),
        ]
    )
    if clip3_side_consumer:
        model_ir.tensors["clip3_side"] = _tensor(
            "clip3_side",
            _ONE_NHWC,
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        )
        operators.append(OperatorIR("RELU", ["clip3_nhwc"], ["clip3_side"]))
        model_ir.outputs.append("clip3_side")
    model_ir.operators = operators
    return model_ir


def _feeds(model_ir: ModelIR) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(29)
    return {
        name: rng.uniform(0.2, 1.0, size=model_ir.tensors[name].shape).astype(
            np.float32
        )
        for name in model_ir.inputs
    }


def _evaluate(model_ir: ModelIR, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = {name: np.asarray(value) for name, value in feeds.items()}
    values.update(
        {
            name: np.asarray(tensor.data).reshape(tensor.shape)
            for name, tensor in model_ir.tensors.items()
            if tensor.data is not None
        }
    )
    for operator in model_ir.operators:
        inputs = [values[name] for name in operator.inputs]
        op_type = str(operator.op_type).upper()
        if op_type == "RESHAPE":
            result = np.reshape(inputs[0], model_ir.tensors[operator.outputs[0]].shape)
        elif op_type == "TRANSPOSE":
            result = np.transpose(inputs[0], tuple(int(value) for value in inputs[1]))
        elif op_type == "MUL":
            result = inputs[0] * inputs[1]
        elif op_type == "SUB":
            result = inputs[0] - inputs[1]
        elif op_type == "ADD":
            result = inputs[0] + inputs[1]
        elif op_type == "DIV":
            result = inputs[0] / inputs[1]
        elif op_type == "LOGISTIC":
            result = 1.0 / (1.0 + np.exp(-inputs[0]))
        elif op_type == "RELU_0_TO_1":
            result = np.clip(inputs[0], 0.0, 1.0)
        elif op_type == "RELU":
            result = np.maximum(inputs[0], 0.0)
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


@pytest.mark.parametrize("logistic_aux", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("rgb_bridge", [False, True])
def test_indexed_singleton_gate_preserves_semantics(
    logistic_aux: bool,
    dynamic: bool,
    rgb_bridge: bool,
) -> None:
    model_ir = _make_model(
        logistic_aux=logistic_aux,
        dynamic=dynamic,
        rgb_bridge=rgb_bridge,
    )
    feeds = _feeds(model_ir)
    before = _evaluate(model_ir, feeds)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    after = _evaluate(model_ir, feeds)
    assert stats == {"optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 1}
    np.testing.assert_allclose(after["y_nhwc"], before["y_nhwc"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(after["div_out"], before["div_out"], rtol=0.0, atol=0.0)
    if rgb_bridge:
        np.testing.assert_allclose(
            after["rgb_mul"],
            np.transpose(before["rgb_mul"], (0, 2, 3, 1)),
            rtol=0.0,
            atol=0.0,
        )
    assert not any(operator.op_type == "RESHAPE" for operator in model_ir.operators)
    assert not any(operator.op_type == "TRANSPOSE" for operator in model_ir.operators)
    assert model_ir.tensors["split_nchw"].shape == _ONE_NHWC
    assert model_ir.tensors["clip3_nchw"].shape == _ONE_NHWC
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_singleton_gate_supports_direct_stale_nchw_rgb_and_side_consumers() -> (
    None
):
    model_ir = _make_model(
        direct_rgb=True,
        clip3_side_consumer=True,
    )
    feeds = _feeds(model_ir)
    before = _evaluate(model_ir, feeds)
    expected_rgb = feeds["clip_nhwc"] * np.asarray(
        model_ir.tensors["rgb_direct_nchw"].data
    ).reshape(_RGB_NHWC)

    stats = optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(model_ir)

    after = _evaluate(model_ir, feeds)
    assert stats["optimized_singleton_gate_conv_concat_nhwc_bridge_blocks"] == 1
    np.testing.assert_allclose(after["y_nhwc"], before["y_nhwc"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        after["rgb_mul"],
        expected_rgb,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        after["clip3_side"], before["clip3_side"], rtol=0.0, atol=0.0
    )
    assert model_ir.tensors["rgb_direct_nchw"].shape == _RGB_NHWC


def test_indexed_singleton_gate_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    original = _fingerprint(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    concat = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    )

    assert optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=0,
    ) == {"optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 0}
    assert _fingerprint(model_ir) == original
    assert optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        candidate=model_ir.operators[0],
    ) == {"optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 0}
    assert _fingerprint(model_ir) == original
    assert optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        candidate=concat,
    ) == {"optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 1}
    after = _fingerprint(model_ir)
    assert optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {"optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 0}
    assert _fingerprint(model_ir) == after


def _operator_with_output(model_ir: ModelIR, name: str) -> OperatorIR:
    return next(
        operator for operator in model_ir.operators if operator.outputs == [name]
    )


def _mutate_unsafe(model_ir: ModelIR, case: str) -> None:
    if case == "non_singleton_clip":
        model_ir.tensors["clip_nhwc"].shape = [1, _H, _W, 2]
    elif case == "non_singleton_aux":
        model_ir.tensors["aux_nhwc"].shape = [1, _H, _W, 2]
    elif case == "non_singleton_split":
        model_ir.tensors["split_nchw"].shape = [1, 2, _H, _W]
    elif case == "wrong_permutation":
        model_ir.tensors["perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int64)
    elif case == "shared_clip_adapter":
        model_ir.tensors["clip_side"] = _tensor("clip_side", _ONE_NCHW)
        model_ir.operators.append(OperatorIR("RELU", ["clip_nchw"], ["clip_side"]))
    elif case == "shared_aux_adapter":
        model_ir.tensors["aux_side"] = _tensor("aux_side", _ONE_NCHW)
        model_ir.operators.append(OperatorIR("RELU", ["aux_nchw"], ["aux_side"]))
    elif case == "shared_gate":
        model_ir.tensors["gate_side"] = _tensor("gate_side", _ONE_NCHW)
        model_ir.operators.append(OperatorIR("RELU", ["gate"], ["gate_side"]))
    elif case == "shared_sub":
        model_ir.tensors["sub_side"] = _tensor("sub_side", _ONE_NCHW)
        model_ir.operators.append(OperatorIR("RELU", ["sub"], ["sub_side"]))
    elif case == "public_adapter_output":
        model_ir.outputs = ["clip3_nhwc"]
    elif case == "public_core":
        model_ir.outputs = ["fused"]
    elif case == "missing_concat_tensor":
        del model_ir.tensors["y_nhwc"]
    elif case == "per_axis_quantization":
        model_ir.tensors["gate"].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3, 0.4],
            zero_point=[0, 0, 0, 0],
            quantized_dimension=3,
        )
    elif case == "duplicate_producer":
        model_ir.operators.append(OperatorIR("RELU", ["sub"], ["gate"]))
    elif case == "consumer_before_producer":
        add_index = model_ir.operators.index(_operator_with_output(model_ir, "fused"))
        signal_index = model_ir.operators.index(
            _operator_with_output(model_ir, "signal_mul")
        )
        model_ir.operators[add_index], model_ir.operators[signal_index] = (
            model_ir.operators[signal_index],
            model_ir.operators[add_index],
        )
    elif case == "fused_activation":
        _operator_with_output(model_ir, "fused").options["fusedActivationFunction"] = (
            "RELU"
        )
    elif case == "second_clip3_candidate":
        model_ir.tensors["clip4_nchw"] = _tensor("clip4_nchw", _ONE_NCHW)
        model_ir.tensors["clip4_shape"] = _shape_tensor("clip4_shape", _ONE_NHWC)
        model_ir.tensors["clip4_nhwc"] = _tensor("clip4_nhwc", _ONE_NHWC)
        model_ir.operators.extend(
            [
                OperatorIR("RELU", ["fused"], ["clip4_nchw"]),
                OperatorIR(
                    "RESHAPE",
                    ["clip4_nchw", "clip4_shape"],
                    ["clip4_nhwc"],
                    options={"newShape": list(_ONE_NHWC)},
                ),
            ]
        )
        concat = next(
            operator
            for operator in model_ir.operators
            if operator.op_type == "CONCATENATION"
        )
        concat.inputs.append("clip4_nhwc")
    elif case == "wrong_concat_axis":
        next(
            operator
            for operator in model_ir.operators
            if operator.op_type == "CONCATENATION"
        ).options["axis"] = 1
    elif case == "mismatched_shape_constant":
        model_ir.tensors["clip_shape"].data = np.asarray([1, 1, _W, _H], dtype=np.int64)
    elif case == "unresolved_keep":
        model_ir.inputs.remove("keep2_nhwc")
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "non_singleton_clip",
        "non_singleton_aux",
        "non_singleton_split",
        "wrong_permutation",
        "shared_clip_adapter",
        "shared_aux_adapter",
        "shared_gate",
        "shared_sub",
        "public_adapter_output",
        "public_core",
        "missing_concat_tensor",
        "per_axis_quantization",
        "duplicate_producer",
        "consumer_before_producer",
        "fused_activation",
        "second_clip3_candidate",
        "wrong_concat_axis",
        "mismatched_shape_constant",
        "unresolved_keep",
    ],
)
def test_indexed_singleton_gate_rejects_unsafe_candidate_transactionally(
    case: str,
) -> None:
    model_ir = _make_model()
    _mutate_unsafe(model_ir, case)
    before = copy.deepcopy(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 0}
    assert _fingerprint(model_ir) == _fingerprint(before)
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)
