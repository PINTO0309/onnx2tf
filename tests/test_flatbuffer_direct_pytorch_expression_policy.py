from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.pytorch_expression_policy import (
    _channel_first_constant_expr_for_buffer_attr_for_codegen,
    _derived_local_var_name_for_codegen,
    _permuted_constant_expr_for_tensor_name_for_codegen,
    _tensor_dtype_name_for_codegen,
    _tensor_expr_for_channel_first_bridge_for_codegen,
    _tensor_expr_for_codegen,
    _transposed_constant_expr_for_tensor_name_for_codegen,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    layout: str = "UNKNOWN",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        logical_layout=layout,
        data=data,
    )


def test_tensor_expression_policy_preserves_alias_and_constant_precedence() -> None:
    model_ir = ModelIR(
        name="tensor_expressions",
        tensors={
            "input": _tensor("input", [1, 3, 4, 5], layout="NCHW"),
            "constant": _tensor(
                "constant",
                [2],
                data=np.asarray([1.0, 2.0], dtype=np.float32),
            ),
        },
        inputs=["input"],
    )

    assert _tensor_dtype_name_for_codegen(
        model_ir=model_ir,
        tensor_name="input",
    ) == "FLOAT32"
    assert _tensor_expr_for_codegen(
        model_ir=model_ir,
        producer_index={},
        tensor_expr_aliases={"input": "explicit_alias"},
        channel_first_tensor_expr_aliases={},
        buffer_attr_names={},
        runtime_imports=set(),
        tensor_var_names={"input": "input_var"},
        tensor_name="input",
    ) == "explicit_alias"

    runtime_imports: set[str] = set()
    assert _tensor_expr_for_codegen(
        model_ir=model_ir,
        producer_index={},
        tensor_expr_aliases={},
        channel_first_tensor_expr_aliases={},
        buffer_attr_names={},
        runtime_imports=runtime_imports,
        tensor_var_names={},
        tensor_name="constant",
    ) == (
        "torch.as_tensor([1.0, 2.0], dtype=torch.float32, "
        "device=_module_device(self))"
    )
    assert runtime_imports == {"_module_device"}


def test_channel_first_bridge_and_constant_alias_expression_lookups() -> None:
    model_ir = ModelIR(
        name="bridge_expressions",
        tensors={"value": _tensor("value", [1, 2, 3, 4])},
    )

    assert _tensor_expr_for_channel_first_bridge_for_codegen(
        model_ir=model_ir,
        channel_first_tensor_expr_aliases={"value": "value_cf"},
        tensor_name="value",
        perm=[0, 3, 1, 2],
    ) == "value_cf"
    assert _channel_first_constant_expr_for_buffer_attr_for_codegen(
        buffer_attr_name_to_tensor_name={"weight": "weight_tensor"},
        channel_first_constant_buffer_alias_exprs={
            "weight_tensor": "self.weight_cf"
        },
        channel_first_rank4_constant_buffer_alias_shape_fn=(
            lambda _name: [1, 4, 1, 1]
        ),
        buffer_expr="self.weight",
        target_shape=[1, 4, 1, 1],
    ) == "self.weight_cf"
    assert _permuted_constant_expr_for_tensor_name_for_codegen(
        permuted_constant_buffer_alias_exprs={
            ("weight_tensor", (0, 2, 3, 1)): "self.weight_nhwc"
        },
        tensor_name="weight_tensor",
        perm=[0, 2, 3, 1],
    ) == "self.weight_nhwc"
    assert _transposed_constant_expr_for_tensor_name_for_codegen(
        transposed_constant_buffer_alias_exprs={
            "weight_tensor": "self.weight_t"
        },
        tensor_name="weight_tensor",
    ) == "self.weight_t"


def test_derived_local_names_are_cached_and_collision_free() -> None:
    cache: dict[str, str] = {}
    used_names = {"decoderoutput"}

    first = _derived_local_var_name_for_codegen(
        synthetic_local_var_names=cache,
        used_local_var_names=used_names,
        base_name="decoder/output",
    )
    second = _derived_local_var_name_for_codegen(
        synthetic_local_var_names=cache,
        used_local_var_names=used_names,
        base_name="decoder/output",
    )

    assert first == "decoderoutput_1"
    assert second == first
