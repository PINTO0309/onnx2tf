from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.pytorch_nms_policy import (
    _is_identity_nms_postprocess_gather_for_codegen,
    _range_only_feeds_identity_nms_postprocess_gathers_for_codegen,
)


def test_nms_range_gather_identity_contract() -> None:
    range_op = OperatorIR(
        op_type="RANGE",
        inputs=["start", "limit", "delta"],
        outputs=["indices"],
    )
    gather_op = OperatorIR(
        op_type="GATHER",
        inputs=["selected", "indices"],
        outputs=["output"],
    )
    model_ir = ModelIR(
        name="nms_identity",
        operators=[range_op, gather_op],
    )
    scalar_literals = {"start": "0", "delta": "1"}

    assert _is_identity_nms_postprocess_gather_for_codegen(
        model_ir=model_ir,
        tensor_expr_aliases={"selected": "_nms_selected_indices_valid_0"},
        producer_index={"indices": 0},
        scalar_literal_expr_fn=lambda name: scalar_literals.get(name),
        params_name="selected",
        indices_name="indices",
    )
    assert _range_only_feeds_identity_nms_postprocess_gathers_for_codegen(
        model_ir=model_ir,
        consumer_index={"indices": [1]},
        is_identity_nms_postprocess_gather_fn=lambda params, indices: (
            params == "selected" and indices == "indices"
        ),
        output_name="indices",
    )


def test_nms_range_policy_rejects_unrelated_alias_and_consumer() -> None:
    model_ir = ModelIR(
        name="nms_reject",
        operators=[
            OperatorIR(op_type="ADD", inputs=["lhs", "rhs"], outputs=["out"]),
        ],
    )

    assert not _is_identity_nms_postprocess_gather_for_codegen(
        model_ir=model_ir,
        tensor_expr_aliases={"selected": "ordinary_alias"},
        producer_index={"indices": 0},
        scalar_literal_expr_fn=lambda _name: "0",
        params_name="selected",
        indices_name="indices",
    )
    assert not _range_only_feeds_identity_nms_postprocess_gathers_for_codegen(
        model_ir=model_ir,
        consumer_index={"indices": [0]},
        is_identity_nms_postprocess_gather_fn=lambda *_args: True,
        output_name="indices",
    )
