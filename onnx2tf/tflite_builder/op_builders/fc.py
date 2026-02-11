from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def build_fully_connected_from_gemm_or_matmul(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    weight_name = node.inputs[1].name
    output_name = node.outputs[0].name
    ctx.ensure_tensor(input_name)
    ctx.ensure_tensor(weight_name)
    ctx.ensure_tensor(output_name)

    input_shape = ctx.get_tensor_shape(input_name)
    if len(input_shape) != 2:
        raise NotImplementedError(
            "Only rank-2 FullyConnected conversion is supported. "
            f"op={node.name} input_shape={input_shape}"
        )

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NotImplementedError(f"FC weight must be constant. op={node.name}")
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2:
        raise NotImplementedError(
            f"FC weight rank must be 2. op={node.name} weight_shape={weights.shape}"
        )

    if node.op == "Gemm":
        alpha = float(node.attrs.get("alpha", 1.0))
        beta = float(node.attrs.get("beta", 1.0))
        trans_a = int(node.attrs.get("transA", 0))
        trans_b = int(node.attrs.get("transB", 0))
        if trans_a != 0:
            raise NotImplementedError(f"Gemm transA=1 is not supported. op={node.name}")
        if trans_b == 0:
            fc_weights = weights.T
        else:
            fc_weights = weights
        if alpha != 1.0:
            fc_weights = fc_weights * alpha

        bias_values = None
        if len(node.inputs) >= 3:
            bias_values = ctx.get_constant_array(node.inputs[2].name)
        if bias_values is None:
            bias_values = np.zeros((fc_weights.shape[0],), dtype=np.float32)
        else:
            bias_values = np.asarray(bias_values, dtype=np.float32).reshape(-1)
        if beta != 1.0:
            bias_values = bias_values * beta
    else:
        fc_weights = weights.T
        bias_values = np.zeros((fc_weights.shape[0],), dtype=np.float32)

    w_name = ctx.add_const_tensor(
        f"{node.name}_fc_weights",
        np.asarray(fc_weights, dtype=np.float32),
    )
    b_name = ctx.add_const_tensor(
        f"{node.name}_fc_bias",
        np.asarray(bias_values, dtype=np.float32),
    )

    ctx.add_operator(
        OperatorIR(
            op_type="FULLY_CONNECTED",
            inputs=[input_name, w_name, b_name],
            outputs=[output_name],
            options={
                "fusedActivationFunction": "NONE",
                "weightsFormat": "DEFAULT",
                "keepNumDims": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )
