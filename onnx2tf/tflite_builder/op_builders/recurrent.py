from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR


def _get_original_node_inputs(node: Any, ctx: Any) -> List[str]:
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None:
        return [str(v.name) for v in node.inputs]
    for graph_node in onnx_model.graph.node:
        graph_node_name = str(graph_node.name) if str(graph_node.name) != "" else str(graph_node.op_type)
        if graph_node_name == str(node.name) and str(graph_node.op_type) == str(node.op):
            return [str(v) for v in graph_node.input]
    return [str(v.name) for v in node.inputs]


def _input_name(inputs: List[str], index: int) -> str:
    if index < 0 or index >= len(inputs):
        return ""
    return str(inputs[index])


def _split_onnx_lstm_gates(
    tensor_2d: np.ndarray,
    hidden_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden = int(hidden_size)
    if tensor_2d.ndim != 2 or int(tensor_2d.shape[0]) != 4 * hidden:
        raise NotImplementedError(
            "LSTM gate tensor must be rank-2 with leading dim=4*hidden_size. "
            f"shape={list(tensor_2d.shape)} hidden_size={hidden}"
        )
    # ONNX gate order is [i, o, f, c]. TFLite expects [i, f, c, o].
    gate_i = np.asarray(tensor_2d[0:hidden, :], dtype=np.float32)
    gate_o = np.asarray(tensor_2d[hidden:2 * hidden, :], dtype=np.float32)
    gate_f = np.asarray(tensor_2d[2 * hidden:3 * hidden, :], dtype=np.float32)
    gate_c = np.asarray(tensor_2d[3 * hidden:4 * hidden, :], dtype=np.float32)
    return gate_i, gate_f, gate_c, gate_o


def _split_onnx_lstm_bias_gates(
    bias_1d: np.ndarray,
    hidden_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden = int(hidden_size)
    bias = np.asarray(bias_1d, dtype=np.float32).reshape(-1)
    if int(bias.size) != 8 * hidden:
        raise NotImplementedError(
            "LSTM bias must have size 8*hidden_size (Wb+Rb). "
            f"size={int(bias.size)} hidden_size={hidden}"
        )
    merged = np.asarray(
        bias[:4 * hidden] + bias[4 * hidden:8 * hidden],
        dtype=np.float32,
    )
    gate_i = np.asarray(merged[0:hidden], dtype=np.float32)
    gate_o = np.asarray(merged[hidden:2 * hidden], dtype=np.float32)
    gate_f = np.asarray(merged[2 * hidden:3 * hidden], dtype=np.float32)
    gate_c = np.asarray(merged[3 * hidden:4 * hidden], dtype=np.float32)
    return gate_i, gate_f, gate_c, gate_o


def _normalized_shape_dim(value: Optional[int], fallback: int = 1) -> int:
    if value is None:
        return int(fallback)
    dim = int(value)
    return dim if dim > 0 else int(fallback)


def build_lstm_op(node: Any, ctx: Any) -> None:
    original_inputs = _get_original_node_inputs(node, ctx)
    x_name = _input_name(original_inputs, 0)
    w_name = _input_name(original_inputs, 1)
    r_name = _input_name(original_inputs, 2)
    b_name = _input_name(original_inputs, 3)
    initial_h_name = _input_name(original_inputs, 5)
    initial_c_name = _input_name(original_inputs, 6)

    y_name = node.outputs[0].name
    y_dtype = str(ctx.get_tensor_dtype(y_name))
    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(y_name)

    w_value = ctx.get_constant_array(w_name)
    r_value = ctx.get_constant_array(r_name)
    if w_value is None or r_value is None:
        raise NotImplementedError(
            f"LSTM weights must be constant for flatbuffer_direct. op={node.name}"
        )
    W = np.asarray(w_value, dtype=np.float32)
    R = np.asarray(r_value, dtype=np.float32)
    if W.ndim != 3 or R.ndim != 3:
        raise NotImplementedError(
            f"LSTM W/R must be rank-3. op={node.name} W={list(W.shape)} R={list(R.shape)}"
        )
    if int(W.shape[0]) != 2 or int(R.shape[0]) != 2:
        raise NotImplementedError(
            f"LSTM bidirectional lowering expects num_directions=2. op={node.name}"
        )

    hidden_size_attr = int(node.attrs.get("hidden_size", 0))
    hidden_size = int(hidden_size_attr) if hidden_size_attr > 0 else int(W.shape[1] // 4)
    if int(W.shape[1]) != 4 * hidden_size or int(R.shape[1]) != 4 * hidden_size:
        raise NotImplementedError(
            "LSTM hidden_size mismatch in W/R. "
            f"op={node.name} hidden_size={hidden_size} "
            f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
        )

    if b_name != "":
        b_value = ctx.get_constant_array(b_name)
        if b_value is None:
            raise NotImplementedError(
                f"LSTM B must be constant when provided. op={node.name} tensor={b_name}"
            )
        B = np.asarray(b_value, dtype=np.float32)
    else:
        B = np.zeros((2, 8 * hidden_size), dtype=np.float32)
    if B.ndim != 2 or int(B.shape[0]) != 2 or int(B.shape[1]) != 8 * hidden_size:
        raise NotImplementedError(
            "LSTM B must have shape [2, 8*hidden_size]. "
            f"op={node.name} hidden_size={hidden_size} B_shape={list(B.shape)}"
        )

    h0_value = ctx.get_constant_array(initial_h_name)
    c0_value = ctx.get_constant_array(initial_c_name)
    if h0_value is None or c0_value is None:
        raise NotImplementedError(
            f"LSTM initial_h/initial_c must be constant for flatbuffer_direct. op={node.name}"
        )
    H0 = np.asarray(h0_value, dtype=np.float32)
    C0 = np.asarray(c0_value, dtype=np.float32)
    if H0.ndim != 3 or C0.ndim != 3:
        raise NotImplementedError(
            f"LSTM initial_h/initial_c must be rank-3. op={node.name}"
        )
    if int(H0.shape[0]) != 2 or int(C0.shape[0]) != 2:
        raise NotImplementedError(
            f"LSTM initial_h/initial_c first dim must be 2. op={node.name}"
        )

    fw_w_i, fw_w_f, fw_w_c, fw_w_o = _split_onnx_lstm_gates(
        np.asarray(W[0], dtype=np.float32),
        hidden_size,
    )
    bw_w_i, bw_w_f, bw_w_c, bw_w_o = _split_onnx_lstm_gates(
        np.asarray(W[1], dtype=np.float32),
        hidden_size,
    )
    fw_r_i, fw_r_f, fw_r_c, fw_r_o = _split_onnx_lstm_gates(
        np.asarray(R[0], dtype=np.float32),
        hidden_size,
    )
    bw_r_i, bw_r_f, bw_r_c, bw_r_o = _split_onnx_lstm_gates(
        np.asarray(R[1], dtype=np.float32),
        hidden_size,
    )
    fw_b_i, fw_b_f, fw_b_c, fw_b_o = _split_onnx_lstm_bias_gates(
        np.asarray(B[0], dtype=np.float32),
        hidden_size,
    )
    bw_b_i, bw_b_f, bw_b_c, bw_b_o = _split_onnx_lstm_bias_gates(
        np.asarray(B[1], dtype=np.float32),
        hidden_size,
    )

    def _add_const(suffix: str, values: np.ndarray, dtype: np.dtype = np.float32) -> str:
        return ctx.add_const_tensor(
            f"{node.name}_{suffix}",
            np.asarray(values, dtype=dtype),
        )

    def _add_zero_variable_state(suffix: str, reference_values: np.ndarray) -> str:
        state_name = ctx.add_intermediate_tensor(
            f"{node.name}_{suffix}",
            dtype="FLOAT32",
            shape=[int(v) for v in list(np.asarray(reference_values).shape)],
        )
        state_tensor = ctx.model_ir.tensors[state_name]
        state_tensor.is_variable = True
        state_tensor.data = None
        return state_name

    fw_h0_name = _add_zero_variable_state("fw_h0_state", np.asarray(H0[0], dtype=np.float32))
    fw_c0_name = _add_zero_variable_state("fw_c0_state", np.asarray(C0[0], dtype=np.float32))
    bw_h0_name = _add_zero_variable_state("bw_h0_state", np.asarray(H0[1], dtype=np.float32))
    bw_c0_name = _add_zero_variable_state("bw_c0_state", np.asarray(C0[1], dtype=np.float32))

    input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    x_tensor = ctx.model_ir.tensors.get(x_name, None)
    input_signature = None
    if x_tensor is not None and x_tensor.shape_signature is not None:
        input_signature = [int(v) for v in list(x_tensor.shape_signature)]
    else:
        input_signature = [int(v) for v in input_shape]
    seq_signature_dim = input_signature[0] if len(input_signature) >= 1 else None
    batch_signature_dim = input_signature[1] if len(input_signature) >= 2 else None
    seq_dim = _normalized_shape_dim(
        seq_signature_dim if seq_signature_dim is not None and int(seq_signature_dim) > 0 else (
            input_shape[0] if len(input_shape) >= 1 else 1
        ),
        fallback=1,
    )
    batch_dim = _normalized_shape_dim(
        batch_signature_dim if batch_signature_dim is not None and int(batch_signature_dim) > 0 else (
            input_shape[1] if len(input_shape) >= 2 else 1
        ),
        fallback=1,
    )
    y_tensor = ctx.model_ir.tensors.get(y_name, None)
    if y_tensor is not None:
        y_tensor.shape = [int(seq_dim), 2, int(batch_dim), int(hidden_size)]
        y_tensor.shape_signature = [
            int(seq_signature_dim) if seq_signature_dim is not None else int(seq_dim),
            2,
            int(batch_signature_dim) if batch_signature_dim is not None else int(batch_dim),
            int(hidden_size),
        ]
    merged_output_name = ctx.add_intermediate_tensor(
        f"{y_name}_bilstm_merged",
        dtype=y_dtype,
        shape=[int(seq_dim), int(batch_dim), int(hidden_size * 2)],
    )
    fw_output_name = ctx.add_intermediate_tensor(
        f"{y_name}_fw",
        dtype=y_dtype,
        shape=[int(seq_dim), int(batch_dim), int(hidden_size)],
    )
    bw_output_name = ctx.add_intermediate_tensor(
        f"{y_name}_bw",
        dtype=y_dtype,
        shape=[int(seq_dim), int(batch_dim), int(hidden_size)],
    )
    fw_expanded_name = ctx.add_intermediate_tensor(
        f"{y_name}_fw_expanded",
        dtype=y_dtype,
        shape=[int(seq_dim), 1, int(batch_dim), int(hidden_size)],
    )
    bw_expanded_name = ctx.add_intermediate_tensor(
        f"{y_name}_bw_expanded",
        dtype=y_dtype,
        shape=[int(seq_dim), 1, int(batch_dim), int(hidden_size)],
    )

    bidirectional_inputs = [
        x_name,
        _add_const("fw_w_i", fw_w_i),
        _add_const("fw_w_f", fw_w_f),
        _add_const("fw_w_c", fw_w_c),
        _add_const("fw_w_o", fw_w_o),
        _add_const("fw_r_i", fw_r_i),
        _add_const("fw_r_f", fw_r_f),
        _add_const("fw_r_c", fw_r_c),
        _add_const("fw_r_o", fw_r_o),
        "",
        "",
        "",
        _add_const("fw_b_i", fw_b_i),
        _add_const("fw_b_f", fw_b_f),
        _add_const("fw_b_c", fw_b_c),
        _add_const("fw_b_o", fw_b_o),
        "",
        "",
        _add_const("bw_w_i", bw_w_i),
        _add_const("bw_w_f", bw_w_f),
        _add_const("bw_w_c", bw_w_c),
        _add_const("bw_w_o", bw_w_o),
        _add_const("bw_r_i", bw_r_i),
        _add_const("bw_r_f", bw_r_f),
        _add_const("bw_r_c", bw_r_c),
        _add_const("bw_r_o", bw_r_o),
        "",
        "",
        "",
        _add_const("bw_b_i", bw_b_i),
        _add_const("bw_b_f", bw_b_f),
        _add_const("bw_b_c", bw_b_c),
        _add_const("bw_b_o", bw_b_o),
        "",
        "",
        fw_h0_name,
        fw_c0_name,
        bw_h0_name,
        bw_c0_name,
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]
    if len(bidirectional_inputs) != 48:
        raise RuntimeError(
            f"Internal error: BIDIRECTIONAL_SEQUENCE_LSTM input count must be 48. got={len(bidirectional_inputs)}"
        )

    ctx.add_operator(
        OperatorIR(
            op_type="BIDIRECTIONAL_SEQUENCE_LSTM",
            inputs=bidirectional_inputs,
            outputs=[merged_output_name],
            options={
                "fusedActivationFunction": "TANH",
                "cellClip": float(node.attrs.get("clip", 0.0)),
                "projClip": 0.0,
                "mergeOutputs": True,
                "timeMajor": True,
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    split_axis_name = _add_const("split_axis", np.asarray(2, dtype=np.int32), dtype=np.int32)
    ctx.add_operator(
        OperatorIR(
            op_type="SPLIT",
            inputs=[split_axis_name, merged_output_name],
            outputs=[fw_output_name, bw_output_name],
            options={
                "numSplits": 2,
            },
        )
    )

    use_reshape_expand = (
        seq_signature_dim is not None
        and batch_signature_dim is not None
        and int(seq_signature_dim) > 0
        and int(batch_signature_dim) > 0
    )
    if use_reshape_expand:
        expand_shape = np.asarray(
            [int(seq_dim), 1, int(batch_dim), int(hidden_size)],
            dtype=np.int32,
        )
        expand_shape_name = _add_const("expand_shape", expand_shape, dtype=np.int32)
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[fw_output_name, expand_shape_name],
                outputs=[fw_expanded_name],
                options={
                    "newShape": [int(v) for v in list(expand_shape)],
                },
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[bw_output_name, expand_shape_name],
                outputs=[bw_expanded_name],
                options={
                    "newShape": [int(v) for v in list(expand_shape)],
                },
            )
        )
    else:
        expand_axis_name = _add_const("expand_axis", np.asarray(1, dtype=np.int32), dtype=np.int32)
        ctx.add_operator(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[fw_output_name, expand_axis_name],
                outputs=[fw_expanded_name],
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[bw_output_name, expand_axis_name],
                outputs=[bw_expanded_name],
            )
        )
    ctx.add_operator(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=[fw_expanded_name, bw_expanded_name],
            outputs=[y_name],
            options={
                "axis": 1,
                "fusedActivationFunction": "NONE",
            },
        )
    )
