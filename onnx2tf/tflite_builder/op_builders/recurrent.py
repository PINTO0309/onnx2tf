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
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "bidirectional"}:
        raise NotImplementedError(
            f"LSTM direction must be forward or bidirectional for flatbuffer_direct. op={node.name} direction={direction}"
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1

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
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NotImplementedError(
            "LSTM W/R num_directions mismatch for flatbuffer_direct. "
            f"op={node.name} direction={direction} expected_num_directions={expected_num_directions} "
            f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
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
        B = np.zeros((expected_num_directions, 8 * hidden_size), dtype=np.float32)
    if (
        B.ndim != 2
        or int(B.shape[0]) != expected_num_directions
        or int(B.shape[1]) != 8 * hidden_size
    ):
        raise NotImplementedError(
            "LSTM B must have shape [num_directions, 8*hidden_size]. "
            f"op={node.name} direction={direction} expected_num_directions={expected_num_directions} "
            f"hidden_size={hidden_size} B_shape={list(B.shape)}"
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
    if int(H0.shape[0]) != expected_num_directions or int(C0.shape[0]) != expected_num_directions:
        raise NotImplementedError(
            "LSTM initial_h/initial_c first dim must match num_directions. "
            f"op={node.name} direction={direction} expected_num_directions={expected_num_directions} "
            f"H0_shape={list(H0.shape)} C0_shape={list(C0.shape)}"
        )

    fw_w_i, fw_w_f, fw_w_c, fw_w_o = _split_onnx_lstm_gates(
        np.asarray(W[0], dtype=np.float32),
        hidden_size,
    )
    fw_r_i, fw_r_f, fw_r_c, fw_r_o = _split_onnx_lstm_gates(
        np.asarray(R[0], dtype=np.float32),
        hidden_size,
    )
    fw_b_i, fw_b_f, fw_b_c, fw_b_o = _split_onnx_lstm_bias_gates(
        np.asarray(B[0], dtype=np.float32),
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
        y_tensor.shape = [int(seq_dim), int(expected_num_directions), int(batch_dim), int(hidden_size)]
        y_tensor.shape_signature = [
            int(seq_signature_dim) if seq_signature_dim is not None else int(seq_dim),
            int(expected_num_directions),
            int(batch_signature_dim) if batch_signature_dim is not None else int(batch_dim),
            int(hidden_size),
        ]
    use_reshape_expand = (
        seq_signature_dim is not None
        and batch_signature_dim is not None
        and int(seq_signature_dim) > 0
        and int(batch_signature_dim) > 0
    )
    if expected_num_directions == 1:
        fw_h0_name = _add_zero_variable_state("fw_h0_state", np.asarray(H0[0], dtype=np.float32))
        fw_c0_name = _add_zero_variable_state("fw_c0_state", np.asarray(C0[0], dtype=np.float32))
        uni_output_name = ctx.add_intermediate_tensor(
            f"{y_name}_lstm_uni",
            dtype=y_dtype,
            shape=[int(seq_dim), int(batch_dim), int(hidden_size)],
        )
        uni_inputs = [
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
            fw_h0_name,
            fw_c0_name,
            "",
            "",
            "",
            "",
        ]
        if len(uni_inputs) != 24:
            raise RuntimeError(
                f"Internal error: UNIDIRECTIONAL_SEQUENCE_LSTM input count must be 24. got={len(uni_inputs)}"
            )
        ctx.add_operator(
            OperatorIR(
                op_type="UNIDIRECTIONAL_SEQUENCE_LSTM",
                inputs=uni_inputs,
                outputs=[uni_output_name],
                options={
                    "fusedActivationFunction": "TANH",
                    "cellClip": float(node.attrs.get("clip", 0.0)),
                    "projClip": 0.0,
                    "timeMajor": True,
                    "asymmetricQuantizeInputs": False,
                    "diagonalRecurrentTensors": False,
                },
            )
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
                    inputs=[uni_output_name, expand_shape_name],
                    outputs=[y_name],
                    options={"newShape": [int(v) for v in list(expand_shape)]},
                )
            )
        else:
            expand_axis_name = _add_const("expand_axis", np.asarray(1, dtype=np.int32), dtype=np.int32)
            ctx.add_operator(
                OperatorIR(
                    op_type="EXPAND_DIMS",
                    inputs=[uni_output_name, expand_axis_name],
                    outputs=[y_name],
                )
            )
        return

    bw_w_i, bw_w_f, bw_w_c, bw_w_o = _split_onnx_lstm_gates(
        np.asarray(W[1], dtype=np.float32),
        hidden_size,
    )
    bw_r_i, bw_r_f, bw_r_c, bw_r_o = _split_onnx_lstm_gates(
        np.asarray(R[1], dtype=np.float32),
        hidden_size,
    )
    bw_b_i, bw_b_f, bw_b_c, bw_b_o = _split_onnx_lstm_bias_gates(
        np.asarray(B[1], dtype=np.float32),
        hidden_size,
    )

    fw_h0_name = _add_zero_variable_state("fw_h0_state", np.asarray(H0[0], dtype=np.float32))
    fw_c0_name = _add_zero_variable_state("fw_c0_state", np.asarray(C0[0], dtype=np.float32))
    bw_h0_name = _add_zero_variable_state("bw_h0_state", np.asarray(H0[1], dtype=np.float32))
    bw_c0_name = _add_zero_variable_state("bw_c0_state", np.asarray(C0[1], dtype=np.float32))

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


def build_rnn_op(node: Any, ctx: Any) -> None:
    original_inputs = _get_original_node_inputs(node, ctx)
    x_name = _input_name(original_inputs, 0)
    w_name = _input_name(original_inputs, 1)
    r_name = _input_name(original_inputs, 2)
    b_name = _input_name(original_inputs, 3)
    sequence_lens_name = _input_name(original_inputs, 4)
    initial_h_name = _input_name(original_inputs, 5)

    y_name = node.outputs[0].name
    y_h_name = node.outputs[1].name if len(node.outputs) > 1 else ""
    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(y_name)
    if y_h_name != "":
        ctx.ensure_tensor(y_h_name)

    w_value = ctx.get_constant_array(w_name)
    r_value = ctx.get_constant_array(r_name)
    if w_value is None or r_value is None:
        raise NotImplementedError(
            f"RNN W/R must be constant for flatbuffer_direct. op={node.name}"
        )
    W = np.asarray(w_value, dtype=np.float32)
    R = np.asarray(r_value, dtype=np.float32)
    if W.ndim != 3 or R.ndim != 3:
        raise NotImplementedError(
            f"RNN W/R must be rank-3. op={node.name} W_shape={list(W.shape)} R_shape={list(R.shape)}"
        )
    if int(W.shape[0]) != 1 or int(R.shape[0]) != 1:
        raise NotImplementedError(
            f"RNN builtin lowering currently supports num_directions=1 only. op={node.name}"
        )
    hidden_size = int(node.attrs.get("hidden_size", int(W.shape[1])))
    if int(W.shape[1]) != hidden_size or int(R.shape[1]) != hidden_size:
        raise NotImplementedError(
            "RNN hidden_size mismatch in W/R for flatbuffer_direct. "
            f"op={node.name} hidden_size={hidden_size} W_shape={list(W.shape)} R_shape={list(R.shape)}"
        )
    input_size = int(W.shape[2])
    if int(R.shape[2]) != hidden_size:
        raise NotImplementedError(
            "RNN recurrent weight shape mismatch for flatbuffer_direct. "
            f"op={node.name} R_shape={list(R.shape)} hidden_size={hidden_size}"
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    if len(input_shape) != 3:
        raise NotImplementedError(
            f"RNN input must be rank-3 [seq,batch,input] in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    if int(input_shape[2]) != input_size and int(input_shape[1]) == input_size:
        transposed_x_name = ctx.add_intermediate_tensor(
            f"{node.name}_rnn_input_sbi",
            dtype=str(ctx.get_tensor_dtype(x_name)).upper(),
            shape=[int(input_shape[0]), int(input_shape[2]), int(input_shape[1])],
        )
        perm_name = ctx.add_const_tensor(
            f"{node.name}_rnn_input_sbi_perm",
            np.asarray([0, 2, 1], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[x_name, perm_name],
                outputs=[transposed_x_name],
            )
        )
        x_name = transposed_x_name
        input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    seq_len = int(input_shape[0]) if int(input_shape[0]) > 0 else 1
    batch = int(input_shape[1]) if int(input_shape[1]) > 0 else 1
    if int(input_shape[2]) != input_size:
        raise NotImplementedError(
            "RNN input size mismatch with W in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape} W_shape={list(W.shape)}"
        )

    if sequence_lens_name != "":
        raise NotImplementedError(
            f"RNN sequence_lens input is not supported in flatbuffer_direct. op={node.name}"
        )

    if b_name != "":
        b_value = ctx.get_constant_array(b_name)
        if b_value is None:
            raise NotImplementedError(
                f"RNN bias must be constant when provided. op={node.name}"
            )
        B = np.asarray(b_value, dtype=np.float32)
        if B.ndim != 2 or int(B.shape[0]) != 1 or int(B.shape[1]) != 2 * hidden_size:
            raise NotImplementedError(
                "RNN bias shape must be [1, 2*hidden_size] in flatbuffer_direct. "
                f"op={node.name} B_shape={list(B.shape)} hidden_size={hidden_size}"
            )
        bias = np.asarray(B[0, :hidden_size] + B[0, hidden_size:], dtype=np.float32)
    else:
        bias = np.zeros((hidden_size,), dtype=np.float32)

    if initial_h_name != "":
        initial_h = ctx.get_constant_array(initial_h_name)
        if initial_h is None:
            raise NotImplementedError(
                f"RNN initial_h must be constant when provided. op={node.name}"
            )
        init_h = np.asarray(initial_h, dtype=np.float32)
        if init_h.ndim != 3 or int(init_h.shape[0]) != 1 or int(init_h.shape[2]) != hidden_size:
            raise NotImplementedError(
                "RNN initial_h shape must be [1, batch, hidden_size] in flatbuffer_direct. "
                f"op={node.name} initial_h_shape={list(init_h.shape)} hidden_size={hidden_size}"
            )

    activations = node.attrs.get("activations", ["Tanh"])
    if not isinstance(activations, (list, tuple)) or len(activations) == 0:
        activations = ["Tanh"]
    activation = str(activations[0]).lower()
    fused_activation = "TANH"
    if activation in {"tanh"}:
        fused_activation = "TANH"
    elif activation in {"relu"}:
        fused_activation = "RELU"
    elif activation in {"sigmoid"}:
        fused_activation = "LOGISTIC"
    else:
        raise NotImplementedError(
            f"RNN activation is not supported in flatbuffer_direct. op={node.name} activation={activation}"
        )
    clip = float(node.attrs.get("clip", 0.0))
    if abs(clip) > 1e-6:
        raise NotImplementedError(
            f"RNN clip is not supported in flatbuffer_direct. op={node.name} clip={clip}"
        )

    input_weights_name = ctx.add_const_tensor(
        f"{node.name}_rnn_input_weights",
        np.asarray(W[0], dtype=np.float32),
    )
    recurrent_weights_name = ctx.add_const_tensor(
        f"{node.name}_rnn_recurrent_weights",
        np.asarray(R[0], dtype=np.float32),
    )
    bias_name = ctx.add_const_tensor(
        f"{node.name}_rnn_bias",
        np.asarray(bias, dtype=np.float32),
    )

    hidden_state_name = ctx.add_intermediate_tensor(
        f"{node.name}_rnn_hidden_state",
        dtype="FLOAT32",
        shape=[int(batch), int(hidden_size)],
    )
    hidden_state_tensor = ctx.model_ir.tensors[hidden_state_name]
    hidden_state_tensor.is_variable = True
    hidden_state_tensor.data = None

    y_seq_name = ctx.add_intermediate_tensor(
        f"{y_name}_rnn_seq",
        dtype=str(ctx.get_tensor_dtype(y_name)).upper(),
        shape=[int(seq_len), int(batch), int(hidden_size)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="UNIDIRECTIONAL_SEQUENCE_RNN",
            inputs=[
                x_name,
                input_weights_name,
                recurrent_weights_name,
                bias_name,
                hidden_state_name,
            ],
            outputs=[y_seq_name],
            options={
                "timeMajor": True,
                "fusedActivationFunction": fused_activation,
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    y_shape = [int(v) for v in ctx.get_tensor_shape(y_name)]
    if len(y_shape) == 4:
        axis_name = ctx.add_const_tensor(
            f"{y_name}_rnn_expand_axis",
            np.asarray(1, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[y_seq_name, axis_name],
                outputs=[y_name],
            )
        )
    elif len(y_shape) == 3:
        shape_name = ctx.add_const_tensor(
            f"{y_name}_rnn_shape",
            np.asarray(y_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[y_seq_name, shape_name],
                outputs=[y_name],
                options={"newShape": y_shape},
            )
        )
    else:
        raise NotImplementedError(
            f"RNN Y output rank must be 3 or 4 for flatbuffer_direct. op={node.name} y_shape={y_shape}"
        )

    if y_h_name != "":
        begin_name = ctx.add_const_tensor(
            f"{y_h_name}_rnn_begin",
            np.asarray([max(seq_len - 1, 0), 0, 0], dtype=np.int32),
        )
        size_name = ctx.add_const_tensor(
            f"{y_h_name}_rnn_size",
            np.asarray([1, int(batch), int(hidden_size)], dtype=np.int32),
        )
        y_h_seq_name = ctx.add_intermediate_tensor(
            f"{y_h_name}_rnn_seq_slice",
            dtype=str(ctx.get_tensor_dtype(y_h_name)).upper(),
            shape=[1, int(batch), int(hidden_size)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SLICE",
                inputs=[y_seq_name, begin_name, size_name],
                outputs=[y_h_seq_name],
            )
        )
        y_h_shape = [int(v) for v in ctx.get_tensor_shape(y_h_name)]
        if y_h_shape != [1, int(batch), int(hidden_size)]:
            y_h_shape_name = ctx.add_const_tensor(
                f"{y_h_name}_rnn_shape",
                np.asarray(y_h_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_h_seq_name, y_h_shape_name],
                    outputs=[y_h_name],
                    options={"newShape": y_h_shape},
                )
            )
        else:
            pass_shape = ctx.add_const_tensor(
                f"{y_h_name}_rnn_identity_shape",
                np.asarray(y_h_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_h_seq_name, pass_shape],
                    outputs=[y_h_name],
                    options={"newShape": y_h_shape},
                )
            )


def build_gru_op(node: Any, ctx: Any) -> None:
    original_inputs = _get_original_node_inputs(node, ctx)
    x_name = _input_name(original_inputs, 0)
    w_name = _input_name(original_inputs, 1)
    r_name = _input_name(original_inputs, 2)
    b_name = _input_name(original_inputs, 3)
    sequence_lens_name = _input_name(original_inputs, 4)
    initial_h_name = _input_name(original_inputs, 5)

    y_name = node.outputs[0].name
    y_h_name = node.outputs[1].name if len(node.outputs) > 1 else ""
    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(y_name)
    if y_h_name != "":
        ctx.ensure_tensor(y_h_name)

    w_value = ctx.get_constant_array(w_name)
    r_value = ctx.get_constant_array(r_name)
    if w_value is None or r_value is None:
        raise NotImplementedError(
            f"GRU W/R must be constant for flatbuffer_direct. op={node.name}"
        )
    W = np.asarray(w_value, dtype=np.float32)
    R = np.asarray(r_value, dtype=np.float32)
    if W.ndim != 3 or R.ndim != 3:
        raise NotImplementedError(
            f"GRU W/R must be rank-3. op={node.name} W_shape={list(W.shape)} R_shape={list(R.shape)}"
        )
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NotImplementedError(
            f"GRU direction is not supported in flatbuffer_direct. op={node.name} direction={direction}"
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NotImplementedError(
            "GRU num_directions mismatch for flatbuffer_direct. "
            f"op={node.name} direction={direction} W_shape={list(W.shape)} R_shape={list(R.shape)}"
        )

    hidden_size = int(node.attrs.get("hidden_size", int(W.shape[1] // 3)))
    if int(W.shape[1]) != 3 * hidden_size or int(R.shape[1]) != 3 * hidden_size:
        raise NotImplementedError(
            "GRU hidden_size mismatch in W/R for flatbuffer_direct. "
            f"op={node.name} hidden_size={hidden_size} W_shape={list(W.shape)} R_shape={list(R.shape)}"
        )
    input_size = int(W.shape[2])
    if int(R.shape[2]) != hidden_size:
        raise NotImplementedError(
            "GRU recurrent weight shape mismatch for flatbuffer_direct. "
            f"op={node.name} R_shape={list(R.shape)} hidden_size={hidden_size}"
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    if len(input_shape) != 3:
        raise NotImplementedError(
            f"GRU input must be rank-3 [seq,batch,input] in flatbuffer_direct. op={node.name} input_shape={input_shape}"
        )
    if int(input_shape[2]) != input_size and int(input_shape[1]) == input_size:
        transposed_x_name = ctx.add_intermediate_tensor(
            f"{node.name}_gru_input_sbi",
            dtype="FLOAT32",
            shape=[int(input_shape[0]), int(input_shape[2]), int(input_shape[1])],
        )
        perm_name = ctx.add_const_tensor(
            f"{node.name}_gru_input_sbi_perm",
            np.asarray([0, 2, 1], dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[x_name, perm_name],
                outputs=[transposed_x_name],
            )
        )
        x_name = transposed_x_name
        input_shape = [int(v) for v in ctx.get_tensor_shape(x_name)]
    seq_len = int(input_shape[0])
    batch = int(input_shape[1])
    if seq_len <= 0 or batch <= 0:
        raise NotImplementedError(
            "GRU builtin lowering requires static positive seq_len and batch in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape}"
        )
    if int(input_shape[2]) != input_size:
        raise NotImplementedError(
            "GRU input size mismatch with W in flatbuffer_direct. "
            f"op={node.name} input_shape={input_shape} W_shape={list(W.shape)}"
        )

    if sequence_lens_name != "":
        raise NotImplementedError(
            f"GRU sequence_lens input is not supported in flatbuffer_direct. op={node.name}"
        )

    linear_before_reset = int(node.attrs.get("linear_before_reset", 0))
    if linear_before_reset not in {0, 1}:
        raise NotImplementedError(
            "GRU linear_before_reset must be 0 or 1 in flatbuffer_direct builtin lowering. "
            f"op={node.name} linear_before_reset={linear_before_reset}"
        )

    W0 = np.asarray(W[0], dtype=np.float32)
    R0 = np.asarray(R[0], dtype=np.float32)
    Wz = np.asarray(W0[0:hidden_size, :], dtype=np.float32)
    Wr = np.asarray(W0[hidden_size:2 * hidden_size, :], dtype=np.float32)
    Wh = np.asarray(W0[2 * hidden_size:3 * hidden_size, :], dtype=np.float32)
    Rz = np.asarray(R0[0:hidden_size, :], dtype=np.float32)
    Rr = np.asarray(R0[hidden_size:2 * hidden_size, :], dtype=np.float32)
    Rh = np.asarray(R0[2 * hidden_size:3 * hidden_size, :], dtype=np.float32)

    if b_name != "":
        b_value = ctx.get_constant_array(b_name)
        if b_value is None:
            raise NotImplementedError(
                f"GRU bias must be constant when provided. op={node.name}"
            )
        B = np.asarray(b_value, dtype=np.float32)
        if (
            B.ndim != 2
            or int(B.shape[0]) != expected_num_directions
            or int(B.shape[1]) != 6 * hidden_size
        ):
            raise NotImplementedError(
                "GRU B must have shape [num_directions, 6*hidden_size] in flatbuffer_direct. "
                f"op={node.name} B_shape={list(B.shape)} hidden_size={hidden_size} "
                f"num_directions={expected_num_directions}"
            )
    else:
        B = np.zeros((expected_num_directions, 6 * hidden_size), dtype=np.float32)

    def _add_const(suffix: str, values: np.ndarray) -> str:
        return ctx.add_const_tensor(
            f"{node.name}_{suffix}",
            np.asarray(values, dtype=np.float32),
        )

    if initial_h_name != "":
        ctx.ensure_tensor(initial_h_name)
        initial_h_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        if initial_h_shape != [expected_num_directions, batch, hidden_size]:
            raise NotImplementedError(
                "GRU initial_h shape must be [num_directions, batch, hidden_size] in flatbuffer_direct. "
                f"op={node.name} initial_h_shape={initial_h_shape} num_directions={expected_num_directions}"
            )

    def _run_gru_direction(
        *,
        dir_index: int,
        reverse: bool,
        dir_tag: str,
    ) -> Tuple[str, str]:
        Wd = np.asarray(W[dir_index], dtype=np.float32)
        Rd = np.asarray(R[dir_index], dtype=np.float32)
        Bd = np.asarray(B[dir_index], dtype=np.float32)
        Wz = np.asarray(Wd[0:hidden_size, :], dtype=np.float32)
        Wr = np.asarray(Wd[hidden_size:2 * hidden_size, :], dtype=np.float32)
        Wh = np.asarray(Wd[2 * hidden_size:3 * hidden_size, :], dtype=np.float32)
        Rz = np.asarray(Rd[0:hidden_size, :], dtype=np.float32)
        Rr = np.asarray(Rd[hidden_size:2 * hidden_size, :], dtype=np.float32)
        Rh = np.asarray(Rd[2 * hidden_size:3 * hidden_size, :], dtype=np.float32)

        Wbz = np.asarray(Bd[0:hidden_size], dtype=np.float32)
        Wbr = np.asarray(Bd[hidden_size:2 * hidden_size], dtype=np.float32)
        Wbh = np.asarray(Bd[2 * hidden_size:3 * hidden_size], dtype=np.float32)
        Rbz = np.asarray(Bd[3 * hidden_size:4 * hidden_size], dtype=np.float32)
        Rbr = np.asarray(Bd[4 * hidden_size:5 * hidden_size], dtype=np.float32)
        Rbh = np.asarray(Bd[5 * hidden_size:6 * hidden_size], dtype=np.float32)
        bz = np.asarray(Wbz + Rbz, dtype=np.float32)
        br = np.asarray(Wbr + Rbr, dtype=np.float32)
        bh = np.asarray(Wbh + Rbh, dtype=np.float32)

        wz_t_name = _add_const(f"gru_{dir_tag}_wz_t", np.asarray(Wz.T, dtype=np.float32))
        wr_t_name = _add_const(f"gru_{dir_tag}_wr_t", np.asarray(Wr.T, dtype=np.float32))
        wh_t_name = _add_const(f"gru_{dir_tag}_wh_t", np.asarray(Wh.T, dtype=np.float32))
        rz_t_name = _add_const(f"gru_{dir_tag}_rz_t", np.asarray(Rz.T, dtype=np.float32))
        rr_t_name = _add_const(f"gru_{dir_tag}_rr_t", np.asarray(Rr.T, dtype=np.float32))
        rh_t_name = _add_const(f"gru_{dir_tag}_rh_t", np.asarray(Rh.T, dtype=np.float32))
        bz_name = _add_const(f"gru_{dir_tag}_bz", bz)
        br_name = _add_const(f"gru_{dir_tag}_br", br)
        wbh_name = _add_const(f"gru_{dir_tag}_wbh", Wbh)
        rbh_name = _add_const(f"gru_{dir_tag}_rbh", Rbh)
        bh_name = _add_const(f"gru_{dir_tag}_bh", bh)
        one_name = _add_const(f"gru_{dir_tag}_one", np.asarray(1.0, dtype=np.float32))

        if initial_h_name != "":
            h_init_slice_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_h_prev_init_slice",
                dtype="FLOAT32",
                shape=[1, batch, hidden_size],
            )
            h_init_begin_name = ctx.add_const_tensor(
                f"{node.name}_gru_{dir_tag}_h_prev_init_begin",
                np.asarray([dir_index, 0, 0], dtype=np.int32),
            )
            h_init_size_name = ctx.add_const_tensor(
                f"{node.name}_gru_{dir_tag}_h_prev_init_size",
                np.asarray([1, batch, hidden_size], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[initial_h_name, h_init_begin_name, h_init_size_name],
                    outputs=[h_init_slice_name],
                )
            )
            h_prev = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_h_prev_init",
                dtype="FLOAT32",
                shape=[batch, hidden_size],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SQUEEZE",
                    inputs=[h_init_slice_name],
                    outputs=[h_prev],
                    options={"squeezeDims": [0]},
                )
            )
        else:
            h_prev = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_h_prev_zero",
                dtype="FLOAT32",
                shape=[batch, hidden_size],
            )
            h_prev_tensor = ctx.model_ir.tensors[h_prev]
            h_prev_tensor.is_variable = True
            h_prev_tensor.data = None

        h_seq_by_time: List[Optional[str]] = [None] * seq_len
        for step in range(seq_len):
            t = (seq_len - 1 - step) if reverse else step
            x_slice_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_x_t_slice_{step}",
            dtype="FLOAT32",
            shape=[1, batch, input_size],
            )
            begin_name = ctx.add_const_tensor(
                f"{node.name}_gru_{dir_tag}_x_t_begin_{step}",
                np.asarray([t, 0, 0], dtype=np.int32),
            )
            size_name = ctx.add_const_tensor(
                f"{node.name}_gru_{dir_tag}_x_t_size_{step}",
                np.asarray([1, batch, input_size], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[x_name, begin_name, size_name],
                    outputs=[x_slice_name],
                )
            )
            x_t_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_x_t_{step}",
                dtype="FLOAT32",
                shape=[batch, input_size],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SQUEEZE",
                    inputs=[x_slice_name],
                    outputs=[x_t_name],
                    options={"squeezeDims": [0]},
                )
            )

            xwz_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_xwz_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            hrz_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_hrz_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            z_pre_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_z_pre_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            z_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_z_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            ctx.add_operator(OperatorIR(op_type="BATCH_MATMUL", inputs=[x_t_name, wz_t_name], outputs=[xwz_name]))
            ctx.add_operator(OperatorIR(op_type="BATCH_MATMUL", inputs=[h_prev, rz_t_name], outputs=[hrz_name]))
            z_sum_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_z_sum_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="ADD",
                    inputs=[xwz_name, hrz_name],
                    outputs=[z_sum_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="ADD",
                    inputs=[z_sum_name, bz_name],
                    outputs=[z_pre_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            ctx.add_operator(OperatorIR(op_type="LOGISTIC", inputs=[z_pre_name], outputs=[z_name]))

            xwr_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_xwr_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            hrr_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_hrr_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            r_pre_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_r_pre_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            r_name_t = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_r_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            ctx.add_operator(OperatorIR(op_type="BATCH_MATMUL", inputs=[x_t_name, wr_t_name], outputs=[xwr_name]))
            ctx.add_operator(OperatorIR(op_type="BATCH_MATMUL", inputs=[h_prev, rr_t_name], outputs=[hrr_name]))
            r_sum_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_r_sum_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="ADD",
                    inputs=[xwr_name, hrr_name],
                    outputs=[r_sum_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="ADD",
                    inputs=[r_sum_name, br_name],
                    outputs=[r_pre_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            ctx.add_operator(OperatorIR(op_type="LOGISTIC", inputs=[r_pre_name], outputs=[r_name_t]))

            xwh_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_xwh_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            rh_in_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_rh_in_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            rh_proj_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_rh_proj_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            rh_proj_bias_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_rh_proj_bias_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            rh_proj_gated_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_rh_proj_gated_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            h_pre_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_h_pre_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            h_cand_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_h_cand_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            ctx.add_operator(OperatorIR(op_type="BATCH_MATMUL", inputs=[x_t_name, wh_t_name], outputs=[xwh_name]))
            ctx.add_operator(
                OperatorIR(
                    op_type="MUL",
                    inputs=[r_name_t, h_prev],
                    outputs=[rh_in_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            if linear_before_reset == 0:
                ctx.add_operator(
                    OperatorIR(
                        op_type="BATCH_MATMUL",
                        inputs=[rh_in_name, rh_t_name],
                        outputs=[rh_proj_name],
                    )
                )
            else:
                ctx.add_operator(
                    OperatorIR(
                        op_type="BATCH_MATMUL",
                        inputs=[h_prev, rh_t_name],
                        outputs=[rh_proj_name],
                    )
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="ADD",
                        inputs=[rh_proj_name, rbh_name],
                        outputs=[rh_proj_bias_name],
                        options={"fusedActivationFunction": "NONE"},
                    )
                )
                ctx.add_operator(
                    OperatorIR(
                        op_type="MUL",
                        inputs=[r_name_t, rh_proj_bias_name],
                        outputs=[rh_proj_gated_name],
                        options={"fusedActivationFunction": "NONE"},
                    )
                )
            h_sum_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_h_sum_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            h_rhs_name = rh_proj_name if linear_before_reset == 0 else rh_proj_gated_name
            ctx.add_operator(
                OperatorIR(
                    op_type="ADD",
                    inputs=[xwh_name, h_rhs_name],
                    outputs=[h_sum_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            h_bias_name = bh_name if linear_before_reset == 0 else wbh_name
            ctx.add_operator(
                OperatorIR(
                    op_type="ADD",
                    inputs=[h_sum_name, h_bias_name],
                    outputs=[h_pre_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            ctx.add_operator(OperatorIR(op_type="TANH", inputs=[h_pre_name], outputs=[h_cand_name]))

            one_minus_z_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_one_minus_z_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            term_new_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_term_new_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            term_prev_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_term_prev_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            h_new_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_h_{step}",
            dtype="FLOAT32",
            shape=[batch, hidden_size],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SUB",
                    inputs=[one_name, z_name],
                    outputs=[one_minus_z_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MUL",
                    inputs=[one_minus_z_name, h_cand_name],
                    outputs=[term_new_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="MUL",
                    inputs=[z_name, h_prev],
                    outputs=[term_prev_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="ADD",
                    inputs=[term_new_name, term_prev_name],
                    outputs=[h_new_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )

            h_step_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_h_step_{step}",
            dtype="FLOAT32",
            shape=[1, batch, hidden_size],
            )
            h_step_shape_name = ctx.add_const_tensor(
                f"{node.name}_gru_{dir_tag}_h_step_shape_{step}",
                np.asarray([1, batch, hidden_size], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[h_new_name, h_step_shape_name],
                    outputs=[h_step_name],
                    options={"newShape": [1, batch, hidden_size]},
                )
            )
            h_seq_by_time[t] = h_step_name
            h_prev = h_new_name

        ordered_steps = [v for v in h_seq_by_time if v is not None]
        if len(ordered_steps) != seq_len:
            raise RuntimeError(
                f"Internal error: GRU direction sequence assembly failed. op={node.name} dir={dir_tag}"
            )
        y_seq_name = ordered_steps[0]
        if len(ordered_steps) > 1:
            y_seq_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_{dir_tag}_seq",
                dtype="FLOAT32",
                shape=[seq_len, batch, hidden_size],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=ordered_steps,
                    outputs=[y_seq_name],
                    options={"axis": 0, "fusedActivationFunction": "NONE"},
                )
            )
        return y_seq_name, h_prev

    y_shape = [int(v) for v in ctx.get_tensor_shape(y_name)]
    y_dtype = str(ctx.get_tensor_dtype(y_name)).upper()
    y_internal_name = y_name
    if y_dtype != "FLOAT32":
        y_internal_name = ctx.add_intermediate_tensor(
            f"{y_name}_gru_f32",
            dtype="FLOAT32",
            shape=y_shape,
        )

    y_h_shape: List[int] = []
    y_h_dtype = "FLOAT32"
    y_h_internal_name = y_h_name
    if y_h_name != "":
        y_h_shape = [int(v) for v in ctx.get_tensor_shape(y_h_name)]
        y_h_dtype = str(ctx.get_tensor_dtype(y_h_name)).upper()
        if y_h_dtype != "FLOAT32":
            y_h_internal_name = ctx.add_intermediate_tensor(
                f"{y_h_name}_gru_f32",
                dtype="FLOAT32",
                shape=y_h_shape,
            )

    if direction == "bidirectional":
        fw_seq_name, fw_last_name = _run_gru_direction(
            dir_index=0,
            reverse=False,
            dir_tag="fw",
        )
        bw_seq_name, bw_last_name = _run_gru_direction(
            dir_index=1,
            reverse=True,
            dir_tag="bw",
        )

        fw_seq_4d_name = ctx.add_intermediate_tensor(
            f"{node.name}_gru_fw_seq_4d",
            dtype="FLOAT32",
            shape=[seq_len, 1, batch, hidden_size],
        )
        bw_seq_4d_name = ctx.add_intermediate_tensor(
            f"{node.name}_gru_bw_seq_4d",
            dtype="FLOAT32",
            shape=[seq_len, 1, batch, hidden_size],
        )
        expand_axis_name = ctx.add_const_tensor(
            f"{node.name}_gru_expand_axis",
            np.asarray(1, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[fw_seq_name, expand_axis_name],
                outputs=[fw_seq_4d_name],
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[bw_seq_name, expand_axis_name],
                outputs=[bw_seq_4d_name],
            )
        )
        y_4d_name = ctx.add_intermediate_tensor(
            f"{node.name}_gru_y_4d",
            dtype="FLOAT32",
            shape=[seq_len, 2, batch, hidden_size],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[fw_seq_4d_name, bw_seq_4d_name],
                outputs=[y_4d_name],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            )
        )

        if len(y_shape) == 4:
            y_shape_name = ctx.add_const_tensor(
                f"{y_name}_gru_shape",
                np.asarray(y_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_4d_name, y_shape_name],
                    outputs=[y_internal_name],
                    options={"newShape": y_shape},
                )
            )
        elif len(y_shape) == 3:
            y_transposed_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_y_transposed",
                dtype="FLOAT32",
                shape=[seq_len, batch, 2, hidden_size],
            )
            y_transpose_perm_name = ctx.add_const_tensor(
                f"{node.name}_gru_y_transpose_perm",
                np.asarray([0, 2, 1, 3], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[y_4d_name, y_transpose_perm_name],
                    outputs=[y_transposed_name],
                )
            )
            y_shape_name = ctx.add_const_tensor(
                f"{y_name}_gru_shape",
                np.asarray(y_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_transposed_name, y_shape_name],
                    outputs=[y_internal_name],
                    options={"newShape": y_shape},
                )
            )
        else:
            raise NotImplementedError(
                f"GRU Y output rank must be 3 or 4 for bidirectional. op={node.name} y_shape={y_shape}"
            )

        if y_h_name != "":
            fw_last_3d_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_fw_last_3d",
                dtype="FLOAT32",
                shape=[1, batch, hidden_size],
            )
            bw_last_3d_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_bw_last_3d",
                dtype="FLOAT32",
                shape=[1, batch, hidden_size],
            )
            fw_last_shape_name = ctx.add_const_tensor(
                f"{node.name}_gru_fw_last_shape",
                np.asarray([1, batch, hidden_size], dtype=np.int32),
            )
            bw_last_shape_name = ctx.add_const_tensor(
                f"{node.name}_gru_bw_last_shape",
                np.asarray([1, batch, hidden_size], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[fw_last_name, fw_last_shape_name],
                    outputs=[fw_last_3d_name],
                    options={"newShape": [1, batch, hidden_size]},
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[bw_last_name, bw_last_shape_name],
                    outputs=[bw_last_3d_name],
                    options={"newShape": [1, batch, hidden_size]},
                )
            )
            y_h_merged_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_y_h_merged",
                dtype="FLOAT32",
                shape=[2, batch, hidden_size],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=[fw_last_3d_name, bw_last_3d_name],
                    outputs=[y_h_merged_name],
                    options={"axis": 0, "fusedActivationFunction": "NONE"},
                )
            )
            y_h_shape_name = ctx.add_const_tensor(
                f"{y_h_name}_gru_shape",
                np.asarray(y_h_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_h_merged_name, y_h_shape_name],
                    outputs=[y_h_internal_name],
                    options={"newShape": y_h_shape},
                )
            )
    else:
        reverse = direction == "reverse"
        dir_tag = "rv" if reverse else "fw"
        seq_name, last_name = _run_gru_direction(
            dir_index=0,
            reverse=reverse,
            dir_tag=dir_tag,
        )

        if len(y_shape) == 4:
            y_seq_4d_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_y_4d",
                dtype="FLOAT32",
                shape=[seq_len, 1, batch, hidden_size],
            )
            axis_name = ctx.add_const_tensor(
                f"{node.name}_gru_expand_axis",
                np.asarray(1, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="EXPAND_DIMS",
                    inputs=[seq_name, axis_name],
                    outputs=[y_seq_4d_name],
                )
            )
            y_shape_name = ctx.add_const_tensor(
                f"{y_name}_gru_shape",
                np.asarray(y_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_seq_4d_name, y_shape_name],
                    outputs=[y_internal_name],
                    options={"newShape": y_shape},
                )
            )
        elif len(y_shape) == 3:
            y_shape_name = ctx.add_const_tensor(
                f"{y_name}_gru_shape",
                np.asarray(y_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[seq_name, y_shape_name],
                    outputs=[y_internal_name],
                    options={"newShape": y_shape},
                )
            )
        else:
            raise NotImplementedError(
                f"GRU Y output rank must be 3 or 4 for flatbuffer_direct. op={node.name} y_shape={y_shape}"
            )

        if y_h_name != "":
            h_last_3d_name = ctx.add_intermediate_tensor(
                f"{node.name}_gru_last_3d",
                dtype="FLOAT32",
                shape=[1, batch, hidden_size],
            )
            h_last_shape_name = ctx.add_const_tensor(
                f"{node.name}_gru_last_shape",
                np.asarray([1, batch, hidden_size], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[last_name, h_last_shape_name],
                    outputs=[h_last_3d_name],
                    options={"newShape": [1, batch, hidden_size]},
                )
            )
            y_h_shape_name = ctx.add_const_tensor(
                f"{y_h_name}_gru_shape",
                np.asarray(y_h_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[h_last_3d_name, y_h_shape_name],
                    outputs=[y_h_internal_name],
                    options={"newShape": y_h_shape},
                )
            )

    if y_internal_name != y_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[y_internal_name],
                outputs=[y_name],
                options={
                    "inDataType": "FLOAT32",
                    "outDataType": y_dtype,
                },
            )
        )

    if y_h_name != "" and y_h_internal_name != y_h_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[y_h_internal_name],
                outputs=[y_h_name],
                options={
                    "inDataType": "FLOAT32",
                    "outDataType": y_h_dtype,
                },
            )
        )
