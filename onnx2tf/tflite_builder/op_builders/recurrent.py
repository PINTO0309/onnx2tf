from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import OperatorIR
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


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


def _tensor_shape_with_signature(ctx: Any, tensor_name: str) -> List[int]:
    shape = [int(v) for v in list(ctx.get_tensor_shape(tensor_name))]
    tensor = ctx.model_ir.tensors.get(tensor_name, None)
    signature = (
        [int(v) for v in list(tensor.shape_signature)]
        if tensor is not None and tensor.shape_signature is not None
        else [int(v) for v in shape]
    )
    if len(signature) != len(shape):
        return [int(v) for v in shape]
    return [
        int(signature[idx]) if int(signature[idx]) < 0 else int(shape[idx])
        for idx in range(len(shape))
    ]


def _add_reshape_operator(
    *,
    ctx: Any,
    input_name: str,
    output_name: str,
    new_shape: List[int],
    preserve_dynamic_shape: bool = False,
) -> None:
    shape_name = ctx.add_const_tensor(
        f"{output_name}_reshape_shape",
        np.asarray([int(v) for v in list(new_shape)], dtype=np.int32),
    )
    options = {
        "newShape": [int(v) for v in list(new_shape)],
    }
    if bool(preserve_dynamic_shape):
        options["preserveDynamicShape"] = True
    ctx.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=[input_name, shape_name],
            outputs=[output_name],
            options=options,
        )
    )
    output_tensor = ctx.model_ir.tensors.get(output_name, None)
    if output_tensor is not None:
        output_tensor.shape_signature = [int(v) for v in list(new_shape)]
        output_tensor.shape = [int(v) if int(v) >= 0 else 1 for v in list(new_shape)]


def build_multi_head_attention_op(node: Any, ctx: Any) -> None:
    original_inputs = _get_original_node_inputs(node, ctx)
    query_name = _input_name(original_inputs, 0)
    key_name = _input_name(original_inputs, 1)
    value_name = _input_name(original_inputs, 2)
    if query_name == "" or key_name == "" or value_name == "":
        raise NotImplementedError(
            "MultiHeadAttention builtin lowering currently requires explicit query/key/value inputs. "
            f"op={node.name}"
        )
    unsupported_optional_inputs = [
        _input_name(original_inputs, idx)
        for idx in range(3, 10)
        if _input_name(original_inputs, idx) != ""
    ]
    if len(unsupported_optional_inputs) > 0:
        raise NotImplementedError(
            "MultiHeadAttention builtin lowering currently supports 3-input form only "
            "(query,key,value; no mask/bias/cache inputs). "
            f"op={node.name} unsupported_optional_inputs={unsupported_optional_inputs}"
        )

    output_name = node.outputs[0].name
    ctx.ensure_tensor(query_name)
    ctx.ensure_tensor(key_name)
    ctx.ensure_tensor(value_name)
    ctx.ensure_tensor(output_name)

    query_dtype = str(ctx.get_tensor_dtype(query_name)).upper()
    key_dtype = str(ctx.get_tensor_dtype(key_name)).upper()
    value_dtype = str(ctx.get_tensor_dtype(value_name)).upper()
    if len({query_dtype, key_dtype, value_dtype}) != 1:
        raise NotImplementedError(
            "MultiHeadAttention builtin lowering requires query/key/value dtypes to match. "
            f"op={node.name} query_dtype={query_dtype} key_dtype={key_dtype} value_dtype={value_dtype}"
        )
    if query_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NotImplementedError(
            "MultiHeadAttention builtin lowering supports FLOAT16/FLOAT32 only. "
            f"op={node.name} dtype={query_dtype}"
        )

    query_shape_sig = _tensor_shape_with_signature(ctx, query_name)
    key_shape_sig = _tensor_shape_with_signature(ctx, key_name)
    value_shape_sig = _tensor_shape_with_signature(ctx, value_name)
    if len(query_shape_sig) != 3 or len(key_shape_sig) != 3 or len(value_shape_sig) != 3:
        raise NotImplementedError(
            "MultiHeadAttention builtin lowering currently supports rank-3 query/key/value only. "
            f"op={node.name} query_shape={query_shape_sig} key_shape={key_shape_sig} value_shape={value_shape_sig}"
        )

    num_heads = int(node.attrs.get("num_heads", 0))
    if num_heads <= 0:
        raise NotImplementedError(
            f"MultiHeadAttention num_heads must be > 0 for builtin lowering. op={node.name} num_heads={num_heads}"
        )
    unidirectional = int(node.attrs.get("unidirectional", 0))
    if unidirectional != 0:
        raise NotImplementedError(
            f"MultiHeadAttention unidirectional=1 is not supported in builtin lowering. op={node.name}"
        )

    batch = int(query_shape_sig[0])
    if batch <= 0:
        batch = int(ctx.get_tensor_shape(query_name)[0])
    if batch <= 0:
        batch = 1

    key_batch = int(key_shape_sig[0]) if int(key_shape_sig[0]) > 0 else int(ctx.get_tensor_shape(key_name)[0])
    value_batch = int(value_shape_sig[0]) if int(value_shape_sig[0]) > 0 else int(ctx.get_tensor_shape(value_name)[0])
    if (key_batch > 0 and key_batch != batch) or (value_batch > 0 and value_batch != batch):
        raise NotImplementedError(
            "MultiHeadAttention builtin lowering requires matching batch dimensions. "
            f"op={node.name} query_batch={batch} key_batch={key_batch} value_batch={value_batch}"
        )

    query_hidden = int(query_shape_sig[2]) if int(query_shape_sig[2]) > 0 else int(ctx.get_tensor_shape(query_name)[2])
    key_hidden = int(key_shape_sig[2]) if int(key_shape_sig[2]) > 0 else int(ctx.get_tensor_shape(key_name)[2])
    value_hidden = int(value_shape_sig[2]) if int(value_shape_sig[2]) > 0 else int(ctx.get_tensor_shape(value_name)[2])
    if query_hidden <= 0 or key_hidden <= 0 or value_hidden <= 0:
        raise NotImplementedError(
            "MultiHeadAttention builtin lowering requires static positive hidden sizes. "
            f"op={node.name} query_hidden={query_hidden} key_hidden={key_hidden} value_hidden={value_hidden}"
        )
    if query_hidden % int(num_heads) != 0 or key_hidden % int(num_heads) != 0 or value_hidden % int(num_heads) != 0:
        raise NotImplementedError(
            "MultiHeadAttention hidden sizes must be divisible by num_heads for builtin lowering. "
            f"op={node.name} num_heads={num_heads} query_hidden={query_hidden} "
            f"key_hidden={key_hidden} value_hidden={value_hidden}"
        )
    query_head_dim = int(query_hidden // int(num_heads))
    key_head_dim = int(key_hidden // int(num_heads))
    value_head_dim = int(value_hidden // int(num_heads))
    if query_head_dim != key_head_dim:
        raise NotImplementedError(
            "MultiHeadAttention requires query/key head dimensions to match for builtin lowering. "
            f"op={node.name} query_head_dim={query_head_dim} key_head_dim={key_head_dim}"
        )

    query_4d_name = ctx.add_intermediate_tensor(
        f"{node.name}_query_4d",
        dtype=query_dtype,
        shape=[int(batch), -1, int(num_heads), int(query_head_dim)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=query_name,
        output_name=query_4d_name,
        new_shape=[int(batch), -1, int(num_heads), int(query_head_dim)],
        preserve_dynamic_shape=True,
    )
    query_bhqd_name = ctx.add_intermediate_tensor(
        f"{node.name}_query_bhqd",
        dtype=query_dtype,
        shape=[int(batch), int(num_heads), -1, int(query_head_dim)],
    )
    query_bhqd_name = make_transpose(
        ctx=ctx,
        input_name=query_4d_name,
        output_name=query_bhqd_name,
        perm_values=[0, 2, 1, 3],
        allow_elide_inverse_chain=False,
    )

    key_4d_name = ctx.add_intermediate_tensor(
        f"{node.name}_key_4d",
        dtype=key_dtype,
        shape=[int(batch), -1, int(num_heads), int(key_head_dim)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=key_name,
        output_name=key_4d_name,
        new_shape=[int(batch), -1, int(num_heads), int(key_head_dim)],
        preserve_dynamic_shape=True,
    )
    key_bhdk_name = ctx.add_intermediate_tensor(
        f"{node.name}_key_bhdk",
        dtype=key_dtype,
        shape=[int(batch), int(num_heads), int(key_head_dim), -1],
    )
    key_bhdk_name = make_transpose(
        ctx=ctx,
        input_name=key_4d_name,
        output_name=key_bhdk_name,
        perm_values=[0, 2, 3, 1],
        allow_elide_inverse_chain=False,
    )

    value_4d_name = ctx.add_intermediate_tensor(
        f"{node.name}_value_4d",
        dtype=value_dtype,
        shape=[int(batch), -1, int(num_heads), int(value_head_dim)],
    )
    _add_reshape_operator(
        ctx=ctx,
        input_name=value_name,
        output_name=value_4d_name,
        new_shape=[int(batch), -1, int(num_heads), int(value_head_dim)],
        preserve_dynamic_shape=True,
    )
    value_bhkd_name = ctx.add_intermediate_tensor(
        f"{node.name}_value_bhkd",
        dtype=value_dtype,
        shape=[int(batch), int(num_heads), -1, int(value_head_dim)],
    )
    value_bhkd_name = make_transpose(
        ctx=ctx,
        input_name=value_4d_name,
        output_name=value_bhkd_name,
        perm_values=[0, 2, 1, 3],
        allow_elide_inverse_chain=False,
    )

    scores_name = ctx.add_intermediate_tensor(
        f"{node.name}_scores",
        dtype=query_dtype,
        shape=[int(batch), int(num_heads), -1, -1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[query_bhqd_name, key_bhdk_name],
            outputs=[scores_name],
            options={
                "adjX": False,
                "adjY": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    scale = float(node.attrs.get("scale", 0.0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(1.0 / math.sqrt(float(query_head_dim)))
    scores_scaled_name = scores_name
    if abs(float(scale) - 1.0) > 1e-12:
        scale_name = ctx.add_const_tensor(
            f"{node.name}_scale",
            np.asarray(scale, dtype=np.float16 if query_dtype == "FLOAT16" else np.float32),
        )
        scores_scaled_name = ctx.add_intermediate_tensor(
            f"{node.name}_scores_scaled",
            dtype=query_dtype,
            shape=[int(batch), int(num_heads), -1, -1],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="MUL",
                inputs=[scores_name, scale_name],
                outputs=[scores_scaled_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )

    probs_name = ctx.add_intermediate_tensor(
        f"{node.name}_probs",
        dtype=query_dtype,
        shape=[int(batch), int(num_heads), -1, -1],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="SOFTMAX",
            inputs=[scores_scaled_name],
            outputs=[probs_name],
            options={"beta": 1.0},
        )
    )

    context_bhqd_name = ctx.add_intermediate_tensor(
        f"{node.name}_context_bhqd",
        dtype=query_dtype,
        shape=[int(batch), int(num_heads), -1, int(value_head_dim)],
    )
    ctx.add_operator(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=[probs_name, value_bhkd_name],
            outputs=[context_bhqd_name],
            options={
                "adjX": False,
                "adjY": False,
                "asymmetricQuantizeInputs": False,
            },
        )
    )

    context_bqhd_name = ctx.add_intermediate_tensor(
        f"{node.name}_context_bqhd",
        dtype=query_dtype,
        shape=[int(batch), -1, int(num_heads), int(value_head_dim)],
    )
    context_bqhd_name = make_transpose(
        ctx=ctx,
        input_name=context_bhqd_name,
        output_name=context_bqhd_name,
        perm_values=[0, 2, 1, 3],
        allow_elide_inverse_chain=False,
    )

    output_compute_name = output_name
    output_dtype = str(ctx.get_tensor_dtype(output_name)).upper()
    if output_dtype != query_dtype:
        output_compute_name = ctx.add_intermediate_tensor(
            f"{output_name}_mha_compute",
            dtype=query_dtype,
            shape=[int(batch), -1, int(value_hidden)],
        )
    _add_reshape_operator(
        ctx=ctx,
        input_name=context_bqhd_name,
        output_name=output_compute_name,
        new_shape=[int(batch), -1, int(value_hidden)],
        preserve_dynamic_shape=True,
    )
    if output_compute_name != output_name:
        ctx.add_operator(
            OperatorIR(
                op_type="CAST",
                inputs=[output_compute_name],
                outputs=[output_name],
                options={
                    "inDataType": query_dtype,
                    "outDataType": output_dtype,
                },
            )
        )


def build_lstm_op(node: Any, ctx: Any) -> None:
    original_inputs = _get_original_node_inputs(node, ctx)
    x_name = _input_name(original_inputs, 0)
    w_name = _input_name(original_inputs, 1)
    r_name = _input_name(original_inputs, 2)
    b_name = _input_name(original_inputs, 3)
    initial_h_name = _input_name(original_inputs, 5)
    initial_c_name = _input_name(original_inputs, 6)
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NotImplementedError(
            f"LSTM direction must be forward/reverse/bidirectional for flatbuffer_direct. op={node.name} direction={direction}"
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1

    y_name = node.outputs[0].name
    y_h_name = node.outputs[1].name if len(node.outputs) > 1 else ""
    y_c_name = node.outputs[2].name if len(node.outputs) > 2 else ""
    y_dtype = str(ctx.get_tensor_dtype(y_name))
    ctx.ensure_tensor(x_name)
    ctx.ensure_tensor(y_name)
    if y_h_name != "":
        ctx.ensure_tensor(y_h_name)
    if y_c_name != "":
        ctx.ensure_tensor(y_c_name)

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

    has_initial_state_inputs = (
        initial_h_name != ""
        and initial_c_name != ""
    )
    if (initial_h_name == "") ^ (initial_c_name == ""):
        raise NotImplementedError(
            f"LSTM initial_h and initial_c must be both present or both absent. op={node.name}"
        )
    initial_h_shape = [expected_num_directions, 1, hidden_size]
    initial_c_shape = [expected_num_directions, 1, hidden_size]
    if has_initial_state_inputs:
        ctx.ensure_tensor(initial_h_name)
        ctx.ensure_tensor(initial_c_name)
        initial_h_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        initial_c_shape = [int(v) for v in ctx.get_tensor_shape(initial_c_name)]
        if len(initial_h_shape) == 3 and len(initial_c_shape) == 3:
            initial_h_sig = [int(v) for v in list(initial_h_shape)]
            initial_c_sig = [int(v) for v in list(initial_c_shape)]
            initial_h_tensor = ctx.model_ir.tensors.get(initial_h_name, None)
            initial_c_tensor = ctx.model_ir.tensors.get(initial_c_name, None)
            if (
                initial_h_tensor is not None
                and initial_h_tensor.shape_signature is not None
                and len(initial_h_tensor.shape_signature) == 3
            ):
                initial_h_sig = [int(v) for v in list(initial_h_tensor.shape_signature)]
            if (
                initial_c_tensor is not None
                and initial_c_tensor.shape_signature is not None
                and len(initial_c_tensor.shape_signature) == 3
            ):
                initial_c_sig = [int(v) for v in list(initial_c_tensor.shape_signature)]
            if (
                int(initial_h_sig[0]) > 0 and int(initial_h_sig[0]) != expected_num_directions
            ) or (
                int(initial_c_sig[0]) > 0 and int(initial_c_sig[0]) != expected_num_directions
            ):
                raise NotImplementedError(
                    "LSTM initial_h/initial_c first dim must match num_directions. "
                    f"op={node.name} direction={direction} expected_num_directions={expected_num_directions} "
                    f"initial_h_shape={initial_h_shape} initial_c_shape={initial_c_shape}"
                )
            if (
                int(initial_h_sig[2]) > 0 and int(initial_h_sig[2]) != hidden_size
            ) or (
                int(initial_c_sig[2]) > 0 and int(initial_c_sig[2]) != hidden_size
            ):
                raise NotImplementedError(
                    "LSTM initial_h/initial_c hidden dim must match hidden_size. "
                    f"op={node.name} hidden_size={hidden_size} "
                    f"initial_h_shape={initial_h_shape} initial_c_shape={initial_c_shape}"
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

    def _prepare_lstm_state_input(
        *,
        state_input_name: str,
        dir_index: int,
        state_tag: str,
    ) -> str:
        if state_input_name == "":
            reference = np.zeros((int(batch_dim), int(hidden_size)), dtype=np.float32)
            return _add_zero_variable_state(f"{state_tag}_state", reference)

        slice_input_name = state_input_name
        state_shape = [int(v) for v in ctx.get_tensor_shape(state_input_name)]
        if len(state_shape) != 3:
            reshaped_state_name = ctx.add_intermediate_tensor(
                f"{node.name}_{state_tag}_dir{dir_index}_reshape3d",
                dtype="FLOAT32",
                shape=[int(expected_num_directions), int(batch_dim), int(hidden_size)],
            )
            reshape_shape_name = _add_const(
                f"{state_tag}_dir{dir_index}_reshape3d_shape",
                np.asarray(
                    [int(expected_num_directions), int(batch_dim), int(hidden_size)],
                    dtype=np.int32,
                ),
                dtype=np.int32,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[state_input_name, reshape_shape_name],
                    outputs=[reshaped_state_name],
                    options={
                        "newShape": [
                            int(expected_num_directions),
                            int(batch_dim),
                            int(hidden_size),
                        ]
                    },
                )
            )
            slice_input_name = reshaped_state_name
            state_shape = [int(expected_num_directions), int(batch_dim), int(hidden_size)]
        if not (expected_num_directions == 1 and int(state_shape[0]) == 1):
            slice_input_name = ctx.add_intermediate_tensor(
                f"{node.name}_{state_tag}_dir{dir_index}_slice",
                dtype="FLOAT32",
                shape=[1, int(batch_dim), int(hidden_size)],
            )
            begin_name = _add_const(
                f"{state_tag}_dir{dir_index}_begin",
                np.asarray([int(dir_index), 0, 0], dtype=np.int32),
                dtype=np.int32,
            )
            size_name = _add_const(
                f"{state_tag}_dir{dir_index}_size",
                np.asarray([1, int(batch_dim), int(hidden_size)], dtype=np.int32),
                dtype=np.int32,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[state_input_name, begin_name, size_name],
                    outputs=[slice_input_name],
                )
            )
        state_2d_name = ctx.add_intermediate_tensor(
            f"{node.name}_{state_tag}_dir{dir_index}_2d",
            dtype="FLOAT32",
            shape=[int(batch_dim), int(hidden_size)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[slice_input_name],
                outputs=[state_2d_name],
                options={"squeezeDims": [0]},
            )
        )
        state_tensor = ctx.model_ir.tensors[state_2d_name]
        state_tensor.is_variable = True
        return state_2d_name

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
    y_h_shape = [int(expected_num_directions), int(batch_dim), int(hidden_size)]
    y_c_shape = [int(expected_num_directions), int(batch_dim), int(hidden_size)]
    if y_h_name != "":
        y_h_tensor = ctx.model_ir.tensors.get(y_h_name, None)
        if y_h_tensor is not None:
            y_h_tensor.shape = [int(v) for v in y_h_shape]
            y_h_tensor.shape_signature = [int(v) for v in y_h_shape]
    if y_c_name != "":
        y_c_tensor = ctx.model_ir.tensors.get(y_c_name, None)
        if y_c_tensor is not None:
            y_c_tensor.shape = [int(v) for v in y_c_shape]
            y_c_tensor.shape_signature = [int(v) for v in y_c_shape]
    use_reshape_expand = (
        seq_signature_dim is not None
        and batch_signature_dim is not None
        and int(seq_signature_dim) > 0
        and int(batch_signature_dim) > 0
    )
    if expected_num_directions == 1:
        lstm_input_name = x_name
        if direction == "reverse":
            reverse_axis_input_name = _add_const(
                "reverse_time_axis_input",
                np.asarray([0], dtype=np.int32),
                dtype=np.int32,
            )
            lstm_input_name = ctx.add_intermediate_tensor(
                f"{y_name}_lstm_reverse_input",
                dtype=str(ctx.get_tensor_dtype(x_name)),
                shape=[int(v) for v in list(input_shape)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="REVERSE_V2",
                    inputs=[x_name, reverse_axis_input_name],
                    outputs=[lstm_input_name],
                )
            )

        fw_h0_name = _prepare_lstm_state_input(
            state_input_name=initial_h_name if has_initial_state_inputs else "",
            dir_index=0,
            state_tag="h0",
        )
        fw_c0_name = _prepare_lstm_state_input(
            state_input_name=initial_c_name if has_initial_state_inputs else "",
            dir_index=0,
            state_tag="c0",
        )
        uni_output_name = ctx.add_intermediate_tensor(
            f"{y_name}_lstm_uni",
            dtype=y_dtype,
            shape=[int(seq_dim), int(batch_dim), int(hidden_size)],
        )
        uni_inputs = [
            lstm_input_name,
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
        uni_output_final_name = uni_output_name
        if direction == "reverse":
            reverse_axis_output_name = _add_const(
                "reverse_time_axis_output",
                np.asarray([0], dtype=np.int32),
                dtype=np.int32,
            )
            uni_output_final_name = ctx.add_intermediate_tensor(
                f"{y_name}_lstm_reverse_output",
                dtype=y_dtype,
                shape=[int(seq_dim), int(batch_dim), int(hidden_size)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="REVERSE_V2",
                    inputs=[uni_output_name, reverse_axis_output_name],
                    outputs=[uni_output_final_name],
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
                    inputs=[uni_output_final_name, expand_shape_name],
                    outputs=[y_name],
                    options={"newShape": [int(v) for v in list(expand_shape)]},
                )
            )
        else:
            expand_axis_name = _add_const("expand_axis", np.asarray(1, dtype=np.int32), dtype=np.int32)
            ctx.add_operator(
                OperatorIR(
                    op_type="EXPAND_DIMS",
                    inputs=[uni_output_final_name, expand_axis_name],
                    outputs=[y_name],
                )
            )
        if y_h_name != "":
            y_h_shape_name = _add_const(
                "y_h_shape",
                np.asarray([1, int(batch_dim), int(hidden_size)], dtype=np.int32),
                dtype=np.int32,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[fw_h0_name, y_h_shape_name],
                    outputs=[y_h_name],
                    options={"newShape": [1, int(batch_dim), int(hidden_size)]},
                )
            )
        if y_c_name != "":
            y_c_shape_name = _add_const(
                "y_c_shape",
                np.asarray([1, int(batch_dim), int(hidden_size)], dtype=np.int32),
                dtype=np.int32,
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[fw_c0_name, y_c_shape_name],
                    outputs=[y_c_name],
                    options={"newShape": [1, int(batch_dim), int(hidden_size)]},
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

    fw_h0_name = _prepare_lstm_state_input(
        state_input_name=initial_h_name if has_initial_state_inputs else "",
        dir_index=0,
        state_tag="h0_fw",
    )
    fw_c0_name = _prepare_lstm_state_input(
        state_input_name=initial_c_name if has_initial_state_inputs else "",
        dir_index=0,
        state_tag="c0_fw",
    )
    bw_h0_name = _prepare_lstm_state_input(
        state_input_name=initial_h_name if has_initial_state_inputs else "",
        dir_index=1,
        state_tag="h0_bw",
    )
    bw_c0_name = _prepare_lstm_state_input(
        state_input_name=initial_c_name if has_initial_state_inputs else "",
        dir_index=1,
        state_tag="c0_bw",
    )

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
    if y_h_name != "":
        fw_h_3d_name = ctx.add_intermediate_tensor(
            f"{y_h_name}_fw_3d",
            dtype="FLOAT32",
            shape=[1, int(batch_dim), int(hidden_size)],
        )
        bw_h_3d_name = ctx.add_intermediate_tensor(
            f"{y_h_name}_bw_3d",
            dtype="FLOAT32",
            shape=[1, int(batch_dim), int(hidden_size)],
        )
        fw_h_shape_name = _add_const(
            "y_h_fw_shape",
            np.asarray([1, int(batch_dim), int(hidden_size)], dtype=np.int32),
            dtype=np.int32,
        )
        bw_h_shape_name = _add_const(
            "y_h_bw_shape",
            np.asarray([1, int(batch_dim), int(hidden_size)], dtype=np.int32),
            dtype=np.int32,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[fw_h0_name, fw_h_shape_name],
                outputs=[fw_h_3d_name],
                options={"newShape": [1, int(batch_dim), int(hidden_size)]},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[bw_h0_name, bw_h_shape_name],
                outputs=[bw_h_3d_name],
                options={"newShape": [1, int(batch_dim), int(hidden_size)]},
            )
        )
        y_h_merged_name = ctx.add_intermediate_tensor(
            f"{y_h_name}_merged",
            dtype="FLOAT32",
            shape=[2, int(batch_dim), int(hidden_size)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[fw_h_3d_name, bw_h_3d_name],
                outputs=[y_h_merged_name],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            )
        )
        y_h_shape_name = _add_const(
            "y_h_shape",
            np.asarray(y_h_shape, dtype=np.int32),
            dtype=np.int32,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[y_h_merged_name, y_h_shape_name],
                outputs=[y_h_name],
                options={"newShape": [int(v) for v in y_h_shape]},
            )
        )
    if y_c_name != "":
        fw_c_3d_name = ctx.add_intermediate_tensor(
            f"{y_c_name}_fw_3d",
            dtype="FLOAT32",
            shape=[1, int(batch_dim), int(hidden_size)],
        )
        bw_c_3d_name = ctx.add_intermediate_tensor(
            f"{y_c_name}_bw_3d",
            dtype="FLOAT32",
            shape=[1, int(batch_dim), int(hidden_size)],
        )
        fw_c_shape_name = _add_const(
            "y_c_fw_shape",
            np.asarray([1, int(batch_dim), int(hidden_size)], dtype=np.int32),
            dtype=np.int32,
        )
        bw_c_shape_name = _add_const(
            "y_c_bw_shape",
            np.asarray([1, int(batch_dim), int(hidden_size)], dtype=np.int32),
            dtype=np.int32,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[fw_c0_name, fw_c_shape_name],
                outputs=[fw_c_3d_name],
                options={"newShape": [1, int(batch_dim), int(hidden_size)]},
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[bw_c0_name, bw_c_shape_name],
                outputs=[bw_c_3d_name],
                options={"newShape": [1, int(batch_dim), int(hidden_size)]},
            )
        )
        y_c_merged_name = ctx.add_intermediate_tensor(
            f"{y_c_name}_merged",
            dtype="FLOAT32",
            shape=[2, int(batch_dim), int(hidden_size)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[fw_c_3d_name, bw_c_3d_name],
                outputs=[y_c_merged_name],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            )
        )
        y_c_shape_name = _add_const(
            "y_c_shape",
            np.asarray(y_c_shape, dtype=np.int32),
            dtype=np.int32,
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[y_c_merged_name, y_c_shape_name],
                outputs=[y_c_name],
                options={"newShape": [int(v) for v in y_c_shape]},
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
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NotImplementedError(
            f"RNN direction is not supported in flatbuffer_direct. op={node.name} direction={direction}"
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1

    y_name = node.outputs[0].name
    y_h_name = node.outputs[1].name if len(node.outputs) > 1 else ""
    y_dtype = str(ctx.get_tensor_dtype(y_name)).upper()
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
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NotImplementedError(
            "RNN num_directions mismatch for flatbuffer_direct. "
            f"op={node.name} direction={direction} expected_num_directions={expected_num_directions} "
            f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
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
    x_tensor = ctx.model_ir.tensors.get(x_name, None)
    input_signature = None
    if x_tensor is not None and x_tensor.shape_signature is not None:
        input_signature = [int(v) for v in list(x_tensor.shape_signature)]
    else:
        input_signature = [int(v) for v in input_shape]
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
        input_signature = [input_signature[0], input_signature[2], input_signature[1]]
    seq_signature_dim = input_signature[0] if len(input_signature) >= 1 else None
    batch_signature_dim = input_signature[1] if len(input_signature) >= 2 else None
    seq_len = _normalized_shape_dim(
        seq_signature_dim if seq_signature_dim is not None and int(seq_signature_dim) > 0 else (
            input_shape[0] if len(input_shape) >= 1 else 1
        ),
        fallback=1,
    )
    batch = _normalized_shape_dim(
        batch_signature_dim if batch_signature_dim is not None and int(batch_signature_dim) > 0 else (
            input_shape[1] if len(input_shape) >= 2 else 1
        ),
        fallback=1,
    )
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
        if (
            B.ndim != 2
            or int(B.shape[0]) != expected_num_directions
            or int(B.shape[1]) != 2 * hidden_size
        ):
            raise NotImplementedError(
                "RNN bias shape must be [num_directions, 2*hidden_size] in flatbuffer_direct. "
                f"op={node.name} direction={direction} expected_num_directions={expected_num_directions} "
                f"B_shape={list(B.shape)} hidden_size={hidden_size}"
            )
    else:
        B = np.zeros((expected_num_directions, 2 * hidden_size), dtype=np.float32)

    if initial_h_name != "":
        ctx.ensure_tensor(initial_h_name)
        initial_h_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        if len(initial_h_shape) != 3:
            raise NotImplementedError(
                "RNN initial_h must be rank-3 [num_directions, batch, hidden_size] in flatbuffer_direct. "
                f"op={node.name} initial_h_shape={initial_h_shape}"
            )
        if int(initial_h_shape[0]) > 0 and int(initial_h_shape[0]) != expected_num_directions:
            raise NotImplementedError(
                "RNN initial_h first dimension must match num_directions in flatbuffer_direct. "
                f"op={node.name} direction={direction} expected_num_directions={expected_num_directions} "
                f"initial_h_shape={initial_h_shape}"
            )
        if int(initial_h_shape[1]) > 0 and int(initial_h_shape[1]) != int(batch):
            raise NotImplementedError(
                "RNN initial_h batch dimension must match input batch in flatbuffer_direct. "
                f"op={node.name} initial_h_shape={initial_h_shape} batch={int(batch)}"
            )
        if int(initial_h_shape[2]) > 0 and int(initial_h_shape[2]) != hidden_size:
            raise NotImplementedError(
                "RNN initial_h hidden dimension must match hidden_size in flatbuffer_direct. "
                f"op={node.name} initial_h_shape={initial_h_shape} hidden_size={hidden_size}"
            )

    activations = node.attrs.get("activations", ["Tanh"])
    if not isinstance(activations, (list, tuple)) or len(activations) == 0:
        activations = ["Tanh"]
    fused_activations: List[str] = []
    for dir_index in range(expected_num_directions):
        activation = str(activations[min(dir_index, len(activations) - 1)]).lower()
        if activation in {"tanh"}:
            fused_activations.append("TANH")
        elif activation in {"relu"}:
            fused_activations.append("RELU")
        elif activation in {"sigmoid"}:
            fused_activations.append("LOGISTIC")
        else:
            raise NotImplementedError(
                "RNN activation is not supported in flatbuffer_direct. "
                f"op={node.name} direction={direction} direction_index={dir_index} activation={activation}"
            )
    clip = float(node.attrs.get("clip", 0.0))
    if abs(clip) > 1e-6:
        raise NotImplementedError(
            f"RNN clip is not supported in flatbuffer_direct. op={node.name} clip={clip}"
        )

    def _prepare_rnn_state_input(*, dir_index: int, dir_tag: str) -> str:
        if initial_h_name == "":
            state_name = ctx.add_intermediate_tensor(
                f"{node.name}_rnn_{dir_tag}_h0_zero",
                dtype="FLOAT32",
                shape=[int(batch), int(hidden_size)],
            )
            state_tensor = ctx.model_ir.tensors[state_name]
            state_tensor.is_variable = True
            state_tensor.data = None
            return state_name

        state_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        slice_input_name = initial_h_name
        if not (expected_num_directions == 1 and int(state_shape[0]) == 1):
            slice_input_name = ctx.add_intermediate_tensor(
                f"{node.name}_rnn_{dir_tag}_h0_slice",
                dtype="FLOAT32",
                shape=[1, int(batch), int(hidden_size)],
            )
            begin_name = ctx.add_const_tensor(
                f"{node.name}_rnn_{dir_tag}_h0_begin",
                np.asarray([int(dir_index), 0, 0], dtype=np.int32),
            )
            size_name = ctx.add_const_tensor(
                f"{node.name}_rnn_{dir_tag}_h0_size",
                np.asarray([1, int(batch), int(hidden_size)], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[initial_h_name, begin_name, size_name],
                    outputs=[slice_input_name],
                )
            )
        state_name = ctx.add_intermediate_tensor(
            f"{node.name}_rnn_{dir_tag}_h0_2d",
            dtype="FLOAT32",
            shape=[int(batch), int(hidden_size)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[slice_input_name],
                outputs=[state_name],
                options={"squeezeDims": [0]},
            )
        )
        state_tensor = ctx.model_ir.tensors[state_name]
        state_tensor.is_variable = True
        return state_name

    def _run_rnn_direction(
        *,
        dir_index: int,
        reverse: bool,
        dir_tag: str,
    ) -> Tuple[str, str]:
        Wd = np.asarray(W[dir_index], dtype=np.float32)
        Rd = np.asarray(R[dir_index], dtype=np.float32)
        Bd = np.asarray(B[dir_index], dtype=np.float32)
        bias = np.asarray(Bd[:hidden_size] + Bd[hidden_size:], dtype=np.float32)

        input_weights_name = ctx.add_const_tensor(
            f"{node.name}_rnn_{dir_tag}_input_weights",
            Wd,
        )
        recurrent_weights_name = ctx.add_const_tensor(
            f"{node.name}_rnn_{dir_tag}_recurrent_weights",
            Rd,
        )
        bias_name = ctx.add_const_tensor(
            f"{node.name}_rnn_{dir_tag}_bias",
            bias,
        )
        hidden_state_name = _prepare_rnn_state_input(dir_index=dir_index, dir_tag=dir_tag)

        rnn_input_name = x_name
        if reverse:
            reverse_axis_name = ctx.add_const_tensor(
                f"{node.name}_rnn_{dir_tag}_reverse_axis_input",
                np.asarray([0], dtype=np.int32),
            )
            reversed_input_name = ctx.add_intermediate_tensor(
                f"{node.name}_rnn_{dir_tag}_input_reversed",
                dtype="FLOAT32",
                shape=[int(seq_len), int(batch), int(input_size)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="REVERSE_V2",
                    inputs=[x_name, reverse_axis_name],
                    outputs=[reversed_input_name],
                )
            )
            rnn_input_name = reversed_input_name

        y_seq_raw_name = ctx.add_intermediate_tensor(
            f"{y_name}_rnn_{dir_tag}_seq_raw",
            dtype=y_dtype,
            shape=[int(seq_len), int(batch), int(hidden_size)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="UNIDIRECTIONAL_SEQUENCE_RNN",
                inputs=[
                    rnn_input_name,
                    input_weights_name,
                    recurrent_weights_name,
                    bias_name,
                    hidden_state_name,
                ],
                outputs=[y_seq_raw_name],
                options={
                    "timeMajor": True,
                    "fusedActivationFunction": fused_activations[dir_index],
                    "asymmetricQuantizeInputs": False,
                },
            )
        )
        y_seq_aligned_name = y_seq_raw_name
        if reverse:
            reverse_axis_output_name = ctx.add_const_tensor(
                f"{node.name}_rnn_{dir_tag}_reverse_axis_output",
                np.asarray([0], dtype=np.int32),
            )
            y_seq_aligned_name = ctx.add_intermediate_tensor(
                f"{y_name}_rnn_{dir_tag}_seq_aligned",
                dtype=y_dtype,
                shape=[int(seq_len), int(batch), int(hidden_size)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="REVERSE_V2",
                    inputs=[y_seq_raw_name, reverse_axis_output_name],
                    outputs=[y_seq_aligned_name],
                )
            )
        return y_seq_aligned_name, hidden_state_name

    if direction == "bidirectional":
        fw_seq_name, fw_h_name = _run_rnn_direction(
            dir_index=0,
            reverse=False,
            dir_tag="fw",
        )
        bw_seq_name, bw_h_name = _run_rnn_direction(
            dir_index=1,
            reverse=True,
            dir_tag="bw",
        )

        fw_axis_name = ctx.add_const_tensor(
            f"{node.name}_rnn_fw_expand_axis",
            np.asarray(1, dtype=np.int32),
        )
        bw_axis_name = ctx.add_const_tensor(
            f"{node.name}_rnn_bw_expand_axis",
            np.asarray(1, dtype=np.int32),
        )
        fw_seq_4d_name = ctx.add_intermediate_tensor(
            f"{y_name}_rnn_fw_seq_4d",
            dtype=y_dtype,
            shape=[int(seq_len), 1, int(batch), int(hidden_size)],
        )
        bw_seq_4d_name = ctx.add_intermediate_tensor(
            f"{y_name}_rnn_bw_seq_4d",
            dtype=y_dtype,
            shape=[int(seq_len), 1, int(batch), int(hidden_size)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[fw_seq_name, fw_axis_name],
                outputs=[fw_seq_4d_name],
            )
        )
        ctx.add_operator(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[bw_seq_name, bw_axis_name],
                outputs=[bw_seq_4d_name],
            )
        )
        y_merged_name = ctx.add_intermediate_tensor(
            f"{y_name}_rnn_bi_merged",
            dtype=y_dtype,
            shape=[int(seq_len), 2, int(batch), int(hidden_size)],
        )
        ctx.add_operator(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=[fw_seq_4d_name, bw_seq_4d_name],
                outputs=[y_merged_name],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            )
        )

        y_shape = [int(v) for v in ctx.get_tensor_shape(y_name)]
        if len(y_shape) == 4:
            y_shape_name = ctx.add_const_tensor(
                f"{y_name}_rnn_bi_y_shape",
                np.asarray(y_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_merged_name, y_shape_name],
                    outputs=[y_name],
                    options={"newShape": [int(v) for v in y_shape]},
                )
            )
        elif len(y_shape) == 3:
            y_shape_name = ctx.add_const_tensor(
                f"{y_name}_rnn_bi_y_shape",
                np.asarray(y_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_merged_name, y_shape_name],
                    outputs=[y_name],
                    options={"newShape": [int(v) for v in y_shape]},
                )
            )
        else:
            raise NotImplementedError(
                f"RNN Y output rank must be 3 or 4 for flatbuffer_direct. op={node.name} y_shape={y_shape}"
            )

        if y_h_name != "":
            fw_h_3d_name = ctx.add_intermediate_tensor(
                f"{y_h_name}_rnn_fw_3d",
                dtype="FLOAT32",
                shape=[1, int(batch), int(hidden_size)],
            )
            bw_h_3d_name = ctx.add_intermediate_tensor(
                f"{y_h_name}_rnn_bw_3d",
                dtype="FLOAT32",
                shape=[1, int(batch), int(hidden_size)],
            )
            fw_h_shape_name = ctx.add_const_tensor(
                f"{y_h_name}_rnn_fw_shape",
                np.asarray([1, int(batch), int(hidden_size)], dtype=np.int32),
            )
            bw_h_shape_name = ctx.add_const_tensor(
                f"{y_h_name}_rnn_bw_shape",
                np.asarray([1, int(batch), int(hidden_size)], dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[fw_h_name, fw_h_shape_name],
                    outputs=[fw_h_3d_name],
                    options={"newShape": [1, int(batch), int(hidden_size)]},
                )
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[bw_h_name, bw_h_shape_name],
                    outputs=[bw_h_3d_name],
                    options={"newShape": [1, int(batch), int(hidden_size)]},
                )
            )
            y_h_merged_name = ctx.add_intermediate_tensor(
                f"{y_h_name}_rnn_bi_merged",
                dtype="FLOAT32",
                shape=[2, int(batch), int(hidden_size)],
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=[fw_h_3d_name, bw_h_3d_name],
                    outputs=[y_h_merged_name],
                    options={"axis": 0, "fusedActivationFunction": "NONE"},
                )
            )
            y_h_shape = [int(v) for v in ctx.get_tensor_shape(y_h_name)]
            y_h_target_shape = [2, int(batch), int(hidden_size)] if len(y_h_shape) == 0 else y_h_shape
            y_h_shape_name = ctx.add_const_tensor(
                f"{y_h_name}_rnn_bi_shape",
                np.asarray(y_h_target_shape, dtype=np.int32),
            )
            ctx.add_operator(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[y_h_merged_name, y_h_shape_name],
                    outputs=[y_h_name],
                    options={"newShape": [int(v) for v in y_h_target_shape]},
                )
            )
        return

    y_seq_name, y_h_state_name = _run_rnn_direction(
        dir_index=0,
        reverse=(direction == "reverse"),
        dir_tag="rev" if direction == "reverse" else "fw",
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
        y_h_shape = [int(v) for v in ctx.get_tensor_shape(y_h_name)]
        y_h_target_shape = [1, int(batch), int(hidden_size)] if len(y_h_shape) == 0 else y_h_shape
        y_h_shape_name = ctx.add_const_tensor(
            f"{y_h_name}_rnn_shape",
            np.asarray(y_h_target_shape, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[y_h_state_name, y_h_shape_name],
                outputs=[y_h_name],
                options={"newShape": [int(v) for v in y_h_target_shape]},
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
