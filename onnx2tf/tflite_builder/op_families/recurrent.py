from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.core.op_contracts import NodeValidationError, get_original_node_inputs as _get_original_node_inputs


def _validate_lstm(node: Any, ctx: Any) -> None:
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "LSTM direction must be one of forward/reverse/bidirectional for builtin lowering. "
                f"direction={direction}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1
    layout = int(node.attrs.get("layout", 0))
    if layout != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LSTM layout must be 0 (time-major). layout={layout}",
            node_name=node.name,
            node_op=node.op,
        )
    input_forget = int(node.attrs.get("input_forget", 0))
    if input_forget != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LSTM input_forget=1 is not supported in builtin lowering. input_forget={input_forget}",
            node_name=node.name,
            node_op=node.op,
        )

    original_inputs = _get_original_node_inputs(node, ctx)

    def _input_name(index: int) -> str:
        if index < 0 or index >= len(original_inputs):
            return ""
        return str(original_inputs[index])

    sequence_lens_name = _input_name(4)
    if sequence_lens_name != "":
        raise NodeValidationError(
            reason_code="unsupported_input",
            message="LSTM sequence_lens is not supported in flatbuffer_direct builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )

    peephole_name = _input_name(7)
    if peephole_name != "":
        raise NodeValidationError(
            reason_code="unsupported_input",
            message="LSTM peephole weights input P is not supported in flatbuffer_direct builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )

    w_name = _input_name(1)
    r_name = _input_name(2)
    if w_name == "" or r_name == "":
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="LSTM requires W and R inputs.",
            node_name=node.name,
            node_op=node.op,
        )
    W = ctx.get_constant_array(w_name)
    R = ctx.get_constant_array(r_name)
    if W is None or R is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message=(
                "LSTM W and R must be constant tensors for builtin lowering. "
                f"W={w_name} W_is_constant={W is not None} "
                f"R={r_name} R_is_constant={R is not None}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    W = np.asarray(W)
    R = np.asarray(R)
    if W.ndim != 3 or R.ndim != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"LSTM W/R must be rank-3. W_shape={list(W.shape)} R_shape={list(R.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM builtin lowering expects weights with num_directions matching direction. "
                f"direction={direction} expected_num_directions={expected_num_directions} "
                f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    hidden_size_attr = int(node.attrs.get("hidden_size", 0))
    hidden_size = int(hidden_size_attr) if hidden_size_attr > 0 else int(W.shape[1] // 4)
    if int(W.shape[1]) != 4 * hidden_size or int(R.shape[1]) != 4 * hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM hidden_size mismatch in W/R. "
                f"hidden_size={hidden_size} W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(R.shape[2]) != hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM projection is not supported in builtin lowering. "
                f"recurrent_output_size={int(R.shape[2])} hidden_size={hidden_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    b_name = _input_name(3)
    if b_name != "":
        B = ctx.get_constant_array(b_name)
        if B is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="LSTM B must be constant when provided.",
                node_name=node.name,
                node_op=node.op,
            )
        B = np.asarray(B)
        if (
            B.ndim != 2
            or int(B.shape[0]) != expected_num_directions
            or int(B.shape[1]) != 8 * hidden_size
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "LSTM B must have shape [num_directions, 8*hidden_size]. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"B_shape={list(B.shape)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    initial_h_name = _input_name(5)
    initial_c_name = _input_name(6)
    if (initial_h_name == "") ^ (initial_c_name == ""):
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="LSTM initial_h and initial_c must be both present or both absent for builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )
    if initial_h_name != "":
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
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "LSTM initial_h and initial_c first dimension must match num_directions. "
                        f"direction={direction} expected_num_directions={expected_num_directions} "
                        f"initial_h_shape={initial_h_shape} initial_c_shape={initial_c_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if (
                int(initial_h_sig[2]) > 0 and int(initial_h_sig[2]) != hidden_size
            ) or (
                int(initial_c_sig[2]) > 0 and int(initial_c_sig[2]) != hidden_size
            ):
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "LSTM initial_h and initial_c hidden dimension must match hidden_size. "
                        f"hidden_size={hidden_size} initial_h_shape={initial_h_shape} initial_c_shape={initial_c_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

def _validate_rnn(node: Any, ctx: Any) -> None:
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "RNN direction must be one of forward/reverse/bidirectional for builtin lowering. "
                f"direction={direction}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1
    layout = int(node.attrs.get("layout", 0))
    if layout != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"RNN layout must be 0 (time-major). layout={layout}",
            node_name=node.name,
            node_op=node.op,
        )

    original_inputs = _get_original_node_inputs(node, ctx)

    def _input_name(index: int) -> str:
        if index < 0 or index >= len(original_inputs):
            return ""
        return str(original_inputs[index])

    if _input_name(4) != "":
        raise NodeValidationError(
            reason_code="unsupported_input",
            message="RNN sequence_lens is not supported in flatbuffer_direct builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )

    w_name = _input_name(1)
    r_name = _input_name(2)
    if w_name == "" or r_name == "":
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="RNN requires W and R inputs.",
            node_name=node.name,
            node_op=node.op,
        )
    W = ctx.get_constant_array(w_name)
    R = ctx.get_constant_array(r_name)
    if W is None or R is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="RNN W and R must be constant tensors for builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )
    W = np.asarray(W)
    R = np.asarray(R)
    if W.ndim != 3 or R.ndim != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"RNN W/R must be rank-3. W_shape={list(W.shape)} R_shape={list(R.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RNN W/R num_directions mismatch for builtin lowering. "
                f"direction={direction} expected_num_directions={expected_num_directions} "
                f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    hidden_size = int(node.attrs.get("hidden_size", int(W.shape[1])))
    if int(W.shape[1]) != hidden_size or int(R.shape[1]) != hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RNN hidden_size mismatch in W/R for builtin lowering. "
                f"hidden_size={hidden_size} W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(R.shape[2]) != hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RNN recurrent weight output size must match hidden_size for builtin lowering. "
                f"R_shape={list(R.shape)} hidden_size={hidden_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    b_name = _input_name(3)
    if b_name != "":
        B = ctx.get_constant_array(b_name)
        if B is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="RNN B must be constant when provided.",
                node_name=node.name,
                node_op=node.op,
            )
        B = np.asarray(B)
        if (
            B.ndim != 2
            or int(B.shape[0]) != expected_num_directions
            or int(B.shape[1]) != 2 * hidden_size
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "RNN B must have shape [num_directions, 2*hidden_size]. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"B_shape={list(B.shape)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    initial_h_name = _input_name(5)
    if initial_h_name != "":
        initial_h_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        if len(initial_h_shape) != 3:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "RNN initial_h must be rank-3 [num_directions, batch, hidden] in builtin lowering. "
                    f"initial_h_shape={initial_h_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if int(initial_h_shape[0]) > 0 and int(initial_h_shape[0]) != expected_num_directions:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "RNN initial_h first dimension must match num_directions in builtin lowering. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"initial_h_shape={initial_h_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if int(initial_h_shape[2]) > 0 and int(initial_h_shape[2]) != hidden_size:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "RNN initial_h hidden dimension must match hidden_size in builtin lowering. "
                    f"initial_h_shape={initial_h_shape} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    activations = node.attrs.get("activations", ["Tanh"])
    if not isinstance(activations, (list, tuple)) or len(activations) == 0:
        activations = ["Tanh"]
    for dir_index in range(expected_num_directions):
        activation = str(activations[min(dir_index, len(activations) - 1)]).lower()
        if activation not in {"tanh", "relu", "sigmoid"}:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "RNN activation must be one of tanh/relu/sigmoid for builtin lowering. "
                    f"direction={direction} direction_index={dir_index} activation={activation}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    clip = float(node.attrs.get("clip", 0.0))
    if abs(clip) > 1e-6:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"RNN clip is not supported in builtin lowering. clip={clip}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_gru(node: Any, ctx: Any) -> None:
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GRU direction must be one of forward/reverse/bidirectional for builtin lowering. "
                f"direction={direction}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1
    layout = int(node.attrs.get("layout", 0))
    if layout != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GRU layout must be 0 (time-major). layout={layout}",
            node_name=node.name,
            node_op=node.op,
        )
    linear_before_reset = int(node.attrs.get("linear_before_reset", 0))
    if linear_before_reset not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GRU linear_before_reset must be 0 or 1 in flatbuffer_direct builtin lowering. "
                f"linear_before_reset={linear_before_reset}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    clip = float(node.attrs.get("clip", 0.0))
    if abs(clip) > 1e-6:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GRU clip is not supported in builtin lowering. clip={clip}",
            node_name=node.name,
            node_op=node.op,
        )
    activations = node.attrs.get("activations", ["Sigmoid", "Tanh"])
    if not isinstance(activations, (list, tuple)) or len(activations) < 2:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GRU activations must contain at least 2 entries. activations={activations}",
            node_name=node.name,
            node_op=node.op,
        )
    act0 = str(activations[0]).lower()
    act1 = str(activations[1]).lower()
    if act0 != "sigmoid" or act1 != "tanh":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GRU builtin lowering supports activations=[Sigmoid,Tanh] only. "
                f"activations={activations}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    original_inputs = _get_original_node_inputs(node, ctx)

    def _input_name(index: int) -> str:
        if index < 0 or index >= len(original_inputs):
            return ""
        return str(original_inputs[index])

    if _input_name(4) != "":
        raise NodeValidationError(
            reason_code="unsupported_input",
            message="GRU sequence_lens is not supported in flatbuffer_direct builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )

    w_name = _input_name(1)
    r_name = _input_name(2)
    if w_name == "" or r_name == "":
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="GRU requires W and R inputs.",
            node_name=node.name,
            node_op=node.op,
        )
    W = ctx.get_constant_array(w_name)
    R = ctx.get_constant_array(r_name)
    if W is None or R is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="GRU W and R must be constant tensors for builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )
    W = np.asarray(W)
    R = np.asarray(R)
    if W.ndim != 3 or R.ndim != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"GRU W/R must be rank-3. W_shape={list(W.shape)} R_shape={list(R.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU W/R num_directions mismatch for builtin lowering. "
                f"direction={direction} expected_num_directions={expected_num_directions} "
                f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    hidden_size = int(node.attrs.get("hidden_size", int(W.shape[1] // 3)))
    if int(W.shape[1]) != 3 * hidden_size or int(R.shape[1]) != 3 * hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU hidden_size mismatch in W/R for builtin lowering. "
                f"hidden_size={hidden_size} W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(R.shape[2]) != hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU recurrent weight output size must match hidden_size for builtin lowering. "
                f"R_shape={list(R.shape)} hidden_size={hidden_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    x_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    if len(x_shape) != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GRU input rank must be 3 [seq,batch,input]. input_shape={x_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(x_shape[0]) <= 0 or int(x_shape[1]) <= 0 or int(x_shape[2]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU builtin lowering requires static positive seq,batch,input dimensions. "
                f"input_shape={x_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    input_size = int(W.shape[2])
    batch_dim = int(x_shape[1])
    if int(x_shape[2]) == input_size:
        batch_dim = int(x_shape[1])
    elif int(x_shape[1]) == input_size:
        batch_dim = int(x_shape[2])
    else:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU input size mismatch with W for builtin lowering. "
                "Expected [seq,batch,input] or [seq,input,batch]. "
                f"input_shape={x_shape} W_shape={list(W.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    b_name = _input_name(3)
    if b_name != "":
        B = ctx.get_constant_array(b_name)
        if B is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="GRU B must be constant when provided.",
                node_name=node.name,
                node_op=node.op,
            )
        B = np.asarray(B)
        if (
            B.ndim != 2
            or int(B.shape[0]) != expected_num_directions
            or int(B.shape[1]) != 6 * hidden_size
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GRU B must have shape [num_directions, 6*hidden_size]. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"B_shape={list(B.shape)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    initial_h_name = _input_name(5)
    if initial_h_name != "":
        initial_h_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        accepted_shapes = {
            (expected_num_directions, int(batch_dim), hidden_size),
            (expected_num_directions, hidden_size, int(batch_dim)),
        }
        if tuple(initial_h_shape) not in accepted_shapes:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GRU initial_h shape must be [num_directions, batch, hidden_size] "
                    "or [num_directions, hidden_size, batch] for builtin lowering. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"initial_h_shape={initial_h_shape} batch={int(batch_dim)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
