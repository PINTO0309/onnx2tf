import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
)
from onnx2tf.utils.enums import (
    ONNX_DTYPES_TO_TF_DTYPES,
)
from typing import Any, Optional


def _is_empty_input(input_var: Any) -> bool:
    if input_var is None:
        return True
    if isinstance(input_var, str) and input_var == "":
        return True
    if hasattr(input_var, 'name') and input_var.name == "":
        return True
    return False


def _to_tf_tensor(input_var: Any, tf_layers_dict: dict) -> Any:
    return tf_layers_dict[input_var.name]['tf_node'] \
        if isinstance(input_var, gs.Variable) else input_var


def _pad_or_slice_last_dim(
    mask_tensor: Any,
    target_len: Any,
    pad_value: Any,
) -> Any:
    mask_shape = tf.shape(mask_tensor)
    last_dim = mask_shape[-1]

    def _pad():
        pad_len = target_len - last_dim
        rank = tf.rank(mask_tensor)
        paddings = tf.concat(
            values=[
                tf.zeros(
                    shape=tf.stack([rank - 1, 2]),
                    dtype=tf.int32,
                ),
                tf.reshape(tf.stack([0, pad_len]), (1, 2)),
            ],
            axis=0,
        )
        return tf.pad(
            tensor=mask_tensor,
            paddings=paddings,
            constant_values=pad_value,
        )

    def _slice():
        return mask_tensor[..., :target_len]

    def _identity():
        return mask_tensor

    return tf.cond(
        pred=last_dim < target_len,
        true_fn=_pad,
        false_fn=lambda: tf.cond(
            pred=last_dim > target_len,
            true_fn=_slice,
            false_fn=_identity,
        ),
    )


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Attention

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_q = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_k = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_v = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_q \
        and before_op_output_shape_trans_k \
        and before_op_output_shape_trans_v

    graph_node_input_q = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_k = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_v = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
    )

    attn_mask_input = None
    past_key_input = None
    past_value_input = None
    nonpad_kv_seqlen_input = None

    if len(graph_node.inputs) >= 4 and not _is_empty_input(graph_node.inputs[3]):
        attn_mask_input = graph_node.inputs[3]
    if len(graph_node.inputs) >= 5 and not _is_empty_input(graph_node.inputs[4]):
        past_key_input = graph_node.inputs[4]
    if len(graph_node.inputs) >= 6 and not _is_empty_input(graph_node.inputs[5]):
        past_value_input = graph_node.inputs[5]
    if len(graph_node.inputs) >= 7 and not _is_empty_input(graph_node.inputs[6]):
        nonpad_kv_seqlen_input = graph_node.inputs[6]

    if (past_key_input is None) != (past_value_input is None):
        past_key_input = None
        past_value_input = None

    Q = _to_tf_tensor(graph_node_input_q, tf_layers_dict)
    K = _to_tf_tensor(graph_node_input_k, tf_layers_dict)
    V = _to_tf_tensor(graph_node_input_v, tf_layers_dict)

    # Pre-process transpose
    Q = pre_process_transpose(
        value_before_transpose=Q,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    K = pre_process_transpose(
        value_before_transpose=K,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    V = pre_process_transpose(
        value_before_transpose=V,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )

    attn_mask = None
    if attn_mask_input is not None:
        graph_node_input_attn = get_constant_or_variable(
            attn_mask_input,
            before_op_output_shape_trans,
        )
        attn_mask = _to_tf_tensor(graph_node_input_attn, tf_layers_dict)
        attn_mask = pre_process_transpose(
            value_before_transpose=attn_mask,
            param_target='inputs',
            param_name=attn_mask_input.name,
            **kwargs,
        )

    past_key = None
    past_value = None
    if past_key_input is not None and past_value_input is not None:
        graph_node_input_past_key = get_constant_or_variable(
            past_key_input,
            before_op_output_shape_trans,
        )
        graph_node_input_past_value = get_constant_or_variable(
            past_value_input,
            before_op_output_shape_trans,
        )
        past_key = _to_tf_tensor(graph_node_input_past_key, tf_layers_dict)
        past_value = _to_tf_tensor(graph_node_input_past_value, tf_layers_dict)
        past_key = pre_process_transpose(
            value_before_transpose=past_key,
            param_target='inputs',
            param_name=past_key_input.name,
            **kwargs,
        )
        past_value = pre_process_transpose(
            value_before_transpose=past_value,
            param_target='inputs',
            param_name=past_value_input.name,
            **kwargs,
        )

    nonpad_kv_seqlen = None
    if nonpad_kv_seqlen_input is not None:
        graph_node_input_nonpad = get_constant_or_variable(
            nonpad_kv_seqlen_input,
            before_op_output_shape_trans,
        )
        nonpad_kv_seqlen = _to_tf_tensor(graph_node_input_nonpad, tf_layers_dict)
        nonpad_kv_seqlen = pre_process_transpose(
            value_before_transpose=nonpad_kv_seqlen,
            param_target='inputs',
            param_name=nonpad_kv_seqlen_input.name,
            **kwargs,
        )

    graph_node_output_y: gs.Variable = graph_node.outputs[0]
    y_shape = graph_node_output_y.shape
    y_dtype = graph_node_output_y.dtype

    graph_node_output_present_key: Optional[gs.Variable] = None
    graph_node_output_present_value: Optional[gs.Variable] = None
    graph_node_output_qk_matmul_output: Optional[gs.Variable] = None
    if len(graph_node.outputs) >= 2 and not _is_empty_input(graph_node.outputs[1]):
        graph_node_output_present_key = graph_node.outputs[1]
    if len(graph_node.outputs) >= 3 and not _is_empty_input(graph_node.outputs[2]):
        graph_node_output_present_value = graph_node.outputs[2]
    if len(graph_node.outputs) >= 4 and not _is_empty_input(graph_node.outputs[3]):
        graph_node_output_qk_matmul_output = graph_node.outputs[3]

    if (graph_node_output_present_key is None) != (graph_node_output_present_value is None):
        graph_node_output_present_key = None
        graph_node_output_present_value = None

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output_y.name] = {
        'optype': graph_node.op,
        'shape': y_shape,
        'dtype': y_dtype,
    }
    if graph_node_output_present_key is not None:
        tf_layers_dict[graph_node_output_present_key.name] = {
            'optype': graph_node.op,
            'shape': graph_node_output_present_key.shape,
            'dtype': graph_node_output_present_key.dtype,
        }
    if graph_node_output_present_value is not None:
        tf_layers_dict[graph_node_output_present_value.name] = {
            'optype': graph_node.op,
            'shape': graph_node_output_present_value.shape,
            'dtype': graph_node_output_present_value.dtype,
        }
    if graph_node_output_qk_matmul_output is not None:
        tf_layers_dict[graph_node_output_qk_matmul_output.name] = {
            'optype': graph_node.op,
            'shape': graph_node_output_qk_matmul_output.shape,
            'dtype': graph_node_output_qk_matmul_output.dtype,
        }

    # Attributes
    opset = kwargs['opset']
    is_causal = bool(graph_node.attrs.get('is_causal', 0))
    qk_matmul_output_mode = int(graph_node.attrs.get('qk_matmul_output_mode', 0))
    softcap = graph_node.attrs.get('softcap', 0.0)
    scale = graph_node.attrs.get('scale', None)
    q_num_heads = graph_node.attrs.get('q_num_heads', None)
    kv_num_heads = graph_node.attrs.get('kv_num_heads', None)
    softmax_precision = graph_node.attrs.get('softmax_precision', None)

    is_causal = replace_parameter(
        value_before_replacement=is_causal,
        param_target='attributes',
        param_name='is_causal',
        **kwargs,
    )
    qk_matmul_output_mode = replace_parameter(
        value_before_replacement=qk_matmul_output_mode,
        param_target='attributes',
        param_name='qk_matmul_output_mode',
        **kwargs,
    )
    softcap = replace_parameter(
        value_before_replacement=softcap,
        param_target='attributes',
        param_name='softcap',
        **kwargs,
    )
    scale = replace_parameter(
        value_before_replacement=scale,
        param_target='attributes',
        param_name='scale',
        **kwargs,
    )
    q_num_heads = replace_parameter(
        value_before_replacement=q_num_heads,
        param_target='attributes',
        param_name='q_num_heads',
        **kwargs,
    )
    kv_num_heads = replace_parameter(
        value_before_replacement=kv_num_heads,
        param_target='attributes',
        param_name='kv_num_heads',
        **kwargs,
    )
    softmax_precision = replace_parameter(
        value_before_replacement=softmax_precision,
        param_target='attributes',
        param_name='softmax_precision',
        **kwargs,
    )

    # Reshape 3D inputs to 4D if needed
    q_rank = len(Q.shape) if Q.shape is not None else None
    k_rank = len(K.shape) if K.shape is not None else None
    v_rank = len(V.shape) if V.shape is not None else None

    input_q_is_3d = q_rank == 3
    input_k_is_3d = k_rank == 3
    input_v_is_3d = v_rank == 3
    input_is_3d = input_q_is_3d and input_k_is_3d and input_v_is_3d

    if input_is_3d:
        if q_num_heads is None:
            q_num_heads = 1
        if kv_num_heads is None:
            kv_num_heads = 1

        q_shape = tf.shape(Q)
        k_shape = tf.shape(K)
        v_shape = tf.shape(V)

        q_num_heads_tensor = \
            tf.constant(q_num_heads, dtype=tf.int32) \
                if isinstance(q_num_heads, int) else tf.cast(q_num_heads, tf.int32)
        kv_num_heads_tensor = \
            tf.constant(kv_num_heads, dtype=tf.int32) \
                if isinstance(kv_num_heads, int) else tf.cast(kv_num_heads, tf.int32)

        q_head_size = tf.math.floordiv(q_shape[2], q_num_heads_tensor)
        k_head_size = tf.math.floordiv(k_shape[2], kv_num_heads_tensor)
        v_head_size = tf.math.floordiv(v_shape[2], kv_num_heads_tensor)

        Q = tf.reshape(
            tensor=Q,
            shape=tf.stack([q_shape[0], q_shape[1], q_num_heads_tensor, q_head_size]),
        )
        K = tf.reshape(
            tensor=K,
            shape=tf.stack([k_shape[0], k_shape[1], kv_num_heads_tensor, k_head_size]),
        )
        V = tf.reshape(
            tensor=V,
            shape=tf.stack([v_shape[0], v_shape[1], kv_num_heads_tensor, v_head_size]),
        )
        Q = transpose_with_flexing_deterrence(
            input_tensor=Q,
            perm=[0, 2, 1, 3],
            **kwargs,
        )
        K = transpose_with_flexing_deterrence(
            input_tensor=K,
            perm=[0, 2, 1, 3],
            **kwargs,
        )
        V = transpose_with_flexing_deterrence(
            input_tensor=V,
            perm=[0, 2, 1, 3],
            **kwargs,
        )

    # Ensure dtype alignment for matmul
    q_dtype = Q.dtype
    if K.dtype != q_dtype:
        K = tf.cast(K, q_dtype)
    if V.dtype != q_dtype:
        V = tf.cast(V, q_dtype)

    if past_key is not None and past_value is not None:
        if past_key.dtype != q_dtype:
            past_key = tf.cast(past_key, q_dtype)
        if past_value.dtype != q_dtype:
            past_value = tf.cast(past_value, q_dtype)
        K = tf.concat([past_key, K], axis=2)
        V = tf.concat([past_value, V], axis=2)

    present_key = K
    present_value = V

    # Heads for GQA/MQA
    q_heads = Q.shape[1] if Q.shape[1] is not None else tf.shape(Q)[1]
    kv_heads = present_key.shape[1] if present_key.shape[1] is not None else tf.shape(present_key)[1]

    attn_key = present_key
    attn_value = present_value
    if isinstance(q_heads, int) and isinstance(kv_heads, int):
        if kv_heads != q_heads:
            repeat = q_heads // kv_heads
            attn_key = tf.repeat(attn_key, repeats=repeat, axis=1)
            attn_value = tf.repeat(attn_value, repeats=repeat, axis=1)
    else:
        repeat = tf.math.floordiv(tf.cast(q_heads, tf.int32), tf.cast(kv_heads, tf.int32))
        attn_key = tf.repeat(attn_key, repeats=repeat, axis=1)
        attn_value = tf.repeat(attn_value, repeats=repeat, axis=1)

    # Scale Q and K
    head_size = tf.shape(Q)[-1]
    if scale is None:
        scale_value = tf.math.rsqrt(tf.cast(head_size, q_dtype))
    else:
        scale_value = tf.cast(scale, q_dtype)

    scale_sqrt = tf.sqrt(scale_value)
    Q = Q * scale_sqrt
    K_scaled = attn_key * scale_sqrt

    # QK^T
    qk_matmul_output = tf.matmul(
        a=Q,
        b=K_scaled,
        transpose_b=True,
    )

    q_seq_len = tf.shape(Q)[2]
    kv_seq_len = tf.shape(present_key)[2]

    attn_bias = None

    if is_causal:
        causal_mask = tf.linalg.band_part(
            tf.ones(shape=tf.stack([q_seq_len, kv_seq_len]), dtype=tf.float32),
            -1,
            0,
        )
        causal_mask = tf.cast(causal_mask, tf.bool)
        neg_inf = tf.constant(-np.inf, dtype=q_dtype)
        causal_bias = tf.where(
            condition=causal_mask,
            x=tf.zeros_like(causal_mask, dtype=q_dtype),
            y=neg_inf,
        )
        attn_bias = causal_bias

    if attn_mask is not None and attn_mask != "":
        if attn_mask.dtype != tf.bool:
            attn_mask = tf.cast(attn_mask, q_dtype)
            pad_value = tf.constant(-np.inf, dtype=q_dtype)
        else:
            pad_value = False

        if opset >= 24:
            attn_mask = _pad_or_slice_last_dim(
                mask_tensor=attn_mask,
                target_len=kv_seq_len,
                pad_value=pad_value,
            )

        if attn_mask.dtype == tf.bool:
            neg_inf = tf.constant(-np.inf, dtype=q_dtype)
            mask_bias = tf.where(
                condition=attn_mask,
                x=tf.zeros_like(attn_mask, dtype=q_dtype),
                y=neg_inf,
            )
        else:
            mask_bias = attn_mask

        attn_bias = mask_bias if attn_bias is None else attn_bias + mask_bias

    if opset >= 24 and nonpad_kv_seqlen is not None and past_key is None and past_value is None:
        nonpad_kv_seqlen = tf.cast(nonpad_kv_seqlen, tf.int32)
        seq_range = tf.range(kv_seq_len, dtype=tf.int32)
        seq_range = tf.reshape(seq_range, shape=[1, -1])
        nonpad_mask = seq_range < tf.reshape(nonpad_kv_seqlen, shape=[-1, 1])
        nonpad_mask = tf.reshape(nonpad_mask, shape=[-1, 1, 1, kv_seq_len])
        neg_inf = tf.constant(-np.inf, dtype=q_dtype)
        nonpad_bias = tf.where(
            condition=nonpad_mask,
            x=tf.zeros_like(nonpad_mask, dtype=q_dtype),
            y=neg_inf,
        )
        attn_bias = nonpad_bias if attn_bias is None else attn_bias + nonpad_bias

    if attn_bias is not None:
        qk_with_bias = qk_matmul_output + attn_bias
    else:
        qk_with_bias = qk_matmul_output

    # Softcap
    qk_softcap = qk_with_bias
    if isinstance(softcap, (float, int, np.floating, np.integer)):
        if softcap > 0:
            softcap_value = tf.cast(softcap, q_dtype)
            qk_softcap = softcap_value * tf.math.tanh(qk_with_bias / softcap_value)
    else:
        softcap_value = tf.cast(softcap, q_dtype)
        safe_softcap = tf.where(
            condition=tf.equal(softcap_value, 0),
            x=tf.ones_like(softcap_value),
            y=softcap_value,
        )
        qk_softcap = tf.where(
            condition=softcap_value > 0,
            x=softcap_value * tf.math.tanh(qk_with_bias / safe_softcap),
            y=qk_with_bias,
        )

    # Softmax
    softmax_dtype = None
    if softmax_precision is not None:
        if softmax_precision in ONNX_DTYPES_TO_TF_DTYPES:
            softmax_dtype = ONNX_DTYPES_TO_TF_DTYPES[softmax_precision]
        elif int(softmax_precision) == 16:
            softmax_dtype = tf.bfloat16

    qk_softmax_input = qk_softcap
    if softmax_dtype is not None and softmax_dtype != qk_softmax_input.dtype:
        qk_softmax_input = tf.cast(qk_softmax_input, softmax_dtype)

    qk_softmax = tf.nn.softmax(
        logits=qk_softmax_input,
        axis=-1,
    )

    if softmax_dtype is not None and qk_softmax.dtype != q_dtype:
        qk_softmax = tf.cast(qk_softmax, q_dtype)

    # Output
    Y = tf.matmul(
        a=qk_softmax,
        b=attn_value,
    )

    if input_is_3d:
        Y = transpose_with_flexing_deterrence(
            input_tensor=Y,
            perm=[0, 2, 1, 3],
            **kwargs,
        )
        y_shape_dyn = tf.shape(Y)
        Y = tf.reshape(
            tensor=Y,
            shape=tf.stack([y_shape_dyn[0], y_shape_dyn[1], y_shape_dyn[2] * y_shape_dyn[3]]),
        )

    tf_layers_dict[graph_node_output_y.name]['tf_node'] = Y

    # Outputs for KV cache
    if graph_node_output_present_key is not None:
        tf_layers_dict[graph_node_output_present_key.name]['tf_node'] = present_key
    if graph_node_output_present_value is not None:
        tf_layers_dict[graph_node_output_present_value.name]['tf_node'] = present_value

    # qk_matmul_output output mode
    if graph_node_output_qk_matmul_output is not None:
        qk_output = qk_matmul_output
        if qk_matmul_output_mode == 1:
            qk_output = qk_with_bias
        elif qk_matmul_output_mode == 2:
            qk_output = qk_softcap
        elif qk_matmul_output_mode == 3:
            qk_output = qk_softmax
        tf_layers_dict[graph_node_output_qk_matmul_output.name]['tf_node'] = qk_output

    # Post-process transpose
    tf_layers_dict[graph_node_output_y.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output_y.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    if graph_node_output_present_key is not None:
        tf_layers_dict[graph_node_output_present_key.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output_present_key.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node_output_present_key.name,
            **kwargs,
        )
    if graph_node_output_present_value is not None:
        tf_layers_dict[graph_node_output_present_value.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output_present_value.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node_output_present_value.name,
            **kwargs,
        )
    if graph_node_output_qk_matmul_output is not None:
        tf_layers_dict[graph_node_output_qk_matmul_output.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output_qk_matmul_output.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node_output_qk_matmul_output.name,
            **kwargs,
        )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output_y.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'Attention',
                'tf_inputs': {
                    'a': qk_softmax,
                    'b': attn_value,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output_y.name]['tf_node'],
                },
            }
        )
