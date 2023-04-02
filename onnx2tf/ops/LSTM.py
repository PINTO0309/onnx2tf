import sys
from typing import List, Dict
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
)
from onnx2tf.utils.colors import Color

from tensorflow.python.framework import ops
from tensorflow.python.util import dispatch

class Affine(tf.keras.layers.Layer):
    def __init__(self, alpha: float, beta: float):
        super(Affine, self).__init__()
        self.alpha = tf.convert_to_tensor(alpha)
        self.beta = tf.convert_to_tensor(beta)

    def call(self, x):
        # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#lstm-14
        # alpha*x + beta
        return self.alpha * x + self.beta

class ScaledTanh(tf.keras.layers.Layer):
    def __init__(self, alpha: float, beta: float):
        super(ScaledTanh, self).__init__()
        self.alpha = tf.convert_to_tensor(alpha)
        self.beta = tf.convert_to_tensor(beta)

    def call(self, x):
        # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#lstm-7
        # alpha*Tanh(beta*x)
        return self.alpha * tf.nn.tanh(self.beta*x)


# {'activateion_name': [tf_activation, default_alpha, default_beta]}
ONNX_ACTIVATION_MAPPING: Dict[str, List] = {
    "Elu": [tf.nn.elu, 1.0, None],
    "HardSigmoid": [tf.keras.backend.hard_sigmoid, 0.2, 0.5],
    "LeakyRelu": [tf.nn.leaky_relu, 0.01, 0.0],
    "Relu": [tf.nn.relu, None, None],
    "Sigmoid": [tf.sigmoid, None, None],
    "Softsign": [tf.nn.softsign, None, None],
    "Softplus": [tf.nn.softplus, None, None],
    "Tanh": [tf.tanh, None, None],
    "ThresholdedRelu": [tf.keras.layers.ThresholdedReLU, 1.0, None],
    "Affine": [Affine, 1.0, 0.0],
    "ScaledTanh": [ScaledTanh, 1.0, 1.0]
}



@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """[WIP][TODO] LSTM
    Need to implement CustomLSTMCell and CustomLSTM to handle activation functions for f, g, and h
    https://zenn.dev/pinto0309/scraps/430cea62b1eb9d

    https://github.com/PINTO0309/onnx2tf/issues/198
    test onnx file: https://s3.ap-northeast-2.wasabisys.com/temp-models/onnx2tf_198/text_recognition_CRNN_EN_2021sep.onnx
    onnx2tf -i text_recognition_CRNN_EN_2021sep.onnx

    test onnx file: https://s3.ap-northeast-2.wasabisys.com/temp-models/onnx2tf_198/LSTM.tanh.bidirectional.onnx
    onnx2tf -i LSTM.tanh.bidirectional.onnx -kat Input3

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_3 = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_4 = True
    before_op_output_shape_trans_5 = True
    before_op_output_shape_trans_6 = True
    before_op_output_shape_trans_7 = True
    before_op_output_shape_trans_8 = True
    if len(graph_node.inputs) >= 4:
        before_op_output_shape_trans_4 = \
            tf_layers_dict.get(graph_node.inputs[3].name, {}).get('before_op_output_shape_trans', True)
    if len(graph_node.inputs) >= 5:
        before_op_output_shape_trans_5 = \
            tf_layers_dict.get(graph_node.inputs[4].name, {}).get('before_op_output_shape_trans', True)
    if len(graph_node.inputs) >= 6:
        before_op_output_shape_trans_6 = \
            tf_layers_dict.get(graph_node.inputs[5].name, {}).get('before_op_output_shape_trans', True)
    if len(graph_node.inputs) >= 7:
        before_op_output_shape_trans_7 = \
            tf_layers_dict.get(graph_node.inputs[6].name, {}).get('before_op_output_shape_trans', True)
    if len(graph_node.inputs) >= 8:
        before_op_output_shape_trans_8 = \
            tf_layers_dict.get(graph_node.inputs[7].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3 \
        and before_op_output_shape_trans_4 \
        and before_op_output_shape_trans_5 \
        and before_op_output_shape_trans_6 \
        and before_op_output_shape_trans_7 \
        and before_op_output_shape_trans_8

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
        is_bias=True,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
        is_bias=True,
    )
    # input
    X = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    # input_weights [num_directions, 4*hidden_size, input_size]
    # num_directions: bidirectional=2, forward or reverse=1
    W = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    # recurrent_weights [num_directions, 4*hidden_size, hidden_size]
    # num_directions: bidirectional=2, forward or reverse=1
    R = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    # Pre-process transpose
    X = pre_process_transpose(
        value_before_transpose=X,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    W = pre_process_transpose(
        value_before_transpose=W,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    R = pre_process_transpose(
        value_before_transpose=R,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )

    graph_node_input_4 = None
    if len(graph_node.inputs) >= 4:
        graph_node_input_4 = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
            is_bias=True,
        )
    graph_node_input_5 = None
    if len(graph_node.inputs) >= 5:
        graph_node_input_5 = get_constant_or_variable(
            graph_node.inputs[4],
            before_op_output_shape_trans,
        )
    graph_node_input_6 = None
    if len(graph_node.inputs) >= 6:
        graph_node_input_6 = get_constant_or_variable(
            graph_node.inputs[5],
            before_op_output_shape_trans,
        )
    graph_node_input_7 = None
    if len(graph_node.inputs) >= 7:
        graph_node_input_7 = get_constant_or_variable(
            graph_node.inputs[6],
            before_op_output_shape_trans,
        )
    graph_node_input_8 = None
    if len(graph_node.inputs) >= 8:
        graph_node_input_8 = get_constant_or_variable(
            graph_node.inputs[7],
            before_op_output_shape_trans,
        )

    # input_biases [num_directions, 8*hidden_size]
    # num_directions: bidirectional=2, forward or reverse=1
    B = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4

    if isinstance(B, np.ndarray) and np.sum(B) != 0.0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'The process for the case where Bias is set to a value greater than zero has not yet been implemented. ' +
            f'https://zenn.dev/pinto0309/scraps/430cea62b1eb9d ' +
            f'B.shape: {B.shape}'
        )
        sys.exit(1)

    # sequence_lens [batch_size]
    sequence_lens = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) and graph_node_input_5.name != '' else graph_node_input_5
    # initial_h [num_directions, batch_size, hidden_size]
    # num_directions: bidirectional=2, forward or reverse=1
    initial_h = tf_layers_dict[graph_node_input_6.name]['tf_node'] \
        if isinstance(graph_node_input_6, gs.Variable) else graph_node_input_6
    # initial_c [num_directions, batch_size, hidden_size]
    # num_directions: bidirectional=2, forward or reverse=1
    initial_c = tf_layers_dict[graph_node_input_7.name]['tf_node'] \
        if isinstance(graph_node_input_7, gs.Variable) else graph_node_input_7
    # P [num_directions, 3*hidden_size]
    # num_directions: bidirectional=2, forward or reverse=1
    P = tf_layers_dict[graph_node_input_8.name]['tf_node'] \
        if isinstance(graph_node_input_8, gs.Variable) and graph_node_input_8.name != '' else graph_node_input_8

    if isinstance(P, np.ndarray) and np.sum(P) != 0.0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'The process for the case where Peepholes is set to a value greater than zero has not yet been implemented. ' +
            f'https://zenn.dev/pinto0309/scraps/430cea62b1eb9d ' +
            f'P.shape: {P.shape}'
        )
        sys.exit(1)

    # Different value ranges for each activation function
    activation_alpha: List[float] = graph_node.attrs.get('activation_alpha', [0.01])
    # Different value ranges for each activation function
    activation_beta: List[float] = graph_node.attrs.get('activation_beta', [])
    # Always three or more present if specified
    activations: List[str] =  graph_node.attrs.get('activations', [])

    if len(activations) > 0:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'The activation functions for f, g, and h are currently not implemented. ' +
            f'https://zenn.dev/pinto0309/scraps/430cea62b1eb9d ' +
            f'https://github.com/microsoft/onnxruntime/blob/c57cf374b67f72575546d7b4c69a1af4972e2b54/onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc#L45-L84 ' +
            f'activations: {activations}'
        )
        sys.exit(1)

    clip: float =  graph_node.attrs.get('clip', None)

    if clip is not None:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'clip is currently not implemented. ' +
            f'clip: {clip}'
        )
        sys.exit(1)

    direction: str =  graph_node.attrs.get('direction', 'forward')
    hidden_size: int =  graph_node.attrs.get('hidden_size', 1)
    input_forget: bool = bool(graph_node.attrs.get('input_forget', 0))

    if input_forget:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'input_forget = 1 is currently not implemented. ' +
            f'input_forget: {input_forget}'
        )
        sys.exit(1)

    layout: int = graph_node.attrs.get('layout', 0)

    # Need transpose for batchwise, X
    if layout == 1:
        X = tf.transpose(X, perm=[1, 0, 2])

    graph_node_output1: gs.Variable = graph_node.outputs[0]
    shape1 = graph_node_output1.shape
    dtype1 = graph_node_output1.dtype
    graph_node_output2 = None
    if len(graph_node.outputs) >= 2:
        graph_node_output2: gs.Variable = graph_node.outputs[1]
        shape2 = graph_node_output2.shape
        dtype2 = graph_node_output2.dtype
    graph_node_output3 = None
    if len(graph_node.outputs) >= 3:
        graph_node_output3: gs.Variable = graph_node.outputs[2]
        shape3 = graph_node_output3.shape
        dtype3 = graph_node_output3.dtype


    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output1.name] = {
        'optype': graph_node.op,
        'shape': shape1,
        'dtype': dtype1,
    }
    if graph_node_output2 is not None:
        tf_layers_dict[graph_node_output2.name] = {
            'optype': graph_node.op,
            'shape': shape2,
            'dtype': dtype2,
        }
    if graph_node_output3 is not None:
        tf_layers_dict[graph_node_output3.name] = {
            'optype': graph_node.op,
            'shape': shape3,
            'dtype': dtype3,
        }

    # Param replacement
    X = replace_parameter(
        value_before_replacement=X,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    W = replace_parameter(
        value_before_replacement=W,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )
    R = replace_parameter(
        value_before_replacement=R,
        param_target='inputs',
        param_name=graph_node.inputs[2].name,
        **kwargs,
    )

    if len(graph_node.inputs) >= 4:
        B = replace_parameter(
            value_before_replacement=B,
            param_target='inputs',
            param_name=graph_node.inputs[3].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 5:
        sequence_lens = replace_parameter(
            value_before_replacement=sequence_lens,
            param_target='inputs',
            param_name=graph_node.inputs[4].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 6:
        initial_h = replace_parameter(
            value_before_replacement=initial_h,
            param_target='inputs',
            param_name=graph_node.inputs[5].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 7:
        initial_c = replace_parameter(
            value_before_replacement=initial_c,
            param_target='inputs',
            param_name=graph_node.inputs[6].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 8:
        P = replace_parameter(
            value_before_replacement=P,
            param_target='inputs',
            param_name=graph_node.inputs[7].name,
            **kwargs,
        )

    activation_alpha = replace_parameter(
        value_before_replacement=activation_alpha,
        param_target='attributes',
        param_name='activation_alpha',
        **kwargs,
    )
    activation_beta = replace_parameter(
        value_before_replacement=activation_beta,
        param_target='attributes',
        param_name='activation_beta',
        **kwargs,
    )
    activations = replace_parameter(
        value_before_replacement=activations,
        param_target='attributes',
        param_name='activations',
        **kwargs,
    )
    clip = replace_parameter(
        value_before_replacement=clip,
        param_target='attributes',
        param_name='clip',
        **kwargs,
    )
    hidden_size = replace_parameter(
        value_before_replacement=hidden_size,
        param_target='attributes',
        param_name='hidden_size',
        **kwargs,
    )
    input_forget = replace_parameter(
        value_before_replacement=input_forget,
        param_target='attributes',
        param_name='input_forget',
        **kwargs,
    )
    layout = replace_parameter(
        value_before_replacement=layout,
        param_target='attributes',
        param_name='layout',
        **kwargs,
    )

    # Generation of TF OP
    def create_lstm_layer(
        *,
        weight,
        recurrence_weight,
        bias=None,
        hidden_size,
        go_backwards,
    ):
        need_bias = True
        if isinstance(bias, np.ndarray) and np.sum(bias) == 0:
            need_bias = False

        if need_bias:
            lstm_layer = tf.keras.layers.LSTM(
                units=hidden_size,
                kernel_initializer=tf.keras.initializers.constant(weight),
                recurrent_initializer=tf.keras.initializers.constant(recurrence_weight),
                bias_initializer=tf.keras.initializers.constant(bias),
                return_sequences=True,
                return_state=True,
                go_backwards=go_backwards,
                dropout=0,
                recurrent_dropout=0,
                stateful=False,
                implementation=2,
            )
        else:
            lstm_layer = tf.keras.layers.LSTM(
                units=hidden_size,
                kernel_initializer=tf.keras.initializers.constant(weight),
                recurrent_initializer=tf.keras.initializers.constant(recurrence_weight),
                return_sequences=True,
                return_state=True,
                go_backwards=go_backwards,
                dropout=0,
                recurrent_dropout=0,
                stateful=False,
                implementation=2,
            )
        return lstm_layer

    forward_lstm = None
    reverse_lstm = None
    input_size = X.shape[-1]

    # Need transpose for batchwise, initial_h/initial_c
    if layout == 1:
        initial_h = tf.transpose(initial_h, perm=[1, 0, 2]) if initial_h is not None else None
        initial_c = tf.transpose(initial_c, perm=[1, 0, 2]) if initial_c is not None else None

    # initial state
    forward_initial_state = None
    backward_initial_state = None
    if direction == 'forward':
        forward_initial_state = [] + [tf.convert_to_tensor(initial_h[0])]
        if initial_c is not None:
            forward_initial_state = forward_initial_state + [tf.convert_to_tensor(initial_c[0])]
        elif initial_h is not None and initial_c is None:
            forward_initial_state = forward_initial_state + [tf.zeros_like(tf.convert_to_tensor(initial_h[0]))]

    elif direction == 'reverse':
        backward_initial_state = [] + [tf.convert_to_tensor(initial_h[0])]
        if initial_c is not None:
            backward_initial_state = backward_initial_state + [tf.convert_to_tensor(initial_c[0])]
        elif initial_h is not None and initial_c is None:
            backward_initial_state = backward_initial_state + [tf.zeros_like(tf.convert_to_tensor(initial_h[0]))]

    elif direction == 'bidirectional':
        forward_initial_state = [] + [tf.convert_to_tensor(initial_h[0])]
        if initial_c is not None:
            forward_initial_state = forward_initial_state + [tf.convert_to_tensor(initial_c[0])]
        elif initial_h is not None and initial_c is None:
            forward_initial_state = forward_initial_state + [tf.zeros_like(tf.convert_to_tensor(initial_h[0]))]
        backward_initial_state = [] + [tf.convert_to_tensor(initial_h[1])]
        if initial_c is not None:
            backward_initial_state = backward_initial_state + [tf.convert_to_tensor(initial_c[1])]
        elif initial_h is not None and initial_c is None:
            backward_initial_state = backward_initial_state + [tf.zeros_like(tf.convert_to_tensor(initial_h[1]))]

    if direction == 'forward':
        forward_weight = tf.reshape(W[0], shape=[4, hidden_size, input_size]) # TensorShape([4, 256, 512])
        forward_recurrence_weight = tf.reshape(R[0], shape=[4, hidden_size, hidden_size]) # TensorShape([4, 256, 256])
        forward_bias = tf.reshape(B[0][:4*hidden_size], shape=[4, hidden_size]) # TensorShape([4, 256])
        fW_i, fW_o, fW_f, fW_c = np.split(forward_weight, 4, axis=0) # (1, 256, 512)
        fR_i, fR_o, fR_f, fR_c = np.split(forward_recurrence_weight, 4, axis=0) # (1, 256, 256)
        fB_i, fB_o, fB_f, fB_c = np.split(forward_bias, 4, axis=0) # (1, 256)
        forward_kernel = np.concatenate([fW_i, fW_f, fW_c, fW_o], axis=1).transpose(2, 0, 1).reshape(input_size, -1) # (512, 1024)
        forward_recurrent_kernel = np.concatenate([fR_i, fR_f, fR_c, fR_o], axis=1).transpose(2, 0, 1).reshape(hidden_size, -1) # (256, 1024)
        forward_bias = np.concatenate([fB_i, fB_f, fB_c, fB_o], axis=1) # (1, 1024)

        forward_lstm = create_lstm_layer(
            weight=forward_kernel,
            recurrence_weight=forward_recurrent_kernel,
            bias=forward_bias,
            hidden_size=hidden_size,
            go_backwards=False,
        )
        output, hidden_state, cell_state = forward_lstm(X, initial_state=forward_initial_state)
        output = tf.expand_dims(output, axis=1)
        hidden_state = tf.expand_dims(hidden_state, axis=0)
        cell_state = tf.expand_dims(cell_state, axis=0)

    elif direction == 'reverse':
        backward_weight = tf.reshape(W[0], shape=[4, hidden_size, input_size]) # TensorShape([4, 256, 512])
        backward_recurrence_weight = tf.reshape(R[0], shape=[4, hidden_size, hidden_size]) # TensorShape([4, 256, 256])
        backward_bias = tf.reshape(B[0][4*hidden_size:4*hidden_size*2], shape=[4, hidden_size]) # TensorShape([4, 256])
        bW_i, bW_o, bW_f, bW_c = np.split(backward_weight, 4, axis=0) # (1, 256, 512)
        bR_i, bR_o, bR_f, bR_c = np.split(backward_recurrence_weight, 4, axis=0) # (1, 256, 256)
        bB_i, bB_o, bB_f, bB_c = np.split(backward_bias, 4, axis=0) # (1, 256)
        backward_kernel = np.concatenate([bW_i, bW_f, bW_c, bW_o], axis=1).transpose(2, 0, 1).reshape(input_size, -1) # (512, 1024)
        backward_recurrent_kernel = np.concatenate([bR_i, bR_f, bR_c, bR_o], axis=1).transpose(2, 0, 1).reshape(hidden_size, -1) # (256, 1024)
        backward_bias = np.concatenate([bB_i, bB_f, bB_c, bB_o], axis=1) # (1, 1024)

        reverse_lstm = create_lstm_layer(
            weight=backward_kernel,
            recurrence_weight=backward_recurrent_kernel,
            bias=backward_bias,
            hidden_size=hidden_size,
            go_backwards=True,
        )
        output, hidden_state, cell_state = reverse_lstm(X, initial_state=backward_initial_state)
        output = tf.expand_dims(output, axis=1)
        hidden_state = tf.expand_dims(hidden_state, axis=0)
        cell_state = tf.expand_dims(cell_state, axis=0)

    elif direction == 'bidirectional':
        forward_weight = tf.reshape(W[0], shape=[4, hidden_size, input_size]) # TensorShape([4, 256, 512])
        forward_recurrence_weight = tf.reshape(R[0], shape=[4, hidden_size, hidden_size]) # TensorShape([4, 256, 256])
        forward_bias = tf.reshape(B[0][:4*hidden_size], shape=[4, hidden_size]) # TensorShape([4, 256])
        fW_i, fW_o, fW_f, fW_c = np.split(forward_weight, 4, axis=0) # (1, 256, 512)
        fR_i, fR_o, fR_f, fR_c = np.split(forward_recurrence_weight, 4, axis=0) # (1, 256, 256)
        fB_i, fB_o, fB_f, fB_c = np.split(forward_bias, 4, axis=0) # (1, 256)

        forward_kernel = np.concatenate([fW_i, fW_f, fW_c, fW_o], axis=1).transpose(2, 0, 1).reshape(input_size, -1) # (512, 1024)
        forward_recurrent_kernel = np.concatenate([fR_i, fR_f, fR_c, fR_o], axis=1).transpose(2, 0, 1).reshape(hidden_size, -1) # (256, 1024)
        forward_bias = np.concatenate([fB_i, fB_f, fB_c, fB_o], axis=1) # (1, 1024)

        reverse_weight = tf.reshape(W[1], shape=[4, hidden_size, input_size])
        reverse_recurrence_weight = tf.reshape(R[1], shape=[4, hidden_size, hidden_size])
        reverse_bias = tf.reshape(B[1][4*hidden_size:4*hidden_size*2], shape=[4, hidden_size])
        rW_i, rW_o, rW_f, rW_c = np.split(reverse_weight, 4, axis=0)
        rR_i, rR_o, rR_f, rR_c = np.split(reverse_recurrence_weight, 4, axis=0)
        rB_i, rB_o, rB_f, rB_c = np.split(reverse_bias, 4, axis=0)

        reverse_kernel = np.concatenate([rW_i, rW_f, rW_c, rW_o], axis=1).transpose(2, 0, 1).reshape(input_size, -1)
        reverse_recurrent_kernel = np.concatenate([rR_i, rR_f, rR_c, rR_o], axis=1).transpose(2, 0, 1).reshape(hidden_size, -1)
        reverse_bias = np.concatenate([rB_i, rB_f, rB_c, rB_o], axis=1) # (1, 1024)

        # forward
        forward_lstm = create_lstm_layer(
            weight=forward_kernel, # (512, 1024)
            recurrence_weight=forward_recurrent_kernel, # (256, 1024)
            bias=forward_bias, # (1, 1024)
            hidden_size=hidden_size, # 256
            go_backwards=False,
        )
        # backward
        reverse_lstm = create_lstm_layer(
            weight=reverse_kernel, # (512, 1024)
            recurrence_weight=reverse_recurrent_kernel, # (256, 1024)
            bias=reverse_bias, # (1, 1024)
            hidden_size=hidden_size, # 256
            go_backwards=True,
        )
        forward_output, forward_h, forward_c = \
            forward_lstm(X, initial_state=forward_initial_state) # [24, 1, 512], [1, 256], [1, 256]
        reverse_output, reverse_h, reverse_c = \
            reverse_lstm(X, initial_state=backward_initial_state) # [24, 1, 512], [1, 256], [1, 256]
        output = tf.concat(
            values=[
                tf.expand_dims(forward_output, axis=1),
                tf.expand_dims(reverse_output, axis=1),
            ],
            axis=1,
        )
        hidden_state = tf.concat(
            values=[
                tf.expand_dims(forward_h, axis=0),
                tf.expand_dims(reverse_h, axis=0),
            ],
            axis=0,
        )
        cell_state = tf.concat(
            values=[
                tf.expand_dims(forward_c, axis=0),
                tf.expand_dims(reverse_c, axis=0),
            ],
            axis=0,
        )

    tf_layers_dict[graph_node_output1.name]['tf_node'] = output
    if graph_node_output2 is not None:
        tf_layers_dict[graph_node_output2.name]['tf_node'] = hidden_state
    if graph_node_output3 is not None:
        tf_layers_dict[graph_node_output3.name]['tf_node'] = cell_state

    # Post-process transpose
    tf_layers_dict[graph_node_output1.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output1.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    if graph_node_output2 is not None:
        tf_layers_dict[graph_node_output2.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output2.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node.outputs[1].name,
            **kwargs,
        )
    if graph_node_output3 is not None:
        tf_layers_dict[graph_node_output3.name]['tf_node'] = post_process_transpose(
            value_before_transpose=tf_layers_dict[graph_node_output3.name]['tf_node'],
            param_target='outputs',
            param_name=graph_node.outputs[2].name,
            **kwargs,
        )

    # Generation of Debug Info
    tf_outputs = {"output1": tf_layers_dict[graph_node_output1.name]['tf_node']}
    if graph_node_output2 is not None:
        tf_outputs["output2"] = tf_layers_dict[graph_node_output2.name]['tf_node']
    if graph_node_output3 is not None:
        tf_outputs["output3"] = tf_layers_dict[graph_node_output3.name]['tf_node']
    tf_layers_dict[graph_node_output1.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.keras.layers.LSTM,
                'tf_inputs': {
                    'X': X,
                    'W': W,
                    'R': R,
                    'B': B,
                    'sequence_lens': sequence_lens,
                    'initial_h': initial_h,
                    'initial_c': initial_c,
                    'P': P,
                    'activation_alpha': activation_alpha,
                    'activation_beta': activation_beta,
                    'activations': activations,
                    'clip': clip,
                    'hidden_size': hidden_size,
                    'input_forget': input_forget,
                    'layout': layout,
                },
                'tf_outputs': tf_outputs,
            }
        )
