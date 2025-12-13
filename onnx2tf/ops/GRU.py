from typing import List, Dict
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
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
from tensorflow.python.keras.layers import Layer


class Elu(Layer):
    def __init__(self, alpha: float, beta: float):
        super(Elu, self).__init__()

    def call(self, x):
        return tf.nn.elu(x)

class HardSigmoid(Layer):
    def __init__(self, alpha: float, beta: float):
        super(HardSigmoid, self).__init__()
        self.alpha = 0.2
        self.beta = 0.5
        self.alpha = tf.convert_to_tensor(alpha) \
            if self.alpha != alpha else tf.convert_to_tensor(self.alpha)
        self.beta = tf.convert_to_tensor(beta) \
            if self.beta != beta else tf.convert_to_tensor(self.beta)

    def call(self, x):
        # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#hardsigmoid-6
        # max(0, min(1, alpha * x + beta))
        return tf.maximum(0.0, tf.minimum(1.0, self.alpha * x + self.beta))

class LeakyReLU(Layer):
    def __init__(self, alpha: float, beta: float):
        super(LeakyReLU, self).__init__()
        self.alpha = 0.01
        self.alpha = tf.convert_to_tensor(alpha) \
            if self.alpha != alpha else tf.convert_to_tensor(self.alpha)

    def call(self, x):
        # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#leakyrelu-16
        return tf.nn.leaky_relu(x, alpha=self.alpha)

class ReLU(Layer):
    def __init__(self, alpha: float, beta: float):
        super(ReLU, self).__init__()

    def call(self, x):
        return tf.nn.relu(x)

class Sigmoid(Layer):
    def __init__(self, alpha: float, beta: float):
        super(Sigmoid, self).__init__()

    def call(self, x):
        return tf.sigmoid(x)

class Softsign(Layer):
    def __init__(self, alpha: float, beta: float):
        super(Softsign, self).__init__()

    def call(self, x):
        return tf.nn.softsign(x)

class Softplus(Layer):
    def __init__(self, alpha: float, beta: float):
        super(Softplus, self).__init__()

    def call(self, x):
        return tf.nn.softplus(x)

class Tanh(Layer):
    def __init__(self, alpha: float, beta: float):
        super(Tanh, self).__init__()

    def call(self, x):
        return tf.tanh(x)

class ThresholdedReLU(Layer):
    def __init__(self, alpha: float, beta: float):
        super(ThresholdedReLU, self).__init__()
        self.alpha = 1.0
        self.alpha = tf.convert_to_tensor(alpha) \
            if self.alpha != alpha else tf.convert_to_tensor(self.alpha)

    def call(self, x):
        return tf_keras.layers.ThresholdedReLU(theta=self.alpha)(x)

class Affine(Layer):
    def __init__(self, alpha: float, beta: float):
        super(Affine, self).__init__()
        self.alpha = 1.0
        self.beta = 0.0
        self.alpha = tf.convert_to_tensor(alpha) \
            if self.alpha != alpha else tf.convert_to_tensor(self.alpha)
        self.beta = tf.convert_to_tensor(beta) \
            if self.beta != beta else tf.convert_to_tensor(self.beta)

    def call(self, x):
        # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#lstm-14
        # alpha*x + beta
        return self.alpha * x + self.beta

class ScaledTanh(Layer):
    def __init__(self, alpha: float, beta: float):
        super(ScaledTanh, self).__init__()
        self.alpha = 1.0
        self.beta = 1.0
        self.alpha = tf.convert_to_tensor(alpha) \
            if self.alpha != alpha else tf.convert_to_tensor(self.alpha)
        self.beta = tf.convert_to_tensor(beta) \
            if self.beta != beta else tf.convert_to_tensor(self.beta)

    def call(self, x):
        # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#lstm-7
        # alpha*Tanh(beta*x)
        return self.alpha * tf.nn.tanh(self.beta * x)



# {'activateion_name': [tf_activation, default_alpha, default_beta]}
ONNX_ACTIVATION_MAPPING: Dict[str, List] = {
    "Elu": [Elu, 1.0, 0.0],
    "HardSigmoid": [HardSigmoid, 0.2, 0.5],
    "LeakyRelu": [LeakyReLU, 0.01, 0.0],
    "Relu": [ReLU, 1.0, 0.0],
    "Sigmoid": [Sigmoid, 1.0, 0.0],
    "Softsign": [Softsign, 1.0, 0.0],
    "Softplus": [Softplus, 1.0, 0.0],
    "Tanh": [Tanh, 1.0, 0.0],
    "ThresholdedRelu": [ThresholdedReLU, 1.0, 0.0],
    "Affine": [Affine, 1.0, 0.0],
    "ScaledTanh": [ScaledTanh, 1.0, 1.0],
}


class CustomGRUCell(tf_keras.layers.AbstractRNNCell):
    def __init__(
        self,
        hidden_size,

        w_z,
        w_r,
        w_h,

        r_z,
        r_r,
        r_h,

        w_bz,
        w_br,
        w_bh,

        r_bz,
        r_br,
        r_bh,

        activations,

        clip,
        linear_before_reset,
        is_bidirectional,
        go_backwards,
        **kwargs
    ):
        super(CustomGRUCell, self).__init__(**kwargs)
        self.hidden_size = hidden_size

        self.w_z=w_z
        self.w_r=w_r
        self.w_h=w_h

        self.r_z=r_z
        self.r_r=r_r
        self.r_h=r_h

        self.w_bz=w_bz
        self.w_br=w_br
        self.w_bh=w_bh

        self.r_bz=r_bz
        self.r_br=r_br
        self.r_bh=r_bh

        self.activations = activations

        self.clip = clip
        self.linear_before_reset = linear_before_reset
        self.is_bidirectional = is_bidirectional
        self.go_backwards = go_backwards

    @property
    def state_size(self):
        return [self.hidden_size]

    def call(self, inputs, states):
        # Custom activation functions
        """
        Default activations: f=Sigmoid, g=Tanh
        ONNX:
            zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
            rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

            ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
            ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
            Ht = (1 - zt) (.) ht + zt (.) Ht-1
        """
        h_prev = states[0]

        # z = self.f(np.dot(row, np.transpose(w_z)) + np.dot(h_prev, np.transpose(r_z)) + w_bz + r_bz)
        # r = self.f(np.dot(row, np.transpose(w_r)) + np.dot(h_prev, np.transpose(r_r)) + w_br + r_br)
        z = tf.matmul(inputs, tf.transpose(self.w_z)) + tf.matmul(h_prev, tf.transpose(self.r_z)) + self.w_bz + self.r_bz
        r = tf.matmul(inputs, tf.transpose(self.w_r)) + tf.matmul(h_prev, tf.transpose(self.r_r)) + self.w_br + self.r_br

        offsetidx = 2 if self.is_bidirectional and self.go_backwards else 0

        if not self.clip:
            z = self.activations[0 + offsetidx](z)
            r = self.activations[0 + offsetidx](r)
        else:
            z = self.activations[0 + offsetidx](
                tf.clip_by_value(
                    z,
                    clip_value_min=-self.clip,
                    clip_value_max=self.clip,
                )
            )
            r = self.activations[0 + offsetidx](
                tf.clip_by_value(
                    r,
                    clip_value_min=-self.clip,
                    clip_value_max=self.clip,
                )
            )

        # h_default = self.g(np.dot(row, np.transpose(w_h)) + np.dot(r * h_prev, np.transpose(r_h)) + w_bh + r_bh)
        # h_linear = self.g(np.dot(row, np.transpose(w_h)) + r * (np.dot(h_prev, np.transpose(r_h)) + r_bh) + w_bh)
        if not self.linear_before_reset:
            h = tf.matmul(inputs, tf.transpose(self.w_h)) + tf.matmul(r * h_prev, tf.transpose(self.r_h)) + self.r_bh + self.w_bh
        else:
            h = tf.matmul(inputs, tf.transpose(self.w_h)) + r * (tf.matmul(h_prev, tf.transpose(self.r_h)) + self.r_bh) + self.w_bh

        if not self.clip:
            h = self.activations[1 + offsetidx](h)
        else:
            h = self.activations[1 + offsetidx](
                tf.clip_by_value(
                    h,
                    clip_value_min=-self.clip,
                    clip_value_max=self.clip,
                )
            )

        h = (1.0 - z) * h + z * h_prev

        return h, [h]


class CustomGRU(Layer):
    def __init__(
        self,
        hidden_size,

        w_z,
        w_r,
        w_h,

        r_z,
        r_r,
        r_h,

        w_bz,
        w_br,
        w_bh,

        r_bz,
        r_br,
        r_bh,

        activations,

        clip,
        linear_before_reset,
        is_bidirectional,
        go_backwards,
        enable_rnn_unroll,
        return_sequences=True,
        **kwargs
    ):
        super(CustomGRU, self).__init__(**kwargs)
        self.hidden_size = hidden_size

        self.w_z=w_z
        self.w_r=w_r
        self.w_h=w_h

        self.r_z=r_z
        self.r_r=r_r
        self.r_h=r_h

        self.w_bz=w_bz
        self.w_br=w_br
        self.w_bh=w_bh

        self.r_bz=r_bz
        self.r_br=r_br
        self.r_bh=r_bh

        self.activations = activations

        self.return_sequences = return_sequences
        self.is_bidirectional = is_bidirectional
        self.go_backwards = go_backwards
        self.enable_rnn_unroll = enable_rnn_unroll

        self.clip = clip
        self.linear_before_reset = linear_before_reset

        self.cell = CustomGRUCell(
            self.hidden_size,

            self.w_z,
            self.w_r,
            self.w_h,

            self.r_z,
            self.r_r,
            self.r_h,

            self.w_bz,
            self.w_br,
            self.w_bh,

            self.r_bz,
            self.r_br,
            self.r_bh,

            self.activations,

            self.clip,
            self.linear_before_reset,
            self.is_bidirectional,
            self.go_backwards,
        )
        self.rnn = tf_keras.layers.RNN(
            self.cell,
            return_sequences=self.return_sequences,
            go_backwards=self.go_backwards,
            return_state=True,
            unroll=self.enable_rnn_unroll,
        )

    def call(self, inputs, initial_state=None):
        outputs, h = self.rnn(inputs, initial_state=initial_state)
        return outputs, h


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """GRU

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
    if len(graph_node.inputs) >= 4:
        before_op_output_shape_trans_4 = \
            tf_layers_dict.get(graph_node.inputs[3].name, {}).get('before_op_output_shape_trans', True)
    if len(graph_node.inputs) >= 5:
        before_op_output_shape_trans_5 = \
            tf_layers_dict.get(graph_node.inputs[4].name, {}).get('before_op_output_shape_trans', True)
    if len(graph_node.inputs) >= 6:
        before_op_output_shape_trans_6 = \
            tf_layers_dict.get(graph_node.inputs[5].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3 \
        and before_op_output_shape_trans_4 \
        and before_op_output_shape_trans_5 \
        and before_op_output_shape_trans_6

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

    # input_biases [num_directions, 8*hidden_size]
    # num_directions: bidirectional=2, forward or reverse=1
    B = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4
    # sequence_lens [batch_size]
    sequence_lens = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) and graph_node_input_5.name != '' else graph_node_input_5
    if isinstance(graph_node_input_5, gs.Variable) and graph_node_input_5.is_empty():
        sequence_lens = None
    # initial_h [num_directions, batch_size, hidden_size]
    # num_directions: bidirectional=2, forward or reverse=1
    initial_h = tf_layers_dict[graph_node_input_6.name]['tf_node'] \
        if isinstance(graph_node_input_6, gs.Variable) and graph_node_input_6.name != '' else graph_node_input_6
    if isinstance(graph_node_input_6, gs.Variable) and graph_node_input_6.is_empty():
        initial_h = None

    # Always three or more present if specified
    #   forward, reverse: 3 items
    #   bidirectional: 6 items
    activations: List[str] =  graph_node.attrs.get('activations', [])
    # Different value ranges for each activation function
    #   forward, reverse: 3 items
    #   bidirectional: 6 items
    activation_alpha: List[float] = graph_node.attrs.get('activation_alpha', [0.01])
    # Different value ranges for each activation function
    #   forward, reverse: 3 items
    #   bidirectional: 6 items
    activation_beta: List[float] = graph_node.attrs.get('activation_beta', [])

    # Default activation function setting
    tf_activations: List = None

    clip: float =  graph_node.attrs.get('clip', None)
    direction: str =  graph_node.attrs.get('direction', 'forward')

    if len(activations) == 0:
        # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#gru-14
        # Equations (Default: f=Sigmoid, g=Tanh)
        default_activations = [
            'Sigmoid', # f (Oblivion Gate)
            'Tanh',    # g (Input Gate)
        ]
        tf_activations = [
            # f (Oblivion Gate)
            ONNX_ACTIVATION_MAPPING[default_activations[0]][0](
                alpha=ONNX_ACTIVATION_MAPPING[default_activations[0]][1], beta=ONNX_ACTIVATION_MAPPING[default_activations[0]][2]
            ),
            # g (Input Gate)
            ONNX_ACTIVATION_MAPPING[default_activations[1]][0](
                alpha=ONNX_ACTIVATION_MAPPING[default_activations[1]][1], beta=ONNX_ACTIVATION_MAPPING[default_activations[1]][2]
            ),
        ]
        tf_activations = tf_activations + [
            # f (Oblivion Gate)
            ONNX_ACTIVATION_MAPPING[default_activations[0]][0](
                alpha=ONNX_ACTIVATION_MAPPING[default_activations[0]][1], beta=ONNX_ACTIVATION_MAPPING[default_activations[0]][2]
            ),
            # g (Input Gate)
            ONNX_ACTIVATION_MAPPING[default_activations[1]][0](
                alpha=ONNX_ACTIVATION_MAPPING[default_activations[1]][1], beta=ONNX_ACTIVATION_MAPPING[default_activations[1]][2]
            ),
        ] if direction == 'bidirectional' else tf_activations

    else:
        tf_activations = [
            # f (Oblivion Gate)
            ONNX_ACTIVATION_MAPPING[activations[0]][0](
                alpha=ONNX_ACTIVATION_MAPPING[activations[0]][1], beta=ONNX_ACTIVATION_MAPPING[activations[0]][2]
            ),
            # g (Input Gate)
            ONNX_ACTIVATION_MAPPING[activations[1]][0](
                alpha=ONNX_ACTIVATION_MAPPING[activations[1]][1], beta=ONNX_ACTIVATION_MAPPING[activations[1]][2]
            ),
        ]
        tf_activations = tf_activations + [
            # f (Oblivion Gate)
            ONNX_ACTIVATION_MAPPING[activations[2]][0](
                alpha=ONNX_ACTIVATION_MAPPING[activations[2]][1], beta=ONNX_ACTIVATION_MAPPING[activations[2]][2]
            ),
            # g (Input Gate)
            ONNX_ACTIVATION_MAPPING[activations[3]][0](
                alpha=ONNX_ACTIVATION_MAPPING[activations[3]][1], beta=ONNX_ACTIVATION_MAPPING[activations[3]][2]
            ),
        ] if direction == 'bidirectional' else tf_activations

    hidden_size: int =  graph_node.attrs.get('hidden_size', 1)
    linear_before_reset: bool = bool(graph_node.attrs.get('linear_before_reset', 0))
    layout: int = graph_node.attrs.get('layout', 0)

    # Need transpose for batchwise, X
    # layout==0
    #   onnx: [seq_length, batch_size, input_size]
    #   tf  : [batch_size, seq_length(timesteps), input_size]
    # layout==1
    #   onnx: [batch_size, seq_length, input_size]
    #   tf  : [batch_size, seq_length(timesteps), input_size]
    if layout == 0:
        X = tf.transpose(X, perm=[1, 0, 2])

    graph_node_output1: gs.Variable = graph_node.outputs[0]
    shape1 = graph_node_output1.shape
    dtype1 = graph_node_output1.dtype
    graph_node_output2 = None
    if len(graph_node.outputs) >= 2:
        graph_node_output2: gs.Variable = graph_node.outputs[1]
        shape2 = graph_node_output2.shape
        dtype2 = graph_node_output2.dtype

    enable_rnn_unroll: bool = kwargs['enable_rnn_unroll']

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
    linear_before_reset = replace_parameter(
        value_before_replacement=linear_before_reset,
        param_target='attributes',
        param_name='linear_before_reset',
        **kwargs,
    )
    layout = replace_parameter(
        value_before_replacement=layout,
        param_target='attributes',
        param_name='layout',
        **kwargs,
    )

    # Generation of TF OP
    forward_lstm = None
    reverse_lstm = None

    # Need transpose for batchwise, initial_h/initial_c
    # layout==0
    #   onnx: [num_directions, batch_size, hidden_size]
    #   tf  : [num_directions, batch_size, hidden_size]
    # layout==1
    #   onnx: [batch_size, num_directions, hidden_size]
    #   tf  : [num_directions, batch_size, hidden_size]
    if layout == 1:
        initial_h = tf.transpose(initial_h, perm=[1, 0, 2]) if initial_h is not None else None

    # initial state
    forward_initial_state = None
    backward_initial_state = None
    if direction == 'forward':
        forward_initial_state = [tf.convert_to_tensor(initial_h[0])] if initial_h is not None else None

    elif direction == 'reverse':
        backward_initial_state = [tf.convert_to_tensor(initial_h[0])] if initial_h is not None else None

    elif direction == 'bidirectional':
        forward_initial_state = [tf.convert_to_tensor(initial_h[0])] if initial_h is not None else None
        backward_initial_state = [tf.convert_to_tensor(initial_h[1])] if initial_h is not None else None

    # LSTM layer
    if direction == 'forward':
        tW = tf.convert_to_tensor(W[0]) # [12, 3]
        tR = tf.convert_to_tensor(R[0]) # [12, 4]
        tB = tf.convert_to_tensor(B[0]) # [24]
        w_z, w_r, w_h = tf.split(tW, num_or_size_splits=3) # [4, 3], [4, 3], [4, 3]
        r_z, r_r, r_h = tf.split(tR, num_or_size_splits=3) # [4, 4], [4, 4], [4, 4]
        w_bz, w_br, w_bh, r_bz, r_br, r_bh = tf.split(tB, num_or_size_splits=6) # [4], [4], [4], [4], [4], [4]

        # forward
        forward_lstm = CustomGRU(
            hidden_size=hidden_size,

            w_z=w_z,
            w_r=w_r,
            w_h=w_h,

            r_z=r_z,
            r_r=r_r,
            r_h=r_h,

            w_bz=w_bz,
            w_br=w_br,
            w_bh=w_bh,

            r_bz=r_bz,
            r_br=r_br,
            r_bh=r_bh,

            activations=tf_activations,

            clip=clip,
            linear_before_reset=linear_before_reset,
            is_bidirectional=False,
            go_backwards=False,
            enable_rnn_unroll=enable_rnn_unroll,
        )
        output, hidden_state = forward_lstm(X, initial_state=forward_initial_state) # [2,5,3], [2, 4]
        output = tf.expand_dims(output, axis=1)
        hidden_state = tf.expand_dims(hidden_state, axis=0)

    elif direction == 'reverse':
        tW = tf.convert_to_tensor(W[0]) # [12, 3]
        tR = tf.convert_to_tensor(R[0]) # [12, 4]
        tB = tf.convert_to_tensor(B[0]) # [24]
        w_z, w_r, w_h = tf.split(tW, num_or_size_splits=3) # [4, 3], [4, 3], [4, 3]
        r_z, r_r, r_h = tf.split(tR, num_or_size_splits=3) # [4, 4], [4, 4], [4, 4]
        w_bz, w_br, w_bh, r_bz, r_br, r_bh = tf.split(tB, num_or_size_splits=6) # [4], [4], [4], [4], [4], [4]

        # reverse
        reverse_lstm = CustomGRU(
            hidden_size=hidden_size,

            w_z=w_z,
            w_r=w_r,
            w_h=w_h,

            r_z=r_z,
            r_r=r_r,
            r_h=r_h,

            w_bz=w_bz,
            w_br=w_br,
            w_bh=w_bh,

            r_bz=r_bz,
            r_br=r_br,
            r_bh=r_bh,

            activations=tf_activations,

            clip=clip,
            linear_before_reset=linear_before_reset,
            is_bidirectional=False,
            go_backwards=True,
            enable_rnn_unroll=enable_rnn_unroll,
        )
        output, hidden_state = reverse_lstm(X, initial_state=backward_initial_state) # [2,5,3], [2, 4]
        output = tf.reverse(output, axis=[1])
        output = tf.expand_dims(output, axis=1)
        hidden_state = tf.expand_dims(hidden_state, axis=0)

    elif direction == 'bidirectional':
        ftW = tf.convert_to_tensor(W[0]) # [12, 3]
        ftR = tf.convert_to_tensor(R[0]) # [12, 4]
        ftB = tf.convert_to_tensor(B[0]) # [24]
        fw_z, fw_r, fw_h = tf.split(ftW, num_or_size_splits=3) # [4, 3], [4, 3], [4, 3]
        fr_z, fr_r, fr_h = tf.split(ftR, num_or_size_splits=3) # [4, 4], [4, 4], [4, 4]
        fw_bz, fw_br, fw_bh, fr_bz, fr_br, fr_bh = tf.split(ftB, num_or_size_splits=6) # [4], [4], [4], [4], [4], [4]

        rtW = tf.convert_to_tensor(W[1]) # [12, 3]
        rtR = tf.convert_to_tensor(R[1]) # [12, 4]
        rtB = tf.convert_to_tensor(B[1]) # [24]
        rw_z, rw_r, rw_h = tf.split(rtW, num_or_size_splits=3) # [4, 3], [4, 3], [4, 3]
        rr_z, rr_r, rr_h = tf.split(rtR, num_or_size_splits=3) # [4, 4], [4, 4], [4, 4]
        rw_bz, rw_br, rw_bh, rr_bz, rr_br, rr_bh = tf.split(rtB, num_or_size_splits=6) # [4], [4], [4], [4], [4], [4]

        # forward
        forward_lstm = CustomGRU(
            hidden_size=hidden_size,

            w_z=fw_z,
            w_r=fw_r,
            w_h=fw_h,

            r_z=fr_z,
            r_r=fr_r,
            r_h=fr_h,

            w_bz=fw_bz,
            w_br=fw_br,
            w_bh=fw_bh,

            r_bz=fr_bz,
            r_br=fr_br,
            r_bh=fr_bh,

            activations=tf_activations,

            clip=clip,
            linear_before_reset=linear_before_reset,
            is_bidirectional=True,
            go_backwards=False,
            enable_rnn_unroll=enable_rnn_unroll,
        )

        # backward
        reverse_lstm = CustomGRU(
            hidden_size=hidden_size,

            w_z=rw_z,
            w_r=rw_r,
            w_h=rw_h,

            r_z=rr_z,
            r_r=rr_r,
            r_h=rr_h,

            w_bz=rw_bz,
            w_br=rw_br,
            w_bh=rw_bh,

            r_bz=rr_bz,
            r_br=rr_br,
            r_bh=rr_bh,

            activations=tf_activations,

            clip=clip,
            linear_before_reset=linear_before_reset,
            is_bidirectional=True,
            go_backwards=True,
            enable_rnn_unroll=enable_rnn_unroll,
        )

        forward_output, forward_h = \
            forward_lstm(X, initial_state=forward_initial_state)
        reverse_output, reverse_h = \
            reverse_lstm(X, initial_state=backward_initial_state)
        output = tf.concat(
            values=[
                tf.expand_dims(forward_output, axis=1),
                tf.expand_dims(tf.reverse(reverse_output, axis=[1]), axis=1),
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

    if len(output.shape) == 4:
        output = tf.transpose(output, perm=[2,1,0,3])

    tf_layers_dict[graph_node_output1.name]['tf_node'] = output
    if graph_node_output2 is not None:
        tf_layers_dict[graph_node_output2.name]['tf_node'] = hidden_state

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

    # Generation of Debug Info
    tf_outputs = {"output1": tf_layers_dict[graph_node_output1.name]['tf_node']}
    if graph_node_output2 is not None:
        tf_outputs["output2"] = tf_layers_dict[graph_node_output2.name]['tf_node']
    tf_layers_dict[graph_node_output1.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_keras.layers.GRU,
                'tf_inputs': {
                    'direction': direction,
                    'X': X,
                    'W': W,
                    'R': R,
                    'B': B,
                    'hidden_size': hidden_size,
                    'sequence_lens': sequence_lens,
                    'initial_h': initial_h,
                    'activations': tf_activations,
                    'clip': clip,
                    'linear_before_reset': linear_before_reset,
                    'layout': layout,
                },
                'tf_outputs': tf_outputs,
            }
        )
