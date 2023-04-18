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
        return tf.keras.layers.ThresholdedReLU(theta=self.alpha)(x)

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


class CustomLSTMCell(tf.keras.layers.AbstractRNNCell):
    def __init__(
        self,
        hidden_size,
        kernel,
        recurrent_kernel,
        activation_alphas,
        activation_betas,
        activations,
        bias_i,
        bias_f,
        bias_c,
        bias_o,
        peephole_i,
        peephole_f,
        peephole_o,
        clip,
        input_forget,
        is_bidirectional,
        go_backwards,
        **kwargs
    ):
        super(CustomLSTMCell, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.activation_alphas = activation_alphas
        self.activation_betas = activation_betas
        self.activations = activations
        self.bi = bias_i
        self.bf = bias_f
        self.bc = bias_c
        self.bo = bias_o
        self.peephole_i = peephole_i
        self.peephole_f = peephole_f
        self.peephole_o = peephole_o
        self.clip = clip
        self.input_forget = input_forget
        self.is_bidirectional = is_bidirectional
        self.go_backwards = go_backwards

    @property
    def state_size(self):
        return [self.hidden_size, self.hidden_size]

    def call(self, inputs, states):
        # Custom activation functions
        """
        Default activations: f=Sigmoid, g=Tanh, h=Tanh
        ONNX:
            it = f( Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi )
            ft = f( Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf )
            ct = g( Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc )

            Ct = ft (.) Ct-1 + it (.) ct

            ot = f( Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo )

            Ht = ot (.) h( Ct )
        """
        h_prev, c_prev = states
        gates = tf.matmul(inputs, self.kernel) + tf.matmul(h_prev, self.recurrent_kernel)
        i, f, c_candidate, o = tf.split(gates, num_or_size_splits=4, axis=-1)
        if self.peephole_i is not None:
            i = i + tf.matmul(self.peephole_i, c_prev)
        if self.peephole_f is not None:
            f = f + tf.matmul(self.peephole_f, c_prev)

        offsetidx = 3 if self.is_bidirectional and self.go_backwards else 0

        if not self.clip:
            i = self.activations[0 + offsetidx](i + self.bi)
            if not self.input_forget:
                f = self.activations[0 + offsetidx](f + self.bf)
            else:
                f = 1.0 - i
            c_candidate = self.activations[1 + offsetidx](c_candidate + self.bc)
        else:
            i = self.activations[0 + offsetidx](
                tf.clip_by_value(
                    i + self.bi,
                    clip_value_min=-self.clip,
                    clip_value_max=self.clip,
                )
            )
            if not self.input_forget:
                f = self.activations[0 + offsetidx](
                    tf.clip_by_value(
                        f + self.bf,
                        clip_value_min=-self.clip,
                        clip_value_max=self.clip,
                    )
                )
            else:
                f = 1.0 - i
            c_candidate = self.activations[1 + offsetidx](
                tf.clip_by_value(
                    c_candidate + self.bc,
                    clip_value_min=-self.clip,
                    clip_value_max=self.clip,
                )
            )

        c = f * c_prev + i * c_candidate

        if self.peephole_o is None:
            o = o + self.bo
        else:
            o = o + tf.matmul(self.peephole_o, c) + self.bo

        if not self.clip:
            o = self.activations[0 + offsetidx](o)
        else:
            o = self.activations[0 + offsetidx](
                tf.clip_by_value(
                    o,
                    clip_value_min=-self.clip,
                    clip_value_max=self.clip,
                )
            )

        h = o * self.activations[2 + offsetidx](c)

        return h, [h, c]


class CustomLSTM(Layer):
    def __init__(
        self,
        hidden_size,
        kernel,
        recurrent_kernel,
        activation_alphas,
        activation_betas,
        activations,
        bias_i,
        bias_f,
        bias_c,
        bias_o,
        peephole_i,
        peephole_f,
        peephole_o,
        clip,
        input_forget,
        is_bidirectional,
        go_backwards,
        enable_rnn_unroll,
        return_sequences=True,
        **kwargs
    ):
        super(CustomLSTM, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.kernel = kernel
        self.recurrent_kernel = recurrent_kernel
        self.activation_alphas = activation_alphas
        self.activation_betas = activation_betas
        self.activations = activations
        self.return_sequences = return_sequences
        self.is_bidirectional = is_bidirectional
        self.go_backwards = go_backwards
        self.enable_rnn_unroll = enable_rnn_unroll
        self.bias_i = bias_i
        self.bias_f = bias_f
        self.bias_c = bias_c
        self.bias_o = bias_o
        self.peephole_i = peephole_i
        self.peephole_f = peephole_f
        self.peephole_o = peephole_o
        self.clip = clip
        self.input_forget = input_forget

        self.cell = CustomLSTMCell(
            self.hidden_size,
            self.kernel,
            self.recurrent_kernel,
            self.activation_alphas,
            self.activation_betas,
            self.activations,
            self.bias_i,
            self.bias_f,
            self.bias_c,
            self.bias_o,
            self.peephole_i,
            self.peephole_f,
            self.peephole_o,
            self.clip,
            self.input_forget,
            self.is_bidirectional,
            self.go_backwards,
        )
        self.rnn = tf.keras.layers.RNN(
            self.cell,
            return_sequences=self.return_sequences,
            go_backwards=self.go_backwards,
            return_state=True,
            unroll=self.enable_rnn_unroll,
        )

    def call(self, inputs, initial_state=None):
        outputs, h, c = self.rnn(inputs, initial_state=initial_state)
        return outputs, h, c


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """LSTM

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
    tf_activation_alphas: List = None
    tf_activation_betas: List = None

    clip: float =  graph_node.attrs.get('clip', None)
    direction: str =  graph_node.attrs.get('direction', 'forward')
    if len(activations) == 0:
        # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LSTM-14
        # Equations (Default: f=Sigmoid, g=Tanh, h=Tanh)
        default_activations = [
            'Sigmoid', # f (Oblivion Gate)
            'Tanh',    # g (Input Gate)
            'Tanh',    # h (Output Gate)
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
            # h (Output Gate)
            ONNX_ACTIVATION_MAPPING[default_activations[2]][0](
                alpha=ONNX_ACTIVATION_MAPPING[default_activations[2]][1], beta=ONNX_ACTIVATION_MAPPING[default_activations[2]][2]
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
            # h (Output Gate)
            ONNX_ACTIVATION_MAPPING[default_activations[2]][0](
                alpha=ONNX_ACTIVATION_MAPPING[default_activations[2]][1], beta=ONNX_ACTIVATION_MAPPING[default_activations[2]][2]
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
            # h (Output Gate)
            ONNX_ACTIVATION_MAPPING[activations[2]][0](
                alpha=ONNX_ACTIVATION_MAPPING[activations[2]][1], beta=ONNX_ACTIVATION_MAPPING[activations[2]][2]
            ),
        ]
        tf_activations = tf_activations + [
            # f (Oblivion Gate)
            ONNX_ACTIVATION_MAPPING[activations[3]][0](
                alpha=ONNX_ACTIVATION_MAPPING[activations[3]][1], beta=ONNX_ACTIVATION_MAPPING[activations[3]][2]
            ),
            # g (Input Gate)
            ONNX_ACTIVATION_MAPPING[activations[4]][0](
                alpha=ONNX_ACTIVATION_MAPPING[activations[4]][1], beta=ONNX_ACTIVATION_MAPPING[activations[4]][2]
            ),
            # h (Output Gate)
            ONNX_ACTIVATION_MAPPING[activations[5]][0](
                alpha=ONNX_ACTIVATION_MAPPING[activations[5]][1], beta=ONNX_ACTIVATION_MAPPING[activations[5]][2]
            ),
        ] if direction == 'bidirectional' else tf_activations

    hidden_size: int =  graph_node.attrs.get('hidden_size', 1)
    input_forget: bool = bool(graph_node.attrs.get('input_forget', 0))
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
    graph_node_output3 = None
    if len(graph_node.outputs) >= 3:
        graph_node_output3: gs.Variable = graph_node.outputs[2]
        shape3 = graph_node_output3.shape
        dtype3 = graph_node_output3.dtype

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
    forward_lstm = None
    reverse_lstm = None
    input_size = X.shape[-1]

    # Need transpose for batchwise, initial_h/initial_c
    # layout==0
    #   onnx: [num_directions, batch_size, hidden_size]
    #   tf  : [num_directions, batch_size, hidden_size]
    # layout==1
    #   onnx: [batch_size, num_directions, hidden_size]
    #   tf  : [num_directions, batch_size, hidden_size]
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

    # LSTM layer
    if direction == 'forward':
        forward_weight = tf.reshape(tf.convert_to_tensor(W[0]), shape=[4, hidden_size, input_size]) # TensorShape([4, 256, 512])
        forward_recurrence_weight = tf.reshape(tf.convert_to_tensor(R[0]), shape=[4, hidden_size, hidden_size]) # TensorShape([4, 256, 256])
        forward_bias_W = tf.reshape(tf.convert_to_tensor(B[0][:4*hidden_size]), shape=[4, hidden_size]) # TensorShape([4, 256])
        forward_bias_R = tf.reshape(tf.convert_to_tensor(B[0][4*hidden_size:4*hidden_size*2]), shape=[4, hidden_size]) # TensorShape([4, 256])
        forward_bias = forward_bias_W + forward_bias_R
        forward_peepholes = None
        if P is not None:
            forward_peepholes = tf.reshape(tf.convert_to_tensor(P[0]), shape=[3, hidden_size])
        fW_i, fW_o, fW_f, fW_c = tf.split(value=forward_weight, num_or_size_splits=4, axis=0) # (1, 256, 512)
        fR_i, fR_o, fR_f, fR_c = tf.split(value=forward_recurrence_weight, num_or_size_splits=4, axis=0) # (1, 256, 256)
        fB_i, fB_o, fB_f, fB_c = tf.split(value=forward_bias, num_or_size_splits=4, axis=0) # (1, 256)
        fP_i = None
        fP_o = None
        fP_f = None
        if forward_peepholes is not None:
            fP_i, fP_o, fP_f = tf.split(value=forward_peepholes, num_or_size_splits=3, axis=0) # (1, 256)
        forward_kernel = tf.reshape(tf.transpose(tf.concat([fW_i, fW_f, fW_c, fW_o], axis=1), perm=[2, 0, 1]), shape=[input_size, -1]) # (1, 256*4, 512) -> (1, 1024, 512) -> (512, 1, 1024) -> (512, 1024)
        forward_recurrent_kernel = tf.reshape(tf.transpose(tf.concat([fR_i, fR_f, fR_c, fR_o], axis=1), perm=[2, 0, 1]), shape=[hidden_size, -1]) # (1, 256*4, 256) -> (256, 1, 1024) -> (256, 1024)
        # forward
        forward_lstm = CustomLSTM(
            hidden_size=hidden_size, # 256
            kernel=forward_kernel, # (512, 1024)
            recurrent_kernel=forward_recurrent_kernel, # (256, 1024)
            activation_alphas=tf_activation_alphas, # [1.0, 1.0, 1.0]
            activation_betas=tf_activation_betas, # [0.0, 0.0, 0.0]
            activations=tf_activations, # [tf.sigmoid, tf.tanh, tf.tanh]
            bias_i=fB_i, # (1, 256)
            bias_f=fB_f, # (1, 256)
            bias_c=fB_c, # (1, 256)
            bias_o=fB_o, # (1, 256)
            peephole_i=fP_i, # (1, 256)
            peephole_f=fP_f, # (1, 256)
            peephole_o=fP_o, # (1, 256)
            clip=clip,
            input_forget=input_forget,
            is_bidirectional=False,
            go_backwards=False,
            enable_rnn_unroll=enable_rnn_unroll,
        )
        output, hidden_state, cell_state = forward_lstm(X, initial_state=forward_initial_state)
        output = tf.expand_dims(output, axis=1)
        hidden_state = tf.expand_dims(hidden_state, axis=0)
        cell_state = tf.expand_dims(cell_state, axis=0)

    elif direction == 'reverse':
        reverse_weight = tf.reshape(tf.convert_to_tensor(W[0]), shape=[4, hidden_size, input_size]) # TensorShape([4, 256, 512])
        reverse_recurrence_weight = tf.reshape(tf.convert_to_tensor(R[0]), shape=[4, hidden_size, hidden_size]) # TensorShape([4, 256, 256])
        reverse_bias_W = tf.reshape(tf.convert_to_tensor(B[0][:4*hidden_size]), shape=[4, hidden_size]) # TensorShape([4, 256])
        reverse_bias_R = tf.reshape(tf.convert_to_tensor(B[0][4*hidden_size:4*hidden_size*2]), shape=[4, hidden_size]) # TensorShape([4, 256])
        reverse_bias = reverse_bias_W + reverse_bias_R
        reverse_peepholes = None
        if P is not None:
            reverse_peepholes = tf.reshape(tf.convert_to_tensor(P[0]), shape=[3, hidden_size])
        rW_i, rW_o, rW_f, rW_c = tf.split(value=reverse_weight, num_or_size_splits=4, axis=0)
        rR_i, rR_o, rR_f, rR_c = tf.split(value=reverse_recurrence_weight, num_or_size_splits=4, axis=0)
        rB_i, rB_o, rB_f, rB_c = tf.split(value=reverse_bias, num_or_size_splits=4, axis=0)
        rP_i = None
        rP_o = None
        rP_f = None
        if reverse_peepholes is not None:
            rP_i, rP_o, rP_f = tf.split(value=reverse_peepholes, num_or_size_splits=3, axis=0) # (1, 256)
        reverse_kernel = tf.reshape(tf.transpose(tf.concat([rW_i, rW_f, rW_c, rW_o], axis=1), perm=[2, 0, 1]), shape=[input_size, -1]) # (1, 256*4, 512) -> (1, 1024, 512) -> (512, 1, 1024) -> (512, 1024)
        reverse_recurrent_kernel = tf.reshape(tf.transpose(tf.concat([rR_i, rR_f, rR_c, rR_o], axis=1), perm=[2, 0, 1]), shape=[hidden_size, -1]) # (1, 256*4, 256) -> (256, 1, 1024) -> (256, 1024)
        # backward
        reverse_lstm = CustomLSTM(
            hidden_size=hidden_size, # 256
            kernel=reverse_kernel, # (512, 1024)
            recurrent_kernel=reverse_recurrent_kernel, # (256, 1024)
            activation_alphas=tf_activation_alphas, # [1.0, 1.0, 1.0]
            activation_betas=tf_activation_betas, # [0.0, 0.0, 0.0]
            activations=tf_activations, # [tf.sigmoid, tf.tanh, tf.tanh]
            bias_i=rB_i, # (1, 256)
            bias_f=rB_f, # (1, 256)
            bias_c=rB_c, # (1, 256)
            bias_o=rB_o, # (1, 256)
            peephole_i=rP_i, # (1, 256)
            peephole_f=rP_f, # (1, 256)
            peephole_o=rP_o, # (1, 256)
            clip=clip,
            input_forget=input_forget,
            is_bidirectional=False,
            go_backwards=True,
            enable_rnn_unroll=enable_rnn_unroll,
        )
        output, hidden_state, cell_state = reverse_lstm(X, initial_state=backward_initial_state)
        output = tf.reverse(output, axis=[1])
        output = tf.expand_dims(output, axis=1)
        hidden_state = tf.expand_dims(hidden_state, axis=0)
        cell_state = tf.expand_dims(cell_state, axis=0)

    elif direction == 'bidirectional':
        forward_weight = tf.reshape(tf.convert_to_tensor(W[0]), shape=[4, hidden_size, input_size]) # TensorShape([4, 256, 512])
        forward_recurrence_weight = tf.reshape(tf.convert_to_tensor(R[0]), shape=[4, hidden_size, hidden_size]) # TensorShape([4, 256, 256])
        forward_bias_W = tf.reshape(tf.convert_to_tensor(B[0][:4*hidden_size]), shape=[4, hidden_size]) # TensorShape([4, 256])
        forward_bias_R = tf.reshape(tf.convert_to_tensor(B[0][4*hidden_size:4*hidden_size*2]), shape=[4, hidden_size]) # TensorShape([4, 256])
        forward_bias = forward_bias_W + forward_bias_R
        forward_peepholes = None
        if P is not None:
            forward_peepholes = tf.reshape(tf.convert_to_tensor(P[0]), shape=[3, hidden_size])
        fW_i, fW_o, fW_f, fW_c = tf.split(value=forward_weight, num_or_size_splits=4, axis=0) # (1, 256, 512)
        fR_i, fR_o, fR_f, fR_c = tf.split(value=forward_recurrence_weight, num_or_size_splits=4, axis=0) # (1, 256, 256)
        fB_i, fB_o, fB_f, fB_c = tf.split(value=forward_bias, num_or_size_splits=4, axis=0) # (1, 256)
        fP_i = None
        fP_o = None
        fP_f = None
        if forward_peepholes is not None:
            fP_i, fP_o, fP_f = tf.split(value=forward_peepholes, num_or_size_splits=3, axis=0) # (1, 256)
        forward_kernel = tf.reshape(tf.transpose(tf.concat([fW_i, fW_f, fW_c, fW_o], axis=1), perm=[2, 0, 1]), shape=[input_size, -1]) # (1, 256*4, 512) -> (1, 1024, 512) -> (512, 1, 1024) -> (512, 1024)
        forward_recurrent_kernel = tf.reshape(tf.transpose(tf.concat([fR_i, fR_f, fR_c, fR_o], axis=1), perm=[2, 0, 1]), shape=[hidden_size, -1]) # (1, 256*4, 256) -> (256, 1, 1024) -> (256, 1024)

        reverse_weight = tf.reshape(tf.convert_to_tensor(W[1]), shape=[4, hidden_size, input_size])
        reverse_recurrence_weight = tf.reshape(tf.convert_to_tensor(R[1]), shape=[4, hidden_size, hidden_size])
        reverse_bias_W = tf.reshape(tf.convert_to_tensor(B[1][:4*hidden_size]), shape=[4, hidden_size]) # TensorShape([4, 256])
        reverse_bias_R = tf.reshape(tf.convert_to_tensor(B[1][4*hidden_size:4*hidden_size*2]), shape=[4, hidden_size]) # TensorShape([4, 256])
        reverse_bias = reverse_bias_W + reverse_bias_R
        reverse_peepholes = None
        if P is not None:
            reverse_peepholes = tf.reshape(tf.convert_to_tensor(P[1]), shape=[3, hidden_size])
        rW_i, rW_o, rW_f, rW_c = tf.split(value=reverse_weight, num_or_size_splits=4, axis=0)
        rR_i, rR_o, rR_f, rR_c = tf.split(value=reverse_recurrence_weight, num_or_size_splits=4, axis=0)
        rB_i, rB_o, rB_f, rB_c = tf.split(value=reverse_bias, num_or_size_splits=4, axis=0)
        rP_i = None
        rP_o = None
        rP_f = None
        if reverse_peepholes is not None:
            rP_i, rP_o, rP_f = tf.split(value=reverse_peepholes, num_or_size_splits=3, axis=0) # (1, 256)
        reverse_kernel = tf.reshape(tf.transpose(tf.concat([rW_i, rW_f, rW_c, rW_o], axis=1), perm=[2, 0, 1]), shape=[input_size, -1]) # (1, 256*4, 512) -> (1, 1024, 512) -> (512, 1, 1024) -> (512, 1024)
        reverse_recurrent_kernel = tf.reshape(tf.transpose(tf.concat([rR_i, rR_f, rR_c, rR_o], axis=1), perm=[2, 0, 1]), shape=[hidden_size, -1]) # (1, 256*4, 256) -> (256, 1, 1024) -> (256, 1024)

        # forward
        forward_lstm = CustomLSTM(
            hidden_size=hidden_size, # 256
            kernel=forward_kernel, # (512, 1024)
            recurrent_kernel=forward_recurrent_kernel, # (256, 1024)
            activation_alphas=tf_activation_alphas, # [1.0, 1.0, 1.0]
            activation_betas=tf_activation_betas, # [0.0, 0.0, 0.0]
            activations=tf_activations, # [tf.sigmoid, tf.tanh, tf.tanh]
            bias_i=fB_i, # (1, 256)
            bias_f=fB_f, # (1, 256)
            bias_c=fB_c, # (1, 256)
            bias_o=fB_o, # (1, 256)
            peephole_i=fP_i, # (1, 256)
            peephole_f=fP_f, # (1, 256)
            peephole_o=fP_o, # (1, 256)
            clip=clip,
            input_forget=input_forget,
            is_bidirectional=True,
            go_backwards=False,
            enable_rnn_unroll=enable_rnn_unroll,
        )

        # backward
        reverse_lstm = CustomLSTM(
            hidden_size=hidden_size, # 256
            kernel=reverse_kernel, # (512, 1024)
            recurrent_kernel=reverse_recurrent_kernel, # (256, 1024)
            activation_alphas=tf_activation_alphas, # [1.0, 1.0, 1.0]
            activation_betas=tf_activation_betas, # [0.0, 0.0, 0.0]
            activations=tf_activations, # [tf.sigmoid, tf.tanh, tf.tanh]
            bias_i=rB_i, # (1, 256)
            bias_f=rB_f, # (1, 256)
            bias_c=rB_c, # (1, 256)
            bias_o=rB_o, # (1, 256)
            peephole_i=rP_i, # (1, 256)
            peephole_f=rP_f, # (1, 256)
            peephole_o=rP_o, # (1, 256)
            clip=clip,
            input_forget=input_forget,
            is_bidirectional=True,
            go_backwards=True,
            enable_rnn_unroll=enable_rnn_unroll,
        )
        forward_output, forward_h, forward_c = \
            forward_lstm(X, initial_state=forward_initial_state) # [1, 24, 512], [[1, 256], [1, 256]] -> [1, 24, 256], [1, 256], [1, 256]
        reverse_output, reverse_h, reverse_c = \
            reverse_lstm(X, initial_state=backward_initial_state) # [1, 24, 512], [[1, 256], [1, 256]] -> [1, 24, 256], [1, 256], [1, 256]
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
                    'direction': direction,
                    'X': X,
                    'W': W,
                    'R': R,
                    'B': B,
                    'hidden_size': hidden_size,
                    'sequence_lens': sequence_lens,
                    'initial_h': initial_h,
                    'initial_c': initial_c,
                    'P': P,
                    'activations': tf_activations,
                    'activation_alpha': tf_activation_alphas,
                    'activation_beta': tf_activation_betas,
                    'clip': clip,
                    'input_forget': input_forget,
                    'layout': layout,
                },
                'tf_outputs': tf_outputs,
            }
        )
