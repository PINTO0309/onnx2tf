import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
from functools import partial
import tensorflow as tf
from tensorflow.python.ops import array_ops
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    print_node_info,
    inverted_operation_enable_disable,
)
from onnx2tf.utils.colors import Color


class RNNMixin(object):
    ONNX_ACTIVATION_MAPPING = {
        # Added from tf 1.8
        # "affine": tf.contrib.distributions.bijectors.AffineScalar,
        # tf.contrib was removed since tf 2.0,
        # Class Affine had been move to the following module
        # "affine": tfp.bijectors.Affine,
        "elu": tf.nn.elu,
        "hard_sigmoid": tf.keras.backend.hard_sigmoid,
        "leaky_relu": tf.nn.leaky_relu,
        "relu": tf.nn.relu,
        "sigmoid": tf.sigmoid,
        "softsign": tf.nn.softsign,
        "softplus": tf.nn.softplus,
        "tanh": tf.tanh,
        "thresholded_relu": tf.keras.layers.ThresholdedReLU,
    }

    rnn_cell = None

    @classmethod
    def rnn(cls, x, cell_class, cell_kwargs, rnn_kwargs, activations, direction):
        cell_kwargs["activation"] = activations[0]

        if cls.rnn_cell is None:
            cls.rnn_cell = [cell_class(**cell_kwargs)]
        rnn_cell = cls.rnn_cell
        cell_fw = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_cell)

        if direction == "bidirectional":
            cell_kwargs["activation"] = activations[1]
            rnn_cell_bw = [cell_class(**cell_kwargs)]
            cell_bw = tf.compat.v1.nn.rnn_cell.MultiRNNCell(rnn_cell_bw)

        if direction == "forward":
            outputs, states = tf.compat.v1.nn.dynamic_rnn(cell_fw, x, **rnn_kwargs)
        elif direction == "bidirectional":
            outputs, states = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, **rnn_kwargs)
        elif direction == "reverse":

            def _reverse(input_, seq_dim):
                return array_ops.reverse(input_, axis=[seq_dim])

            time_dim = 0
            inputs_reverse = _reverse(x, time_dim)
            outputs, states = tf.compat.v1.nn.dynamic_rnn(cell_fw, inputs_reverse, **rnn_kwargs)
            outputs = _reverse(outputs, time_dim)

        return outputs, states

    @classmethod
    def rnn_get_activation(cls, name, alpha, beta):
        if name not in cls.ONNX_ACTIVATION_MAPPING:
            print(
                f'{Color.RED}ERROR:{Color.RESET} ' +
                f'Activation function {name} for {cls.__name__}'
            )
            sys.exit(1)
        activation = cls.ONNX_ACTIVATION_MAPPING[name]
        kwargs = {}
        if name == "affine":
            kwargs["scale"] = alpha
            kwargs["shift"] = beta
            activation = activation(**kwargs)
        elif name == "elu":
            if alpha != 1:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f'Activation function {name} with alpha={alpha} for {cls.__name__}'
                )
                sys.exit(1)
        elif name == "hard_sigmoid":
            if alpha != 0.2 or beta != 0.5:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f'Activation function {name} with alpha={alpha}, beta={beta} for {cls.__name__}'
                )
                sys.exit(1)
        elif name == "leaky_relu":
            kwargs["alpha"] = alpha or 0.01
            activation = partial(activation, **kwargs)
        elif name == "thresholded_relu":
            kwargs["theta"] = alpha
            activation = activation(**kwargs)
        return activation


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """LSTM

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    cls = RNNMixin()

    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_3 = \
        tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)

    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans,
    )

    graph_node_input_4 = None
    if len(graph_node.inputs) >= 4:
        graph_node_input_4 = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
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

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    X = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    X_shape = X.shape
    X_rank = len(X_shape)

    W = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    R = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    B = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4

    sequence_lens = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) and graph_node_input_5.name != '' else graph_node_input_5
    initial_h = tf_layers_dict[graph_node_input_6.name]['tf_node'] \
        if isinstance(graph_node_input_6, gs.Variable) and graph_node_input_6.name != ''  else graph_node_input_6
    initial_c = tf_layers_dict[graph_node_input_7.name]['tf_node'] \
        if isinstance(graph_node_input_7, gs.Variable) and graph_node_input_7.name != ''  else graph_node_input_7
    P = tf_layers_dict[graph_node_input_8.name]['tf_node'] \
        if isinstance(graph_node_input_8, gs.Variable) and graph_node_input_8.name != ''  else graph_node_input_8

    activation_alpha = graph_node.attrs.get('activation_alpha', [None] * 6)
    activation_beta = graph_node.attrs.get('activation_beta', [None] * 6)
    activations = graph_node.attrs.get('activations', None)
    clip = graph_node.attrs.get('clip', None)
    direction = graph_node.attrs.get('direction', 'forward')
    num_directions = 2 if direction == 'bidirectional' else 1
    hidden_size = graph_node.attrs.get('hidden_size', None)
    input_forget = graph_node.attrs.get('input_forget', 0)
    layout = graph_node.attrs.get('layout', 0)
    output_sequence = graph_node.attrs.get("output_sequence", 0)

    if layout == 1:
        X = tf.transpose(X, perm=[1, 0, 2])

    if len(X_shape) == 4 and X_shape[1] == 1:
        x = tf.squeeze(x)

    sequence_length = sequence_lens

    cell_kwargs = {}

    if clip is not None:
        cell_kwargs["cell_clip"] = clip

    tf_activations = [tf.nn.tanh] * num_directions

    if activations is not None:
        activations = list(map(lambda x: x.lower(), activations))
        activation_idxs = [1, 4] if num_directions == 2 else [1]

        tf_activations = [
            cls.rnn_get_activation(activations[i], activation_alpha[i], activation_beta[i]) for i in activation_idxs
        ]









    # # Preserving Graph Structure (Dict)
    # tf_layers_dict[graph_node_output.name] = {
    #     'optype': graph_node.op,
    #     'shape': shape,
    #     'dtype': dtype,
    # }

    # # Generation of TF OP
    # tf_layers_dict[graph_node_output.name]['tf_node'] = \
    #     tf.squeeze(
    #         input=input_tensor,
    #         axis=list(axes) if axes is not None else None,
    #         name=graph_node.name,
    #     )
