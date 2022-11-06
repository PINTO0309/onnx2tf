import sys
import uuid
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


from functools import partial

import tensorflow as tf
from tensorflow.python.ops import array_ops

def get_variable_name(
    node: gs.Node,
    var_name: str,
):
    """ Get variable name.
    :param node: ONNX NodeProto object
    :param var_name: name of the variable
    :return: unique variable name
    """
    v_name = node.op.lower() + '_' + var_name
    return v_name + '_' + node.name.lower() if node.name else v_name

def get_unique_suffix():
    """ Get unique suffix by using first 8 chars from uuid.uuid4
    to make unique identity name.
    :return: Unique suffix string.
    """
    return str(uuid.uuid4())[:8]

class RNNMixin(object):
    ONNX_ACTIVATION_MAPPING = {
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
            outputs, states = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, **rnn_kwargs)
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
            raise ValueError("Activation function {} for {}".format(name, cls.__name__))
        activation = cls.ONNX_ACTIVATION_MAPPING[name]
        kwargs = {}
        if name == "affine":
            kwargs["scale"] = alpha
            kwargs["shift"] = beta
            activation = activation(**kwargs)
        elif name == "elu":
            if alpha != 1:
                raise ValueError("Activation function {} with alpha={} for {}".format(name, alpha, cls.__name__))
        elif name == "hard_sigmoid":
            if alpha != 0.2 or beta != 0.5:
                raise ValueError("Activation function {} with alpha={}, beta={} for {}".format(name, alpha, beta, cls.__name__))
        elif name == "leaky_relu":
            kwargs["alpha"] = alpha or 0.01
            activation = partial(activation, **kwargs)
        elif name == "thresholded_relu":
            kwargs["theta"] = alpha
            activation = activation(**kwargs)
        return activation


class LSTM(RNNMixin):
    # declare variable names for custom getters
    weight_var_name = 'kernel'
    bias_var_name = 'bias'
    peephole_weight_forget_var_name = 'w_f_diag'
    peephole_weight_input_var_name = 'w_i_diag'
    peephole_weight_output_var_name = 'w_o_diag'

    @classmethod
    def get_req_vars_template(cls, node, init_dict):
        """ Get required variables template, which is a dictionary of
            variable names with initial value and shape
            :return: Dict.
        """
        b_shape = node.attrs["hidden_size"] * 4
        return {
            cls.weight_var_name: [
                tf.constant([[0.]], dtype=tf.float32),
                tf.TensorShape([None, None])
            ],
            cls.bias_var_name: [
                tf.constant([0.] * b_shape, dtype=tf.float32),
                tf.TensorShape([b_shape])
            ],
            cls.peephole_weight_forget_var_name: [
                tf.constant([[0.]], dtype=tf.float32),
                tf.TensorShape([None, None])
            ],
            cls.peephole_weight_input_var_name: [
                tf.constant([[0.]], dtype=tf.float32),
                tf.TensorShape([None, None])
            ],
            cls.peephole_weight_output_var_name: [
                tf.constant([[0.]], dtype=tf.float32),
                tf.TensorShape([None, None])
            ]
        }

    @classmethod
    def args_check(cls, node, **kwargs):
        direction = node.attrs.get("direction", "forward")
        num_directions = 2 if direction == "bidirectional" else 1
        if node.attrs.get("input_forget", 0):
            pass
        if "activations" in node.attrs:
            activations = list(map(lambda x: x.lower(), node.attrs["activations"]))
            if activations[0] != "sigmoid":
                raise ValueError("LSTM without sigmoid for `f`")
            if activations[1] != activations[2]:
                raise ValueError("LSTM without same activation for `g` and `h`")
            if num_directions == 2:
                if activations[3] != "sigmoid":
                    raise ValueError("LSTM without sigmoid for Tensorflow")
                if activations[4] != activations[5]:
                    raise ValueError("LSTM without same activation for `g` and `h`")

    @classmethod
    def _custom_getter(
        cls,
        getter,
        name,
        node=None,
        # tf_layers_dict=None,
        X=None,
        W=None,
        R=None,
        B=None,
        P=None,
        is_bidirectional=None,
        *args,
        **kwargs,
    ):
        names = name.split("/")
        if is_bidirectional:
            if "fw" in names:
                index = 0
            elif "bw" in names:
                index = 1
            else:
                raise RuntimeError("Can not get {} for bidirectional. Either fw and bw is not in name scope.".format(names[-1]))

        if names[-1] == "kernel":
            weight_variable = tensor_dict[get_variable_name(node, cls.weight_var_name)]
            # onnx W[iofc], R[iofc]
            if is_bidirectional:
                w = tf.split(W, 2)[index]
                r = tf.split(R, 2)[index]
            else:
                w = W
                r = R
            w_i, w_o, w_f, w_c = tf.split(tf.squeeze(w), 4)
            r_i, r_o, r_f, r_c = tf.split(tf.squeeze(r), 4)
            new_w = tf.transpose(tf.concat([w_i, w_c, w_f, w_o], 0))
            new_r = tf.transpose(tf.concat([r_i, r_c, r_f, r_o], 0))
            kernel = tf.concat([new_w, new_r], 0)
            weight_variable.assign(kernel)
            return weight_variable

        if names[-1] == "bias":
            bias_variable = tensor_dict[get_variable_name(node, cls.bias_var_name)]
            if len(node.inputs) >= 4:
                # onnx Wb[iofc], Rb[iofc]
                if is_bidirectional:
                    b = tf.split(B, 2)[index]
                else:
                    b = B
                w_b, r_b = tf.split(tf.squeeze(b), 2)
                w_b_i, w_b_o, w_b_f, w_b_c = tf.split(w_b, 4)
                r_b_i, r_b_o, r_b_f, r_b_c = tf.split(r_b, 4)
                w_b = tf.transpose(tf.concat([w_b_i, w_b_c, w_b_f, w_b_o], 0))
                r_b = tf.transpose(tf.concat([r_b_i, r_b_c, r_b_f, r_b_o], 0))
                bias_variable.assign(tf.add(w_b, r_b))

            return bias_variable

        # Only use_peepholes is True,
        # will try to get w_f_diag, w_i_diag, w_o_diag
        # onnx P[iof]
        if names[-1] in ["w_f_diag", "w_i_diag", "w_o_diag"]:
            if is_bidirectional:
                p = tf.split(P, 2)[index]
            else:
                p = P
            if names[-1] == "w_f_diag":
                w_f_variable = tensor_dict[get_variable_name(node, cls.peephole_weight_forget_var_name)]
                w_f_variable.assign(tf.split(p, 3, axis=1)[2])
                return w_f_variable
            if names[-1] == "w_i_diag":
                w_i_variable = tensor_dict[get_variable_name(node, cls.peephole_weight_input_var_name)]
                w_i_variable.assign(tf.split(p, 3, axis=1)[0])
                return w_i_variable
            if names[-1] == "w_o_diag":
                w_o_variable = tensor_dict[get_variable_name(node, cls.peephole_weight_output_var_name)]
                w_o_variable.assign(tf.split(p, 3, axis=1)[1])
                return w_o_variable
        return getter(name, *args, **kwargs)

    @classmethod
    def _common(
        cls,
        graph_node,
        X,
        W,
        R,
        B,
        sequence_length,
        initial_h,
        initial_c,
        P,
        activation_alpha,
        activation_beta,
        activations,
        clip,
        direction,
        num_directions,
        hidden_size,
        input_forget,
        layout,
        output_sequence,
        tf_layers_dict,
    ):
        x = X
        input_shape = x.shape
        input_size = len(graph_node.inputs)
        # Need transpose for batchwise
        if layout == 1:
            x = tf.transpose(x, perm=[1, 0, 2])
        if len(input_shape) == 4 and input_shape[1] == 1:
            x = tf.squeeze(x)
        cell_kwargs = {}
        cell_kwargs["cell_clip"] = clip

        tf_activations = [tf.nn.tanh] * num_directions
        if activations is not None:
            activation_idxs = [1, 4] if num_directions == 2 else [1]
            tf_activations = [
                    cls.rnn_get_activation(
                        activations[i],
                        activation_alpha[i],
                        activation_beta[i]
                    ) for i in activation_idxs
            ]

        with tf.compat.v1.variable_scope(
            "LSTM_" + get_unique_suffix(),
            custom_getter=partial(
                cls._custom_getter,
                node=graph_node,
                # tensor_dict=tf_layers_dict,
                X=X,
                W=W,
                R=R,
                B=B,
                P=P,
                is_bidirectional=num_directions == 2,
            )
        ):
            cell_kwargs["use_peepholes"] = P is not None
            cell_kwargs["forget_bias"] = 0.
            cell_kwargs["num_units"] = hidden_size
            initial_state = None
            initial_state_bw = None
            if input_size >= 6:
                if layout == 1:
                    initial_h = tf.transpose(initial_h, perm=[1, 0, 2])
                    initial_c = tf.transpose(initial_c, perm=[1, 0, 2])
                if initial_h is not None and initial_c is not None:
                    initial_state = (tf.compat.v1.nn.rnn_cell.LSTMStateTuple(initial_c[0], initial_h[0]),)
                    if num_directions == 2:
                        initial_state_bw = (tf.compat.v1.nn.rnn_cell.LSTMStateTuple(initial_c[1], initial_h[1]),)
                rnn_kwargs = {}
                if num_directions == 1:
                    rnn_kwargs["initial_state"] = initial_state
                elif num_directions == 2:
                    rnn_kwargs["initial_state_fw"] = initial_state
                    rnn_kwargs["initial_state_bw"] = initial_state_bw
                rnn_kwargs["sequence_length"] = sequence_length
                rnn_kwargs["time_major"] = True
                rnn_kwargs["dtype"] = tf.float32
                outputs, states = cls.rnn(
                    x,
                    tf.compat.v1.nn.rnn_cell.LSTMCell,
                    cell_kwargs,
                    rnn_kwargs,
                    tf_activations,
                    direction,
                )

        if num_directions == 1:
            state = states[0]
            c = tf.expand_dims(state[0], 0)
            h = tf.expand_dims(state[1], 0)
            output = tf.expand_dims(outputs, 1)
        else:
            state_fw = states[0][0]
            state_bw = states[1][0]
            output_fw = outputs[0]
            output_bw = outputs[1]
            c_fw = tf.expand_dims(state_fw[0], 0)
            c_bw = tf.expand_dims(state_bw[0], 0)
            c = tf.concat((c_fw, c_bw), axis=0)
            h_fw = tf.expand_dims(state_fw[1], 0)
            h_bw = tf.expand_dims(state_bw[1], 0)
            h = tf.concat((h_fw, h_bw), axis=0)
            output_fw = tf.expand_dims(output_fw, 1)
            output_bw = tf.expand_dims(output_bw, 1)
            output = tf.concat((output_fw, output_bw), axis=1)

        # Need transpose for batchwise
        if layout == 1:
            output = tf.transpose(output, perm=[2, 0, 1, 3])
            h = tf.transpose(h, perm=[1, 0, 2])
            c = tf.transpose(c, perm=[1, 0, 2])

        return [output, h, c] if output_sequence == 0 else [h, c]


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

    W = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    R = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3
    B = tf_layers_dict[graph_node_input_4.name]['tf_node'] \
        if isinstance(graph_node_input_4, gs.Variable) else graph_node_input_4

    sequence_length = tf_layers_dict[graph_node_input_5.name]['tf_node'] \
        if isinstance(graph_node_input_5, gs.Variable) and graph_node_input_5.name != '' else None
    initial_h = tf_layers_dict[graph_node_input_6.name]['tf_node'] \
        if isinstance(graph_node_input_6, gs.Variable) and graph_node_input_6.name != ''  else graph_node_input_6
    initial_c = tf_layers_dict[graph_node_input_7.name]['tf_node'] \
        if isinstance(graph_node_input_7, gs.Variable) and graph_node_input_7.name != ''  else graph_node_input_7
    initial_c = initial_c if initial_c is not None else tf.zeros_like(initial_h)
    P = tf_layers_dict[graph_node_input_8.name]['tf_node'] \
        if isinstance(graph_node_input_8, gs.Variable) and graph_node_input_8.name != ''  else graph_node_input_8

    activation_alpha = graph_node.attrs.get('activation_alpha', [None] * 6)
    activation_beta = graph_node.attrs.get('activation_beta', [None] * 6)
    activations = graph_node.attrs.get('activations', None)
    activations = list(map(lambda x: x.lower(), activations)) if activations is not None else None
    clip = graph_node.attrs.get('clip', None)
    direction = graph_node.attrs.get('direction', 'forward')
    num_directions = 2 if direction == 'bidirectional' else 1
    hidden_size = graph_node.attrs.get('hidden_size', None)
    input_forget = graph_node.attrs.get('input_forget', 0)
    layout = graph_node.attrs.get('layout', 0)
    output_sequence = graph_node.attrs.get("output_sequence", 0)

    lstm = LSTM()
    result = lstm._common(
        graph_node,
        X,
        W,
        R,
        B,
        sequence_length,
        initial_h,
        initial_c,
        P,
        activation_alpha,
        activation_beta,
        activations,
        clip,
        direction,
        num_directions,
        hidden_size,
        input_forget,
        layout,
        output_sequence,
        tf_layers_dict,
    )
    if len(result) == 2:
        pass # [h, c] if output_sequence > 0
    elif len(result) == 3:
        pass # [output, h, c] if output_sequence == 0






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
