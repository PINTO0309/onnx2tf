import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import importlib
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)
from onnx2tf.utils.colors import Color


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """FusedConv

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_output: gs.Variable = graph_node.outputs[0]
    op = importlib.import_module(f'onnx2tf.ops.Conv')
    op.make_node(
        graph_node=graph_node,
        tf_layers_dict=tf_layers_dict,
        **kwargs,
    )
    conv_op = tf_layers_dict[graph_node_output.name]['tf_node']

    activation = graph_node.attrs.get('activation', 'Relu')
    activation_params = graph_node.attrs.get('activation_params', [])

    # Generation of TF OP
    # FusedConv activations: https://zenn.dev/pinto0309/scraps/1ac42ec0518d3b
    # 1. Relu
    # 2. Tanh
    # 3. Sigmoid
    # 4. LeakyRelu
    # 5. Clip
    # 6. HardSigmoid
    conv_act_op = None
    if activation == 'Relu':
        conv_act_op = tf.nn.relu(
            features=conv_op,
        )
    elif activation == 'Tanh':
        conv_act_op = tf.nn.tanh(
            x=conv_op,
        )
    elif activation == 'Sigmoid':
        conv_act_op = tf.nn.sigmoid(
            x=conv_op,
        )
    elif activation == 'LeakyRelu':
        conv_act_op = tf.nn.leaky_relu(
            features=conv_op,
            alpha=activation_params[0] \
                if (isinstance(activation_params, list) or isinstance(activation_params, np.ndarray)) \
                    and len(activation_params) > 0 \
                else activation_params,
        )
    elif activation == 'Clip':
        min_value = activation_params[0]
        max_value = activation_params[1]
        if (isinstance(min_value, np.ndarray) or isinstance(min_value, float)) and min_value == 0.0 \
            and (isinstance(max_value, np.ndarray)  or isinstance(max_value, float)) and max_value == 6.0:
            conv_act_op = tf.nn.relu6(features=conv_op)
        elif (isinstance(min_value, np.ndarray) or isinstance(min_value, float)) and min_value == 0.0 \
            and (max_value is None or max_value.shape is None):
            conv_act_op = tf.nn.relu(features=conv_op)
        else:
            if (isinstance(min_value, np.ndarray) and min_value.shape is not None) \
                and (isinstance(max_value, np.ndarray) and max_value.shape is not None):
                conv_act_op = \
                    tf.clip_by_value(
                        t=conv_op,
                        clip_value_min=min_value,
                        clip_value_max=max_value,
                    )
            elif (isinstance(min_value, np.ndarray) and min_value.shape is not None) \
                and (max_value is None or max_value.shape is None):
                conv_act_op = \
                    tf.maximum(
                        x=conv_op,
                        y=min_value,
                    )
            elif (min_value is None or min_value.shape is None) \
                and (max_value is not None and max_value.shape is not None):
                conv_act_op = \
                    tf.minimum(
                        x=conv_op,
                        y=max_value,
                    )
    elif activation == 'HardSigmoid':
        alpha = activation_params[0]
        beta = activation_params[1]
        conv_act_op = tf.maximum(0.0, tf.minimum(1.0, alpha * conv_op + beta))
    else:
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'FusedConv {activation} is not yet supported. ' +
            f'graph_node.name: {graph_node.name}'
        )
        sys.exit(1)

    tf_layers_dict[graph_node_output.name]['tf_node'] = conv_act_op

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': f'FusedConv_{activation}',
                'tf_inputs': {
                    'input': conv_op,
                    'activation': activation,
                    'activation_params': activation_params,
                },
                'tf_outputs': {
                    'output': conv_act_op,
                },
            }
        )
