import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """BatchNormalization

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    # Inputs
    X: gs.Variable = graph_node.inputs[0]
    scale: gs.Constant = graph_node.inputs[1]
    B: gs.Constant = graph_node.inputs[2]
    input_mean: gs.Constant = graph_node.inputs[3]
    input_var: gs.Constant = graph_node.inputs[4]
    # Outputs
    Y: gs.Variable = graph_node.outputs[0]
    # running_mean: gs.Variable = graph_node.outputs[1] # disuse
    # running_var: gs.Variable = graph_node.outputs[2] # disuse

    if hasattr(scale, 'inputs') \
        and len(scale.inputs) > 0 \
        and hasattr(scale.inputs[0], 'attrs') \
        and 'value' in scale.inputs[0].attrs \
        and hasattr(scale.inputs[0].attrs['value'], 'values'):
        scale = scale.inputs[0].attrs['value']

    if hasattr(B, 'inputs') \
        and len(B.inputs) > 0 \
        and hasattr(B.inputs[0], 'attrs') \
        and 'value' in B.inputs[0].attrs \
        and hasattr(B.inputs[0].attrs['value'], 'values'):
        B = B.inputs[0].attrs['value']

    if hasattr(input_mean, 'inputs') \
        and len(input_mean.inputs) > 0 \
        and hasattr(input_mean.inputs[0], 'attrs') \
        and 'value' in input_mean.inputs[0].attrs \
        and hasattr(input_mean.inputs[0].attrs['value'], 'values'):
        input_mean = input_mean.inputs[0].attrs['value']

    if hasattr(input_var, 'inputs') \
        and len(input_var.inputs) > 0 \
        and hasattr(input_var.inputs[0], 'attrs') \
        and 'value' in input_var.inputs[0].attrs \
        and hasattr(input_var.inputs[0].attrs['value'], 'values'):
        input_var = input_var.inputs[0].attrs['value']

    shape = Y.shape
    dtype = Y.dtype

    epsilon = graph_node.attrs.get('epsilon', 1e-05)
    momentum = graph_node.attrs.get('momentum', 0.9)
    training_mode = bool(graph_node.attrs.get('training_mode', 0)) # disuse

    # Preserving Graph Structure (Dict)
    tf_layers_dict[Y.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    mul_values = 1.0 / np.sqrt(input_var.values + epsilon) * scale.values

    if np.count_nonzero(input_mean.values) > 0:
        tf_layers_dict[Y.name]['tf_node'] = \
            (tf_layers_dict[X.name]['tf_node'] - input_mean.values) * mul_values + B.values
    else:
        tf_layers_dict[Y.name]['tf_node'] = \
            tf_layers_dict[X.name]['tf_node'] * mul_values + B.values

    # Generation of Debug Info
    tf_layers_dict[Y.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'BatchNormalization',
                'tf_inputs': {
                    'X': tf_layers_dict[X.name]['tf_node'],
                    'B': B,
                    'mean': input_mean,
                    'var': input_var,
                    'scale': scale,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[Y.name]['tf_node'],
                },
            }
        )
