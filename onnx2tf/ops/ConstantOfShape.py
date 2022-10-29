import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from onnx import numpy_helper
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    print_node_info,
    make_tf_node_info,
)


@print_node_info
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ConstantOfShape

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input: gs.Variable = graph_node.inputs[0]
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # https://github.com/onnx/onnx-tensorflow/blob/main/onnx_tf/handlers/backend/constant_of_shape.py
    shape = tf_layers_dict[graph_node_input.name]['tf_node']

    # make sure the shape dtype is either int32 or int64
    if shape.dtype not in [tf.int64, tf.int32]:
        shape = tf.cast(shape, tf.int64)

    # the default value is 0, float32
    constant_tensor = None
    if "value" in graph_node.attrs:
        attr_value = graph_node.attrs['value']
        value = attr_value.values
        constant_tensor = value[0]
    else:
        constant_tensor = 0.

    cons = tf.fill(
        dims=shape,
        value=constant_tensor,
        name=graph_node.name,
    )

    tf_layers_dict[graph_node_output.name]['tf_node'] = cons

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.constant,
                'tf_inputs': {
                    'value': constant_tensor,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
