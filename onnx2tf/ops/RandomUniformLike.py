import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
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
    """RandomUniformLike

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    rdtype = graph_node.attrs.get('dtype', 1)
    rhigh = graph_node.attrs.get('high', 1.0)
    rlow = graph_node.attrs.get('low', 0.0)
    rseed = graph_node.attrs.get('seed', 0)
    rshape = graph_node_input.shape \
        if input_tensor is None else input_tensor.shape
    rshape = [
        s if not isinstance(s, str) else None for s in rshape
    ]

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    generator = tf.random.Generator.from_seed(rseed)
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        generator.uniform(
            shape=rshape,
            minval=rlow,
            maxval=rhigh,
            dtype=ONNX_DTYPES_TO_TF_DTYPES[rdtype],
            name=graph_node.name,
        )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.random.uniform,
                'tf_inputs': {
                    'shape': shape,
                    'minval': rlow,
                    'maxval': rhigh,
                    'dtype': ONNX_DTYPES_TO_TF_DTYPES[rdtype],
                    'seed': rseed,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
