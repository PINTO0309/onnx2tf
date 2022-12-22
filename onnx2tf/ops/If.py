import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
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
    """If
    
    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node
    
    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    # Determines type of transposition
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1
    
    # Transposition of input values and conversion process to Numpy.ndarray
    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    
    # Type annotation for debugging efficiency
    input_tensor_1 = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    
    # Preserving Graph Structure (Dict) section
    graph_node_output: gs.Variable = graph_node.outputs[0]
    
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }
    
    # Define the condition
    cond = tf.constant(True)
    
    # Generation of TF OP section
    # TODO: Add the logic to generate the TF OP. Note: ONNX If is equivalent to TF cond.
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.cond(cond, 
                lambda: input_tensor_1, 
                lambda: tf.constant(False), 
                name=graph_node.name
        )
    
    # 5. Generation of Debug Info section
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
    make_tf_node_info(
        node_info={
            'tf_op_type': tf.cond,
            'tf_inputs': {
                'x': input_tensor_1,
            },
            'tf_outputs': {
                'output': tf_layers_dict[graph_node_output.name]['tf_node'],
            },
        }
    )