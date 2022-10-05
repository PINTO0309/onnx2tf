import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    convert_axis,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
)


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ArgMin

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
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get('axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(graph_node_input.shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # 0: False, 1: True
    keepdims = bool(graph_node.attrs.get('keepdims', 0))

    # 0: False, 1: True
    select_last_index = bool(graph_node.attrs.get('select_last_index', 0))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    reversed_tensor = None
    if not select_last_index:
        reversed_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
            if isinstance(graph_node_input, gs.Variable) else graph_node_input
    else:
        reversed_tensor = \
            tf.reverse(
                tensor=tf_layers_dict[graph_node_input.name]['tf_node'] \
                    if isinstance(graph_node_input, gs.Variable) else graph_node_input,
                axis=axis,
                name=f'{graph_node.name}_reverse',
            )

    final_tensor = None
    argmined_tensor = tf.math.argmin(
        input=reversed_tensor,
        axis=axis,
        output_type=dtype,
        name=f'{graph_node.name}_argmin',
    )
    if keepdims:
        final_tensor = \
            tf.expand_dims(
                input=argmined_tensor,
                axis=axis,
                name=f'{graph_node.name}_expand_dims',
            )
    else:
        final_tensor = argmined_tensor

    tf_layers_dict[graph_node_output.name]['tf_node'] = final_tensor
