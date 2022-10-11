import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    convert_axis,
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
    """ConcatFromSequence

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    graph_node_input: gs.Variable = graph_node.inputs[0]
    graph_node_output: gs.Variable = graph_node.outputs[0]

    input_sequence = tf_layers_dict[graph_node_input.name]['tf_node']
    output_tensor = tf.sparse.to_dense(input_sequence.to_sparse())
    i_min = 0
    i_max = tf.shape(output_tensor)[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    axis = graph_node.attrs.get('axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(shape),
    )
    new_axis = graph_node.attrs.get('new_axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    new_axis = convert_axis(
        axis=new_axis,
        tensor_rank=len(shape),
        before_op_output_shape_trans=True,
    )

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # https://github.com/onnx/onnx-tensorflow/blob/main/onnx_tf/handlers/backend/concat_from_sequence.py
    cond_less = lambda i1, i2, i3, axis, o1: tf.less(i1, i2)
    body_concat = lambda i1, i2, i3, axis, o1: [
        i1 + 1, i2, i3, axis,
        tf.concat([o1, tf.gather(i3, i1)], axis=axis)
    ]

    # initialize with the first element
    t = tf.gather(output_tensor, 0)

    # setup inputs for the while loop
    input_tensor = tf.gather(output_tensor, tf.range(1, i_max))
    i_max = i_max - 1

    # loop through the rest of elements
    _, _, _, _, output_tensor = tf.while_loop(
        cond_less,
        body_concat, [i_min, i_max, input_tensor, axis, t],
        shape_invariants=[
            tf.TensorShape(None),
            i_max.get_shape(),
            input_tensor.get_shape(),
            tf.TensorShape(None),
            tf.TensorShape(None)
        ],
        parallel_iterations=1,
    )

    tf_layers_dict[graph_node_output.name]['tf_node'] = output_tensor

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'ConcatFromSequence',
                'tf_inputs': {
                    'input_sequence': input_sequence,
                    'axis': axis,
                    'new_axis': new_axis,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
