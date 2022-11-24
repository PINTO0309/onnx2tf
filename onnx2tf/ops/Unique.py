import random
from typing import List

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
from onnx2tf.utils.colors import Color


class tfUnique(tf.keras.layers.Layer):

    def __init__(self):
        super(tfUnique, self).__init__()
        self.unique_ops = tf.raw_ops.UniqueWithCountsV2

    def call(self, x, axis):
        return self.unique_ops(x=x, axis=[axis], out_idx=tf.int64)


@print_node_info
@inverted_operation_enable_disable
def make_node(
        *,
        graph_node: gs.Node,
        tf_layers_dict: dict,
        **kwargs: dict,
):
    """Unique

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_outputs: List[gs.Variable] = [
        graph_node_output for graph_node_output in graph_node.outputs
    ]

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    axis = graph_node.attrs.get('axis', None)
    sorted = graph_node.attrs.get('sorted', 1)

    # Preserving Graph Structure (Dict)
    for graph_node_output in graph_node_outputs:
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': graph_node_output.shape,
            'dtype': graph_node_output.dtype,
        }

    # Generation of TF OP
    # tensorflow raw_ops does not support direct call to KerasTensor, need to call through keras layer
    tf_unique_ops = tfUnique()

    # flatten tensor if axis is not specified
    if axis is None:
        axis = 0
        input_tensor = tf.reshape(input_tensor, [-1])

    # CAUTION: tensorflow unique returns inverse indices only
    y, inverse_indices, count = tf_unique_ops(x=input_tensor, axis=axis)

    # use tf.unique again to get true unique indices
    rey, reidx = tf.unique(inverse_indices)
    num_segments = tf.shape(rey)[0]
    num_elems = tf.shape(inverse_indices)[0]
    indices = tf.math.unsorted_segment_min(tf.range(num_elems), reidx, num_segments)
    indices = tf.cast(indices, dtype=inverse_indices.dtype)

    # tf unique returns unsorted tensor, need to sort if option is enabled
    if sorted:
        # TODO: implement sort
        error_msg = f'' + \
                    f'{Color.RED}WARNING:{Color.RESET} ' + \
                    f'Sort option in Unique ops is not implemented yet.'
        print(error_msg)
        assert False, error_msg

    tf_layers_dict[graph_node_outputs[0].name]['tf_node'] = y
    tf_layers_dict[graph_node_outputs[1].name]['tf_node'] = indices
    tf_layers_dict[graph_node_outputs[2].name]['tf_node'] = inverse_indices
    tf_layers_dict[graph_node_outputs[3].name]['tf_node'] = count

    # Generation of Debug Info
    tf_outputs = {f"output{idx}": value for idx, value in enumerate([y, indices, inverse_indices, count])}
    tf_layers_dict[graph_node_outputs[0].name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.raw_ops.UniqueWithCountsV2,
                'tf_inputs': {
                    'value': input_tensor,
                    'axis': axis,
                    'sorted': sorted
                },
                'tf_outputs': tf_outputs,
            }
        )
