import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
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
    """Unsqueeze

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

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = None
    if len(graph_node.inputs) >= 2:
        graph_node_input_2 = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_tensor_shape = list(input_tensor.shape)
    tensor_rank = len(input_tensor_shape)

    axes = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if axes is not None and axes.shape is None:
        axes = None

    axes = graph_node.attrs.get('axes', axes)

    if isinstance(axes, list) or (isinstance(axes, np.ndarray) and len(axes.shape) > 0):
        axes = [
            convert_axis(
                axis=idx,
                tensor_rank=tensor_rank+len(axes),
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in axes
        ]
    elif axes is not None and isinstance(axes, np.ndarray) and len(axes.shape) == 0:
        axes = convert_axis(
            axis=axes,
            tensor_rank=tensor_rank+1,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )
        axes = list(axes[np.newaxis])

    if axes is not None and isinstance(axes, list) and len(axes) > 0:
        axes.sort()

    new_shape = copy.deepcopy(input_tensor_shape)
    # TODO: Dynamic Tensor
    for idx in axes:
        new_shape.insert(idx, 1)

    new_shape = [dim if dim is not None else -1 for dim in new_shape]

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#unsqueeze-13
    """
    [2,3,4,5,6,7]
    test pattern.1 : axes=0, [1,2,3,4,5,6,7]
    test pattern.2 : axes=1, [2,1,3,4,5,6,7]
    test pattern.3 : axes=5, [2,3,4,5,6,1,7]
    test pattern.4 : axes=6, [2,3,4,5,6,7,1]
    test pattern.5 : axes=[0,1], [1,1,2,3,4,5,6,7]
    test pattern.6 : axes=[1,4], [2,1,3,4,1,5,6,7]
    test pattern.7 : axes=[6,7], [2,3,4,5,6,7,1,1]
    test pattern.8 : axes=[3,6], [2,3,4,1,5,6,1,7]
    test pattern.9 : axes=[3,-1], [2,3,4,1,5,6,1,7]
    """
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.reshape(
            tensor=input_tensor,
            shape=new_shape,
            name=graph_node.name,
        )
