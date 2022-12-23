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
    """ReduceMin

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

    tensor_rank = len(graph_node_input_1.shape)

    axes = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if axes is not None and axes.shape is None:
        axes = None

    axes = graph_node.attrs.get('axes', axes)
    noop_with_empty_axes = bool(graph_node.attrs.get('noop_with_empty_axes', 0))
    if noop_with_empty_axes:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'TensorFlow does not support noop_with_empty_axes=1 (True).'
        print(error_msg)
        assert not noop_with_empty_axes, error_msg

    # NCHW->NHWC, NCDHW->NDHWC
    if isinstance(axes, list) or (isinstance(axes, np.ndarray) and len(axes.shape) > 0):
        axes = [
            convert_axis(
                axis=idx,
                tensor_rank=tensor_rank,
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in axes
        ]
    elif axes is not None and isinstance(axes, np.ndarray) and len(axes.shape) == 0:
        axes = convert_axis(
            axis=axes,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )

    # 0: False, 1: True
    keepdims = bool(graph_node.attrs.get('keepdims', 1))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    reducemined_tensor = input_tensor
    reducemined_tensor = tf.math.reduce_min(
        input_tensor=reducemined_tensor,
        axis=axes,
        keepdims=keepdims,
        name=f'{graph_node.name}',
    )
    tf_layers_dict[graph_node_output.name]['tf_node'] = reducemined_tensor

    # Generation of Debug Info
    tf_inputs = {f"axis{idx}": value for idx, value in enumerate(axes)} if axes is not None else {"axis": None}
    tf_inputs['input_tensor'] = input_tensor
    tf_inputs['keepdims'] = keepdims

    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.reduce_min,
                'tf_inputs': tf_inputs,
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
