import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    convert_axis,
    convert_reverse_axis,
    print_node_info,
    inverted_operation_enable_disable,
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
    """Reshape

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
    tensor_rank = len(input_tensor.shape)

    reshape_shape = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    if reshape_shape is not None and reshape_shape.shape is None:
        reshape_shape = []
    if reshape_shape is None:
        reshape_shape = []

    allowzero = bool(graph_node.attrs.get('allowzero', 0))
    if allowzero:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'TensorFlow does not support allowzero=1 (True).'
        print(error_msg)
        assert not allowzero, error_msg

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    # NWC->NCW, NHWC->NCHW, NDHWC->NCDHW Transpose
    perm = [
        convert_axis(
            axis=idx,
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        ) for idx in range(tensor_rank)
    ]
    transposed_tensor = tf.transpose(
        a=input_tensor,
        perm=list(perm) if perm is not None else None,
    )
    if isinstance(reshape_shape, np.ndarray):
        perm_shape = [
            convert_axis(
                axis=idx,
                tensor_rank=len(reshape_shape),
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in range(len(reshape_shape))
        ]
        transposed_reshape_shape = list(reshape_shape)
        if before_op_output_shape_trans:
            transposed_reshape_shape = [
                transposed_reshape_shape[perm_shape_dim] for perm_shape_dim in list(perm_shape)
            ]
    else:
        transposed_reshape_shape = reshape_shape

    # Reshape
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.reshape(
            tensor=transposed_tensor,
            shape=transposed_reshape_shape,
            name=graph_node.name,
        )
