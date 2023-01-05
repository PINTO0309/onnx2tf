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
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """DepthToSpace

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
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape

    blocksize = graph_node.attrs.get('blocksize', 1)
    mode = graph_node.attrs.get('mode', 'DCR')

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    if mode == "DCR":
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.nn.depth_to_space(
                input=input_tensor,
                block_size=blocksize,
                name=graph_node.name,
            )

    elif mode == "CRD":
        batch, channel = input_tensor_shape[0], input_tensor_shape[-1]
        height, width = input_tensor_shape[1], input_tensor_shape[2]
        csize = channel // (blocksize**2)

        reshape_node = tf.reshape(
            tensor=input_tensor,
            shape=[batch, height, width, csize, blocksize, blocksize]
        )
        transpose_node = transpose_with_flexing_deterrence(
            input_tensor=reshape_node,
            perm=[0,1,4,2,5,3],
            **kwargs,
        )
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.reshape(
                tensor=transpose_node,
                shape=[batch, height * blocksize, width * blocksize, csize],
                name=graph_node.name,
        )

    # Post-process transpose
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.nn.depth_to_space,
                'tf_inputs': {
                    'input': input_tensor,
                    'block_size': blocksize,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
