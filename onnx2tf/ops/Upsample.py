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
    """Upsample

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

    input_tensor = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    scales = None
    if len(graph_node.inputs) >= 2:
        scales = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    else:
        scales = get_constant_or_variable(
            graph_node.attrs.get('scales', scales),
            before_op_output_shape_trans,
        )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape
    scales = tf_layers_dict[scales.name]['tf_node'] \
        if isinstance(scales, gs.Variable) else scales

    mode = graph_node.attrs.get('mode', 'nearest')

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[input_tensor.name]['nhwc'] \
            if isinstance(input_tensor, gs.Variable) \
                and 'nhwc' in tf_layers_dict[input_tensor.name].keys() else False
    }

    # Generation of TF OP
    new_size = None
    if hasattr(graph_node.outputs[0], 'shape') \
        and graph_node.outputs[0].shape is not None \
        and isinstance(graph_node.outputs[0].shape[-2], int) \
        and isinstance(graph_node.outputs[0].shape[-1], int):
        new_size = graph_node.outputs[0].shape[-2:len(graph_node.outputs[0].shape)] # Estimated from ONNX output shape
    else:
        h_w_scale = scales[1:3]
        h_w_shape = input_tensor_shape[1:3]
        new_size = tf.cast(h_w_scale * tf.cast(h_w_shape, scales.dtype), tf.int32)

    if hasattr(new_size, 'set_shape'):
        new_size.set_shape([2])

    if hasattr(new_size, '_inferred_value'):
        new_size_values = new_size._inferred_value
        if new_size_values.count(None) == len(new_size_values):
            tensor_rank = len(graph_node_output.shape)
            convertion_table = [0] + [i for i in range(2, tensor_rank)] + [1]
            new_values = [0] * tensor_rank
            for new_idx, idx in enumerate(convertion_table):
                new_values[new_idx] = graph_node_output.shape[idx]
            new_size = new_values[-3:-1]

    resized_tensor = None
    tf_op_type = None
    if mode.lower() == "bilinear" or mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
    else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR

    resized_tensor = tf.image.resize(
        images=input_tensor,
        size=new_size,
        method=mode,
        name=graph_node.name,
    )
    tf_op_type = tf.image.resize

    tf_layers_dict[graph_node_output.name]['tf_node'] = resized_tensor

    # Post-process transpose
    before_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    tf_layers_dict[graph_node_output.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[graph_node_output.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )
    after_trans_shape = tf_layers_dict[graph_node_output.name]['tf_node'].shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'images': input_tensor,
                    'new_size/crop_size': new_size,
                    'method': mode,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
