import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
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


class Col2ImLayer(tf_keras.layers.Layer):
    def __init__(self):
        super(Col2ImLayer, self).__init__()

    def call(
        self,
        input_tensor,
        input_block_shape,
        strides,
        input_image_shape,
        pads,
        dilations,
        output_shape = None,
    ):
        N, _, L = input_tensor.shape
        if output_shape is not None:
            C = tf.convert_to_tensor(output_shape[1])
            output_shape = tf.convert_to_tensor([output_shape[0]] + [s for s in output_shape[2:]] + [output_shape[1]])
        else:
            C = tf.convert_to_tensor(input_tensor.shape[1] // (input_block_shape[0] * input_block_shape[1]))
            output_shape = tf.convert_to_tensor([N] + input_image_shape + [C])
        im = tf.TensorArray(input_tensor.dtype, size=N, dynamic_size=True)

        def loop_over_l(n, im_n, l):
            row_idx = tf.convert_to_tensor((l // ((input_image_shape[1] - (input_block_shape[1] - 1) * dilations[1] - 1 + strides[1]) // strides[1])) * strides[0])
            col_idx = tf.convert_to_tensor((l % ((input_image_shape[1] - (input_block_shape[1] - 1) * dilations[1] - 1 + strides[1]) // strides[1])) * strides[1])

            def loop_over_c(c, im_n):
                patch_idx = tf.convert_to_tensor(c * input_block_shape[0] * input_block_shape[1])
                patch = tf.reshape(input_tensor[n, patch_idx:patch_idx + input_block_shape[0] * input_block_shape[1], l], input_block_shape)
                if dilations[0] > 1 or dilations[1] > 1:
                    dilated_patch = tf.zeros([(input_block_shape[0] - 1) * dilations[0] + 1, (input_block_shape[1] - 1) * dilations[1] + 1], dtype=patch.dtype)
                    for i in tf.range(input_block_shape[0]):
                        for j in tf.range(input_block_shape[1]):
                            dilated_patch = tf.tensor_scatter_nd_update(dilated_patch, [[i * dilations[0], j * dilations[1]]], [patch[i, j]])
                    patch = dilated_patch
                patch_shape = tf.shape(patch)
                indices = tf.reshape(tf.stack(tf.meshgrid(row_idx + tf.range(patch_shape[0]), col_idx + tf.range(patch_shape[1]), c, indexing='ij'), axis=-1), [-1, 3])
                updates = tf.reshape(patch, [-1])
                return tf.tensor_scatter_nd_add(im_n, indices, updates)
            return tf.reduce_sum(tf.map_fn(lambda c: loop_over_c(c, im_n), tf.range(C), dtype=im_n.dtype), axis=0)

        def loop_over_n(n):
            im_n = tf.zeros(output_shape[1:], dtype=input_tensor.dtype)
            return tf.reduce_sum(tf.map_fn(lambda l: loop_over_l(n, im_n, l), tf.range(L), dtype=im_n.dtype), axis=0)

        im = tf.map_fn(loop_over_n, tf.range(N), dtype=input_tensor.dtype)

        return im[:, pads[0]:input_image_shape[0]-pads[2], pads[1]:input_image_shape[1]-pads[3], :]


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Col2Im

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = before_op_output_shape_trans_1

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans=False,
    )
    graph_node_input_3 = get_constant_or_variable(
        graph_node.inputs[2],
        before_op_output_shape_trans=False,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    input_image_shape = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    input_block_shape = tf_layers_dict[graph_node_input_3.name]['tf_node'] \
        if isinstance(graph_node_input_3, gs.Variable) else graph_node_input_3

    spatial_size = len(input_image_shape)
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    pads = graph_node.attrs.get('pads', [0, 0] * spatial_size)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Pre-process transpose
    before_trans_shape = input_tensor.shape
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    after_trans_shape = input_tensor.shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    # Generation of TF OP
    col2im = Col2ImLayer()
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        col2im(
            input_tensor=input_tensor,
            input_block_shape=input_block_shape,
            strides=strides,
            input_image_shape=input_image_shape,
            pads=pads,
            dilations=dilations,
            output_shape=shape,
        )

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
                'tf_op_type': 'Col2Im',
                'tf_inputs': {
                    'input': input_tensor,
                    'image_shape': input_image_shape,
                    'block_shape': input_block_shape,
                    'dilations': dilations,
                    'pads': pads,
                    'strides': strides,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
