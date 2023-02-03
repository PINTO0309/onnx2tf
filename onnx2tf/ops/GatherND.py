import sys
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
    make_tf_node_info,
    process_neg_idx,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
from onnx2tf.utils.colors import Color
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """GatherND

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = \
        tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    indices_tensor = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    indices_tensor = pre_process_transpose(
        value_before_transpose=indices_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    tensor_rank = len(input_tensor.shape)

    batch_dims = graph_node.attrs.get('batch_dims', 0)
    batch_dims = convert_axis(
        axis=batch_dims,
        tensor_rank=tensor_rank,
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    replace_gathernd_to_pseudo_gathernd = "gathernd" in kwargs['replace_to_pseudo_operators']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    indices_tensor = process_neg_idx(
        data=input_tensor,
        indices=indices_tensor,
        batch_dims=batch_dims,
    )

    if not replace_gathernd_to_pseudo_gathernd:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.gather_nd(
                params=input_tensor,
                indices=indices_tensor,
                batch_dims=batch_dims,
                name=graph_node.name,
            )
    else:
        if batch_dims != 0:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'--replace_gathernd_to_pseudo_gathernd is supported only if batch_dims=0.'+
                f'graph_node.name: {graph_node.name}'
            )
            sys.exit(1)

        params_shape = input_tensor.shape
        idx_shape = indices_tensor.shape
        idx_dims = idx_shape[-1]
        gather_shape = params_shape[idx_dims:]
        params_flat = tf.reshape(
            input_tensor,
            tf.concat([[-1], gather_shape], axis=0),
        )
        axis_step = tf.math.cumprod(
            params_shape[:idx_dims],
            exclusive=True,
            reverse=True,
        )
        mul = tf.math.multiply(
            indices_tensor,
            tf.cast(
                axis_step,
                dtype= NUMPY_DTYPES_TO_TF_DTYPES[indices_tensor.dtype] \
                    if isinstance(indices_tensor.dtype, np.dtype) else indices_tensor.dtype,
            ),
        )
        indices_flat = tf.reduce_sum(
            mul,
            axis=-1,
        )
        result_flat = tf.gather(
            params_flat,
            indices_flat,
        )
        pseudo_gathernd = None
        if len(idx_shape) > 0 and len(idx_shape[:-1]) > 0 and idx_shape[:-1][0] is not None:
            pseudo_gathernd = tf.reshape(
                result_flat,
                tf.concat([idx_shape[:-1], gather_shape], axis=0),
            )
        else:
            pseudo_gathernd = result_flat
        tf_layers_dict[graph_node_output.name]['tf_node'] = pseudo_gathernd

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
                'tf_op_type': tf.gather_nd,
                'tf_inputs': {
                    'params': input_tensor,
                    'indices': indices_tensor,
                    'batch_dims': batch_dims,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
