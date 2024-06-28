import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    convert_axis,
    alternative_argmax,
    alternative_fused_argmax,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
)
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
    """ArgMax

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

    replace_argmax_to_reducemax_and_indices_is_int64 = \
        kwargs['replace_argmax_to_reducemax_and_indices_is_int64']
    replace_argmax_to_reducemax_and_indices_is_float32 = \
        kwargs['replace_argmax_to_reducemax_and_indices_is_float32']
    replace_argmax_to_fused_argmax_and_indices_is_int64 = \
        kwargs['replace_argmax_to_fused_argmax_and_indices_is_int64']
    replace_argmax_to_fused_argmax_and_indices_is_float32 = \
        kwargs['replace_argmax_to_fused_argmax_and_indices_is_float32']

    axis = graph_node.attrs.get('axis', 0)
    # NCHW->NHWC, NCDHW->NDHWC
    axis = convert_axis(
        axis=axis,
        tensor_rank=len(graph_node_input.shape),
        before_op_output_shape_trans=before_op_output_shape_trans,
    )

    # 0: False, 1: True
    keepdims = bool(graph_node.attrs.get('keepdims', 1))

    # 0: False, 1: True
    select_last_index = bool(graph_node.attrs.get('select_last_index', 0))

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    }

    # Generation of TF OP
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    ### 1. Select last index or first index
    reversed_tensor = None
    if not select_last_index:
        reversed_tensor = input_tensor
    else:
        reversed_tensor = \
            tf.reverse(
                tensor=input_tensor,
                axis=axis,
                name=f'{graph_node.name}_reverse',
            )

    ### 2. Replace ArgMax with a ReduceMax or no replace
    final_tensor = None
    tf_op_type = None
    if not replace_argmax_to_reducemax_and_indices_is_int64 \
        and not replace_argmax_to_reducemax_and_indices_is_float32 \
        and not replace_argmax_to_fused_argmax_and_indices_is_int64 \
        and not replace_argmax_to_fused_argmax_and_indices_is_float32:
        argmaxed_tensor = tf.math.argmax(
            input=reversed_tensor,
            axis=axis,
            output_type=NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
                if isinstance(dtype, np.dtype) else dtype,
            name=f'{graph_node.name}_argmax',
        )
        if keepdims:
            final_tensor = \
                tf.expand_dims(
                    input=argmaxed_tensor,
                    axis=axis,
                    name=f'{graph_node.name}_expand_dims',
                )
        else:
            final_tensor = argmaxed_tensor
        tf_op_type = tf.math.argmax
    elif replace_argmax_to_reducemax_and_indices_is_int64 \
        or replace_argmax_to_reducemax_and_indices_is_float32:
        final_tensor = alternative_argmax(
            input_tensor=reversed_tensor,
            axis=axis,
            output_type=tf.int64 if replace_argmax_to_reducemax_and_indices_is_int64 else tf.float32,
            keepdims=keepdims,
            replace_argmax_to_reducemax_and_indices_is_int64=replace_argmax_to_reducemax_and_indices_is_int64,
            replace_argmax_to_reducemax_and_indices_is_float32=replace_argmax_to_reducemax_and_indices_is_float32,
        )
        tf_op_type = 'alternative_argmax'
    elif (
            replace_argmax_to_fused_argmax_and_indices_is_int64 \
            or replace_argmax_to_fused_argmax_and_indices_is_float32
        ) and graph_node.i().op == 'Resize':
        final_tensor = alternative_fused_argmax(
            input_tensor=reversed_tensor,
            original_shape=graph_node.inputs[0].shape,
            axis=axis,
            output_type=NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
                if isinstance(dtype, np.dtype) else dtype,
            keepdims=keepdims,
            replace_argmax_to_fused_argmax_and_indices_is_int64=replace_argmax_to_fused_argmax_and_indices_is_int64,
            replace_argmax_to_fused_argmax_and_indices_is_float32=replace_argmax_to_fused_argmax_and_indices_is_float32,
        )
        tf_op_type = 'alternative_fused_argmax'
    else:
        argmaxed_tensor = tf.math.argmax(
            input=reversed_tensor,
            axis=axis,
            output_type=NUMPY_DTYPES_TO_TF_DTYPES[dtype] \
                if isinstance(dtype, np.dtype) else dtype,
            name=f'{graph_node.name}_argmax',
        )
        if keepdims:
            final_tensor = \
                tf.expand_dims(
                    input=argmaxed_tensor,
                    axis=axis,
                    name=f'{graph_node.name}_expand_dims',
                )
        else:
            final_tensor = argmaxed_tensor
        tf_op_type = tf.math.argmax

    tf_layers_dict[graph_node_output.name]['tf_node'] = final_tensor

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
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'input': input_tensor,
                    'axis': axis,
                    'output_type': tf.float32 if replace_argmax_to_reducemax_and_indices_is_float32 else dtype,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
