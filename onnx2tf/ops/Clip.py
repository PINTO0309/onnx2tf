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
    """Clip

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans_1 = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_2 = True
    if len(graph_node.inputs) >= 2:
        before_op_output_shape_trans_2 = \
            tf_layers_dict.get(graph_node.inputs[1].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_3 = True
    if len(graph_node.inputs) >= 3:
        before_op_output_shape_trans_3 = \
            tf_layers_dict.get(graph_node.inputs[2].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    min_value_node = None
    if len(graph_node.inputs) >= 2:
        min_value_node = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    max_value_node = None
    if len(graph_node.inputs) >= 3:
        max_value_node = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]

    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    min_value_node = graph_node.attrs.get('min', min_value_node)
    max_value_node = graph_node.attrs.get('max', max_value_node)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    }

    relu_relu6_merge_op_names: dict = kwargs['relu_relu6_merge_op_names']
    relu6_op_names = [op_name for op_names in relu_relu6_merge_op_names.values() for op_name in op_names]
    enable_relu_relu6_merge = graph_node.name in relu6_op_names

    # Generation of TF OP
    features = None
    if isinstance(graph_node_input, gs.Variable):
        features = tf_layers_dict[graph_node_input.name]['tf_node']
    else:
        features = graph_node_input
    min_value = None
    if isinstance(min_value_node, gs.Variable) and min_value_node.shape is not None:
        min_value = tf_layers_dict[min_value_node.name]['tf_node']
    else:
        min_value = min_value_node
    max_value = None
    if isinstance(max_value_node, gs.Variable) and max_value_node.shape is not None:
        max_value = tf_layers_dict[max_value_node.name]['tf_node']
    else:
        max_value = max_value_node

    # Pre-process transpose
    before_trans_shape = features.shape
    features = pre_process_transpose(
        value_before_transpose=features,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    after_trans_shape = features.shape
    if 'nhwc' in tf_layers_dict[graph_node_output.name].keys() \
        and tf_layers_dict[graph_node_output.name]['nhwc'] == True \
        and before_trans_shape != after_trans_shape:
        tf_layers_dict[graph_node_output.name].pop('nhwc')

    if min_value is not None and (isinstance(min_value, float) or isinstance(min_value, tf.Tensor)):
        min_value = np.asarray([min_value])
    if max_value is not None and (isinstance(max_value, float) or isinstance(max_value, tf.Tensor)):
        max_value = np.asarray([max_value])

    tf_op_type = None

    if not enable_relu_relu6_merge:
        features_dtype = NUMPY_DTYPES_TO_TF_DTYPES[features.dtype] \
            if isinstance(features.dtype, np.dtype) else features.dtype
        if (isinstance(min_value, np.ndarray) or isinstance(min_value, float)) and min_value == 0.0 \
            and (isinstance(max_value, np.ndarray)  or isinstance(max_value, float)) and max_value == 6.0:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.nn.relu6(features=features)
            tf_op_type = tf.nn.relu6
        elif (isinstance(min_value, np.ndarray) or isinstance(min_value, float)) and min_value == 0.0 \
            and (max_value is None or max_value.shape is None or (hasattr(max_value, '_inferred_value') and max_value._inferred_value is None)):
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.nn.relu(features=features)
            tf_op_type = tf.nn.relu
        else:
            if (isinstance(min_value, np.ndarray) and min_value.shape is not None) \
                and (isinstance(max_value, np.ndarray) and max_value.shape is not None):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.clip_by_value(
                        t=features,
                        clip_value_min=tf.convert_to_tensor(min_value, dtype=features_dtype) \
                            if isinstance(min_value, np.ndarray) else min_value,
                        clip_value_max=tf.convert_to_tensor(max_value, dtype=features_dtype) \
                            if isinstance(max_value, np.ndarray) else max_value,
                    )
                tf_op_type = tf.clip_by_value
            elif (isinstance(min_value, np.ndarray) and min_value.shape is not None) \
                and (max_value is None or max_value.shape is None or (hasattr(max_value, '_inferred_value') and max_value._inferred_value is None)):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.maximum(
                        x=features,
                        y=tf.convert_to_tensor(min_value, dtype=features_dtype) \
                            if isinstance(min_value, np.ndarray) else min_value,
                    )
                tf_op_type = tf.maximum
            elif (min_value is None or min_value.shape is None or (hasattr(min_value, '_inferred_value') and min_value._inferred_value is None)) \
                and (max_value is not None and max_value.shape is not None):
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.minimum(
                        x=features,
                        y=tf.convert_to_tensor(max_value, dtype=features_dtype) \
                            if isinstance(max_value, np.ndarray) else max_value,
                    )
                tf_op_type = tf.minimum
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.identity(
                input=features,
                name=graph_node.name,
            )
        tf_op_type = tf.identity

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
                    'features': features,
                    'min_value': min_value,
                    'max_value': max_value,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
