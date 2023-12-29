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
    """Relu

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

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': tf_layers_dict[graph_node_input.name]['nhwc'] \
            if isinstance(graph_node_input, gs.Variable) \
                and 'nhwc' in tf_layers_dict[graph_node_input.name].keys() else False
    }

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

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

    # Relu + Relu6 -> Relu6, opset >= 11
    enable_relu_relu6_merge = False
    try:
        opset = int(kwargs['opset'])
        if opset >= 11:
            relu6_node: gs.Node = None
            if graph_node.o().op == 'Clip' \
                and len(graph_node.o().inputs) == 3 \
                and isinstance(graph_node.o().inputs[1], gs.Constant) \
                and isinstance(graph_node.o().inputs[2], gs.Constant) \
                and hasattr(graph_node.o().inputs[1], 'values') \
                and hasattr(graph_node.o().inputs[2], 'values') \
                and graph_node.o().inputs[1].values == 0 \
                and graph_node.o().inputs[2].values == 6:

                relu6_node = graph_node.o()
                if 'relu_relu6_merge_op_names' not in kwargs:
                    kwargs['relu_relu6_merge_op_names'] = {}
                kwargs['relu_relu6_merge_op_names'][graph_node.name] = [
                    relu6_node.name,
                ]
                relu6_node = graph_node.o()
                enable_relu_relu6_merge = True

        elif opset <= 10:
            min_value = graph_node.attrs.get('min', None)
            max_value = graph_node.attrs.get('max', None)
            if graph_node.o().op == 'Clip' \
                and min_value is not None \
                and max_value is not None \
                and min_value == 0.0 \
                and max_value == 6.0:

                relu6_node = graph_node.o()
                if 'relu_relu6_merge_op_names' not in kwargs:
                    kwargs['relu_relu6_merge_op_names'] = {}
                kwargs['relu_relu6_merge_op_names'][graph_node.name] = [
                    relu6_node.name,
                ]
                enable_relu_relu6_merge = True
    except:
        pass

    # Generation of TF OP
    if not enable_relu_relu6_merge:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.nn.relu(
                features=input_tensor,
                name=graph_node.name,
            )
    else:
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.nn.relu6(
                features=input_tensor,
                name=graph_node.name,
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
                'tf_op_type': tf.nn.relu,
                'tf_inputs': {
                    'features': input_tensor,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
