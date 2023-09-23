import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
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
from onnx2tf.utils.enums import ONNX_DTYPES_TO_TF_DTYPES


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Bernoulli

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

    bernoulli_dtype = graph_node.attrs.get('dtype', None)
    if bernoulli_dtype is not None:
        bernoulli_dtype = ONNX_DTYPES_TO_TF_DTYPES[bernoulli_dtype]
    else:
        bernoulli_dtype = input_tensor.dtype

    seed = graph_node.attrs.get('seed', 0)

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    def Bernoulli(
        input_tensor,
        bernoulli_dtype,
        seed,
        input_tensor_shape,
        name,
    ):
        return tf.reshape(
            tf.compat.v1.distributions.Bernoulli(
                probs=input_tensor,
                dtype=bernoulli_dtype,
                name=name,
            ).sample(seed=seed),
            input_tensor_shape,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        Lambda(
            Bernoulli,
            arguments={
                'bernoulli_dtype': bernoulli_dtype,
                'seed': seed,
                'input_tensor_shape': input_tensor_shape,
                'name': graph_node.name,
            }
        )(input_tensor)

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
                'tf_op_type': 'Bernoulli',
                'tf_inputs': {
                    'input': input_tensor,
                    'seed': seed,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
