import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    explicit_broadcast,
    pre_explicit_broadcast,
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
    """BatchNormalization

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    # Inputs
    X: gs.Variable = graph_node.inputs[0]
    scale = graph_node.inputs[1]
    B = graph_node.inputs[2]
    input_mean = graph_node.inputs[3]
    input_var = graph_node.inputs[4]
    # Outputs
    Y: gs.Variable = graph_node.outputs[0]
    if len(graph_node.outputs) > 1:
        graph_node.outputs = [graph_node.outputs[0]]

    if hasattr(scale, 'inputs') \
        and len(scale.inputs) > 0 \
        and hasattr(scale.inputs[0], 'attrs') \
        and 'value' in scale.inputs[0].attrs \
        and hasattr(scale.inputs[0].attrs['value'], 'values'):
        scale = scale.inputs[0].attrs['value']

    if hasattr(B, 'inputs') \
        and len(B.inputs) > 0 \
        and hasattr(B.inputs[0], 'attrs') \
        and 'value' in B.inputs[0].attrs \
        and hasattr(B.inputs[0].attrs['value'], 'values'):
        B = B.inputs[0].attrs['value']

    if hasattr(input_mean, 'inputs') \
        and len(input_mean.inputs) > 0 \
        and hasattr(input_mean.inputs[0], 'attrs') \
        and 'value' in input_mean.inputs[0].attrs \
        and hasattr(input_mean.inputs[0].attrs['value'], 'values'):
        input_mean = input_mean.inputs[0].attrs['value']

    if hasattr(input_var, 'inputs') \
        and len(input_var.inputs) > 0 \
        and hasattr(input_var.inputs[0], 'attrs') \
        and 'value' in input_var.inputs[0].attrs \
        and hasattr(input_var.inputs[0].attrs['value'], 'values'):
        input_var = input_var.inputs[0].attrs['value']

    shape = Y.shape
    dtype = Y.dtype

    epsilon = graph_node.attrs.get('epsilon', 1e-05)
    momentum = graph_node.attrs.get('momentum', 0.9)
    training_mode = bool(graph_node.attrs.get('training_mode', 0)) # disuse

    nhwc: bool = tf_layers_dict[X.name]['nhwc'] \
        if isinstance(X, gs.Variable) and 'nhwc' in tf_layers_dict[X.name].keys() else False

    # Preserving Graph Structure (Dict)
    tf_layers_dict[Y.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': nhwc,
    }

    # Generation of TF OP
    input_tensor = tf_layers_dict[X.name]['tf_node']

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    mean = input_mean.values \
        if not isinstance(input_mean, gs.Variable) \
            else tf_layers_dict[input_mean.name]['tf_node']
    var = input_var.values \
        if not isinstance(input_var, gs.Variable) \
            else tf_layers_dict[input_var.name]['tf_node']
    offset = B.values \
        if not isinstance(B, gs.Variable) \
            else tf_layers_dict[B.name]['tf_node']
    scale = scale.values \
        if not isinstance(scale, gs.Variable) \
            else tf_layers_dict[scale.name]['tf_node']

    # Broadcast
    if len(input_tensor.shape) == 3:
        input_tensor, mean = pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=mean,
        )
        input_tensor, mean = explicit_broadcast(
            const_or_var_1=input_tensor,
            const_or_var_2=mean,
            graph_node=graph_node,
            tf_layers_dict= tf_layers_dict,
        )
        input_tensor, var = pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=var,
        )
        input_tensor, var = explicit_broadcast(
            const_or_var_1=input_tensor,
            const_or_var_2=var,
            graph_node=graph_node,
            tf_layers_dict= tf_layers_dict,
        )
        input_tensor, offset = pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=offset,
        )
        input_tensor, offset = explicit_broadcast(
            const_or_var_1=input_tensor,
            const_or_var_2=offset,
            graph_node=graph_node,
            tf_layers_dict= tf_layers_dict,
        )
        input_tensor, scale = pre_explicit_broadcast(
            input_tensor_1=input_tensor,
            input_tensor_2=scale,
        )
        input_tensor, scale = explicit_broadcast(
            const_or_var_1=input_tensor,
            const_or_var_2=scale,
            graph_node=graph_node,
            tf_layers_dict= tf_layers_dict,
        )

    try:
        tf_layers_dict[Y.name]['tf_node'] = \
            tf.nn.batch_normalization(
                x=input_tensor,
                mean=mean \
                    if not isinstance(mean, np.ndarray) \
                        else tf.convert_to_tensor(mean),
                variance=var \
                    if not isinstance(var, np.ndarray) \
                        else tf.convert_to_tensor(var),
                offset=offset \
                    if not isinstance(offset, np.ndarray) \
                        else tf.convert_to_tensor(offset),
                scale=scale \
                    if not isinstance(scale, np.ndarray) \
                        else tf.convert_to_tensor(scale),
                variance_epsilon=epsilon,
            )
        # Shape correction workaround when final output shapes do not match
        if not nhwc \
            and graph_node.outputs[0].shape is not None \
            and sum([1 if isinstance(s, str) else 0 for s in graph_node.outputs[0].shape]) == 0 \
            and np.prod(graph_node.outputs[0].shape) != np.prod(tf_layers_dict[Y.name]['tf_node'].shape) \
            and (
                (mean.shape is not None and len(mean.shape) == 1) \
                    or (var.shape is not None and len(var.shape) == 1) \
                    or (offset.shape is not None and len(offset.shape) == 1) \
                    or (scale.shape is not None and len(scale.shape) == 1)
            ):
                if len(input_tensor.shape) == 3:
                    input_tensor = \
                        transpose_with_flexing_deterrence(
                            input_tensor=input_tensor,
                            perm=[0,2,1],
                            **kwargs,
                        )
                    tf_layers_dict[Y.name]['tf_node'] = \
                        tf.nn.batch_normalization(
                            x=input_tensor,
                            mean=mean \
                                if not isinstance(mean, np.ndarray) \
                                    else tf.convert_to_tensor(mean),
                            variance=var \
                                if not isinstance(var, np.ndarray) \
                                    else tf.convert_to_tensor(var),
                            offset=offset \
                                if not isinstance(offset, np.ndarray) \
                                    else tf.convert_to_tensor(offset),
                            scale=scale \
                                if not isinstance(scale, np.ndarray) \
                                    else tf.convert_to_tensor(scale),
                            variance_epsilon=epsilon,
                        )
                elif len(input_tensor.shape) == 4:
                    input_tensor = \
                        transpose_with_flexing_deterrence(
                            input_tensor=input_tensor,
                            perm=[0,2,3,1],
                            **kwargs,
                        )
                    tf_layers_dict[Y.name]['tf_node'] = \
                        tf.nn.batch_normalization(
                            x=input_tensor,
                            mean=mean \
                                if not isinstance(mean, np.ndarray) \
                                    else tf.convert_to_tensor(mean),
                            variance=var \
                                if not isinstance(var, np.ndarray) \
                                    else tf.convert_to_tensor(var),
                            offset=offset \
                                if not isinstance(offset, np.ndarray) \
                                    else tf.convert_to_tensor(offset),
                            scale=scale \
                                if not isinstance(scale, np.ndarray) \
                                    else tf.convert_to_tensor(scale),
                            variance_epsilon=epsilon,
                        )
                elif len(input_tensor.shape) == 5:
                    input_tensor = \
                        transpose_with_flexing_deterrence(
                            input_tensor=input_tensor,
                            perm=[0,2,3,4,1],
                            **kwargs,
                        )
                    tf_layers_dict[Y.name]['tf_node'] = \
                        tf.nn.batch_normalization(
                            x=input_tensor,
                            mean=mean \
                                if not isinstance(mean, np.ndarray) \
                                    else tf.convert_to_tensor(mean),
                            variance=var \
                                if not isinstance(var, np.ndarray) \
                                    else tf.convert_to_tensor(var),
                            offset=offset \
                                if not isinstance(offset, np.ndarray) \
                                    else tf.convert_to_tensor(offset),
                            scale=scale \
                                if not isinstance(scale, np.ndarray) \
                                    else tf.convert_to_tensor(scale),
                            variance_epsilon=epsilon,
                        )

    except Exception as e:
        # Workaround for inconsistent "C" position
        input_tensor_rank = len(input_tensor.shape)
        if input_tensor_rank >= 3 \
            and input_tensor.shape[1] is not None \
            and input_tensor.shape[1] == offset.shape[0]:

            perm = [0] + [i for i in range(2, input_tensor_rank)] + [1]
            tf_layers_dict[Y.name]['tf_node'] = \
                tf.nn.batch_normalization(
                    x=tf.transpose(input_tensor, perm=perm),
                    mean=mean \
                        if not isinstance(mean, np.ndarray) \
                            else tf.convert_to_tensor(mean),
                    variance=var \
                        if not isinstance(var, np.ndarray) \
                            else tf.convert_to_tensor(var),
                    offset=offset \
                        if not isinstance(offset, np.ndarray) \
                            else tf.convert_to_tensor(offset),
                    scale=scale \
                        if not isinstance(scale, np.ndarray) \
                            else tf.convert_to_tensor(scale),
                    variance_epsilon=epsilon,
                )
        else:
            raise


    # Post-process transpose
    tf_layers_dict[Y.name]['tf_node'] = post_process_transpose(
        value_before_transpose=tf_layers_dict[Y.name]['tf_node'],
        param_target='outputs',
        param_name=graph_node.outputs[0].name,
        **kwargs,
    )

    # Generation of Debug Info
    tf_layers_dict[Y.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'BatchNormalization',
                'tf_inputs': {
                    'X': tf_layers_dict[X.name]['tf_node'],
                    'mean': input_mean,
                    'variance': input_var,
                    'offset': B,
                    'scale': scale,
                    'variance_epsilon': epsilon,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[Y.name]['tf_node'],
                },
            }
        )
