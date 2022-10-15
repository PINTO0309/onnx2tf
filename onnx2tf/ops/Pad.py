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
)


def _check_positive(
    *,
    pads,
):
    p = tf.greater_equal(
        x=pads,
        y=tf.zeros(
            shape=(1),
            dtype=pads.dtype,
        )
    )
    r = tf.reduce_all(input_tensor=p)
    return r


def _process_neg_pads(
    *,
    x,
    paddings,
    constant_values,
    mode,
    tensor_rank,
    name,
):
    i_shape = tf.shape(
        input=x,
        out_type=paddings.dtype,
    )
    i_rank = tf.cast(
        x=tf.rank(x),
        dtype=paddings.dtype,
    )
    begins = tf.negative(
        x=tf.gather(
            params=paddings,
            indices=tf.range(i_rank),
        )
    )
    ends = i_shape + tf.gather(
        params=paddings,
        indices=tf.range(
            start=i_rank,
            limit=i_rank*2,
        ),
    )
    sizes = ends - begins

    return tf.slice(
        input_=x,
        begin=begins,
        size=sizes,
        name=name,
    )


def _process_pos_pads(
    *,
    x,
    paddings,
    constant_value,
    mode,
    tensor_rank,
    name,
):

    def _symmetric_pad(i, x):
        paddings_i = tf.map_fn(
            fn=lambda e: tf.where(i < e, 1, 0),
            elems=paddings,
        )
        paddings_i = tf.reshape(
            tensor=paddings_i,
            shape=[tensor_rank, 2],
        )
        x = tf.pad(
            tensor=x,
            paddings=paddings_i,
            mode='SYMMETRIC'
        )
        return i + 1, x

    # tf requires int32 paddings
    paddings = tf.cast(
        x=tf.transpose(
            a=tf.reshape(
                tensor=paddings,
                shape=[2, tensor_rank],
            )
        ),
        dtype=tf.int32,
    )

    if mode.lower() == "edge":
        # Tensorflow doesn't support edge mode so we need to implement the
        # np.pad(x, paddings, mode="edge") logic using Tensorflow ops. A
        # while loop is used to go through the tf.pad 'SYMMETRIC' mode to pad
        # one value at a time for both sides and all dimensions.
        paddings = tf.reshape(paddings, [-1])
        max_i = tf.reduce_max(paddings)
        _, x = tf.while_loop(
            lambda i, x: tf.less(i, max_i), _symmetric_pad, [0, x],
            [tf.TensorShape([]), tf.TensorShape(None)])
        return x

    return tf.pad(
        tensor=x,
        paddings=paddings,
        mode=mode,
        constant_values=constant_value,
        name=name,
    )


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Pad

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

    input_tensor = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    paddings = get_constant_or_variable(
        graph_node.inputs[1],
        before_op_output_shape_trans,
    )
    constant_value = 0
    if len(graph_node.inputs) >= 3  and graph_node.inputs[2].name != '':
        constant_value = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor
    tensor_rank = len(input_tensor.shape)
    paddings = tf_layers_dict[paddings.name]['tf_node'] \
        if isinstance(paddings, gs.Variable) else paddings

    constant_value = tf_layers_dict[constant_value.name]['tf_node'] \
        if isinstance(constant_value, gs.Variable) else constant_value

    # Transpose pads values
    paddings = graph_node.inputs[1]
    if hasattr(paddings, 'values'):
        values = paddings.values
        paddings = values.reshape([2, tensor_rank]).transpose()
        paddings_rank = paddings.shape[0]
        if paddings_rank > 2:
            convertion_table = [0] + [i for i in range(2, paddings_rank)] + [1]
            new_paddings = []
            for idx in convertion_table:
                new_paddings.append(paddings[idx, :])
            paddings = np.asarray(new_paddings)

    mode = graph_node.attrs.get('mode', 'constant')

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Generation of TF OP
    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        tf.cond(
            _check_positive(
                pads=paddings,
            ),
            lambda: _process_pos_pads(
                x=input_tensor,
                paddings=paddings,
                constant_value=constant_value,
                mode=mode,
                tensor_rank=tensor_rank,
                name= graph_node.name,
            ),
            lambda: _process_neg_pads(
                x=input_tensor,
                paddings=paddings,
                constant_value=constant_value,
                mode=mode,
                tensor_rank=tensor_rank,
                name=graph_node.name,
            ),
        )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': 'Pad',
                'tf_inputs': {
                    'x': input_tensor,
                    'paddings': paddings,
                    'constant_value': constant_value,
                    'mode': mode,
                    'tensor_rank': tensor_rank,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
