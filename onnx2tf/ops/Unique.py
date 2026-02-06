import random
from typing import List

random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tf_keras
import onnx2tf.gs as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    get_replacement_parameter,
    pre_process_transpose,
    convert_axis,
)
from onnx2tf.utils.logging import Color


class tfUnique(tf_keras.layers.Layer):

    def __init__(self):
        super(tfUnique, self).__init__()
        self.unique_ops = tf.raw_ops.UniqueWithCountsV2

    def call(self, x, axis):
        return self.unique_ops(x=x, axis=[axis], out_idx=tf.int64)


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
        *,
        graph_node: gs.Node,
        tf_layers_dict: dict,
        **kwargs: dict,
):
    """Unique

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    before_op_output_shape_trans = \
        tf_layers_dict.get(graph_node.inputs[0].name, {}).get('before_op_output_shape_trans', True)

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_outputs: List[gs.Variable] = [
        graph_node_output for graph_node_output in graph_node.outputs
    ]

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape
    tensor_rank = len(input_tensor_shape) \
        if input_tensor_shape != tf.TensorShape(None) else 1

    axis = graph_node.attrs.get('axis', None)
    sorted = graph_node.attrs.get('sorted', 1)
    if axis is not None:
        if isinstance(axis, np.ndarray) and axis.shape == ():
            axis = int(axis)
        axis = convert_axis(
            axis=int(axis),
            tensor_rank=tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )

    # Preserving Graph Structure (Dict)
    for graph_node_output in graph_node_outputs:
        tf_layers_dict[graph_node_output.name] = {
            'optype': graph_node.op,
            'shape': graph_node_output.shape,
            'dtype': graph_node_output.dtype,
        }

    # Generation of TF OP
    # tensorflow raw_ops does not support direct call to KerasTensor, need to call through keras layer
    tf_unique_ops = tfUnique()

    # flatten tensor if axis is not specified
    if axis is None:
        axis = 0
        input_tensor = tf.reshape(input_tensor, [-1])

    # CAUTION: tensorflow unique returns inverse indices only
    y, inverse_indices, count = tf_unique_ops(x=input_tensor, axis=axis)

    # use tf.unique again to get true unique indices
    rey, reidx = tf.unique(inverse_indices)
    num_segments = tf.shape(rey)[0]
    num_elems = tf.shape(inverse_indices)[0]
    indices = tf.math.unsorted_segment_min(tf.range(num_elems), reidx, num_segments)
    indices = tf.cast(indices, dtype=inverse_indices.dtype)

    # tf unique returns unsorted tensor, need to sort if option is enabled
    if sorted:
        # Sort unique outputs to match ONNX sorted behavior.
        def _argsort_supported(dtype):
            return dtype.is_floating or dtype.is_integer or dtype == tf.bool

        y_rank = y.shape.rank
        axis_ = axis
        if axis_ is None:
            axis_ = 0
        if axis_ < 0 and y_rank is not None:
            axis_ = axis_ + y_rank

        def _lexsort_perm(flat_2d):
            if not _argsort_supported(flat_2d.dtype):
                return None
            cols = flat_2d.shape[1]
            if cols is None:
                return None
            order = tf.range(tf.shape(flat_2d)[0])
            for col in reversed(range(cols)):
                col_vals = tf.gather(flat_2d, order)[:, col]
                if col_vals.dtype == tf.bool:
                    col_vals = tf.cast(col_vals, tf.int32)
                order = tf.gather(order, tf.argsort(col_vals, stable=True))
            return order

        order = None
        if y_rank is not None and y_rank == 1:
            if _argsort_supported(y.dtype):
                sort_vals = y
                if sort_vals.dtype == tf.bool:
                    sort_vals = tf.cast(sort_vals, tf.int32)
                order = tf.argsort(sort_vals, stable=True)
        elif y_rank is not None and axis_ is not None and 0 <= axis_ < y_rank:
            perm = [axis_] + [i for i in range(y_rank) if i != axis_]
            y_t = tf.transpose(y, perm)
            flat = tf.reshape(y_t, [tf.shape(y_t)[0], -1])
            order = _lexsort_perm(flat)

        if order is None:
            warn_msg = f'' + \
                       Color.YELLOW(f'WARNING:') + ' ' + \
                       f'Unique sort fallback to unsorted due to dynamic shape or unsupported dtype.'
            print(warn_msg)
        else:
            y = tf.gather(y, order, axis=axis_)
            count = tf.gather(count, order)
            indices = tf.gather(indices, order)
            inv_order = tf.argsort(order)
            inverse_indices = tf.gather(inv_order, inverse_indices)

    if len(graph_node_outputs) >= 1:
        tf_layers_dict[graph_node_outputs[0].name]['tf_node'] = y
    if len(graph_node_outputs) >= 2:
        tf_layers_dict[graph_node_outputs[1].name]['tf_node'] = indices
    if len(graph_node_outputs) >= 3:
        tf_layers_dict[graph_node_outputs[2].name]['tf_node'] = inverse_indices
    if len(graph_node_outputs) >= 4:
        tf_layers_dict[graph_node_outputs[3].name]['tf_node'] = count

    # Generation of Debug Info
    tf_outputs = {f"output{idx}": value for idx, value in enumerate([y, indices, inverse_indices, count])}
    tf_layers_dict[graph_node_outputs[0].name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.raw_ops.UniqueWithCountsV2,
                'tf_inputs': {
                    'value': input_tensor,
                    'axis': axis,
                    'sorted': sorted
                },
                'tf_outputs': tf_outputs,
            }
        )
