import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.python.keras.layers import Lambda
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    convert_axis,
)
from onnx2tf.utils.enums import NUMPY_DTYPES_TO_TF_DTYPES
from onnx2tf.utils.colors import Color


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Slice

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
    before_op_output_shape_trans_4 = True
    if len(graph_node.inputs) >= 4:
        before_op_output_shape_trans_4 = \
            tf_layers_dict.get(graph_node.inputs[3].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans_5 = True
    if len(graph_node.inputs) >= 5:
        before_op_output_shape_trans_5 = \
            tf_layers_dict.get(graph_node.inputs[4].name, {}).get('before_op_output_shape_trans', True)
    before_op_output_shape_trans = \
        before_op_output_shape_trans_1 \
        and before_op_output_shape_trans_2 \
        and before_op_output_shape_trans_3 \
        and before_op_output_shape_trans_4 \
        and before_op_output_shape_trans_5

    input_tensor = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[input_tensor.name]['tf_node'] \
        if isinstance(input_tensor, gs.Variable) else input_tensor
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    starts = None
    if len(graph_node.inputs) >= 2:
        starts = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
        starts = tf_layers_dict[starts.name]['tf_node'] \
            if isinstance(starts, gs.Variable) else starts

    ends = None
    if len(graph_node.inputs) >= 3:
        ends = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
        ends = tf_layers_dict[ends.name]['tf_node'] \
            if isinstance(ends, gs.Variable) else ends

    axes = None
    if len(graph_node.inputs) >= 4:
        axes = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    axes = tf_layers_dict[axes.name]['tf_node'] \
        if isinstance(axes, gs.Variable) else axes
    if isinstance(axes, np.ndarray):
        axes = axes if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends.dtype)
    elif isinstance(axes, list):
        axes = np.asarray(axes, dtype=ends.dtype) if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends.dtype)
    elif axes is not None:
        axes = axes if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends.dtype)

    steps = None
    if len(graph_node.inputs) >= 5:
        steps = get_constant_or_variable(
            graph_node.inputs[4],
            before_op_output_shape_trans,
        )
    steps = tf_layers_dict[steps.name]['tf_node'] \
        if isinstance(steps, gs.Variable) else steps
    if isinstance(steps, np.ndarray):
        steps = tf.constant(steps, dtype=NUMPY_DTYPES_TO_TF_DTYPES[steps.dtype])

    axes = graph_node.attrs.get('axes', axes)

    if isinstance(axes, list) or (isinstance(axes, np.ndarray) and len(axes.shape) > 0):
        axes = [
            convert_axis(
                axis=idx,
                tensor_rank=input_tensor_rank,
                before_op_output_shape_trans=before_op_output_shape_trans,
            ) for idx in axes
        ]
    elif axes is not None and isinstance(axes, np.ndarray) and len(axes.shape) == 0:
        axes = convert_axis(
            axis=axes,
            tensor_rank=input_tensor_rank,
            before_op_output_shape_trans=before_op_output_shape_trans,
        )
    if isinstance(axes, list):
        axes = np.asarray(axes)

    starts = graph_node.attrs.get('starts', starts)
    if isinstance(starts, list):
        starts = np.asarray(starts)
    ends = graph_node.attrs.get('ends', ends)
    if isinstance(ends, list):
        ends = np.asarray(ends)

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    # Param replacement - OP replacement
    """
    Slice implements special replacements separately at this time
    Ignore all automatic conversions and generate tf.strided_slice directly
    by specifying all parameters of tf.strided_slice directly
    https://www.tensorflow.org/api_docs/python/tf/strided_slice

    import numpy as np
    n = np.asarray(
        [
            [
                [1, 1, 1],
                [2, 2, 2],
            ],
            [
                [3, 3, 3],
                [4, 4, 4],
            ],
            [
                [5, 5, 5],
                [6, 6, 6],
            ],
        ]
    )
    n.shape: (3, 2, 3)

    import tensorflow as tf
    t = tf.constant(
        [
            [
                [1, 1, 1],
                [2, 2, 2],
            ],
            [
                [3, 3, 3],
                [4, 4, 4],
            ],
            [
                [5, 5, 5],
                [6, 6, 6],
            ],
        ]
    )
    t.shape: TensorShape([3, 2, 3])

    # Numpy [begin0:end0:step0, begin1:end1:step1, begin2:end2:step2, ...]
        n[1:2, 0:1, 0:3] -> [[[3, 3, 3]]]
        n[1:2, 0:2, 0:3] -> [[[3, 3, 3], [4, 4, 4]]]
        n[1:2:1, 0:1:1, 0:3:1] -> [[[3, 3, 3]]]

    # TensorFlow [begin0,begin1,begin2, ...], [end0,end1,end2, ...], [strides0,strides1,strides2, ...]
        tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1]) -> [[[3, 3, 3]]]
        tf.strided_slice(t, [1, 0, 0], [2, 2, 3], [1, 1, 1]) -> [[[3, 3, 3], [4, 4, 4]]]
    """

    op_rep_params = kwargs.get('op_rep_params', [])
    begin_ = None
    for op_rep_param in op_rep_params:
        if op_rep_param['param_target'] == 'op':
            begin_ = op_rep_param.get('begin', None)
            end_ = op_rep_param.get('end', None)
            strides_ = op_rep_param.get('strides', None)
            begin_mask_ = op_rep_param.get('begin_mask', 0)
            end_mask_ = op_rep_param.get('end_mask', 0)
            ellipsis_mask_ = op_rep_param.get('ellipsis_mask', 0)
            new_axis_mask_ = op_rep_param.get('new_axis_mask', 0)
            shrink_axis_mask_ = op_rep_param.get('shrink_axis_mask', 0)

            if begin_ is None or end_ is None:
                print(
                    f'{Color.RED}ERROR:{Color.RESET} ' +
                    f'When replacing Slice OP, "begin" and "end" must be specified in replace.json. ' +
                    f'Check the specification of tf.strided_slice in TensorFlow and specify the appropriate parameters. ' +
                    f'https://www.tensorflow.org/api_docs/python/tf/strided_slice'
                )
                sys.exit(1)

    # Generation of TF OP
    if begin_ is None:
        if None not in input_tensor_shape:
            # first of all, get the input tensor shape
            if axes is not None:
                is_axes_negative = tf.less(axes, tf.zeros_like(axes))
                axes = tf.where(is_axes_negative, axes + tf.cast(input_tensor_rank, axes.dtype), axes)
            else:
                axes = [i for i in range(input_tensor_rank)]

            # expand a dimension of 1 at the end
            sparse_indices = tf.cast(tf.expand_dims(axes, -1), tf.int64)

            # build the indexed dimension sizes as sparse_shape
            sparse_shape = tf.gather_nd(
                params=input_tensor_shape,
                indices=sparse_indices,
            )
            sparse_shape = tf.cast(sparse_shape, tf.int64)

            # take care of starts, ends that are larger than the dim size.
            starts_min = tf.minimum(starts, sparse_shape)
            ends_min = tf.minimum(ends, sparse_shape)

            # need to densify everything for the inputs to slice
            # the output shape is the input_tensor rank
            output_shape = tf.reshape(input_tensor_rank, [1])
            output_shape = tf.cast(output_shape, tf.int64)

            def tf_SparseTensor(sparse_indices, pos, sparse_shape, start_or_end):
                dense = None
                # take care of starts, ends that are negative
                if start_or_end == 'start':
                    is_starts_negative = tf.less(pos, tf.zeros_like(pos))
                    final = tf.where(is_starts_negative, pos + sparse_shape, pos)
                    sparse_tensor = tf.sparse.reorder(
                        sp_input=tf.sparse.SparseTensor(
                            indices=sparse_indices,
                            values=final,
                            dense_shape=sparse_shape,
                        )
                    )
                    dense = tf.sparse.to_dense(sparse_tensor)
                elif start_or_end == 'end':
                    is_ends_negative = tf.less(pos, tf.zeros_like(pos))
                    final = tf.where(is_ends_negative, pos + sparse_shape, pos)
                    sparse_tensor = tf.sparse.reorder(
                        sp_input=tf.sparse.SparseTensor(
                            indices=sparse_indices,
                            values=final,
                            dense_shape=sparse_shape,
                        )
                    )
                    dense_ends = tf.sparse.to_dense(
                        sparse_tensor,
                        default_value=tf.constant(-1, dtype=dense_begins.dtype)
                    )
                    dense = tf.where(
                        tf.equal(dense_ends, tf.constant(-1, dtype=dense_begins.dtype)),
                        input_tensor_shape,
                        dense_ends,
                    )
                return dense

            # create dense tensor, pad 0 as default begins
            dense_begins = Lambda(
                tf_SparseTensor,
                arguments={
                    'pos': starts_min,
                    'sparse_shape': output_shape,
                    'start_or_end': 'start',
                }
            )(sparse_indices)
            # create dense tensor, pad -1 for next step
            dense_ends = Lambda(
                tf_SparseTensor,
                arguments={
                    'pos': ends_min,
                    'sparse_shape': output_shape,
                    'start_or_end': 'end'
                }
            )(sparse_indices)

            # create dense tensor for steps if not already so
            if steps is not None:
                dense_steps = tf.sparse.reorder(
                    sp_input=tf.sparse.SparseTensor(
                        sparse_indices,
                        steps,
                        output_shape,
                    )
                )
                dense_steps = tf.sparse.to_dense(
                    dense_steps,
                    default_value=tf.constant(1, dtype=steps.dtype)
                )
            else:
                dense_steps = tf.ones(input_tensor_rank, ends.dtype)

            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.strided_slice(
                    input_=input_tensor,
                    begin=dense_begins,
                    end=dense_ends,
                    strides=tf.cast(dense_steps, dtype=dense_begins.dtype),
                )
        else:
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.strided_slice(
                    input_=input_tensor,
                    begin=starts,
                    end=ends,
                    strides=steps,
                    name=graph_node.name,
                )
    else:
        # OP replacement
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.strided_slice(
                input_=input_tensor,
                begin=begin_,
                end=end_,
                strides=strides_,
                begin_mask=begin_mask_,
                end_mask=end_mask_,
                ellipsis_mask=ellipsis_mask_,
                new_axis_mask=new_axis_mask_,
                shrink_axis_mask=shrink_axis_mask_,
                name=graph_node.name,
            )


    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf.strided_slice,
                'tf_inputs': {
                    'input_': input_tensor,
                    'begin': starts,
                    'end': ends,
                    'strides': steps,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
