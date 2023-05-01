import sys
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
    convert_axis,
    replace_max_values_negative_values,
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    stridedslice_with_flexing_deterrence,
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

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape) \
        if input_tensor_shape != tf.TensorShape(None) else 1

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

    starts = graph_node.attrs.get('starts', starts)
    if isinstance(starts, list):
        starts = tf.convert_to_tensor(np.asarray(starts))
    ends = graph_node.attrs.get('ends', ends)
    if isinstance(ends, list):
        ends = tf.convert_to_tensor(np.asarray(ends))
    ends_dtype = NUMPY_DTYPES_TO_TF_DTYPES[ends.dtype] \
        if isinstance(ends.dtype, np.dtype) else ends.dtype

    axes = None
    if len(graph_node.inputs) >= 4:
        axes = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    axes = tf_layers_dict[axes.name]['tf_node'] \
        if isinstance(axes, gs.Variable) else axes

    if isinstance(axes, np.ndarray):
        axes = axes \
            if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends_dtype)
    elif isinstance(axes, list):
        axes = np.asarray(axes, dtype=ends.dtype) \
            if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends_dtype)
    elif axes is not None:
        axes = axes \
            if len(graph_node.inputs) >= 4 else tf.range(tf.shape(starts)[0], dtype=ends_dtype)

    steps = None
    if len(graph_node.inputs) >= 5:
        steps = get_constant_or_variable(
            graph_node.inputs[4],
            before_op_output_shape_trans,
        )
    steps = tf_layers_dict[steps.name]['tf_node'] \
        if isinstance(steps, gs.Variable) else steps
    if isinstance(steps, np.ndarray):
        steps_dtype = NUMPY_DTYPES_TO_TF_DTYPES[steps.dtype] \
            if isinstance(steps.dtype, np.dtype) else steps.dtype
        steps = tf.constant(steps, dtype=steps_dtype)

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
        axes = tf.convert_to_tensor(np.asarray(axes))

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
        ##### begin
        if isinstance(starts, tf.Tensor) and hasattr(starts, "numpy"):
            begin_ = [dim for dim in starts.numpy()]
        elif not isinstance(starts, np.ndarray) and tf.keras.backend.is_keras_tensor(starts):
            begin_ = starts
        else:
            begin_ = [dim for dim in starts]
        ##### end
        if isinstance(ends, tf.Tensor) and hasattr(ends, "numpy"):
            end_ = [dim for dim in ends.numpy()]
        elif not isinstance(ends, np.ndarray) and tf.keras.backend.is_keras_tensor(ends):
            end_ = ends
        else:
            end_ = [dim for dim in ends]
        ##### strides
        strides_ = None
        if steps is not None:
            if isinstance(steps, tf.Tensor) and hasattr(steps, "numpy"):
                strides_ = [dim for dim in steps.numpy()]
            elif not isinstance(steps, np.ndarray) and tf.keras.backend.is_keras_tensor(steps):
                strides_ = steps
            else:
                strides_ = [dim for dim in steps]

        # Adjust the number of dimensions of the input data according to the number of axes [List]
        ##### Replace max values
        if isinstance(begin_, list):
            if axes is not None:
                unsqueeze_mask = [1] * input_tensor_rank
                for axis in axes:
                    unsqueeze_mask[axis] = 0
            else:
                unsqueeze_mask = [0] * input_tensor_rank
            for axis, maskbit in enumerate(unsqueeze_mask):
                if maskbit == 1:
                    begin_.insert(axis, 0)
            begin_ = replace_max_values_negative_values(
                input_tensor_shape=input_tensor_shape,
                index_list=begin_,
                axes=axes,
            )
        ##### Replace negative values
        if isinstance(end_, list):
            if axes is not None:
                unsqueeze_mask = [1] * input_tensor_rank
                for axis in axes:
                    unsqueeze_mask[axis] = 0
            else:
                unsqueeze_mask = [0] * input_tensor_rank
            for axis, maskbit in enumerate(unsqueeze_mask):
                if maskbit == 1:
                    end_.insert(axis, 0)
            end_ = replace_max_values_negative_values(
                input_tensor_shape=input_tensor_shape,
                index_list=end_,
                axes=axes,
            )
        if strides_ is not None:
            if isinstance(strides_, list):
                if axes is not None:
                    unsqueeze_mask = [1] * input_tensor_rank
                    for axis in axes:
                        unsqueeze_mask[axis] = 0
                else:
                    unsqueeze_mask = [0] * input_tensor_rank
                for axis, maskbit in enumerate(unsqueeze_mask):
                    if maskbit == 1:
                        strides_.insert(axis, 1)

        # Adjust the number of dimensions of the input data according to the number of axes [Tensor]
        if not isinstance(begin_, list) and input_tensor_rank > begin_.shape[0]:
            begin_zeros = tf.zeros(shape=input_tensor_rank, dtype=tf.int64)
            begin_ = tf.tensor_scatter_nd_update(
                tensor=begin_zeros,
                indices=[[axis] for axis in axes],
                updates=begin_,
            )
        begin_ = tf.cast(x=begin_, dtype=tf.int64)
        if not isinstance(end_, list) and input_tensor_rank > end_.shape[0]:
            end_zeros = tf.zeros(input_tensor_rank, dtype=tf.int64)
            end_ = tf.tensor_scatter_nd_update(
                tensor=end_zeros,
                indices=[[axis] for axis in axes],
                updates=end_,
            )
        end_ = tf.cast(x=end_, dtype=tf.int64)
        if strides_ is not None and not isinstance(strides_, list) and input_tensor_rank > strides_.shape[0]:
            strides_ones = tf.ones(input_tensor_rank, dtype=tf.int64)
            strides_ = tf.tensor_scatter_nd_update(
                tensor=strides_ones,
                indices=[[axis] for axis in axes],
                updates=strides_,
            )
            strides_ = tf.cast(x=strides_, dtype=tf.int64)

        ##### begin_mask
        begin_bit_mask = tf.constant([2**idx for idx in range(input_tensor_rank)], dtype=tf.int32)
        cliped_values = tf.cast(1-tf.clip_by_value(t=begin_,clip_value_min=0,clip_value_max=1), dtype=tf.int32)
        begin_mask_ = tf.cast(
            tf.math.reduce_sum(
                input_tensor=tf.math.multiply(x=cliped_values, y=begin_bit_mask),
            ),
            dtype=tf.int32,
        )
        if hasattr(begin_mask_, '_inferred_value') and begin_mask_._inferred_value == [None]:
            begin_mask_ = 0

        ##### end_mask
        end_bit_mask = tf.constant([2**idx for idx in range(input_tensor_rank)], dtype=tf.int32)
        cliped_values = tf.cast(1-tf.clip_by_value(t=end_,clip_value_min=0,clip_value_max=1), dtype=tf.int32)
        end_mask_ = tf.cast(
            tf.math.reduce_sum(
                input_tensor=tf.math.multiply(x=cliped_values, y=end_bit_mask),
            ),
            dtype=tf.int32,
        )
        if hasattr(end_mask_, '_inferred_value') and end_mask_._inferred_value == [None]:
            end_mask_ = 0

        # strided_slice
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            tf.strided_slice(
                input_=input_tensor,
                begin=begin_,
                end=end_,
                strides=strides_,
                begin_mask=begin_mask_,
                end_mask=end_mask_,
                name=graph_node.name,
            )
        # FlexStridedSlice generation suppression process
        if input_tensor.shape != tf.TensorShape(None):
            COMPRESSION_DEFAULT_VALUE = 5
            onnx_slice_dims_count = 0
            if isinstance(starts, np.ndarray):
                onnx_slice_dims_count = len(starts)
            elif hasattr(starts, 'numpy'):
                onnx_slice_dims_count = len(starts.numpy())
            elif isinstance(starts, int):
                onnx_slice_dims_count = 1
            elif tf.keras.backend.is_keras_tensor(starts):
                onnx_slice_dims_count = len(starts.shape)
            else:
                onnx_slice_dims_count = len(starts)

            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                stridedslice_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    begin=begin_,
                    end=end_,
                    strides=strides_,
                    begin_mask=begin_mask_,
                    end_mask=end_mask_,
                    ignore_axes=axes,
                    compression_defult_value=COMPRESSION_DEFAULT_VALUE,
                    onnx_slice_dims_count=onnx_slice_dims_count,
                    output_shape=tf_layers_dict[graph_node_output.name]['tf_node'].shape,
                    name=graph_node.name,
                    **kwargs,
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
