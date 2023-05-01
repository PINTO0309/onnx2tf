import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    pre_process_transpose,
    post_process_transpose,
    make_tf_partial_model_inputs,
    dummy_tf_inference,
    transpose_with_flexing_deterrence,
)
from typing import Any, Dict, List
from onnx2tf.utils.colors import Color
from onnx2tf.utils.enums import (
    NUMPY_DTYPES_TO_TF_DTYPES,
    TF_DTYPES_TO_NUMPY_DTYPES,
)

INF_INDEX_VALUE: int = 4294967296


@print_node_info
@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ScaleAndTranslate

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
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    # Workaround to avoid as many Resize failures as possible
    # for models with useless Transpose immediately before them.
    # If the input geometry of the ONNX and the input geometry of the TF model match,
    # the input geometry on the TF model side is forcibly transposed to the NWC or NHWC or NDHWC format.
    # However, if all dimensions of CW or CHW or CDHW have the same value,
    # the forced transposition process is skipped because it may destroy the structure of the model.
    onnx_input_shape = [
        dim if isinstance(dim, int) else None for dim in graph_node.inputs[0].shape
    ] if graph_node.inputs[0].shape is not None else None
    tf_input_shape = [
        dim if isinstance(dim, int) else None for dim in input_tensor_shape
    ]
    if onnx_input_shape is not None \
        and len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 4:
                # 2D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,1],
                    **kwargs,
                )
                before_op_output_shape_trans = True

            elif len(onnx_input_shape) == 5:
                # 3D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )
                before_op_output_shape_trans = True

    sizes = None
    if len(graph_node.inputs) >= 2:
        sizes = get_constant_or_variable(
            graph_node.inputs[1],
            before_op_output_shape_trans,
        )
    scales = None
    if len(graph_node.inputs) >= 3:
        scales = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    translation = None
    if len(graph_node.inputs) >= 4:
        translation = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )

    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    sizes = tf_layers_dict[sizes.name]['tf_node'] \
        if isinstance(sizes, gs.Variable) else sizes
    scales = tf_layers_dict[scales.name]['tf_node'] \
        if (isinstance(scales, gs.Variable) and scales.name != '') else scales
    translation = tf_layers_dict[translation.name]['tf_node'] \
        if (isinstance(translation, gs.Variable) and translation.name != '') else translation

    antialias = bool(graph_node.attrs.get('antialias', 1))
    kernel_type = graph_node.attrs.get('kernel_type', 'lanczos3')
    if kernel_type == 'triangle':
        kernel_type = 'bilinear'

    replace_argmax_to_fused_argmax_and_indicies_is_int64 = \
        kwargs['replace_argmax_to_fused_argmax_and_indicies_is_int64']
    replace_argmax_to_fused_argmax_and_indicies_is_float32 = \
        kwargs['replace_argmax_to_fused_argmax_and_indicies_is_float32']
    fused_argmax_scale_ratio = \
        kwargs['fused_argmax_scale_ratio']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Generation of TF OP
    if 4 <= input_tensor_rank <= 5:
        pass
    else:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'Currently, ScaleAndTranslate operations other than 4D and 5D are not supported. '+
            'Pull requests are welcome. \n'+
            f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
        )
        sys.exit(1)

    if sizes is not None:
        # sizes is defined
        # The number of elements of 'sizes' should be the same as the rank of input 'X'
        if isinstance(sizes, gs.Variable):
            sizes = sizes.set_shape(input_tensor_shape.shape)
            new_size = tf.cast(sizes[1:input_tensor_rank-1], tf.int32)
        elif isinstance(sizes, np.ndarray):
            new_size = tf.cast(sizes, tf.int32)
        elif tf.keras.backend.is_keras_tensor(sizes):
            new_size = tf.cast(tf.slice(sizes, [1], [input_tensor_rank-2]), tf.int32)
    elif scales is not None:
        # only scales is defined
        if hasattr(graph_node_output, 'shape') \
            and graph_node_output.shape is not None:
            numeric_bools = np.asarray([isinstance(graph_node_output.shape[-(idx+1)], int) for idx in range(input_tensor_rank-2)])
            if numeric_bools.all():
                new_size = graph_node_output.shape[-2:len(graph_node_output.shape)] # Estimated from ONNX output shape
            else:
                h_w_scale = scales[1:input_tensor_rank-1]
                h_w_shape = input_tensor_shape[1:input_tensor_rank-1]
                new_size = tf.cast(
                    h_w_scale * tf.cast(
                        h_w_shape,
                        NUMPY_DTYPES_TO_TF_DTYPES[scales.dtype] \
                            if isinstance(scales.dtype, np.dtype) else scales.dtype,
                    ),
                    tf.int32,
                )
        else:
            h_w_scale = scales[1:input_tensor_rank-1]
            h_w_shape = input_tensor_shape[1:input_tensor_rank-1]
            new_size = tf.cast(
                h_w_scale * tf.cast(
                    h_w_shape,
                    NUMPY_DTYPES_TO_TF_DTYPES[scales.dtype] \
                        if isinstance(scales.dtype, np.dtype) else scales.dtype,
                ),
                tf.int32,
            )

    if hasattr(new_size, '_inferred_value'):
        new_size_values = new_size._inferred_value
        if (new_size_values is None or new_size_values.count(None) == len(new_size_values)) \
            and sum([1 if isinstance(s, str) else 0 for s in graph_node_output.shape[1:input_tensor_rank-1]]) == 0:
            tensor_rank = len(graph_node_output.shape)
            convertion_table = [0] + [i for i in range(2, tensor_rank)] + [1]
            new_values = [0] * tensor_rank
            for new_idx, idx in enumerate(convertion_table):
                new_values[new_idx] = graph_node_output.shape[idx]
            new_size = new_values[-(input_tensor_rank-1):-1]

    if (replace_argmax_to_fused_argmax_and_indicies_is_int64 \
        or replace_argmax_to_fused_argmax_and_indicies_is_float32) \
        and graph_node.o().op == 'ArgMax' \
        and input_tensor_rank == 4:
        new_size = tf.cast(
            tf.cast(new_size, dtype=tf.float32) * fused_argmax_scale_ratio,
            dtype=tf.int32,
        )

    # Param replacement
    input_tensor = replace_parameter(
        value_before_replacement=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    if len(graph_node.inputs) >= 2:
        new_size = replace_parameter(
            value_before_replacement=new_size,
            param_target='inputs',
            param_name=graph_node.inputs[1].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 3:
        scales = replace_parameter(
            value_before_replacement=scales,
            param_target='inputs',
            param_name=graph_node.inputs[2].name,
            **kwargs,
        )
    if len(graph_node.inputs) >= 4:
        translation = replace_parameter(
            value_before_replacement=translation,
            param_target='inputs',
            param_name=graph_node.inputs[3].name,
            **kwargs,
        )

    antialias = replace_parameter(
        value_before_replacement=antialias,
        param_target='attributes',
        param_name='antialias',
        **kwargs,
    )
    kernel_type = replace_parameter(
        value_before_replacement=kernel_type,
        param_target='attributes',
        param_name='kernel_type',
        **kwargs,
    )

    # Pre-process transpose
    input_tensor = pre_process_transpose(
        value_before_transpose=input_tensor,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )

    resized_tensor = None
    org_dtype = input_tensor.dtype
    resized_tensor = tf.image.resize(
        images=input_tensor,
        size=new_size,
        method=kernel_type,
        preserve_aspect_ratio=False,
        antialias=antialias,
        name=graph_node.name,
    )

    # TensorFlow's Resize operation casts to Float32 on its own,
    # so we have to change it back to the original type.
    if org_dtype != resized_tensor.dtype:
        resized_tensor = tf.cast(
            x=resized_tensor,
            dtype=org_dtype,
        )

    tf_layers_dict[graph_node_output.name]['tf_node'] = resized_tensor

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
                'tf_op_type': tf.image.resize,
                'tf_inputs': {
                    'images': input_tensor,
                    'size': new_size,
                    'method': kernel_type,
                    'preserve_aspect_ratio': False,
                    'antialias': antialias,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
