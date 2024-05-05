import sys
import copy
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import itertools
import tensorflow as tf
import tf_keras
from tensorflow.python.keras.layers import (
    Conv1D,
    Conv2D,
    Conv3D,
)
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_constant_or_variable,
    get_weights_constant_or_variable,
    get_padding_as_op,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    dummy_tf_inference,
    transpose_with_flexing_deterrence,
    get_tf_model_inputs,
    onnx_tf_tensor_validation,
)
from typing import Any, Dict
from onnx2tf.utils.logging import *

INF_INDEX_VALUE: int = 4294967296


def _apply_zero_point(base, zero_point):
    base = tf.cast(base, tf.float32)
    zero_point = tf.cast(zero_point, tf.float32)
    return base - zero_point


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """ConvInteger

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
    kernel_shape = graph_node.attrs.get('kernel_shape', [])
    kernel_size = len(kernel_shape)
    try:
        input_weights = get_weights_constant_or_variable(
            const_or_var=graph_node.inputs[1] \
                if graph_node.i(1).op != 'DequantizeLinear' \
                    else tf.transpose(
                            tf_layers_dict[graph_node.inputs[1].name]['tf_node'],
                            perm=[0]+[len(tf_layers_dict[graph_node.inputs[1].name]['tf_node'].shape)-1]+[i for i in range(1, len(tf_layers_dict[graph_node.inputs[1].name]['tf_node'].shape)-1)]
                        ),
            kernel_size=kernel_size,
        )
    except Exception as ex:
        input_weights = get_weights_constant_or_variable(
            const_or_var=graph_node.inputs[1],
            kernel_size=kernel_size,
        )
    x_zero_point = None
    if len(graph_node.inputs) >= 3:
        x_zero_point = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
            is_bias=True,
        )
    w_zero_point = None
    if len(graph_node.inputs) >= 4:
        w_zero_point = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
            is_bias=True,
        )


    graph_node_output: gs.Variable = graph_node.outputs[0]
    output_tensor_shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_weights = tf_layers_dict[input_weights.name]['tf_node'] \
        if isinstance(input_weights, gs.Variable) else input_weights

    x_zero_point = tf_layers_dict[x_zero_point.name]['tf_node'] \
        if isinstance(x_zero_point, gs.Variable) else x_zero_point
    w_zero_point = tf_layers_dict[w_zero_point.name]['tf_node'] \
        if isinstance(w_zero_point, gs.Variable) else w_zero_point
    w_zero_point_shape = w_zero_point.shape
    w_zero_point_rank = len(w_zero_point_shape)

    # Apply x_zero_point first
    input_tensor = _apply_zero_point(input_tensor, x_zero_point) \
        if x_zero_point is not None else tf.cast(input_tensor, tf.float32)

    # Apply w_zero_point next
    if w_zero_point is None:
        input_weights = tf.cast(input_weights, tf.float32)
    elif w_zero_point_rank == 0:
        input_weights = _apply_zero_point(input_weights, w_zero_point)
    elif w_zero_point_rank == 1:
        tensor_list = []
        process_shape = [1] + [input_weights.shape[i] for i in range(1, len(input_weights.shape))]
        for i in range(input_weights.shape[0]):
            out_tensor = _apply_zero_point(input_weights[i], w_zero_point[i])
        tensor_list.append(tf.reshape(out_tensor, process_shape))
        input_weights = tf.concat(tensor_list, 0)

    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)
    spatial_size = input_tensor_rank - 2
    input_weights_shape = input_weights.shape
    auto_pad = graph_node.attrs.get('auto_pad', 'NOTSET')
    dilations = graph_node.attrs.get('dilations', [1] * spatial_size)
    group = graph_node.attrs.get('group', 1)
    pads = graph_node.attrs.get('pads', [0, 0] * spatial_size)
    strides = graph_node.attrs.get('strides', [1] * spatial_size)

    disable_group_convolution: bool = kwargs['disable_group_convolution']
    onnx_tensor_infos_for_validation: Dict[str: np.ndarray] = kwargs['onnx_tensor_infos_for_validation']
    test_data_nhwc: np.ndarray = kwargs['test_data_nhwc']
    custom_input_op_name_np_data_path: str = kwargs['custom_input_op_name_np_data_path']
    disable_strict_mode: bool = kwargs['disable_strict_mode']

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': output_tensor_shape,
        'dtype': dtype,
        'nhwc': True,
    }

    # Generation of TF OP

    # Workaround to avoid as many conversion failures as possible
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

    all_axes_same = False
    if onnx_input_shape is not None \
        and len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 3:
                # 1D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,1],
                    **kwargs,
                )

            elif len(onnx_input_shape) == 4:
                # 2D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,1],
                    **kwargs,
                )

            elif len(onnx_input_shape) == 5:
                # 3D
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )
        else:
            all_axes_same = True
            # Get the output tensor of one previous OP of TensorFlow only once
            if not disable_strict_mode:
                tf_model_inputs = get_tf_model_inputs(
                    tf_layers_dict=tf_layers_dict,
                )
                val_model = None
                if not isinstance(input_tensor, np.ndarray):
                    val_model = tf_keras.Model(
                        inputs=tf_model_inputs,
                        outputs=[
                            input_tensor,
                        ],
                    )
                else:
                    pass

            # TF dummy inference
            #   Get the output tensor of the previous layer of MatMul
            #   If input.1 and input.2 are both layers, tf_pre_tensor_infos is 2 cases
            #   If one of input.1 or input.2 is np.ndarray, tf_pre_tensor_infos is 1 case
            tf_pre_tensor_infos = {}
            if not disable_strict_mode:
                try:
                    tf_pre_tensor_infos: Dict[Any] = dummy_tf_inference(
                        model=val_model,
                        inputs=tf_model_inputs,
                        test_data_nhwc=test_data_nhwc,
                        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
                    )
                except Exception as ex:
                    pass
                del val_model

            # Get np.ndarray for validation
            validation_data = None
            if not disable_strict_mode:
                if len(tf_pre_tensor_infos) == 1:
                    if not isinstance(input_tensor, np.ndarray):
                        validation_data = list(tf_pre_tensor_infos.values())[0]
                    else:
                        validation_data = copy.deepcopy(input_tensor)

                # Get ONNX inference results
                onnx_tensor_infos = None
                if onnx_tensor_infos_for_validation is not None \
                    and onnx_tensor_infos_for_validation.get(graph_node_output.name, None) is not None:
                    onnx_tensor_infos = {
                        graph_node_output.name: onnx_tensor_infos_for_validation[graph_node_output.name]
                    }
                    del onnx_tensor_infos_for_validation
    """
    Conv1D
    Conv2D
    Conv3D
    DepthwiseConv2D
    GroupConv1D
    GroupConv2D
    GroupConv3D
    SeparableConv
    """
    # Check auto_pad nonexistent or NOTSET first
    pad_mode = 'VALID'
    padded = False

    # for onnx pads that diff on the side of axes
    # for example [3,1] or [3,3,3,1]
    pads_axes_opposite_same = True
    for axis_begin, axis_end in zip(pads[0:spatial_size:1], pads[spatial_size::1]):
        if axis_begin != axis_end:
            pads_axes_opposite_same = False
            break

    if auto_pad == 'NOTSET':
        if pads_axes_opposite_same \
            and input_tensor_rank >=2 \
            and graph_node.inputs[0].shape is not None \
            and graph_node.inputs[0].shape[2:] == output_tensor_shape[2:]:
            pad_mode = "SAME"
        elif pads != [0, 0] * spatial_size:
            input_tensor = get_padding_as_op(
                x=input_tensor,
                pads=pads,
            )
            pad_mode = 'VALID'
            padded = True
        else:
            pad_mode = 'VALID'
    # Then we use auto_pad to setup pad_mode
    elif auto_pad == "SAME_UPPER":
        pad_mode = "SAME"
    elif auto_pad == "VALID":
        pad_mode = "VALID"
    elif auto_pad == "SAME_LOWER":
        error_msg = f'' +\
            Color.RED(f'ERROR:') + ' ' +\
            f'Invalid auto_pad attribute: {auto_pad}'
        print(error_msg)
        assert False, error_msg
    else:
        error_msg = f'' +\
            Color.RED(f'ERROR:') + ' ' +\
            f'Invalid auto_pad attribute: {auto_pad}'
        print(error_msg)
        assert False, error_msg

    # DepthwiseConv2D
    #   1. rank=4
    #   2. group>1
    #   3. No undefined dimension
    #   4. All strides spatial shape are the same number
    depthwise = (
        input_tensor_rank == 4 \
        and len(input_weights_shape) == 4 \
        and group != 1 \
        and not (None in input_weights_shape) \
        and sum([1 if s == strides[0] else 0 for s in strides]) == len(strides)
    )
    if depthwise and input_tensor_shape[-1] != None:
        depthwise = bool(group == input_tensor_shape[-1])

    if depthwise is True:
        depthwise_filter_shape = list(input_weights_shape[0:2]) + [-1, input_weights_shape[3] // group]
        input_weights = tf.reshape(input_weights, depthwise_filter_shape)

    input_weights = input_weights \
        if not isinstance(input_weights, np.ndarray) \
            else tf.convert_to_tensor(input_weights)


    def conv_nobias(input_tensor, input_weights, strides, pad_mode, dilations):
        return \
            tf.nn.convolution(
                input=input_tensor,
                filters=input_weights,
                strides=strides,
                padding=pad_mode,
                dilations=dilations,
            )

    def group_conv1d_nobias(input_weights, strides, pad_mode, dilations, group, graph_node, input_tensor):
        return \
            Conv1D(
                filters=input_weights.shape[-1],
                kernel_size=input_weights.shape[:1],
                strides=strides,
                padding=pad_mode.lower(),
                dilation_rate=dilations,
                groups=group,
                use_bias=False,
                kernel_initializer=tf_keras.initializers.constant(input_weights),
                name=graph_node.name,
            )(input_tensor)

    def group_conv2d_nobias(input_weights, strides, pad_mode, dilations, group, graph_node, input_tensor):
        return \
            Conv2D(
                filters=input_weights.shape[-1],
                kernel_size=input_weights.shape[:2],
                strides=strides,
                padding=pad_mode.lower(),
                dilation_rate=dilations,
                groups=group,
                use_bias=False,
                kernel_initializer=tf_keras.initializers.constant(input_weights),
                name=graph_node.name,
            )(input_tensor)

    def sep_conv_nobias(input_tensor, input_weights, pad_mode, strides, dilations):
        input_tensor_splits = tf.split(input_tensor, num_or_size_splits=group, axis=-1)
        weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)
        return \
            tf.concat(
                values=[
                    tf.nn.convolution(
                        input=input_tensor_split,
                        filters=weight_split,
                        padding=pad_mode,
                        strides=strides,
                        dilations=dilations,
                    ) for (input_tensor_split, weight_split) in zip(input_tensor_splits, weight_splits)
                ],
                axis=-1
            )

    def depth_conv_nobias(input_tensor, input_weights, pad_mode, strides, dilations):
        return \
            tf.nn.depthwise_conv2d(
                input=input_tensor,
                filter=input_weights,
                padding=pad_mode,
                strides=strides,
                dilations=dilations,
            )


    # Conv
    tf_op_type = None
    error_check_tf_op_type = ''

    if not depthwise:
        if group == 1:
            # Conv1D, Conv2D, Conv3D - No Bias
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                conv_nobias(
                    input_tensor,
                    input_weights,
                    strides,
                    pad_mode,
                    dilations,
                )
            tf_op_type = tf.nn.convolution
            error_check_tf_op_type = 'conv_nobias'

        else:
            if kernel_size in (1, 2, 3) and not disable_group_convolution:
                warn(
                    f'This model contains GroupConvolution and is automatically optimized for TFLite, ' +
                    f'but is not output because saved_model does not support GroupConvolution. ' +
                    f'If saved_model is needed, specify --disable_group_convolution to retransform the model.'
                )
            # GroupedConvolution - Conv1D, Conv2D, Conv3D - No Bias
            if kernel_size == 1 and not disable_group_convolution:
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    group_conv1d_nobias(
                        input_weights,
                        strides,
                        pad_mode,
                        dilations,
                        group,
                        graph_node,
                        input_tensor,
                    )
                tf_op_type = 'GroupedConvolution1D'
                error_check_tf_op_type = 'group_conv1d_nobias'

            elif kernel_size == 2 and not disable_group_convolution:
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    group_conv2d_nobias(
                        input_weights,
                        strides,
                        pad_mode,
                        dilations,
                        group,
                        graph_node,
                        input_tensor,
                    )
                tf_op_type = 'GroupedConvolution2D'
                error_check_tf_op_type = 'group_conv2d_nobias'

            # TODO: As of TensorFlow Lite v2.10.0, GroupedConvolution3D is converted to FlexConv3D.
            # TODO: Uncomment out when TensorFlow Lite officially supports GroupedConvolution3D.
            # elif kernel_size == 3 and not disable_group_convolution:
            #     tf_layers_dict[graph_node_output.name]['tf_node'] = \
            #         Conv3D(
            #             filters=input_weights.shape[-1],
            #             kernel_size=input_weights.shape[:3],
            #             strides=strides,
            #             padding=pad_mode.lower(),
            #             dilation_rate=dilations,
            #             groups=group,
            #             use_bias=False,
            #             kernel_initializer=tf_keras.initializers.constant(input_weights),
            #             name=graph_node.name,
            #         )(input_tensor)
            #     tf_op_type = 'GroupedConvolution3D'
            #     error_check_tf_op_type = 'group_conv3d_nobias'

            else:
                # SeparableConv
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    sep_conv_nobias(
                        input_tensor,
                        input_weights,
                        pad_mode,
                        strides,
                        dilations,
                    )
                tf_op_type = tf.nn.convolution
                error_check_tf_op_type = 'sep_conv_nobias'

    else:
        # DepthwiseConv2D
        strides = [1] + strides + [1]
        tf_layers_dict[graph_node_output.name]['tf_node'] = \
            depth_conv_nobias(
                input_tensor,
                input_weights,
                pad_mode,
                strides,
                dilations,
            )
        tf_op_type = tf.nn.depthwise_conv2d
        error_check_tf_op_type = 'depth_conv_nobias'

    # Automatic correction of accuracy degradation
    min_abs_err = sys.maxsize
    min_abs_err_perm_1: int = [idx for idx in range(input_tensor_rank)]

    if not disable_strict_mode and all_axes_same:
        if onnx_tensor_infos is not None and validation_data is not None:
            tensor_1_candidate_for_transpositions = list(itertools.permutations(range(input_tensor_rank)))
            tensor_1_candidate_for_transpositions = [
                trans_perm for trans_perm in tensor_1_candidate_for_transpositions \
                    if trans_perm[0] == 0
            ]
            # Search for the axis with the smallest error
            for tensor_1_candidate_for_transposition in tensor_1_candidate_for_transpositions:
                try:
                    target_validation_data = validation_data.transpose(tensor_1_candidate_for_transposition)
                    # Build TF dummy model
                    input = tf_keras.Input(
                        shape=target_validation_data.shape[1:],
                        batch_size=target_validation_data.shape[0] \
                            if isinstance(target_validation_data.shape[0], int) else None,
                        name='dummy_input',
                        dtype=target_validation_data.dtype,
                    )
                    # op_type
                    tmp_conv_op = None
                    if error_check_tf_op_type == 'conv_nobias':
                        tmp_conv_op = \
                            conv_nobias(
                                input,
                                input_weights,
                                strides,
                                pad_mode,
                                dilations,
                            )
                    elif error_check_tf_op_type == 'group_conv1d_nobias':
                        tmp_conv_op = \
                            group_conv1d_nobias(
                                input_weights,
                                strides,
                                pad_mode,
                                dilations,
                                group,
                                graph_node,
                                input,
                            )
                    elif error_check_tf_op_type == 'group_conv2d_nobias':
                        tmp_conv_op = \
                            group_conv2d_nobias(
                                input_weights,
                                strides,
                                pad_mode,
                                dilations,
                                group,
                                graph_node,
                                input,
                            )
                    elif error_check_tf_op_type == 'sep_conv_nobias':
                        tmp_conv_op = \
                            sep_conv_nobias(
                                input,
                                input_weights,
                                pad_mode,
                                strides,
                                dilations,
                            )
                    elif error_check_tf_op_type == 'depth_conv_nobias':
                        tmp_conv_op = \
                            depth_conv_nobias(
                                input,
                                input_weights,
                                pad_mode,
                                strides,
                                dilations,
                            )
                    # define model
                    val_model = tf_keras.Model(
                        inputs=[
                            input,
                        ],
                        outputs=[
                            tmp_conv_op
                        ],
                    )
                    # TF dummy inference
                    tf_tensor_infos: Dict[Any] = dummy_tf_inference(
                        model=val_model,
                        inputs=[
                            input,
                        ],
                        verification_datas=[
                            target_validation_data,
                        ],
                    )
                    del input
                    del val_model

                    # Validation
                    onnx_tf_output_pairs = {
                        (oi[0], ti[0]): (oi[1], ti[1]) \
                            for oi, ti in zip(onnx_tensor_infos.items(), tf_tensor_infos.items())
                    }
                    """
                    check_results: Dict[str, List[np.ndarray, int, float|int]]
                        {
                            onnx_output_name: [
                                onnx_tensor,
                                matched_flg, <--- 0: Unmatched, 1: Matched, 2: Skipped (Deleted or Shape Unmatched)
                                max_abs_err,
                            ]
                        }
                    """
                    check_results = onnx_tf_tensor_validation(
                        output_pairs=onnx_tf_output_pairs,
                        rtol=0.0,
                        atol=0.0,
                    )
                    result_err = sum([val[2] for val in check_results.values()])
                    if result_err < min_abs_err:
                        min_abs_err = result_err
                        min_abs_err_perm_1 = list(tensor_1_candidate_for_transposition)
                        if min_abs_err < 1e-3:
                            break
                except Exception as ex:
                    pass

        input_tensor = tf.transpose(a=input_tensor, perm=min_abs_err_perm_1)
        if error_check_tf_op_type == 'conv_nobias':
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                conv_nobias(
                    input_tensor,
                    input_weights,
                    strides,
                    pad_mode,
                    dilations,
                )
        elif error_check_tf_op_type == 'group_conv1d_nobias':
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                group_conv1d_nobias(
                    input_weights,
                    strides,
                    pad_mode,
                    dilations,
                    group,
                    graph_node,
                    input_tensor,
                )
        elif error_check_tf_op_type == 'group_conv2d_nobias':
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                group_conv2d_nobias(
                    input_weights,
                    strides,
                    pad_mode,
                    dilations,
                    group,
                    graph_node,
                    input_tensor,
                )
        elif error_check_tf_op_type == 'sep_conv_nobias':
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                sep_conv_nobias(
                    input_tensor,
                    input_weights,
                    pad_mode,
                    strides,
                    dilations,
                )
        elif error_check_tf_op_type == 'depth_conv_nobias':
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                depth_conv_nobias(
                    input_tensor,
                    input_weights,
                    pad_mode,
                    strides,
                    dilations,
                )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'input': input_tensor,
                    'weights': input_weights,
                    'strides': strides,
                    'dilations': dilations,
                    'padding': pad_mode,
                    'group': group,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
