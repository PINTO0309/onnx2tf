import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
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
    make_tf_partial_model_inputs,
    dummy_tf_inference,
    transpose_with_flexing_deterrence,
)
from typing import Any, Dict, List
from onnx2tf.utils.colors import Color

INF_INDEX_VALUE: int = 4294967296


@print_node_info
@inverted_operation_enable_disable
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Conv

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
    input_weights = get_weights_constant_or_variable(
        const_or_var=graph_node.inputs[1],
        kernel_size=kernel_size,
    )
    input_bias = None
    if len(graph_node.inputs) >= 3:
        input_bias = get_constant_or_variable(
            graph_node.inputs[2],
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
    input_bias = tf_layers_dict[input_bias.name]['tf_node'] \
        if isinstance(input_bias, gs.Variable) else input_bias

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
    ]
    tf_input_shape = [
        dim if isinstance(dim, int) else None for dim in input_tensor_shape
    ]
    tf_transposed_perm = None
    test_data_transposed_perm = None
    if len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 3:
                # 1D - Overall model
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,1],
                    **kwargs,
                )
                tf_transposed_perm = [0,2,1]
            elif len(onnx_input_shape) == 4:
                # 2D - Overall model
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,1],
                    **kwargs,
                )
                tf_transposed_perm = [0,2,3,1]
            elif len(onnx_input_shape) == 5:
                # 3D - Overall model
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )
                tf_transposed_perm = [0,2,3,4,1]
    else:
        if len(onnx_input_shape) == 3:
            test_data_transposed_perm = [0,2,1]
        elif len(onnx_input_shape) == 4:
            test_data_transposed_perm = [0,2,3,1]
        elif len(onnx_input_shape) == 5:
            test_data_transposed_perm = [0,2,3,4,1]

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
    if auto_pad == 'NOTSET':
        if input_tensor_rank >=2 \
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
            f'{Color.RED}ERROR:{Color.RESET} ' +\
            f'Invalid auto_pad attribute: {auto_pad}'
        print(error_msg)
        assert False, error_msg
    else:
        error_msg = f'' +\
            f'{Color.RED}ERROR:{Color.RESET} ' +\
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

    # Generate input OPs for TensorFlow subgraphs
    # For inference testing on OP stand-alone
    if kwargs['acc_check']:
        tf_partial_model_inputs: List[tf.keras.Input] = \
            make_tf_partial_model_inputs(
                input_tensors=[
                    input_tensor,
                ]
            )
        tf_partial_model_tensors = tf_partial_model_inputs[0] \
            if tf_partial_model_inputs is not None else None
        tf_partial_model_outputs = None

    def tf_partial_model_inference(
        *,
        tf_partial_model_inputs,
        tf_partial_model_outputs,
        tf_layers_dict,
    ):
        tf_partial_model = tf.keras.Model(
            inputs=tf_partial_model_inputs,
            outputs=tf_partial_model_outputs,
        )
        test_data = None
        if not isinstance(input_tensor, np.ndarray):
            if not isinstance(graph_node_input, np.ndarray) \
                and graph_node_input.name in tf_layers_dict \
                and 'verification_data' in tf_layers_dict[graph_node_input.name].keys():
                test_data: np.ndarray = tf_layers_dict[graph_node_input.name]['verification_data']
            elif isinstance(graph_node_input, np.ndarray):
                test_data: np.ndarray = graph_node_input
            else:
                test_data = None
        else:
            test_data = input_tensor
        if tf_transposed_perm is not None \
            and isinstance(test_data, np.ndarray):
            test_data = test_data.transpose(tf_transposed_perm)
        elif test_data_transposed_perm is not None \
            and isinstance(test_data, np.ndarray) \
            and tf_input_shape != list(test_data.shape):
            test_data = test_data.transpose(test_data_transposed_perm)
        if padded \
            and isinstance(test_data, np.ndarray):
            test_data = get_padding_as_op(
                x=test_data,
                pads=pads,
            )
        tf_partial_model_result_infos: Dict[Any] = dummy_tf_inference(
            model=tf_partial_model,
            inputs=tf_partial_model_inputs,
            verification_datas=[
                test_data
            ]
        )
        tf_layers_dict[graph_node_output.name]['verification_data'] = \
            list(tf_partial_model_result_infos.values())[0]
        del tf_partial_model
        del tf_partial_model_inputs
        del tf_partial_model_outputs
        del test_data

    # Conv
    tf_op_type = None
    if input_bias is not None:
        if not depthwise:
            if group == 1:
                # Conv1D, Conv2D, Conv3D - Bias Add
                ### Overall model
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.add(
                        tf.nn.convolution(
                            input=input_tensor,
                            filters=input_weights,
                            strides=strides,
                            padding=pad_mode,
                            dilations=dilations,
                        ),
                        input_bias,
                    )
                tf_op_type = tf.nn.convolution
                ### Partial model
                if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                    tf_partial_model_outputs = \
                        [
                            tf.add(
                                tf.nn.convolution(
                                    input=tf_partial_model_tensors,
                                    filters=input_weights,
                                    strides=strides,
                                    padding=pad_mode,
                                    dilations=dilations,
                                ),
                                input_bias,
                            )
                        ]
                    tf_partial_model_inference(
                        tf_partial_model_inputs=tf_partial_model_inputs,
                        tf_partial_model_outputs=tf_partial_model_outputs,
                        tf_layers_dict=tf_layers_dict,
                    )

            else:
                if kernel_size in (1, 2, 3) and not disable_group_convolution:
                    print(
                        f'{Color.YELLOW}WARNING:{Color.RESET} ' +\
                        f'This model contains GroupConvolution and is automatically optimized for TFLite, ' +
                        f'but is not output because saved_model does not support GroupConvolution. ' +
                        f'If saved_model is needed, specify --disable_group_convolution to retransform the model.'
                    )
                # GroupedConvolution - Conv1D, Conv2D, Conv3D - Bias Add
                if kernel_size == 1 and not disable_group_convolution:
                    ### Overall model
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.add(
                            x=Conv1D(
                                filters=input_weights.shape[-1],
                                kernel_size=input_weights.shape[:1],
                                strides=strides,
                                padding=pad_mode.lower(),
                                dilation_rate=dilations,
                                groups=group,
                                use_bias=False,
                                kernel_initializer=tf.keras.initializers.constant(input_weights),
                                name=graph_node.name,
                            )(input_tensor),
                            y=input_bias,
                        )
                    tf_op_type = 'GroupedConvolution1D'
                    ### Partial model
                    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                        tf_partial_model_outputs = \
                            [
                                tf.add(
                                    x=Conv1D(
                                        filters=input_weights.shape[-1],
                                        kernel_size=input_weights.shape[:1],
                                        strides=strides,
                                        padding=pad_mode.lower(),
                                        dilation_rate=dilations,
                                        groups=group,
                                        use_bias=False,
                                        kernel_initializer=tf.keras.initializers.constant(input_weights),
                                        name=graph_node.name,
                                    )(tf_partial_model_tensors),
                                    y=input_bias,
                                )
                            ]
                        tf_partial_model_inference(
                            tf_partial_model_inputs=tf_partial_model_inputs,
                            tf_partial_model_outputs=tf_partial_model_outputs,
                            tf_layers_dict=tf_layers_dict,
                        )

                elif kernel_size == 2 and not disable_group_convolution:
                    ### Overall model
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.add(
                            x=Conv2D(
                                filters=input_weights.shape[-1],
                                kernel_size=input_weights.shape[:2],
                                strides=strides,
                                padding=pad_mode.lower(),
                                dilation_rate=dilations,
                                groups=group,
                                use_bias=False,
                                kernel_initializer=tf.keras.initializers.constant(input_weights),
                                name=graph_node.name,
                            )(input_tensor),
                            y=input_bias,
                        )
                    tf_op_type = 'GroupedConvolution2D'
                    ### Partial model
                    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                        tf_partial_model_outputs = \
                            [
                                tf.add(
                                    x=Conv2D(
                                        filters=input_weights.shape[-1],
                                        kernel_size=input_weights.shape[:2],
                                        strides=strides,
                                        padding=pad_mode.lower(),
                                        dilation_rate=dilations,
                                        groups=group,
                                        use_bias=False,
                                        kernel_initializer=tf.keras.initializers.constant(input_weights),
                                        name=graph_node.name,
                                    )(tf_partial_model_tensors),
                                    y=input_bias,
                                )
                            ]
                        tf_partial_model_inference(
                            tf_partial_model_inputs=tf_partial_model_inputs,
                            tf_partial_model_outputs=tf_partial_model_outputs,
                            tf_layers_dict=tf_layers_dict,
                        )

                # TODO: As of TensorFlow Lite v2.10.0, GroupedConvolution3D is converted to FlexConv3D.
                # TODO: Uncomment out when TensorFlow Lite officially supports GroupedConvolution3D.
                # elif kernel_size == 3 and not disable_group_convolution:
                #     ### Overall model
                #     tf_layers_dict[graph_node_output.name]['tf_node'] = \
                #         tf.add(
                #             x=Conv3D(
                #                 filters=input_weights.shape[-1],
                #                 kernel_size=input_weights.shape[:3],
                #                 strides=strides,
                #                 padding=pad_mode.lower(),
                #                 dilation_rate=dilations,
                #                 groups=group,
                #                 use_bias=False,
                #                 kernel_initializer=tf.keras.initializers.constant(input_weights),
                #                 name=graph_node.name,
                #             )(input_tensor),
                #             y=input_bias,
                #         )
                #     tf_op_type = 'GroupedConvolution3D'
                #     ### Partial model
                #     if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                #         tf_partial_model_outputs = \
                #             [
                #                 tf.add(
                #                     x=Conv3D(
                #                         filters=input_weights.shape[-1],
                #                         kernel_size=input_weights.shape[:3],
                #                         strides=strides,
                #                         padding=pad_mode.lower(),
                #                         dilation_rate=dilations,
                #                         groups=group,
                #                         use_bias=False,
                #                         kernel_initializer=tf.keras.initializers.constant(input_weights),
                #                         name=graph_node.name,
                #                     )(tf_partial_model_tensors),
                #                     y=input_bias,
                #                 )
                #             ]
                #         tf_partial_model_inference(
                #             tf_partial_model_inputs=tf_partial_model_inputs,
                #             tf_partial_model_outputs=tf_partial_model_outputs,
                #             tf_layers_dict=tf_layers_dict,
                #         )

                else:
                    # SeparableConv
                    ### Overall model
                    input_tensor_splits = tf.split(input_tensor, num_or_size_splits=group, axis=-1)
                    weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        tf.add(
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
                            ),
                            input_bias,
                        )
                    tf_op_type = tf.nn.convolution
                    ### Partial model
                    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                        partial_input_tensor_splits = tf.split(tf_partial_model_tensors, num_or_size_splits=group, axis=-1)
                        partial_weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)
                        tf_partial_model_outputs = \
                            [
                                tf.add(
                                    tf.concat(
                                        values=[
                                            tf.nn.convolution(
                                                input=partial_input_tensor_split,
                                                filters=partial_weight_split,
                                                padding=pad_mode,
                                                strides=strides,
                                                dilations=dilations,
                                            ) for (partial_input_tensor_split, partial_weight_split) in zip(partial_input_tensor_splits, partial_weight_splits)
                                        ],
                                        axis=-1
                                    ),
                                    input_bias,
                                )
                            ]
                        tf_partial_model_inference(
                            tf_partial_model_inputs=tf_partial_model_inputs,
                            tf_partial_model_outputs=tf_partial_model_outputs,
                            tf_layers_dict=tf_layers_dict,
                        )

        else:
            # DepthwiseConv2D
            ### Overall model
            strides = [1] + strides + [1]
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.add(
                    tf.nn.depthwise_conv2d(
                        input=input_tensor,
                        filter=input_weights,
                        padding=pad_mode,
                        strides=strides,
                        dilations=dilations,
                    ),
                    input_bias,
                )
            tf_op_type = tf.nn.depthwise_conv2d
            ### Partial model
            if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                tf_partial_model_outputs = \
                    [
                        tf.add(
                            tf.nn.depthwise_conv2d(
                                input=tf_partial_model_tensors,
                                filter=input_weights,
                                padding=pad_mode,
                                strides=strides,
                                dilations=dilations,
                            ),
                            input_bias,
                        )
                    ]
                tf_partial_model_inference(
                    tf_partial_model_inputs=tf_partial_model_inputs,
                    tf_partial_model_outputs=tf_partial_model_outputs,
                    tf_layers_dict=tf_layers_dict,
                )

    else:
        if not depthwise:
            if group == 1:
                # Conv1D, Conv2D, Conv3D - No Bias
                ### Overall model
                tf_layers_dict[graph_node_output.name]['tf_node'] = \
                    tf.nn.convolution(
                        input=input_tensor,
                        filters=input_weights,
                        strides=strides,
                        padding=pad_mode,
                        dilations=dilations,
                    )
                tf_op_type = tf.nn.convolution
                ### Partial model
                if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                    tf_partial_model_outputs = \
                        [
                            tf.nn.convolution(
                                input=tf_partial_model_tensors,
                                filters=input_weights,
                                strides=strides,
                                padding=pad_mode,
                                dilations=dilations,
                            )
                        ]
                    tf_partial_model_inference(
                        tf_partial_model_inputs=tf_partial_model_inputs,
                        tf_partial_model_outputs=tf_partial_model_outputs,
                        tf_layers_dict=tf_layers_dict,
                    )

            else:
                if kernel_size in (1, 2, 3) and not disable_group_convolution:
                    print(
                        f'{Color.YELLOW}WARNING:{Color.RESET} ' +\
                        f'This model contains GroupConvolution and is automatically optimized for TFLite, ' +
                        f'but is not output because saved_model does not support GroupConvolution. ' +
                        f'If saved_model is needed, specify --disable_group_convolution to retransform the model.'
                    )
                # GroupedConvolution - Conv1D, Conv2D, Conv3D - No Bias
                ### Overall model
                if kernel_size == 1 and not disable_group_convolution:
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        Conv1D(
                            filters=input_weights.shape[-1],
                            kernel_size=input_weights.shape[:1],
                            strides=strides,
                            padding=pad_mode.lower(),
                            dilation_rate=dilations,
                            groups=group,
                            use_bias=False,
                            kernel_initializer=tf.keras.initializers.constant(input_weights),
                            name=graph_node.name,
                        )(input_tensor)
                    tf_op_type = 'GroupedConvolution1D'
                    ### Partial model
                    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                        tf_partial_model_outputs = \
                            [
                                Conv1D(
                                    filters=input_weights.shape[-1],
                                    kernel_size=input_weights.shape[:1],
                                    strides=strides,
                                    padding=pad_mode.lower(),
                                    dilation_rate=dilations,
                                    groups=group,
                                    use_bias=False,
                                    kernel_initializer=tf.keras.initializers.constant(input_weights),
                                    name=graph_node.name,
                                )(tf_partial_model_tensors)
                            ]
                        tf_partial_model_inference(
                            tf_partial_model_inputs=tf_partial_model_inputs,
                            tf_partial_model_outputs=tf_partial_model_outputs,
                            tf_layers_dict=tf_layers_dict,
                        )

                elif kernel_size == 2 and not disable_group_convolution:
                    ### Overall model
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
                        Conv2D(
                            filters=input_weights.shape[-1],
                            kernel_size=input_weights.shape[:2],
                            strides=strides,
                            padding=pad_mode.lower(),
                            dilation_rate=dilations,
                            groups=group,
                            use_bias=False,
                            kernel_initializer=tf.keras.initializers.constant(input_weights),
                            name=graph_node.name,
                        )(input_tensor)
                    tf_op_type = 'GroupedConvolution2D'
                    ### Partial model
                    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                        tf_partial_model_outputs = \
                            [
                                Conv2D(
                                    filters=input_weights.shape[-1],
                                    kernel_size=input_weights.shape[:2],
                                    strides=strides,
                                    padding=pad_mode.lower(),
                                    dilation_rate=dilations,
                                    groups=group,
                                    use_bias=False,
                                    kernel_initializer=tf.keras.initializers.constant(input_weights),
                                    name=graph_node.name,
                                )(tf_partial_model_tensors)
                            ]
                        tf_partial_model_inference(
                            tf_partial_model_inputs=tf_partial_model_inputs,
                            tf_partial_model_outputs=tf_partial_model_outputs,
                            tf_layers_dict=tf_layers_dict,
                        )

                # TODO: As of TensorFlow Lite v2.10.0, GroupedConvolution3D is converted to FlexConv3D.
                # TODO: Uncomment out when TensorFlow Lite officially supports GroupedConvolution3D.
                # elif kernel_size == 3 and not disable_group_convolution:
                #     ### Overall model
                #     tf_layers_dict[graph_node_output.name]['tf_node'] = \
                #         Conv3D(
                #             filters=input_weights.shape[-1],
                #             kernel_size=input_weights.shape[:3],
                #             strides=strides,
                #             padding=pad_mode.lower(),
                #             dilation_rate=dilations,
                #             groups=group,
                #             use_bias=False,
                #             kernel_initializer=tf.keras.initializers.constant(input_weights),
                #             name=graph_node.name,
                #         )(input_tensor)
                #     tf_op_type = 'GroupedConvolution3D'
                #     ### Partial model
                #     if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                #         tf_partial_model_outputs = \
                #             [
                #                 Conv3D(
                #                     filters=input_weights.shape[-1],
                #                     kernel_size=input_weights.shape[:3],
                #                     strides=strides,
                #                     padding=pad_mode.lower(),
                #                     dilation_rate=dilations,
                #                     groups=group,
                #                     use_bias=False,
                #                     kernel_initializer=tf.keras.initializers.constant(input_weights),
                #                     name=graph_node.name,
                #                 )(tf_partial_model_tensors)
                #             ]
                #         tf_partial_model_inference(
                #             tf_partial_model_inputs=tf_partial_model_inputs,
                #             tf_partial_model_outputs=tf_partial_model_outputs,
                #             tf_layers_dict=tf_layers_dict,
                #         )

                else:
                    # SeparableConv
                    ### Overall model
                    input_tensor_splits = tf.split(input_tensor, num_or_size_splits=group, axis=-1)
                    weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)
                    tf_layers_dict[graph_node_output.name]['tf_node'] = \
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
                    tf_op_type = tf.nn.convolution
                    ### Partial model
                    if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                        partial_input_tensor_splits = tf.split(tf_partial_model_tensors, num_or_size_splits=group, axis=-1)
                        partial_weight_splits = tf.split(input_weights, num_or_size_splits=group, axis=-1)
                        tf_partial_model_outputs = \
                            [
                                tf.concat(
                                    values=[
                                        tf.nn.convolution(
                                            input=partial_input_tensor_split,
                                            filters=partial_weight_split,
                                            padding=pad_mode,
                                            strides=strides,
                                            dilations=dilations,
                                        ) for (partial_input_tensor_split, partial_weight_split) in zip(partial_input_tensor_splits, partial_weight_splits)
                                    ],
                                    axis=-1
                                )
                            ]
                        tf_partial_model_inference(
                            tf_partial_model_inputs=tf_partial_model_inputs,
                            tf_partial_model_outputs=tf_partial_model_outputs,
                            tf_layers_dict=tf_layers_dict,
                        )

        else:
            # DepthwiseConv2D
            ### Overall model
            strides = [1] + strides + [1]
            tf_layers_dict[graph_node_output.name]['tf_node'] = \
                tf.nn.depthwise_conv2d(
                    input=input_tensor,
                    filter=input_weights,
                    padding=pad_mode,
                    strides=strides,
                    dilations=dilations,
                )
            tf_op_type = tf.nn.depthwise_conv2d
            ### Partial model
            if kwargs['acc_check'] and tf_partial_model_inputs is not None:
                tf_partial_model_outputs = \
                    [
                        tf.nn.depthwise_conv2d(
                            input=tf_partial_model_tensors,
                            filter=input_weights,
                            padding=pad_mode,
                            strides=strides,
                            dilations=dilations,
                        )
                    ]
                tf_partial_model_inference(
                    tf_partial_model_inputs=tf_partial_model_inputs,
                    tf_partial_model_outputs=tf_partial_model_outputs,
                    tf_layers_dict=tf_layers_dict,
                )

    # Generation of Debug Info
    tf_layers_dict[graph_node_output.name]['tf_node_info'] = \
        make_tf_node_info(
            node_info={
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'input': input_tensor,
                    'weights': input_weights,
                    'bias': input_bias,
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
