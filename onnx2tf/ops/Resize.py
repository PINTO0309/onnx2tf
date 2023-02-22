import sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
from tensorflow.keras.layers import Lambda # type: ignore
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    get_replacement_parameter,
    replace_parameter,
    get_constant_or_variable,
    print_node_info,
    inverted_operation_enable_disable,
    make_tf_node_info,
    upsampling2d_bilinear,
    upsampling2d_bicubic,
    upsampling2d_nearest,
    upsampling3d_bilinear,
    upsampling3d_bicubic,
    upsampling3d_nearest,
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
    """Resize

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

    opset = kwargs['opset']

    graph_node_input = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    input_tensor = tf_layers_dict[graph_node_input.name]['tf_node'] \
        if isinstance(graph_node_input, gs.Variable) else graph_node_input
    input_tensor_shape = input_tensor.shape
    input_tensor_rank = len(input_tensor_shape)

    # Generate input OPs for TensorFlow subgraphs
    # For inference testing on OP stand-alone
    tf_partial_model_inputs: List[tf.keras.Input] = \
        make_tf_partial_model_inputs(
            input_tensors=[input_tensor]
        )
    tf_partial_model_tensors = None
    tf_partial_model_outputs = None

    # Workaround to avoid as many Resize failures as possible
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
    if len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 4:
                # 2D - Overall model
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,1],
                    **kwargs,
                )
                before_op_output_shape_trans = True
                # 2D - Partial model
                if tf_partial_model_inputs is not None:
                    tf_partial_model_tensors = \
                        transpose_with_flexing_deterrence(
                            input_tensor=tf_partial_model_inputs[0],
                            perm=[0,2,3,1],
                            **kwargs,
                        )
            elif len(onnx_input_shape) == 5:
                # 3D - Overall model
                input_tensor = transpose_with_flexing_deterrence(
                    input_tensor=input_tensor,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )
                before_op_output_shape_trans = True
                # 3D - Partial model
                if tf_partial_model_inputs is not None:
                    tf_partial_model_tensors = \
                        transpose_with_flexing_deterrence(
                            input_tensor=tf_partial_model_inputs[0],
                            perm=[0,2,3,4,1],
                            **kwargs,
                        )

    roi = None
    scales = None
    if len(graph_node.inputs) >= 2:
        if opset > 10:
            roi = get_constant_or_variable(
                graph_node.inputs[1],
                before_op_output_shape_trans,
            )
        else:
            scales = get_constant_or_variable(
                graph_node.inputs[1],
                before_op_output_shape_trans,
            )
    if len(graph_node.inputs) >= 3:
        scales = get_constant_or_variable(
            graph_node.inputs[2],
            before_op_output_shape_trans,
        )
    sizes = None
    if len(graph_node.inputs) >= 4:
        sizes = get_constant_or_variable(
            graph_node.inputs[3],
            before_op_output_shape_trans,
        )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    roi = tf_layers_dict[roi.name]['tf_node'] \
        if (isinstance(roi, gs.Variable) and roi.name != '') else roi
    scales = tf_layers_dict[scales.name]['tf_node'] \
        if (isinstance(scales, gs.Variable) and scales.name != '') else scales
    sizes = tf_layers_dict[sizes.name]['tf_node'] \
        if isinstance(sizes, gs.Variable) else sizes

    coordinate_transformation_mode = graph_node.attrs.get('coordinate_transformation_mode', 'half_pixel')
    extrapolation_value = graph_node.attrs.get('extrapolation_value', 0.0)
    mode = graph_node.attrs.get('mode', 'nearest')

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
    if mode.lower() == "linear":
        mode = tf.image.ResizeMethod.BILINEAR
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_bilinear
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_bilinear
        else:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
                'Pull requests are welcome. \n'+
                f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
            )
            sys.exit(1)
    elif mode.lower() == "cubic":
        mode = tf.image.ResizeMethod.BICUBIC
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_bicubic
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_bicubic
        else:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
                'Pull requests are welcome. \n'+
                f'graph_node.name: {graph_node.name} shape: {input_tensor_shape}'
            )
            sys.exit(1)
    else:
        mode = tf.image.ResizeMethod.NEAREST_NEIGHBOR
        if input_tensor_rank == 4:
            tf_resize = upsampling2d_nearest
        elif input_tensor_rank == 5:
            tf_resize = upsampling3d_nearest
        else:
            print(
                f'{Color.RED}ERROR:{Color.RESET} '+
                f'Currently, Resize operations other than 4D and 5D are not supported. '+
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
            new_size = tf.cast(sizes[1:input_tensor_rank-1], tf.int32)
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
        roi = replace_parameter(
            value_before_replacement=roi,
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
        new_size = replace_parameter(
            value_before_replacement=new_size,
            param_target='inputs',
            param_name=graph_node.inputs[3].name,
            **kwargs,
        )

    coordinate_transformation_mode = replace_parameter(
        value_before_replacement=coordinate_transformation_mode,
        param_target='attributes',
        param_name='coordinate_transformation_mode',
        **kwargs,
    )
    extrapolation_value = replace_parameter(
        value_before_replacement=extrapolation_value,
        param_target='attributes',
        param_name='extrapolation_value',
        **kwargs,
    )
    mode = replace_parameter(
        value_before_replacement=mode,
        param_target='attributes',
        param_name='mode',
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
    boxes = None
    box_indices = None
    tf_op_type = None
    align_corners = None
    half_pixel_centers = None
    org_dtype = input_tensor.dtype
    if coordinate_transformation_mode == "tf_crop_and_resize":
        # get boxes for crop
        indices = [1,2,5,6]
        boxes = tf.expand_dims(tf.gather(roi, indices, axis=0), 0)
        # get box_indices for crop
        box_indices = tf.cast(tf.range(0, input_tensor_shape[0]), dtype=tf.int32)
        ### Overall model
        # run crop and resize
        resized_tensor = tf.image.crop_and_resize(
            images=input_tensor,
            boxes=boxes,
            box_indices=box_indices,
            crop_size=new_size,
            method=mode,
            extrapolation_value=extrapolation_value,
            name=graph_node.name,
        )
        tf_op_type = tf.image.crop_and_resize
        ### Partial model
        if tf_partial_model_inputs is not None:
            tf_partial_model_outputs = \
                [
                    tf.image.crop_and_resize(
                        images=tf_partial_model_tensors \
                            if tf_partial_model_tensors is not None else tf_partial_model_inputs[0],
                        boxes=boxes,
                        box_indices=box_indices,
                        crop_size=new_size,
                        method=mode,
                        extrapolation_value=extrapolation_value,
                    )
                ]

    elif coordinate_transformation_mode == "align_corners":
        align_corners = True
        half_pixel_centers = False
        ### Overall model
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': graph_node.name,
            }
        )(input_tensor)
        tf_op_type = tf_resize
        ### Partial model
        if tf_partial_model_inputs is not None:
            tf_partial_model_outputs = \
                [
                    Lambda(
                        tf_resize,
                        arguments={
                            'new_size': new_size,
                            'align_corners': align_corners,
                            'half_pixel_centers': half_pixel_centers,
                            'name': graph_node.name,
                        }
                    )(tf_partial_model_tensors \
                        if tf_partial_model_tensors is not None else tf_partial_model_inputs[0]
                    )
                ]

    elif coordinate_transformation_mode == "asymmetric":
        align_corners = False
        half_pixel_centers = False
        ### Overall model
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': graph_node.name,
            }
        )(input_tensor)
        tf_op_type = tf_resize
        ### Partial model
        if tf_partial_model_inputs is not None:
            tf_partial_model_outputs = \
                [
                    Lambda(
                        tf_resize,
                        arguments={
                            'new_size': new_size,
                            'align_corners': align_corners,
                            'half_pixel_centers': half_pixel_centers,
                            'name': graph_node.name,
                        }
                    )(tf_partial_model_tensors \
                        if tf_partial_model_tensors is not None else tf_partial_model_inputs[0]
                    )
                ]

    elif coordinate_transformation_mode == "half_pixel":
        align_corners = False
        half_pixel_centers = True
        ### Overall model
        resized_tensor = Lambda(
            tf_resize,
            arguments={
                'new_size': new_size,
                'align_corners': align_corners,
                'half_pixel_centers': half_pixel_centers,
                'name': graph_node.name,
            }
        )(input_tensor)
        tf_op_type = tf_resize
        ### Partial model
        if tf_partial_model_inputs is not None:
            tf_partial_model_outputs = \
                [
                    Lambda(
                        tf_resize,
                        arguments={
                            'new_size': new_size,
                            'align_corners': align_corners,
                            'half_pixel_centers': half_pixel_centers,
                            'name': graph_node.name,
                        }
                    )(tf_partial_model_tensors \
                        if tf_partial_model_tensors is not None else tf_partial_model_inputs[0]
                    )
                ]

    else:
        ### Overall model
        resized_tensor = tf.image.resize(
            images=input_tensor,
            size=new_size,
            method=mode,
            name=graph_node.name,
        )
        tf_op_type = tf.image.resize
        ### Partial model
        if tf_partial_model_inputs is not None:
            tf_partial_model_outputs = \
                [
                    tf.image.resize(
                        images=tf_partial_model_tensors \
                            if tf_partial_model_tensors is not None else tf_partial_model_inputs[0],
                        size=new_size,
                        method=mode,
                    )
                ]

    ### Partial model
    if tf_partial_model_inputs is not None:
        tf_partial_model = tf.keras.Model(
            inputs=tf_partial_model_inputs,
            outputs=tf_partial_model_outputs,
        )
        test_data = None
        if not isinstance(input_tensor, np.ndarray):
            if not isinstance(graph_node_input, np.ndarray) \
                and 'verification_data' in tf_layers_dict[graph_node_input.name].keys():
                test_data: np.ndarray = tf_layers_dict[graph_node_input.name]['verification_data']
            elif isinstance(graph_node_input, np.ndarray):
                test_data: np.ndarray = graph_node_input
            else:
                test_data = None
        else:
            test_data = input_tensor
        # TF dummy inference
        tf_tensor_infos: Dict[Any] = dummy_tf_inference(
            model=tf_partial_model,
            inputs=tf_partial_model_inputs,
            verification_datas=[
                test_data
            ]
        )
        tf_partial_model_result: np.ndarray = list(tf_tensor_infos.values())[0]
        cast_dtype = TF_DTYPES_TO_NUMPY_DTYPES[org_dtype] \
            if isinstance(org_dtype, tf.dtypes.DType) else org_dtype
        tf_partial_model_result = tf_partial_model_result.astype(cast_dtype)
        tf_layers_dict[graph_node_output.name]['verification_data'] = tf_partial_model_result
        del tf_partial_model
        del tf_partial_model_inputs
        del tf_partial_model_outputs
        del test_data

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
                'tf_op_type': tf_op_type,
                'tf_inputs': {
                    'images': input_tensor,
                    'boxes': boxes,
                    'box_indices': box_indices,
                    'new_size/crop_size': new_size,
                    'method': mode,
                    'extrapolation_value': extrapolation_value,
                    'align_corners': align_corners,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
