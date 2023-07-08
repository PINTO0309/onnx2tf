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
    get_replacement_parameter,
    pre_process_transpose,
    post_process_transpose,
    transpose_with_flexing_deterrence,
    transpose_with_flexing_deterrence,
)
from onnx2tf.utils.colors import Color

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
    """GridSample

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

    graph_node_input_1 = get_constant_or_variable(
        graph_node.inputs[0],
        before_op_output_shape_trans,
    )
    graph_node_input_2 = get_constant_or_variable(
        graph_node.inputs[1],
        False \
            if hasattr(graph_node.inputs[1], 'values') \
                and isinstance(graph_node.inputs[1].values, np.ndarray) \
                    else before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
        'nhwc': True,
    }

    image = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    grid = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2
    input_tensor_shape = image.shape

    # Pre-process transpose
    image = pre_process_transpose(
        value_before_transpose=image,
        param_target='inputs',
        param_name=graph_node.inputs[0].name,
        **kwargs,
    )
    grid = pre_process_transpose(
        value_before_transpose=grid,
        param_target='inputs',
        param_name=graph_node.inputs[1].name,
        **kwargs,
    )

    align_corners = bool(graph_node.attrs.get('align_corners', 0))
    mode = graph_node.attrs.get('mode', 'bilinear')
    padding_mode = graph_node.attrs.get('padding_mode', 'zeros')

    ENABLE_MODES = ['bilinear']
    if mode not in ENABLE_MODES:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The current implementation of GridSample supports only mode={ENABLE_MODES}. '+
            f'Pull requests are welcome. \n'+
            f'mode: {mode}'
        )
        sys.exit(1)

    ENABLE_PADDING_MODES = ['zeros']
    if padding_mode not in ENABLE_PADDING_MODES:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The current implementation of GridSample supports only mode={ENABLE_PADDING_MODES}. '+
            f'Pull requests are welcome. \n'+
            f'mode: {padding_mode}'
        )
        sys.exit(1)

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
    if onnx_input_shape is not None \
        and len(onnx_input_shape) > 1 and len(tf_input_shape) > 1 \
        and onnx_input_shape == tf_input_shape:

        shape_for_judging_skip = [
            dim if dim is not None else INF_INDEX_VALUE for dim in onnx_input_shape[1:]
        ]
        if shape_for_judging_skip.count(shape_for_judging_skip[0]) != len(shape_for_judging_skip):
            if len(onnx_input_shape) == 3:
                # 1D
                image = transpose_with_flexing_deterrence(
                    input_tensor=image,
                    perm=[0,2,1],
                    **kwargs,
                )
            elif len(onnx_input_shape) == 4:
                # 2D
                image = transpose_with_flexing_deterrence(
                    input_tensor=image,
                    perm=[0,2,3,1],
                    **kwargs,
                )
            elif len(onnx_input_shape) == 5:
                # 3D
                image = transpose_with_flexing_deterrence(
                    input_tensor=image,
                    perm=[0,2,3,4,1],
                    **kwargs,
                )

    # Generation of TF OP
    """
    image
        [N, H, W, C]
    grid
        [N, grid_H, grid_W, 2]
    """
    split11, split12 = tf.split(grid, num_or_size_splits=2, axis=3) # x, y

    if align_corners:
        add1 = tf.math.add(split11, tf.convert_to_tensor(1.0)) # Add_output_0
        mul1 = tf.math.multiply(add1, tf.convert_to_tensor((image.shape[2]-1)*0.5, dtype=tf.float32)) # Mul_output_0
    else:
        add1 = tf.math.add(split11, tf.convert_to_tensor(1.0)) # Add_output_0
        mul00 = tf.math.multiply(add1, tf.convert_to_tensor(image.shape[2], dtype=tf.float32)) # Mul_output_0
        sub1 = tf.math.subtract(mul00, tf.convert_to_tensor(1, dtype=tf.float32)) # Sub_output_0
        mul1 = tf.math.multiply(sub1, tf.convert_to_tensor(0.5, dtype=tf.float32)) # Div_output_0
    reshape1 = tf.reshape(mul1, [tf.shape(mul1)[0], tf.reduce_prod(tf.shape(mul1)[1:])]) # Reshape_output_0

    if align_corners:
        add2 = tf.math.add(split12, tf.convert_to_tensor(1.0)) # Add_1_output_0
        mul2 = tf.math.multiply(add2, tf.convert_to_tensor((image.shape[1]-1)*0.5, dtype=tf.float32)) # Mul_1_output_0
    else:
        add2 = tf.math.add(split12, tf.convert_to_tensor(1.0)) # Add_output_0
        mul01 = tf.math.multiply(add2, tf.convert_to_tensor(image.shape[1], dtype=tf.float32)) # Mul_output_0
        sub2 = tf.math.subtract(mul01, tf.convert_to_tensor(1, dtype=tf.float32)) # Sub_output_0
        mul2 = tf.math.multiply(sub2, tf.convert_to_tensor(0.5, dtype=tf.float32)) # Div_output_0
    reshape2 = tf.reshape(mul2, [tf.shape(mul2)[0], tf.reduce_prod(tf.shape(mul2)[1:])]) # Reshape_1_output_0

    floor1 = tf.math.floor(reshape1) # Floor_output_0
    sub11 = tf.math.subtract(reshape1, floor1) # Sub_3_output_0
    add12 = tf.math.add(floor1, tf.convert_to_tensor(1, dtype=tf.float32)) # Add_2_output_0
    sub12 = tf.math.subtract(add12, reshape1) # Sub_output_0

    floor2 = tf.math.floor(reshape2) # Floor_1_output_0
    sub21 = tf.math.subtract(reshape2, floor2) # Sub_2_output_0
    add22 = tf.math.add(floor2, tf.convert_to_tensor(1, dtype=tf.float32)) # Add_3_output_0
    sub22 = tf.math.subtract(add22, reshape2) # Sub_1_output_0


    # Sub_output_0 Sub_1_output_0 -> Mul_2_output_0
    mul11 = tf.math.multiply(sub12, sub22) # Mul_2_output_0
    unsqueeze11 = tf.expand_dims(mul11, axis=1) # Unsqueeze_output_0

    # Sub_output_0 Sub_2_output_0 -> Mul_3_output_0
    mul12 = tf.math.multiply(sub12, sub21) # Mul_3_output_0
    unsqueeze12 = tf.expand_dims(mul12, axis=1) # Unsqueeze_1_output_0

    # Sub_3_output_0 Sub_2_output_0 -> Mul_5_output_0
    mul21 = tf.math.multiply(sub11, sub21) # Mul_5_output_0
    unsqueeze21 = tf.expand_dims(mul21, axis=1) # Unsqueeze_3_output_0

    # Sub_3_output_0 Sub_1_output_0 -> Mul_4_output_0
    mul22 = tf.math.multiply(sub11, sub22) # Mul_4_output_0
    unsqueeze22 = tf.expand_dims(mul22, axis=1) # Unsqueeze_2_output_0


    # Add_2_output_0 Constant_1_output_0 -> Add_4_output_0
    add31 = tf.math.add(add12, tf.convert_to_tensor(1, dtype=tf.float32)) # Add_4_output_0
    # Add_4_output_0 -> Cast_2_output_0
    cast31 = tf.cast(add31, dtype=tf.int64) # Cast_2_output_0
    # Cast_2_output_0 Constant_26_output_0 -> Less_1_output_0
    less31 = tf.less(cast31, tf.convert_to_tensor(0, dtype=tf.int64)) # Less_1_output_0
    # Less_1_output_0 Constant_26_output_0 Cast_2_output_0 -> Where_2_output_0
    where311 = \
        tf.where(
            condition=less31,
            x=tf.convert_to_tensor(0, dtype=tf.int64),
            y=cast31,
        )
    # Where_2_output_0 Constant_27_output_0 -> Greater_1_output_0
    greter31 = tf.greater(where311, tf.convert_to_tensor(image.shape[2]+1, dtype=tf.int64)) # Greater_1_output_0
    # Greater_1_output_0 Constant_27_output_0 Where_2_output_0 -> Where_3_output_0
    where312 = \
        tf.where(
            condition=greter31,
            x=tf.convert_to_tensor(image.shape[2]+1, dtype=tf.int64),
            y=where311,
        )


    # Add_2_output_0 -> Cast_1_output_0
    cast32 = tf.cast(add12, dtype=tf.int64) # Cast_1_output_0
    # Cast_1_output_0 Constant_26_output_0 -> Less_output_0
    less32 = tf.less(cast32, tf.convert_to_tensor(0, dtype=tf.int64)) # Less_output_0
    # Less_output_0 Constant_26_output_0 Cast_1_output_0 -> Where_output_0
    where321 = \
        tf.where(
            condition=less32,
            x=tf.convert_to_tensor(0, dtype=tf.int64),
            y=cast32,
        )
    # Where_output_0 Constant_27_output_0 -> Greater_output_0
    greter32 = tf.greater(where321, tf.convert_to_tensor(image.shape[2]+1, dtype=tf.int64)) # Greater_output_0
    # Greater_output_0 Constant_27_output_0 Where_output_0 -> Where_1_output_0
    where322 = \
        tf.where(
            condition=greter32,
            x=tf.convert_to_tensor(image.shape[2]+1, dtype=tf.int64),
            y=where321,
        )


    # Add_3_output_0 Constant_1_output_0 -> Add_5_output_0
    add33 = tf.math.add(add22, tf.convert_to_tensor(1, dtype=tf.float32)) # Add_5_output_0
    # Add_5_output_0 -> Cast_4_output_0
    cast33 = tf.cast(add33, dtype=tf.int64) # Cast_4_output_0
    # Cast_4_output_0 Constant_26_output_0 -> Less_3_output_0
    less33 = tf.less(cast33, tf.convert_to_tensor(0, dtype=tf.int64)) # Less_3_output_0
    # Less_3_output_0 Constant_26_output_0 Cast_4_output_0 -> Where_6_output_0
    where331 = \
        tf.where(
            condition=less33,
            x=tf.convert_to_tensor(0, dtype=tf.int64),
            y=cast33,
        )
    # Where_6_output_0 Constant_32_output_0 -> Greater_3_output_0
    greter33 = tf.greater(where331, tf.convert_to_tensor(image.shape[1]+1, dtype=tf.int64)) # Greater_3_output_0
    # Greater_3_output_0 Constant_32_output_0 Where_6_output_0 -> Where_7_output_0
    where332 = \
        tf.where(
            condition=greter33,
            x=tf.convert_to_tensor(image.shape[1]+1, dtype=tf.int64),
            y=where331,
        )
    # Where_7_output_0 Constant_37_output_0 -> Mul_8_output_0
    mul33 = tf.math.multiply(where332, tf.convert_to_tensor(image.shape[2]+2, dtype=tf.int64)) # Mul_8_output_0


    # Add_3_output_0 -> Cast_3_output_0
    cast34 = tf.cast(add22, dtype=tf.int64) # Cast_3_output_0
    # Cast_3_output_0 Constant_26_output_0 -> Less_2_output_0
    less34 = tf.less(cast34, tf.convert_to_tensor(0, dtype=tf.int64)) # Less_2_output_0
    # Less_2_output_0 Constant_26_output_0 Cast_3_output_0 -> Where_4_output_0
    where341 = \
        tf.where(
            condition=less34,
            x=tf.convert_to_tensor(0, dtype=tf.int64),
            y=cast34,
        )
    # Where_4_output_0 Constant_32_output_0 -> Greater_2_output_0
    greter34 = tf.greater(where341, tf.convert_to_tensor(image.shape[1]+1, dtype=tf.int64)) # Greater_2_output_0
    # Greater_2_output_0 Constant_32_output_0 Where_4_output_0 -> Where_5_output_0
    where342 = \
        tf.where(
            condition=greter34,
            x=tf.convert_to_tensor(image.shape[1]+1, dtype=tf.int64),
            y=where341,
        )
    # Where_5_output_0 Constant_37_output_0 -> Mul_6_output_0
    mul34 = tf.math.multiply(where342, tf.convert_to_tensor(image.shape[2]+2, dtype=tf.int64)) # Mul_6_output_0


    # Where_3_output_0 Mul_6_output_0 -> Add_8_output_0
    add41 = tf.math.add(where312, mul34) # Add_8_output_0
    # Add_8_output_0 Constant_11_output_0 -> Unsqueeze_6_output_0
    unsqueeze41 = tf.expand_dims(add41, axis=1) # Unsqueeze_6_output_0
    # Unsqueeze_6_output_0 Where_8_output_0 -> Expand_2_output_0
    expand41_ones = tf.ones([1] + [image.shape[3]] + [1], dtype=tf.int64)
    expand41 = tf.math.multiply(unsqueeze41, expand41_ones) # Expand_2_output_0

    # Where_1_output_0 Mul_6_output_0 -> Add_6_output_0
    add42 = tf.math.add(where322, mul34) # Add_6_output_0
    # Add_6_output_0 Constant_11_output_0 -> Unsqueeze_4_output_0
    unsqueeze42 = tf.expand_dims(add42, axis=1) # Unsqueeze_4_output_0
    # Unsqueeze_4_output_0 Where_8_output_0 -> Expand_output_0
    expand42_ones = tf.ones([1] + [image.shape[3]] + [1], dtype=tf.int64)
    expand42 = tf.math.multiply(unsqueeze42, expand42_ones) # Expand_output_0

    # Where_3_output_0 Mul_8_output_0 -> Add_9_output_0
    add43 = tf.math.add(where312, mul33) # Add_9_output_0
    # Add_9_output_0 Constant_11_output_0 -> Unsqueeze_7_output_0
    unsqueeze43 = tf.expand_dims(add43, axis=1) # Unsqueeze_7_output_0
    # Unsqueeze_7_output_0 Where_8_output_0 -> Expand_3_output_0
    expand43_ones = tf.ones([1] + [image.shape[3]] + [1], dtype=tf.int64)
    expand43 = tf.math.multiply(unsqueeze43, expand43_ones) # Expand_3_output_0

    # Where_1_output_0 Mul_8_output_0 -> Add_7_output_0
    add44 = tf.math.add(where322, mul33) # Add_7_output_0
    # Add_7_output_0 Constant_11_output_0 -> Unsqueeze_5_output_0
    unsqueeze44 = tf.expand_dims(add44, axis=1) # Unsqueeze_5_output_0
    # Unsqueeze_5_output_0 Where_8_output_0 -> Expand_1_output_0
    expand44_ones = tf.ones([1] + [image.shape[3]] + [1], dtype=tf.int64)
    expand44 = tf.math.multiply(unsqueeze44, expand44_ones) # Expand_1_output_0


    ################################################## image
    image_padded = tf.pad(image, paddings=[[0,0],[1,1],[1,1],[0,0]]) # Pad_output_0
    # Pad_output_0 Constant_36_output_0 -> Reshape_4_output_0
    image_reshape = tf.reshape(image_padded, shape=[image_padded.shape[0]] + [np.prod(image_padded.shape[1:3])] + [image_padded.shape[3]])
    image_reshape_transpose = tf.transpose(image_reshape, perm=[0,2,1])


    # Reshape_4_output_0 Expand_3_output_0 -> GatherElements_3_output_0
    axis_perm1 = tf.tensor_scatter_nd_update(
        tf.range(tf.rank(image_reshape_transpose)),
        tf.constant([[0], [2]]),
        tf.constant([2, 0])
    )
    data_swaped1 = tf.transpose(image_reshape_transpose, perm=axis_perm1)
    index_swaped1 = tf.transpose(expand43, perm=axis_perm1)
    idx_tensors_per_axis1 = [
        tf.range(tf.shape(index_swaped1, index_swaped1.dtype)[i]) \
            for i in range(index_swaped1.shape.rank)
    ]
    idx_tensors_per_axis1 = tf.meshgrid(
        *idx_tensors_per_axis1,
        indexing='ij',
    )
    idx_tensors_per_axis1[0] = index_swaped1
    dim_expanded_idx_tensors_per_axis1 = [
        tf.expand_dims(idx_tensor, axis=-1)
        for idx_tensor in idx_tensors_per_axis1
    ]
    index_expanded1 = tf.concat(dim_expanded_idx_tensors_per_axis1, axis=-1)
    gathernd1 = tf.gather_nd(data_swaped1, index_expanded1)
    gatherelements1 = tf.transpose(gathernd1, perm=[2,1,0]) # GatherElements_3_output_0


    # Reshape_4_output_0 Expand_2_output_0 -> GatherElements_2_output_0
    axis_perm2 = tf.tensor_scatter_nd_update(
        tf.range(tf.rank(image_reshape_transpose)),
        tf.constant([[0], [2]]),
        tf.constant([2, 0])
    )
    data_swaped2 = tf.transpose(image_reshape_transpose, perm=axis_perm2)
    index_swaped2 = tf.transpose(expand41, perm=axis_perm2)
    idx_tensors_per_axis2 = [
        tf.range(tf.shape(index_swaped2, index_swaped2.dtype)[i]) \
            for i in range(index_swaped2.shape.rank)
    ]
    idx_tensors_per_axis2 = tf.meshgrid(
        *idx_tensors_per_axis2,
        indexing='ij',
    )
    idx_tensors_per_axis2[0] = index_swaped2
    dim_expanded_idx_tensors_per_axis2 = [
        tf.expand_dims(idx_tensor, axis=-1)
        for idx_tensor in idx_tensors_per_axis2
    ]
    index_expanded2 = tf.concat(dim_expanded_idx_tensors_per_axis2, axis=-1)
    gathernd2 = tf.gather_nd(data_swaped2, index_expanded2)
    gatherelements2 = tf.transpose(gathernd2, perm=[2,1,0]) # GatherElements_2_output_0


    # Reshape_4_output_0 Expand_1_output_0 -> GatherElements_1_output_0
    axis_perm3 = tf.tensor_scatter_nd_update(
        tf.range(tf.rank(image_reshape_transpose)),
        tf.constant([[0], [2]]),
        tf.constant([2, 0])
    )
    data_swaped3 = tf.transpose(image_reshape_transpose, perm=axis_perm3)
    index_swaped3 = tf.transpose(expand44, perm=axis_perm3)
    idx_tensors_per_axis3 = [
        tf.range(tf.shape(index_swaped3, index_swaped3.dtype)[i]) \
            for i in range(index_swaped3.shape.rank)
    ]
    idx_tensors_per_axis3 = tf.meshgrid(
        *idx_tensors_per_axis3,
        indexing='ij',
    )
    idx_tensors_per_axis3[0] = index_swaped3
    dim_expanded_idx_tensors_per_axis3 = [
        tf.expand_dims(idx_tensor, axis=-1)
        for idx_tensor in idx_tensors_per_axis3
    ]
    index_expanded3 = tf.concat(dim_expanded_idx_tensors_per_axis3, axis=-1)
    gathernd3 = tf.gather_nd(data_swaped3, index_expanded3)
    gatherelements3 = tf.transpose(gathernd3, perm=[2,1,0]) # GatherElements_1_output_0


    # Reshape_4_output_0 Expand_output_0 -> GatherElements_output_0
    axis_perm4 = tf.tensor_scatter_nd_update(
        tf.range(tf.rank(image_reshape_transpose)),
        tf.constant([[0], [2]]),
        tf.constant([2, 0])
    )
    data_swaped4 = tf.transpose(image_reshape_transpose, perm=axis_perm4)
    index_swaped4 = tf.transpose(expand42, perm=axis_perm4)
    idx_tensors_per_axis4 = [
        tf.range(tf.shape(index_swaped4, index_swaped4.dtype)[i]) \
            for i in range(index_swaped4.shape.rank)
    ]
    idx_tensors_per_axis4 = tf.meshgrid(
        *idx_tensors_per_axis4,
        indexing='ij',
    )
    idx_tensors_per_axis4[0] = index_swaped4
    dim_expanded_idx_tensors_per_axis4 = [
        tf.expand_dims(idx_tensor, axis=-1)
        for idx_tensor in idx_tensors_per_axis4
    ]
    index_expanded4 = tf.concat(dim_expanded_idx_tensors_per_axis4, axis=-1)
    gathernd4 = tf.gather_nd(data_swaped4, index_expanded4)
    gatherelements4 = tf.transpose(gathernd4, perm=[2,1,0]) # GatherElements_output_0


    # GatherElements_3_output_0 Unsqueeze_3_output_0 -> Mul_15_output_0
    mul51 = tf.math.multiply(gatherelements1, unsqueeze21)
    # GatherElements_2_output_0 Unsqueeze_2_output_0 -> Mul_14_output_0
    mul52 = tf.math.multiply(gatherelements2, unsqueeze22)
    # GatherElements_1_output_0 Unsqueeze_1_output_0 -> Mul_13_output_0
    mul53 = tf.math.multiply(gatherelements3, unsqueeze12)
    # GatherElements_output_0 Unsqueeze_output_0 -> Mul_12_output_0
    mul54 = tf.math.multiply(gatherelements4, unsqueeze11)


    # Mul_12_output_0 Mul_13_output_0 -> Add_10_output_0
    add61 = tf.math.add(mul54, mul53)
    # Add_10_output_0 Mul_14_output_0 -> Add_11_output_0
    add62 = tf.math.add(add61, mul52)
    # Add_11_output_0 Mul_15_output_0 -> Add_12_output_0
    add63 = tf.math.add(add62, mul51)

    # Add_12_output_0 Constant_55_output_0 -> output_tensor
    output_shape = [
        image.shape[0],
        image.shape[3],
        grid.shape[1],
        grid.shape[2],
    ]
    final_reshape = tf.reshape(add63, shape=output_shape)
    # NCHW -> NHWC
    output = tf.transpose(final_reshape, perm=[0,2,3,1])

    tf_layers_dict[graph_node_output.name]['tf_node'] = output

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
                'tf_op_type': 'GridSample',
                'tf_inputs': {
                    'image': image,
                    'grid': grid,
                    'align_corners': align_corners,
                    'mode': mode,
                    'padding_mode': padding_mode,
                },
                'tf_outputs': {
                    'output': tf_layers_dict[graph_node_output.name]['tf_node'],
                },
            }
        )
