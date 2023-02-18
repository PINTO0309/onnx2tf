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
)
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
        before_op_output_shape_trans,
    )
    graph_node_output: gs.Variable = graph_node.outputs[0]
    shape = graph_node_output.shape
    dtype = graph_node_output.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_node_output.name] = {
        'optype': graph_node.op,
        'shape': shape,
        'dtype': dtype,
    }

    image = tf_layers_dict[graph_node_input_1.name]['tf_node'] \
        if isinstance(graph_node_input_1, gs.Variable) else graph_node_input_1
    grid = tf_layers_dict[graph_node_input_2.name]['tf_node'] \
        if isinstance(graph_node_input_2, gs.Variable) else graph_node_input_2

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

    if not align_corners:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'The current implementation of GridSample supports only align_corners=1. '+
            f'Pull requests are welcome. \n'+
            f'align_corners: 0'
        )
        sys.exit(1)

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


    # Generation of TF OP
    """
    image
        [N, H, W, C]
    grid
        [N, grid_H, grid_W, 2]
    """
    Nt, H, W, C = image.shape
    grid_H = grid.shape[1]
    grid_W = grid.shape[2]
    xgrid, ygrid = tf.split(
        value=grid,
        num_or_size_splits=2,
        axis=-1,
    )
    mask = tf.cast(
        (xgrid >= 0) & (ygrid >= 0) & (xgrid < W - 1) & (ygrid < H - 1),
        dtype=tf.float32,
    )
    x0 = tf.math.floor(xgrid)
    x1 = x0 + 1
    y0 = tf.math.floor(ygrid)
    y1 = y0 + 1

    wa = transpose_with_flexing_deterrence(
        input_tensor=(x1 - xgrid) * (y1 - ygrid),
        perm=[3, 0, 1, 2],
        **kwargs,
    )
    wb = transpose_with_flexing_deterrence(
        input_tensor=(x1 - xgrid) * (ygrid - y0),
        perm=[3, 0, 1, 2],
        **kwargs,
    )
    wc = transpose_with_flexing_deterrence(
        input_tensor=(xgrid - x0) * (y1 - ygrid),
        perm=[3, 0, 1, 2],
        **kwargs,
    )
    wd = transpose_with_flexing_deterrence(
        input_tensor=(xgrid - x0) * (ygrid - y0),
        perm=[3, 0, 1, 2],
        **kwargs,
    )

    x0 = tf.cast(
        tf.reshape(
            tensor=(x0 * mask),
            shape=[Nt, grid_H, grid_W],
        ),
        dtype=tf.int64,
    )
    y0 = tf.cast(
        tf.reshape(
            tensor=(y0 * mask),
            shape=[Nt, grid_H, grid_W]
        ),
        dtype=tf.int64,
    )
    x1 = tf.cast(
        tf.reshape(
            tensor=(x1 * mask),
            shape=[Nt, grid_H, grid_W]
        ),
        dtype=tf.int64,
    )
    y1 = tf.cast(
        tf.reshape(
            tensor=(y1 * mask),
            shape=[Nt, grid_H, grid_W]
        ),
        dtype=tf.int64,
    )

    ind = np.arange(Nt)
    ind = tf.reshape(tensor=ind, shape=[Nt, 1])
    ind = tf.tile(input=ind, multiples=[1, grid_H])
    ind = tf.reshape(tensor=ind, shape=[Nt, grid_H, 1])
    ind = tf.tile(input=ind, multiples=[1, 1, grid_W])
    ind = tf.cast(ind, dtype=tf.int64)

    ### common
    temp_image = tf.reshape(
        tensor=image,
        shape=[-1, image.shape[-1]],
    )
    temp_ind = ind * H * W
    ### wa
    temp_y0 = y0 * W
    temp_x0 = x0
    temp_x0y0ind = temp_x0 + temp_y0 + temp_ind
    temp_gather_wa = tf.gather(
        params=temp_image,
        indices=temp_x0y0ind,
    )
    temp_reshape1_wa = tf.reshape(
        tensor=temp_gather_wa,
        shape=[-1, temp_gather_wa.shape[3]],
    )
    temp_traspose_wa = transpose_with_flexing_deterrence(
        input_tensor=temp_reshape1_wa,
        perm=[1,0],
        **kwargs,
    )
    temp_reshape2_wa = tf.reshape(
        tensor=temp_traspose_wa,
        shape=[
            temp_gather_wa.shape[3],
            temp_gather_wa.shape[0],
            temp_gather_wa.shape[1],
            temp_gather_wa.shape[2],
        ],
    )
    temp_wa = temp_reshape2_wa * wa
    ### wb
    temp_y1 = y1 * W
    temp_x0 = x0
    temp_ind_y1x0 = temp_x0 + temp_y1 + temp_ind
    temp_gather_wb = tf.gather(
        params=temp_image,
        indices=temp_ind_y1x0,
    )
    temp_reshape1_wb = tf.reshape(
        tensor=temp_gather_wb,
        shape=[-1, temp_gather_wb.shape[3]],
    )
    temp_traspose_wb = transpose_with_flexing_deterrence(
        input_tensor=temp_reshape1_wb,
        perm=[1,0],
        **kwargs,
    )
    temp_reshape2_wb = tf.reshape(
        tensor=temp_traspose_wb,
        shape=[
            temp_gather_wb.shape[3],
            temp_gather_wb.shape[0],
            temp_gather_wb.shape[1],
            temp_gather_wb.shape[2],
        ],
    )
    temp_wb = temp_reshape2_wb * wb
    ### wc
    temp_y0 = y0 * W
    temp_x1 = x1
    temp_ind_y0x1 = temp_x1 + temp_y0 + temp_ind
    temp_gather_wc = tf.gather(
        params=temp_image,
        indices=temp_ind_y0x1,
    )
    temp_reshape1_wc = tf.reshape(
        tensor=temp_gather_wc,
        shape=[-1, temp_gather_wc.shape[3]],
    )
    temp_traspose_wc = transpose_with_flexing_deterrence(
        input_tensor=temp_reshape1_wc,
        perm=[1,0],
        **kwargs,
    )
    temp_reshape2_wc = tf.reshape(
        tensor=temp_traspose_wc,
        shape=[
            temp_gather_wc.shape[3],
            temp_gather_wc.shape[0],
            temp_gather_wc.shape[1],
            temp_gather_wc.shape[2],
        ],
    )
    temp_wc = temp_reshape2_wc * wc
    ### wd
    temp_y1 = y1 * W
    temp_x1 = x1
    temp_ind_y1x1 = temp_x1 + temp_y1 + temp_ind
    temp_gather_wd = tf.gather(
        params=temp_image,
        indices=temp_ind_y1x1,
    )
    temp_reshape1_wd = tf.reshape(
        tensor=temp_gather_wd,
        shape=[-1, temp_gather_wd.shape[3]],
    )
    temp_traspose_wd = transpose_with_flexing_deterrence(
        input_tensor=temp_reshape1_wd,
        perm=[1,0],
        **kwargs,
    )
    temp_reshape2_wd = tf.reshape(
        tensor=temp_traspose_wd,
        shape=[
            temp_gather_wd.shape[3],
            temp_gather_wd.shape[0],
            temp_gather_wd.shape[1],
            temp_gather_wd.shape[2],
        ],
    )
    temp_wd = temp_reshape2_wd * wd
    ### wa + wb + wc + wd
    output_tensor = temp_wa + temp_wb + temp_wc + temp_wd

    output_tensor = transpose_with_flexing_deterrence(
        input_tensor=output_tensor,
        perm=[1,2,3,0],
        **kwargs,
    )
    mask = tf.tile(
        input=mask,
        multiples=[1,1,1,C],
    )

    tf_layers_dict[graph_node_output.name]['tf_node'] = \
        output_tensor = output_tensor * mask

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
