import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
import onnx_graphsurgeon as gs
from typing import List
from onnx2tf.utils.colors import Color
from onnx2tf.utils.common_functions import (
    print_node_info,
)


@print_node_info
def make_node(
    *,
    graph_input: gs.Variable,
    tf_layers_dict: dict,
    keep_ncw_or_nchw_or_ncdhw_input_names: List[str],
    keep_nwc_or_nhwc_or_ndhwc_input_names: List[str],
    keep_shape_absolutely_input_names: List[str],
    **kwargs: dict,
):
    """

    Parameters
    ----------
    graph_input: gs.Variable
        graph_surgeon Variable

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph

    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]]
        Holds the NCW or NCHW or NCDHW of the input shape for the specified INPUT OP names.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n
        Valid only for 3D, 4D and 5D input tensors.\n\n
        e.g. \n
        keep_ncw_or_nchw_or_ncdhw_input_names=['input0','input1','input2']

    keep_nwc_or_nhwc_or_ndhwc_input_names: Optional[List[str]]
        Holds the NWC or NHWC or NDHWC of the input shape for the specified INPUT OP names.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n
        If the input OP name is the same as the input OP name specified\n
        in the keep_ncw_or_nchw_or_ncdhw_input_names option, it is ignored.\n
        Valid only for 3D, 4D and 5D input tensors.\n\n
        e.g. \n
        keep_nwc_or_nhwc_or_ndhwc_input_names=['input0','input1','input2']

    keep_shape_absolutely_input_names: Optional[List[str]]
        Name of the INPUT that unconditionally maintains its shape.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n\n
        e.g.\n
        keep_shape_absolutely_input_names=['input0','input1','input2']
    """
    ncw_nchw_ncdhw_keep = False
    nwc_nhwc_ndhwc_keep = False
    absolutely_keep = False
    batch_size = kwargs.get('batch_size', None)
    graph_input_name = kwargs.get('subgraph_input_name', graph_input.name)

    shape = graph_input.shape
    dtype = graph_input.dtype

    # Overwrite batch or shapes
    if batch_size is not None \
        and len(shape) > 0 \
        and (isinstance(shape[0], str) or shape[0] == -1):
        shape[0] = batch_size

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_input_name] = {
        'optype': 'Input',
        'shape': shape,
        'dtype': dtype,
        'op': None,
    }
    if keep_shape_absolutely_input_names and graph_input_name in keep_shape_absolutely_input_names:
        absolutely_keep = True
    else:
        if graph_input.shape != tf.TensorShape(None) and len(graph_input.shape) in [3, 4, 5] \
            and keep_ncw_or_nchw_or_ncdhw_input_names:
            if graph_input_name in keep_ncw_or_nchw_or_ncdhw_input_names:
                ncw_nchw_ncdhw_keep = True
            else:
                ncw_nchw_ncdhw_keep = False
        elif graph_input.shape != tf.TensorShape(None) and len(graph_input.shape) in [3, 4, 5] \
            and not keep_ncw_or_nchw_or_ncdhw_input_names:
            ncw_nchw_ncdhw_keep = False
        else:
            ncw_nchw_ncdhw_keep = True

        if graph_input.shape != tf.TensorShape(None) and len(graph_input.shape) in [3, 4, 5] \
            and keep_nwc_or_nhwc_or_ndhwc_input_names:
            if graph_input_name in keep_nwc_or_nhwc_or_ndhwc_input_names:
                nwc_nhwc_ndhwc_keep = True
            else:
                nwc_nhwc_ndhwc_keep = False
        elif graph_input.shape != tf.TensorShape(None) and len(graph_input.shape) in [3, 4, 5] \
            and not keep_nwc_or_nhwc_or_ndhwc_input_names:
            nwc_nhwc_ndhwc_keep = False
        else:
            nwc_nhwc_ndhwc_keep = True
        if ncw_nchw_ncdhw_keep and nwc_nhwc_ndhwc_keep:
            nwc_nhwc_ndhwc_keep = False

    tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_keep'] = ncw_nchw_ncdhw_keep
    tf_layers_dict[graph_input_name]['nwc_nhwc_ndhwc_keep'] = nwc_nhwc_ndhwc_keep

    # Generation of TF OP
    tf_input_shape = None
    if graph_input.shape != tf.TensorShape(None) and len(shape) == 3:
        # 3D
        if not ncw_nchw_ncdhw_keep and not nwc_nhwc_ndhwc_keep and not absolutely_keep:
            tf_layers_dict[graph_input_name]['tf_node'] = \
                tf.keras.Input(
                    shape=[
                        shape[2] if isinstance(shape[2], int) else None,
                        shape[1] if isinstance(shape[1], int) else None,
                    ],
                    batch_size=shape[0] if isinstance(shape[0], int) else None,
                    name=graph_input_name,
                    dtype=dtype,
                )
            tf_layers_dict[graph_input_name]['op'] = tf_layers_dict[graph_input_name]['tf_node']
            tf_input_shape = [
                shape[0] if isinstance(shape[0], int) else None,
                shape[2] if isinstance(shape[2], int) else None,
                shape[1] if isinstance(shape[1], int) else None,
            ]
            tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,2,1]
        else:
            ncw = tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in shape[1:]
                ],
                batch_size=shape[0] if isinstance(shape[0], int) else None,
                name=graph_input_name,
                dtype=dtype,
            )
            if not absolutely_keep:
                tf_layers_dict[graph_input_name]['tf_node'] = \
                    tf.transpose(ncw, perm=[0,2,1])
                tf_input_shape = [
                    shape[0] if isinstance(shape[0], int) else None,
                    shape[2] if isinstance(shape[2], int) else None,
                    shape[1] if isinstance(shape[1], int) else None,
                ]
                tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,2,1]
            else:
                tf_layers_dict[graph_input_name]['tf_node'] = ncw
                tf_input_shape = [
                    shape[0] if isinstance(shape[0], int) else None,
                    shape[1] if isinstance(shape[1], int) else None,
                    shape[2] if isinstance(shape[2], int) else None,
                ]
                tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,1,2]
            tf_layers_dict[graph_input_name]['op'] = ncw

    elif graph_input.shape != tf.TensorShape(None) and len(shape) == 4:
        # 4D
        if not ncw_nchw_ncdhw_keep and not nwc_nhwc_ndhwc_keep and not absolutely_keep:
            tf_layers_dict[graph_input_name]['tf_node'] = \
                tf.keras.Input(
                    shape=[
                        shape[2] if isinstance(shape[2], int) else None,
                        shape[3] if isinstance(shape[3], int) else None,
                        shape[1] if isinstance(shape[1], int) else None,
                    ],
                    batch_size=shape[0] if isinstance(shape[0], int) else None,
                    name=graph_input_name,
                    dtype=dtype,
                )
            tf_layers_dict[graph_input_name]['op'] = tf_layers_dict[graph_input_name]['tf_node']
            tf_input_shape = [
                shape[0] if isinstance(shape[0], int) else None,
                shape[2] if isinstance(shape[2], int) else None,
                shape[3] if isinstance(shape[3], int) else None,
                shape[1] if isinstance(shape[1], int) else None,
            ]
            tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,3,1,2]
        else:
            nchw = tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in shape[1:]
                ],
                batch_size=shape[0] if isinstance(shape[0], int) else None,
                name=graph_input_name,
                dtype=dtype,
            )
            if not absolutely_keep:
                tf_layers_dict[graph_input_name]['tf_node'] = \
                    tf.transpose(nchw, perm=[0,2,3,1])
                tf_input_shape = [
                    shape[0] if isinstance(shape[0], int) else None,
                    shape[2] if isinstance(shape[2], int) else None,
                    shape[3] if isinstance(shape[3], int) else None,
                    shape[1] if isinstance(shape[1], int) else None,
                ]
                tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,3,1,2]
            else:
                tf_layers_dict[graph_input_name]['tf_node'] = nchw
                tf_input_shape = [
                    shape[0] if isinstance(shape[0], int) else None,
                    shape[1] if isinstance(shape[1], int) else None,
                    shape[2] if isinstance(shape[2], int) else None,
                    shape[3] if isinstance(shape[3], int) else None,
                ]
                tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,1,2,3]
            tf_layers_dict[graph_input_name]['op'] = nchw

    elif graph_input.shape != tf.TensorShape(None) and len(shape) == 5:
        # 5D
        if not ncw_nchw_ncdhw_keep and not nwc_nhwc_ndhwc_keep and not absolutely_keep:
            tf_layers_dict[graph_input_name]['tf_node'] = \
                tf.keras.Input(
                    shape=[
                        shape[2] if isinstance(shape[2], int) else None,
                        shape[3] if isinstance(shape[3], int) else None,
                        shape[4] if isinstance(shape[4], int) else None,
                        shape[1] if isinstance(shape[1], int) else None,
                    ],
                    batch_size=shape[0] if isinstance(shape[0], int) else None,
                    name=graph_input_name,
                    dtype=dtype,
                )
            tf_layers_dict[graph_input_name]['op'] = tf_layers_dict[graph_input_name]['tf_node']
            tf_input_shape = [
                shape[0] if isinstance(shape[0], int) else None,
                shape[2] if isinstance(shape[2], int) else None,
                shape[3] if isinstance(shape[3], int) else None,
                shape[4] if isinstance(shape[4], int) else None,
                shape[1] if isinstance(shape[1], int) else None,
            ]
            tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,4,1,2,3]
        else:
            ncdhw = tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in shape[1:]
                ],
                batch_size=shape[0],
                name=graph_input_name,
                dtype=dtype,
            )
            if not absolutely_keep:
                tf_layers_dict[graph_input_name]['tf_node'] = \
                    tf.transpose(ncdhw, perm=[0,2,3,4,1])
                tf_input_shape = [
                    shape[0] if isinstance(shape[0], int) else None,
                    shape[2] if isinstance(shape[2], int) else None,
                    shape[3] if isinstance(shape[3], int) else None,
                    shape[4] if isinstance(shape[4], int) else None,
                    shape[1] if isinstance(shape[1], int) else None,
                ]
                tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,4,1,2,3]
            else:
                tf_layers_dict[graph_input_name]['tf_node'] = ncdhw
                tf_input_shape = [
                    shape[0] if isinstance(shape[0], int) else None,
                    shape[1] if isinstance(shape[1], int) else None,
                    shape[2] if isinstance(shape[2], int) else None,
                    shape[3] if isinstance(shape[3], int) else None,
                    shape[4] if isinstance(shape[4], int) else None,
                ]
                tf_layers_dict[graph_input_name]['ncw_nchw_ncdhw_perm'] = [0,1,2,3,4]
            tf_layers_dict[graph_input_name]['op'] = ncdhw

    elif graph_input.shape != tf.TensorShape(None) and len(shape) > 0:
        # Except scalar, 4D and 5D
        if ncw_nchw_ncdhw_keep \
            and keep_ncw_or_nchw_or_ncdhw_input_names \
            and graph_input_name in keep_ncw_or_nchw_or_ncdhw_input_names:
            error_msg = f'' +\
                f'{Color.RED}ERROR:{Color.RESET} ' +\
                f'The keep_ncw_or_nchw_or_ncdhw_input_names parameter only supports 3D/4D/5D input. ' +\
                f'INPUT name: {graph_input_name} input_shape: {graph_input.shape}'
            print(error_msg)
            assert not ncw_nchw_ncdhw_keep, error_msg

        if nwc_nhwc_ndhwc_keep \
            and keep_nwc_or_nhwc_or_ndhwc_input_names \
            and graph_input_name in keep_nwc_or_nhwc_or_ndhwc_input_names:
            error_msg = f'' +\
                f'{Color.RED}ERROR:{Color.RESET} ' +\
                f'The keep_nwc_or_nhwc_or_ndhwc_input_names parameter only supports 3D/4D/5D input. ' +\
                f'INPUT name: {graph_input_name} input_shape: {graph_input.shape}'
            print(error_msg)
            assert not nwc_nhwc_ndhwc_keep, error_msg

        tf_layers_dict[graph_input_name]['tf_node'] = \
            tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in shape[1:]
                ],
                batch_size=shape[0] if isinstance(shape[0], int) else None,
                name=graph_input_name,
                dtype=dtype,
            )
        tf_layers_dict[graph_input_name]['op'] = tf_layers_dict[graph_input_name]['tf_node']
        tf_input_shape = [
            shape[0] if isinstance(shape[0], int) else None,
        ] + [inp if isinstance(inp, int) else None for inp in shape[1:]]

    else:
        # Scalar
        if ncw_nchw_ncdhw_keep \
            and keep_ncw_or_nchw_or_ncdhw_input_names \
            and graph_input_name in keep_ncw_or_nchw_or_ncdhw_input_names:
            error_msg = f''+\
                f'{Color.RED}ERROR:{Color.RESET} ' +\
                f'The keep_ncw_or_nchw_or_ncdhw_input_names parameter only supports 3D/4D/5D input. ' +\
                f'INPUT name: {graph_input_name} input_shape: {graph_input.shape}'
            print(error_msg)
            assert not ncw_nchw_ncdhw_keep, error_msg

        if nwc_nhwc_ndhwc_keep \
            and keep_nwc_or_nhwc_or_ndhwc_input_names \
            and graph_input_name in keep_nwc_or_nhwc_or_ndhwc_input_names:
            error_msg = f'' +\
                f'{Color.RED}ERROR:{Color.RESET} ' +\
                f'The keep_nwc_or_nhwc_or_ndhwc_input_names parameter only supports 3D/4D/5D input. ' +\
                f'INPUT name: {graph_input_name} input_shape: {graph_input.shape}'
            print(error_msg)
            assert not nwc_nhwc_ndhwc_keep, error_msg

        tf_layers_dict[graph_input_name]['tf_node'] = \
            tf.keras.Input(
                shape=shape if graph_input.shape != tf.TensorShape(None) else [None],
                name=graph_input_name,
                dtype=dtype,
            )
        tf_layers_dict[graph_input_name]['op'] = tf_layers_dict[graph_input_name]['tf_node']
        tf_input_shape = shape if graph_input.shape != tf.TensorShape(None) else [None]

    # The output_shape_trans stores the result of determining
    output_shape_trans = False
    if shape is not None and tf_input_shape is not None:
        for onnx_dim, tf_dim in zip(shape, tf_input_shape):
            onnx_dim = onnx_dim if isinstance(onnx_dim, int) else None
            if onnx_dim is None and tf_dim is None:
                pass
            elif onnx_dim is None and tf_dim is not None:
                output_shape_trans = True
            elif onnx_dim is not None and tf_dim is None:
                output_shape_trans = True
            elif onnx_dim is not None and tf_dim is not None:
                if onnx_dim == tf_dim:
                    pass
                else:
                    output_shape_trans = True
            else:
                pass
    tf_layers_dict[graph_input_name]['before_op_output_shape_trans'] = output_shape_trans