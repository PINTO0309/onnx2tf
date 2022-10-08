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
        --keep_ncw_or_nchw_or_ncdhw_input_names=['input0', 'input1', 'input2']
    """
    ncw_nchw_ncdhw_keep = False

    shape = graph_input.shape
    dtype = graph_input.dtype

    # Preserving Graph Structure (Dict)
    tf_layers_dict[graph_input.name] = {
        'optype': 'Input',
        'shape': shape,
        'dtype': dtype,
        'op': None,
    }
    if len(graph_input.shape) in [3, 4, 5] and keep_ncw_or_nchw_or_ncdhw_input_names:
        if graph_input.name in keep_ncw_or_nchw_or_ncdhw_input_names:
            ncw_nchw_ncdhw_keep = True
        else:
            ncw_nchw_ncdhw_keep = False
    elif len(graph_input.shape) in [3, 4, 5] and not keep_ncw_or_nchw_or_ncdhw_input_names:
        ncw_nchw_ncdhw_keep = False
    else:
        ncw_nchw_ncdhw_keep = True
    tf_layers_dict[graph_input.name]['ncw_nchw_ncdhw_keep'] = ncw_nchw_ncdhw_keep

    # Generation of TF OP
    if len(shape) == 3:
        # 3D
        if not ncw_nchw_ncdhw_keep:
            tf_layers_dict[graph_input.name]['tf_node'] = \
                tf.keras.Input(
                    shape=[
                        shape[2] if isinstance(shape[2], int) else None,
                        shape[1] if isinstance(shape[1], int) else None,
                    ],
                    batch_size=shape[0] if isinstance(shape[0], int) else None,
                    name=graph_input.name,
                    dtype=dtype,
                )
            tf_layers_dict[graph_input.name]['op'] = tf_layers_dict[graph_input.name]['tf_node']
        else:
            nchw = tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in shape[1:]
                ],
                batch_size=shape[0] if isinstance(shape[0], int) else None,
                name=graph_input.name,
                dtype=dtype,
            )
            tf_layers_dict[graph_input.name]['tf_node'] = \
                tf.transpose(nchw, perm=[0,2,1])
            tf_layers_dict[graph_input.name]['op'] = nchw

    elif len(shape) == 4:
        # 4D
        if not ncw_nchw_ncdhw_keep:
            tf_layers_dict[graph_input.name]['tf_node'] = \
                tf.keras.Input(
                    shape=[
                        shape[2] if isinstance(shape[2], int) else None,
                        shape[3] if isinstance(shape[3], int) else None,
                        shape[1] if isinstance(shape[1], int) else None,
                    ],
                    batch_size=shape[0] if isinstance(shape[0], int) else None,
                    name=graph_input.name,
                    dtype=dtype,
                )
            tf_layers_dict[graph_input.name]['op'] = tf_layers_dict[graph_input.name]['tf_node']
        else:
            nchw = tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in shape[1:]
                ],
                batch_size=shape[0] if isinstance(shape[0], int) else None,
                name=graph_input.name,
                dtype=dtype,
            )
            tf_layers_dict[graph_input.name]['tf_node'] = \
                tf.transpose(nchw, perm=[0,2,3,1])
            tf_layers_dict[graph_input.name]['op'] = nchw

    elif len(shape) == 5:
        # 5D
        if not ncw_nchw_ncdhw_keep:
            tf_layers_dict[graph_input.name]['tf_node'] = \
                tf.keras.Input(
                    shape=[
                        shape[2] if isinstance(shape[2], int) else None,
                        shape[3] if isinstance(shape[3], int) else None,
                        shape[4] if isinstance(shape[4], int) else None,
                        shape[1] if isinstance(shape[5], int) else None,
                    ],
                    batch_size=shape[0] if isinstance(shape[0], int) else None,
                    name=graph_input.name,
                    dtype=dtype,
                )
            tf_layers_dict[graph_input.name]['op'] = tf_layers_dict[graph_input.name]['tf_node']
        else:
            ncdhw = tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in shape[1:]
                ],
                batch_size=shape[0],
                name=graph_input.name,
                dtype=dtype,
            )
            tf_layers_dict[graph_input.name]['tf_node'] = \
                tf.transpose(ncdhw, perm=[0,2,3,4,1])
            tf_layers_dict[graph_input.name]['op'] = ncdhw

    elif len(shape) > 0:
        # Except scalar, 4D and 5D
        if ncw_nchw_ncdhw_keep and keep_ncw_or_nchw_or_ncdhw_input_names and graph_input.name in keep_ncw_or_nchw_or_ncdhw_input_names:
            error_msg = f'' +\
                f'{Color.RED}ERROR:{Color.RESET} ' +\
                f'The keep_ncw_or_nchw_or_ncdhw_input_names parameter only supports 3D/4D/5D input. ' +\
                f'INPUT name: {graph_input.name} input_shape: {graph_input.shape}'
            print(error_msg)
            assert not ncw_nchw_ncdhw_keep, error_msg

        tf_layers_dict[graph_input.name]['tf_node'] = \
            tf.keras.Input(
                shape=[
                    inp if isinstance(inp, int) else None for inp in shape[1:]
                ],
                batch_size=shape[0] if isinstance(shape[0], int) else None,
                name=graph_input.name,
                dtype=dtype,
            )
        tf_layers_dict[graph_input.name]['op'] = tf_layers_dict[graph_input.name]['tf_node']

    else:
        # Scalar
        if ncw_nchw_ncdhw_keep and keep_ncw_or_nchw_or_ncdhw_input_names and graph_input.name in keep_ncw_or_nchw_or_ncdhw_input_names:
            error_msg = f''+\
                f'{Color.RED}ERROR:{Color.RESET} ' +\
                f'The keep_ncw_or_nchw_or_ncdhw_input_names parameter only supports 3D/4D/5D input. ' +\
                f'INPUT name: {graph_input.name} input_shape: {graph_input.shape}'
            print(error_msg)
            assert not ncw_nchw_ncdhw_keep, error_msg

        tf_layers_dict[graph_input.name]['tf_node'] = \
            tf.keras.Input(
                shape=shape,
                name=graph_input.name,
                dtype=dtype,
            )
        tf_layers_dict[graph_input.name]['op'] = tf_layers_dict[graph_input.name]['tf_node']
