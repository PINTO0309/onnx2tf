import random
random.seed(0)
import numpy as np
np.random.seed(0)
import importlib
import onnx_graphsurgeon as gs
from onnx2tf.utils.common_functions import (
    inverted_operation_enable_disable,
    get_replacement_parameter,
)


@inverted_operation_enable_disable
@get_replacement_parameter
def make_node(
    *,
    graph_node: gs.Node,
    tf_layers_dict: dict,
    **kwargs: dict,
):
    """Scatter

    Parameters
    ----------
    graph_node: gs.Node
        graph_surgeon Node

    tf_layers_dict: dict
        optype, shape, dtype, tensorflow graph
    """
    op = importlib.import_module(f'onnx2tf.ops.ScatterElements')
    op.make_node(
        graph_node=graph_node,
        tf_layers_dict=tf_layers_dict,
        **kwargs,
    )
