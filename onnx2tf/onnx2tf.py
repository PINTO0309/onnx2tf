#! /usr/bin/env python

import os
import sys

import random
random.seed(0)

import numpy as np
np.random.seed(0)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.random.set_seed(0)

import onnx
import onnx_graphsurgeon as gs

from typing import Optional, List
from argparse import ArgumentParser

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


def convert(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
    output_folder_path: Optional[str] = 'saved_model',
    keep_nchw_or_ncdhw_input_names: Optional[List[str]] = None,
    non_verbose: Optional[bool] = False,
):
    """Convert ONNX to TensorFlow models.

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n
        Either input_onnx_file_path or onnx_graph must be specified.

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n
        Either input_onnx_file_path or onnx_graph must be specified.\n
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.

    output_folder_path: Optional[str]
        Output tensorflow model folder path.\n
        Default: "saved_model"

    keep_nchw_or_ncdhw_input_names: Optional[List[str]]
        Holds the NCHW or NCDHW of the input shape for the specified INPUT OP names.\n
        If a nonexistent INPUT OP name is specified, it is ignored.\n
        Valid only for 4D and 5D input tensors.\n\n
        e.g. \n
        --keep_nchw_or_ncdhw_input_names=['input0', 'input1', 'input2']

    non_verbose: Optional[bool]
        Do not show all information logs. Only error logs are displayed.\n
        Default: False
    """

    # Either designation required
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # If output_folder_path is empty, set the initial value
    if not output_folder_path:
        output_folder_path = 'saved_model'

    # Escape
    input_onnx_file_path = fr'{input_onnx_file_path}'
    output_folder_path = fr'{output_folder_path}'

    # Input file existence check
    if not os.path.exists(input_onnx_file_path):
        print(
            f'{Color.RED}ERROR:{Color.RESET} ' +
            f'The specified *.onnx file does not exist.' + \
            f'input_onnx_file_path: {input_onnx_file_path}'
        )
        sys.exit(1)

    # Create Output folder
    os.makedirs(output_folder_path, exist_ok=True)


    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)
    graph = gs.import_onnx(onnx_graph)

    a=0


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    parser.add_argument(
        '-o',
        '--output_folder_path',
        type=str,
        help=\
            'Output folder path. \n' +
            'Default: "saved_model"'
    )
    parser.add_argument(
        '-k',
        '--keep_nchw_or_ncdhw_input_names',
        type=str,
        nargs='+',
        help=\
            'Holds the NCHW or NCDHW of the input shape for the specified INPUT OP names. \n' +
            'If a nonexistent INPUT OP name is specified, it is ignored. \n' +
            'Valid only for 4D and 5D input tensors. \n\n' +
            'e.g. \n' +
            '--keep_nchw_or_ncdhw_input_names "input0" "input1" "input2"'
    )
    parser.add_argument(
        '-n',
        '--non_verbose',
        action='store_true',
        help='Do not show all information logs. Only error logs are displayed.'
    )
    args = parser.parse_args()

    # Convert
    tf_graph = convert(
        input_onnx_file_path=args.input_onnx_file_path,
        output_folder_path=args.output_folder_path,
        keep_nchw_or_ncdhw_input_names=args.keep_nchw_or_ncdhw_input_names,
        non_verbose=args.non_verbose,
    )


if __name__ == '__main__':
    main()
