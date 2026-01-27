#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)


class Model1D(nn.Module):
    def __init__(
        self,
    ):
        super(Model1D, self).__init__()

    def forward(self, x):
        lppool1d = nn.LPPool1d(
            norm_type=2,
            kernel_size=3,
            stride=2,
            ceil_mode=True,
        )(x)
        return lppool1d


class Model2D(nn.Module):
    def __init__(
        self,
    ):
        super(Model2D, self).__init__()

    def forward(self, x):
        lppool2d = nn.LPPool2d(
            norm_type=2,
            kernel_size=(3, 2),
            stride=(2, 1),
            ceil_mode=False,
        )(x)
        return lppool2d


class Model3D(nn.Module):
    def __init__(
        self,
    ):
        super(Model3D, self).__init__()

    def forward(self, x):
        lppool3d = nn.LPPool3d(
            norm_type=2,
            kernel_size=(3, 2, 2),
            stride=(2, 1, 2),
            ceil_mode=False,
        )(x)
        return lppool3d


if __name__ == "__main__":
    MODEL = f'LpPool1D'
    model = Model1D()
    onnx_file = f"{MODEL}.onnx"
    x = torch.tensor([[[1.,2,3,4,5,6,7]]])
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=22,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    MODEL = f'LpPool2D'
    model = Model2D()
    onnx_file = f"{MODEL}.onnx"
    x = torch.randn(20, 16, 50, 32)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=22,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    MODEL = f'LpPool3D'
    model = Model3D()
    onnx_file = f"{MODEL}.onnx"
    x = torch.randn(20, 16, 50, 44, 31)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=22,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
