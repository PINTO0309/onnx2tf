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
        avgpool1d = nn.AvgPool1d(
            kernel_size=3,
            stride=2,
            padding=0,
            ceil_mode=True,
            count_include_pad=True,
        )(x)
        return avgpool1d

class Model2D(nn.Module):
    def __init__(
        self,
    ):
        super(Model2D, self).__init__()

    def forward(self, x):
        avgpool2d = nn.AvgPool2d(
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(0, 0),
            ceil_mode=False,
            count_include_pad=True,
        )(x)
        return avgpool2d

class Model3D(nn.Module):
    def __init__(
        self,
    ):
        super(Model3D, self).__init__()

    def forward(self, x):
        avgpool3d = nn.AvgPool3d(
            kernel_size=(3, 2, 2),
            stride=(2, 1, 2),
            padding=(0, 0, 0),
            ceil_mode=False,
            count_include_pad=True,
        )(x)
        return avgpool3d


if __name__ == "__main__":
    MODEL = f'AvgPool1D'
    model = Model1D()
    onnx_file = f"{MODEL}.onnx"
    x = torch.tensor([[[1.,2,3,4,5,6,7]]])
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
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

    MODEL = f'AvgPool2D'
    model = Model2D()
    onnx_file = f"{MODEL}.onnx"
    x = torch.randn(20, 16, 50, 32)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
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

    MODEL = f'AvgPool3D'
    model = Model3D()
    onnx_file = f"{MODEL}.onnx"
    x = torch.randn(20, 16, 50, 44, 31)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
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