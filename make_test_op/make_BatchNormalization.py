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
        batchnorm1d = nn.BatchNorm1d(
            num_features=100,
            eps=1e-05,
            momentum=0.1,
            affine=False,
        )(x)
        return batchnorm1d

class Model2D(nn.Module):
    def __init__(
        self,
    ):
        super(Model2D, self).__init__()

    def forward(self, x):
        batchnorm2d = nn.BatchNorm2d(
            num_features=100,
            eps=1e-05,
            momentum=0.1,
            affine=False,
        )(x)
        return batchnorm2d

class Model3D(nn.Module):
    def __init__(
        self,
    ):
        super(Model3D, self).__init__()

    def forward(self, x):
        batchnorm3d = nn.BatchNorm3d(
            num_features=100,
            eps=1e-05,
            momentum=0.1,
            affine=False,
        )(x)
        return batchnorm3d


if __name__ == "__main__":
    OPSET=15
    MODEL = f'BatchNorm1D'
    model = Model1D()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(20, 100)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    MODEL = f'BatchNorm2D'
    model = Model2D()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(20, 100, 35, 45)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    MODEL = f'BatchNorm3D'
    model = Model3D()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(20, 100, 35, 45, 10)
    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            f'{MODEL}_input',
        ],
        output_names=[
            f'{MODEL}_output',
        ],
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
