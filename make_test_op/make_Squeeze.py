#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)


class Model1(nn.Module):
    def __init__(
        self,
    ):
        super(Model1, self).__init__()

    def forward(self, x, y):
        return torch.squeeze(x, dim=y)

class Model2(nn.Module):
    def __init__(
        self,
    ):
        super(Model2, self).__init__()

    def forward(self, x):
        return torch.squeeze(x, dim=0)

class Model3(nn.Module):
    def __init__(
        self,
    ):
        super(Model3, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


if __name__ == "__main__":
    OPSET=13
    MODEL = f'Squeeze_var'
    model = Model1()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn([1,2,3,4,1,1], dtype=torch.float32)
    y = torch.tensor(0, dtype=torch.int64)
    torch.onnx.export(
        model,
        args=(x,y),
        f=onnx_file,
        opset_version=OPSET,
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

    OPSET=11
    MODEL = f'Squeeze'
    model = Model2()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
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
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    OPSET=13
    MODEL = f'Squeeze_all'
    model = Model3()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
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
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
