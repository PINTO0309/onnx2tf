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

    def forward(self, x):
        return torch.clamp(x, min=0.0, max=6.0)

class Model2(nn.Module):
    def __init__(
        self,
    ):
        super(Model2, self).__init__()

    def forward(self, x, min, max):
        return torch.clamp(x, min=min, max=max)

class Model3(nn.Module):
    def __init__(
        self,
    ):
        super(Model3, self).__init__()

    def forward(self, x):
        return torch.clamp(x, min=torch.tensor(1.0, dtype=torch.float32))

if __name__ == "__main__":
    OPSET=11
    MODEL = f'Clip'
    model = Model1()
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

    MODEL = f'Clip_var'
    model = Model2()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(20, 100)
    min = torch.tensor(0.0, dtype=torch.float32)
    max = torch.tensor(6.0, dtype=torch.float32)
    torch.onnx.export(
        model,
        args=(x,min,max),
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

    MODEL = f'Clip_inf'
    model = Model3()
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
