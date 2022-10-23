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
        return torch.add(x, y)

class Model2(nn.Module):
    def __init__(
        self,
    ):
        super(Model2, self).__init__()

    def forward(self, x):
        return torch.add(x, torch.tensor([0.6], dtype=torch.float32))

if __name__ == "__main__":
    OPSET=11
    MODEL = f'Add_var'
    model = Model1()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.randn(20, 100)
    y = torch.randn(20, 100)
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
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    MODEL = f'Add'
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
        do_constant_folding=False,
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)
    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)
