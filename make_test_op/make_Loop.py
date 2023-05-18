#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)


class Model_1(nn.Module):
    def __init__(
        self,
    ):
        super(Model_1, self).__init__()

    def forward(self, x):
        for i in range(64):
            x = torch.sigmoid(x)
        return x

class Model_2(nn.Module):
    def __init__(
        self,
    ):
        super(Model_2, self).__init__()

    def forward(self, x, counter):
        while (counter < 64):
            x = torch.sigmoid(x)
            counter = counter + 1
        return x


class Model_3(nn.Module):
    def __init__(
        self,
    ):
        super(Model_3, self).__init__()

    def forward(self, x, counter):
        while True:
            x = torch.sigmoid(x)
            counter = counter + 1
            if counter >= 64:
                break
        return x


if __name__ == "__main__":
    """
    パターン1: ループ回数指定=8, ループ継続条件=None
    パターン2: ループ回数指定==None, ループ継続条件=ループ変数<8
    パターン3: ループ回数指定==None, ループ継続条件==None -> 特殊条件でbodyの中で強制ブレークする条件を定義？
    """
    OPSET=16
    MODEL = f'Loop_1'
    model = Model_1()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.Tensor([1,2,3]).float()
    torch.onnx.export(
        torch.jit.script(model),
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

    MODEL = f'Loop_2'
    model = Model_2()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.Tensor([1,2,3]).float()
    counter = torch.tensor(0)
    torch.onnx.export(
        torch.jit.script(model),
        args=(x, counter),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            f'{MODEL}_input',
            f'{MODEL}_counter',
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

    MODEL = f'Loop_3'
    model = Model_3()
    onnx_file = f"{MODEL}_{OPSET}.onnx"
    x = torch.Tensor([1,2,3]).float()
    counter = torch.tensor(0)
    torch.onnx.export(
        torch.jit.script(model),
        args=(x, counter),
        f=onnx_file,
        opset_version=OPSET,
        input_names=[
            f'{MODEL}_input',
            f'{MODEL}_counter',
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
