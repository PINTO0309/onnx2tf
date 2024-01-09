#! /usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
import onnx
from onnxsim import simplify
import numpy as np
np.random.seed(0)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


class STFT(nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window(self.win_length)

    def forward(self, x):
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            normalized=False,
            return_complex=False # torch.onnx.export: STFT does not currently support complex types
        )
        
        return spec

if __name__ == "__main__":
    OPSET=[17]

    for opset in OPSET:
        MODEL = f'STFT'
        batch, signal_length = 2, 320000
        n_fft, hop_length, win_length = 1024, 320, 1024
        input = torch.randn(
            size=(batch, signal_length),
        )
        model = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        onnx_file = f"{MODEL}_{opset}.onnx"
        
        torch.onnx.export(
            model,
            args=(input),
            f=onnx_file,
            opset_version=opset,
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

       
