import torch
import numpy as np
from os import path
from torch import nn

save_dir = '/data/ojw/onnx2tf/test_cind/'

# model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        pass

    def forward(self, x_1, x_2):
        x_1[x_1==1] = 0
        
        return x_1+x_2

# input data
x_1 = torch.tensor([2,3,4,5])
x_2 = torch.tensor([0,0,0,0])
m = Model().eval()

# export input
np.save(path.join(save_dir, 'x_1'), x_1.numpy())
np.save(path.join(save_dir, 'x_2'), x_2.numpy())

# export onnx
torch.onnx.export(
    m,
    (x_1, x_2),
    path.join(save_dir, 'model.onnx'),
    verbose=True,
    opset_version=16
)

