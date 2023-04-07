import torch
import pickle
from os import path
from torch import nn

save_dir = '/data/ojw/onnx2tf/test/'

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
with open(path.join(save_dir, 'input.pkl'), 'wb') as f:
    pickle.dump([x_1.numpy(),x_2.numpy()], f)

# export onnx
torch.onnx.export(
    m,
    (x_1, x_2),
    path.join(save_dir, 'model.onnx'),
    verbose=True,
    opset_version=16
)

