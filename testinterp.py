import torch
import torch.nn.functional as F
from time import time
import numpy as np

Nd = 10**6
Ni = 10

i = torch.rand(Ni)
x = torch.rand(Nd)
y = torch.range(1,Nd)/Nd

xN = x.numpy()
yN = y.numpy()
iN = i.numpy()
# Time numpy
start = time()
result = np.interp(iN,xN,yN)
