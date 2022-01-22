

import torch

a = torch.tensor([1,2,3])
b = torch.tensor([2,3,4]).to('cuda')
c = torch.add(a,b)
print(c)