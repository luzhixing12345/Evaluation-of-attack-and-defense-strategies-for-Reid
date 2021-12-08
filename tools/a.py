


import torch
import torch.nn.functional as F

a = torch.randint(10,(12,20)).float()
print(a.shape)
b = a[0]
b = b.unsqueeze(0)
features = F.normalize(b, p=2, dim=1)
print(features)