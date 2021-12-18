

import torch
a = torch.randint(10,(4,3))

b = torch.randint(10,(1,3))

c = torch.cat((a,b))
print(a)
print(b)
print(c)
print(c.shape)