

import torch

a=torch.randint(10,(64,3,256,128))
print(a.shape)
s = [ a for i in range(10)]
s = torch.cat(s,dim=0)
q = s[:20]
g = s[20:]
print(q.shape,g.shape)