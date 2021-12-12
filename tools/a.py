

import torch


a = torch.randint(10,(10,4,2))
num = 5

b = torch.cat((a[0:num,:,:],a[num:,:,:]),1)
print(b.shape)