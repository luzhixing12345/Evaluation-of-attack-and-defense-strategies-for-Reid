
import torch

a = [torch.tensor([[i,i+1],[i,i+1],[i+1,i+2]]) for i in range(10)]
b= torch.stack(a)

print(b.shape)