
import torch

x = torch.tensor([2.], requires_grad=True)

a = torch.add(x, 1).detach()
a.requires_grad=True
b = torch.add(a, 2).detach() 
b.requires_grad=True
y = torch.mul(a, b)
y.backward()

print("requires_grad: ", x.requires_grad, a.requires_grad, b.requires_grad, y.requires_grad)
print("is_leaf: ", x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
print("grad: ", a.grad.data,b.grad.data)
