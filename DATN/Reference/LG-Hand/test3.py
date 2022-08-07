import torch
import math
x = torch.tensor([math.pi/3], requires_grad=True)

y =  - torch.cos(x) + torch.sin(x) + torch.sin(x)/torch.cos(x)

y.backward()
print(x.grad)