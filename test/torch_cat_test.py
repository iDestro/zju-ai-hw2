import torch


a = torch.Tensor([[1, 2, 3]])
b = [1, 2, 3]

print(torch.cat([a, torch.zeros(1, 3)], dim=1))