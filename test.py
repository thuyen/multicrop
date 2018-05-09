import torch
import multicrop

x = torch.randint(0, 5, (5, 5, 1), dtype=torch.int16)
f = torch.tensor([[2, 2]], dtype=torch.int16)

a = multicrop.crop2d(x, f, 3, 3, 1)

x = x.view(5, 5)
a = a.view(3, 3)

print(x)
print(a)

#x = torch.randint(0, 5, (5, 5), dtype=torch.int16)
#f = torch.tensor([[2, 2]], dtype=torch.int16)
#
#a = multicrop.crop2d(x, f, 3, 3, 1)
#
#print(x)
#print(a)

#x = torch.randint(0, 5, (5, 5, 5, 1), dtype=torch.int16)
#f = torch.tensor([[2, 2, 2]], dtype=torch.int16)
#
#a = multicrop.crop3d(x, f, 3, 3, 3, 1)
#
#x = x.view(5, 5, 5)
#a = a.view(3, 3, 3)
#print(x)
#print(a)
