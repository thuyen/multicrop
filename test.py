import torch
import multicrop

def test2d(op, data, coord, target, stride=1, flag=True):
    if flag and data.dim() > 2:
        # if channels first
        #hwc -> chw
        data = data.permute(2, 0, 1).contiguous()
        #nhwc -> nchw
        target = target.permute(0, 3, 1, 2).contiguous()
    out = op(data, coord, 3, 3, stride, flag)
    print((out == target).all().item())
    print('-'*10)

def test3d(op, data, coord, target, stride=1, flag=True):
    if flag and data.dim() > 3:
        # if channels first
        #hwtc -> chwt
        data = data.permute(3, 0, 1, 2).contiguous()
        #nhwtc -> nchwt
        target = target.permute(0, 4, 1, 2, 3).contiguous()
    out = op(data, coord, 3, 3, 3, stride, flag)
    print((out == target).all().item())
    print('-'*10)


## test 2d
x = torch.randint(0, 5, (5, 5), dtype=torch.int16)
f = torch.tensor([[2, 2]], dtype=torch.int16)
t = x[1:4, 1:4].unsqueeze(0)

test2d(multicrop.crop2d, x, f, t, 1, True)  # channel first
test2d(multicrop.crop2d, x, f, t, 1, False) # channel last

test2d(multicrop.crop2d_cpu, x, f, t, 1, True)  # channel first
test2d(multicrop.crop2d_cpu, x, f, t, 1, False) # channel last

x, f, t = x.cuda(), f.cuda(), t.cuda()
test2d(multicrop.crop2d_gpu, x, f, t, 1, True)  # channel first
test2d(multicrop.crop2d_gpu, x, f, t, 1, False) # channel last

x = torch.randint(0, 5, (5, 5, 2), dtype=torch.int16)
f = torch.tensor([[2, 2]], dtype=torch.int16)
t = x[1:4, 1:4].unsqueeze(0)

test2d(multicrop.crop2d, x, f, t, 1, True)  # channel first
test2d(multicrop.crop2d, x, f, t, 1, False) # channel last

test2d(multicrop.crop2d_cpu, x, f, t, 1, True)  # channel first
test2d(multicrop.crop2d_cpu, x, f, t, 1, False) # channel last

x, f, t = x.cuda(), f.cuda(), t.cuda()
test2d(multicrop.crop2d_gpu, x, f, t, 1, True)  # channel first
test2d(multicrop.crop2d_gpu, x, f, t, 1, False) # channel last

## test 3d
x = torch.randint(0, 5, (5, 5, 5), dtype=torch.int16)
f = torch.tensor([[2, 2, 2]], dtype=torch.int16)
t = x[1:4, 1:4, 1:4].unsqueeze(0)

test3d(multicrop.crop3d_cpu, x, f, t, 1, True)  # channel first
test3d(multicrop.crop3d_cpu, x, f, t, 1, False) # channel last

x, f, t = x.cuda(), f.cuda(), t.cuda()
test3d(multicrop.crop3d_gpu, x, f, t, 1, True)  # channel first
test3d(multicrop.crop3d_gpu, x, f, t, 1, False) # channel last

x = torch.randint(0, 5, (5, 5, 5, 2), dtype=torch.int16)
f = torch.tensor([[2, 2, 2]], dtype=torch.int16)
t = x[1:4, 1:4, 1:4].unsqueeze(0)


test3d(multicrop.crop3d_cpu, x, f, t, 1, True)  # channel first
test3d(multicrop.crop3d_cpu, x, f, t, 1, False) # channel last

x, f, t = x.cuda(), f.cuda(), t.cuda()
test3d(multicrop.crop3d_gpu, x, f, t, 1, True)  # channel first
test3d(multicrop.crop3d_gpu, x, f, t, 1, False) # channel last


