import torch
import torch.nn as nn


class FixedPooling(nn.Module):
    def __init__(self, fixed_size):
        super().__init__()
        self.fixed_size = fixed_size

    def forward(self, x):
        b, w, h = x.shape
        p_w = self.fixed_size * ((w + self.fixed_size - 1) // self.fixed_size) - w
        p_h = self.fixed_size * ((h + self.fixed_size - 1) // self.fixed_size) - h
        x = nn.functional.pad(x, (0, p_h, 0, p_w))
        pool_size = (((w + self.fixed_size - 1) // self.fixed_size), ((h + self.fixed_size - 1) // self.fixed_size))
        pool = nn.MaxPool2d(pool_size, stride=pool_size)
        return pool(x)


if __name__ == '__main__':
    x = torch.randn(64, 1, 1)
    model = FixedPooling(6)
    print(model(x))

    
    