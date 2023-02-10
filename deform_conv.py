from torch import nn
import torch
import math
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d
import torchvision
import numpy as np

class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, offset_in_channels=32,kernel_size=3, stride=1, padding=1, bias=True, extra_offset_mask=False):
        super(ModulatedDeformConv, self).__init__()
        
        self.padding = padding
        self.extra_offset_mask = extra_offset_mask
        
        self.conv_offset_mask = nn.Conv2d(
            offset_in_channels,
            3 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding,
            bias=True)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                bias=bias)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        if self.extra_offset_mask:
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return deform_conv2d(input=x, offset=offset,
                                weight=self.regular_conv.weight,
                                bias=self.regular_conv.bias,
                                padding=self.padding,
                                mask=mask)