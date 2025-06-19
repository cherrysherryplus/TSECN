import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OutlierAwareLoss(nn.Module):
    def __init__(self, kernel_size=False):
        super(OutlierAwareLoss, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size)
        self.kernel = kernel_size

    def forward(self, out, lab):
        b, c, h, w = out.shape
        p = self.kernel // 2
        delta = out - lab
        if self.kernel:
            delta_ = torch.nn.functional.pad(delta, (p, p, p, p))
            patch = self.unfold(delta_).reshape(b, c,
                                                self.kernel, self.kernel,
                                                h, w).detach()
            var = patch.std((2, 3)) / (2 ** .5)
            avg = patch.mean((2, 3))
        else:
            var = delta.std((2, 3), keepdims=True) / (2 ** .5)
            avg = delta.mean((2, 3), True)
        weight = 1 - (-((delta - avg).abs() / var)).exp().detach()
        # weight = 1 - (-1 / var.detach()-1/ delta.abs().detach()).exp()
        loss = (delta.abs() * weight).mean()
        return loss
    