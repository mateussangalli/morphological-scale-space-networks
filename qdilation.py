# module that defines scale equivariant model based on the scale-space of quadratic erosions

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from Dconv import Dconv2d

class Dilation2dSingle(torch.nn.Module):
    def __init__(self, scale, width):
        super(Dilation2dSingle, self).__init__()
        se = torch.tensor([(1/(4*scale))*(x**2) for x in range(-width, width+1)], dtype=torch.float32)
        se = se.view([-1, 1])
        self.se = nn.Parameter(se, requires_grad=False)

    def forward(self, im, se_coef):
        # the 2d quadratic structuring elemet can be decomposed into the Minkowski sum of two 1d quadratic structuring elements
        # with orthogonal lines as support
        n = self.se.size()[0]
        pad1 = n//2-1+n%2
        pad2 = n//2
        tmp = im
        
        tmp = torch.nn.functional.pad(tmp, [pad1, pad2, 0, 0], mode='constant', value=-1e4)
        tmp = F.unfold(tmp, [1, n], padding=0).view(im.shape[0], im.shape[1], n, -1)
        tmp = torch.max(tmp - se_coef*self.se, 2)[0].view(im.shape)
        
        tmp = torch.nn.functional.pad(tmp, [0, 0, pad1, pad2], mode='constant', value=-1e4)
        tmp = F.unfold(tmp, [n, 1], padding=0).view(im.shape[0], im.shape[1], n, -1)
        tmp = torch.max(tmp - se_coef*self.se, 2)[0].view(im.shape)
        
        return tmp



class Dilation2d(torch.nn.Module):
    # function that performs the lifting based on the quadratic dilations scale-space
    def __init__(self, se_coef=0.5, base=2., zero_scale=1., n_scales=8):
        super(Dilation2d, self).__init__()

        self.se_coef = torch.nn.Parameter(torch.tensor(se_coef), requires_grad=True)
        


        self.base = base
        self.zero_scale = zero_scale
        self.n_scales = n_scales
        k = np.arange(1, n_scales)
        dilations = np.power(base, k)
        self.scales = (zero_scale**2)*(dilations**2 - 1.)
        self.scales = list(self.scales)
        print('qdilations scales: ', end='')
        print(self.scales)

        # the widths(point where the structuring functions are truncated) are also computed in the same way the Gaussian ones are
        self.widths = np.asarray([4*int(np.ceil(np.sqrt(scale))) for scale in self.scales])
        print('widths:  ', end='')
        print(self.widths)

        self.dilations = nn.ModuleList(
            [Dilation2dSingle(s, w) for s, w in zip(self.scales, self.widths)])
        

                                            
    def forward(self, inputs):
        output = [dilation(inputs, self.se_coef) for dilation in self.dilations]
        output = [inputs] + output
        output = torch.stack(output, 2)
        return output
