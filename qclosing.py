# module that defines scale equivariant model based on the scale-space of quadratic erosions

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from Dconv import Dconv2d


class Closing2d(torch.nn.Module):
    # function that performs the lifting based on the quadratic dilations scale-space
    def __init__(self, se_coef=0.5, base=2., zero_scale=1., n_scales=8, scales=None):
        super(Closing2d, self).__init__()

        self.se_coef = torch.nn.Parameter(torch.tensor(se_coef), requires_grad=True)
        self.n_scales = n_scales
        


        # computes the scales used
        self.base = base
        self.zero_scale = zero_scale
        self.n_scales = n_scales
        k = np.arange(0, n_scales)
        dilations = np.power(base, k)
        self.scales = (zero_scale**2)*(dilations**2)
        print('qclosing scales: ', end='')
        print(self.scales)

        # the widths(point where the structuring functions are truncated) are also computed in the same way the Gaussian ones are
        self.widths = np.asarray([4*int(np.ceil(np.sqrt(scale))) for scale in self.scales])
        print('widths:  ', end='')
        print(self.widths)

        self.ses = nn.ParameterList([])
        for i in range(len(self.scales)):
            self.ses.append(self._make_se(self.scales[i], self.widths[i]))
        

                                            
    def _make_se(self, t, w):
        # creates a list of the 1d quadratic structuring elements for each scale
        se = torch.tensor([(1/(4*t))*(x**2) for x in range(-w, w+1)], dtype=torch.float32)
        se = se.view([-1, 1])
        return nn.Parameter(se, requires_grad=False)

    def _sep_dilation(self, im, se):
        # the 2d quadratic structuring elemet can be decomposed into the Minkowski sum of two 1d quadratic structuring elements
        # with orthogonal lines as support
        n = se.size()[0]
        if n % 2 == 0:
            pad1 = n//2-1
            pad2 = n//2
        else:
            pad1 = n//2
            pad2 = n//2
        tmp = im
        
        tmp = torch.nn.functional.pad(tmp, [pad1, pad2, 0, 0], mode='constant', value=-1e4)
        tmp = F.unfold(tmp, [1, n], padding=0).view(im.shape[0], im.shape[1], n, -1)
        tmp = torch.max(tmp - self.se_coef*se, 2)[0].view(im.shape)
        
        tmp = torch.nn.functional.pad(tmp, [0, 0, pad1, pad2], mode='constant', value=-1e4)
        tmp = F.unfold(tmp, [n, 1], padding=0).view(im.shape[0], im.shape[1], n, -1)
        tmp = torch.max(tmp - self.se_coef*se, 2)[0].view(im.shape)
        
        return tmp

    def _sep_erosion(self, im, se):
        # the 2d quadratic structuring elemet can be decomposed into the Minkowski sum of two 1d quadratic structuring elements
        # with orthogonal lines as support
        n = se.size()[0]
        if n % 2 == 0:
            pad1 = n//2-1
            pad2 = n//2
        else:
            pad1 = n//2
            pad2 = n//2
        tmp = im
        
        tmp = torch.nn.functional.pad(tmp, [pad1, pad2, 0, 0], mode='constant', value=1e4)
        tmp = F.unfold(tmp, [1, n], padding=0).view(im.shape[0], im.shape[1], n, -1)
        tmp = torch.min(tmp + self.se_coef*se, 2)[0].view(im.shape)
        
        tmp = torch.nn.functional.pad(tmp, [0, 0, pad1, pad2], mode='constant', value=1e4)
        tmp = F.unfold(tmp, [n, 1], padding=0).view(im.shape[0], im.shape[1], n, -1)
        tmp = torch.min(tmp + self.se_coef*se, 2)[0].view(im.shape)
        
        return tmp
    
    
    def forward(self, inputs):
        output_list = list()
        output = [self._sep_dilation(inputs, se) for se in self.ses]
        output = [self._sep_erosion(output[i], self.ses[i]) for i in range(self.n_scales)]
        output = torch.stack(output, 2)
        return output
