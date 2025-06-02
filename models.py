import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
import numpy as np


###################################################
# LAYERS / BLOCKS
###################################################

class DegWeights(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(DegWeights, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError("Expected 4D input tensor (batch_size, channels, height, width), got {}D tensor instead".format(x.dim()))
        device = x.device
        x_down = F.avg_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)
        #print(x_down)
        count = (x_down == 0.5).sum(dim=(-2, -1)).view(x_down.size(0), 1)
        degW = torch.pow(2., -count).float().to(device=device).unsqueeze(1).unsqueeze(2)
        return degW




class MajorityPooling2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, pooling_half='FullRandom'):
        super(MajorityPooling2d, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size, stride, padding)
        self.pooling_half = pooling_half

    def forward(self, x):
        device = x.device
        # average the +1s and -1s of the lattice with a (2,2) sliding window with stride 2
        x_down = self.avg_pool(x.to(dtype=torch.float, device=device))

        # assign values to downsampled lattice depending on the average obtained
        # majority of +1s in the window             -->     +1
        # majority of -1s in the window             -->     -1
        # equality of +1s and -1s in the window     -->     random
        x_down[x_down > 0.5] = 1.
        x_down[x_down < 0.5] = 0.
        if self.pooling_half == 'FullRandom':
            x_down[x_down == 0.5] = (torch.rand(x_down[x_down == 0.5].numel()) * 2 - 1 >= 0).to(dtype=int).to(
                dtype=torch.float).to(device=device)
        elif self.pooling_half == 'ConfRandom':
            x_down[x_down == 0.5] = (torch.rand(1) * 2 - 1 >= 0)
        elif self.pooling_half == '1':
            x_down[x_down == 0.5] = 1.
        elif self.pooling_half == '0':
            x_down[x_down == 0.5] = 0.
        else:
            raise NotImplementedError(f"pooling_half =  {self.pooling_half} not implemented")
        return x_down




class ParametrizedConv2d(nn.Module):
    """
    Parametrizes a convolution layer's kernel and bias to be compatible with the physics:
     - for a kernel size of 3, the kernel will be [[a,b,a], [b,c,b], [a,b,a]] w/ a, b and c learnable parameters. This
       kernel is symmetric w/ respect to the x and y axes, which is to be expected naturally.
     - for any kernel size, bias = -sum(kernel)/2

     TODO:
      - support kernel_size > 7
      - support in_channel > 1
      - support in_channel > 1

    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding='same', bias=True,
                 padding_mode='circular'):
        super(ParametrizedConv2d, self).__init__()
        assert in_channels == 1, "only in_channel of 1 is currently supported"
        assert out_channels == 1, "only out_channel of 1 is currently supported"
#        assert kernel_size == 3, "only kernel_size of 3 is currently supported"
        assert padding == 'same', "only padding 'same' is currently supported"
        assert padding_mode == 'circular', "only padding_mode 'circular' is currently supported"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if padding == 'same':
            self.padding = (self.kernel_size - 1) // 2
        self.padding_mode = padding_mode
        self.bias = bias

        if self.kernel_size == 3:     
           n_params = 3
        elif self.kernel_size == 5: 
           n_params = 6
        elif self.kernel_size == 7: 
           n_params = 10
        else:
           raise NotImplementedError("Kernel size not supported")
           
        self.params = Parameter(torch.rand(n_params))  # [a, b, c]

    def build_kernel_and_bias(self, device):
        kernel = torch.ones(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, device=device)
        # TODO: find a more elegant way for bigger kernel sizes...
        if self.kernel_size == 3 : 
           kernel[:, :, 0, 0] *= self.params[0]  # a, corner elements of the kernel
           kernel[:, :, 0, 2] *= self.params[0]
           kernel[:, :, 2, 0] *= self.params[0]
           kernel[:, :, 2, 2] *= self.params[0]
           kernel[:, :, 0, 1] *= self.params[1]  # b, side elements of the kernel
           kernel[:, :, 1, 0] *= self.params[1]
           kernel[:, :, 2, 1] *= self.params[1]
           kernel[:, :, 1, 2] *= self.params[1]
           kernel[:, :, 1, 1] *= self.params[2]  # c, central element of the kernel
        elif self.kernel_size == 5 :
           kernel[:, :, 1, 1] *= self.params[0]  # a, corner elements of the kernel 3x3
           kernel[:, :, 1, 3] *= self.params[0]
           kernel[:, :, 3, 1] *= self.params[0]
           kernel[:, :, 3, 3] *= self.params[0]
           kernel[:, :, 1, 2] *= self.params[1]  # b, side elements of the kernel 3x3
           kernel[:, :, 2, 1] *= self.params[1]
           kernel[:, :, 3, 2] *= self.params[1]
           kernel[:, :, 2, 3] *= self.params[1]
           kernel[:, :, 2, 2] *= self.params[2] # c, central element of the kernel
           kernel[:, :, 0, 0] *= self.params[3]  # d, corner elements of the kernel 5x5
           kernel[:, :, 0, 4] *= self.params[3]
           kernel[:, :, 4, 0] *= self.params[3]
           kernel[:, :, 4, 4] *= self.params[3]
           kernel[:, :, 0, 2] *= self.params[4]  # e, central side elements of the kernel 5x5
           kernel[:, :, 2, 0] *= self.params[4]
           kernel[:, :, 4, 2] *= self.params[4]
           kernel[:, :, 2, 4] *= self.params[4]
           kernel[:, :, 0, 1] *= self.params[5]  # f, rest of the side element of the kernel 5x5
           kernel[:, :, 0, 3] *= self.params[5]
           kernel[:, :, 1, 0] *= self.params[5]
           kernel[:, :, 3, 0] *= self.params[5]
           kernel[:, :, 1, 4] *= self.params[5]  
           kernel[:, :, 3, 4] *= self.params[5]
           kernel[:, :, 4, 1] *= self.params[5]
           kernel[:, :, 4, 3] *= self.params[5]
        elif self.kernel_size == 7 :
           kernel[:, :, 2, 2] *= self.params[0]  # a, corner elements of the kernel 3x3
           kernel[:, :, 2, 4] *= self.params[0]
           kernel[:, :, 4, 2] *= self.params[0]
           kernel[:, :, 4, 4] *= self.params[0]
           kernel[:, :, 2, 3] *= self.params[1]  # b, side elements of the kernel 3x3
           kernel[:, :, 3, 2] *= self.params[1]
           kernel[:, :, 4, 3] *= self.params[1]
           kernel[:, :, 3, 4] *= self.params[1]
           kernel[:, :, 3, 3] *= self.params[2] # c, central element of the kernel
           kernel[:, :, 1, 1] *= self.params[3]  # d, corner elements of the kernel 5x5
           kernel[:, :, 1, 5] *= self.params[3]
           kernel[:, :, 5, 1] *= self.params[3]
           kernel[:, :, 5, 5] *= self.params[3]
           kernel[:, :, 1, 3] *= self.params[4]  # e, central side elements of the kernel 5x5
           kernel[:, :, 3, 1] *= self.params[4]
           kernel[:, :, 5, 3] *= self.params[4]
           kernel[:, :, 3, 5] *= self.params[4]
           kernel[:, :, 1, 2] *= self.params[5]  # f, rest of the side element of the kernel 5x5
           kernel[:, :, 1, 4] *= self.params[5]
           kernel[:, :, 2, 1] *= self.params[5]
           kernel[:, :, 4, 1] *= self.params[5]
           kernel[:, :, 2, 5] *= self.params[5]  
           kernel[:, :, 4, 5] *= self.params[5]
           kernel[:, :, 5, 2] *= self.params[5]
           kernel[:, :, 5, 4] *= self.params[5]
           kernel[:, :, 0, 0] *= self.params[6]  # g, corner elements of the kernel 7x7
           kernel[:, :, 0, 6] *= self.params[6]
           kernel[:, :, 6, 0] *= self.params[6]
           kernel[:, :, 6, 6] *= self.params[6]
           kernel[:, :, 0, 3] *= self.params[7]  # h, central side elements of the kernel 7x7
           kernel[:, :, 3, 0] *= self.params[7]
           kernel[:, :, 6, 3] *= self.params[7]
           kernel[:, :, 3, 6] *= self.params[7]
           kernel[:, :, 0, 1] *= self.params[8]  # i, closer to corner side elements of the kernel 7x7
           kernel[:, :, 1, 0] *= self.params[8]
           kernel[:, :, 0, 5] *= self.params[8]
           kernel[:, :, 5, 0] *= self.params[8]
           kernel[:, :, 1, 6] *= self.params[8]  
           kernel[:, :, 5, 6] *= self.params[8]
           kernel[:, :, 6, 1] *= self.params[8]
           kernel[:, :, 6, 5] *= self.params[8]
           kernel[:, :, 0, 2] *= self.params[9]  # j, closer to center side elements of the kernel 7x7
           kernel[:, :, 2, 0] *= self.params[9]
           kernel[:, :, 0, 4] *= self.params[9]
           kernel[:, :, 4, 0] *= self.params[9]
           kernel[:, :, 2, 6] *= self.params[9]  
           kernel[:, :, 4, 6] *= self.params[9]
           kernel[:, :, 6, 2] *= self.params[9]
           kernel[:, :, 6, 4] *= self.params[9]


        bias = -torch.ones(self.out_channels, device=device) * torch.sum(kernel) / 2

        return kernel, bias

    def forward(self, x):
        device = x.device

        kernel, bias = self.build_kernel_and_bias(device)
        out = F.pad(x, pad=(self.padding, self.padding, self.padding, self.padding), mode=self.padding_mode)
        out = F.conv2d(out, kernel, bias, self.stride) if self.bias else F.conv2d(out, kernel, stride=self.stride)

        return out


class NearestNeighborsConv2d(nn.Module):
    """
    Parametrizes a convolution layer's kernel and bias to be compatible with the physics:
     - the kernel is always 3x3 and only probes the nearest neighbors symmetrically
            kernel = [[0,b,0],[b,a,b],[0,b,0]]
     - for any kernel size, bias = -sum(kernel)/2
    """

    def __init__(self, in_channels: int, out_channels: int, stride=1, padding='same', bias=True,
                 padding_mode='circular'):
        super(NearestNeighborsConv2d, self).__init__()
        super().__init__()
        assert in_channels == 1, "only in_channel of 1 is currently supported"
        assert out_channels == 1, "only out_channel of 1 is currently supported"
        assert padding == 'same', "only padding 'same' is currently supported"
        assert padding_mode == 'circular', "only padding_mode 'circular' is currently supported"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.stride = stride
        if padding == 'same':
            self.padding = (self.kernel_size - 1) // 2
        self.padding_mode = padding_mode
        self.bias = bias

        n_params = 2
        self.params = Parameter(torch.rand(n_params))  # [a, b]

    def build_kernel_and_bias(self, device):
        kernel = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, device=device)
        kernel[:, :, 1, 1] = self.params[0]  # a, central element of the kernel
        kernel[:, :, 0, 1] = self.params[1]  # b, side elements of the kernel
        kernel[:, :, 1, 0] = self.params[1]
        kernel[:, :, 2, 1] = self.params[1]
        kernel[:, :, 1, 2] = self.params[1]

        bias = -torch.ones(self.out_channels, device=device) * torch.sum(kernel) / 2

        return kernel, bias

    def forward(self, x):
        device = x.device

        kernel, bias = self.build_kernel_and_bias(device)
        out = F.pad(x, pad=(self.padding, self.padding, self.padding, self.padding), mode=self.padding_mode)
        out = F.conv2d(out, kernel, bias, self.stride) if self.bias else F.conv2d(out, kernel, stride=self.stride)

        return out


###################################################
# MODELS
###################################################


class Decoder(nn.Module):
    """
    Upsamples an Ising lattice by a factor 2 in all dimensions.

    Input tensor of shape (1, H, W)
    Output tensor of shape (1, 2H, 2W)  with H and W are powers of 2
    """

    def __init__(self, kernel_size=3, upsampling_method='nearest'):
        super(Decoder, self).__init__()

        if upsampling_method == 'nearest':
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        elif upsampling_method == 'bilinear':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            raise NotImplementedError(f"choose between 'nearest' or 'bilinear' for upsampling, not {upsampling_method}")

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      padding_mode='circular'),

            # nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, padding_mode='circular'),
        )

    def forward(self, x):
        return self.convs(self.up(x))

    def upsample(self, x, factor=2):
        assert factor % 2 == 0, f"argument 'factor' of method 'upsample' should be a power of 2, not {factor}. Default: 2"
        while factor % 2 == 0:
            x = self.forward(x)
            x = (torch.rand_like(x) < torch.sigmoid(x)).float()
            factor /= 2
        return x


class DecoderSymmetrizedConv(nn.Module):
    """
    Upsamples an Ising lattice by a factor 2 in all dimensions. Uses a convolution which has the symmetries of the Ising model

    """

    def __init__(self, upsampling_method='nearest', kernel_size: int = 3):
        super(DecoderSymmetrizedConv, self).__init__()
        #trying to fix the problem that this decoder will not accept up.params
#        self.params=ups

        if upsampling_method == 'nearest':
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        elif upsampling_method == 'bilinear':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        elif upsampling_method == 'probabilisticNN':
            self.up = UpsamplingProbabilisticNearest2d(scale_factor=2)
        else:
            raise NotImplementedError(f"choose between 'nearest' or 'bilinear' for upsampling, not {upsampling_method}")

        self.convs = ParametrizedConv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=1,
                                        padding='same',
                                        padding_mode='circular')

    def forward(self, x):
        return self.convs(self.up(x))

    def upsample(self, x, factor=2):
        assert factor % 2 == 0, f"argument 'factor' of method 'upsample' should be a power of 2, not {factor}. Default: 2"
        while factor % 2 == 0:
            x = self.forward(x)
            x = (torch.rand_like(x) < torch.sigmoid(x)).float()
            factor /= 2
        return x


class DecoderNearestNeighborsConv(nn.Module):
    """
    Upsamples an Ising lattice by a factor 2 in all dimensions. Uses a convolution which has the symmetries of the Iisng model

    """

    def __init__(self, upsampling_method='nearest'):
        super(DecoderNearestNeighborsConv, self).__init__()

        if upsampling_method == 'nearest':
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        elif upsampling_method == 'bilinear':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            raise NotImplementedError(f"choose between 'nearest' or 'bilinear' for upsampling, not {upsampling_method}")

        self.convs = NearestNeighborsConv2d(in_channels=1, out_channels=1, stride=1, padding='same',
                                            padding_mode='circular')

    def forward(self, x):
        return self.convs(self.up(x))

    def upsample(self, x, factor=2):
        assert factor % 2 == 0, f"argument 'factor' of method 'upsample' should be a power of 2, not {factor}. Default: 2"
        while factor % 2 == 0:
            x = self.forward(x)
            x = (torch.rand_like(x) < torch.sigmoid(x)).float()
            factor /= 2
        return x


class Decoder1layer(nn.Module):
    """
    Upsamples an Ising lattice by a factor 2 in all dimensions.

    Input tensor of shape (1, H, W)
    Output tensor of shape (1, 2H, 2W)  with H and W are powers of 2
    """

    def __init__(self, upsampling_method='nearest', kernel_size: int = 5, int_channels: int = 32):
        super(Decoder1layer, self).__init__()

        if upsampling_method == 'nearest':
            self.up = nn.UpsamplingNearest2d(scale_factor=2)
        elif upsampling_method == 'bilinear':
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            raise NotImplementedError(f"choose between 'nearest' or 'bilinear' for upsampling, not {upsampling_method}")

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=int_channels, kernel_size=kernel_size, stride=1, padding='same',
                      padding_mode='circular'),
            nn.ReLU(),
            nn.Conv2d(in_channels=int_channels, out_channels=1, kernel_size=kernel_size, stride=1, padding='same',
                      padding_mode='circular'),
        )

    def forward(self, x):
        return self.convs(self.up(x))

    def upsample(self, x, factor=2):
        assert factor % 2 == 0, f"argument 'factor' of method 'upsample' should be a power of 2, not {factor}. Default: 2"
        while factor % 2 == 0:
            x = self.forward(x)
            x = (torch.rand_like(x) < torch.sigmoid(x)).float()
            factor /= 2
        return x



class LossMagnetization(nn.Module):
    """
    Compute logisitc loss + cost if the magnetization is not the same
    TODO: does not seem to work if gamma != 0
    """

    def __init__(self, gamma=1):
        super(LossMagnetization, self).__init__()
        self.gamma = gamma

#        self.mainloss = nn.BCEWithLogitsLoss()
        
        #self.weights=DegWeights(kernel_size=(2, 2),stride=2,padding=0)

    def forward(self, ypred, ytrue):
        #degW=self.weights(ytrue)
        #print(degW)
        #print(degW[1,0].float())
        #degW/=degW.max().float()
        #print(degW.size())
        #print(ypred.size())
        #print(ytrue.size())
        self.mainloss = nn.BCEWithLogitsLoss()
        
        return self.mainloss(ypred, ytrue) + self.gamma * torch.pow(torch.sum(ypred) - torch.sum(ytrue), 2)

class LossMagnetizationW(nn.Module):
    """
    Compute logisitc loss + cost if the magnetization is not the same
    TODO: does not seem to work if gamma != 0
    """

    def __init__(self, gamma=1):
        super(LossMagnetizationW, self).__init__()
        self.gamma = gamma

#        self.mainloss = nn.BCEWithLogitsLoss()
        
        self.weights=DegWeights(kernel_size=(2, 2),stride=2,padding=0)

    def forward(self, ypred, ytrue):
        degW=self.weights(ytrue)
        #print(degW)
        #print(degW[1,0].float())
        degW/=degW.mean().float()
        #print(degW.size())
        #print(ypred.size())
        #print(ytrue.size())
        self.mainloss = nn.BCEWithLogitsLoss(weight=degW)
        
        return self.mainloss(ypred, ytrue) + self.gamma * torch.pow(torch.sum(ypred) - torch.sum(ytrue), 2)

