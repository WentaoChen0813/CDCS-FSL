# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

# Basic ResNet model

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

class distLinear(nn.Module):
    def __init__(self, indim, outdim, scale=2):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        # if outdim <=200:
        #     self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github
        # else:
        #     self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.
        self.scale_factor = scale

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

class protoLinear(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.prototypes = nn.Parameter(torch.ones(indim, outdim))
        nn.init.kaiming_uniform_(self.prototypes)

    def forward(self, x):
        dist = (x.unsqueeze(-1) - self.prototypes.unsqueeze(0)) ** 2
        dist = dist.sum(dim=1)
        score = -dist
        return score

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out

class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out
            
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight 
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, pool = True, padding = 1):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding)
            self.BN     = BatchNorm2d_fw(outdim)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding)
            self.BN     = nn.BatchNorm2d(outdim)
        self.relu   = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)


    def forward(self,x):
        out = self.trunk(x)
        return out


class DANormalization(nn.BatchNorm2d):

    def set_use_cache(self, mode=False):
        self.use_cache = mode

    def forward(self, input):
        if not hasattr(self, 'use_cache'):
            self.use_cache = False
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats and not self.use_cache:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.use_cache:
            # only use cache once until set it manually
            self.use_cache = False
            return F.batch_norm(
            input, self.cache_mean, self.cache_var, self.weight, self.bias,
            False, exponential_average_factor, self.eps)
        else:
            with torch.no_grad():
                self.cache_mean = torch.mean(input, dim=(0, 2, 3))
                self.cache_var = torch.var(input, dim=(0, 2, 3), unbiased=False)
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)


class DSBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.choice = 'a'

    def forward(self, input):
        if self.choice == 'a':
            return self.bn1(input)
        elif self.choice == 'b':
            # use bn1 as default, use bn2 only once after manually setting.
            self.choice = 'a'
            return self.bn2(input)
        elif self.choice == 'split':
            self.choice = 'a'
            x1, x2 = torch.chunk(input, 2, dim=0)
            x1 = self.bn1(x1)
            x2 = self.bn2(x2)
            return torch.cat([x1, x2], dim=0)

    def set_choice(self, choice):
        assert choice in ['a', 'b', 'split']
        self.choice = choice


class AdaBatchNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn2.weight = self.bn1.weight
        self.bn2.bias = self.bn1.bias
        self.choice = 'a'

    def forward(self, input):
        if self.choice == 'a':
            return self.bn1(input)
        elif self.choice == 'b':
            # use bn1 as default, use bn2 only once after manually setting.
            self.choice = 'a'
            return self.bn2(input)
        elif self.choice == 'split':
            self.choice = 'a'
            x1, x2 = torch.chunk(input, 2, dim=0)
            x1 = self.bn1(x1)
            x2 = self.bn2(x2)
            return torch.cat([x1, x2], dim=0)

    def set_choice(self, choice):
        assert choice in ['a', 'b', 'split']
        self.choice = choice


class AdaptiveSharedBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:
            alpha = torch.clamp(self.alpha, 0., 1.)
            x1, x2 = torch.chunk(input, 2, dim=0)
            mean1 = torch.mean(x1, dim=(0, 2, 3))
            mean2 = torch.mean(x2, dim=(0, 2, 3))
            mean = alpha * mean1 + (1-alpha) * mean2
            var1 = ((x1 - mean.view(1, -1, 1, 1))**2).mean(dim=(0, 2, 3))
            var2 = ((x2 - mean.view(1, -1, 1, 1))**2).mean(dim=(0, 2, 3))
            var = alpha * var1 + (1-alpha) * var2
            input = (input - mean.view(1, -1, 1, 1)) / torch.sqrt(var.view(1, -1, 1, 1) + self.eps)
            input = self.weight.view(1, -1, 1, 1) * input + self.bias.view(1, -1, 1, 1)
            with torch.no_grad():
                self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * mean
                self.running_var = (1 - exponential_average_factor) * self.running_var + exponential_average_factor * var
            return input
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.choice = '0'

    def forward(self, input):
        if self.choice == '0':
            return self.conv0(input)
        elif self.choice == 'a':
            self.choice = '0'
            return self.conv0(input) + self.conv1(input)
        elif self.choice == 'b':
            self.choice = '0'
            return self.conv0(input) + self.conv2(input)
        elif self.choice == 'split':
            self.choice = '0'
            x1, x2 = torch.chunk(input, 2)
            x1 = self.conv0(x1) + self.conv1(x1)
            x2 = self.conv0(x2) + self.conv2(x2)
            return torch.cat([x1, x2])

    def set_choice(self, choice):
        assert choice in ['0', 'a', 'b', 'split']
        self.choice = choice


class DSWBNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.weight2 = nn.Parameter(torch.ones_like(self.weight), requires_grad=True)
        self.bias2 = nn.Parameter(torch.zeros_like(self.bias), requires_grad=True)
        self.choice = 'a'

    def forward(self, input):
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.choice == 'a':
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        elif self.choice == 'b':
            self.choice = 'a'
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight2, self.bias2,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        elif self.choice == 'split':
            self.choice = 'a'
            x1, x2 = torch.chunk(input, 2, dim=0)
            x1 = F.batch_norm(
                x1, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
            x2 = F.batch_norm(
                x2, self.running_mean, self.running_var, self.weight2, self.bias2,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
            return torch.cat([x1, x2])

    def set_choice(self, choice):
        assert choice in ['a', 'b', 'split']
        self.choice = choice

# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False #Default
    bn = 'normal'
    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            if self.bn in ['normal', 'dsconv']:
                self.BN1 = nn.BatchNorm2d(outdim)
                self.BN2 = nn.BatchNorm2d(outdim)
            elif self.bn == 'dan':
                self.BN1 = DANormalization(outdim)
                self.BN2 = DANormalization(outdim)
            elif self.bn == 'dsbn':
                self.BN1 = DSBatchNorm(outdim)
                self.BN2 = DSBatchNorm(outdim)
            elif self.bn == 'adabn':
                self.BN1 = AdaBatchNorm(outdim)
                self.BN2 = AdaBatchNorm(outdim)
            elif self.bn == 'asbn':
                self.BN1 = AdaptiveSharedBatchNorm(outdim)
                self.BN2 = AdaptiveSharedBatchNorm(outdim)
            elif self.bn == 'dswb':
                self.BN1 = DSWBNorm(outdim)
                self.BN2 = DSWBNorm(outdim)
            else:
                raise ValueError(self.bn)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                if self.bn in ['normal', 'dsconv']:
                    self.BNshortcut = nn.BatchNorm2d(outdim)
                elif self.bn == 'dan':
                    self.BNshortcut = DANormalization(outdim)
                elif self.bn == 'dsbn':
                    self.BNshortcut = DSBatchNorm(outdim)
                elif self.bn == 'adabn':
                    self.BNshortcut = AdaBatchNorm(outdim)
                elif self.bn == 'asbn':
                    self.BNshortcut = AdaptiveSharedBatchNorm(outdim)
                elif self.bn == 'dswb':
                    self.BNshortcut = DSWBNorm(outdim)
                else:
                    raise ValueError(self.bn)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False #Default
    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten = True):
        super(ConvNet,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,19,19]

    def forward(self,x):
        out = self.trunk(x)
        return out

class ConvNetS(nn.Module): #For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten = True):
        super(ConvNetS,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i <4 ) ) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ConvNetSNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super(ConvNetSNopool,self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1  ) #only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64,5,5]

    def forward(self,x):
        out = x[:,0:1,:,:] #only use the first dimension
        out = self.trunk(out)
        return out

class ResNet(nn.Module):
    maml = False #Default
    bn = 'normal'
    def __init__(self,block,list_of_num_layers, list_of_out_dims, flatten = True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        # bn_cache = SimpleBlock.bn
        # SimpleBlock.bn = 'normal'
        # self.bn = 'normal'
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False)
            if self.bn == 'normal':
                bn1 = nn.BatchNorm2d(64)
            elif self.bn == 'dan':
                bn1 = DANormalization(64)
            elif self.bn == 'dsbn':
                bn1 = DSBatchNorm(64)
            elif self.bn == 'adabn':
                bn1 = AdaBatchNorm(64)
            elif self.bn == 'asbn':
                bn1 = AdaptiveSharedBatchNorm(64)
            elif self.bn == 'dsconv':
                conv1 = DSConv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                bn1 = nn.BatchNorm2d(64)
            elif self.bn == 'dswb':
                bn1 = DSWBNorm(64)
            else:
                raise ValueError(self.bn)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)


        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):
            # if i == 2:
            #     SimpleBlock.bn = bn_cache
            for j in range(list_of_num_layers[i]):
                half_res = (i>=1) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out

    def set_bn_use_cache(self, mode):
        for module in self.modules():
            if isinstance(module, DANormalization):
                module.set_use_cache(mode)

    def set_bn_choice(self, choice):
        for module in self.modules():
            if isinstance(module, DSBatchNorm) or isinstance(module, AdaBatchNorm) or isinstance(module, DSWBNorm):
                module.set_choice(choice)

    def set_conv_choice(self, choice):
        for module in self.modules():
            if isinstance(module, DSConv):
                module.set_choice(choice)

def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)

def Conv4NP():
    return ConvNetNopool(4)

def Conv6NP():
    return ConvNetNopool(6)

def Conv4S():
    return ConvNetS(4)

def Conv4SNP():
    return ConvNetSNopool(4)

def ResNet10( flatten = True):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], flatten)

def ResNet18( flatten = True):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], flatten)

def ResNet34( flatten = True):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], flatten)

def ResNet50( flatten = True):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], flatten)

def ResNet101( flatten = True):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], flatten)




