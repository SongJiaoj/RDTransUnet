import torch.nn as nn
import torch
import torch.nn.functional as F
from .CTrans5 import ChannelTransformer
from torch import Tensor
from typing import List

#获取激活函数的实例
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

#创建包含多个卷积层的序列
def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

    
#下采样操作
class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.maxpool(x)
        x0 = self.conv1(out)
        out1 = self.nConvs(out)
        out = x0 + out1
        return out

#将输入张量展平为一维向量
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

#实现通道注意力机制
class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out
    
class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    
#实现上采样操作，并结合通道注意力机制进行特征融合
class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        #self.coatt 是一个 CCA 对象，用于应用通道注意力机制。它接受输入通道数的一半 F_g 和输入通道数的一半 F_x
        #self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.eca=eca_layer(in_channels // 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        up=self.eca(up)
        skin_x=self.eca(skip_x)
        #skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skin_x, up], dim=1)
        #x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


class skip5(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=560,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 16, nb_Conv=2)

        self.maxpool = nn.MaxPool2d(2)
        
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8, in_channels*16],
                                     patchSize=config.patch_sizes)
        self.up5 = _make_nConv(in_channels*16, in_channels*8, nb_Conv=2)
        self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss

    def forward(self, x):  
        x = x.float()
        #print('x',x.shape)
        x1 = self.inc(x)
        #print('x1', x1.shape)
        x2 = self.down1(x1)
        #print('x2', x2.shape)
        x3 = self.down2(x2)
        #print('x3', x3.shape)
        x4 = self.down3(x3)
        #print('x4', x4.shape)
        x5 = self.down4(x4)
        #print('x5', x5.shape)
        
        x1,x2,x3,x4,x5,att_weights = self.mtc(x1,x2,x3,x4,x5)
        # print('x1', x1.shape)
        # print('x2', x2.shape)
        # print('x3', x3.shape)
        # print('x4', x4.shape)
        # print('x5', x5.shape)
        x5 = self.up5(x5)
        #print('x5', x5.shape)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes ==1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x) # if nusing BCEWithLogitsLoss or class>1
        if self.vis: # visualize the attention maps
            return logits, att_weights
        else:
            return logits




