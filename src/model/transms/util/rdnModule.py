import torch
from torch import nn
import numpy as np

class UpSampleBlock(nn.Module):
    def __init__(self,input_channels,scale_factor=2):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels,input_channels*scale_factor**2,kernel_size=3,stride=1,padding=1)
        self.shuffler = nn.PixelShuffle(scale_factor)
        self.lrelu = nn.LeakyReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.shuffler(x)
        return self.lrelu(x)
    
class dense_block(nn.Module):
    def __init__(self, in_channels, addition_channels):
        super(dense_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=addition_channels, kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))],dim=1)
    
class rdb(nn.Module):
    def __init__(self, in_channels, C, growth_at_each_dense):
        super(rdb, self).__init__()
        denses = nn.ModuleList()
        for i in range(0,C):
            denses.append(dense_block(in_channels+i*growth_at_each_dense,growth_at_each_dense))
        self.local_res_block = nn.Sequential(*denses)
        self.last_conv = nn.Conv2d(in_channels=in_channels+C*growth_at_each_dense,out_channels=in_channels,kernel_size=1,stride=1,padding=0)
    def forward(self,x):
        return x + self.last_conv(self.local_res_block(x))
    

class rdn1x(nn.Module):
    def __init__(self,input_channels, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate):
        super(rdn1x,self).__init__()
        self.conv0 = nn.Conv2d(input_channels,nb_of_features, kernel_size = 3, stride = 1, padding=1)
        self.conv1 = nn.Conv2d(nb_of_features, nb_of_features, kernel_size = 3, stride = 1, padding=1)
        self.rdbs = nn.ModuleList()
        for i in range(0,nb_of_blocks):
            self.rdbs.append(rdb(nb_of_features, layer_in_each_block, growth_rate))
        self.conv2 = nn.Conv2d(in_channels=nb_of_blocks*nb_of_features, out_channels= nb_of_features,kernel_size=1,stride=1,padding=0)
        self.conv3 = nn.Conv2d(in_channels=nb_of_features, out_channels= nb_of_features,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        x = self.conv0(x)
        residual0 = x
        x = self.conv1(x)
        rdb_outs = list()
        for layer in self.rdbs:
            x = layer(x)
            rdb_outs.append(x)
        x = torch.cat(rdb_outs, dim=1)
        x = self.conv2(x)
        x = self.conv3(x) +residual0
        return x