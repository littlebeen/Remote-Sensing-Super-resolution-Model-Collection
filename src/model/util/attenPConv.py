import torch
import torch.nn as nn

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        #self.a = torch.nn.Parameter(torch.Tensor([0]))
        #self.a.requires_grad=True
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class AttPConv(nn.Module):

    def __init__(self, inplans=64, planes=64,  pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(AttPConv, self).__init__()
        self.conv2_1 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = nn.Conv2d(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = nn.Conv2d(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        x = torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)
        return x


class AttPConv2(nn.Module): #含CA
    def __init__(self,in_channels):
        super(AttPConv, self).__init__()
        self.BN=nn.BatchNorm2d(in_channels)
        self.LR=nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.con1= nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=7,padding=3, groups=8)
        self.con2= nn.Conv2d(in_channels=in_channels, out_channels=in_channels//4, kernel_size=5,padding=2, groups=4)
        self.con3= nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3,padding=1, groups=1)
        self.ca=nn.ModuleList()
        for i in range(3):
            if(i!=2):
                self.ca.append(CALayer(in_channels//4))
            else:
                self.ca.append(CALayer(in_channels//2))
        self.BN2=nn.BatchNorm2d(in_channels)
        self.LR2=nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.local_fuse = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.BN3=self.BN=nn.BatchNorm2d(in_channels)
    def forward(self, x):
        x=self.BN(x)
        x = self.LR(x)
        x1 = self.con1(x) 
        x1=self.ca[0](x1)  
        x2 = self.con2(x)
        x2=self.ca[1](x2)  
        x3 = self.con3(x)
        x3=self.ca[2](x3)  #64 32 32
        x = torch.concat([x1,x2,x3],dim=1)
        x=self.BN2(x)
        x = self.LR2(x)
        x = self.local_fuse(x)
        x=self.BN3(x)
        return x
