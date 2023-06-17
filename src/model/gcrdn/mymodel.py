# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
import torch
import torch.nn as nn
from ..util import nlsa
from ..util.rdn import RDB_Conv
from ..util.attenPConv import AttPConv
from ..util.dbpn import DenseProjection
from utils.tools import draw_features

def make_model(args, parent=False):
    return MYMODEL(args)

class RDB(nn.Module):
    def __init__(self, args, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers  #8
        #nlsa
        n_feats=64
        chunk_size=args.chunk_size
        res_scale =0.1
        n_hashes=4 

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Sequential(*[nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)])
        #nlsa
        self.nlsa=nlsa.NonLocalSparseAttention(
              channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=res_scale)

    def forward(self, x):
        x=self.LFF(self.convs(x)) +x
        x= self.nlsa(x)
        return x

class MYMODEL(nn.Module):
    def __init__(self, args):
        super(MYMODEL, self).__init__()
        r = int(args.scale)
        G0 = args.G0  #64
        kSize = args.RDNkSize  #3

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]  #B

        # Shallow feature extraction net
        rgb_std = (1.0, 1.0, 1.0)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)



        #Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(args, growRate0 = G0, growRate = G, nConvLayers = C)
            )
          
        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        #dbpn
        nr = 64
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = G0
        self.depth = args.dbdepth  #6
        for i in range(self.depth):
            self.upmodules.append(
                DenseProjection(channels, nr, r, True, i > 1)
            )
            channels += nr
        
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(
                DenseProjection(channels, nr, r, False, i != 0)
            )
            channels += nr
        
         # Up-sampling net
        if r == 2 or r == 3 or r == 4:
              self.UPNet = nn.Sequential(*[
                nn.Conv2d(self.depth * nr, G0, 1, padding=0, stride=1),
                nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        # elif r == 4:
        #     self.UPNet = nn.Sequential(*[
        #         nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
        #         nn.PixelShuffle(2),
        #         nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
        #         nn.PixelShuffle(2),
        #         nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
        #     ])
        # else:
        #     raise ValueError("scale must be 2 or 3 or 4.")


    def forward(self, x):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1
        #draw_features(64, x,"{}/encoder.png".format('../image'))

        h_list = [] #64 96 96
        l_list = [x] #64 32 32
        for i in range(self.depth - 1):
            l = torch.cat(l_list, dim=1)
            h_list.append(self.upmodules[i](l)) # 16 32 96 96
            l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))
        
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))

        #hotmap
        # x1 = self.UPNet[0](torch.cat(h_list, dim=1))
        # x1 = self.UPNet[1](x1)
        # draw_features(64, x1,"{}/decoder.png".format('../image'))
        
        x = self.UPNet(torch.cat(h_list, dim=1))
        x = self.add_mean(x)
        return x
