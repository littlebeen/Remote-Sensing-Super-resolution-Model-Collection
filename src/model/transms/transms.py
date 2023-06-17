
from distutils.command.config import config
import torch
import torch.nn as nn
from .util.cvtModule import UpSampleBlock,CvTNonSquare
from .util.rdnModule import rdn1x
from math import log2
n1=32
config = {
    'n1':32,
    'inChannel' :3,
    'outChannel':3,
    'initialConvFeatures' : 24, #### free param
    #'scaleFactor':4, 
    'rdn_nb_of_features' : 64,   
    'rdn_nb_of_blocks' : 32,
    'rdn_layer_in_each_block' : 18, #### untitled.png
    'rdn_growth_rate' : 6,
    'img_size_train' : n1, #e.g 8 or (6,8) input image size ####
    "cvt_out_channels": 64,#C1 ####
    "cvt_dim" : 64,   #or256
    "convAfterConcatLayerFeatures" : 48 #C3 ###
}


class TRSMSMS(nn.Module):
    """config = {
    'initialConvFeatures' : 32, #### free param
    'scaleFactor':scale_factor, 
    'rdn_nb_of_features' : 24,
    'rdn_nb_of_blocks' : 4,
    'rdn_layer_in_each_block' : 5, #### untitled.png
    'rdn_growth_rate' : 6,
    'img_size1' : 8, #e.g 8 or (6,8) input image size ####
    'img_size2' : 8, #e.g 8 or (6,8) input image size ####
    "cvt_out_channels": 32,#C1 ####
    "cvt_dim" : 32
    "convAfterConcatLayerFeatures" : 32 #C3 ###
    }"""
    def __init__(self,args):
        super(TRSMSMS, self).__init__()
        self.initialConv = nn.Conv2d(config['inChannel'],config['initialConvFeatures'],3,1,1)
        self.rdn = rdn1x(input_channels = config['initialConvFeatures'], 
                         nb_of_features = config['rdn_nb_of_features'], 
                         nb_of_blocks = config['rdn_nb_of_blocks'],
                        layer_in_each_block = config["rdn_layer_in_each_block"],
                        growth_rate = config["rdn_growth_rate"])
        self.transformer = CvTNonSquare(image_size1 = config["img_size_train"], image_size2 = config["img_size_train"], in_channels = config['initialConvFeatures'], out_channels = config["cvt_out_channels"],dim =config["cvt_dim"] )
        self.convAfterConcat = nn.Conv2d(config['rdn_nb_of_features']+config['cvt_out_channels'],
                                      config["convAfterConcatLayerFeatures"],3,1,1)
        upSamplersList = []

        if(args.scale==3):
             upSamplersList.append(UpSampleBlock(config["convAfterConcatLayerFeatures"],args.scale))
        if args.scale==4:
            for _ in range(int(log2(args.scale))):   
                upSamplersList.append(UpSampleBlock(config["convAfterConcatLayerFeatures"],2))
        
        self.upSampler = nn.Sequential(*upSamplersList)
        self.lastConv = nn.Conv2d(config['convAfterConcatLayerFeatures'],config['outChannel'],3,1,1)
    def forward(self,x):
        x = self.initialConv(x)  #24 32 32
        rdnSkip = self.rdn(x) #24 32 32
        x = self.transformer(x) #64 32 32
        x = torch.cat([x, rdnSkip], dim=1) #88 32 32
        x = self.convAfterConcat(x) #48 32 32
        x = self.upSampler(x)# 48 96 96
        x = self.lastConv(x) #3 96 96
        return x