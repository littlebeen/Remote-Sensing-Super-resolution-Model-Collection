import torch
from torch import nn, einsum
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
import numpy as np


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)
            
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)

#         print("Q shape: ",q.shape)
#         print("K shape: ",k.shape)
        dots = torch.matmul(q, k.transpose(-1, -2))*self.scale
        attn = torch.nn.Softmax(dim=-1)(dots)
        out = torch.matmul(attn, v)
            
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    

class ConvAttentionNonSq(nn.Module):
    def __init__(self, dim, img_size1, img_size2, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False):

        super().__init__()
        self.last_stage = last_stage
        self.img_size1 = img_size1
        self.img_size2 = img_size2
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h)
            
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size1, w=self.img_size2)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)

#         print("Q shape: ",q.shape)
#         print("K shape: ",k.shape)
        dots = torch.matmul(q, k.transpose(-1, -2))*self.scale
        attn = torch.nn.Softmax(dim=-1)(dots)
        out = torch.matmul(attn, v)
            
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    
class TransformerNonSq(nn.Module):
    def __init__(self, dim, img_size1, img_size2, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttentionNonSq(dim, img_size1, img_size2, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, img_size, depth, heads, dim_head, mlp_dim, dropout=0., last_stage=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, ConvAttention(dim, img_size, heads=heads, dim_head=dim_head, dropout=dropout, last_stage=last_stage)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

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


class CvT(nn.Module):
    def __init__(self, image_size, in_channels,out_channels = 32 , dim=54, kernels=[3, 3, 3], strides=[1,1,2],
                 heads=[4,4,4] , depth = [1, 1, 1], pool='cls', dropout=0., emb_dropout=0., scale_dim=2,):
        super().__init__()

        self.dim = dim

        ##### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[0], strides[0], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size//1, w = image_size//1),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size//1,depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size//1, w = image_size//1)
        )

        ##### Stage 2 #######
        in_channels = dim
        scale = heads[1]//heads[0]
        dim = scale*dim
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size, w = image_size),
            nn.LayerNorm(dim)
        )
        self.stage2_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size//1, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size//1, w = image_size//1)
        )

        ##### Stage 3 #######
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[2], strides[2], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size//2, w = image_size//2),
            nn.LayerNorm(dim)
        )
        self.stage3_transformer = nn.Sequential(
            Transformer(dim=dim, img_size=image_size//2, depth=depth[2], heads=heads[2], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout, last_stage=False),
        )


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.lastConv = nn.Conv2d(dim,out_channels,3,1,1)
        self.upsampler = UpSampleBlock(out_channels)
    def forward(self, img):
        
        xs = self.stage1_conv_embed(img)
#         print('xs before pool',xs.shape)        

        xs = self.stage1_transformer(xs)
#         print('xs before pool',xs.shape)
        xs = self.stage2_conv_embed(xs)
#         print('xs before pool',xs.shape)
        xs = self.stage2_transformer(xs)
#         print('xs before pool',xs.shape)

        xs = self.stage3_conv_embed(xs)
#         print('xs before pool',xs.shape)
#         b, n, _ = xs.shape
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage3_transformer(xs)
#         print('xs before pool',xs.shape)        
#         xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0])
#         xs = self.mlp_head(xs)
        xs = xs.permute(0,2,1)
#         print('xs before pool',xs.shape)        
        xs = xs.contiguous().view(xs.shape[0],xs.shape[1],img.shape[2]//2,img.shape[3]//2)
#         print('xs before pool',xs.shape)        
        xs = self.lastConv(xs)
        xs = self.upsampler(xs)   
        
#         print('xs after mlp',xs.shape)
#         xs = xs.contiguous().view(img.shape[0],1,img.shape[2]//2,img.shape[3]//2)
#         xs = self.lastConv(xs)
        return xs


class CvTNonSquare(nn.Module):
    def __init__(self, image_size1, image_size2, in_channels,out_channels = 32 , dim=54, kernels=[3, 3, 3], strides=[1,1,2],
                 heads=[4,4,4] , depth = [1, 1, 1], pool='cls', dropout=0., emb_dropout=0., scale_dim=2,):
        super().__init__()

        self.dim = dim

        ##### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[0], strides[0], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size1//1, w = image_size2//1),
            nn.LayerNorm(dim)
        )
        self.stage1_transformer = nn.Sequential(
            TransformerNonSq(dim=dim, img_size1=image_size1//1, img_size2=image_size2//1, depth=depth[0], heads=heads[0], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size1//1, w = image_size2//1)
        )

        ##### Stage 2 #######
        in_channels = dim
        scale = heads[1]//heads[0]
        dim = scale*dim
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[1], strides[1], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size1, w = image_size2),
            nn.LayerNorm(dim)
        )
        self.stage2_transformer = nn.Sequential(
            TransformerNonSq(dim=dim, img_size1=image_size1//1, img_size2=image_size2//1, depth=depth[1], heads=heads[1], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout),
            Rearrange('b (h w) c -> b c h w', h = image_size1//1, w = image_size2//1)
        )

        ##### Stage 3 #######
        in_channels = dim
        scale = heads[2] // heads[1]
        dim = scale * dim
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels[2], strides[2], 1),
            Rearrange('b c h w -> b (h w) c', h = image_size1//2, w = image_size2//2),
            nn.LayerNorm(dim)
        )
        self.stage3_transformer = nn.Sequential(
            TransformerNonSq(dim=dim, img_size1=image_size1//2, img_size2=image_size2//2, depth=depth[2], heads=heads[2], dim_head=self.dim,
                                              mlp_dim=dim * scale_dim, dropout=dropout, last_stage=False),
        )


        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout_large = nn.Dropout(emb_dropout)

        self.lastConv = nn.Conv2d(dim,out_channels,3,1,1)
        self.upsampler = UpSampleBlock(out_channels)
    def forward(self, img):
        
        xs = self.stage1_conv_embed(img)
#         print('xs before pool',xs.shape)        

        xs = self.stage1_transformer(xs)
#         print('xs before pool',xs.shape)
        xs = self.stage2_conv_embed(xs)
#         print('xs before pool',xs.shape)
        xs = self.stage2_transformer(xs)
#         print('xs before pool',xs.shape)

        xs = self.stage3_conv_embed(xs)
#         print('xs before pool',xs.shape)
#         b, n, _ = xs.shape
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage3_transformer(xs)
#         print('xs before pool',xs.shape)        
#         xs = xs.mean(dim=1) if self.pool == 'mean' else xs[:, 0])
#         xs = self.mlp_head(xs)
        xs = xs.permute(0,2,1)
#         print('xs before pool',xs.shape)        
        xs = xs.contiguous().view(xs.shape[0],xs.shape[1],img.shape[2]//2,img.shape[3]//2)
#         print('xs before pool',xs.shape)        
        xs = self.lastConv(xs)
        xs = self.upsampler(xs)   
        
#         print('xs after mlp',xs.shape)
#         xs = xs.contiguous().view(img.shape[0],1,img.shape[2]//2,img.shape[3]//2)
#         xs = self.lastConv(xs)
        return xs