import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from ..base_model import BaseModel
from ..modules import FeedForward, HorizontalPool, VerticalPool, TemporalPool, Transformer, PositionalEncodings, PatchEmbeddings, ConvolutionalStream, CopyInterleave, PyramidPooling
from utils import visualisation as TSNE_Visual
import numpy as np



class ViViT(BaseModel):
    def __init__(self, cfgs, training):
        super(ViViT, self).__init__(cfgs, training)
        

    def build_network(self, model_cfg):

        self.image_size = 64
        # self.num_frames = 32
        self.depth = np.array(model_cfg['tub_depth']) ## [4, 8, 16]
        self.channels = model_cfg['channels']
        self.patch_size = np.array(model_cfg['patch_size']) ## [4, 4, 8]
        self.kernel = np.array(model_cfg['kernel_size'])
        self.dropout = 0.
        self.emb_dropout = 0.
        self.scale_dim = 4
        self.cls_num = 74
        dim = np.multiply(self.depth, self.patch_size**2) ## [64, 128, 1024]

        self.convStream_L1 = ConvolutionalStream(self.kernel[2:5], in_c=self.channels[0], out_c=self.channels[1])
        self.convStream_L2 = ConvolutionalStream(self.kernel[1:4], in_c=self.channels[1], out_c=self.channels[1])
        self.convStream_L3 = ConvolutionalStream(self.kernel[0:3], in_c=self.channels[1], out_c=self.channels[1])

        self.dropout = nn.Dropout(self.emb_dropout)

        self.FCNN = FeedForward(128, self.cls_num)
        self.FCNN_TL = FeedForward(128, 128)
    

    def forward(self, x4):
        ipts, labs, t, v, seqL = x4
        x = ipts[0].unsqueeze(1)
        
        '''First Layer'''
        x = self.convStream_L1(x)

        
        ''' Second Layer '''
        x = self.convStream_L2(x)



        ''' Third Layer'''
        x = self.convStream_L3(x)
        x = torch.max(x, dim = 2)[0]
        x= rearrange(x, 'b c h w -> b (h w) c') ## [6, 64, 128]
           


        '''
        Implementing for CE Loss
        
        '''
        embed1 = self.FCNN(x) ## [6, 64, 74]
        embed1 = rearrange(embed1, 'b p c -> b c p').contiguous() ## [6, 74, 64]



        '''
        Implementing for Triplet Loss
        
        ''' 
        embed2 = self.FCNN_TL(x)
        embed2 = rearrange(embed2, 'b p c -> b c p').contiguous() ## [6, 128, 64]


        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed2, 'labels': labs},
                'softmax': {'logits': embed1, 'labels': labs}
                # 'temporal_attn': {'attn': temporal_adj_mat, 'labels': labs},
                # 'spatial_attn': {'attn': space_adj_mat, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(ipts, 'n c t h w -> (n c t) 1 h w') #ipts == [6, 1, 30, 64, 64]
            },
            'inference_feat': {
                'embeddings': embed2
            },
            'tsne_plot' :{
                'tsne_feature' : embed2
            },
            'tsne_plot_ca' : {
                'tsne_feature' : rearrange(embed2, 'b p c -> b (p c)')
            }
        }

        return retval