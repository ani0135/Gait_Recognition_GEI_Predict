import torch
import torch.nn as nn
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks
from torchvision.utils import save_image

from einops import rearrange

def double_convolution(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the 
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv_op

class GeiPredict(BaseModel):
    def __init__(self, *args, **kargs):
        super(GeiPredict, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        channels    = model_cfg['channels']
        self.max_pool2d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(1, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)

        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose3d(
            in_channels=256, out_channels=128,
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2))
        
        self.up_convolution_1 = double_convolution(256, 128)
        self.up_transpose_2 = nn.ConvTranspose3d(
            in_channels=128, out_channels=64,
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2))
        self.up_convolution_2 = double_convolution(128, 64)
        self.out = nn.Conv3d(
            in_channels=64, out_channels=1, 
            kernel_size=1
        )
        self.FCs = SeparateFCs(16, channels[2], channels[1])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num']) 
    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        # sils = ipts[0]
        # if len(sils.size()) == 4:
        #     sils = sils.unsqueeze(1)
        # else:
        #     sils = rearrange(sils, 'n s c h w -> n c s h w')

        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        ## Down Convolution
        down_1 = self.down_convolution_1(sils)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)      
        
        ## Up Convolution
        up_1 = self.up_transpose_1(down_5)
        x = self.up_convolution_1(torch.cat([down_3, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_1, up_2], 1)) # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(down_5, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]

        out = self.out(x).squeeze(1)
        embed = torch.mean(out, dim = 1).squeeze(1)
        org_gei = torch.mean(sils, dim = 2, dtype=torch.float16).squeeze(1)

        retval = {
           'training_feat': {
               'triplet': {'embeddings': embed_1, 'labels': labs},
               'gei': {'pred_gei': embed, 'org_gei': org_gei}
                   },
           'visual_summary': {
               'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
           },
           'inference_feat': {
               'embeddings': embed_1
           }
       }
        return retval