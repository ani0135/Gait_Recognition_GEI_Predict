# import torch.nn.functional as F
import torch

from .base import BaseLoss
from einops import rearrange


class EuclidianDist(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(EuclidianDist, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, pred_gei, org_gei):
        """
            pred_gei: [n, h, w]
            gei: [n, h, w]
        """
        # pred_gei = pred_gei
        # pred_gei = rearrange(pred_gei, 'n h w -> n (h w)')
        # org_gei = rearrange(org_gei, 'n h w -> n (h w)')
        loss = torch.cdist(pred_gei, org_gei, compute_mode='use_mm_for_euclid_dist_if_necessary')
        self.info.update({'loss': loss.detach().clone()})
        return loss, self.info
