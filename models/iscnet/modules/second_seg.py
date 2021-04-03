# Back propogate box features to input points.
# author: ynie
# date: March, 2020
# cite: PointNet++

from models.registers import MODULES
import torch
from torch import nn
from models.iscnet.modules.pointseg import PointSeg, get_loss

@MODULES.register_module
class Seg_2nd(nn.Module):
    ''' Back-Propagte box proposal features to input points
    '''
    def __init__(self, cfg, optim_spec=None):
        super(Seg_2nd, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Network parameters'''
        self.input_feature_dim = int(cfg.config['data']['use_color_completion']) * 3 + int(not cfg.config['data']['no_height']) * 1

        '''Modules'''
        self.point_seg = PointSeg(num_class=2, channel=self.input_feature_dim + 3 + 1)
        self.mask_loss_func = get_loss()

    def forward(self, input_for_seg, gt_for_seg, probs_for_query, point_masks):

        input_features = torch.cat([input_for_seg, probs_for_query.unsqueeze(1)], dim=1)
        seg_pred, trans_feat = self.point_seg(input_features)
        seg_pred = seg_pred.contiguous().view(-1, 2)
        point_masks = point_masks.contiguous().view(-1)
        point_mask_loss = self.mask_loss_func(seg_pred, gt_for_seg, point_masks, trans_feat, weight=None)

        return point_mask_loss