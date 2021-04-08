# Back propogate box features to input points.
# author: ynie
# date: March, 2020
# cite: PointNet++

from models.registers import MODULES
import torch
from torch import nn
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import STN_Group
from models.iscnet.modules.layers import ResnetPointnet
from models.iscnet.modules.pointseg import PointSeg, get_loss

@MODULES.register_module
class SkipPropagation(nn.Module):
    ''' Back-Propagte box proposal features to input points
    '''
    def __init__(self, cfg, optim_spec=None):
        super(SkipPropagation, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Network parameters'''
        self.input_feature_dim = int(cfg.config['data']['use_color_completion']) * 3 + int(not cfg.config['data']['no_height']) * 1

        '''Modules'''
        self.stn = STN_Group(
                radius=1.,
                nsample=1024,
                use_xyz=False,
                normalize_xyz=True
            )

        self.encoder = ResnetPointnet(c_dim=cfg.config['data']['c_dim'],
                                      dim=self.input_feature_dim + 3 + 128,
                                      hidden_dim=cfg.config['data']['hidden_dim'])
        self.point_seg = PointSeg(num_class=2, channel=self.input_feature_dim + 3)
        self.mask_loss_func = get_loss()
        # self.self_attn = SelfAttention(input_layer=cfg.config['data']['c_dim'], hidden_layer=256)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:3+self.input_feature_dim].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def generate(self, box_xyz, box_orientations, box_feature, input_point_cloud):
        xyz, features = self._break_up_pc(input_point_cloud)

        point_instance_labels = torch.zeros_like(features) # labels are not used in generation.
        features = torch.cat([features, point_instance_labels], dim=1)

        xyz, features = self.stn(xyz, features, box_xyz, box_orientations)

        batch_size, _, N_proposals, N_points = features.size()

        features = features[:, 0].unsqueeze(1)
        input_features = torch.cat([xyz, features], dim=1)

        input_features = input_features.permute([0, 2, 3, 1]).contiguous().view(batch_size * N_proposals, N_points, -1)

        # use PointNet to predict masks
        seg_pred, trans_feat = self.point_seg(input_features.transpose(1,2).contiguous())
        seg_pred = seg_pred.contiguous().view(batch_size * N_proposals * N_points, 2)

        box_feature = box_feature.transpose(1, 2).contiguous().view(batch_size * N_proposals, -1).unsqueeze(1)
        box_feature = box_feature.repeat(1, N_points, 1)
        input_features = torch.cat([input_features, box_feature], dim=2)

        # get segmented masks
        point_seg_mask = torch.argmax(seg_pred, dim=1).view(batch_size*N_proposals, N_points)
        point_seg_mask = point_seg_mask.unsqueeze(-1).expand(batch_size * N_proposals, N_points, input_features.shape[-1])
        input_features = input_features * point_seg_mask.float()

        input_features = self.encoder(input_features)
        input_features = input_features.view(batch_size, N_proposals, -1).transpose(1, 2)

        # input_features = self.self_attn(input_features)

        return input_features

    def forward(self, box_xyz, box_orientations, box_feature, input_point_cloud, point_instance_labels, proposal_instance_labels):
        '''
        Extract point features from input pointcloud, and propagate to box xyz.
        :param box_xyz: (Batch size x N points x 3) point coordinates
        :param box_feature: (Batch size x Feature dim x Num of boxes) box features.
        :param input_point_cloud: (Batch size x Num of pointcloud points x feature dim) box features.
        :return:
        '''

        xyz, features = self._break_up_pc(input_point_cloud)

        features = torch.cat([features, point_instance_labels.unsqueeze(1)], dim=1)
        xyz, features = self.stn(xyz, features, box_xyz, box_orientations)

        batch_size, _, N_proposals, N_points = features.size()

        # get point mask
        instance_labels = features[:, 1]
        instance_point_masks = instance_labels==proposal_instance_labels.unsqueeze(-1).repeat(1,1,N_points)
        instance_point_masks = instance_point_masks.view(batch_size * N_proposals * N_points)

        features = features[:, 0].unsqueeze(1)
        input_features = torch.cat([xyz, features], dim=1)

        input_features = input_features.permute([0, 2, 3, 1]).contiguous().view(batch_size * N_proposals, N_points, -1)

        # use PointNet to predict masks
        seg_pred, trans_feat = self.point_seg(input_features.transpose(1,2).contiguous())
        seg_pred = seg_pred.contiguous().view(batch_size * N_proposals * N_points, 2)
        point_mask_loss = self.mask_loss_func(seg_pred, instance_point_masks.long(), trans_feat, weight=None)

        box_feature = box_feature.transpose(1, 2).contiguous().view(batch_size * N_proposals, -1).unsqueeze(1)
        box_feature = box_feature.repeat(1, N_points, 1)
        input_features = torch.cat([input_features, box_feature], dim=2)

        # get segmented masks
        point_seg_mask = torch.argmax(seg_pred, dim=1).view(batch_size*N_proposals, N_points)
        point_seg_mask = point_seg_mask.unsqueeze(-1).expand(batch_size * N_proposals, N_points, input_features.shape[-1])
        input_features = input_features * point_seg_mask.float()

        input_features = self.encoder(input_features)
        input_features = input_features.view(batch_size, N_proposals, -1).transpose(1, 2)

        # input_features = self.self_attn(input_features)

        return input_features, point_mask_loss
