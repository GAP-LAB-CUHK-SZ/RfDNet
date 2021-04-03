# pointnet backbone
# author: ynie
# date: March, 2020
# cite: VoteNet
import torch
import torch.nn as nn
from models.registers import MODULES
import torch.nn.functional as F
from models.iscnet.modules.CGNL import SpatialCGNL

@MODULES.register_module
class MLCV_VotingModule(nn.Module):
    def __init__(self, cfg, optim_spec = None):
        '''
        Skeleton Extraction Net to obtain partial skeleton from a partial scan (refer to PointNet++).
        :param config: configuration file.
        :param optim_spec: optimizer parameters.
        '''
        super(MLCV_VotingModule, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Modules'''
        self.vote_factor = cfg.config['data']['vote_factor']
        self.in_dim = 256
        self.out_dim = self.in_dim # due to residual feature, in_dim has to be == out_dim
        self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.in_dim, (3+self.out_dim) * self.vote_factor, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
        self.sa1 = SpatialCGNL(self.in_dim, int(self.in_dim / 2), use_scale=False, groups=4)

    def forward(self, seed_xyz, seed_features):
        """ Forward pass.

        Arguments:
            seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
            seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
        Returns:
            vote_xyz: (batch_size, num_seed*vote_factor, 3)
            vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
        """
        batch_size = seed_xyz.shape[0]
        num_seed = seed_xyz.shape[1]
        num_vote = num_seed * self.vote_factor

        feature_dim = seed_features.shape[1]
        net = seed_features.view(batch_size, feature_dim, 32, 32)
        net = self.sa1(net)
        net = net.view(batch_size, feature_dim, num_seed)
        net = F.relu(self.bn1(self.conv1(net)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.conv3(net)  # (batch_size, (3+out_dim)*vote_factor, num_seed)

        net = net.transpose(2, 1).view(batch_size, num_seed, self.vote_factor, 3 + self.out_dim)
        offset = net[:, :, :, 0:3]
        vote_xyz = seed_xyz.unsqueeze(2) + offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, num_vote, 3)

        residual_features = net[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
        vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
        vote_features = vote_features.contiguous().view(batch_size, num_vote, self.out_dim)
        vote_features = vote_features.transpose(2, 1).contiguous()

        return vote_xyz, vote_features

