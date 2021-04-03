# PFDiscriminator
# author: ynie
# date: March, 2020
import torch.nn as nn
from models.registers import MODULES
from .pcn_submodules import Autoencoder
from torch.autograd import Variable
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


@MODULES.register_module
class PCN_Autoencoder(nn.Module):

    def __init__(self, cfg, optim_spec=None):
        super(PCN_Autoencoder, self).__init__()
        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec
        self.threshold = 0.5

        '''Input channel.'''
        in_channel =  3 + int(cfg.config['data']['use_color']) * 3 + int(not cfg.config['data']['no_height']) * 1

        '''Modules'''
        num_points = cfg.config['data']['object_input_num_point']
        self.num_coarses = cfg.config['data']['object_input_num_point']
        num_fines = cfg.config['data']['object_input_num_point'] * cfg.config['data']['upsample_rate']
        grid_size = 4
        self.PCN = Autoencoder(num_points,
                               self.num_coarses,
                               num_fines,
                               grid_size,
                               in_channel=in_channel)

    def prepare_data(self, partial_scan, full_scan):
        # for gt
        full_scan = Variable(full_scan, requires_grad=True).contiguous()
        centroid_ids = furthest_point_sample(full_scan, self.num_coarses)
        full_scan_key1 = gather_operation(full_scan.transpose(1,2).contiguous(), centroid_ids).transpose(1,2).contiguous()
        full_scan_key1 = Variable(full_scan_key1, requires_grad=True)

        # for partial scan
        partial_scan = Variable(partial_scan, requires_grad=True)

        outputs = [full_scan, full_scan_key1]

        return partial_scan, outputs

    def forward(self, input_points, gt_points):
        '''
        shape completion for each instance
        :param end_points:
        :return:
        '''
        inputs, outputs = self.prepare_data(input_points, gt_points[..., :3])
        pred_coarses, pred_fine = self.PCN(inputs)
        return pred_fine, pred_coarses, outputs[0], outputs[1]

