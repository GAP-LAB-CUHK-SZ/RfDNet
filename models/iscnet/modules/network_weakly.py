# ISCNet: model loader
# author: ynie
# date: Feb, 2020

from models.registers import METHODS, MODULES, LOSSES
from models.network import BaseNetwork
import torch
from net_utils.nn_distance import nn_distance
import numpy as np
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls
from external.common import compute_iou
from net_utils.libs import flip_axis_to_depth, extract_pc_in_box3d, flip_axis_to_camera
from torch import optim
from models.loss import chamfer_func
from net_utils.box_util import get_3d_box

from .network import ISCNet

@METHODS.register_module
class ISCNet_WEAK(BaseNetwork):
    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        phase_names = []
        if cfg.config[cfg.config['mode']]['phase'] in ['detection']:
            phase_names += ['backbone', 'voting', 'detection']
        if cfg.config[cfg.config['mode']]['phase'] in ['completion']:
            phase_names += ['backbone', 'voting', 'detection', 'completion']
            if cfg.config['data']['skip_propagate']:
                phase_names += ['skip_propagation']
        if cfg.config[cfg.config['mode']]['phase'] in ['prior']:
            phase_names += ['completion', 'class_encode']
                
        if (not cfg.config['model']) or (not phase_names):
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')
        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1)))

        '''freeze submodules or not'''
        self.freeze_modules(cfg)
    def sample_for_prior(inputs, sample_size):
        # temporary naive implementation should be either random
        # or sampled by removing some chungs of the input pC so it has holes
        # instead of being evenly distributed
        return inputs[:, 0:sample_size]
    def forward(self, data, export_shape=False):
        '''
        Forward pass of the network
        :param data (dict): contains the data for training.
        :param export_shape: if output shape voxels for visualization
        :return: end_points: dict
        '''
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'prior':
            pc = data['point_clouds']
            input_points = self.sample_for_prior(pc, 256) # not sure where the no. of points per object is set
            logits, features_for_completion = self.class_encode(input_points)
            completion_loss, shape_example = self.completion.compute_loss(features_for_completion,
                                                                        data['xyz_query'], # just a 3D grid
                                                                        data['object_labels'], # class labels for ShapeNet
                                                                        data['xyz_query'], # Labels in out
                                                                        export_shape) 
            return logits, features_for_completion, completion_loss, shape_example # end_points,  BATCH_PROPOSAL_IDs removed
        else:
            pass
            # not yet decided
            # return super(ISCNet_WEAK, self).forward(data, export_shape)


    def loss(self, est_data, gt_data):
        '''
        calculate loss of est_out given gt_out.
        '''
        completion_loss = est_data[2]
        total_loss = {}
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'prior':
            completion_loss = self.completion_loss(completion_loss)
            class_loss = self.class_encode(est_data, gt_data)
            total_loss = {'completion_loss': completion_loss,
                          'class_loss':class_loss}
            total_loss['total'] = class_loss + completion_loss

        return total_loss