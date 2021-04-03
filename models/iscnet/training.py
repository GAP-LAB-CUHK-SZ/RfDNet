# Trainer for Total3D.
# author: ynie
# date: Feb, 2020
from models.training import BaseTrainer
import torch
import numpy as np
import os
from net_utils import visualization as vis

class Trainer(BaseTrainer):
    '''
    Trainer object for total3d.
    '''

    def eval_step(self, data):
        '''
        performs a step in evaluation
        :param data (dict): data dictionary
        :return:
        '''
        loss = self.compute_loss(data)
        loss['total'] = loss['total'].item()
        return loss

    def visualize_step(self, epoch, phase, iter, data):
        ''' Performs a visualization step.
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        with torch.no_grad():
            '''network forwarding'''
            est_data = self.net({**data, 'export_shape':True})
            voxels_out, proposal_to_gt_box_w_cls_list = est_data[2:4]

        if proposal_to_gt_box_w_cls_list is None:
            return

        sample_ids = np.random.choice(voxels_out.shape[0], 3, replace=False) if voxels_out.shape[0]>=3 else range(voxels_out.shape[0])
        n_shapes_per_batch = self.cfg.config['data']['completion_limit_in_train']
        for idx, i in enumerate(sample_ids):
            voxel_path = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s_%03d_pred.png' % (epoch, phase, iter, idx))
            vis.visualize_voxels(voxels_out[i].cpu().numpy(), voxel_path)

            batch_index = i // n_shapes_per_batch
            in_batch_id = i % n_shapes_per_batch
            box_id = proposal_to_gt_box_w_cls_list[batch_index][in_batch_id][1].item()
            cls_id = proposal_to_gt_box_w_cls_list[batch_index][in_batch_id][2].item()

            voxels_gt = data['object_voxels'][batch_index][box_id].cpu().numpy()
            voxel_path = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s_%03d_gt_cls%d.png' % (epoch, phase, iter, idx, cls_id))
            vis.visualize_voxels(voxels_gt, voxel_path)

    def to_device(self, data):
        device = self.device
        for key in data:
            if key not in ['object_voxels', 'shapenet_catids', 'shapenet_ids']:
                data[key] = data[key].to(device)
        return data

    def compute_loss(self, data):
        '''
        compute the overall loss.
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        '''computer losses'''
        loss = self.net.module.loss(est_data, data)
        return loss
