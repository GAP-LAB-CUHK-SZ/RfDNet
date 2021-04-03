# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from configs.path_config import SHAPENETCLASSES
from configs.path_config import ScanNet_OBJ_CLASS_IDS as OBJ_CLASS_IDS
import torch

class ScannetConfig(object):
    def __init__(self):
        self.num_class = len(OBJ_CLASS_IDS)
        self.num_heading_bin = 12
        self.num_size_cluster = len(OBJ_CLASS_IDS)

        self.type2class = {SHAPENETCLASSES[cls]:index for index, cls in enumerate(OBJ_CLASS_IDS)}
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.class_ids = OBJ_CLASS_IDS
        self.shapenetid2class = {class_id: i for i, class_id in enumerate(list(self.class_ids))}
        self.mean_size_arr = np.load('datasets/scannet/scannet_means.npz')['arr_0']
        self.type_mean_size = {}
        self.data_path = 'datasets/scannet/processed_data'
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i, :]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        num_class = self.num_heading_bin
        angle = angle % (2 * np.pi)
        assert False not in (angle >= 0) * (angle <= 2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = np.int16(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class '''
        num_class = self.num_heading_bin
        angle_per_class = 2*np.pi/float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle>np.pi:
            angle = angle - 2*np.pi
        return angle

    def class2angle_cuda(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.'''
        num_class = self.num_heading_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls.float() * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            angle = angle - 2*np.pi*(angle>np.pi).float()
        return angle

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def class2size_cuda(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        mean_size_arr = torch.from_numpy(self.mean_size_arr).to(residual.device).float()
        return mean_size_arr[pred_cls.view(-1), :].view(*pred_cls.size(), 3) + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle
        return obb


def rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:, 0] = crnr[0] * dx
        crnrs[:, 1] = crnr[1] * dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:, i] = crnrs[:, 0]
        new_y[:, i] = crnrs[:, 1]

    new_dx = 2.0 * np.max(new_x, 1)
    new_dy = 2.0 * np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)

if __name__ == '__main__':
    cfg = ScannetConfig()