# Dataloader of ISCNet.
# author: ynie
# date: Feb, 2020
# Cite: VoteNet

import torch.utils.data
from torch.utils.data import DataLoader
from net_utils.libs import random_sampling_by_instance, rotz, flip_axis_to_camera
import numpy as np
from models.datasets import ScanNet
import os
from net_utils.box_util import get_3d_box
from utils import pc_util
from utils.scannet.tools import get_box_corners
from net_utils.transforms import SubsamplePoints
from external import binvox_rw
import pickle

default_collate = torch.utils.data.dataloader.default_collate
MAX_NUM_OBJ = 64
MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673])

class ISCNet_ScanNet(ScanNet):
    def __init__(self, cfg, mode):
        super(ISCNet_ScanNet, self).__init__(cfg, mode)
        self.num_points = cfg.config['data']['num_point']
        self.use_color = cfg.config['data']['use_color_detection'] or cfg.config['data']['use_color_completion']
        self.use_height = not cfg.config['data']['no_height']
        self.augment = mode == 'train'
        self.shapenet_path = cfg.config['data']['shapenet_path']
        self.points_unpackbits = cfg.config['data']['points_unpackbits']
        self.n_points_object = cfg.config['data']['points_subsample']
        self.points_transform = SubsamplePoints(cfg.config['data']['points_subsample'], mode)
        self.phase = cfg.config[self.mode]['phase']

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        data_path = self.split[idx]
        with open(data_path['bbox'], 'rb') as file:
            box_info = pickle.load(file)
        boxes3D = []
        classes = []
        shapenet_catids = []
        shapenet_ids = []
        object_instance_ids = []
        for item in box_info:
            object_instance_ids.append(item['instance_id'])
            boxes3D.append(item['box3D'])
            classes.append(item['cls_id'])
            shapenet_catids.append(item['shapenet_catid'])
            shapenet_ids.append(item['shapenet_id'])
        boxes3D = np.array(boxes3D)
        scan_data = np.load(data_path['scan'])
        point_cloud = scan_data['mesh_vertices']
        point_votes = scan_data['point_votes']
        point_instance_labels = scan_data['instance_labels']

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]  # do not use color for now
        else:
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        '''Augment'''
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                boxes3D[:, 0] = -1 * boxes3D[:, 0]
                boxes3D[:, 6] = np.sign(boxes3D[:, 6]) * np.pi - boxes3D[:, 6]
                point_votes[:, [1, 4, 7]] = -1 * point_votes[:, [1, 4, 7]]
            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                boxes3D[:, 1] = -1 * boxes3D[:, 1]
                boxes3D[:, 6] = -1 * boxes3D[:, 6]
                point_votes[:, [2, 5, 8]] = -1 * point_votes[:, [2, 5, 8]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 2) - np.pi / 4  # -45 ~ +45 degree
            rot_mat = rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))
            point_votes_end[:,4:7] = np.dot(point_cloud[:,0:3] + point_votes[:,4:7], np.transpose(rot_mat))
            point_votes_end[:,7:10] = np.dot(point_cloud[:,0:3] + point_votes[:,7:10], np.transpose(rot_mat))

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            boxes3D[:, 0:3] = np.dot(boxes3D[:, 0:3], np.transpose(rot_mat))
            boxes3D[:, 6] += rot_angle
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]
            point_votes[:,4:7] = point_votes_end[:,4:7] - point_cloud[:,0:3]
            point_votes[:,7:10] = point_votes_end[:,7:10] - point_cloud[:,0:3]

            '''Normalize angles to [-pi, pi]'''
            boxes3D[:, 6] = np.mod(boxes3D[:, 6] + np.pi, 2 * np.pi) - np.pi

        class_ind = [self.dataset_config.shapenetid2class[x] for x in classes]

        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        object_instance_labels = np.zeros((MAX_NUM_OBJ, ))

        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:boxes3D.shape[0]] = class_ind
        size_residuals[0:boxes3D.shape[0], :] = boxes3D[:, 3:6] - self.dataset_config.mean_size_arr[class_ind, :]
        target_bboxes_mask[0:boxes3D.shape[0]] = 1
        target_bboxes[0:boxes3D.shape[0], :] = boxes3D[:,0:6]
        object_instance_labels[0:boxes3D.shape[0]] = object_instance_ids

        obj_angle_class, obj_angle_residuals = self.dataset_config.angle2class(boxes3D[:, 6])
        angle_classes[0:boxes3D.shape[0]] = obj_angle_class
        angle_residuals[0:boxes3D.shape[0]] = obj_angle_residuals

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]
        point_instance_labels = point_instance_labels[choices]

        '''For Object Detection'''
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:boxes3D.shape[0]] = class_ind
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)

        '''For Object Completion'''
        if self.phase == 'completion':
            object_points = np.zeros((MAX_NUM_OBJ, np.sum(self.n_points_object), 3))
            object_points_occ = np.zeros((MAX_NUM_OBJ, np.sum(self.n_points_object)))
            points_data = self.get_shapenet_points(shapenet_catids, shapenet_ids, transform=self.points_transform)
            object_points[0:boxes3D.shape[0]] = points_data['points']
            object_points_occ[0:boxes3D.shape[0]] = points_data['occ']

            ret_dict['object_points'] = object_points.astype(np.float32)
            ret_dict['object_points_occ'] = object_points_occ.astype(np.float32)
            ret_dict['object_instance_labels'] = object_instance_labels.astype(np.float32)
            ret_dict['point_instance_labels'] = point_instance_labels.astype(np.float32)

            '''Get Voxels for Visualization'''
            voxels_data = self.get_shapenet_voxels(shapenet_catids, shapenet_ids)
            object_voxels = np.zeros((MAX_NUM_OBJ, *voxels_data.shape[1:]))
            object_voxels[0:boxes3D.shape[0]] = voxels_data
            ret_dict['object_voxels'] = object_voxels.astype(np.float32)

            if self.mode in ['test']:
                points_iou_data = self.get_shapenet_points(shapenet_catids, shapenet_ids, transform=None)

                n_iou_points = points_iou_data['occ'].shape[-1]
                object_points_iou = np.zeros((MAX_NUM_OBJ, n_iou_points, 3))
                object_points_iou_occ = np.zeros((MAX_NUM_OBJ, n_iou_points))
                object_points_iou[0:boxes3D.shape[0]] = points_iou_data['points']
                object_points_iou_occ[0:boxes3D.shape[0]] = points_iou_data['occ']

                ret_dict['object_points_iou'] = object_points_iou.astype(np.float32)
                ret_dict['object_points_iou_occ'] = object_points_iou_occ.astype(np.float32)
                ret_dict['shapenet_catids'] = shapenet_catids
                ret_dict['shapenet_ids'] = shapenet_ids
        return ret_dict

    def get_shapenet_voxels(self, shapenet_catids, shapenet_ids):
        '''Load object voxels.'''
        shape_data_list = []
        for shapenet_catid, shapenet_id in zip(shapenet_catids, shapenet_ids):
            voxel_file = os.path.join(self.shapenet_path, 'voxel/16', shapenet_catid, shapenet_id + '.binvox')
            with open(voxel_file, 'rb') as f:
                voxels = binvox_rw.read_as_3d_array(f)
            voxels = voxels.data.astype(np.float32)
            shape_data_list.append(voxels[np.newaxis])
        return np.concatenate(shape_data_list, axis=0)

    def get_shapenet_points(self, shapenet_catids, shapenet_ids, transform=None):
        '''Load points and corresponding occ values.'''
        shape_data_list = []
        for shapenet_catid, shapenet_id in zip(shapenet_catids, shapenet_ids):
            points_dict = np.load(os.path.join(self.shapenet_path, 'point', shapenet_catid, shapenet_id + '.npz'))
            points = points_dict['points']
            # Break symmetry if given in float16:
            if points.dtype == np.float16 and self.mode == 'train':
                points = points.astype(np.float32)
                points += 1e-4 * np.random.randn(*points.shape)
            else:
                points = points.astype(np.float32)
            occupancies = points_dict['occupancies']
            if self.points_unpackbits:
                occupancies = np.unpackbits(occupancies)[:points.shape[0]]
            occupancies = occupancies.astype(np.float32)
            data = {'points':points, 'occ': occupancies}
            if transform is not None:
                data = transform(data)
            shape_data_list.append(data)

        return recursive_cat_to_numpy(shape_data_list)

def recursive_cat_to_numpy(data_list):
    '''Covert a list of dict to dict of numpy arrays.'''
    out_dict = {}
    for key, value in data_list[0].items():
        if isinstance(value, np.ndarray):
            out_dict = {**out_dict, key: np.concatenate([data[key][np.newaxis] for data in data_list], axis=0)}
        elif isinstance(value, dict):
            out_dict =  {**out_dict, **recursive_cat_to_numpy(value)}
        elif np.isscalar(value):
            out_dict = {**out_dict, key: np.concatenate([np.array([data[key]])[np.newaxis] for data in data_list], axis=0)}
        elif isinstance(value, list):
            out_dict = {**out_dict, key: np.concatenate([np.array(data[key])[np.newaxis] for data in data_list], axis=0)}
    return out_dict

def collate_fn(batch):
    '''
    data collater
    :param batch:
    :return:
    '''
    collated_batch = {}
    for key in batch[0]:
        if key not in ['shapenet_catids', 'shapenet_ids']:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        else:
            collated_batch[key] = [elem[key] for elem in batch]

    return collated_batch

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def ISCNet_dataloader(cfg, mode='train'):
    if cfg.config['data']['dataset'] == 'scannet':
        dataset = ISCNet_ScanNet(cfg, mode)
    else:
        raise NotImplementedError

    dataloader = DataLoader(dataset=dataset,
                            num_workers=cfg.config['device']['num_workers'],
                            batch_size=cfg.config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn,
                            worker_init_fn=my_worker_init_fn)
    return dataloader
