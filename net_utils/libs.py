# Lib functions in data processing and calculation.
# author: ynie
# date: Feb, 2020

import torch
import torch.nn as nn
import numpy as np
from external.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        if hasattr(m, 'weight') and hasattr(m.weight, 'data'):
            torch.nn.init.xavier_normal_(m.weight.data)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'):
            torch.nn.init.constant_(m.bias.data, 0.0)

# ----------------------------------------
# Point Cloud Sampling
# ----------------------------------------
def random_sampling_by_instance(pc, instance_labels, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0] < num_sample)

    sample_rate = num_sample/len(instance_labels)
    unique_instance_labels = np.unique(instance_labels)

    instance_point_choices = []
    for instance_id in unique_instance_labels:
        point_ids = np.where(instance_labels==instance_id)[0]
        ids = np.random.choice(len(point_ids), int(len(point_ids) * sample_rate) + 1, replace=replace)
        point_choices = point_ids[ids]
        instance_point_choices.append(point_choices)

    total_choices = np.hstack(instance_point_choices)
    keep_ids = np.random.choice(len(total_choices), num_sample, replace=False)
    total_choices = total_choices[keep_ids]

    if return_choices:
        return pc[total_choices], total_choices
    else:
        return pc[total_choices]

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


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

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

def flip_axis_to_camera_cuda(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = pc.clone()
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]]
    pc2[..., 1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[...,2] *= -1
    return pc2

def flip_axis_to_depth_cuda(pc):
    pc2 = pc.clone()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2

def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def extract_pc_in_box3d_cuda(pointclouds, boxes3d):
    ''' pc: (N,3), box3d: (8,3) '''
    boxes3d_flatten = torch.cat([torch.min(boxes3d, dim=2)[0], torch.max(boxes3d, dim=2)[0]], dim=-1)
    b_size = boxes3d.size(0)
    masks = []
    for b_id in range(b_size):
        box3d = boxes3d_flatten[b_id]
        pc = pointclouds[b_id]
        pc = pc.unsqueeze(1).expand(pc.size(0), box3d.size(0), pc.size(1))

        c1 = pc[:, :, 0] <= box3d[:, 3]
        c2 = pc[:, :, 0] >= box3d[:, 0]
        c3 = pc[:, :, 1] <= box3d[:, 4]
        c4 = pc[:, :, 1] >= box3d[:, 1]
        c5 = pc[:, :, 2] <= box3d[:, 5]
        c6 = pc[:, :, 2] >= box3d[:, 2]

        mask = c1 + c2 + c3 + c4 + c5 + c6
        masks.append((mask == 6).unsqueeze(0))

    return torch.cat(masks, dim=0).transpose(1,2)

def sample_points(box_point_masks, num_points):
    box_point_masks_float = box_point_masks.float()
    box_point_masks_float = box_point_masks_float.view(-1, box_point_masks.size(-1))
    # there are no points in some boxes thus we set equal probability for each point in a scene.
    empty_box_masks = box_point_masks_float.sum(dim=-1) == 0
    box_point_masks_float[empty_box_masks] = 1.

    sampled_point_ids = torch.multinomial(box_point_masks_float, num_samples=num_points, replacement=True)

    return sampled_point_ids.view(*box_point_masks.size()[:2], num_points), empty_box_masks.view(box_point_masks.size()[:2])

def extract_points(data, pred_center, candidate_pred_box_masks, box_point_masks, empty_box_masks, num_points, upsample_rate):
    b_size, MAX_NUM_OBJ, CONDIDATE_NUM_OBJ = candidate_pred_box_masks.size()
    gt_point_num = num_points * upsample_rate

    input_object_points = []
    gt_object_points = []
    object_cls_labels = []

    for batch_id in range(b_size):
        for obj_id in range(MAX_NUM_OBJ):
            if data['box_label_mask'][batch_id, obj_id] == 0: continue
            pred_box_mask = candidate_pred_box_masks[batch_id, obj_id] * (1 - empty_box_masks[batch_id])
            if pred_box_mask.sum()==0: continue
            points_in_pred_boxes = data['point_clouds'][batch_id, box_point_masks[batch_id, pred_box_mask, :]]
            sample_inds = furthest_point_sample(points_in_pred_boxes[..., :3].contiguous(), num_points)

            # obtain input points for each object_id
            points_in_pred_boxes = gather_operation(points_in_pred_boxes.transpose(1, 2).contiguous(), sample_inds).transpose(1, 2).contiguous()

            # obtain ground_truth points for each object_id
            gt_instance_point_ids = (data['gt_instance_labels'][batch_id] == data['box_instance_labels'][batch_id][obj_id]).astype(np.uint8)
            gt_points = data['gt_point_clouds'][batch_id, gt_instance_point_ids]

            if gt_points.size(0) < gt_point_num:
                extra_inds = torch.randint(0, gt_points.size(0), (gt_point_num - gt_points.size(0),))
                gt_points = torch.cat([gt_points, gt_points[extra_inds]], dim=0).unsqueeze(0)
            else:
                gt_sample_inds = furthest_point_sample(gt_points[..., :3].unsqueeze(0).contiguous(), gt_point_num)
                gt_points = gather_operation(gt_points.unsqueeze(0).transpose(1, 2).contiguous(), gt_sample_inds).transpose(1, 2).contiguous()

            gt_points = gt_points.expand(points_in_pred_boxes.size(0), gt_point_num, gt_points.size(2))

            # move to pred box center
            points_in_pred_boxes_coordinates = points_in_pred_boxes[..., :3] - pred_center[batch_id, pred_box_mask].unsqueeze(1).detach()
            points_in_pred_boxes = torch.cat([points_in_pred_boxes_coordinates, points_in_pred_boxes[..., 3].unsqueeze(-1)], dim=-1)
            input_object_points.append(points_in_pred_boxes)

            # move to pred box center
            gt_points_coordinates = gt_points[..., :3] - pred_center[batch_id, pred_box_mask].unsqueeze(1).detach()
            gt_points = torch.cat([gt_points_coordinates, gt_points[..., 3].unsqueeze(-1)], dim=-1)
            gt_object_points.append(gt_points)

            # obtain class label for each gt object
            object_cls_labels.append(data['sem_cls_label'][batch_id][obj_id].unsqueeze(0).expand(points_in_pred_boxes.size(0)))

    if len(input_object_points):
        return torch.cat(input_object_points, dim=0), torch.cat(gt_object_points, dim=0), torch.cat(object_cls_labels, dim=0)
    else:
        return input_object_points, gt_object_points, object_cls_labels