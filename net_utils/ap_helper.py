# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import numpy as np
from net_utils.eval_det import eval_det_multiprocessing_wo_mesh, eval_det_multiprocessing_w_mesh, get_iou_obb, \
    compute_mesh_iou
import torch
from net_utils.box_util import get_3d_box
from net_utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from net_utils.libs import softmax, flip_axis_to_camera, flip_axis_to_depth, extract_pc_in_box3d
import trimesh
from trimesh.exchange.binvox import voxelize_mesh
import os
from multiprocessing import Pool
from functools import partial
from utils.shapenet import ShapeNetv2_Watertight_Scaled_Simplified_path

transform_shapenet = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None, evaluate_mesh=False):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.evaluate_mesh = evaluate_mesh
        self.reset()

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.

        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self):
        if self.evaluate_mesh:
            return self.compute_metrics_w_mesh()
        else:
            return self.compute_metrics_wo_mesh()

    def compute_metrics_wo_mesh(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing_wo_mesh(self.pred_map_cls, self.gt_map_cls,
                                                         ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def compute_metrics_w_mesh(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh) = eval_det_multiprocessing_w_mesh(self.pred_map_cls,
                                                                                          self.gt_map_cls,
                                                                                          ovthresh=self.ap_iou_thresh,
                                                                                          get_iou_func=get_iou_obb,
                                                                                          get_iou_mesh=compute_mesh_iou)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)

        # for mesh
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision_mesh' % (clsname)] = ap_mesh[key]
        ret_dict['mAP_mesh'] = np.mean(list(ap_mesh.values()))
        rec_list_mesh = []
        for key in sorted(ap_mesh.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall_mesh' % (clsname)] = rec_mesh[key][-1]
                rec_list_mesh.append(rec_mesh[key][-1])
            except:
                ret_dict['%s Recall_mesh' % (clsname)] = 0
                rec_list_mesh.append(0)
        ret_dict['AR_mesh'] = np.mean(rec_list_mesh)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0


def parse_predictions(est_data, gt_data, config_dict):
    """ Parse predictions to OBB parameters and suppress overlapping boxes

    Args:
        est_data, gt_data: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    eval_dict = {}

    pred_center = est_data['center']  # B,num_proposal,3
    pred_heading_class = torch.argmax(est_data['heading_scores'], -1)  # B,num_proposal
    heading_residuals = est_data['heading_residuals_normalized'] * (
                np.pi / config_dict['dataset_config'].num_heading_bin)  # Bxnum_proposalxnum_heading_bin
    pred_heading_residual = torch.gather(heading_residuals, 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(est_data['size_scores'], -1)  # B,num_proposal
    size_residuals = est_data['size_residuals_normalized'] * torch.from_numpy(
        config_dict['dataset_config'].mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
    pred_size_residual = torch.gather(size_residuals, 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(est_data['sem_cls_scores'], -1)  # B,num_proposal
    sem_cls_probs = softmax(est_data['sem_cls_scores'].detach().cpu().numpy())  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal

    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())
    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict['dataset_config'].class2angle( \
                pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size( \
                int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, -heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = gt_data['point_clouds'].cpu().numpy()[:, :, 0:3]  # B,N,3
        for i in range(bsize):
            pc = batch_pc[i, :, :]  # (N,3)
            for j in range(K):
                box3d = pred_corners_3d_upright_camera[i, j, :, :]  # (8,3)
                box3d = flip_axis_to_depth(box3d)
                pc_in_box, inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i, j] = 0
        # -------------------------------------

    obj_logits = est_data['objectness_scores'].detach().cpu().numpy()
    obj_prob = softmax(obj_logits)[:, :, 1]  # (B,K)
    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K), dtype=np.uint8)
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        eval_dict['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K), dtype=np.uint8)
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        eval_dict['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K), dtype=np.uint8)
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        eval_dict['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    return eval_dict, {'pred_corners_3d_upright_camera': pred_corners_3d_upright_camera,
                       'sem_cls_probs': sem_cls_probs,
                       'obj_prob': obj_prob,
                       'pred_sem_cls': pred_sem_cls}


def assembly_pred_map_cls(eval_dict, parsed_predictions, config_dict, mesh_outputs=None, voxel_size=0.047):
    pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera']
    sem_cls_probs = parsed_predictions['sem_cls_probs']
    obj_prob = parsed_predictions['obj_prob']
    pred_mask = eval_dict['pred_mask']
    pred_sem_cls = parsed_predictions['pred_sem_cls']
    bsize, N_proposals = pred_sem_cls.shape
    if mesh_outputs is not None:
        assert bsize == 1
        meshes = mesh_outputs['meshes']
        proposal_ids = mesh_outputs['proposal_ids'].cpu().numpy()

    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        if config_dict['per_class_proposal']:
            if mesh_outputs is None:
                cur_list = []
                for ii in range(config_dict['dataset_config'].num_class):
                    cur_list += [(ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j]) \
                                 for j in range(N_proposals) if
                                 pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
            else:
                sample_idx = [(ii, j) for ii in range(config_dict['dataset_config'].num_class) for j in
                              range(N_proposals) if
                              pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]

                p = Pool(processes=16)
                cur_list = p.map(partial(batch_load_pred_data, proposal_ids=proposal_ids,
                                         batch_id=i, pred_corners=pred_corners_3d_upright_camera,
                                         sem_cls_probs=sem_cls_probs, obj_prob=obj_prob, meshes=meshes, voxel_size=voxel_size), sample_idx)
                p.close()
                p.join()

            batch_pred_map_cls.append(cur_list)
        else:
            if mesh_outputs is None:
                batch_pred_map_cls.append([(pred_sem_cls[i, j].item(),
                                            pred_corners_3d_upright_camera[i, j],
                                            obj_prob[i, j]) \
                                           for j in range(N_proposals) if
                                           pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
            else:
                sample_idx = [j for j in range(N_proposals) if
                              pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
                p = Pool(processes=16)
                temp_list = p.map(partial(batch_load_pred_data_wo_cls, proposal_ids=proposal_ids, meshes=meshes,
                                          pred_corners=pred_corners_3d_upright_camera, batch_id=i,
                                          pred_sem_cls=pred_sem_cls.cpu().numpy(),
                                          obj_prob=obj_prob, voxel_size=voxel_size), sample_idx)
                p.close()
                p.join()

                batch_pred_map_cls.append(temp_list)

    eval_dict['batch_pred_map_cls'] = batch_pred_map_cls

    return eval_dict


def parse_groundtruths(gt_data, config_dict):
    """ Parse groundtruth labels to OBB parameters.

    Args:
        gt_data: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    center_label = gt_data['center_label']
    heading_class_label = gt_data['heading_class_label']
    heading_residual_label = gt_data['heading_residual_label']
    size_class_label = gt_data['size_class_label']
    size_residual_label = gt_data['size_residual_label']
    box_label_mask = gt_data['box_label_mask']
    sem_cls_label = gt_data['sem_cls_label']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i, j] == 0: continue
            heading_angle = config_dict['dataset_config'].class2angle(heading_class_label[i, j].detach().cpu().numpy(),
                                                                      heading_residual_label[
                                                                          i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size(int(size_class_label[i, j].detach().cpu().numpy()),
                                                                size_residual_label[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, -heading_angle, gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    return {'sem_cls_label': sem_cls_label,
            'gt_corners_3d_upright_camera': gt_corners_3d_upright_camera,
            'box_label_mask': box_label_mask}


def assembly_gt_map_cls(parsed_gts, mesh_outputs=None, voxel_size=0.047):
    sem_cls_label = parsed_gts['sem_cls_label']
    gt_corners_3d_upright_camera = parsed_gts['gt_corners_3d_upright_camera']
    box_label_mask = parsed_gts['box_label_mask']
    bsize = sem_cls_label.shape[0]
    MAX_OBJs = gt_corners_3d_upright_camera.shape[1]

    if mesh_outputs is not None:
        assert bsize == 1
        shapenet_catids = mesh_outputs['shapenet_catids'][0]
        shapenet_ids = mesh_outputs['shapenet_ids'][0]
        meshes = [
            trimesh.load(os.path.join(ShapeNetv2_Watertight_Scaled_Simplified_path, shapenet_catid, shapenet_id + '.off'),
                         process=False) for shapenet_catid, shapenet_id in zip(shapenet_catids, shapenet_ids)]

    batch_gt_map_cls = []
    for i in range(bsize):
        if mesh_outputs is None:
            batch_gt_map_cls.append([(sem_cls_label[i, j].item(), gt_corners_3d_upright_camera[i, j]) for j in
                                     range(MAX_OBJs) if box_label_mask[i, j] == 1])
        else:
            sample_idx = [j for j in range(MAX_OBJs) if box_label_mask[i, j] == 1]

            p = Pool(processes=16)
            temp_list = p.map(partial(batch_load_gt_data, meshes=meshes, gt_corners=gt_corners_3d_upright_camera,
                                      sem_cls_label=sem_cls_label.cpu().numpy(), batch_id=i, voxel_size=voxel_size), sample_idx)
            p.close()
            p.join()
            batch_gt_map_cls.append(temp_list)

    return batch_gt_map_cls


def fit_shapenet_obj_to_votenet_box(points, box_corners):
    '''
    Fit points from shapenet objects to box corners produced from votenet.
    '''
    # recover box corners to 7-d coordinates
    corners_3d_depth = flip_axis_to_depth(box_corners)
    center = (np.max(corners_3d_depth, axis=0) + np.min(corners_3d_depth, axis=0)) / 2.
    forward_vector = corners_3d_depth[1] - corners_3d_depth[2]
    left_vector = corners_3d_depth[0] - corners_3d_depth[1]
    up_vector = corners_3d_depth[6] - corners_3d_depth[2]
    orientation = np.arctan2(forward_vector[1], forward_vector[0])
    sizes = np.linalg.norm([forward_vector, left_vector, up_vector], axis=1)

    # transform obj points to boxes
    obj_points = points - (points.max(0) + points.min(0)) / 2.
    obj_points = obj_points.dot(transform_shapenet.T)
    obj_points = obj_points.dot(np.diag(1 / (obj_points.max(0) - obj_points.min(0)))).dot(np.diag(sizes))

    axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0],
                               [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
    obj_points = obj_points.dot(axis_rectified) + center

    return obj_points


def batch_load_pred_data(idx, proposal_ids, batch_id, pred_corners, sem_cls_probs, obj_prob, meshes, voxel_size):
    ii, j = idx

    mesh_data = meshes[list(proposal_ids[batch_id, :, 0]).index(j)]
    obj_points = mesh_data.vertices

    obj_points = fit_shapenet_obj_to_votenet_box(obj_points, pred_corners[batch_id, j])
    mesh_data.vertices = obj_points

    dimension = int(max((obj_points.max(0) - obj_points.min(0))) / voxel_size)
    dimension = max(dimension, 2)
    # internal voxels
    voxel_data_internal = voxelize_mesh(mesh_data, dimension=dimension, wireframe=True, dilated_carving=True)
    # surface voxels
    voxel_data_surface = voxelize_mesh(mesh_data, exact=True, dimension=dimension)

    return (ii, pred_corners[batch_id, j], sem_cls_probs[batch_id, j, ii] * obj_prob[batch_id, j],
            (voxel_data_internal, voxel_data_surface))


def batch_load_pred_data_wo_cls(j, proposal_ids, meshes, pred_corners, batch_id, pred_sem_cls, obj_prob, voxel_size):
    mesh_data = meshes[list(proposal_ids[batch_id, :, 0]).index(j)]
    obj_points = mesh_data.vertices
    obj_points = fit_shapenet_obj_to_votenet_box(obj_points, pred_corners[batch_id, j])
    mesh_data.vertices = obj_points

    dimension = int(max((obj_points.max(0) - obj_points.min(0))) / voxel_size)
    dimension = max(dimension, 2)
    # internal voxels
    voxel_data_internal = voxelize_mesh(mesh_data, dimension=dimension, wireframe=True, dilated_carving=True)
    # surface voxels
    voxel_data_surface = voxelize_mesh(mesh_data, exact=True, dimension=dimension)

    return (pred_sem_cls[batch_id, j], pred_corners[batch_id, j], obj_prob[batch_id, j],
            (voxel_data_internal, voxel_data_surface))


def batch_load_gt_data(j, meshes, gt_corners, sem_cls_label, batch_id, voxel_size):
    mesh_data = meshes[j]
    obj_points = mesh_data.vertices
    obj_points = fit_shapenet_obj_to_votenet_box(obj_points, gt_corners[batch_id, j])
    mesh_data.vertices = obj_points

    dimension = int(max((obj_points.max(0) - obj_points.min(0))) / voxel_size)
    dimension = max(dimension, 2)
    # internal voxels
    voxel_data_internal = voxelize_mesh(mesh_data, dimension=dimension, wireframe=True, dilated_carving=True)
    # surface voxels
    voxel_data_surface = voxelize_mesh(mesh_data, exact=True, dimension=dimension)
    return (sem_cls_label[batch_id, j], gt_corners[batch_id, j], (voxel_data_internal, voxel_data_surface))