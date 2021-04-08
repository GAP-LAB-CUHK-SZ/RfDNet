# Tester for Total3D
# author: ynie
# date: April, 2020
from models.testing import BaseTester
from .training import Trainer
from net_utils.ap_helper import parse_predictions, parse_groundtruths, assembly_pred_map_cls, assembly_gt_map_cls
import os
import torch
import numpy as np
from net_utils.libs import softmax
from utils import pc_util
from models.loss import compute_objectness_loss
from utils.read_and_write import read_json
from net_utils.libs import flip_axis_to_depth, flip_axis_to_camera

class Tester(BaseTester, Trainer):
    '''
    Tester object for ISCNet.
    '''

    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)

    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''
        if est_data[4] is not None:
            eval_dict = est_data[4]
        else:
            eval_dict, parsed_predictions = parse_predictions(est_data[0], gt_data, self.cfg.eval_config)
            eval_dict = assembly_pred_map_cls(eval_dict, parsed_predictions, self.cfg.eval_config)

        parsed_gts = parse_groundtruths(gt_data, self.cfg.eval_config)
        batch_gt_map_cls = assembly_gt_map_cls(parsed_gts)
        eval_dict['batch_gt_map_cls'] = batch_gt_map_cls
        return eval_dict

    def evaluate_step(self, est_data, data):
        eval_metrics = {}

        cls_iou_stat = est_data[6]
        if cls_iou_stat is not None:
            cls_iou_stat_out = {}
            for cls, iou in zip(cls_iou_stat['cls'], cls_iou_stat['iou']):
                if str(cls) + '_voxel_iou' not in cls_iou_stat_out:
                    cls_iou_stat_out[str(cls) + '_voxel_iou'] = []
                cls_iou_stat_out[str(cls) + '_voxel_iou'].append(iou)

            eval_metrics = {**eval_metrics, **cls_iou_stat_out}

        return eval_metrics

    def test_step(self, data):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net.module.generate(data)

        '''computer losses'''
        loss = self.net.module.loss(est_data, data)
        eval_metrics = self.evaluate_step(est_data, data)

        loss['total'] = loss['total'].item()
        loss = {**loss, **eval_metrics}
        return loss, est_data

    def visualize_step(self, phase, iter, gt_data, our_data, eval_dict, inference_switch=False):
        ''' Performs a visualization step.
        '''
        split_file = os.path.join(self.cfg.config['data']['split'], 'scannetv2_' + phase + '.json')
        scene_name = read_json(split_file)[gt_data['scan_idx']]['scan'].split('/')[3]

        dump_dir = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s_%s'%(phase, iter, scene_name))
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)

        DUMP_CONF_THRESH = self.cfg.config['generation']['dump_threshold'] # Dump boxes with obj prob larger than that.
        batch_id = 0

        '''Predict meshes'''
        pred_sem_cls = our_data[7]['pred_sem_cls'][batch_id].cpu().numpy()
        if our_data[5] is not None:
            meshes = our_data[5]
            BATCH_PROPOSAL_IDs = our_data[3][0].cpu().numpy()
            for mesh_data, map_data in zip(meshes, BATCH_PROPOSAL_IDs):
                str_nums = (map_data[0], map_data[1], pred_sem_cls[map_data[0]])
                object_mesh = os.path.join(dump_dir, 'proposal_%d_target_%d_class_%d_mesh.ply' % str_nums)
                mesh_data.export(object_mesh)
        else:
            BATCH_PROPOSAL_IDs = np.empty(0)

        '''Predict boxes'''
        est_data = our_data[0]
        pred_corners_3d_upright_camera = our_data[7]['pred_corners_3d_upright_camera']
        objectness_prob = our_data[7]['obj_prob'][batch_id]

        # INPUT
        point_clouds = gt_data['point_clouds'].cpu().numpy()

        # NETWORK OUTPUTS
        seed_xyz = est_data['seed_xyz'].detach().cpu().numpy()  # (B,num_seed,3)
        if 'vote_xyz' in est_data:
            # aggregated_vote_xyz = est_data['aggregated_vote_xyz'].detach().cpu().numpy()
            # vote_xyz = est_data['vote_xyz'].detach().cpu().numpy()  # (B,num_seed,3)
            aggregated_vote_xyz = est_data['aggregated_vote_xyz'].detach().cpu().numpy()

        box_corners_cam = pred_corners_3d_upright_camera[batch_id]
        box_corners_depth = flip_axis_to_depth(box_corners_cam)
        centroid = (np.max(box_corners_depth, axis=1) + np.min(box_corners_depth, axis=1)) / 2.

        forward_vector = box_corners_depth[:,1] - box_corners_depth[:,2]
        left_vector = box_corners_depth[:,0] - box_corners_depth[:,1]
        up_vector = box_corners_depth[:,6] - box_corners_depth[:,2]
        orientation = np.arctan2(forward_vector[:,1], forward_vector[:,0])
        forward_size = np.linalg.norm(forward_vector, axis=1)
        left_size = np.linalg.norm(left_vector, axis=1)
        up_size = np.linalg.norm(up_vector, axis=1)
        sizes = np.vstack([forward_size, left_size, up_size]).T

        box_params = np.hstack([centroid, sizes, orientation[:,np.newaxis]])

        # OTHERS
        pred_mask = eval_dict['pred_mask']  # B,num_proposal

        pc = point_clouds[batch_id, :, :]

        # Dump various point clouds
        pc_util.write_ply(pc, os.path.join(dump_dir, '%06d_pc.ply' % (batch_id)))
        pc_util.write_ply(seed_xyz[batch_id, :, :], os.path.join(dump_dir, '%06d_seed_pc.ply' % (batch_id)))
        if 'vote_xyz' in est_data:
            pc_util.write_ply(est_data['vote_xyz'][batch_id, :, :],
                              os.path.join(dump_dir, '%06d_vgen_pc.ply' % (batch_id)))
            pc_util.write_ply(aggregated_vote_xyz[batch_id, :, :],
                              os.path.join(dump_dir, '%06d_aggregated_vote_pc.ply' % (batch_id)))
        pc_util.write_ply(box_params[:, 0:3], os.path.join(dump_dir, '%06d_proposal_pc.ply' % (batch_id)))
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            pc_util.write_ply(box_params[objectness_prob > DUMP_CONF_THRESH, 0:3],
                              os.path.join(dump_dir, '%06d_confident_proposal_pc.ply' % (batch_id)))

        # Dump predicted bounding boxes
        if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
            num_proposal = box_params.shape[0]
            if len(box_params) > 0:
                pc_util.write_oriented_bbox(box_params[objectness_prob > DUMP_CONF_THRESH, :],
                                            os.path.join(dump_dir, '%06d_pred_confident_bbox.ply' % (batch_id)))
                pc_util.write_oriented_bbox(
                    box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
                    os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.ply' % (batch_id)))
                pc_util.write_oriented_bbox(box_params[pred_mask[batch_id, :] == 1, :],
                                            os.path.join(dump_dir, '%06d_pred_nms_bbox.ply' % (batch_id)))
                pc_util.write_oriented_bbox(box_params, os.path.join(dump_dir, '%06d_pred_bbox.ply' % (batch_id)))

                save_path = os.path.join(dump_dir, '%06d_pred_confident_nms_bbox.npz' % (batch_id))
                np.savez(save_path, obbs=box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
                         proposal_map = BATCH_PROPOSAL_IDs)

        # Return if it is at inference time. No dumping of groundtruths
        if inference_switch:
            return

        objectness_loss, objectness_label, objectness_mask, object_assignment = \
            compute_objectness_loss(est_data, gt_data)

        # LABELS
        gt_center = gt_data['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
        gt_mask = gt_data['box_label_mask'].cpu().numpy()  # B,K2
        gt_heading_class = gt_data['heading_class_label'].cpu().numpy()  # B,K2
        gt_heading_residual = gt_data['heading_residual_label'].cpu().numpy()  # B,K2
        gt_size_class = gt_data['size_class_label'].cpu().numpy()  # B,K2
        gt_size_residual = gt_data['size_residual_label'].cpu().numpy()  # B,K2,3
        objectness_label = objectness_label.detach().cpu().numpy()  # (B,K,)
        objectness_mask = objectness_mask.detach().cpu().numpy()  # (B,K,)

        if np.sum(objectness_label[batch_id, :]) > 0:
            pc_util.write_ply(box_params[objectness_label[batch_id, :] > 0, 0:3],
                              os.path.join(dump_dir, '%06d_gt_positive_proposal_pc.ply' % (batch_id)))
        if np.sum(objectness_mask[batch_id, :]) > 0:
            pc_util.write_ply(box_params[objectness_mask[batch_id, :] > 0, 0:3],
                              os.path.join(dump_dir, '%06d_gt_mask_proposal_pc.ply' % (batch_id)))
        pc_util.write_ply(gt_center[batch_id, :, 0:3], os.path.join(dump_dir, '%06d_gt_centroid_pc.ply' % (batch_id)))
        pc_util.write_ply_color(box_params[:, 0:3], objectness_label[batch_id, :],
                                os.path.join(dump_dir, '%06d_proposal_pc_objectness_label.ply' % (batch_id)))

        # Dump GT bounding boxes
        obbs = []
        for j in range(gt_center.shape[1]):
            if gt_mask[batch_id, j] == 0: continue
            obb = self.cfg.dataset_config.param2obb(gt_center[batch_id, j, 0:3], gt_heading_class[batch_id, j], gt_heading_residual[batch_id, j],
                                   gt_size_class[batch_id, j], gt_size_residual[batch_id, j])
            obbs.append(obb)
        if len(obbs) > 0:
            obbs = np.vstack(tuple(obbs))  # (num_gt_objects, 7)
            pc_util.write_oriented_bbox(obbs, os.path.join(dump_dir, '%06d_gt_bbox.ply' % (batch_id)))

        # OPTIONALL, also dump prediction and gt details
        if 'batch_pred_map_cls' in eval_dict:
            fout = open(os.path.join(dump_dir, '%06d_pred_map_cls.txt' % (batch_id)), 'w')
            for t in eval_dict['batch_pred_map_cls'][batch_id]:
                fout.write(str(t[0]) + ' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write(' ' + str(t[2]))
                fout.write('\n')
            fout.close()
        if 'batch_gt_map_cls' in eval_dict:
            fout = open(os.path.join(dump_dir, '%06d_gt_map_cls.txt' % (batch_id)), 'w')
            for t in eval_dict['batch_gt_map_cls'][batch_id]:
                fout.write(str(t[0]) + ' ')
                fout.write(",".join([str(x) for x in list(t[1].flatten())]))
                fout.write('\n')
            fout.close()