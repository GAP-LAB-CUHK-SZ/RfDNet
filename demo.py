# demo file.
# author: ynie
# date: July, 2020
from net_utils.utils import load_device, load_model
from net_utils.utils import CheckpointIO
from configs.config_utils import mount_external_config
from time import time
import trimesh
import numpy as np
from utils import pc_util
from models.iscnet.dataloader import collate_fn
import torch
from net_utils.ap_helper import parse_predictions
from net_utils.libs import flip_axis_to_depth, extract_pc_in_box3d, flip_axis_to_camera
from net_utils.box_util import get_3d_box
from torch import optim
from models.loss import chamfer_func
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils.scannet.visualization.vis_for_demo import Vis_base


def load_demo_data(cfg, device):
    point_cloud = trimesh.load(cfg.config['demo_path']).vertices
    use_color = cfg.config['data']['use_color_detection'] or cfg.config['data']['use_color_completion']
    MEAN_COLOR_RGB = np.array([121.87661, 109.73591, 95.61673])
    use_height = not cfg.config['data']['no_height']
    num_points = cfg.config['data']['num_point']

    if not use_color:
        point_cloud = point_cloud[:, 0:3]  # do not use color for now
    else:
        point_cloud = point_cloud[:, 0:6]
        point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0

    if use_height:
        floor_height = np.percentile(point_cloud[:, 2], 0.99)
        height = point_cloud[:, 2] - floor_height
        point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

    point_cloud, choices = pc_util.random_sampling(point_cloud, num_points, return_choices=True)
    data = collate_fn([{'point_clouds': point_cloud.astype(np.float32)}])

    for key in data:
        if key not in ['object_voxels', 'shapenet_catids', 'shapenet_ids']:
            data[key] = data[key].to(device)
    return data

def get_proposal_id(cfg, end_points, data, mode='random', batch_sample_ids=None, DUMP_CONF_THRESH=-1.):
    '''
    Get the proposal ids for completion training for the limited GPU RAM.
    :param end_points: estimated data from votenet.
    :param data: data source which contains gt contents.
    :return:
    '''
    batch_size = 1
    device = end_points['center'].device
    NUM_PROPOSALS = end_points['center'].size(1)
    proposal_id_list = []

    if mode == 'objectness' or batch_sample_ids is not None:
        objectness_probs = torch.softmax(end_points['objectness_scores'], dim=2)[..., 1]

    for batch_id in range(batch_size):

        proposal_to_gt_box_w_cls = torch.arange(0, NUM_PROPOSALS).unsqueeze(-1).to(device).long()

        sample_ids = (objectness_probs[batch_id] > DUMP_CONF_THRESH).cpu().numpy()*batch_sample_ids[batch_id]
        sample_ids = sample_ids.astype(np.bool)

        proposal_to_gt_box_w_cls = proposal_to_gt_box_w_cls[sample_ids].long()
        proposal_id_list.append(proposal_to_gt_box_w_cls.unsqueeze(0))

    return torch.cat(proposal_id_list, dim=0)

def chamfer_dist(obj_points, obj_points_masks, pc_in_box, pc_in_box_masks, centroid_params, orientation_params):
    b_s = obj_points.size(0)
    axis_rectified = torch.zeros(size=(b_s, 3, 3)).to(obj_points.device)
    axis_rectified[:, 2, 2] = 1
    axis_rectified[:, 0, 0] = torch.cos(orientation_params)
    axis_rectified[:, 0, 1] = torch.sin(orientation_params)
    axis_rectified[:, 1, 0] = -torch.sin(orientation_params)
    axis_rectified[:, 1, 1] = torch.cos(orientation_params)
    obj_points_after = torch.bmm(obj_points, axis_rectified) + centroid_params.unsqueeze(-2)
    dist1, dist2 = chamfer_func(obj_points_after, pc_in_box)
    return torch.mean(dist2 * pc_in_box_masks)*1e3

def fit_mesh_to_scan(cfg, pred_mesh_dict, parsed_predictions, eval_dict, input_scan, dump_threshold):
    '''fit meshes to input scan'''
    pred_corners_3d_upright_camera = parsed_predictions['pred_corners_3d_upright_camera']
    pred_sem_cls = parsed_predictions['pred_sem_cls']
    bsize, N_proposals = pred_sem_cls.shape
    pred_mask = eval_dict['pred_mask']
    obj_prob = parsed_predictions['obj_prob']
    device = input_scan.device
    input_scan = input_scan.cpu().numpy()
    transform_shapenet = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    index_list = []
    box_params_list = []
    max_obj_points = 10000
    max_pc_in_box = 50000
    obj_points_list = []
    obj_points_mask_list = []
    pc_in_box_list = []
    pc_in_box_mask_list = []
    for i in range(bsize):
        for j in range(N_proposals):
            if not (pred_mask[i, j] == 1 and obj_prob[i, j] > dump_threshold):
                continue
            # get mesh points
            mesh_data = pred_mesh_dict['meshes'][list(pred_mesh_dict['proposal_ids'][i,:,0]).index(j)]
            obj_points = mesh_data.vertices
            obj_points = obj_points - (obj_points.max(0) + obj_points.min(0)) / 2.
            obj_points = obj_points.dot(transform_shapenet.T)
            obj_points = obj_points / (obj_points.max(0) - obj_points.min(0))

            obj_points_matrix = np.zeros((max_obj_points, 3))
            obj_points_mask = np.zeros((max_obj_points,), dtype=np.uint8)
            obj_points_matrix[:obj_points.shape[0], :] = obj_points
            obj_points_mask[:obj_points.shape[0]] = 1

            # box corners
            box_corners_cam = pred_corners_3d_upright_camera[i, j]
            box_corners_depth = flip_axis_to_depth(box_corners_cam)
            # box vector form
            centroid = (np.max(box_corners_depth, axis=0) + np.min(box_corners_depth, axis=0)) / 2.
            forward_vector = box_corners_depth[1] - box_corners_depth[2]
            left_vector = box_corners_depth[0] - box_corners_depth[1]
            up_vector = box_corners_depth[6] - box_corners_depth[2]
            orientation = np.arctan2(forward_vector[1], forward_vector[0])
            sizes = np.linalg.norm([forward_vector, left_vector, up_vector], axis=1)
            box_params = np.array([*centroid, *sizes, orientation])

            # points in larger boxes (remove grounds)
            larger_box = flip_axis_to_depth(get_3d_box(1.2*sizes, -orientation, flip_axis_to_camera(centroid)))
            height = np.percentile(input_scan[i, :, 2], 5)
            scene_scan = input_scan[i, input_scan[i, :, 2] >= height, :3]
            pc_in_box, inds = extract_pc_in_box3d(scene_scan, larger_box)
            if len(pc_in_box) < 5:
                continue

            pc_in_box_matrix = np.zeros((max_pc_in_box, 3))
            pc_in_box_mask = np.zeros((max_pc_in_box,), dtype=np.uint8)
            pc_in_box_matrix[:pc_in_box.shape[0], :] = pc_in_box
            pc_in_box_mask[:pc_in_box.shape[0]] = 1

            index_list.append((i, j))
            obj_points_list.append(obj_points_matrix)
            obj_points_mask_list.append(obj_points_mask)
            box_params_list.append(box_params)
            pc_in_box_list.append(pc_in_box_matrix)
            pc_in_box_mask_list.append(pc_in_box_mask)

    obj_points_list = np.array(obj_points_list)
    pc_in_box_list = np.array(pc_in_box_list)
    obj_points_mask_list = np.array(obj_points_mask_list)
    pc_in_box_mask_list = np.array(pc_in_box_mask_list)
    box_params_list = np.array(box_params_list)

    # scale to predicted sizes
    obj_points_list = obj_points_list * box_params_list[:, np.newaxis, 3:6]

    obj_points_list = torch.from_numpy(obj_points_list).to(device).float()
    pc_in_box_list = torch.from_numpy(pc_in_box_list).to(device).float()
    pc_in_box_mask_list = torch.from_numpy(pc_in_box_mask_list).to(device).float()
    '''optimize box center and orientation'''
    centroid_params = box_params_list[:, :3]
    orientation_params = box_params_list[:, 6]
    centroid_params = torch.from_numpy(centroid_params).to(device).float()
    orientation_params = torch.from_numpy(orientation_params).to(device).float()
    centroid_params.requires_grad = True
    orientation_params.requires_grad = True

    lr = 0.01
    iterations = 100
    optimizer = optim.Adam([centroid_params, orientation_params], lr=lr)

    centroid_params_cpu, orientation_params_cpu, best_loss = None, None, 1e6
    for iter in range(iterations):
        optimizer.zero_grad()
        loss = chamfer_dist(obj_points_list, obj_points_mask_list, pc_in_box_list, pc_in_box_mask_list,
                                 centroid_params, orientation_params)
        if loss < best_loss:
            centroid_params_cpu = centroid_params.data.cpu().numpy()
            orientation_params_cpu = orientation_params.data.cpu().numpy()
            best_loss = loss
        loss.backward()
        optimizer.step()

    for idx in range(box_params_list.shape[0]):
        i, j = index_list[idx]
        best_box_corners_cam = get_3d_box(box_params_list[idx, 3:6], -orientation_params_cpu[idx], flip_axis_to_camera(centroid_params_cpu[idx]))
        pred_corners_3d_upright_camera[i, j] = best_box_corners_cam

    parsed_predictions['pred_corners_3d_upright_camera'] = pred_corners_3d_upright_camera
    return parsed_predictions

def generate(cfg, net, data, post_processing):
    with torch.no_grad():
        '''For Detection'''
        mode = cfg.config['mode']
        inputs = {'point_clouds': data['point_clouds']}
        end_points = {}
        end_points = net.backbone(inputs['point_clouds'], end_points)
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features

        xyz, features = net.voting(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features
        # --------- DETECTION ---------
        if_proposal_feature = cfg.config[mode]['phase'] == 'completion'
        end_points, proposal_features = net.detection(xyz, features, end_points, if_proposal_feature)

        eval_dict, parsed_predictions = parse_predictions(end_points, data, cfg.eval_config)

        '''For Completion'''
        # use 3D NMS to generate sample ids.
        batch_sample_ids = eval_dict['pred_mask']

        dump_threshold = cfg.config['generation']['dump_threshold']

        BATCH_PROPOSAL_IDs = get_proposal_id(cfg, end_points, data, mode='random', batch_sample_ids=batch_sample_ids,
                                             DUMP_CONF_THRESH=dump_threshold)
        # Skip propagate point clouds to box centers.
        device = end_points['center'].device
        if not cfg.config['data']['skip_propagate']:
            gather_ids = BATCH_PROPOSAL_IDs[..., 0].unsqueeze(1).repeat(1, 128, 1).long().to(device)
            object_input_features = torch.gather(proposal_features, 2, gather_ids)
        else:
            # gather proposal features
            gather_ids = BATCH_PROPOSAL_IDs[..., 0].unsqueeze(1).repeat(1, 128, 1).long().to(device)
            proposal_features = torch.gather(proposal_features, 2, gather_ids)

            # gather proposal centers
            gather_ids = BATCH_PROPOSAL_IDs[..., 0].unsqueeze(-1).repeat(1, 1, 3).long().to(device)
            pred_centers = torch.gather(end_points['center'], 1, gather_ids)

            # gather proposal orientations
            pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # B,num_proposal
            heading_residuals = end_points['heading_residuals_normalized'] * (np.pi / cfg.eval_config[
                'dataset_config'].num_heading_bin)  # Bxnum_proposalxnum_heading_bin
            pred_heading_residual = torch.gather(heading_residuals, 2,
                                                 pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
            pred_heading_residual.squeeze_(2)
            heading_angles = cfg.eval_config['dataset_config'].class2angle_cuda(pred_heading_class,
                                                                                pred_heading_residual)
            heading_angles = torch.gather(heading_angles, 1, BATCH_PROPOSAL_IDs[..., 0])

            object_input_features = net.skip_propagation.generate(pred_centers, heading_angles, proposal_features,
                                                                  inputs['point_clouds'])

        batch_size, feat_dim, N_proposals = object_input_features.size()
        object_input_features = object_input_features.transpose(1, 2).contiguous().view(batch_size * N_proposals,
                                                                                        feat_dim)

        gather_ids = BATCH_PROPOSAL_IDs[..., 0].unsqueeze(-1).repeat(1, 1, end_points['sem_cls_scores'].size(2))
        cls_codes_for_completion = torch.gather(end_points['sem_cls_scores'], 1, gather_ids)
        cls_codes_for_completion = (
                    cls_codes_for_completion >= torch.max(cls_codes_for_completion, dim=2, keepdim=True)[0]).float()
        cls_codes_for_completion = cls_codes_for_completion.view(batch_size * N_proposals, -1)

        meshes = net.completion.generator.generate_mesh(object_input_features, cls_codes_for_completion)

    if post_processing:
        pred_mesh_dict = {'meshes': meshes, 'proposal_ids': BATCH_PROPOSAL_IDs}
        parsed_predictions = fit_mesh_to_scan(cfg, pred_mesh_dict, parsed_predictions, eval_dict, inputs['point_clouds'], dump_threshold)
    return end_points, BATCH_PROPOSAL_IDs, eval_dict, meshes, parsed_predictions

def save_visualization(cfg, input_data, our_data, output_dir):
    DUMP_CONF_THRESH = cfg.config['generation']['dump_threshold']  # Dump boxes with obj prob larger than that.

    '''Dump meshes'''
    meshes = our_data[3]
    BATCH_PROPOSAL_IDs = our_data[1][0].cpu().numpy()
    for mesh_data, map_data in zip(meshes, BATCH_PROPOSAL_IDs):
        object_mesh = os.path.join(output_dir, 'proposal_%d_mesh.ply' % tuple(map_data))
        mesh_data.export(object_mesh)

    '''Dump boxes'''
    batch_id = 0
    pred_corners_3d_upright_camera = our_data[4]['pred_corners_3d_upright_camera']
    objectness_prob = our_data[4]['obj_prob'][batch_id]

    # INPUT
    point_clouds = input_data['point_clouds'].cpu().numpy()

    # Box params
    box_corners_cam = pred_corners_3d_upright_camera[batch_id]
    box_corners_depth = flip_axis_to_depth(box_corners_cam)
    centroid = (np.max(box_corners_depth, axis=1) + np.min(box_corners_depth, axis=1)) / 2.

    forward_vector = box_corners_depth[:, 1] - box_corners_depth[:, 2]
    left_vector = box_corners_depth[:, 0] - box_corners_depth[:, 1]
    up_vector = box_corners_depth[:, 6] - box_corners_depth[:, 2]
    orientation = np.arctan2(forward_vector[:, 1], forward_vector[:, 0])
    forward_size = np.linalg.norm(forward_vector, axis=1)
    left_size = np.linalg.norm(left_vector, axis=1)
    up_size = np.linalg.norm(up_vector, axis=1)
    sizes = np.vstack([forward_size, left_size, up_size]).T

    box_params = np.hstack([centroid, sizes, orientation[:, np.newaxis]])

    # OTHERS
    eval_dict = our_data[2]
    pred_mask = eval_dict['pred_mask']  # B,num_proposal

    pc = point_clouds[batch_id, :, :]

    '''Dump point cloud'''
    pc_util.write_ply(pc, os.path.join(output_dir, '%06d_pc.ply' % (batch_id)))

    '''Dump boxes'''
    if np.sum(objectness_prob > DUMP_CONF_THRESH) > 0:
        if len(box_params) > 0:
            save_path = os.path.join(output_dir, '%06d_pred_confident_nms_bbox.npz' % (batch_id))
            np.savez(save_path,
                     obbs=box_params[np.logical_and(objectness_prob > DUMP_CONF_THRESH, pred_mask[batch_id, :] == 1), :],
                     proposal_map=BATCH_PROPOSAL_IDs)

def visualize(output_dir, offline):
    predicted_boxes = np.load(os.path.join(output_dir, '000000_pred_confident_nms_bbox.npz'))
    input_point_cloud = pc_util.read_ply(os.path.join(output_dir, '000000_pc.ply'))
    bbox_params = predicted_boxes['obbs']
    proposal_map = predicted_boxes['proposal_map']
    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    instance_models = []
    center_list = []
    vector_list = []

    for map_data, bbox_param in zip(proposal_map, bbox_params):
        mesh_file = os.path.join(output_dir, 'proposal_%d_mesh.ply' % tuple(map_data))
        ply_reader = vtk.vtkPLYReader()
        ply_reader.SetFileName(mesh_file)
        ply_reader.Update()
        # get points from object
        polydata = ply_reader.GetOutput()
        # read points using vtk_to_numpy
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)

        '''Fit obj points to bbox'''
        center = bbox_param[:3]
        orientation = bbox_param[6]
        sizes = bbox_param[3:6]

        obj_points = obj_points - (obj_points.max(0) + obj_points.min(0))/2.
        obj_points = obj_points.dot(transform_m.T)
        obj_points = obj_points.dot(np.diag(1/(obj_points.max(0) - obj_points.min(0)))).dot(np.diag(sizes))

        axis_rectified = np.array([[np.cos(orientation), np.sin(orientation), 0], [-np.sin(orientation), np.cos(orientation), 0], [0, 0, 1]])
        obj_points = obj_points.dot(axis_rectified) + center

        points_array = numpy_to_vtk(obj_points[..., :3], deep=True)
        polydata.GetPoints().SetData(points_array)
        ply_reader.Update()

        '''draw bboxes'''
        vectors = np.diag(sizes/2.).dot(axis_rectified)

        instance_models.append(ply_reader)
        center_list.append(center)
        vector_list.append(vectors)

    scene = Vis_base(scene_points=input_point_cloud, instance_models=instance_models, center_list=center_list,
                     vector_list=vector_list)

    camera_center = np.array([0, -3, 3])
    scene.visualize(centroid=camera_center, offline=offline, save_path=os.path.join(output_dir, 'pred.png'))

def run(cfg):
    '''Begin to run network.'''
    checkpoint = CheckpointIO(cfg)

    '''Mount external config data'''
    cfg = mount_external_config(cfg)

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load net'''
    cfg.log_string('Loading model.')
    net = load_model(cfg, device=device)
    checkpoint.register_modules(net=net)
    cfg.log_string(net)

    '''Load existing checkpoint'''
    checkpoint.parse_checkpoint()

    '''Load data'''
    cfg.log_string('Loading data.')
    input_data = load_demo_data(cfg, device)

    '''Run demo'''
    net.train(cfg.config['mode'] == 'train')
    start = time()
    our_data = generate(cfg, net.module, input_data, post_processing=False)
    end = time()
    print('Time elapsed: %s.' % (end - start))

    '''Save visualization'''
    scene_name = os.path.splitext(os.path.basename(cfg.config['demo_path']))[0]
    output_dir = os.path.join('demo/outputs', scene_name)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    save_visualization(cfg, input_data, our_data, output_dir)
    visualize(output_dir, offline=False)



