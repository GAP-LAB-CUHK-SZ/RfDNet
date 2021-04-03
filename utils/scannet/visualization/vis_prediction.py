import sys
sys.path.append('.')
import numpy as np
import vtk
from utils import pc_util
import os
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from utils.scannet.visualization.vis_gt import Vis_base

if __name__ == '__main__':
    root_path = '/home/ynie/Project/SceneCompletion/out/iscnet/2020-10-30T15:40:24.913443/visualization/test_90_scene0575_00'
    predicted_boxes = np.load(os.path.join(root_path, '000000_pred_confident_nms_bbox.npz'))
    input_point_cloud = pc_util.read_ply(os.path.join(root_path, '000000_pc.ply'))
    bbox_params = predicted_boxes['obbs']
    proposal_map = predicted_boxes['proposal_map']

    transform_m = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

    instance_models = []
    center_list = []
    vector_list = []
    class_ids = []
    for map_data, bbox_param in zip(proposal_map, bbox_params):
        mesh_file = os.path.join(root_path, 'proposal_%d_target_%d_class_%d_mesh.ply' % tuple(map_data))
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
        class_ids.append(map_data[2])

    input_point_cloud = np.hstack([input_point_cloud, np.zeros_like(input_point_cloud)])

    scene = Vis_base(scene_points=input_point_cloud, instance_models=instance_models, center_list=center_list,
                     vector_list=vector_list, class_ids=class_ids)
    save_path = '/home/ynie/Project/SceneCompletion/out/selected_samples'
    save_path = os.path.join(save_path, '_'.join(os.path.basename(root_path).split('_')[2:]))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    camera_center = np.array([-3, 2, 3])
    scene.visualize(centroid=camera_center, save_path=os.path.join(save_path, 'pred.png'))
    scene.visualize(centroid=camera_center, save_path=os.path.join(save_path, 'points.png'), only_points=True)




