'''
Visualization tools for Scannet.
author: ynie
date: July, 2020
'''
import sys

sys.path.append('.')
import os
from configs.path_config import PathConfig
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from utils.scannet.tools import make_M_from_tqs, calc_Mbbox, get_iou_cuboid, normalize, json_read
from utils.scannet.visualization.vis_scannet import Vis_Scannet
import random
from utils.scannet.load_scannet_data import export
import pickle
import seaborn as sns
from configs.path_config import SHAPENETCLASSES, ShapeNetIDMap
from utils.shapenet import ShapeNetv2_path
from utils.read_and_write import read_obj

path_config = PathConfig('scannet')


class Vis_Scan2CAD(Vis_Scannet):
    '''
    visualization class for scannet frames.
    '''

    def __init__(self, gt_dirname: str = None, shapenet_instances: list = None):
        scene_name = os.path.basename(gt_dirname)
        self._mesh_file = os.path.join(gt_dirname, scene_name + '_vh_clean_2.ply')
        agg_file = os.path.join(gt_dirname, scene_name + '.aggregation.json')
        seg_file = os.path.join(gt_dirname, scene_name + '_vh_clean_2.0.010000.segs.json')
        meta_file = os.path.join(gt_dirname, scene_name + '.txt')  # includes axis

        with open(path_config.raw_label_map_file, 'rb') as file:
            label_map = pickle.load(file)

        self._vertices, semantic_labels, instance_labels, instance_bboxes, instance2semantic = \
            export(self._mesh_file, agg_file, seg_file, meta_file, label_map, None)

        self.palette_cls = np.array([(0., 0., 0.), *sns.color_palette("hls", len(SHAPENETCLASSES))])

        instances_scan2cad = []
        for scan2cad_object in shapenet_instances:
            center = scan2cad_object['box3D'][:, 3][:3]
            vectors = scan2cad_object['box3D'][:3, :3].T
            scan2cad_corners = np.array(self.get_box_corners(center, vectors)[0])

            best_iou_score = 0.
            best_instance_id = None
            best_cls_id = None
            for inst_id, instance_bbox in enumerate(instance_bboxes):
                center = instance_bbox[:3]
                vectors = np.diag(instance_bbox[3:6]) / 2.
                scannet_corners = np.array(self.get_box_corners(center, vectors)[0])
                iou_score = get_iou_cuboid(scan2cad_corners, scannet_corners)

                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_instance_id = inst_id + 1
                    best_cls_id = int(instance_bbox[-1])
            if best_iou_score == 0.:
                continue

            cls_id = SHAPENETCLASSES.index(ShapeNetIDMap[scan2cad_object['shapenet_catid'][1:]])
            instances_scan2cad.append(
                {'inst_id': best_instance_id, 'bbox': scan2cad_object['box3D'],
                 'cls_id': cls_id, 'vtk_object':scan2cad_object['vtk_object']})
        self.instances = instances_scan2cad
        self.cam_K = np.array([[600, 0, 800], [0, 600, 600], [0, 0, 1]])

    def set_arrow_actor(self, startpoint, vector):
        '''
        Design an actor to draw an arrow from startpoint to startpoint + vector.
        :param startpoint: 3D point
        :param vector: 3D vector
        :return: an vtk arrow actor
        '''
        arrow_source = vtk.vtkArrowSource()
        arrow_source.SetTipLength(0.2)
        arrow_source.SetTipRadius(0.08)
        arrow_source.SetShaftRadius(0.02)

        vector = vector / np.linalg.norm(vector) * 0.5

        endpoint = startpoint + vector

        # compute a basis
        normalisedX = [0 for i in range(3)]
        normalisedY = [0 for i in range(3)]
        normalisedZ = [0 for i in range(3)]

        # the X axis is a vector from start to end
        math = vtk.vtkMath()
        math.Subtract(endpoint, startpoint, normalisedX)
        length = math.Norm(normalisedX)
        math.Normalize(normalisedX)

        # the Z axis is an arbitrary vector cross X
        arbitrary = [0 for i in range(3)]
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)
        math.Cross(normalisedX, arbitrary, normalisedZ)
        math.Normalize(normalisedZ)

        # the Y axis is Z cross X
        math.Cross(normalisedZ, normalisedX, normalisedY)

        # create the direction cosine matrix
        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalisedX[i])
            matrix.SetElement(i, 1, normalisedY[i])
            matrix.SetElement(i, 2, normalisedZ[i])

        # apply the transform
        transform = vtk.vtkTransform()
        transform.Translate(startpoint)
        transform.Concatenate(matrix)
        transform.Scale(length, length, length)

        # create a mapper and an actor for the arrow
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()

        mapper.SetInputConnection(arrow_source.GetOutputPort())
        actor.SetUserMatrix(transform.GetMatrix())
        actor.SetMapper(mapper)

        return actor

    def set_render(self, *args, vis_pc=True):
        renderer = vtk.vtkRenderer()

        '''draw world system'''
        renderer.AddActor(self.set_axes_actor())

        '''load ply mesh file'''
        ply_actor = self.set_actor(self.set_mapper(self.set_ply_property(self.mesh_file), 'model'))
        ply_actor.GetProperty().SetInterpolationToPBR()
        renderer.AddActor(ply_actor)

        '''load bounding boxes'''
        for box3D in self.instances:
            cls_id = box3D['cls_id']
            if cls_id not in path_config.OBJ_CLASS_IDS:
                continue
            center = box3D['bbox'][:, 3][:3]
            vectors = box3D['bbox'][:3, :3].T
            corners, faces = self.get_box_corners(center, vectors)
            color = self.palette_cls[cls_id] * 255
            bbox_actor = self.set_actor(self.set_mapper(self.set_cube_prop(corners, faces, color), 'box'))
            bbox_actor.GetProperty().SetOpacity(0.3)
            bbox_actor.GetProperty().SetInterpolationToPBR()
            renderer.AddActor(bbox_actor)

            # draw obj model
            if box3D['vtk_object'] is not None:
                object_actor = self.set_actor(self.set_mapper(box3D['vtk_object'], 'model'))
                object_actor.GetProperty().SetColor(color/255)
                object_actor.GetProperty().SetInterpolationToPBR()
                renderer.AddActor(object_actor)
            # draw orientations
            color = [[1, 0, 0], [0, 1, 0], [0., 0., 1.]]

            for index in range(vectors.shape[0]):
                arrow_actor = self.set_arrow_actor(center, vectors[index])
                arrow_actor.GetProperty().SetColor(color[index])
                renderer.AddActor(arrow_actor)

        '''light'''
        positions = [(10, 10, 10), (-10, 10, 10), (10, -10, 10), (-10, -10, 10)]
        for position in positions:
            light = vtk.vtkLight()
            light.SetIntensity(1.5)
            light.SetPosition(*position)
            light.SetPositional(True)
            light.SetFocalPoint(0, 0, 0)
            light.SetColor(1., 1., 1.)
            renderer.AddLight(light)

        renderer.SetBackground(1., 1., 1.)
        return renderer


if __name__ == '__main__':

    filename_json = path_config.scan2cad_annotation_path

    scene_dirname = 'scene0001_00'
    gt_dirname = os.path.join(path_config.metadata_root, 'scans', scene_dirname)
    assert os.path.exists(gt_dirname)

    # calculate transform matrix from scan2cad to scannet
    meta_file = os.path.join(gt_dirname, os.path.basename(gt_dirname) + '.txt')
    assert os.path.exists(meta_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) \
                                 for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break

    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    r = [r for r in json_read(filename_json) if r["id_scan"] == scene_dirname][0]
    id_scan = r["id_scan"]

    Mscan = make_M_from_tqs(r["trs"]["translation"], r["trs"]["rotation"], r["trs"]["scale"])
    R_transform = np.array(axis_align_matrix).reshape((4, 4)).dot(np.linalg.inv(Mscan))

    shapenet_instances = []
    for model in r['aligned_models']:
        # read corresponding shapenet scanned points
        id_cad = model["id_cad"]
        catid_cad = model["catid_cad"]
        obj_path = os.path.join(ShapeNetv2_path, catid_cad, id_cad + '/models/model_normalized.obj')
        vtk_object = vtk.vtkOBJReader()
        vtk_object.SetFileName(obj_path)
        vtk_object.Update()
        # get points from object
        polydata = vtk_object.GetOutput()
        # read points using vtk_to_numpy
        obj_points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float)
        # read transformation
        t = model["trs"]["translation"]
        q = model["trs"]["rotation"]
        s = model["trs"]["scale"]
        Mcad = make_M_from_tqs(t, q, s)
        transform_shape = R_transform.dot(Mcad)
        '''get transformed axes'''
        center = (obj_points.max(0) + obj_points.min(0)) / 2.
        axis_points = np.array([center,
                                center - np.array([0, 0, 1]),
                                center - np.array([1, 0, 0]),
                                center + np.array([0, 1, 0])])
        axis_points_transformed = np.hstack([axis_points, np.ones((axis_points.shape[0], 1))]).dot(transform_shape.T)[..., :3]
        center_transformed = axis_points_transformed[0]
        forward_transformed = axis_points_transformed[1] - axis_points_transformed[0]
        left_transformed = axis_points_transformed[2] - axis_points_transformed[0]
        up_transformed = axis_points_transformed[3] - axis_points_transformed[0]
        forward_transformed = normalize(forward_transformed)
        left_transformed = normalize(left_transformed)
        up_transformed = normalize(up_transformed)
        axis_transformed = np.array([forward_transformed, left_transformed, up_transformed])
        '''get rectified axis'''
        axis_rectified = np.zeros_like(axis_transformed)
        up_rectified_id = np.argmax(axis_transformed[:, 2])
        forward_rectified_id = 0 if up_rectified_id !=0 else (up_rectified_id + 1) % 3
        left_rectified_id = np.setdiff1d([0, 1, 2], [up_rectified_id, forward_rectified_id])[0]
        up_rectified = np.array([0, 0, 1])
        forward_rectified = axis_transformed[forward_rectified_id]
        forward_rectified = np.array([*forward_rectified[:2], 0.])
        forward_rectified = normalize(forward_rectified)
        left_rectified = np.cross(up_rectified, forward_rectified)
        axis_rectified[forward_rectified_id] = forward_rectified
        axis_rectified[left_rectified_id] = left_rectified
        axis_rectified[up_rectified_id] = up_rectified
        if np.linalg.det(axis_rectified) < 0:
            axis_rectified[left_rectified_id] *= -1
        '''deploy points'''
        obj_points = np.hstack([obj_points, np.ones((obj_points.shape[0], 1))]).dot(transform_shape.T)[..., :3]
        coordinates = (obj_points - center_transformed).dot(axis_transformed.T)
        obj_points = coordinates.dot(axis_rectified) + center_transformed
        '''define bounding boxes'''
        sizes = (coordinates.max(0) - coordinates.min(0)) / 2
        vectors = np.diag(sizes[[forward_rectified_id, left_rectified_id, up_rectified_id]]).dot(np.array([forward_rectified, left_rectified, up_rectified]))
        box3D = np.eye(4)
        box3D[:3, :] = np.hstack([vectors.T, center_transformed[np.newaxis].T])

        points_array = numpy_to_vtk(obj_points[..., :3], deep=True)
        polydata.GetPoints().SetData(points_array)
        vtk_object.Update()

        shapenet_instances.append({'box3D':box3D, 'shapenet_catid': catid_cad, 'shapenet_id': id_cad, 'vtk_object':vtk_object})

    scene = Vis_Scan2CAD(gt_dirname=gt_dirname, shapenet_instances=shapenet_instances)
    scene.visualize()
