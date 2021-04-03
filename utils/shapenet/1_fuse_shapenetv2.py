import sys
sys.path.append('.')
from utils.shapenet import ShapeNetv2_path, ShapeNetv2_Watertight_path
import os
from utils.read_and_write import read_obj, write_obj
import math
import numpy as np
from external.librender import pyrender
from configs.path_config import SHAPENETCLASSES, ShapeNetIDMap, ScanNet_OBJ_CLASS_IDS
from scipy import ndimage
from external import pyfusion
import mcubes
from multiprocessing import Pool
from glob import glob

def get_points(n_views=100):
    """
    See https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere.

    :param n_points: number of points
    :type n_points: int
    :return: list of points
    :rtype: numpy.ndarray
    """

    rnd = 1.
    points = []
    offset = 2. / n_views
    increment = math.pi * (3. - math.sqrt(5.))

    for i in range(n_views):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % n_views) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    # visualization.plot_point_cloud(np.array(points))
    return np.array(points)

def get_views(n_camera_views):
    """
    Generate a set of views to generate depth maps from.

    :param n_views: number of views per axis
    :type n_views: int
    :return: rotation matrices
    :rtype: [numpy.ndarray]
    """

    Rs = []
    points = get_points(n_camera_views)

    for i in range(points.shape[0]):
        # https://math.stackexchange.com/questions/1465611/given-a-point-on-a-sphere-how-do-i-find-the-angles-needed-to-point-at-its-ce
        longitude = - math.atan2(points[i, 0], points[i, 1])
        latitude = math.atan2(points[i, 2], math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2))

        R_x = np.array([[1, 0, 0],
                        [0, math.cos(latitude), -math.sin(latitude)],
                        [0, math.sin(latitude), math.cos(latitude)]])
        R_y = np.array([[math.cos(longitude), 0, math.sin(longitude)],
                        [0, 1, 0],
                        [-math.sin(longitude), 0, math.cos(longitude)]])

        R = R_y.dot(R_x)
        Rs.append(R)

    return Rs

def render(vertices, faces, camera_Rs):
    depth_maps = []
    for camera_R in camera_Rs:
        np_vertices = camera_R.dot(vertices.astype(np.float64).T)
        np_vertices[2, :] += 1
        np_faces = faces.astype(np.float64)

        depthmap, mask, img = pyrender.render(np_vertices.copy(), np_faces.T.copy(), render_intrinsics, znf, image_size)
        depthmap -= 1.5 * voxel_size
        depthmap = ndimage.morphology.grey_erosion(depthmap, size=(3, 3))
        depth_maps.append(depthmap)
    return depth_maps

def fuse_depthmaps(depthmaps, Rs):
    Ks = fusion_intrisics.reshape((1, 3, 3))
    Ks = np.repeat(Ks, len(depthmaps), axis=0).astype(np.float32)

    Ts = []
    for i in range(len(Rs)):
        Rs[i] = Rs[i]
        Ts.append(np.array([0, 0, 1]))

    Ts = np.array(Ts).astype(np.float32)
    Rs = np.array(Rs).astype(np.float32)

    depthmaps = np.array(depthmaps).astype(np.float32)
    views = pyfusion.PyViews(depthmaps, Ks, Rs, Ts)
    tsdf = pyfusion.tsdf_gpu(views, resolution, resolution, resolution, voxel_size, truncation, False)
    mask_grid = pyfusion.tsdfmask_gpu(views, resolution, resolution, resolution, voxel_size, truncation, False)
    tsdf[mask_grid==0.] = truncation
    tsdf = np.transpose(tsdf[0], [2, 1, 0])
    return tsdf

def batch_scan(input_obj):
    output_path = os.path.join(ShapeNetv2_Watertight_path, '/'.join(input_obj.split('/')[2:4])+'.off')
    if os.path.exists(output_path):
        return
    else:
        if not os.path.exists(os.path.dirname(output_path)):
            os.mkdir(os.path.dirname(output_path))

    obj_file = read_obj(input_obj, flags=('v', 'f'))
    vertices = obj_file['v']
    faces = obj_file['f']
    faces = np.array([[int(face_id.split('/')[0]) for face_id in item] for item in faces])

    '''Scale to [-0.5, 0.5], center at 0.'''
    center = (vertices.max(0) + vertices.min(0)) / 2.
    scale = max(vertices.max(0) - vertices.min(0))
    scale = scale / (1 - padding)
    vertices_normalized = (vertices - center) / scale

    '''Render depth maps'''
    camera_Rs = get_views(n_camera_views)
    depths = render(vertices=vertices_normalized, faces=faces, camera_Rs=camera_Rs)

    '''Fuse depth maps'''
    tsdf = fuse_depthmaps(depths, camera_Rs)
    # To ensure that the final mesh is indeed watertight
    tsdf = np.pad(tsdf, 1, 'constant', constant_values=1e6)
    vertices, triangles = mcubes.marching_cubes(-tsdf, 0)
    # Remove padding offset
    vertices -= 1
    # Normalize to [-0.5, 0.5]^3 cube
    vertices /= resolution
    vertices -= 0.5

    '''scale back'''
    vertices = vertices * scale + center
    mcubes.export_off(vertices, triangles, output_path)

if __name__ == '__main__':
    focal_length_x, focal_length_y = 640., 640.
    principal_point_x, principal_point_y = 320., 320.
    image_height, image_width = 640, 640
    n_camera_views = 200
    render_intrinsics = np.array([focal_length_x, focal_length_y, principal_point_x, principal_point_y])
    image_size = np.array([image_height, image_width], dtype=np.int32)
    znf = np.array([1 - 0.75, 1 + 0.75], dtype=float)
    fusion_intrisics = np.array([
        [focal_length_x, 0, principal_point_x],
        [0, focal_length_y, principal_point_y],
        [0, 0, 1]
    ])
    resolution = 256
    truncation_factor = 10
    voxel_size = 1. / resolution
    truncation = truncation_factor*voxel_size
    padding = 0.1
    GT_N_Points = 100000
    name_id_map = {name: id for id, name in ShapeNetIDMap.items()}
    ShapeNetIds = [name_id_map[SHAPENETCLASSES[idx]] for idx in ScanNet_OBJ_CLASS_IDS]

    ShapeNet_objs = [glob(os.path.join(ShapeNetv2_path, '0' + idx, '*','models', 'model_normalized.obj')) for idx in ShapeNetIds]
    ShapeNet_objs = sum(ShapeNet_objs, [])

    if not os.path.isdir(ShapeNetv2_Watertight_path):
        os.makedirs(ShapeNetv2_Watertight_path)

    p = Pool(processes=8)
    p.map(batch_scan, ShapeNet_objs)
    p.close()
    p.join()
