import json

import numpy as np
import quaternion
from shapely.geometry.polygon import Polygon


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M


def normalize(a, axis=-1, order=2):
    '''
    Normalize any kinds of tensor data along a specific axis
    :param a: source tensor data.
    :param axis: data on this axis will be normalized.
    :param order: Norm order, L0, L1 or L2.
    :return:
    '''
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1

    if len(a.shape) == 1:
        return a / l2
    else:
        return a / np.expand_dims(l2, axis)


def get_iou_cuboid(cu1, cu2):
    """
        Calculate the Intersection over Union (IoU) of two 3D cuboid.

        Parameters
        ----------
        cu1 : numpy array, 8x3
        cu2 : numpy array, 8x3

        Returns
        -------
        float
            in [0, 1]
    """

    # 2D projection on the horizontal plane (x-y plane)
    polygon2D_1 = Polygon(
        [(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])

    polygon2D_2 = Polygon(
        [(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])

    # 2D intersection area of the two projections.
    intersect_2D = polygon2D_1.intersection(polygon2D_2).area

    # the volume of the intersection part of cu1 and cu2
    inter_vol = intersect_2D * max(0.0, min(cu1[4][2], cu2[4][2]) - max(cu1[0][2], cu2[0][2]))

    # the volume of cu1 and cu2
    vol1 = polygon2D_1.area * (cu1[4][2] - cu1[0][2])
    vol2 = polygon2D_2.area * (cu2[4][2] - cu2[0][2])

    # return 3D IoU
    return inter_vol / (vol1 + vol2 - inter_vol)


def json_write(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def json_read(filename):
    with open(filename, 'r') as infile:
        return json.load(infile)

def get_box_corners(center, vectors):
    '''
    Convert box center and vectors to the corner-form
    :param center:
    :param vectors:
    :return: corner points related to the box
    '''
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

    corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
    corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    return corner_pnts
