# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Generic Code for Object Detection Evaluation

    Input:
    For each class:
        For each image:
            Predictions: box, score
            Groundtruths: box

    Output:
    For each class:
        precision-recal and average precision

    Author: Charles R. Qi

    Ref: https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/lib/datasets/voc_eval.py
"""
from multiprocessing import Pool
from net_utils.box_util import box3d_iou
import numpy as np
from net_utils.metric_util import calc_iou # axis-aligned 3D box IoU

def compute_mesh_iou(voxel1, voxel2):
    voxel1_internal, voxel1_surface = voxel1
    voxel2_internal, voxel2_surface = voxel2

    if voxel1_surface.filled_count ==0 or voxel2_surface.filled_count == 0:
        return 0.

    # (Note: internal voxels would be empty)
    if voxel1_internal.filled_count > 0 and voxel2_internal.filled_count > 0:
        v1_internal_points = voxel1_internal.points
        # v1 surface points that are not belong to internal.
        v1_surface_points = voxel1_surface.points[voxel1_internal.is_filled(voxel1_surface.points) == False]
        v1_points = np.vstack([v1_internal_points, v1_surface_points])

        v2_internal_points = voxel2_internal.points
        # v2 surface points that are not belong to internal.
        v2_surface_points = voxel2_surface.points[voxel2_internal.is_filled(voxel2_surface.points) == False]
        v2_points = np.vstack([v2_internal_points, v2_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points) + voxel2_internal.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points) + voxel1_internal.is_filled(v2_points))

    elif voxel1_internal.filled_count == 0 and voxel2_internal.filled_count > 0:
        v1_points = voxel1_surface.points

        v2_internal_points = voxel2_internal.points
        # v2 surface points that are not belong to internal.
        v2_surface_points = voxel2_surface.points[voxel2_internal.is_filled(voxel2_surface.points) == False]
        v2_points = np.vstack([v2_internal_points, v2_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points) + voxel2_internal.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points))

    elif voxel1_internal.filled_count > 0 and voxel2_internal.filled_count == 0:
        v2_points = voxel2_surface.points

        v1_internal_points = voxel1_internal.points
        # v1 surface points that are not belong to internal.
        v1_surface_points = voxel1_surface.points[voxel1_internal.is_filled(voxel1_surface.points) == False]
        v1_points = np.vstack([v1_internal_points, v1_surface_points])

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points) + voxel1_internal.is_filled(v2_points))
    else:
        v1_points = voxel1_surface.points
        v2_points = voxel2_surface.points

        v1_in_v2 = sum(voxel2_surface.is_filled(v1_points))
        v2_in_v1 = sum(voxel1_surface.is_filled(v2_points))

    if v1_in_v2 == 0 or v2_in_v1 == 0:
        return 0.

    alpha1 = v1_in_v2 / v1_points.shape[0]
    alpha2 = v2_in_v1 / v2_points.shape[0]

    return (alpha1 * alpha2) / (alpha1 + alpha2 - alpha1 * alpha2)


def get_iou_obb(bb1,bb2):
    iou3d, iou2d = box3d_iou(bb1,bb2)
    return iou3d

def get_iou_main(get_iou_func, args):
    return get_iou_func(*args)

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_iou(bb1, bb2):
    """ Compute IoU of two bounding boxes.
        ** Define your bod IoU function HERE **
    """
    #pass
    iou3d = calc_iou(bb1, bb2)
    return iou3d

def eval_det_cls_w_mesh(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou, get_iou_mesh=compute_mesh_iou):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array([item[0] for item in gt[img_id]])
        mesh = [item[1] for item in gt[img_id]]
        det = [False] * len(bbox)
        det_mesh = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det, 'mesh':mesh, 'det_mesh': det_mesh}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': [], 'mesh':[], 'det_mesh': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    meshes = []
    for img_id in pred.keys():
        for box,score,mesh in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
            meshes.append(mesh)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    meshes = [meshes[x] for x in sorted_ind]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    tp_mesh = np.zeros(nd)
    fp_mesh = np.zeros(nd)
    for d in range(nd):
        if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        mesh_pred = meshes[d]

        ovmax = -np.inf
        ovmax_mesh = -np.inf
        BBGT = R['bbox'].astype(float)
        MESH_GT = R['mesh']

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

                iou_mesh = get_iou_main(get_iou_mesh, (mesh_pred, MESH_GT[j]))
                if iou_mesh > ovmax_mesh:
                    ovmax_mesh = iou_mesh
                    jmax_mesh = j

            # jmax_mesh = jmax
            # ovmax_mesh = get_iou_main(get_iou_mesh, (mesh_pred, MESH_GT[jmax_mesh]))

        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

        #print d, ovmax for mesh
        if ovmax_mesh > ovthresh:
            if not R['det_mesh'][jmax_mesh]:
                tp_mesh[d] = 1.
                R['det_mesh'][jmax_mesh] = 1
            else:
                fp_mesh[d] = 1.
        else:
            fp_mesh[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # for mesh
    # compute precision recall
    fp_mesh = np.cumsum(fp_mesh)
    tp_mesh = np.cumsum(tp_mesh)
    rec_mesh = tp_mesh / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec_mesh = tp_mesh / np.maximum(tp_mesh + fp_mesh, np.finfo(np.float64).eps)
    ap_mesh = voc_ap(rec_mesh, prec_mesh, use_07_metric)

    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)

def eval_det_cls_wo_mesh(pred, gt, ovthresh=0.25, use_07_metric=False, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box,score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #if d%100==0: print(d)
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get_iou_main(get_iou_func, (bb, BBGT[j,...]))
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        #print d, ovmax
        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    #print('NPOS: ', npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def eval_det_cls_wrapper_w_mesh(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func, get_iou_mesh = arguments
    (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh) = eval_det_cls_w_mesh(pred, gt, ovthresh, use_07_metric, get_iou_func, get_iou_mesh)
    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)

def eval_det_cls_wrapper_wo_mesh(arguments):
    pred, gt, ovthresh, use_07_metric, get_iou_func = arguments
    rec, prec, ap = eval_det_cls_wo_mesh(pred, gt, ovthresh, use_07_metric, get_iou_func)
    return (rec, prec, ap)

def eval_det_multiprocessing_w_mesh(pred_all, gt_all, ovthresh=0.25, use_07_metric=True, get_iou_func=get_iou, get_iou_mesh=compute_mesh_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score, mesh in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score, mesh))
    for img_id in gt_all.keys():
        for classname, bbox, mesh in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append((bbox, mesh))

    rec = {}
    prec = {}
    ap = {}
    rec_mesh = {}
    prec_mesh = {}
    ap_mesh = {}

    try:
        p = Pool(processes=8)
        ret_values = p.map(eval_det_cls_wrapper_w_mesh,
                           [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func, get_iou_mesh) for classname in
                            gt.keys() if classname in pred])
        p.close()
        p.join()
    except:
        ret_values = []
        for classname in gt.keys():
            if classname not in pred:
                continue
            ret_value = eval_det_cls_wrapper_w_mesh((pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func, get_iou_mesh))
            ret_values.append(ret_value)

    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            (rec[classname], prec[classname], ap[classname]), (rec_mesh[classname], prec_mesh[classname], ap_mesh[classname]) = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0

            rec_mesh[classname] = 0
            prec_mesh[classname] = 0
            ap_mesh[classname] = 0
        print(classname, 'box', ap[classname])
        print(classname, 'mesh', ap_mesh[classname])

    return (rec, prec, ap), (rec_mesh, prec_mesh, ap_mesh)

def eval_det_multiprocessing_wo_mesh(pred_all, gt_all, ovthresh=0.25, use_07_metric=True, get_iou_func=get_iou):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {}  # map {classname: pred}
    gt = {}  # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox, score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    p = Pool(processes=10)
    ret_values = p.map(eval_det_cls_wrapper_wo_mesh,
                       [(pred[classname], gt[classname], ovthresh, use_07_metric, get_iou_func) for classname in
                        gt.keys() if classname in pred])
    p.close()
    p.join()
    for i, classname in enumerate(gt.keys()):
        if classname in pred:
            rec[classname], prec[classname], ap[classname] = ret_values[i]
        else:
            rec[classname] = 0
            prec[classname] = 0
            ap[classname] = 0
        print(classname, ap[classname])

    return rec, prec, ap