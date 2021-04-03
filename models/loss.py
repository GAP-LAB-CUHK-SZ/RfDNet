# loss function library.
# author: ynie
# date: Feb, 2020
import numpy as np
import torch
import torch.nn as nn

from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
from models.registers import LOSSES
from net_utils.nn_distance import nn_distance, huber_loss

chamfer_func = ChamferDistance()


FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
objectness_criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
criterion_size_class = nn.CrossEntropyLoss(reduction='none')
criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')

class BaseLoss(object):
    '''base loss class'''

    def __init__(self, weight=1):
        '''initialize loss module'''
        self.weight = weight


@LOSSES.register_module
class Null(BaseLoss):
    '''This loss function is for modules where a loss preliminary calculated.'''

    def __call__(self, loss):
        return self.weight * torch.mean(loss)


def compute_vote_loss(est_data, gt_data):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        est_data, gt_data: dict (read-only)

    Returns:
        vote_loss: scalar Tensor

    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = est_data['seed_xyz'].shape[0]
    num_seed = est_data['seed_xyz'].shape[1]  # B,num_seed,3
    vote_xyz = est_data['vote_xyz']  # B,num_seed*vote_factor,3
    seed_inds = est_data['seed_inds'].long()  # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(gt_data['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size, num_seed, 1).repeat(1, 1, 3 * GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(gt_data['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += est_data['seed_xyz'].repeat(1, 1, 3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size * num_seed, -1,
                                     3)  # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size * num_seed, GT_VOTE_FACTOR,
                                               3)  # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)  # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist * seed_gt_votes_mask.float()) / (torch.sum(seed_gt_votes_mask.float()) + 1e-6)
    return vote_loss

def compute_objectness_loss(est_data, gt_data):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = est_data['aggregated_vote_xyz']
    gt_center = gt_data['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = est_data['objectness_scores']
    objectness_loss = objectness_criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(est_data, gt_data, meta_data, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        est_data, gt_data, meta_data: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = meta_data['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = est_data['center']
    gt_center = gt_data['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = gt_data['box_label_mask']
    objectness_label = meta_data['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(gt_data['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_class_loss = criterion_heading_class(est_data['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(gt_data['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(est_data['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(gt_data['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    size_class_loss = criterion_size_class(est_data['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(gt_data['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(est_data['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3)
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(gt_data['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    sem_cls_loss = criterion_sem_cls(est_data['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


@LOSSES.register_module
class DetectionLoss(BaseLoss):
    def __call__(self, est_data, gt_data, dataset_config):
        """ Loss functions

        Args:
            end_points: dict
                {
                    seed_xyz, seed_inds, vote_xyz,
                    center,
                    heading_scores, heading_residuals_normalized,
                    size_scores, size_residuals_normalized,
                    sem_cls_scores, #seed_logits,#
                    center_label,
                    heading_class_label, heading_residual_label,
                    size_class_label, size_residual_label,
                    sem_cls_label,
                    box_label_mask,
                    vote_label, vote_label_mask
                }
            config: dataset config instance
        Returns:
            loss: pytorch scalar tensor
            end_points: dict
        """
        # Vote loss
        vote_loss = compute_vote_loss(est_data, gt_data)

        # Obj loss
        objectness_loss, objectness_label, objectness_mask, object_assignment = \
            compute_objectness_loss(est_data, gt_data)

        total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
        pos_ratio = \
            torch.sum(objectness_label.float().cuda()) / float(total_num_proposal)
        neg_ratio = \
            torch.sum(objectness_mask.float()) / float(total_num_proposal) - pos_ratio

        # Box loss and sem cls loss
        meta_data = {'object_assignment':object_assignment,
                     'objectness_label':objectness_label}
        center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
            compute_box_and_sem_cls_loss(est_data, gt_data, meta_data, dataset_config)
        box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss
        # Final loss function
        loss = vote_loss + 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
        loss *= 10

        # --------------------------------------------
        # Some other statistics
        obj_pred_val = torch.argmax(est_data['objectness_scores'], 2)  # B,K
        obj_acc = torch.sum((obj_pred_val == objectness_label.long()).float() * objectness_mask) / (
                    torch.sum(objectness_mask) + 1e-6)

        return {'total':loss,
                'vote_loss': vote_loss.item(),
                'objectness_loss': objectness_loss.item(),
                'box_loss': box_loss.item(),
                'sem_cls_loss': sem_cls_loss.item(),
                'pos_ratio': pos_ratio.item(),
                'neg_ratio': neg_ratio.item(),
                'center_loss': center_loss.item(),
                'heading_cls_loss': heading_cls_loss.item(),
                'heading_reg_loss': heading_reg_loss.item(),
                'size_cls_loss': size_cls_loss.item(),
                'size_reg_loss': size_reg_loss.item(),
                'obj_acc': obj_acc.item()}

@LOSSES.register_module
class ChamferDist(BaseLoss):
    def __call__(self, pointset1, pointset2):
        '''
        calculate the chamfer distance between two point sets.
        :param pointset1 (B x N x 3): torch.FloatTensor
        :param pointset2 (B x N x 3): torch.FloatTensor
        :return:
        '''
        dist1, dist2 = chamfer_func(pointset1, pointset2)[:2]
        loss = self.weight * ((torch.mean(dist1)) + (torch.mean(dist2)))
        return loss


@LOSSES.register_module
class PCN_Loss(BaseLoss):
    def __init__(self, weight):
        super(PCN_Loss, self).__init__(weight)
        self.chamfer_distance = ChamferDist()

    def __call__(self, pred_fine, pred_coarses, full_scan, full_scan_coarse):
        CD_LOSS = self.chamfer_distance(pred_fine, full_scan)
        errG = CD_LOSS + 0.1 * self.chamfer_distance(pred_coarses, full_scan_coarse)
        return self.weight * errG, CD_LOSS.item()

@LOSSES.register_module
class ONet_Loss(BaseLoss):
    def __call__(self, value):
        completion_loss = torch.mean(value[:,0])
        mask_loss = torch.mean(value[:,1])
        total_loss = self.weight * (completion_loss + 100*mask_loss)
        return {'total_loss': total_loss,
                'completion_loss': completion_loss.item(),
                'mask_loss': mask_loss.item()}

def compute_objectness_loss_boxnet(est_data, gt_data):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = est_data['aggregated_vote_xyz']
    gt_center = gt_data['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # NOTE: Different from VoteNet, here we use seed label as objectness label.
    seed_inds = est_data['seed_inds'].long() # B,num_seed in [0,num_points-1]
    seed_gt_votes_mask = torch.gather(gt_data['vote_label_mask'], 1, seed_inds)
    est_data['seed_labels'] = seed_gt_votes_mask
    aggregated_vote_inds = est_data['aggregated_vote_inds']
    objectness_label = torch.gather(est_data['seed_labels'], 1, aggregated_vote_inds.long()) # select (B,K) from (B,1024)
    objectness_mask = torch.ones((objectness_label.shape[0], objectness_label.shape[1])).cuda() # no ignore zone anymore

    # Compute objectness loss
    objectness_scores = est_data['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

@LOSSES.register_module
class BoxNetDetectionLoss(BaseLoss):
    def __call__(self, est_data, gt_data, dataset_config):
        """ Loss functions

        Args:
            end_points: dict
                {
                    seed_xyz, seed_inds,
                    center,
                    heading_scores, heading_residuals_normalized,
                    size_scores, size_residuals_normalized,
                    sem_cls_scores, #seed_logits,#
                    center_label,
                    heading_class_label, heading_residual_label,
                    size_class_label, size_residual_label,
                    sem_cls_label,
                    box_label_mask,
                    vote_label, vote_label_mask
                }
            config: dataset config instance
        Returns:
            loss: pytorch scalar tensor
            end_points: dict
        """

        # Obj loss
        objectness_loss, objectness_label, objectness_mask, object_assignment = \
            compute_objectness_loss_boxnet(est_data, gt_data)
        total_num_proposal = objectness_label.shape[0] * objectness_label.shape[1]
        pos_ratio = \
            torch.sum(objectness_label.float().cuda()) / float(total_num_proposal)
        neg_ratio = \
            torch.sum(objectness_mask.float()) / float(total_num_proposal) - pos_ratio

        # Box loss and sem cls loss
        meta_data = {'object_assignment':object_assignment,
                     'objectness_label':objectness_label}
        center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
            compute_box_and_sem_cls_loss(est_data, gt_data, meta_data, dataset_config)

        box_loss = center_loss + 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * size_cls_loss + size_reg_loss

        # Final loss function
        loss = 0.5 * objectness_loss + box_loss + 0.1 * sem_cls_loss
        loss *= 10

        # --------------------------------------------
        # Some other statistics
        obj_pred_val = torch.argmax(est_data['objectness_scores'], 2)  # B,K
        obj_acc = torch.sum((obj_pred_val == objectness_label.long()).float() * objectness_mask) / (
                    torch.sum(objectness_mask) + 1e-6)

        return {'total':loss,
                'objectness_loss': objectness_loss.item(),
                'box_loss': box_loss.item(),
                'sem_cls_loss': sem_cls_loss.item(),
                'pos_ratio': pos_ratio.item(),
                'neg_ratio': neg_ratio.item(),
                'center_loss': center_loss.item(),
                'heading_cls_loss': heading_cls_loss.item(),
                'heading_reg_loss': heading_reg_loss.item(),
                'size_cls_loss': size_cls_loss.item(),
                'size_reg_loss': size_reg_loss.item(),
                'obj_acc': obj_acc.item()}
