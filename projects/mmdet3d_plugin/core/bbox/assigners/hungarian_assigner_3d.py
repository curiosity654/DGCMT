# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.models.utils.transformer import inverse_sigmoid
from scipy.optimize import linear_sum_assignment

from projects.mmdet3d_plugin.core.bbox.util import (
    normalize_bbox, 
    denormalize_bbox
)


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None,
                 code_weights=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range
        self.code_weights = code_weights
        if self.code_weights:
            self.code_weights = torch.tensor(self.code_weights)[None, :].cuda()

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes, 
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7,
               code_weights=None):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)

        if self.code_weights is not None:
            bbox_pred = bbox_pred * self.code_weights
            normalized_gt_bboxes = normalized_gt_bboxes * self.code_weights
        
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
      
        # weighted sum of above two costs
        cost = cls_cost + reg_cost
        
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3DV3(BaseAssigner):
    """Hungarian Assigner for 3D bounding boxes with IoU calculation.
    
    This version computes IoU between predictions and ground truths,
    which is useful for monitoring matching quality.
    
    Args:
        cls_cost (dict): Classification cost config.
        reg_cost (dict): Regression cost config.
        iou_cost (dict): IoU cost config.
        iou_calculator (dict): IoU calculator config.
        pc_range (list): Point cloud range.
    """
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBox3DL1Cost', weight=1.0),
                 iou_cost=dict(type='IoU3DCost', weight=0.0),
                 iou_calculator=dict(type='BboxOverlaps3D'),
                 pc_range=None,
                 code_weights=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.pc_range = pc_range
        self.code_weights = code_weights
        if self.code_weights:
            self.code_weights = torch.tensor(self.code_weights)[None, :].cuda()
        
    def assign(self,
               bboxes,
               cls_pred,
               gt_bboxes, 
               gt_labels, 
               gt_bboxes_ignore=None,
               code_weights=None,
               with_velo=False):
        """Assign gt to bboxes.
        
        Args:
            bboxes (Tensor): Predicted bboxes (normalized), shape [num_query, ...]
            cls_pred (Tensor): Predicted classification, shape [num_query, num_classes]
            gt_bboxes (Tensor): Ground truth bboxes, shape [num_gt, ...]
            gt_labels (Tensor): Ground truth labels, shape [num_gt]
            gt_bboxes_ignore (Tensor, optional): Ignored gt bboxes.
            code_weights (Tensor, optional): Code weights for bbox regression.
            with_velo (bool): Whether predictions include velocity.
        
        Returns:
            AssignResult: Assignment result with max_overlaps containing IoU values.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bboxes.new_full((num_bboxes,),
                                           -1,
                                           dtype=torch.long)
        assigned_labels = bboxes.new_full((num_bboxes,),
                                          -1,
                                          dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        # Normalize gt_bboxes
        gt_bboxes_normalized = normalize_bbox(gt_bboxes, self.pc_range)
        
        # Apply code_weights if provided (use instance parameter or init parameter)
        _code_weights = code_weights if code_weights is not None else self.code_weights
        bboxes_weighted = bboxes
        gt_bboxes_normalized_weighted = gt_bboxes_normalized
        if _code_weights is not None:
            bboxes_weighted = bboxes * _code_weights
            gt_bboxes_normalized_weighted = gt_bboxes_normalized * _code_weights

        # Compute regression cost (only pass first 8 dimensions like original HungarianAssigner3D)
        reg_cost = self.reg_cost(bboxes_weighted[:, :8], gt_bboxes_normalized_weighted[:, :8])

        # Denormalize predictions for IoU calculation
        bboxes_denormalized = denormalize_bbox(bboxes, self.pc_range)

        # Compute IoU
        iou = self.iou_calculator(bboxes_denormalized, gt_bboxes)
        iou_cost = self.iou_cost(iou)

        # Weighted sum of costs
        cost = cls_cost + reg_cost + iou_cost
        pairwise_cost = cost.clone()

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        try:
            cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
            matched_row_inds = torch.from_numpy(matched_row_inds).to(bboxes.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(bboxes.device)
        except Exception as e:
            print(f"Hungarian matching failed: {e}")
            matched_row_inds = torch.tensor([], dtype=torch.long, device=bboxes.device)
            matched_col_inds = torch.tensor([], dtype=torch.long, device=bboxes.device)

        # 4. assign backgrounds and foregrounds
        assigned_gt_inds[:] = 0
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        # Store IoU values for matched queries
        max_overlaps = torch.zeros_like(iou.max(1).values)
        max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
        
        assign_result = AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        assign_result.pairwise_cost = pairwise_cost
        return assign_result

