# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult, StaticSamplingResult


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, static=False, **kwargs):
        self.static = static

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        if self.static:
            return self.sample_statically(assign_result, bboxes, gt_bboxes,
                                          *args, **kwargs)
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(
            pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result

    def sample_statically(self, assign_result, bboxes, gt_bboxes,
                          *args, **kwargs):
        """Directly returns the padded positive and negative indices of samples.
        

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`StaticSamplingResult`: sampler results
        """
        # Suppose an gt bbox corresponds to at most two predicted bboxes
        # and we having N gt bboxes, then the number of pred_bboxes with
        # ground truth label will not surpass 2N, so we will pad pos_inds
        # to (2N,) and pad value will be -1
        # neg_inds will be padded to the number of bboxes
        N = gt_bboxes.shape[0]
        inds = torch.range(0, assign_result.gt_inds.shape[0]-1,
                           dtype=torch.long)
        pos_flags = (assign_result.gt_inds > 0).long()
        pos_inds = pos_flags * inds + (pos_flags - 1)
        pos_inds = torch.sort(pos_inds, descending=True).values[:2*N]
        gt_flags = pos_inds >= 0

        neg_inds = (assign_result.gt_inds == 0).long()
        neg_inds = neg_inds * inds + (neg_inds - 1)
        neg_inds = torch.sort(neg_inds, descending=True).values

        sampling_result = StaticSamplingResult(
            pos_inds, neg_inds, bboxes, gt_bboxes, assign_result, gt_flags)
        return sampling_result
