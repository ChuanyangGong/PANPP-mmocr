# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
import cv2
from mmengine.structures import InstanceData
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from mmocr.registry import MODELS
from mmocr.utils import ConfigType, OptMultiConfig, check_argument
from mmengine.model import BaseModule


@MODELS.register_module()
class MaskRoIExtractor(BaseModule):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (:obj:`ConfigDict` or dict): Specify RoI layer type and
            arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
            Defaults to 56.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 output_size: List[int] = [8, 32],
                 scale_factor: int = 4,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        assert check_argument.is_type_list(in_channels, int)
        assert isinstance(out_channels, int)

        in_channels = sum(in_channels)
        self.output_size = output_size
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.scale_factor = scale_factor

    def _upsample(self, x, output_size):
        return F.interpolate(x, size=output_size, mode='bilinear')

    def _make_diff(self, lower, upper, maximum):
        assert maximum > 0
        upper = min(upper, maximum)
        lower = max(0, min(lower, upper))
        if lower == upper:
            if upper == maximum:
                lower = maximum - 1
            else:
                upper += 1
        return lower, upper

    def forward(self, feats: Tuple[Tensor],
                proposal_instances: List[InstanceData]) -> Tensor:
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            proposal_instances(List[InstanceData]): Proposal instances.

        Returns:
            Tensor: RoI feature.
        """

        if isinstance(feats, tuple):
            feats = torch.cat(feats, dim=1)
        feats = self.conv1(feats)
        feats = self.relu1(self.bn1(feats))
        image_size = np.array(feats.size()[2:], dtype=np.int32) * self.scale_factor
        feats = self._upsample(feats, image_size.tolist())

        batch_size, _, H, W = feats.size()
        polygons = [p_i.polygons for p_i in proposal_instances]

        out_size = self.output_size
        roi_feats_list = []
        for i in range(feats.size(0)):
            cur_feat = feats[i]
            for idx, polygon in enumerate(polygons[i]):
                # prepossess polygon
                polygon = np.round(polygon).reshape(-1, 2).astype(np.int32)

                # get min rectangle
                l, t = polygon.min(axis=0) + \
                       np.array([-1, -1]) + np.random.randint(-1, 2)
                r, b = polygon.max(axis=0) + \
                       np.array([1, 1]) + np.random.randint(-1, 2)  # [] => [)
                # avoid t == l or b == r
                t, b = self._make_diff(t, b, H)
                l, r = self._make_diff(l, r, W)

                # make text mask
                mask = np.zeros((b-t, r-l), dtype=np.uint8)
                polygon = polygon - np.array([l, t])
                mask = cv2.drawContours(mask, [polygon], -1, 1, thickness=cv2.FILLED)
                mask = torch.from_numpy(mask).to(cur_feat.device).float()

                mask = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0), kernel_size=(3, 3),
                                    stride=1, padding=1)[0, 0]

                feat_crop = cur_feat[:, t:b, l:r] * mask
                _, h, w = feat_crop.size()
                if h > w * 1.5:
                    feat_crop = feat_crop.transpose(1, 2)

                feat_crop = F.interpolate(feat_crop.unsqueeze(0), out_size,
                                       mode='bilinear')
                roi_feats_list.append(feat_crop)
        roi_feats = feats.new_zeros((0,)) if len(roi_feats_list) == 0 else torch.cat(roi_feats_list)
        return roi_feats
