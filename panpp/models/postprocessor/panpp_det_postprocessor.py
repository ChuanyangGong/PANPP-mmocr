from functools import partial
from typing import List, Sequence, Union

import cv2
import numpy as np
import torch
from torch import Tensor
from mmcv.ops import pixel_group
from mmengine.structures import InstanceData

from mmocr.registry import MODELS
from mmocr.structures import TextDetDataSample
from mmocr.models.textdet.postprocessors.pan_postprocessor import PANPostprocessor
from mmocr.utils import rescale_polygons


@MODELS.register_module()
class PANPPDetPostprocessor(PANPostprocessor):
    def get_text_instances(self, pred_results: torch.Tensor,
                           data_sample: TextDetDataSample,
                           **kwargs) -> TextDetDataSample:
        """Get text instance predictions of one image.

        Args:
            pred_result (torch.Tensor): Prediction results of an image which
                is a tensor of shape :math:`(N, H, W)`.
            data_sample (TextDetDataSample): Datasample of an image.

        Returns:
            TextDetDataSample: A new DataSample with predictions filled in.
            Polygons and results are saved in
            ``TextDetDataSample.pred_instances.polygons``. The confidence
            scores are saved in ``TextDetDataSample.pred_instances.scores``.
        """
        assert pred_results.dim() == 3

        pred_results[:2, :, :] = torch.sigmoid(pred_results[:2, :, :])
        pred_results = pred_results.detach().cpu().numpy()

        text_score = pred_results[0].astype(np.float32)
        text = pred_results[0] > self.min_text_confidence
        kernel = (pred_results[1] > self.min_kernel_confidence) * text
        embeddings = pred_results[2:] * text.astype(np.float32)
        embeddings = embeddings.transpose((1, 2, 0))  # (h, w, 4)

        region_num, labels = cv2.connectedComponents(
            kernel.astype(np.uint8), connectivity=4)
        contours, _ = cv2.findContours((kernel * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        kernel_contours = np.zeros(text.shape, dtype='uint8')
        cv2.drawContours(kernel_contours, contours, -1, 255)
        text_points = pixel_group(text_score, text, embeddings, labels,
                                  kernel_contours, region_num,
                                  self.distance_threshold)

        polygons = []
        scores = []
        for text_point in text_points:
            text_confidence = text_point[0]
            text_point = text_point[2:]
            text_point = np.array(text_point, dtype=int).reshape(-1, 2)
            area = text_point.shape[0]
            if (area < self.min_text_area
                    or text_confidence <= self.score_threshold):
                continue

            polygon = self._points2boundary(text_point)
            if len(polygon) > 0:
                polygons.append(polygon)
                scores.append(text_confidence)
        pred_instances = InstanceData()
        pred_instances.polygons = rescale_polygons(
            polygons,
            np.array([self.downsample_ratio, self.downsample_ratio]),
            mode='div'
        )
        pred_instances.scores = torch.FloatTensor(scores)
        data_sample.pred_instances = pred_instances
        return data_sample

    def __call__(self,
                 pred_results: Union[Tensor, List[Tensor]],
                 data_samples: Sequence[TextDetDataSample],
                 training: bool = False) -> Sequence[TextDetDataSample]:
        if training:
            return data_samples
        cfg = self.train_cfg if training else self.test_cfg
        if cfg is None:
            cfg = {}
        pred_results = self.split_results(pred_results)
        process_single = partial(self._process_single, **cfg)
        results = list(map(process_single, pred_results, data_samples))

        return results
