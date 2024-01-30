import os.path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from shapely.geometry import Polygon

from mmocr.registry import METRICS
from mmocr.utils import poly_intersection, poly_iou, polys2shapely, list_from_file
from rapidfuzz.distance import Levenshtein


@METRICS.register_module()
class E2EOneMinusNEDIOUMetric(BaseMetric):
    # TODO docstring
    """OneMinusNED with IOU metric.

    Args:
        ignored_characters_file: A txt file containing all the characters that
            need to be ignored, one character per line. Defaults to None.
        reserved_characters_file: A txt file containing all the characters that
            need to be retained, one character per line. Ignore characters not
            contained in this file when calculating edit distance. Defaults to
            None.
        match_iou_thr (float): IoU threshold for a match. Defaults to 0.5.
        ignore_precision_thr (float): Precision threshold when prediction and\
            gt ignored polygons are matched. Defaults to 0.5.
        pred_score_thrs (dict): Best prediction score threshold searching
            space. Defaults to dict(start=0.3, stop=0.9, step=0.1).
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None
    """
    default_prefix: Optional[str] = 'e2e_1-NED'

    def __init__(self,
                 ignored_characters_file: str = None,
                 reserved_characters_file: str = None,
                 match_iou_thr: float = 0.5,
                 ignore_precision_thr: float = 0.5,
                 pred_score_thrs: Dict = dict(start=0.3, stop=0.9, step=0.1),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ignored_characters = self._get_character_list(ignored_characters_file)
        self.reserved_characters = self._get_character_list(reserved_characters_file)
        self.match_iou_thr = match_iou_thr
        self.ignore_precision_thr = ignore_precision_thr
        self.pred_score_thrs = np.arange(**pred_score_thrs)

    def process(self, data_batch: Sequence[Dict],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[Dict]): A batch of data from dataloader.
            data_samples (Sequence[Dict]): A batch of outputs from
                the model.
        """
        for data_sample in data_samples:

            pred_instances = data_sample.get('pred_instances')
            pred_polygons = pred_instances.get('polygons')
            pred_scores = pred_instances.get('scores')
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            pred_scores = np.array(pred_scores, dtype=np.float32)
            pred_texts = pred_instances.get('texts')

            gt_instances = data_sample.get('gt_instances')
            gt_polys = gt_instances.get('polygons')
            gt_ignore_flags = gt_instances.get('ignored')
            gt_texts = gt_instances.get('texts')
            if isinstance(gt_ignore_flags, torch.Tensor):
                gt_ignore_flags = gt_ignore_flags.cpu().numpy()
            gt_polys = polys2shapely(gt_polys)
            pred_polys = polys2shapely(pred_polygons)
            pred_ignore_flags = self._filter_preds(pred_polys, gt_polys,
                                                   pred_scores,
                                                   gt_ignore_flags)
            pred_indexes = self._true_indexes(~pred_ignore_flags)
            gt_indexes = self._true_indexes(~gt_ignore_flags)
            pred_texts = [pred_texts[i] for i in pred_indexes]
            gt_texts = [gt_texts[i] for i in gt_indexes]

            gt_num = np.sum(~gt_ignore_flags)
            pred_num = np.sum(~pred_ignore_flags)
            iou_metric = np.zeros([gt_num, pred_num])

            # Compute IoU scores amongst kept pred and gt polygons
            for pred_mat_id, pred_poly_id in enumerate(pred_indexes):
                for gt_mat_id, gt_poly_id in enumerate(gt_indexes):
                    iou_metric[gt_mat_id, pred_mat_id] = poly_iou(
                        gt_polys[gt_poly_id], pred_polys[pred_poly_id])

            # Todo: remove careless characters in gt_texts & pred_texts
            gt_texts = [self._remove_special_character(gt_text) for gt_text in gt_texts]
            pred_texts = [self._remove_special_character(gt_text) for gt_text in pred_texts]
            result = dict(
                gt_texts=gt_texts,
                pred_texts=pred_texts,
                iou_metric=iou_metric,
                pred_scores=pred_scores[~pred_ignore_flags])
            self.results.append(result)

    def compute_metrics(self, results: List[Dict]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list[dict]): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        best_eval_results = dict(oneMinusNED=-1)
        logger.info('Evaluating e2e-1-NED-iou...')

        overall_metrics = []  # store every matched texts' edit distance
        for i, pred_score_thr in enumerate(self.pred_score_thrs):
            norm_eds = []
            for result in results:
                iou_metric = np.array(result['iou_metric'])  # (gt_num, pred_num)
                pred_scores = np.array(result['pred_scores'])  # (pred_num)
                gt_texts = result['gt_texts']
                pred_texts = result['pred_texts']

                pred_ignore_flags = pred_scores < pred_score_thr
                filtered_pred_texts = [
                    pred_texts[j]
                    for j in self._true_indexes(~pred_ignore_flags)
                ]
                csr_matched_metric = iou_metric[:, ~pred_ignore_flags]
                csr_matched_metric[csr_matched_metric < self.match_iou_thr] = 0.0

                used_pred_text_idxs = set()
                for gt_idx, gt_preds_iou in enumerate(csr_matched_metric):
                    pred_text = None
                    gt_text = gt_texts[gt_idx]
                    sortedIdx = np.argsort(-gt_preds_iou)
                    for pred_idx in sortedIdx:
                        if pred_idx in used_pred_text_idxs:
                            continue
                        if gt_preds_iou[pred_idx] < self.match_iou_thr:
                            break
                        pred_text = filtered_pred_texts[pred_idx]
                        used_pred_text_idxs.add(pred_idx)

                    if pred_text is None:  # Prediction not matched
                        norm_ed = 1
                    else:
                        norm_ed = Levenshtein.normalized_distance(pred_text, gt_text)
                    norm_eds.append(norm_ed)

                for pred_idx in range(len(pred_texts)): # Ground Truth not matched
                    if pred_idx not in used_pred_text_idxs:
                        norm_ed = 1
                        norm_eds.append(norm_ed)
            overall_metrics.append(1 - np.sum(norm_eds) / max(len(norm_eds), 1))

        for i, pred_score_thr in enumerate(self.pred_score_thrs):
            eval_results = dict(oneMinusNED=overall_metrics[i])
            logger.info(f'prediction score threshold: {pred_score_thr:.2f}, '
                        f'1-NED: {eval_results["oneMinusNED"]:.4f}\n')
            if eval_results['oneMinusNED'] > best_eval_results['oneMinusNED']:
                best_eval_results = eval_results
        return best_eval_results

    def _filter_preds(self, pred_polys: List[Polygon], gt_polys: List[Polygon],
                      pred_scores: List[float],
                      gt_ignore_flags: np.ndarray) -> np.ndarray:
        """Filter out the predictions by score threshold and whether it
        overlaps ignored gt polygons.

        Args:
            pred_polys (list[Polygon]): Pred polygons.
            gt_polys (list[Polygon]): GT polygons.
            pred_scores (list[float]): Pred scores of polygons.
            gt_ignore_flags (np.ndarray): 1D boolean array indicating
                the positions of ignored gt polygons.

        Returns:
            np.ndarray: 1D boolean array indicating the positions of ignored
            pred polygons.
        """

        # Filter out predictions based on the minimum score threshold
        pred_ignore_flags = pred_scores < self.pred_score_thrs.min()
        pred_indexes = self._true_indexes(~pred_ignore_flags)
        gt_indexes = self._true_indexes(gt_ignore_flags)
        # Filter out pred polygons which overlaps any ignored gt polygons
        for pred_id in pred_indexes:
            for gt_id in gt_indexes:
                # Match pred with ignored gt
                precision = poly_intersection(
                    gt_polys[gt_id], pred_polys[pred_id]) / (
                        pred_polys[pred_id].area + 1e-5)
                if precision > self.ignore_precision_thr:
                    pred_ignore_flags[pred_id] = True
                    break

        return pred_ignore_flags

    def _true_indexes(self, array: np.ndarray) -> np.ndarray:
        """Get indexes of True elements from a 1D boolean array."""
        return np.where(array)[0]

    def _get_character_list(self, dict_file):
        _dict = set() if dict_file is not None and os.path.exists(dict_file) else None
        if _dict is not None:
            for line_num, line in enumerate(list_from_file(dict_file)):
                line = line.strip('\r\n')
                if len(line) > 1:
                    raise ValueError('Expect each line has 0 or 1 character, '
                                     f'got {len(line)} characters '
                                     f'at line {line_num + 1}')
                if line != '':
                    _dict.add(line)
        return _dict

    def _remove_special_character(self, _str):
        """remove special character and replace unknown character to $"""
        if self.ignored_characters is None and self.reserved_characters is None:
            return _str

        _str_list = [c for c in _str]
        for idx, c in enumerate(_str_list):
            if self.ignored_characters is not None and c in self.ignored_characters:
                _str_list[idx] = ""
            elif self.reserved_characters is not None and c not in self.reserved_characters:
                _str_list[idx] = ""
        return "".join(_str_list)
