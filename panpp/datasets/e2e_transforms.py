# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Tuple, Optional, List, Union, Any, Sequence

import random
import imgaug
import imgaug.augmenters as iaa
from shapely.geometry import Polygon as sPolygon
import cv2
import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmocr.utils import crop_polygon, poly2bbox, poly_intersection
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmcv.transforms import to_tensor

from mmocr.registry import TRANSFORMS
from mmocr.structures import (TextDetDataSample)
from mmengine.structures import InstanceData


@TRANSFORMS.register_module()
class E2ERemoveIllegalSample(BaseTransform):
    def transform(self, results: Dict) -> Dict:
        hasValid = ~results['gt_ignored']
        if 'gt_text_ignore' in results:
            hasValid = hasValid & ~results['gt_text_ignore']
        if hasValid.sum() == 0:
            return None
        return results


@TRANSFORMS.register_module()
class E2ERandomRotate(BaseTransform):
    """Randomly rotate the image, boxes, and polygons. For recognition task,
    only the image will be rotated. If set ``use_canvas`` as True, the shape of
    rotated image might be modified based on the rotated angle size, otherwise,
    the image will keep the shape before rotation.

    Required Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_polygons (optional)

    Modified Keys:

    - img
    - img_shape (optional)
    - gt_bboxes (optional)
    - gt_polygons (optional)

    Added Keys:

    - rotated_angle

    Args:
        max_angle (int): The maximum rotation angle (can be bigger than 180 or
            a negative). Defaults to 10.
        pad_with_fixed_color (bool): The flag for whether to pad rotated
            image with fixed value. Defaults to False.
        pad_value (tuple[int, int, int]): The color value for padding rotated
            image. Defaults to (0, 0, 0).
        use_canvas (bool): Whether to create a canvas for rotated image.
            Defaults to False. If set true, the image shape may be modified.
    """

    def __init__(
        self,
        max_angle: int = 10,
        pad_with_fixed_color: bool = False,
        pad_value: Tuple[int, int, int] = (0, 0, 0),
        use_canvas: bool = False,
    ) -> None:
        if not isinstance(max_angle, int):
            raise TypeError('`max_angle` should be an integer'
                            f', but got {type(max_angle)} instead')
        if not isinstance(pad_with_fixed_color, bool):
            raise TypeError('`pad_with_fixed_color` should be a bool, '
                            f'but got {type(pad_with_fixed_color)} instead')
        if not isinstance(pad_value, (list, tuple)):
            raise TypeError('`pad_value` should be a list or tuple, '
                            f'but got {type(pad_value)} instead')
        if len(pad_value) != 3:
            raise ValueError('`pad_value` should contain three integers')
        if not isinstance(pad_value[0], int) or not isinstance(
                pad_value[1], int) or not isinstance(pad_value[2], int):
            raise ValueError('`pad_value` should contain three integers')

        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value
        self.use_canvas = use_canvas

    @cache_randomness
    def _sample_angle(self, max_angle: int) -> float:
        """Sampling a random angle for rotation.

        Args:
            max_angle (int): Maximum rotation angle

        Returns:
            float: The random angle used for rotation
        """
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    @staticmethod
    def _cal_canvas_size(ori_size: Tuple[int, int],
                         degree: int) -> Tuple[int, int]:
        """Calculate the canvas size.

        Args:
            ori_size (Tuple[int, int]): The original image size (height, width)
            degree (int): The rotation angle

        Returns:
            Tuple[int, int]: The size of the canvas
        """
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    @staticmethod
    def _rotate_points(center: Tuple[float, float],
                       points: np.array,
                       theta: float,
                       center_shift: Tuple[int, int] = (0, 0)) -> np.array:
        """Rotating a set of points according to the given theta.

        Args:
            center (Tuple[float, float]): The coordinate of the canvas center
            points (np.array): A set of points needed to be rotated
            theta (float): Rotation angle
            center_shift (Tuple[int, int]): The shifting offset of the center
                coordinate

        Returns:
            np.array: The rotated coordinates of the input points
        """
        (center_x, center_y) = center
        center_y = -center_y
        x, y = points[::2], points[1::2]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[::2], points[1::2] = _x, _y
        return points

    def _rotate_img(self, results: Dict) -> Tuple[int, int]:
        """Rotating the input image based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            Tuple[int, int]: The shifting offset of the center point.
        """
        if results.get('img', None) is not None:
            h = results['img'].shape[0]
            w = results['img'].shape[1]
            rotation_matrix = cv2.getRotationMatrix2D(
                (w / 2, h / 2), results['rotated_angle'], 1)

            canvas_size = self._cal_canvas_size((h, w),
                                                results['rotated_angle'])
            center_shift = (int(
                (canvas_size[1] - w) / 2), int((canvas_size[0] - h) / 2))
            rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
            rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)
            if self.pad_with_fixed_color:
                rotated_img = cv2.warpAffine(
                    results['img'],
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    flags=cv2.INTER_NEAREST,
                    borderValue=self.pad_value)
            else:
                mask = np.zeros_like(results['img'])
                (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                                  np.random.randint(0, w * 7 // 8))
                img_cut = results['img'][h_ind:(h_ind + h // 9),
                                         w_ind:(w_ind + w // 9)]
                img_cut = mmcv.imresize(img_cut,
                                        (canvas_size[1], canvas_size[0]))
                mask = cv2.warpAffine(
                    mask,
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    borderValue=[1, 1, 1])
                rotated_img = cv2.warpAffine(
                    results['img'],
                    rotation_matrix, (canvas_size[1], canvas_size[0]),
                    borderValue=[0, 0, 0])
                rotated_img = rotated_img + img_cut * mask

            results['img'] = rotated_img
        else:
            raise ValueError('`img` is not found in results')

        return center_shift

    def _rotate_bboxes(self, results: Dict, center_shift: Tuple[int,
                                                                int]) -> None:
        """Rotating the bounding boxes based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point
        """
        if results.get('gt_bboxes', None) is not None:
            height, width = results['img_shape']
            box_list = []
            for box in results['gt_bboxes']:
                rotated_box = self._rotate_points((width / 2, height / 2),
                                                  bbox2poly(box),
                                                  results['rotated_angle'],
                                                  center_shift)
                rotated_box = poly2bbox(rotated_box)
                box_list.append(rotated_box)

            results['gt_bboxes'] = np.array(
                box_list, dtype=np.float32).reshape(-1, 4)

    def _rotate_polygons(self, results: Dict,
                         center_shift: Tuple[int, int]) -> None:
        """Rotating the polygons based on the given angle.

        Args:
            results (dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point
        """
        if results.get('gt_polygons', None) is not None:
            height, width = results['img_shape']
            polygon_list = []
            for poly in results['gt_polygons']:
                rotated_poly = self._rotate_points(
                    (width / 2, height / 2), poly, results['rotated_angle'],
                    center_shift)
                polygon_list.append(rotated_poly)
            results['gt_polygons'] = polygon_list

    def transform(self, results: Dict) -> Dict:
        """Applying random rotate on results.

        Args:
            results (Dict): Result dict containing the data to transform.
            center_shift (Tuple[int, int]): The shifting offset of the
                center point

        Returns:
            dict: The transformed data
        """
        # TODO rotate char_quads & char_rects for SegOCR
        if self.use_canvas:
            results['rotated_angle'] = self._sample_angle(self.max_angle)
            # rotate image
            center_shift = self._rotate_img(results)
            # rotate gt_bboxes
            self._rotate_bboxes(results, center_shift)
            # rotate gt_polygons
            self._rotate_polygons(results, center_shift)

            results['img_shape'] = (results['img'].shape[0],
                                    results['img'].shape[1])
        else:
            args = [
                dict(
                    cls='Affine',
                    rotate=[-self.max_angle, self.max_angle],
                    backend='cv2',
                    order=0)  # order=0 -> cv2.INTER_NEAREST
            ]
            imgaug_transform = E2EImgAugWrapper(args)
            results = imgaug_transform(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_angle = {self.max_angle}'
        repr_str += f', pad_with_fixed_color = {self.pad_with_fixed_color}'
        repr_str += f', pad_value = {self.pad_value}'
        repr_str += f', use_canvas = {self.use_canvas})'
        return repr_str


@TRANSFORMS.register_module()
class E2EImgAugWrapper(BaseTransform):
    """A wrapper around imgaug https://github.com/aleju/imgaug.

    Find available augmenters at
    https://imgaug.readthedocs.io/en/latest/source/overview_of_augmenters.html.

    Required Keys:

    - img
    - gt_polygons (optional for text recognition)
    - gt_bboxes (optional for text recognition)
    - gt_bboxes_labels (optional for text recognition)
    - gt_ignored (optional for text recognition)
    - gt_texts (optional)

    Modified Keys:

    - img
    - gt_polygons (optional for text recognition)
    - gt_bboxes (optional for text recognition)
    - gt_bboxes_labels (optional for text recognition)
    - gt_ignored (optional for text recognition)
    - img_shape (optional)
    - gt_texts (optional)

    Args:
        args (list[list or dict]], optional): The argumentation list. For
            details, please refer to imgaug document. Take
            args=[['Fliplr', 0.5], dict(cls='Affine', rotate=[-10, 10]),
            ['Resize', [0.5, 3.0]]] as an example. The args horizontally flip
            images with probability 0.5, followed by random rotation with
            angles in range [-10, 10], and resize with an independent scale in
            range [0.5, 3.0] for each side of images. Defaults to None.
        fix_poly_trans (dict): The transform configuration to fix invalid
            polygons. Set it to None if no fixing is needed.
            Defaults to dict(type='FixInvalidPolygon').
    """

    def __init__(
        self,
        args: Optional[List[Union[List, Dict]]] = None,
        fix_poly_trans: Optional[dict] = dict(type='FixInvalidPolygon')
    ) -> None:
        assert args is None or isinstance(args, list) and len(args) > 0
        if args is not None:
            for arg in args:
                assert isinstance(arg, (list, dict)), \
                    'args should be a list of list or dict'
        self.args = args
        self.augmenter = self._build_augmentation(args)
        self.fix_poly_trans = fix_poly_trans
        if fix_poly_trans is not None:
            self.fix = TRANSFORMS.build(fix_poly_trans)

    def transform(self, results: Dict) -> Dict:
        """Transform the image and annotation data.

        Args:
            results (dict): Result dict containing the data to transform.

        Returns:
            dict: The transformed data.
        """
        # img is bgr
        image = results['img']
        aug = None
        ori_shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            if not self._augment_annotations(aug, ori_shape, results):
                return None
            results['img'] = aug.augment_image(image)
            results['img_shape'] = (results['img'].shape[0],
                                    results['img'].shape[1])
        if getattr(self, 'fix', None) is not None:
            results = self.fix(results)
        return results

    def _augment_annotations(self, aug: imgaug.augmenters.meta.Augmenter,
                             ori_shape: Tuple[int,
                                              int], results: Dict) -> Dict:
        """Augment annotations following the pre-defined augmentation sequence.

        Args:
            aug (imgaug.augmenters.meta.Augmenter): The imgaug augmenter.
            ori_shape (tuple[int, int]): The ori_shape of the original image.
            results (dict): Result dict containing annotations to transform.

        Returns:
            bool: Whether the transformation has been successfully applied. If
            the transform results in empty polygon/bbox annotations, return
            False.
        """
        # Assume co-existence of `gt_polygons`, `gt_bboxes` and `gt_ignored`
        # for text detection
        if 'gt_polygons' in results:

            # augment polygons
            transformed_polygons, removed_poly_inds, ignore_poly_inds = self._augment_polygons(
                aug, ori_shape, results['gt_polygons'])
            if len(transformed_polygons) == 0:
                return False
            results['gt_polygons'] = transformed_polygons

            # remove instances that are no longer inside the augmented image
            if 'gt_text_ignore' not in results:
                results['gt_text_ignore'] = np.zeros(results['gt_ignored'].shape) == 1
            results['gt_text_ignore'][ignore_poly_inds] = True
            results['gt_bboxes_labels'] = np.delete(
                results['gt_bboxes_labels'], removed_poly_inds, axis=0)
            results['gt_ignored'] = np.delete(
                results['gt_ignored'], removed_poly_inds, axis=0)
            results['gt_text_ignore'] = np.delete(
                results['gt_text_ignore'], removed_poly_inds, axis=0)
            # TODO: deal with gt_texts corresponding to clipped polygons
            if 'gt_texts' in results:
                results['gt_texts'] = [
                    text for i, text in enumerate(results['gt_texts'])
                    if i not in removed_poly_inds
                ]

            # Generate new bboxes
            bboxes = [poly2bbox(poly) for poly in transformed_polygons]
            results['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
            if len(bboxes) > 0:
                results['gt_bboxes'] = np.stack(bboxes)

        return True

    def _augment_polygons(self, aug: imgaug.augmenters.meta.Augmenter,
                          ori_shape: Tuple[int, int], polys: List[np.ndarray]
                          ) -> Tuple[List[np.ndarray], List[int]]:
        """Augment polygons.

        Args:
            aug (imgaug.augmenters.meta.Augmenter): The imgaug augmenter.
            ori_shape (tuple[int, int]): The shape of the original image.
            polys (list[np.ndarray]): The polygons to be augmented.

        Returns:
            tuple(list[np.ndarray], list[int]): The augmented polygons, and the
            indices of polygons removed as they are out of the augmented image.
        """
        imgaug_polys = []
        for poly in polys:
            poly = poly.reshape(-1, 2)
            imgaug_polys.append(imgaug.Polygon(poly))
        imgaug_polys = aug.augment_polygons(
            [imgaug.PolygonsOnImage(imgaug_polys, shape=ori_shape)])[0]

        new_polys = []
        removed_poly_inds = []
        ignore_poly_inds = []
        for i, poly in enumerate(imgaug_polys.polygons):
            # Sometimes imgaug may produce some invalid polygons with no points
            if not poly.is_valid or poly.is_out_of_image(imgaug_polys.shape):
                removed_poly_inds.append(i)
                continue
            new_poly = []
            ori_area = poly.area
            try:
                poly = poly.clip_out_of_image(imgaug_polys.shape)[0]
            except Exception as e:
                warnings.warn(f'Failed to clip polygon out of image: {e}')
            # ignore most part out of image's instance
            if poly.area / ori_area < 0.9:
                ignore_poly_inds.append(i)
            for point in poly:
                new_poly.append(np.array(point, dtype=np.float32))
            new_poly = np.array(new_poly, dtype=np.float32).flatten()
            # Under some conditions, imgaug can generate "polygon" with only
            # two points, which is not a valid polygon.
            if len(new_poly) <= 4:
                removed_poly_inds.append(i)
                continue
            new_polys.append(new_poly)

        return new_polys, removed_poly_inds, ignore_poly_inds

    def _build_augmentation(self, args, root=True):
        """Build ImgAugWrapper augmentations.

        Args:
            args (dict): Arguments to be passed to imgaug.
            root (bool): Whether it's building the root augmenter.

        Returns:
            imgaug.augmenters.meta.Augmenter: The built augmenter.
        """
        if args is None:
            return None
        if isinstance(args, (int, float, str)):
            return args
        if isinstance(args, list):
            if root:
                sequence = [
                    self._build_augmentation(value, root=False)
                    for value in args
                ]
                return iaa.Sequential(sequence)
            arg_list = [self._to_tuple_if_list(a) for a in args[1:]]
            return getattr(iaa, args[0])(*arg_list)
        if isinstance(args, dict):
            if 'cls' in args:
                cls = getattr(iaa, args['cls'])
                return cls(
                    **{
                        k: self._to_tuple_if_list(v)
                        for k, v in args.items() if not k == 'cls'
                    })
            else:
                return {
                    key: self._build_augmentation(value, root=False)
                    for key, value in args.items()
                }
        raise RuntimeError('unknown augmenter arg: ' + str(args))

    def _to_tuple_if_list(self, obj: Any) -> Any:
        """Convert an object into a tuple if it is a list."""
        if isinstance(obj, list):
            return tuple(obj)
        return obj

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(args = {self.args}, '
        repr_str += f'fix_poly_trans = {self.fix_poly_trans})'
        return repr_str


@TRANSFORMS.register_module()
@avoid_cache_randomness
class E2ETextDetRandomCrop(BaseTransform):
    """Randomly select a region and crop images to a target size and make sure
    to contain text region. This transform may break up text instances, and for
    broken text instances, we will crop it's bbox and polygon coordinates. This
    transform is recommend to be used in segmentation-based network.

    Required Keys:

    - img
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Modified Keys:

    - img
    - img_shape
    - gt_polygons
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignored

    Args:
        target_size (tuple(int, int) or int): Target size for the cropped
            image. If it's a tuple, then target width and target height will be
            ``target_size[0]`` and ``target_size[1]``, respectively. If it's an
            integer, them both target width and target height will be
            ``target_size``.
        positive_sample_ratio (float): The probability of sampling regions
            that go through text regions. Defaults to 5. / 8.
    """

    def __init__(self,
                 target_size: Tuple[int, int] or int,
                 positive_sample_ratio: float = 5.0 / 8.0) -> None:
        self.target_size = target_size if isinstance(
            target_size, tuple) else (target_size, target_size)
        self.positive_sample_ratio = positive_sample_ratio

    def _get_postive_prob(self) -> float:
        """Get the probability to do positive sample.

        Returns:
            float: The probability to do positive sample.
        """
        return np.random.random_sample()

    def _sample_num(self, start, end):
        """Sample a number in range [start, end].

        Args:
            start (int): Starting point.
            end (int): Ending point.

        Returns:
            (int): Sampled number.
        """
        return random.randint(start, end)

    def _sample_offset(self, gt_polygons: Sequence[np.ndarray],
                       img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Samples the top-left coordinate of a crop region, ensuring that the
        cropped region contains at least one polygon.

        Args:
            gt_polygons (list(ndarray)) : Polygons.
            img_size (tuple(int, int)) : Image size in the format of
                (height, width).

        Returns:
            tuple(int, int): Top-left coordinate of the cropped region.
        """
        h, w = img_size
        t_w, t_h = self.target_size

        # target size is bigger than origin size
        t_h = t_h if t_h < h else h
        t_w = t_w if t_w < w else w
        if (gt_polygons is not None and len(gt_polygons) > 0
                and self._get_postive_prob() < self.positive_sample_ratio):

            # make sure to crop the positive region

            # the minimum top left to crop positive region (h,w)
            tl = np.array([h + 1, w + 1], dtype=np.int32)
            for gt_polygon in gt_polygons:
                temp_point = np.min(gt_polygon.reshape(2, -1), axis=1)
                if temp_point[0] <= tl[0]:
                    tl[0] = temp_point[0]
                if temp_point[1] <= tl[1]:
                    tl[1] = temp_point[1]
            tl = tl - (t_h, t_w)
            tl[tl < 0] = 0
            # the maximum bottum right to crop positive region
            br = np.array([0, 0], dtype=np.int32)
            for gt_polygon in gt_polygons:
                temp_point = np.max(gt_polygon.reshape(2, -1), axis=1)
                if temp_point[0] > br[0]:
                    br[0] = temp_point[0]
                if temp_point[1] > br[1]:
                    br[1] = temp_point[1]
            br = br - (t_h, t_w)
            br[br < 0] = 0

            # if br is too big so that crop the outside region of img
            br[0] = min(br[0], h - t_h)
            br[1] = min(br[1], w - t_w)
            #
            h = self._sample_num(tl[0], br[0]) if tl[0] < br[0] else 0
            w = self._sample_num(tl[1], br[1]) if tl[1] < br[1] else 0
        else:
            # make sure not to crop outside of img

            h = self._sample_num(0, h - t_h) if h - t_h > 0 else 0
            w = self._sample_num(0, w - t_w) if w - t_w > 0 else 0

        return (h, w)

    def _crop_img(self, img: np.ndarray, offset: Tuple[int, int],
                  target_size: Tuple[int, int]) -> np.ndarray:
        """Crop the image given an offset and a target size.

        Args:
            img (ndarray): Image.
            offset (Tuple[int. int]): Coordinates of the starting point.
            target_size: Target image size.
        """
        h, w = img.shape[:2]
        target_size = target_size[::-1]
        br = np.min(
            np.stack((np.array(offset) + np.array(target_size), np.array(
                (h, w)))),
            axis=0)
        return img[offset[0]:br[0], offset[1]:br[1]], np.array(
            [offset[1], offset[0], br[1], br[0]])

    def _crop_polygons(self, polygons: Sequence[np.ndarray],
                       crop_bbox: np.ndarray) -> Sequence[np.ndarray]:
        """Crop polygons to be within a crop region. If polygon crosses the
        crop_bbox, we will keep the part left in crop_bbox by cropping its
        boardline.

        Args:
            polygons (list(ndarray)): List of polygons [(N1, ), (N2, ), ...].
            crop_bbox (ndarray): Cropping region. [x1, y1, x2, y1].

        Returns
            tuple(List(ArrayLike), list[int]):
                - (List(ArrayLike)): The rest of the polygons located in the
                    crop region.
                - (list[int]): Index list of the reserved polygons.
        """
        polygons_cropped = []
        kept_idx = []
        ignore_idx = []
        for idx, polygon in enumerate(polygons):
            if polygon.size < 6:
                continue
            poly = crop_polygon(polygon, crop_bbox)
            if poly is not None:
                poly = poly.reshape(-1, 2) - (crop_bbox[0], crop_bbox[1])
                if sPolygon(poly.reshape(-1, 2)).area / sPolygon(polygon.reshape(-1, 2)).area < 0.9:
                    ignore_idx.append(idx)
                polygons_cropped.append(poly.reshape(-1))
                kept_idx.append(idx)
        return (polygons_cropped, kept_idx, ignore_idx)

    def transform(self, results: Dict) -> Dict:
        """Applying random crop on results.
        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The transformed data
        """
        if self.target_size == results['img'].shape[:2][::-1]:
            return results
        gt_polygons = results['gt_polygons']
        crop_offset = self._sample_offset(gt_polygons,
                                          results['img'].shape[:2])
        img, crop_bbox = self._crop_img(results['img'], crop_offset,
                                        self.target_size)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        gt_polygons, polygon_kept_idx, polygon_ignore_idx = self._crop_polygons(
            gt_polygons, crop_bbox)
        bboxes = [poly2bbox(poly) for poly in gt_polygons]
        results['gt_bboxes'] = np.array(
            bboxes, dtype=np.float32).reshape(-1, 4)

        results['gt_polygons'] = gt_polygons
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
            polygon_kept_idx]
        results['gt_ignored'] = results['gt_ignored'][polygon_kept_idx]
        if 'gt_text_ignore' not in results:
            results['gt_text_ignore'] = np.zeros(results['gt_ignored'].shape) == 1
        if 'gt_texts' in results:
            results['gt_texts'] = [
                text for i, text in enumerate(results['gt_texts'])
                if i in polygon_kept_idx
            ]
        results['gt_text_ignore'][polygon_ignore_idx] = True
        results['gt_text_ignore'] = results['gt_text_ignore'][polygon_kept_idx]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(target_size = {self.target_size}, '
        repr_str += f'positive_sample_ratio = {self.positive_sample_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class E2EPackTextInputs(BaseTransform):
    """Pack the inputs data for text detection.

    The type of outputs is `dict`:

    - inputs: image converted to tensor, whose shape is (C, H, W).
    - data_samples: Two components of ``TextDetDataSample`` will be updated:

      - gt_instances (InstanceData): Depending on annotations, a subset of the
        following keys will be updated:

        - bboxes (torch.Tensor((N, 4), dtype=torch.float32)): The groundtruth
          of bounding boxes in the form of [x1, y1, x2, y2]. Renamed from
          'gt_bboxes'.
        - labels (torch.LongTensor(N)): The labels of instances.
          Renamed from 'gt_bboxes_labels'.
        - polygons(list[np.array((2k,), dtype=np.float32)]): The
          groundtruth of polygons in the form of [x1, y1,..., xk, yk]. Each
          element in polygons may have different number of points. Renamed from
          'gt_polygons'. Using numpy instead of tensor is that polygon usually
          is not the output of model and operated on cpu.
        - ignored (torch.BoolTensor((N,))): The flag indicating whether the
          corresponding instance should be ignored. Renamed from
          'gt_ignored'.
        - texts (list[str]): The groundtruth texts. Renamed from 'gt_texts'.

      - metainfo (dict): 'metainfo' is always populated. The contents of the
        'metainfo' depends on ``meta_keys``. By default it includes:

        - "img_path": Path to the image file.
        - "img_shape": Shape of the image input to the network as a tuple
          (h, w). Note that the image may be zero-padded afterward on the
          bottom/right if the batch tensor is larger than this shape.
        - "scale_factor": A tuple indicating the ratio of width and height
          of the preprocessed image to the original one.
        - "ori_shape": Shape of the preprocessed image as a tuple
          (h, w).
        - "pad_shape": Image shape after padding (if any Pad-related
          transform involved) as a tuple (h, w).
        - "flip": A boolean indicating if the image has been flipped.
        - ``flip_direction``: the flipping direction.

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            the metainfo of ``TextDetSample``. Defaults to ``('img_path',
            'ori_shape', 'img_shape', 'scale_factor', 'flip',
            'flip_direction')``.
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_polygons': 'polygons',
        'gt_texts': 'texts',
        'gt_ignored': 'ignored',
        'gt_text_ignore': 'text_ignored'
    }

    def __init__(self,
                 meta_keys=('img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): Data for model forwarding.
            - 'data_samples' (obj:`DetDataSample`): The annotation info of the
              sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # A simple trick to speedup formatting by 3-5 times when
            # OMP_NUM_THREADS != 1
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if img.flags.c_contiguous:
                img = to_tensor(img)
                img = img.permute(2, 0, 1).contiguous()
            else:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            packed_results['inputs'] = img

        data_sample = TextDetDataSample()
        instance_data = InstanceData()
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key in ['gt_bboxes', 'gt_bboxes_labels', 'gt_ignored', 'gt_text_ignore']:
                instance_data[self.mapping_table[key]] = to_tensor(
                    results[key])
            else:
                instance_data[self.mapping_table[key]] = results[key]
        data_sample.gt_instances = instance_data

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

