# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.registry import MODELS
from .two_stage_text_spotting import TwoStageTextSpotter


@MODELS.register_module()
class PANPP(TwoStageTextSpotter):
    """Attention based recognizer."""
