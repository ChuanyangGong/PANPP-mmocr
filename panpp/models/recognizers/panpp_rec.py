# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.textrecog import EncoderDecoderRecognizer
from mmocr.registry import MODELS


@MODELS.register_module()
class PANPPRec(EncoderDecoderRecognizer):
    """CTC-loss based recognizer."""
