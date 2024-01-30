dictionary = dict(
    type='Dictionary',
    dict_file='{{ fileDirname }}/../../dicts/english_characters.txt',
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True
)

model = dict(
    type='PANPP',
    data_preprocessor=dict(
        type='TextDetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        stem_channels=128,
        deep_stem=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch'),
    neck=dict(type='FPEMv2_FFM', in_channels=[64, 128, 256, 512]),
    det_head=dict(
        type='PANHead',
        in_channels=[128, 128, 128, 128],
        hidden_dim=128,
        out_channel=6,
        module_loss=dict(
            type='PANModuleLoss',
            loss_text=dict(type='MaskedSquareDiceLoss'),
            loss_kernel=dict(type='MaskedSquareDiceLoss'),
        ),
        postprocessor=dict(
            type='PANPPDetPostprocessor',
            text_repr_type='poly',
            rescale_fields=[])),
    roi_head=dict(
        type='RecRoIHead',
        roi_extractor=dict(
            type='MaskRoIExtractor',
            in_channels=[128, 128, 128, 128],
            output_size=(8, 32),
            out_channels=128),
        rec_head=dict(
            type='PANPPRec',
            decoder=dict(
                type='PANPPRecDecoder',
                dictionary=dictionary,
                postprocessor=dict(
                    type='AttentionPostprocessor',
                    ignore_chars=['padding', 'unknown']),
                module_loss=dict(
                    type='PANPPCEModuleLoss',
                    ignore_first_char=True,
                    ignore_char=-1,
                    reduction='none'),
                max_seq_len=32,
                lstm_num=2))),
    postprocessor=dict(type='PANPPPostprocessor')
)

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='ShortScaleAspectJitter',
        short_size=736,
        scale_divisor=1,
        ratio_range=(1.0, 1.0),
        aspect_ratio_range=(1.0, 1.0)),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True),
    dict(type='FixInvalidPolygon'),
    dict(
        type='PackTextDetInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='color_ignore_orientation'),
    dict(
        type='LoadOCRAnnotations',
        with_polygon=True,
        with_bbox=True,
        with_label=True,
        with_text=True,
    ),
    dict(type='FixInvalidPolygon'),
    dict(type='ShortScaleAspectJitter', short_size=736, scale_divisor=32),
    dict(type='E2ERandomRotate', max_angle=10),
    dict(type='E2ETextDetRandomCrop', target_size=(736, 736)),
    dict(type='Pad', size=(736, 736)),
    dict(
        type='TorchVisionWrapper',
        op='ColorJitter',
        brightness=32.0 / 255,
        saturation=0.5),
    dict(type='E2ERemoveIllegalSample'),
    dict(
        type='E2EPackTextInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
