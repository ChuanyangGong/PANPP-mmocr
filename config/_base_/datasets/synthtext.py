synthtext_textspotting_data_root = 'data/synthtext'

synthtext_textspotting_train = dict(
    type='OCRDataset',
    data_root=synthtext_textspotting_data_root,
    ann_file='textspotting_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)
