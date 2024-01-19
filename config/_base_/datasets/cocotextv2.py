cocotextv2_textspotting_data_root = 'data/cocotextv2'

cocotextv2_textspotting_train = dict(
    type='OCRDataset',
    data_root=cocotextv2_textspotting_data_root,
    ann_file='textspotting_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

cocotextv2_textspotting_test = dict(
    type='OCRDataset',
    data_root=cocotextv2_textspotting_data_root,
    ann_file='textspotting_val.json',
    test_mode=True,
    pipeline=None)
