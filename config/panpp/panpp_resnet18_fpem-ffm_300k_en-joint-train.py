"""
This config is jointly trained on 5 datasets:
    synthtext : ctw1500 : totaltext : icdar2015 : cocotextv2 = 1 : 1 : 1 : 1 : 1
Test on Total-Text
"""
_base_ = [
    '_base_panpp_resnet18_fpem-ffm_english.py',
    '../_base_/datasets/synthtext.py',
    '../_base_/datasets/ctw1500.py',
    '../_base_/datasets/totaltext.py',
    '../_base_/datasets/icdar2015.py',
    '../_base_/datasets/cocotextv2.py',
    '../_base_/iteration_runtime.py',
    '../_base_/schedules/schedule_adam_300k.py',
]

# dataset settings
totaltext_textspotting_test = _base_.totaltext_textspotting_test
totaltext_textspotting_test.pipeline = _base_.test_pipeline

# multi dataset
train_list = [
    _base_.synthtext,  # 858,750 train image
    _base_.ctw1500,    # 1,000 train image
    _base_.totaltext,  # 1,255 train image
    _base_.icdar2015,  # 1,000 train image
    _base_.cocotextv2, # 19,039 train image
]

train_list = [
    dict(
        type='ConcatDataset',
        datasets=train_list[:1],
        pipeline=_base_.train_pipeline),
    dict(
        type='RepeatDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=train_list[1:2],
            pipeline=_base_.train_pipeline),
        times=859),
    dict(
        type='RepeatDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=train_list[2:3],
            pipeline=_base_.train_pipeline),
        times=684),
    dict(
        type='RepeatDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=train_list[3:4],
            pipeline=_base_.train_pipeline),
        times=859),
    dict(
        type='RepeatDataset',
        dataset=dict(
            type='ConcatDataset',
            datasets=train_list[4:5],
            pipeline=_base_.train_pipeline),
        times=45),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='ConcatDataset', datasets=train_list, verify_meta=False))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=totaltext_textspotting_test)

test_dataloader = val_dataloader

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

auto_scale_lr = dict(base_batch_size=16)

custom_imports = dict(imports=['panpp'], allow_failed_imports=False)
