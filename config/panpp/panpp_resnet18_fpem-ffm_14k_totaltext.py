_base_ = [
    '_base_panpp_resnet18_fpem-ffm_english.py',
    '../_base_/datasets/totaltext.py',
    '../_base_/iteration_runtime.py',
    '../_base_/schedules/schedule_adam_300k.py',
]

# dataset settings
totaltext_textspotting_test = _base_.totaltext_textspotting_test
totaltext_textspotting_test.pipeline = _base_.test_pipeline
totaltext_textspotting_train = _base_.totaltext_textspotting_train
totaltext_textspotting_train.pipeline = _base_.train_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=totaltext_textspotting_train)

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

#load_from = ''  # load pretrained model from file
