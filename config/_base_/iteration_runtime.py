_base_ = 'epoch_runtime.py'

_base_.default_hooks.checkpoint = dict(
        type='CheckpointHook',
        interval=1000,
        by_epoch=False,
        max_keep_ckpts=1
)

_base_.log_processor.by_epoch = False