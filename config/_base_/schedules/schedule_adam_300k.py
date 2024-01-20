# optimizer
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='Adam', lr=1e-3))
train_cfg = dict(type='IterBasedTrainLoop', max_iters=300000, val_interval=10000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning rate
param_scheduler = [
    dict(type='PolyLR', power=0.9, by_epoch=False, end=300000),
]
