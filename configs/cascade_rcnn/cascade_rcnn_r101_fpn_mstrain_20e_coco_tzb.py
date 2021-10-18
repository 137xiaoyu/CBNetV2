_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_tzb.py',
    '../_base_/datasets/coco_detection_mstrain_tzb.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

# learning policy
lr_config = dict(step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=200)

optimizer = dict(type='SGD', lr=0.0025)

log_config = dict(interval=50)

checkpoint_config = dict(interval=5)

evaluation = dict(interval=50,
                  metric='bbox',
                  metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
                                'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000'])
