_base_ = [
    '../_base_/models/cascade_mask_rcnn_r50_fpn_cell.py',
    '../_base_/datasets/coco_instance_cell.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2)

optimizer = dict(type='SGD', lr=0.0025)

log_config = dict(interval=50)

checkpoint_config = dict(interval=5)

evaluation = dict(interval=5)
