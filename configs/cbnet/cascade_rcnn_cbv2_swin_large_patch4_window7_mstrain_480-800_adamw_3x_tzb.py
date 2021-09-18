_base_ = [
    '../swin/cascade_rcnn_swin_large_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_tzb.py'
]

model = dict(
    backbone=dict(
        type='CBSwinTransformer',
    ),
    neck=dict(
        type='CBFPN',
    ),
    test_cfg = dict(
        rcnn=dict(
            # score_thr=0.001,
            nms=dict(type='nms'),
            # nms=dict(type='soft_nms'),
        )
    )
)
