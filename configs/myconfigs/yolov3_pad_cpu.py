_base_ = ['../yolo/yolov3_mobilenetv2_320_300e_coco.py']

optimizer_config = dict(_delete_=True)

options_cfg = dict(
    randomSeed=42,
    partialsType='half',
    enableExecutableCaching='engine_cache',
    train_cfg=dict(
        executionStrategy='SameAsIpu',
        Training=dict(gradientAccumulation=8),
        availableMemoryProportion=[0.3, 0.3, 0.3, 0.3],
    ),
    eval_cfg=dict(deviceIterations=1, ),
)

ipu_model_cfg = dict(
    train_split_edges=[
        # dict(layer_to_call='backbone.conv2', ipu_id=0),
        dict(layer_to_call='backbone.layer3', ipu_id=1),
        dict(layer_to_call='backbone.layer5', ipu_id=2),
        dict(layer_to_call='backbone.layer7', ipu_id=3)
    ])

# runner = dict(type='IPUEpochBasedRunner',
#               ipu_model_cfg=ipu_model_cfg,
#               options_cfg=options_cfg,)

# fp16 = dict(loss_scale=256.0, velocity_accum_type='half', accum_type='half')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='IPUFormatBundle',
         pad_dic=dict(gt_bboxes=dict(dim=0,num=50),
                      gt_labels=dict(dim=0,num=50),
                      gt_bboxes_ignore=dict(dim=0,num=20))),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=24,
    train=dict(
        dataset=dict(
            pipeline=train_pipeline
        )
    ))

model = dict(
    bbox_head=dict(
        static=True,),
    train_cfg=dict(
        assigner=dict(
            static=True,
        ),
    )
)
