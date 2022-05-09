_base_ = '../centernet/centernet_resnet18_dcnv2_140e_coco.py'

model = dict(neck=dict(use_dcn=False))

runner = dict(type='IPUEpochBasedRunner', max_epochs=28)

optimizer_config = dict(_delete_=True)
