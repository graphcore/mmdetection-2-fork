# python3 tools/train.py configs/myconfigs/detr_ipu_debug.py --ipu-replicas 1
# python3 tools/train.py configs/myconfigs/ssd_ipu.py --ipu-replicas 1
python3 tools/train.py configs/myconfigs/yolov3_ipu.py --ipu-replicas 1
# python3 tools/train.py configs/myconfigs/yolov3_cpu.py
