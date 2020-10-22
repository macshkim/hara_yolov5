#!/bin/bash

python train.py --data ./data/fld.yaml --weights '' --cfg ./models/yolov5s.yaml --epochs 160 --batch-size 64 --logdir ./runs/ --device 0 &> ./logs/train_yolov5s_fld_v2.log &

#python train.py --data ./data/fld.yaml --weights '' --cfg ./models/yolov5m.yaml --epochs 160 --batch-size 40 --logdir ./runs/ --device 0 &> ./logs/train_yolov5m_fld.log &
