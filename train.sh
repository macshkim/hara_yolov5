#!/bin/bash

python train.py --img 640 --batch 64 --epochs 30 --data ./data/fld.yaml --cfg ./models/yolov5s.yaml --weights '' --logdir /storage/sjiphrwm/models/yolov5/ --device 0 &> train_yolov5s_fld.log &