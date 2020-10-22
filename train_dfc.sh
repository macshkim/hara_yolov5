#!/bin/bash

python train.py --data ./data/deepfashion_c.yaml --weights '' --cfg ./models/yolov5s.yaml --epochs 160 --batch-size 64 --logdir /storage/sjiphrwm/models/yolov5/dfc/ --device 0

