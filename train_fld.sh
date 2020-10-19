#!/bin/bash

#python train.py --data ./data/fld.yaml --weights '' --cfg ./models/yolov5s.yaml --batch-size 64 --logdir ./runs/ --device 0 &> ./logs/train_yolov5s_fld.log &

python train.py --data ./data/fld.yaml --weights '' --cfg ./models/yolov5m.yaml --batch-size 40 --logdir ./runs/ --device 0 
