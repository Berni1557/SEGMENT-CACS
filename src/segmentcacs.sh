#!/bin/sh

python segment_cacs_predict.py \
    -m ../model/SegmentCACS_0001619_unet.pt \
    -d ../data \
    -p ../prediction \
    -f mhd \
    -gpu cuda \

