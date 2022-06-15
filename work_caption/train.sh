#!/bin/bash

fairseq-train \
 --user-dir scripts/customs \
 --arch image_caption \
 --task captioning \
 --save-dir result/captioning/ \
 --captions-dir data/datalab-cup3-reverse-image-caption-2021 \
 --criterion focal \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model data/sp/sp_oall_32k.model \
 --batch-size 4 \
 --update-freq 16\
 --save-interval 5 \
 --lr 0.001 \
 --max-epoch 20 \
 --optimizer adafactor 