#!/bin/bash

fairseq-train data/perchat/bin/ \
 --arch image_caption \
 --task captioning \
 --save-dir result/captioning/ \
 --criterion cross_entropy \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model data/sentencepiece/sp_oall_32k.model \
 --batch-size 4 \
 --update-freq 16\
 --save-interval 5 \
 --lr 0.001 \
 --max-epoch 20 \
 --optimizer adafactor \
 --user-dir ./scripts/customs