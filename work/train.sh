#!/bin/bash

CUDA_VISIBLE_DEVICES=1 fairseq-train data/perchat/bin/ \
 --arch transformer \
 --finetune-from-model data/pretrained/japanese-dialog-transformer-1.6B.pt \
 --task translation \
 --save-dir result/perchat/ \
 --criterion cross_entropy \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model data/sentencepiece/sp_oall_32k.model \
 --batch-size 4 \
 --update-freq 16\
 --encoder-embed-dim 1920 --decoder-embed-dim 1920 \
 --encoder-attention-heads 32 --decoder-attention-heads 32 \
 --encoder-ffn-embed-dim 7680 --decoder-ffn-embed-dim 7680 \
 --encoder-layers 2 --decoder-layers 24 \
 --encoder-normalize-before --decoder-normalize-before \
 --save-interval 5 \
 --lr 0.000001 \
 --max-epoch 20 \
 --optimizer adafactor \