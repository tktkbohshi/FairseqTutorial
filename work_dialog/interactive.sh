#!/bin/bash

fairseq-interactive data/perchat/bin/ \
 --path data/pretrained/japanese-dialog-transformer-1.6B.pt
 --beam 10 \
 --seed 0 \
 --min-len 10 \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model data/sp/sp_oall_32k.model \
 --no-repeat-ngram-size 3 \
 --nbest 10 \
 --sampling \
 --sampling-topp 0.9 \
 --temperature 1.0