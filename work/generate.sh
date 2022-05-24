#!/bin/bash

fairseq-interactive data/sample/bin/ \
 --path checkpoints/persona50k-flat_1.6B_33avog1i_4.16.pt\
 --beam 10 \
 --seed 0 \
 --min-len 10 \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model data/dicts/sp_oall_32k.model \
 --no-repeat-ngram-size 3 \
 --nbest 10 \
 --sampling \
 --sampling-topp 0.9 \
 --temperature 1.0