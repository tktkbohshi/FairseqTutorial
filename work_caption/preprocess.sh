#!/bin/bash

fairseq-preprocess \
  --trainpref data/datalab-cup3-reverse-image-caption-2021/train --validpref data/datalab-cup3-reverse-image-caption-2021/valid --testpref data/datalab-cup3-reverse-image-caption-2021/test \
  --source-lang src \
  --destdir data/datalab-cup3-reverse-image-caption-2021/bin \
  --tokenizer space \
  --srcdict /home/is/natsuno-t/projects/japanese-dialog-transformers/data/dicts/sp_oall_32k.txt 
