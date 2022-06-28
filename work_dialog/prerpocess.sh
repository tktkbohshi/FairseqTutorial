#!/bin/bash

set -eu

# extrct data into raw texts
python scripts/extract_persona_chat.py japanese_persona_chat.xlsx data/perchat/raw/

# divide texts into word segments
python scripts/tokenize_sp.py data/perchat/raw data/perchat/spaced

# run fairseq preprocess
fairseq-preprocess \
    --trainpref data/perchat/spaced/train \
    --validpref data/perchat/spaced/valid \
    --testpref data/perchat/spaced/test \
    --source-lang src \
    --target-lang dst \
    --destdir data/perchat/bin \
    --tokenizer space \
    --srcdict data/sp/sp_oall_32k.txt \
    --tgtdict data/sp/sp_oall_32k.txt

