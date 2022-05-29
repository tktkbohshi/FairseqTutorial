#!/bin/bash
fairseq-preprocess \
--trainpref data/train --validpref data/valid --testpref data/test \
--source-lang src --target-lang dst \
--destdir data/perchat/bin \
--tokenizer space \
--srcdict data/sp/dicts/sp_oall_32k.txt \
--tgtdict data/sp/dicts/sp_oall_32k.txt