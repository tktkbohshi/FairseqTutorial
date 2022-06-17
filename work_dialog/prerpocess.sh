#!/bin/bash
fairseq-preprocess \
--trainpref data/perchat/train --validpref data/perchat/valid --testpref data/perchat/test \
--source-lang src --target-lang dst \
--destdir data/perchat/bin \
--tokenizer space \
--srcdict data/sp/sp_oall_32k.txt \
--tgtdict data/sp/sp_oall_32k.txt