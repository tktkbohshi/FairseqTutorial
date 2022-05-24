#!/bin/bash
git clone https://github.com/nttcslab/japanese-dialog-transformers.git
cp japanese-dialog-transformers/data/dicts/* data/sp
rm -R japanese-dialog-transformers