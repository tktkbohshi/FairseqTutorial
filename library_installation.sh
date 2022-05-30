#!/bin/bash

pip install --upgrade pip

##########################################################################################################################################
# Before installing Fairseq, please install Pytorch of the appropriate version for your environment as the following website.
# https://pytorch.org/get-started/locally/
##########################################################################################################################################

# Install the latest fairseq
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

cd -

# Install sentencepiece
pip install sentencepiece
